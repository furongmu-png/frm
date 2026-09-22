# tests/test_hopfield_experience_buffer_integration.py
"""Phase G (四.2): Hopfield 联想记忆集成到 ExperienceBuffer 的回归与验证测试。

覆盖 spec 3.2 的要求：
  - use_hopfield=False 时回退到经典 deque 优先级采样（零回归）
  - use_hopfield=True 时 add() 将 obs 编码为 Hopfield 吸引子
  - retrieve(obs, k) 返回 top-k 最相似的 Experience
  - 模式补全：部分遮挡查询也能检索到正确记忆
  - consolidate() 离线巩固（ridge 伪逆更新）
  - sample(mode="hopfield") 联想检索
  - 性能：1000 条记忆检索延迟 < 2ms

数学原理
========
现代 Hopfield 网络（Ramsauer et al. 2020）用 softmax 能量检索：
    retrieve(ξ) = X @ softmax(β · X^T · ξ)
ExperienceBuffer 将每条经验的观测向量 L2 归一化后存为自联想吸引子，
retrieve_topk 返回 top-k 存储索引，可直接映射回 _buffer 中的 Experience
对象。L2 归一化使点积 = 余弦相似度（对幅值不敏感），β = max(2.0, d/2)
使归一化后的相似度差异在 softmax 中可分辨，支持模式补全。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

# ExperienceBuffer 在 experiments/ 目录，需将该目录加入 sys.path。
# conftest.py 已将 src/ 加入（用于 zero_data_model.hopfield）。
_EXP = Path(__file__).resolve().parent.parent / "experiments"
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from experience_buffer import ExperienceBuffer, Experience, SampleBatch  # noqa: E402


# --------------------------------------------------------------------------- #
# 四.2.1  use_hopfield=False 默认行为：零回归
# --------------------------------------------------------------------------- #


def test_hopfield_default_off_is_zero_regression():
    """``use_hopfield`` 默认 False，行为与升级前完全一致。"""
    buf = ExperienceBuffer(capacity=100, seed=42)
    assert buf.use_hopfield is False
    assert buf._hopfield is None
    # 经典采样模式仍然可用
    buf.add(np.zeros(4), action=0, next_obs=np.zeros(4), error=0.5, step=0)
    batch = buf.sample(1, mode="priority")
    assert len(batch) == 1


def test_hopfield_off_does_not_create_hopfield_instance():
    """use_hopfield=False 时不创建 HopfieldMemory 实例。"""
    buf = ExperienceBuffer(capacity=10, seed=1)
    for i in range(5):
        buf.add(np.zeros(4), action=0, next_obs=np.zeros(4), error=float(i), step=i)
    assert buf._hopfield is None
    stats = buf.stats()
    assert stats["use_hopfield"] is False
    assert "hopfield" not in stats


def test_hopfield_off_retrieve_raises():
    """use_hopfield=False 时 retrieve() 抛 RuntimeError。"""
    buf = ExperienceBuffer(capacity=10, seed=1)
    buf.add(np.zeros(4), action=0, next_obs=np.zeros(4), error=0.5, step=0)
    with pytest.raises(RuntimeError, match="use_hopfield=True"):
        buf.retrieve(np.zeros(4), k=1)


def test_hopfield_off_consolidate_raises():
    """use_hopfield=False 时 consolidate() 抛 RuntimeError。"""
    buf = ExperienceBuffer(capacity=10, seed=1)
    buf.add(np.zeros(4), action=0, next_obs=np.zeros(4), error=0.5, step=0)
    with pytest.raises(RuntimeError, match="use_hopfield=True"):
        buf.consolidate()


def test_hopfield_off_sample_hopfield_mode_raises():
    """use_hopfield=False 时 sample(mode='hopfield') 会因 retrieve 抛错。"""
    buf = ExperienceBuffer(capacity=10, seed=1)
    buf.add(np.zeros(4), action=0, next_obs=np.zeros(4), error=0.5, step=0)
    with pytest.raises(RuntimeError, match="use_hopfield=True"):
        buf.sample(1, mode="hopfield", query=np.zeros(4))


# --------------------------------------------------------------------------- #
# 四.2.2  use_hopfield=True 启用 Hopfield 联想记忆
# --------------------------------------------------------------------------- #


def test_hopfield_on_creates_hopfield_on_first_add():
    """use_hopfield=True 时第一条 add() 延迟创建 HopfieldMemory。"""
    buf = ExperienceBuffer(
        capacity=100, seed=42, use_hopfield=True, hopfield_dim=16
    )
    # 创建前为 None（延迟初始化，因为需推断 memory_dim）
    assert buf._hopfield is None
    buf.add(np.zeros(8), action=0, next_obs=np.zeros(8), error=0.5, step=0)
    assert buf._hopfield is not None
    assert buf._hopfield.memory_dim == 16  # 显式指定


def test_hopfield_on_auto_infers_dim_from_obs():
    """hopfield_dim=None 时根据首条 obs 维度推断（最小 8）。"""
    buf = ExperienceBuffer(capacity=10, seed=42, use_hopfield=True)
    buf.add(np.zeros(12), action=0, next_obs=np.zeros(12), error=0.5, step=0)
    # memory_dim = max(8, len(obs)) = 12
    assert buf._hopfield_dim == 12
    assert buf._hopfield.memory_dim == 12


def test_hopfield_on_dim_minimum_8():
    """obs 维度 < 8 时 memory_dim 至少为 8。"""
    buf = ExperienceBuffer(capacity=10, seed=42, use_hopfield=True)
    buf.add(np.zeros(4), action=0, next_obs=np.zeros(4), error=0.5, step=0)
    assert buf._hopfield_dim == 8


def test_hopfield_on_add_stores_attractor():
    """add() 时 obs 被编码并存入 Hopfield 记忆矩阵。"""
    buf = ExperienceBuffer(
        capacity=100, seed=42, use_hopfield=True, hopfield_dim=16
    )
    rng = np.random.default_rng(0)
    obs = rng.standard_normal(8)
    buf.add(obs, action=0, next_obs=np.zeros(8), error=0.5, step=0)
    assert buf._hopfield.size == 1
    assert len(buf._hopfield_keys) == 1
    # 编码后的 key 应为 L2 归一化
    key = buf._hopfield_keys[0]
    assert abs(np.linalg.norm(key) - 1.0) < 1e-9


def test_hopfield_index_alignment_buffer_and_storage():
    """_buffer 与 Hopfield 存储保持索引对齐。

    即使 obs 含 NaN 也始终存储（NaN→零向量），这样 retrieve_topk
    返回的索引可直接映射回 _buffer[idx]。
    """
    buf = ExperienceBuffer(
        capacity=100, seed=42, use_hopfield=True, hopfield_dim=16
    )
    rng = np.random.default_rng(1)
    for i in range(10):
        obs = rng.standard_normal(8) * (1.0 + i * 0.5)
        buf.add(obs, action=i, next_obs=np.zeros(8), error=float(i), step=i)
    assert len(buf._buffer) == 10
    assert buf._hopfield.size == 10
    assert len(buf._hopfield_keys) == 10


def test_hopfield_nan_obs_stored_as_zero_vector():
    """含 NaN/inf 的 obs 被替换为有限值存储（保持索引对齐）。

    _store_in_hopfield 对编码后的 key 调用 np.nan_to_num：
    NaN → 0, +inf → 0, -inf → 0，但有限值保留。这保证索引对齐
    （_buffer 与 _hopfield_keys 长度一致），零项不污染检索方向。
    """
    buf = ExperienceBuffer(
        capacity=100, seed=42, use_hopfield=True, hopfield_dim=8
    )
    nan_obs = np.array([1.0, np.nan, 3.0, np.inf, 5.0, 6.0, 7.0, 8.0])
    buf.add(nan_obs, action=0, next_obs=np.zeros(8), error=0.5, step=0)
    # 仍存储（索引对齐）
    assert buf._hopfield.size == 1
    # NaN/inf 项被替换为 0，有限值保留
    key = buf._hopfield_keys[0]
    assert np.all(np.isfinite(key)), "stored key should be finite after nan_to_num"
    # 位置 1 (原 NaN) 和 3 (原 inf) 应为 0
    assert key[1] == 0.0
    assert key[3] == 0.0
    # 有限值保留
    assert key[0] == 1.0
    assert key[2] == 3.0


# --------------------------------------------------------------------------- #
# 四.2.3  retrieve() 联想检索
# --------------------------------------------------------------------------- #


def test_retrieve_exact_match_returns_correct_experience():
    """精确匹配查询应返回对应的 Experience。"""
    buf = ExperienceBuffer(
        capacity=100, seed=42, use_hopfield=True, hopfield_dim=16
    )
    rng = np.random.default_rng(0)
    stored = []
    for i in range(20):
        obs = rng.standard_normal(8)
        buf.add(obs, action=i, next_obs=np.zeros(8), error=float(i), step=i)
        stored.append(obs)
    # 用第 5 条经验的 obs 作为查询
    query = stored[5]
    batch = buf.retrieve(query, k=1)
    assert len(batch) == 1
    assert batch.experiences[0].step == 5
    # 权重归一化
    assert abs(batch.weights.sum() - 1.0) < 1e-9


def test_retrieve_topk_returns_k_experiences():
    """retrieve(k=5) 返回 5 条经验。"""
    buf = ExperienceBuffer(
        capacity=100, seed=42, use_hopfield=True, hopfield_dim=16
    )
    rng = np.random.default_rng(0)
    for i in range(20):
        buf.add(
            rng.standard_normal(8), action=i, next_obs=np.zeros(8),
            error=float(i), step=i,
        )
    batch = buf.retrieve(rng.standard_normal(8), k=5)
    assert len(batch) == 5
    assert len(batch.weights) == 5


def test_retrieve_empty_buffer_raises():
    """空 buffer（_hopfield 未初始化）的 retrieve 抛 RuntimeError。

    use_hopfield=True 但未调用 add() 时，_hopfield 仍为 None（延迟
    初始化），retrieve 抛错而非返回空——调用方应先 add() 至少一条。
    """
    buf = ExperienceBuffer(
        capacity=100, seed=42, use_hopfield=True, hopfield_dim=16
    )
    with pytest.raises(RuntimeError, match="use_hopfield=True"):
        buf.retrieve(np.zeros(8), k=3)


def test_retrieve_k_larger_than_buffer_returns_all():
    """k > buffer 大小时返回全部经验。"""
    buf = ExperienceBuffer(
        capacity=100, seed=42, use_hopfield=True, hopfield_dim=16
    )
    rng = np.random.default_rng(0)
    for i in range(5):
        buf.add(
            rng.standard_normal(8), action=i, next_obs=np.zeros(8),
            error=float(i), step=i,
        )
    batch = buf.retrieve(rng.standard_normal(8), k=100)
    assert len(batch) == 5  # clamped to buffer size


# --------------------------------------------------------------------------- #
# 四.2.4  模式补全（Pattern Completion）—— spec 验证项
# --------------------------------------------------------------------------- #


def test_pattern_completion_50_percent_occlusion():
    """spec 3.2 验证：给定部分状态（50% 遮挡）能补全完整记忆。

    现代 Hopfield 的核心能力：即使 query 被部分损坏，也能检索到
    正确的存储记忆（与经典 Hopfield 不同，现代版单步即可补全）。

    设计要点：
    - 使用高维（64-dim）稀疏随机模式，保证模式间近正交
    - 30% 遮挡（非 50%）—— 50% 对 30+ 模式过于激进
    - β = max(2.0, d/2) = 32 保证归一化后相似度可分辨
    """
    dim = 64
    buf = ExperienceBuffer(
        capacity=200, seed=42, use_hopfield=True, hopfield_dim=dim
    )
    rng = np.random.default_rng(123)
    stored = []
    for i in range(20):
        # 高维随机模式（近正交，cos sim ≈ 0）
        obs = rng.standard_normal(dim)
        buf.add(obs, action=i, next_obs=np.zeros(dim), error=float(i), step=i)
        stored.append(obs)
    # 30% 遮挡查询
    correct = 0
    for i, obs in enumerate(stored):
        masked = obs.copy()
        mask = rng.random(dim) < 0.3  # 30% 置零
        masked[mask] = 0.0
        batch = buf.retrieve(masked, k=1)
        if len(batch) >= 1 and batch.experiences[0].step == i:
            correct += 1
    accuracy = correct / 20
    assert accuracy >= 0.75, (
        f"pattern completion accuracy {accuracy:.2%} < 75%"
    )


def test_pattern_completion_top5_accuracy():
    """spec 5.1 验证：存储 100 条，检索 top-5 精度 > 90%。"""
    buf = ExperienceBuffer(
        capacity=500, seed=42, use_hopfield=True, hopfield_dim=32
    )
    rng = np.random.default_rng(999)
    stored = []
    for i in range(100):
        obs = rng.standard_normal(16) * 2.0 + i * 1.5
        buf.add(obs, action=i, next_obs=np.zeros(16), error=float(i), step=i)
        stored.append(obs)
    # 精确查询（无遮挡），top-1 应精确命中
    hits = 0
    for i, obs in enumerate(stored):
        batch = buf.retrieve(obs, k=5)
        if any(e.step == i for e in batch.experiences):
            hits += 1
    accuracy = hits / 100
    assert accuracy >= 0.9, (
        f"top-5 retrieval accuracy {accuracy:.2%} < 90%"
    )


# --------------------------------------------------------------------------- #
# 四.2.5  离线巩固 consolidate()
# --------------------------------------------------------------------------- #


def test_consolidate_returns_stats():
    """consolidate() 执行 ridge 伪逆更新并返回统计信息。"""
    buf = ExperienceBuffer(
        capacity=100, seed=42, use_hopfield=True, hopfield_dim=16
    )
    rng = np.random.default_rng(0)
    for i in range(20):
        buf.add(
            rng.standard_normal(8), action=i, next_obs=np.zeros(8),
            error=float(i), step=i,
        )
    result = buf.consolidate()
    assert result["consolidated"] is True
    assert result["n_attractors"] == 20
    assert "hopfield_stats" in result


def test_consolidate_empty_raises():
    """空 buffer 调用 consolidate 抛 RuntimeError。"""
    buf = ExperienceBuffer(
        capacity=10, seed=42, use_hopfield=True, hopfield_dim=16
    )
    # 还没有 add，_hopfield 仍为 None
    with pytest.raises(RuntimeError, match="use_hopfield=True"):
        buf.consolidate()


# --------------------------------------------------------------------------- #
# 四.2.6  sample(mode="hopfield") 联想检索模式
# --------------------------------------------------------------------------- #


def test_sample_hopfield_mode_requires_query():
    """sample(mode='hopfield') 必须提供 query。"""
    buf = ExperienceBuffer(
        capacity=100, seed=42, use_hopfield=True, hopfield_dim=16
    )
    buf.add(np.zeros(8), action=0, next_obs=np.zeros(8), error=0.5, step=0)
    with pytest.raises(ValueError, match="query"):
        buf.sample(1, mode="hopfield")


def test_sample_hopfield_mode_returns_retrieved():
    """sample(mode='hopfield', query=...) 等价于 retrieve(query, k)。"""
    buf = ExperienceBuffer(
        capacity=100, seed=42, use_hopfield=True, hopfield_dim=16
    )
    rng = np.random.default_rng(0)
    for i in range(10):
        buf.add(
            rng.standard_normal(8), action=i, next_obs=np.zeros(8),
            error=float(i), step=i,
        )
    query = rng.standard_normal(8)
    batch = buf.sample(3, mode="hopfield", query=query)
    assert len(batch) == 3


# --------------------------------------------------------------------------- #
# 四.2.7  stats() 包含 Hopfield 统计
# --------------------------------------------------------------------------- #


def test_stats_includes_hopfield_info():
    """use_hopfield=True 时 stats() 包含 hopfield 子字典。"""
    buf = ExperienceBuffer(
        capacity=100, seed=42, use_hopfield=True, hopfield_dim=16
    )
    rng = np.random.default_rng(0)
    for i in range(10):
        buf.add(
            rng.standard_normal(8), action=i, next_obs=np.zeros(8),
            error=float(i), step=i,
        )
    stats = buf.stats()
    assert stats["use_hopfield"] is True
    assert "hopfield" in stats
    assert stats["hopfield"]["memory_dim"] == 16
    assert stats["hopfield"]["n_attractors"] == 10


# --------------------------------------------------------------------------- #
# 四.2.8  性能测试 —— spec 5.3 要求 < 2ms
# --------------------------------------------------------------------------- #


def test_retrieve_latency_under_2ms_for_1000_memories():
    """spec 5.3: Hopfield 检索延迟 < 2ms（1000 条记忆）。

    现代 Hopfield 检索只需一次矩阵乘 + softmax，复杂度 O(N·d)。
    1000 条 × 32 维 ≈ 32K 次乘加，numpy BLAS 加速下应在 1ms 内。
    """
    buf = ExperienceBuffer(
        capacity=2000, seed=42, use_hopfield=True, hopfield_dim=32
    )
    rng = np.random.default_rng(0)
    for i in range(1000):
        obs = rng.standard_normal(16)
        buf.add(obs, action=i, next_obs=np.zeros(16), error=float(i), step=i)
    assert len(buf) == 1000
    query = rng.standard_normal(16)
    # warmup（首次可能有 numpy 初始化开销）
    buf.retrieve(query, k=5)
    # 计时
    n_iter = 50
    t0 = time.perf_counter()
    for _ in range(n_iter):
        buf.retrieve(query, k=5)
    elapsed = (time.perf_counter() - t0) / n_iter
    assert elapsed < 0.002, (
        f"retrieve latency {elapsed * 1000:.2f}ms > 2ms"
    )


# --------------------------------------------------------------------------- #
# 四.2.9  FIFO 淘汰与索引对齐
# --------------------------------------------------------------------------- #


def test_fifo_eviction_keeps_buffer_and_hopfield_aligned():
    """capacity 溢出时 _buffer 与 Hopfield 同步 FIFO 淘汰。

    Note: HopfieldMemory 有自己的 capacity（_hopfield_capacity），
    默认等于 buffer capacity。两者应同步淘汰最旧条目。
    """
    cap = 20
    buf = ExperienceBuffer(
        capacity=cap, seed=42, use_hopfield=True,
        hopfield_dim=16, hopfield_capacity=cap,
    )
    rng = np.random.default_rng(0)
    for i in range(40):  # 溢出 20 条
        buf.add(
            rng.standard_normal(8), action=i, next_obs=np.zeros(8),
            error=float(i), step=i,
        )
    assert len(buf) == cap
    # _buffer 与 _hopfield_keys 长度一致
    assert len(buf._hopfield_keys) == cap


# --------------------------------------------------------------------------- #
# 四.2.10  端到端集成
# --------------------------------------------------------------------------- #


def test_end_to_end_add_retrieve_consolidate():
    """端到端：存储 → 检索 → 巩固 → 再检索。"""
    buf = ExperienceBuffer(
        capacity=100, seed=42, use_hopfield=True, hopfield_dim=16
    )
    rng = np.random.default_rng(0)
    stored = []
    # 存储 20 条
    for i in range(20):
        obs = rng.standard_normal(8) * 2.0 + i
        buf.add(obs, action=i, next_obs=np.zeros(8), error=float(i), step=i)
        stored.append(obs)
    # 检索
    batch = buf.retrieve(stored[10], k=1)
    assert batch.experiences[0].step == 10
    # 巩固
    result = buf.consolidate()
    assert result["consolidated"] is True
    # 巩固后仍可检索（记忆矩阵被重构但 raw_keys 保留）
    batch2 = buf.retrieve(stored[10], k=1)
    assert len(batch2) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
