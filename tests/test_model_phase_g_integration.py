# tests/test_model_phase_g_integration.py
"""Phase G (四.3): 三项核心认知升级在 ZeroDataModel 中的集成测试。

覆盖 spec 4.1-4.2 的要求：
  - use_s4 / use_pcn / use_hopfield 开关（默认 False，零回归）
  - use_s4=True 时 S4 隐状态暴露在 cognitive_upgrades.s4_state
  - use_pcn=True 时 PCN 层级误差暴露在 cognitive_upgrades.pcn
  - use_hopfield=True 时 Hopfield 检索暴露在 cognitive_upgrades.memory_retrieved
  - 环境变量 ZDM_USE_* 作为默认开关
  - pickle 支持新属性
  - think() 整体延迟增加 < 20%（性能要求）
"""
from __future__ import annotations

import os
import pickle
import time

import numpy as np
import pytest

from zero_data_model.model import ZeroDataModel


# --------------------------------------------------------------------------- #
# 四.3.1  默认开关为 False：零回归
# --------------------------------------------------------------------------- #


def test_default_flags_all_false():
    """不传任何 use_* 参数时，三个开关全为 False。"""
    m = ZeroDataModel(dim=16, seed=42)
    assert m._use_s4 is False
    assert m._use_pcn is False
    assert m._use_hopfield is False
    assert m.pcn_hierarchy is None
    assert m.hopfield_memory is None
    assert m.s4_state_cache is None
    # S4 层不在 active_inference 内
    assert m.active_inference.generative_model._s4_layer is None


def test_default_think_no_cognitive_upgrades():
    """默认配置 think() 不产生 cognitive_upgrades（零回归）。"""
    m = ZeroDataModel(dim=16, seed=42)
    sig = m.think()
    assert "cognitive_upgrades" not in sig.metadata


# --------------------------------------------------------------------------- #
# 四.3.2  use_s4=True
# --------------------------------------------------------------------------- #


def test_use_s4_creates_s4_layer():
    """use_s4=True 时 ActiveInferenceEngine 内部创建 S4 层。"""
    m = ZeroDataModel(dim=16, seed=42, use_s4=True)
    assert m._use_s4 is True
    s4 = m.active_inference.generative_model._s4_layer
    assert s4 is not None
    assert s4.state_dim == 16


def test_use_s4_exposes_state_in_metadata():
    """think() 后 cognitive_upgrades 含 s4_state。"""
    m = ZeroDataModel(dim=16, seed=42, use_s4=True)
    sig = m.think()
    cu = sig.metadata["cognitive_upgrades"]
    assert "s4_state" in cu
    assert isinstance(cu["s4_state"], list)
    assert len(cu["s4_state"]) <= 8
    assert "s4_spectral_radius" in cu


def test_use_s4_state_advances_across_cycles():
    """多个 think() 周期后 S4 隐状态应变化。"""
    m = ZeroDataModel(dim=16, seed=42, use_s4=True)
    sig1 = m.think()
    state1 = sig1.metadata["cognitive_upgrades"]["s4_state"]
    sig2 = m.think()
    state2 = sig2.metadata["cognitive_upgrades"]["s4_state"]
    assert state1 != state2, "S4 state should advance across think() cycles"


# --------------------------------------------------------------------------- #
# 四.3.3  use_pcn=True
# --------------------------------------------------------------------------- #


def test_use_pcn_creates_hierarchy():
    """use_pcn=True 时构造 HierarchicalZeroDataModel（3 层 PCN）。"""
    m = ZeroDataModel(dim=16, seed=42, use_pcn=True)
    assert m._use_pcn is True
    assert m.pcn_hierarchy is not None
    layers = m.pcn_hierarchy.layers
    assert "L0" in layers
    assert "L1" in layers
    assert "L2" in layers


def test_use_pcn_exposes_layer_errors_in_metadata():
    """think() 后 cognitive_upgrades.pcn 含 layer_errors。"""
    m = ZeroDataModel(dim=16, seed=42, use_pcn=True)
    sig = m.think()
    cu = sig.metadata["cognitive_upgrades"]
    assert "pcn" in cu
    assert "layer_errors" in cu["pcn"]
    le = cu["pcn"]["layer_errors"]
    assert "L0" in le and "L1" in le and "L2" in le
    assert all(isinstance(v, float) for v in le.values())


def test_use_pcn_state_norms_in_metadata():
    """think() 后 cognitive_upgrades.pcn 含 state_norms。"""
    m = ZeroDataModel(dim=16, seed=42, use_pcn=True)
    sig = m.think()
    cu = sig.metadata["cognitive_upgrades"]
    assert "state_norms" in cu["pcn"]
    sn = cu["pcn"]["state_norms"]
    assert all(isinstance(v, float) for v in sn.values())


# --------------------------------------------------------------------------- #
# 四.3.4  use_hopfield=True
# --------------------------------------------------------------------------- #


def test_use_hopfield_creates_memory():
    """use_hopfield=True 时构造 HopfieldMemory。"""
    m = ZeroDataModel(dim=16, seed=42, use_hopfield=True)
    assert m._use_hopfield is True
    assert m.hopfield_memory is not None
    assert m.hopfield_memory.memory_dim == 16


def test_use_hopfield_empty_memory_no_retrieve():
    """空记忆时 think() 不暴露 memory_retrieved（无记忆可检索）。"""
    m = ZeroDataModel(dim=16, seed=42, use_hopfield=True)
    sig = m.think()
    cu = sig.metadata["cognitive_upgrades"]
    assert "memory_retrieved" not in cu


def test_use_hopfield_with_memories_exposes_retrieval():
    """有记忆时 think() 暴露 memory_retrieved。"""
    m = ZeroDataModel(dim=16, seed=42, use_hopfield=True)
    rng = np.random.default_rng(0)
    for _ in range(5):
        obs = rng.standard_normal(16)
        m.hopfield_memory.store(obs / np.linalg.norm(obs))
    sig = m.think()
    cu = sig.metadata["cognitive_upgrades"]
    assert "memory_retrieved" in cu
    mr = cu["memory_retrieved"]
    assert mr["n_memories"] == 5
    assert 0.0 <= mr["top1_similarity"] <= 1.0


# --------------------------------------------------------------------------- #
# 四.3.5  三开关组合
# --------------------------------------------------------------------------- #


def test_all_three_flags_enabled():
    """同时启用三个开关，think() 暴露全部 metadata。"""
    m = ZeroDataModel(
        dim=16, seed=42, use_s4=True, use_pcn=True, use_hopfield=True
    )
    # 存一些 Hopfield 记忆
    rng = np.random.default_rng(0)
    for _ in range(3):
        obs = rng.standard_normal(16)
        m.hopfield_memory.store(obs / np.linalg.norm(obs))

    sig = m.think()
    cu = sig.metadata["cognitive_upgrades"]
    assert "s4_state" in cu
    assert "pcn" in cu
    assert "memory_retrieved" in cu


def test_all_three_json_serializable():
    """三开关启用时 cognitive_upgrades 可 JSON 序列化。"""
    import json

    m = ZeroDataModel(
        dim=16, seed=42, use_s4=True, use_pcn=True, use_hopfield=True
    )
    rng = np.random.default_rng(0)
    for _ in range(3):
        obs = rng.standard_normal(16)
        m.hopfield_memory.store(obs / np.linalg.norm(obs))
    sig = m.think()
    # 不应抛 JSON 序列化错误
    json.dumps(sig.metadata)


# --------------------------------------------------------------------------- #
# 四.3.6  环境变量默认值
# --------------------------------------------------------------------------- #


def test_env_var_use_s4(monkeypatch):
    """ZDM_USE_S4=1 时 use_s4 默认为 True。"""
    monkeypatch.setenv("ZDM_USE_S4", "1")
    m = ZeroDataModel(dim=16, seed=42)
    assert m._use_s4 is True
    assert m.active_inference.generative_model._s4_layer is not None


def test_env_var_use_pcn(monkeypatch):
    """ZDM_USE_PCN=1 时 use_pcn 默认为 True。"""
    monkeypatch.setenv("ZDM_USE_PCN", "1")
    m = ZeroDataModel(dim=16, seed=42)
    assert m._use_pcn is True
    assert m.pcn_hierarchy is not None


def test_env_var_use_hopfield(monkeypatch):
    """ZDM_USE_HOPFIELD=1 时 use_hopfield 默认为 True。"""
    monkeypatch.setenv("ZDM_USE_HOPFIELD", "1")
    m = ZeroDataModel(dim=16, seed=42)
    assert m._use_hopfield is True
    assert m.hopfield_memory is not None


def test_explicit_param_overrides_env(monkeypatch):
    """显式参数优先于环境变量。"""
    monkeypatch.setenv("ZDM_USE_S4", "1")
    m = ZeroDataModel(dim=16, seed=42, use_s4=False)
    assert m._use_s4 is False
    assert m.active_inference.generative_model._s4_layer is None


# --------------------------------------------------------------------------- #
# 四.3.7  pickle 支持
# --------------------------------------------------------------------------- #


def test_pickle_with_all_flags():
    """三开关启用时模型可 pickle（pcn_hierarchy 重建）。"""
    m = ZeroDataModel(
        dim=16, seed=42, use_s4=True, use_pcn=True, use_hopfield=True
    )
    m.think()  # 推进一些状态
    data = pickle.dumps(m)
    m2 = pickle.loads(data)
    # 基本属性恢复
    assert m2.dim == 16
    assert m2._use_s4 is True
    assert m2._use_pcn is True
    assert m2._use_hopfield is True
    # PCN 层级重建
    assert m2.pcn_hierarchy is not None
    # Hopfield 记忆恢复（有 __getstate__/__setstate__）
    assert m2.hopfield_memory is not None
    # S4 层在 active_inference 内部恢复
    assert m2.active_inference.generative_model._s4_layer is not None
    # think() 仍能运行
    sig = m2.think()
    assert "cognitive_upgrades" in sig.metadata


def test_pickle_default_model():
    """默认配置模型可 pickle（零回归）。"""
    m = ZeroDataModel(dim=16, seed=42)
    m.think()
    data = pickle.dumps(m)
    m2 = pickle.loads(data)
    assert m2.dim == 16
    sig = m2.think()
    assert "cognitive_upgrades" not in sig.metadata


# --------------------------------------------------------------------------- #
# 四.3.8  dim=1 边界
# --------------------------------------------------------------------------- #


def test_dim_one_with_all_flags():
    """dim=1 + 三开关启用不崩溃（L2 维度 floor 到 1）。"""
    m = ZeroDataModel(
        dim=1, seed=42, use_s4=True, use_pcn=True, use_hopfield=True
    )
    sig = m.think()
    cu = sig.metadata["cognitive_upgrades"]
    assert "s4_state" in cu
    assert "pcn" in cu
    # hopfield 为空时不暴露 memory_retrieved
    assert "memory_retrieved" not in cu


# --------------------------------------------------------------------------- #
# 四.3.9  性能：think() 延迟增加 < 20%
# --------------------------------------------------------------------------- #


def test_think_latency_increase_under_20_percent():
    """spec 5.3: 整体 think() 延迟增加不超过 20%。

    对比：默认配置 vs 三开关全开，think() 延迟增加 < 20%。
    S4 的 step()/update() 是 O(dim^2) 矩阵运算；PCN 的 run_pcn_cycle
    是 3 层 O(dim^2) 运算；Hopfield retrieve 是 O(N·dim) 矩阵乘。
    在 dim=64 下总开销应在 ms 级。
    """
    dim = 64
    n_warmup = 3
    n_measure = 20

    # 基准：默认配置
    m_base = ZeroDataModel(dim=dim, seed=42)
    for _ in range(n_warmup):
        m_base.think()
    t0 = time.perf_counter()
    for _ in range(n_measure):
        m_base.think()
    base_latency = (time.perf_counter() - t0) / n_measure

    # 升级：三开关全开
    m_upgraded = ZeroDataModel(
        dim=dim, seed=42, use_s4=True, use_pcn=True, use_hopfield=True
    )
    # 存一些 Hopfield 记忆
    rng = np.random.default_rng(0)
    for _ in range(50):
        obs = rng.standard_normal(dim)
        m_upgraded.hopfield_memory.store(obs / np.linalg.norm(obs))
    for _ in range(n_warmup):
        m_upgraded.think()
    t0 = time.perf_counter()
    for _ in range(n_measure):
        m_upgraded.think()
    upgraded_latency = (time.perf_counter() - t0) / n_measure

    # 升级后延迟增加比例
    increase = (upgraded_latency - base_latency) / base_latency
    # 阈值 20%，但允许一些波动（CI 环境）
    assert increase < 0.20, (
        f"think() latency increased by {increase:.1%} > 20% "
        f"(base={base_latency*1000:.2f}ms, upgraded={upgraded_latency*1000:.2f}ms)"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
