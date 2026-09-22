"""Phase G (四.4): snapshot.py 新字段 + 前端面板测试。

覆盖 spec 4.3 的要求：
  - Snapshot dataclass 新增 s4_state / layer_errors / memory_retrieved
  - to_dict() 包含三个新字段（默认空容器）
  - _extract_phase_g_state 从 signal.metadata["cognitive_upgrades"] 读取
  - 回退路径：metadata=None 时从 model.s4_state_cache 读取 S4 状态
  - NaN/inf 值被过滤
  - collect() 的 metadata 参数默认 None（向后兼容）
  - 端到端：真实模型 + think() + collect() 产生已填充的字段
  - JSON 可序列化
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Make src/ importable (matches pytest rootdir convention).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visualization.snapshot import Snapshot, SnapshotCollector  # noqa: E402
from zero_data_model.model import ZeroDataModel  # noqa: E402


# --------------------------------------------------------------------------- #
# 4.4.1  Snapshot dataclass 默认值（零回归）
# --------------------------------------------------------------------------- #


class TestSnapshotDefaults:
    def test_default_snapshot_has_empty_phase_g_fields(self):
        """新建 Snapshot 的三个 Phase G 字段为空容器。"""
        s = Snapshot()
        assert s.s4_state == []
        assert s.layer_errors == {}
        assert s.memory_retrieved == {}

    def test_to_dict_includes_phase_g_keys(self):
        """to_dict() 必须包含三个新键。"""
        d = Snapshot().to_dict()
        assert "s4_state" in d
        assert "layer_errors" in d
        assert "memory_retrieved" in d
        assert d["s4_state"] == []
        assert d["layer_errors"] == {}
        assert d["memory_retrieved"] == {}

    def test_to_dict_rounds_floats_to_6dp(self):
        """浮点数应被截断到 6 位小数，避免 payload 膨胀。"""
        s = Snapshot(
            s4_state=[0.123456789, -0.987654321],
            layer_errors={"L0": 1.23456789, "L1": 0.0000001},
            memory_retrieved={"n_memories": 5, "top1_similarity": 0.87654321},
        )
        d = s.to_dict()
        assert d["s4_state"] == [0.123457, -0.987654]
        assert d["layer_errors"]["L0"] == 1.234568
        assert d["layer_errors"]["L1"] == 0.0
        assert d["memory_retrieved"]["n_memories"] == 5
        assert d["memory_retrieved"]["top1_similarity"] == 0.876543

    def test_to_dict_is_json_serializable(self):
        """完整 snapshot dict 必须可被 json.dumps 序列化。"""
        s = Snapshot(
            s4_state=[0.1, -0.2, 0.3],
            layer_errors={"L0": 1.0, "L1": 0.5, "L2": 0.25},
            memory_retrieved={"n_memories": 10, "top1_similarity": 0.9},
        )
        # 不应抛出 TypeError
        json.dumps(s.to_dict())


# --------------------------------------------------------------------------- #
# 4.4.2  _extract_phase_g_state 解析逻辑
# --------------------------------------------------------------------------- #


class TestExtractPhaseGState:
    """直接测试 _extract_phase_g_state 静态方法。"""

    def test_metadata_with_all_three_upgrades(self):
        """metadata 含完整 cognitive_upgrades 时三个字段都被填充。"""
        metadata = {
            "cognitive_upgrades": {
                "s4_state": [0.1, -0.2, 0.3],
                "pcn": {
                    "layer_errors": {"L0": 1.0, "L1": 0.5, "L2": 0.25},
                    "state_norms": {"L0": 2.0, "L1": 1.0, "L2": 0.5},
                },
                "memory_retrieved": {"n_memories": 5, "top1_similarity": 0.9},
            }
        }
        s4, le, mem = SnapshotCollector._extract_phase_g_state(None, metadata)
        assert s4 == [0.1, -0.2, 0.3]
        assert le == {"L0": 1.0, "L1": 0.5, "L2": 0.25}
        assert mem == {"n_memories": 5, "top1_similarity": 0.9}

    def test_metadata_none_falls_back_to_s4_cache(self):
        """metadata=None 时从 model.s4_state_cache 回退读取 S4 状态。"""
        class FakeModel:
            s4_state_cache = [0.5, -0.5, 0.0]
        s4, le, mem = SnapshotCollector._extract_phase_g_state(FakeModel(), None)
        assert s4 == [0.5, -0.5, 0.0]
        # PCN/Hopfield 没有缓存属性，回退路径下保持空
        assert le == {}
        assert mem == {}

    def test_metadata_none_no_s4_cache_returns_empty(self):
        """model 无 s4_state_cache 属性时返回空。"""
        class FakeModel:
            pass
        s4, le, mem = SnapshotCollector._extract_phase_g_state(FakeModel(), None)
        assert s4 == []
        assert le == {}
        assert mem == {}

    def test_nan_values_filtered_from_s4_state(self):
        """S4 状态中的 NaN/inf 应被过滤。"""
        metadata = {
            "cognitive_upgrades": {
                "s4_state": [0.1, float("nan"), float("inf"), -0.2, float("-inf")],
            }
        }
        s4, _, _ = SnapshotCollector._extract_phase_g_state(None, metadata)
        assert s4 == [0.1, -0.2]

    def test_nan_values_filtered_from_layer_errors(self):
        """layer_errors 中的 NaN 应被过滤。"""
        metadata = {
            "cognitive_upgrades": {
                "pcn": {"layer_errors": {"L0": 1.0, "L1": float("nan"), "L2": 0.5}},
            }
        }
        _, le, _ = SnapshotCollector._extract_phase_g_state(None, metadata)
        assert le == {"L0": 1.0, "L2": 0.5}

    def test_partial_metadata_only_populates_present_fields(self):
        """metadata 只有 S4 信息时，PCN/Hopfield 字段为空。"""
        metadata = {"cognitive_upgrades": {"s4_state": [0.1, 0.2]}}
        s4, le, mem = SnapshotCollector._extract_phase_g_state(None, metadata)
        assert s4 == [0.1, 0.2]
        assert le == {}
        assert mem == {}

    def test_metadata_not_dict_returns_empty(self):
        """metadata 不是 dict 时安全返回空。"""
        s4, le, mem = SnapshotCollector._extract_phase_g_state(None, "not a dict")
        assert s4 == []
        assert le == {}
        assert mem == {}

    def test_cognitive_upgrades_not_dict_returns_empty(self):
        """cognitive_upgrades 不是 dict 时安全返回空。"""
        metadata = {"cognitive_upgrades": "not a dict"}
        s4, le, mem = SnapshotCollector._extract_phase_g_state(None, metadata)
        assert s4 == []
        assert le == {}
        assert mem == {}


# --------------------------------------------------------------------------- #
# 4.4.3  collect() 向后兼容 + 端到端集成
# --------------------------------------------------------------------------- #


class TestCollectBackwardCompat:
    def test_collect_without_metadata_arg_still_works(self):
        """旧调用方不传 metadata 参数时不应报错（默认 None）。"""
        m = ZeroDataModel(dim=8, seed=42)  # 所有升级关闭
        collector = SnapshotCollector()
        # 不传 metadata —— 应使用默认 None
        snap = collector.collect(model=m, step=0, modality="physics")
        d = snap.to_dict()
        # 升级关闭时三个字段应为空
        assert d["s4_state"] == []
        assert d["layer_errors"] == {}
        assert d["memory_retrieved"] == {}

    def test_collect_history_stores_phase_g_fields(self):
        """collect 后 history 中的快照包含 Phase G 字段。"""
        m = ZeroDataModel(dim=8, seed=42)
        collector = SnapshotCollector()
        collector.collect(model=m, step=0, modality="physics")
        hist = collector.get_history()
        assert len(hist) == 1
        assert "s4_state" in hist[0]
        assert "layer_errors" in hist[0]
        assert "memory_retrieved" in hist[0]


# --------------------------------------------------------------------------- #
# 4.4.4  端到端：真实模型 + think() + collect()
# --------------------------------------------------------------------------- #


class TestEndToEndPhaseG:
    """用真实 ZeroDataModel 验证 Phase G 字段在 collect() 后被填充。"""

    @pytest.fixture
    def model_with_all_upgrades(self):
        """构造启用全部三项升级的模型，并预存 Hopfield 记忆。"""
        m = ZeroDataModel(dim=16, seed=42, use_s4=True, use_pcn=True, use_hopfield=True)
        # 预存两条记忆以便 retrieve 返回非空结果
        m.hopfield_memory.store(np.ones(16), np.ones(16))
        m.hopfield_memory.store(-np.ones(16), -np.ones(16))
        return m

    def test_s4_state_populated_after_think(self, model_with_all_upgrades):
        """think() 后 s4_state 应有 ≤8 个有限浮点数。"""
        m = model_with_all_upgrades
        obs = np.random.default_rng(0).standard_normal(16)
        signal = m.think(obs)
        collector = SnapshotCollector()
        snap = collector.collect(
            model=m, step=0, modality="physics",
            metadata=getattr(signal, "metadata", None),
        )
        d = snap.to_dict()
        assert len(d["s4_state"]) > 0
        assert len(d["s4_state"]) <= 8
        assert all(isinstance(x, float) for x in d["s4_state"])
        assert all(np.isfinite(x) for x in d["s4_state"])

    def test_layer_errors_populated_after_think(self, model_with_all_upgrades):
        """think() 后 layer_errors 应含 L0/L1/L2 三个有限值。"""
        m = model_with_all_upgrades
        obs = np.random.default_rng(1).standard_normal(16)
        signal = m.think(obs)
        collector = SnapshotCollector()
        snap = collector.collect(
            model=m, step=0, modality="physics",
            metadata=getattr(signal, "metadata", None),
        )
        d = snap.to_dict()
        assert "L0" in d["layer_errors"]
        assert "L1" in d["layer_errors"]
        assert "L2" in d["layer_errors"]
        for v in d["layer_errors"].values():
            assert isinstance(v, float)
            assert np.isfinite(v)

    def test_memory_retrieved_populated_after_think(self, model_with_all_upgrades):
        """think() 后 memory_retrieved 应含 n_memories 和 top1_similarity。"""
        m = model_with_all_upgrades
        obs = np.random.default_rng(2).standard_normal(16)
        signal = m.think(obs)
        collector = SnapshotCollector()
        snap = collector.collect(
            model=m, step=0, modality="physics",
            metadata=getattr(signal, "metadata", None),
        )
        d = snap.to_dict()
        assert d["memory_retrieved"]["n_memories"] == 2
        assert isinstance(d["memory_retrieved"]["top1_similarity"], float)
        # 查询接近 ±1 的存储模式，相似度应较高
        assert d["memory_retrieved"]["top1_similarity"] > 0.5

    def test_full_snapshot_json_serializable(self, model_with_all_upgrades):
        """端到端快照必须可被 json.dumps 序列化（前端可消费）。"""
        m = model_with_all_upgrades
        obs = np.random.default_rng(3).standard_normal(16)
        signal = m.think(obs)
        collector = SnapshotCollector()
        snap = collector.collect(
            model=m, step=0, modality="physics",
            metadata=getattr(signal, "metadata", None),
        )
        # 不应抛出 TypeError
        json.dumps(snap.to_dict())

    def test_metadata_arg_overrides_s4_cache_fallback(self, model_with_all_upgrades):
        """metadata 优先级高于 model.s4_state_cache 回退。"""
        m = model_with_all_upgrades
        # 先跑一次 think() 让 s4_state_cache 被填充
        obs = np.random.default_rng(4).standard_normal(16)
        signal = m.think(obs)
        cached = m.s4_state_cache
        assert cached is not None and len(cached) > 0
        # 现在传入一个 metadata，其 s4_state 与 cache 不同
        override_metadata = {
            "cognitive_upgrades": {"s4_state": [0.999, -0.999]},
        }
        collector = SnapshotCollector()
        snap = collector.collect(
            model=m, step=0, modality="physics",
            metadata=override_metadata,
        )
        d = snap.to_dict()
        # metadata 中的值应优先
        assert d["s4_state"] == [0.999, -0.999]


# --------------------------------------------------------------------------- #
# 4.4.5  跨快照一致性：连续 collect 不互相干扰
# --------------------------------------------------------------------------- #


class TestMultiSnapshotConsistency:
    def test_collect_multiple_cycles_independent(self):
        """连续多次 collect 应产生独立的 Phase G 字段。"""
        m = ZeroDataModel(dim=16, seed=42, use_s4=True, use_pcn=True, use_hopfield=True)
        m.hopfield_memory.store(np.ones(16), np.ones(16))
        collector = SnapshotCollector()
        rng = np.random.default_rng(99)
        s4_lens = []
        for i in range(5):
            obs = rng.standard_normal(16)
            signal = m.think(obs)
            snap = collector.collect(
                model=m, step=i, modality="physics",
                metadata=getattr(signal, "metadata", None),
            )
            d = snap.to_dict()
            s4_lens.append(len(d["s4_state"]))
        # 每次 s4_state 都应非空（S4 层一旦初始化就持续维护隐状态）
        assert all(n > 0 for n in s4_lens)
        # history 应累积 5 条
        assert collector.history_size == 5
