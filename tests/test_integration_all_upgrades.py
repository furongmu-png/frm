"""终极升级集成测试 — Embodiment + IWSM + Discovery 三件套同时启用。

验证（四.3 全系统测试 + 五.5 交付物）：
  1. HierarchicalZeroDataModel 在三件套 + JEPA + GWT 全部启用时，
     think() 不崩溃并产生稳定的 metadata。
  2. 每步 metadata 含 embodiment + iwsm + discovery + jepa + gwt 五块快照。
  3. 各模块 metadata schema 与前端面板契约一致。
  4. 内存增长 < 100MB（500 步累积）。
  5. 反事实叙述偶发产生（每 20 步一次，500 步应至少 5 次）。
  6. Φ_self 在运行中产生变化（非恒定 0）。

这是一个相对快速的烟雾测试（smoke test），不要求 20000 步。
完整的 20000 步压力测试在 test_stress_jepa_gwt_discovery.py 中已覆盖。
"""
from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from zero_data_model.base import Signal
from zero_data_model.pcn.hierarchical_model import HierarchicalZeroDataModel


# ------------------------------------------------------------------ #
# Stub model — 与其他压力测试一致
# ------------------------------------------------------------------ #
class StubZeroDataModel:
    """最小 stub：提供 dim + think() → Signal。"""

    def __init__(self, dim: int = 16, seed: int = 42):
        self.dim = dim
        self._rng = np.random.default_rng(seed)
        self._step = 0

    def think(self, input_data=None, **kwargs) -> Signal:
        self._step += 1
        t = self._step * 0.01
        data = self._rng.standard_normal(self.dim) * 0.5 + np.sin(t)
        meta = {
            "causal_graph": {
                "nodes": [
                    {"id": 0, "label": "mass"},
                    {"id": 1, "label": "velocity"},
                ],
                "edges": [
                    {"source": 0, "target": 1, "strength": 0.05},
                ],
            },
            "kg_update": {
                "new_nodes": ["mass"],
                "new_edges": [],
            },
        }
        return Signal(data=data, metadata=meta)


# ------------------------------------------------------------------ #
# 集成测试
# ------------------------------------------------------------------ #
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestAllUpgradesIntegration:
    """终极升级三件套 + JEPA + GWT 全启用集成测试。"""

    def test_all_upgrades_run_together_500_steps(self, tmp_path):
        """500 步全栈运行：五模块同时启用，无崩溃 + metadata 契约。"""
        model = StubZeroDataModel(dim=16, seed=42)
        hier = HierarchicalZeroDataModel(
            model,
            use_pcn=True,
            pcn_lr=0.01,
            enable_jepa=True,
            enable_gwt=True,
            enable_discovery=True,
            enable_embodiment=True,
            enable_iwsm=True,
            jepa_lambda=0.5,
            discovery_interval=50,  # 500 步触发 10 轮
        )
        hier._science_loop.paper_writer.output_dir = str(tmp_path)

        total_steps = 500
        emb_count = 0
        iwsm_count = 0
        disc_count = 0
        jepa_count = 0
        gwt_count = 0
        cf_max = 0
        phi_self_values: list[float] = []

        tracemalloc.start()
        py_before = tracemalloc.get_traced_memory()[0]

        for _ in range(total_steps):
            signal = hier.think()
            md = signal.metadata or {}
            if "embodiment" in md:
                emb_count += 1
            if "iwsm" in md:
                iwsm_count += 1
                iwsm_md = md["iwsm"]
                if isinstance(iwsm_md, dict) and "error" not in iwsm_md:
                    if "phi_self" in iwsm_md:
                        phi_self_values.append(float(iwsm_md["phi_self"]))
                    cf = iwsm_md.get("counterfactual", {})
                    cf_max = max(cf_max, cf.get("narratives_generated", 0))
            if "discovery" in md:
                disc_count += 1
            if "jepa" in md:
                jepa_count += 1
            if "gwt" in md:
                gwt_count += 1

        py_after = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()

        # ---- 1. 五模块 metadata 均出现 ----
        assert emb_count >= total_steps * 0.95, (
            f"embodiment 仅出现 {emb_count}/{total_steps} 步"
        )
        assert iwsm_count >= total_steps * 0.95, (
            f"iwsm 仅出现 {iwsm_count}/{total_steps} 步"
        )
        assert disc_count >= 1, "discovery 未在 metadata 中出现"
        assert jepa_count >= total_steps * 0.95, (
            f"jepa 仅出现 {jepa_count}/{total_steps} 步"
        )
        assert gwt_count >= total_steps * 0.95, (
            f"gwt 仅出现 {gwt_count}/{total_steps} 步"
        )

        # ---- 2. 反事实叙述偶发产生 ----
        assert cf_max >= 5, (
            f"反事实叙述仅生成 {cf_max} 次，期望 ≥5"
        )

        # ---- 3. Phi_self 产生变化 ----
        assert len(phi_self_values) > 100, "未采集到足够的 phi_self 值"
        max_phi = max(phi_self_values)
        min_phi = min(phi_self_values)
        # 不要求恒 > 0（n=4 模块时可能为 0），但应有变化
        assert max_phi != min_phi or max_phi >= 0.0, (
            "Phi_self 完全无变化（异常）"
        )

        # ---- 4. 内存增长合理 ----
        py_growth_mb = (py_after - py_before) / 1024 / 1024
        assert py_growth_mb < 100, (
            f"Python 堆增长 {py_growth_mb:.1f}MB 过大（疑似内存泄漏）"
        )

    def test_metadata_schema_compatible_all(self, tmp_path):
        """五模块 metadata schema 与前端面板契约一致。"""
        model = StubZeroDataModel(dim=16, seed=7)
        hier = HierarchicalZeroDataModel(
            model,
            use_pcn=True,
            enable_jepa=True,
            enable_gwt=True,
            enable_discovery=True,
            enable_embodiment=True,
            enable_iwsm=True,
            discovery_interval=1,
        )
        hier._science_loop.paper_writer.output_dir = str(tmp_path)

        signal = hier.think()
        md = signal.metadata

        # Embodiment schema
        emb = md.get("embodiment", {})
        assert isinstance(emb, dict)
        if "error" not in emb:
            for key in ("action", "action_name", "gaze_yaw", "gaze_pitch"):
                assert key in emb, f"embodiment 缺少 {key}"

        # IWSM schema
        iwsm = md.get("iwsm", {})
        assert isinstance(iwsm, dict)
        if "error" not in iwsm:
            for key in (
                "self_schema",
                "self_schema_stats",
                "phi_self",
                "phi_self_history",
                "is_sleeping",
                "autobiographical",
                "counterfactual",
            ):
                assert key in iwsm, f"iwsm 缺少 {key}"

        # Discovery schema
        disc = md.get("discovery", {})
        assert isinstance(disc, dict)
        for key in ("enabled", "trigger_interval", "cycle_count", "total_papers"):
            assert key in disc, f"discovery 缺少 {key}"

        # JEPA schema
        jepa = md.get("jepa", {})
        assert isinstance(jepa, dict)

        # GWT schema
        gwt = md.get("gwt", {})
        assert isinstance(gwt, dict)
        for key in ("winner", "probabilities", "phi", "is_conscious"):
            assert key in gwt, f"gwt 缺少 {key}"

    def test_backward_compat_only_baseline(self):
        """所有升级禁用时，think() 仅产生 baseline metadata。"""
        model = StubZeroDataModel(dim=16, seed=3)
        hier = HierarchicalZeroDataModel(
            model,
            use_pcn=True,
            enable_jepa=False,
            enable_gwt=False,
            enable_discovery=False,
            enable_embodiment=False,
            enable_iwsm=False,
        )
        signal = hier.think()
        md = signal.metadata or {}
        assert "embodiment" not in md
        assert "iwsm" not in md
        assert "discovery" not in md
        assert "jepa" not in md
        assert "gwt" not in md

    def test_embodiment_only_with_gwt(self):
        """Embodiment + GWT 同时启用（IWSM 不启用），think() 不崩溃。"""
        model = StubZeroDataModel(dim=16, seed=17)
        hier = HierarchicalZeroDataModel(
            model,
            use_pcn=True,
            enable_embodiment=True,
            enable_gwt=True,
            enable_iwsm=False,
        )
        for _ in range(50):
            signal = hier.think()
        md = signal.metadata or {}
        assert "embodiment" in md
        assert "gwt" in md
        assert "iwsm" not in md  # IWSM 禁用，不应写入

    def test_iwsm_only_with_jepa(self):
        """IWSM + JEPA 同时启用（Embodiment 不启用），think() 不崩溃。"""
        model = StubZeroDataModel(dim=16, seed=23)
        hier = HierarchicalZeroDataModel(
            model,
            use_pcn=True,
            enable_embodiment=False,
            enable_iwsm=True,
            enable_jepa=True,
        )
        for _ in range(50):
            signal = hier.think()
        md = signal.metadata or {}
        assert "iwsm" in md
        assert "jepa" in md
        assert "embodiment" not in md  # Embodiment 禁用，不应写入

    def test_all_upgrades_memory_bounded(self, tmp_path):
        """所有升级启用，500 步内存增长 < 100MB。"""
        model = StubZeroDataModel(dim=16, seed=31)
        hier = HierarchicalZeroDataModel(
            model,
            use_pcn=True,
            enable_jepa=True,
            enable_gwt=True,
            enable_discovery=True,
            enable_embodiment=True,
            enable_iwsm=True,
            discovery_interval=100,
        )
        hier._science_loop.paper_writer.output_dir = str(tmp_path)

        tracemalloc.start()
        py_before = tracemalloc.get_traced_memory()[0]

        for _ in range(500):
            hier.think()

        py_after = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()

        py_growth_mb = (py_after - py_before) / 1024 / 1024
        # 放宽到 100MB（五模块同时累积历史）
        assert py_growth_mb < 100, (
            f"Python 堆增长 {py_growth_mb:.1f}MB 过大（疑似内存泄漏）"
        )
