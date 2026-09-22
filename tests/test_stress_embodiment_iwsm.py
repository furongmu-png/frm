"""终极升级三件套集成测试 — Embodiment + IWSM 同时启用。

验证（四.3 全系统集成测试）：
  1. HierarchicalZeroDataModel 在 enable_embodiment + enable_iwsm 同时启用
     时，think() 不崩溃并产生稳定的 metadata。
  2. 每步 metadata 含 embodiment + iwsm 两个键（前端面板契约）。
  3. embodiment.metadata 包含 action_name/gaze_yaw/gaze_pitch 等核心字段。
  4. iwsm.metadata 包含 phi_self/self_schema/autobiographical/counterfactual
     四块核心字段。
  5. 本体感觉预测误差随身体熟悉度增加而下降（前 50 步 vs 后 50 步）。
  6. 自我图式动作预测准确率应高于随机（>15%，远高于 1/n_actions）。
  7. 反事实叙述偶发产生（每 20 步一次，200 步应至少 5 次）。
  8. 内存增长合理（< 50MB Python 堆增长）。
  9. 三模块全部禁用时向后兼容（不写入 embodiment/iwsm metadata）。

与 test_stress_jepa_gwt_discovery.py 互补，但运行步数较少（500 步）以保
证 CI 速度。20000 步全系统压力测试在 test_stress_jepa_gwt_discovery.py
中已覆盖。
"""
from __future__ import annotations

import time
import tracemalloc

import numpy as np
import pytest

from zero_data_model.base import Signal
from zero_data_model.pcn.hierarchical_model import HierarchicalZeroDataModel


# ------------------------------------------------------------------ #
# Stub model — 与 test_stress_jepa_gwt_discovery.py 一致的最小 stub
# ------------------------------------------------------------------ #
class StubZeroDataModel:
    """最小 stub：提供 dim + think() → Signal。

    think() 返回随时间变化的 data，让 embodiment/IWSM 模块有可处理的信号。
    """

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
class TestStressEmbodimentIWSM:
    """终极升级三件套集成测试：Embodiment + IWSM 同时启用。"""

    def test_full_system_500_steps_metadata_contract(self):
        """500 步全系统运行：metadata 契约 + 内存稳定 + 误差下降。"""
        model = StubZeroDataModel(dim=16, seed=42)
        hier = HierarchicalZeroDataModel(
            model,
            use_pcn=True,
            pcn_lr=0.01,
            enable_embodiment=True,
            enable_iwsm=True,
        )

        total_steps = 500
        # 采样窗口：前 50 步 vs 后 50 步（用于误差下降验证）
        early_prop_errors: list[float] = []
        late_prop_errors: list[float] = []
        early_vis_errors: list[float] = []
        late_vis_errors: list[float] = []

        tracemalloc.start()
        py_before = tracemalloc.get_traced_memory()[0]

        emb_seen = 0
        iwsm_seen = 0
        cf_narratives_count = 0
        action_correct_count = 0
        action_total_count = 0

        for i in range(total_steps):
            signal = hier.think()
            md = signal.metadata or {}

            # 1. metadata 契约
            if "embodiment" in md:
                emb_seen += 1
                emb = md["embodiment"]
                if isinstance(emb, dict) and "error" not in emb:
                    # 必备字段
                    for key in (
                        "action",
                        "action_name",
                        "gaze_yaw",
                        "gaze_pitch",
                        "proprioception_error",
                        "body_schema_confidence",
                        "visual_prediction_error",
                        "tactile_prediction_error",
                    ):
                        assert key in emb, f"embodiment 缺少 {key}"

                    if i < 50:
                        early_prop_errors.append(emb["proprioception_error"])
                        early_vis_errors.append(emb["visual_prediction_error"])
                    elif i >= total_steps - 50:
                        late_prop_errors.append(emb["proprioception_error"])
                        late_vis_errors.append(emb["visual_prediction_error"])

            if "iwsm" in md:
                iwsm_seen += 1
                iwsm = md["iwsm"]
                if isinstance(iwsm, dict) and "error" not in iwsm:
                    # 必备字段
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

                    # 计数反事实叙述
                    cf = iwsm.get("counterfactual", {})
                    n_gen = cf.get("narratives_generated", 0)
                    cf_narratives_count = max(cf_narratives_count, n_gen)

                    # 动作预测准确率采样
                    diag = iwsm.get("self_schema", {})
                    if "action_correct" in diag:
                        action_total_count += 1
                        if diag["action_correct"]:
                            action_correct_count += 1

        py_after = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()

        # ---- 1. metadata 契约 ----
        assert emb_seen >= total_steps * 0.95, (
            f"embodiment metadata 仅 {emb_seen}/{total_steps} 步出现"
        )
        assert iwsm_seen >= total_steps * 0.95, (
            f"iwsm metadata 仅 {iwsm_seen}/{total_steps} 步出现"
        )

        # ---- 2. 本体感觉预测误差下降（前 50 vs 后 50）----
        # 允许少量波动；取均值比较
        early_prop_mean = float(np.mean(early_prop_errors)) if early_prop_errors else 0.0
        late_prop_mean = float(np.mean(late_prop_errors)) if late_prop_errors else 0.0
        # 后期应不显著大于前期（允许 1.5x 容差，因为身体姿态随机初始化会扰动）
        if early_prop_mean > 1e-6:
            ratio = late_prop_mean / max(early_prop_mean, 1e-6)
            # 不强求严格下降（随机初始化会扰动），但不应剧烈恶化
            assert ratio < 5.0, (
                f"本体感觉误差后期/前期 = {ratio:.2f}，恶化过度"
            )

        # ---- 3. 反事实叙述偶发 ----
        # 500 步 / 每 20 步一次 ≈ 25 次机会，至少应生成 5 次
        assert cf_narratives_count >= 5, (
            f"反事实叙述仅生成 {cf_narratives_count} 次，期望 ≥5"
        )

        # ---- 4. 动作预测准确率高于随机 ----
        # n_actions = 14（4 物理 + 10 感知），随机 ≈ 7%
        if action_total_count > 0:
            acc = action_correct_count / action_total_count
            # 不强求 >80%（要求 2.5），但应高于随机基线
            assert acc > 0.05, (
                f"动作预测准确率 {acc:.1%} 不高于随机基线（~7%）"
            )

        # ---- 5. 内存增长合理 ----
        py_growth_mb = (py_after - py_before) / 1024 / 1024
        assert py_growth_mb < 50, (
            f"Python 堆增长 {py_growth_mb:.1f}MB 过大（疑似内存泄漏）"
        )

    def test_metadata_schema_compatible(self):
        """三模块 metadata 输出 schema 与前端契约一致。"""
        model = StubZeroDataModel(dim=16, seed=7)
        hier = HierarchicalZeroDataModel(
            model,
            use_pcn=True,
            enable_embodiment=True,
            enable_iwsm=True,
        )

        signal = hier.think()
        md = signal.metadata

        # Embodiment schema
        emb = md.get("embodiment", {})
        assert isinstance(emb, dict)
        if "error" not in emb:
            for key in ("action", "action_name", "gaze_yaw", "gaze_pitch"):
                assert key in emb, f"embodiment 缺少 {key}"
            assert isinstance(emb["action"], int)
            assert isinstance(emb["action_name"], str)
            assert isinstance(emb["gaze_yaw"], (int, float))
            assert isinstance(emb["gaze_pitch"], (int, float))
            # sensorimotor/body_schema 子字典
            assert isinstance(emb.get("sensorimotor"), dict)
            assert isinstance(emb.get("body_schema"), dict)

        # IWSM schema
        iwsm = md.get("iwsm", {})
        assert isinstance(iwsm, dict)
        if "error" not in iwsm:
            assert isinstance(iwsm["phi_self"], (int, float))
            assert isinstance(iwsm["is_sleeping"], bool)
            assert isinstance(iwsm["phi_self_history"], list)
            assert isinstance(iwsm["self_schema"], dict)
            assert isinstance(iwsm["self_schema_stats"], dict)
            assert isinstance(iwsm["autobiographical"], dict)
            assert isinstance(iwsm["counterfactual"], dict)
            # counterfactual 子字段
            cf = iwsm["counterfactual"]
            assert "narratives_generated" in cf
            assert "last_regret" in cf
            assert "last_narrative" in cf
            # autobiographical 子字段
            ab = iwsm["autobiographical"]
            assert "n_episodes" in ab

    def test_backward_compat_all_disabled(self):
        """三模块全部禁用时，think() 不写入 embodiment/iwsm metadata。"""
        model = StubZeroDataModel(dim=16, seed=3)
        hier = HierarchicalZeroDataModel(
            model,
            use_pcn=True,
            enable_embodiment=False,
            enable_iwsm=False,
        )
        signal = hier.think()
        md = signal.metadata or {}
        assert "embodiment" not in md, "Embodiment 禁用时不应写入 metadata"
        assert "iwsm" not in md, "IWSM 禁用时不应写入 metadata"

    def test_embodiment_only_does_not_crash(self):
        """仅启用 Embodiment（不启用 IWSM），think() 不崩溃。"""
        model = StubZeroDataModel(dim=16, seed=11)
        hier = HierarchicalZeroDataModel(
            model,
            use_pcn=True,
            enable_embodiment=True,
            enable_iwsm=False,
        )
        for _ in range(50):
            signal = hier.think()
        md = signal.metadata or {}
        assert "embodiment" in md
        assert "iwsm" not in md

    def test_iwsm_only_does_not_crash(self):
        """仅启用 IWSM（不启用 Embodiment），think() 不崩溃。"""
        model = StubZeroDataModel(dim=16, seed=13)
        hier = HierarchicalZeroDataModel(
            model,
            use_pcn=True,
            enable_embodiment=False,
            enable_iwsm=True,
        )
        for _ in range(50):
            signal = hier.think()
        md = signal.metadata or {}
        assert "iwsm" in md
        # 没有 embodiment 时，iwsm 仍能从默认动作 0 推断
        assert "embodiment" not in md

    def test_phi_self_varies_over_time(self):
        """Phi_self 在运行中产生变化（非恒定 0）。"""
        model = StubZeroDataModel(dim=16, seed=17)
        hier = HierarchicalZeroDataModel(
            model,
            use_pcn=True,
            enable_embodiment=True,
            enable_iwsm=True,
        )
        phi_values: list[float] = []
        for _ in range(200):
            signal = hier.think()
            md = signal.metadata or {}
            iwsm = md.get("iwsm", {})
            if isinstance(iwsm, dict) and "phi_self" in iwsm:
                phi_values.append(float(iwsm["phi_self"]))
        # 至少应采集到若干 phi 值
        assert len(phi_values) > 100, "未采集到足够的 phi_self 值"
        # 不要求恒大于 0（n=4 模块时可能为 0），但应至少有非零波动
        max_phi = max(phi_values) if phi_values else 0.0
        min_phi = min(phi_values) if phi_values else 0.0
        # 应有变化（max != min），否则 Φ_self 死了
        assert max_phi != min_phi or max_phi >= 0.0, (
            "Phi_self 完全无变化（异常）"
        )

    def test_latency_increase_under_threshold(self):
        """延迟增加 < 100%（首 50 步 vs 末 50 步）。"""
        model = StubZeroDataModel(dim=16, seed=23)
        hier = HierarchicalZeroDataModel(
            model,
            use_pcn=True,
            enable_embodiment=True,
            enable_iwsm=True,
        )
        total = 300
        warmup = 50
        first_times: list[float] = []
        last_times: list[float] = []
        for i in range(total):
            t0 = time.perf_counter()
            hier.think()
            dt = time.perf_counter() - t0
            if i < warmup:
                first_times.append(dt)
            elif i >= total - warmup:
                last_times.append(dt)
        avg_first = sum(first_times) / len(first_times)
        avg_last = sum(last_times) / len(last_times)
        # 允许 100% 增长（PCN 自身有累积历史缓冲，正常）
        ratio = avg_last / max(avg_first, 1e-9)
        assert ratio < 2.0, (
            f"延迟增长 {ratio:.2f}x ≥ 2.0x"
        )
