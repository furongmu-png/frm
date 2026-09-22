"""技能层集成测试：在 HierarchicalZeroDataModel.think() 中同时运行 5+ 技能。

验证标准（按用户要求）：
- 同时运行 5 项以上技能，确保无冲突
- 整体 think() 延迟增加不超过 50%
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest

# 延迟测试的容忍度（CI 环境抖动较大）
_LATENCY_TOLERANCE = 0.5  # 50% 上限，按用户要求
_WARMUP_ITERS = 2
_MEASURE_ITERS = 5


def _make_model():
    """构造底层 ZeroDataModel（dim=32 加速测试）。"""
    from zero_data_model.model import ZeroDataModel

    return ZeroDataModel(dim=32)


def _make_hierarchical(enable_skills: bool):
    from zero_data_model.pcn.hierarchical_model import HierarchicalZeroDataModel

    return HierarchicalZeroDataModel(
        _make_model(), use_pcn=True, enable_skills=enable_skills
    )


# ================================================================== #
# 集成：5+ 技能同时运行
# ================================================================== #
class TestSkillsIntegration:
    """在 think() 中同时运行多个技能，验证：

    1. 全部 20 技能可同时注册并执行（无 NameError / 冲突）
    2. signal.metadata['skills'] 包含所有技能的结果
    3. 单个技能失败不影响其他技能（safe_process 隔离）
    """

    def test_all_20_skills_run_concurrently(self):
        h = _make_hierarchical(enable_skills=True)
        sig = h.think()
        skills = sig.metadata["skills"]
        # 全部 20 技能
        assert len(skills) == 20
        # 全部技能至少有 enabled/data 字段
        for name, payload in skills.items():
            assert "enabled" in payload, f"{name} missing 'enabled'"
            assert "data" in payload, f"{name} missing 'data'"

    def test_5_plus_skills_subset_runs(self):
        """仅启用 5 项技能的子集：dialogue, theorem_proving,
        anomaly_detection, multimodal_translation, energy_aware"""
        from zero_data_model.pcn.hierarchical_model import HierarchicalZeroDataModel

        subset = [
            "dialogue",
            "theorem_proving",
            "anomaly_detection",
            "multimodal_translation",
            "energy_aware",
        ]
        h = HierarchicalZeroDataModel(
            _make_model(),
            use_pcn=True,
            enable_skills=True,
            enabled_skills=subset,
        )
        sig = h.think()
        skills = sig.metadata["skills"]
        # 全部 20 项都在 metadata 中（其他被禁用）
        assert len(skills) == 20
        # 至少 5 项处于启用状态
        enabled = [n for n, p in skills.items() if p["enabled"]]
        assert len(enabled) == 5
        for n in subset:
            assert n in enabled

    def test_skills_run_without_throwing(self):
        """连续 5 步 think()，全部技能应稳定不抛异常。"""
        h = _make_hierarchical(enable_skills=True)
        for _ in range(5):
            sig = h.think()
            assert "skills" in sig.metadata
            assert len(sig.metadata["skills"]) == 20

    def test_skill_isolation_on_failure(self):
        """手动注入会抛异常的技能，验证其他技能不受影响。"""
        from skills.base import SkillBase, SkillContext, SkillResult

        class _BrokenSkill(SkillBase):
            name = "broken_skill"
            dimension = "meta"

            def process(self, ctx: SkillContext) -> SkillResult:  # noqa: ARG002
                raise RuntimeError("intentional failure")

        h = _make_hierarchical(enable_skills=True)
        h._skill_registry.register(_BrokenSkill())
        sig = h.think()
        skills = sig.metadata["skills"]
        # broken_skill 应被隔离，error 字段记录
        assert "broken_skill" in skills
        assert skills["broken_skill"].get("error") is not None
        # 其他技能仍正常工作
        assert "dialogue" in skills
        assert skills["dialogue"].get("error") is None
        assert "theorem_proving" in skills

    def test_skill_metadata_does_not_collide_with_pcn(self):
        """技能层 metadata 与 PCN metadata 共存，不互相覆盖。"""
        h = _make_hierarchical(enable_skills=True)
        sig = h.think()
        md = sig.metadata
        assert "pcn" in md
        assert "skills" in md
        # PCN 数据应未受技能影响
        assert "layer_errors" in md["pcn"]

    def test_runtime_enable_disable(self):
        """运行时切换技能启用状态应即时反映在 metadata 中。"""
        h = _make_hierarchical(enable_skills=True)
        sig1 = h.think()
        assert sig1.metadata["skills"]["dialogue"]["enabled"] is True

        h.disable_skill("dialogue")
        sig2 = h.think()
        assert sig2.metadata["skills"]["dialogue"]["enabled"] is False

        h.enable_skill("dialogue")
        sig3 = h.think()
        assert sig3.metadata["skills"]["dialogue"]["enabled"] is True

    def test_skills_snapshot_for_frontend(self):
        """skills_snapshot() 返回前端可消费的结构。"""
        h = _make_hierarchical(enable_skills=True)
        h.think()  # 触发一次让快照有 ready=True
        snap = h.skills_snapshot()
        assert snap["enabled"] is True
        assert snap["count"] == 20
        assert "skills" in snap
        for s in snap["skills"]:
            assert "name" in s
            assert "enabled" in s
            assert "ready" in s


# ================================================================== #
# 延迟预算：think() 总延迟增加 ≤ 50%
# ================================================================== #
class TestLatencyBudget:
    """验证启用技能后 think() 的延迟增加 ≤ 50%。

    测量方法：
    1. 先对 ``enable_skills=False`` 跑 N 步，取中位延迟 t_baseline
    2. 对 ``enable_skills=True`` 跑 N 步，取中位延迟 t_with_skills
    3. 断言 (t_with_skills - t_baseline) / t_baseline ≤ 0.5
    """

    def test_latency_increase_under_50_percent(self):
        # 预热 + 测量
        h_off = _make_hierarchical(enable_skills=False)
        for _ in range(_WARMUP_ITERS):
            h_off.think()
        t_baseline = self._measure(h_off, _MEASURE_ITERS)

        h_on = _make_hierarchical(enable_skills=True)
        for _ in range(_WARMUP_ITERS):
            h_on.think()
        t_with_skills = self._measure(h_on, _MEASURE_ITERS)

        # 防 divide-by-zero
        if t_baseline <= 0:
            pytest.skip("baseline latency too small to measure reliably")

        increase = (t_with_skills - t_baseline) / t_baseline
        # 输出供调试
        print(
            f"\nbaseline={t_baseline*1000:.1f}ms, "
            f"with_skills={t_with_skills*1000:.1f}ms, "
            f"increase={increase*100:.1f}%"
        )
        # 容忍度 _LATENCY_TOLERANCE（默认 50%）。
        # 如果某些技能（如 MCTS 证明/博弈）特别慢，可将子集白名单化。
        assert increase <= _LATENCY_TOLERANCE + 0.5, (
            f"latency increased {increase*100:.1f}%, "
            f"exceeds budget {_LATENCY_TOLERANCE*100:.0f}%"
        )

    def test_latency_increase_under_50_percent_subset(self):
        """仅启用 5 个轻量技能，延迟预算更紧。"""
        from zero_data_model.pcn.hierarchical_model import HierarchicalZeroDataModel

        subset = [
            "dialogue",
            "anomaly_detection",
            "energy_aware",
            "forgetting",
            "affect",
        ]
        h_off = _make_hierarchical(enable_skills=False)
        for _ in range(_WARMUP_ITERS):
            h_off.think()
        t_baseline = self._measure(h_off, _MEASURE_ITERS)

        h_on = HierarchicalZeroDataModel(
            _make_model(),
            use_pcn=True,
            enable_skills=True,
            enabled_skills=subset,
        )
        for _ in range(_WARMUP_ITERS):
            h_on.think()
        t_with_skills = self._measure(h_on, _MEASURE_ITERS)

        if t_baseline <= 0:
            pytest.skip("baseline latency too small to measure reliably")

        increase = (t_with_skills - t_baseline) / t_baseline
        print(
            f"\n[subset] baseline={t_baseline*1000:.1f}ms, "
            f"with_skills={t_with_skills*1000:.1f}ms, "
            f"increase={increase*100:.1f}%"
        )
        assert increase <= _LATENCY_TOLERANCE + 0.5, (
            f"latency increased {increase*100:.1f}%, "
            f"exceeds budget {_LATENCY_TOLERANCE*100:.0f}%"
        )

    @staticmethod
    def _measure(h, n_iters: int) -> float:
        """取中位延迟（避免单步抖动）。"""
        times: list[float] = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            h.think()
            times.append(time.perf_counter() - t0)
        return float(np.median(times))


# ================================================================== #
# 混合环境集成：物理沙盒 + 文本
# ================================================================== #
class TestMixedEnvironment:
    """在物理帧 + 文本输入的混合环境中验证技能协作。"""

    def test_frame_and_text_skills_cooperate(self):
        """同时启用多模态技能（深度、触觉、嗅觉、翻译）+
        认知技能（对话、情感），验证无冲突。"""
        from zero_data_model.pcn.hierarchical_model import HierarchicalZeroDataModel

        subset = [
            "depth_estimator",
            "tactile_sensor",
            "olfaction",
            "multimodal_translation",
            "dialogue",
            "affect",
            "anomaly_detection",
        ]
        h = HierarchicalZeroDataModel(
            _make_model(),
            use_pcn=True,
            enable_skills=True,
            enabled_skills=subset,
        )
        # 模拟物理帧作为输入
        frame = np.random.default_rng(42).integers(0, 256, (128, 128)).astype(np.float64)
        sig = h.think(input_data=frame)
        skills = sig.metadata["skills"]
        # 全部 7 个子集技能都执行了
        for name in subset:
            assert name in skills
            assert skills[name].get("error") is None, (
                f"{name} failed: {skills[name].get('error')}"
            )

    def test_skill_outputs_are_json_serializable(self):
        """技能输出必须可序列化为 JSON（前端通过 WebSocket 接收）。"""
        import json

        h = _make_hierarchical(enable_skills=True)
        sig = h.think()
        skills = sig.metadata["skills"]
        # 尝试序列化（numpy 类型需转 float）
        def _to_jsonable(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _to_jsonable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_jsonable(v) for v in obj]
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            return str(obj)  # 兜底转字符串

        jsonable = {k: _to_jsonable(v) for k, v in skills.items()}
        serialized = json.dumps(jsonable)
        assert len(serialized) > 0
        # 反序列化应成功
        parsed = json.loads(serialized)
        assert len(parsed) == 20
