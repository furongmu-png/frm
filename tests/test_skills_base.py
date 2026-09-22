"""技能基类（SkillBase）单元测试：节流机制与异常隔离。

覆盖：
- ``process_interval`` 节流：非刷新步返回缓存，刷新步真正执行
- 首次调用必刷新（即使 interval > 1），保证有数据
- 禁用技能不调用 process
- safe_process 异常隔离
"""
from __future__ import annotations

import numpy as np

from skills.base import SkillBase, SkillContext, SkillResult


class _CountingSkill(SkillBase):
    """记录 process() 真正执行次数的测试技能。"""

    name = "counting_skill"
    dimension = "meta"

    def __init__(self, *, interval: int = 1) -> None:
        super().__init__(enabled=True)
        self.process_interval = interval
        self.call_count = 0

    def process(self, ctx: SkillContext) -> SkillResult:
        self.call_count += 1
        return SkillResult(
            name=self.name,
            data={"step": ctx.step, "call": self.call_count},
        )


class _BrokenSkill(SkillBase):
    name = "broken_skill"
    dimension = "meta"

    def process(self, ctx: SkillContext) -> SkillResult:  # noqa: ARG002
        raise RuntimeError("boom")


def _ctx(step: int = 0) -> SkillContext:
    return SkillContext(
        belief=np.zeros(8),
        prediction_error=0.1,
        step=step,
    )


# ================================================================== #
# 节流机制
# ================================================================== #
class TestProcessThrottle:
    def test_interval_1_runs_every_step(self):
        skill = _CountingSkill(interval=1)
        for step in range(5):
            skill.safe_process(_ctx(step))
        assert skill.call_count == 5

    def test_interval_3_runs_only_on_refresh_steps(self):
        """interval=3：步 0/3/6 真正执行，其余返回缓存。"""
        skill = _CountingSkill(interval=3)
        for step in range(7):  # steps 0..6
            skill.safe_process(_ctx(step))
        # 刷新步：0, 3, 6 → 3 次真正执行
        assert skill.call_count == 3

    def test_first_call_always_refreshes(self):
        """首次调用（step=0）必刷新，保证有数据可返回。"""
        skill = _CountingSkill(interval=10)
        result = skill.safe_process(_ctx(0))
        assert skill.call_count == 1
        assert result.data["call"] == 1

    def test_non_refresh_step_returns_cached_result(self):
        """非刷新步返回缓存的 SkillResult（零计算）。"""
        skill = _CountingSkill(interval=3)
        r0 = skill.safe_process(_ctx(0))  # 刷新 → call=1
        r1 = skill.safe_process(_ctx(1))  # 缓存
        r2 = skill.safe_process(_ctx(2))  # 缓存
        assert skill.call_count == 1
        # 缓存返回的是同一个 SkillResult 对象
        assert r1 is r0
        assert r2 is r0
        # 缓存结果携带首次执行的 step
        assert r1.data["step"] == 0

    def test_refresh_step_re_executes(self):
        """刷新步真正重新执行，更新缓存。"""
        skill = _CountingSkill(interval=3)
        skill.safe_process(_ctx(0))  # call=1
        skill.safe_process(_ctx(1))  # cached
        r3 = skill.safe_process(_ctx(3))  # 刷新 → call=2
        assert skill.call_count == 2
        assert r3.data["call"] == 2


# ================================================================== #
# safe_process 异常隔离与禁用
# ================================================================== #
class TestSafeProcessIsolation:
    def test_disabled_skill_returns_disabled_result(self):
        skill = _CountingSkill()
        skill.enabled = False
        result = skill.safe_process(_ctx(0))
        assert result.enabled is False
        assert skill.call_count == 0

    def test_exception_is_captured_not_raised(self):
        skill = _BrokenSkill()
        result = skill.safe_process(_ctx(0))
        # 不抛异常，错误写入 result.error
        assert result.error is not None
        assert "RuntimeError" in result.error
        assert "boom" in result.error

    def test_snapshot_reports_ready_after_process(self):
        skill = _CountingSkill()
        assert skill.snapshot()["ready"] is False
        skill.safe_process(_ctx(0))
        assert skill.snapshot()["ready"] is True
        assert skill.snapshot()["data"]["call"] == 1

    def test_snapshot_reports_error_for_broken_skill(self):
        skill = _BrokenSkill()
        skill.safe_process(_ctx(0))
        snap = skill.snapshot()
        assert snap["ready"] is True
        assert "error" in snap
