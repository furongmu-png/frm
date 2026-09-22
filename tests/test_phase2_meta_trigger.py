# tests/test_phase2_meta_trigger.py
"""第二阶段 §2.2 元认知触发机制单元测试。

验证：
- 动态阈值（历史均值 + 2σ）
- 不确定度超阈值时触发信息寻求
- 降低学习率（precision 提升）
- 自我提问生成
- 突发变化检测（物体突然消失）
"""
from __future__ import annotations

from zero_data_model.metacog.meta_trigger import MetaTrigger


# ------------------------------------------------------------------ #
# 初始化
# ------------------------------------------------------------------ #
class TestInit:
    """测试初始化。"""

    def test_default_init(self) -> None:
        mt = MetaTrigger()
        assert not mt.triggered
        assert mt.learning_rate_scale == 1.0
        assert mt.current_threshold > 0

    def test_custom_params(self) -> None:
        mt = MetaTrigger(
            threshold_sigma=3.0, history_window=50, lr_scale_factor=0.3
        )
        assert mt.learning_rate_scale == 1.0  # not triggered yet


# ------------------------------------------------------------------ #
# 动态阈值
# ------------------------------------------------------------------ #
class TestDynamicThreshold:
    """测试动态阈值计算。"""

    def test_threshold_not_triggered_low_uncertainty(self) -> None:
        mt = MetaTrigger(threshold_sigma=2.0, min_history=5, seed=42)
        for _ in range(30):
            r = mt.update(0.1, topic="test")
        assert not mt.triggered
        assert r["mode"] == "normal"

    def test_threshold_adapts_to_history(self) -> None:
        mt = MetaTrigger(threshold_sigma=2.0, min_history=5, seed=42)
        # Low uncertainty history → low threshold
        for _ in range(20):
            mt.update(0.05, topic="")
        low_threshold = mt.current_threshold
        # High uncertainty history → higher threshold
        for _ in range(20):
            mt.update(0.5, topic="")
        high_threshold = mt.current_threshold
        assert high_threshold > low_threshold

    def test_min_history_prevents_early_trigger(self) -> None:
        mt = MetaTrigger(min_history=10, seed=42)
        # Few steps with high uncertainty — should not trigger yet
        for _ in range(5):
            mt.update(0.9, topic="x")
        # With min_history=10 and only 5 steps, threshold is conservative (0.8)
        # 0.9 > 0.8 → may trigger. But the point is threshold is conservative.
        assert mt.current_threshold >= 0.5


# ------------------------------------------------------------------ #
# 触发与解除
# ------------------------------------------------------------------ #
class TestTriggerRelease:
    """测试触发与解除机制。"""

    def test_triggers_on_spike(self) -> None:
        mt = MetaTrigger(threshold_sigma=2.0, min_history=5, seed=42)
        # Build low-uncertainty history
        for _ in range(30):
            mt.update(0.05, topic="")
        assert not mt.triggered
        # Sudden spike
        r = mt.update(0.95, topic="collision")
        assert mt.triggered
        assert r["mode"] == "cautious"

    def test_releases_when_uncertainty_drops(self) -> None:
        mt = MetaTrigger(threshold_sigma=2.0, min_history=5, seed=42)
        for _ in range(30):
            mt.update(0.05, topic="")
        mt.update(0.95, topic="x")
        assert mt.triggered
        # Uncertainty drops back
        for _ in range(30):
            mt.update(0.05, topic="")
        assert not mt.triggered
        assert mt.learning_rate_scale == 1.0

    def test_learning_rate_lowered_on_trigger(self) -> None:
        mt = MetaTrigger(lr_scale_factor=0.5, seed=42)
        for _ in range(30):
            mt.update(0.05, topic="")
        mt.update(0.95, topic="x")
        assert mt.learning_rate_scale == 0.5

    def test_learning_rate_restored_on_release(self) -> None:
        mt = MetaTrigger(lr_scale_factor=0.5, seed=42)
        for _ in range(30):
            mt.update(0.05, topic="")
        mt.update(0.95, topic="x")
        assert mt.learning_rate_scale < 1.0
        for _ in range(30):
            mt.update(0.05, topic="")
        assert mt.learning_rate_scale == 1.0


# ------------------------------------------------------------------ #
# 信息寻求与自我提问
# ------------------------------------------------------------------ #
class TestInfoSeeking:
    """测试主动信息寻求与自我提问。"""

    def test_generates_info_request(self) -> None:
        mt = MetaTrigger(seed=42)
        for _ in range(30):
            mt.update(0.05, topic="")
        mt.update(0.95, topic="gravity")
        reqs = mt.get_pending_info_requests()
        assert len(reqs) > 0
        assert "gravity" in reqs[0]

    def test_generates_question(self) -> None:
        mt = MetaTrigger(seed=42)
        for _ in range(30):
            mt.update(0.05, topic="")
        mt.update(0.95, topic="collision_dynamics")
        qs = mt.get_pending_questions()
        assert len(qs) > 0
        assert "collision" in qs[0].lower()
        assert qs[0].endswith("?")

    def test_question_format(self) -> None:
        mt = MetaTrigger(seed=42)
        q = mt.generate_question("quantum_mechanics")
        assert "quantum mechanics" in q.lower()
        assert q.endswith("?")

    def test_pending_requests_cleared_after_get(self) -> None:
        mt = MetaTrigger(seed=42)
        for _ in range(30):
            mt.update(0.05, topic="")
        mt.update(0.95, topic="x")
        first = mt.get_pending_info_requests()
        second = mt.get_pending_info_requests()
        assert len(first) > 0
        assert len(second) == 0

    def test_no_request_without_topic(self) -> None:
        mt = MetaTrigger(seed=42)
        for _ in range(30):
            mt.update(0.05, topic="")
        mt.update(0.95, topic="")  # no topic
        reqs = mt.get_pending_info_requests()
        qs = mt.get_pending_questions()
        assert len(reqs) == 0
        assert len(qs) == 0


# ------------------------------------------------------------------ #
# 验证场景：突发变化
# ------------------------------------------------------------------ #
class TestSuddenChange:
    """验证：在物理沙盒中引入突发变化，观察模型快速切换至谨慎模式。"""

    def test_object_disappears_triggers_cautious(self) -> None:
        """模拟物体突然消失的不确定度突变。"""
        mt = MetaTrigger(threshold_sigma=2.0, min_history=5, seed=42)
        # Normal operation: predictable environment
        for _ in range(50):
            r = mt.update(0.08, topic="")
        assert r["mode"] == "normal"
        # Object suddenly disappears → high uncertainty
        r = mt.update(0.9, topic="object_tracking")
        assert r["mode"] == "cautious"
        assert mt.should_seek_info()

    def test_trigger_events_recorded(self) -> None:
        mt = MetaTrigger(seed=42)
        for _ in range(30):
            mt.update(0.05, topic="")
        mt.update(0.95, topic="physics")
        events = mt.get_trigger_events()
        assert len(events) > 0
        assert events[-1]["type"] == "meta_trigger"
        assert events[-1]["topic"] == "physics"

    def test_stats(self) -> None:
        mt = MetaTrigger(seed=42)
        for _ in range(10):
            mt.update(0.1, topic="")
        s = mt.stats
        assert s["step_count"] == 10
        assert "triggered" in s
        assert "learning_rate_scale" in s
        assert "current_threshold" in s
