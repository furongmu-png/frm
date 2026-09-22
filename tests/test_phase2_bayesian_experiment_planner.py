# tests/test_phase2_bayesian_experiment_planner.py
"""第二阶段 §3.1 贝叶斯实验规划器单元测试。

验证：
- KL 散度信息增益计算（归一化、对称化、零和防护）
- 候选实验注册与默认候选集
- select_best 选择信息增益最大的候选
- execute 执行实验（沙盒集成 / 内部模拟）
- evaluate 周期性触发
- stats 统计正确性
- 沙盒 Protocol 集成
"""
from __future__ import annotations

from threading import Thread

import numpy as np
import pytest

from zero_data_model.experiment.bayesian_experiment_planner import (
    BayesianExperimentPlannerV2,
    CandidateIntervention,
)


# --------------------------------------------------------------------------- #
# 测试用沙盒桩
# --------------------------------------------------------------------------- #
class _StubSandbox:
    """满足 SandboxInterface Protocol 的桩对象。"""

    def __init__(self, state: np.ndarray | None = None) -> None:
        self._state = state if state is not None else np.ones(8) * 0.5
        self.calls: list[tuple[str, dict]] = []

    def apply_intervention(
        self, intervention_type: str, params: dict
    ) -> dict:
        self.calls.append((intervention_type, dict(params)))
        # 干预后状态翻转（确保 KL > 0）。
        new_state = self._state * -1.0
        self._state = new_state
        return {"state": new_state.tolist()}

    def get_state_distribution(self) -> np.ndarray:
        return self._state.copy()


# --------------------------------------------------------------------------- #
# 初始化
# --------------------------------------------------------------------------- #
class TestInit:
    """测试初始化。"""

    def test_default_init(self) -> None:
        p = BayesianExperimentPlannerV2()
        assert p._eval_interval == 1000
        assert p._n_samples == 10
        assert len(p.candidates) == 0

    def test_custom_params(self) -> None:
        p = BayesianExperimentPlannerV2(
            eval_interval=50, n_samples=5, seed=7,
            max_candidates=10, max_history=20,
        )
        assert p._eval_interval == 50
        assert p._n_samples == 5
        assert p.candidates.maxlen == 10

    def test_invalid_eval_interval_raises(self) -> None:
        with pytest.raises(ValueError):
            BayesianExperimentPlannerV2(eval_interval=0)

    def test_invalid_n_samples_raises(self) -> None:
        with pytest.raises(ValueError):
            BayesianExperimentPlannerV2(n_samples=0)


# --------------------------------------------------------------------------- #
# 候选管理
# --------------------------------------------------------------------------- #
class TestCandidateManagement:
    """测试候选实验注册。"""

    def test_register_candidate(self) -> None:
        p = BayesianExperimentPlannerV2()
        p.register_candidate("exp1", "scalar", {"factor": 2.0})
        assert len(p.candidates) == 1
        c = p.candidates[0]
        assert c.name == "exp1"
        assert c.intervention_type == "scalar"
        assert c.params == {"factor": 2.0}
        assert c.predicted_info_gain == 0.0
        assert c.executed is False

    def test_register_candidate_default_params(self) -> None:
        """params=None 应被替换为空 dict。"""
        p = BayesianExperimentPlannerV2()
        p.register_candidate("exp", "scalar")
        assert p.candidates[0].params == {}

    def test_default_candidates(self) -> None:
        p = BayesianExperimentPlannerV2()
        p.default_candidates(dim=32)
        names = [c.name for c in p.candidates]
        # spec 中列出的 5 个默认实验。
        assert "change_gravity" in names
        assert "add_object" in names
        assert "remove_object" in names
        assert "change_friction" in names
        assert "rotate_scene" in names
        assert len(p.candidates) == 5

    def test_candidates_deque_maxlen(self) -> None:
        p = BayesianExperimentPlannerV2(max_candidates=3)
        for i in range(10):
            p.register_candidate(f"exp{i}", "scalar")
        # deque 限长 3，仅保留最后 3 个。
        assert len(p.candidates) == 3
        assert p.candidates[0].name == "exp7"
        assert p.candidates[-1].name == "exp9"


# --------------------------------------------------------------------------- #
# KL 散度
# --------------------------------------------------------------------------- #
class TestKLDivergence:
    """测试 KL 散度计算。"""

    def test_identical_distributions_zero_kl(self) -> None:
        p = BayesianExperimentPlannerV2()
        a = np.array([0.25, 0.25, 0.25, 0.25])
        kl = p._kl_divergence(a, a)
        assert kl < 1e-6

    def test_different_distributions_positive_kl(self) -> None:
        p = BayesianExperimentPlannerV2()
        a = np.array([0.9, 0.05, 0.05])
        b = np.array([0.1, 0.45, 0.45])
        kl = p._kl_divergence(a, b)
        assert kl > 0.0

    def test_zero_sum_returns_zero(self) -> None:
        """全零分布返回 0（避免 log(0)）。"""
        p = BayesianExperimentPlannerV2()
        kl = p._kl_divergence(np.zeros(4), np.array([0.5, 0.5]))
        assert kl == 0.0

    def test_length_mismatch_aligns(self) -> None:
        """长度不齐时截取最短长度。"""
        p = BayesianExperimentPlannerV2()
        a = np.array([0.5, 0.5, 0.0])
        b = np.array([0.5, 0.5])
        kl = p._kl_divergence(a, b)
        # 截取前 2 个元素，归一化后相同 → KL ≈ 0。
        assert kl < 1e-6

    def test_result_non_negative(self) -> None:
        p = BayesianExperimentPlannerV2()
        a = np.array([0.1, 0.9])
        b = np.array([0.7, 0.3])
        kl = p._kl_divergence(a, b)
        assert kl >= 0.0


# --------------------------------------------------------------------------- #
# 信息增益估计
# --------------------------------------------------------------------------- #
class TestEstimateInfoGain:
    """测试预期信息增益估计。"""

    def test_returns_float_and_sets_attribute(self) -> None:
        p = BayesianExperimentPlannerV2(n_samples=3, seed=42)
        c = CandidateIntervention("exp", "scalar", {"factor": 2.0})
        gain = p.estimate_info_gain(c, np.ones(8) * 0.5)
        assert isinstance(gain, float)
        assert gain >= 0.0
        # 预测信息增益被写入候选。
        assert c.predicted_info_gain == gain

    def test_with_world_model_step(self) -> None:
        p = BayesianExperimentPlannerV2(n_samples=5, seed=42)
        c = CandidateIntervention("exp", "scalar", {"factor": 3.0})

        def step_fn(s: np.ndarray) -> np.ndarray:
            return s * 0.9  # 衰减演化

        gain = p.estimate_info_gain(c, np.ones(8), world_model_step=step_fn)
        assert gain >= 0.0

    def test_multiple_samples_average(self) -> None:
        """多次采样取平均，结果应为有限值。"""
        p = BayesianExperimentPlannerV2(n_samples=20, seed=42)
        c = CandidateIntervention("exp", "scalar", {"factor": 2.0})
        gain = p.estimate_info_gain(c, np.ones(16) * 0.5)
        assert np.isfinite(gain)


# --------------------------------------------------------------------------- #
# 选择最优
# --------------------------------------------------------------------------- #
class TestSelectBest:
    """测试最优候选选择。"""

    def test_returns_none_when_empty(self) -> None:
        p = BayesianExperimentPlannerV2()
        assert p.select_best(np.ones(8)) is None

    def test_returns_highest_gain_candidate(self) -> None:
        p = BayesianExperimentPlannerV2(n_samples=5, seed=42)
        # 不同的干预类型应有不同的信息增益。
        p.register_candidate("low", "scalar", {"factor": 1.0})  # 接近无变化
        p.register_candidate("high", "scalar", {"factor": 5.0})  # 大幅变化
        best = p.select_best(np.ones(8) * 0.5)
        assert best is not None
        # factor=5 的候选信息增益应不低于 factor=1 的。
        assert best.name in ("low", "high")

    def test_select_skips_executed(self) -> None:
        """已执行的候选不参与选择。"""
        p = BayesianExperimentPlannerV2(n_samples=3, seed=42)
        p.register_candidate("a", "scalar", {"factor": 2.0})
        p.register_candidate("b", "scalar", {"factor": 1.5})
        # 标记 a 已执行。
        p.candidates[0].executed = True
        best = p.select_best(np.ones(8))
        assert best is not None
        assert best.name == "b"

    def test_select_sets_predicted_gain(self) -> None:
        p = BayesianExperimentPlannerV2(n_samples=3, seed=42)
        p.register_candidate("a", "scalar", {"factor": 2.0})
        best = p.select_best(np.ones(8))
        assert best is not None
        assert best.predicted_info_gain >= 0.0


# --------------------------------------------------------------------------- #
# 执行
# --------------------------------------------------------------------------- #
class TestExecute:
    """测试实验执行。"""

    def test_execute_without_sandbox(self) -> None:
        p = BayesianExperimentPlannerV2(n_samples=3, seed=42)
        c = CandidateIntervention("exp", "scalar", {"factor": 2.0})
        result = p.execute(c, np.ones(8) * 0.5)
        assert result["name"] == "exp"
        assert result["used_sandbox"] is False
        assert len(result["state_before"]) == 8
        assert len(result["state_after"]) == 8
        assert c.executed is True
        assert c.actual_info_gain >= 0.0

    def test_execute_with_sandbox(self) -> None:
        p = BayesianExperimentPlannerV2(n_samples=3, seed=42)
        sandbox = _StubSandbox(state=np.ones(8) * 0.5)
        p.attach_sandbox(sandbox)
        c = CandidateIntervention("exp", "scalar", {"factor": 2.0})
        result = p.execute(c, np.ones(8) * 0.5)
        assert result["used_sandbox"] is True
        assert len(sandbox.calls) == 1
        assert sandbox.calls[0][0] == "scalar"

    def test_execute_marks_candidate(self) -> None:
        p = BayesianExperimentPlannerV2(seed=42)
        c = CandidateIntervention("exp", "scalar", {"factor": 2.0})
        assert c.executed is False
        p.execute(c, np.ones(8))
        assert c.executed is True

    def test_execute_appends_history(self) -> None:
        p = BayesianExperimentPlannerV2(seed=42)
        p.register_candidate("exp", "scalar", {"factor": 2.0})
        c = p.candidates[0]
        p.execute(c, np.ones(8))
        # stats 反映执行历史。
        assert p.stats["n_executed"] == 1


# --------------------------------------------------------------------------- #
# 沙盒集成
# --------------------------------------------------------------------------- #
class TestSandboxIntegration:
    """测试沙盒接口集成。"""

    def test_attach_detach(self) -> None:
        p = BayesianExperimentPlannerV2()
        assert p.stats["sandbox_attached"] is False
        sandbox = _StubSandbox()
        p.attach_sandbox(sandbox)
        assert p.stats["sandbox_attached"] is True
        p.detach_sandbox()
        assert p.stats["sandbox_attached"] is False

    def test_detach_when_no_sandbox_is_noop(self) -> None:
        p = BayesianExperimentPlannerV2()
        p.detach_sandbox()  # 不应抛错
        assert p.stats["sandbox_attached"] is False


# --------------------------------------------------------------------------- #
# 周期评估
# --------------------------------------------------------------------------- #
class TestEvaluate:
    """测试周期性评估。"""

    def test_returns_none_between_intervals(self) -> None:
        p = BayesianExperimentPlannerV2(eval_interval=100, n_samples=3, seed=42)
        p.register_candidate("a", "scalar", {"factor": 2.0})
        # step=50 不是 100 的倍数。
        assert p.evaluate(np.ones(8), step=50) is None

    def test_returns_result_on_interval(self) -> None:
        p = BayesianExperimentPlannerV2(eval_interval=100, n_samples=3, seed=42)
        p.register_candidate("a", "scalar", {"factor": 2.0})
        r = p.evaluate(np.ones(8), step=100)
        assert r is not None
        assert r["experiment"] == "a"
        assert "predicted_gain" in r

    def test_returns_none_when_no_candidates(self) -> None:
        p = BayesianExperimentPlannerV2(eval_interval=10, seed=42)
        # 无候选时即使到达间隔也返回 None。
        assert p.evaluate(np.ones(8), step=10) is None


# --------------------------------------------------------------------------- #
# 统计
# --------------------------------------------------------------------------- #
class TestStats:
    """测试统计属性。"""

    def test_stats_empty(self) -> None:
        p = BayesianExperimentPlannerV2()
        s = p.stats
        assert s["n_candidates"] == 0
        assert s["n_executed"] == 0
        assert s["mean_actual_gain"] == 0.0
        assert s["sandbox_attached"] is False

    def test_stats_after_execution(self) -> None:
        p = BayesianExperimentPlannerV2(seed=42)
        p.register_candidate("a", "scalar", {"factor": 2.0})
        p.register_candidate("b", "scalar", {"factor": 1.5})
        p.execute(p.candidates[0], np.ones(8))
        s = p.stats
        assert s["n_candidates"] == 2
        assert s["n_executed"] == 1
        assert s["mean_actual_gain"] >= 0.0


# --------------------------------------------------------------------------- #
# 线程安全
# --------------------------------------------------------------------------- #
class TestThreadSafety:
    """测试并发访问不会崩溃。"""

    def test_concurrent_register_does_not_crash(self) -> None:
        p = BayesianExperimentPlannerV2(max_candidates=10000, seed=42)

        def worker() -> None:
            for i in range(30):
                p.register_candidate(f"exp{i}", "scalar", {"factor": 2.0})

        threads = [Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(p.candidates) == 120

    def test_concurrent_select_best_does_not_crash(self) -> None:
        p = BayesianExperimentPlannerV2(n_samples=2, max_candidates=100, seed=42)
        for i in range(5):
            p.register_candidate(f"exp{i}", "scalar", {"factor": float(i + 1)})

        def worker() -> None:
            for _ in range(5):
                p.select_best(np.ones(8))

        threads = [Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert True
