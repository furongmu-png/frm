"""元技能单元测试：学习如何学习、课程调度、遗忘管理、能耗感知。"""
from __future__ import annotations

import numpy as np
import pytest

from skills.base import SkillContext
from skills.meta.meta_learning import (
    MetaLearner,
    Hyperparameter,
    HyperparameterSpace,
    GaussianProcess,
    expected_improvement,
)
from skills.meta.curriculum import (
    CurriculumScheduler,
    CurriculumTask,
)
from skills.meta.forgetting import (
    MemoryDecay,
    MemoryItem,
)
from skills.meta.energy_aware import (
    EnergyMonitor,
    EnergyPolicy,
    _policy_for,
)


# ================================================================== #
# 17. MetaLearner
# ================================================================== #
class TestMetaLearner:
    def test_default_space(self):
        ml = MetaLearner()
        params = ml.space.default_values()
        assert "learning_rate" in params
        assert "beta_decay" in params
        assert "consolidation_freq" in params
        assert "exploration_beta" in params

    def test_observe_updates_best(self):
        ml = MetaLearner()
        ml.observe(ml.space.default_values(), loss=0.5)
        assert ml._best_y == 0.5
        assert ml._best_params is not None

    def test_observe_tracks_minimum(self):
        ml = MetaLearner()
        ml.observe(ml.space.default_values(), loss=0.8)
        ml.observe(ml.space.default_values(), loss=0.3)
        ml.observe(ml.space.default_values(), loss=0.6)
        assert ml._best_y == 0.3

    def test_suggest_returns_random_in_initial_phase(self):
        ml = MetaLearner(n_initial_random=5)
        s1 = ml.suggest()
        s2 = ml.suggest()
        # 前几次应该是随机不同的
        assert isinstance(s1, dict)
        assert set(s1.keys()) == set(ml.space.default_values().keys())

    def test_suggest_finds_optimal_after_observations(self):
        """通过若干观测后，应该能找到接近最优的超参数。"""
        ml = MetaLearner(n_initial_random=3)
        # 目标：learning_rate 接近 0.1 时损失最低
        def loss_fn(p):
            return (p["learning_rate"] - 0.1) ** 2 + 0.05
        for _ in range(15):
            p = ml.suggest()
            ml.observe(p, loss_fn(p))
        assert ml._best_y < 0.3  # 应该找到比随机好得多的点
        assert abs(ml._best_params["learning_rate"] - 0.1) < 0.2

    def test_process_via_context(self):
        ml = MetaLearner()
        ctx = SkillContext(belief=np.zeros(32), prediction_error=0.5)
        result = ml.safe_process(ctx)
        assert result.error is None
        assert "suggested_params" in result.data
        assert "n_observations" in result.data

    def test_snapshot(self):
        ml = MetaLearner()
        ml.observe(ml.space.default_values(), 0.5)
        snap = ml.snapshot()
        assert snap["ready"] is True
        assert snap["data"]["n_observations"] == 1
        assert snap["data"]["best_loss"] == 0.5

    # --- 内部组件 ---

    def test_hyperparameter_log_scale(self):
        p = Hyperparameter("lr", 1e-3, 1.0, 0.1, log_scale=True)
        # 默认值 0.1 应该映射到某个 [0, 1] 值
        u = p.to_unit(0.1)
        assert 0.0 <= u <= 1.0
        # 来回转换应该一致
        assert abs(p.from_unit(u) - 0.1) < 1e-9

    def test_hyperparameter_linear(self):
        p = Hyperparameter("beta", 0.0, 1.0, 0.5, log_scale=False)
        assert abs(p.to_unit(0.5) - 0.5) < 1e-9
        assert abs(p.from_unit(0.5) - 0.5) < 1e-9

    def test_space_to_unit_round_trip(self):
        space = HyperparameterSpace(
            [
                Hyperparameter("a", 0.0, 1.0, 0.5),
                Hyperparameter("b", -10.0, 10.0, 0.0),
            ]
        )
        vals = {"a": 0.7, "b": -3.0}
        u = space.to_unit(vals)
        recovered = space.from_unit(u)
        assert abs(recovered["a"] - 0.7) < 1e-9
        assert abs(recovered["b"] - (-3.0)) < 1e-9

    def test_gaussian_process_predicts_near_data(self):
        gp = GaussianProcess()
        X = np.array([[0.1, 0.2], [0.5, 0.5], [0.9, 0.8]])
        y = np.array([0.1, 0.5, 0.9])
        gp.fit(X, y)
        # 在已知点附近预测应该接近 y
        m, _ = gp.predict(np.array([0.5, 0.5]))
        assert abs(m - 0.5) < 0.2

    def test_gaussian_process_variance_decreases_with_data(self):
        gp = GaussianProcess()
        # 无数据 → 高方差
        _, v1 = gp.predict(np.array([0.5]))
        assert v1 == 1.0
        X = np.array([[0.1, 0.2], [0.5, 0.5], [0.9, 0.8]])
        y = np.array([0.1, 0.5, 0.9])
        gp.fit(X, y)
        # 在已知点附近 → 低方差
        _, v2 = gp.predict(np.array([0.5, 0.5]))
        assert v2 < v1

    def test_expected_improvement_returns_nonneg(self):
        gp = GaussianProcess()
        X = np.array([[0.1, 0.2], [0.5, 0.5], [0.9, 0.8]])
        y = np.array([0.1, 0.5, 0.9])
        gp.fit(X, y)
        candidates = np.array([[0.3, 0.3], [0.7, 0.7]])
        ei = expected_improvement(gp, candidates, best_y=0.1)
        assert ei.shape == (2,)
        assert np.all(ei >= 0)

    def test_expected_improvement_zero_at_observed_point(self):
        """已观测点处 EI 应接近 0。"""
        gp = GaussianProcess()
        X = np.array([[0.1, 0.2]])
        y = np.array([0.1])
        gp.fit(X, y)
        ei = expected_improvement(gp, X, best_y=0.1)
        # 由于 noise_var > 0，方差非 0，EI 不严格为 0，但应远小于 1
        assert ei[0] < 0.05

    def test_history_tracking(self):
        ml = MetaLearner()
        ml.observe(ml.space.default_values(), 0.5)
        ml.observe(ml.space.default_values(), 0.3)
        h = ml.history()
        assert len(h) == 2
        assert h[0]["loss"] == 0.5


# ================================================================== #
# 18. CurriculumScheduler
# ================================================================== #
class TestCurriculumScheduler:
    def test_add_task(self):
        cs = CurriculumScheduler()
        t = cs.add_task("t1", difficulty=0.5)
        assert t.task_id == "t1"
        assert t.difficulty == 0.5
        assert cs.stats()["n_tasks"] == 1

    def test_add_duplicate_raises(self):
        cs = CurriculumScheduler()
        cs.add_task("t1", 0.5)
        with pytest.raises(ValueError):
            cs.add_task("t1", 0.5)

    def test_record_attempt_updates_error(self):
        cs = CurriculumScheduler()
        cs.add_task("t1", 0.5)
        cs.record_attempt("t1", error=0.8)
        cs.record_attempt("t1", error=0.5)
        task = cs._tasks["t1"]
        assert task.last_error == 0.5
        assert task.n_attempts == 2

    def test_record_unknown_task_raises(self):
        cs = CurriculumScheduler()
        cs.add_task("t1", 0.5)
        with pytest.raises(KeyError):
            cs.record_attempt("missing", error=0.5)

    def test_learning_progress_positive_when_improving(self):
        cs = CurriculumScheduler()
        cs.add_task("t1", 0.5)
        # 误差递减 → LP > 0
        for e in [1.0, 0.8, 0.6, 0.4, 0.2]:
            cs.record_attempt("t1", e)
        assert cs._tasks["t1"].learning_progress() > 0.0

    def test_learning_progress_zero_with_insufficient_data(self):
        cs = CurriculumScheduler()
        cs.add_task("t1", 0.5)
        assert cs._tasks["t1"].learning_progress() == 0.0
        cs.record_attempt("t1", 0.5)
        # 只有一个样本 → LP=0
        assert cs._tasks["t1"].learning_progress() == 0.0

    def test_select_next_returns_existing_task(self):
        cs = CurriculumScheduler(exploration_rate=0.0)
        cs.add_task("t1", 0.1)
        cs.add_task("t2", 0.5)
        # 没有 LP → 难度低的优先（t1）
        chosen = cs.select_next()
        assert chosen in ("t1", "t2")

    def test_select_next_favors_high_progress(self):
        """学习进展高的任务应被更多选中（exploitation 模式）。"""
        cs = CurriculumScheduler(exploration_rate=0.0)
        cs.add_task("t1", 0.1)  # 简单
        cs.add_task("t2", 0.5)
        # t2 进展快
        for e in [1.0, 0.9, 0.5, 0.1]:
            cs.record_attempt("t2", e)
        # t1 进展慢
        for e in [0.5, 0.5, 0.5, 0.5]:
            cs.record_attempt("t1", e)
        # 应选 t2
        chosen = cs.select_next()
        assert chosen == "t2"

    def test_empty_pool_raises(self):
        cs = CurriculumScheduler()
        with pytest.raises(RuntimeError):
            cs.select_next()

    def test_mastery_clipped_to_unit(self):
        cs = CurriculumScheduler()
        cs.add_task("t1", 0.5)
        cs.record_attempt("t1", error=2.0)  # 1 - 2 = -1 → 0
        cs.record_attempt("t1", error=-1.0)  # 1 - (-1) = 2 → 1
        m = cs._tasks["t1"].mastery()
        assert 0.0 <= m <= 1.0

    def test_process_via_context(self):
        cs = CurriculumScheduler()
        cs.add_task("t1", 0.5)
        ctx = SkillContext(belief=np.zeros(32), prediction_error=0.5)
        result = cs.safe_process(ctx)
        assert result.error is None
        assert "current_task" in result.data
        assert result.data["n_tasks"] == 1

    def test_snapshot(self):
        cs = CurriculumScheduler()
        cs.add_task("t1", 0.5)
        snap = cs.snapshot()
        assert snap["ready"] is True
        assert snap["data"]["n_tasks"] == 1

    def test_difficulty_clipped(self):
        cs = CurriculumScheduler()
        t = cs.add_task("t1", difficulty=1.5)
        assert t.difficulty == 1.0
        t2 = cs.add_task("t2", difficulty=-0.5)
        assert t2.difficulty == 0.0

    def test_task_to_dict(self):
        cs = CurriculumScheduler()
        cs.add_task("t1", 0.3, description="push ball")
        cs.record_attempt("t1", 0.5)
        d = cs._tasks["t1"].to_dict()
        assert d["task_id"] == "t1"
        assert d["description"] == "push ball"
        assert d["n_attempts"] == 1
        assert 0.0 <= d["mastery"] <= 1.0


# ================================================================== #
# 19. MemoryDecay
# ================================================================== #
class TestMemoryDecay:
    def test_store_and_retrieve(self):
        md = MemoryDecay()
        md.store("m1", np.array([1.0, 0.0, 0.0]))
        md.store("m2", np.array([0.0, 1.0, 0.0]))
        results = md.retrieve(np.array([1.0, 0.0, 0.0]), k=1)
        assert len(results) == 1
        assert results[0].item_id == "m1"

    def test_store_update_existing(self):
        md = MemoryDecay()
        md.store("m1", np.array([1.0, 0.0]))
        md.store("m1", np.array([0.0, 1.0]))  # 更新
        assert len(md._memories) == 1
        # 更新后 age=0
        assert md._memories["m1"].age == 0

    def test_retrieve_updates_age_and_count(self):
        md = MemoryDecay()
        md.store("m1", np.array([1.0, 0.0]))
        md.store("m2", np.array([0.0, 1.0]))
        md.retrieve(np.array([1.0, 0.0]), k=1)
        assert md._memories["m1"].n_retrievals == 1
        assert md._memories["m1"].age == 0
        assert md._memories["m2"].age == 0  # 未被检索的初始也是 0
        # 衰减一次 → 未检索项 age=1
        md.decay()
        assert md._memories["m2"].age == 1
        # m1 也 age+=1，但下次检索会归 0
        assert md._memories["m1"].age == 1

    def test_decay_reduces_weight(self):
        md = MemoryDecay(decay_factor=0.5)
        md.store("m1", np.array([1.0, 0.0]))
        for _ in range(3):
            md.decay()
        # weight = 1 * 0.5^3 = 0.125
        assert abs(md._memories["m1"].weight - 0.125) < 1e-9

    def test_decay_factor_must_be_in_range(self):
        with pytest.raises(ValueError):
            MemoryDecay(decay_factor=0.0)
        with pytest.raises(ValueError):
            MemoryDecay(decay_factor=1.5)

    def test_prune_threshold_must_be_in_range(self):
        with pytest.raises(ValueError):
            MemoryDecay(prune_threshold=-0.1)
        with pytest.raises(ValueError):
            MemoryDecay(prune_threshold=1.5)

    def test_prune_removes_low_weight(self):
        md = MemoryDecay(decay_factor=0.5, prune_threshold=0.3)
        md.store("m1", np.array([1.0, 0.0]))
        md.store("m2", np.array([0.0, 1.0]))
        # 检索 m1 一次，使其 age 归 0
        # 实际上 age 不影响 weight（weight 只随 decay 衰减）
        for _ in range(3):
            md.decay()
        n_pruned = md.prune()
        # 3 次衰减后 weight = 0.125 < 0.3 → 应该都被剪
        assert n_pruned == 2

    def test_eviction_when_capacity_full(self):
        md = MemoryDecay(decay_factor=0.99, max_capacity=3)
        for i in range(5):
            md.store(f"m{i}", np.array([float(i), 0.0]))
        assert len(md._memories) == 3
        assert md._n_evicted == 2

    def test_retrieve_empty_returns_empty(self):
        md = MemoryDecay()
        results = md.retrieve(np.array([1.0, 0.0]), k=1)
        assert results == []

    def test_retrieve_topk_limits(self):
        md = MemoryDecay()
        # 使用四个不同方向（基本正交）的向量
        md.store("m0", np.array([1.0, 0.0]))
        md.store("m1", np.array([0.0, 1.0]))
        md.store("m2", np.array([-1.0, 0.0]))
        md.store("m3", np.array([0.0, -1.0]))
        md.store("m4", np.array([0.7, 0.7]))
        results = md.retrieve(np.array([1.0, 0.1]), k=3)
        assert len(results) == 3
        # 最相似的是 m0（向量 [1, 0]）
        assert results[0].item_id == "m0"

    def test_usage_rate(self):
        md = MemoryDecay()
        md.store("m1", np.array([1.0, 0.0]))
        md.store("m2", np.array([0.0, 1.0]))
        md.store("m3", np.array([0.5, 0.5]))
        md.retrieve(np.array([1.0, 0.0]), k=1)  # 只命中 m1
        assert md.usage_rate() == 1.0 / 3.0

    def test_step_and_prune(self):
        md = MemoryDecay(decay_factor=0.5, prune_threshold=0.3)
        md.store("m1", np.array([1.0, 0.0]))
        md.store("m2", np.array([0.0, 1.0]))
        result = md.step_and_prune()
        assert "active_memories" in result
        assert "pruned" in result

    def test_process_via_context(self):
        md = MemoryDecay()
        md.store("m1", np.array([1.0, 0.0]))
        ctx = SkillContext(belief=np.zeros(32))
        result = md.safe_process(ctx)
        assert result.error is None
        assert "stats" in result.data

    def test_snapshot_includes_counts(self):
        md = MemoryDecay()
        md.store("m1", np.array([1.0, 0.0]))
        snap = md.snapshot()
        assert snap["data"]["n_memories"] == 1
        assert snap["data"]["capacity"] == 1000

    def test_metadata_persisted(self):
        md = MemoryDecay()
        md.store("m1", np.array([1.0, 0.0]), metadata={"label": "cat"})
        assert md._memories["m1"].metadata == {"label": "cat"}


# ================================================================== #
# 20. EnergyMonitor
# ================================================================== #
class TestEnergyMonitor:
    def test_default_mode_balanced(self):
        em = EnergyMonitor()
        assert em.current_mode() == "balanced"

    def test_update_reads_system_state(self):
        em = EnergyMonitor()
        em.set_readers(cpu=lambda: 50.0, mem=lambda: 60.0, battery=lambda: 80.0)
        state = em.update()
        assert state["cpu"] == 50.0
        assert state["memory"] == 60.0
        assert state["battery"] == 80.0

    def test_decide_mode_high_cpu(self):
        em = EnergyMonitor(cpu_threshold_high=80.0)
        em.set_readers(cpu=lambda: 90.0, mem=lambda: 50.0, battery=lambda: 80.0)
        em.update()
        # 高 CPU 但电量充足 → balanced（降负载）
        assert em.decide_mode() == "balanced"

    def test_decide_mode_low_cpu_balanced_battery(self):
        em = EnergyMonitor(cpu_threshold_low=30.0)
        em.set_readers(cpu=lambda: 10.0, mem=lambda: 30.0, battery=lambda: 80.0)
        em.update()
        # 低 CPU 且电量充足 → performance
        assert em.decide_mode() == "performance"

    def test_decide_mode_low_battery(self):
        em = EnergyMonitor(battery_threshold_low=30.0)
        em.set_readers(cpu=lambda: 10.0, mem=lambda: 30.0, battery=lambda: 25.0)
        em.update()
        # 电量低 → power_saver
        assert em.decide_mode() == "power_saver"

    def test_decide_mode_critical_battery(self):
        em = EnergyMonitor(battery_threshold_critical=10.0)
        em.set_readers(cpu=lambda: 10.0, mem=lambda: 30.0, battery=lambda: 5.0)
        em.update()
        # 电量严重低 → critical
        assert em.decide_mode() == "critical"

    def test_force_mode_overrides_decide(self):
        em = EnergyMonitor()
        em.set_readers(cpu=lambda: 90.0, mem=lambda: 90.0, battery=lambda: 100.0)
        em.update()
        em.set_force_mode("critical")
        assert em.current_mode() == "critical"
        assert em.recommend().mode == "critical"

    def test_force_mode_none_clears_override(self):
        em = EnergyMonitor()
        em.set_force_mode("critical")
        em.set_force_mode(None)
        # 不再强制
        assert em.stats()["forced"] is False

    def test_force_mode_invalid_raises(self):
        em = EnergyMonitor()
        with pytest.raises(ValueError):
            em.set_force_mode("turbo")

    def test_recommend_returns_policy(self):
        em = EnergyMonitor()
        em.set_readers(cpu=lambda: 10.0, mem=lambda: 30.0, battery=lambda: 80.0)
        em.update()
        policy = em.recommend()
        assert isinstance(policy, EnergyPolicy)
        assert policy.mode == "performance"
        assert policy.max_iterations > 0
        assert 0 < policy.latent_dim_scale <= 1.0

    def test_policy_preset_variations(self):
        perf = _policy_for("performance")
        saver = _policy_for("power_saver")
        crit = _policy_for("critical")
        assert perf.max_iterations > saver.max_iterations
        assert saver.max_iterations > crit.max_iterations

    def test_estimate_battery_remaining(self):
        em = EnergyMonitor()
        # 模拟电量线性下降
        levels = [80.0, 79.0, 78.0, 77.0, 76.0, 75.0]
        em.set_readers(battery=lambda: levels.pop(0) if levels else 75.0)
        for _ in range(6):
            em.update()
        remaining = em.estimate_battery_remaining_min()
        assert remaining is not None
        assert remaining > 0.0

    def test_estimate_battery_returns_none_if_stable(self):
        em = EnergyMonitor()
        em.set_readers(battery=lambda: 100.0)
        for _ in range(10):
            em.update()
        # 电量稳定 → None
        assert em.estimate_battery_remaining_min() is None

    def test_history_capped(self):
        em = EnergyMonitor(history_size=5)
        em.set_readers(cpu=lambda: 50.0, mem=lambda: 60.0, battery=lambda: 80.0)
        for _ in range(10):
            em.update()
        assert len(em._cpu_history) == 5

    def test_process_via_context(self):
        em = EnergyMonitor()
        em.set_readers(cpu=lambda: 50.0, mem=lambda: 60.0, battery=lambda: 80.0)
        ctx = SkillContext(belief=np.zeros(32))
        result = em.safe_process(ctx)
        assert result.error is None
        assert "stats" in result.data
        assert "policy" in result.data

    def test_snapshot_includes_mode(self):
        em = EnergyMonitor()
        em.set_readers(cpu=lambda: 10.0, mem=lambda: 30.0, battery=lambda: 80.0)
        em.update()
        snap = em.snapshot()
        assert snap["data"]["mode"] == "performance"

    def test_stats_returns_dict(self):
        em = EnergyMonitor()
        em.set_readers(cpu=lambda: 50.0, mem=lambda: 60.0, battery=lambda: 80.0)
        em.update()
        s = em.stats()
        assert s["cpu"] == 50.0
        assert s["memory"] == 60.0
        assert s["battery"] == 80.0
        assert s["forced"] is False
        assert s["step"] == 1
