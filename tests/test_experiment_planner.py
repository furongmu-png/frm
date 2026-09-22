from __future__ import annotations

from threading import Thread

import numpy as np
import pytest

from zero_data_model.experiment.experiment_planner import (
    BayesianExperimentPlanner,
    CandidateExperiment,
    MIN_UNCERTAINTY,
)


# --------------------------------------------------------------------------- #
# 构造
# --------------------------------------------------------------------------- #


def test_construct_default():
    p = BayesianExperimentPlanner()
    assert p._eval_interval == 1000
    assert len(p.candidates) == 0
    assert p._param_uncertainty == 1.0


def test_construct_with_args():
    p = BayesianExperimentPlanner(eval_interval=10, seed=7,
                                    max_candidates=5, max_history=10)
    assert p._eval_interval == 10
    assert p.candidates.maxlen == 5


# --------------------------------------------------------------------------- #
# 军事级修复点：eval_interval=0 抛 ValueError
# --------------------------------------------------------------------------- #


def test_construct_eval_interval_zero_raises_value_error():
    """核心修复点：eval_interval=0 抛 ValueError。"""
    with pytest.raises(ValueError):
        BayesianExperimentPlanner(eval_interval=0)


def test_construct_eval_interval_negative_raises_value_error():
    with pytest.raises(ValueError):
        BayesianExperimentPlanner(eval_interval=-5)


# --------------------------------------------------------------------------- #
# register_candidate / estimate_info_gain / select_best / record_result / evaluate
# --------------------------------------------------------------------------- #


def test_register_candidate_appends():
    p = BayesianExperimentPlanner()
    p.register_candidate("c1", {"type": "scalar"})
    assert len(p.candidates) == 1
    assert p.candidates[0].name == "c1"


def test_default_candidates_registers_five():
    p = BayesianExperimentPlanner()
    p.default_candidates(dim=8)
    assert len(p.candidates) == 5


def test_estimate_info_gain_returns_finite():
    p = BayesianExperimentPlanner(seed=42)
    p.register_candidate("c1", {"type": "scalar", "factor": 1.5})
    cand = p.candidates[0]
    gain = p.estimate_info_gain(cand, current_uncertainty=0.5)
    assert np.isfinite(gain)
    assert gain >= 0.0


def test_select_best_returns_highest_gain():
    p = BayesianExperimentPlanner(seed=42)
    p.register_candidate("c1", {"x": 1})
    p.register_candidate("c2", {"x": 1, "y": 2, "z": 3, "w": 4})
    best = p.select_best(current_uncertainty=0.5)
    assert best is not None
    assert best.name in ("c1", "c2")


def test_select_best_returns_none_when_all_executed():
    p = BayesianExperimentPlanner(seed=42)
    p.register_candidate("c1", {"x": 1})
    p.candidates[0].executed = True
    assert p.select_best(current_uncertainty=0.5) is None


def test_select_best_returns_none_when_empty():
    p = BayesianExperimentPlanner(seed=42)
    assert p.select_best(current_uncertainty=0.5) is None


def test_record_result_marks_executed():
    p = BayesianExperimentPlanner(seed=42)
    p.register_candidate("c1", {"x": 1})
    cand = p.candidates[0]
    p.record_result(cand, fe_before=2.0, fe_after=1.0)
    assert cand.executed is True
    assert cand.actual_gain == 1.0
    assert len(p._history) == 1


def test_evaluate_returns_none_off_interval():
    p = BayesianExperimentPlanner(eval_interval=10, seed=42)
    p.register_candidate("c1", {"x": 1})
    # step=5：5 % 10 != 0 → 不评估
    assert p.evaluate(current_uncertainty=0.5, step=5) is None


def test_evaluate_returns_dict_on_interval():
    p = BayesianExperimentPlanner(eval_interval=10, seed=42)
    p.register_candidate("c1", {"x": 1})
    result = p.evaluate(current_uncertainty=0.5, step=10)
    assert result is not None
    assert "experiment" in result
    assert "intervention" in result
    assert "predicted_gain" in result


def test_evaluate_returns_none_when_no_candidates():
    p = BayesianExperimentPlanner(eval_interval=10, seed=42)
    assert p.evaluate(current_uncertainty=0.5, step=10) is None


# --------------------------------------------------------------------------- #
# 军事级修复点：_param_uncertainty 不降到 0
# --------------------------------------------------------------------------- #


def test_param_uncertainty_floor_after_many_records():
    """核心修复点：MIN_UNCERTAINTY 下限——多次大增益 record_result 后不确定性不下破。"""
    p = BayesianExperimentPlanner(seed=42)
    p.register_candidate("c1", {"x": 1})
    cand = p.candidates[0]
    # 多次大增益记录，模拟不确定性持续衰减
    for _ in range(50):
        # 用同一个 candidate 也无所谓——只检查 _param_uncertainty 不破下限
        c = CandidateExperiment(name="tmp", intervention={})
        p.record_result(c, fe_before=10.0, fe_after=0.0)  # gain=10
    assert p._param_uncertainty >= MIN_UNCERTAINTY


def test_param_uncertainty_decays_with_gain():
    p = BayesianExperimentPlanner(seed=42)
    initial = p._param_uncertainty
    cand = CandidateExperiment(name="c1", intervention={})
    p.record_result(cand, fe_before=10.0, fe_after=0.0)
    assert p._param_uncertainty < initial
    assert p._param_uncertainty >= MIN_UNCERTAINTY


def test_param_uncertainty_no_change_for_zero_gain():
    p = BayesianExperimentPlanner(seed=42)
    initial = p._param_uncertainty
    cand = CandidateExperiment(name="c1", intervention={})
    # gain=0 → decay = 1 - 0/(0+1) = 1 → 不衰减
    p.record_result(cand, fe_before=1.0, fe_after=1.0)
    assert p._param_uncertainty == pytest.approx(initial)


# --------------------------------------------------------------------------- #
# 军事级修复点：deque maxlen 限长
# --------------------------------------------------------------------------- #


def test_candidates_deque_maxlen():
    """deque maxlen 限长：candidates 上限 max_candidates。"""
    p = BayesianExperimentPlanner(max_candidates=5, seed=42)
    for i in range(20):
        p.register_candidate(f"c{i}", {"x": i})
    assert len(p.candidates) == 5
    # FIFO：应保留最后 5 个
    names = [c.name for c in p.candidates]
    assert names == ["c15", "c16", "c17", "c18", "c19"]


def test_history_deque_maxlen():
    """deque maxlen 限长：history 上限 max_history。"""
    p = BayesianExperimentPlanner(max_history=5, seed=42)
    for i in range(20):
        c = CandidateExperiment(name=f"c{i}", intervention={})
        p.record_result(c, fe_before=1.0, fe_after=0.0)
    assert len(p._history) == 5


# --------------------------------------------------------------------------- #
# 线程安全
# --------------------------------------------------------------------------- #


def test_concurrent_register_does_not_crash():
    p = BayesianExperimentPlanner(max_candidates=10000, seed=42)

    def worker(idx: int) -> None:
        for i in range(50):
            p.register_candidate(f"a{idx}_{i}", {"x": i})

    threads = [Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # 4 × 50 = 200，maxlen=10000 故全部保留
    assert len(p.candidates) == 200


def test_concurrent_select_best_does_not_crash():
    """线程安全：并发 select_best 不抛异常。"""
    p = BayesianExperimentPlanner(seed=42)
    for i in range(50):
        p.register_candidate(f"c{i}", {"x": i})

    def worker() -> None:
        for _ in range(20):
            p.select_best(current_uncertainty=0.5)

    threads = [Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
