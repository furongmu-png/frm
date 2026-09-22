from __future__ import annotations

from threading import Thread

import numpy as np
import pytest

from zero_data_model.experiment.hypothesis_tester import (
    Hypothesis,
    HypothesisTester,
)


# --------------------------------------------------------------------------- #
# 构造
# --------------------------------------------------------------------------- #


def test_construct_default():
    ht = HypothesisTester()
    assert ht._edge_threshold == 0.05
    assert len(ht.hypotheses) == 0


def test_construct_with_args():
    ht = HypothesisTester(edge_threshold=0.1, seed=42, max_hypotheses=10)
    assert ht._edge_threshold == 0.1
    assert ht.hypotheses.maxlen == 10


# --------------------------------------------------------------------------- #
# generate_from_causal_graph / design_experiment / update_with_data / get_supported
# --------------------------------------------------------------------------- #


def test_generate_from_causal_graph_returns_hypotheses():
    ht = HypothesisTester(edge_threshold=0.05, seed=42)
    m = np.array([[0.0, 0.5, 0.0],
                  [0.0, 0.0, 0.01],
                  [0.3, 0.0, 0.0]])
    hyps = ht.generate_from_causal_graph(m)
    assert isinstance(hyps, list)
    # m[0,1]=0.5 (>0.2) → causes；m[1,2]=0.01 (<0.05) → no_effect；m[2,0]=0.3 (>0.2) → causes
    assert len(hyps) > 0
    directions = [h.direction for h in hyps]
    assert "causes" in directions
    assert "no_effect" in directions


def test_generate_respects_max_hyps():
    ht = HypothesisTester(seed=42)
    # 全部 off-diagonal 都触发假设 → 远超 max_hyps=2
    m = np.full((4, 4), 0.5)
    np.fill_diagonal(m, 0.0)
    hyps = ht.generate_from_causal_graph(m, max_hyps=2)
    assert len(hyps) == 2


def test_generate_skips_diagonal():
    """对角元素被跳过（i==j）。"""
    ht = HypothesisTester(seed=42)
    # 仅对角线非零，所有非对角为 0 → 但 0 < edge_threshold=0.05 触发 no_effect
    m = np.array([[1.0, 0.0],
                  [0.0, 1.0]])
    hyps = ht.generate_from_causal_graph(m)
    # 非对角 2 个元素，都为 0 → 都触发 no_effect
    assert len(hyps) == 2
    for h in hyps:
        assert h.direction == "no_effect"
        assert h.cause_var != h.effect_var


def test_design_experiment_returns_dict():
    ht = HypothesisTester(seed=42)
    hyp = Hypothesis(
        statement="x causes y",
        cause_var=1, effect_var=2, direction="causes",
    )
    design = ht.design_experiment(hyp)
    assert "fix_var" in design
    assert "observe_var" in design
    assert "n_trials" in design
    assert "intervention_value" in design
    assert design["fix_var"] == 1
    assert design["observe_var"] == 2
    assert np.isfinite(design["intervention_value"])


def test_update_with_data_marks_supported_for_causes():
    """方向 causes + 方差小 → 支持。"""
    ht = HypothesisTester(seed=42)
    hyp = Hypothesis(
        statement="x causes y",
        cause_var=0, effect_var=1, direction="causes",
    )
    # 方差很小（<0.01）→ BF=5 → posterior=5 > 1 → supported=True
    obs = np.array([0.5, 0.5, 0.5, 0.5])
    ht.update_with_data(hyp, obs)
    assert hyp.tested is True
    assert hyp.supported is True


def test_update_with_data_marks_unsupported_when_variance_high():
    """方向 causes + 方差大 → 不支持（posterior=0.2 < 1）。"""
    ht = HypothesisTester(seed=42)
    hyp = Hypothesis(
        statement="x causes y",
        cause_var=0, effect_var=1, direction="causes",
    )
    obs = np.array([0.0, 1.0, 2.0, 3.0])  # 高方差
    ht.update_with_data(hyp, obs)
    assert hyp.tested is True
    assert hyp.supported is False


def test_get_supported_returns_only_supported_tested():
    ht = HypothesisTester(seed=42)
    m = np.array([[0.0, 0.5],
                  [0.0, 0.0]])
    ht.generate_from_causal_graph(m)
    # 对所有假设提供低方差数据 → 全部 supported（如果是 causes）
    for hyp in list(ht.hypotheses):
        if hyp.direction == "causes":
            ht.update_with_data(hyp, np.array([0.5, 0.5, 0.5, 0.5]))
    supported = ht.get_supported()
    assert all(h.tested and h.supported for h in supported)


def test_get_supported_empty_when_nothing_tested():
    ht = HypothesisTester(seed=42)
    ht.generate_from_causal_graph(np.array([[0.0, 0.5], [0.0, 0.0]]))
    assert ht.get_supported() == []


# --------------------------------------------------------------------------- #
# 军事级修复点
# --------------------------------------------------------------------------- #


def test_generate_with_none_transition_raises_type_error():
    """核心修复点：None 入参抛 TypeError。"""
    ht = HypothesisTester(seed=42)
    with pytest.raises(TypeError):
        ht.generate_from_causal_graph(None)


def test_update_with_data_none_observations_raises_type_error():
    """核心修复点：None 入参抛 TypeError。"""
    ht = HypothesisTester(seed=42)
    hyp = Hypothesis(
        statement="x causes y",
        cause_var=0, effect_var=1, direction="causes",
    )
    with pytest.raises(TypeError):
        ht.update_with_data(hyp, None)


def test_generate_with_non_2d_raises_value_error():
    """核心修复点：非二维 transition_matrix 抛 ValueError。"""
    ht = HypothesisTester(seed=42)
    with pytest.raises(ValueError):
        ht.generate_from_causal_graph(np.array([1.0, 2.0, 3.0]))


def test_generate_with_non_square_raises_value_error():
    """非方阵抛 ValueError。"""
    ht = HypothesisTester(seed=42)
    with pytest.raises(ValueError):
        ht.generate_from_causal_graph(np.zeros((2, 3)))


def test_update_with_data_short_obs_returns_no_crash():
    """观测不足 2 个 → 方差按 0 处理。"""
    ht = HypothesisTester(seed=42)
    hyp = Hypothesis(
        statement="x causes y",
        cause_var=0, effect_var=1, direction="causes",
    )
    # 单点观测 → 方差=0.0 < 0.01 → BF=5（causes）→ supported
    ht.update_with_data(hyp, np.array([0.5]))
    assert hyp.tested is True
    assert hyp.supported is True


def test_hypotheses_deque_maxlen():
    """deque maxlen 限长：hypotheses 上限 max_hypotheses。"""
    ht = HypothesisTester(max_hypotheses=5, seed=42)
    # 每次生成 12 个假设，但 deque maxlen=5
    m = np.full((4, 4), 0.5)
    np.fill_diagonal(m, 0.0)
    ht.generate_from_causal_graph(m, max_hyps=20)
    assert len(ht.hypotheses) == 5


# --------------------------------------------------------------------------- #
# 线程安全
# --------------------------------------------------------------------------- #


def test_concurrent_generate_does_not_crash():
    ht = HypothesisTester(max_hypotheses=10000, seed=42)
    m = np.full((3, 3), 0.5)
    np.fill_diagonal(m, 0.0)

    def worker() -> None:
        for _ in range(20):
            ht.generate_from_causal_graph(m)

    threads = [Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # 至少有些假设被生成
    assert len(ht.hypotheses) > 0


def test_concurrent_update_with_data_does_not_crash():
    ht = HypothesisTester(seed=42)
    m = np.array([[0.0, 0.5], [0.0, 0.0]])
    ht.generate_from_causal_graph(m)

    def worker() -> None:
        for _ in range(20):
            for hyp in list(ht.hypotheses):
                ht.update_with_data(hyp, np.array([0.5, 0.5, 0.5]))

    threads = [Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
