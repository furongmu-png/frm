from __future__ import annotations

from threading import Thread

import numpy as np
import pytest

from zero_data_model.knowledge.causal_inference import CausalInference


# --------------------------------------------------------------------------- #
# 构造与 set_transition
# --------------------------------------------------------------------------- #


def test_construct_default_no_matrix():
    ci = CausalInference()
    assert ci.transition_matrix is None


def test_construct_with_matrix():
    m = np.eye(3)
    ci = CausalInference(transition_matrix=m)
    assert ci.transition_matrix is not None
    assert ci.transition_matrix.shape == (3, 3)


def test_set_transition_replaces_matrix():
    ci = CausalInference()
    ci.set_transition(np.eye(2))
    assert ci.transition_matrix.shape == (2, 2)
    # 替换
    ci.set_transition(np.eye(4))
    assert ci.transition_matrix.shape == (4, 4)


# --------------------------------------------------------------------------- #
# 军事级修复点：非方阵抛 ValueError
# --------------------------------------------------------------------------- #


def test_set_transition_rejects_non_square_matrix():
    """非方阵抛 ValueError。"""
    ci = CausalInference()
    with pytest.raises(ValueError):
        ci.set_transition(np.zeros((2, 3)))


def test_set_transition_rejects_1d_array():
    """一维数组抛 ValueError。"""
    ci = CausalInference()
    with pytest.raises(ValueError):
        ci.set_transition(np.array([1.0, 2.0, 3.0]))


# --------------------------------------------------------------------------- #
# do_calculus
# --------------------------------------------------------------------------- #


def test_do_calculus_returns_predicted_state():
    ci = CausalInference()
    ci.set_transition(np.eye(3))
    result = ci.do_calculus({0: 5.0})
    assert "predicted_state" in result
    assert "intervened_vars" in result
    assert "changed_vars" in result
    # predicted_state 应为一维
    assert isinstance(result["predicted_state"], np.ndarray)
    assert result["predicted_state"].ndim == 1
    assert result["predicted_state"].shape == (3,)


def test_do_calculus_without_matrix_returns_empty():
    ci = CausalInference()
    result = ci.do_calculus({0: 5.0})
    assert result["predicted_state"].size == 0
    assert result["changed_vars"] == []


def test_do_calculus_identifies_changed_vars():
    """非对角转移矩阵：干预 var 0 → 通过 m 改变其他变量。"""
    ci = CausalInference()
    # m[i, j] 表示 var i 对 var j 的影响
    m = np.array([[0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0],
                  [1.0, 0.0, 0.0]])
    ci.set_transition(m)
    # baseline: state=[1,1,1] @ m = [1,1,1]
    # 干预 var 0 = 5 → modified=[5,1,1] → predicted = [1, 5, 1]
    # → var 1 改变（5 vs 1）
    result = ci.do_calculus({0: 5.0})
    assert 1 in result["changed_vars"]
    # var 0 自身 baseline=predicted=1 → 不变
    assert 0 not in result["changed_vars"]


# --------------------------------------------------------------------------- #
# counterfactual
# --------------------------------------------------------------------------- #


def test_counterfactual_applies_intervention():
    ci = CausalInference()
    ci.set_transition(np.eye(3))
    observed = np.array([1.0, 2.0, 3.0])
    result = ci.counterfactual(observed, {1: 99.0})
    assert result[0] == 1.0
    assert result[1] == 99.0
    assert result[2] == 3.0


def test_counterfactual_returns_1d_array():
    """2-D 输入 ravel 归一化：返回一维数组。"""
    ci = CausalInference()
    ci.set_transition(np.eye(3))
    observed_2d = np.array([[1.0, 2.0, 3.0]])
    result = ci.counterfactual(observed_2d, {1: 9.0})
    assert result.ndim == 1
    assert result.shape == (3,)


def test_counterfactual_returns_copy_of_observed():
    """counterfactual 不修改原 observed 数组。"""
    ci = CausalInference()
    ci.set_transition(np.eye(3))
    observed = np.array([1.0, 2.0, 3.0])
    observed_copy = observed.copy()
    ci.counterfactual(observed, {1: 99.0})
    # 原 observed 不变
    assert np.allclose(observed, observed_copy)


def test_counterfactual_out_of_range_intervention_ignored():
    """干预变量索引越界被忽略。"""
    ci = CausalInference()
    ci.set_transition(np.eye(3))
    result = ci.counterfactual(np.array([1.0, 2.0, 3.0]),
                                {5: 99.0, -1: 99.0})
    # 没有变化
    assert np.allclose(result, [1.0, 2.0, 3.0])


def test_counterfactual_none_observed_raises_type_error():
    """核心修复点：None 入参抛 TypeError。"""
    ci = CausalInference()
    ci.set_transition(np.eye(3))
    with pytest.raises(TypeError):
        ci.counterfactual(None, {0: 1.0})


def test_counterfactual_none_intervention_raises_type_error():
    """核心修复点：None intervention 抛 TypeError。"""
    ci = CausalInference()
    ci.set_transition(np.eye(3))
    with pytest.raises(TypeError):
        ci.counterfactual(np.array([1.0, 2.0, 3.0]), None)


def test_counterfactual_without_matrix_raises_type_error():
    """未设置 transition_matrix 抛 TypeError。"""
    ci = CausalInference()
    with pytest.raises(TypeError):
        ci.counterfactual(np.array([1.0, 2.0, 3.0]), {0: 1.0})


# --------------------------------------------------------------------------- #
# identify_confounders
# --------------------------------------------------------------------------- #


def test_identify_confounders_finds_common_cause():
    ci = CausalInference()
    # var 0 同时影响 var 1 和 var 2 → 0 是 (1,2) 的混杂因子
    m = np.array([[0.0, 1.0, 1.0],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0]])
    ci.set_transition(m)
    confs = ci.identify_confounders(1, 2)
    assert 0 in confs


def test_identify_confounders_empty_without_matrix():
    ci = CausalInference()
    assert ci.identify_confounders(0, 1) == []


def test_identify_confounders_out_of_range_returns_empty():
    ci = CausalInference()
    ci.set_transition(np.eye(3))
    assert ci.identify_confounders(0, 99) == []
    assert ci.identify_confounders(-1, 1) == []


# --------------------------------------------------------------------------- #
# 线程安全
# --------------------------------------------------------------------------- #


def test_concurrent_set_transition_does_not_crash():
    ci = CausalInference()

    def worker(i: int) -> None:
        ci.set_transition(np.eye(i + 2))

    threads = [Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # 最终 transition_matrix 应是某次写入的方阵
    assert ci.transition_matrix is not None
    s = ci.transition_matrix.shape
    assert s[0] == s[1]


def test_concurrent_do_calculus_does_not_crash():
    ci = CausalInference()
    ci.set_transition(np.eye(4))

    def worker() -> None:
        for _ in range(20):
            ci.do_calculus({0: 1.0, 2: 3.0})

    threads = [Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def test_concurrent_counterfactual_does_not_crash():
    ci = CausalInference()
    ci.set_transition(np.eye(4))
    obs = np.array([1.0, 2.0, 3.0, 4.0])

    def worker() -> None:
        for _ in range(20):
            ci.counterfactual(obs, {0: 9.0})

    threads = [Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
