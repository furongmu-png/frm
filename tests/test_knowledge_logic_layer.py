from __future__ import annotations

from threading import Thread

import numpy as np
import pytest

from zero_data_model.knowledge.logic_layer import LogicLayer, LogicRule


# --------------------------------------------------------------------------- #
# 构造
# --------------------------------------------------------------------------- #


def test_construct_default():
    layer = LogicLayer()
    assert layer.rules == []
    assert layer.predicate_values == {}


# --------------------------------------------------------------------------- #
# add_rule / set_predicate / t_norm / t_conorm / evaluate_rule / check_all / get_penalty_signal
# --------------------------------------------------------------------------- #


def test_add_rule_appends_to_rules():
    layer = LogicLayer()
    layer.add_rule("r1", ["a", "b"], "c", weight=2.0, description="d")
    assert len(layer.rules) == 1
    r = layer.rules[0]
    assert r.name == "r1"
    assert r.antecedents == ["a", "b"]
    assert r.consequent == "c"
    assert r.weight == 2.0
    assert r.description == "d"


def test_set_predicate_stores_value():
    layer = LogicLayer()
    layer.set_predicate("p", 0.5)
    assert layer.predicate_values["p"] == 0.5


def test_set_predicate_clamps_to_unit_interval():
    """set_predicate 范围钳制 [0,1]：超出范围被钳到 0 或 1。"""
    layer = LogicLayer()
    layer.set_predicate("high", 1.5)
    assert layer.predicate_values["high"] == 1.0
    layer.set_predicate("low", -0.5)
    assert layer.predicate_values["low"] == 0.0
    # 边界值
    layer.set_predicate("z0", 0.0)
    assert layer.predicate_values["z0"] == 0.0
    layer.set_predicate("z1", 1.0)
    assert layer.predicate_values["z1"] == 1.0


def test_set_predicate_rejects_none():
    layer = LogicLayer()
    with pytest.raises(TypeError):
        layer.set_predicate("p", None)


def test_set_predicate_rejects_non_string_name():
    layer = LogicLayer()
    with pytest.raises(TypeError):
        layer.set_predicate(123, 0.5)


def test_set_predicate_rejects_nan_and_inf():
    layer = LogicLayer()
    with pytest.raises(ValueError):
        layer.set_predicate("p", float("nan"))
    with pytest.raises(ValueError):
        layer.set_predicate("p", float("inf"))


def test_set_predicate_rejects_bool():
    """bool 是 int 的子类但不视为合法数值。"""
    layer = LogicLayer()
    with pytest.raises(TypeError):
        layer.set_predicate("p", True)


def test_t_norm_returns_min():
    layer = LogicLayer()
    assert layer.t_norm(0.3, 0.7) == 0.3
    assert layer.t_norm(0.9, 0.1) == 0.1


def test_t_conorm_returns_max():
    layer = LogicLayer()
    assert layer.t_conorm(0.3, 0.7) == 0.7
    assert layer.t_conorm(0.1, 0.9) == 0.9


def test_evaluate_rule_returns_inferred_and_penalty():
    """前件真值 t-范数 → 后件推断真值；后件低于推断则产生惩罚。"""
    layer = LogicLayer()
    layer.add_rule("r1", ["a", "b"], "c", weight=1.0)
    layer.set_predicate("a", 0.8)
    layer.set_predicate("b", 0.6)
    layer.set_predicate("c", 0.4)
    inferred, penalty = layer.evaluate_rule(layer.rules[0])
    # inferred = min(0.8, 0.6) = 0.6
    assert inferred == pytest.approx(0.6)
    # penalty = (0.6 - 0.4) * 1.0 = 0.2
    assert penalty == pytest.approx(0.2)


def test_evaluate_rule_no_penalty_when_actual_meets_inferred():
    layer = LogicLayer()
    layer.add_rule("r1", ["a"], "c")
    layer.set_predicate("a", 0.5)
    layer.set_predicate("c", 0.5)
    _inferred, penalty = layer.evaluate_rule(layer.rules[0])
    assert penalty == 0.0


def test_evaluate_rule_empty_antecedents_returns_vacuous_truth():
    """空前件规则返回空真：推断为 1.0，无惩罚。"""
    layer = LogicLayer()
    layer.add_rule("r1", [], "c")
    inferred, penalty = layer.evaluate_rule(layer.rules[0])
    assert inferred == 1.0
    assert penalty == 0.0


def test_evaluate_rule_missing_consequent_treated_as_zero():
    """缺失后件按 0.0 处理：产生违反惩罚。"""
    layer = LogicLayer()
    layer.add_rule("r1", ["a"], "c")  # c 未设置
    layer.set_predicate("a", 0.5)
    inferred, penalty = layer.evaluate_rule(layer.rules[0])
    # inferred = 0.5, actual = 0.0 → penalty = 0.5
    assert inferred == pytest.approx(0.5)
    assert penalty == pytest.approx(0.5)


def test_evaluate_rule_missing_antecedent_defaults_to_zero():
    """前件谓词缺失按 0.0 处理：t-范数与 0 取 min = 0。"""
    layer = LogicLayer()
    layer.add_rule("r1", ["missing", "b"], "c")
    layer.set_predicate("b", 0.9)
    layer.set_predicate("c", 0.0)
    inferred, penalty = layer.evaluate_rule(layer.rules[0])
    # inferred = min(min(1.0, 0.0), 0.9) = 0.0
    assert inferred == 0.0
    # actual=0.0, inferred=0.0 → actual 不低于 inferred → penalty=0
    assert penalty == 0.0


def test_check_all_returns_violations_summary():
    layer = LogicLayer()
    layer.add_rule("r1", ["a"], "c", weight=1.0)
    layer.add_rule("r2", ["b"], "d", weight=2.0)
    # r1: a=0.5 → inferred=0.5, c=0.0 → penalty=0.5
    # r2: b=0.7 → inferred=0.7, d=0.3 → penalty=(0.7-0.3)*2=0.8
    layer.set_predicate("a", 0.5)
    layer.set_predicate("b", 0.7)
    layer.set_predicate("c", 0.0)
    layer.set_predicate("d", 0.3)
    result = layer.check_all()
    assert result["n_violations"] == 2
    assert result["total_penalty"] == pytest.approx(0.5 + 0.8)
    assert len(result["violations"]) == 2


def test_check_all_no_violations():
    layer = LogicLayer()
    layer.add_rule("r1", ["a"], "c")
    layer.set_predicate("a", 0.5)
    layer.set_predicate("c", 0.5)
    result = layer.check_all()
    assert result["n_violations"] == 0
    assert result["total_penalty"] == 0.0


def test_get_penalty_signal_returns_array():
    layer = LogicLayer()
    layer.add_rule("r1", ["a"], "c")
    layer.add_rule("r2", ["b"], "d")
    layer.set_predicate("a", 0.5)
    layer.set_predicate("c", 0.0)
    layer.set_predicate("b", 0.4)
    layer.set_predicate("d", 0.4)  # 不违反
    sig = layer.get_penalty_signal()
    assert isinstance(sig, np.ndarray)
    assert sig.shape == (2,)
    # 第一条规则有违反（>0），第二条无（=0）
    assert sig[0] > 0.0
    assert sig[1] == 0.0


# --------------------------------------------------------------------------- #
# 军事级修复点：线程安全
# --------------------------------------------------------------------------- #


def test_concurrent_add_rule_does_not_crash():
    layer = LogicLayer()

    def worker(idx: int) -> None:
        for i in range(50):
            layer.add_rule(f"r{idx}_{i}", ["a"], "c")

    threads = [Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # 4 workers × 50 rules = 200
    assert len(layer.rules) == 200


def test_concurrent_set_predicate_does_not_crash():
    layer = LogicLayer()

    def worker(idx: int) -> None:
        for i in range(50):
            layer.set_predicate(f"p{idx}_{i}", 0.5)

    threads = [Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(layer.predicate_values) == 200


def test_concurrent_check_all_does_not_crash():
    """线程安全：并发 check_all 不抛异常。"""
    layer = LogicLayer()
    for i in range(20):
        layer.add_rule(f"r{i}", ["a"], "c")
    layer.set_predicate("a", 0.5)
    layer.set_predicate("c", 0.3)

    def reader() -> None:
        for _ in range(20):
            layer.check_all()
            layer.get_penalty_signal()

    threads = [Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
