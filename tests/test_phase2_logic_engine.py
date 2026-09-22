# tests/test_phase2_logic_engine.py
"""第二阶段 §1.1 逻辑规则引擎单元测试。

验证：
- 字典式规则声明与模糊推理
- 矛盾度计算（0-1）与 Hebbian 权重调整
- Gödel t-范数与蕴含
- 苏格拉底三段论推理
- 无摩擦环境触发高矛盾度
"""
from __future__ import annotations

import numpy as np
import pytest

from zero_data_model.knowledge.logic_engine import LogicEngine


# ------------------------------------------------------------------ #
# 规则声明与验证
# ------------------------------------------------------------------ #
class TestAddRule:
    """测试字典式规则声明。"""

    def test_add_simple_rule(self) -> None:
        engine = LogicEngine(predicate_dim=8, seed=42)
        engine.add_rule({
            "if": ["a", "b"],
            "then": "c",
            "weight": 0.8,
        })
        assert len(engine.rules) == 1
        rule = engine.rules[0]
        assert rule.antecedents == ["a", "b"]
        assert rule.consequent == "c"
        assert rule.weight == 0.8

    def test_add_rule_creates_predicate_embeddings(self) -> None:
        engine = LogicEngine(predicate_dim=16, seed=42)
        engine.add_rule({"if": ["p1", "p2"], "then": "p3"})
        assert "p1" in engine.predicate_embeddings
        assert "p2" in engine.predicate_embeddings
        assert "p3" in engine.predicate_embeddings
        assert engine.predicate_embeddings["p1"].shape == (16,)

    def test_add_rule_default_weight(self) -> None:
        engine = LogicEngine(seed=42)
        engine.add_rule({"if": ["a"], "then": "b"})
        assert engine.rules[0].weight == 1.0

    def test_add_rule_weight_clamped(self) -> None:
        engine = LogicEngine(seed=42)
        engine.add_rule({"if": ["a"], "then": "b", "weight": 5.0})
        assert engine.rules[0].weight == 2.0
        engine.add_rule({"if": ["c"], "then": "d", "weight": -1.0})
        assert engine.rules[1].weight == 0.0

    def test_add_rule_missing_if_raises(self) -> None:
        engine = LogicEngine(seed=42)
        with pytest.raises(KeyError):
            engine.add_rule({"then": "b"})

    def test_add_rule_missing_then_raises(self) -> None:
        engine = LogicEngine(seed=42)
        with pytest.raises(KeyError):
            engine.add_rule({"if": ["a"]})

    def test_add_rule_empty_antecedents_raises(self) -> None:
        engine = LogicEngine(seed=42)
        with pytest.raises(ValueError):
            engine.add_rule({"if": [], "then": "b"})

    def test_add_rule_non_dict_raises(self) -> None:
        engine = LogicEngine(seed=42)
        with pytest.raises(TypeError):
            engine.add_rule("not a dict")  # type: ignore[arg-type]


# ------------------------------------------------------------------ #
# 模糊逻辑算子
# ------------------------------------------------------------------ #
class TestFuzzyOperators:
    """测试 Gödel t-范数与蕴含。"""

    def test_t_norm_is_min(self) -> None:
        assert LogicEngine.t_norm(0.3, 0.7) == 0.3
        assert LogicEngine.t_norm(0.9, 0.1) == 0.1
        assert LogicEngine.t_norm(0.5, 0.5) == 0.5

    def test_implication_satisfied(self) -> None:
        # A <= B → implication = 1
        assert LogicEngine.implication(0.3, 0.7) == 1.0
        assert LogicEngine.implication(0.5, 0.5) == 1.0

    def test_implication_violated(self) -> None:
        # A > B → implication = B
        assert LogicEngine.implication(0.9, 0.3) == 0.3
        assert LogicEngine.implication(0.8, 0.2) == 0.2


# ------------------------------------------------------------------ #
# 一致性检查
# ------------------------------------------------------------------ #
class TestCheckConsistency:
    """测试矛盾度计算。"""

    def test_no_rules_zero_contradiction(self) -> None:
        engine = LogicEngine(seed=42)
        assert engine.check_consistency({"a": 0.5}) == 0.0

    def test_consistent_state_low_contradiction(self) -> None:
        engine = LogicEngine(seed=42)
        engine.add_rule({"if": ["a"], "then": "b", "weight": 0.8})
        # a=0.5, b=0.9 → b >= a → satisfied
        c = engine.check_consistency({"a": 0.5, "b": 0.9})
        assert c < 0.1

    def test_violated_state_high_contradiction(self) -> None:
        engine = LogicEngine(seed=42)
        engine.add_rule({"if": ["a"], "then": "b", "weight": 0.8})
        # a=0.9, b=0.1 → violated
        c = engine.check_consistency({"a": 0.9, "b": 0.1})
        assert c > 0.2

    def test_contradiction_in_range(self) -> None:
        engine = LogicEngine(seed=42)
        engine.add_rule({"if": ["a"], "then": "b", "weight": 1.0})
        for _ in range(50):
            a = float(np.random.random())
            b = float(np.random.random())
            c = engine.check_consistency({"a": a, "b": b})
            assert 0.0 <= c <= 1.0

    def test_multiple_rules_aggregate(self) -> None:
        engine = LogicEngine(seed=42)
        engine.add_rule({"if": ["a"], "then": "b", "weight": 1.0})
        engine.add_rule({"if": ["c"], "then": "d", "weight": 1.0})
        # Both violated
        c = engine.check_consistency({
            "a": 0.9, "b": 0.1, "c": 0.8, "d": 0.2
        })
        assert c > 0.3

    def test_non_dict_belief_raises(self) -> None:
        engine = LogicEngine(seed=42)
        with pytest.raises(TypeError):
            engine.check_consistency("not a dict")  # type: ignore[arg-type]

    def test_missing_consequent_treated_as_violated(self) -> None:
        engine = LogicEngine(seed=42)
        engine.add_rule({"if": ["a"], "then": "b", "weight": 1.0})
        # b missing → treated as 0 → high contradiction
        c = engine.check_consistency({"a": 0.9})
        assert c > 0.2


# ------------------------------------------------------------------ #
# 矛盾消解
# ------------------------------------------------------------------ #
class TestResolveContradiction:
    """测试矛盾驱动的额外误差。"""

    def test_returns_same_shape(self) -> None:
        engine = LogicEngine(seed=42)
        engine.add_rule({"if": ["a"], "then": "b"})
        engine.check_consistency({"a": 0.9, "b": 0.1})
        error = np.array([0.1, 0.2, 0.3])
        extra = engine.resolve_contradiction({"a": 0.9, "b": 0.1}, error)
        assert extra.shape == error.shape

    def test_zero_contradiction_zero_extra(self) -> None:
        engine = LogicEngine(seed=42)
        engine.add_rule({"if": ["a"], "then": "b"})
        # Consistent state
        engine.check_consistency({"a": 0.3, "b": 0.9})
        error = np.array([0.1, 0.2, 0.3])
        extra = engine.resolve_contradiction({"a": 0.3, "b": 0.9}, error)
        assert np.allclose(extra, 0.0)

    def test_high_contradiction_positive_extra(self) -> None:
        engine = LogicEngine(contradiction_gain=1.0, seed=42)
        engine.add_rule({"if": ["a"], "then": "b", "weight": 1.0})
        engine.check_consistency({"a": 0.9, "b": 0.1})
        error = np.array([0.5, 0.5, 0.5])
        extra = engine.resolve_contradiction({"a": 0.9, "b": 0.1}, error)
        assert np.all(extra > 0)
        assert np.all(extra <= error)


# ------------------------------------------------------------------ #
# Hebbian 权重调整
# ------------------------------------------------------------------ #
class TestHebbianAdjustment:
    """测试规则权重的 Hebbian 动态调整。"""

    def test_satisfied_rule_strengthens(self) -> None:
        engine = LogicEngine(lr=0.05, seed=42)
        engine.add_rule({"if": ["a"], "then": "b", "weight": 0.5})
        w_before = engine.rules[0].weight
        # Satisfied: consequent >= antecedent
        for _ in range(20):
            engine.check_consistency({"a": 0.3, "b": 0.9})
        w_after = engine.rules[0].weight
        assert w_after > w_before

    def test_violated_rule_weakens(self) -> None:
        engine = LogicEngine(lr=0.05, seed=42)
        engine.add_rule({"if": ["a"], "then": "b", "weight": 1.5})
        w_before = engine.rules[0].weight
        # Violated: antecedent >> consequent
        for _ in range(20):
            engine.check_consistency({"a": 0.9, "b": 0.1})
        w_after = engine.rules[0].weight
        assert w_after < w_before

    def test_weight_stays_in_range(self) -> None:
        engine = LogicEngine(lr=0.5, seed=42)
        engine.add_rule({"if": ["a"], "then": "b", "weight": 1.0})
        for _ in range(100):
            engine.check_consistency({"a": 0.9, "b": 0.1})
        assert 0.0 <= engine.rules[0].weight <= 2.0


# ------------------------------------------------------------------ #
# 谓词嵌入
# ------------------------------------------------------------------ #
class TestPredicateEmbeddings:
    """测试可学习谓词嵌入。"""

    def test_embedding_dim(self) -> None:
        engine = LogicEngine(predicate_dim=32, seed=42)
        engine.add_rule({"if": ["a"], "then": "b"})
        assert engine.predicate_embeddings["a"].shape == (32,)

    def test_get_predicate_embedding(self) -> None:
        engine = LogicEngine(predicate_dim=8, seed=42)
        emb = engine.get_predicate_embedding("new_pred")
        assert emb.shape == (8,)
        assert "new_pred" in engine.predicate_embeddings

    def test_different_predicates_different_embeddings(self) -> None:
        engine = LogicEngine(predicate_dim=16, seed=42)
        e1 = engine.get_predicate_embedding("p1")
        e2 = engine.get_predicate_embedding("p2")
        assert not np.allclose(e1, e2)


# ------------------------------------------------------------------ #
# 集成验证：苏格拉底三段论
# ------------------------------------------------------------------ #
class TestSyllogism:
    """测试苏格拉底三段论推理。

    苏格拉底是人 → 人都会死 → 苏格拉底会死。
    """

    def test_socrates_syllogism(self) -> None:
        engine = LogicEngine(predicate_dim=8, seed=42)
        # Rule 1: if socrates is human, then socrates is mortal
        # Rule 2: if human, then mortal (general rule)
        engine.add_rule({
            "if": ["socrates_is_human"],
            "then": "socrates_is_mortal",
            "weight": 1.0,
            "name": "socrates_mortal",
        })

        # Socrates is human (high truth), but is he mortal?
        belief = {"socrates_is_human": 0.9, "socrates_is_mortal": 0.1}
        c = engine.check_consistency(belief)
        # If Socrates is human but not mortal → high contradiction
        assert c > 0.2, f"Expected contradiction, got {c}"

        # Now Socrates is mortal → consistent
        belief2 = {"socrates_is_human": 0.9, "socrates_is_mortal": 0.85}
        c2 = engine.check_consistency(belief2)
        assert c2 < c, "Consistent state should have lower contradiction"


# ------------------------------------------------------------------ #
# 验证场景：无摩擦环境
# ------------------------------------------------------------------ #
class TestFrictionlessEnv:
    """验证：'所有移动物体最终停止'在无摩擦环境触发高矛盾度。"""

    def test_frictionless_high_contradiction(self) -> None:
        engine = LogicEngine(predicate_dim=8, seed=42)
        engine.add_rule({
            "if": ["object_is_moving"],
            "then": "object_will_stop",
            "weight": 0.8,
            "name": "moving_objects_stop",
        })
        # Frictionless: moving but won't stop
        frictionless = {"object_is_moving": 0.9, "object_will_stop": 0.1}
        c_frictionless = engine.check_consistency(frictionless)
        # Normal: moving and will stop
        normal = {"object_is_moving": 0.9, "object_will_stop": 0.8}
        c_normal = engine.check_consistency(normal)
        assert c_frictionless > c_normal
        assert c_frictionless > 0.2

    def test_stats(self) -> None:
        engine = LogicEngine(seed=42)
        engine.add_rule({"if": ["a"], "then": "b"})
        engine.check_consistency({"a": 0.9, "b": 0.1})
        s = engine.stats
        assert s["n_rules"] == 1
        assert s["n_predicates"] == 2
        assert s["total_fired"] == 1
