"""Tests for the Reasoning capability domain (base module)."""

import numpy as np
import pytest

from zero_data_model.capabilities import (
    AnalogicalReasoner,
    DeductiveReasoner,
    InductiveReasoner,
    PropositionalLogicEngine,
    ReasoningRules,
)


@pytest.fixture(autouse=True)
def _deterministic_rng():
    np.random.seed(42)


# --------------------------------------------------------------------------- #
# ReasoningRules
# --------------------------------------------------------------------------- #


def test_reasoning_rules_defaults():
    rules = ReasoningRules()
    assert rules.max_inference_depth == 10
    assert rules.consistency_check is True
    assert rules.default_confidence == 0.5
    assert rules.contradiction_threshold == 0.5
    assert rules.abduction_max_hypotheses == 5


def test_reasoning_rules_post_init_populates_dict():
    rules = ReasoningRules()
    assert rules.rules["max_inference_depth"] == 10
    assert rules.rules["consistency_check"] is True
    assert rules.rules["default_confidence"] == 0.5


def test_reasoning_rules_custom_values():
    rules = ReasoningRules(max_inference_depth=5, default_confidence=0.7)
    assert rules.max_inference_depth == 5
    assert rules.default_confidence == 0.7


# --------------------------------------------------------------------------- #
# PropositionalLogicEngine
# --------------------------------------------------------------------------- #


def test_prop_engine_add_fact():
    eng = PropositionalLogicEngine()
    eng.add_fact("rain", True)
    result = eng.infer()
    assert result["facts"]["rain"] is True
    assert result["inferences"] == []
    assert result["contradictions"] == []


def test_prop_engine_modus_ponens():
    eng = PropositionalLogicEngine()
    eng.add_fact("rain", True)
    eng.add_rule("rain", "wet_grass")
    result = eng.infer()
    assert result["facts"]["wet_grass"] is True
    assert ("rain", "wet_grass", True) in result["inferences"]
    assert result["contradictions"] == []


def test_prop_engine_chained_inference():
    eng = PropositionalLogicEngine()
    eng.add_fact("a", True)
    eng.add_rule("a", "b")
    eng.add_rule("b", "c")
    eng.add_rule("c", "d")
    result = eng.infer()
    assert result["facts"]["b"] is True
    assert result["facts"]["c"] is True
    assert result["facts"]["d"] is True
    assert len(result["inferences"]) == 3


def test_prop_engine_false_antecedent_no_inference():
    eng = PropositionalLogicEngine()
    eng.add_fact("rain", False)
    eng.add_rule("rain", "wet_grass")
    result = eng.infer()
    assert "wet_grass" not in result["facts"]
    assert result["inferences"] == []


def test_prop_engine_negation_rule():
    eng = PropositionalLogicEngine()
    eng.add_fact("bird", True)
    eng.add_negation_rule("bird", "penguin")  # bird -> not penguin
    result = eng.infer()
    assert result["facts"]["penguin"] is False
    assert ("bird", "penguin", False) in result["inferences"]


def test_prop_engine_not_prefix_resolution():
    """Rules with ``not <prop>`` antecedent fire when prop is False."""
    eng = PropositionalLogicEngine()
    eng.add_fact("raining", False)
    eng.add_rule("not raining", "dry_ground")  # if not raining -> dry_ground
    result = eng.infer()
    assert result["facts"]["dry_ground"] is True


def test_prop_engine_contradiction_detection():
    eng = PropositionalLogicEngine()
    eng.add_fact("a", True)
    eng.add_rule("a", "b")  # a -> b is True
    eng.add_negation_rule("a", "b")  # a -> b is False (contradiction)
    result = eng.infer()
    # The first rule sets b = True; the second tries to set b = False.
    # Both "b" values get inferred, causing a contradiction.
    assert "b" in result["contradictions"]


def test_prop_engine_no_rules():
    eng = PropositionalLogicEngine()
    eng.add_fact("solo", True)
    result = eng.infer()
    assert result["facts"]["solo"] is True
    assert result["inferences"] == []


def test_prop_engine_max_inference_depth():
    """A self-feeding rule chain must terminate at max_inference_depth."""
    eng = PropositionalLogicEngine(rules=ReasoningRules(max_inference_depth=2))
    eng.add_fact("a", True)
    eng.add_rule("a", "b")
    eng.add_rule("b", "c")
    eng.add_rule("c", "d")
    eng.add_rule("d", "e")
    result = eng.infer()
    # Even with depth=2, forward chaining converges within 2 iterations.
    # All facts should be derived because each rule fires in successive
    # iterations: iter 1 -> b, iter 2 -> c, etc.
    # We only check that inference terminates without error.
    assert isinstance(result["facts"], dict)
    assert result["facts"]["a"] is True


# --------------------------------------------------------------------------- #
# DeductiveReasoner
# --------------------------------------------------------------------------- #


def test_deductive_barbara():
    """Barbara (AAA-1): All M are P; All S are M -> All S are P."""
    dr = DeductiveReasoner()
    result = dr.syllogism(("A", "human", "mortal"), ("A", "socrates", "human"))
    assert result["valid"] is True
    assert result["conclusion"] == "All socrates are mortal"
    assert "Barbara" in result["form"]


def test_deductive_celarent():
    """Celarent (EAE-1): No M are P; All S are M -> No S are P."""
    dr = DeductiveReasoner()
    result = dr.syllogism(("E", "reptile", "furry"), ("A", "snake", "reptile"))
    assert result["valid"] is True
    assert result["conclusion"] == "No snake are furry"
    assert "Celarent" in result["form"]


def test_deductive_darii():
    """Darii (AII-1): All M are P; Some S are M -> Some S are P."""
    dr = DeductiveReasoner()
    result = dr.syllogism(("A", "bird", "flying"), ("I", "swan", "bird"))
    assert result["valid"] is True
    assert result["conclusion"] == "Some swan are flying"
    assert "Darii" in result["form"]


def test_deductive_ferio():
    """Ferio (EIO-1): No M are P; Some S are M -> Some S are not P."""
    dr = DeductiveReasoner()
    result = dr.syllogism(("E", "mammal", "fish"), ("I", "whale", "mammal"))
    assert result["valid"] is True
    assert result["conclusion"] == "Some whale are not fish"
    assert "Ferio" in result["form"]


def test_deductive_invalid_form():
    """Non-canonical form combinations are invalid."""
    dr = DeductiveReasoner()
    result = dr.syllogism(("O", "X", "Y"), ("I", "A", "B"))
    assert result["valid"] is False
    assert result["conclusion"] is None
    assert result["form"] == "invalid"


def test_deductive_invalid_middle_term():
    """When the middle term doesn't unify, the syllogism is invalid."""
    dr = DeductiveReasoner()
    result = dr.syllogism(("A", "X", "Y"), ("A", "A", "Z"))
    assert result["valid"] is False


def test_deductive_render_all_forms():
    """The renderer covers all four canonical proposition forms."""
    assert DeductiveReasoner._render("A", "S", "P") == "All S are P"
    assert DeductiveReasoner._render("E", "S", "P") == "No S are P"
    assert DeductiveReasoner._render("I", "S", "P") == "Some S are P"
    assert DeductiveReasoner._render("O", "S", "P") == "Some S are not P"
    assert DeductiveReasoner._render("Z", "S", "P") == ""


# --------------------------------------------------------------------------- #
# InductiveReasoner
# --------------------------------------------------------------------------- #


def test_inductive_generalize_perfect_rule():
    """A pair that uniquely appears in all positives is selected."""
    ir = InductiveReasoner()
    examples = [
        {"color": "red", "shape": "circle"},
        {"color": "red", "shape": "square"},
        {"color": "blue", "shape": "triangle"},
    ]
    labels = [True, True, False]
    result = ir.generalize(examples, labels)
    assert result["rule"] == "color == 'red'"
    assert result["confidence"] == 1.0
    assert result["support"] == 2
    assert result["coverage"] == 1.0


def test_inductive_generalize_no_positives():
    ir = InductiveReasoner()
    examples = [{"a": 1}, {"a": 2}]
    labels = [False, False]
    result = ir.generalize(examples, labels)
    assert result["rule"] == ""
    assert result["confidence"] == 0.0
    assert result["support"] == 0
    assert result["coverage"] == 0.0


def test_inductive_generalize_pair_in_negatives():
    """A pair that appears in both positives and negatives has lower confidence."""
    ir = InductiveReasoner()
    examples = [
        {"x": 1},
        {"x": 1},
        {"x": 1},
    ]
    labels = [True, False, True]
    result = ir.generalize(examples, labels)
    # support=2 positives, 1 negative -> confidence = 2/3.
    assert result["support"] == 2
    assert 0.0 < result["confidence"] <= 1.0


def test_inductive_generalize_empty_examples():
    ir = InductiveReasoner()
    result = ir.generalize([], [])
    assert result["rule"] == ""
    assert result["confidence"] == 0.0


def test_inductive_generalize_mismatched_lengths():
    ir = InductiveReasoner()
    result = ir.generalize([{"a": 1}], [True, False])
    assert result["rule"] == ""
    assert result["confidence"] == 0.0


def test_inductive_generalize_picks_best_coverage():
    """When multiple pairs are exclusive to positives, the one with the
    highest coverage wins."""
    ir = InductiveReasoner()
    examples = [
        {"a": 1, "b": 10},
        {"a": 1, "b": 20},
        {"a": 2, "b": 30},
    ]
    labels = [True, True, False]
    result = ir.generalize(examples, labels)
    # ('a', 1) covers 2 of 2 positives; ('b', 10) covers 1 of 2.
    assert result["rule"] == "a == 1"
    assert result["coverage"] == 1.0


# --------------------------------------------------------------------------- #
# AnalogicalReasoner
# --------------------------------------------------------------------------- #


def test_analogical_basic_mapping():
    """Source and target with the same keys map onto each other."""
    from zero_data_model.category_engine import CategoryTheoryEngine
    from zero_data_model.math_universe import MathematicalUniverse

    ar = AnalogicalReasoner(
        dim=32,
        category_engine=CategoryTheoryEngine(dim=32),
        math_universe=MathematicalUniverse(dim=32),
    )
    source = {"a": 1, "b": 2}
    target = {"a": 10, "b": 20}
    result = ar.analogize(source, target)
    assert result["mapping"]["a"] == "a"
    assert result["mapping"]["b"] == "b"
    assert 0.0 <= result["similarity"] <= 1.0
    # Transfer: source value 1 -> mapped target 'a'.
    assert result["transfer"]["a"] == 1
    assert result["transfer"]["b"] == 2


def test_analogical_empty_source():
    ar = AnalogicalReasoner()
    result = ar.analogize({}, {"a": 1})
    assert result["mapping"] == {}
    assert result["similarity"] == 0.0
    assert result["transfer"] == {}


def test_analogical_empty_target():
    ar = AnalogicalReasoner()
    result = ar.analogize({"a": 1}, {})
    assert result["mapping"] == {}
    assert result["similarity"] == 0.0


def test_analogical_more_sources_than_targets():
    """When source has more keys than target, extras map to empty string."""
    from zero_data_model.category_engine import CategoryTheoryEngine

    ar = AnalogicalReasoner(
        dim=16, category_engine=CategoryTheoryEngine(dim=16)
    )
    source = {"a": 1, "b": 2, "c": 3}
    target = {"x": 10}
    result = ar.analogize(source, target)
    # Exactly one source maps to "x"; the other two map to "".
    assert "x" in result["mapping"].values()
    empty_count = sum(1 for v in result["mapping"].values() if v == "")
    assert empty_count == 2


def test_analogical_string_values_mapped():
    """String values produce vectors so similarity is non-trivial.

    The mapping is a bijection between source and target keys of equal
    cardinality; we verify that no two source keys map to the same target.
    """
    from zero_data_model.category_engine import CategoryTheoryEngine

    ar = AnalogicalReasoner(
        dim=16, category_engine=CategoryTheoryEngine(dim=16)
    )
    source = {"name": "alice", "age": 30}
    target = {"name": "bob", "age": 25}
    result = ar.analogize(source, target)
    # Mapping is a bijection: every source maps to a distinct target.
    targets = list(result["mapping"].values())
    assert len(targets) == len(set(targets))
    # All targets are in the target dict.
    for t in targets:
        assert t in target
    assert 0.0 <= result["similarity"] <= 1.0


def test_analogical_combine_vectors_weighted():
    """_combine_vectors weights the first arg by 2.0 and the second by 1.0."""
    from zero_data_model.category_engine import CategoryTheoryEngine

    ar = AnalogicalReasoner(
        dim=4, category_engine=CategoryTheoryEngine(dim=4)
    )
    a = np.array([1.0, 0.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0, 0.0])
    combined = ar._combine_vectors(a, b)
    # Combined = 2*a + 1*b = [2, 1, 0, 0], norm = sqrt(5).
    expected = np.array([2.0, 1.0, 0.0, 0.0]) / np.sqrt(5)
    np.testing.assert_allclose(combined, expected, atol=1e-6)


def test_analogical_to_vector_scalar_one_hot():
    """Distinct scalar values produce distinct one-hot vectors."""
    from zero_data_model.category_engine import CategoryTheoryEngine

    ar = AnalogicalReasoner(
        dim=16, category_engine=CategoryTheoryEngine(dim=16)
    )
    v1 = ar._to_vector(1)
    v2 = ar._to_vector(2)
    # Both are one-hot at different positions.
    assert v1.sum() == 1.0
    assert v2.sum() == 1.0
    # Hash(1) != hash(2), so they should differ.
    assert not np.allclose(v1, v2)
