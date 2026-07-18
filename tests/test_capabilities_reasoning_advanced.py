"""Tests for the Reasoning capability domain (advanced module)."""

import numpy as np
import pytest

from zero_data_model.capabilities import (
    AbductiveReasoner,
    CausalChainReasoner,
    DefeasibleReasoner,
)


@pytest.fixture(autouse=True)
def _deterministic_rng():
    np.random.seed(42)


# --------------------------------------------------------------------------- #
# AbductiveReasoner
# --------------------------------------------------------------------------- #


def test_abductive_explain_best_match():
    """The hypothesis with highest keyword overlap wins."""
    abr = AbductiveReasoner()
    hypotheses = ["common cold with fever", "broken leg", "flu with fever and cough"]
    result = abr.explain(
        "patient has fever and cough",
        hypotheses,
        priors=[0.3, 0.1, 0.6],
    )
    assert result["best"] == "flu with fever and cough"
    assert len(result["scores"]) == 3
    assert 0.0 <= result["confidence"] <= 1.0
    # The winning hypothesis should have the highest score.
    best_idx = result["scores"].index(max(result["scores"]))
    assert result["best"] == hypotheses[best_idx]


def test_abductive_explain_no_hypotheses():
    abr = AbductiveReasoner()
    result = abr.explain("anything", [], None)
    assert result["best"] is None
    assert result["scores"] == []
    assert result["confidence"] == 0.0


def test_abductive_explain_default_priors():
    """Without priors, hypotheses are weighted uniformly."""
    abr = AbductiveReasoner()
    result = abr.explain("fever", ["flu fever", "cold"])
    assert len(result["scores"]) == 2
    # The flu hypothesis has overlap with "fever", the cold does not.
    assert result["scores"][0] > result["scores"][1]
    assert result["best"] == "flu fever"


def test_abductive_explain_no_overlap():
    """When nothing overlaps, the first hypothesis wins with zero confidence."""
    abr = AbductiveReasoner()
    result = abr.explain("alpha", ["beta", "gamma"], [0.5, 0.5])
    assert result["best"] == "beta"
    assert result["confidence"] == 0.0
    assert result["scores"] == [0.0, 0.0]


def test_abductive_explain_mismatched_priors():
    abr = AbductiveReasoner()
    result = abr.explain("x", ["a", "b"], [0.5])
    assert result["best"] is None
    assert result["scores"] == []
    assert result["confidence"] == 0.0


def test_abductive_tokenize():
    tokens = AbductiveReasoner._tokenize("Hello, World! 123")
    assert tokens == {"hello", "world", "123"}


def test_abductive_tokenize_empty():
    assert AbductiveReasoner._tokenize("") == set()


def test_abductive_keyword_overlap():
    a = {"a", "b", "c"}
    b = {"b", "c", "d"}
    # Intersection = {b, c}, union = {a, b, c, d} -> 2/4 = 0.5.
    assert AbductiveReasoner._keyword_overlap(a, b) == 0.5


def test_abductive_keyword_overlap_empty():
    assert AbductiveReasoner._keyword_overlap(set(), {"a"}) == 0.0
    assert AbductiveReasoner._keyword_overlap({"a"}, set()) == 0.0


def test_abductive_confidence_normalization():
    """Confidence = best_score / sum(scores), in [0, 1]."""
    abr = AbductiveReasoner()
    result = abr.explain(
        "fever",
        ["fever flu", "cold"],
        [1.0, 1.0],
    )
    # Only "fever flu" overlaps with "fever".
    assert result["scores"][0] > 0
    assert result["scores"][1] == 0.0
    assert result["confidence"] == 1.0  # best / (best + 0) = 1.0


# --------------------------------------------------------------------------- #
# DefeasibleReasoner
# --------------------------------------------------------------------------- #


def test_defeasible_basic_default():
    df = DefeasibleReasoner()
    df.add_default(("bird", "flies", True))
    result = df.conclude({"bird": True})
    assert result["conclusions"]["flies"] is True
    assert result["defeated"] == []
    assert result["ambiguous"] == []


def test_defeasible_exception_blocks_rule():
    df = DefeasibleReasoner()
    df.add_default(("bird", "flies", True), exception=("penguin",))
    result = df.conclude({"bird": True, "penguin": True})
    assert "flies" not in result["conclusions"]
    assert ("bird", "flies") in result["defeated"]


def test_defeasible_antecedent_false_no_fire():
    df = DefeasibleReasoner()
    df.add_default(("bird", "flies", True))
    result = df.conclude({"bird": False})
    assert "flies" not in result["conclusions"]
    assert result["defeated"] == []


def test_defeasible_ambiguous_conflict():
    """Two rules concluding opposite values for the same prop -> ambiguous."""
    df = DefeasibleReasoner()
    df.add_default(("bird", "flies", True))
    df.add_default(("bird", "flies", False))
    result = df.conclude({"bird": True})
    assert "flies" in result["ambiguous"]
    assert "flies" not in result["conclusions"]


def test_defeasible_no_exception_when_exception_is_false():
    """Rule fires when antecedent is True and exception is False/absent."""
    df = DefeasibleReasoner()
    df.add_default(("bird", "flies", True), exception=("penguin",))
    result = df.conclude({"bird": True, "penguin": False})
    assert result["conclusions"]["flies"] is True


def test_defeasible_add_default_invalid_rule():
    """A rule with fewer than 2 elements is silently dropped."""
    df = DefeasibleReasoner()
    df.add_default(("bird",))  # too short
    df.add_default(())  # empty
    df.add_default(("bird", "flies", True))  # valid
    result = df.conclude({"bird": True})
    assert result["conclusions"]["flies"] is True


def test_defeasible_default_value_true_when_omitted():
    """The value defaults to True when not provided."""
    df = DefeasibleReasoner()
    df.add_default(("bird", "flies"))  # value omitted
    result = df.conclude({"bird": True})
    assert result["conclusions"]["flies"] is True


def test_defeasible_multiple_conclusions():
    df = DefeasibleReasoner()
    df.add_default(("bird", "flies", True))
    df.add_default(("bird", "has_feathers", True))
    df.add_default(("bird", "lays_eggs", True))
    result = df.conclude({"bird": True})
    assert len(result["conclusions"]) == 3
    assert result["conclusions"]["flies"] is True
    assert result["conclusions"]["has_feathers"] is True
    assert result["conclusions"]["lays_eggs"] is True


def test_defeasible_empty_rules():
    df = DefeasibleReasoner()
    result = df.conclude({"bird": True})
    assert result["conclusions"] == {}
    assert result["defeated"] == []
    assert result["ambiguous"] == []


# --------------------------------------------------------------------------- #
# CausalChainReasoner
# --------------------------------------------------------------------------- #


def test_causal_chain_simple_trace():
    cc = CausalChainReasoner()
    cc.add_causal("a", "b")
    cc.add_causal("b", "c")
    result = cc.trace("a", max_depth=5)
    assert result["chain"] == ["a", "b", "c"]
    assert "b" in result["effects"]
    assert "c" in result["effects"]
    assert result["depth"] == 2
    assert result["cycles"] is False


def test_causal_chain_unknown_start():
    cc = CausalChainReasoner()
    result = cc.trace("nonexistent", max_depth=5)
    assert result["chain"] == ["nonexistent"]
    assert result["effects"] == []
    assert result["depth"] == 0
    assert result["cycles"] is False


def test_causal_chain_cycle_detection():
    cc = CausalChainReasoner()
    cc.add_causal("a", "b")
    cc.add_causal("b", "c")
    cc.add_causal("c", "a")  # back-edge: cycle.
    result = cc.trace("a", max_depth=10)
    assert result["cycles"] is True
    # The chain should not infinitely loop.
    assert "a" in result["chain"]
    assert "b" in result["chain"]
    assert "c" in result["chain"]


def test_causal_chain_max_depth():
    """Trace stops at max_depth even if more effects exist."""
    cc = CausalChainReasoner()
    cc.add_causal("a", "b")
    cc.add_causal("b", "c")
    cc.add_causal("c", "d")
    cc.add_causal("d", "e")
    result = cc.trace("a", max_depth=1)
    # max_depth=1 means we visit a, then go 1 deep to b, but not beyond.
    assert "a" in result["chain"]
    assert "b" in result["chain"]
    # depth should be at most 1.
    assert result["depth"] <= 1


def test_causal_chain_branching():
    cc = CausalChainReasoner()
    cc.add_causal("a", "b")
    cc.add_causal("a", "c")
    cc.add_causal("b", "d")
    cc.add_causal("c", "e")
    result = cc.trace("a", max_depth=5)
    # All children should be visited.
    assert set(result["effects"]) == {"b", "c", "d", "e"}
    assert result["cycles"] is False


def test_causal_chain_empty():
    cc = CausalChainReasoner()
    result = cc.trace("nothing", max_depth=5)
    assert result["chain"] == ["nothing"]
    assert result["effects"] == []


def test_causal_chain_add_causal_appends():
    """Multiple edges from the same cause are all traced."""
    cc = CausalChainReasoner()
    cc.add_causal("a", "b")
    cc.add_causal("a", "c")
    cc.add_causal("a", "d")
    result = cc.trace("a", max_depth=5)
    assert set(result["effects"]) == {"b", "c", "d"}


def test_causal_chain_string_coercion():
    """Non-string inputs are coerced to str."""
    cc = CausalChainReasoner()
    cc.add_causal(1, 2)
    cc.add_causal(2, 3)
    result = cc.trace(1, max_depth=5)
    assert "1" in result["chain"]
    assert "2" in result["chain"]
    assert "3" in result["chain"]


def test_causal_chain_returns_dict_structure():
    cc = CausalChainReasoner()
    cc.add_causal("x", "y")
    result = cc.trace("x", max_depth=5)
    assert set(result.keys()) == {"chain", "effects", "depth", "cycles"}
    assert isinstance(result["chain"], list)
    assert isinstance(result["effects"], list)
    assert isinstance(result["depth"], int)
    assert isinstance(result["cycles"], bool)


def test_causal_chain_self_loop():
    """A self-edge a -> a is detected as a cycle."""
    cc = CausalChainReasoner()
    cc.add_causal("a", "a")
    result = cc.trace("a", max_depth=5)
    assert result["cycles"] is True
    assert result["chain"][0] == "a"
