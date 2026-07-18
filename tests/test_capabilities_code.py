"""Tests for the Code capability domain (base module)."""

import numpy as np
import pytest

from zero_data_model.capabilities import (
    ASTAnalyzer,
    CodeEncoder,
    CodeRules,
    CodeSimilarityChecker,
    DefectPatternDetector,
)


@pytest.fixture(autouse=True)
def _deterministic_rng():
    np.random.seed(42)


SIMPLE_SRC = """
import os
import numpy as np

def foo(x):
    if x is None:
        return 0
    return x + 1

class Bar:
    def method_a(self):
        return 42
"""

# --------------------------------------------------------------------------- #
# CodeRules
# --------------------------------------------------------------------------- #


def test_code_rules_defaults():
    rules = CodeRules()
    assert rules.max_line_length == 100
    assert rules.indent_size == 4
    assert rules.max_function_complexity == 10
    assert rules.max_function_lines == 50
    assert rules.naming_convention == "snake_case"
    assert rules.defect_patterns == ("mutable_default", "bare_except", "equals_none")


def test_code_rules_post_init_populates_dict():
    rules = CodeRules()
    assert rules.rules["max_line_length"] == 100
    assert rules.rules["naming_convention"] == "snake_case"


def test_code_rules_custom_values():
    rules = CodeRules(max_line_length=80, indent_size=2)
    assert rules.max_line_length == 80
    assert rules.indent_size == 2


# --------------------------------------------------------------------------- #
# CodeEncoder
# --------------------------------------------------------------------------- #


class TestCodeEncoder:
    def test_encode_shape_and_norm(self):
        enc = CodeEncoder(dim=32)
        v = enc.encode(SIMPLE_SRC)
        assert v.shape == (32,)
        norm = float(np.linalg.norm(v))
        np.testing.assert_allclose(norm, 1.0, atol=1e-6)

    def test_encode_empty_returns_zeros(self):
        enc = CodeEncoder(dim=16)
        v = enc.encode("")
        assert v.shape == (16,)
        np.testing.assert_allclose(v, 0.0)

    def test_encode_no_nan_for_valid_source(self):
        enc = CodeEncoder(dim=16)
        v = enc.encode("x = 1")
        assert np.all(np.isfinite(v))

    def test_encode_deterministic(self):
        enc = CodeEncoder(dim=16)
        v1 = enc.encode(SIMPLE_SRC)
        v2 = enc.encode(SIMPLE_SRC)
        np.testing.assert_allclose(v1, v2)

    def test_encode_different_sources_different_vectors(self):
        enc = CodeEncoder(dim=32)
        v1 = enc.encode("def f():\n    pass\n")
        v2 = enc.encode("class C:\n    pass\n" * 5)
        # Two very different sources should not produce identical encodings.
        assert not np.allclose(v1, v2)

    def test_encode_syntax_error_no_crash(self):
        # Invalid Python should not crash; falls back to no AST nodes.
        enc = CodeEncoder(dim=16)
        v = enc.encode("def def def ::::")
        assert v.shape == (16,)
        assert np.all(np.isfinite(v))


# --------------------------------------------------------------------------- #
# ASTAnalyzer
# --------------------------------------------------------------------------- #


class TestASTAnalyzer:
    def test_analyze_extracts_functions(self):
        ana = ASTAnalyzer(dim=16)
        result = ana.analyze(SIMPLE_SRC)
        assert "foo" in result["functions"]
        assert "method_a" in result["functions"]

    def test_analyze_extracts_classes(self):
        ana = ASTAnalyzer(dim=16)
        result = ana.analyze(SIMPLE_SRC)
        assert "Bar" in result["classes"]

    def test_analyze_extracts_imports(self):
        ana = ASTAnalyzer(dim=16)
        result = ana.analyze(SIMPLE_SRC)
        assert "os" in result["imports"]
        assert "numpy" in result["imports"]

    def test_analyze_complexity_positive(self):
        ana = ASTAnalyzer(dim=16)
        result = ana.analyze(SIMPLE_SRC)
        # Base complexity = 1; the ``if`` adds 1.
        assert result["complexity"] >= 2

    def test_analyze_max_depth_positive(self):
        ana = ASTAnalyzer(dim=16)
        result = ana.analyze(SIMPLE_SRC)
        assert result["max_depth"] > 0
        assert result["node_count"] > 0

    def test_analyze_empty_source_returns_zeros(self):
        ana = ASTAnalyzer(dim=16)
        result = ana.analyze("")
        assert result["functions"] == []
        assert result["classes"] == []
        assert result["imports"] == []
        assert result["complexity"] == 0
        assert result["max_depth"] == 0
        assert result["node_count"] == 0

    def test_analyze_syntax_error_returns_zeros(self):
        ana = ASTAnalyzer(dim=16)
        result = ana.analyze("def def def ::::")
        assert result["functions"] == []
        assert result["complexity"] == 0

    def test_analyze_comprehension_adds_complexity(self):
        ana = ASTAnalyzer(dim=16)
        src = "y = [x for x in range(10) if x > 0]\n"
        result = ana.analyze(src)
        # ListComp + IfExp inside comp -> complexity > 1.
        assert result["complexity"] >= 2

    def test_analyze_boolean_op_adds_complexity(self):
        ana = ASTAnalyzer(dim=16)
        src = "z = a and b and c\n"
        result = ana.analyze(src)
        # BoolOp with 3 values -> +2 paths.
        assert result["complexity"] >= 3


# --------------------------------------------------------------------------- #
# CodeSimilarityChecker
# --------------------------------------------------------------------------- #


class TestCodeSimilarityChecker:
    def test_compare_identical_sources_similarity_one(self):
        sim = CodeSimilarityChecker(dim=16)
        result = sim.compare(SIMPLE_SRC, SIMPLE_SRC)
        np.testing.assert_allclose(result["similarity"], 1.0, atol=1e-9)
        np.testing.assert_allclose(result["token_overlap"], 1.0, atol=1e-9)
        np.testing.assert_allclose(result["structure_similarity"], 1.0, atol=1e-9)

    def test_compare_disjoint_sources_low_overlap(self):
        sim = CodeSimilarityChecker(dim=16)
        result = sim.compare("def alpha():\n    pass\n", "def beta():\n    pass\n")
        # Tokens differ but structure is similar; overall < 1.
        assert result["similarity"] < 1.0

    def test_compare_empty_sources_structure_identical(self):
        # Two empty sources parse to a trivial Module node each -> AST
        # structures are identical; token overlap is 0; overall similarity 0.5.
        sim = CodeSimilarityChecker(dim=16)
        result = sim.compare("", "")
        np.testing.assert_allclose(result["structure_similarity"], 1.0, atol=1e-9)
        np.testing.assert_allclose(result["token_overlap"], 0.0, atol=1e-9)
        np.testing.assert_allclose(result["similarity"], 0.5, atol=1e-9)

    def test_compare_returns_in_unit_interval(self):
        sim = CodeSimilarityChecker(dim=16)
        result = sim.compare("def f():\n    pass\n", "class C:\n    pass\n")
        assert 0.0 <= result["similarity"] <= 1.0
        assert 0.0 <= result["token_overlap"] <= 1.0
        assert 0.0 <= result["structure_similarity"] <= 1.0

    def test_compare_symmetric(self):
        sim = CodeSimilarityChecker(dim=16)
        a = "def foo():\n    return 1\n"
        b = "def bar():\n    return 2\n"
        r1 = sim.compare(a, b)
        r2 = sim.compare(b, a)
        np.testing.assert_allclose(r1["similarity"], r2["similarity"])

    def test_compare_syntax_error_no_crash(self):
        sim = CodeSimilarityChecker(dim=16)
        result = sim.compare("def ::::", "valid = 1")
        # First source fails to parse; structure_similarity falls back to 0.
        assert result["similarity"] >= 0.0


# --------------------------------------------------------------------------- #
# DefectPatternDetector
# --------------------------------------------------------------------------- #


class TestDefectPatternDetector:
    def test_detect_mutable_default_list(self):
        det = DefectPatternDetector(dim=16)
        result = det.detect("def f(x=[]):\n    return x\n")
        patterns = [d["pattern"] for d in result["defects"]]
        assert "mutable_default" in patterns

    def test_detect_mutable_default_dict(self):
        det = DefectPatternDetector(dim=16)
        result = det.detect("def f(x={}):\n    return x\n")
        patterns = [d["pattern"] for d in result["defects"]]
        assert "mutable_default" in patterns

    def test_detect_mutable_default_set(self):
        det = DefectPatternDetector(dim=16)
        # Use a literal set ``{1, 2}`` (ast.Set), not ``set()`` (ast.Call).
        result = det.detect("def f(x={1, 2}):\n    return x\n")
        patterns = [d["pattern"] for d in result["defects"]]
        assert "mutable_default" in patterns

    def test_detect_bare_except(self):
        det = DefectPatternDetector(dim=16)
        result = det.detect("try:\n    pass\nexcept:\n    pass\n")
        patterns = [d["pattern"] for d in result["defects"]]
        assert "bare_except" in patterns

    def test_detect_typed_except_not_flagged(self):
        det = DefectPatternDetector(dim=16)
        result = det.detect("try:\n    pass\nexcept ValueError:\n    pass\n")
        patterns = [d["pattern"] for d in result["defects"]]
        assert "bare_except" not in patterns

    def test_detect_equals_none(self):
        det = DefectPatternDetector(dim=16)
        result = det.detect("def f(x):\n    if x == None:\n        return 0\n")
        patterns = [d["pattern"] for d in result["defects"]]
        assert "equals_none" in patterns

    def test_detect_is_none_not_flagged(self):
        det = DefectPatternDetector(dim=16)
        result = det.detect("def f(x):\n    if x is None:\n        return 0\n")
        patterns = [d["pattern"] for d in result["defects"]]
        assert "equals_none" not in patterns

    def test_detect_clean_source_no_defects(self):
        det = DefectPatternDetector(dim=16)
        result = det.detect("def f(x=None):\n    if x is None:\n        return 0\n    return x\n")
        assert result["total"] == 0

    def test_detect_empty_source_returns_empty(self):
        det = DefectPatternDetector(dim=16)
        result = det.detect("")
        assert result["defects"] == []
        assert result["total"] == 0

    def test_detect_syntax_error_returns_empty(self):
        det = DefectPatternDetector(dim=16)
        result = det.detect("def ::::")
        assert result["total"] == 0

    def test_detect_defect_has_line_and_snippet(self):
        det = DefectPatternDetector(dim=16)
        result = det.detect("def f(x=[]):\n    return x\n")
        defect = next(d for d in result["defects"] if d["pattern"] == "mutable_default")
        assert defect["line"] == 1
        assert "x=[]" in defect["snippet"]
