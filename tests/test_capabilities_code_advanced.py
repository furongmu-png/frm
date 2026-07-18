"""Tests for the Code capability domain (advanced module)."""

import numpy as np
import pytest

from zero_data_model.capabilities import (
    CodeRules,
    CodeStyleAnalyzer,
    ControlFlowAnalyzer,
    DependencyGraphBuilder,
)


@pytest.fixture(autouse=True)
def _deterministic_rng():
    np.random.seed(42)


# Source with two functions of differing complexity.
SAMPLE_SRC = """
import os
import numpy as np
from typing import List

def simple(x):
    return x + 1

def complex_func(items):
    total = 0
    for item in items:
        if item > 0:
            total += item
        else:
            total -= 1
    return total

class MyClass:
    def method_one(self):
        pass
"""

# --------------------------------------------------------------------------- #
# ControlFlowAnalyzer
# --------------------------------------------------------------------------- #


class TestControlFlowAnalyzer:
    def test_analyze_returns_function_list(self):
        cfa = ControlFlowAnalyzer(dim=16)
        result = cfa.analyze(SAMPLE_SRC)
        names = [f["name"] for f in result["functions"]]
        assert "simple" in names
        assert "complex_func" in names
        assert "method_one" in names

    def test_analyze_complex_function_higher_complexity(self):
        cfa = ControlFlowAnalyzer(dim=16)
        result = cfa.analyze(SAMPLE_SRC)
        by_name = {f["name"]: f for f in result["functions"]}
        assert by_name["complex_func"]["complexity"] > by_name["simple"]["complexity"]

    def test_analyze_max_complexity_correct(self):
        cfa = ControlFlowAnalyzer(dim=16)
        result = cfa.analyze(SAMPLE_SRC)
        # complex_func has for + if + else (except handler none) -> base 1 + 3 = 4.
        assert result["max_complexity"] >= 3

    def test_analyze_avg_complexity_is_mean(self):
        cfa = ControlFlowAnalyzer(dim=16)
        result = cfa.analyze(SAMPLE_SRC)
        expected = float(np.mean([f["complexity"] for f in result["functions"]]))
        np.testing.assert_allclose(result["avg_complexity"], expected)

    def test_analyze_returns_total_functions(self):
        cfa = ControlFlowAnalyzer(dim=16)
        result = cfa.analyze(SAMPLE_SRC)
        assert result["total_functions"] == len(result["functions"])
        assert result["total_functions"] == 3

    def test_analyze_empty_source_returns_zeros(self):
        cfa = ControlFlowAnalyzer(dim=16)
        result = cfa.analyze("")
        assert result["functions"] == []
        assert result["avg_complexity"] == 0.0
        assert result["max_complexity"] == 0
        assert result["total_functions"] == 0

    def test_analyze_syntax_error_returns_zeros(self):
        cfa = ControlFlowAnalyzer(dim=16)
        result = cfa.analyze("def ::::")
        assert result["total_functions"] == 0

    def test_analyze_basic_blocks_is_complexity_plus_one(self):
        cfa = ControlFlowAnalyzer(dim=16)
        result = cfa.analyze(SAMPLE_SRC)
        for f in result["functions"]:
            assert f["basic_blocks"] == f["complexity"] + 1

    def test_analyze_lines_positive(self):
        cfa = ControlFlowAnalyzer(dim=16)
        result = cfa.analyze(SAMPLE_SRC)
        for f in result["functions"]:
            assert f["lines"] >= 1

    def test_analyze_no_functions_returns_zeros(self):
        cfa = ControlFlowAnalyzer(dim=16)
        result = cfa.analyze("x = 1\ny = 2\n")
        assert result["total_functions"] == 0
        assert result["max_complexity"] == 0


# --------------------------------------------------------------------------- #
# CodeStyleAnalyzer
# --------------------------------------------------------------------------- #


class TestCodeStyleAnalyzer:
    def test_analyze_clean_source_high_score(self):
        csa = CodeStyleAnalyzer(dim=16)
        result = csa.analyze("def foo():\n    return 1\n")
        # Clean source: no violations.
        assert result["total"] == 0
        np.testing.assert_allclose(result["score"], 1.0)

    def test_analyze_long_line_violation(self):
        csa = CodeStyleAnalyzer(dim=16)
        long_line = "x = " + "a" * 200
        result = csa.analyze(long_line)
        rules = [v["rule"] for v in result["violations"]]
        assert "line_too_long" in rules

    def test_analyze_trailing_whitespace_violation(self):
        csa = CodeStyleAnalyzer(dim=16)
        result = csa.analyze("x = 1   \n")
        rules = [v["rule"] for v in result["violations"]]
        assert "trailing_whitespace" in rules

    def test_analyze_mixed_indent_violation(self):
        csa = CodeStyleAnalyzer(dim=16)
        # Mix of tab and space in indentation.
        result = csa.analyze("def f():\n\t  return 1\n")
        rules = [v["rule"] for v in result["violations"]]
        assert "mixed_indent" in rules

    def test_analyze_camel_case_naming_violation(self):
        csa = CodeStyleAnalyzer(dim=16)
        result = csa.analyze("def camelCase():\n    pass\n")
        rules = [v["rule"] for v in result["violations"]]
        assert "naming_convention" in rules

    def test_analyze_snake_case_no_naming_violation(self):
        csa = CodeStyleAnalyzer(dim=16)
        result = csa.analyze("def my_func():\n    pass\n")
        rules = [v["rule"] for v in result["violations"]]
        assert "naming_convention" not in rules

    def test_analyze_score_in_unit_interval(self):
        csa = CodeStyleAnalyzer(dim=16)
        result = csa.analyze(SAMPLE_SRC)
        assert 0.0 <= result["score"] <= 1.0

    def test_analyze_empty_source_returns_perfect_score(self):
        csa = CodeStyleAnalyzer(dim=16)
        result = csa.analyze("")
        assert result["total"] == 0
        np.testing.assert_allclose(result["score"], 1.0)

    def test_analyze_custom_max_line_length(self):
        rules = CodeRules(max_line_length=10)
        csa = CodeStyleAnalyzer(dim=16, rules=rules)
        result = csa.analyze("x = 12345678901\n")  # 15 chars
        rules_violated = [v["rule"] for v in result["violations"]]
        assert "line_too_long" in rules_violated

    def test_analyze_violations_include_line_numbers(self):
        csa = CodeStyleAnalyzer(dim=16)
        result = csa.analyze("x = 1   \n")
        for v in result["violations"]:
            assert v["line"] >= 1


# --------------------------------------------------------------------------- #
# DependencyGraphBuilder
# --------------------------------------------------------------------------- #


class TestDependencyGraphBuilder:
    def test_build_extracts_imports_as_nodes(self):
        dgb = DependencyGraphBuilder(dim=16)
        result = dgb.build(SAMPLE_SRC)
        assert "os" in result["nodes"]
        assert "numpy" in result["nodes"]
        assert "typing" in result["nodes"]

    def test_build_includes_main_node(self):
        dgb = DependencyGraphBuilder(dim=16)
        result = dgb.build(SAMPLE_SRC)
        assert "__main__" in result["nodes"]

    def test_build_adjacency_shape(self):
        dgb = DependencyGraphBuilder(dim=16)
        result = dgb.build(SAMPLE_SRC)
        n = result["n_modules"]
        assert result["adjacency"].shape == (n, n)

    def test_build_main_imports_all(self):
        dgb = DependencyGraphBuilder(dim=16)
        result = dgb.build(SAMPLE_SRC)
        # __main__ is at index 0; it should import every other module.
        adj = result["adjacency"]
        for j in range(1, result["n_modules"]):
            assert adj[0, j] == 1.0

    def test_build_modules_dont_import_each_other(self):
        dgb = DependencyGraphBuilder(dim=16)
        result = dgb.build(SAMPLE_SRC)
        adj = result["adjacency"]
        # Modules other than __main__ should not import anything.
        n = result["n_modules"]
        for i in range(1, n):
            for j in range(n):
                assert adj[i, j] == 0.0

    def test_build_returns_edges_list(self):
        dgb = DependencyGraphBuilder(dim=16)
        result = dgb.build(SAMPLE_SRC)
        # Each edge is a tuple ("__main__", "<module>").
        assert len(result["edges"]) >= 3
        for edge in result["edges"]:
            assert edge[0] == "__main__"

    def test_build_empty_source_returns_empty(self):
        dgb = DependencyGraphBuilder(dim=16)
        result = dgb.build("")
        assert result["nodes"] == []
        assert result["edges"] == []
        assert result["n_modules"] == 0
        assert result["adjacency"].shape == (0, 0)

    def test_build_syntax_error_returns_empty(self):
        dgb = DependencyGraphBuilder(dim=16)
        result = dgb.build("def ::::")
        assert result["n_modules"] == 0

    def test_build_deduplicates_imports(self):
        dgb = DependencyGraphBuilder(dim=16)
        src = "import os\nimport os\nfrom os import path\n"
        result = dgb.build(src)
        # os should appear once in nodes.
        assert result["nodes"].count("os") == 1

    def test_build_no_imports_returns_empty(self):
        dgb = DependencyGraphBuilder(dim=16)
        result = dgb.build("x = 1\ny = 2\n")
        assert result["n_modules"] == 0
        assert result["edges"] == []
