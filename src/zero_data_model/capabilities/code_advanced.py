# src/zero_data_model/capabilities/code_advanced.py
"""Advanced code analysis capabilities for the zero-data cognitive model.

Composes the core cognitive modules (math_universe for fractal features)
with ``CodeRules``. All operators use Python's stdlib ``ast`` -- no
external linting dependencies.
"""

from __future__ import annotations

import ast
import re

import numpy as np

from .code import ASTAnalyzer
from .rules import CodeRules


class ControlFlowAnalyzer:
    """Extract basic blocks and per-function cyclomatic complexity.

    For each function in the source, computes its cyclomatic complexity
    (decision points + 1) and line count. Returns the average and max
    complexity across all functions.
    """

    def __init__(self, dim: int = 64, rules: CodeRules | None = None):
        self.dim = dim
        self.rules = rules or CodeRules()
        self._ast = ASTAnalyzer(dim=dim, rules=rules)

    def analyze(self, source: str) -> dict:
        """Analyze control flow in ``source``.

        Returns ``{'functions', 'avg_complexity', 'max_complexity',
        'total_functions'}`` where each function is a dict with
        ``{'name', 'complexity', 'lines', 'basic_blocks'}``.
        """
        if not source:
            return {
                "functions": [],
                "avg_complexity": 0.0,
                "max_complexity": 0,
                "total_functions": 0,
            }
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return {
                "functions": [],
                "avg_complexity": 0.0,
                "max_complexity": 0,
                "total_functions": 0,
            }
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                complexity = self._function_complexity(node)
                # Line count: from function start to last child's end.
                end_line = node.lineno
                for child in ast.walk(node):
                    if hasattr(child, "lineno") and child.lineno is not None:
                        end_line = max(end_line, child.lineno)
                lines = end_line - node.lineno + 1
                # Basic blocks approximation: complexity + 1 (entry block).
                basic_blocks = complexity + 1
                functions.append({
                    "name": node.name,
                    "complexity": int(complexity),
                    "lines": int(lines),
                    "basic_blocks": int(basic_blocks),
                })
        total = len(functions)
        if total == 0:
            return {
                "functions": [],
                "avg_complexity": 0.0,
                "max_complexity": 0,
                "total_functions": 0,
            }
        complexities = [f["complexity"] for f in functions]
        avg = float(np.mean(complexities)) if complexities else 0.0
        max_c = int(max(complexities)) if complexities else 0
        return {
            "functions": functions,
            "avg_complexity": avg,
            "max_complexity": max_c,
            "total_functions": total,
        }

    @staticmethod
    def _function_complexity(node: ast.AST) -> int:
        """Cyclomatic complexity: decision points + 1, scoped to a function."""
        complexity = 1
        for child in ast.walk(node):
            if child is node:
                continue
            if isinstance(
                child,
                ast.If
                | ast.IfExp
                | ast.For
                | ast.While
                | ast.AsyncFor
                | ast.ExceptHandler,
            ):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += max(0, len(child.values) - 1)
            elif isinstance(
                child,
                ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp,
            ):
                complexity += 1
        return complexity


class CodeStyleAnalyzer:
    """Rule-based style checks: line length, naming, indentation, whitespace.

    Returns a list of style violations and a normalized score in ``[0, 1]``
    (1.0 = no violations).
    """

    def __init__(self, dim: int = 64, rules: CodeRules | None = None):
        self.dim = dim
        self.rules = rules or CodeRules()

    def analyze(self, source: str) -> dict:
        """Analyze style in ``source``.

        Returns ``{'violations', 'total', 'score'}`` where each violation
        is ``{'rule', 'line', 'message'}``.
        """
        if not source:
            return {
                "violations": [],
                "total": 0,
                "score": 1.0,
            }
        lines = source.splitlines()
        violations = []
        for i, line in enumerate(lines, start=1):
            # Line length.
            if len(line) > self.rules.max_line_length:
                violations.append({
                    "rule": "line_too_long",
                    "line": i,
                    "message": f"line exceeds {self.rules.max_line_length} chars",
                })
            # Trailing whitespace.
            if line != line.rstrip():
                violations.append({
                    "rule": "trailing_whitespace",
                    "line": i,
                    "message": "trailing whitespace",
                })
            # Mixed tabs and spaces in indentation.
            leading = line[: len(line) - len(line.lstrip())]
            if "\t" in leading and " " in leading:
                violations.append({
                    "rule": "mixed_indent",
                    "line": i,
                    "message": "mixed tabs and spaces in indentation",
                })
        # Naming conventions: function names should be snake_case.
        try:
            tree = ast.parse(source)
        except SyntaxError:
            tree = None
        if tree is not None:
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
                    and not self._is_snake_case(node.name)
                ):
                    violations.append({
                        "rule": "naming_convention",
                        "line": node.lineno,
                        "message": f"function '{node.name}' not in snake_case",
                    })
        total = len(violations)
        # Score: 1 - violations / max(1, total_lines).
        n_lines = max(1, len(lines))
        score = max(0.0, 1.0 - total / n_lines)
        return {
            "violations": violations,
            "total": int(total),
            "score": float(np.clip(score, 0.0, 1.0)),
        }

    @staticmethod
    def _is_snake_case(name: str) -> bool:
        """Check if ``name`` follows snake_case convention."""
        return bool(re.match(r"^[a-z_][a-z0-9_]*$", name))


class DependencyGraphBuilder:
    """Build a module-level import dependency graph.

    Walks the AST for Import and ImportFrom nodes, extracts module names,
    and builds an adjacency matrix where ``adj[i, j] = 1`` if module i
    imports module j.
    """

    def __init__(self, dim: int = 64, rules: CodeRules | None = None):
        self.dim = dim
        self.rules = rules or CodeRules()

    def build(self, source: str) -> dict:
        """Build a dependency graph from imports in ``source``.

        Returns ``{'nodes', 'edges', 'adjacency', 'n_modules'}``.
        """
        if not source:
            return {
                "nodes": [],
                "edges": [],
                "adjacency": np.zeros((0, 0)),
                "n_modules": 0,
            }
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return {
                "nodes": [],
                "edges": [],
                "adjacency": np.zeros((0, 0)),
                "n_modules": 0,
            }
        # Collect imports.
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module:
                    imports.append(module)
        # Unique nodes (sorted for determinism).
        nodes = sorted(set(imports))
        if not nodes:
            return {
                "nodes": [],
                "edges": [],
                "adjacency": np.zeros((0, 0)),
                "n_modules": 0,
            }
        # Treat the source file as a central node "__main__" that imports
        # each imported module.
        all_nodes = ["__main__"] + nodes
        n_total = len(all_nodes)
        adjacency = np.zeros((n_total, n_total), dtype=float)
        edges = []
        for imp in imports:
            i = 0  # __main__
            j = all_nodes.index(imp)
            adjacency[i, j] = 1.0
            edges.append((all_nodes[i], all_nodes[j]))
        return {
            "nodes": all_nodes,
            "edges": edges,
            "adjacency": adjacency,
            "n_modules": int(n_total),
        }
