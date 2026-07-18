# src/zero_data_model/capabilities/code.py
"""Code analysis capability module for the zero-data cognitive model.

Composes the core cognitive modules (math universe for fractal compression)
with a small rule library (``CodeRules``: line length, complexity limits,
naming conventions). Uses only Python's stdlib ``ast`` module for parsing --
no external linting libraries (pylint / flake8 / black) are required.
"""

from __future__ import annotations

import ast
import re
from collections import Counter

import numpy as np

from ..math_universe import MathematicalUniverse
from .rules import CodeRules


class CodeEncoder:
    """Encode Python source code into a ``dim``-length L2-normalized vector.

    Pipeline: tokenize -> count features (token count, char count, line
    count, indent depth, keyword density, AST node count) -> fractal
    compression -> dim vector.
    """

    def __init__(
        self,
        dim: int = 64,
        math_universe: MathematicalUniverse | None = None,
        rules: CodeRules | None = None,
    ):
        self.dim = dim
        self.rules = rules or CodeRules()
        self.math_universe = math_universe or MathematicalUniverse(dim=dim)

    def encode(self, source: str) -> np.ndarray:
        """Encode ``source`` into a ``dim``-length L2-normalized vector."""
        if not source:
            return np.zeros(self.dim, dtype=float)
        scalars = self._extract_scalars(source)
        if scalars.size == 0:
            return np.zeros(self.dim, dtype=float)
        # Fractal compression of the scalar feature vector. Suppress numpy
        # divide-by-zero warnings from np.corrcoef inside
        # math_universe.fractal.compress when the input is degenerate; we
        # sanitize the output below.
        with np.errstate(invalid="ignore", divide="ignore"):
            compressed = self.math_universe.fractal.compress(scalars)
        # Combine raw scalars with fractal-derived statistics (matches the
        # vision encoder pattern: raw features + fractal stats). The fractal
        # dict from math_universe has keys {'mean', 'std', 'self_similarity'}.
        fractal_stats = (
            np.array(
                [
                    float(compressed.get("mean", 0.0)),
                    float(compressed.get("std", 0.0)),
                    float(compressed.get("self_similarity", 0.0)),
                ],
                dtype=float,
            )
            if isinstance(compressed, dict)
            else np.asarray(compressed, dtype=float).flatten()
        )
        combined = np.concatenate([scalars, fractal_stats])
        combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
        vec = self._to_dim_vector(combined)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-12:
            return vec
        return vec / norm

    def _extract_scalars(self, source: str) -> np.ndarray:
        """Extract numerical feature vector from source code.

        Returns a small set of structural scalars that are sensitive to the
        source's size, complexity, and shape. These are tiled to a reasonable
        length so fractal compression has enough samples to bite on.
        """
        lines = source.splitlines()
        n_lines = len(lines)
        n_chars = len(source)
        tokens = re.findall(r"\b\w+\b", source)
        n_tokens = len(tokens)
        # Indent depth: average leading whitespace per non-empty line.
        indent_depths = []
        for line in lines:
            if line.strip():
                stripped = line.lstrip()
                indent_chars = len(line) - len(stripped)
                # Treat tabs as 4 spaces (rules.indent_size).
                indent = (
                    indent_chars
                    if " " * indent_chars == line[:indent_chars]
                    else indent_chars * 4
                )
                indent_depths.append(indent)
        avg_indent = float(np.mean(indent_depths)) if indent_depths else 0.0
        # Keyword density: fraction of tokens that are Python keywords.
        keywords = {
            "def", "class", "if", "else", "elif", "for", "while", "try",
            "except", "finally", "return", "yield", "import", "from", "as",
            "with", "lambda", "and", "or", "not", "in", "is", "pass", "break",
            "continue", "raise", "assert", "del", "global", "nonlocal",
        }
        keyword_count = sum(1 for t in tokens if t in keywords)
        keyword_density = float(keyword_count / max(1, n_tokens))
        # AST node count.
        try:
            tree = ast.parse(source)
            node_count = sum(1 for _ in ast.walk(tree))
        except SyntaxError:
            node_count = 0
        # Build a feature vector: the 6 raw scalars plus log-scaled versions
        # (for scale diversity) and a normalised version. Tile to 32 elements
        # so fractal compression has enough samples.
        scalars = np.array(
            [n_lines, n_chars, n_tokens, avg_indent, keyword_density, node_count],
            dtype=float,
        )
        # Add log-scaled features (skip zero to avoid -inf).
        log_scalars = np.array(
            [
                float(np.log(max(1, s))) if s > 0 else 0.0
                for s in scalars
            ],
            dtype=float,
        )
        # Add a mean-centered copy (captures deviation from average).
        centered = scalars - float(np.mean(scalars))
        combined = np.concatenate([scalars, log_scalars, centered])
        # Tile to 32 elements (or truncate if shorter).
        if combined.size < 32:
            combined = np.tile(combined, (32 // combined.size) + 1)[:32]
        return combined[:32]

    def _to_dim_vector(self, vec: np.ndarray) -> np.ndarray:
        """Pad or truncate ``vec`` to ``dim`` length."""
        if vec.size < self.dim:
            vec = np.pad(vec, (0, self.dim - vec.size))
        elif vec.size > self.dim:
            vec = vec[: self.dim]
        return vec.astype(float)


class ASTAnalyzer:
    """Parse Python AST and extract structural features.

    Returns function/class names, imports, cyclomatic complexity, max
    nesting depth, and total node count.
    """

    def __init__(self, dim: int = 64, rules: CodeRules | None = None):
        self.dim = dim
        self.rules = rules or CodeRules()

    def analyze(self, source: str) -> dict:
        """Analyze ``source`` and return structural features.

        Returns ``{'functions', 'classes', 'imports', 'complexity',
        'max_depth', 'node_count'}``.
        """
        if not source:
            return {
                "functions": [],
                "classes": [],
                "imports": [],
                "complexity": 0,
                "max_depth": 0,
                "node_count": 0,
            }
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return {
                "functions": [],
                "classes": [],
                "imports": [],
                "complexity": 0,
                "max_depth": 0,
                "node_count": 0,
            }
        functions = []
        classes = []
        imports = []
        complexity = 1  # base complexity
        node_count = 0
        for node in ast.walk(tree):
            node_count += 1
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}" if module else alias.name)
            # Decision points add to cyclomatic complexity.
            if isinstance(
                node,
                ast.If
                | ast.IfExp
                | ast.For
                | ast.While
                | ast.AsyncFor
                | ast.ExceptHandler,
            ):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                # Each boolean operator adds one path.
                complexity += max(0, len(node.values) - 1)
            elif isinstance(node, ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp):
                complexity += 1
        max_depth = self._max_depth(tree)
        return {
            "functions": functions,
            "classes": classes,
            "imports": imports,
            "complexity": int(complexity),
            "max_depth": int(max_depth),
            "node_count": int(node_count),
        }

    def _max_depth(self, tree: ast.AST) -> int:
        """Compute the maximum nesting depth of the AST."""
        max_depth = 0

        def walk_depth(node, depth):
            nonlocal max_depth
            if depth > max_depth:
                max_depth = depth
            for child in ast.iter_child_nodes(node):
                walk_depth(child, depth + 1)

        walk_depth(tree, 0)
        return max_depth


class CodeSimilarityChecker:
    """Compare two code snippets via token + structure similarity.

    Token overlap is the Jaccard index of the token sets. Structure
    similarity is the cosine similarity of AST node-type count vectors.
    The overall similarity is the average of the two.
    """

    def __init__(self, dim: int = 64, rules: CodeRules | None = None):
        self.dim = dim
        self.rules = rules or CodeRules()

    def compare(self, source_a: str, source_b: str) -> dict:
        """Compare two code snippets.

        Returns ``{'similarity', 'token_overlap', 'structure_similarity'}``.
        """
        tokens_a = set(re.findall(r"\b\w+\b", source_a or ""))
        tokens_b = set(re.findall(r"\b\w+\b", source_b or ""))
        if tokens_a or tokens_b:
            token_overlap = float(
                len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
            )
        else:
            token_overlap = 0.0
        struct_a = self._node_type_counts(source_a)
        struct_b = self._node_type_counts(source_b)
        structure_similarity = self._cosine_similarity(struct_a, struct_b)
        similarity = 0.5 * token_overlap + 0.5 * structure_similarity
        return {
            "similarity": float(np.clip(similarity, 0.0, 1.0)),
            "token_overlap": float(np.clip(token_overlap, 0.0, 1.0)),
            "structure_similarity": float(np.clip(structure_similarity, 0.0, 1.0)),
        }

    def _node_type_counts(self, source: str) -> Counter:
        """Count AST node types in ``source``."""
        try:
            tree = ast.parse(source or "")
        except SyntaxError:
            return Counter()
        return Counter(type(node).__name__ for node in ast.walk(tree))

    @staticmethod
    def _cosine_similarity(a: Counter, b: Counter) -> float:
        """Cosine similarity between two count vectors."""
        keys = set(a.keys()) | set(b.keys())
        if not keys:
            return 0.0
        vec_a = np.array([a.get(k, 0) for k in keys], dtype=float)
        vec_b = np.array([b.get(k, 0) for k in keys], dtype=float)
        norm_a = float(np.linalg.norm(vec_a))
        norm_b = float(np.linalg.norm(vec_b))
        if norm_a < 1e-12 or norm_b < 1e-12:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


class DefectPatternDetector:
    """Detect rule-based defect patterns via AST walking.

    Detects:
    - ``mutable_default``: ``def f(x=[])`` or ``def f(x={})``
    - ``bare_except``: ``except:``
    - ``equals_none``: ``if x == None`` (should be ``is None``)
    """

    def __init__(self, dim: int = 64, rules: CodeRules | None = None):
        self.dim = dim
        self.rules = rules or CodeRules()

    def detect(self, source: str) -> dict:
        """Detect defect patterns in ``source``.

        Returns ``{'defects', 'total'}`` where ``defects`` is a list of
        ``{'pattern', 'line', 'col', 'snippet'}``.
        """
        if not source:
            return {"defects": [], "total": 0}
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return {"defects": [], "total": 0}
        defects = []
        lines = source.splitlines()
        for node in ast.walk(tree):
            # mutable_default
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                for default in node.args.defaults + node.args.kw_defaults:
                    if default is None:
                        continue
                    if isinstance(
                        default,
                        ast.List | ast.Dict | ast.Set | ast.ListComp,
                    ):
                        defects.append({
                            "pattern": "mutable_default",
                            "line": node.lineno,
                            "col": node.col_offset,
                            "snippet": self._safe_line(lines, node.lineno - 1),
                        })
            # bare_except
            elif isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    defects.append({
                        "pattern": "bare_except",
                        "line": node.lineno,
                        "col": node.col_offset,
                        "snippet": self._safe_line(lines, node.lineno - 1),
                    })
            # equals_none: `x == None` (Compare with Eq operator + Constant None)
            elif isinstance(node, ast.Compare):
                for op, comparator in zip(node.ops, node.comparators, strict=False):
                    if isinstance(op, ast.Eq) and self._is_none_constant(comparator):
                        defects.append({
                            "pattern": "equals_none",
                            "line": node.lineno,
                            "col": node.col_offset,
                            "snippet": self._safe_line(lines, node.lineno - 1),
                        })
        return {"defects": defects, "total": len(defects)}

    @staticmethod
    def _is_none_constant(node: ast.AST) -> bool:
        """Check if a node is a ``None`` constant."""
        if isinstance(node, ast.Constant):
            return node.value is None
        # Python 3.7 compat: NameConstant with value None.
        if hasattr(ast, "NameConstant") and isinstance(node, ast.NameConstant):
            return node.value is None
        return False

    @staticmethod
    def _safe_line(lines: list[str], idx: int) -> str:
        """Return the line at ``idx`` or empty string."""
        if 0 <= idx < len(lines):
            return lines[idx].strip()
        return ""
