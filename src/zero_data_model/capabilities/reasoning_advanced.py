# src/zero_data_model/capabilities/reasoning_advanced.py
"""Advanced reasoning capabilities for the zero-data cognitive model.

Composes the core cognitive modules (active inference for free-energy scoring)
with ``ReasoningRules``. Provides pure-Python, rule-based abductive,
defeasible, and causal-chain reasoning -- no external LLM or training data
is required.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .rules import ReasoningRules


class AbductiveReasoner:
    """Best-explanation inference: pick the most plausible hypothesis.

    Given an observation and a list of candidate hypotheses, scores each
    hypothesis by ``prior * likelihood`` where ``likelihood`` is a rule-based
    keyword-overlap score between the observation and the hypothesis text.
    Returns the best-scoring hypothesis with a normalized confidence.
    """

    def __init__(
        self,
        dim: int = 64,
        active_inference=None,
        rules: ReasoningRules | None = None,
    ):
        self.dim = dim
        self.rules = rules or ReasoningRules()
        self.active_inference = active_inference

    def explain(
        self,
        observation: str,
        hypotheses: list[str],
        priors: list[float] | None = None,
    ) -> dict:
        """Find the best explanation for ``observation``.

        Returns ``{'best', 'scores', 'confidence'}`` where ``scores`` is a
        list of floats aligned with ``hypotheses``, ``best`` is the
        hypothesis with the highest score, and ``confidence`` is the
        normalized score ``best_score / sum(scores)``.
        """
        if not hypotheses:
            return {"best": None, "scores": [], "confidence": 0.0}
        if priors is None:
            priors = [1.0 / len(hypotheses)] * len(hypotheses)
        if len(priors) != len(hypotheses):
            return {"best": None, "scores": [], "confidence": 0.0}
        obs_tokens = self._tokenize(observation)
        scores = []
        for hyp, prior in zip(hypotheses, priors, strict=True):
            hyp_tokens = self._tokenize(hyp)
            likelihood = self._keyword_overlap(obs_tokens, hyp_tokens)
            scores.append(float(prior) * float(likelihood))
        total = float(np.sum(scores))
        if total < 1e-12:
            # No information: pick the first hypothesis with zero confidence.
            return {
                "best": hypotheses[0],
                "scores": [0.0] * len(hypotheses),
                "confidence": 0.0,
            }
        best_idx = int(np.argmax(scores))
        confidence = float(scores[best_idx] / total)
        return {
            "best": hypotheses[best_idx],
            "scores": [float(s) for s in scores],
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
        }

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Tokenize ``text`` into a set of lowercase word tokens."""
        if not text:
            return set()
        tokens = []
        current = []
        for ch in text.lower():
            if ch.isalnum():
                current.append(ch)
            else:
                if current:
                    tokens.append("".join(current))
                    current = []
        if current:
            tokens.append("".join(current))
        return set(tokens)

    @staticmethod
    def _keyword_overlap(a: set[str], b: set[str]) -> float:
        """Jaccard overlap between two token sets in ``[0, 1]``."""
        if not a or not b:
            return 0.0
        intersection = len(a & b)
        union = len(a | b)
        return float(intersection / union) if union > 0 else 0.0


class DefeasibleReasoner:
    """Default logic with exceptions: ``All X are Y, unless Z``.

    Each rule has an antecedent, a consequent, and an optional exception.
    A rule fires only when its antecedent is True AND its exception is
    False. Conflicts (multiple rules concluding opposite values) are
    reported as ``ambiguous``.
    """

    def __init__(self, dim: int = 64, rules: ReasoningRules | None = None):
        self.dim = dim
        self.rules = rules or ReasoningRules()
        # Each rule: (antecedent, consequent, value, exception_or_None).
        self._defaults: list[tuple[str, str, bool, str | None]] = []

    def add_default(
        self,
        rule: tuple,
        exception: tuple | None = None,
    ) -> None:
        """Add a default rule ``rule -> consequent`` with optional exception.

        ``rule`` is ``(antecedent, consequent, value)`` where value is True
        for "consequent is True" and False for "consequent is False".
        ``exception`` is ``(exception_prop, ...)`` -- if the first element
        is True in facts, the rule is defeated.
        """
        if not rule or len(rule) < 2:
            return
        antecedent = str(rule[0])
        consequent = str(rule[1])
        value = bool(rule[2]) if len(rule) > 2 else True
        exc = str(exception[0]) if exception else None
        self._defaults.append((antecedent, consequent, value, exc))

    def conclude(self, facts: dict) -> dict:
        """Apply defaults to ``facts``.

        Returns ``{'conclusions', 'defeated', 'ambiguous'}`` where
        ``conclusions`` is a dict of inferred propositions, ``defeated``
        lists rules blocked by their exceptions, and ``ambiguous`` lists
        propositions where conflicting rules fire.
        """
        conclusions: dict[str, bool] = {}
        defeated: list[tuple[str, str]] = []
        # Track which rule produced each conclusion for conflict detection.
        sources: dict[str, list[tuple[str, bool]]] = defaultdict(list)
        for antecedent, consequent, value, exception in self._defaults:
            antecedent_true = bool(facts.get(antecedent, False))
            if not antecedent_true:
                continue
            # Check exception: if exception prop is True in facts, defeat.
            if exception is not None and bool(facts.get(exception, False)):
                defeated.append((antecedent, consequent))
                continue
            sources[consequent].append((antecedent, value))
            # Tentatively apply.
            current = conclusions.get(consequent)
            if current is None:
                conclusions[consequent] = value
        # Detect ambiguous: same proposition concluded both True and False.
        ambiguous: list[str] = []
        for prop, srcs in sources.items():
            values = {v for _, v in srcs}
            if len(values) > 1:
                ambiguous.append(prop)
                # Remove from conclusions: ambiguous means no firm conclusion.
                conclusions.pop(prop, None)
        return {
            "conclusions": conclusions,
            "defeated": defeated,
            "ambiguous": sorted(set(ambiguous)),
        }


class CausalChainReasoner:
    """Chain causal implications: A -> B -> C.

    Maintains a causal graph as an adjacency list. ``trace(start)`` performs
    a DFS from ``start`` and returns the chain of effects, with cycle
    detection.
    """

    def __init__(self, dim: int = 64, rules: ReasoningRules | None = None):
        self.dim = dim
        self.rules = rules or ReasoningRules()
        # cause -> list of effects.
        self._graph: dict[str, list[str]] = defaultdict(list)

    def add_causal(self, cause: str, effect: str) -> None:
        """Add a causal edge ``cause -> effect``."""
        self._graph[str(cause)].append(str(effect))

    def trace(self, start: str, max_depth: int = 5) -> dict:
        """Trace the causal chain starting from ``start``.

        Returns ``{'chain', 'effects', 'depth', 'cycles'}`` where ``chain``
        is the list of nodes visited in DFS order, ``effects`` is the set
        of distinct downstream effects (excluding ``start``), ``depth`` is
        the maximum depth reached, and ``cycles`` is True if a cycle was
        detected.
        """
        start = str(start)
        if start not in self._graph:
            return {
                "chain": [start],
                "effects": [],
                "depth": 0,
                "cycles": False,
            }
        chain: list[str] = [start]
        effects: set[str] = set()
        visited: set[str] = {start}
        max_depth_reached = 0
        cycles = False

        def dfs(node: str, depth: int) -> None:
            nonlocal max_depth_reached, cycles
            if depth > max_depth:
                return
            max_depth_reached = max(max_depth_reached, depth)
            for child in self._graph.get(node, []):
                if child in visited:
                    cycles = True
                    continue
                visited.add(child)
                chain.append(child)
                effects.add(child)
                dfs(child, depth + 1)

        dfs(start, 0)
        return {
            "chain": chain,
            "effects": sorted(effects),
            "depth": int(max_depth_reached),
            "cycles": bool(cycles),
        }
