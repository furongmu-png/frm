# src/zero_data_model/capabilities/causal.py
"""Causal/Decision capability module for the zero-data cognitive model.

Composes the core cognitive modules (active inference for free-energy
scoring) with a small rule library (``CausalRules``: tree depth, Nash
iterations, bandit epsilon). Provides pure-Python, rule-based decision
trees, game-theoretic analysis, counterfactual reasoning, and multi-armed
bandits -- no external ML libraries (scikit-learn / nashpy) are required.
"""

from __future__ import annotations

import numpy as np

from .rules import CausalRules


class DecisionTreeBuilder:
    """ID3-style decision tree (entropy-based splits).

    Builds a recursive tree from labeled feature data by repeatedly
    selecting the feature with the highest information gain. Stops at
    ``max_tree_depth``, pure nodes, or when fewer than ``min_samples_split``
    samples remain.
    """

    def __init__(self, dim: int = 64, rules: CausalRules | None = None):
        self.dim = dim
        self.rules = rules or CausalRules()

    def fit(self, features: np.ndarray, labels: np.ndarray) -> dict:
        """Fit a decision tree to ``(features, labels)``.

        Returns ``{'tree', 'depth', 'n_leaves', 'features_used'}`` where
        ``tree`` is a nested dict (internal nodes have ``feature`` and
        ``children``; leaves have ``label``).
        """
        features = np.asarray(features, dtype=float)
        labels = np.asarray(labels)
        if features.size == 0 or labels.size == 0:
            return {
                "tree": {"label": None, "n_samples": 0},
                "depth": 0,
                "n_leaves": 1,
                "features_used": [],
            }
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        n_samples, n_features = features.shape
        if n_samples != labels.shape[0]:
            return {
                "tree": {"label": None, "n_samples": 0},
                "depth": 0,
                "n_leaves": 1,
                "features_used": [],
            }
        features_used: set[int] = set()

        def build(X: np.ndarray, y: np.ndarray, depth: int) -> dict:
            n = y.shape[0]
            # Pure or stop condition.
            unique = np.unique(y)
            if (
                len(unique) <= 1
                or depth >= self.rules.max_tree_depth
                or n < self.rules.min_samples_split
            ):
                # Leaf: majority label.
                if n == 0:
                    return {"label": None, "n_samples": 0}
                values, counts = np.unique(y, return_counts=True)
                label = values[np.argmax(counts)]
                return {"label": label, "n_samples": int(n)}
            # Pick best feature by information gain.
            best_feat = self._best_feature(X, y)
            if best_feat is None:
                values, counts = np.unique(y, return_counts=True)
                label = values[np.argmax(counts)]
                return {"label": label, "n_samples": int(n)}
            features_used.add(int(best_feat))
            node = {
                "feature": int(best_feat),
                "n_samples": int(n),
                "children": {},
            }
            feature_col = X[:, best_feat]
            for value in np.unique(feature_col):
                mask = feature_col == value
                child_X = X[mask]
                child_y = y[mask]
                node["children"][float(value)] = build(child_X, child_y, depth + 1)
            return node

        tree = build(features, labels, 0)

        def count_leaves(node: dict) -> int:
            if "children" not in node:
                return 1
            return sum(count_leaves(c) for c in node["children"].values())

        def max_depth(node: dict) -> int:
            if "children" not in node:
                return 0
            return 1 + max((max_depth(c) for c in node["children"].values()), default=0)

        return {
            "tree": tree,
            "depth": int(max_depth(tree)),
            "n_leaves": int(count_leaves(tree)),
            "features_used": sorted(features_used),
        }

    def _best_feature(self, X: np.ndarray, y: np.ndarray) -> int | None:
        """Pick the feature with the highest information gain."""
        n_samples, n_features = X.shape
        if n_samples == 0:
            return None
        base_entropy = self._entropy(y)
        best_gain = -1.0
        best_feat = None
        for f in range(n_features):
            col = X[:, f]
            unique_vals = np.unique(col)
            if len(unique_vals) <= 1:
                continue
            subset_entropy = 0.0
            for v in unique_vals:
                mask = col == v
                weight = mask.sum() / n_samples
                subset_entropy += weight * self._entropy(y[mask])
            gain = base_entropy - subset_entropy
            if gain > best_gain:
                best_gain = gain
                best_feat = f
        return best_feat

    @staticmethod
    def _entropy(y: np.ndarray) -> float:
        """Shannon entropy of a label vector."""
        if y.size == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        p = counts / y.size
        # Use log2; suppress warning for p=0 via masking.
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))


class GameTheoryAnalyzer:
    """2-player normal-form game analyzer.

    Finds pure-strategy Nash equilibria by scanning the payoff matrix for
    cells where player A's payoff is the row maximum AND player B's payoff
    is the column maximum. If ``payoff_b`` is omitted, the game is
    assumed to be zero-sum (``payoff_b = -payoff_a``).
    """

    def __init__(self, dim: int = 64, rules: CausalRules | None = None):
        self.dim = dim
        self.rules = rules or CausalRules()

    def analyze(
        self,
        payoff_a: np.ndarray,
        payoff_b: np.ndarray | None = None,
    ) -> dict:
        """Analyze a 2-player normal-form game.

        Returns ``{'nash_equilibria', 'value_a', 'value_b', 'is_zero_sum'}``
        where ``nash_equilibria`` is a list of ``(row, col)`` tuples.
        """
        a = np.asarray(payoff_a, dtype=float)
        if a.ndim != 2:
            return {
                "nash_equilibria": [],
                "value_a": 0.0,
                "value_b": 0.0,
                "is_zero_sum": False,
            }
        is_zero_sum = payoff_b is None
        b = -a if is_zero_sum else np.asarray(payoff_b, dtype=float)
        if b.shape != a.shape:
            return {
                "nash_equilibria": [],
                "value_a": 0.0,
                "value_b": 0.0,
                "is_zero_sum": is_zero_sum,
            }
        # Pure-strategy Nash equilibria: cells where a is row-max and b is
        # col-max.
        row_max = a.max(axis=1, keepdims=True)
        col_max = b.max(axis=0, keepdims=True)
        a_best = a == row_max
        b_best = b == col_max
        nash_mask = a_best & b_best
        equilibria: list[tuple[int, int]] = []
        rows, cols = np.where(nash_mask)
        for r, c in zip(rows, cols, strict=True):
            equilibria.append((int(r), int(c)))
        if equilibria:
            # Use first equilibrium for value (deterministic order).
            r, c = equilibria[0]
            value_a = float(a[r, c])
            value_b = float(b[r, c])
        else:
            value_a = 0.0
            value_b = 0.0
        return {
            "nash_equilibria": equilibria,
            "value_a": value_a,
            "value_b": value_b,
            "is_zero_sum": bool(is_zero_sum),
        }


class CounterfactualReasoner:
    """Counterfactual reasoning: 'what if the input had been different?'.

    Given an observed input and an intervention (replace one element with
    a new value), computes the factual vs counterfactual outcome. The
    outcome defaults to ``mean(observed)``; if a ``model_fn`` is provided,
    it is applied instead. The surprisal of the change is scored by
    ``active_inference.compute_free_energy``.
    """

    def __init__(self, dim: int = 64, active_inference=None, rules: CausalRules | None = None):
        self.dim = dim
        self.rules = rules or CausalRules()
        if active_inference is None:
            from ..active_inference import ActiveInferenceEngine

            active_inference = ActiveInferenceEngine(state_dim=dim, obs_dim=dim)
        self.active_inference = active_inference

    def counterfactual(
        self,
        observed: np.ndarray,
        intervention: dict,
        model_fn=None,
    ) -> dict:
        """Compute the counterfactual outcome under ``intervention``.

        ``intervention`` is ``{'index': int, 'value': float}``.
        Returns ``{'factual', 'counterfactual', 'effect', 'surprisal'}``.
        """
        observed = np.asarray(observed, dtype=float)
        if observed.size == 0:
            return {
                "factual": 0.0,
                "counterfactual": 0.0,
                "effect": 0.0,
                "surprisal": 0.0,
            }
        idx = int(intervention.get("index", 0))
        value = float(intervention.get("value", 0.0))
        # Factual outcome: model_fn or mean.
        factual = (
            float(model_fn(observed)) if model_fn is not None else float(np.mean(observed))
        )
        # Counterfactual input: replace the value at idx.
        modified = observed.copy()
        if 0 <= idx < modified.size:
            modified.flat[idx] = value
        counterfactual = (
            float(model_fn(modified)) if model_fn is not None else float(np.mean(modified))
        )
        effect = counterfactual - factual
        # Surprisal of the difference vector (length 1).
        diff_vec = np.array([effect], dtype=float)
        surprisal = float(self.active_inference.compute_free_energy(diff_vec))
        if not np.isfinite(surprisal):
            surprisal = 0.0
        return {
            "factual": float(factual),
            "counterfactual": float(counterfactual),
            "effect": float(effect),
            "surprisal": float(surprisal),
        }


class MultiArmedBandit:
    """epsilon-greedy + UCB1 multi-armed bandit selector.

    Given a history of per-arm rewards, selects the next arm to pull.
    With probability ``epsilon`` explores randomly; otherwise exploits
    the best arm. Also computes UCB1 confidence bounds.
    """

    def __init__(self, dim: int = 64, rules: CausalRules | None = None):
        self.dim = dim
        self.rules = rules or CausalRules()

    def select(self, rewards_history: list[list[float]]) -> dict:
        """Select the next arm given ``rewards_history``.

        ``rewards_history[i]`` is the list of rewards observed for arm i.
        Returns ``{'arm', 'method', 'expected_values', 'confidence_bounds'}``.
        """
        if not rewards_history:
            return {
                "arm": 0,
                "method": "none",
                "expected_values": np.zeros(0),
                "confidence_bounds": np.zeros(0),
            }
        n_arms = len(rewards_history)
        # Per-arm mean reward (0 if never pulled).
        means = np.zeros(n_arms, dtype=float)
        counts = np.zeros(n_arms, dtype=float)
        for i, history in enumerate(rewards_history):
            if history:
                means[i] = float(np.mean(history))
                counts[i] = float(len(history))
        total = float(counts.sum())
        # UCB1: mean + sqrt(2 * ln(N) / n_i) for arms with n_i > 0.
        ucb = np.zeros(n_arms, dtype=float)
        if total > 0:
            log_n = float(np.log(max(1.0, total)))
            for i in range(n_arms):
                if counts[i] > 0:
                    ucb[i] = means[i] + float(np.sqrt(2.0 * log_n / counts[i]))
                else:
                    # Unpulled arm: infinite UCB (force exploration).
                    ucb[i] = float("inf")
        # Epsilon-greedy selection.
        rng = np.random.default_rng()
        if rng.random() < self.rules.bandit_epsilon:
            # Explore: pick a random arm.
            arm = int(rng.integers(0, n_arms))
            method = "explore"
        else:
            # Exploit: pick argmax mean (break ties by lowest index).
            arm = int(np.argmax(means))
            method = "exploit"
        return {
            "arm": arm,
            "method": method,
            "expected_values": means,
            "confidence_bounds": ucb,
        }
