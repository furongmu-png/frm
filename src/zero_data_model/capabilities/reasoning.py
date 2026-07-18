# src/zero_data_model/capabilities/reasoning.py
"""Reasoning capability module for the zero-data cognitive model.

Composes the core cognitive modules (category engine for structural
similarity) with a small rule library (``ReasoningRules``: inference depth,
contradiction thresholds). Provides pure-Python, rule-based propositional,
deductive, inductive, and analogical reasoning -- no external LLM or
training data is required.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .rules import ReasoningRules


class PropositionalLogicEngine:
    """Rule-based propositional logic with modus ponens inference.

    Maintains a knowledge base of facts (proposition -> bool) and rules
    (antecedent -> list of consequents). Inference iterates forward until
    a fixpoint is reached or ``max_inference_depth`` iterations elapse.
    Contradictions (a fact inferred as both True and False) are reported.
    """

    def __init__(self, dim: int = 64, rules: ReasoningRules | None = None):
        self.dim = dim
        self.rules = rules or ReasoningRules()
        self._facts: dict[str, bool] = {}
        # antecedent -> list of (consequent, value) pairs.
        self._rules: dict[str, list[tuple[str, bool]]] = defaultdict(list)

    def add_fact(self, proposition: str, value: bool) -> None:
        """Add a fact to the knowledge base."""
        self._facts[proposition] = bool(value)

    def add_rule(self, antecedent: str, consequent: str) -> None:
        """Add a rule ``antecedent -> consequent`` (modus ponens).

        The rule asserts that if ``antecedent`` is True, then ``consequent``
        is True. To express ``not consequent``, add a rule with the
        antecedent being ``not <antecedent>``.
        """
        self._rules[antecedent].append((consequent, True))

    def add_negation_rule(self, antecedent: str, consequent: str) -> None:
        """Add a rule ``antecedent -> not consequent``."""
        self._rules[antecedent].append((consequent, False))

    def infer(self) -> dict:
        """Run forward inference until fixpoint or max depth.

        Returns ``{'facts', 'inferences', 'contradictions'}`` where
        ``inferences`` is the list of ``(antecedent, consequent, value)``
        tuples derived, and ``contradictions`` is a list of propositions
        inferred as both True and False.
        """
        inferences: list[tuple[str, str, bool]] = []
        contradictions: list[str] = []
        max_depth = self.rules.max_inference_depth
        for _ in range(max_depth):
            new_inferences: list[tuple[str, str, bool]] = []
            changed = False
            for antecedent, consequents in self._rules.items():
                # Resolve antecedent truth value.
                antecedent_value = self._resolve(antecedent)
                if antecedent_value is not True:
                    continue
                for consequent, value in consequents:
                    current = self._facts.get(consequent)
                    if current is None:
                        new_inferences.append((antecedent, consequent, value))
                    elif current != value:
                        contradictions.append(consequent)
            if not new_inferences and not contradictions:
                break
            for ant, cons, val in new_inferences:
                if self._facts.get(cons) is None:
                    self._facts[cons] = val
                    inferences.append((ant, cons, val))
                    changed = True
            if not changed and not contradictions:
                break
        # Deduplicate contradictions.
        contradictions = sorted(set(contradictions))
        return {
            "facts": dict(self._facts),
            "inferences": inferences,
            "contradictions": contradictions,
        }

    def _resolve(self, proposition: str) -> bool | None:
        """Resolve a proposition's truth value, handling ``not`` prefix."""
        prop = proposition.strip()
        if prop.lower().startswith("not "):
            inner = prop[4:].strip()
            val = self._facts.get(inner)
            if val is None:
                return None
            return not val
        return self._facts.get(prop)


class DeductiveReasoner:
    """Term logic: classical categorical syllogisms.

    Supports the four canonical forms:
    - **A** (universal affirmative): "All S are P"
    - **E** (universal negative): "No S are P"
    - **I** (particular affirmative): "Some S are P"
    - **O** (particular negative): "Some S are not P"

    And the four first-figure syllogisms: Barbara (AAA), Celarent (EAE),
    Darii (AII), Ferio (EIO).
    """

    def __init__(self, dim: int = 64, rules: ReasoningRules | None = None):
        self.dim = dim
        self.rules = rules or ReasoningRules()

    def syllogism(self, major: tuple, minor: tuple) -> dict:
        """Apply a categorical syllogism.

        ``major`` and ``minor`` are tuples of ``(form, subject, predicate)``
        where ``form`` is one of ``"A" | "E" | "I" | "O"``.

        Returns ``{'conclusion', 'valid', 'form'}``.
        """
        if len(major) != 3 or len(minor) != 3:
            return {"conclusion": None, "valid": False, "form": "invalid"}
        m_form, m_subj, m_pred = major
        s_form, s_subj, s_pred = minor
        # In a categorical syllogism, the minor's predicate is the middle
        # term M, the major's subject is also M, and the conclusion links
        # the minor's subject S to the major's predicate P.
        if s_pred != m_subj:
            # Try the alternate figure: major's predicate is M.
            if m_pred == s_pred:
                m_form, m_subj, m_pred = (m_form, m_pred, m_subj)
            else:
                return {"conclusion": None, "valid": False, "form": "invalid"}
        if s_pred != m_subj:
            return {"conclusion": None, "valid": False, "form": "invalid"}
        # Determine the valid figure / mood.
        figure = self._classify_figure(m_form, s_form)
        if figure is None:
            return {"conclusion": None, "valid": False, "form": "invalid"}
        conclusion_form, name = figure
        # Build the conclusion string.
        conclusion = self._render(conclusion_form, s_subj, m_pred)
        return {
            "conclusion": conclusion,
            "valid": True,
            "form": f"{name} ({m_form}{s_form}{conclusion_form}-1)",
        }

    @staticmethod
    def _classify_figure(m_form: str, s_form: str) -> tuple[str, str] | None:
        """Map (major_form, minor_form) to a valid syllogism.

        Returns ``(conclusion_form, name)`` or None.
        """
        valid = {
            # First figure.
            ("A", "A"): ("A", "Barbara"),
            ("E", "A"): ("E", "Celarent"),
            ("A", "I"): ("I", "Darii"),
            ("E", "I"): ("O", "Ferio"),
        }
        return valid.get((m_form, s_form))

    @staticmethod
    def _render(form: str, subj: str, pred: str) -> str:
        """Render a categorical proposition as a string."""
        if form == "A":
            return f"All {subj} are {pred}"
        if form == "E":
            return f"No {subj} are {pred}"
        if form == "I":
            return f"Some {subj} are {pred}"
        if form == "O":
            return f"Some {subj} are not {pred}"
        return ""


class InductiveReasoner:
    """Generalize a rule from labeled examples.

    Given a list of example feature dicts and boolean labels, finds the
    ``(key, value)`` pair that appears exclusively in positive examples
    (highest support + coverage). Returns the rule as a string, with
    confidence = support / total positive examples.
    """

    def __init__(self, dim: int = 64, rules: ReasoningRules | None = None):
        self.dim = dim
        self.rules = rules or ReasoningRules()

    def generalize(
        self,
        examples: list[dict],
        labels: list[bool],
    ) -> dict:
        """Induce a rule from labeled ``examples``.

        Returns ``{'rule', 'confidence', 'support', 'coverage'}``.
        """
        if not examples or len(examples) != len(labels):
            return {
                "rule": "",
                "confidence": 0.0,
                "support": 0,
                "coverage": 0.0,
            }
        positives = [ex for ex, lbl in zip(examples, labels, strict=True) if lbl]
        negatives = [ex for ex, lbl in zip(examples, labels, strict=True) if not lbl]
        n_pos = len(positives)
        if n_pos == 0:
            return {
                "rule": "",
                "confidence": 0.0,
                "support": 0,
                "coverage": 0.0,
            }
        # Tally (key, value) pairs across positives.
        pos_pairs: dict[tuple, int] = defaultdict(int)
        for ex in positives:
            for k, v in ex.items():
                pos_pairs[(k, v)] += 1
        # Tally (key, value) pairs across negatives.
        neg_pairs: dict[tuple, int] = defaultdict(int)
        for ex in negatives:
            for k, v in ex.items():
                neg_pairs[(k, v)] += 1
        # Find the pair that maximizes support and excludes all negatives.
        best_pair = None
        best_score = -1.0
        best_support = 0
        for pair, support in pos_pairs.items():
            neg_count = neg_pairs.get(pair, 0)
            # Coverage = fraction of positives containing this pair.
            coverage = support / n_pos
            # Confidence = positives with pair / all examples with pair.
            total = support + neg_count
            confidence = support / total if total > 0 else 0.0
            # Prefer pairs with high coverage and high confidence.
            score = coverage * (1.0 if neg_count == 0 else confidence)
            if score > best_score:
                best_score = score
                best_pair = pair
                best_support = support
        if best_pair is None:
            return {
                "rule": "",
                "confidence": 0.0,
                "support": 0,
                "coverage": 0.0,
            }
        key, value = best_pair
        rule_str = f"{key} == {value!r}"
        coverage = best_support / n_pos
        total_with_pair = best_support + neg_pairs.get(best_pair, 0)
        confidence = best_support / total_with_pair if total_with_pair > 0 else 0.0
        return {
            "rule": rule_str,
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "support": int(best_support),
            "coverage": float(np.clip(coverage, 0.0, 1.0)),
        }


class AnalogicalReasoner:
    """Analogy via structural similarity (category theory).

    Given a source and target structure (each a dict of attributes), find
    the best mapping of source attributes to target attributes by structural
    similarity, and transfer the source's "relation" to the target.
    """

    def __init__(
        self,
        dim: int = 64,
        category_engine=None,
        math_universe=None,
        rules: ReasoningRules | None = None,
    ):
        self.dim = dim
        self.rules = rules or ReasoningRules()
        # Lazy-import to avoid a circular dependency at module import time.
        if category_engine is None:
            from ..category_engine import CategoryTheoryEngine

            category_engine = CategoryTheoryEngine(dim=dim)
        self.category_engine = category_engine
        self.math_universe = math_universe

    def analogize(self, source: dict, target: dict) -> dict:
        """Map ``source`` attributes to ``target`` attributes by similarity.

        Returns ``{'mapping', 'similarity', 'transfer'}``. The mapping is
        based on structural similarity of the (key, value) attribute as a
        whole -- the key contributes the structural role, and the value
        contributes the content.
        """
        if not source or not target:
            return {"mapping": {}, "similarity": 0.0, "transfer": {}}
        src_keys = list(source.keys())
        tgt_keys = list(target.keys())
        # Encode each attribute as a (key, value) combined vector. The key
        # dominates the structural role (string-hash vector), and the value
        # adds content; both contribute to similarity.
        src_vecs = {
            k: self._combine_vectors(self._to_vector(k), self._to_vector(v))
            for k, v in source.items()
        }
        tgt_vecs = {
            k: self._combine_vectors(self._to_vector(k), self._to_vector(v))
            for k, v in target.items()
        }
        # For each source key, find the closest target key.
        mapping: dict[str, str] = {}
        similarities: list[float] = []
        used_targets: set[str] = set()
        # Greedy matching: sort source keys by their best similarity (desc),
        # then assign each source to the best unused target.
        candidates: list[tuple[str, str, float]] = []
        for sk in src_keys:
            sv = src_vecs[sk]
            for tk in tgt_keys:
                tv = tgt_vecs[tk]
                sim = self.category_engine.structural_similarity(sv, tv)
                candidates.append((sk, tk, max(0.0, sim)))
        # Sort by similarity descending; assign greedily.
        candidates.sort(key=lambda x: x[2], reverse=True)
        assigned_sources: set[str] = set()
        for sk, tk, sim in candidates:
            if sk in assigned_sources or tk in used_targets:
                continue
            mapping[sk] = tk
            similarities.append(sim)
            assigned_sources.add(sk)
            used_targets.add(tk)
            if len(assigned_sources) == len(src_keys):
                break
        # Sources without any match (e.g., more sources than targets).
        for sk in src_keys:
            if sk not in mapping:
                mapping[sk] = ""  # no match
        avg_sim = float(np.mean(similarities)) if similarities else 0.0
        avg_sim = float(np.clip(avg_sim, 0.0, 1.0))
        # Transfer: source's numeric values can be transferred to mapped target.
        transfer: dict[str, object] = {}
        for sk, tk in mapping.items():
            if not tk:
                continue
            sv = source[sk]
            tv = target[tk]
            # If both source and target values are numeric, transfer source value
            # as a "predicted" target value (analogy: if source.attr = X, then
            # target.mapped_attr should also be X).
            if isinstance(sv, int | float) and isinstance(tv, int | float):
                transfer[tk] = sv
        return {
            "mapping": mapping,
            "similarity": avg_sim,
            "transfer": transfer,
        }

    def _combine_vectors(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Combine two vectors by weighted sum + L2 normalization.

        The first vector (key) gets weight 2.0 to dominate the structural
        role; the second (value) gets weight 1.0 for content. This ensures
        same-key attributes match strongly even when their values differ.
        """
        combined = 2.0 * a + 1.0 * b
        norm = float(np.linalg.norm(combined))
        if norm > 1e-12:
            return combined / norm
        return combined

    def _to_vector(self, value: object) -> np.ndarray:
        """Convert any value to a ``dim``-length L2-normalized float vector.

        Different scalar values are encoded as one-hot positions (so 1, 2,
        3 produce distinct vectors). Strings and dicts use deterministic
        hashing of their content.
        """
        if isinstance(value, int | float):
            # One-hot by value hash: distinct values -> distinct positions.
            v = np.zeros(self.dim, dtype=float)
            idx = abs(hash(value)) % self.dim
            v[idx] = 1.0
            return v
        if isinstance(value, str):
            # Hash each character to a bin (deterministic per Python session).
            v = np.zeros(self.dim, dtype=float)
            for ch in value:
                idx = abs(hash(ch)) % self.dim
                v[idx] += 1.0
            norm = float(np.linalg.norm(v))
            return v / norm if norm > 1e-12 else v
        if isinstance(value, dict):
            v = np.zeros(self.dim, dtype=float)
            for k, val in value.items():
                sub = self._to_vector(val)
                idx = abs(hash(str(k))) % self.dim
                v[idx] += float(np.mean(sub))
            norm = float(np.linalg.norm(v))
            return v / norm if norm > 1e-12 else v
        if isinstance(value, list | tuple):
            if not value:
                return np.zeros(self.dim, dtype=float)
            sub_vecs = [self._to_vector(item) for item in value]
            mean = np.mean(np.array(sub_vecs), axis=0)
            norm = float(np.linalg.norm(mean))
            return mean / norm if norm > 1e-12 else mean
        # Fallback: hash the string repr.
        v = np.zeros(self.dim, dtype=float)
        for ch in repr(value):
            idx = abs(hash(ch)) % self.dim
            v[idx] += 1.0
        norm = float(np.linalg.norm(v))
        return v / norm if norm > 1e-12 else v
