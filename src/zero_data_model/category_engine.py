# src/zero_data_model/category_engine.py
"""Category Theory Foundation for cross-domain reasoning."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base import CognitiveModule, Prediction, Signal

# JIT kernels -- graceful fallback to pure numpy if numba missing.
try:
    from .hardware.kernels import _cosine_similarity, _topos_classify
    _HAS_JIT = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_JIT = False


@dataclass
class Category:
    """A category with objects and morphisms."""
    name: str
    objects: dict[str, np.ndarray] = field(default_factory=dict)
    morphisms: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)

    def add_object(self, name: str, representation: np.ndarray) -> None:
        self.objects[name] = representation

    def add_morphism(self, source: str, target: str, transform: np.ndarray) -> None:
        self.morphisms[(source, target)] = transform

    def compose(self, source: str, intermediate: str, target: str) -> np.ndarray | None:
        m1 = self.morphisms.get((source, intermediate))
        m2 = self.morphisms.get((intermediate, target))
        if m1 is not None and m2 is not None:
            return m2 @ m1
        return None


@dataclass
class Functor:
    """Structure-preserving map between categories."""
    source: str
    target: str
    object_map: dict[str, str]
    morphism_map: dict[tuple[str, str], np.ndarray]

    def apply(self, obj: np.ndarray) -> np.ndarray:
        for transform in self.morphism_map.values():
            if transform.shape[1] == obj.shape[0]:
                return transform @ obj
        return obj


class ToposEngine:
    """Topos theory: subobject classifier for truth values."""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.truth_values = np.linspace(0, 1, dim)
        self.classifier = np.random.randn(dim, dim) * 0.1

    def classify(self, signal: np.ndarray) -> np.ndarray:
        s = signal[: self.dim]
        if len(s) < self.dim:
            s = np.pad(s, (0, self.dim - len(s)))
        if _HAS_JIT:
            return _topos_classify(
                np.ascontiguousarray(s, dtype=float),
                np.ascontiguousarray(self.classifier, dtype=float),
            )
        return 1.0 / (1.0 + np.exp(-s @ self.classifier))


class CategoryTheoryEngine(CognitiveModule):
    """
    Category Theory reasoning engine.
    - Defines categories for different problem domains
    - Uses functors for cross-domain transfer
    - Finds isomorphisms between problems
    """

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.categories: dict[str, Category] = {}
        self.functors: list[Functor] = []
        self.topos = ToposEngine(dim)
        self._init_default_categories()

    def _init_default_categories(self):
        nlp = Category(name="NLP")
        cv = Category(name="CV")
        analytics = Category(name="Analytics")
        for cat in [nlp, cv, analytics]:
            for i in range(5):
                cat.add_object(f"concept_{i}", np.random.randn(self.dim) * 0.1)
            self.categories[cat.name] = cat
        transfer_nlp_cv = np.random.randn(self.dim, self.dim) * 0.05
        self.functors.append(Functor(
            source="NLP", target="CV",
            object_map={f"concept_{i}": f"concept_{i}" for i in range(5)},
            morphism_map={("concept_0", "concept_1"): transfer_nlp_cv},
        ))

    def find_isomorphism(self, problem_a: np.ndarray, problem_b: np.ndarray) -> float:
        """Compute structural similarity between two problems."""
        a = problem_a[: self.dim]
        b = problem_b[: self.dim]
        if len(a) < self.dim:
            a = np.pad(a, (0, self.dim - len(a)))
        if len(b) < self.dim:
            b = np.pad(b, (0, self.dim - len(b)))
        if _HAS_JIT:
            return float(_cosine_similarity(
                np.ascontiguousarray(a, dtype=float),
                np.ascontiguousarray(b, dtype=float),
            ))
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        return float(cos_sim)

    def transfer_solution(
        self, source_cat: str, target_cat: str, solution: np.ndarray
    ) -> np.ndarray:
        """Transfer a solution between domains via functor."""
        for functor in self.functors:
            if functor.source == source_cat and functor.target == target_cat:
                return functor.apply(solution)
        return solution

    def process(self, signal: Signal) -> Signal:
        truth = self.topos.classify(signal.data)
        for cat_name, cat in self.categories.items():
            for _obj_name, obj_repr in cat.objects.items():
                similarity = self.find_isomorphism(signal.data, obj_repr)
                if similarity > 0.8:
                    for functor in self.functors:
                        if functor.source == cat_name:
                            transferred = functor.apply(obj_repr)
                            return Signal(data=transferred, metadata={"transferred_from": cat_name})
        return Signal(data=truth, metadata={"classified": True})

    def predict(self, signal: Signal) -> Prediction:
        truth = self.topos.classify(signal.data)
        return Prediction(value=truth, uncertainty=float(1.0 - np.mean(np.abs(truth))))

    def update(self, prediction_error: float) -> None:
        noise = np.random.randn(self.dim, self.dim) * prediction_error * 0.001
        self.topos.classifier += noise
