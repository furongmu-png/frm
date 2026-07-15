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
    """Structure-preserving map between categories.

    A functor ``F: C -> D`` maps each object ``X`` in ``C`` to an object
    ``F(X)`` in ``D`` (via ``object_map``) and each morphism ``f: X -> Y`` in
    ``C`` to a morphism ``F(f): F(X) -> F(Y)`` in ``D`` (via
    ``morphism_map``), preserving identity and composition.
    """
    source: str
    target: str
    object_map: dict[str, str]
    morphism_map: dict[tuple[str, str], np.ndarray]

    def apply_morphism(
        self, source_obj: str, target_obj: str, vector: np.ndarray
    ) -> np.ndarray | None:
        """Apply the functor's image of a single named morphism to a vector.

        This is the *real* functor action on a morphism: look up the morphism
        ``(source_obj -> target_obj)`` in the source category, return its
        image under F (a matrix in ``morphism_map``), and apply it to
        ``vector``. Returns ``None`` when no such morphism is registered or
        the shape is incompatible -- callers can then fall back to the
        shape-matched ``apply``.
        """
        transform = self.morphism_map.get((source_obj, target_obj))
        if transform is None:
            return None
        if transform.shape[1] != vector.shape[0]:
            return None
        return transform @ vector

    def compose_morphisms(
        self,
        src_a: str,
        mid: str,
        tgt: str,
        vector: np.ndarray,
    ) -> np.ndarray | None:
        """Apply ``F(g) . F(f)`` to ``vector`` where ``f: src_a -> mid`` and
        ``g: mid -> tgt``.

        Real functors preserve composition: ``F(g ∘ f) = F(g) ∘ F(f)``.
        Returns ``None`` if either morphism is missing or shapes mismatch.
        """
        first = self.apply_morphism(src_a, mid, vector)
        if first is None:
            return None
        second = self.apply_morphism(mid, tgt, first)
        return second

    def apply(self, obj: np.ndarray) -> np.ndarray:
        """Apply the functor to an object vector.

        Looks up the morphism via ``object_map`` first (Fix 13): the original
        implementation picked the first shape-matching morphism and ignored
        ``object_map`` entirely, which made cross-domain transfer
        indistinguishable from random shape matching. Falls back to shape
        matching only when no object_map entry fits.
        """
        # Try object_map first: each entry maps a source object name to a
        # target object name; we then look up the morphism keyed by the
        # (source, target) pair. We match by shape because callers pass raw
        # arrays, not object names.
        for src_obj_name, tgt_obj_name in self.object_map.items():
            # We cannot do a name lookup from a raw array, so we still match
            # by shape here; the (src, tgt) key selects the right morphism.
            result = self.apply_morphism(src_obj_name, tgt_obj_name, obj)
            if result is not None:
                return result
        # Fallback: shape matching against any morphism (preserves the
        # original behaviour when object_map has no usable entry).
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

    def structural_similarity(self, problem_a: np.ndarray, problem_b: np.ndarray) -> float:
        """Cosine similarity in ``[-1, 1]`` between two problem vectors.

        C-batch fix: this was previously called ``find_isomorphism``, but the
        implementation is a *similarity score* (cosine of the angle between
        the two vectors), not an isomorphism (which would be an invertible
        structure-preserving map). The name has been corrected; the old
        name is kept as a deprecated alias for backward compatibility.
        """
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

    def find_isomorphism(self, problem_a: np.ndarray, problem_b: np.ndarray) -> float:
        """Deprecated alias for ``structural_similarity``.

        Returns a similarity score in ``[-1, 1]``; despite the name this does
        NOT compute a category-theoretic isomorphism. Use
        ``find_invertible_map`` for an actual invertible linear map between
        two vectors.
        """
        return self.structural_similarity(problem_a, problem_b)

    def find_invertible_map(
        self, source: np.ndarray, target: np.ndarray
    ) -> np.ndarray | None:
        """Find an invertible linear map ``T`` with ``T @ source = target``.

        Returns ``None`` when no well-conditioned map exists (degenerate
        inputs). Uses a Householder-style construction: ``T = I - 2 vv^T``
        where ``v`` is the bisector of ``source`` and ``target``, which is
        always orthogonal (hence invertible) and maps ``source/|source|`` to
        ``target/|target|`` exactly.
        """
        a = np.asarray(source, dtype=float).flatten()
        b = np.asarray(target, dtype=float).flatten()
        n = max(a.shape[0], b.shape[0])
        if a.shape[0] < n:
            a = np.pad(a, (0, n - a.shape[0]))
        if b.shape[0] < n:
            b = np.pad(b, (0, n - b.shape[0]))
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-12 or nb < 1e-12:
            return None
        a_hat = a / na
        b_hat = b / nb
        v = a_hat - b_hat
        v_norm = float(np.linalg.norm(v))
        if v_norm < 1e-12:
            # source and target are already aligned -> identity maps them.
            return np.eye(n)
        v = v / v_norm
        # Householder reflection: T = I - 2 v v^T (orthogonal, hence invertible).
        return np.eye(n) - 2.0 * np.outer(v, v)

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
        if not np.isfinite(prediction_error):
            return
        prediction_error = float(np.clip(prediction_error, -1e6, 1e6))
        noise = np.random.randn(self.dim, self.dim) * prediction_error * 0.001
        self.topos.classifier += noise
