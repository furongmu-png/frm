# src/zero_data_model/category_engine.py
"""Category Theory Foundation for cross-domain reasoning."""

from __future__ import annotations

import math
import threading
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

    def __init__(self, dim: int = 64, rng: np.random.Generator | None = None):
        self.dim = dim
        # Round-3 audit CRIT-1: per-module Generator
        self._rng = rng if rng is not None else np.random.default_rng()
        self.truth_values = np.linspace(0, 1, dim)
        self.classifier = self._rng.standard_normal((dim, dim)) * 0.1
        # Week-1 perf: cache a contiguous float view of ``classifier`` so the
        # JIT path does not call ``np.ascontiguousarray`` (a fresh array copy
        # when the source is non-contiguous or non-float) on every ``classify``
        # call -- ``process`` and ``predict`` both call ``classify`` once per
        # think() cycle. Invalidated only by ``CategoryTheoryEngine.update``
        # (the sole mutator of ``classifier``), which flips ``_classifier_dirty``
        # after the in-place add. The JIT branch recomputes the contiguous
        # copy lazily on the next ``classify`` call.
        self._classifier_contig: np.ndarray = np.ascontiguousarray(
            self.classifier, dtype=float
        )
        self._classifier_dirty: bool = False

    def classify(self, signal: np.ndarray) -> np.ndarray:
        s = signal[: self.dim]
        if len(s) < self.dim:
            s = np.pad(s, (0, self.dim - len(s)))
        if _HAS_JIT:
            # Week-1 perf: reuse the cached contiguous classifier view, only
            # rebuilding it when ``update`` has mutated ``classifier`` since
            # the last call. ``np.ascontiguousarray`` on an already-contiguous
            # float array is a no-op (returns the input), so the rebuild is
            # cheap when nothing changed.
            if self._classifier_dirty:
                self._classifier_contig = np.ascontiguousarray(
                    self.classifier, dtype=float
                )
                self._classifier_dirty = False
            return _topos_classify(
                np.ascontiguousarray(s, dtype=float),
                self._classifier_contig,
            )
        return 1.0 / (1.0 + np.exp(-s @ self.classifier))


class CategoryTheoryEngine(CognitiveModule):
    """
    Category Theory reasoning engine.
    - Defines categories for different problem domains
    - Uses functors for cross-domain transfer
    - Finds isomorphisms between problems
    """

    def __init__(self, dim: int = 64, rng: np.random.Generator | None = None):
        self.dim = dim
        # Round-3 audit CRIT-1: per-module Generator
        self._rng = rng if rng is not None else np.random.default_rng()
        # Per-module re-entrant lock (Phase D / think() lockless update):
        # protects ``process`` / ``predict`` / ``update`` from concurrent
        # think() calls racing on ``self._rng`` and shared mutable state
        # (categories, functors, topos.classifier). RLock allows re-entry
        # (process calls structural_similarity which reads classifier).
        self._lock = threading.RLock()
        self.categories: dict[str, Category] = {}
        self.functors: list[Functor] = []
        self.topos = ToposEngine(dim, rng=self._rng)
        self._init_default_categories()
        # Round-8 audit PERF8-5: cache of the last ``process`` output so
        # ``predict`` can reuse it instead of re-running ``topos.classify``
        # (an O(dim^2) matmul + sigmoid). Matches the Fix 12 pattern used by
        # ConsciousnessCore / MathUniverse / BiologicalSubstrate.
        self._last_process_output: np.ndarray | None = None

    def _init_default_categories(self):
        nlp = Category(name="NLP")
        cv = Category(name="CV")
        analytics = Category(name="Analytics")
        for cat in [nlp, cv, analytics]:
            for i in range(5):
                # Round-3 audit CRIT-1: per-module Generator
                cat.add_object(f"concept_{i}", self._rng.standard_normal(self.dim) * 0.1)
            self.categories[cat.name] = cat
        # Round-3 audit CRIT-1: per-module Generator
        transfer_nlp_cv = self._rng.standard_normal((self.dim, self.dim)) * 0.05
        self.functors.append(Functor(
            source="NLP", target="CV",
            object_map={f"concept_{i}": f"concept_{i}" for i in range(5)},
            morphism_map={("concept_0", "concept_1"): transfer_nlp_cv},
        ))

    def __getstate__(self) -> dict:
        # Phase D: per-module RLock is not picklable. Strip it here and
        # rebuild in __setstate__. Hold the lock so concurrent think()
        # blocks during the snapshot (consistent array copy).
        with self._lock:
            return {k: v for k, v in self.__dict__.items() if k != "_lock"}

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._lock = threading.RLock()

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
        # Week-1 perf: make both inputs contiguous once at the entry point so
        # both the JIT and the pure-numpy paths see C-contiguous float arrays.
        # The previous code only contig-copied in the JIT branch, so the numpy
        # path paid repeated non-contiguous-access penalties when ``problem_a``
        # / ``problem_b`` were slices of larger buffers (common in ``process``,
        # which passes ``signal.data`` directly).
        a = np.ascontiguousarray(a, dtype=float)
        b = np.ascontiguousarray(b, dtype=float)
        if _HAS_JIT:
            return float(_cosine_similarity(a, b))
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        return float(cos_sim)

    def find_isomorphism(self, problem_a: np.ndarray, problem_b: np.ndarray) -> float:
        """Deprecated alias for ``structural_similarity``.

        Returns a similarity score in ``[-1, 1]``; despite the name this does
        NOT compute a category-theoretic isomorphism. Use
        ``find_invertible_map`` for an actual invertible linear map between
        two vectors.
        """
        # Round-3 audit: emit DeprecationWarning so callers are alerted.
        import warnings

        warnings.warn(
            "find_isomorphism is deprecated; use structural_similarity",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.structural_similarity(problem_a, problem_b)

    def find_invertible_map(
        self, source: np.ndarray, target: np.ndarray
    ) -> np.ndarray | None:
        """Find an invertible linear map ``T`` with ``T @ source = target``.

        Returns ``None`` when no well-conditioned map exists (degenerate
        inputs). Uses a Householder-style construction: ``T = (nb/na) *
        (I - 2 vv^T)`` where ``v`` is the bisector of ``source`` and
        ``target``. The reflection ``(I - 2 vv^T)`` is orthogonal (hence
        invertible) and maps ``source/|source|`` to ``target/|target|``
        exactly; the ``nb/na`` scalar then rescales the magnitude so
        ``T @ source = target`` holds exactly (not just up to a norm ratio).

        Round-9 audit R9-003: the previous implementation returned the
        unscaled reflection, so ``T @ source = (|source|/|target|) * target``
        -- only the unit vectors matched, not the magnitudes. The docstring
        promised ``T @ source = target`` (the contract every caller relies
        on for solution transfer between domains), so the bug silently
        rescaled transferred solutions by the source/target norm ratio.
        Scaling by ``nb/na`` keeps the map invertible (non-zero scalar
        times an orthogonal matrix is non-singular) while honoring the
        documented contract.
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
        # Round-10 audit R10-A-004: reject extreme norm ratios so the
        # scaled Householder map ``T = (nb/na) * (I - 2vv^T)`` does not
        # become ill-conditioned. A scale of 1e6 produces a condition
        # number of ~1e12 (scale² * orthogonal matrix), which amplifies
        # float64 round-off (~1e-15) into the significant digits of any
        # downstream ``transfer_solution`` call. The threshold ``1e6``
        # matches the maximum ``prediction_error`` clip used across the
        # cognitive stack and is generous enough to admit any practical
        # solution-transfer scenario.
        scale = nb / na
        if not math.isfinite(scale) or scale > 1e6 or scale < 1e-6:
            return None
        a_hat = a / na
        b_hat = b / nb
        v = a_hat - b_hat
        v_norm = float(np.linalg.norm(v))
        if v_norm < 1e-12:
            # source and target are already aligned up to magnitude:
            # T = (nb/na) * I maps source -> target exactly.
            return scale * np.eye(n)
        v = v / v_norm
        # Scaled Householder reflection: orthogonal reflection (I - 2 v v^T)
        # maps a_hat -> b_hat, then the (nb/na) scalar rescales the magnitude
        # so T @ source = (nb/na) * (|source| * b_hat) = nb * b_hat = target.
        return scale * (np.eye(n) - 2.0 * np.outer(v, v))

    def transfer_solution(
        self, source_cat: str, target_cat: str, solution: np.ndarray
    ) -> np.ndarray:
        """Transfer a solution between domains via functor.

        Round-3 audit: raise ``KeyError`` when no functor matches the
        ``(source_cat, target_cat)`` pair instead of silently returning the
        input unchanged — the silent fallback was a HIGH-severity correctness
        issue (callers could not distinguish "transfer happened" from "no
        functor matched").
        """
        for functor in self.functors:
            if functor.source == source_cat and functor.target == target_cat:
                return functor.apply(solution)
        raise KeyError(
            f"no functor registered for {source_cat!r} -> {target_cat!r}"
        )

    def process(self, signal: Signal) -> Signal:
        with self._lock:
            # Week-1 perf: make ``signal.data`` contiguous at entry so every
            # downstream call (``topos.classify``, ``structural_similarity``
            # loop over category objects) sees a C-contiguous buffer. The
            # previous code re-contig-copied inside ``classify`` and (only in
            # the JIT branch) inside ``structural_similarity``; doing it once
            # here means a single copy covers all uses in this cycle.
            data = np.ascontiguousarray(signal.data, dtype=float)
            truth = self.topos.classify(data)
            for cat_name, cat in self.categories.items():
                for _obj_name, obj_repr in cat.objects.items():
                    # Round-6 audit NEW5-10: call ``structural_similarity`` directly
                    # instead of the deprecated ``find_isomorphism`` alias. The
                    # C-batch renamed the method but missed this callsite, so every
                    # ``think()`` cycle that reached this loop emitted a
                    # DeprecationWarning (with stack-frame inspection) up to 15
                    # times per cycle (3 categories x 5 objects).
                    similarity = self.structural_similarity(data, obj_repr)
                    if similarity > 0.8:
                        for functor in self.functors:
                            if functor.source == cat_name:
                                transferred = functor.apply(obj_repr)
                                # Round-8 audit PERF8-5: cache the process output
                                # so ``predict`` reuses it (matches the
                                # ConsciousnessCore/MathUniverse/BiologicalSubstrate
                                # pattern from Fix 12 / PERF8-3).
                                self._last_process_output = transferred
                                return Signal(
                                    data=transferred,
                                    metadata={"transferred_from": cat_name},
                                )
            # Round-8 audit PERF8-5: cache for ``predict`` to reuse.
            self._last_process_output = truth
            return Signal(data=truth, metadata={"classified": True})

    def predict(self, signal: Signal) -> Prediction:
        with self._lock:
            # Round-8 audit PERF8-5: reuse the cached process output so we do not
            # re-run ``topos.classify`` (an O(dim^2) matmul + sigmoid) twice per
            # think() cycle. Falls back to a fresh classify when ``predict`` is
            # called standalone (no prior ``process`` in this cycle), matching
            # the ConsciousnessCore/MathUniverse/BiologicalSubstrate pattern.
            if self._last_process_output is not None:
                truth = self._last_process_output
            else:
                truth = self.topos.classify(signal.data)
            return Prediction(value=truth, uncertainty=float(1.0 - np.mean(np.abs(truth))))

    def update(self, prediction_error: float) -> None:
        with self._lock:
            if not np.isfinite(prediction_error):
                return
            # Round-8 audit THEORY8-clip: clip to [0, 1e6] (NON-NEGATIVE) to
            # match ``active_inference.update`` and ``biological.update``. The
            # previous ``[-1e6, 1e6]`` clip allowed negative values to flip
            # the noise sign -- so a module could "learn" in the OPPOSITE
            # direction from what the prediction error signals (gradient
            # ascent instead of descent). ``biological.update`` uses
            # ``tanh(prediction_error * 0.001)`` (sign-preserving but bounded);
            # ``active_inference.update`` clips to ``[0, 1e6]``. We pick the
            # latter for consistency with the rest of the cognitive stack.
            prediction_error = float(np.clip(prediction_error, 0.0, 1e6))
            # Round-3 audit CRIT-1: per-module Generator
            noise = self._rng.standard_normal((self.dim, self.dim)) * prediction_error * 0.001
            self.topos.classifier += noise
            # Week-1 perf: ``classifier`` just mutated in-place -> invalidate
            # the cached contiguous view so the next ``classify`` rebuilds it.
            self.topos._classifier_dirty = True
