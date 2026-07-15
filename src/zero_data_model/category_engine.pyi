from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base import CognitiveModule, Prediction, Signal

@dataclass
class Category:
    """A category with objects and morphisms."""
    name: str
    objects: dict[str, np.ndarray] = field(default_factory=dict)
    morphisms: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)

    def add_object(self, name: str, representation: np.ndarray) -> None: ...

    def add_morphism(
        self, source: str, target: str, transform: np.ndarray
    ) -> None: ...

    def compose(
        self, source: str, intermediate: str, target: str
    ) -> np.ndarray | None: ...


@dataclass
class Functor:
    """Structure-preserving map between categories."""
    source: str
    target: str
    object_map: dict[str, str]
    morphism_map: dict[tuple[str, str], np.ndarray]

    def apply_morphism(
        self, source_obj: str, target_obj: str, vector: np.ndarray
    ) -> np.ndarray | None: ...

    def compose_morphisms(
        self,
        src_a: str,
        mid: str,
        tgt: str,
        vector: np.ndarray,
    ) -> np.ndarray | None: ...

    def apply(self, obj: np.ndarray) -> np.ndarray: ...


class ToposEngine:
    """Topos theory: subobject classifier for truth values."""

    dim: int
    truth_values: np.ndarray
    classifier: np.ndarray

    def __init__(self, dim: int = 64, rng: np.random.Generator | None = None) -> None: ...

    def classify(self, signal: np.ndarray) -> np.ndarray: ...


class CategoryTheoryEngine(CognitiveModule):
    """
    Category Theory reasoning engine.
    - Defines categories for different problem domains
    - Uses functors for cross-domain transfer
    - Finds isomorphisms between problems
    """

    dim: int
    categories: dict[str, Category]
    functors: list[Functor]
    topos: ToposEngine

    def __init__(self, dim: int = 64, rng: np.random.Generator | None = None) -> None: ...

    def _init_default_categories(self) -> None: ...

    def structural_similarity(
        self, problem_a: np.ndarray, problem_b: np.ndarray
    ) -> float:
        """Cosine similarity in ``[-1, 1]`` between two problem vectors."""
        ...

    def find_isomorphism(
        self, problem_a: np.ndarray, problem_b: np.ndarray
    ) -> float:
        """Deprecated alias for ``structural_similarity``.

        Returns a similarity score in ``[-1, 1]``; despite the name this does
        NOT compute a category-theoretic isomorphism. Use
        ``find_invertible_map`` for an actual invertible linear map between
        two vectors.
        """
        ...

    def find_invertible_map(
        self, source: np.ndarray, target: np.ndarray
    ) -> np.ndarray | None:
        """Find an invertible linear map ``T`` with ``T @ source = target``."""
        ...

    def transfer_solution(
        self,
        source_cat: str,
        target_cat: str,
        solution: np.ndarray,
    ) -> np.ndarray:
        """Transfer a solution between domains via functor."""
        ...

    def process(self, signal: Signal) -> Signal: ...

    def predict(self, signal: Signal) -> Prediction: ...

    def update(self, prediction_error: float) -> None: ...
