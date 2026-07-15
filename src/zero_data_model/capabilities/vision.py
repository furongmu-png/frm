# src/zero_data_model/capabilities/vision.py
"""Computer Vision capability module for the zero-data cognitive model.

Composes the core cognitive modules (math universe, biological substrate,
active inference, consciousness, category theory) with a small rule library
(``VisionRules``: Sobel kernels, geometry) used as prior knowledge. No external
training data and no learned weights are required -- every operator below is a
deterministic, rule-based prior.
"""

from __future__ import annotations

import numpy as np

from ..active_inference import ActiveInferenceEngine
from ..biological import BiologicalSubstrate
from ..category_engine import CategoryTheoryEngine
from ..consciousness_core import ConsciousnessCore
from ..math_universe import MathematicalUniverse
from .rules import VisionRules


def _as_2d_float(image: np.ndarray) -> np.ndarray:
    """Coerce any array-like into a 2D float array.

    1D inputs are reshaped to ``(1, -1)`` so 2D convolution operators work.
    """
    img = np.asarray(image, dtype=float)
    if img.ndim == 1:
        img = img.reshape(1, -1)
    elif img.ndim != 2:
        # Collapse any extra leading dimensions onto the last two axes.
        img = img.reshape(-1, img.shape[-1])
    return img


class ImageEncoder:
    """Encode a 2D grayscale image into a fixed-length, L2-normalized vector.

    The representation is built entirely from self-generated structure:
    topological features (betti numbers + moments) and fractal compression
    statistics. No learned weights are involved.
    """

    def __init__(self, dim: int = 64, math_universe=None, rules: VisionRules | None = None):
        self.dim = dim
        self.rules = rules or VisionRules()
        self.math_universe = math_universe or MathematicalUniverse(dim=dim)

    def _downsample(self, img: np.ndarray) -> np.ndarray:
        """Strided downsample so the flattened length is on the order of ``dim``."""
        h, w = img.shape
        target_side = max(1, int(round(np.sqrt(self.dim))))
        sh = max(1, h // target_side)
        sw = max(1, w // target_side)
        return img[::sh, ::sw]

    def encode(self, image: np.ndarray) -> np.ndarray:
        """Encode a 2D image (grayscale, any size) into a ``dim``-length vector."""
        img = self._downsample(_as_2d_float(image))
        flat = img.flatten()

        topo = self.math_universe.topology.topological_features(flat)
        fractal = self.math_universe.fractal.compress(flat)
        stats = np.array(
            [fractal["mean"], fractal["std"], fractal["self_similarity"]],
            dtype=float,
        )
        # Connected-components / correlation stats can be NaN for degenerate
        # (constant) inputs; sanitize before combining.
        stats = np.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)

        combined = np.concatenate([stats, np.asarray(topo, dtype=float)])
        if combined.shape[0] > self.dim:
            combined = combined[: self.dim]
        elif combined.shape[0] < self.dim:
            combined = np.pad(combined, (0, self.dim - combined.shape[0]))

        norm = float(np.linalg.norm(combined))
        if norm > 1e-8:
            combined = combined / norm
        return combined


class FeatureExtractor:
    """Extract rule-based features from an image using no learned weights."""

    def __init__(self, dim: int = 64, biological=None, rules: VisionRules | None = None):
        self.dim = dim
        self.rules = rules or VisionRules()
        self.biological = biological or BiologicalSubstrate(dim=dim)

    def extract(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Extract edges, texture, morphology, and statistics features."""
        img = _as_2d_float(image)

        # Edges: Sobel magnitude (rule-based gradient).
        gx = self.rules.convolve(img, self.rules.sobel_x)
        gy = self.rules.convolve(img, self.rules.sobel_y)
        edges = np.sqrt(gx ** 2 + gy ** 2).flatten()

        # Texture: cellular automaton evolution seeded by a binarized image row.
        binary = (img >= img.mean()).astype(int)
        ca = self.biological.automata
        row = binary[0] if binary.shape[0] > 0 else np.zeros(1, dtype=int)
        state = np.zeros(ca.size, dtype=int)
        n = min(len(row), ca.size)
        state[:n] = row[:n]
        ca.state = state
        history = ca.evolve(n_steps=5)
        texture = history.flatten().astype(float)

        # Morphology: self-organizing morphogenetic pattern.
        morph = self.biological.morphogenetic.develop(n_steps=10)
        morphology = morph.flatten().astype(float)

        # Stats: basic intensity summary.
        stats = np.array(
            [float(img.mean()), float(img.std()), float(img.min()), float(img.max())]
        )

        return {
            "edges": edges,
            "texture": texture,
            "morphology": morphology,
            "stats": stats,
        }


class PatternRecognizer:
    """Recognize a shape by matching against self-synthesized rule-based prototypes."""

    def __init__(
        self,
        dim: int = 64,
        active_inference: ActiveInferenceEngine | None = None,
        consciousness: ConsciousnessCore | None = None,
        encoder: ImageEncoder | None = None,
        rules: VisionRules | None = None,
    ):
        self.dim = dim
        self.rules = rules or VisionRules()
        self.active_inference = active_inference
        self.consciousness = consciousness
        self.encoder = encoder or ImageEncoder(dim=dim)
        # Precompute each prototype's encoding once (P-HIGH-06): recognize()
        # previously re-synthesized and re-encoded every prototype per call.
        self._proto_encodings: dict[str, np.ndarray] = {
            shape: self.encoder.encode(self._synthesize_prototype(shape))
            for shape in self.rules.shapes
        }

    def _synthesize_prototype(self, shape: str, size: int = 16) -> np.ndarray:
        """Deterministically draw a named shape on a square grid (no learning)."""
        img = np.zeros((size, size), dtype=float)
        cy, cx = size // 2, size // 2
        yy, xx = np.indices((size, size))

        if shape == "circle":
            img[(yy - cy) ** 2 + (xx - cx) ** 2 <= (size // 3) ** 2] = 1.0
        elif shape == "square":
            img[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 1.0
        elif shape == "triangle":
            # Lower-triangular fill (a right triangle), as a rule-based prior.
            img = np.tri(size, dtype=float)
        elif shape == "line":
            for i in range(size):
                img[i, i] = 1.0
        elif shape == "blob":
            # Deterministic RNG seeded by a stable hash of the shape name.
            seed = sum(ord(c) for c in shape)
            rng = np.random.default_rng(seed)
            max_r = max(2, size // 4)
            for _ in range(5):
                by = int(rng.integers(0, size))
                bx = int(rng.integers(0, size))
                r = int(rng.integers(1, max_r))
                img[(yy - by) ** 2 + (xx - bx) ** 2 <= r ** 2] = 1.0
        return img

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def recognize(self, image: np.ndarray) -> tuple[str, float]:
        """Recognize a pattern by comparing the encoding to prototype encodings."""
        input_enc = self.encoder.encode(image)

        best_shape: str | None = None
        best_score = -float("inf")
        for shape in self.rules.shapes:
            proto_enc = self._proto_encodings[shape]
            score = self._cosine(input_enc, proto_enc)
            if self.active_inference is not None:
                # Free energy of the DIFFERENCE between input and prototype
                # (Fix 6): the original used ``input_enc`` alone, which is the
                # same for every prototype and so had zero effect on ranking.
                free_energy = float(
                    self.active_inference.compute_free_energy(input_enc - proto_enc)
                )
                score = score - 0.01 * free_energy
            if score > best_score:
                best_score = score
                best_shape = shape

        # Map the (possibly active-inference-adjusted) score into [0, 1].
        confidence = float(np.clip((best_score + 1.0) / 2.0, 0.0, 1.0))
        assert best_shape is not None  # rules.shapes is non-empty by construction
        return best_shape, confidence


class ShapeAnalyzer:
    """Analyze geometric/topological shape properties using category theory + rules."""

    def __init__(
        self,
        dim: int = 64,
        category_engine: CategoryTheoryEngine | None = None,
        rules: VisionRules | None = None,
    ):
        self.dim = dim
        self.rules = rules or VisionRules()
        self.category_engine = category_engine

    def analyze(self, image: np.ndarray) -> dict:
        """Return aspect ratio, symmetry, complexity, and betti-0 of the image."""
        img = _as_2d_float(image)
        return {
            "aspect_ratio": self._aspect_ratio(img),
            "symmetry": self._symmetry(img),
            "complexity": self._complexity(img),
            "beti0": self._betti0(img),
        }

    def _aspect_ratio(self, img: np.ndarray) -> float:
        nonzero = np.argwhere(img != 0)
        if len(nonzero) == 0:
            return 1.0
        min_y, min_x = nonzero.min(axis=0)
        max_y, max_x = nonzero.max(axis=0)
        height = int(max_y - min_y + 1)
        width = int(max_x - min_x + 1)
        if width == 0:
            return 1.0
        return float(height) / float(width)

    def _symmetry(self, img: np.ndarray) -> float:
        mirrored = np.fliplr(img)
        a = img.flatten()
        b = mirrored.flatten()
        sa = float(np.std(a))
        sb = float(np.std(b))
        if sa < 1e-8 or sb < 1e-8:
            return 1.0 if np.allclose(a, b) else 0.0
        corr = float(np.corrcoef(a, b)[0, 1])
        if not np.isfinite(corr):
            return 0.0
        return float(np.clip(corr, 0.0, 1.0))

    def _complexity(self, img: np.ndarray) -> float:
        gx = self.rules.convolve(img, self.rules.sobel_x)
        gy = self.rules.convolve(img, self.rules.sobel_y)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        return float(np.std(mag))

    def _betti0(self, img: np.ndarray) -> float:
        # Use a category-engine-backed topology analyzer when one is available;
        # otherwise fall back to a rule-based connected-components count.
        if self.category_engine is not None and hasattr(self.category_engine, "topology"):
            betti = self.category_engine.topology.compute_betti_numbers(img.flatten())
            return float(betti.get(0, 1))
        binary = (img >= img.mean()).astype(int)
        # scipy.ndimage.label is ~200-800x faster than the previous
        # double for + DFS implementation (P-HIGH-01). The cross structure
        # preserves the previous 4-connectivity semantics.
        from scipy.ndimage import label as _scipy_label  # noqa: PLC0415

        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        _, num_features = _scipy_label(binary, structure=structure)
        return float(num_features)
