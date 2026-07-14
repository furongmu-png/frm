"""Domain rule libraries — prior knowledge encoded as rules, not data.

These are not learned from external datasets. They are minimal inductive
biases (linguistic universals, image-processing kernels, statistical priors)
that the zero-data capabilities compose with self-generated representations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class DomainRules:
    """Base class for domain-specific rule libraries."""

    def __init__(self):
        self.rules: dict[str, object] = {}

    def get(self, name: str, default=None):
        return self.rules.get(name, default)


@dataclass
class NLPRules(DomainRules):
    """Linguistic priors: stop-words, punctuation, char categories."""

    stopwords: set[str] = field(default_factory=lambda: {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "and", "or", "but", "if", "then", "of", "to", "in", "on", "at", "by",
        "for", "with", "from", "as", "it", "its", "this", "that", "these",
        "those", "i", "you", "he", "she", "we", "they", "me", "him", "her",
        "us", "them", "my", "your", "his", "hers", "our", "their",
    })
    punctuation: str = ".,;:!?-\"'()[]{}"
    positive_lexicon: set[str] = field(default_factory=lambda: {
        "good", "great", "excellent", "happy", "love", "wonderful", "best",
        "amazing", "perfect", "beautiful", "bright", "calm", "kind",
    })
    negative_lexicon: set[str] = field(default_factory=lambda: {
        "bad", "terrible", "awful", "sad", "hate", "worst", "horrible",
        "ugly", "dark", "angry", "broken", "wrong", "fail",
    })
    topic_keywords: dict[str, set[str]] = field(default_factory=lambda: {
        "tech": {"code", "data", "model", "system", "compute", "algorithm", "network"},
        "nature": {"tree", "river", "mountain", "sky", "ocean", "forest", "flower"},
        "emotion": {"happy", "sad", "love", "fear", "joy", "anger", "hope"},
        "science": {"energy", "force", "mass", "light", "atom", "quantum", "field"},
    })

    def __post_init__(self):
        super().__init__()
        self.rules = {
            "stopwords": self.stopwords,
            "punctuation": self.punctuation,
            "positive": self.positive_lexicon,
            "negative": self.negative_lexicon,
            "topics": self.topic_keywords,
        }

    def tokenize(self, text: str) -> list[str]:
        """Lightweight rule-based tokenizer (no external library)."""
        cleaned = text.lower()
        for ch in self.punctuation:
            cleaned = cleaned.replace(ch, " ")
        return [t for t in cleaned.split() if t and t not in self.stopwords]

    def sentiment_prior(self, tokens: list[str]) -> float:
        """Return a prior sentiment score in [-1, 1] from lexicon rules."""
        pos = sum(1 for t in tokens if t in self.positive_lexicon)
        neg = sum(1 for t in tokens if t in self.negative_lexicon)
        total = pos + neg
        if total == 0:
            return 0.0
        return (pos - neg) / total


@dataclass
class VisionRules(DomainRules):
    """Image-processing priors: kernels, geometry constants."""

    sobel_x: np.ndarray = field(default_factory=lambda: np.array([
        [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]
    ], dtype=float))
    sobel_y: np.ndarray = field(default_factory=lambda: np.array([
        [-1, -2, -1], [0, 0, 0], [1, 2, 1]
    ], dtype=float))
    gaussian_kernel: np.ndarray = field(default_factory=lambda: np.array([
        [1, 2, 1], [2, 4, 2], [1, 2, 1]
    ], dtype=float) / 16.0)
    shapes: tuple[str, ...] = ("circle", "square", "triangle", "line", "blob")

    def __post_init__(self):
        super().__init__()
        self.rules = {
            "sobel_x": self.sobel_x,
            "sobel_y": self.sobel_y,
            "gaussian": self.gaussian_kernel,
            "shapes": self.shapes,
        }

    def convolve(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """2D convolution with zero padding (rule-based operator)."""
        kh, kw = kernel.shape
        ph, pw = kh // 2, kw // 2
        padded = np.pad(image, ((ph, ph), (pw, pw)), mode="constant")
        out = np.zeros_like(image, dtype=float)
        h, w = image.shape
        for i in range(h):
            for j in range(w):
                out[i, j] = np.sum(padded[i:i + kh, j:j + kw] * kernel)
        return out


@dataclass
class AnalyticsRules(DomainRules):
    """Statistical priors for time-series analysis."""

    default_window: int = 5
    anomaly_z_threshold: float = 2.0
    seasons: tuple[int, ...] = (4, 7, 12, 24)
    trend_threshold: float = 0.05

    def __post_init__(self):
        super().__init__()
        self.rules = {
            "window": self.default_window,
            "z_threshold": self.anomaly_z_threshold,
            "seasons": self.seasons,
            "trend_threshold": self.trend_threshold,
        }

    def moving_average(self, series: np.ndarray, window: int | None = None) -> np.ndarray:
        """Rule-based moving average (no learned parameters)."""
        w = window or self.default_window
        if len(series) < w:
            return np.array([float(np.mean(series))] * len(series))
        cumsum = np.cumsum(np.insert(series, 0, 0))
        return (cumsum[w:] - cumsum[:-w]) / w

    def zscore(self, series: np.ndarray) -> np.ndarray:
        mu = float(np.mean(series))
        sigma = float(np.std(series)) + 1e-8
        return (series - mu) / sigma
