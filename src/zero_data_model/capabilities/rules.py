"""Domain rule libraries — prior knowledge encoded as rules, not data.

These are not learned from external datasets. They are minimal inductive
biases (linguistic universals, image-processing kernels, statistical priors)
that the zero-data capabilities compose with self-generated representations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import correlate as _nd_correlate


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
        # Round-6 audit NEW5-4: bound the input length so a direct (non-API)
        # caller cannot pass a multi-MB string that would make the
        # ``cleaned.replace(ch, " ")`` loop O(n · |punct|) with no early exit.
        # The API layer caps text at 8192 chars; this is a deeper guard.
        _MAX_TOKENIZE_CHARS = 65536
        if len(text) > _MAX_TOKENIZE_CHARS:
            raise ValueError(
                f"text length {len(text)} exceeds tokenize cap "
                f"{_MAX_TOKENIZE_CHARS}"
            )
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
        """2D correlation (zero padding) via ``scipy.ndimage`` (Fix 10).

        The previous pure-Python double loop was the dominant cost in
        ``FeatureExtractor.extract`` and ``ShapeAnalyzer._complexity``;
        ``scipy.ndimage.correlate`` is a compiled C kernel and is already a
        dependency (scipy is required by ``pyproject.toml``).
        """
        return _nd_correlate(image.astype(float), kernel, mode="constant")


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


@dataclass
class AudioRules(DomainRules):
    """Audio-processing priors: STFT parameters, mel scale, onset thresholds.

    All defaults follow common speech-processing conventions (16 kHz mono,
    1024-point STFT, 26 mel bins). No learned parameters are involved.
    """

    sample_rate: int = 16000
    frame_size: int = 1024
    hop_size: int = 512
    n_mels: int = 26
    onset_threshold: float = 0.3
    pitch_min_hz: float = 80.0
    pitch_max_hz: float = 500.0
    texture_labels: tuple[str, ...] = ("speech", "music", "noise", "silence")

    def __post_init__(self):
        super().__init__()
        self.rules = {
            "sample_rate": self.sample_rate,
            "frame_size": self.frame_size,
            "hop_size": self.hop_size,
            "n_mels": self.n_mels,
            "onset_threshold": self.onset_threshold,
            "pitch_min_hz": self.pitch_min_hz,
            "pitch_max_hz": self.pitch_max_hz,
            "texture_labels": self.texture_labels,
        }

    @staticmethod
    def hz_to_mel(hz: float) -> float:
        """Convert Hz to mel scale (rule-based, 1127 * ln(1 + hz/700))."""
        return 1127.0 * float(np.log1p(hz / 700.0))

    @staticmethod
    def mel_to_hz(mel: float) -> float:
        """Convert mel to Hz (inverse of ``hz_to_mel``)."""
        return 700.0 * (float(np.exp(mel / 1127.0)) - 1.0)


@dataclass
class GraphRules(DomainRules):
    """Graph-processing priors: community resolution, path heuristics."""

    default_weight: float = 1.0
    community_resolution: float = 1.0
    path_heuristic_weight: float = 1.0
    centrality_normalized: bool = True
    isomorphism_max_iter: int = 5

    def __post_init__(self):
        super().__init__()
        self.rules = {
            "default_weight": self.default_weight,
            "community_resolution": self.community_resolution,
            "path_heuristic_weight": self.path_heuristic_weight,
            "centrality_normalized": self.centrality_normalized,
            "isomorphism_max_iter": self.isomorphism_max_iter,
        }


@dataclass
class RoboticsRules(DomainRules):
    """Robotics priors: control timestep, velocity/acceleration limits."""

    dt: float = 0.01
    max_velocity: float = 1.0
    max_acceleration: float = 5.0
    arm_segments: int = 3
    arm_length: float = 1.0
    safety_margin: float = 0.05
    mpc_horizon: int = 10

    def __post_init__(self):
        super().__init__()
        self.rules = {
            "dt": self.dt,
            "max_velocity": self.max_velocity,
            "max_acceleration": self.max_acceleration,
            "arm_segments": self.arm_segments,
            "arm_length": self.arm_length,
            "safety_margin": self.safety_margin,
            "mpc_horizon": self.mpc_horizon,
        }
