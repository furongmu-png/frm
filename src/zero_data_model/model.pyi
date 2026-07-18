from __future__ import annotations

import threading
from typing import Any

import numpy as np

from .active_inference import ActiveInferenceEngine
from .base import CognitiveModule, Signal
from .biological import BiologicalSubstrate
from .capabilities.analytics import (
    AnomalyDetector,
    PatternMiner,
    TimeSeriesForecaster,
    TrendAnalyzer,
)
from .capabilities.analytics_advanced import (
    BayesianEstimator,
    CausalInference,
    ChangePointDetector,
)
from .capabilities.nlp import (
    SemanticComparator,
    TextEncoder,
    TextGenerator,
    ZeroShotClassifier,
)
from .capabilities.nlp_advanced import (
    MultiLingualEncoder,
    SentenceEncoder,
    SyntacticAnalyzer,
)
from .capabilities.rules import AnalyticsRules, NLPRules, VisionRules
from .capabilities.vision import (
    FeatureExtractor,
    ImageEncoder,
    PatternRecognizer,
    ShapeAnalyzer,
)
from .capabilities.vision_advanced import (
    DepthEstimator,
    PointCloudEncoder,
    VideoFrameAnalyzer,
)
from .category_engine import CategoryTheoryEngine
from .consciousness_core import ConsciousnessCore
from .hardware.parallel import ParallelExecutor
from .math_universe import MathematicalUniverse
from .quantum_hybrid import QuantumClassicalHybrid

class ZeroDataModel:
    """
    A self-sufficient cognitive system requiring no external data.

    Core Architecture:
    - Consciousness Core: perception, attention, self-reflection
    - Active Inference: free energy minimization, epistemic foraging
    - Category Theory: cross-domain reasoning, isomorphism detection
    - Quantum-Classical Hybrid: parallel exploration, optimization
    - Biological Substrate: DNA storage, morphogenesis, cellular automata
    - Mathematical Universe: information geometry, topology, fractals
    """

    dim: int
    # Seeded-RNG / concurrency-safety fields (Fix 9 / Fix 7 / B-CRIT-01).
    _seed: int | None
    _rng: np.random.Generator
    _lock: threading.RLock
    consciousness: ConsciousnessCore
    active_inference: ActiveInferenceEngine
    category_engine: CategoryTheoryEngine
    quantum_hybrid: QuantumClassicalHybrid
    biological: BiologicalSubstrate
    math_universe: MathematicalUniverse
    modules: list[CognitiveModule]
    nlp_text_encoder: TextEncoder
    nlp_comparator: SemanticComparator
    nlp_classifier: ZeroShotClassifier
    nlp_generator: TextGenerator
    vision_encoder: ImageEncoder
    vision_features: FeatureExtractor
    vision_recognizer: PatternRecognizer
    vision_analyzer: ShapeAnalyzer
    analytics_forecaster: TimeSeriesForecaster
    analytics_anomaly: AnomalyDetector
    analytics_miner: PatternMiner
    analytics_trend: TrendAnalyzer
    nlp_rules: NLPRules
    vision_rules: VisionRules
    analytics_rules: AnalyticsRules
    nlp_multilingual: MultiLingualEncoder
    nlp_syntactic: SyntacticAnalyzer
    nlp_sentence_encoder: SentenceEncoder
    vision_point_cloud: PointCloudEncoder
    vision_video: VideoFrameAnalyzer
    vision_depth: DepthEstimator
    analytics_causal: CausalInference
    analytics_bayesian: BayesianEstimator
    analytics_changepoint: ChangePointDetector
    parallel_executor: ParallelExecutor
    cycle_count: int

    def __init__(self, dim: int = 64, seed: int | None = None) -> None: ...

    # Round-8 audit R8-HIGH-3: pickle / deepcopy support. Drop the
    # unpicklable ``_lock`` + ``parallel_executor`` and rebuild on restore.
    def __getstate__(self) -> dict[str, Any]: ...
    def __setstate__(self, state: dict[str, Any]) -> None: ...

    @property
    def hardware_info(self) -> dict[str, Any]:
        """Report the active hardware backends for diagnostics."""
        ...

    def think(self, input_data: np.ndarray | None = None) -> Signal:
        """Process a thought cycle. If no input, self-generates from internal state."""
        ...

    def _self_generate(self) -> Signal: ...
    def _integrate(
        self,
        signals: list[Signal],
        uncertainties: np.ndarray | None = None,
    ) -> Signal: ...

    def solve(self, problem: np.ndarray) -> Signal:
        """Solve an optimization problem using quantum annealing."""
        ...

    def find_analogies(self, problem_a: np.ndarray, problem_b: np.ndarray) -> float:
        """Find structural similarity between two problems."""
        ...

    def generate_knowledge(self, query: str = "") -> Signal:
        """Self-generate knowledge without external data."""
        ...

    # --- NLP capabilities ---

    def encode_text(self, text: str) -> np.ndarray:
        """Encode text into a fixed-dim vector (zero-data)."""
        ...

    def text_similarity(self, a: str, b: str) -> float:
        """Compute semantic similarity between two texts in [0, 1]."""
        ...

    def classify_text(self, text: str) -> tuple[str, float]:
        """Zero-shot classify text into a topic (tech/nature/emotion/science)."""
        ...

    def generate_text(self, seed: str, length: int = 32) -> str:
        """Generate text from a seed with no external data."""
        ...

    # --- Vision capabilities ---

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """Encode a 2D image into a fixed-dim vector (zero-data)."""
        ...

    def extract_image_features(self, image: np.ndarray) -> dict[str, Any]:
        """Extract rule-based features (edges, texture, morphology, stats)."""
        ...

    def recognize_pattern(self, image: np.ndarray) -> tuple[str, float]:
        """Recognize a shape/pattern via self-synthesized prototypes."""
        ...

    def analyze_shape(self, image: np.ndarray) -> dict[str, Any]:
        """Analyze geometric/topological shape properties."""
        ...

    # --- Analytics capabilities ---

    def forecast(self, series: np.ndarray, horizon: int = 5) -> np.ndarray:
        """Forecast future values of a 1D series (zero-data)."""
        ...

    def detect_anomalies(self, series: np.ndarray) -> np.ndarray:
        """Detect anomalies in a 1D series (returns bool mask)."""
        ...

    def mine_patterns(self, series: np.ndarray) -> dict[str, Any]:
        """Mine structural patterns (self-similarity, topology, CA rule, periodicity)."""
        ...

    def analyze_trend(self, series: np.ndarray) -> dict[str, Any]:
        """Analyze trend, regime, curvature, geodesic deviation, isomorphism."""
        ...

    # --- Advanced NLP capabilities ---

    def detect_script(self, text: str) -> str:
        """Detect the dominant script of ``text``.

        Returns one of 11 recognized script names (latin, cyrillic, greek,
        hebrew, arabic, devanagari, thai, hiragana, katakana, cjk, hangul),
        ``'mixed'`` (when no single script dominates the recognized chars),
        or ``'unknown'`` (when no recognized-script character is present).

        Round-10 audit R10-C-006: the previous docstring listed only 5
        scripts (latin/cyrillic/cjk/arabic/mixed); the implementation has
        supported 11 scripts since Fix 18, but the docstring was never
        updated.
        """
        ...

    def encode_multilingual(self, text: str) -> np.ndarray:
        """Encode ``text`` into a script-aware dim-length vector (L2-normalized)."""
        ...

    def analyze_syntax(self, text: str) -> dict[str, Any]:
        """Rule-based syntactic analysis (sentences, POS guesses, SVO hint)."""
        ...

    def encode_sentences(self, text: str) -> np.ndarray:
        """Encode each sentence of ``text`` into its own dim-length vector."""
        ...

    # --- Advanced Vision capabilities ---

    def encode_point_cloud(self, points: np.ndarray) -> np.ndarray:
        """Encode a 3D point cloud (Nx3) into a dim-length L2-normalized vector."""
        ...

    def analyze_video(self, frames: list[np.ndarray] | np.ndarray) -> dict[str, Any]:
        """Analyze a sequence of 2D frames (motion, keyframes, temporal encoding)."""
        ...

    def estimate_depth(self, image: np.ndarray) -> dict[str, Any]:
        """Estimate a monocular depth map from a single 2D image (rule-based)."""
        ...

    # --- Advanced Analytics capabilities ---

    def infer_cause(
        self, cause: np.ndarray, effect: np.ndarray, max_lag: int = 5
    ) -> dict[str, Any]:
        """Granger-style causal inference between two series (no statsmodels)."""
        ...

    def bayesian_update(self, obs: float | np.ndarray) -> None:
        """Online Bayesian posterior update from a single observation."""
        ...

    def bayesian_predictive(self) -> tuple[float, float]:
        """Return ``(mean, std)`` of the Bayesian posterior predictive distribution."""
        ...

    def detect_change_points(self, series: np.ndarray) -> np.ndarray:
        """Detect distributional change points in ``series`` (returns int indices)."""
        ...
