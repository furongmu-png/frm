# src/zero_data_model/model.py
"""ZeroDataModel — Full system integration."""

from __future__ import annotations

import threading

import numpy as np

from .active_inference import ActiveInferenceEngine
from .base import Signal
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
from .capabilities.nlp import SemanticComparator, TextEncoder, TextGenerator, ZeroShotClassifier
from .capabilities.nlp_advanced import (
    MultiLingualEncoder,
    SentenceEncoder,
    SyntacticAnalyzer,
)
from .capabilities.rules import AnalyticsRules, NLPRules, VisionRules
from .capabilities.vision import FeatureExtractor, ImageEncoder, PatternRecognizer, ShapeAnalyzer
from .capabilities.vision_advanced import (
    DepthEstimator,
    PointCloudEncoder,
    VideoFrameAnalyzer,
)
from .category_engine import CategoryTheoryEngine
from .consciousness_core import ConsciousnessCore
from .hardware import accel as _accel
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

    Domain Capabilities (compose the core modules + rule priors):
    - NLP: text encoding, semantic similarity, zero-shot classification, text generation
    - Vision: image encoding, feature extraction, pattern recognition, shape analysis
    - Analytics: time-series forecasting, anomaly detection, pattern mining, trend analysis
    """

    def __init__(self, dim: int = 64, seed: int | None = None):
        self.dim = dim
        # Optional RNG seed (Fix 9). The actual concurrency-safety guarantee
        # for the global ``np.random`` RNG used inside ``think`` comes from
        # ``self._lock`` (Fix 7): it serializes every state-mutating cycle so
        # the legacy global RNG is never accessed concurrently. The read-only
        # analytics methods (classify_text, detect_anomalies, ...) use the
        # now-pure ``compute_free_energy`` and do not touch the RNG, so they
        # remain safe to call concurrently with each other.
        self._seed = seed
        if seed is not None:
            np.random.seed(seed)
        self._rng = np.random.default_rng(seed)
        # Re-entrant lock around every state-mutating think/solve cycle (Fix 7).
        self._lock = threading.RLock()
        self.consciousness = ConsciousnessCore(dim=dim)
        self.active_inference = ActiveInferenceEngine(
            state_dim=dim, obs_dim=dim, action_dim=dim // 2
        )
        self.category_engine = CategoryTheoryEngine(dim=dim)
        # When a seed is set (Fix 9), force the deterministic pure-NumPy
        # ``SimulatorQuantumBackend`` instead of the Qiskit ``StatevectorSampler``.
        # The Qiskit sampler performs stochastic shot-based measurement that
        # does NOT draw from numpy's global RNG, so cycle re-seeding cannot
        # make it reproducible. The pure-NumPy simulator evolves a closed-form
        # state vector with no sampling, so two seeded models produce identical
        # ``think()`` sequences. When no seed is set, prefer the real Qiskit
        # backend (if installed) for production use.
        self.quantum_hybrid = QuantumClassicalHybrid(
            dim=dim, quantum_backend="simulator" if seed is not None else None
        )
        self.biological = BiologicalSubstrate(dim=dim)
        self.math_universe = MathematicalUniverse(dim=dim)
        self.modules = [
            self.consciousness,
            self.active_inference,
            self.category_engine,
            self.quantum_hybrid,
            self.biological,
            self.math_universe,
        ]
        # Domain capabilities built on top of the core modules.
        self.nlp_text_encoder = TextEncoder(dim=dim)
        self.nlp_comparator = SemanticComparator(
            self.nlp_text_encoder, self.math_universe, self.category_engine
        )
        self.nlp_classifier = ZeroShotClassifier(
            dim=dim,
            active_inference=self.active_inference,
            category_engine=self.category_engine,
        )
        self.nlp_generator = TextGenerator(
            dim=dim, biological=self.biological, math_universe=self.math_universe
        )
        self.vision_encoder = ImageEncoder(dim=dim, math_universe=self.math_universe)
        self.vision_features = FeatureExtractor(dim=dim, biological=self.biological)
        self.vision_recognizer = PatternRecognizer(
            dim=dim,
            active_inference=self.active_inference,
            consciousness=self.consciousness,
            encoder=self.vision_encoder,
        )
        self.vision_analyzer = ShapeAnalyzer(dim=dim, category_engine=self.category_engine)
        self.analytics_forecaster = TimeSeriesForecaster(
            dim=dim,
            active_inference=self.active_inference,
            quantum_hybrid=self.quantum_hybrid,
        )
        self.analytics_anomaly = AnomalyDetector(dim=dim, active_inference=self.active_inference)
        self.analytics_miner = PatternMiner(
            dim=dim, biological=self.biological, math_universe=self.math_universe
        )
        self.analytics_trend = TrendAnalyzer(
            dim=dim, math_universe=self.math_universe, category_engine=self.category_engine
        )
        # Rule libraries reused by the advanced capabilities below.
        self.nlp_rules = NLPRules()
        self.vision_rules = VisionRules()
        self.analytics_rules = AnalyticsRules()
        # Advanced domain capabilities (reuse the existing core module
        # instances: math_universe, biological, active_inference, rules).
        self.nlp_multilingual = MultiLingualEncoder(dim=dim, rules=self.nlp_rules)
        self.nlp_syntactic = SyntacticAnalyzer(dim=dim, rules=self.nlp_rules)
        self.nlp_sentence_encoder = SentenceEncoder(dim=dim, rules=self.nlp_rules)
        self.vision_point_cloud = PointCloudEncoder(
            dim=dim, math_universe=self.math_universe
        )
        self.vision_video = VideoFrameAnalyzer(
            dim=dim, biological=self.biological, math_universe=self.math_universe
        )
        self.vision_depth = DepthEstimator(dim=dim, math_universe=self.math_universe)
        self.analytics_causal = CausalInference(
            dim=dim, active_inference=self.active_inference, rules=self.analytics_rules
        )
        self.analytics_bayesian = BayesianEstimator(
            dim=dim, active_inference=self.active_inference
        )
        self.analytics_changepoint = ChangePointDetector(
            dim=dim,
            active_inference=self.active_inference,
            rules=self.analytics_rules,
        )
        # Hardware acceleration: parallel module execution + GPU-aware arrays.
        # When a seed is set, force sequential execution so the legacy global
        # numpy RNG (used inside module process/predict) is never accessed
        # concurrently across threads — guaranteeing reproducibility.
        self.parallel_executor = ParallelExecutor(
            n_workers=1 if seed is not None else None
        )
        self.cycle_count = 0

    @property
    def hardware_info(self) -> dict:
        """Report the active hardware backends for diagnostics."""
        return {
            "array_backend": _accel.backend_name(),
            "gpu": _accel.has_gpu,
            "quantum_backend": self.quantum_hybrid.quantum_backend_name,
            "annealer_jit": getattr(self.quantum_hybrid.annealer, "jit", False),
            **self.parallel_executor.info,
        }

    def think(self, input_data: np.ndarray | None = None) -> Signal:
        """
        Process a thought cycle. If no input, self-generates from internal state.

        Module ``process`` and ``predict`` steps run concurrently via the
        parallel executor when multiple cores are available.

        The whole cycle is serialized by ``self._lock`` (Fix 7) so concurrent
        ``think`` calls do not race on the legacy global numpy RNG or on
        shared module state.
        """
        with self._lock:
            # When a seed is set, re-seed the global numpy RNG from the
            # model's private Generator so each think() cycle is deterministic
            # AND progresses across cycles. This makes two models with the
            # same seed produce identical think() sequences.
            if self._seed is not None:
                np.random.seed(int(self._rng.integers(0, 2**31)))

            if input_data is None:
                signal = self._self_generate()
            else:
                padded = np.zeros(self.dim)
                padded[: len(input_data)] = input_data[: self.dim]
                signal = Signal(data=padded)

            # Parallel module processing.
            results = self.parallel_executor.map_modules(self.modules, signal)

            integrated = self._integrate(results)
            reflection = self.consciousness.reflect()

            # Parallel prediction.
            preds = self.parallel_executor.map(
                lambda m: m.predict(integrated), self.modules
            )
            # Per-module prediction error (Fix 16): the original code passed
            # every module the same mean error; pass each module its own
            # ``pred.uncertainty`` so update() is meaningfully per-module.
            for module, pred in zip(self.modules, preds, strict=False):
                module.update(pred.uncertainty)

            self.cycle_count += 1
            return Signal(
                data=integrated.data,
                metadata={
                    "cycle": self.cycle_count,
                    "self_reflection": reflection.metadata,
                    "module_count": len(self.modules),
                },
            )

    def _self_generate(self) -> Signal:
        """Self-generate input from internal knowledge."""
        bio_signal = self.biological.dna_storage.generate(Signal(data=np.zeros(self.dim)))
        fractal = self.math_universe.fractal.generate(np.zeros(self.dim), n_iterations=3)
        combined = 0.5 * bio_signal.data[: self.dim] + 0.5 * fractal
        if len(combined) < self.dim:
            combined = np.pad(combined, (0, self.dim - len(combined)))
        return Signal(data=combined[: self.dim], metadata={"self_generated": True})

    def _integrate(self, signals: list[Signal]) -> Signal:
        """Integrate signals from all modules."""
        max_len = max(len(s.data) for s in signals)
        padded = np.zeros((len(signals), max_len))
        for i, s in enumerate(signals):
            padded[i, : len(s.data)] = s.data
        mean_signal = np.mean(padded, axis=0)
        return Signal(data=mean_signal[: self.dim], metadata={"integrated": True})

    def solve(self, problem: np.ndarray) -> Signal:
        """Solve an optimization problem using quantum annealing.

        The ``problem`` vector is encoded into the annealer's cost matrix as
        ``outer(p, p) + 0.1 * I`` before optimizing (Fix 14), so the solution
        actually depends on the input rather than being independent of it.
        """
        with self._lock:
            n = self.quantum_hybrid.annealer.cost_matrix.shape[0]
            problem_flat = np.zeros(n)
            p = np.asarray(problem, dtype=float).flatten()
            problem_flat[: min(len(p), n)] = p[:n]
            self.quantum_hybrid.annealer.cost_matrix = (
                np.outer(problem_flat, problem_flat) + np.eye(n) * 0.1
            )
            # Re-symmetrize defensively (outer product is already symmetric).
            cm = self.quantum_hybrid.annealer.cost_matrix
            self.quantum_hybrid.annealer.cost_matrix = (cm + cm.T) / 2.0
            solution, energy = self.quantum_hybrid.solve_optimization()
            return Signal(data=solution, metadata={"energy": energy, "type": "optimization"})

    def find_analogies(self, problem_a: np.ndarray, problem_b: np.ndarray) -> float:
        """Find structural similarity between two problems."""
        # ``find_isomorphism`` is read-only (no module state mutation), so the
        # lock is not strictly required; we acquire it for consistency with
        # the rest of the public API and to avoid surprising re-entry.
        with self._lock:
            return self.category_engine.find_isomorphism(problem_a, problem_b)

    def generate_knowledge(self, query: str = "") -> Signal:
        """Self-generate knowledge without external data."""
        with self._lock:
            return self._self_generate()

    # --- NLP capabilities ---

    def encode_text(self, text: str) -> np.ndarray:
        """Encode text into a fixed-dim vector (zero-data)."""
        return self.nlp_text_encoder.encode(text)

    def text_similarity(self, a: str, b: str) -> float:
        """Compute semantic similarity between two texts in [0, 1]."""
        return self.nlp_comparator.similarity(a, b)

    def classify_text(self, text: str) -> tuple[str, float]:
        """Zero-shot classify text into a topic (tech/nature/emotion/science)."""
        return self.nlp_classifier.classify(text)

    def generate_text(self, seed: str, length: int = 32) -> str:
        """Generate text from a seed with no external data."""
        return self.nlp_generator.generate(seed, length=length)

    # --- Vision capabilities ---

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """Encode a 2D image into a fixed-dim vector (zero-data)."""
        return self.vision_encoder.encode(image)

    def extract_image_features(self, image: np.ndarray) -> dict:
        """Extract rule-based features (edges, texture, morphology, stats)."""
        return self.vision_features.extract(image)

    def recognize_pattern(self, image: np.ndarray) -> tuple[str, float]:
        """Recognize a shape/pattern via self-synthesized prototypes."""
        return self.vision_recognizer.recognize(image)

    def analyze_shape(self, image: np.ndarray) -> dict:
        """Analyze geometric/topological shape properties."""
        return self.vision_analyzer.analyze(image)

    # --- Analytics capabilities ---

    def forecast(self, series: np.ndarray, horizon: int = 5) -> np.ndarray:
        """Forecast future values of a 1D series (zero-data)."""
        return self.analytics_forecaster.forecast(series, horizon=horizon)

    def detect_anomalies(self, series: np.ndarray) -> np.ndarray:
        """Detect anomalies in a 1D series (returns bool mask)."""
        return self.analytics_anomaly.detect(series)

    def mine_patterns(self, series: np.ndarray) -> dict:
        """Mine structural patterns (self-similarity, topology, CA rule, periodicity)."""
        return self.analytics_miner.mine(series)

    def analyze_trend(self, series: np.ndarray) -> dict:
        """Analyze trend, regime, curvature, geodesic deviation, isomorphism."""
        return self.analytics_trend.analyze(series)

    # --- Advanced NLP capabilities ---

    def detect_script(self, text: str) -> str:
        """Detect the dominant script of ``text`` (latin/cyrillic/cjk/arabic/mixed)."""
        return self.nlp_multilingual.detect_script(text)

    def encode_multilingual(self, text: str) -> np.ndarray:
        """Encode ``text`` into a script-aware dim-length vector (L2-normalized)."""
        return self.nlp_multilingual.encode(text)

    def analyze_syntax(self, text: str) -> dict:
        """Rule-based syntactic analysis (sentences, POS guesses, SVO hint)."""
        return self.nlp_syntactic.analyze(text)

    def encode_sentences(self, text: str) -> np.ndarray:
        """Encode each sentence of ``text`` into its own dim-length vector."""
        return self.nlp_sentence_encoder.encode(text)

    # --- Advanced Vision capabilities ---

    def encode_point_cloud(self, points: np.ndarray) -> np.ndarray:
        """Encode a 3D point cloud (Nx3) into a dim-length L2-normalized vector."""
        return self.vision_point_cloud.encode(points)

    def analyze_video(self, frames) -> dict:
        """Analyze a sequence of 2D frames (motion, keyframes, temporal encoding)."""
        return self.vision_video.analyze(frames)

    def estimate_depth(self, image: np.ndarray) -> dict:
        """Estimate a monocular depth map from a single 2D image (rule-based)."""
        return self.vision_depth.estimate(image)

    # --- Advanced Analytics capabilities ---

    def infer_cause(self, cause: np.ndarray, effect: np.ndarray, max_lag: int = 5) -> dict:
        """Granger-style causal inference between two series (no statsmodels)."""
        return self.analytics_causal.infer_cause(cause, effect, max_lag=max_lag)

    def bayesian_update(self, obs) -> None:
        """Online Bayesian posterior update from a single observation."""
        self.analytics_bayesian.update(obs)

    def bayesian_predictive(self) -> tuple[float, float]:
        """Return ``(mean, std)`` of the Bayesian posterior predictive distribution."""
        return self.analytics_bayesian.predictive()

    def detect_change_points(self, series: np.ndarray) -> np.ndarray:
        """Detect distributional change points in ``series`` (returns int indices)."""
        return self.analytics_changepoint.detect(series)
