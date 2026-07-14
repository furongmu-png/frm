# src/zero_data_model/model.py
"""ZeroDataModel — Full system integration."""

from __future__ import annotations
import numpy as np
from .base import Signal
from .consciousness_core import ConsciousnessCore
from .active_inference import ActiveInferenceEngine
from .category_engine import CategoryTheoryEngine
from .quantum_hybrid import QuantumClassicalHybrid
from .biological import BiologicalSubstrate
from .math_universe import MathematicalUniverse
from .capabilities.nlp import TextEncoder, SemanticComparator, ZeroShotClassifier, TextGenerator
from .capabilities.vision import ImageEncoder, FeatureExtractor, PatternRecognizer, ShapeAnalyzer
from .capabilities.analytics import TimeSeriesForecaster, AnomalyDetector, PatternMiner, TrendAnalyzer
from .hardware.parallel import ParallelExecutor
from .hardware import accel as _accel


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

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.consciousness = ConsciousnessCore(dim=dim)
        self.active_inference = ActiveInferenceEngine(state_dim=dim, obs_dim=dim, action_dim=dim // 2)
        self.category_engine = CategoryTheoryEngine(dim=dim)
        self.quantum_hybrid = QuantumClassicalHybrid(dim=dim)
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
        self.nlp_comparator = SemanticComparator(self.nlp_text_encoder, self.math_universe, self.category_engine)
        self.nlp_classifier = ZeroShotClassifier(dim=dim, active_inference=self.active_inference, category_engine=self.category_engine)
        self.nlp_generator = TextGenerator(dim=dim, biological=self.biological, math_universe=self.math_universe)
        self.vision_encoder = ImageEncoder(dim=dim, math_universe=self.math_universe)
        self.vision_features = FeatureExtractor(dim=dim, biological=self.biological)
        self.vision_recognizer = PatternRecognizer(dim=dim, active_inference=self.active_inference, consciousness=self.consciousness, encoder=self.vision_encoder)
        self.vision_analyzer = ShapeAnalyzer(dim=dim, category_engine=self.category_engine)
        self.analytics_forecaster = TimeSeriesForecaster(dim=dim, active_inference=self.active_inference, quantum_hybrid=self.quantum_hybrid)
        self.analytics_anomaly = AnomalyDetector(dim=dim, active_inference=self.active_inference)
        self.analytics_miner = PatternMiner(dim=dim, biological=self.biological, math_universe=self.math_universe)
        self.analytics_trend = TrendAnalyzer(dim=dim, math_universe=self.math_universe, category_engine=self.category_engine)
        # Hardware acceleration: parallel module execution + GPU-aware arrays.
        self.parallel_executor = ParallelExecutor()
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
        """
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
        pred_errors = [p.uncertainty for p in preds]
        mean_err = float(np.mean(pred_errors)) if pred_errors else 0.0
        for module in self.modules:
            module.update(mean_err)

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
        """Solve an optimization problem using quantum annealing."""
        solution, energy = self.quantum_hybrid.solve_optimization()
        return Signal(data=solution, metadata={"energy": energy, "type": "optimization"})

    def find_analogies(self, problem_a: np.ndarray, problem_b: np.ndarray) -> float:
        """Find structural similarity between two problems."""
        return self.category_engine.find_isomorphism(problem_a, problem_b)

    def generate_knowledge(self, query: str = "") -> Signal:
        """Self-generate knowledge without external data."""
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
