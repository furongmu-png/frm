# src/zero_data_model/model.py
"""ZeroDataModel — Full system integration."""

from __future__ import annotations

import os
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
        # Round-3 audit: validate dim to prevent OOM / confusing downstream
        # errors. ``dim`` drives every cognitive module to allocate ``dim x
        # dim`` arrays; unbounded values cause silent OOM, and ``dim <= 0``
        # surfaces as a confusing numpy shape error deep in a module.
        if not isinstance(dim, int) or dim < 1 or dim > 4096:
            raise ValueError(
                f"dim must be an int in [1, 4096], got {dim!r}"
            )
        self.dim = dim
        # Optional RNG seed (Fix 9). Round-3 audit CRIT-1: each cognitive
        # module now holds its own ``np.random.Generator`` (``self._rng``)
        # seeded from this seed, instead of drawing from the global
        # ``np.random`` RNG. The per-module generators advance their own state
        # on every draw, so seeded models are reproducible without per-cycle
        # re-seeding. ``self._lock`` (Fix 7) still serializes every
        # state-mutating cycle so the shared generators are never accessed
        # concurrently. The read-only analytics methods (classify_text,
        # detect_anomalies, ...) use the now-pure ``compute_free_energy`` and
        # do not touch the RNG, so they remain safe to call concurrently.
        # ``np.random.seed(seed)`` below is retained only for backward
        # compatibility (some capability modules may still use the global RNG).
        #
        # When a seed is set, also pin the BLAS thread count to 1 (B-CRIT-01):
        # multi-threaded BLAS (OpenBLAS/MKL) parallelises matmuls across cores
        # and the thread-local work partition is not deterministic across
        # runs, which breaks reproducibility even when the numpy RNG is
        # seeded. ``setdefault`` only takes effect when the variable is not
        # already set in the environment, and only before numpy initialises
        # its BLAS context -- so this is best-effort for processes that
        # construct a seeded model before any heavy numpy work.
        if seed is not None:
            for _var in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ):
                os.environ.setdefault(_var, "1")
        self._seed = seed
        if seed is not None:
            np.random.seed(seed)
        self._rng = np.random.default_rng(seed)
        # Re-entrant lock around every state-mutating think/solve cycle (Fix 7).
        self._lock = threading.RLock()
        # Round-4 audit RNG-1: spawn an INDEPENDENT child Generator for each
        # cognitive module. ``np.random.Generator`` is NOT thread-safe (NumPy
        # docs: "using the same Generator from multiple threads is problematic"),
        # so the Round-3 approach of sharing one ``self._rng`` across all
        # modules was still a race when ``parallel_executor`` dispatched
        # ``module.process()`` to ThreadPoolExecutor workers in unseeded mode.
        # ``spawn(n)`` produces n bit-stream-independent child Generators that
        # can be used concurrently from different threads. Seeded mode keeps
        # reproducibility (spawn is deterministic given the parent seed); the
        # children are consumed in a fixed order so a seeded model's
        # ``think()`` sequence remains identical across runs.
        # Round-5 audit RNG5-3: the count MUST match the number of modules
        # indexed below (_child_rngs[0]..[5]). If a 7th module is added,
        # bump _N_COGNITIVE_MODULES or the indexing will raise IndexError.
        _N_COGNITIVE_MODULES = 6
        _child_rngs = self._rng.spawn(_N_COGNITIVE_MODULES)
        self.consciousness = ConsciousnessCore(dim=dim, rng=_child_rngs[0])
        self.active_inference = ActiveInferenceEngine(
            state_dim=dim, obs_dim=dim, action_dim=dim // 2, rng=_child_rngs[1]
        )
        self.category_engine = CategoryTheoryEngine(dim=dim, rng=_child_rngs[2])
        # When a seed is set (Fix 9), force the deterministic pure-NumPy
        # ``SimulatorQuantumBackend`` instead of the Qiskit ``StatevectorSampler``.
        # The Qiskit sampler performs stochastic shot-based measurement that
        # does NOT draw from numpy's global RNG, so cycle re-seeding cannot
        # make it reproducible. The pure-NumPy simulator evolves a closed-form
        # state vector with no sampling, so two seeded models produce identical
        # ``think()`` sequences. When no seed is set, prefer the real Qiskit
        # backend (if installed) for production use.
        self.quantum_hybrid = QuantumClassicalHybrid(
            dim=dim,
            quantum_backend="simulator" if seed is not None else None,
            rng=_child_rngs[3],
        )
        self.biological = BiologicalSubstrate(dim=dim, rng=_child_rngs[4])
        self.math_universe = MathematicalUniverse(dim=dim, rng=_child_rngs[5])
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
        # When a seed is set, force sequential execution so the per-module
        # child Generators (spawned in ``__init__``) are consumed in a fixed
        # order — guaranteeing reproducibility. In unseeded mode the children
        # are independent bit-streams (see RNG-1), so multi-worker parallelism
        # is safe.
        self.parallel_executor = ParallelExecutor(
            n_workers=1 if seed is not None else None
        )
        self.cycle_count = 0

    # Round-8 audit R8-HIGH-3: pickle support. ``ZeroDataModel`` holds two
    # objects that are NOT picklable:
    #
    #   * ``self._lock`` (``threading.RLock``) — locks are tied to a
    #     process and cannot be sent across a pipe / socket.
    #   * ``self.parallel_executor._pool`` (``ThreadPoolExecutor``) — its
    #     worker threads hold ``_thread.lock`` instances.
    #
    # Without these hooks, ``pickle.dumps(model)`` raises
    # ``TypeError: cannot pickle '_thread.lock' object``. Some downstream
    # paths (e.g. ``multiprocessing`` pools, ``copy.deepcopy`` in tests,
    # frameworks that ship the model to worker processes) call pickle, so
    # the model silently fails to integrate. ``__getstate__`` drops the
    # two unpicklable attrs; ``__setstate__`` rebuilds them from the
    # recorded seed (``self._seed`` is picklable) so the unpickled model
    # behaves identically to a freshly-constructed one.
    #
    # Note: the model's authoritative persistence path remains
    # ``ModelSerializer.save/load`` (npz + json), which has its own
    # validation. These hooks only enable pickle / deepcopy.
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # Drop the unpicklable concurrency primitives. ``_seed`` survives
        # in the state so ``__setstate__`` can rebuild ``parallel_executor``
        # with the same n_workers policy (1 for seeded, auto for unseeded).
        state["_lock"] = None
        state["parallel_executor"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        # Rebuild the concurrency primitives that ``__getstate__`` dropped.
        self._lock = threading.RLock()
        seed = self._seed
        self.parallel_executor = ParallelExecutor(
            n_workers=1 if seed is not None else None
        )

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
        ``think`` calls do not race on the shared per-module Generators
        (CRIT-1) or on shared module state.

        C-7: Per-module prediction uncertainties are computed from the input
        signal (not the integrated signal) and used to weight the integration
        via ``softmax(1/uncertainty)``. Modules that are more confident about
        the input contribute more to the integrated output, and the same
        predictions drive the ``update()`` step — avoiding a redundant second
        predict pass and aligning with predictive-processing theory (bottom-up
        prediction errors drive both integration and learning).
        """
        with self._lock:
            # Round-3 audit CRIT-1: per-module Generator — each module holds its
            # own ``self._rng`` (seeded once in ``__init__``), so the per-cycle
            # global ``np.random.seed`` re-seed is no longer needed. The
            # generators advance their own state on every draw, so two seeded
            # models still produce identical ``think()`` sequences.

            if input_data is None:
                signal = self._self_generate()
            else:
                padded = np.zeros(self.dim)
                padded[: len(input_data)] = input_data[: self.dim]
                signal = Signal(data=padded)

            # Parallel module processing.
            results = self.parallel_executor.map_modules(self.modules, signal)

            # C-7: Predict from the INPUT signal to obtain per-module
            # uncertainties, which then weight the integration. Modules that
            # are more confident (lower uncertainty) about the input contribute
            # more to the integrated output. The same predictions drive the
            # ``update()`` step below, avoiding a redundant second predict
            # pass. ``ConsciousnessCore.predict`` reuses its process cache,
            # so this call is cheap for that module.
            preds = self.parallel_executor.map(
                lambda m: m.predict(signal), self.modules
            )
            # Round-3 audit: ``max(nan, 1e-8)`` returns ``nan`` (because
            # ``nan > 1e-8`` is False, so the first argument wins). Explicitly
            # reject non-finite uncertainties so a single NaN-poisoned module
            # cannot corrupt the entire softmax weighting.
            uncertainties = np.array(
                [
                    max(float(p.uncertainty), 1e-8)
                    if np.isfinite(float(p.uncertainty))
                    else 1e8
                    for p in preds
                ]
            )

            # C-7: Weighted integration by inverse uncertainty.
            integrated = self._integrate(results, uncertainties)
            reflection = self.consciousness.reflect()

            # Per-module prediction error (Fix 16 + Round-7 THEORY7-4): the
            # original code passed every module the same mean error; Fix 16
            # changed it to ``pred.uncertainty`` — but ``uncertainty`` is the
            # VARIANCE of the predicted output, not a prediction error.
            # Modules' ``update()`` methods interpret the argument as a
            # prediction-error magnitude and scale learning rates by it
            # (e.g. ``lr = 0.001 * prediction_error``). Passing variance
            # instead of error means a module with high-variance but accurate
            # predictions gets a large learning rate, while a module with
            # low-variance but wrong predictions barely updates — the opposite
            # of what's intended.
            # Round-7 fix: compute the ACTUAL per-module prediction error as
            # the mean-squared distance between the predicted observation and
            # the input signal. Fall back to ``pred.uncertainty`` if the
            # prediction is non-finite (defensive — keeps update() callable).
            for module, pred in zip(self.modules, preds, strict=False):
                pred_val = np.asarray(pred.value, dtype=float).flatten()
                sig_slice = signal.data[: len(pred_val)]
                if len(sig_slice) < len(pred_val):
                    sig_slice = np.pad(sig_slice, (0, len(pred_val) - len(sig_slice)))
                diff = pred_val - sig_slice
                error = float(np.dot(diff, diff)) / max(len(pred_val), 1)
                if not np.isfinite(error):
                    error = float(pred.uncertainty)
                module.update(error)

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

    def _integrate(
        self,
        signals: list[Signal],
        uncertainties: np.ndarray | None = None,
    ) -> Signal:
        """Integrate signals from all modules.

        C-7: When ``uncertainties`` is provided (one per module), weight each
        module's output by ``softmax(1 / uncertainty)`` so more confident
        modules (lower prediction uncertainty) contribute more to the
        integrated signal. When ``uncertainties`` is None (backward-compatible
        path used by callers that do not compute predictions), fall back to
        the original equal-weight mean.

        The softmax normalises weights to sum to 1 and is numerically stable
        when some uncertainties are very small (via max-subtraction). When all
        uncertainties are equal, softmax(1/c) produces uniform weights — the
        same result as the equal-weight mean — so the weighted path is a
        strict generalisation of the original behaviour.
        """
        # Round-3 audit: guard against an empty ``signals`` list —
        # ``max()`` of an empty sequence raises ``ValueError``.
        if not signals:
            return Signal(
                data=np.zeros(self.dim),
                metadata={"integrated": True, "empty": True},
            )
        max_len = max(len(s.data) for s in signals)
        padded = np.zeros((len(signals), max_len))
        for i, s in enumerate(signals):
            padded[i, : len(s.data)] = s.data

        if uncertainties is None or len(uncertainties) != len(signals):
            # Equal-weight mean (backward-compatible path).
            mean_signal = np.mean(padded, axis=0)
        else:
            # C-7: Inverse-uncertainty weighting via softmax(1/uncertainty).
            # Confident modules (low uncertainty) weigh more; the softmax
            # normalises weights to sum to 1 and is numerically stable when
            # some uncertainties are very small (via max-subtraction).
            inv_unc = 1.0 / np.asarray(uncertainties, dtype=float)
            inv_unc = inv_unc - np.max(inv_unc)  # numerical stability
            weights = np.exp(inv_unc)
            weights = weights / (np.sum(weights) + 1e-12)
            # Weighted sum: each row weighted by its module's confidence.
            mean_signal = (weights[:, None] * padded).sum(axis=0)

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
        # Round-7 audit NEW5-10 (Round-2 + Round-7): use the renamed
        # ``structural_similarity`` instead of the deprecated
        # ``find_isomorphism`` alias. The method is read-only (no module
        # state mutation), so the lock is not strictly required; we acquire
        # it for consistency with the rest of the public API and to avoid
        # surprising re-entry.
        with self._lock:
            return self.category_engine.structural_similarity(problem_a, problem_b)

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
        with self._lock:
            return self.nlp_comparator.similarity(a, b)

    def classify_text(self, text: str) -> tuple[str, float]:
        """Zero-shot classify text into a topic (tech/nature/emotion/science)."""
        with self._lock:
            return self.nlp_classifier.classify(text)

    def generate_text(self, seed: str, length: int = 32) -> str:
        """Generate text from a seed with no external data."""
        with self._lock:
            return self.nlp_generator.generate(seed, length=length)

    # --- Vision capabilities ---

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """Encode a 2D image into a fixed-dim vector (zero-data)."""
        return self.vision_encoder.encode(image)

    def extract_image_features(self, image: np.ndarray) -> dict:
        """Extract rule-based features (edges, texture, morphology, stats)."""
        with self._lock:
            return self.vision_features.extract(image)

    def recognize_pattern(self, image: np.ndarray) -> tuple[str, float]:
        """Recognize a shape/pattern via self-synthesized prototypes."""
        with self._lock:
            return self.vision_recognizer.recognize(image)

    def analyze_shape(self, image: np.ndarray) -> dict:
        """Analyze geometric/topological shape properties."""
        with self._lock:
            return self.vision_analyzer.analyze(image)

    # --- Analytics capabilities ---

    def forecast(self, series: np.ndarray, horizon: int = 5) -> np.ndarray:
        """Forecast future values of a 1D series (zero-data)."""
        with self._lock:
            return self.analytics_forecaster.forecast(series, horizon=horizon)

    def detect_anomalies(self, series: np.ndarray) -> np.ndarray:
        """Detect anomalies in a 1D series (returns bool mask)."""
        with self._lock:
            return self.analytics_anomaly.detect(series)

    def mine_patterns(self, series: np.ndarray) -> dict:
        """Mine structural patterns (self-similarity, topology, CA rule, periodicity)."""
        with self._lock:
            return self.analytics_miner.mine(series)

    def analyze_trend(self, series: np.ndarray) -> dict:
        """Analyze trend, regime, curvature, geodesic deviation, isomorphism."""
        with self._lock:
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
        with self._lock:
            return self.nlp_syntactic.analyze(text)

    def encode_sentences(self, text: str) -> np.ndarray:
        """Encode each sentence of ``text`` into its own dim-length vector."""
        with self._lock:
            return self.nlp_sentence_encoder.encode(text)

    # --- Advanced Vision capabilities ---

    def encode_point_cloud(self, points: np.ndarray) -> np.ndarray:
        """Encode a 3D point cloud (Nx3) into a dim-length L2-normalized vector."""
        return self.vision_point_cloud.encode(points)

    def analyze_video(self, frames: list[np.ndarray] | np.ndarray) -> dict:
        """Analyze a sequence of 2D frames (motion, keyframes, temporal encoding)."""
        with self._lock:
            return self.vision_video.analyze(frames)

    def estimate_depth(self, image: np.ndarray) -> dict:
        """Estimate a monocular depth map from a single 2D image (rule-based)."""
        with self._lock:
            return self.vision_depth.estimate(image)

    # --- Advanced Analytics capabilities ---

    def infer_cause(self, cause: np.ndarray, effect: np.ndarray, max_lag: int = 5) -> dict:
        """Granger-style causal inference between two series (no statsmodels)."""
        with self._lock:
            return self.analytics_causal.infer_cause(cause, effect, max_lag=max_lag)

    def bayesian_update(self, obs: float | np.ndarray) -> None:
        """Online Bayesian posterior update from a single observation."""
        with self._lock:
            self.analytics_bayesian.update(obs)

    def bayesian_predictive(self) -> tuple[float, float]:
        """Return ``(mean, std)`` of the Bayesian posterior predictive distribution."""
        with self._lock:
            return self.analytics_bayesian.predictive()

    def detect_change_points(self, series: np.ndarray) -> np.ndarray:
        """Detect distributional change points in ``series`` (returns int indices)."""
        with self._lock:
            return self.analytics_changepoint.detect(series)
