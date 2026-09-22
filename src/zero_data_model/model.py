# src/zero_data_model/model.py
"""ZeroDataModel — Full system integration."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

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
from .capabilities.audio import (
    AudioClassifier,
    AudioEncoder,
    OnsetDetector,
    PitchDetector,
)
from .capabilities.audio_advanced import (
    MusicAnalyzer,
    SpeakerRecognizer,
    SpeechSegmenter,
)
from .capabilities.causal import (
    CounterfactualReasoner,
    DecisionTreeBuilder,
    GameTheoryAnalyzer,
    MultiArmedBandit,
)
from .capabilities.causal_advanced import (
    CausalGraphBuilder,
    InterventionAnalyzer,
    POMDPApproximator,
)
from .capabilities.code import (
    ASTAnalyzer,
    CodeEncoder,
    CodeSimilarityChecker,
    DefectPatternDetector,
)
from .capabilities.code_advanced import (
    CodeStyleAnalyzer,
    ControlFlowAnalyzer,
    DependencyGraphBuilder,
)
from .capabilities.graph import (
    CentralityAnalyzer,
    CommunityDetector,
    GraphEncoder,
    PathFinder,
)
from .capabilities.graph_advanced import (
    DynamicGraphTracker,
    GraphIsomorphismDetector,
    SpanningTreeExtractor,
)
from .capabilities.memory import (
    ContextMemory,
    EpisodicMemory,
    MemoryConsolidator,
    WorkingMemory,
)
from .capabilities.memory_advanced import (
    ForgetfulMemory,
    HierarchicalMemory,
    MemoryIndexer,
    SpreadingActivationMemory,
)
from .capabilities.multimodal import (
    CrossModalAligner,
    ModalityEncoder,
    ModalityFuser,
    SharedLatentSpace,
)
from .capabilities.multimodal_advanced import (
    AttentionBasedFuser,
    ContrastiveAligner,
    MultimodalRetriever,
)
from .capabilities.nlp import SemanticComparator, TextEncoder, TextGenerator, ZeroShotClassifier
from .capabilities.nlp_advanced import (
    MultiLingualEncoder,
    SentenceEncoder,
    SyntacticAnalyzer,
)
from .capabilities.planning import (
    ActionSequencer,
    GoalDecomposer,
    HierarchicalPlanner,
    TrajectoryPlanner,
)
from .capabilities.planning_advanced import (
    ContingencyPlanner,
    MonteCarloTreePlanner,
    PolicyGradientPlanner,
    SymbolicPlanner,
)
from .capabilities.reasoning import (
    AnalogicalReasoner,
    DeductiveReasoner,
    InductiveReasoner,
    PropositionalLogicEngine,
)
from .capabilities.reasoning_advanced import (
    AbductiveReasoner,
    CausalChainReasoner,
    DefeasibleReasoner,
)
from .capabilities.rl import (
    PolicyOptimizer,
    QLearner,
    SyntheticMDP,
    ValueFunction,
)
from .capabilities.rl_advanced import (
    DynaQ,
    MonteCarloTreeSearch,
    PosteriorSampling,
)
from .capabilities.robotics import (
    GaitGenerator,
    KinematicsSolver,
    MotionPlanner,
    SensorFuser,
)
from .capabilities.robotics_advanced import (
    CollisionChecker,
    MPCController,
    TrajectoryOptimizer,
)
from .capabilities.rules import (
    AnalyticsRules,
    AudioRules,
    CausalRules,
    CodeRules,
    GraphRules,
    MemoryRules,
    MultimodalRules,
    NLPRules,
    PlanningRules,
    ReasoningRules,
    RLRules,
    RoboticsRules,
    TimeRules,
    VisionRules,
)
from .capabilities.time import (
    EventTimestampAnalyzer,
    FrequencyAnalyzer,
    SeasonalityDetector,
    TimeSeriesEncoder,
)
from .capabilities.time_advanced import (
    AnomalyTimingDetector,
    CyclePhaseTracker,
    ForecastabilityScorer,
)
from .capabilities.vision import FeatureExtractor, ImageEncoder, PatternRecognizer, ShapeAnalyzer
from .capabilities.vision_advanced import (
    DepthEstimator,
    PointCloudEncoder,
    VideoFrameAnalyzer,
)
from .category_engine import CategoryTheoryEngine
from .causal_emergence import CausalEmergenceEngine, EmergenceRules
from .consciousness_core import ConsciousnessCore
from .hardware import accel as _accel
from .hardware.parallel import ParallelExecutor
from .math_universe import MathematicalUniverse
from .metrics import ZDM_METRICS
from .quantum_hybrid import QuantumClassicalHybrid

# Module-level logger used by the cognitive-upgrade hooks in ``think()``.
# Each hook is wrapped in try/except and logs a warning on failure so a
# single optional optimization can never crash a full think() cycle.
_logger = logging.getLogger(__name__)


def _sanitize_for_json(obj: Any) -> Any:
    """Convert numpy types/arrays in ``obj`` to plain Python for JSON serialization.

    The cognitive-upgrade hooks (architect, layered_predictor, meta_cognition,
    ...) frequently return numpy scalars / ndarrays inside their result dicts.
    Those values are surfaced via ``Signal.metadata`` and may be serialized to
    JSON by API callers (FastAPI's ``jsonable_encoder`` raises on ``np.float64``
    / ``np.ndarray``). This helper walks the structure recursively and
    replaces every numpy leaf with its native Python equivalent, mapping
    NaN/inf to ``None`` (JSON ``null``) so the output is always JSON-safe.
    """
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if np.isfinite(v) else None  # NaN/inf -> null
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


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

    def __init__(
        self,
        dim: int = 64,
        seed: int | None = None,
        *,
        enable_architect: bool = False,
        enable_layered_predictor: bool = False,
        enable_episodic_memory: bool = False,
        enable_logic_layer: bool = False,
        enable_meta_cognition: bool = False,
        enable_experiment_planner: bool = False,
        enable_multiagent: bool = False,
        # --- Phase G (四.3): 三项核心认知升级开关 ----------------------- #
        # 默认 False 保证零回归；显式传 True 或设环境变量 ZDM_USE_* 启用。
        # 详见 zero_data_model.config。
        use_s4: bool | None = None,
        use_pcn: bool | None = None,
        use_hopfield: bool | None = None,
    ):
        # Round-3 audit: validate dim to prevent OOM / confusing downstream
        # errors. ``dim`` drives every cognitive module to allocate ``dim x
        # dim`` arrays; unbounded values cause silent OOM, and ``dim <= 0``
        # surfaces as a confusing numpy shape error deep in a module.
        if not isinstance(dim, int) or dim < 1 or dim > 4096:
            raise ValueError(
                f"dim must be an int in [1, 4096], got {dim!r}"
            )
        self.dim = dim

        # Phase G (四.3): 解析三项认知升级开关。
        # 显式参数优先；None 时回退到 config 模块的环境变量默认值。
        from zero_data_model.config import (
            get_default_use_s4,
            get_default_use_pcn,
            get_default_use_hopfield,
        )
        self._use_s4 = bool(use_s4) if use_s4 is not None else get_default_use_s4()
        self._use_pcn = bool(use_pcn) if use_pcn is not None else get_default_use_pcn()
        self._use_hopfield = (
            bool(use_hopfield) if use_hopfield is not None
            else get_default_use_hopfield()
        )
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
        # spec §2.3 (causal emergence): the emergence engine is NOT a
        # cognitive module — it stays out of ``self.modules`` and does NOT
        # bump ``_N_COGNITIVE_MODULES``. It uses ``_child_rngs[6]``, so we
        # spawn one extra child here.
        _N_COGNITIVE_MODULES = 6
        _child_rngs = self._rng.spawn(_N_COGNITIVE_MODULES + 1)
        self.consciousness = ConsciousnessCore(dim=dim, rng=_child_rngs[0])
        self.active_inference = ActiveInferenceEngine(
            state_dim=dim, obs_dim=dim, action_dim=dim // 2, rng=_child_rngs[1],
            # Phase G (四.1): 透传 S4 开关到生成模型
            use_s4=self._use_s4, s4_seed=self._seed,
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
        # Causal emergence engine (spec §2.3): composes 5 modules (topology,
        # causal discovery, differential, HMC, chaotic memory) into a single
        # recursive perception-causal-counterfactual-posterior-memory loop.
        # NOT in ``self.modules`` (preserves the 6-module cognitive count
        # contract) and does NOT change ``_N_COGNITIVE_MODULES``. Uses its
        # own child rng at index 6.
        self.emergence_rules = EmergenceRules()
        self.emergence = CausalEmergenceEngine(
            dim=dim,
            active_inference=self.active_inference,
            math_universe=self.math_universe,
            rules=self.emergence_rules,
            rng=_child_rngs[6],
        )
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
        # Audio domain capabilities (reuse existing core module instances).
        self.audio_rules = AudioRules()
        self.audio_encoder = AudioEncoder(
            dim=dim, math_universe=self.math_universe, rules=self.audio_rules
        )
        self.audio_onset = OnsetDetector(dim=dim, rules=self.audio_rules)
        self.audio_pitch = PitchDetector(dim=dim, rules=self.audio_rules)
        self.audio_classifier = AudioClassifier(dim=dim, rules=self.audio_rules)
        self.audio_segmenter = SpeechSegmenter(
            dim=dim, biological=self.biological, rules=self.audio_rules
        )
        self.audio_music = MusicAnalyzer(
            dim=dim, biological=self.biological, rules=self.audio_rules
        )
        self.audio_speaker = SpeakerRecognizer(
            dim=dim, math_universe=self.math_universe, rules=self.audio_rules
        )
        # Graph domain capabilities (reuse existing core module instances).
        self.graph_rules = GraphRules()
        self.graph_encoder = GraphEncoder(
            dim=dim, math_universe=self.math_universe, rules=self.graph_rules
        )
        self.graph_community = CommunityDetector(dim=dim, rules=self.graph_rules)
        self.graph_path = PathFinder(dim=dim, rules=self.graph_rules)
        self.graph_centrality = CentralityAnalyzer(dim=dim, rules=self.graph_rules)
        self.graph_isomorphism = GraphIsomorphismDetector(dim=dim, rules=self.graph_rules)
        self.graph_dynamic = DynamicGraphTracker(dim=dim, rules=self.graph_rules)
        self.graph_spanning = SpanningTreeExtractor(dim=dim, rules=self.graph_rules)
        # Robotics domain capabilities (reuse existing core module instances).
        self.robotics_rules = RoboticsRules()
        self.robotics_motion = MotionPlanner(
            dim=dim, math_universe=self.math_universe, rules=self.robotics_rules
        )
        self.robotics_kinematics = KinematicsSolver(dim=dim, rules=self.robotics_rules)
        self.robotics_sensor = SensorFuser(dim=dim, rules=self.robotics_rules)
        self.robotics_gait = GaitGenerator(
            dim=dim, biological=self.biological, rules=self.robotics_rules
        )
        self.robotics_trajectory = TrajectoryOptimizer(
            dim=dim, math_universe=self.math_universe, rules=self.robotics_rules
        )
        self.robotics_collision = CollisionChecker(dim=dim, rules=self.robotics_rules)
        self.robotics_mpc = MPCController(
            dim=dim,
            active_inference=self.active_inference,
            rules=self.robotics_rules,
        )
        # Time-series domain capabilities (reuse existing core module instances).
        self.time_rules = TimeRules()
        self.time_encoder = TimeSeriesEncoder(
            dim=dim, math_universe=self.math_universe, rules=self.time_rules
        )
        self.time_seasonality = SeasonalityDetector(dim=dim, rules=self.time_rules)
        self.time_frequency = FrequencyAnalyzer(
            dim=dim, math_universe=self.math_universe, rules=self.time_rules
        )
        self.time_events = EventTimestampAnalyzer(dim=dim, rules=self.time_rules)
        self.time_anomaly_timing = AnomalyTimingDetector(
            dim=dim, active_inference=self.active_inference, rules=self.time_rules
        )
        self.time_cycle_phase = CyclePhaseTracker(
            dim=dim, biological=self.biological, rules=self.time_rules
        )
        self.time_forecastability = ForecastabilityScorer(
            dim=dim, math_universe=self.math_universe, rules=self.time_rules
        )
        # Code-analysis domain capabilities (reuse existing core module instances).
        self.code_rules = CodeRules()
        self.code_encoder = CodeEncoder(
            dim=dim, math_universe=self.math_universe, rules=self.code_rules
        )
        self.code_ast = ASTAnalyzer(dim=dim, rules=self.code_rules)
        self.code_similarity = CodeSimilarityChecker(dim=dim, rules=self.code_rules)
        self.code_defects = DefectPatternDetector(dim=dim, rules=self.code_rules)
        self.code_control_flow = ControlFlowAnalyzer(dim=dim, rules=self.code_rules)
        self.code_style = CodeStyleAnalyzer(dim=dim, rules=self.code_rules)
        self.code_dependency = DependencyGraphBuilder(dim=dim, rules=self.code_rules)
        # Reasoning domain capabilities (reuse existing core module instances).
        self.reasoning_rules = ReasoningRules()
        self.reasoning_propositional = PropositionalLogicEngine(
            dim=dim, rules=self.reasoning_rules
        )
        self.reasoning_deductive = DeductiveReasoner(
            dim=dim, rules=self.reasoning_rules
        )
        self.reasoning_inductive = InductiveReasoner(
            dim=dim, rules=self.reasoning_rules
        )
        self.reasoning_analogical = AnalogicalReasoner(
            dim=dim,
            category_engine=self.category_engine,
            math_universe=self.math_universe,
            rules=self.reasoning_rules,
        )
        self.reasoning_abductive = AbductiveReasoner(
            dim=dim,
            active_inference=self.active_inference,
            rules=self.reasoning_rules,
        )
        self.reasoning_defeasible = DefeasibleReasoner(
            dim=dim, rules=self.reasoning_rules
        )
        self.reasoning_causal_chain = CausalChainReasoner(
            dim=dim, rules=self.reasoning_rules
        )
        # Causal/Decision domain capabilities (reuse existing core module
        # instances for free-energy scoring where relevant).
        self.causal_rules = CausalRules()
        self.causal_tree = DecisionTreeBuilder(dim=dim, rules=self.causal_rules)
        self.causal_game = GameTheoryAnalyzer(dim=dim, rules=self.causal_rules)
        self.causal_counterfactual = CounterfactualReasoner(
            dim=dim,
            active_inference=self.active_inference,
            rules=self.causal_rules,
        )
        self.causal_bandit = MultiArmedBandit(dim=dim, rules=self.causal_rules)
        self.causal_pomdp = POMDPApproximator(
            dim=dim,
            active_inference=self.active_inference,
            rules=self.causal_rules,
        )
        self.causal_graph = CausalGraphBuilder(dim=dim, rules=self.causal_rules)
        self.causal_intervention = InterventionAnalyzer(
            dim=dim,
            active_inference=self.active_inference,
            rules=self.causal_rules,
        )
        # ------------------------------------------------------------------
        # Phase 6 capabilities: Memory / Planning / Multimodal / RL.
        # Reuse the emergence engine's chaotic memory, causal engine,
        # differential generator, and HMC sampler where applicable; reuse
        # the existing nlp sentence encoder for ContextMemory.
        # ------------------------------------------------------------------
        self.memory_rules = MemoryRules()
        self.planning_rules = PlanningRules()
        self.multimodal_rules = MultimodalRules()
        self.rl_rules = RLRules()
        # Memory (8 classes).
        self.memory_episodic = EpisodicMemory(
            dim=dim,
            chaotic_memory=self.emergence.memory,
            rules=self.memory_rules,
        )
        self.memory_working = WorkingMemory(dim=dim, rules=self.memory_rules)
        self.memory_context = ContextMemory(
            dim=dim,
            sentence_encoder=self.nlp_sentence_encoder,
            rules=self.memory_rules,
        )
        self.memory_consolidator = MemoryConsolidator(
            dim=dim, rules=self.memory_rules
        )
        self.memory_hierarchical = HierarchicalMemory(
            dim=dim,
            chaotic_memory=self.emergence.memory,
            rules=self.memory_rules,
        )
        self.memory_spreading = SpreadingActivationMemory(
            dim=dim, rules=self.memory_rules
        )
        self.memory_forgetful = ForgetfulMemory(
            dim=dim, rules=self.memory_rules
        )
        self.memory_indexer = MemoryIndexer(dim=dim, rules=self.memory_rules)
        # Planning (8 classes).
        self.planning_trajectory = TrajectoryPlanner(
            dim=dim,
            differential_generator=self.emergence.differential,
            rules=self.planning_rules,
        )
        self.planning_goal_decomposer = GoalDecomposer(
            dim=dim, rules=self.planning_rules
        )
        self.planning_action_sequencer = ActionSequencer(
            dim=dim, rules=self.planning_rules
        )
        self.planning_hierarchical = HierarchicalPlanner(
            dim=dim,
            causal_inference_engine=self.emergence.causal,
            rules=self.planning_rules,
        )
        self.planning_mcts = MonteCarloTreePlanner(
            dim=dim,
            hamiltonian_sampler=self.emergence.hmc,
            rules=self.planning_rules,
            rng=_child_rngs[6],
        )
        self.planning_symbolic = SymbolicPlanner(
            dim=dim, rules=self.planning_rules
        )
        self.planning_policy_gradient = PolicyGradientPlanner(
            n_states=int(self.rl_rules.rl_n_states),
            n_actions=int(self.rl_rules.rl_n_actions),
            rules=self.planning_rules,
            rng=_child_rngs[6],
        )
        self.planning_contingency = ContingencyPlanner(
            dim=dim, rules=self.planning_rules
        )
        # Multimodal (7 classes).
        self.multimodal_aligner = CrossModalAligner(
            dim=dim, rules=self.multimodal_rules
        )
        self.multimodal_shared_space = SharedLatentSpace(
            dim_modal=dim, rules=self.multimodal_rules, rng=_child_rngs[6]
        )
        self.multimodal_fuser = ModalityFuser(
            dim=dim, rules=self.multimodal_rules
        )
        self.multimodal_encoder = ModalityEncoder(
            dim=dim, rules=self.multimodal_rules
        )
        self.multimodal_attention_fuser = AttentionBasedFuser(
            dim=dim,
            rules=self.multimodal_rules,
            rng=_child_rngs[6],
        )
        self.multimodal_contrastive = ContrastiveAligner(
            dim=dim,
            rules=self.multimodal_rules,
            hamiltonian_sampler=self.emergence.hmc,
        )
        self.multimodal_retriever = MultimodalRetriever(
            dim=dim,
            shared_latent_space=self.multimodal_shared_space,
            rules=self.multimodal_rules,
        )
        # RL (7 classes). SyntheticMDP is shared across QLearner /
        # PolicyOptimizer / DynaQ / MonteCarloTreeSearch / PosteriorSampling
        # so that all algorithms operate on the same environment.
        self.rl_mdp = SyntheticMDP(
            n_states=int(self.rl_rules.rl_n_states),
            n_actions=int(self.rl_rules.rl_n_actions),
            rules=self.rl_rules,
            seed=seed,
            rng=_child_rngs[6],
        )
        self.rl_q_learner = QLearner(
            n_states=int(self.rl_rules.rl_n_states),
            n_actions=int(self.rl_rules.rl_n_actions),
            mdp=self.rl_mdp,
            rules=self.rl_rules,
            rng=_child_rngs[6],
        )
        self.rl_policy_optimizer = PolicyOptimizer(
            n_states=int(self.rl_rules.rl_n_states),
            n_actions=int(self.rl_rules.rl_n_actions),
            mdp=self.rl_mdp,
            rules=self.rl_rules,
            rng=_child_rngs[6],
        )
        self.rl_value_function = ValueFunction(
            n_states=int(self.rl_rules.rl_n_states),
            rules=self.rl_rules,
            rng=_child_rngs[6],
        )
        self.rl_dyna_q = DynaQ(
            n_states=int(self.rl_rules.rl_n_states),
            n_actions=int(self.rl_rules.rl_n_actions),
            mdp=self.rl_mdp,
            rules=self.rl_rules,
            rng=_child_rngs[6],
        )
        self.rl_mcts = MonteCarloTreeSearch(
            mdp=self.rl_mdp,
            rules=self.rl_rules,
            rng=_child_rngs[6],
        )
        self.rl_posterior = PosteriorSampling(
            n_states=int(self.rl_rules.rl_n_states),
            n_actions=int(self.rl_rules.rl_n_actions),
            mdp=self.rl_mdp,
            rules=self.rl_rules,
            rng=_child_rngs[6],
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
        # ================================================================ #
        # 第七阶段认知升级模块（可插拔，默认关闭以保持向后兼容）。
        #
        # 各模块通过 feature flag 启用。启用后在 think() 末尾被调用，
        # 但不会修改核心 6 模块的计算逻辑——它们只读取状态、记录
        # 历史、或返回建议性元数据。这样确保已有测试零回归。
        # ================================================================ #
        # LOW-1 audit fix: ``_seed_val`` previously diverged from ``self._seed``
        # (it was forced to ``42`` when ``seed is None``). Now both hold the
        # raw user-supplied seed (``None`` when unseeded). All cognitive-upgrade
        # module constructors below call ``np.random.default_rng(seed)``, which
        # accepts ``None`` (seeds from OS entropy) — so passing ``None`` here is
        # safe and consistent with the rest of the model's unseeded semantics.
        self._seed_val = self._seed
        # 第一阶段：架构可塑性
        self.architect = None
        if enable_architect:
            from zero_data_model.plasticity.architect import ArchitectureOptimizer
            self.architect = ArchitectureOptimizer(dim=dim, seed=self._seed_val)
        # 第二阶段：层次化时间
        self.layered_predictor = None
        self.temporal_memory = None
        if enable_layered_predictor:
            from zero_data_model.cogtime.layered_predictor import LayeredPredictor
            from zero_data_model.cogtime.temporal_memory import TemporalMemory
            # LOW-2 audit fix: when ``dim`` is small (e.g. ``dim=1``),
            # ``dim // 2`` and ``dim // 4`` collapse to 0, producing
            # degenerate 0-width L1/L2 layers (``np.zeros(0)``,
            # ``np.eye(0)``) that later crash on indexing / weighting.
            # Floor each sub-dim at 1 so the layered predictor remains
            # functional for any valid ``dim >= 1``.
            _dim_l1 = max(dim // 2, 1)
            _dim_l2 = max(dim // 4, 1)
            self.layered_predictor = LayeredPredictor(
                dim_l0=dim, dim_l1=_dim_l1, dim_l2=_dim_l2, seed=self._seed_val
            )
            self.temporal_memory = TemporalMemory(
                input_dim=dim, hidden_dim=32, output_dim=dim, seed=self._seed_val
            )
        # 第三阶段：结构化记忆
        self.episodic_graph = None
        self.semantic_index = None
        if enable_episodic_memory:
            from zero_data_model.cogmem.episodic_graph import EpisodicGraph
            from zero_data_model.cogmem.semantic_index import SemanticIndex
            self.episodic_graph = EpisodicGraph()
            self.semantic_index = SemanticIndex(dim=dim)
        # 第四阶段：神经符号融合
        self.logic_layer = None
        self.causal_inference = None
        if enable_logic_layer:
            # Alias to avoid shadowing the top-level ``CausalInference``
            # import (from ``.capabilities.analytics_advanced``) used at
            # ``self.analytics_causal = CausalInference(...)`` above. A bare
            # ``from ... import CausalInference`` here would make
            # ``CausalInference`` a local variable throughout ``__init__``
            # (Python compile-time scope rule), triggering
            # ``UnboundLocalError`` at line 405 when ``enable_logic_layer``
            # is False (the default).
            from zero_data_model.knowledge.causal_inference import (
                CausalInference as _KnowledgeCausalInference,
            )
            from zero_data_model.knowledge.logic_layer import LogicLayer
            self.logic_layer = LogicLayer()
            self.causal_inference = _KnowledgeCausalInference()
        # 第五阶段：元认知
        self.meta_cognition = None
        if enable_meta_cognition:
            from zero_data_model.metacog.meta_cognition import MetaCognition
            self.meta_cognition = MetaCognition(dim=dim, seed=self._seed_val)
        # 第六阶段：主动实验设计
        self.experiment_planner = None
        self.hypothesis_tester = None
        if enable_experiment_planner:
            from zero_data_model.experiment.experiment_planner import (
                BayesianExperimentPlanner,
            )
            from zero_data_model.experiment.hypothesis_tester import HypothesisTester
            self.experiment_planner = BayesianExperimentPlanner(seed=self._seed_val)
            self.experiment_planner.default_candidates(dim=dim)
            self.hypothesis_tester = HypothesisTester(seed=self._seed_val)
        # 第七阶段 b：多智能体协作——承载多智能体世界、符号通信与文化传承。
        # 与其他升级模块一致：默认 None（opt-in），启用时惰性导入并构造。
        # ``MultiAgentWorld`` 内部为每个智能体构造独立 ZeroDataModel（使用
        # 默认 flag，故不会递归触发 multiagent hook，无无限递归）。注意
        # ``MultiAgentWorld`` 用 ``seed + i`` 派生各智能体种子，因此 seed
        # 不能为 None；当模型未播种时从 ``self._rng`` 派生一个具体种子。
        self.multiagent_world = None
        self.multiagent_communication = None
        self.multiagent_culture = None
        if enable_multiagent:
            from zero_data_model.multiagent.communication import (
                CommunicationChannel,
            )
            from zero_data_model.multiagent.culture import CulturePropagation
            from zero_data_model.multiagent.world import MultiAgentWorld

            _ma_seed = (
                self._seed_val
                if self._seed_val is not None
                else int(self._rng.integers(0, 2_000_000_000))
            )
            self.multiagent_world = MultiAgentWorld(
                n_agents=2, dim=dim, seed=_ma_seed
            )
            self.multiagent_communication = CommunicationChannel(seed=_ma_seed)
            self.multiagent_culture = CulturePropagation(seed=_ma_seed)

        # ---------------------------------------------------------------- #
        # Phase G (四.3): 三项核心认知升级模块构造
        # ---------------------------------------------------------------- #
        # 1. S4 状态空间模型：已在 ActiveInferenceEngine 内部通过 use_s4
        #    透传构造（见上方）。此处仅保留属性引用供 think() 钩子读取。
        self.s4_state_cache = None  # think() 中缓存 S4 隐状态前几维

        # 2. 层次化预测编码网络（PCN）：3 层 L0/L1/L2
        #    HierarchicalZeroDataModel 组合持有 ZeroDataModel（self），
        #    但 think() 钩子通过 run_pcn_cycle() 调用 PCN 层级（不递归
        #    调用 self.think()），避免循环。
        self.pcn_hierarchy = None
        if self._use_pcn:
            from zero_data_model.pcn.hierarchical_model import (
                HierarchicalZeroDataModel,
            )
            self.pcn_hierarchy = HierarchicalZeroDataModel(
                self, use_pcn=True, pcn_lr=0.01,
            )

        # 3. 现代 Hopfield 联想记忆：模型内部用于快速模式补全。
        #    ExperienceBuffer 的 Hopfield 集成在 experiments/ 中处理；
        #    此处的 HopfieldMemory 供 think() 钩子做在线联想检索。
        self.hopfield_memory = None
        if self._use_hopfield:
            from zero_data_model.hopfield import HopfieldMemory
            # memory_dim = dim，容量 = 1024（可扩展）
            _hf_beta = max(2.0, dim / 2.0)  # 归一化向量的可分辨 beta
            self.hopfield_memory = HopfieldMemory(
                memory_dim=dim, capacity=1024, beta=_hf_beta,
                seed=self._seed_val,
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
    # Round-8 audit SIDE-1: ``__getstate__`` takes ``self._lock`` so a
    # concurrent ``think()`` cannot mutate the numpy arrays (in-place
    # ``emission[:, :] += ...``, ``layer.weights += ...``) while pickle
    # walks the dict. Without the lock, pickle could serialise a
    # half-mutated array — the bytes on disk would be inconsistent and
    # the unpickled model would be silently corrupted. ``RLock`` is
    # reentrant so a thread that already holds the lock (e.g. ``think``
    # calling ``pickle.dumps`` internally — not currently done but
    # possible in user code) does not deadlock.
    #
    # Note: the model's authoritative persistence path remains
    # ``ModelSerializer.save/load`` (npz + json), which has its own
    # validation. These hooks only enable pickle / deepcopy.
    def __getstate__(self) -> dict:
        # Round-9 audit R9-005: the previous SIDE-1 fix took the lock only
        # while copying ``__dict__`` (a shallow dict copy), then released
        # the lock BEFORE returning. Pickle walks the returned dict (and
        # the numpy arrays it references) AFTER the lock release, so a
        # concurrent ``think()`` (which acquires ``_lock`` and performs
        # in-place ``emission[:, :] += ...``, ``layer.weights += ...``)
        # could mutate the very arrays pickle was serializing, producing
        # a torn snapshot.
        #
        # Phase D (think() lockless update): each core cognitive module
        # now holds its own per-module ``threading.RLock``. We acquire
        # all 7 locks (model + 6 modules) in a fixed order so concurrent
        # ``think()`` calls block at the per-module lock acquisition
        # until the snapshot completes. Each module implements its own
        # ``__getstate__`` that strips its own ``_lock`` from the state
        # dict — so we do NOT mutate the live lock attributes here (the
        # previous approach set them to ``None`` during deepcopy, which
        # raced with concurrent ``think()``: the background thread found
        # ``module._lock = None`` and raised ``TypeError: 'NoneType'
        # object does not support the context manager protocol``).
        # Pickle recursively calls each module's ``__getstate__`` when
        # walking the returned dict, so per-module locks are stripped at
        # the right level without any live-object mutation.
        #
        # We exclude ``_lock`` (model-level RLock, not picklable) and
        # ``parallel_executor`` (wraps a ThreadPoolExecutor, not picklable)
        # from the top-level dict; ``__setstate__`` rebuilds them.
        # Phase G: also exclude ``pcn_hierarchy`` — it holds a reference to
        # ``self`` (circular) and its own ``_lock`` (not picklable). It is
        # rebuilt in ``__setstate__`` when ``_use_pcn`` is True.
        with self._lock, \
             self.consciousness._lock, \
             self.active_inference._lock, \
             self.category_engine._lock, \
             self.quantum_hybrid._lock, \
             self.biological._lock, \
             self.math_universe._lock:
            return {
                k: v for k, v in self.__dict__.items()
                if k not in ("_lock", "parallel_executor", "pcn_hierarchy")
            }

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        # Rebuild the model-level concurrency primitives that
        # ``__getstate__`` excluded. Per-module ``_lock`` attributes are
        # rebuilt by each module's own ``__setstate__`` (called by
        # pickle when it restores the module objects in ``state``).
        self._lock = threading.RLock()
        seed = self._seed
        self.parallel_executor = ParallelExecutor(
            n_workers=1 if seed is not None else None
        )
        # Phase G (四.3): rebuild PCN hierarchy if it was enabled.
        # The old instance was excluded from pickle state (circular ref +
        # _lock); reconstruct a fresh wrapper around the restored self.
        self.pcn_hierarchy = None
        if getattr(self, "_use_pcn", False):
            from zero_data_model.pcn.hierarchical_model import (
                HierarchicalZeroDataModel,
            )
            self.pcn_hierarchy = HierarchicalZeroDataModel(
                self, use_pcn=True, pcn_lr=0.01,
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

        Phase D (think() lockless update): previously the whole cycle was
        serialized by ``self._lock`` (Fix 7), making concurrent API
        requests sequential. We now drop the global lock and rely on
        per-module ``threading.RLock`` (added to all 6 core cognitive
        modules) to protect ``process`` / ``predict`` / ``update`` /
        ``reflect`` from racing on the per-module Generators (CRIT-1)
        and on shared module state. Different modules can now run in
        parallel up to module-level contention; same-module calls
        serialize on that module's lock.

        Risk control: the only remaining shared state in ``think()`` is
        ``self.cycle_count`` (an int counter); we protect it with a brief
        ``self._lock`` acquisition at the end (sub-microsecond). All
        other setter methods (``load`` / ``save`` / ``solve`` / ...) still
        hold ``self._lock`` for now — "break through one by one" per the
        user's instruction. ``_self_generate`` acquires the per-module
        locks of the two modules it touches (``biological`` and
        ``math_universe``) so concurrent ``think()`` calls do not race
        on their RNGs.

        C-7: Per-module prediction uncertainties are computed from the input
        signal (not the integrated signal) and used to weight the integration
        via ``softmax(1/uncertainty)``. Modules that are more confident about
        the input contribute more to the integrated output, and the same
        predictions drive the ``update()`` step — avoiding a redundant second
        predict pass and aligning with predictive-processing theory (bottom-up
        prediction errors drive both integration and learning).
        """
        # Phase 1: READ — build the input Signal (no shared state mutation).
        # ``_self_generate`` acquires ``biological._lock`` and
        # ``math_universe._lock`` internally so it is safe to call
        # concurrently. When ``input_data`` is provided, no lock is needed
        # — the Signal is a fresh ndarray copied from the caller's input.
        if input_data is None:
            signal = self._self_generate()
        elif isinstance(input_data, str):
            # Tolerate raw-string callers (e.g. ``think("test")``): encode
            # the text via the existing NLP text encoder into a ``dim``-length
            # vector so the rest of the cycle operates on a numeric signal.
            # The ndarray / None paths below are unchanged (zero regression).
            signal = Signal(data=self.nlp_text_encoder.encode(input_data))
        else:
            padded = np.zeros(self.dim)
            padded[: len(input_data)] = input_data[: self.dim]
            signal = Signal(data=padded)

        # Phase 2: COMPUTE — parallel module processing + prediction.
        # Each module's ``process`` / ``predict`` acquires its own per-module
        # RLock internally, so concurrent ``think()`` calls can run in
        # parallel up to module-level contention (different modules in
        # parallel; same module serialised). No global lock held here.
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
        #
        # P2.11 性能优化: 向量化替代 Python 列表解析。原代码每周期对
        # 6 个模块逐个 float() + isfinite() + max()，产生 6 次属性
        # 访问与 Python 层分支。改为一次性 np.fromiter 提取，再用
        # np.where + np.maximum 在 numpy 层完成，避免 Python 循环。
        raw_unc = np.fromiter(
            (float(p.uncertainty) for p in preds),
            dtype=np.float64,
            count=len(preds),
        )
        finite_mask = np.isfinite(raw_unc)
        # 非有限值 -> 1e8（上限惩罚），有限值 -> max(raw, 1e-8)。
        uncertainties = np.where(
            finite_mask,
            np.maximum(raw_unc, 1e-8),
            1e8,
        )

        # C-7: Weighted integration by inverse uncertainty. ``_integrate``
        # is purely functional (no module state mutation) — no lock needed.
        integrated = self._integrate(results, uncertainties)
        # ``reflect`` reads ``consciousness.self_model.state``; protected
        # by ``ConsciousnessCore._lock`` internally.
        reflection = self.consciousness.reflect()

        # Phase 3: WRITE — per-module update. Each module's ``update``
        # acquires its own per-module RLock internally, so writes to
        # different modules are independent (no global lock needed).
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
        # Week-1 perf: parallelise the per-module ``update`` dispatch. Each
        # module's ``update`` holds its own per-module RLock internally, so
        # writes to different modules are independent and safe to run
        # concurrently -- the previous sequential ``for module, pred in zip(...)``
        # loop ran each ``module.update(error)`` in series even though the
        # modules never contend on each other's state. The per-module error
        # computation (cheap numpy ops reading ``pred.value`` and
        # ``signal.data``) is kept identical to the original zipping logic;
        # only the final ``module.update(error)`` calls move onto the thread
        # pool via ``parallel_executor.map``.
        errors: list[float] = []
        # P2.11 内存优化: 原代码每模块分配一个临时 ``diff`` 数组
        # (``diff = pred_val - sig_slice``) 仅为 ``np.dot(diff, diff)`` 读取
        # 一次。由于 ``np.asarray(...).flatten()`` 总返回新拷贝（非 view），
        # 可安全地用它作为 ``np.subtract`` 的 ``out=`` 目标，原地写入差值
        # 并复用于 ``np.dot``，省去每模块一次 dim 长度数组分配。
        # 数值结果与原实现逐位一致（同一次减法 + 同一次点积，仅落点不同）。
        # 原实现（保留为 fallback 注释）:
        #   diff = pred_val - sig_slice
        #   error = float(np.dot(diff, diff)) / max(len(pred_val), 1)
        # float32 评估: 此 error 直接驱动 ``module.update(error)`` 的学习率
        # (lr = 0.001 * error) 与 architect 的 split/prune 决策，float32 的
        # ~7 位有效数字会改变学习动力学与结构可塑性触发点 → 跳过 float32。
        for _module, pred in zip(self.modules, preds, strict=False):
            pred_val = np.asarray(pred.value, dtype=float).flatten()
            sig_slice = signal.data[: len(pred_val)]
            if len(sig_slice) < len(pred_val):
                sig_slice = np.pad(sig_slice, (0, len(pred_val) - len(sig_slice)))
            np.subtract(pred_val, sig_slice, out=pred_val)
            error = float(np.dot(pred_val, pred_val)) / max(len(pred_val), 1)
            if not np.isfinite(error):
                error = float(pred.uncertainty)
            errors.append(error)
        self.parallel_executor.map(
            lambda me: me[0].update(me[1]),
            list(zip(self.modules, errors, strict=False)),
        )

        # ``cycle_count`` is the only remaining shared state in think().
        # Use a brief global lock to serialise just the counter increment
        # (sub-microsecond) so concurrent ``think()`` calls do not lose
        # updates. Other API setters still hold ``self._lock`` for their
        # full body — they will block on this brief acquisition only if
        # they happen to run during this tiny window.
        with self._lock:
            self.cycle_count += 1
            cycle = self.cycle_count

        # Round-8 audit THEORY8-9: previously the returned Signal left
        # ``confidence`` at its default of 1.0 (base.Signal dataclass),
        # so every ``/think`` response reported ``confidence: 1.0`` to
        # clients regardless of how uncertain the model actually was.
        # Derive a meaningful confidence in (0, 1] from the mean
        # per-module uncertainty: when modules are very confident
        # (uncertainty -> 0), confidence -> 1; when they are uncertain,
        # confidence -> 0. The 1/(1+x) form keeps it bounded and smooth.
        mean_uncertainty = float(np.mean(uncertainties))
        confidence = float(1.0 / (1.0 + mean_uncertainty))

        # ================================================================ #
        # 第七阶段认知升级钩子（仅在对应 feature flag 启用时调用）。
        #
        # 所有钩子都在 think() 末尾执行，只读取已计算的状态，
        # 不修改核心 6 模块的逻辑。结果收集到 upgrade_meta dict
        # 并附加到返回的 Signal.metadata 中，供可视化前端使用。
        # ================================================================ #
        upgrade_meta: dict[str, Any] = {}
        # 第一阶段：架构可塑性——记录各模块误差，定期评估分裂/剪枝。
        # HIGH-1 (try/except): the architect hook is the only cognitive-upgrade
        # hook that was NOT wrapped in try/except. If ``architect.evaluate()``
        # raises (e.g. ``RuntimeError: list changed size during iteration``),
        # it crashed the entire ``think()`` cycle. The architect is an optional
        # optimization and must never crash think() — match the pattern used by
        # the other hooks below (which already have try/except + ``pass``).
        #
        # HIGH-2 (pass a copy): ``architect.evaluate`` historically mutated
        # ``self.modules`` (append/pop), which later broke
        # ``zip([type(m).__name__ for m in self.modules], errors)`` (silent
        # truncation when a module was added, or positional mislabeling when
        # one was pruned, because ``errors`` still had N entries). The
        # architect.py post-batch-1-fix now operates on an internal copy and
        # does NOT mutate the input — but pass a defensive copy here so even a
        # future regression in architect.py cannot corrupt the live list.
        #
        # HIGH-3 (self._lock): Phase D made think() lockless except for the
        # ``cycle_count`` counter (per-module RLocks protect ``process`` /
        # ``predict`` / ``update``). The architect hook is the one remaining
        # write-adjacent operation that touches ``self.modules`` without any
        # lock; if a concurrent metadata reader (or another ``think()``'s
        # ``record_errors`` call) iterates ``self.modules`` while the architect
        # reads it, the read could see a torn snapshot. ``self._lock`` is an
        # RLock so acquisition is safe even if a future caller already holds
        # it; here think() does NOT hold ``self._lock`` at the hook point
        # (Phase D dropped the global lock), so acquiring it serializes the
        # architect against any concurrent reader that also takes
        # ``self._lock`` (every public read API in this class does).
        if self.architect is not None:
            with self._lock:
                try:
                    module_names = [type(m).__name__ for m in self.modules]
                    self.architect.record_errors(module_names, errors)
                    # Pass a copy so architect cannot mutate the live
                    # modules list (HIGH-2 audit fix; architect.py also
                    # self-defends by copying internally).
                    arch_result = self.architect.evaluate(
                        list(self.modules), cycle
                    )
                    if arch_result:
                        upgrade_meta["architecture"] = arch_result
                except Exception as exc:
                    _logger.warning(
                        "architect hook failed: %s: %s",
                        type(exc).__name__,
                        exc,
                    )
            # Best-effort specialized gauge update (never crashes think()).
            try:
                _arch_meta = upgrade_meta.get("architecture") or {}
                ZDM_METRICS.architect_dormant_count.set(
                    int(_arch_meta.get("dormant_count", 0))
                )
                for _action in _arch_meta.get("actions", []):
                    ZDM_METRICS.architect_actions_total.labels(
                        action_type=_action.get("action", "unknown")
                    ).inc()
            except Exception as exc:
                _logger.warning(
                    "architect metric update failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )
        # 第二阶段：层次化时间——多层预测编码 + 时态记忆。
        if self.layered_predictor is not None:
            try:
                lp_result = self.layered_predictor.update(signal.data, cycle)
                upgrade_meta["layered_predictor"] = lp_result
            except Exception:
                pass
            try:
                _lp_meta = upgrade_meta.get("layered_predictor") or {}
                ZDM_METRICS.layered_belief_norm.set(
                    float(_lp_meta.get("l2_belief_norm", 0.0))
                )
            except Exception as exc:
                _logger.warning(
                    "layered_predictor metric update failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )
        if self.temporal_memory is not None:
            try:
                tm_error = self.temporal_memory.update(
                    signal.data, signal.data, lr=0.01
                )
                upgrade_meta["temporal_memory_error"] = float(tm_error)
            except Exception:
                pass
            try:
                ZDM_METRICS.temporal_spectral_radius.set(
                    float(self.temporal_memory.spectral_radius)
                )
                ZDM_METRICS.temporal_mse.set(
                    float(upgrade_meta.get("temporal_memory_error", 0.0))
                )
            except Exception as exc:
                _logger.warning(
                    "temporal_memory metric update failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )
        # 第三阶段：结构化记忆——情节图插入 + 语义索引。
        if self.episodic_graph is not None:
            try:
                belief = self.active_inference.generative_model.belief_state
                action = (
                    self.active_inference.action_history[-1]
                    if self.active_inference.action_history
                    else None
                )
                fe = (
                    float(self.active_inference.free_energy_history[-1])
                    if self.active_inference.free_energy_history
                    else 0.0
                )
                self.episodic_graph.insert(
                    state=belief, action=action, next_state=None,
                    free_energy=fe, step=cycle,
                )
                upgrade_meta["episodic_nodes"] = self.episodic_graph.node_count
            except Exception:
                pass
            try:
                ZDM_METRICS.episodic_node_count.set(
                    int(self.episodic_graph.node_count)
                )
                ZDM_METRICS.episodic_edge_count.set(
                    int(self.episodic_graph.edge_count)
                )
            except Exception as exc:
                _logger.warning(
                    "episodic_graph metric update failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )
        # 第三阶段 b：语义索引——为当前 belief 向量建立可检索索引。
        # MEDIUM-1 audit fix: ``semantic_index`` was constructed in ``__init__``
        # (gated by ``enable_episodic_memory``) but never invoked, so it was
        # dead weight. Wire it up as an OPTIONAL hook (guarded by ``is not
        # None``) that indexes the current belief vector each cycle. ``Signal``
        # (base.py) only carries ``data`` / ``metadata`` / ``confidence`` —
        # there is no ``belief_state`` field — so we index ``signal.data``
        # (the integrated belief vector for this cycle).
        if self.semantic_index is not None:
            try:
                _vec = np.asarray(signal.data).ravel()
                self.semantic_index.add(
                    _vec, node_id=cycle,
                    metadata={"cycle": cycle, "label": "belief"},
                )
                # ``size`` is a @property on SemanticIndex (not a method).
                upgrade_meta["semantic_index_size"] = self.semantic_index.size
            except Exception as exc:
                _logger.warning(
                    "semantic_index hook failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )
            try:
                ZDM_METRICS.semantic_index_size.set(
                    int(self.semantic_index.size)
                )
            except Exception as exc:
                _logger.warning(
                    "semantic_index metric update failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )
        # 第四阶段：逻辑约束层——检查规则违反并产生惩罚。
        # P2.11 性能优化: ``check_all()`` 与 ``get_penalty_signal()`` 各自遍历
        # 全部规则。原实现每周期调用 ``check_all()`` 两次（一次检测违反、一次
        # 更新指标）外加 ``get_penalty_signal()`` 一次，共 3 次规则遍历。改为
        # 复用首次 ``check_all()`` 的结果更新指标，消除冗余的第二次遍历；保留
        # fallback：仅当首次调用异常时才在指标块里重新调用（与原实现的独立
        # 异常隔离语义一致）。``get_penalty_signal()`` 仍需独立调用以获取每条
        # 规则的惩罚数组（``check_all`` 只返回汇总 ``total_penalty``）。
        if self.logic_layer is not None:
            try:
                logic_result = self.logic_layer.check_all()
            except Exception:
                logic_result = None
            if logic_result is not None:
                try:
                    if logic_result["n_violations"] > 0:
                        upgrade_meta["logic_violations"] = logic_result
                except Exception:
                    pass
            try:
                # 复用已计算的 logic_result；仅在首次调用失败时 fallback 重算。
                _lr = (
                    logic_result
                    if logic_result is not None
                    else self.logic_layer.check_all()
                )
                ZDM_METRICS.logic_violation_count.set(
                    int(_lr.get("n_violations", 0))
                )
                _ps = np.asarray(self.logic_layer.get_penalty_signal(), dtype=float)
                ZDM_METRICS.logic_penalty.set(
                    float(_ps.mean()) if _ps.size else 0.0
                )
            except Exception as exc:
                _logger.warning(
                    "logic_layer metric update failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )
        # 第四阶段 b：因果推断——表面当前转移矩阵维度，保持模型"热"。
        # MEDIUM-1 audit fix: ``causal_inference`` was constructed in
        # ``__init__`` (gated by ``enable_logic_layer``) but never invoked.
        # Wire it up as an OPTIONAL hook that surfaces the transition-matrix
        # dimension in metadata. The model constructs ``CausalInference()`` with
        # no args, so ``transition_matrix`` may be ``None``; guard explicitly.
        if self.causal_inference is not None:
            try:
                _tm = getattr(self.causal_inference, "transition_matrix", None)
                _n = (
                    int(_tm.shape[0])
                    if _tm is not None and hasattr(_tm, "shape") and _tm.ndim >= 1
                    else 0
                )
                upgrade_meta["causal_dim"] = _n
            except Exception as exc:
                _logger.warning(
                    "causal_inference hook failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )
        # 第五阶段：元认知——更新不确定性估计。
        if self.meta_cognition is not None:
            try:
                fe = (
                    float(self.active_inference.free_energy_history[-1])
                    if self.active_inference.free_energy_history
                    else 0.0
                )
                param_norm = float(
                    np.linalg.norm(
                        self.active_inference.generative_model.transition
                    )
                )
                meta_result = self.meta_cognition.update(
                    prediction_error=fe, param_update_norm=param_norm
                )
                upgrade_meta["meta_cognition"] = meta_result
            except Exception:
                pass
            try:
                _mc_meta = upgrade_meta.get("meta_cognition") or {}
                ZDM_METRICS.metacog_confidence.set(
                    float(_mc_meta.get("confidence", 0.0))
                )
                ZDM_METRICS.metacog_uncertainty.set(
                    float(_mc_meta.get("mean_uncertainty", 0.0))
                )
            except Exception as exc:
                _logger.warning(
                    "meta_cognition metric update failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )
        # 第六阶段：主动实验设计——定期评估候选实验。
        if self.experiment_planner is not None:
            try:
                exp_result = self.experiment_planner.evaluate(
                    current_uncertainty=mean_uncertainty, step=cycle
                )
                if exp_result is not None:
                    upgrade_meta["experiment"] = exp_result
            except Exception:
                pass
            try:
                ZDM_METRICS.experiment_candidates.set(
                    int(len(self.experiment_planner.candidates))
                )
                _exp_meta = upgrade_meta.get("experiment") or {}
                ZDM_METRICS.experiment_info_gain.set(
                    float(_exp_meta.get("predicted_gain", 0.0))
                )
            except Exception as exc:
                _logger.warning(
                    "experiment_planner metric update failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )
        # 第六阶段 b：假设检验器——报告当前已支持的假设数量。
        # MEDIUM-1 audit fix: ``hypothesis_tester`` was constructed in
        # ``__init__`` (gated by ``enable_experiment_planner``) but never
        # invoked. Wire it up as an OPTIONAL hook that surfaces the number
        # of currently-supported hypotheses in metadata.
        if self.hypothesis_tester is not None:
            try:
                _supported = self.hypothesis_tester.get_supported()
                upgrade_meta["supported_hypotheses"] = len(_supported)
            except Exception as exc:
                _logger.warning(
                    "hypothesis_tester hook failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )
            try:
                ZDM_METRICS.hypothesis_supported.set(
                    int(upgrade_meta.get("supported_hypotheses", 0))
                )
            except Exception as exc:
                _logger.warning(
                    "hypothesis_tester metric update failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )
        # 第七阶段 c：多智能体协作——推进多智能体世界、驱动符号通信与
        # 文化传承，并把三个特化指标（collaboration_events /
        # communication_usage / culture_generations）更新到 Prometheus。
        # 与其他 hook 一致：try/except 包裹，异常绝不拖垮 think()。
        # ``MultiAgentWorld.step`` 返回信号列表（非 dict），故这里通过
        # 各组件的 stats 属性聚合所需字段。
        if self.multiagent_world is not None:
            try:
                # 推进世界一步：让每个内部智能体思考并检测协作。
                self.multiagent_world.step()
                # 驱动通信通道：本轮记录一次符号使用，使"语言涌现"
                # 统计随周期推进。符号按 cycle 取模选取，保证落在词表内。
                _sym = int(cycle) % int(
                    self.multiagent_communication.vocab_size
                )
                self.multiagent_communication.record_usage(
                    _sym, event_type="think_cycle"
                )
                # 记录一代文化快照：用当前自由能与（可选）情节图规模
                # 作为该代的知识量代理，n_steps 用当前 cycle。
                _fe = (
                    float(self.active_inference.free_energy_history[-1])
                    if self.active_inference.free_energy_history
                    else 0.0
                )
                _kg = (
                    int(self.episodic_graph.node_count)
                    if self.episodic_graph is not None
                    else 0
                )
                self.multiagent_culture.record_generation(
                    knowledge_graph_size=_kg,
                    mean_free_energy=_fe,
                    n_steps=int(cycle),
                )
                # 聚合三个组件的统计到 metadata。
                _collab = self.multiagent_world.get_collaboration_stats()
                _comm = self.multiagent_communication.stats
                _cult = self.multiagent_culture.stats
                _collab_events = int(_collab.get("n_events", 0))
                _comm_usage = int(_comm.get("total_usage", 0))
                _culture_gens = int(_cult.get("n_generations", 0))
                upgrade_meta["phase7_multiagent"] = {
                    "collaboration_events": _collab_events,
                    "communication_usage": _comm_usage,
                    "culture_generations": _culture_gens,
                    "agents": int(self.multiagent_world.agent_count),
                    "step": int(self.multiagent_world.step_count),
                }
            except Exception as exc:
                _logger.warning(
                    "multiagent hook failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )
            try:
                _ma_meta = upgrade_meta.get("phase7_multiagent") or {}
                ZDM_METRICS.multiagent_collaboration_events.set(
                    int(_ma_meta.get("collaboration_events", 0))
                )
                ZDM_METRICS.communication_usage.set(
                    int(_ma_meta.get("communication_usage", 0))
                )
                ZDM_METRICS.culture_generations.set(
                    int(_ma_meta.get("culture_generations", 0))
                )
            except Exception as exc:
                _logger.warning(
                    "multiagent metric update failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )

        # ---------------------------------------------------------------- #
        # Phase G (四.3): S4 / PCN / Hopfield think() 钩子
        # ---------------------------------------------------------------- #
        # 与其他认知升级钩子一致：每个钩子 try/except 包裹，绝不拖垮
        # think() 主流程。结果写入 upgrade_meta，经 _sanitize_for_json
        # 后暴露在 signal.metadata["cognitive_upgrades"] 中。

        # --- G.1: S4 隐状态暴露 ----------------------------------------- #
        # S4 的 step()/update() 已在 active_inference.process() 内部完成
        # （predict_next_state + update_belief 的延迟 Hebbian 更新）。
        # 此钩子仅读取 S4 隐状态前几维供可视化/诊断。
        if self._use_s4 and self.active_inference is not None:
            try:
                _s4 = getattr(
                    self.active_inference.generative_model, "_s4_layer", None
                )
                if _s4 is not None:
                    _s4_state = getattr(_s4, "_state", None)
                    if _s4_state is not None and np.all(np.isfinite(_s4_state)):
                        # 缓存前 8 维（或更少）用于快照
                        _n_show = min(8, len(_s4_state))
                        self.s4_state_cache = [
                            float(x) for x in _s4_state[:_n_show]
                        ]
                        upgrade_meta["s4_state"] = self.s4_state_cache
                        upgrade_meta["s4_spectral_radius"] = float(
                            getattr(_s4, "spectral_radius", 0.0)
                        )
            except Exception as exc:
                _logger.warning(
                    "S4 state hook failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )

        # --- G.2: PCN 层次化预测编码 ------------------------------------ #
        # 将集成后的观测喂入 L0/L1/L2 层级，执行自顶向下预测 + 自底向上
        # 误差更新。run_pcn_cycle 不调用 self.think()（避免循环递归）。
        if self._use_pcn and self.pcn_hierarchy is not None:
            try:
                _pcn_obs = integrated.data
                _pcn_result = self.pcn_hierarchy.run_pcn_cycle(_pcn_obs)
                if _pcn_result is not None:
                    upgrade_meta["pcn"] = _pcn_result
            except Exception as exc:
                _logger.warning(
                    "PCN hook failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )

        # --- G.3: Hopfield 联想记忆检索 --------------------------------- #
        # 用当前观测查询 Hopfield 记忆矩阵，返回 top-k 最相似的记忆
        # 摘要（存储数量、top-1 相似度）。若记忆库为空则跳过。
        if self._use_hopfield and self.hopfield_memory is not None:
            try:
                _hf = self.hopfield_memory
                if _hf.size > 0:
                    _query = integrated.data
                    # 归一化查询向量（与存储时一致）
                    _q_norm = np.linalg.norm(_query)
                    _query = _query / _q_norm if _q_norm > 1e-12 else _query
                    _retrieved, _sims = _hf.retrieve(_query, k=1)
                    upgrade_meta["memory_retrieved"] = {
                        "n_memories": int(_hf.size),
                        "top1_similarity": (
                            float(_sims[0]) if len(_sims) > 0 else 0.0
                        ),
                    }
            except Exception as exc:
                _logger.warning(
                    "Hopfield retrieve hook failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )

        # ---------------------------------------------------------------- #
        # Build the return Signal metadata dict.
        # ---------------------------------------------------------------- #
        # HIGH-2 (recompute names): ``errors`` is computed from the modules
        # list at the START of think(). If an upgrade hook mutated
        # ``self.modules`` (split / prune) between then and now, the lengths
        # diverge. Recompute ``_final_module_names`` from the CURRENT
        # ``self.modules`` right before the zip, and truncate to the shorter
        # of (names, errors) — preferring ``errors``' length since it
        # reflects the modules that actually ran this cycle. This prevents
        # both the silent truncation (KeyError) and the positional
        # mislabeling (errors[N-1] mapped to a different module's name).
        #
        # MEDIUM-2: disambiguate split modules that share a class name so
        # ``dict(zip(...))`` does not collapse them (last one wins). After a
        # split the architect may produce two modules of the same class —
        # append an ``#index`` suffix to keep both entries visible.
        _seen: dict[str, int] = {}
        _final_module_names: list[str] = []
        for _m in self.modules:
            _nm = type(_m).__name__
            if _nm in _seen:
                _seen[_nm] += 1
                _nm = f"{_nm}#{_seen[_nm]}"
            else:
                _seen[_nm] = 0
            _final_module_names.append(_nm)
        _n = min(len(_final_module_names), len(errors))

        # MEDIUM-4: only surface ``module_errors`` / ``cognitive_upgrades``
        # when at least one cognitive-upgrade module is enabled. The default
        # (no-upgrades) configuration must keep the original metadata
        # key-set so strict-key-set tests do not regress.
        _cognitive_active = any(
            getattr(self, _attr, None) is not None
            for _attr in (
                "architect",
                "layered_predictor",
                "episodic_graph",
                "logic_layer",
                "meta_cognition",
                "experiment_planner",
                "semantic_index",
                "causal_inference",
                "hypothesis_tester",
                "multiagent_world",
                # Phase G (四.3): PCN 与 Hopfield 模块
                "pcn_hierarchy",
                "hopfield_memory",
            )
        ) or (
            # Phase G (四.1): S4 层在 active_inference.generative_model
            # 内部，非 self 直接属性，单独检查
            self._use_s4
            and self.active_inference is not None
            and getattr(
                self.active_inference.generative_model, "_s4_layer", None
            ) is not None
        )

        _metadata: dict[str, Any] = {
            "cycle": cycle,
            "self_reflection": reflection.metadata,
            "module_count": len(self.modules),
            # Round-10 audit R10-C-010: propagate the
            # ``self_generated`` flag from ``_self_generate`` so
            # callers can tell whether this cycle ran on a real
            # input or on internally-generated content. Previously
            # the metadata was rebuilt here and the flag was lost,
            # so any downstream code reading
            # ``result.metadata["self_generated"]`` raised KeyError.
            "self_generated": input_data is None,
        }
        if _cognitive_active:
            # 第七阶段：暴露各模块预测误差供架构优化器使用。
            # MEDIUM-3: sanitize numpy floats / arrays so the dict is
            # always JSON-serializable (FastAPI ``jsonable_encoder`` raises
            # on ``np.float64``).
            _metadata["module_errors"] = _sanitize_for_json(
                dict(
                    zip(
                        _final_module_names[:_n],
                        errors[:_n],
                        strict=False,
                    )
                )
            )
            # 第七阶段：认知升级钩子结果（仅启用时存在）。
            _metadata["cognitive_upgrades"] = _sanitize_for_json(upgrade_meta)

        return Signal(
            data=integrated.data,
            metadata=_metadata,
            confidence=confidence,
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
            # P2.11 内存优化: 用 ``out=`` 原地写入 inv_unc / weights，省去
            # 中间临时数组（原代码每次 rebind 都新建一个数组）。数值结果
            # 与原实现逐位一致（同序同算，仅落点不同）。
            # 原实现（保留为 fallback 注释）:
            #   inv_unc = 1.0 / np.asarray(uncertainties, dtype=float)
            #   inv_unc = inv_unc - np.max(inv_unc)  # numerical stability
            #   weights = np.exp(inv_unc)
            #   weights = weights / (np.sum(weights) + 1e-12)
            # float32 评估: softmax 的 max-subtraction 数值稳定性依赖
            # float64 的 ~15 位有效数字；当某模块 uncertainty 极小(1e-8)
            # 时 float32 (~7 位) 会损失精度导致权重偏移 → 跳过 float32。
            unc_arr = np.asarray(uncertainties, dtype=float)
            inv_unc = np.empty_like(unc_arr)
            np.divide(1.0, unc_arr, out=inv_unc)
            np.subtract(inv_unc, np.max(inv_unc), out=inv_unc)
            weights = np.empty_like(inv_unc)
            np.exp(inv_unc, out=weights)
            np.divide(weights, np.sum(weights) + 1e-12, out=weights)
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

    # ------------------------------------------------------------------ #
    # Audio domain facades
    # ------------------------------------------------------------------ #

    def encode_audio(self, signal: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Encode a 1D audio signal into a ``dim``-length L2-normalized vector."""
        with self._lock:
            return self.audio_encoder.encode(signal, sample_rate=sample_rate)

    def detect_onsets(self, signal: np.ndarray, sample_rate: int = 16000) -> dict:
        """Detect note/onset events via spectral flux."""
        with self._lock:
            return self.audio_onset.detect(signal, sample_rate=sample_rate)

    def detect_pitch(self, signal: np.ndarray, sample_rate: int = 16000) -> dict:
        """Detect the fundamental frequency via autocorrelation."""
        with self._lock:
            return self.audio_pitch.detect(signal, sample_rate=sample_rate)

    def classify_audio(self, signal: np.ndarray, sample_rate: int = 16000) -> dict:
        """Classify audio texture (speech/music/noise/silence)."""
        with self._lock:
            return self.audio_classifier.classify(signal, sample_rate=sample_rate)

    def segment_speech(self, signal: np.ndarray, sample_rate: int = 16000) -> dict:
        """Segment audio into speech/silence regions via VAD."""
        with self._lock:
            return self.audio_segmenter.segment(signal, sample_rate=sample_rate)

    def analyze_music(self, signal: np.ndarray, sample_rate: int = 16000) -> dict:
        """Analyze music for tempo and beats."""
        with self._lock:
            return self.audio_music.analyze(signal, sample_rate=sample_rate)

    def extract_speaker_features(self, signal: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Extract speaker-discriminative LPC + formant features."""
        with self._lock:
            return self.audio_speaker.extract_features(signal, sample_rate=sample_rate)

    # ------------------------------------------------------------------ #
    # Graph domain facades
    # ------------------------------------------------------------------ #

    def encode_graph(
        self,
        adjacency: np.ndarray,
        node_features: np.ndarray | None = None,
    ) -> np.ndarray:
        """Encode a graph into a ``dim``-length L2-normalized vector."""
        with self._lock:
            return self.graph_encoder.encode(adjacency, node_features=node_features)

    def detect_communities(self, adjacency: np.ndarray) -> dict:
        """Detect communities via modularity optimization."""
        with self._lock:
            return self.graph_community.detect(adjacency)

    def find_path(self, adjacency: np.ndarray, source: int, target: int) -> dict:
        """Find the shortest path via Dijkstra."""
        with self._lock:
            return self.graph_path.find(adjacency, source, target)

    def analyze_centrality(self, adjacency: np.ndarray) -> dict:
        """Analyze degree, betweenness, and closeness centrality."""
        with self._lock:
            return self.graph_centrality.analyze(adjacency)

    def check_isomorphism(self, adj_a: np.ndarray, adj_b: np.ndarray) -> dict:
        """Check if two graphs are likely isomorphic (Weisfeiler-Lehman)."""
        with self._lock:
            return self.graph_isomorphism.check(adj_a, adj_b)

    def track_dynamic_graph(self, snapshots: list) -> dict:
        """Track community drift across graph snapshots."""
        with self._lock:
            return self.graph_dynamic.track(snapshots)

    def extract_spanning_tree(self, adjacency: np.ndarray) -> dict:
        """Extract the minimum spanning tree via Kruskal."""
        with self._lock:
            return self.graph_spanning.extract(adjacency)

    # ------------------------------------------------------------------ #
    # Robotics domain facades
    # ------------------------------------------------------------------ #

    def plan_motion(self, waypoints: np.ndarray, n_steps: int = 100) -> dict:
        """Plan a smooth trajectory through ``waypoints`` (cubic spline)."""
        with self._lock:
            return self.robotics_motion.plan(waypoints, n_steps=n_steps)

    def forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        """Compute the end-effector position of a planar N-DOF arm."""
        with self._lock:
            return self.robotics_kinematics.forward(joint_angles)

    def inverse_kinematics(
        self, target: np.ndarray, seed: np.ndarray | None = None
    ) -> dict:
        """Solve inverse kinematics via damped least squares."""
        with self._lock:
            return self.robotics_kinematics.inverse(target, seed=seed)

    def fuse_sensors(
        self,
        measurements: list[np.ndarray],
        variances: list[float],
    ) -> np.ndarray:
        """Fuse multiple sensor readings by inverse-variance weighting."""
        with self._lock:
            return self.robotics_sensor.fuse(measurements, variances)

    def update_kalman(
        self,
        prior: np.ndarray,
        prior_var: float,
        measurement: np.ndarray,
        meas_var: float,
    ) -> dict:
        """Sequential Kalman-style Bayesian update."""
        with self._lock:
            return self.robotics_sensor.update(
                prior, prior_var, measurement, meas_var
            )

    def generate_gait(self, n_steps: int = 100, gait_type: str = "walk") -> dict:
        """Generate a periodic gait pattern (walk/trot/bound)."""
        with self._lock:
            return self.robotics_gait.generate(n_steps=n_steps, gait_type=gait_type)

    def optimize_trajectory(self, trajectory: np.ndarray, n_iter: int = 10) -> dict:
        """Smooth a trajectory by minimizing jerk (gradient descent)."""
        with self._lock:
            return self.robotics_trajectory.optimize(trajectory, n_iter=n_iter)

    def check_collision(
        self,
        obstacles: list,
        position: np.ndarray,
        radius: float = 0.1,
    ) -> dict:
        """Check collision between a body and circular obstacles."""
        with self._lock:
            return self.robotics_collision.check(obstacles, position, radius=radius)

    def check_path_collision(
        self,
        obstacles: list,
        path: np.ndarray,
        radius: float = 0.1,
    ) -> dict:
        """Check collisions along a path of body positions."""
        with self._lock:
            return self.robotics_collision.check_path(obstacles, path, radius=radius)

    def control_mpc(
        self,
        current_state: np.ndarray,
        target_state: np.ndarray,
        obstacles: list | None = None,
    ) -> dict:
        """Pick the next control action via model predictive control."""
        with self._lock:
            return self.robotics_mpc.control(
                current_state, target_state, obstacles=obstacles
            )

    # ------------------------------------------------------------------ #
    # Time-series domain facades
    # ------------------------------------------------------------------ #

    def encode_time_series(self, series: np.ndarray) -> np.ndarray:
        """Encode a 1D time series into a ``dim``-length L2-normalized vector."""
        with self._lock:
            return self.time_encoder.encode(series)

    def detect_seasonality(self, series: np.ndarray) -> dict:
        """Detect seasonal periods via autocorrelation peaks."""
        with self._lock:
            return self.time_seasonality.detect(series)

    def analyze_frequency(self, series: np.ndarray) -> dict:
        """Analyze the frequency content of a series (FFT + spectral entropy)."""
        with self._lock:
            return self.time_frequency.analyze(series)

    def analyze_event_timestamps(self, timestamps: np.ndarray) -> dict:
        """Analyze event timing (inter-arrival, rate, burstiness)."""
        with self._lock:
            return self.time_events.analyze(timestamps)

    def detect_anomalous_timing(self, timestamps: np.ndarray) -> dict:
        """Detect anomalous inter-arrival gaps via z-score."""
        with self._lock:
            return self.time_anomaly_timing.detect(timestamps)

    def track_cycle_phase(
        self, series: np.ndarray, period: int | None = None
    ) -> dict:
        """Track phase within a periodic cycle."""
        with self._lock:
            return self.time_cycle_phase.track(series, period=period)

    def score_forecastability(self, series: np.ndarray) -> dict:
        """Score how forecastable a series is (entropy + stationarity + autocorr)."""
        with self._lock:
            return self.time_forecastability.score(series)

    # ------------------------------------------------------------------ #
    # Code-analysis domain facades
    # ------------------------------------------------------------------ #

    def encode_code(self, source: str) -> np.ndarray:
        """Encode Python source code into a ``dim``-length L2-normalized vector."""
        with self._lock:
            return self.code_encoder.encode(source)

    def analyze_ast(self, source: str) -> dict:
        """Analyze AST structure (functions, classes, imports, complexity)."""
        with self._lock:
            return self.code_ast.analyze(source)

    def compare_code(self, source_a: str, source_b: str) -> dict:
        """Compare two code snippets via token + structure similarity."""
        with self._lock:
            return self.code_similarity.compare(source_a, source_b)

    def detect_code_defects(self, source: str) -> dict:
        """Detect rule-based defect patterns (mutable_default, bare_except, equals_none)."""
        with self._lock:
            return self.code_defects.detect(source)

    def analyze_control_flow(self, source: str) -> dict:
        """Analyze control flow (per-function cyclomatic complexity, basic blocks)."""
        with self._lock:
            return self.code_control_flow.analyze(source)

    def analyze_code_style(self, source: str) -> dict:
        """Analyze code style (line length, naming, whitespace)."""
        with self._lock:
            return self.code_style.analyze(source)

    def build_dependency_graph(self, source: str) -> dict:
        """Build a module-level import dependency graph."""
        with self._lock:
            return self.code_dependency.build(source)

    # ------------------------------------------------------------------ #
    # Reasoning domain facades
    # ------------------------------------------------------------------ #

    def add_logical_fact(self, proposition: str, value: bool) -> None:
        """Add a fact to the propositional logic knowledge base."""
        with self._lock:
            self.reasoning_propositional.add_fact(proposition, value)

    def add_logical_rule(self, antecedent: str, consequent: str) -> None:
        """Add a modus-ponens rule ``antecedent -> consequent``."""
        with self._lock:
            self.reasoning_propositional.add_rule(antecedent, consequent)

    def infer_logical(self) -> dict:
        """Run forward inference (modus ponens) until fixpoint."""
        with self._lock:
            return self.reasoning_propositional.infer()

    def syllogism(self, major: tuple, minor: tuple) -> dict:
        """Apply a categorical syllogism (Barbara / Celarent / Darii / Ferio)."""
        with self._lock:
            return self.reasoning_deductive.syllogism(major, minor)

    def induct_rule(
        self, examples: list[dict], labels: list[bool]
    ) -> dict:
        """Induce the best (key, value) rule from labeled examples."""
        with self._lock:
            return self.reasoning_inductive.generalize(examples, labels)

    def analogize(self, source: dict, target: dict) -> dict:
        """Map source attributes to target attributes by structural similarity."""
        with self._lock:
            return self.reasoning_analogical.analogize(source, target)

    def abduce(
        self,
        observation: str,
        hypotheses: list[str],
        priors: list[float] | None = None,
    ) -> dict:
        """Find the best explanation for ``observation`` among ``hypotheses``."""
        with self._lock:
            return self.reasoning_abductive.explain(
                observation, hypotheses, priors=priors
            )

    def add_default_rule(
        self, rule: tuple, exception: tuple | None = None
    ) -> None:
        """Add a defeasible default rule with optional exception."""
        with self._lock:
            self.reasoning_defeasible.add_default(rule, exception=exception)

    def conclude_defaults(self, facts: dict) -> dict:
        """Apply defaults to ``facts`` (returns conclusions, defeated, ambiguous)."""
        with self._lock:
            return self.reasoning_defeasible.conclude(facts)

    def add_causal_link(self, cause: str, effect: str) -> None:
        """Add a causal edge ``cause -> effect``."""
        with self._lock:
            self.reasoning_causal_chain.add_causal(cause, effect)

    def trace_causal_chain(self, start: str, max_depth: int = 5) -> dict:
        """Trace the causal chain starting from ``start`` (DFS with cycle detection)."""
        with self._lock:
            return self.reasoning_causal_chain.trace(start, max_depth=max_depth)

    # ------------------------------------------------------------------ #
    # Causal / Decision domain facades.
    # ------------------------------------------------------------------ #

    def fit_decision_tree(self, features: np.ndarray, labels: np.ndarray) -> dict:
        """Fit an ID3-style decision tree to ``(features, labels)``."""
        with self._lock:
            return self.causal_tree.fit(features, labels)

    def analyze_game(
        self,
        payoff_a: np.ndarray,
        payoff_b: np.ndarray | None = None,
    ) -> dict:
        """Analyze a 2-player normal-form game; find pure Nash equilibria."""
        with self._lock:
            return self.causal_game.analyze(payoff_a, payoff_b)

    def counterfactual(
        self,
        observed: np.ndarray,
        intervention: dict,
        model_fn=None,
    ) -> dict:
        """Estimate the counterfactual outcome under ``intervention``."""
        with self._lock:
            return self.causal_counterfactual.counterfactual(
                observed, intervention, model_fn=model_fn
            )

    def select_bandit_arm(self, rewards_history: list[list[float]]) -> dict:
        """Select the next bandit arm (epsilon-greedy + UCB1)."""
        with self._lock:
            return self.causal_bandit.select(rewards_history)

    def solve_pomdp(
        self,
        transitions: np.ndarray,
        observations: np.ndarray,
        rewards: np.ndarray,
    ) -> dict:
        """Solve a (PO)MDP via value iteration."""
        with self._lock:
            return self.causal_pomdp.solve(transitions, observations, rewards)

    def discover_causal_graph(
        self,
        data: np.ndarray,
        var_names: list[str] | None = None,
    ) -> dict:
        """Discover a causal graph from observational data (PC-style)."""
        with self._lock:
            return self.causal_graph.discover(data, var_names=var_names)

    def intervene(
        self,
        data: np.ndarray,
        intervention_var: int,
        intervention_value: float,
    ) -> dict:
        """Estimate the effect of ``do(X[intervention_var] = value)``."""
        with self._lock:
            return self.causal_intervention.intervene(
                data, intervention_var, intervention_value
            )

    # ------------------------------------------------------------------
    # Causal emergence engine facade (spec §2.3)
    # ------------------------------------------------------------------
    def perceive_topology(
        self, data: np.ndarray, max_dim: int | None = None
    ) -> dict:
        """Perceive topological invariants of ``data`` (spec §2.3).

        Delegates to ``CausalEmergenceEngine.perceive_topology`` under
        ``self._lock`` (fix M9: lock granularity is the full method body).
        Returns dict with ``betti_numbers``, ``persistence_entropy``,
        ``euler_characteristic``, ``n_points``, ``max_eps``,
        ``persistence_diagram``.
        """
        with self._lock:
            return self.emergence.perceive_topology(data, max_dim=max_dim)

    def discover_causal_dynamics(
        self,
        data: np.ndarray,
        var_names: list[str] | None = None,
        method: str | None = None,
    ) -> dict:
        """Discover causal DAG from observational data (spec §2.3).

        Distinct from ``discover_causal_graph`` (which uses the basic
        CausalGraphBuilder capability); this facade uses the emergence
        engine's PC/LiNGAM/correlation pipeline.
        """
        with self._lock:
            return self.emergence.discover_causal_dynamics(
                data, var_names=var_names, method=method
            )

    def generate_trajectory(
        self,
        start_state: np.ndarray,
        end_state: np.ndarray,
        n_steps: int = 32,
        constraints: dict | None = None,
    ) -> dict:
        """Generate damped least-action trajectory (spec §2.3)."""
        with self._lock:
            return self.emergence.generate_trajectory(
                start_state, end_state, n_steps=n_steps, constraints=constraints
            )

    def sample_posterior(
        self,
        log_prob_fn,
        initial_position: np.ndarray,
        n_samples: int | None = None,
        step_size: float | None = None,
        n_leapfrog: int | None = None,
        grad_fn=None,
    ) -> dict:
        """Sample posterior via HMC (spec §2.3)."""
        with self._lock:
            return self.emergence.sample_posterior(
                log_prob_fn=log_prob_fn,
                initial_position=initial_position,
                n_samples=n_samples,
                step_size=step_size,
                n_leapfrog=n_leapfrog,
                grad_fn=grad_fn,
            )

    def recall_memory(self, query: np.ndarray, n_steps: int = 100) -> dict:
        """Recall from chaotic associative memory (spec §2.3)."""
        with self._lock:
            return self.emergence.recall_memory(query, n_steps=n_steps)

    def emergence_cycle(self, observation: np.ndarray) -> dict:
        """Run the full recursive emergence loop (spec §2.3, §8.2).

        Observation must be 2D ``(n_samples, n_features)`` with both
        dimensions >= 2; otherwise returns ``{'emergence_score': 0.0,
        'reason': 'insufficient_data'}``.

        Returns dict with ``perception``, ``causal_graph``,
        ``counterfactual``, ``posterior``, ``memory``,
        ``emergence_score``, ``warnings``.
        """
        with self._lock:
            return self.emergence.emergence_cycle(observation)

    # ------------------------------------------------------------------
    # Phase 6 — Memory domain facades
    # ------------------------------------------------------------------

    def encode_memory(
        self, observation: np.ndarray, label: object = None
    ) -> dict:
        """Encode an observation into the episodic memory store."""
        with self._lock:
            return self.memory_episodic.encode(observation, label=label)

    def retrieve_memory(self, query: np.ndarray, top_k: int = 5) -> dict:
        """Retrieve the top-k most similar episodic memories."""
        with self._lock:
            return self.memory_episodic.retrieve(query, top_k=top_k)

    def push_working(self, observation: np.ndarray) -> dict:
        """Push an observation onto the working-memory window."""
        with self._lock:
            return self.memory_working.push(observation)

    def working_attention(self) -> dict:
        """Return softmax attention weights over the working memory."""
        with self._lock:
            return self.memory_working.attention_weights()

    def add_context_turn(
        self, user_msg: str, agent_msg: str = ""
    ) -> dict:
        """Append a (user, agent) turn to the context memory."""
        with self._lock:
            return self.memory_context.add_turn(user_msg, agent_msg)

    def consolidate_memory(self) -> dict:
        """Promote high-weight working-memory items into episodic memory."""
        with self._lock:
            return self.memory_consolidator.consolidate(
                self.memory_working, self.memory_episodic
            )

    def retrieve_hierarchical(
        self, query: np.ndarray, top_k: int = 5
    ) -> dict:
        """Retrieve from the hierarchical 3-tier memory."""
        with self._lock:
            return self.memory_hierarchical.retrieve(query, top_k=top_k)

    def retrieve_spreading(
        self, query: np.ndarray, top_k: int | None = None
    ) -> dict:
        """Retrieve via spreading activation over the similarity graph."""
        with self._lock:
            return self.memory_spreading.retrieve(query, top_k=top_k)

    def retrieve_forgetful(self, query: np.ndarray, top_k: int = 5) -> dict:
        """Retrieve with Ebbinghaus forgetting-curve decay weighting."""
        with self._lock:
            return self.memory_forgetful.retrieve(query, top_k=top_k)

    def review_memory(self, ids: list[int]) -> dict:
        """Reset the timestamp of reviewed items (Ebbinghaus spaced-repetition)."""
        with self._lock:
            return self.memory_forgetful.review(ids)

    def index_memory(
        self, observations: np.ndarray, labels: list | None = None
    ) -> dict:
        """Batch-build the inverted index over observations."""
        with self._lock:
            return self.memory_indexer.build_from(observations, labels=labels)

    def search_index(self, query: np.ndarray, top_k: int = 5) -> dict:
        """Top-k search via the normalized matrix-multiply index."""
        with self._lock:
            return self.memory_indexer.search(query, top_k=top_k)

    # ------------------------------------------------------------------
    # Phase 6 — Planning domain facades
    # ------------------------------------------------------------------

    def plan_trajectory(
        self,
        start_state: np.ndarray,
        goal_state: np.ndarray,
        obstacles: np.ndarray | None = None,
        n_steps: int = 32,
        margin: float | None = None,
    ) -> dict:
        """Plan a damped least-action trajectory with per-step actions."""
        with self._lock:
            return self.planning_trajectory.plan(
                start_state, goal_state, obstacles=obstacles,
                n_steps=n_steps, margin=margin,
            )

    def decompose_goal(
        self, goal: str, max_depth: int | None = None
    ) -> dict:
        """Decompose an abstract goal into an AND/OR tree of actions."""
        with self._lock:
            return self.planning_goal_decomposer.decompose(
                goal, max_depth=max_depth
            )

    def sequence_actions(
        self, adjacency: np.ndarray, labels: list | None = None
    ) -> dict:
        """Topologically sort a DAG of actions (greedy cycle breaking)."""
        with self._lock:
            return self.planning_action_sequencer.sequence(
                adjacency, labels=labels
            )

    def plan_hierarchical(
        self,
        observation: np.ndarray,
        var_names: list[str] | None = None,
        goal_vars: list[int] | None = None,
    ) -> dict:
        """Discover causal DAG + topologically sort into an action sequence."""
        with self._lock:
            return self.planning_hierarchical.plan_hierarchy(
                observation, var_names=var_names, goal_vars=goal_vars
            )

    def search_mcts(
        self,
        root_state,
        n_simulations: int | None = None,
        max_depth: int | None = None,
    ) -> dict:
        """Run a UCT MCTS search over a planning model."""
        with self._lock:
            return self.planning_mcts.search(
                root_state,
                n_simulations=n_simulations,
                max_depth=max_depth,
            )

    def plan_symbolic(
        self,
        initial_state: dict,
        goal_state: dict,
        max_expansions: int = 1000,
    ) -> dict:
        """A* STRIPS-style plan from ``initial_state`` to ``goal_state``."""
        with self._lock:
            return self.planning_symbolic.plan(
                initial_state, goal_state, max_expansions=max_expansions
            )

    def update_policy_gradient(
        self,
        state: int,
        action: int,
        return_value: float,
        baseline: float = 0.0,
    ) -> dict:
        """REINFORCE-style update of the policy-gradient planner."""
        with self._lock:
            return self.planning_policy_gradient.update(
                state, action, return_value, baseline=baseline
            )

    def select_policy_gradient_action(self, state) -> dict:
        """Sample an action from the policy-gradient planner's softmax."""
        with self._lock:
            return self.planning_policy_gradient.select_action(state)

    def add_contingency_plan(
        self, name: str, actions: list, precondition_fn=None
    ) -> dict:
        """Register a named plan with a precondition predicate."""
        with self._lock:
            return self.planning_contingency.add_plan(
                name, actions, precondition_fn=precondition_fn
            )

    def contingency_fallback(self, state=None) -> dict:
        """Pick the first registered plan whose precondition matches."""
        with self._lock:
            return self.planning_contingency.fallback(state)

    # ------------------------------------------------------------------
    # Phase 6 — Multimodal domain facades
    # ------------------------------------------------------------------

    def fit_cross_modal(
        self, observations_a: np.ndarray, observations_b: np.ndarray
    ) -> dict:
        """Fit CCA on paired observations across two modalities."""
        with self._lock:
            return self.multimodal_aligner.fit(
                observations_a, observations_b
            )

    def align_cross_modal(
        self, observations: np.ndarray, source: str = "a"
    ) -> dict:
        """Project ``observations`` into the aligned cross-modal space."""
        with self._lock:
            return self.multimodal_aligner.align(observations, source=source)

    def update_shared_space(
        self, x: np.ndarray, y: np.ndarray
    ) -> dict:
        """Online CCA update of the shared latent space."""
        with self._lock:
            return self.multimodal_shared_space.update(x, y)

    def project_to_shared(self, x: np.ndarray) -> np.ndarray:
        """Project ``x`` into the shared latent space."""
        with self._lock:
            return self.multimodal_shared_space.project(x)

    def fuse_modalities(
        self, embeddings: list[np.ndarray], strategy: str | None = None
    ) -> dict:
        """Fuse multiple modality embeddings (mean/concat/weighted)."""
        with self._lock:
            return self.multimodal_fuser.fuse(embeddings, strategy=strategy)

    def fit_modality_encoder(
        self, sequences: list, labels: list | None = None
    ) -> dict:
        """Fit the modality encoder vocabulary / IDF / PCA basis."""
        with self._lock:
            return self.multimodal_encoder.fit(sequences, labels=labels)

    def encode_modality(self, sequence) -> np.ndarray:
        """Encode a raw modality sequence into a fixed-dim vector."""
        with self._lock:
            return self.multimodal_encoder.encode(sequence)

    def fuse_with_attention(
        self, queries: np.ndarray, keys: np.ndarray, values: np.ndarray
    ) -> dict:
        """Multi-head scaled-dot-product attention fusion."""
        with self._lock:
            return self.multimodal_attention_fuser.fuse(queries, keys, values)

    def contrastive_loss(
        self, batch_a: np.ndarray, batch_b: np.ndarray
    ) -> dict:
        """InfoNCE contrastive alignment loss between paired batches."""
        with self._lock:
            return self.multimodal_contrastive.loss(batch_a, batch_b)

    def build_multimodal_index(
        self, items: np.ndarray, labels: list | None = None
    ) -> dict:
        """Index a corpus of modality-b items for cross-modal retrieval."""
        with self._lock:
            return self.multimodal_retriever.build(items, labels=labels)

    def query_multimodal(
        self, query: np.ndarray, top_k: int = 5
    ) -> dict:
        """Cross-modal top-k retrieval."""
        with self._lock:
            return self.multimodal_retriever.query(query, top_k=top_k)

    # ------------------------------------------------------------------
    # Phase 6 — RL domain facades
    # ------------------------------------------------------------------

    def step_mdp(self, state: int, action: int) -> dict:
        """Take one step in the synthetic MDP."""
        with self._lock:
            return self.rl_mdp.step(state, action)

    def reset_mdp(self) -> dict:
        """Reset the synthetic MDP to a random non-terminal state."""
        with self._lock:
            return self.rl_mdp.reset()

    def update_q(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
    ) -> dict:
        """One tabular Q-learning TD update."""
        with self._lock:
            return self.rl_q_learner.update(
                state, action, reward, next_state
            )

    def select_q_action(
        self, state: int, epsilon: float | None = None
    ) -> dict:
        """Epsilon-greedy action selection from the Q-table."""
        with self._lock:
            return self.rl_q_learner.select_action(state, epsilon=epsilon)

    def train_q_learner(
        self, n_episodes: int = 100, max_steps_per_episode: int = 100
    ) -> dict:
        """Full tabular Q-learning training loop."""
        with self._lock:
            return self.rl_q_learner.train(
                n_episodes=n_episodes,
                max_steps_per_episode=max_steps_per_episode,
            )

    def evaluate_q_learner(
        self, n_episodes: int = 10, max_steps_per_episode: int = 100
    ) -> dict:
        """Evaluate the greedy Q-policy without updating Q."""
        with self._lock:
            return self.rl_q_learner.evaluate(
                n_episodes=n_episodes,
                max_steps_per_episode=max_steps_per_episode,
            )

    def train_policy_optimizer(
        self, n_episodes: int = 100, max_steps_per_episode: int = 100
    ) -> dict:
        """REINFORCE-style policy-gradient training loop."""
        with self._lock:
            return self.rl_policy_optimizer.train(
                n_episodes=n_episodes,
                max_steps_per_episode=max_steps_per_episode,
            )

    def value_td0_update(
        self,
        state: int,
        reward: float,
        next_state: int,
        done: bool = False,
    ) -> dict:
        """TD(0) update of the tabular value function."""
        with self._lock:
            return self.rl_value_function.td0_update(
                state, reward, next_state, done=done
            )

    def value_td_lambda_update(
        self, trajectory: list, lambda_: float = 0.9
    ) -> dict:
        """TD(λ) update of the tabular value function over a trajectory."""
        with self._lock:
            return self.rl_value_function.td_lambda_update(
                trajectory, lambda_=lambda_
            )

    def value_monte_carlo_update(self, trajectory: list) -> dict:
        """First-visit MC update of the tabular value function."""
        with self._lock:
            return self.rl_value_function.monte_carlo_update(trajectory)

    def train_dyna_q(
        self, n_episodes: int = 100, max_steps_per_episode: int = 100
    ) -> dict:
        """Dyna-Q training (Q-learning + model + imaginary rollouts)."""
        with self._lock:
            return self.rl_dyna_q.train(
                n_episodes=n_episodes,
                max_steps_per_episode=max_steps_per_episode,
            )

    def search_rl_mcts(
        self,
        root_state: int,
        n_simulations: int | None = None,
        max_depth: int | None = None,
    ) -> dict:
        """UCT MCTS over the known synthetic MDP."""
        with self._lock:
            return self.rl_mcts.search(
                root_state,
                n_simulations=n_simulations,
                max_depth=max_depth,
            )

    def update_rl_posterior(
        self, state: int, action: int, reward: float
    ) -> dict:
        """Bayesian update of the PSRL reward posterior."""
        with self._lock:
            return self.rl_posterior.update(state, action, reward)

    def sample_rl_posterior(self) -> dict:
        """Sample a reward matrix from the PSRL posterior."""
        with self._lock:
            return self.rl_posterior.sample_reward()

    def value_iterate_rl(
        self, reward: np.ndarray, n_iters: int = 50
    ) -> dict:
        """Value iteration on the MDP with a sampled reward."""
        with self._lock:
            return self.rl_posterior.value_iteration(reward, n_iters=n_iters)

    def train_posterior_sampling(
        self, n_episodes: int = 50, max_steps_per_episode: int = 100
    ) -> dict:
        """PSRL training loop (sample reward → value iteration → act)."""
        with self._lock:
            return self.rl_posterior.train(
                n_episodes=n_episodes,
                max_steps_per_episode=max_steps_per_episode,
            )
