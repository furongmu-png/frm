"""Domain capabilities built on top of the zero-data core modules."""

import contextlib

from .rules import (
    AnalyticsRules,
    AudioRules,
    CausalRules,
    CodeRules,
    DomainRules,
    GraphRules,
    NLPRules,
    ReasoningRules,
    RoboticsRules,
    TimeRules,
    VisionRules,
)
from .vision import (
    FeatureExtractor,
    ImageEncoder,
    PatternRecognizer,
    ShapeAnalyzer,
)

# Sibling capability modules (NLP, Analytics) are not implemented yet.
# Guard their imports so this package stays importable; the names below are
# only exported once the corresponding modules exist.
with contextlib.suppress(ImportError):  # pragma: no cover - optional sibling module
    from .nlp import (
        SemanticComparator,
        TextEncoder,
        TextGenerator,
        ZeroShotClassifier,
    )

with contextlib.suppress(ImportError):  # pragma: no cover - optional sibling module
    from .analytics import (
        AnomalyDetector,
        PatternMiner,
        TimeSeriesForecaster,
        TrendAnalyzer,
    )

# Advanced capability modules (multilingual NLP, point clouds / video / depth,
# causal inference / Bayesian updating / change-point detection). Same optional
# import guard as the base modules so the package stays importable if a sibling
# advanced module is removed.
with contextlib.suppress(ImportError):  # pragma: no cover - optional sibling module
    from .nlp_advanced import (
        MultiLingualEncoder,
        SentenceEncoder,
        SyntacticAnalyzer,
    )

with contextlib.suppress(ImportError):  # pragma: no cover - optional sibling module
    from .vision_advanced import (
        DepthEstimator,
        PointCloudEncoder,
        VideoFrameAnalyzer,
    )

with contextlib.suppress(ImportError):  # pragma: no cover - optional sibling module
    from .analytics_advanced import (
        BayesianEstimator,
        CausalInference,
        ChangePointDetector,
    )

# Audio capability modules (STFT encoding, onset / pitch detection, audio
# texture classification, speech segmentation, music analysis, speaker
# feature extraction). Same optional import guard.
with contextlib.suppress(ImportError):  # pragma: no cover - optional sibling module
    from .audio import (
        AudioClassifier,
        AudioEncoder,
        OnsetDetector,
        PitchDetector,
    )

with contextlib.suppress(ImportError):  # pragma: no cover - optional sibling module
    from .audio_advanced import (
        MusicAnalyzer,
        SpeakerRecognizer,
        SpeechSegmenter,
    )

# Graph capability modules (encoding, community detection, path finding,
# centrality, isomorphism, dynamic tracking, spanning tree).
with contextlib.suppress(ImportError):  # pragma: no cover - optional sibling module
    from .graph import (
        CentralityAnalyzer,
        CommunityDetector,
        GraphEncoder,
        PathFinder,
    )

with contextlib.suppress(ImportError):  # pragma: no cover - optional sibling module
    from .graph_advanced import (
        DynamicGraphTracker,
        GraphIsomorphismDetector,
        SpanningTreeExtractor,
    )

# Robotics capability modules (motion planning, kinematics, sensor fusion,
# gait generation, trajectory optimization, collision checking, MPC).
with contextlib.suppress(ImportError):  # pragma: no cover - optional sibling module
    from .robotics import (
        GaitGenerator,
        KinematicsSolver,
        MotionPlanner,
        SensorFuser,
    )

with contextlib.suppress(ImportError):  # pragma: no cover - optional sibling module
    from .robotics_advanced import (
        CollisionChecker,
        MPCController,
        TrajectoryOptimizer,
    )

# Time-series capability modules (encoding, seasonality, frequency, events,
# anomaly timing, cycle phase, forecastability).
with contextlib.suppress(ImportError):  # pragma: no cover - optional sibling module
    from .time import (
        EventTimestampAnalyzer,
        FrequencyAnalyzer,
        SeasonalityDetector,
        TimeSeriesEncoder,
    )

with contextlib.suppress(ImportError):  # pragma: no cover - optional sibling module
    from .time_advanced import (
        AnomalyTimingDetector,
        CyclePhaseTracker,
        ForecastabilityScorer,
    )

# Code-analysis capability modules (encoding, AST, similarity, defects,
# control flow, style, dependency graph).
with contextlib.suppress(ImportError):  # pragma: no cover - optional sibling module
    from .code import (
        ASTAnalyzer,
        CodeEncoder,
        CodeSimilarityChecker,
        DefectPatternDetector,
    )

with contextlib.suppress(ImportError):  # pragma: no cover - optional sibling module
    from .code_advanced import (
        CodeStyleAnalyzer,
        ControlFlowAnalyzer,
        DependencyGraphBuilder,
    )

# Reasoning capability modules (propositional logic, deduction, induction,
# analogy, abduction, defeasible reasoning, causal chains).
with contextlib.suppress(ImportError):  # pragma: no cover - optional sibling module
    from .reasoning import (
        AnalogicalReasoner,
        DeductiveReasoner,
        InductiveReasoner,
        PropositionalLogicEngine,
    )

with contextlib.suppress(ImportError):  # pragma: no cover - optional sibling module
    from .reasoning_advanced import (
        AbductiveReasoner,
        CausalChainReasoner,
        DefeasibleReasoner,
    )

# Causal/Decision capability modules (decision trees, game theory,
# counterfactual reasoning, bandits, POMDP, causal graph discovery,
# intervention analysis).
with contextlib.suppress(ImportError):  # pragma: no cover - optional sibling module
    from .causal import (
        CounterfactualReasoner,
        DecisionTreeBuilder,
        GameTheoryAnalyzer,
        MultiArmedBandit,
    )

with contextlib.suppress(ImportError):  # pragma: no cover - optional sibling module
    from .causal_advanced import (
        CausalGraphBuilder,
        InterventionAnalyzer,
        POMDPApproximator,
    )

__all__ = [
    "DomainRules",
    "NLPRules",
    "VisionRules",
    "AnalyticsRules",
    "AudioRules",
    "GraphRules",
    "RoboticsRules",
    "TimeRules",
    "CodeRules",
    "ReasoningRules",
    "CausalRules",
    "TextEncoder",
    "SemanticComparator",
    "ZeroShotClassifier",
    "TextGenerator",
    "ImageEncoder",
    "FeatureExtractor",
    "PatternRecognizer",
    "ShapeAnalyzer",
    "TimeSeriesForecaster",
    "AnomalyDetector",
    "PatternMiner",
    "TrendAnalyzer",
    # Advanced NLP capabilities.
    "MultiLingualEncoder",
    "SyntacticAnalyzer",
    "SentenceEncoder",
    # Advanced CV capabilities.
    "PointCloudEncoder",
    "VideoFrameAnalyzer",
    "DepthEstimator",
    # Advanced Analytics capabilities.
    "CausalInference",
    "BayesianEstimator",
    "ChangePointDetector",
    # Audio capabilities.
    "AudioEncoder",
    "OnsetDetector",
    "PitchDetector",
    "AudioClassifier",
    "MusicAnalyzer",
    "SpeechSegmenter",
    "SpeakerRecognizer",
    # Graph capabilities.
    "GraphEncoder",
    "CommunityDetector",
    "PathFinder",
    "CentralityAnalyzer",
    "GraphIsomorphismDetector",
    "DynamicGraphTracker",
    "SpanningTreeExtractor",
    # Robotics capabilities.
    "MotionPlanner",
    "KinematicsSolver",
    "SensorFuser",
    "GaitGenerator",
    "TrajectoryOptimizer",
    "CollisionChecker",
    "MPCController",
    # Time capabilities.
    "TimeSeriesEncoder",
    "SeasonalityDetector",
    "FrequencyAnalyzer",
    "EventTimestampAnalyzer",
    "AnomalyTimingDetector",
    "CyclePhaseTracker",
    "ForecastabilityScorer",
    # Code capabilities.
    "CodeEncoder",
    "ASTAnalyzer",
    "CodeSimilarityChecker",
    "DefectPatternDetector",
    "ControlFlowAnalyzer",
    "CodeStyleAnalyzer",
    "DependencyGraphBuilder",
    # Reasoning capabilities.
    "PropositionalLogicEngine",
    "DeductiveReasoner",
    "InductiveReasoner",
    "AnalogicalReasoner",
    "AbductiveReasoner",
    "DefeasibleReasoner",
    "CausalChainReasoner",
    # Causal/Decision capabilities.
    "DecisionTreeBuilder",
    "GameTheoryAnalyzer",
    "CounterfactualReasoner",
    "MultiArmedBandit",
    "POMDPApproximator",
    "CausalGraphBuilder",
    "InterventionAnalyzer",
]
