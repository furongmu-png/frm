from __future__ import annotations

from .analytics import (
    AnomalyDetector,
    PatternMiner,
    TimeSeriesForecaster,
    TrendAnalyzer,
)
from .analytics_advanced import (
    BayesianEstimator,
    CausalInference,
    ChangePointDetector,
)
from .nlp import (
    SemanticComparator,
    TextEncoder,
    TextGenerator,
    ZeroShotClassifier,
)
from .nlp_advanced import (
    MultiLingualEncoder,
    SentenceEncoder,
    SyntacticAnalyzer,
)
from .rules import (
    AnalyticsRules,
    AudioRules,
    CausalRules,
    CodeRules,
    DomainRules,
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
from .vision import (
    FeatureExtractor,
    ImageEncoder,
    PatternRecognizer,
    ShapeAnalyzer,
)
from .vision_advanced import (
    DepthEstimator,
    PointCloudEncoder,
    VideoFrameAnalyzer,
)

# Phase 6 capability modules (optional import guards are reflected by
# re-importing here; if a module is missing the name is absent at runtime).
from .memory import (
    ContextMemory,
    EpisodicMemory,
    MemoryConsolidator,
    WorkingMemory,
)
from .memory_advanced import (
    ForgetfulMemory,
    HierarchicalMemory,
    MemoryIndexer,
    SpreadingActivationMemory,
)
from .planning import (
    ActionSequencer,
    GoalDecomposer,
    HierarchicalPlanner,
    TrajectoryPlanner,
)
from .planning_advanced import (
    ContingencyPlanner,
    MonteCarloTreePlanner,
    PolicyGradientPlanner,
    SymbolicPlanner,
)
from .multimodal import (
    CrossModalAligner,
    ModalityEncoder,
    ModalityFuser,
    SharedLatentSpace,
)
from .multimodal_advanced import (
    AttentionBasedFuser,
    ContrastiveAligner,
    MultimodalRetriever,
)
from .rl import (
    PolicyOptimizer,
    QLearner,
    SyntheticMDP,
    ValueFunction,
)
from .rl_advanced import (
    DynaQ,
    MonteCarloTreeSearch,
    PosteriorSampling,
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
    "MemoryRules",
    "PlanningRules",
    "MultimodalRules",
    "RLRules",
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
    # Phase 6 Memory capabilities.
    "EpisodicMemory",
    "WorkingMemory",
    "ContextMemory",
    "MemoryConsolidator",
    "HierarchicalMemory",
    "SpreadingActivationMemory",
    "ForgetfulMemory",
    "MemoryIndexer",
    # Phase 6 Planning capabilities.
    "HierarchicalPlanner",
    "GoalDecomposer",
    "TrajectoryPlanner",
    "ActionSequencer",
    "MonteCarloTreePlanner",
    "SymbolicPlanner",
    "PolicyGradientPlanner",
    "ContingencyPlanner",
    # Phase 6 Multimodal capabilities.
    "CrossModalAligner",
    "SharedLatentSpace",
    "ModalityFuser",
    "ModalityEncoder",
    "AttentionBasedFuser",
    "ContrastiveAligner",
    "MultimodalRetriever",
    # Phase 6 RL capabilities.
    "SyntheticMDP",
    "QLearner",
    "PolicyOptimizer",
    "ValueFunction",
    "DynaQ",
    "MonteCarloTreeSearch",
    "PosteriorSampling",
]
