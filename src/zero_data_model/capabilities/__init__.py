"""Domain capabilities built on top of the zero-data core modules."""

from .rules import DomainRules, NLPRules, VisionRules, AnalyticsRules
from .vision import (
    ImageEncoder,
    FeatureExtractor,
    PatternRecognizer,
    ShapeAnalyzer,
)

# Sibling capability modules (NLP, Analytics) are not implemented yet.
# Guard their imports so this package stays importable; the names below are
# only exported once the corresponding modules exist.
try:  # pragma: no cover - optional sibling module
    from .nlp import (
        TextEncoder,
        SemanticComparator,
        ZeroShotClassifier,
        TextGenerator,
    )
except ImportError:
    pass

try:  # pragma: no cover - optional sibling module
    from .analytics import (
        TimeSeriesForecaster,
        AnomalyDetector,
        PatternMiner,
        TrendAnalyzer,
    )
except ImportError:
    pass

# Advanced capability modules (multilingual NLP, point clouds / video / depth,
# causal inference / Bayesian updating / change-point detection). Same optional
# import guard as the base modules so the package stays importable if a sibling
# advanced module is removed.
try:  # pragma: no cover - optional sibling module
    from .nlp_advanced import (
        MultiLingualEncoder,
        SyntacticAnalyzer,
        SentenceEncoder,
    )
except ImportError:
    pass

try:  # pragma: no cover - optional sibling module
    from .vision_advanced import (
        PointCloudEncoder,
        VideoFrameAnalyzer,
        DepthEstimator,
    )
except ImportError:
    pass

try:  # pragma: no cover - optional sibling module
    from .analytics_advanced import (
        CausalInference,
        BayesianEstimator,
        ChangePointDetector,
    )
except ImportError:
    pass

__all__ = [
    "DomainRules",
    "NLPRules",
    "VisionRules",
    "AnalyticsRules",
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
]
