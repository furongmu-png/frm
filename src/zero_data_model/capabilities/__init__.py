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
]
