"""Domain capabilities built on top of the zero-data core modules."""

import contextlib

from .rules import (
    AnalyticsRules,
    AudioRules,
    DomainRules,
    GraphRules,
    NLPRules,
    RoboticsRules,
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

__all__ = [
    "DomainRules",
    "NLPRules",
    "VisionRules",
    "AnalyticsRules",
    "AudioRules",
    "GraphRules",
    "RoboticsRules",
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
]
