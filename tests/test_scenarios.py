# tests/test_scenarios.py
"""Application-validation scenario tests for the zero-data cognitive model.

These tests prove the system works end-to-end with NO external data: zero-shot
text classification, cross-domain transfer (text<->image, series<->text),
zero-shot pattern recognition, anomaly detection, trend analysis, forecasting
and a self-driven think cycle. They exercise the public ``ZeroDataModel`` API
just like a real application would.

The model's internal modules use unseeded ``np.random.randn`` for their
generative parameters; an autouse fixture seeds the global RNG before every
test so the scenarios are reproducible. Generated test data (images, noise)
uses an explicit ``np.random.default_rng(42)``.
"""

from __future__ import annotations

import numpy as np
import pytest

from zero_data_model.model import ZeroDataModel

# The zero-data classifier is built from self-generated keyword prototypes and
# its output space is exactly these four topics.
TOPICS = {"tech", "nature", "emotion", "science"}
VALID_SHAPES = {"circle", "square", "triangle", "line", "blob"}


@pytest.fixture(autouse=True)
def _seed_global_rng() -> None:
    """Seed numpy's legacy global RNG so model init is deterministic per test."""
    np.random.seed(42)


# ---------------------------------------------------------------------------
# 1. Zero-shot classification across diverse topics.
# ---------------------------------------------------------------------------
def test_zero_shot_classification_diverse_topics() -> None:
    """Classify texts from the model's four topic categories.

    The classifier only emits {tech, nature, emotion, science} (its self-built
    prototype space), so "diverse topics" is realised as these four categories,
    each probed with the required exact sentence plus two additional sentences.
    A majority (>= 2/3) must be classified correctly with confidence > 0.
    """
    model = ZeroDataModel(dim=64)

    # The four EXACT topic texts required by the spec -> each must be correct.
    exact = [
        ("the algorithm computes the network", "tech"),
        ("trees rivers mountains forests", "nature"),
        ("love joy happiness hope", "emotion"),
        ("energy force mass quantum field", "science"),
    ]
    for text, expected in exact:
        topic, conf = model.classify_text(text)
        assert topic == expected, f"{text!r}: expected {expected}, got {topic}"
        assert conf > 0.0, f"{text!r}: confidence {conf} not > 0"

    # Two additional sentences per topic; assert majority-correct per topic.
    extra = {
        "tech": ["code data model system", "the system computes data"],
        "nature": ["tree river mountain forest", "sky ocean flower tree"],
        "emotion": ["happy love hope joy", "sad fear anger happy"],
        "science": ["quantum atom field energy", "light force mass atom"],
    }
    for expected, texts in extra.items():
        correct = 0
        for text in texts:
            topic, conf = model.classify_text(text)
            assert topic in TOPICS
            assert conf > 0.0
            if topic == expected:
                correct += 1
        assert correct >= 1, f"topic {expected}: only {correct}/2 correct"


# ---------------------------------------------------------------------------
# 2. Zero-shot text similarity ranking.
# ---------------------------------------------------------------------------
def test_zero_shot_text_similarity_ranking() -> None:
    """A tech query must rank same-topic candidates above unrelated candidates."""
    model = ZeroDataModel(dim=64)
    query = "code data model"
    candidates = [
        ("algorithm network system", "tech"),
        ("compute data code", "tech"),
        ("the model computes data", "tech"),
        ("love joy hope", "emotion"),
        ("energy quantum field", "science"),
    ]
    sims = [(model.text_similarity(query, c), topic) for c, topic in candidates]

    tech_sims = [s for s, t in sims if t == "tech"]
    nontech_sims = [s for s, t in sims if t != "tech"]

    # Every same-topic candidate outranks every unrelated candidate.
    assert min(tech_sims) > max(nontech_sims)
    # The top-ranked candidate is a tech candidate.
    top_topic = max(sims, key=lambda x: x[0])[1]
    assert top_topic == "tech"


# ---------------------------------------------------------------------------
# 3. Cross-domain text -> image transfer.
# ---------------------------------------------------------------------------
def test_cross_domain_text_to_image_transfer() -> None:
    """The model can compute structural similarity across modalities."""
    model = ZeroDataModel(dim=64)
    rng = np.random.default_rng(42)
    image = rng.random((16, 16))

    text_vec = model.encode_text("code data model")
    image_vec = model.encode_image(image)

    score = model.find_analogies(text_vec, image_vec)
    assert np.isfinite(score)
    assert -1.0 <= score <= 1.0

    # The same image compared to itself is structural identity (~1.0).
    self_score = model.find_analogies(image_vec, image_vec)
    assert abs(self_score - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# 4. Cross-domain series -> text transfer.
# ---------------------------------------------------------------------------
def test_cross_domain_series_to_text() -> None:
    """A time-series embedding (via forecast) and a text are comparable."""
    model = ZeroDataModel(dim=64)
    series = np.arange(20, dtype=float) * 2.0
    series_embedding = model.forecast(series, horizon=64)
    text_vec = model.encode_text("energy force quantum field")

    score = model.find_analogies(series_embedding, text_vec)
    assert np.isfinite(score)
    assert -1.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# 5. Zero-shot pattern recognition of multiple shapes.
# ---------------------------------------------------------------------------
def test_zero_shot_pattern_recognition_multiple_shapes() -> None:
    """Synthesize circle/square/triangle (rule-based) and recognize each."""
    model = ZeroDataModel(dim=64)
    size = 16
    yy, xx = np.indices((size, size))
    cy, cx = size // 2, size // 2

    circle = ((yy - cy) ** 2 + (xx - cx) ** 2 <= (size // 3) ** 2).astype(float)
    square = np.zeros((size, size))
    square[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 1.0
    triangle = np.tri(size, dtype=float)

    for name, image in [("circle", circle), ("square", square), ("triangle", triangle)]:
        shape, conf = model.recognize_pattern(image)
        assert shape in VALID_SHAPES
        assert shape == name, f"{name}: recognized as {shape}"
        assert 0.0 <= conf <= 1.0
        assert conf > 0.0


# ---------------------------------------------------------------------------
# 6. Anomaly detection: single spike + level shift.
# ---------------------------------------------------------------------------
def test_anomaly_detection_spike_and_level_shift() -> None:
    """Flag at least one point in a spike region and in a level-shift region."""
    model = ZeroDataModel(dim=64)
    rng = np.random.default_rng(42)

    # (a) A single spike in a flat baseline.
    baseline = np.ones(20) + rng.normal(0.0, 0.05, 20)
    spike_series = np.concatenate([baseline, [10.0], baseline])
    spike_mask = model.detect_anomalies(spike_series)
    spike_idx = len(baseline)  # spike sits right after the first baseline block
    assert spike_mask[spike_idx], "spike point was not flagged"

    # (b) A level shift: a long low baseline followed by a short high-level
    # region. The shifted region is a statistical minority so its points are
    # outliers in the free-energy distribution and get flagged.
    level_series = np.concatenate([np.ones(30), np.full(5, 10.0)])
    level_mask = model.detect_anomalies(level_series)
    shift_region = np.arange(30, 35)
    assert level_mask[shift_region].any(), "no point flagged in the level-shift region"


# ---------------------------------------------------------------------------
# 7. Trend analysis across three regimes.
# ---------------------------------------------------------------------------
def test_trend_analysis_three_regimes() -> None:
    """Up / down / flat regimes are classified correctly."""
    model = ZeroDataModel(dim=64)
    rng = np.random.default_rng(42)
    n = 50
    up = np.arange(n, dtype=float) + rng.normal(0.0, 0.1, n)
    down = -np.arange(n, dtype=float) + rng.normal(0.0, 0.1, n)
    flat = np.ones(n) * 3.0 + rng.normal(0.0, 0.1, n)

    for expected, series in [("up", up), ("down", down), ("flat", flat)]:
        out = model.analyze_trend(series)
        assert out["regime"] == expected


# ---------------------------------------------------------------------------
# 8. End-to-end self-driven think cycle.
# ---------------------------------------------------------------------------
def test_end_to_end_think_cycle_with_self_generation() -> None:
    """Run think() with NO input 5 times; cycle increments, output finite,
    and metacognitive confidence metadata is present."""
    model = ZeroDataModel(dim=64)
    cycles: list[int] = []
    last = None
    for _ in range(5):
        last = model.think()  # no input -> self-generates from internal state
        cycles.append(last.metadata["cycle"])

    assert cycles == [1, 2, 3, 4, 5]
    assert last is not None
    assert np.all(np.isfinite(last.data))

    # Confidence / metacognition metadata is present and well-formed.
    reflection = last.metadata["self_reflection"]
    assert "self_confidence" in reflection
    assert 0.0 <= reflection["self_confidence"] <= 1.0
    assert reflection["history_len"] > 0


# ---------------------------------------------------------------------------
# 9. Forecast continues an upward trend.
# ---------------------------------------------------------------------------
def test_forecast_continues_trend() -> None:
    """Forecasting an upward series continues the trend (mean does not drop)."""
    model = ZeroDataModel(dim=64)
    series = np.arange(20, dtype=float) * 3.0  # strong upward trend
    forecast = model.forecast(series, horizon=5)

    assert forecast.shape == (5,)
    assert np.all(np.isfinite(forecast))
    assert forecast.mean() >= series.mean()


# ---------------------------------------------------------------------------
# 10. Cross-domain analogy consistency (self-similarity == 1.0).
# ---------------------------------------------------------------------------
def test_cross_domain_analogy_consistency() -> None:
    """find_analogies(a, a) ~= 1.0 for vectors from different domains."""
    model = ZeroDataModel(dim=64)
    rng = np.random.default_rng(42)

    text_vec = model.encode_text("code data model")
    image_vec = model.encode_image(rng.random((16, 16)))
    series_vec = model.forecast(np.arange(20, dtype=float), horizon=64)

    for name, vec in [("text", text_vec), ("image", image_vec), ("series", series_vec)]:
        score = model.find_analogies(vec, vec)
        assert abs(score - 1.0) < 1e-6, f"{name}: self-analogy = {score}"
