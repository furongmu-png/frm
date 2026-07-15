# tests/test_model.py
import numpy as np

from zero_data_model.model import ZeroDataModel


def test_model_creation():
    model = ZeroDataModel(dim=16)
    assert len(model.modules) == 6


def test_model_think_no_input():
    model = ZeroDataModel(dim=16)
    result = model.think()
    assert result.data.shape == (16,)
    assert result.metadata["cycle"] == 1


def test_model_think_with_input():
    # The autouse conftest fixture re-seeds np.random to 42 before every
    # test, so np.random.randn(16) here is deterministic.
    model = ZeroDataModel(dim=16)
    result = model.think(np.random.randn(16))
    assert result.data.shape == (16,)
    assert np.all(np.isfinite(result.data))


def test_model_solve():
    model = ZeroDataModel(dim=16)
    result = model.solve(np.random.randn(16))
    assert "energy" in result.metadata
    # Strengthened: the solution vector must be dim-length and finite, and
    # the reported annealer energy must be a finite scalar.
    assert result.data.shape == (16,)
    assert np.all(np.isfinite(result.data))
    assert np.isfinite(result.metadata["energy"])


def test_model_find_analogies():
    model = ZeroDataModel(dim=16)
    a = np.random.randn(16)
    score = model.find_analogies(a, a)
    assert abs(score - 1.0) < 1e-6


def test_model_find_analogies_divergent_inputs_score_lower():
    """Divergent inputs produce a lower isomorphism score than identical
    inputs. This guards against a regression where find_analogies ignored
    its arguments and always returned 1.0.

    find_analogies delegates to find_isomorphism which returns a cosine
    similarity in [-1, 1]; identical vectors score ~1.0, divergent vectors
    score strictly lower (and may go negative)."""
    model = ZeroDataModel(dim=16)
    a = np.random.randn(16)
    # Orthogonal, scaled-up vector: structurally very different from ``a``.
    b = np.random.randn(16) + 10.0
    identical = model.find_analogies(a, a)
    divergent = model.find_analogies(a, b)
    assert -1.0 <= divergent <= 1.0
    assert divergent < identical


def test_model_generate_knowledge():
    model = ZeroDataModel(dim=16)
    knowledge = model.generate_knowledge()
    assert knowledge.data.shape[0] > 0


def test_model_multiple_cycles():
    model = ZeroDataModel(dim=16)
    for i in range(5):
        result = model.think()
        assert result.metadata["cycle"] == i + 1


# --- Domain capability integration tests ---

def test_model_encode_text():
    model = ZeroDataModel(dim=16)
    vec = model.encode_text("code data model")
    assert vec.shape == (16,)
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-3


def test_model_text_similarity():
    model = ZeroDataModel(dim=16)
    s = model.text_similarity("code data model", "code data model")
    assert s > 0.9


def test_model_classify_text():
    model = ZeroDataModel(dim=16)
    topic, conf = model.classify_text("the algorithm computes the network")
    assert topic in {"tech", "nature", "emotion", "science"}
    assert 0.0 <= conf <= 1.0


def test_model_generate_text():
    model = ZeroDataModel(dim=16)
    out = model.generate_text("seed", 32)
    assert len(out) == 32
    assert all(32 <= ord(c) <= 126 for c in out)


def test_model_encode_image():
    model = ZeroDataModel(dim=16)
    img = np.random.rand(8, 8)
    vec = model.encode_image(img)
    assert vec.shape == (16,)
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-3


def test_model_extract_image_features():
    model = ZeroDataModel(dim=16)
    feats = model.extract_image_features(np.random.rand(8, 8))
    assert set(feats.keys()) == {"edges", "texture", "morphology", "stats"}
    for v in feats.values():
        assert isinstance(v, np.ndarray) and v.ndim == 1


def test_model_recognize_pattern():
    model = ZeroDataModel(dim=16)
    shape, conf = model.recognize_pattern(np.random.rand(8, 8))
    assert shape in {"circle", "square", "triangle", "line", "blob"}
    assert 0.0 <= conf <= 1.0


def test_model_analyze_shape():
    model = ZeroDataModel(dim=16)
    out = model.analyze_shape(np.ones((10, 10)))
    assert set(out.keys()) == {"aspect_ratio", "symmetry", "complexity", "beti0"}


def test_model_forecast():
    model = ZeroDataModel(dim=16)
    series = np.arange(20, dtype=float)
    pred = model.forecast(series, horizon=5)
    assert pred.shape == (5,)
    assert np.all(np.isfinite(pred))


def test_model_detect_anomalies():
    model = ZeroDataModel(dim=16)
    series = np.array([1, 1, 1, 1, 100, 1, 1, 1, 1], dtype=float)
    mask = model.detect_anomalies(series)
    assert mask.dtype == bool
    assert mask.shape == (9,)
    assert mask.any()


def test_model_mine_patterns():
    model = ZeroDataModel(dim=16)
    out = model.mine_patterns(np.arange(20, dtype=float))
    assert set(out.keys()) == {"self_similarity", "topology", "automaton_rule", "periodicity"}


def test_model_analyze_trend():
    model = ZeroDataModel(dim=16)
    out = model.analyze_trend(np.arange(50, dtype=float))
    assert out["regime"] == "up"
    assert set(out.keys()) == {
        "trend_slope",
        "regime",
        "curvature",
        "geodesic_deviation",
        "isomorphism_score",
    }
