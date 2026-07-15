# tests/test_capabilities_rules.py
"""Direct tests for the domain rule libraries in ``capabilities.rules``.

The whole ``rules`` module previously had zero direct tests; the rule
libraries (``NLPRules``, ``VisionRules``, ``AnalyticsRules``) were only
exercised transitively through the higher-level capability classes. These
tests pin the documented behaviour of each rule operator directly.
"""

from __future__ import annotations

import numpy as np

from zero_data_model.capabilities.rules import (
    AnalyticsRules,
    DomainRules,
    NLPRules,
    VisionRules,
)

# --------------------------------------------------------------------------- #
# DomainRules base class
# --------------------------------------------------------------------------- #


def test_domain_rules_get_returns_value_when_present():
    """get(name) returns the stored value for a known rule name."""
    rules = NLPRules()
    # ``__post_init__`` populates ``rules`` with the stopword set.
    assert rules.get("stopwords") is rules.stopwords


def test_domain_rules_get_returns_default_when_absent():
    """get(name, default) returns the default for an unknown rule name."""
    rules = NLPRules()
    sentinel = object()
    assert rules.get("does_not_exist", sentinel) is sentinel
    # And ``None`` when no default is supplied.
    assert rules.get("does_not_exist") is None


def test_domain_rules_subclasses_share_get_protocol():
    """All rule subclasses expose the same ``get(name, default)`` protocol."""
    for cls in (NLPRules, VisionRules, AnalyticsRules):
        rules = cls()
        assert isinstance(rules, DomainRules)
        assert callable(rules.get)


# --------------------------------------------------------------------------- #
# NLPRules
# --------------------------------------------------------------------------- #


def test_nlp_rules_tokenize_lowercases_and_strips_punctuation():
    """tokenize lowercases, strips punctuation and splits on whitespace."""
    rules = NLPRules()
    # Punctuation in NLPRules includes ',' and '!' so 'Hello, World!' becomes
    # 'hello  world ' which splits to ['hello', 'world'].
    assert rules.tokenize("Hello, World!") == ["hello", "world"]


def test_nlp_rules_tokenize_empty_string():
    """An empty string tokenizes to an empty list."""
    rules = NLPRules()
    assert rules.tokenize("") == []


def test_nlp_rules_tokenize_multiple_spaces():
    """Multiple consecutive spaces do not produce empty tokens."""
    rules = NLPRules()
    assert rules.tokenize("hello    world") == ["hello", "world"]


def test_nlp_rules_tokenize_strips_stopwords():
    """Stop-words ('the', 'is', ...) are filtered out by the tokenizer."""
    rules = NLPRules()
    tokens = rules.tokenize("the cat is on the mat")
    assert "the" not in tokens
    assert "is" not in tokens
    assert tokens == ["cat", "mat"]


def test_nlp_rules_sentiment_prior_positive_token():
    """A known-positive token yields a +1 prior."""
    rules = NLPRules()
    assert rules.sentiment_prior(["good"]) == 1.0


def test_nlp_rules_sentiment_prior_negative_token():
    """A known-negative token yields a -1 prior."""
    rules = NLPRules()
    assert rules.sentiment_prior(["bad"]) == -1.0


def test_nlp_rules_sentiment_prior_neutral_for_unknown():
    """Unknown tokens yield a 0.0 (neutral) prior."""
    rules = NLPRules()
    assert rules.sentiment_prior(["chair"]) == 0.0


def test_nlp_rules_sentiment_prior_mixed_balanced():
    """Equal positive + negative counts average to 0.0."""
    rules = NLPRules()
    assert rules.sentiment_prior(["good", "bad"]) == 0.0


# --------------------------------------------------------------------------- #
# VisionRules.convolve (now backed by scipy.ndimage.correlate)
# --------------------------------------------------------------------------- #


def test_vision_rules_convolve_identity_returns_image():
    """A 3x3 identity kernel returns the input image unchanged."""
    rules = VisionRules()
    image = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])
    identity = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    out = rules.convolve(image, identity)
    np.testing.assert_allclose(out, image, atol=1e-10)


def test_vision_rules_convolve_constant_image_identity_returns_constant():
    """A constant image filtered by the identity kernel is unchanged."""
    rules = VisionRules()
    image = np.full((5, 5), 7.0)
    identity = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    out = rules.convolve(image, identity)
    np.testing.assert_allclose(out, image, atol=1e-10)


def test_vision_rules_convolve_impulse_spreads_kernel():
    """A single-pixel impulse convolved with a kernel yields the kernel."""
    rules = VisionRules()
    image = np.zeros((5, 5))
    image[2, 2] = 1.0
    kernel = np.array([
        [1.0, 2.0, 1.0],
        [2.0, 4.0, 2.0],
        [1.0, 2.0, 1.0],
    ])
    out = rules.convolve(image, kernel)
    # The kernel pattern should land centred on (2, 2).
    np.testing.assert_allclose(out[1:4, 1:4], kernel, atol=1e-10)
    # And the surrounding ring should remain zero.
    assert out[0, 0] == 0.0
    assert out[4, 4] == 0.0


def test_vision_rules_convolve_sobel_y_detects_horizontal_edge():
    """The Sobel-Y kernel produces a signed response across a horizontal edge."""
    rules = VisionRules()
    image = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    ])
    out = rules.convolve(image, rules.sobel_y)
    # The middle row should have a strong positive response (gradient up->down).
    assert out[1, 1] > 0.0
    # The response should be finite everywhere.
    assert np.all(np.isfinite(out))


# --------------------------------------------------------------------------- #
# AnalyticsRules.moving_average
# --------------------------------------------------------------------------- #


def test_analytics_rules_moving_average_normal_case():
    """Normal case returns len(series) - w + 1 values."""
    rules = AnalyticsRules()
    series = np.arange(10, dtype=float)
    out = rules.moving_average(series, window=3)
    # 10 - 3 + 1 = 8 outputs.
    assert out.shape == (8,)
    # First window mean: (0 + 1 + 2) / 3 = 1.0
    np.testing.assert_allclose(out[0], 1.0, atol=1e-10)
    np.testing.assert_allclose(out[1], 2.0, atol=1e-10)


def test_analytics_rules_moving_average_window_one_returns_series():
    """window=1 returns the original series unchanged."""
    rules = AnalyticsRules()
    series = np.array([3.0, 5.0, 7.0, 9.0])
    out = rules.moving_average(series, window=1)
    np.testing.assert_allclose(out, series, atol=1e-10)


def test_analytics_rules_moving_average_series_shorter_than_window():
    """Series shorter than the window returns repeated overall mean."""
    rules = AnalyticsRules()
    series = np.array([1.0, 2.0, 3.0])
    out = rules.moving_average(series, window=10)
    # Falls back to mean(series) repeated for every element.
    np.testing.assert_allclose(out, np.array([2.0, 2.0, 2.0]), atol=1e-10)


def test_analytics_rules_moving_average_empty_series():
    """An empty series returns an empty array (mean of empty is nan -> falls
    into the ``len < w`` branch producing an empty list)."""
    rules = AnalyticsRules()
    out = rules.moving_average(np.array([], dtype=float), window=3)
    assert out.shape == (0,)


# --------------------------------------------------------------------------- #
# AnalyticsRules.zscore
# --------------------------------------------------------------------------- #


def test_analytics_rules_zscore_constant_series_returns_zeros():
    """A constant series produces all zeros (no NaN from division by zero)."""
    rules = AnalyticsRules()
    out = rules.zscore(np.full(10, 5.0))
    assert out.shape == (10,)
    np.testing.assert_allclose(out, np.zeros(10), atol=1e-6)
    assert np.all(np.isfinite(out))


def test_analytics_rules_zscore_normal_series_mean_zero_std_one():
    """A normal series z-scores to mean ~0 and std ~1."""
    rules = AnalyticsRules()
    series = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = rules.zscore(series)
    assert abs(float(np.mean(out))) < 1e-6
    assert abs(float(np.std(out)) - 1.0) < 1e-6


def test_analytics_rules_zscore_empty_series_does_not_crash():
    """An empty series does not crash; the result is finite (empty or nan-safe)."""
    rules = AnalyticsRules()
    # The implementation adds 1e-8 to std so a truly empty input may produce
    # an empty or zero-length output; either is acceptable as long as we do
    # not raise.
    out = rules.zscore(np.array([], dtype=float))
    assert isinstance(out, np.ndarray)
