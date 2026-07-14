# tests/test_capabilities_nlp.py
"""Tests for the zero-data NLP capability module.

These capabilities compose the already-implemented core cognitive modules
(math universe, biological substrate, active inference, category theory) with
a small rule library (``NLPRules``) used as prior knowledge. No external NLP
libraries and no learned weights are required.
"""

from __future__ import annotations

import numpy as np

from zero_data_model.capabilities.nlp import (
    SemanticComparator,
    TextEncoder,
    TextGenerator,
    ZeroShotClassifier,
)


def test_text_encoder_shape():
    enc = TextEncoder(dim=64)
    vec = enc.encode("hello world this is a test sentence")
    assert vec.shape == (64,)
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-6


def test_text_encoder_deterministic():
    enc = TextEncoder(dim=64)
    v1 = enc.encode("the quick brown fox")
    v2 = enc.encode("the quick brown fox")
    np.testing.assert_allclose(v1, v2, atol=1e-8)


def test_semantic_similarity_identical():
    comp = SemanticComparator(TextEncoder(dim=64))
    sim = comp.similarity("machine learning models", "machine learning models")
    assert sim > 0.95


def test_semantic_similarity_different():
    comp = SemanticComparator(TextEncoder(dim=64))
    same_domain = comp.similarity("code data model", "algorithm network compute")
    other_domain = comp.similarity("code data model", "tree river mountain")
    assert other_domain < same_domain


def test_zero_shot_classifier_topic():
    clf = ZeroShotClassifier(dim=64)
    topic, conf = clf.classify("the algorithm computes the network")
    assert topic == "tech"
    assert conf > 0.0


def test_text_generator_length():
    gen = TextGenerator(dim=64)
    out = gen.generate("seed", 32)
    assert isinstance(out, str)
    assert len(out) == 32
    for ch in out:
        assert 32 <= ord(ch) <= 126
