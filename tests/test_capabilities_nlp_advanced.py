# tests/test_capabilities_nlp_advanced.py
"""Tests for the advanced zero-data NLP capability module.

The advanced NLP capabilities (multilingual encoding, rule-based syntactic
analysis, sentence-level encoding) compose the existing ``TextEncoder`` with
Unicode-block script detection and suffix-rule POS guessing. No external NLP
libraries and no learned weights are required.
"""

from __future__ import annotations

import numpy as np

from zero_data_model.capabilities.nlp_advanced import (
    MultiLingualEncoder,
    SentenceEncoder,
    SyntacticAnalyzer,
)


def test_multilingual_encoder_shape_and_norm():
    """encode(text) returns a dim-length, L2-normalized vector."""
    enc = MultiLingualEncoder(dim=64)
    vec = enc.encode("hello world this is a test")
    assert vec.shape == (64,)
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-6


def test_detect_script_latin():
    """Latin ASCII input is detected as 'latin'."""
    enc = MultiLingualEncoder(dim=64)
    assert enc.detect_script("hello world") == "latin"


def test_detect_script_cyrillic():
    """Cyrillic input is detected as 'cyrillic'."""
    enc = MultiLingualEncoder(dim=64)
    assert enc.detect_script("привет мир") == "cyrillic"


def test_detect_script_cjk():
    """CJK input is detected as 'cjk'."""
    enc = MultiLingualEncoder(dim=64)
    assert enc.detect_script("你好世界") == "cjk"


def test_detect_script_arabic_and_mixed():
    """Arabic input is detected as 'arabic'; mixed input as 'mixed'."""
    enc = MultiLingualEncoder(dim=64)
    assert enc.detect_script("مرحبا بالعالم") == "arabic"
    # Roughly equal Latin + Cyrillic + CJK should be flagged as mixed.
    assert enc.detect_script("hello мир 你好") == "mixed"


def test_syntactic_analyzer_keys_and_types():
    """analyze(text) returns the five required keys with correct types."""
    sa = SyntacticAnalyzer(dim=64)
    out = sa.analyze("The cat is running. She quickly painted the creation!")
    assert set(out.keys()) == {
        "sentence_count",
        "avg_word_length",
        "punctuation_density",
        "pos_guesses",
        "dependency_hint",
    }
    assert isinstance(out["sentence_count"], int)
    assert isinstance(out["avg_word_length"], float)
    assert isinstance(out["punctuation_density"], float)
    assert isinstance(out["pos_guesses"], dict)
    assert isinstance(out["dependency_hint"], dict)


def test_syntactic_analyzer_sentence_count():
    """Sentence splitting on '.', '!' counts the right number of sentences."""
    sa = SyntacticAnalyzer(dim=64)
    out = sa.analyze("First sentence. Second one! Third?")
    assert out["sentence_count"] == 3


def test_syntactic_pos_guesses_correct_suffix_entries():
    """Suffix rules guess the right POS for known suffix words."""
    sa = SyntacticAnalyzer(dim=64)
    out = sa.analyze("running quickly creation beautiful painted")
    pos = out["pos_guesses"]
    # -ing -> verb
    assert pos.get("running") == "verb"
    # -ly -> adverb
    assert pos.get("quickly") == "adverb"
    # -tion -> noun
    assert pos.get("creation") == "noun"
    # -ful -> adj
    assert pos.get("beautiful") == "adj"
    # -ed -> verb
    assert pos.get("painted") == "verb"


def test_syntactic_dependency_hint_svo():
    """The SVO dependency hint picks out subject / verb / object candidates."""
    sa = SyntacticAnalyzer(dim=64)
    out = sa.analyze("cat running creation")
    hint = out["dependency_hint"]
    assert set(hint.keys()) == {"subject", "verb", "object"}
    # First noun-able token becomes the subject.
    assert hint["subject"] == "cat"
    # First verb-guessed token becomes the verb.
    assert hint["verb"] == "running"
    # First noun after the verb becomes the object.
    assert hint["object"] == "creation"


def test_sentence_encoder_shape():
    """encode(text) returns an (n_sentences, dim) array."""
    se = SentenceEncoder(dim=64)
    arr = se.encode("Hello world. Foo bar baz! Third sentence here?")
    assert arr.ndim == 2
    assert arr.shape[0] == 3
    assert arr.shape[1] == 64
    # Each row should be L2-normalized (TextEncoder property).
    for row in arr:
        assert abs(float(np.linalg.norm(row)) - 1.0) < 1e-6


def test_sentence_encoder_empty_returns_zero_rows():
    """encode('') returns an (0, dim) array instead of raising."""
    se = SentenceEncoder(dim=32)
    arr = se.encode("")
    assert arr.shape == (0, 32)


def test_multilingual_encoder_deterministic():
    """Encoding the same text twice yields identical vectors."""
    enc = MultiLingualEncoder(dim=64)
    v1 = enc.encode("hello world")
    v2 = enc.encode("hello world")
    np.testing.assert_allclose(v1, v2, atol=1e-8)
