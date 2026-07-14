# src/zero_data_model/capabilities/nlp_advanced.py
"""Advanced Natural Language Processing capabilities for the zero-data model.

Like the base ``nlp`` module, every operator here is a deterministic, rule-based
prior composed with the existing core cognitive modules. No external NLP
libraries and no learned weights are required -- multilingual awareness,
syntactic analysis and sentence-level encoding are all derived from Unicode
block ranges, suffix rules and the ``TextEncoder`` hashing scheme.
"""

from __future__ import annotations

import numpy as np

from .nlp import TextEncoder
from .rules import NLPRules


# Unicode block ranges used for rule-based script detection. Each entry maps a
# script name to a tuple of (low, high) codepoint inclusive bounds; ranges are
# ordered from most-specific to least so the first match wins for any char.
_SCRIPT_RANGES: list[tuple[str, tuple[int, int]]] = [
    ("cyrillic", (0x0400, 0x04FF)),
    ("cjk", (0x4E00, 0x9FFF)),
    ("arabic", (0x0600, 0x06FF)),
]


def _script_of_char(ch: str) -> str:
    """Return the script name for a single character using rule-based ranges.

    Latin is the default for ASCII letters and Latin-1 supplement; punctuation
    and whitespace return 'other' so they do not bias the script histogram.
    """
    cp = ord(ch)
    for name, (lo, hi) in _SCRIPT_RANGES:
        if lo <= cp <= hi:
            return name
    # Latin: basic ASCII letters + Latin-1 supplement letters.
    if (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A) or (0x00C0 <= cp <= 0x024F):
        return "latin"
    return "other"


class MultiLingualEncoder:
    """Extend ``TextEncoder`` with rule-based multilingual awareness.

    Script detection is performed via Unicode block ranges (Latin / Cyrillic /
    CJK / Arabic). Per-script character statistics are blended with the base
    ``TextEncoder`` representation to produce a script-aware embedding of length
    ``dim`` (L2-normalized).
    """

    def __init__(self, dim: int = 64, rules: NLPRules | None = None):
        self.dim = dim
        self.rules = rules if rules is not None else NLPRules()
        self.encoder = TextEncoder(dim, self.rules)

    def detect_script(self, text: str) -> str:
        """Return the dominant script name for ``text``.

        Returns one of ``'latin'``, ``'cyrillic'``, ``'cjk'``, ``'arabic'`` or
        ``'mixed'`` (when no single script dominates the recognized characters).
        """
        counts = {"latin": 0, "cyrillic": 0, "cjk": 0, "arabic": 0}
        for ch in text:
            s = _script_of_char(ch)
            if s in counts:
                counts[s] += 1
        total = sum(counts.values())
        if total == 0:
            return "latin"  # default for punctuation/whitespace-only input
        max_count = max(counts.values())
        # 'mixed' when the top script covers less than 70% of recognized chars.
        if max_count / total < 0.7 and sum(1 for v in counts.values() if v > 0) > 1:
            return "mixed"
        # Return the name with the highest count.
        return max(counts.items(), key=lambda kv: kv[1])[0]

    def _script_stats(self, text: str) -> np.ndarray:
        """Return a 4-vector of per-script proportions (latin/cyrillic/cjk/arabic)."""
        counts = {"latin": 0, "cyrillic": 0, "cjk": 0, "arabic": 0}
        for ch in text:
            s = _script_of_char(ch)
            if s in counts:
                counts[s] += 1
        total = sum(counts.values())
        if total == 0:
            return np.zeros(4, dtype=float)
        return np.array(
            [counts["latin"], counts["cyrillic"], counts["cjk"], counts["arabic"]],
            dtype=float,
        ) / float(total)

    def encode(self, text: str) -> np.ndarray:
        """Encode ``text`` into a ``dim``-length L2-normalized script-aware vector."""
        base = self.encoder.encode(text)
        stats = self._script_stats(text)
        vec = base.copy()
        # Place the 4 per-script proportions into dedicated trailing bins.
        n_stats = stats.shape[0]
        base_idx = (self.dim - n_stats) % self.dim
        for i, s in enumerate(stats):
            vec[(base_idx + i) % self.dim] += float(s) * 2.0
        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec = vec / norm
        return vec


class SyntacticAnalyzer:
    """Rule-based syntactic analysis with no external parser.

    Suffix rules guess part-of-speech for each token; a stopword-position
    heuristic produces a coarse subject-verb-object (SVO) dependency hint. The
    output is a fully deterministic dictionary suitable for downstream
    zero-data reasoning.
    """

    # Suffix -> guessed POS, ordered so longer suffixes are matched first.
    _SUFFIX_RULES: list[tuple[str, str]] = [
        ("tion", "noun"),
        ("ness", "noun"),
        ("ment", "noun"),
        ("ity", "noun"),
        ("ing", "verb"),
        ("ed", "verb"),
        ("ly", "adverb"),
        ("ful", "adj"),
        ("ous", "adj"),
        ("able", "adj"),
        ("ive", "adj"),
    ]

    def __init__(self, dim: int = 64, rules: NLPRules | None = None):
        self.dim = dim
        self.rules = rules if rules is not None else NLPRules()

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences on '.', '!', '?' (rule-based)."""
        sentences: list[str] = []
        current: list[str] = []
        for ch in text:
            current.append(ch)
            if ch in ".!?":
                sent = "".join(current).strip()
                if sent:
                    sentences.append(sent)
                current = []
        tail = "".join(current).strip()
        if tail:
            sentences.append(tail)
        return sentences

    def _guess_pos(self, word: str) -> str:
        """Guess the POS of ``word`` from suffix rules; default is 'noun'."""
        lowered = word.lower()
        if lowered in self.rules.stopwords:
            # Stopwords are mostly determiners / pronouns / prepositions.
            return "stop"
        for suffix, pos in self._SUFFIX_RULES:
            if lowered.endswith(suffix) and len(lowered) > len(suffix):
                return pos
        return "noun"

    def analyze(self, text: str) -> dict:
        """Return a rule-based syntactic analysis dictionary for ``text``."""
        sentences = self._split_sentences(text)
        sentence_count = len(sentences)

        # Punctuation density is computed on the original text before any
        # punctuation stripping, so it actually reflects the input.
        total_chars = max(1, len(text))
        punct_count = sum(1 for ch in text if ch in self.rules.punctuation)
        punctuation_density = float(punct_count) / float(total_chars)

        # Tokenize by replacing punctuation with whitespace.
        cleaned = text
        for ch in self.rules.punctuation:
            cleaned = cleaned.replace(ch, " ")
        raw_tokens = [t for t in cleaned.split() if t]

        if raw_tokens:
            avg_word_length = float(np.mean([len(t) for t in raw_tokens]))
        else:
            avg_word_length = 0.0

        # POS guesses per token (lowercased word -> POS string).
        pos_guesses: dict[str, str] = {}
        for tok in raw_tokens:
            pos_guesses[tok.lower()] = self._guess_pos(tok)

        # Coarse SVO dependency hint from stopword positions and POS guesses.
        dependency_hint = self._dependency_hint(raw_tokens)

        return {
            "sentence_count": int(sentence_count),
            "avg_word_length": float(avg_word_length),
            "punctuation_density": float(punctuation_density),
            "pos_guesses": pos_guesses,
            "dependency_hint": dependency_hint,
        }

    def _dependency_hint(self, tokens: list[str]) -> dict:
        """Coarse subject-verb-object heuristic from rules + stopword positions.

        The first non-stopword token is treated as the subject candidate, the
        first verb-guessed token as the verb candidate, and the first noun-guessed
        token after the verb as the object candidate. Each is returned as a
        lowercased string (or empty string if none was found).
        """
        subject = ""
        verb = ""
        obj = ""
        verb_seen = False
        for tok in tokens:
            pos = self._guess_pos(tok)
            lower = tok.lower()
            if pos == "stop":
                continue
            if subject == "" and pos in ("noun", "adj"):
                subject = lower
            elif verb == "" and pos == "verb":
                verb = lower
                verb_seen = True
            elif verb_seen and pos in ("noun", "adj") and obj == "":
                obj = lower
        return {"subject": subject, "verb": verb, "object": obj}


class SentenceEncoder:
    """Encode each sentence of a text into its own ``dim``-length vector.

    Sentence splitting is rule-based (on '.', '!', '?'); each sentence is then
    encoded via the shared ``TextEncoder``. The output is an
    ``(n_sentences, dim)`` array, or a ``(0, dim)`` array when no sentence is
    found.
    """

    def __init__(self, dim: int = 64, rules: NLPRules | None = None):
        self.dim = dim
        self.rules = rules if rules is not None else NLPRules()
        self.encoder = TextEncoder(dim, self.rules)

    def encode(self, text: str) -> np.ndarray:
        """Encode each sentence; return an ``(n_sentences, dim)`` array."""
        sentences = SyntacticAnalyzer._split_sentences(text)
        if not sentences:
            return np.zeros((0, self.dim), dtype=float)
        vecs = np.stack([self.encoder.encode(s) for s in sentences])
        return vecs
