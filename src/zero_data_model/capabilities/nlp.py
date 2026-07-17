# src/zero_data_model/capabilities/nlp.py
"""Natural Language Processing capability module for the zero-data cognitive model.

Composes the core cognitive modules (math universe, biological substrate,
active inference, category theory) with a small rule library (``NLPRules``)
used as prior knowledge. No external NLP libraries and no learned weights are
required -- every operator below is a deterministic, rule-based prior.
"""

from __future__ import annotations

import numpy as np

from ..active_inference import ActiveInferenceEngine
from ..base import Signal
from ..biological import BiologicalSubstrate
from ..category_engine import CategoryTheoryEngine
from ..math_universe import MathematicalUniverse
from .rules import NLPRules


class TextEncoder:
    """Encode text into a fixed-length, L2-normalized vector with no training data.

    The representation combines three self-generated feature groups:
      1. char-level n-gram hashing (n=2, 3) into ``dim`` bins,
      2. Unicode codepoint statistics (mean, std, max),
      3. rule-based token features (token count, avg token length, sentiment
         prior) plus topic-membership indicators derived from
         ``NLPRules.topic_keywords`` -- the latter is what lets texts that share
         a topic (e.g. two "tech" sentences) land closer together even when they
         share no surface vocabulary.
    """

    def __init__(self, dim: int = 64, rules: NLPRules | None = None):
        self.dim = dim
        self.rules = rules if rules is not None else NLPRules()

    @staticmethod
    def _hash_ngram(gram: str) -> int:
        # Deterministic polynomial rolling hash (stable across processes,
        # unlike Python's salted ``hash`` for strings).
        h = 0
        for ch in gram:
            h = (h * 131 + ord(ch)) & 0xFFFFFFFF
        return h

    def encode(self, text: str) -> np.ndarray:
        """Encode ``text`` into a ``dim``-length float array, L2-normalized."""
        vec = np.zeros(self.dim, dtype=float)
        lowered = text.lower()

        # 1. char-level n-gram hashing (n=2, 3) into `dim` bins.
        for n in (2, 3):
            for i in range(len(lowered) - n + 1):
                gram = lowered[i:i + n]
                idx = self._hash_ngram(gram) % self.dim
                vec[idx] += 1.0

        # 2. Unicode codepoint statistics (mean, std, max), scaled to ~[0, 1].
        cps = [ord(c) for c in text]
        if cps:
            cp_arr = np.asarray(cps, dtype=float)
            cp_mean = float(cp_arr.mean()) / 128.0
            cp_std = float(cp_arr.std()) / 30.0
            cp_max = float(cp_arr.max()) / 128.0
        else:
            cp_mean = cp_std = cp_max = 0.0

        # 3. rule-based token features.
        tokens = self.rules.tokenize(text)
        token_count = float(len(tokens)) / 20.0
        avg_len = (float(np.mean([len(t) for t in tokens])) if tokens else 0.0) / 10.0
        sentiment = (self.rules.sentiment_prior(tokens) + 1.0) / 2.0

        # 3b. topic-membership indicators from rules.topic_keywords.
        topics = sorted(self.rules.topic_keywords.keys())
        topic_counts = dict.fromkeys(topics, 0)
        for tok in tokens:
            for topic in topics:
                if tok in self.rules.topic_keywords[topic]:
                    topic_counts[topic] += 1

        # Place the scalar + topic features into dedicated trailing bins so they
        # do not collide with each other. Topic bins are weighted so that
        # same-topic texts align strongly in a shared dimension.
        scalars = [cp_mean, cp_std, cp_max, token_count, avg_len, sentiment]
        n_scalar = len(scalars)
        n_topic = len(topics)
        topic_weight = 3.0
        base = self.dim - (n_scalar + n_topic)
        for i, s in enumerate(scalars):
            vec[(base + i) % self.dim] += s
        for j, topic in enumerate(topics):
            vec[(base + n_scalar + j) % self.dim] += topic_counts[topic] * topic_weight

        # L2-normalize.
        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec = vec / norm
        return vec


class SemanticComparator:
    """Compare two texts by blending cosine similarity with KL divergence."""

    def __init__(self, encoder: TextEncoder, math_universe=None, category_engine=None):
        self.encoder = encoder
        self.dim = encoder.dim
        self.math_universe = (
            math_universe if math_universe is not None else MathematicalUniverse(dim=self.dim)
        )
        self.category_engine = (
            category_engine if category_engine is not None else CategoryTheoryEngine(dim=self.dim)
        )

    def similarity(self, a: str, b: str) -> float:
        """Blend cosine similarity with KL divergence; return a float in [0, 1].

        Cosine is computed via the category engine's isomorphism finder when
        available, else directly. KL divergence is converted to a similarity
        via ``exp(-KL)``. Identical texts yield ~1.0.
        """
        va = self.encoder.encode(a)
        vb = self.encoder.encode(b)

        if self.category_engine is not None:
            # Round-7 audit NEW5-10 (Round-2 + Round-7): use the renamed
            # ``structural_similarity`` instead of the deprecated
            # ``find_isomorphism`` alias.
            cos = self.category_engine.structural_similarity(va, vb)
        else:
            na = float(np.linalg.norm(va))
            nb = float(np.linalg.norm(vb))
            cos = float(np.dot(va, vb) / (na * nb + 1e-8))
        cos_01 = (cos + 1.0) / 2.0

        if self.math_universe is not None:
            kl = self.math_universe.info_geometry.kl_divergence(va, vb)
            kl_sim = float(np.exp(-kl)) if np.isfinite(kl) else 0.0
            result = 0.5 * cos_01 + 0.5 * kl_sim
        else:
            result = cos_01

        return float(np.clip(result, 0.0, 1.0))


class ZeroShotClassifier:
    """Zero-shot text classifier built from self-generated topic prototypes."""

    def __init__(
        self,
        dim: int = 64,
        active_inference=None,
        category_engine=None,
        rules: NLPRules | None = None,
    ):
        self.dim = dim
        self.rules = rules if rules is not None else NLPRules()
        self.encoder = TextEncoder(dim, self.rules)
        self.active_inference = (
            active_inference
            if active_inference is not None
            else ActiveInferenceEngine(state_dim=dim, obs_dim=dim, action_dim=16)
        )
        self.category_engine = (
            category_engine if category_engine is not None else CategoryTheoryEngine(dim=dim)
        )
        self.prototypes = self._build_prototypes()

    def _build_prototypes(self) -> dict[str, np.ndarray]:
        """Build one prototype vector per topic from rules.topic_keywords.

        Each prototype is the L2-normalized mean of the encoded keywords for
        that topic -- i.e. a self-generated prototype with no external data.
        """
        protos: dict[str, np.ndarray] = {}
        for topic, keywords in self.rules.topic_keywords.items():
            vecs = [self.encoder.encode(kw) for kw in keywords]
            if vecs:
                proto = np.mean(np.stack(vecs), axis=0)
                n = float(np.linalg.norm(proto))
                if n > 1e-8:
                    proto = proto / n
                protos[topic] = proto
        return protos

    def classify(self, text: str) -> tuple[str, float]:
        """Classify ``text`` by scoring its encoding against each topic prototype.

        Score blends cosine similarity (via the category engine) with free
        energy from active inference (lower free energy = better fit). Returns
        ``(best_topic_name, confidence in [0, 1])``.
        """
        vec = self.encoder.encode(text)

        best_topic: str | None = None
        best_score = -float("inf")
        best_cos = 0.0
        for topic in sorted(self.prototypes.keys()):
            proto = self.prototypes[topic]
            cos = self.category_engine.structural_similarity(vec, proto)
            score = cos
            if self.active_inference is not None:
                fe = float(self.active_inference.compute_free_energy(vec - proto))
                score = cos - 0.01 * fe
            if score > best_score:
                best_score = score
                best_topic = topic
                best_cos = cos

        confidence = float(np.clip(best_cos, 0.0, 1.0))
        assert best_topic is not None  # prototypes are non-empty by construction
        return (best_topic, confidence)


class TextGenerator:
    """Generate text with no external data, using the biological substrate."""

    def __init__(
        self,
        dim: int = 64,
        biological=None,
        math_universe=None,
        rules: NLPRules | None = None,
    ):
        self.dim = dim
        self.rules = rules if rules is not None else NLPRules()
        self.encoder = TextEncoder(dim, self.rules)
        self.biological = biological if biological is not None else BiologicalSubstrate(dim=dim)
        self.math_universe = (
            math_universe if math_universe is not None else MathematicalUniverse(dim=dim)
        )

    @staticmethod
    def _to_printable(value: float) -> str:
        """Deterministically map a scalar to one printable ASCII char (32..126)."""
        v = float(np.tanh(value))
        mapped = int(np.floor(((v + 1.0) / 2.0) * 95.0)) + 32
        mapped = max(32, min(126, mapped))
        return chr(mapped)

    def generate(self, seed: str, length: int = 32) -> str:
        """Generate a ``length``-char string of printable ASCII from ``seed``."""
        seed_vec = self.encoder.encode(seed)

        # Store at least two entries so dna_storage.generate can recombine
        # them via crossover (it requires >=2 stored sequences).
        self.biological.dna_storage.store("seed_a", seed_vec)
        self.biological.dna_storage.store("seed_b", seed_vec + 0.01)

        signal = Signal(data=seed_vec.copy(), metadata={"source": "text_generator"})
        generated = self.biological.dna_storage.generate(signal)
        vec = np.asarray(generated.data, dtype=float).flatten()
        if vec.size == 0:
            vec = np.zeros(self.dim)

        # Iteratively refine via the fractal generator when available.
        if self.math_universe is not None:
            vec = self.math_universe.fractal.generate(vec, n_iterations=5)
        vec = np.asarray(vec, dtype=float).flatten()
        if vec.size == 0:
            vec = np.zeros(self.dim)
        vec = np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=-1.0)

        # Decode the vector back to characters, cycling so the output always
        # has exactly `length` printable ASCII characters.
        chars = [self._to_printable(float(vec[i % vec.size])) for i in range(length)]
        return "".join(chars)
