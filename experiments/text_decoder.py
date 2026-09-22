# experiments/text_decoder.py
"""Generative text decoder (Phase K, Task 2.1).

A ZERO-BACKPROP text decoder that maps a hidden state back to natural
language. Project philosophy forbids learned-from-scratch RNN / LM
training, so we use a hybrid strategy:

  1. **Retrieval**: given a query latent, find the most similar
     previously-read text block in a ``RetrievalPool`` (cosine sim of
     the latent vectors). The retrieved block becomes the seed.

  2. **Markov continuation**: continue the seed text using an n-gram
     Markov chain learned ON-THE-FLY from the retrieval pool. We never
     train weights — we just count token transitions from the
     already-read text. The first token after the seed comes from the
     seed's last token; subsequent tokens are sampled by a fixed
     ``temperature``.

  3. **Template substitution** (optional): if the caller supplies a
     ``causal_graph`` (a list of (entity, role) pairs from the
     knowledge graph), substitute the most-similar retrieved entities
     with the current context entities.

  4. **Truncation / cleanup**: trim incomplete trailing sentences,
     collapse repeated whitespace.

The decoder is paired with a ``RetrievalPool`` that grows as the model
reads more text. The pool stores (latent, text) pairs and supports
fast cosine-similarity search via brute force (the pool is small
enough for a smoke run; a production version would use an ANN index).

Usage
------
    decoder = TextDecoder(text_encoder, seed=42)
    decoder.ingest("hello world", latent_vector)
    decoder.ingest("physics is the study of matter", latent2)
    text = decoder.decode(query_latent, max_length=64)
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


# ------------------------------------------------------------------ #
# Tokenisation
# ------------------------------------------------------------------ #
# A simple word-level tokeniser. Splits on whitespace and punctuation
# but KEEPS the punctuation as separate tokens (so the Markov chain
# can learn sentence boundaries).
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Word-level tokeniser that preserves punctuation as tokens."""
    return TOKEN_RE.findall(text)


def detokenize(tokens: list[str]) -> str:
    """Inverse of ``tokenize`` — joins tokens back to text.

    Inserts a space between two word tokens, no space before
    punctuation, no space after opening punctuation.
    """
    if not tokens:
        return ""
    out = [tokens[0]]
    for prev, cur in zip(tokens, tokens[1:]):
        if _is_word(prev) and _is_word(cur):
            out.append(" " + cur)
        elif prev in "([{":
            out.append(cur)
        elif cur in ")]}.,;:!?":
            out.append(cur)
        else:
            out.append(" " + cur if not cur.isspace() else cur)
    return "".join(out)


def _is_word(tok: str) -> bool:
    """True iff ``tok`` consists only of word characters."""
    return bool(tok) and all(c.isalnum() or c == "_" for c in tok)


# ------------------------------------------------------------------ #
# RetrievalPool
# ------------------------------------------------------------------ #
@dataclass
class RetrievalPoolEntry:
    latent: np.ndarray
    text: str
    source: str = ""
    tokens: list[str] = field(default_factory=list)
    free_energy: float = 0.0


class RetrievalPool:
    """An append-only store of (latent, text) pairs.

    The pool is the decoder's "memory" of what the model has read.
    Cosine similarity search is brute-force — sufficient for pools of
    up to ~10K entries (sub-millisecond per query).
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = int(max_size)
        self.entries: list[RetrievalPoolEntry] = []
        # Pre-computed normalised latent matrix (updated lazily).
        self._matrix: Optional[np.ndarray] = None
        self._dirty: bool = True

    def __len__(self) -> int:
        return len(self.entries)

    def ingest(self, latent: np.ndarray, text: str,
               source: str = "", free_energy: float = 0.0) -> None:
        """Add a (latent, text) pair to the pool."""
        if not text or not text.strip():
            return
        latent = np.asarray(latent, dtype=np.float64).flatten()
        entry = RetrievalPoolEntry(
            latent=latent,
            text=text,
            source=source,
            tokens=tokenize(text),
            free_energy=float(free_energy),
        )
        self.entries.append(entry)
        # Enforce capacity (FIFO eviction).
        if len(self.entries) > self.max_size:
            self.entries.pop(0)
        self._dirty = True

    def extend(self, items: Iterable[tuple[np.ndarray, str]]) -> None:
        """Bulk-ingest an iterable of (latent, text) pairs."""
        for latent, text in items:
            self.ingest(latent, text)

    def retrieve(self, query: np.ndarray, k: int = 5) -> list[RetrievalPoolEntry]:
        """Return the k most similar entries to ``query`` (cosine sim)."""
        if not self.entries:
            return []
        if self._dirty:
            self._rebuild_matrix()
        q = np.asarray(query, dtype=np.float64).flatten()
        qn = np.linalg.norm(q)
        if qn < 1e-12:
            return list(self.entries[:k])
        q_unit = q / qn
        sims = self._matrix @ q_unit
        top = np.argsort(sims)[::-1][:k]
        return [self.entries[i] for i in top]

    def most_similar_text(self, query: np.ndarray) -> Optional[str]:
        results = self.retrieve(query, k=1)
        return results[0].text if results else None

    # ------------------------------------------------------------------ #
    def _rebuild_matrix(self) -> None:
        if not self.entries:
            self._matrix = None
            self._dirty = False
            return
        mat = np.stack([e.latent for e in self.entries], axis=0)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        self._matrix = mat / norms
        self._dirty = False


# ------------------------------------------------------------------ #
# MarkovChain
# ------------------------------------------------------------------ #
class MarkovChain:
    """N-gram Markov chain over word tokens.

    Trained on a list of token sequences. Given a context of (n-1)
    tokens, samples the next token from the observed distribution.
    """

    def __init__(self, n: int = 2, seed: int = 42):
        self.n = int(n)
        self._rng = np.random.default_rng(seed)
        # transitions[context_tuple] -> {token: count}
        self.transitions: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # Starter contexts: those that appear at the beginning of a sequence.
        self.starters: list[tuple] = []
        # Vocabulary: for fallback when no transition matches.
        self.vocab: list[str] = []

    def train(self, token_sequences: Iterable[list[str]]) -> None:
        """Train on an iterable of token sequences."""
        for tokens in token_sequences:
            if len(tokens) < self.n:
                continue
            self.starters.append(tuple(tokens[:self.n - 1]))
            # Pad with a sentinel at the start so the (n-1) context
            # always has tokens for the first generation step.
            padded = ["<s>"] * (self.n - 1) + tokens + ["</s>"]
            for i in range(len(padded) - self.n + 1):
                context = tuple(padded[i:i + self.n - 1])
                nxt = padded[i + self.n - 1]
                self.transitions[context][nxt] += 1
            for tok in tokens:
                if tok not in self.vocab:
                    self.vocab.append(tok)

    def sample_next(self, context: tuple, temperature: float = 1.0) -> Optional[str]:
        """Sample the next token given a context tuple.

        ``temperature`` > 1 flattens the distribution; < 1 sharpens it.
        Returns None if no transition matches the context.

        Uses Katz-style backoff: try the full (n-1)-token context, then
        progressively shorter suffixes, then fall back to a unigram
        sample from the whole vocabulary.
        """
        # Try progressively shorter suffixes of the context.
        max_ctx_len = self.n - 1
        for ctx_len in range(max_ctx_len, -1, -1):
            if ctx_len == 0:
                # Unigram fallback: sample from any token in vocab.
                if not self.vocab:
                    return None
                # Build a unigram distribution by summing all transition
                # counts (cheap if transitions are small).
                all_counts: dict[str, int] = {}
                for dist in self.transitions.values():
                    for tok, c in dist.items():
                        all_counts[tok] = all_counts.get(tok, 0) + c
                if not all_counts:
                    # Last resort: uniform sample from vocab.
                    return str(self._rng.choice(self.vocab))
                tokens = list(all_counts.keys())
                counts = np.array(
                    [all_counts[t] for t in tokens], dtype=np.float64
                )
                probs = counts / counts.sum()
                if temperature != 1.0:
                    probs = probs ** (1.0 / max(temperature, 1e-3))
                    probs = probs / probs.sum()
                idx = int(self._rng.choice(len(tokens), p=probs))
                return tokens[idx]
            ctx = tuple(context[-ctx_len:]) if ctx_len > 0 else ()
            if ctx in self.transitions:
                dist = self.transitions[ctx]
                if dist:
                    tokens = list(dist.keys())
                    counts = np.array(
                        [dist[t] for t in tokens], dtype=np.float64
                    )
                    if temperature != 1.0:
                        probs = counts / counts.sum()
                        probs = probs ** (1.0 / max(temperature, 1e-3))
                        probs = probs / probs.sum()
                    else:
                        probs = counts / counts.sum()
                    idx = int(self._rng.choice(len(tokens), p=probs))
                    return tokens[idx]
        return None

    def generate(self, max_tokens: int = 50, temperature: float = 1.0,
                 seed_context: Optional[tuple] = None) -> list[str]:
        """Generate a token sequence up to ``max_tokens`` long.

        Stops at the ``</s>`` sentinel if reached.
        """
        if not self.transitions:
            return []
        if seed_context is None:
            if not self.starters:
                seed_context = ("<s>",) * (self.n - 1)
            else:
                seed_context = self.starters[int(self._rng.integers(0, len(self.starters)))]
        context = list(seed_context)
        out: list[str] = []
        for _ in range(max_tokens):
            nxt = self.sample_next(tuple(context), temperature=temperature)
            if nxt is None or nxt == "</s>":
                break
            if nxt != "<s>":
                out.append(nxt)
            context.append(nxt)
            if len(context) > self.n - 1:
                context = context[-(self.n - 1):]
        return out


# ------------------------------------------------------------------ #
# TextDecoder
# ------------------------------------------------------------------ #
class TextDecoder:
    """Hybrid retrieval + Markov text decoder.

    Parameters
    ----------
    text_encoder : TextEncoder
        Used to encode any freshly-ingested raw text into a latent.
    n : int
        Markov chain n-gram order. Default 2 (bigram).
    pool_max_size : int
        Maximum number of (latent, text) pairs retained for retrieval.
    seed : int
        Seed for the Markov chain RNG.
    temperature : float
        Default sampling temperature for the Markov chain.
    """

    def __init__(
        self,
        text_encoder=None,
        n: int = 2,
        pool_max_size: int = 10000,
        seed: int = 42,
        temperature: float = 1.0,
    ):
        self.text_encoder = text_encoder
        self.n = int(n)
        self.pool = RetrievalPool(max_size=pool_max_size)
        self.seed = int(seed)
        self.temperature = float(temperature)
        # Markov chain is rebuilt on demand when the pool changes.
        self._markov: Optional[MarkovChain] = None
        self._pool_signature: int = 0

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #
    def ingest(self, text: str, latent: Optional[np.ndarray] = None,
               source: str = "", free_energy: float = 0.0) -> None:
        """Add a text block (+ its latent) to the retrieval pool.

        If ``latent`` is None, the decoder encodes ``text`` via its
        ``text_encoder`` (must be provided at construction time).
        """
        if not text or not text.strip():
            return
        if latent is None:
            if self.text_encoder is None:
                raise ValueError("latent is None and no text_encoder available")
            latent = self.text_encoder.encode(text)
        self.pool.ingest(latent, text, source=source, free_energy=free_energy)
        self._pool_signature += 1

    def ingest_many(self, items: Iterable[tuple]) -> None:
        """Bulk-ingest an iterable of (text, latent, source?) tuples."""
        for item in items:
            text = item[0]
            latent = item[1] if len(item) > 1 else None
            source = item[2] if len(item) > 2 else ""
            fe = item[3] if len(item) > 3 else 0.0
            self.ingest(text, latent=latent, source=source, free_energy=fe)

    # ------------------------------------------------------------------ #
    # Decoding
    # ------------------------------------------------------------------ #
    def decode(
        self,
        latent: np.ndarray,
        max_length: int = 128,
        temperature: Optional[float] = None,
        n_seed_tokens: int = 8,
    ) -> str:
        """Decode a latent into a text string.

        Strategy:
          1. Retrieve the most similar text block from the pool.
          2. Tokenise it; use its first ``n_seed_tokens`` tokens as
             the seed context for the Markov chain.
          3. Generate up to ``max_length`` tokens via the chain.
          4. Detokenise + clean up the output.
        """
        if not self.pool.entries:
            return ""
        if temperature is None:
            temperature = self.temperature
        # Refresh Markov chain if the pool has changed.
        self._refresh_markov_if_needed()
        # Retrieve the most similar block.
        retrieved = self.pool.retrieve(latent, k=1)
        if not retrieved:
            return ""
        seed_entry = retrieved[0]
        # Use the first n_seed_tokens of the retrieved text as the seed.
        seed_tokens = seed_entry.tokens[:n_seed_tokens] if seed_entry.tokens else []
        if not seed_tokens:
            return seed_entry.text[:max_length]
        # Generate continuation with the Markov chain.
        if self._markov is None or not self._markov.transitions:
            # No chain yet — just return the retrieved text (truncated).
            return self._truncate(seed_entry.text, max_length)
        # Build the seed context (n-1 tokens).
        ctx = (["<s>"] * (self.n - 1)) + seed_tokens[-(self.n - 1):]
        generated = self._markov.generate(
            max_tokens=max_length,
            temperature=temperature,
            seed_context=tuple(ctx),
        )
        # Compose: seed + generated tokens, deduplicated.
        combined = seed_tokens + generated
        # Deduplicate consecutive duplicates (Markov chains can stutter).
        deduped: list[str] = []
        for t in combined:
            if deduped and deduped[-1] == t:
                continue
            deduped.append(t)
        text = detokenize(deduped)
        return self._truncate(text, max_length)

    def decode_with_template(
        self,
        latent: np.ndarray,
        template: str,
        substitutions: Optional[dict[str, str]] = None,
        max_length: int = 128,
    ) -> str:
        """Retrieve a seed, then apply template substitution.

        ``template`` is a string with ``{placeholder}`` fields. The
        decoder fills the placeholders from ``substitutions`` (a dict
        mapping placeholder names to entity names from the current
        context). The seed text itself is appended to the substituted
        template, providing a "generated hypothesis" feel.
        """
        # Get the seed.
        seed = self.decode(latent, max_length=max_length)
        # Apply substitutions to the template.
        out = template
        if substitutions:
            for k, v in substitutions.items():
                out = out.replace("{" + k + "}", str(v))
        # Append the retrieved seed for grounding.
        if seed:
            out = out + " " + seed
        return self._truncate(out, max_length)

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #
    def _refresh_markov_if_needed(self) -> None:
        # Track the pool size as the signature.
        sig = len(self.pool)
        if self._markov is not None and sig == self._pool_signature:
            return
        self._markov = MarkovChain(n=self.n, seed=self.seed)
        self._markov.train([e.tokens for e in self.pool.entries if e.tokens])
        self._pool_signature = sig

    @staticmethod
    def _truncate(text: str, max_length: int) -> str:
        """Truncate to ``max_length`` chars at a sentence boundary."""
        if len(text) <= max_length:
            return text.strip()
        truncated = text[:max_length]
        # Try to cut at the last sentence-ending punctuation.
        for punct in (". ", "? ", "! ", ".\n", "?\n", "!\n"):
            idx = truncated.rfind(punct)
            if idx > 0:
                return truncated[:idx + 1].strip()
        # Fall back to last whitespace.
        idx = truncated.rfind(" ")
        if idx > 0:
            return truncated[:idx].strip()
        return truncated.strip()


# ------------------------------------------------------------------ #
# Self-test
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    # Build a tiny pool of "encyclopedia-like" text.
    samples = [
        "Water is a liquid substance composed of hydrogen and oxygen.",
        "Electric current is the flow of electric charge through a conductor.",
        "Gravity is a force that attracts objects with mass toward each other.",
        "Light is electromagnetic radiation within a portion of the spectrum.",
        "Photosynthesis is the process by which plants convert sunlight into energy.",
        "A river is a natural flowing watercourse, usually freshwater.",
        "A voltage is the difference in electric potential between two points.",
        "Mass is a property of a physical body and a measure of its resistance to acceleration.",
        "Sound is a vibration that propagates as an acoustic wave through a medium.",
        "Heat is energy transferred between systems by thermal interaction.",
    ]
    # Reuse the Phase H TextEncoder to produce latents.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from text_encoder import TextEncoder
    text_encoder = TextEncoder(block_size=512, output_dim=32, seed=42)
    decoder = TextDecoder(text_encoder=text_encoder, n=2, seed=42, temperature=1.0)
    for s in samples:
        latent = text_encoder.encode(s)
        decoder.ingest(s, latent=latent, source="self-test")

    print(f"[self-test] pool size = {len(decoder.pool)}")
    print(f"[self-test] Markov transitions = {len(decoder._markov.transitions) if decoder._markov else 0}")

    # Decode from each query latent.
    for q_text in [
        "What is water?",
        "Tell me about electricity.",
        "What is the force that pulls things down?",
        "Describe sound.",
    ]:
        q_latent = text_encoder.encode(q_text)
        generated = decoder.decode(q_latent, max_length=120, temperature=0.9)
        print(f"  Q: {q_text}")
        print(f"  A: {generated}")
        print()

    # Template substitution test.
    q = text_encoder.encode("river")
    out = decoder.decode_with_template(
        q,
        template="{entity} behaves like water in some respects.",
        substitutions={"entity": "current"},
        max_length=160,
    )
    print(f"  template output: {out}")

    # Determinism with same seed.
    decoder2 = TextDecoder(text_encoder=text_encoder, n=2, seed=42, temperature=1.0)
    for s in samples:
        decoder2.ingest(s, latent=text_encoder.encode(s))
    q = text_encoder.encode("What is gravity?")
    a1 = decoder.decode(q, max_length=80)
    a2 = decoder2.decode(q, max_length=80)
    assert a1 == a2, f"decoder should be deterministic given seed; got {a1!r} vs {a2!r}"
    print("[self-test] determinism: OK")
    print("[self-test] OK")
