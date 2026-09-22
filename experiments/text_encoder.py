# experiments/text_encoder.py
"""Character-block encoder for ZeroDataModel's text-reading loop.

A deliberately minimal, deterministic, numpy-only encoder that maps a
fixed-length string to a fixed-dim observation vector. No pre-trained
embeddings, no tokeniser, no external NLP library — just a random
projection of one-hot character histograms.

Design (matches the task spec):

  1. Build a fixed ``CharEmbedding`` table of shape ``(256, emb_dim)``
     where each row is the (deterministic, seed-controlled) random
     vector for one byte value 0..255. We use byte values rather than
     Unicode code points so the encoder is encoding-agnostic: any
     UTF-8 string can be ``.encode('utf-8', errors='replace')``'d to a
     byte sequence and looked up directly. The first 128 rows cover
     ASCII; the rest cover high bytes.

  2. For an input string of length ``block_size``:
       a. Convert to bytes (UTF-8, errors='replace').
       b. Look up each byte's embedding — ``(block_size, emb_dim)``.
       c. Aggregate by MEAN over the block axis. Mean (rather than
          sum) keeps the magnitude stable across blocks of different
          effective lengths (e.g. if a future caller passes a shorter
          string). For the standard ``TextStream`` the length is
          always exactly ``block_size``, so mean and sum differ only
          by a constant scale.

  3. Apply a fixed random projection ``(emb_dim, output_dim)`` to
     reduce / reshape to the model's ``dim``. This decouples the
     embedding width (which we want wide enough to separate all 256
     bytes) from the model dim (which the user controls).

The result is a ``(output_dim,)`` vector. The same string always
encodes to the same vector (deterministic). Different strings
producing the same byte histogram will collide — this is intentional:
the encoder is a BAG-OF-CHARACTERS, not a sequence model. The
sequence information is left for the model to discover from the
temporal order of consecutive blocks.

Why bag-of-characters (not sequence)?
  - Phase H is the FIRST text experiment. We want to verify the model
    can learn character-level statistics (letter frequencies, word
    boundaries via n-gram co-occurrence) before adding positional
    encodings. A bag-of-characters encoder is the simplest baseline
    that exposes this signal.
  - Positional encoding (sinusoidal, learned, or chunked) is a
    natural Phase H+ extension — see the ``positional`` parameter
    stub below.

Usage:
    from text_encoder import TextEncoder
    enc = TextEncoder(block_size=128, output_dim=32, seed=42)
    obs = enc.encode("The quick brown fox")
"""

from __future__ import annotations

import numpy as np


# Default embedding width. 64 is wide enough to separate 256 bytes
# with high probability under a random Gaussian projection, while
# keeping the encoder lightweight. The downstream model dim is
# typically 32-64 — we project to that after aggregation.
DEFAULT_EMB_DIM = 64


class TextEncoder:
    """Bag-of-characters encoder with a fixed random projection.

    Parameters
    ----------
    block_size : int
        Expected length of input strings. Used only for the optional
        positional encoding (not yet enabled by default); the encoder
        itself handles any length.
    output_dim : int
        Dimensionality of the output observation vector. Should match
        the model's ``dim``.
    emb_dim : int
        Width of the per-character embedding table. Defaults to
        ``DEFAULT_EMB_DIM`` (64).
    seed : int | None
        Seed for the random projection. ``None`` = non-deterministic.
        Same seed + same input string => same output vector.
    positional : bool
        If True, ADD a sinusoidal positional encoding to the
        per-character embedding before aggregation. Default False —
        the bag-of-characters baseline comes first. (Stub for Phase H+.)
    """

    def __init__(
        self,
        block_size: int = 128,
        output_dim: int = 32,
        emb_dim: int = DEFAULT_EMB_DIM,
        seed: int | None = 42,
        positional: bool = False,
    ):
        if output_dim <= 0:
            raise ValueError(f"output_dim must be > 0, got {output_dim}")
        if emb_dim <= 0:
            raise ValueError(f"emb_dim must be > 0, got {emb_dim}")
        self.block_size = int(block_size)
        self.output_dim = int(output_dim)
        self.emb_dim = int(emb_dim)
        self.seed = seed
        self.positional = bool(positional)

        rng = np.random.default_rng(seed)
        # Per-character embedding table: (256, emb_dim).
        # We use a Gaussian (mean 0, std 1/sqrt(emb_dim)) so the
        # expected L2 norm of each row is 1, and the average of
        # ``block_size`` rows has L2 norm ~ 1/sqrt(block_size) — well
        # within the model's expected observation magnitude (the
        # vision encoder produces vectors of L2 norm ~ 1-10).
        self.char_table = rng.standard_normal((256, self.emb_dim)) / np.sqrt(self.emb_dim)
        # Fixed random projection: (emb_dim, output_dim). Same seed
        # => same matrix => deterministic encoding.
        self.projection = rng.standard_normal((self.emb_dim, self.output_dim)) / np.sqrt(self.emb_dim)
        # Pre-build the positional encoding if requested (Phase H+ stub).
        if self.positional:
            # Standard transformer sinusoidal positional encoding.
            pos = np.arange(self.block_size)[:, None]  # (block_size, 1)
            div = np.exp(np.arange(0, self.emb_dim, 2) * (-np.log(10000.0) / self.emb_dim))
            self._pos_enc = np.zeros((self.block_size, self.emb_dim))
            self._pos_enc[:, 0::2] = np.sin(pos * div)
            self._pos_enc[:, 1::2] = np.cos(pos * div)
        else:
            self._pos_enc = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def encode(self, text: str) -> np.ndarray:
        """Encode ``text`` into a ``(output_dim,)`` observation vector.

        The encoding is deterministic: the same string always produces
        the same vector (given the same constructor seed).

        Empty strings return a zero vector — the caller should treat
        a zero-norm observation as an end-of-stream signal.
        """
        if not text:
            return np.zeros(self.output_dim, dtype=float)
        # Convert to bytes. ``errors='replace'`` substitutes the
        # Unicode replacement char (U+FFFD, which becomes 0xEF 0xBF
        # 0xBD in UTF-8) for any un-encodable code point — so the
        # encoder never raises on weird input.
        b = text.encode("utf-8", errors="replace")
        # Convert to a numpy uint8 array for fancy indexing.
        arr = np.frombuffer(b, dtype=np.uint8)
        # Look up per-byte embeddings: (len(b), emb_dim).
        embeds = self.char_table[arr]
        # Optional positional encoding (Phase H+). We add it BEFORE
        # aggregation so sequence information survives into the mean.
        # NOTE: this changes the encoder's semantics from "bag" to
        # "position-aware bag" — only enable for Phase H+ experiments.
        if self._pos_enc is not None and embeds.shape[0] == self.block_size:
            embeds = embeds + self._pos_enc
        # Aggregate by mean. Mean (not sum) keeps magnitude stable
        # across variable block lengths.
        agg = embeds.mean(axis=0)  # (emb_dim,)
        # Project to output_dim.
        out = agg @ self.projection  # (output_dim,)
        # Sanity: replace any NaN/Inf with 0. The random projection is
        # well-conditioned by construction (1/sqrt(emb_dim) scaling),
        # but a degenerate input (e.g. all-zero bytes) could produce
        # 0/0 if we divided somewhere. Be defensive.
        if not np.all(np.isfinite(out)):
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    # ------------------------------------------------------------------ #
    # Introspection helpers
    # ------------------------------------------------------------------ #
    def char_histogram(self, text: str) -> np.ndarray:
        """Return the raw 256-d byte histogram (un-normalised).

        Useful for analysis scripts that want to inspect WHAT the
        encoder is "seeing" without the random projection.
        """
        if not text:
            return np.zeros(256, dtype=float)
        b = text.encode("utf-8", errors="replace")
        arr = np.frombuffer(b, dtype=np.uint8)
        hist = np.zeros(256, dtype=float)
        for byte in arr:
            hist[byte] += 1.0
        return hist

    def __repr__(self) -> str:
        return (
            f"TextEncoder(block_size={self.block_size}, "
            f"output_dim={self.output_dim}, emb_dim={self.emb_dim}, "
            f"seed={self.seed}, positional={self.positional})"
        )


# ------------------------------------------------------------------ #
# Self-test
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    enc = TextEncoder(block_size=128, output_dim=32, seed=42)
    print(enc)
    v1 = enc.encode("The quick brown fox jumps over the lazy dog.")
    v2 = enc.encode("The quick brown fox jumps over the lazy dog.")
    v3 = enc.encode("A totally different sentence about physics!")
    print(f"v1 shape: {v1.shape}, L2 norm: {np.linalg.norm(v1):.4f}")
    print(f"v1 == v2 (deterministic): {np.allclose(v1, v2)}")
    print(f"v1 vs v3 cosine: "
          f"{float(np.dot(v1, v3) / (np.linalg.norm(v1) * np.linalg.norm(v3))):.4f}")
    # Empty string returns zero vector.
    v0 = enc.encode("")
    print(f"empty string L2: {np.linalg.norm(v0):.4f}")
    print("\nTextEncoder self-test OK")
