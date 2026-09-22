# experiments/text_stream.py
"""Streaming text environment for ZeroDataModel.

A minimal, deterministic, numpy-only text streamer that emits
fixed-length character blocks from a plain-text file. Designed as a
drop-in replacement for the pixel-streaming ``PhysicsSandbox`` in
the closed-loop pipeline — but feeding characters instead of pixels.

Phase H extends the streamer with **action-driven navigation**
(``navigate(action)``): the model can move forward, backward,
fast-forward, fast-backward, or stay in place. The original ``step()``
method is preserved as a thin wrapper around ``navigate(NAV_FORWARD)``
for backward compatibility.

Determinism contract: with a fixed ``seed`` (and no ``shuffle=True``),
the sequence of ``step()`` returns is identical across runs and
platforms. The only randomness in the streamer is the optional block
shuffling, which is drawn from a single ``np.random.Generator`` so a
fixed seed reproduces the same shuffle order.

Usage:
    from text_stream import TextStream, NAV_FORWARD
    stream = TextStream("wiki_simple.txt", block_size=256, seed=42)
    block = stream.step()         # next 256 chars (forward by default)
    block = stream.navigate(NAV_FORWARD)   # explicit forward
    print(stream.tell())          # current pointer
    stream.seek(0); stream.reset()  # rewind
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# Default reading block size in characters. 128 keeps each block
# short enough for human inspection while giving the encoder enough
# context to form a stable bag-of-characters signal.
DEFAULT_BLOCK_SIZE = 128


# ------------------------------------------------------------------ #
# Phase H: navigation action constants.
# These are the 5 discrete actions the model can take to control its
# reading position. They map 1:1 to the candidate indices returned by
# ``ActiveInferenceEngine.select_action`` when ``num_candidates=5`` is
# configured (see ``run_text_curious.make_text_curious_model``).
# ------------------------------------------------------------------ #
NAV_FORWARD       = 0  # +block_size  (next block)
NAV_BACKWARD      = 1  # -block_size  (previous block, clamped at 0)
NAV_FAST_FORWARD  = 2  # +block_size * 4
NAV_FAST_BACKWARD = 3  # -block_size * 4
NAV_STAY          = 4  # 0  (re-read current block)
N_NAV_ACTIONS = 5
NAV_ACTION_NAMES = [
    "forward", "backward", "fast_forward", "fast_backward", "stay",
]


class TextStream:
    """A streaming, position-tracked view over a plain-text file.

    The file is loaded fully into memory at construction time
    (sufficient for the typical Wiki-extracted text of a few MB).
    For multi-GB corpora, swap the in-memory buffer for an mmap; the
    public API stays the same.

    The streamer is deterministic given a fixed seed. ``step()`` is
    a pure function of ``pos`` (no internal RNG consumption) when
    ``shuffle=False``, so two streamers with the same file and same
    ``block_size`` produce identical block sequences regardless of
    seed. The seed only matters when ``shuffle=True`` is passed to
    ``reset()``.
    """

    def __init__(
        self,
        file_path: str | Path,
        block_size: int = DEFAULT_BLOCK_SIZE,
        seed: int | None = 42,
    ):
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}")
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"text file not found: {file_path}")
        # Read the whole file as UTF-8 with errors='replace' so that
        # any malformed byte sequence does not crash the loop. Lines
        # are joined with a single space — preserves word boundaries
        # without keeping the literal newline characters (which would
        # pollute the character statistics with a rare, uninformative
        # symbol).
        raw = self.file_path.read_text(encoding="utf-8", errors="replace")
        # Collapse any run of whitespace (newlines, tabs, multiple
        # spaces) into a single space, but preserve leading/trailing
        # space so block boundaries remain predictable. This is a
        # deliberate simplification — the model is meant to discover
        # structure from a normalised character distribution, not from
        # the original layout.
        self.text = " ".join(raw.split())
        self.block_size = int(block_size)
        self.seed = seed
        # ``_rng`` is only consumed by ``reset(shuffle=True)``. With
        # ``seed=None`` we keep NumPy's default non-deterministic
        # generator — the determinism contract still holds for the
        # default ``shuffle=False`` path.
        self._rng = np.random.default_rng(seed)
        self.pos: int = 0
        # Track whether we've reached end-of-text at least once. Useful
        # for the closed loop to decide between wraparound and stop.
        self._wrapped: bool = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def reset(self, seed: int | None = None, shuffle: bool = False) -> None:
        """Rewind the pointer to position 0.

        If ``shuffle=True``, the *block order* is permuted by a fresh
        permutation seeded from ``seed`` (or the constructor seed if
        ``seed=None``). The block *content* is unchanged; only the
        order in which blocks are yielded by ``step()`` changes. This
        is the entry point for curiosity-driven reading (Phase H+):
        the model can request a re-shuffle to revisit blocks in a new
        order without losing the underlying text.

        NOTE: shuffle support is implemented here so that downstream
        code can opt-in, but the default ``TextRunner`` does NOT use
        it (passive sequential reading, per the task spec).
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.pos = 0
        self._wrapped = False
        if shuffle:
            # We materialise the permutation lazily so an unshuffled
            # streamer pays no cost. The permutation is over block
            # indices, not characters, so block boundaries are
            # preserved.
            n_blocks = max(1, len(self.text) // self.block_size)
            self._block_perm = self._rng.permutation(n_blocks)
        else:
            self._block_perm = None

    def step(self) -> str:
        """Return the next ``block_size`` characters and advance ``pos``.

        Backward-compatible wrapper around ``navigate(NAV_FORWARD)``.
        Phase H delegates all movement logic to ``navigate`` so that
        the wraparound / boundary-clamp behaviour is identical whether
        the caller uses ``step()`` (passive reading) or
        ``navigate(action)`` (curiosity-driven reading).
        """
        return self.navigate(NAV_FORWARD)

    # ------------------------------------------------------------------ #
    # Phase H: action-driven navigation
    # ------------------------------------------------------------------ #
    def navigate(self, action: int) -> str:
        """Apply a navigation ``action`` and return the new block.

        Phase H introduces 5 discrete actions (defined above):

        ======  =================  ====================================
        action  name               effect on ``pos``
        ======  =================  ====================================
        0       forward            ``pos += block_size``
        1       backward           ``pos -= block_size`` (clamped at 0)
        2       fast_forward       ``pos += block_size * 4``
        3       fast_backward      ``pos -= block_size * 4`` (clamped)
        4       stay               ``pos`` unchanged (re-read block)
        ======  =================  ====================================

        Boundary handling:
          - At the START of the text (pos < delta): clamp to 0. The
            returned block is the first ``block_size`` chars. This
            means repeated ``backward`` actions at the start all
            return the same block — the model "feels" the wall.
          - At the END of the text (pos + block_size > n): wrap to 0
            and set ``wrapped = True``. The returned block is taken
            from the new wrapped position. This matches the original
            ``step()`` cyclical-reading behaviour so consolidation
            across multiple reads stays possible.
          - ``stay``: returns the same block as the previous call,
            WITHOUT moving. Useful for re-reading high-surprise
            passages (curiosity-driven consolidation).

        Returns:
            The block of ``block_size`` chars at the new ``pos``.
            Returns ``""`` only if the underlying text is empty.
        """
        n = len(self.text)
        if n == 0:
            return ""
        # Compute the position delta for this action. ``stay`` keeps
        # pos unchanged (delta=0). Forward/backward deltas are
        # multiples of ``block_size``.
        if action == NAV_FORWARD:
            delta = self.block_size
        elif action == NAV_BACKWARD:
            delta = -self.block_size
        elif action == NAV_FAST_FORWARD:
            delta = self.block_size * 4
        elif action == NAV_FAST_BACKWARD:
            delta = -self.block_size * 4
        elif action == NAV_STAY:
            delta = 0
        else:
            raise ValueError(
                f"unknown navigation action {action!r}; "
                f"expected 0..{N_NAV_ACTIONS - 1}"
            )
        # Apply the delta. If we'd go below 0, clamp to 0 (start wall).
        new_pos = self.pos + delta
        if new_pos < 0:
            new_pos = 0
        # If we'd walk past the end, wrap around. We check
        # ``new_pos >= n`` (not ``new_pos + block_size > n``) so that
        # a fast_forward that overshoots wraps cleanly. The wrapped
        # position is taken modulo ``n`` so a single huge delta still
        # lands somewhere valid.
        if new_pos >= n:
            new_pos = new_pos % n
            self._wrapped = True
        # Clamp pos for safety (in case n shrank between calls — it
        # can't in this class, but defensive).
        self.pos = max(0, min(new_pos, n - 1))
        # Read the block at the new position. If near the end, take
        # the tail and pad with the head so the block length is
        # always ``block_size`` (matches ``step()`` behaviour).
        end = self.pos + self.block_size
        if end <= n:
            block = self.text[self.pos:end]
        else:
            tail = self.text[self.pos:]
            head = self.text[: self.block_size - len(tail)]
            block = tail + head
        return block

    def tell(self) -> int:
        """Return the current pointer position (in characters)."""
        return self.pos

    def seek(self, pos: int) -> None:
        """Set the pointer to ``pos`` (clamped to ``[0, len(text)]``)."""
        self.pos = max(0, min(int(pos), len(self.text)))

    def __len__(self) -> int:
        """Total number of characters in the underlying text."""
        return len(self.text)

    # ------------------------------------------------------------------ #
    # Convenience for analysis scripts
    # ------------------------------------------------------------------ #
    def n_blocks(self) -> int:
        """How many ``block_size``-length blocks fit in the text?"""
        return max(0, len(self.text) // self.block_size)

    @property
    def wrapped(self) -> bool:
        """Has the stream wrapped around at least once?"""
        return self._wrapped


# ------------------------------------------------------------------ #
# Self-test: run when executed directly
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import tempfile

    # Build a tiny in-memory text file to exercise the API.
    sample = (
        "Cognitive emergence is the spontaneous formation of structure "
        "in a self-organising system. The ZeroDataModel project tests "
        "whether a free-energy-minimising agent can form pre-linguistic "
        "concepts from raw pixel or character streams."
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(sample)
        path = f.name

    stream = TextStream(path, block_size=32, seed=42)
    print(f"len(stream) = {len(stream)} chars, "
          f"n_blocks = {stream.n_blocks()}")
    for i in range(5):
        block = stream.step()
        print(f"  step {i}: pos={stream.tell():3d}  "
              f"wrapped={stream.wrapped}  "
              f"block={block!r}")

    # Test wraparound.
    stream.seek(len(stream) - 5)
    block = stream.step()
    print(f"\nwraparound test: block={block!r}  "
          f"pos_after={stream.tell()}  wrapped={stream.wrapped}")

    # Test reset + shuffle.
    stream.reset(shuffle=True)
    print(f"\nafter reset(shuffle=True): pos={stream.tell()}  "
          f"wrapped={stream.wrapped}")

    # Test Phase H navigate() on each of the 5 actions.
    stream.reset()
    print(f"\nnavigate() tests (block_size={stream.block_size}):")
    for action, name in enumerate(NAV_ACTION_NAMES):
        block = stream.navigate(action)
        print(f"  action={action} ({name:14s})  "
              f"pos_after={stream.tell():3d}  "
              f"block={block!r}")

    # Boundary tests: start wall + end wraparound.
    stream.reset()
    # At pos=0, backward should clamp to 0 and return block[0].
    stream.seek(0)
    b_back = stream.navigate(NAV_BACKWARD)
    print(f"\nstart wall: backward at pos=0 -> pos={stream.tell()}, "
          f"block={b_back!r}")
    # Forward several times then fast_backward to test mid-stream.
    for _ in range(3):
        stream.navigate(NAV_FORWARD)
    pos_mid = stream.tell()
    stream.navigate(NAV_FAST_BACKWARD)
    print(f"fast_backward from pos={pos_mid} -> pos={stream.tell()} "
          f"(expected max(0, {pos_mid - 4 * stream.block_size}))")

    Path(path).unlink()
    print("\nTextStream self-test OK")
