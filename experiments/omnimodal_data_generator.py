# experiments/omnimodal_data_generator.py
"""Omni-modal data generator (Phase K, Task 1.4).

A single multi-stream sampler that wraps:

  - The Phase H ``TextStream`` (curiosity-driven text reading).
  - The Phase J ``EncyclopediaReader`` (large-corpus articles).
  - The Phase K ``ImageStream``, ``AudioStream``, ``CodeStream`` (this phase).
  - A physics-sandbox adapter (re-uses ``run_sandbox_curious`` infrastructure).

The generator's primary role is to provide ``get_batch(modality)`` and
``sample(modality=None)`` methods that return (encoded_vector, modality,
source_label) tuples suitable for feeding into ZeroDataModel's cognitive
cycle. The modality mix is configurable per the Phase K spec:

    text        60 %   (default)
    physics     20 %
    image       10 %
    audio        5 %
    code         5 %

Design philosophy
------------------
This module is a thin ORCHESTRATOR. Each modality stream owns its own
encoder; the generator only decides WHICH stream to sample from next
and forwards the encoded vector. Streams are LIVED (they advance their
internal pointer on each step) so a long-running generator produces a
realistic temporal pattern.

The generator is deliberately ROBUST to missing modalities:
  - If a stream is None or its directory is empty, the generator silently
    re-distributes that modality's probability mass to the others.
  - This lets the smoke test run without every corpus pre-installed.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np


# ------------------------------------------------------------------ #
# Modality labels (strings used as the ``modality`` field in returned tuples)
# ------------------------------------------------------------------ #
MOD_TEXT    = "text"
MOD_PHYSICS = "physics"
MOD_IMAGE   = "image"
MOD_AUDIO   = "audio"
MOD_CODE    = "code"

DEFAULT_RATIOS = {
    MOD_TEXT:    0.60,
    MOD_PHYSICS: 0.20,
    MOD_IMAGE:   0.10,
    MOD_AUDIO:   0.05,
    MOD_CODE:    0.05,
}


# ------------------------------------------------------------------ #
# Sample record
# ------------------------------------------------------------------ #
@dataclass
class ModalitySample:
    """One observation returned by the generator."""
    vector: np.ndarray         # (dim,) encoded observation
    modality: str              # one of MOD_*
    source: str                # human-readable label (path / article / step)
    metadata: dict = field(default_factory=dict)  # modality-specific extras


# ------------------------------------------------------------------ #
# OmniModalGenerator
# ------------------------------------------------------------------ #
class OmniModalGenerator:
    """Multi-stream sampler with weighted modality mix.

    Parameters
    ----------
    streams : dict[str, object]
        Each value must implement ``.step() -> (vector, source)`` and
        ``.__len__``. May be ``None`` for absent modalities.
    ratios : dict[str, float]
        Per-modality sampling probability. Need not sum to 1; values
        are normalised. Modalities with stream=None are dropped and
        their mass is redistributed proportionally.
    seed : int
        Seed for the choice RNG (so the modality mix is reproducible).
    """

    def __init__(
        self,
        streams: Optional[dict] = None,
        ratios: Optional[dict] = None,
        seed: int = 42,
    ):
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        streams = dict(streams or {})
        # Drop missing streams. We DON'T drop len()==0 streams because
        # some streams are unbounded (physics sandbox) or may report
        # len=0 for legitimate reasons (e.g. an encyclopedia reader
        # whose ``__len__`` semantics differ from a list's).
        active = {}
        for k, v in streams.items():
            if v is None:
                continue
            active[k] = v
        self.streams = active
        # Resolve ratios.
        ratios = dict(ratios or DEFAULT_RATIOS)
        # Keep only active modalities (so missing streams get 0 mass).
        active_ratios = {k: float(ratios.get(k, 0.0)) for k in active}
        total = sum(active_ratios.values())
        if total <= 0:
            # Uniform fallback.
            active_ratios = {k: 1.0 / len(active) for k in active}
            total = 1.0
        # Normalise.
        self.ratios = {k: v / total for k, v in active_ratios.items()}
        self._modalities = list(self.ratios.keys())
        self._probs = np.array([self.ratios[m] for m in self._modalities], dtype=np.float64)
        # Bookkeeping.
        self.step_count = 0
        self.modality_counts: dict[str, int] = {m: 0 for m in self._modalities}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def sample(self) -> ModalitySample:
        """Draw one sample from a randomly-chosen modality.

        The modality is chosen according to the weighted ratios.
        Returns a ``ModalitySample`` with the encoded vector + metadata.
        """
        if not self._modalities:
            raise RuntimeError("no active streams in the generator")
        modality = self._choose_modality()
        return self.get_batch(modality)

    def get_batch(self, modality: str) -> ModalitySample:
        """Sample the next observation from a specific modality."""
        if modality not in self.streams:
            raise KeyError(f"modality {modality!r} not active; "
                           f"choose from {self._modalities}")
        stream = self.streams[modality]
        # Streams all expose .step() -> (vector, source).
        try:
            vector, source = stream.step()
        except Exception as exc:  # noqa: BLE001 — defensive
            # If the stream blows up, fall back to a zero vector with
            # an error marker in the source. This keeps the consumer
            # (the learning loop) running.
            return ModalitySample(
                vector=np.zeros(0, dtype=np.float64),
                modality=modality,
                source=f"<error:{type(exc).__name__}>",
                metadata={"error": str(exc)},
            )
        self.modality_counts[modality] += 1
        self.step_count += 1
        return ModalitySample(
            vector=np.asarray(vector, dtype=np.float64),
            modality=modality,
            source=str(source),
            metadata={"step": self.step_count},
        )

    def reset(self) -> None:
        """Reset every active stream's internal pointer."""
        for stream in self.streams.values():
            try:
                stream.reset()
            except Exception:
                pass
        self.step_count = 0
        self.modality_counts = {m: 0 for m in self._modalities}

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def active_modalities(self) -> list[str]:
        return list(self._modalities)

    def effective_ratios(self) -> dict[str, float]:
        return dict(self.ratios)

    def __repr__(self) -> str:
        return (f"OmniModalGenerator(active={self._modalities}, "
                f"ratios={self.ratios}, steps={self.step_count})")

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #
    def _choose_modality(self) -> str:
        # np.random.Generator.choice over a string array.
        idx = int(self._rng.choice(len(self._modalities), p=self._probs))
        return self._modalities[idx]


# ------------------------------------------------------------------ #
# Factory: build the default generator from existing infrastructure
# ------------------------------------------------------------------ #
def build_default_generator(
    text_stream=None,
    encyclopedia_reader=None,
    physics_stream=None,
    image_dir: str | Path | None = None,
    audio_dir: str | Path | None = None,
    code_dir: str | Path | None = None,
    output_dim: int = 32,
    ratios: Optional[dict] = None,
    seed: int = 42,
) -> OmniModalGenerator:
    """Wire together an OmniModalGenerator from the Phase K encoders.

    Parameters
    ----------
    text_stream : object, optional
        A stream with ``.step() -> (vec, src)`` for the text modality.
        Typically a Phase H ``TextStream`` wrapped via an adapter.
    encyclopedia_reader : object, optional
        A Phase J ``EncyclopediaReader`` (or a navigator wrapping it).
        Used as the text stream if ``text_stream`` is None.
    physics_stream : object, optional
        A stream for the physics-sandbox modality. Usually an adapter
        over ``run_sandbox_curious`` that yields (vec, src) tuples.
    image_dir / audio_dir / code_dir : path or None
        Directories containing the corresponding modality files.
    output_dim : int
        Encoder output dim. Must match the model's ``dim``.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from image_encoder import ImageEncoder, ImageStream, make_synthetic_image_dir
    from audio_encoder import AudioEncoder, AudioStream, make_synthetic_audio_dir
    from code_encoder import CodeEncoder, CodeStream, make_synthetic_code_dir

    streams: dict = {}

    # --- Text modality ----------------------------------------------- #
    if text_stream is not None:
        streams[MOD_TEXT] = text_stream
    elif encyclopedia_reader is not None:
        streams[MOD_TEXT] = EncyclopediaStreamAdapter(encyclopedia_reader)

    # --- Physics modality -------------------------------------------- #
    if physics_stream is not None:
        streams[MOD_PHYSICS] = physics_stream

    # --- Image modality ---------------------------------------------- #
    image_encoder = ImageEncoder(output_dim=output_dim, seed=seed)
    if image_dir is not None:
        try:
            streams[MOD_IMAGE] = ImageStream(image_dir, image_encoder, seed=seed)
        except Exception:
            pass
    if MOD_IMAGE not in streams:
        # Fall back to a synthetic image dir.
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="omni_img_"))
        make_synthetic_image_dir(tmp, n_per_shape=3, seed=seed)
        streams[MOD_IMAGE] = ImageStream(tmp, image_encoder, seed=seed)

    # --- Audio modality ----------------------------------------------- #
    audio_encoder = AudioEncoder(output_dim=output_dim, seed=seed)
    if audio_dir is not None:
        try:
            streams[MOD_AUDIO] = AudioStream(audio_dir, audio_encoder, seed=seed)
        except Exception:
            pass
    if MOD_AUDIO not in streams:
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="omni_aud_"))
        make_synthetic_audio_dir(tmp, n_per_kind=2, seed=seed)
        streams[MOD_AUDIO] = AudioStream(tmp, audio_encoder, seed=seed)

    # --- Code modality ------------------------------------------------ #
    code_encoder = CodeEncoder(output_dim=output_dim, seed=seed)
    if code_dir is not None:
        try:
            streams[MOD_CODE] = CodeStream(code_dir, code_encoder, seed=seed)
        except Exception:
            pass
    if MOD_CODE not in streams:
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="omni_code_"))
        make_synthetic_code_dir(tmp, n_per_snippet=2, seed=seed)
        streams[MOD_CODE] = CodeStream(tmp, code_encoder, seed=seed)

    return OmniModalGenerator(streams=streams, ratios=ratios, seed=seed)


# ------------------------------------------------------------------ #
# Adapters — turn Phase H / J infrastructure into "streams"
# ------------------------------------------------------------------ #
class EncyclopediaStreamAdapter:
    """Adapt an ``EncyclopediaReader`` to the ``.step()`` interface.

    The adapter reads the next text block from the reader and encodes
    it via a TextEncoder. It also exposes the article title in the
    source label so the consumer (knowledge graph builder) can use it.
    """

    def __init__(
        self,
        reader,
        encoder=None,
        block_size: int = 512,
        output_dim: int = 32,
        seed: int = 42,
    ):
        self.reader = reader
        if encoder is None:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from text_encoder import TextEncoder
            encoder = TextEncoder(block_size=block_size, output_dim=output_dim, seed=seed)
        self.encoder = encoder
        self._idx = 0
        self._current_title = ""

    def __len__(self) -> int:
        # Number of articles in the reader's index (or 0 if unknown).
        try:
            return int(self.reader.n_articles())
        except Exception:
            return 0

    def reset(self) -> None:
        self._idx = 0

    def step(self) -> tuple[np.ndarray, str]:
        # Read the next block from the reader.
        block = self.reader.read_block()
        if not block:
            # Wrap around: seek to a random article.
            try:
                self.reader.seek_to_random_article()
            except Exception:
                pass
            block = self.reader.read_block() or ""
        # Capture the article title via the reader's public API.
        title = ""
        for method in ("get_title", "current_title"):
            fn = getattr(self.reader, method, None)
            if callable(fn):
                try:
                    title = str(fn() or "")
                    if title:
                        break
                except Exception:
                    pass
        self._current_title = title
        vec = self.encoder.encode(block)
        source = f"wiki:{title}" if title else "wiki:?"
        return vec, source


class PhysicsStreamAdapter:
    """Adapt a physics sandbox step to the ``.step()`` interface.

    The adapter calls the sandbox's ``step(action)`` method (with a
    random or model-driven action) and returns the observation vector.
    For the omni-modal generator's purposes, we use a RANDOM action —
    the model itself is the one driving the sandbox via the active
    inference engine in ``run_omni_learning``. This adapter is only
    used when the omni generator samples a physics observation
    INDEPENDENTLY of the model's chosen action.
    """

    def __init__(self, sandbox, action_dim: int = 4, seed: int = 42):
        self.sandbox = sandbox
        self.action_dim = int(action_dim)
        self._rng = np.random.default_rng(seed)
        self._idx = 0

    def __len__(self) -> int:
        return 0  # unbounded

    def reset(self) -> None:
        self._idx = 0
        try:
            self.sandbox.reset()
        except Exception:
            pass

    def step(self) -> tuple[np.ndarray, str]:
        # Random action.
        action = self._rng.standard_normal(self.action_dim) * 0.5
        try:
            obs, info = self.sandbox.step(action)
        except Exception:
            obs = np.zeros(self.action_dim, dtype=np.float64)
            info = {}
        self._idx += 1
        source = f"physics:{self._idx}"
        return np.asarray(obs, dtype=np.float64), source


# ------------------------------------------------------------------ #
# Self-test
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    print("[self-test] building default generator (synthetic modalities only)...")
    gen = build_default_generator(output_dim=32, seed=42)
    print(f"  {gen}")
    print(f"  active modalities: {gen.active_modalities}")
    print(f"  effective ratios : {gen.effective_ratios()}")

    # Draw 20 samples and check the modality distribution roughly
    # matches the configured ratios.
    samples = []
    for _ in range(200):
        s = gen.sample()
        samples.append(s)
        assert s.vector.shape == (32,), f"bad vector shape: {s.vector.shape}"
        assert s.modality in gen.active_modalities
        assert isinstance(s.source, str) and s.source

    counts = gen.modality_counts
    total = sum(counts.values())
    print(f"[self-test] drew {total} samples. Per-modality:")
    for mod in gen.active_modalities:
        c = counts.get(mod, 0)
        ratio = c / total if total > 0 else 0.0
        target = gen.ratios[mod]
        print(f"  {mod:8s} n={c:3d}  actual={ratio:.3f}  target={target:.3f}")

    # Verify get_batch(modality) works for each modality.
    for mod in gen.active_modalities:
        s = gen.get_batch(mod)
        assert s.modality == mod
        print(f"  get_batch({mod!r}) -> source={s.source!r}  norm={np.linalg.norm(s.vector):.4f}")

    # Test error recovery: replace one stream with a broken one.
    class BrokenStream:
        def __len__(self): return 5
        def reset(self): pass
        def step(self): raise RuntimeError("simulated failure")
    gen.streams[MOD_TEXT] = BrokenStream()
    s = gen.get_batch(MOD_TEXT)
    assert s.source.startswith("<error"), f"expected error marker, got {s.source}"
    print(f"[self-test] error recovery: OK  (source={s.source!r})")

    print("[self-test] OK")
