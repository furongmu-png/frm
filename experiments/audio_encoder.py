# experiments/audio_encoder.py
"""Audio encoder for the omnimodal phase (Phase K, Task 1.2).

A ZERO-PRETRAIN audio encoder with two complementary pathways:

  Path A — Spectrogram pathway (default):
    1. Load a WAV file (or accept an ndarray waveform).
    2. Compute a log-Mel-ish spectrogram with ``scipy.signal.spectrogram``
       (we approximate Mel scaling with a fixed random filter bank, since
       ``librosa`` is not assumed available).
    3. Resize the (freq, time) spectrogram to a fixed (n_mels, n_frames)
       canvas, flatten, standardise (zero-mean, unit-norm).
    4. Project through a FIXED random Gaussian matrix to ``output_dim``.

  Path B — Raw-waveform pathway (lightweight fallback):
    1. Resample / pad / truncate the waveform to ``frame_size`` samples.
    2. Apply ``n_filters`` fixed random 1D-conv filters (length
       ``filter_len``) with stride, take the temporal mean per filter.
    3. Project through a fixed random matrix to ``output_dim``.

Both pathways share the same Johnson-Lindenstrauss spirit as the image
encoder: a fixed random projection reduces a high-dimensional modality
to the model's ``dim``, leaving the structure-discovery to downstream
belief updating.

Why two pathways?
  - Spectrogram captures time-frequency structure (good for music / speech).
  - Raw-waveform is cheaper and robust when SciPy is unavailable.

An ``AudioStream`` class loads WAV files (via ``wave`` from the stdlib,
so no external audio dependency is required) and yields encoded vectors
+ the source path. Synthesised tone/noise generators are provided for
smoke runs without real audio corpora.

Determinism: with a fixed ``seed``, all random projection matrices
and filter banks are identical across runs, so the same WAV file →
same vector.
"""

from __future__ import annotations

import math
import sys
import wave
from pathlib import Path

import numpy as np

# SciPy's spectrogram — used only for Path A.
try:
    from scipy.signal import spectrogram as _scipy_spectrogram
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False


DEFAULT_OUTPUT_DIM = 32          # match ZeroDataModel.dim
DEFAULT_PROJ_DIM = 256           # internal projection dim
DEFAULT_N_MELS = 32             # spectrogram freq axis after Mel-ish bank
DEFAULT_N_FRAMES = 64           # spectrogram time axis
DEFAULT_FRAME_SIZE = 16000      # raw-waveform pathway sample count
DEFAULT_FILTER_LEN = 64         # raw-waveform pathway conv filter length
DEFAULT_N_FILTERS = 64          # raw-waveform pathway conv filter count
SUPPORTED_EXTS = {".wav", ".wave"}


# ------------------------------------------------------------------ #
# AudioEncoder
# ------------------------------------------------------------------ #
class AudioEncoder:
    """Fixed random-projection encoder for mono audio.

    Parameters
    ----------
    output_dim : int
        Dimensionality of the output vector. Must match the model's dim.
    proj_dim : int
        Internal projection width (intermediate between spectrogram
        size and ``output_dim``). Acts as a Johnson-Lindenstrauss reducer.
    n_mels : int
        Number of Mel-ish frequency bins after the random filter bank.
    n_frames : int
        Number of time frames after resampling the spectrogram.
    frame_size : int
        Sample count used by the raw-waveform pathway.
    filter_len : int
        Length of the 1D random conv filters in the raw-waveform pathway.
    n_filters : int
        Number of 1D random conv filters in the raw-waveform pathway.
    seed : int
        Seed for all random matrices / filter banks.
    pathway : str
        ``"spectrogram"`` (default) or ``"waveform"``. Selects which
        pathway the encoder uses. ``"auto"`` prefers spectrogram and
        falls back to waveform if SciPy is unavailable.
    """

    def __init__(
        self,
        output_dim: int = DEFAULT_OUTPUT_DIM,
        proj_dim: int = DEFAULT_PROJ_DIM,
        n_mels: int = DEFAULT_N_MELS,
        n_frames: int = DEFAULT_N_FRAMES,
        frame_size: int = DEFAULT_FRAME_SIZE,
        filter_len: int = DEFAULT_FILTER_LEN,
        n_filters: int = DEFAULT_N_FILTERS,
        seed: int = 42,
        pathway: str = "auto",
    ):
        self.output_dim = int(output_dim)
        self.proj_dim = int(proj_dim)
        self.n_mels = int(n_mels)
        self.n_frames = int(n_frames)
        self.frame_size = int(frame_size)
        self.filter_len = int(filter_len)
        self.n_filters = int(n_filters)
        self.seed = int(seed)
        # Resolve pathway.
        if pathway == "auto":
            pathway = "spectrogram" if _HAS_SCIPY else "waveform"
        if pathway not in ("spectrogram", "waveform"):
            raise ValueError(f"unknown pathway {pathway!r}")
        if pathway == "spectrogram" and not _HAS_SCIPY:
            pathway = "waveform"
        self.pathway = pathway

        rng = np.random.default_rng(seed)
        # Spectrogram pathway: project a (n_mels * n_frames) flat vector
        # through proj_dim → output_dim.
        spec_flat = self.n_mels * self.n_frames
        # Random Mel-ish filter bank: maps scipy's raw freq bins to
        # ``n_mels`` bins. We don't know the number of freq bins in
        # advance (depends on signal length), so we use a SOFT bank
        # applied after spectrogram computation by resampling.
        # We just keep a single projection matrix for the resampled
        # spectrogram.
        self._spec_proj1 = rng.standard_normal((self.proj_dim, spec_flat)) / np.sqrt(spec_flat)
        self._spec_proj2 = rng.standard_normal((self.output_dim, self.proj_dim)) / np.sqrt(self.proj_dim)
        # Waveform pathway: n_filters random 1D conv filters of length
        # filter_len, plus a projection from n_filters → output_dim.
        self._wav_filters = rng.standard_normal((self.n_filters, self.filter_len)) / np.sqrt(self.filter_len)
        self._wav_proj = rng.standard_normal((self.output_dim, self.n_filters)) / np.sqrt(self.n_filters)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def encode(self, audio_or_path, sr: int | None = None) -> np.ndarray:
        """Encode a waveform (ndarray / WAV path) -> (output_dim,) vector.

        Parameters
        ----------
        audio_or_path : np.ndarray | str | Path
            Mono waveform as a 1D float ndarray in [-1, 1], or a path
            to a WAV file. Multi-channel WAVs are mixed to mono.
        sr : int, optional
            Sample rate of ``audio_or_path`` if it is an ndarray. If
            the rate differs from the encoder's expected rate
            (DEFAULT_FRAME_SIZE samples per "frame"), the signal is
            resampled by naive decimation / interpolation.
        """
        wav = self._to_mono_float(audio_or_path)
        if wav.size == 0:
            return np.zeros(self.output_dim, dtype=np.float64)
        if self.pathway == "spectrogram":
            return self._encode_spectrogram(wav)
        return self._encode_waveform(wav)

    # ------------------------------------------------------------------ #
    # Spectrogram pathway
    # ------------------------------------------------------------------ #
    def _encode_spectrogram(self, wav: np.ndarray) -> np.ndarray:
        # Compute STFT via scipy.signal.spectrogram. We use the
        # default Hann window; nperseg scales with the signal length
        # but is capped to avoid empty spectrograms on short signals.
        nperseg = min(256, max(16, len(wav) // 8))
        if nperseg >= len(wav):
            nperseg = max(4, len(wav))
        f, t, Sxx = _scipy_spectrogram(
            wav, fs=1.0, nperseg=nperseg, noverlap=max(1, nperseg // 2)
        )
        # Sxx: (n_freq, n_time). Convert to log magnitude (add eps).
        log_spec = np.log10(Sxx + 1e-12)
        # Resample to (n_mels, n_frames) via nearest-neighbour.
        spec = self._resize_2d(log_spec, self.n_mels, self.n_frames)
        flat = spec.flatten().astype(np.float64)
        flat -= flat.mean()
        n = float(np.linalg.norm(flat))
        if n > 1e-12:
            flat /= n
        h = np.tanh(self._spec_proj1 @ flat)
        out = self._spec_proj2 @ h
        return out.astype(np.float64)

    # ------------------------------------------------------------------ #
    # Waveform pathway (no SciPy required)
    # ------------------------------------------------------------------ #
    def _encode_waveform(self, wav: np.ndarray) -> np.ndarray:
        # Pad / truncate to frame_size.
        x = self._pad_truncate(wav.astype(np.float64), self.frame_size)
        # Normalise.
        n = float(np.linalg.norm(x))
        if n > 1e-12:
            x = x / n
        # Apply n_filters random 1D conv filters with mean-pooling.
        # Valid-mode convolution via stride tricks.
        conv_out = np.empty(self.n_filters, dtype=np.float64)
        fl = self.filter_len
        # Build sliding-window view (frame_size - fl + 1, fl).
        n_windows = x.shape[0] - fl + 1
        if n_windows <= 0:
            # Very short signal — just dot once with zero-padded input.
            xp = np.zeros(fl, dtype=np.float64)
            xp[: x.shape[0]] = x
            conv_out = self._wav_filters @ xp
        else:
            # strided view: shape (n_windows, fl)
            windows = np.lib.stride_tricks.sliding_window_view(x, fl)
            # (n_filters, n_windows) = filters @ windows.T
            conv = self._wav_filters @ windows.T
            # Absolute-mean pool over time. A plain ``mean`` collapses
            # periodic signals (sine waves have zero DC component) and
            # yields ~0; |·|.mean() retains the per-filter activation
            # magnitude regardless of phase — a much more informative
            # summary for non-pretrained random-filter banks.
            conv_out = np.abs(conv).mean(axis=1)
        out = self._wav_proj @ conv_out
        return out.astype(np.float64)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _to_mono_float(self, audio_or_path) -> np.ndarray:
        if isinstance(audio_or_path, np.ndarray):
            arr = audio_or_path.astype(np.float64, copy=False)
            if arr.ndim == 2:
                # (n_samples, n_channels) → mono by mean.
                arr = arr.mean(axis=1)
            # Assume already in [-1, 1]; do not rescale.
            return arr
        # Treat as path.
        path = Path(audio_or_path)
        if not path.exists():
            raise FileNotFoundError(f"audio file not found: {path}")
        return self._load_wav(path)

    @staticmethod
    def _load_wav(path: Path) -> np.ndarray:
        """Load a WAV file via the stdlib ``wave`` module."""
        with wave.open(str(path), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
        if not raw:
            return np.zeros(0, dtype=np.float64)
        # Map sampwidth → numpy dtype.
        if sampwidth == 1:
            dt = np.uint8
            scale = 128.0
            offset = 128.0
        elif sampwidth == 2:
            dt = np.int16
            scale = 32768.0
            offset = 0.0
        elif sampwidth == 4:
            dt = np.int32
            scale = 2147483648.0
            offset = 0.0
        else:
            raise ValueError(f"unsupported WAV sampwidth: {sampwidth}")
        arr = np.frombuffer(raw, dtype=dt).astype(np.float64)
        arr = arr.reshape(-1, n_channels)
        if offset:
            arr -= offset
        arr /= scale
        # Mix to mono.
        if arr.shape[1] > 1:
            arr = arr.mean(axis=1)
        else:
            arr = arr[:, 0]
        return arr

    @staticmethod
    def _pad_truncate(x: np.ndarray, n: int) -> np.ndarray:
        if x.shape[0] >= n:
            return x[:n]
        out = np.zeros(n, dtype=x.dtype)
        out[: x.shape[0]] = x
        return out

    @staticmethod
    def _resize_2d(arr: np.ndarray, h: int, w: int) -> np.ndarray:
        """Nearest-neighbour resize of a 2D array to (h, w)."""
        H, W = arr.shape
        if H == 0 or W == 0:
            return np.zeros((h, w), dtype=arr.dtype)
        ys = (np.arange(h) * H / h).astype(int)
        xs = (np.arange(w) * W / w).astype(int)
        return arr[np.ix_(ys, xs)]


# ------------------------------------------------------------------ #
# Synthetic audio generators
# ------------------------------------------------------------------ #
def make_synthetic_audio(
    kind: str = "sine",
    duration_s: float = 1.0,
    sr: int = 16000,
    freq: float = 440.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a synthetic mono waveform (float64 in [-1, 1]).

    Kinds:
      - ``sine`` : pure-tone sine wave at ``freq`` Hz.
      - ``chord`` : sum of 3 sine waves (root + major third + fifth).
      - ``noise`` : white Gaussian noise.
      - ``chirp`` : linear chirp from ``freq`` to ``2*freq``.
      - ``click`` : impulse train (sparse 1.0 spikes) at 8 Hz.
    """
    n = int(duration_s * sr)
    t = np.arange(n) / sr
    if kind == "sine":
        return np.sin(2 * np.pi * freq * t)
    if kind == "chord":
        return (np.sin(2 * np.pi * freq * t)
                + np.sin(2 * np.pi * freq * 1.25 * t)
                + np.sin(2 * np.pi * freq * 1.5 * t)) / 3.0
    if kind == "noise":
        rng = np.random.default_rng(seed)
        return rng.standard_normal(n) * 0.3
    if kind == "chirp":
        f1, f2 = freq, 2.0 * freq
        phase = 2 * np.pi * (f1 * t + (f2 - f1) * t * t / (2 * duration_s))
        return np.sin(phase)
    if kind == "click":
        out = np.zeros(n, dtype=np.float64)
        click_period = max(1, int(sr / 8.0))   # 8 Hz impulse train
        out[::click_period] = 1.0
        return out
    raise ValueError(f"unknown kind {kind!r}")


def make_synthetic_audio_dir(
    output_dir: Path,
    sr: int = 16000,
    duration_s: float = 1.0,
    n_per_kind: int = 3,
    kinds: tuple[str, ...] = ("sine", "chord", "noise", "chirp", "click"),
    seed: int = 42,
) -> list[Path]:
    """Write a synthetic audio dataset to ``output_dir`` (WAV files).

    Returns the list of saved WAV paths. Uses the stdlib ``wave`` module
    so no external dependency is needed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = []
    # Frequency choices spanning low / mid / high registers.
    freqs = [220.0, 440.0, 880.0]
    for kind in kinds:
        for i in range(n_per_kind):
            freq = freqs[i % len(freqs)]
            wav = make_synthetic_audio(
                kind=kind, duration_s=duration_s, sr=sr,
                freq=freq,
                seed=int(rng.integers(0, 2 ** 31 - 1)),
            )
            fname = f"{kind}_{i:02d}_{int(freq)}hz.wav"
            path = output_dir / fname
            _save_wav(path, wav, sr)
            paths.append(path)
    return paths


def _save_wav(path: Path, wav: np.ndarray, sr: int) -> None:
    """Write a mono float64 waveform to a 16-bit PCM WAV file."""
    # Clip + scale to int16.
    clipped = np.clip(wav, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


# ------------------------------------------------------------------ #
# AudioStream
# ------------------------------------------------------------------ #
class AudioStream:
    """Stream WAV files from a directory, encoding each on demand.

    Iteration order is deterministic (sorted by filename).
    """

    def __init__(
        self,
        audio_dir: str | Path,
        encoder: AudioEncoder,
        shuffle: bool = False,
        seed: int = 42,
    ):
        self.audio_dir = Path(audio_dir)
        self.encoder = encoder
        self._paths = sorted(
            p for p in self.audio_dir.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTS and p.is_file()
        )
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(self._paths)
        self._idx = 0

    def __len__(self) -> int:
        return len(self._paths)

    def reset(self) -> None:
        self._idx = 0

    def step(self) -> tuple[np.ndarray, str]:
        """Return the next (encoded_vector, path) pair. Wraps around."""
        if not self._paths:
            raise RuntimeError("no audio files in directory")
        path = self._paths[self._idx % len(self._paths)]
        self._idx += 1
        return self.encoder.encode(path), str(path)

    def random(self) -> tuple[np.ndarray, str]:
        if not self._paths:
            raise RuntimeError("no audio files in directory")
        rng = np.random.default_rng(None)
        path = self._paths[int(rng.integers(0, len(self._paths)))]
        return self.encoder.encode(path), str(path)


# ------------------------------------------------------------------ #
# Self-test
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import tempfile

    enc = AudioEncoder(output_dim=32, seed=42)
    print(f"[self-test] encoder built (pathway={enc.pathway})")
    print(f"  spec_proj1={enc._spec_proj1.shape} spec_proj2={enc._spec_proj2.shape}")
    print(f"  wav_filters={enc._wav_filters.shape} wav_proj={enc._wav_proj.shape}")

    # Encode two synthetic tones.
    a = make_synthetic_audio("sine", freq=440.0, seed=0)
    b = make_synthetic_audio("sine", freq=880.0, seed=1)
    va = enc.encode(a)
    vb = enc.encode(b)
    print(f"  encoded 440Hz sine: shape={va.shape}  norm={np.linalg.norm(va):.4f}")
    print(f"  encoded 880Hz sine: shape={vb.shape}  norm={np.linalg.norm(vb):.4f}")
    cos = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))
    print(f"  cos(440, 880)={cos:.4f}")

    # White noise should be more dissimilar to a pure tone.
    noise = make_synthetic_audio("noise", seed=2)
    vn = enc.encode(noise)
    cos_n = float(np.dot(va, vn) / (np.linalg.norm(va) * np.linalg.norm(vn) + 1e-12))
    print(f"  cos(440, noise)={cos_n:.4f}")

    # Determinism check.
    va_again = enc.encode(make_synthetic_audio("sine", freq=440.0, seed=0))
    assert np.allclose(va, va_again), "encoder should be deterministic given seed"
    print("[self-test] determinism: OK")

    # Synthetic dir + AudioStream.
    tmp = Path(tempfile.mkdtemp())
    paths = make_synthetic_audio_dir(tmp, n_per_kind=2, seed=42)
    print(f"[self-test] wrote {len(paths)} synthetic WAVs to {tmp}")
    stream = AudioStream(tmp, enc, seed=42)
    print(f"  stream len = {len(stream)}")
    v, p = stream.step()
    print(f"  first: {Path(p).name}  vec norm = {np.linalg.norm(v):.4f}")

    # Waveform pathway self-test.
    enc_w = AudioEncoder(output_dim=32, seed=42, pathway="waveform")
    vw = enc_w.encode(a)
    print(f"[self-test] waveform pathway: shape={vw.shape}  norm={np.linalg.norm(vw):.4f}")

    print("[self-test] OK")
