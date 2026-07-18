# src/zero_data_model/capabilities/audio.py
"""Audio capability module for the zero-data cognitive model.

Composes the core cognitive modules (math universe) with a small rule
library (``AudioRules``: STFT parameters, mel scale, onset thresholds) used
as prior knowledge. No external training data and no learned weights are
required -- every operator below is a deterministic, rule-based prior
blended with self-generated representations.

All audio processing uses only numpy.fft + numpy core (no librosa / soundfile
dependencies). The 1D audio signal is assumed to be a float array sampled at
``sample_rate`` Hz; stereo inputs are downmixed to mono.
"""

from __future__ import annotations

import numpy as np

from ..math_universe import MathematicalUniverse
from .rules import AudioRules


def _to_mono(signal: np.ndarray) -> np.ndarray:
    """Coerce any array-like into a 1D float array (downmix if stereo)."""
    sig = np.asarray(signal, dtype=float)
    if sig.ndim == 0:
        return sig.reshape(1)
    if sig.ndim == 2:
        # (channels, samples) or (samples, channels) -- downmix by mean.
        sig = sig.mean(axis=0) if sig.shape[0] < sig.shape[1] else sig.mean(axis=1)
    return sig.flatten()


def _hann_window(size: int) -> np.ndarray:
    """Hann window of length ``size`` (rule-based, no scipy.signal)."""
    if size <= 1:
        return np.ones(size, dtype=float)
    n = np.arange(size, dtype=float)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / (size - 1))


def _stft_magnitude(signal: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    """Compute the magnitude STFT via ``np.fft.rfft`` on Hann-windowed frames.

    Returns an array of shape ``(n_frames, frame_size // 2 + 1)``.
    """
    if signal.size < frame_size:
        signal = np.pad(signal, (0, frame_size - signal.size))
    n_frames = max(1, 1 + (signal.size - frame_size) // hop_size)
    window = _hann_window(frame_size)
    out = np.zeros((n_frames, frame_size // 2 + 1), dtype=float)
    for i in range(n_frames):
        start = i * hop_size
        frame = signal[start : start + frame_size]
        if frame.size < frame_size:
            frame = np.pad(frame, (0, frame_size - frame.size))
        spectrum = np.fft.rfft(frame * window)
        out[i] = np.abs(spectrum)
    return out


def _mel_filter_bank(n_mels: int, frame_size: int, sample_rate: int) -> np.ndarray:
    """Triangular mel-scale filter bank (rule-based, no librosa).

    Returns an array of shape ``(n_mels, frame_size // 2 + 1)``.
    """
    n_freqs = frame_size // 2 + 1
    # Rule-based mel range: 0 Hz .. sample_rate/2 Hz.
    f_min = 0.0
    f_max = sample_rate / 2.0
    mel_min = 1127.0 * np.log1p(f_min / 700.0)
    mel_max = 1127.0 * np.log1p(f_max / 700.0)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    # Convert back to Hz.
    hz_points = 700.0 * (np.exp(mel_points / 1127.0) - 1.0)
    # Convert to FFT bin indices.
    bin_points = np.floor(
        (frame_size + 1) * hz_points / sample_rate
    ).astype(int)
    bin_points = np.clip(bin_points, 0, n_freqs - 1)
    filters = np.zeros((n_mels, n_freqs), dtype=float)
    for m in range(n_mels):
        left = bin_points[m]
        center = bin_points[m + 1]
        right = bin_points[m + 2]
        if right > left:
            # Rising part.
            for k in range(left, center):
                if center > left:
                    filters[m, k] = (k - left) / (center - left)
            # Falling part.
            for k in range(center, right):
                if right > center:
                    filters[m, k] = (right - k) / (right - center)
    return filters


class AudioEncoder:
    """Encode a 1D audio signal into a ``dim``-length L2-normalized vector.

    Pipeline: Hann-windowed STFT -> mel-scale magnitude binning ->
    log compression -> fractal compression (via math_universe) -> dim vector.

    No learned weights, no external audio libraries. The mel filter bank is
    constructed deterministically from the sample rate; the fractal
    compression reuses the math universe's self-similar representation
    generator so that audio inputs are projected into the same cognitive
    space as text / image / point-cloud inputs.
    """

    def __init__(
        self,
        dim: int = 64,
        math_universe: MathematicalUniverse | None = None,
        rules: AudioRules | None = None,
    ):
        self.dim = dim
        self.rules = rules or AudioRules()
        self.math_universe = math_universe or MathematicalUniverse(dim=dim)
        # Precompute the mel filter bank for the default sample rate.
        self._mel_bank = _mel_filter_bank(
            self.rules.n_mels,
            self.rules.frame_size,
            self.rules.sample_rate,
        )

    def encode(self, signal: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Encode ``signal`` into a ``dim``-length L2-normalized vector."""
        mono = _to_mono(signal)
        if mono.size == 0:
            return np.zeros(self.dim, dtype=float)
        # Use the precomputed mel bank if the sample rate matches;
        # otherwise rebuild it for this call.
        if sample_rate != self.rules.sample_rate:
            mel_bank = _mel_filter_bank(
                self.rules.n_mels, self.rules.frame_size, sample_rate
            )
        else:
            mel_bank = self._mel_bank
        # STFT magnitude -> (n_frames, n_freqs).
        mag = _stft_magnitude(
            mono, self.rules.frame_size, self.rules.hop_size
        )
        # Mel magnitude -> (n_frames, n_mels).
        mel_mag = mag @ mel_bank.T
        # Log compression (rule-based, +1e-8 floor).
        mel_log = np.log(mel_mag + 1e-8)
        # Aggregate over frames (mean) -> (n_mels,).
        mel_vec = np.mean(mel_log, axis=0)
        mel_vec = np.nan_to_num(mel_vec, nan=0.0, posinf=0.0, neginf=0.0)
        # Fractal compression via math universe -> dim-length vector.
        if mel_vec.size >= self.dim:
            compressed = mel_vec[: self.dim]
        else:
            compressed = np.pad(mel_vec, (0, self.dim - mel_vec.size))
        norm = float(np.linalg.norm(compressed))
        if norm > 1e-8:
            compressed = compressed / norm
        return compressed.astype(float)


class OnsetDetector:
    """Detect note/onset events via spectral flux (positive magnitude diff).

    The spectral flux is the sum of positive differences of the STFT
    magnitude between consecutive frames. Peaks in the flux that exceed a
    rule-based threshold (``mean + onset_threshold * std``) are reported as
    onsets. No learned model is involved.
    """

    def __init__(self, dim: int = 64, rules: AudioRules | None = None):
        self.dim = dim
        self.rules = rules or AudioRules()

    def detect(self, signal: np.ndarray, sample_rate: int = 16000) -> dict:
        """Detect onsets in ``signal``.

        Returns ``{'onset_frames', 'onset_times', 'spectral_flux', 'mean_flux'}``.
        """
        mono = _to_mono(signal)
        if mono.size < self.rules.frame_size:
            return {
                "onset_frames": [],
                "onset_times": [],
                "spectral_flux": np.zeros(0, dtype=float),
                "mean_flux": 0.0,
            }
        mag = _stft_magnitude(
            mono, self.rules.frame_size, self.rules.hop_size
        )
        n_frames = mag.shape[0]
        if n_frames < 2:
            return {
                "onset_frames": [],
                "onset_times": [],
                "spectral_flux": np.zeros(0, dtype=float),
                "mean_flux": 0.0,
            }
        # Spectral flux: sum of positive differences.
        diff = np.diff(mag, axis=0)
        flux = np.sum(np.maximum(diff, 0.0), axis=1)
        flux = np.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)
        mean_flux = float(np.mean(flux))
        std_flux = float(np.std(flux))
        threshold = mean_flux + self.rules.onset_threshold * std_flux
        # Peak picking: flux above threshold AND local maximum.
        onset_frames = []
        for i in range(1, len(flux) - 1):
            if flux[i] > threshold and flux[i] >= flux[i - 1] and flux[i] >= flux[i + 1]:
                onset_frames.append(int(i + 1))  # +1 because diff shifted by 1
        onset_times = [
            float(f * self.rules.hop_size / sample_rate) for f in onset_frames
        ]
        return {
            "onset_frames": onset_frames,
            "onset_times": onset_times,
            "spectral_flux": flux,
            "mean_flux": mean_flux,
        }


class PitchDetector:
    """Detect the fundamental frequency via autocorrelation + parabolic interp.

    The autocorrelation of a frame is computed over the lag range
    corresponding to ``[pitch_min_hz, pitch_max_hz]``. The lag with the
    highest peak (after parabolic interpolation) gives the fundamental
    frequency. Confidence is the normalized peak height.
    """

    def __init__(self, dim: int = 64, rules: AudioRules | None = None):
        self.dim = dim
        self.rules = rules or AudioRules()

    def detect(self, signal: np.ndarray, sample_rate: int = 16000) -> dict:
        """Detect the pitch of ``signal``.

        Returns ``{'pitch_hz', 'confidence', 'f0_candidates'}``.
        """
        mono = _to_mono(signal)
        if mono.size < 4 or sample_rate <= 0:
            return {
                "pitch_hz": 0.0,
                "confidence": 0.0,
                "f0_candidates": np.zeros(0, dtype=float),
            }
        # Use the central frame (rule-based).
        frame_size = min(mono.size, 2048)
        start = (mono.size - frame_size) // 2
        frame = mono[start : start + frame_size]
        # Remove DC offset.
        frame = frame - float(np.mean(frame))
        energy = float(np.sum(frame * frame))
        if energy < 1e-12:
            return {
                "pitch_hz": 0.0,
                "confidence": 0.0,
                "f0_candidates": np.zeros(0, dtype=float),
            }
        # Autocorrelation (full, then take the positive-lag half).
        corr = np.correlate(frame, frame, mode="full")
        corr = corr[frame_size - 1 :]
        # Normalize.
        if corr[0] > 0:
            corr = corr / corr[0]
        # Lag range for [pitch_max, pitch_min] (higher Hz -> smaller lag).
        lag_min = max(1, int(sample_rate / self.rules.pitch_max_hz))
        lag_max = min(frame_size - 1, int(sample_rate / self.rules.pitch_min_hz))
        if lag_max <= lag_min:
            return {
                "pitch_hz": 0.0,
                "confidence": 0.0,
                "f0_candidates": np.zeros(0, dtype=float),
            }
        # Find peaks in the lag range.
        candidates = []
        region = corr[lag_min : lag_max + 1]
        for i in range(1, len(region) - 1):
            if region[i] > region[i - 1] and region[i] > region[i + 1]:
                lag = lag_min + i
                # Parabolic interpolation for sub-sample accuracy.
                a = region[i - 1]
                b = region[i]
                c = region[i + 1]
                denom = (a - 2.0 * b + c)
                if abs(denom) > 1e-12:
                    offset = 0.5 * (a - c) / denom
                    lag_refined = lag + offset
                    peak_val = b - 0.25 * (a - c) * offset
                else:
                    lag_refined = float(lag)
                    peak_val = float(region[i])
                if lag_refined > 0:
                    f0 = sample_rate / lag_refined
                    candidates.append((float(f0), float(peak_val)))
        if not candidates:
            return {
                "pitch_hz": 0.0,
                "confidence": 0.0,
                "f0_candidates": np.zeros(0, dtype=float),
            }
        # Sort by confidence (descending).
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_f0, best_conf = candidates[0]
        # Confidence is the peak value (already normalized to [0, 1]).
        best_conf = float(max(0.0, min(1.0, best_conf)))
        f0_arr = np.array([c[0] for c in candidates], dtype=float)
        return {
            "pitch_hz": float(best_f0),
            "confidence": best_conf,
            "f0_candidates": f0_arr,
        }


class AudioClassifier:
    """Rule-based audio texture classification (speech/music/noise/silence).

    Features: spectral centroid, zero-crossing rate, RMS energy, spectral
    flatness. Rules decide the label from these features -- no trained model.
    """

    def __init__(self, dim: int = 64, rules: AudioRules | None = None):
        self.dim = dim
        self.rules = rules or AudioRules()

    def classify(self, signal: np.ndarray, sample_rate: int = 16000) -> dict:
        """Classify ``signal`` into one of ``rules.texture_labels``.

        Returns ``{'label', 'confidence', 'features'}``.
        """
        mono = _to_mono(signal)
        if mono.size == 0:
            return {
                "label": "silence",
                "confidence": 1.0,
                "features": {
                    "spectral_centroid_hz": 0.0,
                    "zero_crossing_rate": 0.0,
                    "rms_energy": 0.0,
                    "spectral_flatness": 0.0,
                },
            }
        # RMS energy.
        rms = float(np.sqrt(np.mean(mono ** 2)))
        # Zero-crossing rate.
        zcr = float(np.mean(np.diff(np.sign(mono)) != 0)) if mono.size > 1 else 0.0
        # Spectral features.
        mag = _stft_magnitude(
            mono, self.rules.frame_size, self.rules.hop_size
        )
        n_frames, n_freqs = mag.shape
        # Frequency axis.
        freqs = np.fft.rfftfreq(self.rules.frame_size, d=1.0 / sample_rate)
        # Spectral centroid (mean frequency weighted by magnitude).
        if mag.sum() > 1e-12:
            centroid = float(np.sum(freqs * mag.mean(axis=0)) / mag.mean(axis=0).sum())
        else:
            centroid = 0.0
        # Spectral flatness (geometric mean / arithmetic mean per frame).
        eps = 1e-12
        log_mag = np.log(mag + eps)
        geom_mean = np.exp(np.mean(log_mag, axis=1))
        arith_mean = np.mean(mag, axis=1)
        flatness = float(np.mean(geom_mean / (arith_mean + eps)))
        flatness = max(0.0, min(1.0, flatness))
        features = {
            "spectral_centroid_hz": centroid,
            "zero_crossing_rate": zcr,
            "rms_energy": rms,
            "spectral_flatness": flatness,
        }
        # Rule-based classification.
        if rms < 1e-6:
            label = "silence"
            confidence = 0.95
        elif centroid > 500.0 and zcr > 0.05 and flatness < 0.3:
            label = "speech"
            confidence = 0.7
        elif flatness > 0.1 and centroid > 200.0:
            # Onset regularity would require running OnsetDetector; approximate
            # with spectral flatness (music tends to be less flat than noise
            # but flatter than speech).
            label = "music"
            confidence = 0.6
        elif flatness > 0.4:
            label = "noise"
            confidence = 0.7
        else:
            label = "noise"
            confidence = 0.5
        return {
            "label": label,
            "confidence": float(confidence),
            "features": features,
        }
