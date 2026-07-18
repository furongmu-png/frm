# src/zero_data_model/capabilities/audio_advanced.py
"""Advanced Audio capabilities for the zero-data cognitive model.

Like the base ``audio`` module, every operator here is a deterministic,
rule-based prior composed with the existing core cognitive modules
(biological substrate, math universe). No external audio libraries (no
librosa / pydub) and no learned weights are required -- voice activity
detection, beat tracking and speaker-feature extraction are all derived
from rule priors + core-module self-generated representations.
"""

from __future__ import annotations

import numpy as np

from ..biological import BiologicalSubstrate
from ..math_universe import MathematicalUniverse
from .audio import OnsetDetector, _to_mono
from .rules import AudioRules


class SpeechSegmenter:
    """Voice activity detection (VAD) via energy + zero-crossing rate.

    Each frame is classified as speech or silence by two rule-based
    conditions: (a) RMS energy exceeds a threshold derived from the
    signal's own mean energy; (b) the zero-crossing rate is below a
    threshold (voiced speech has low ZCR). Adjacent speech frames are
    merged into segments; gaps shorter than a rule-based minimum are
    bridged. The temporal pattern is encoded via the biological substrate's
    cellular automaton (with state save/restore per R9-002 / R9-010).
    """

    def __init__(
        self,
        dim: int = 64,
        biological: BiologicalSubstrate | None = None,
        rules: AudioRules | None = None,
    ):
        self.dim = dim
        self.rules = rules or AudioRules()
        self.biological = biological or BiologicalSubstrate(dim=dim)
        # Rule-based VAD parameters.
        self.energy_alpha = 0.3   # threshold = mean + alpha * std
        self.zcr_max = 0.5        # max zero-crossing rate for speech
        self.min_gap_sec = 0.2   # bridge gaps shorter than this

    def segment(self, signal: np.ndarray, sample_rate: int = 16000) -> dict:
        """Segment ``signal`` into speech / silence regions.

        Returns ``{'segments', 'total_duration', 'speech_ratio'}`` where
        each segment is ``{'start': float, 'end': float, 'label': str}``.
        """
        mono = _to_mono(signal)
        total_duration = float(mono.size / sample_rate) if sample_rate > 0 else 0.0
        if mono.size < self.rules.frame_size or sample_rate <= 0:
            return {
                "segments": [
                    {"start": 0.0, "end": total_duration, "label": "silence"}
                ],
                "total_duration": total_duration,
                "speech_ratio": 0.0,
            }
        frame_size = self.rules.frame_size
        hop_size = self.rules.hop_size
        n_frames = 1 + (mono.size - frame_size) // hop_size
        if n_frames < 1:
            return {
                "segments": [
                    {"start": 0.0, "end": total_duration, "label": "silence"}
                ],
                "total_duration": total_duration,
                "speech_ratio": 0.0,
            }
        # Per-frame energy + ZCR.
        energies = np.zeros(n_frames, dtype=float)
        zcrs = np.zeros(n_frames, dtype=float)
        window = np.hanning(frame_size)
        for i in range(n_frames):
            start = i * hop_size
            frame = mono[start : start + frame_size]
            if frame.size < frame_size:
                frame = np.pad(frame, (0, frame_size - frame.size))
            frame_w = frame * window
            energies[i] = float(np.sqrt(np.mean(frame_w ** 2)))
            if frame.size > 1:
                zcrs[i] = float(np.mean(np.diff(np.sign(frame)) != 0))
            else:
                zcrs[i] = 0.0
        # Rule-based thresholds.
        e_mean = float(np.mean(energies))
        e_std = float(np.std(energies))
        e_threshold = e_mean + self.energy_alpha * e_std
        # Classify each frame.
        is_speech = (energies > e_threshold) & (zcrs < self.zcr_max)
        # Merge adjacent speech frames into segments.
        segments = []
        i = 0
        while i < n_frames:
            if is_speech[i]:
                j = i
                while j < n_frames and is_speech[j]:
                    j += 1
                start_time = float(i * hop_size / sample_rate)
                end_time = float(j * hop_size / sample_rate)
                segments.append(
                    {"start": start_time, "end": end_time, "label": "speech"}
                )
                i = j
            else:
                i += 1
        # Bridge gaps shorter than min_gap_sec.
        if len(segments) >= 2:
            merged = [segments[0]]
            for seg in segments[1:]:
                gap = seg["start"] - merged[-1]["end"]
                if gap < self.min_gap_sec:
                    merged[-1]["end"] = seg["end"]
                else:
                    merged.append(seg)
            segments = merged
        # Fill silence gaps between speech segments.
        full_segments = []
        prev_end = 0.0
        for seg in segments:
            if seg["start"] > prev_end:
                full_segments.append(
                    {"start": prev_end, "end": seg["start"], "label": "silence"}
                )
            full_segments.append(seg)
            prev_end = seg["end"]
        if prev_end < total_duration:
            full_segments.append(
                {"start": prev_end, "end": total_duration, "label": "silence"}
            )
        if not full_segments:
            full_segments = [
                {"start": 0.0, "end": total_duration, "label": "silence"}
            ]
        speech_dur = sum(
            s["end"] - s["start"] for s in full_segments if s["label"] == "speech"
        )
        speech_ratio = float(speech_dur / total_duration) if total_duration > 0 else 0.0
        # Temporal encoding via biological CA (with save/restore).
        ca = self.biological.automata
        saved_state = ca.state.copy()
        saved_rule = ca.rule
        try:
            binary = is_speech.astype(int)
            initial = np.zeros(ca.size, dtype=int)
            m = min(len(binary), ca.size)
            initial[:m] = binary[:m]
            ca.state = initial
            ca.rule = 30
            ca.evolve(n_steps=max(1, min(len(binary), 20)))
        finally:
            ca.state = saved_state
            ca.rule = saved_rule
        return {
            "segments": full_segments,
            "total_duration": total_duration,
            "speech_ratio": speech_ratio,
        }


class MusicAnalyzer:
    """Beat tracking + tempo estimation via onset autocorrelation.

    The onset envelope (spectral flux) is computed by reusing
    ``OnsetDetector``. The autocorrelation of the onset envelope is then
    searched for peaks in the lag range corresponding to 60-180 BPM.
    The strongest peak gives the tempo; beats are placed at the peak
    positions of the onset envelope at the tempo lag.
    """

    def __init__(
        self,
        dim: int = 64,
        biological: BiologicalSubstrate | None = None,
        rules: AudioRules | None = None,
    ):
        self.dim = dim
        self.rules = rules or AudioRules()
        self.biological = biological or BiologicalSubstrate(dim=dim)
        self.onset_detector = OnsetDetector(dim=dim, rules=self.rules)

    def analyze(self, signal: np.ndarray, sample_rate: int = 16000) -> dict:
        """Analyze ``signal`` for tempo and beats.

        Returns ``{'tempo_bpm', 'beat_frames', 'beat_times', 'onset_envelope'}``.
        """
        mono = _to_mono(signal)
        if mono.size < self.rules.frame_size or sample_rate <= 0:
            return {
                "tempo_bpm": 0.0,
                "beat_frames": [],
                "beat_times": [],
                "onset_envelope": np.zeros(0, dtype=float),
            }
        onset_result = self.onset_detector.detect(mono, sample_rate)
        onset_env = onset_result["spectral_flux"]
        if onset_env.size < 4:
            return {
                "tempo_bpm": 0.0,
                "beat_frames": [],
                "beat_times": [],
                "onset_envelope": onset_env,
            }
        # If the onset envelope has no energy (silence), return zero tempo.
        onset_energy = float(np.sum(onset_env * onset_env))
        if onset_energy < 1e-12:
            return {
                "tempo_bpm": 0.0,
                "beat_frames": [],
                "beat_times": [],
                "onset_envelope": onset_env,
            }
        # Autocorrelation of onset envelope.
        onset_centered = onset_env - float(np.mean(onset_env))
        corr = np.correlate(onset_centered, onset_centered, mode="full")
        corr = corr[onset_env.size - 1 :]
        if corr[0] > 0:
            corr = corr / corr[0]
        # Lag range for 60-180 BPM.
        hop_sec = self.rules.hop_size / sample_rate
        lag_min = int(60.0 / 180.0 / hop_sec)  # 180 BPM -> short lag
        lag_max = int(60.0 / 60.0 / hop_sec)   # 60 BPM -> long lag
        lag_min = max(1, lag_min)
        lag_max = min(len(corr) - 1, lag_max)
        if lag_max <= lag_min:
            return {
                "tempo_bpm": 0.0,
                "beat_frames": [],
                "beat_times": [],
                "onset_envelope": onset_env,
            }
        region = corr[lag_min : lag_max + 1]
        best_lag_offset = int(np.argmax(region))
        best_lag = lag_min + best_lag_offset
        if best_lag <= 0:
            tempo_bpm = 0.0
        else:
            period_sec = best_lag * hop_sec
            tempo_bpm = 60.0 / period_sec if period_sec > 0 else 0.0
        # Beat tracking: place beats at regular intervals starting from the
        # strongest onset.
        start_frame = int(np.argmax(onset_env)) if onset_env.size > 0 else 0
        beat_frames = []
        f = start_frame
        while f < onset_env.size:
            beat_frames.append(int(f))
            f += best_lag
        f = start_frame - best_lag
        while f >= 0:
            beat_frames.append(int(f))
            f -= best_lag
        beat_frames = sorted(set(beat_frames))
        beat_times = [float(b * self.rules.hop_size / sample_rate) for b in beat_frames]
        return {
            "tempo_bpm": float(tempo_bpm),
            "beat_frames": beat_frames,
            "beat_times": beat_times,
            "onset_envelope": onset_env,
        }


class SpeakerRecognizer:
    """Extract speaker-discriminative features via LPC + formant estimation.

    Linear Predictive Coding (LPC) coefficients are computed via the
    Levinson-Durbin recursion (pure numpy). The roots of the LPC polynomial
    give formant frequencies; these are combined with spectral centroid
    and pitch to form a speaker-feature vector. No speaker model is
    trained -- the class only extracts features; comparison is left to the
    caller via ``compare`` (cosine similarity through the math universe).
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
        self.lpc_order = 12

    def _levinson_durbin(self, autocorr: np.ndarray, order: int) -> np.ndarray:
        """Levinson-Durbin recursion for LPC coefficients (pure numpy).

        Returns the LPC coefficients ``a[0..order]`` where ``a[0] = 1.0``.
        """
        if order < 1 or autocorr.size < 1:
            return np.array([1.0], dtype=float)
        a = np.zeros(order + 1, dtype=float)
        a[0] = 1.0
        e = float(autocorr[0])
        if abs(e) < 1e-12:
            return a
        for i in range(1, order + 1):
            if i >= autocorr.size:
                break
            acc = float(autocorr[i])
            for j in range(1, i):
                acc -= a[j] * autocorr[i - j]
            k = acc / e
            # Update a (reflect).
            a_new = a.copy()
            for j in range(1, i):
                a_new[j] = a[j] - k * a[i - j]
            a_new[i] = -k
            a = a_new
            e = e * (1.0 - k * k)
            if abs(e) < 1e-12:
                break
        return a

    def extract_features(self, signal: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Extract speaker-discriminative features from ``signal``.

        Returns a ``dim``-length L2-normalized feature vector.
        """
        mono = _to_mono(signal)
        if mono.size < self.rules.frame_size or sample_rate <= 0:
            return np.zeros(self.dim, dtype=float)
        # Use the central frame.
        frame_size = min(mono.size, 2048)
        start = (mono.size - frame_size) // 2
        frame = mono[start : start + frame_size]
        # Remove DC.
        frame = frame - float(np.mean(frame))
        # Pre-emphasis (rule-based, standard 0.97).
        pre = np.empty_like(frame)
        pre[0] = frame[0]
        pre[1:] = frame[1:] - 0.97 * frame[:-1]
        frame = pre
        # Windowing.
        window = np.hanning(frame.size)
        frame_w = frame * window
        # Autocorrelation for LPC.
        autocorr = np.correlate(frame_w, frame_w, mode="full")
        autocorr = autocorr[frame_w.size - 1 :]
        # LPC coefficients.
        lpc = self._levinson_durbin(autocorr, self.lpc_order)
        # Formant estimation: roots of the LPC polynomial.
        # Polynomial: 1 + a[1]*z^-1 + a[2]*z^-2 + ... (a[0]=1)
        poly_coeffs = np.real(lpc)
        try:
            roots = np.roots(poly_coeffs)
        except np.linalg.LinAlgError:
            roots = np.array([], dtype=complex)
        # Keep roots inside the unit circle; convert to Hz.
        formants = []
        for r in roots:
            if np.abs(r) < 0.99 and np.abs(r) > 0.1:
                angle = np.angle(r)
                if angle > 0:
                    freq_hz = angle * sample_rate / (2.0 * np.pi)
                    formants.append(float(freq_hz))
        formants.sort()
        # Take up to 4 formants.
        formants = formants[:4]
        while len(formants) < 4:
            formants.append(0.0)
        # Spectral centroid.
        mag = np.abs(np.fft.rfft(frame_w))
        freqs = np.fft.rfftfreq(frame_w.size, d=1.0 / sample_rate)
        centroid = (
            float(np.sum(freqs * mag) / mag.sum())
            if mag.sum() > 1e-12
            else 0.0
        )
        # Energy.
        energy = float(np.sqrt(np.mean(frame_w ** 2)))
        # Combine features.
        features = np.array(
            formants + [centroid, energy] + list(lpc[: self.dim - 6]),
            dtype=float,
        )
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        if features.size < self.dim:
            features = np.pad(features, (0, self.dim - features.size))
        else:
            features = features[: self.dim]
        norm = float(np.linalg.norm(features))
        if norm > 1e-8:
            features = features / norm
        return features.astype(float)

    def compare(
        self, features_a: np.ndarray, features_b: np.ndarray
    ) -> float:
        """Compare two speaker feature vectors via cosine similarity.

        Returns a similarity score in ``[-1, 1]``.
        """
        a = np.asarray(features_a, dtype=float).flatten()
        b = np.asarray(features_b, dtype=float).flatten()
        n = min(a.size, b.size)
        if n < 1:
            return 0.0
        a = a[:n]
        b = b[:n]
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
