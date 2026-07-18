# src/zero_data_model/capabilities/time.py
"""Time-series capability module for the zero-data cognitive model.

Composes the core cognitive modules (math universe for fractal compression,
biological substrate for CA-based smoothing) with a small rule library
(``TimeRules``: FFT parameters, seasonality thresholds). No external
training data and no learned weights are required -- every operator below
is a deterministic, rule-based prior blended with self-generated
representations.
"""

from __future__ import annotations

import numpy as np

from ..math_universe import MathematicalUniverse
from .rules import TimeRules


class TimeSeriesEncoder:
    """Encode a 1D time series into a ``dim``-length L2-normalized vector.

    Pipeline: z-score normalize -> Hann-windowed STFT magnitude -> log
    compression -> fractal compression (math_universe) -> dim vector.
    """

    def __init__(
        self,
        dim: int = 64,
        math_universe: MathematicalUniverse | None = None,
        rules: TimeRules | None = None,
    ):
        self.dim = dim
        self.rules = rules or TimeRules()
        self.math_universe = math_universe or MathematicalUniverse(dim=dim)

    def encode(self, series: np.ndarray) -> np.ndarray:
        """Encode ``series`` into a ``dim``-length L2-normalized vector."""
        data = np.asarray(series, dtype=float).flatten()
        if data.size == 0:
            return np.zeros(self.dim, dtype=float)
        # z-score normalize (guard against zero std for constant series).
        mu = float(np.mean(data))
        sigma = float(np.std(data))
        if sigma < 1e-12:
            normalized = data - mu
        else:
            normalized = (data - mu) / sigma
        # STFT magnitude.
        mag = self._stft_magnitude(normalized)
        if mag.size == 0:
            mag = normalized
        # Log compression.
        log_mag = np.log(mag + 1e-8)
        # Fractal compression to dim. Suppress numpy divide-by-zero warnings
        # from np.corrcoef inside math_universe.fractal.compress when the
        # input is constant or degenerate; we sanitize the output below.
        with np.errstate(invalid="ignore", divide="ignore"):
            compressed = self.math_universe.fractal.compress(log_mag)
        # Extract a fixed-length feature vector.
        vec = self._to_dim_vector(compressed)
        # Sanitize + L2 normalize.
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-12:
            return vec
        return vec / norm

    def _stft_magnitude(self, data: np.ndarray) -> np.ndarray:
        """Compute STFT magnitude averaged across frames."""
        n = data.size
        win = self.rules.fft_window_size
        hop = self.rules.fft_hop_size
        if n < win:
            # Pad with zeros.
            padded = np.zeros(win, dtype=float)
            padded[:n] = data
            frame = padded * self._hann_window(win)
            spec = np.abs(np.fft.rfft(frame))
            return spec
        # Hann window.
        window = self._hann_window(win)
        magnitudes = []
        for start in range(0, max(1, n - win + 1), max(1, hop)):
            frame = data[start : start + win] * window
            spec = np.abs(np.fft.rfft(frame))
            magnitudes.append(spec)
        if not magnitudes:
            return np.zeros(win // 2 + 1, dtype=float)
        # Average magnitude across frames.
        return np.mean(np.array(magnitudes), axis=0)

    @staticmethod
    def _hann_window(n: int) -> np.ndarray:
        """Hann window of length ``n``."""
        if n <= 1:
            return np.ones(1, dtype=float)
        return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1))

    def _to_dim_vector(self, compressed: dict | np.ndarray) -> np.ndarray:
        """Extract a fixed-length feature vector from fractal output."""
        if isinstance(compressed, dict):
            # Pull numerical fields.
            fields = []
            for key in ("self_similarity", "entropy", "fractal_dimension"):
                if key in compressed and isinstance(compressed[key], int | float):
                    fields.append(float(compressed[key]))
            # Topological features if present.
            topo = compressed.get("topology")
            if topo is not None:
                arr = np.asarray(topo, dtype=float).flatten()
                fields.extend(arr.tolist())
            vec = np.array(fields, dtype=float)
        else:
            vec = np.asarray(compressed, dtype=float).flatten()
        # Pad/truncate to dim.
        if vec.size < self.dim:
            vec = np.pad(vec, (0, self.dim - vec.size))
        elif vec.size > self.dim:
            vec = vec[: self.dim]
        return vec.astype(float)


class SeasonalityDetector:
    """Detect seasonal periods via autocorrelation peaks.

    Computes the autocorrelation of the series for lags ``1..max_lag``
    and finds local maxima exceeding ``seasonality_threshold``. The
    strongest peaks are returned as candidate seasonal periods.
    """

    def __init__(self, dim: int = 64, rules: TimeRules | None = None):
        self.dim = dim
        self.rules = rules or TimeRules()

    def detect(self, series: np.ndarray) -> dict:
        """Detect seasonal periods in ``series``.

        Returns ``{'periods', 'strengths', 'dominant_period'}``.
        """
        data = np.asarray(series, dtype=float).flatten()
        n = data.size
        if n < 4:
            return {
                "periods": [],
                "strengths": [],
                "dominant_period": 0,
            }
        max_lag = min(self.rules.seasonality_max_lag, n // 2)
        centered = data - float(np.mean(data))
        denom = float(np.sum(centered * centered))
        if denom < 1e-12:
            return {
                "periods": [],
                "strengths": [],
                "dominant_period": 0,
            }
        # Autocorrelation for lags 1..max_lag.
        autocorr = np.zeros(max_lag + 1, dtype=float)
        autocorr[0] = 1.0
        for lag in range(1, max_lag + 1):
            a = centered[: n - lag]
            b = centered[lag:]
            num = float(np.sum(a * b))
            autocorr[lag] = num / denom
        # Find local maxima exceeding threshold.
        periods = []
        strengths = []
        threshold = self.rules.seasonality_threshold
        for lag in range(2, max_lag):
            if autocorr[lag] <= threshold:
                continue
            # Local maximum: greater than both neighbors.
            if autocorr[lag] > autocorr[lag - 1] and autocorr[lag] > autocorr[lag + 1]:
                periods.append(int(lag))
                strengths.append(float(autocorr[lag]))
        # Sort by strength descending.
        if periods:
            order = np.argsort(strengths)[::-1]
            periods = [periods[i] for i in order]
            strengths = [strengths[i] for i in order]
        dominant = periods[0] if periods else 0
        return {
            "periods": periods,
            "strengths": strengths,
            "dominant_period": int(dominant),
        }


class FrequencyAnalyzer:
    """Spectral analysis via FFT: dominant frequencies, power spectrum.

    Returns the FFT power spectrum, the dominant non-DC frequency, and the
    spectral entropy (a measure of how spread the signal's energy is
    across frequencies).
    """

    def __init__(
        self,
        dim: int = 64,
        math_universe: MathematicalUniverse | None = None,
        rules: TimeRules | None = None,
    ):
        self.dim = dim
        self.rules = rules or TimeRules()
        self.math_universe = math_universe or MathematicalUniverse(dim=dim)

    def analyze(self, series: np.ndarray) -> dict:
        """Analyze the frequency content of ``series``.

        Returns ``{'frequencies', 'power', 'dominant_freq', 'spectral_entropy'}``.
        """
        data = np.asarray(series, dtype=float).flatten()
        n = data.size
        if n < 2:
            return {
                "frequencies": np.zeros(0, dtype=float),
                "power": np.zeros(0, dtype=float),
                "dominant_freq": 0.0,
                "spectral_entropy": 0.0,
            }
        # Remove DC component.
        centered = data - float(np.mean(data))
        # FFT.
        spectrum = np.fft.rfft(centered)
        power = np.abs(spectrum) ** 2
        frequencies = np.fft.rfftfreq(n, d=1.0 / self.rules.sample_rate)
        # Dominant frequency (excluding DC, which is index 0).
        if power.size > 1 and float(np.max(power[1:])) > 1e-12:
            dominant_idx = int(np.argmax(power[1:])) + 1
            dominant_freq = float(frequencies[dominant_idx])
        else:
            dominant_freq = 0.0
        # Spectral entropy: normalize power -> p, -sum(p * log(p)).
        total = float(np.sum(power))
        if total < 1e-12:
            spectral_entropy = 0.0
        else:
            p = power / total
            p = np.clip(p, 1e-12, 1.0)
            spectral_entropy = float(-np.sum(p * np.log(p)))
            # Normalize by log(n) so it's in [0, 1].
            max_entropy = float(np.log(max(2, power.size)))
            if max_entropy > 1e-12:
                spectral_entropy = spectral_entropy / max_entropy
        return {
            "frequencies": frequencies,
            "power": power,
            "dominant_freq": dominant_freq,
            "spectral_entropy": float(np.clip(spectral_entropy, 0.0, 1.0)),
        }


class EventTimestampAnalyzer:
    """Extract event timing: inter-arrival times, rate, burstiness.

    Given a series of event timestamps, computes the inter-arrival times,
    the overall event rate (events per time unit), and a burstiness
    coefficient in ``[-1, 1]`` (high values = bursty, low = regular).
    """

    def __init__(self, dim: int = 64, rules: TimeRules | None = None):
        self.dim = dim
        self.rules = rules or TimeRules()

    def analyze(self, timestamps: np.ndarray) -> dict:
        """Analyze event timing.

        Returns ``{'inter_arrival', 'rate', 'burstiness', 'total_events'}``.
        """
        ts = np.asarray(timestamps, dtype=float).flatten()
        n = ts.size
        if n < 2:
            return {
                "inter_arrival": np.zeros(0, dtype=float),
                "rate": 0.0,
                "burstiness": 0.0,
                "total_events": int(n),
            }
        ts_sorted = np.sort(ts)
        inter = np.diff(ts_sorted)
        total_time = float(ts_sorted[-1] - ts_sorted[0])
        rate = float(n / total_time) if total_time > 1e-12 else 0.0
        mean_inter = float(np.mean(inter))
        std_inter = float(np.std(inter))
        # Burstiness coefficient: (std - mean) / (std + mean).
        denom = std_inter + mean_inter
        burstiness = float((std_inter - mean_inter) / denom) if denom > 1e-12 else 0.0
        return {
            "inter_arrival": inter,
            "rate": rate,
            "burstiness": float(np.clip(burstiness, -1.0, 1.0)),
            "total_events": int(n),
        }
