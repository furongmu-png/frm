# src/zero_data_model/capabilities/time_advanced.py
"""Advanced time-series capabilities for the zero-data cognitive model.

Composes the core cognitive modules (active inference for surprisal,
biological substrate for CA smoothing) with ``TimeRules``. All operators
are pure numpy -- no statsmodels / tslearn dependencies.
"""

from __future__ import annotations

import numpy as np

from ..active_inference import ActiveInferenceEngine
from ..biological import BiologicalSubstrate
from ..math_universe import MathematicalUniverse
from .rules import TimeRules
from .time import SeasonalityDetector


class AnomalyTimingDetector:
    """Detect anomalous inter-arrival gaps (rare timing patterns).

    Combines a z-score rule (|z| > event_threshold_std flags an anomaly)
    with an active-inference surprisal score. The surprisal is computed
    on the inter-arrival series and added as a secondary signal.
    """

    def __init__(
        self,
        dim: int = 64,
        active_inference: ActiveInferenceEngine | None = None,
        rules: TimeRules | None = None,
    ):
        self.dim = dim
        self.rules = rules or TimeRules()
        self.active_inference = active_inference or ActiveInferenceEngine(
            state_dim=dim, obs_dim=dim, action_dim=dim // 2
        )

    def detect(self, timestamps: np.ndarray) -> dict:
        """Detect anomalous inter-arrival gaps.

        Returns ``{'anomalies', 'scores', 'threshold'}`` where ``anomalies``
        is a boolean mask over inter-arrival gaps and ``scores`` are the
        |z|-scores.
        """
        ts = np.asarray(timestamps, dtype=float).flatten()
        if ts.size < 3:
            return {
                "anomalies": np.zeros(max(0, ts.size - 1), dtype=bool),
                "scores": np.zeros(max(0, ts.size - 1), dtype=float),
                "threshold": float(self.rules.event_threshold_std),
            }
        ts_sorted = np.sort(ts)
        inter = np.diff(ts_sorted)
        mean_inter = float(np.mean(inter))
        std_inter = float(np.std(inter))
        scores = (
            np.zeros_like(inter)
            if std_inter < 1e-12
            else (inter - mean_inter) / std_inter
        )
        abs_scores = np.abs(scores)
        threshold = float(self.rules.event_threshold_std)
        anomalies = abs_scores > threshold
        # Sanitize.
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        return {
            "anomalies": anomalies,
            "scores": abs_scores,
            "threshold": threshold,
        }


class CyclePhaseTracker:
    """Track phase within a periodic cycle.

    Given a series and an optional period (auto-detected via
    ``SeasonalityDetector`` if not provided), computes the phase of each
    sample within its cycle (in ``[0, 1)``) and a phase coherence measure
    (how concentrated the phases are around a single angle).
    """

    def __init__(
        self,
        dim: int = 64,
        biological: BiologicalSubstrate | None = None,
        rules: TimeRules | None = None,
    ):
        self.dim = dim
        self.rules = rules or TimeRules()
        self.biological = biological or BiologicalSubstrate(dim=dim)
        self._seasonality = SeasonalityDetector(dim=dim, rules=self.rules)

    def track(self, series: np.ndarray, period: int | None = None) -> dict:
        """Track cycle phase.

        Returns ``{'phases', 'period', 'phase_coherence'}``.
        """
        data = np.asarray(series, dtype=float).flatten()
        n = data.size
        if n < 4:
            return {
                "phases": np.zeros(n, dtype=float),
                "period": 0,
                "phase_coherence": 0.0,
            }
        if period is None or period <= 0:
            result = self._seasonality.detect(data)
            period = int(result.get("dominant_period", 0))
        if period <= 0:
            # Fall back to a default of n // 2.
            period = max(2, n // 2)
        # Phase: (t mod period) / period in [0, 1).
        t = np.arange(n, dtype=float)
        phases = (t % period) / float(period)
        # Phase coherence: treat phases as angles, compute mean resultant length.
        angles = 2.0 * np.pi * phases
        real_part = float(np.mean(np.cos(angles)))
        imag_part = float(np.mean(np.sin(angles)))
        coherence = float(np.sqrt(real_part * real_part + imag_part * imag_part))
        # CA temporal smoothing (with save/restore per R9-010).
        ca = self.biological.automata
        saved_state = ca.state.copy()
        saved_rule = ca.rule
        try:
            initial = np.zeros(ca.size, dtype=int)
            m = min(n, ca.size)
            initial[:m] = (phases[:m] * 2).astype(int)
            ca.state = initial
            ca.rule = 30
            ca.evolve(n_steps=max(1, min(n, 20)))
        finally:
            ca.state = saved_state
            ca.rule = saved_rule
        return {
            "phases": phases,
            "period": int(period),
            "phase_coherence": float(np.clip(coherence, 0.0, 1.0)),
        }


class ForecastabilityScorer:
    """Score how forecastable a series is (entropy + stationarity + autocorr).

    Combines three signals into a single ``[0, 1]`` forecastability score:
    - Sample entropy (normalized to ``[0, 1]``): lower entropy = more
      forecastable.
    - Stationarity: how similar the first and second halves' means are.
    - Autocorrelation strength: max autocorrelation at non-zero lag.
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

    def score(self, series: np.ndarray) -> dict:
        """Score forecastability.

        Returns ``{'forecastability', 'entropy', 'stationarity',
        'autocorr_strength'}``.
        """
        data = np.asarray(series, dtype=float).flatten()
        n = data.size
        if n < self.rules.forecast_min_samples:
            return {
                "forecastability": 0.0,
                "entropy": 1.0,
                "stationarity": 0.0,
                "autocorr_strength": 0.0,
            }
        # Sample entropy (normalized): use histogram-based Shannon entropy.
        entropy = self._shannon_entropy(data)
        # Stationarity: 1 - |mean(first_half) - mean(second_half)| / (std + eps).
        mid = n // 2
        first_mean = float(np.mean(data[:mid]))
        second_mean = float(np.mean(data[mid:]))
        std = float(np.std(data)) + 1e-8
        stationarity = 1.0 - min(1.0, abs(first_mean - second_mean) / std)
        # Autocorrelation strength: max autocorr at lag 1..n//2.
        autocorr_strength = self._max_autocorr(data)
        # Forecastability: weighted blend.
        forecastability = (
            0.4 * (1.0 - entropy)
            + 0.3 * stationarity
            + 0.3 * autocorr_strength
        )
        forecastability = float(np.clip(forecastability, 0.0, 1.0))
        return {
            "forecastability": forecastability,
            "entropy": float(entropy),
            "stationarity": float(np.clip(stationarity, 0.0, 1.0)),
            "autocorr_strength": float(np.clip(autocorr_strength, 0.0, 1.0)),
        }

    @staticmethod
    def _shannon_entropy(data: np.ndarray) -> float:
        """Histogram-based Shannon entropy, normalized to [0, 1]."""
        n = data.size
        if n < 2:
            return 1.0
        # Bin into sqrt(n) bins.
        n_bins = max(2, int(np.sqrt(n)))
        counts, _ = np.histogram(data, bins=n_bins)
        total = float(np.sum(counts))
        if total < 1e-12:
            return 1.0
        p = counts[counts > 0] / total
        entropy = float(-np.sum(p * np.log(p)))
        max_entropy = float(np.log(n_bins))
        if max_entropy < 1e-12:
            return 1.0
        return float(np.clip(entropy / max_entropy, 0.0, 1.0))

    @staticmethod
    def _max_autocorr(data: np.ndarray) -> float:
        """Max autocorrelation at non-zero lag, in [0, 1]."""
        n = data.size
        if n < 4:
            return 0.0
        centered = data - float(np.mean(data))
        denom = float(np.sum(centered * centered))
        if denom < 1e-12:
            return 0.0
        max_lag = min(n // 2, 64)
        best = 0.0
        for lag in range(1, max_lag + 1):
            a = centered[: n - lag]
            b = centered[lag:]
            num = float(np.sum(a * b))
            corr = num / denom
            if corr > best:
                best = corr
        return float(max(0.0, best))
