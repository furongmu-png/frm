# src/zero_data_model/capabilities/analytics.py
"""Data Analytics capability module for the zero-data cognitive model.

Composes the core cognitive modules (active inference, quantum hybrid,
biological substrate, math universe, category theory) with a small rule
library (``AnalyticsRules``: moving averages, z-scores, trend thresholds)
used as prior knowledge. No external training data and no learned weights
are required -- every operator below is a deterministic, rule-based prior
blended with self-generated representations.
"""

from __future__ import annotations

import numpy as np

from ..active_inference import ActiveInferenceEngine
from ..biological import BiologicalSubstrate
from ..category_engine import CategoryTheoryEngine
from ..math_universe import MathematicalUniverse
from ..quantum_hybrid import QuantumClassicalHybrid
from .rules import AnalyticsRules


class TimeSeriesForecaster:
    """Forecast future values of a 1D series with no trained model.

    The series is embedded into an active-inference belief state and rolled
    forward via the generative model; the result is blended 50/50 with a
    rule-based moving-average trend extrapolated into the future.
    """

    def __init__(
        self,
        dim: int = 64,
        active_inference: ActiveInferenceEngine | None = None,
        quantum_hybrid: QuantumClassicalHybrid | None = None,
        rules: AnalyticsRules | None = None,
    ):
        self.dim = dim
        self.rules = rules or AnalyticsRules()
        self.active_inference = active_inference or ActiveInferenceEngine(
            state_dim=dim, obs_dim=dim, action_dim=dim // 2
        )
        self.quantum_hybrid = quantum_hybrid or QuantumClassicalHybrid(
            dim=dim, n_qubits=min(8, dim)
        )

    def forecast(self, series: np.ndarray, horizon: int = 5) -> np.ndarray:
        """Forecast ``horizon`` future values of ``series``.

        - Embed the series into the belief state (pad/truncate to state_dim).
        - Iteratively predict the next state and decode each to a scalar
          (mean of the predicted observation).
        - Blend 0.5/0.5 with a moving-average trend extrapolated forward.
        """
        data = np.asarray(series, dtype=float).flatten()
        gm = self.active_inference.generative_model
        state_dim = gm.state_dim

        # Embed the series into the belief state (pad/truncate to state_dim).
        state = np.zeros(state_dim, dtype=float)
        n = min(len(data), state_dim)
        state[:n] = data[:n]
        gm.belief_state = state.copy()

        # Roll the generative model forward and decode each state to a scalar.
        ai_forecast = np.zeros(horizon, dtype=float)
        current = state.copy()
        for t in range(horizon):
            current = gm.predict_next_state(current, None)
            observation = gm.predict_observation(current)
            ai_forecast[t] = float(np.mean(observation))

        # Rule-based moving-average trend, linearly extrapolated forward.
        ma = np.asarray(self.rules.moving_average(data), dtype=float)
        if ma.size >= 2:
            slope, intercept = np.polyfit(np.arange(ma.size, dtype=float), ma, 1)
        elif ma.size == 1:
            slope, intercept = 0.0, float(ma[0])
        else:
            slope, intercept = 0.0, 0.0
        future_idx = np.arange(ma.size, ma.size + horizon, dtype=float)
        ma_trend = slope * future_idx + intercept

        blended = 0.5 * ai_forecast + 0.5 * ma_trend
        blended = np.nan_to_num(blended, nan=0.0, posinf=0.0, neginf=0.0)
        return blended.astype(float)


class AnomalyDetector:
    """Detect anomalies using free-energy surprisal combined with z-score rules.

    A local free energy is computed per point by feeding a most-local window
    (the point itself) into the active-inference engine; the belief state is
    reset before each point so each surprisal is estimated independently. The
    resulting free-energy series is z-scored, and points whose |z| exceeds the
    rule threshold are flagged.
    """

    def __init__(
        self,
        dim: int = 64,
        active_inference: ActiveInferenceEngine | None = None,
        rules: AnalyticsRules | None = None,
    ):
        self.dim = dim
        self.rules = rules or AnalyticsRules()
        self.active_inference = active_inference or ActiveInferenceEngine(
            state_dim=dim, obs_dim=dim, action_dim=dim // 2
        )

    def detect(self, series: np.ndarray) -> np.ndarray:
        """Return a boolean mask (same length as ``series``) of anomalies."""
        data = np.asarray(series, dtype=float).flatten()
        n = data.shape[0]
        gm = self.active_inference.generative_model
        state_dim = gm.state_dim

        free_energies = np.zeros(n, dtype=float)
        for i in range(n):
            # Most-local window: the point itself, so each point's surprisal is
            # estimated independently (avoids a single spike diluting the
            # z-score through many overlapping windows).
            local = data[i : i + 1]
            gm.belief_state = np.zeros(state_dim, dtype=float)
            free_energies[i] = float(self.active_inference.compute_free_energy(local))

        z = self.rules.zscore(free_energies)
        anomalies = np.abs(z) > self.rules.anomaly_z_threshold
        return np.asarray(anomalies, dtype=bool)


class PatternMiner:
    """Mine structural patterns from a series with no learned model.

    Combines fractal self-similarity, topological features, a Wolfram cellular
    automaton rule whose evolution best reproduces the binarized series, and an
    autocorrelation-based periodicity estimate.
    """

    def __init__(
        self,
        dim: int = 64,
        biological: BiologicalSubstrate | None = None,
        math_universe: MathematicalUniverse | None = None,
        rules: AnalyticsRules | None = None,
    ):
        self.dim = dim
        self.rules = rules or AnalyticsRules()
        self.biological = biological or BiologicalSubstrate(dim=dim)
        self.math_universe = math_universe or MathematicalUniverse(dim=dim)

    def mine(self, series: np.ndarray) -> dict:
        """Return self-similarity, topology, automaton rule and periodicity."""
        data = np.asarray(series, dtype=float).flatten()

        # Fractal self-similarity score (sanitize degenerate/constant inputs).
        compressed = self.math_universe.fractal.compress(data)
        self_similarity = float(compressed["self_similarity"])
        if not np.isfinite(self_similarity):
            self_similarity = 0.0

        # Topological features (betti numbers, moments, ...).
        topology = np.asarray(
            self.math_universe.topology.topological_features(data), dtype=float
        )

        # Wolfram CA rule (0-255) whose evolution best matches the binarized
        # series: encode the binarized series as the initial CA state, evolve
        # under each rule, and pick the one with lowest MSE to the pattern.
        automaton_rule = self._best_automaton_rule(data)

        # Autocorrelation peak lag in 1..len//2.
        periodicity = self._autocorrelation_period(data)

        return {
            "self_similarity": self_similarity,
            "topology": topology,
            "automaton_rule": int(automaton_rule),
            "periodicity": int(periodicity),
        }

    def _best_automaton_rule(self, data: np.ndarray) -> int:
        ca = self.biological.automata
        size = ca.size
        threshold = float(np.mean(data)) if data.size > 0 else 0.0
        binary = (data > threshold).astype(int)
        initial = np.zeros(size, dtype=int)
        m = min(len(binary), size)
        initial[:m] = binary[:m]
        target = initial.astype(float)
        n_steps = max(1, min(len(data) - 1, 20))

        best_rule = 0
        best_mse = float("inf")
        for rule in range(256):
            ca.rule = rule
            ca.state = initial.copy()
            history = ca.evolve(n_steps=n_steps)
            mse = float(np.mean((history.astype(float) - target) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_rule = rule
        return best_rule

    def _autocorrelation_period(self, data: np.ndarray) -> int:
        n = data.shape[0]
        if n < 2:
            return 1
        centered = data - np.mean(data)
        max_lag = n // 2
        best_lag = 1
        best_corr = -float("inf")
        for lag in range(1, max_lag + 1):
            a = centered[: n - lag]
            b = centered[lag:]
            denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
            if denom < 1e-12:
                continue
            corr = float(np.sum(a * b) / denom)
            if not np.isfinite(corr):
                continue
            if corr > best_corr:
                best_corr = corr
                best_lag = lag
        return int(best_lag)


class TrendAnalyzer:
    """Analyze trends using information geometry and category theory.

    Combines a linear regression slope/regime classification, a curvature
    estimate, a KL-divergence geodesic deviation between the two halves of the
    series, and a category-theoretic isomorphism (self-similarity) score.
    """

    def __init__(
        self,
        dim: int = 64,
        math_universe: MathematicalUniverse | None = None,
        category_engine: CategoryTheoryEngine | None = None,
        rules: AnalyticsRules | None = None,
    ):
        self.dim = dim
        self.rules = rules or AnalyticsRules()
        self.math_universe = math_universe or MathematicalUniverse(dim=dim)
        self.category_engine = category_engine or CategoryTheoryEngine(dim=dim)

    def analyze(self, series: np.ndarray) -> dict:
        """Return trend slope, regime, curvature, geodesic deviation and
        isomorphism score for ``series``."""
        data = np.asarray(series, dtype=float).flatten()
        n = data.shape[0]

        # Linear regression slope (degree-1 polyfit).
        slope = float(np.polyfit(np.arange(n, dtype=float), data, 1)[0])

        threshold = self.rules.trend_threshold
        if slope > threshold:
            regime = "up"
        elif slope < -threshold:
            regime = "down"
        else:
            regime = "flat"

        # Second-derivative mean (curvature).
        curvature = float(np.mean(np.diff(data, n=2))) if n >= 3 else 0.0

        # KL divergence between the abs-valued, normalized halves.
        mid = n // 2
        first_half = np.abs(data[:mid])
        second_half = np.abs(data[mid:])
        geodesic_deviation = float(
            self.math_universe.info_geometry.kl_divergence(first_half, second_half)
        )

        # Structural similarity between the two halves (cosine isomorphism).
        isomorphism_score = float(
            self.category_engine.find_isomorphism(first_half, second_half)
        )

        return {
            "trend_slope": slope,
            "regime": regime,
            "curvature": curvature,
            "geodesic_deviation": geodesic_deviation,
            "isomorphism_score": isomorphism_score,
        }
