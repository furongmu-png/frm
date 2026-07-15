# src/zero_data_model/capabilities/analytics_advanced.py
"""Advanced Data Analytics capabilities for the zero-data cognitive model.

Like the base ``analytics`` module, every operator here is a deterministic,
rule-based prior composed with the existing core cognitive modules (active
inference, rules). No external statistical libraries (no statsmodels / pyro)
and no learned weights are required -- causal inference, Bayesian updating and
change-point detection are all derived from rule priors + active-inference
free energy.
"""

from __future__ import annotations

import contextlib
import math

import numpy as np

from ..active_inference import ActiveInferenceEngine
from .rules import AnalyticsRules


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two equal-length 1D arrays (rule-based)."""
    a = np.asarray(a, dtype=float).flatten()
    b = np.asarray(b, dtype=float).flatten()
    n = min(a.shape[0], b.shape[0])
    if n < 2:
        return 0.0
    a = a[:n]
    b = b[:n]
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    centered_a = a - np.mean(a)
    centered_b = b - np.mean(b)
    denom = math.sqrt(float(np.sum(centered_a ** 2)) * float(np.sum(centered_b ** 2)))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(centered_a * centered_b) / denom)


def _partial_correlation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Rule-based partial correlation of x and y given z (no statsmodels).

    Uses the standard recursive formula:
        r(x,y|z) = (r(x,y) - r(x,z) * r(y,z)) / sqrt((1 - r(x,z)^2)(1 - r(y,z)^2))
    """
    rxy = _pearson(x, y)
    rxz = _pearson(x, z)
    ryz = _pearson(y, z)
    denom = math.sqrt(max(0.0, 1.0 - rxz ** 2) * max(0.0, 1.0 - ryz ** 2))
    if denom < 1e-12:
        return 0.0
    return float((rxy - rxz * ryz) / denom)


class CausalInference:
    """Granger-causality-style causal inference between two series (no statsmodels).

    Combines a lagged cross-correlation scan with a rule-based partial
    correlation conditional-independence test. The result is a causal strength
    score in [-1, 1] and a rule-based p-value approximation derived from sample
    size and the lagged correlation.
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

    def infer_cause(
        self,
        cause_series: np.ndarray,
        effect_series: np.ndarray,
        max_lag: int = 5,
    ) -> dict:
        """Infer the causal direction cause -> effect via lagged correlation.

        Returns ``{'best_lag', 'causal_strength', 'p_value_approx'}``:

          * ``best_lag``: lag in 1..max_lag with the strongest absolute
            correlation; 0 if no usable overlap.
          * ``causal_strength``: signed correlation in [-1, 1] at ``best_lag``,
            adjusted by a rule-based partial-correlation conditional
            independence check.
          * ``p_value_approx``: rule-based significance approximation in
            [0, 1] derived from sample size and ``|causal_strength|``.
        """
        cause = np.asarray(cause_series, dtype=float).flatten()
        effect = np.asarray(effect_series, dtype=float).flatten()
        n = min(cause.shape[0], effect.shape[0])
        # Truncate to common length so lagged slicing stays in-bounds.
        cause = cause[:n]
        effect = effect[:n]

        if n < 4 or max_lag < 1:
            return {
                "best_lag": 0,
                "causal_strength": 0.0,
                "p_value_approx": 1.0,
            }

        max_lag = min(max_lag, n - 2)

        best_lag = 1
        best_corr = -float("inf")
        best_abs = 0.0
        for lag in range(1, max_lag + 1):
            # cause[t-lag] -> effect[t]
            a = cause[: n - lag]
            b = effect[lag:]
            r = _pearson(a, b)
            if not math.isfinite(r):
                continue
            if abs(r) > best_abs:
                best_abs = abs(r)
                best_corr = r
                best_lag = lag

        # Conditional-independence check via partial correlation: regress out
        # the effect's own recent past (a proxy autoregressor) from both the
        # lagged cause and the effect. If the partial correlation collapses to
        # ~0, the apparent causality is likely spurious.
        lag = best_lag
        if n - lag - 1 >= 3:
            lagged_cause = cause[: n - lag - 1]
            effect_future = effect[lag + 1:]
            # Use the effect's own past as the conditioning series z.
            effect_past = effect[: n - lag - 1]
            partial = _partial_correlation(lagged_cause, effect_future, effect_past)
        else:
            partial = best_corr

        # Blend: 70% lagged correlation + 30% partial correlation (rule-based
        # shrinkage toward the conditional-independence-adjusted estimate).
        causal_strength = (
            0.7 * best_corr + 0.3 * partial if math.isfinite(partial) else best_corr
        )
        causal_strength = float(np.clip(causal_strength, -1.0, 1.0))

        # Rule-based p-value approximation from sample size + |r|.
        # Uses the standard Student-t approximation: t = r * sqrt((n-2)/(1-r^2))
        # and converts the two-tailed survival to a p-value via the normal CDF
        # (a closed-form, learning-free approximation that needs no statsmodels).
        sample_size = max(3, n - lag)
        r_eff = abs(causal_strength)
        denom = max(1e-8, 1.0 - r_eff * r_eff)
        t_stat = r_eff * math.sqrt((sample_size - 2) / denom)
        # Two-tailed p-value via the standard normal survival function.
        p_value = float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(t_stat / math.sqrt(2.0)))))
        p_value = float(np.clip(p_value, 0.0, 1.0))

        return {
            "best_lag": int(best_lag),
            "causal_strength": causal_strength,
            "p_value_approx": p_value,
        }


class BayesianEstimator:
    """Online Bayesian updating with a conjugate-prior rule library (no pyro).

    Two conjugate models are supported, dispatched by observation type:

      * **Normal-Normal** (unknown mean, known variance): observations are
        scalars. Posterior mean / variance follow the standard conjugate
        update; the predictive is a Student-t approximation collapsed to a
        Normal with the posterior mean and combined variance.
      * **Beta-Binomial** (rate estimation): observations are
        ``(successes, trials)`` tuples or single booleans. Posterior is a
        Beta distribution; the predictive is Bernoulli/Binomial with the
        posterior mean.

    All updates are closed-form and require no external sampling library.
    """

    def __init__(
        self,
        dim: int = 64,
        active_inference: ActiveInferenceEngine | None = None,
    ):
        self.dim = dim
        self.active_inference = active_inference or ActiveInferenceEngine(
            state_dim=dim, obs_dim=dim, action_dim=dim // 2
        )
        # Known variance for the Normal-Normal model (rule prior).
        self.known_var = 1.0
        self.reset(prior_mean=0.0, prior_var=1.0)
        # Beta-Binomial state (Beta(1, 1) uninformative prior).
        self.beta_a = 1.0
        self.beta_b = 1.0
        self._beta_n = 0  # number of Beta observations seen

    def reset(self, prior_mean: float, prior_var: float) -> None:
        """Reset the Normal-Normal posterior to the supplied prior."""
        self.normal_mean = float(prior_mean)
        self.normal_var = float(max(prior_var, 1e-8))
        self._normal_n = 0
        # Also reset the Beta-Binomial state to an uninformative prior.
        self.beta_a = 1.0
        self.beta_b = 1.0
        self._beta_n = 0

    def _update_normal(self, observation: float) -> None:
        """Closed-form Normal-Normal conjugate update with known variance."""
        prior_prec = 1.0 / self.normal_var
        lik_prec = 1.0 / self.known_var
        # Online update: treat each scalar as a single observation.
        new_prec = prior_prec + lik_prec
        new_mean = (self.normal_mean * prior_prec + float(observation) * lik_prec) / new_prec
        self.normal_mean = float(new_mean)
        self.normal_var = float(1.0 / new_prec)
        self._normal_n += 1

    def _update_beta(self, successes: int, trials: int) -> None:
        """Closed-form Beta-Binomial conjugate update."""
        if trials <= 0:
            return
        s = max(0, min(int(successes), int(trials)))
        f = int(trials) - s
        self.beta_a += float(s)
        self.beta_b += float(f)
        self._beta_n += 1

    def update(self, observation) -> None:
        """Update the posterior from a single observation.

        - Scalar (int/float): Normal-Normal update.
        - Tuple ``(successes, trials)``: Beta-Binomial update.
        - Boolean: Beta-Binomial update with one trial.
        """
        if isinstance(observation, tuple) and len(observation) == 2:
            self._update_beta(int(observation[0]), int(observation[1]))
            return
        if isinstance(observation, bool):
            self._update_beta(1 if observation else 0, 1)
            return
        if isinstance(observation, (int, float)):
            self._update_normal(float(observation))
            return
        # Fallback: try to coerce to float (e.g. numpy scalars).
        with contextlib.suppress(TypeError, ValueError):
            self._update_normal(float(observation))

    def posterior_mean(self) -> float:
        """Return the posterior mean of the dominant updated model.

        Defaults to the Normal-Normal posterior mean; switches to the Beta
        posterior mean when only Beta-Binomial observations have been seen.
        """
        if self._normal_n == 0 and self._beta_n > 0:
            return float(self.beta_a / (self.beta_a + self.beta_b))
        return float(self.normal_mean)

    def posterior_var(self) -> float:
        """Return the posterior variance of the dominant updated model."""
        if self._normal_n == 0 and self._beta_n > 0:
            # Beta variance: ab / ((a+b)^2 (a+b+1)).
            a, b = self.beta_a, self.beta_b
            denom = (a + b) ** 2 * (a + b + 1.0)
            return float(a * b / denom) if denom > 0 else 0.0
        return float(self.normal_var)

    def predictive(self) -> tuple[float, float]:
        """Return ``(mean, std)`` of the posterior predictive distribution."""
        if self._normal_n == 0 and self._beta_n > 0:
            mean = float(self.beta_a / (self.beta_a + self.beta_b))
            var = self.posterior_var()
            return (mean, float(math.sqrt(max(var, 0.0))))
        # Normal-Normal predictive: N(posterior_mean, posterior_var + known_var).
        var = self.normal_var + self.known_var
        return (float(self.normal_mean), float(math.sqrt(max(var, 0.0))))


class ChangePointDetector:
    """Detect distributional change points via free-energy surprisal + CUSUM.

    The detector combines two learning-free signals:

      * **Free-energy surprisal** from the active-inference engine, computed
        per-point by feeding a small local window into the engine.
      * **Rule-based CUSUM** on the surprisal series: a cumulative deviation
        from a reference level, with reset on detection.

    Points where the CUSUM statistic exceeds a rule-based threshold are
    reported as change points. The detector is deterministic given the
    active-inference engine's internal matrices (pin the RNG in tests).
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
        # CUSUM rule priors.
        self.cusum_k = 0.5  # slack (allowance) per step
        self.cusum_threshold = 3.0  # rule-of-three-sigma threshold

    def _free_energy_series(self, series: np.ndarray) -> np.ndarray:
        """Compute a per-point free-energy surprisal signal."""
        data = np.asarray(series, dtype=float).flatten()
        n = data.shape[0]
        if n == 0:
            return np.zeros(0, dtype=float)
        # ``compute_free_energy`` is now pure (Fix 3): it does not mutate
        # ``belief_state``, so the per-point reset is no longer needed.
        fe = np.zeros(n, dtype=float)
        for i in range(n):
            local = data[i : i + 1]
            fe[i] = float(self.active_inference.compute_free_energy(local))
        return fe

    def detect(self, series: np.ndarray) -> np.ndarray:
        """Return an array of int indices at which change points occur.

        The algorithm: compute a free-energy surprisal series, z-score it,
        then run a one-sided CUSUM. Indices where the CUSUM statistic first
        exceeds the rule threshold are reported; the CUSUM is reset after each
        detected change point so multiple changes can be found.
        """
        data = np.asarray(series, dtype=float).flatten()
        n = data.shape[0]
        if n < 4:
            return np.zeros(0, dtype=int)

        fe = self._free_energy_series(data)
        # Z-score the free energy so the CUSUM threshold is scale-invariant.
        mu = float(np.mean(fe))
        sigma = float(np.std(fe)) + 1e-8
        z = (fe - mu) / sigma

        cusum = 0.0
        change_points: list[int] = []
        for i in range(n):
            cusum = max(0.0, cusum + (z[i] - self.cusum_k))
            if cusum > self.cusum_threshold:
                # Report the current index as a change point and reset CUSUM
                # so a single long plateau only produces one detection.
                change_points.append(int(i))
                cusum = 0.0

        # Also augment with a simple mean-shift rule: large |diff| in raw data
        # z-score (a complementary, free-energy-independent signal).
        diffs = np.abs(np.diff(data))
        if diffs.size > 0:
            d_mu = float(np.mean(diffs))
            d_sigma = float(np.std(diffs)) + 1e-8
            for i, d in enumerate(diffs):
                if d > d_mu + 3.0 * d_sigma and (i + 1) not in change_points:
                    change_points.append(int(i + 1))

        # De-duplicate and sort, then enforce a minimum gap so the detector
        # does not report a long run of adjacent points for one shift.
        change_points = sorted(set(change_points))
        min_gap = max(2, n // 10)
        pruned: list[int] = []
        for cp in change_points:
            if not pruned or cp - pruned[-1] >= min_gap:
                pruned.append(cp)
            else:
                # Keep the larger-surprisal one of the two close neighbours.
                if fe[cp] > fe[pruned[-1]]:
                    pruned[-1] = cp
        return np.asarray(pruned, dtype=int)

    def detect_count(self, series: np.ndarray) -> int:
        """Return the number of detected change points in ``series``."""
        return int(self.detect(series).shape[0])
