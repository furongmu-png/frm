# src/zero_data_model/causal_emergence/hmc.py
"""Module D: HamiltonianSampler.

Hamiltonian Monte Carlo sampler with leapfrog integration and central-
difference gradients. Provides calibrated uncertainty estimates from
arbitrary log-posterior functions.

Algorithm (spec §6):
- Potential energy U(q) = -log_prob_fn(q).
- Gradient: analytical if ``grad_fn`` provided; otherwise central
  difference with step ``h = rules.hmc_finite_diff_h`` (default 1e-5,
  fix M3).
- Leapfrog integrator (unit mass matrix).
- Metropolis accept/reject with Hamiltonian H = U + 0.5*||p||^2.
- ESS via Geyer (1992) initial monotone sequence estimator (fix M4).

Edge cases (fix L4, M5):
- ``log_prob_fn`` raising → caught, proposal rejected, warning logged.
- ``log_prob_fn`` returning -inf or NaN → rejected.
- ``initial_position`` non-finite → ValueError.
- ``n_samples == 0`` → empty arrays.
- ``accept_rate`` outside [0.5, 0.95] → ``converged = False``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .rules import EmergenceRules


class HamiltonianSampler:
    """Hamiltonian Monte Carlo sampler (leapfrog + Metropolis)."""

    def __init__(
        self,
        dim: int = 64,
        rules: EmergenceRules | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.dim = dim
        self.rules = rules or EmergenceRules()
        self.rng = rng or np.random.default_rng()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def sample(
        self,
        log_prob_fn: Callable[[np.ndarray], float],
        initial_position: np.ndarray,
        n_samples: int = 100,
        step_size: float | None = None,
        n_leapfrog: int | None = None,
        grad_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> dict:
        """Sample from the posterior defined by ``log_prob_fn``.

        Single-chain ESS estimates have large uncertainty; recommend
        multi-chain runs with Gelman-Rubin R-hat for production use
        (fix L5).

        Returns dict with ``samples``, ``mean``, ``std``,
        ``accept_rate``, ``ess``, ``converged``, ``warnings``.
        """
        if step_size is None:
            step_size = self.rules.hmc_step_size
        if n_leapfrog is None:
            n_leapfrog = self.rules.hmc_n_leapfrog

        q = self._sanitize_position(initial_position)
        dim = q.shape[0]

        # Edge case: n_samples == 0 -> empty arrays
        if n_samples <= 0:
            return self._empty_result(dim)

        warnings: list[str] = []
        samples = np.zeros((n_samples, dim))
        n_accept = 0

        # Current Hamiltonian
        current_logp = self._safe_log_prob(log_prob_fn, q, warnings)
        if not np.isfinite(current_logp):
            # If the very first log_prob is invalid, we cannot proceed
            # meaningfully — fill zeros and flag.
            warnings.append(
                "initial_position has non-finite log_prob; "
                "all samples will be the initial position"
            )
            for i in range(n_samples):
                samples[i] = q
            return self._finalize(samples, 0.0, warnings)

        current_U = -current_logp

        for i in range(n_samples):
            # Momentum resample: p ~ N(0, I)
            p0 = self.rng.standard_normal(dim)
            current_H = current_U + 0.5 * float(np.sum(p0 * p0))

            # Leapfrog integration
            q_new, p_new = self._leapfrog(
                log_prob_fn, q, p0, step_size, n_leapfrog, grad_fn, warnings
            )

            # Compute new Hamiltonian
            new_logp = self._safe_log_prob(log_prob_fn, q_new, warnings)
            if not np.isfinite(new_logp):
                # Reject: q stays
                samples[i] = q
                continue
            new_U = -new_logp
            new_H = new_U + 0.5 * float(np.sum(p_new * p_new))

            # Metropolis accept/reject
            delta_H = current_H - new_H
            if np.log(self.rng.uniform()) < delta_H:
                q = q_new
                current_U = new_U
                current_logp = new_logp
                n_accept += 1

            samples[i] = q

        accept_rate = n_accept / n_samples

        return self._finalize(samples, accept_rate, warnings)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _sanitize_position(self, position: Any) -> np.ndarray:
        """Coerce to 1D float ndarray; raise on non-finite."""
        arr = np.ascontiguousarray(position, dtype=float).flatten()
        if arr.size == 0:
            raise ValueError("initial_position must be non-empty")
        if not np.all(np.isfinite(arr)):
            raise ValueError("initial_position must be finite")
        return arr

    def _safe_log_prob(
        self,
        log_prob_fn: Callable[[np.ndarray], float],
        q: np.ndarray,
        warnings: list[str],
    ) -> float:
        """Call log_prob_fn with exception/NaN guard (fix L4)."""
        try:
            val = float(log_prob_fn(q))
        except Exception as exc:  # noqa: BLE001 — spec requires catch-all
            warnings.append(f"log_prob_fn raised: {type(exc).__name__}: {exc}")
            return float("-inf")
        if not np.isfinite(val):
            # -inf is acceptable (just rejects); NaN is not.
            if np.isnan(val):
                warnings.append("log_prob_fn returned NaN; rejecting proposal")
                return float("-inf")
            return val
        return val

    def _grad_U(
        self,
        log_prob_fn: Callable[[np.ndarray], float],
        q: np.ndarray,
        grad_fn: Callable[[np.ndarray], np.ndarray] | None,
        warnings: list[str],
    ) -> np.ndarray:
        """Gradient of U = -log_prob.

        If ``grad_fn`` is provided, use it (and apply the same safety
        guard). Otherwise, central difference with step ``h``.
        """
        if grad_fn is not None:
            try:
                g = np.ascontiguousarray(grad_fn(q), dtype=float).flatten()
                if not np.all(np.isfinite(g)):
                    warnings.append("grad_fn returned non-finite; using zero")
                    return np.zeros_like(q)
                return -g  # U = -log_prob
            except Exception as exc:  # noqa: BLE001
                warnings.append(
                    f"grad_fn raised: {type(exc).__name__}: {exc}; "
                    "using central difference"
                )
        # Central difference (fix M3)
        h = self.rules.hmc_finite_diff_h
        grad = np.zeros_like(q)
        for i in range(q.shape[0]):
            qp = q.copy()
            qp[i] += h
            logp_p = self._safe_log_prob(log_prob_fn, qp, warnings)
            qm = q.copy()
            qm[i] -= h
            logp_m = self._safe_log_prob(log_prob_fn, qm, warnings)
            # grad_U = -grad_logp = -(logp_p - logp_m) / (2h)
            grad[i] = -(logp_p - logp_m) / (2.0 * h)
        # NaN guard (e.g., both -inf)
        if not np.all(np.isfinite(grad)):
            grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        return grad

    def _leapfrog(
        self,
        log_prob_fn: Callable[[np.ndarray], float],
        q: np.ndarray,
        p: np.ndarray,
        step_size: float,
        n_leapfrog: int,
        grad_fn: Callable[[np.ndarray], np.ndarray] | None,
        warnings: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Leapfrog integration of Hamiltonian dynamics.

        Canonical form:
            p -= (eps/2) * grad(q)            # initial half step
            for k in range(L):
                q += eps * p                   # full position step
                if k < L - 1:
                    p -= eps * grad(q)         # interior full momentum step
            p -= (eps/2) * grad(q)            # final half step
            p = -p                            # reversibility convention
        """
        q = q.copy()
        p = p.copy()

        # Initial half step for momentum
        grad = self._grad_U(log_prob_fn, q, grad_fn, warnings)
        p = p - 0.5 * step_size * grad

        for k in range(n_leapfrog):
            q = q + step_size * p
            if k < n_leapfrog - 1:
                grad = self._grad_U(log_prob_fn, q, grad_fn, warnings)
                p = p - step_size * grad

        # Final half step
        grad = self._grad_U(log_prob_fn, q, grad_fn, warnings)
        p = p - 0.5 * step_size * grad

        # Negate momentum (HMC reversibility convention; does not affect
        # the Hamiltonian since H = U + 0.5*||p||^2).
        p = -p

        # NaN/Inf guard
        q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        return q, p

    def _empty_result(self, dim: int) -> dict:
        return {
            "samples": np.zeros((0, dim)),
            "mean": np.zeros(dim),
            "std": np.zeros(dim),
            "accept_rate": 0.0,
            "ess": 0.0,
            "converged": False,
            "warnings": ["n_samples=0; no samples collected"],
        }

    def _finalize(
        self, samples: np.ndarray, accept_rate: float, warnings: list[str]
    ) -> dict:
        mean = samples.mean(axis=0) if samples.size else np.zeros(samples.shape[1])
        std = samples.std(axis=0) if samples.size else np.zeros(samples.shape[1])
        ess = self._ess_geyer(samples) if samples.size else 0.0
        converged = 0.5 <= accept_rate <= 0.95
        return {
            "samples": samples,
            "mean": np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0),
            "std": np.nan_to_num(std, nan=0.0, posinf=0.0, neginf=0.0),
            "accept_rate": float(accept_rate),
            "ess": float(ess),
            "converged": bool(converged),
            "warnings": warnings,
        }

    def _ess_geyer_per_dim(self, samples: np.ndarray) -> list[float]:
        """Per-dimension ESS via Geyer (1992) initial monotone sequence.

        fix R2-NEW-L3 (test surface): exposes per-dimension ESS values so
        tests can verify the constant-dim branch directly, instead of
        relying on the mean (which is too coarse to distinguish
        ``ess_d = 1.0`` from a hypothetical ``ess_d = 0`` regression when
        the other dim carries ~75 effective samples).

        For each dimension, compute the autocorrelation function and
        sum pairs (lag 2k-1, 2k) until the sum first becomes negative.
        ESS_d = n / (1 + 2 * sum). Constant dimensions (variance < 1e-12)
        return ESS_d = 1.0 — a conservative middle ground between 0 (no
        info) and n (perfectly representative).
        """
        n, dim = samples.shape
        if n < 2:
            return [float(n)] * dim

        ess_values: list[float] = []
        for d in range(dim):
            x = samples[:, d] - samples[:, d].mean()
            var = float(np.sum(x * x)) / n
            if var < 1e-12:
                # fix NEW-L3: constant series carries zero posterior
                # information — a single sample suffices to represent it.
                # Returning ``n`` (the previous behavior) inflated the mean
                # ESS for posteriors that are constant in some dimensions,
                # masking under-sampling in the others. ``1.0`` is the
                # conservative choice: count one effective sample per
                # constant dim.
                ess_values.append(1.0)
                continue

            # Autocorrelation via FFT for speed
            acf = self._autocorr(x, n)
            # Geyer initial monotone sequence: sum pairs until negative
            # pairs[k] = acf[2k-1] + acf[2k]
            # Stop at first non-positive pair sum
            s = 0.0
            k = 1
            while 2 * k < n:
                pair = acf[2 * k - 1] + acf[2 * k]
                if pair <= 0:
                    break
                s += pair
                k += 1
            ess = n / (1.0 + 2.0 * s)
            ess_values.append(max(1.0, min(float(n), ess)))

        return ess_values

    def _ess_geyer(self, samples: np.ndarray) -> float:
        """ESS via Geyer (1992) initial monotone sequence estimator.

        Returns the mean across dimensions of ``_ess_geyer_per_dim``.

        fix R2-NEW-L4: Constant dimensions (variance < 1e-12) return
        ESS_d = 1.0 — a conservative middle ground between 0 (no info)
        and n (perfectly representative). Returning ``n`` would inflate the
        mean ESS for mixed posteriors and mask under-sampling in
        well-sampled dimensions.
        """
        ess_values = self._ess_geyer_per_dim(samples)
        return float(np.mean(ess_values))

    @staticmethod
    def _autocorr(x: np.ndarray, n: int) -> np.ndarray:
        """Autocorrelation function via FFT, normalized so acf[0] = 1."""
        # Pad to next power of 2 for FFT efficiency
        size = 1
        while size < 2 * n:
            size *= 2
        f = np.fft.rfft(x, size)
        acf = np.fft.irfft(f * np.conjugate(f), size)[:n]
        acf = acf / acf[0]
        return acf
