# src/zero_data_model/active_inference.py
"""Active Inference Engine based on Free Energy Principle."""

from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass

import numpy as np

from .base import CognitiveModule, Prediction, Signal

# Round-5 audit TEST5-5: named constant instead of a magic 1e6 number.
# Returned by ``compute_free_energy`` / ``update_belief`` when the
# observation or internal state is non-finite. Chosen large enough that
# ``AnomalyDetector``'s z-score flags it as an outlier, but finite so it
# does not poison downstream argmin / softmax.
_FREE_ENERGY_SENTINEL: float = 1e6


@dataclass
class MarkovBlanket:
    """Defines the boundary between internal and external states."""
    sensory_dim: int
    active_dim: int
    internal_dim: int
    sensory_weights: np.ndarray
    active_weights: np.ndarray

    @classmethod
    def create(
        cls,
        sensory_dim: int = 32,
        active_dim: int = 16,
        internal_dim: int = 64,
        rng: np.random.Generator | None = None,
    ):
        # Round-3 audit CRIT-1: per-module Generator
        _rng = rng if rng is not None else np.random.default_rng()
        return cls(
            sensory_dim=sensory_dim,
            active_dim=active_dim,
            internal_dim=internal_dim,
            sensory_weights=_rng.standard_normal((sensory_dim, internal_dim)) * 0.1,
            active_weights=_rng.standard_normal((internal_dim, active_dim)) * 0.1,
        )


class GenerativeModel:
    """Internal generative model — predicts sensory inputs from hidden states."""

    def __init__(
        self, state_dim: int = 64, obs_dim: int = 32, rng: np.random.Generator | None = None
    ):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        # Round-3 audit CRIT-1: per-module Generator
        self._rng = rng if rng is not None else np.random.default_rng()
        self.transition = self._rng.standard_normal((state_dim, state_dim)) * 0.05
        self.emission = self._rng.standard_normal((state_dim, obs_dim)) * 0.1
        # C-batch fix: initialise ``belief_state`` to a small non-zero vector
        # so the very first ``update()`` (called before any ``process``) has a
        # non-zero state to compute a real gradient from -- the previous
        # ``zeros`` initialisation made the gradient ``outer(0, error) = 0``,
        # so the only way ``update`` could change ``emission`` was the random
        # noise walk, which is exactly what we are removing.
        self.belief_state = self._rng.standard_normal(state_dim) * 0.01
        # Cache of the last inference context so ``update`` can do real
        # gradient descent instead of a random walk. Populated by
        # ``update_belief``; falls back to ``belief_state`` + zero obs when
        # ``update`` is called before any observation has been processed.
        self._last_state: np.ndarray | None = None
        self._last_observation: np.ndarray | None = None
        self._last_error: np.ndarray | None = None

    def predict_observation(self, state: np.ndarray) -> np.ndarray:
        return state @ self.emission

    def predict_next_state(self, state: np.ndarray, action: np.ndarray | None = None) -> np.ndarray:
        next_state = state @ self.transition
        if action is not None:
            padded = np.zeros(self.state_dim)
            padded[: len(action)] = action
            next_state += padded
        return next_state

    def infer_state(self, observation: np.ndarray) -> tuple[np.ndarray, float]:
        """Pure: returns (inferred_state, prediction_error) WITHOUT mutating
        ``belief_state``.

        The returned state is ``belief_state + 0.1 * gradient``; callers that
        want to actually update the belief must call ``update_belief`` instead.
        Keeping this pure is what lets ``compute_free_energy`` run from
        read-only analytics methods (classify_text, detect_anomalies, ...)
        without corrupting the shared belief state.
        """
        predicted_obs = self.predict_observation(self.belief_state)
        error = observation[: self.obs_dim] - predicted_obs[: len(observation)]
        if len(error) < self.obs_dim:
            error = np.pad(error, (0, self.obs_dim - len(error)))
        prediction_error = float(np.mean(error ** 2))
        gradient = error @ self.emission.T
        return self.belief_state + 0.1 * gradient, prediction_error

    def update_belief(self, observation: np.ndarray) -> tuple[np.ndarray, float]:
        """Mutate ``belief_state`` from a new observation.

        Returns the same ``(new_state, prediction_error)`` pair as
        ``infer_state`` for convenience. Called only from the active-inference
        ``process`` / ``update`` cycle so analytics methods can use
        ``infer_state`` (pure) without corrupting shared state.

        C-batch fix: caches ``(state, observation, error)`` so ``update`` can
        do real gradient descent on ``emission`` instead of a noise walk.

        Round-5 audit TEST5-2: guard against NaN/Inf observation. Unlike
        ``compute_free_energy`` (read-only, returns the sentinel), this method
        MUTATES ``belief_state``. A NaN observation would permanently corrupt
        the belief (every subsequent cycle inherits the NaN), and the cached
        context would poison the next ``emission_gradient_step``. Reject the
        update entirely: return the current (unchanged) belief and the
        sentinel prediction_error so ``process`` records a large free energy
        that ``AnomalyDetector`` picks up as an anomaly.
        """
        obs_arr = np.asarray(observation, dtype=float)
        if not np.all(np.isfinite(obs_arr)):
            # Round-5 audit TEST5-4: emit a warning so the rejection is
            # observable rather than silent. Without this, a NaN upstream
            # module would be invisible — the belief stays unchanged and
            # the only signal is the sentinel free energy (which
            # AnomalyDetector picks up, but the operator does not see why).
            warnings.warn(
                "update_belief received a non-finite observation "
                f"(contains NaN={np.any(np.isnan(obs_arr))}, "
                f"Inf={np.any(np.isinf(obs_arr))}); "
                "rejecting the belief update to prevent permanent "
                "corruption of belief_state",
                stacklevel=2,
            )
            # Do NOT mutate belief_state or the inference cache.
            return self.belief_state.copy(), _FREE_ENERGY_SENTINEL
        predicted_obs = self.predict_observation(self.belief_state)
        error = observation[: self.obs_dim] - predicted_obs[: len(observation)]
        if len(error) < self.obs_dim:
            error = np.pad(error, (0, self.obs_dim - len(error)))
        prediction_error = float(np.mean(error ** 2))
        gradient = error @ self.emission.T
        new_state = self.belief_state + 0.1 * gradient
        self.belief_state = new_state
        # Cache the inference context for ``update``'s gradient descent.
        self._last_state = new_state.copy()
        self._last_observation = observation[: self.obs_dim].astype(float).copy()
        if len(self._last_observation) < self.obs_dim:
            self._last_observation = np.pad(
                self._last_observation, (0, self.obs_dim - len(self._last_observation))
            )
        self._last_error = error.copy()
        return new_state, prediction_error

    def emission_gradient_step(self, lr: float) -> None:
        """One gradient-descent step on ``emission`` using the cached context.

        Minimises ``||obs - state @ emission||^2``; the gradient w.r.t.
        ``emission`` is ``-2 * outer(state, error)``, so the descent update is
        ``emission += 2 * lr * outer(state, error)`` (which moves ``state @
        emission`` toward ``obs``). When no inference has been cached yet,
        falls back to using ``belief_state`` and a zero target observation --
        the resulting gradient is non-zero as long as ``belief_state`` is
        non-zero (which it is after the C-batch init fix above).
        """
        if lr == 0.0:
            return
        state = self._last_state if self._last_state is not None else self.belief_state
        if self._last_error is not None and self._last_observation is not None:
            # Recompute the error against the *current* emission so the
            # gradient reflects the latest emission (the cached error was
            # computed against the emission at inference time).
            predicted_obs = state @ self.emission
            obs = self._last_observation
            error = obs[: self.obs_dim] - predicted_obs[: self.obs_dim]
            if len(error) < self.obs_dim:
                error = np.pad(error, (0, self.obs_dim - len(error)))
        else:
            # No cached observation: target a zero observation. The gradient
            # is then ``-2 * outer(state, -state @ emission)`` which is still
            # a meaningful descent step toward zero prediction.
            predicted_obs = state @ self.emission
            error = -predicted_obs[: self.obs_dim]
            if len(error) < self.obs_dim:
                error = np.pad(error, (0, self.obs_dim - len(error)))
        # Descent step: emission += 2 * lr * outer(state, error).
        # ``emission`` is (state_dim, obs_dim); outer(state, error) matches.
        self.emission[:, : self.obs_dim] += 2.0 * lr * np.outer(state, error)


class HomeostaticController:
    """Maintains internal equilibrium."""

    def __init__(self, dim: int = 64, target: np.ndarray | None = None):
        self.target = target if target is not None else np.zeros(dim)
        self.dim = dim
        self.tolerance = 0.5

    def deviation(self, state: np.ndarray) -> float:
        s = state[: self.dim]
        if len(s) < self.dim:
            s = np.pad(s, (0, self.dim - len(s)))
        return float(np.linalg.norm(s - self.target))

    def regulate(self, state: np.ndarray) -> np.ndarray:
        s = state[: self.dim]
        if len(s) < self.dim:
            s = np.pad(s, (0, self.dim - len(s)))
        correction = (self.target - s) * 0.1
        return correction


class ActiveInferenceEngine(CognitiveModule):
    """
    Active Inference Engine based on Free Energy Principle.
    - Minimizes variational free energy (prediction error + complexity)
    - Epistemic foraging: actively seeks information
    - Homeostatic regulation
    """

    def __init__(
        self,
        state_dim: int = 64,
        obs_dim: int = 32,
        action_dim: int = 16,
        rng: np.random.Generator | None = None,
    ):
        # Round-3 audit CRIT-1: per-module Generator
        self._rng = rng if rng is not None else np.random.default_rng()
        self.blanket = MarkovBlanket.create(obs_dim, action_dim, state_dim, rng=self._rng)
        self.generative_model = GenerativeModel(state_dim, obs_dim, rng=self._rng)
        self.homeostasis = HomeostaticController(state_dim)
        # Bounded deques so long-running engines do not leak memory (Fix 8).
        self.action_history: deque = deque(maxlen=1000)
        self.free_energy_history: deque = deque(maxlen=1000)

    def compute_free_energy(self, observation: np.ndarray) -> float:
        """Pure variational free energy ``F = KL(q || p) + E_q[prediction_error]``.

        Does NOT mutate ``belief_state`` (uses ``infer_state`` which now
        returns a new state). Safe to call from read-only analytics methods.

        C-batch fix: the previous implementation returned
        ``prediction_error + complexity`` where ``complexity`` was just
        ``||belief||^2 * 0.01`` -- a magnitude penalty, NOT a KL divergence.
        The true variational free energy under the FEP is

            F = KL(q(s) || p(s)) + E_q[ - log p(o | s) ]

        where ``q(s)`` is the (Gaussian) variational posterior over states and
        ``p(s) = N(0, I)`` is the standard-normal prior. We approximate
        ``q(s) = N(belief, sigma_q^2 I)`` with ``sigma_q^2`` estimated from
        the variance of recent actions (a proxy for the engine's state
        uncertainty), falling back to ``sigma_q^2 = 1`` when no actions have
        been recorded yet.

        The KL term for ``q = N(b, sigma^2 I)`` vs ``p = N(0, I)`` is

            KL = 0.5 * (||b||^2 + sigma^2 * dim - dim - dim * log(sigma^2))

        which reduces to ``0.5 * ||b||^2`` when ``sigma = 1`` -- so the new
        term is a strict superset of the old complexity penalty, and existing
        tests that only check energy decreases still hold.

        Round-6 audit API6-3-1: returns ``_FREE_ENERGY_SENTINEL`` (1e6) when
        ``observation`` is non-finite (NaN/Inf) or when the computed FE is
        non-finite. Callers reading this as a finite float should treat
        ``fe == _FREE_ENERGY_SENTINEL`` as the anomaly signal (used by
        ``AnomalyDetector``'s z-score). The type contract is preserved
        (always ``float``), but the value-domain on bad inputs changed in
        Round-5 TEST5-1/TEST5-5.
        """
        # Round-5 audit TEST5-1: do NOT sanitize the observation at entry.
        # The Round-4 NEW-2 fix added ``np.nan_to_num`` here, but that
        # silently turned a NaN observation into a zeros vector, producing a
        # "normal" small free energy — masking the very anomaly that
        # ``AnomalyDetector`` (which z-scores this value) needs to see. The
        # downstream guards below already handle NaN/Inf correctly: a NaN
        # observation makes ``pred_error`` NaN, which the guard clamps to the
        # sentinel value, so the free energy spikes and the anomaly is
        # flagged. ``infer_state`` is pure (does not mutate belief_state), so
        # letting NaN flow through it is safe.
        observation = np.asarray(observation, dtype=float)
        inferred, pred_error = self.generative_model.infer_state(observation)
        # Guard the result so callers never receive a NaN free energy (which
        # would break argmin in select_action). The sentinel IS the anomaly
        # signal — do not collapse it to a small value.
        if not np.isfinite(pred_error):
            pred_error = _FREE_ENERGY_SENTINEL
        if not np.all(np.isfinite(inferred)):
            inferred = np.nan_to_num(inferred, nan=0.0, posinf=0.0, neginf=0.0)
        # Estimate variational posterior variance sigma_q^2 from recent
        # action variance (uncertainty about the next state -> uncertainty
        # about the posterior). Fall back to 1.0 when no actions recorded.
        if len(self.action_history) >= 2:
            recent = np.asarray(list(self.action_history)[-32:], dtype=float)
            # Round-3 audit: sanitize NaN/Inf before np.var, and guard the
            # result — the ``+ 1e-6`` floor only helps for small positive
            # values, not for NaN (which propagates through np.log).
            recent = np.nan_to_num(recent, nan=0.0, posinf=0.0, neginf=0.0)
            sigma_q2 = float(np.mean(np.var(recent, axis=0))) + 1e-6
            if not np.isfinite(sigma_q2) or sigma_q2 <= 0:
                sigma_q2 = 1.0
        else:
            sigma_q2 = 1.0
        dim = float(self.generative_model.state_dim)
        b_norm_sq = float(np.dot(inferred, inferred))
        if not np.isfinite(b_norm_sq):
            b_norm_sq = _FREE_ENERGY_SENTINEL
        # KL(N(belief, sigma^2 I) || N(0, I))
        kl_qp = 0.5 * (
            b_norm_sq
            + sigma_q2 * dim
            - dim
            - dim * float(np.log(sigma_q2))
        )
        # Pragmatic term: scaled prediction error (negative log-likelihood proxy).
        fe = pred_error + kl_qp
        # Final guard: if anything still escaped (shouldn't happen, but
        # defense-in-depth), return the sentinel rather than NaN.
        return float(fe) if np.isfinite(fe) else _FREE_ENERGY_SENTINEL

    def select_action(self, belief: np.ndarray) -> np.ndarray:
        """Select action that minimizes expected free energy.

        C-batch fix: integrates three previously-disconnected pieces of the
        active-inference architecture:

        1. ``MarkovBlanket.active_weights`` -- the previous implementation
           sampled purely random candidate actions, completely ignoring the
           blanket's ``active_weights`` projection from internal states to
           actions. Now the candidate actions are drawn *around* the blanket
           projection ``belief @ active_weights`` so the action selection
           actually uses the learned sensory-active coupling.

        2. ``epistemic_foraging`` -- the previous ``epistemic_foraging``
           method existed but was never called from ``select_action``. Now
           the expected free energy includes an *epistemic* term
           (``-lambda * var(predicted_state)``) so the engine prefers actions
           that visit uncertain regions (information gain), matching the FEP
           principle of active inference.

        3. ``compute_free_energy`` (pure) -- unchanged, used for the pragmatic
           term.

        Uses ``compute_free_energy`` (pure) so this never mutates
        ``belief_state`` either.
        """
        best_action = None
        best_efep = float("inf")
        # Markov blanket projection: belief -> mean action. Shape
        # (active_dim,) = belief @ active_weights.
        aw = self.blanket.active_weights  # (internal_dim, active_dim)
        if belief.shape[0] == aw.shape[0]:
            mean_action = belief @ aw
        else:
            mean_action = np.zeros(self.blanket.active_dim)
        for _ in range(8):
            # Sample around the blanket-projected mean rather than around 0,
            # so the action selection uses the sensory-active coupling learned
            # by the Markov blanket.
            # Round-3 audit CRIT-1: per-module Generator
            candidate = mean_action + self._rng.standard_normal(self.blanket.active_dim) * 0.5
            predicted_state = self.generative_model.predict_next_state(belief, candidate)
            predicted_obs = self.generative_model.predict_observation(predicted_state)
            # Pragmatic term: expected prediction error under this action.
            efe = self.compute_free_energy(predicted_obs)
            # Homeostatic term: deviation from the target state.
            homeostatic_dev = self.homeostasis.deviation(predicted_state)
            # Epistemic term: information gain = -var(predicted_state).
            # High variance -> high information potential -> lower EFE.
            # Scaled by a small lambda so the pragmatic term still dominates.
            epistemic_bonus = -0.05 * float(np.var(predicted_state))
            total = efe + 0.1 * homeostatic_dev + epistemic_bonus
            if total < best_efep:
                best_efep = total
                best_action = candidate
        return best_action if best_action is not None else np.zeros(self.blanket.active_dim)

    def epistemic_foraging(self, belief: np.ndarray) -> Signal | None:
        """Actively seek information when uncertain.

        Still callable standalone (preserves the existing API and tests), but
        is now also integrated into ``select_action`` via the epistemic bonus
        term -- so the engine performs epistemic foraging *implicitly* on
        every action selection rather than only when this method is called
        explicitly.
        """
        uncertainty = float(np.var(belief))
        # Round-3 audit: np.var of empty/NaN arrays returns NaN; guard.
        if not np.isfinite(uncertainty):
            uncertainty = 1.0
        if uncertainty > 0.1:
            # Round-3 audit CRIT-1: per-module Generator
            exploration = self._rng.standard_normal(self.blanket.active_dim) * uncertainty
            return Signal(
                data=exploration,
                metadata={"type": "epistemic", "uncertainty": uncertainty},
            )
        return None

    def process(self, signal: Signal) -> Signal:
        # ``update_belief`` mutates the shared belief_state (this is the only
        # place analytics-callable code paths intentionally update the belief).
        belief, pred_error = self.generative_model.update_belief(signal.data)
        # Round-6 audit THEORY6-1: use the SAME KL-based free-energy formula
        # as ``compute_free_energy`` so anomaly detection (which consumes
        # ``free_energy_history``) and action selection (which calls
        # ``compute_free_energy``) agree on what "free energy" means. The old
        # ``pred_error + ||belief||^2 * 0.01`` was the legacy magnitude penalty
        # that the C-batch replaced inside ``compute_free_energy`` — but
        # ``process`` was never updated, so the two paths reported different
        # FE values for the same observation.
        free_energy = self.compute_free_energy(signal.data)
        # ``compute_free_energy`` may return the sentinel when ``signal.data``
        # is non-finite; fall back to the raw prediction error so the history
        # always carries a finite value for downstream AnomalyDetector.
        if free_energy == _FREE_ENERGY_SENTINEL or not np.isfinite(free_energy):
            free_energy = float(pred_error) if np.isfinite(pred_error) else 0.0
        self.free_energy_history.append(free_energy)
        action = self.select_action(belief)
        self.action_history.append(action)
        correction = self.homeostasis.regulate(belief)
        output = belief + correction
        return Signal(data=output, metadata={"free_energy": free_energy})

    def predict(self, signal: Signal) -> Prediction:
        # Seed the prediction from the incoming signal rather than the (shared)
        # internal belief_state, so predictions actually reflect the input.
        # ``predict_observation`` does ``state @ emission`` and expects a
        # state_dim-length vector, so we pad/pad the signal to state_dim
        # (not obs_dim) to avoid a shape mismatch when obs_dim != state_dim.
        gm = self.generative_model
        state = signal.data[: gm.state_dim]
        if len(state) < gm.state_dim:
            state = np.pad(state, (0, gm.state_dim - len(state)))
        predicted_obs = gm.predict_observation(state)
        # Round-3 audit: np.var of empty/NaN returns NaN; guard so a single
        # bad module cannot poison the integration softmax in think().
        var = float(np.var(predicted_obs))
        uncertainty = var if np.isfinite(var) else 1.0
        return Prediction(value=predicted_obs, uncertainty=uncertainty)

    def update(self, prediction_error: float) -> None:
        """Update the generative model from a prediction error signal.

        C-batch fix: the previous implementation added
        ``randn(*emission.shape) * prediction_error * 0.001`` to ``emission`` --
        a *random walk* in emission space scaled by the error, which has no
        gradient-descent interpretation and would only converge by accident.
        The new implementation performs a real gradient-descent step on
        ``emission`` using the cached inference context
        (``(state, observation, error)`` from the last ``update_belief``
        call), minimising ``||obs - state @ emission||^2``:

            emission += 2 * lr * outer(state, error)

        where ``lr = 0.001 * prediction_error`` so the step magnitude scales
        with the error signal (preserving the test contract that
        ``update(0.0)`` is a no-op). When no inference has been cached yet
        (``update`` called before any ``process``), the gradient is computed
        from ``belief_state`` and a zero target observation -- still a
        well-defined descent step as long as ``belief_state`` is non-zero
        (guaranteed by the C-batch init fix).
        """
        if not np.isfinite(prediction_error):
            return
        prediction_error = float(np.clip(prediction_error, -1e6, 1e6))
        if prediction_error == 0.0:
            return
        # Round-3 audit: clamp lr to prevent divergence when prediction_error
        # is near the 1e6 ceiling (lr=1000 would overshoot wildly).
        lr = float(np.clip(0.001 * prediction_error, -0.1, 0.1))
        self.generative_model.emission_gradient_step(lr)
