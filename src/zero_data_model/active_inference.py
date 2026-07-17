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
        # Round-8 audit THEORY8-10: clip the gradient's Frobenius norm so a
        # large ``state`` or ``error`` (e.g. an upstream module emitting a
        # near-sentinel value through ``Signal``) cannot blow up ``emission``
        # in a single step. ``emission`` starts at scale 0.1 and the lr is
        # already clipped to [0, 0.1], so a per-step change of magnitude 1.0
        # is more than enough headroom for normal learning while capping the
        # pathological case (|state|*|error| up to 1e12 without the clip).
        grad = np.outer(state, error)
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm > 1.0:
            grad = grad / grad_norm
        self.emission[:, : self.obs_dim] += 2.0 * lr * grad


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
        # Round-8 audit PERF8-7: dedicated recent-only deque (maxlen=32) so
        # ``_compute_sigma_q2`` does not have to materialise the full
        # ``action_history`` (O(1000)) just to slice off the last 32 entries
        # every CFE call. Kept in lockstep with ``action_history`` by
        # ``process`` (both append the same action back-to-back). Matches the
        # SelfModel._recent pattern (P-CRIT-02).
        self._recent_actions: deque = deque(maxlen=32)
        # Round-8 audit PERF8-2: cache sigma_q2 across the 9 CFE calls per
        # think cycle (1 from process + 8 from select_action). Invalidated
        # only when action_history changes (i.e. once per cycle in process).
        # Without this cache, every CFE call re-materialised
        # ``list(action_history)`` (O(N), N<=1000) and recomputed np.var —
        # 9x redundant work per cycle. ``_sigma_q2_dirty`` is set True by
        # ``process`` after appending to action_history.
        self._cached_sigma_q2: float = 1.0
        self._sigma_q2_dirty: bool = True

    def _compute_sigma_q2(self) -> float:
        """Estimate the variational posterior variance ``sigma_q^2`` from
        recent action history, with caching.

        Round-8 audit PERF8-2: previously every ``compute_free_energy`` call
        re-materialised ``list(action_history)`` and recomputed ``np.var``.
        Since the result is invariant across the 9 CFE calls per cycle
        (action_history is only mutated by ``process`` after the CFE burst),
        we cache it and invalidate only when ``process`` appends a new action.

        Round-8 audit PERF8-7: ``list(action_history)[-32:]`` still
        materialised the entire 1000-entry deque (then sliced to 32). The
        dedicated ``_recent_actions: deque(maxlen=32)`` is kept in lockstep
        with ``action_history`` by ``process``, so we only ever materialise
        up to 32 entries — O(32) instead of O(1000) per CFE burst (×9 per
        cycle when the cache is cold).
        """
        if not self._sigma_q2_dirty:
            return self._cached_sigma_q2
        if len(self.action_history) >= 2:
            recent = np.asarray(list(self._recent_actions), dtype=float)
            recent = np.nan_to_num(recent, nan=0.0, posinf=0.0, neginf=0.0)
            sigma_q2 = float(np.mean(np.var(recent, axis=0))) + 1e-6
            if not np.isfinite(sigma_q2) or sigma_q2 <= 0:
                sigma_q2 = 1.0
        else:
            sigma_q2 = 1.0
        self._cached_sigma_q2 = sigma_q2
        self._sigma_q2_dirty = False
        return sigma_q2

    def compute_free_energy(
        self, observation: np.ndarray, state: np.ndarray | None = None
    ) -> float:
        """Pure variational free energy ``F = KL(q || p) + E_q[prediction_error]``.

        Does NOT mutate ``belief_state``. Safe to call from read-only analytics
        methods.

        C-batch fix: the previous implementation returned
        ``prediction_error + complexity`` where ``complexity`` was just
        ``||belief||^2 * 0.01`` -- a magnitude penalty, NOT a KL divergence.
        The true variational free energy under the FEP is

            F = KL(q(s) || p(s)) + E_q[ - log p(o | s) ]

        where ``q(s)`` is the (Gaussian) variational posterior over states and
        ``p(s) = N(0, I)`` is the standard-normal prior. We approximate
        ``q(s) = N(state, sigma_q^2 I)`` with ``sigma_q^2`` estimated from
        the variance of recent actions (a proxy for the engine's state
        uncertainty), falling back to ``sigma_q^2 = 1`` when no actions have
        been recorded yet.

        The KL term for ``q = N(b, sigma^2 I)`` vs ``p = N(0, I)`` is

            KL = 0.5 * (||b||^2 + sigma^2 * dim - dim - dim * log(sigma^2))

        which reduces to ``0.5 * ||b||^2`` when ``sigma = 1`` -- so the new
        term is a strict superset of the old complexity penalty, and existing
        tests that only check energy decreases still hold.

        Round-7 audit THEORY7-1: BOTH the KL term (``||state||^2``) and the
        pragmatic term (``||obs - state @ emission||^2``) now use the SAME
        ``state`` reference, so the two terms share a single posterior
        ``q(s) = N(state, sigma_q^2 I)``. The Round-6 fix used ``||inferred||^2``
        (one gradient step ahead) for KL but ``pred_error`` from
        ``belief_state`` for the NLL — mixing two posteriors.

        Round-7 audit THEORY7-2: accepts an optional ``state`` argument so
        ``select_action`` can evaluate the EFE under the predicted next state
        rather than the current belief. When ``state is None``, defaults to
        ``generative_model.belief_state``.

        Round-7 audit API6-3-1: returns ``_FREE_ENERGY_SENTINEL`` (1e6) when
        ``observation`` is non-finite (NaN/Inf) or when the computed FE is
        non-finite. Callers reading this as a finite float should treat
        ``fe == _FREE_ENERGY_SENTINEL`` as the anomaly signal (used by
        ``AnomalyDetector``'s z-score). The type contract is preserved
        (always ``float``), but the value-domain on bad inputs changed in
        Round-5 TEST5-1/TEST5-5.
        """
        observation = np.asarray(observation, dtype=float)
        gm = self.generative_model
        # Round-7 audit THEORY7-1: use ``state`` for BOTH terms so they share
        # the same posterior q(s) = N(state, sigma_q^2 I).
        state = gm.belief_state if state is None else np.asarray(state, dtype=float)
        # Round-7 audit TEST5-1/TEST5-3 (regression guard): when the
        # observation is non-finite, return the EXACT sentinel value
        # immediately — do NOT compute pred_error + kl_qp, which would yield
        # ``sentinel + small_kl_term`` (slightly above 1e6) and break the
        # ``fe == _FREE_ENERGY_SENTINEL`` contract that ``process`` and the
        # Round-4 regression tests rely on for anomaly detection.
        if not np.all(np.isfinite(observation)):
            return _FREE_ENERGY_SENTINEL
        predicted_obs = gm.predict_observation(state)
        error = observation[: gm.obs_dim] - predicted_obs[: len(observation)]
        if len(error) < gm.obs_dim:
            error = np.pad(error, (0, gm.obs_dim - len(error)))
        # Round-7 audit PERF7-8: np.dot(error, error) avoids the temporary
        # ``error ** 2`` array that np.mean(error ** 2) materialises.
        pred_error = float(np.dot(error, error)) / error.size
        # Guard the result so callers never receive a NaN free energy (which
        # would break argmin in select_action). The sentinel IS the anomaly
        # signal — do not collapse it to a small value.
        if not np.isfinite(pred_error):
            return _FREE_ENERGY_SENTINEL
        if not np.all(np.isfinite(state)):
            state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        # Estimate variational posterior variance sigma_q^2 from recent
        # action variance (uncertainty about the next state -> uncertainty
        # about the posterior). Fall back to 1.0 when no actions recorded.
        # Round-8 audit PERF8-2: use the cached value — ``_compute_sigma_q2``
        # only recomputes when ``action_history`` changes (once per cycle in
        # ``process``), avoiding 9x redundant O(N) re-materialisations.
        sigma_q2 = self._compute_sigma_q2()
        dim = float(gm.state_dim)
        b_norm_sq = float(np.dot(state, state))
        if not np.isfinite(b_norm_sq):
            b_norm_sq = _FREE_ENERGY_SENTINEL
        # KL(N(state, sigma^2 I) || N(0, I))
        kl_qp = 0.5 * (
            b_norm_sq
            + sigma_q2 * dim
            - dim
            - dim * float(np.log(sigma_q2))
        )
        # Round-8 audit THEORY8-2: under the variational posterior
        # ``q(s) = N(state, sigma_q^2 I)``, the expected negative
        # log-likelihood (pragmatic term) is
        #   E_q[||obs - s @ emission||^2] = ||obs - state @ emission||^2
        #                                  + sigma_q^2 * ||emission||_F^2
        # The previous code included only the first term (``pred_error``),
        # so when ``sigma_q2`` was large (high posterior uncertainty) the
        # KL term grew but the pragmatic term did NOT -- the posterior
        # variance had no effect on the pragmatic surprise, biasing
        # ``select_action`` toward over-confident actions. Adding the
        # missing ``sigma_q2 * ||emission||_F^2`` term makes the KL and
        # pragmatic terms consistent: higher posterior uncertainty now
        # RAISES the expected surprise, so ``select_action`` is rewarded
        # for visiting well-resolved states (low sigma_q2) -- the
        # epistemic-drive behaviour active inference predicts.
        emission_fro_sq = float(
            np.dot(gm.emission.ravel(), gm.emission.ravel())
        )
        pragmatic = pred_error + sigma_q2 * emission_fro_sq
        fe = pragmatic + kl_qp
        # Final guard: if anything still escaped (shouldn't happen, but
        # defense-in-depth), return the sentinel rather than NaN.
        return float(fe) if np.isfinite(fe) else _FREE_ENERGY_SENTINEL

    def select_action(
        self,
        belief: np.ndarray,
        current_observation: np.ndarray | None = None,
    ) -> np.ndarray:
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

        Round-8 audit THEORY8-1 (CRIT): the Round-7 THEORY7-2 fix passed
        ``predicted_obs = predicted_state @ emission`` as the observation to
        ``compute_free_energy(state=predicted_state)``, but CFE recomputes
        ``state @ emission`` internally — so the NLL term was identically
        zero for every candidate, silently disabling the pragmatic term of
        active inference. The fix is to evaluate the EFE against the
        *current* observation (the one ``process`` just received) so the
        pragmatic term measures how surprising the current sensory input
        would be under the predicted next state. When no observation is
        available (standalone callers, tests), fall back to the prior
        observation target ``predicted_state @ emission`` so the EFE is
        still well-defined (degenerates to the Round-7 behaviour — kept for
        backward compat).

        Round-8 audit PERF8-4: hoist ``belief @ transition`` out of the
        8-candidate loop — it is invariant across candidates, only the
        ``+ action`` term changes. Saves 7 redundant O(dim^2) matmuls per
        ``select_action`` call.
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
        # Round-8 audit PERF8-4: belief @ transition is invariant across the
        # 8 candidate actions (only the additive action term changes), so
        # hoist it out of the loop. Saves 7 redundant O(dim^2) matmuls.
        base_next_state = self.generative_model.predict_next_state(belief, action=None)
        # Round-8 audit THEORY8-12: the epistemic bonus must measure the
        # spread of predicted states ACROSS the 8 candidate actions (i.e.
        # how much each candidate resolves the uncertainty about which
        # action to take), NOT the variance across the COMPONENTS of one
        # candidate's predicted-state vector. The previous
        # ``np.var(predicted_state)`` computed the within-candidate
        # dimension-spread -- dominated by the magnitude profile of the
        # state vector, not by the action's information value. We collect
        # all candidate (action, predicted_state, efe, homeostatic_dev)
        # tuples first, then compute the cross-candidate variance per
        # state-dimension and assign each candidate its share of the
        # epistemic bonus.
        candidates: list[np.ndarray] = []
        predicted_states: list[np.ndarray] = []
        efes: list[float] = []
        homeostatic_devs: list[float] = []
        for _ in range(8):
            # Sample around the blanket-projected mean rather than around 0,
            # so the action selection uses the sensory-active coupling learned
            # by the Markov blanket.
            # Round-3 audit CRIT-1: per-module Generator
            candidate = mean_action + self._rng.standard_normal(self.blanket.active_dim) * 0.5
            # Apply only the action's additive contribution (transition part
            # already in base_next_state). Pads to state_dim.
            padded = np.zeros(self.generative_model.state_dim)
            padded[: len(candidate)] = candidate
            predicted_state = base_next_state + padded
            # Pragmatic term: expected prediction error under this action.
            # Round-8 audit THEORY8-1: evaluate the EFE against the CURRENT
            # observation (not the predicted observation) so the NLL term is
            # non-degenerate. If no current observation is available, fall
            # back to the prior prediction target so the EFE is well-defined.
            if current_observation is not None:
                efe_obs = current_observation
            else:
                efe_obs = self.generative_model.predict_observation(predicted_state)
            efe = self.compute_free_energy(efe_obs, state=predicted_state)
            # Homeostatic term: deviation from the target state.
            homeostatic_dev = self.homeostasis.deviation(predicted_state)
            candidates.append(candidate)
            predicted_states.append(predicted_state)
            efes.append(efe)
            homeostatic_devs.append(homeostatic_dev)
        # THEORY8-12: cross-candidate variance per state dimension. Each
        # candidate's epistemic bonus is proportional to how much ITS
        # predicted state contributes to the cross-candidate spread --
        # candidates that move the predicted state away from the mean of
        # the other candidates have higher information potential.
        predicted_stack = np.stack(predicted_states)  # (8, state_dim)
        cross_candidate_var = np.var(predicted_stack, axis=0)  # (state_dim,)
        candidate_mean = np.mean(predicted_stack, axis=0)  # (state_dim,)
        for i in range(8):
            # Distance of this candidate's predicted state from the
            # cross-candidate mean, weighted by the per-dimension variance.
            # Candidates in high-variance dimensions that are far from the
            # mean have higher information value.
            deviation = predicted_states[i] - candidate_mean
            epistemic_bonus = -0.05 * float(
                np.dot(deviation * deviation, cross_candidate_var)
                / (np.sum(cross_candidate_var) + 1e-12)
            )
            total = efes[i] + 0.1 * homeostatic_devs[i] + epistemic_bonus
            if total < best_efep:
                best_efep = total
                best_action = candidates[i]
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
        # Round-7 audit THEORY7-3 + PERF7-2: compute the free energy BEFORE
        # ``update_belief`` mutates ``belief_state``. The Round-6 THEORY6-1
        # fix called ``compute_free_energy`` AFTER ``update_belief``, so the
        # recorded FE was the *posterior* surprise (after the belief had
        # already moved toward the observation), not the *prior* surprise
        # that ``AnomalyDetector`` (which z-scores this value) expects.
        # Computing FE first also eliminates the redundant ``infer_state``
        # call (PERF7-2): ``compute_free_energy`` computes its own
        # prediction error from ``belief_state @ emission`` — no need for
        # ``update_belief``'s return value.
        free_energy = self.compute_free_energy(signal.data)
        # ``update_belief`` mutates the shared belief_state (this is the only
        # place analytics-callable code paths intentionally update the belief).
        belief, pred_error = self.generative_model.update_belief(signal.data)
        # ``compute_free_energy`` may return the sentinel when ``signal.data``
        # is non-finite; fall back to the raw prediction error so the history
        # always carries a finite value for downstream AnomalyDetector.
        if free_energy == _FREE_ENERGY_SENTINEL or not np.isfinite(free_energy):
            free_energy = float(pred_error) if np.isfinite(pred_error) else 0.0
        self.free_energy_history.append(free_energy)
        # Round-8 audit THEORY8-1: pass the current observation to
        # ``select_action`` so the EFE's pragmatic term is non-degenerate
        # (evaluates the predicted state's surprise against the current
        # sensory input rather than against its own prediction).
        action = self.select_action(belief, current_observation=signal.data)
        self.action_history.append(action)
        # Round-8 audit PERF8-7: keep ``_recent_actions`` in lockstep with
        # ``action_history`` so ``_compute_sigma_q2`` can read from the
        # bounded 32-entry deque instead of materialising the full
        # 1000-entry ``action_history`` every CFE call.
        self._recent_actions.append(action)
        # Round-8 audit PERF8-2: invalidate the sigma_q2 cache now that
        # action_history has a new entry; the next CFE burst will recompute.
        self._sigma_q2_dirty = True
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

        Round-8 audit THEORY8-3 (HIGH) + THEORY8-7 (MED): the previous
        ``lr = 0.001 * prediction_error`` mixed two objectives — the lr
        scale came from ``prediction_error`` (computed by ``think()`` against
        ``signal.data`` as the state, i.e. objective A) while the gradient
        direction came from ``emission_gradient_step`` using the cached
        ``(_last_state, _last_observation)`` from ``update_belief`` (i.e.
        objective B). When ``signal.data != belief_state`` (the normal
        case), the lr and the gradient were for different problems.
        Additionally, the symmetric ``(-1e6, 1e6)`` clip allowed negative
        ``prediction_error`` to invert the gradient (gradient ascent).

        The fix: derive ``lr`` from the SAME cached context that drives the
        gradient direction — specifically, from ``||_last_error||^2 / dim``
        (the belief-state-relative MSE that ``update_belief`` already
        computed). The ``prediction_error`` argument becomes a "should-update"
        gate (no-op when zero or non-finite, matching the test contract).
        This makes the lr scale consistent with the gradient objective,
        and removes the negative-lr hazard.
        """
        if not np.isfinite(prediction_error):
            return
        # Round-8 audit THEORY8-7: clip to NON-NEGATIVE — a negative
        # prediction_error would invert the gradient (ascent, not descent).
        # MSE is non-negative by construction; this is defense-in-depth.
        prediction_error = float(np.clip(prediction_error, 0.0, 1e6))
        if prediction_error == 0.0:
            return
        # Round-8 audit THEORY8-3: derive lr from the cached inference
        # context (the same one ``emission_gradient_step`` will use) so the
        # lr scale and the gradient direction are from the SAME objective.
        # ``_last_error`` is the error ``update_belief`` computed against
        # ``belief_state`` (objective B), so the lr matches the gradient
        # direction. Falls back to the argument when no inference has been
        # cached yet (preserves the pre-process ``update()`` contract).
        cached_error = self.generative_model._last_error
        if cached_error is not None and cached_error.size > 0:
            scale_error = float(np.dot(cached_error, cached_error)) / cached_error.size
            if not np.isfinite(scale_error) or scale_error <= 0:
                scale_error = prediction_error
        else:
            scale_error = prediction_error
        # Round-3 audit: clamp lr to prevent divergence when prediction_error
        # is near the 1e6 ceiling (lr=1000 would overshoot wildly).
        lr = float(np.clip(0.001 * scale_error, 0.0, 0.1))
        if lr == 0.0:
            return
        self.generative_model.emission_gradient_step(lr)
