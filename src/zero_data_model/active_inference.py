# src/zero_data_model/active_inference.py
"""Active Inference Engine based on Free Energy Principle."""

from __future__ import annotations

import threading
import warnings
from collections import deque
from dataclasses import dataclass

import numpy as np

from .base import CognitiveModule, Prediction, Signal
from .s4 import S4Layer

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
        self,
        state_dim: int = 64,
        obs_dim: int = 32,
        rng: np.random.Generator | None = None,
        # --- Phase G (S4): optional structured state-space predictor ------- #
        # 当 use_s4=True 时，启用 S4 层替代固定转移矩阵 ``belief @ transition``。
        # S4 通过 HiPPO 对角初始化 + Hebbian 局部更新捕获长程时序依赖，
        # 默认 False 保证零回归（现有 1889+ 测试不受影响）。
        use_s4: bool = False,
        s4_dt: float = 0.1,
        s4_lr: float = 0.01,
        s4_seed: int | None = None,
    ):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        # Round-3 audit CRIT-1: per-module Generator
        self._rng = rng if rng is not None else np.random.default_rng()
        self.transition = self._rng.standard_normal((state_dim, state_dim)) * 0.05
        # Week-1 perf: ``emission`` is a property (see below) whose setter
        # refreshes ``_cached_emission_fro`` on every assignment, so both the
        # ``__init__`` assignment here and any external replacement (e.g. a
        # test that zero-outs the emission to isolate a term) keep the cache
        # in sync. In-place mutation (``emission[:, :obs] += ...`` inside
        # ``emission_gradient_step``) bypasses the setter, so that path
        # refreshes the cache explicitly at the end of the step.
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

        # --- Phase G (S4): structured state-space predictor ---------------- #
        # S4 替代固定转移矩阵 ``belief @ transition``。S4 的 step() 是有状态的
        # （维护隐状态 x_k 捕获历史），因此在线学习时每步调用 step() 会推进
        # 隐状态。为了实现 Hebbian 更新（预测误差驱动），需要延迟一步：
        #   cycle T:   predict_next_state(belief_T) → s4.step(belief_T) → y_T
        #              缓存 (belief_T, y_T)
        #   cycle T+1: update_belief(obs_{T+1}) → new_belief_{T+1}
        #              s4_error = y_T - new_belief_{T+1}
        #              s4.update(s4_error, u=belief_T)
        # 这样 S4 的预测 y_T（基于 belief_T）与实际的 new_belief_{T+1} 比较，
        # 误差驱动 C/B 的局部 Hebbian 更新（无反向传播）。
        self.use_s4 = bool(use_s4)
        if self.use_s4:
            # S4 层：state_dim = input_dim = output_dim = state_dim
            # （S4 的输入是 belief，输出是预测的下一 belief，维度一致）
            self._s4_layer = S4Layer(
                state_dim=state_dim,
                input_dim=state_dim,
                output_dim=state_dim,
                dt=s4_dt,
                lr=s4_lr,
                seed=s4_seed,
            )
        else:
            self._s4_layer = None
        # 延迟 Hebbian 更新缓存：上一周期 S4 的输入与输出
        self._s4_prev_input: np.ndarray | None = None
        self._s4_prev_output: np.ndarray | None = None

    def predict_observation(self, state: np.ndarray) -> np.ndarray:
        return state @ self.emission

    @property
    def emission(self) -> np.ndarray:
        """Emission matrix ``state -> observation``.

        Week-1 perf: backed by ``self._emission``. The setter refreshes
        ``_cached_emission_fro`` on every assignment so that external
        replacement (e.g. ``gm.emission = np.zeros(...)`` in tests) keeps
        the Frobenius-norm cache in sync. In-place mutation
        (``emission[:, :obs] += ...`` in ``emission_gradient_step``)
        bypasses the setter; that path refreshes the cache explicitly.
        """
        return self._emission

    @emission.setter
    def emission(self, value: np.ndarray) -> None:
        self._emission = np.asarray(value, dtype=float)
        self._cached_emission_fro = float(np.linalg.norm(self._emission))

    def predict_next_state(self, state: np.ndarray, action: np.ndarray | None = None) -> np.ndarray:
        # Phase G (S4): 当 use_s4=True 时，用 S4 层的有状态 step() 替代
        # ``state @ transition``。S4 的隐状态 x_k 捕获长程时序依赖（HiPPO
        # 对角初始化保证稳定衰减），输出 y_k = C x_k + D u_k 是对下一状态
        # 的预测。step() 推进隐状态（在线学习），并缓存 (input, output) 对
        # 供下一周期的延迟 Hebbian 更新使用。
        if self.use_s4 and self._s4_layer is not None:
            next_state = self._s4_layer.step(state)
            # 缓存本周期 S4 的输入与输出，供下一周期 update_belief 计算
            # 延迟预测误差 s4_error = y_T - new_belief_{T+1}
            self._s4_prev_input = np.asarray(state, dtype=float).copy()
            self._s4_prev_output = np.asarray(next_state, dtype=float).copy()
        else:
            next_state = state @ self.transition
        if action is not None:
            padded = np.zeros(self.state_dim)
            padded[: len(action)] = action
            next_state = next_state + padded
        return next_state

    def predict_next_state_batch(
        self, states: np.ndarray, actions: np.ndarray
    ) -> np.ndarray:
        """Vectorised batch form of ``predict_next_state``.

        ``states`` is ``(B, state_dim)`` and ``actions`` is ``(B, action_dim)``;
        returns ``(B, state_dim)`` via ``states @ self.transition +
        padded_actions`` (each action zero-padded to ``state_dim``). Mirrors the
        single-sample math exactly so callers can substitute the per-sample
        loop without changing results. ``select_action`` keeps its own hoisted
        ``base_next_state`` (PERF8-4) and only adds the padded actions in batch,
        so it does not recompute the invariant ``belief @ transition`` matmul.

        Phase G (S4): 当 use_s4=True 时，batch 模式采用 *peek*（查询）语义——
        基于 S4 当前隐状态 x_k 计算每个候选状态的预测输出，但不推进隐状态
        （counterfactual 评估不应污染在线状态）。公式：
            y_i = C @ (A_bar * x_k + B_bar @ states[i]) + D @ states[i]
        这与 step() 的数学一致，只是不写入 _state。
        """
        states = np.asarray(states, dtype=float)
        actions = np.asarray(actions, dtype=float)
        B = states.shape[0]
        padded = np.zeros((B, self.state_dim), dtype=float)
        padded[:, : actions.shape[1]] = actions
        if self.use_s4 and self._s4_layer is not None:
            # Peek 模式：不推进 S4 隐状态
            s4 = self._s4_layer
            with s4._lock:
                x_k = s4._state
                # 对每个候选状态计算 y_i = C @ (A_bar*x + B_bar@states[i]) + D@states[i]
                # 批量化：(B, state_dim) @ B_bar.T → (B, state_dim)
                new_x = s4._A_bar * x_k + states @ s4._B_bar.T  # (B, state_dim)
                y = new_x @ s4._C.T + states @ s4._D.T  # (B, output_dim=state_dim)
            return y + padded
        return states @ self.transition + padded

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
        # Phase G (S4): 延迟一步的 Hebbian 更新。上一周期 predict_next_state
        # 缓存了 S4 的输入 (belief_T) 与输出 (y_T)。本周期 update_belief 得到
        # 新的 new_belief_{T+1}，即可计算 S4 的预测误差：
        #     s4_error = y_T - new_belief_{T+1}
        # 并用此误差驱动 S4 的 C/B 局部 Hebbian 更新（无反向传播）。
        # 这与 spec 1.1 "update(error) 使用预测误差驱动 A, B, C 的局部 Hebbian
        # 更新" 一致——A 不更新（HiPPO 稳定性保证），B/C 用误差驱动。
        if (
            self.use_s4
            and self._s4_layer is not None
            and self._s4_prev_output is not None
            and self._s4_prev_input is not None
        ):
            s4_error = self._s4_prev_output - new_state
            # 仅在误差有限时更新（NaN 防护）
            if np.all(np.isfinite(s4_error)):
                self._s4_layer.update(s4_error, u=self._s4_prev_input)
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
        # Week-1 perf: ``emission`` just mutated -> recompute the cached
        # Frobenius norm so downstream ``compute_free_energy`` calls see the
        # new value. This is the sole invalidation point.
        self._cached_emission_fro = float(np.linalg.norm(self.emission))


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
        # --- Curiosity / exploration (Phase E) ----------------------- #
        # All defaults preserve the pre-Phase-E behaviour when
        # ``exploration_beta_start == 0`` (the -beta*info_gain term
        # vanishes and select_action reduces to its original form).
        exploration_beta_start: float = 1.0,
        exploration_beta_min: float = 0.01,
        exploration_decay_steps: int = 5000,
        exploration_decay_type: str = "linear",
        n_action_bins: int = 8,
        action_error_window: int = 10,
        # Phase H: number of candidate actions sampled in ``select_action``.
        # Pre-Phase-H this was hardcoded to 8 (the value used by the sandbox
        # loop). Phase H (text-reading) needs 5 discrete navigation actions
        # (forward / backward / fast-forward / fast-backward / stay) and
        # uses ``num_candidates=5`` so that ``best_idx`` (the chosen
        # candidate index) directly maps to a navigation action.
        # Backward-compat: default 8 keeps the sandbox behaviour unchanged.
        num_candidates: int = 8,
        # --- Phase G (S4): structured state-space predictor --------------- #
        # use_s4=False 时使用固定转移矩阵 ``belief @ transition``（零回归）；
        # use_s4=True 时启用 S4 层（HiPPO 对角初始化 + Hebbian 局部更新），
        # 捕获长程时序依赖。详见 ``GenerativeModel`` 的 S4 集成注释。
        use_s4: bool = False,
        s4_dt: float = 0.1,
        s4_lr: float = 0.01,
        s4_seed: int | None = None,
    ):
        # Round-3 audit CRIT-1: per-module Generator
        self._rng = rng if rng is not None else np.random.default_rng()
        # Per-module re-entrant lock (Phase D / think() lockless update):
        # protects ``process`` / ``predict`` / ``update`` /
        # ``compute_free_energy`` / ``select_action`` from concurrent
        # think() calls racing on ``self._rng`` and shared mutable
        # state (blanket, generative_model, action_history,
        # free_energy_history). RLock allows re-entry (process calls
        # compute_free_energy + select_action internally).
        self._lock = threading.RLock()
        self.blanket = MarkovBlanket.create(obs_dim, action_dim, state_dim, rng=self._rng)
        self.generative_model = GenerativeModel(
            state_dim,
            obs_dim,
            rng=self._rng,
            use_s4=use_s4,
            s4_dt=s4_dt,
            s4_lr=s4_lr,
            s4_seed=s4_seed,
        )
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

        # --- Phase E: intrinsic curiosity (Plan A info-gain proxy) --- #
        # Theory: EFE ≈ pragmatic (prediction error) - epistemic (info gain).
        # We add an explicit temporal info-gain term: -beta * IG(a), where
        # IG(a) is approximated by the std of recent prediction errors for
        # actions in the same angular bin as ``a`` (Plan A in the spec).
        # High std => unpredictable outcomes => high learning potential.
        # The existing ``epistemic_bonus`` (cross-candidate variance of
        # predicted states) is a ONE-SHOT spatial measure; this new term
        # is a TEMPORAL measure of outcome unpredictability. They are
        # complementary and both are kept.
        #
        # Action binning: continuous action vectors are bucketed by angle
        # into ``n_action_bins`` sectors (default 8, 45° each). This keeps
        # the proxy action-space-agnostic (decoupled from any specific
        # downstream discretisation like the sandbox's 0-3) while giving
        # enough granularity to distinguish exploration directions.
        self.exploration_beta_start = float(exploration_beta_start)
        self.exploration_beta_min = float(exploration_beta_min)
        self.exploration_decay_steps = int(exploration_decay_steps)
        self.exploration_decay_type = str(exploration_decay_type)
        self._n_action_bins = int(n_action_bins)
        # Per-bin sliding window of recent prediction errors (pragmatic
        # term only, NOT the total EFE — recording EFE would create
        # circular feedback since EFE already includes -beta*IG).
        self._action_error_history: list[deque] = [
            deque(maxlen=action_error_window) for _ in range(self._n_action_bins)
        ]
        # Step counter for beta scheduling. Incremented once per
        # ``select_action`` call. NOT incremented by analytics-only
        # ``compute_free_energy`` calls, so beta reflects the agent's
        # actual interaction history, not read-only inspection.
        self._exploration_step = 0
        # Phase H: configurable candidate count (was hardcoded 8).
        # Validated to >= 1 so an accidental 0 doesn't crash the loop.
        if num_candidates < 1:
            raise ValueError(
                f"num_candidates must be >= 1, got {num_candidates}"
            )
        self._num_candidates = int(num_candidates)
        # Phase H: last selected candidate index (0..num_candidates-1).
        # Exposed so callers (e.g. ``run_text_curious.py``) can map the
        # chosen index to a discrete navigation action without re-running
        # the selection. Updated by ``select_action`` under the lock.
        self._last_selected_idx: int = 0

    def __getstate__(self) -> dict:
        # Phase D: per-module RLock is not picklable. Strip it here and
        # rebuild in __setstate__. Hold the lock so concurrent think()
        # blocks during the snapshot (consistent array copy).
        with self._lock:
            return {k: v for k, v in self.__dict__.items() if k != "_lock"}

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._lock = threading.RLock()

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

        Round-10 audit R10-A-002 (sigma_q^2 semantic refactor): the previous
        code added a tiny ``1e-6`` floor for numerical safety (so ``log``
        wouldn't divide by zero in the KL term), but this let ``sigma_q^2``
        collapse to ``~1e-6`` whenever recent actions were uniform — which
        then blew up the KL term in ``compute_free_energy``
        (``0.5 * (||b||^2 + sigma^2 * dim - dim - dim * log(sigma^2))``;
        for ``sigma^2 = 1e-6`` and ``dim = 64`` this is
        ``0.5 * (||b||^2 + 883) ≈ 441`` even with ``b = 0``). A uniform
        recent-action stream is a poor proxy for "extreme state certainty":
        it more likely means the policy is *stuck* (a degenerate attractor)
        than that the posterior is genuinely tight. The new clamp
        ``[1e-3, 1.0]`` enforces two semantic invariants:

          * **Lower bound ``1e-3``**: even with zero action variance, retain
            a small but non-trivial residual uncertainty about the state
            (1% of the prior variance). This caps the KL contribution of
            ``-dim * log(sigma^2)`` at ``-dim * log(1e-3) = dim * 6.9``
            (~442 for ``dim = 64``) instead of ``dim * 13.8`` (~884), and
            keeps the EFE dominated by genuine prediction error rather
            than by the proxy's collapse.
          * **Upper bound ``1.0``**: matches the uninformative ``N(0, I)``
            prior — Bayesian updating cannot make the posterior *wider*
            than the prior. Action variance ``> 1`` (e.g. large motor
            commands) used to inflate ``sigma_q^2`` above 1, incorrectly
            signalling more posterior uncertainty than the prior carries.
        """
        if not self._sigma_q2_dirty:
            return self._cached_sigma_q2
        if len(self.action_history) >= 2:
            recent = np.asarray(list(self._recent_actions), dtype=float)
            recent = np.nan_to_num(recent, nan=0.0, posinf=0.0, neginf=0.0)
            raw_var = float(np.mean(np.var(recent, axis=0)))
            sigma_q2 = float(np.clip(raw_var, 1e-3, 1.0))
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
        with self._lock:
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
            # Week-1 perf: ``_cached_emission_fro`` is maintained by the
            # ``emission`` property setter (which fires on replacement) and by
            # ``emission_gradient_step`` (the sole in-place mutator in the
            # production think() cycle, which refreshes the cache at the end).
            # However, CFE recomputes ``||emission||_F^2`` from the LIVE
            # ``emission`` array rather than trusting the cache, because numpy
            # in-place slice mutation (``emission[:] = 0.0``, ``emission[...] =
            # ...``) bypasses the property setter and would leave the cache
            # stale -- and several regression tests (e.g. test_p11 which zeroes
            # the emission to isolate the KL term) rely on CFE reflecting such
            # mutations. The recomputation is O(state_dim * obs_dim) and runs
            # once per CFE call; at the default (64, 32) dims this is a
            # sub-microsecond ``np.dot(ravel, ravel)``.
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
        with self._lock:
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
            # Phase H: ``num_candidates`` was previously hardcoded to 8.
            # Configurable via ``__init__`` so text-reading loops can use
            # 5 candidates (matching the 5 navigation actions).
            n_cand = self._num_candidates
            # Week-1 perf: vectorise the candidate loop. Sample all candidates
            # at once -- ``standard_normal((n_cand, active_dim))`` consumes the
            # same RNG stream (n_cand * active_dim draws, same order) as the
            # previous ``n_cand`` calls to ``standard_normal(active_dim)``, so
            # the candidate distribution is bit-for-bit identical.
            # Sample around the blanket-projected mean rather than around 0,
            # so the action selection uses the sensory-active coupling learned
            # by the Markov blanket.
            # Round-3 audit CRIT-1: per-module Generator
            candidates = mean_action + self._rng.standard_normal(
                (n_cand, self.blanket.active_dim)
            ) * 0.5  # (n_cand, active_dim)
            # Apply only the action's additive contribution (transition part
            # already in base_next_state via the PERF8-4 hoist). Pads to
            # state_dim in batch -- ``predict_next_state_batch`` exposes the
            # same math; here we keep the hoist so the invariant
            # ``belief @ transition`` is NOT recomputed per candidate.
            state_dim = self.generative_model.state_dim
            padded = np.zeros((n_cand, state_dim))
            padded[:, : self.blanket.active_dim] = candidates
            predicted_states = base_next_state[None, :] + padded  # (n_cand, state_dim)
            # Week-1 perf: the predicted states are computed in batch above
            # (one matmul for all candidates via the PERF8-4 hoist +
            # ``predict_next_state_batch``), but the EFE itself is evaluated
            # per-candidate via ``compute_free_energy``. This preserves the
            # testable contract that CFE is called once per candidate with
            # ``state=predicted_state`` and the current observation (Round-7
            # THEORY7-2, Round-8 THEORY8-1) -- monkeypatching CFE to record
            # its arguments must observe all ``n_cand`` calls -- while the
            # batched predicted-state computation eliminates the 8 redundant
            # ``belief @ transition`` matmuls. The cached emission Frobenius
            # norm (recomputed only by ``emission_gradient_step``) saves the
            # ``np.dot(emission.ravel(), emission.ravel())`` scan inside each
            # CFE call.
            # Round-8 audit THEORY8-1: evaluate the EFE against the CURRENT
            # observation (not the predicted observation) so the NLL term is
            # non-degenerate. If no current observation is available, fall
            # back to the prior prediction target so the EFE is well-defined.
            gm = self.generative_model
            efes = np.empty(n_cand, dtype=float)
            for i in range(n_cand):
                if current_observation is not None:
                    # THEORY8-1: evaluate the EFE against the CURRENT
                    # observation so the NLL term is non-degenerate.
                    obs_for_efe = current_observation
                else:
                    # THEORY8-1 fallback: no observation available -- evaluate
                    # against the predicted observation (degenerate but
                    # well-defined EFE, kept for backward compat).
                    obs_for_efe = predicted_states[i] @ gm.emission
                efes[i] = self.compute_free_energy(
                    obs_for_efe, state=predicted_states[i]
                )
            # Homeostatic term: deviation from the target state (batched).
            # ``predicted_states`` is (n_cand, state_dim) and
            # ``homeostasis.dim == state_dim``, so no per-row padding needed.
            homeostatic_devs = np.linalg.norm(
                predicted_states - self.homeostasis.target, axis=1
            )  # (n_cand,)
            # THEORY8-12: cross-candidate variance per state dimension. Each
            # candidate's epistemic bonus is proportional to how much ITS
            # predicted state contributes to the cross-candidate spread --
            # candidates that move the predicted state away from the mean of
            # the other candidates have higher information potential.
            predicted_stack = predicted_states  # (n_cand, state_dim)
            cross_candidate_var = np.var(predicted_stack, axis=0)  # (state_dim,)
            candidate_mean = np.mean(predicted_stack, axis=0)  # (state_dim,)
            # Phase E: compute the exploration weight beta for this call
            # (decays over time so the agent shifts from exploration to
            # exploitation as it accumulates experience). beta=0 disables
            # the curiosity term entirely, recovering the pre-Phase-E
            # behaviour (backwards compatibility).
            beta = self._compute_beta()
            # Phase E: per-candidate information-gain proxy (Plan A). For each
            # candidate, look up the std of recent prediction errors in its
            # angular bin. High std => unpredictable outcomes => high
            # learning potential => the -beta*IG term lowers the EFE,
            # encouraging the agent to prefer actions with uncertain
            # outcomes (exploration).
            info_gains = [
                self._compute_information_gain_proxy(c) for c in candidates
            ]
            best_idx = 0
            for i in range(n_cand):
                # Distance of this candidate's predicted state from the
                # cross-candidate mean, weighted by the per-dimension variance.
                # Candidates in high-variance dimensions that are far from the
                # mean have higher information value.
                deviation = predicted_states[i] - candidate_mean
                epistemic_bonus = -0.05 * float(
                    np.dot(deviation * deviation, cross_candidate_var)
                    / (np.sum(cross_candidate_var) + 1e-12)
                )
                # Phase E: total EFE = pragmatic + homeostatic + epistemic
                # (one-shot spatial) - beta * info_gain (temporal). The
                # minus sign on the info_gain term is the active-inference
                # decomposition EFE = risk - epistemic_value: lower EFE =
                # better, so subtracting IG lowers EFE for informative
                # actions, making them preferred.
                total = (
                    efes[i]
                    + 0.1 * homeostatic_devs[i]
                    + epistemic_bonus
                    - beta * info_gains[i]
                )
                if total < best_efep:
                    best_efep = total
                    best_action = candidates[i]
                    best_idx = i
            # Phase E: record the chosen action's pragmatic prediction error
            # (NOT the total EFE — recording EFE would create circular
            # feedback since EFE already includes -beta*IG) into its
            # angular bin's sliding window. This populates the history that
            # the NEXT select_action call will read. The chosen action is
            # the one we actually commit to, so its realized error (proxied
            # by the candidate's efe against the current observation) is
            # the most informative sample to record.
            chosen_bin = self._action_bin(candidates[best_idx])
            self._action_error_history[chosen_bin].append(float(efes[best_idx]))
            # Phase E: advance the step counter so beta decays on the next
            # call. Incrementing AFTER the selection means the first call
            # uses beta_start (full exploration), as intended.
            self._exploration_step += 1
            # Phase H: expose the chosen candidate index for callers that
            # want a discrete action. With ``num_candidates=5`` (text
            # navigation), ``best_idx`` directly maps to navigation action
            # 0-4 (forward/backward/fast_forward/fast_backward/stay).
            self._last_selected_idx = best_idx
            return best_action if best_action is not None else np.zeros(self.blanket.active_dim)

    # ------------------------------------------------------------------ #
    # Phase E: intrinsic curiosity helpers
    # ------------------------------------------------------------------ #
    def _compute_beta(self) -> float:
        """Current exploration weight, decaying over ``_exploration_step``.

        Supports three decay schedules (selected by
        ``self.exploration_decay_type``):

        - ``"linear"`` (default): beta = max(beta_min, beta_start -
          (step / decay_steps) * (beta_start - beta_min)). Reaches
          beta_min at step == decay_steps, then stays there.
        - ``"exponential"``: beta = beta_min + (beta_start - beta_min)
          * exp(-step / decay_steps). Smooth, never quite reaches
          beta_min. The ``decay_steps`` is interpreted as the time
          constant (1/e at step == decay_steps).
        - ``"stage"``: beta = beta_start if step < decay_steps else
          beta_min. Step function — full exploration until
          ``decay_steps``, then full exploitation.

        Returns 0.0 when ``beta_start == 0`` (disables curiosity
        entirely, recovering pre-Phase-E behaviour for backwards
        compatibility).
        """
        s = self._exploration_step
        if self.exploration_beta_start <= 0.0:
            return 0.0
        b_start = self.exploration_beta_start
        b_min = self.exploration_beta_min
        T = max(1, self.exploration_decay_steps)
        decay_type = self.exploration_decay_type
        if decay_type == "exponential":
            return b_min + (b_start - b_min) * float(np.exp(-s / T))
        if decay_type == "stage":
            return b_start if s < T else b_min
        # Default: linear.
        return max(b_min, b_start - (s / T) * (b_start - b_min))

    def _action_bin(self, action_vec: np.ndarray) -> int:
        """Bucket a continuous action vector by its dominant direction.

        Uses the angle of the (vx, vy) projection (first two components)
        to assign one of ``self._n_action_bins`` angular sectors. This
        is action-space-agnostic: it does not assume any specific
        downstream discretisation (e.g. the sandbox's 0-3), only that
        the first two action dimensions carry the dominant directional
        signal. Actions with near-zero magnitude fall in bin 0
        (arbitrary but deterministic).
        """
        if action_vec.shape[0] < 2:
            return 0
        vx = float(action_vec[0])
        vy = float(action_vec[1])
        if abs(vx) < 1e-9 and abs(vy) < 1e-9:
            return 0
        angle = float(np.arctan2(vy, vx))  # (-pi, pi]
        # Shift to [0, 2pi) then bin.
        if angle < 0.0:
            angle += 2.0 * float(np.pi)
        bin_width = 2.0 * float(np.pi) / self._n_action_bins
        return int(angle / bin_width) % self._n_action_bins

    def _compute_information_gain_proxy(self, action_vec: np.ndarray) -> float:
        """Plan A info-gain proxy: std of recent prediction errors for
        actions in the same angular bin as ``action_vec``.

        Returns a default of 1.0 when the bin has fewer than 2 samples
        (early exploration — encourages trying unvisited directions by
        giving them maximal info gain). Once 2+ samples accumulate,
        returns the standard deviation of the recorded pragmatic
        prediction errors: high std => the outcome of this kind of
        action is unpredictable => high learning potential.

        This is a temporal complement to the existing spatial
        ``epistemic_bonus`` (cross-candidate variance): the bonus
        measures how much THIS candidate spreads the prediction across
        the 8 candidates (one-shot), while this proxy measures how
        unpredictable THIS direction's outcomes have been HISTORICALLY.
        """
        bin_idx = self._action_bin(action_vec)
        history = self._action_error_history[bin_idx]
        if len(history) < 2:
            # Cold-start: maximal info gain to encourage visiting
            # unexplored directions. This is the "optimism in the face
            # of uncertainty" heuristic (à la UCB).
            return 1.0
        return float(np.std(history))

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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
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
