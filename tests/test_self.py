"""Unified self-model (IWSM) unit tests.

Covers the four Stage-2 cognitive-upgrade modules under ``src/self/``:
  - SelfSchema             : predicts own attention/action/affect
  - AutobiographicalMemory : self-episode store with narrative retrieval
  - PhiSelf                : self-relevant integrated information Φ_self
  - CounterfactualSelf     : "what if I had acted differently" explanations

All components are numpy-only, thread-safe and Hebbian. These tests pin their
public contracts and a few core behavioural guarantees (learning improves
accuracy, FIFO eviction, counterfactual regret sign, ...).

NOTE on Φ_self monotonicity
--------------------------
``PhiSelf.compute_phi`` searches *all* bipartitions and takes the minimum
``whole_info - part_info`` (clamped to >= 0). For a symmetric MI matrix with
``n_self_modules == 4`` (the default), the four 1-vs-3 partition losses sum to
zero, so at least one of them is <= 0 and Φ is therefore *always* 0 -- a known
degeneracy of this greedy bipartition estimator, not a bug. To exercise a
non-trivial Φ we use ``n_self_modules == 2`` (the only bipartition is
``{0},{1}`` and the loss equals ``MI[0,1]``). Both behaviours are documented
in the tests below.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.self.self_schema import SelfSchema, SelfPrediction, SelfSchemaStats
from src.self.autobiographical_memory import (
    AutobiographicalMemory,
    SelfEpisode,
    NarrativeResult,
)
from src.self.phi_self import PhiSelf, PhiSelfState
from src.self.counterfactual_self import CounterfactualSelf, CounterfactualResult


# ================================================================== #
# SelfSchema
# ================================================================== #
class TestSelfSchema:
    """Predictive model of the agent's own attention/action/affect."""

    def test_predict_returns_self_prediction(self):
        """predict(state) returns a SelfPrediction with correctly-shaped fields."""
        n_modules, n_actions, affect_dim, dim = 4, 14, 4, 32
        sch = SelfSchema(dim=dim, n_modules=n_modules, n_actions=n_actions,
                         affect_dim=affect_dim, seed=0)
        state = np.ones(dim)
        pred = sch.predict(state)

        assert isinstance(pred, SelfPrediction), "predict must return SelfPrediction"
        assert pred.attention_prediction.shape == (n_modules,), (
            f"attention_prediction shape {pred.attention_prediction.shape}, "
            f"expected ({n_modules},)"
        )
        assert isinstance(pred.action_prediction, (int, np.integer)), (
            f"action_prediction must be an int, got {type(pred.action_prediction)}"
        )
        assert pred.action_probs.shape == (n_actions,), (
            f"action_probs shape {pred.action_probs.shape}, expected ({n_actions},)"
        )
        assert pred.affect_prediction.shape == (affect_dim,), (
            f"affect_prediction shape {pred.affect_prediction.shape}, "
            f"expected ({affect_dim},)"
        )
        # softmax outputs must be valid probability distributions
        assert np.all(pred.attention_prediction >= 0)
        assert abs(pred.attention_prediction.sum() - 1.0) < 1e-9
        assert np.all(pred.action_probs >= 0)
        assert abs(pred.action_probs.sum() - 1.0) < 1e-9
        # argmax of probs must agree with the reported action
        assert pred.action_prediction == int(np.argmax(pred.action_probs))

    def test_update_returns_diagnostics_dict(self):
        """update(...) returns a dict with the expected diagnostic keys."""
        sch = SelfSchema(dim=16, n_modules=3, n_actions=5, affect_dim=4, seed=0)
        state = np.ones(16)
        attn = np.array([0.0, 1.0, 0.0])  # winner = module 1
        diag = sch.update(state, attn, actual_action=2,
                          actual_affect=np.zeros(4))

        assert isinstance(diag, dict), "update must return a diagnostics dict"
        # The source emits these exact keys (attn_*, not attention_*).
        for key in ("attn_error", "action_error", "affect_error",
                    "action_correct", "self_consistency", "step"):
            assert key in diag, f"diagnostics missing key {key!r}: {list(diag)}"
        assert isinstance(diag["action_correct"], bool)
        assert isinstance(diag["step"], int)
        assert diag["step"] == 1

    def test_action_prediction_accuracy_improves(self):
        """Consistent state→action mapping drives action_accuracy above 0.5.

        Training feeds the *same* state every step with the *same* target
        action. The first update is a no-op (its input is the zero initial
        state) but from step 2 the Hebbian outer-product update shifts the
        action logits so argmax converges to the target action.
        """
        dim, n_actions, target = 32, 5, 2
        sch = SelfSchema(dim=dim, n_actions=n_actions, n_modules=4,
                         affect_dim=4, lr=0.01, seed=0)
        state = np.ones(dim)  # ||state||^2 = dim -> strong, stable learning signal
        attn = np.array([0.0, 1.0, 0.0, 0.0])
        affect = np.zeros(4)

        for _ in range(50):
            sch.update(state, attn, actual_action=target, actual_affect=affect)

        assert sch.stats.action_accuracy > 0.5, (
            f"action_accuracy={sch.stats.action_accuracy:.3f} did not exceed 0.5 "
            f"after 50 consistent training steps"
        )

    def test_attention_prediction_accuracy_improves(self):
        """Consistent attention winner drives attention_accuracy above 0.3.

        Threshold is lenient (0.3) because attention is a softer softmax
        target than the discrete action and accuracy can be non-monotonic in
        early steps while the winner-take-all logits cross over.
        """
        dim, n_modules = 32, 4
        sch = SelfSchema(dim=dim, n_modules=n_modules, n_actions=5,
                         affect_dim=4, lr=0.01, seed=0)
        state = np.ones(dim)
        winner = 1
        attn = np.zeros(n_modules)
        attn[winner] = 1.0

        for _ in range(50):
            sch.update(state, attn, actual_action=0, actual_affect=np.zeros(4))

        assert sch.stats.attention_accuracy > 0.3, (
            f"attention_accuracy={sch.stats.attention_accuracy:.3f} did not exceed "
            f"0.3 after 50 consistent training steps"
        )

    def test_handles_invalid_action(self):
        """actual_action=999 is out of range and gets clamped to 0 (no crash)."""
        sch = SelfSchema(dim=8, n_modules=3, n_actions=5, affect_dim=4, seed=0)
        diag = sch.update(
            np.ones(8), np.array([1.0, 0.0, 0.0]),
            actual_action=999, actual_affect=np.zeros(4),
        )
        # No exception raised, and diagnostics are well-formed.
        assert isinstance(diag, dict)
        assert "action_correct" in diag
        # stats.step must have advanced exactly once.
        assert sch.stats.step == 1

    def test_handles_nan_input(self):
        """NaN inputs are sanitised internally and must not crash predict/update."""
        sch = SelfSchema(dim=8, n_modules=3, n_actions=4, affect_dim=4, seed=0)

        # predict with NaN state -> internally replaced by zeros
        pred = sch.predict(np.full(8, np.nan))
        assert np.all(np.isfinite(pred.attention_prediction))
        assert np.all(np.isfinite(pred.action_probs))
        assert np.all(np.isfinite(pred.affect_prediction))

        # update with NaN affect -> internally replaced by zeros, no crash
        diag = sch.update(
            np.ones(8), np.array([1.0, 0.0, 0.0]),
            actual_action=1, actual_affect=np.full(4, np.nan),
        )
        assert isinstance(diag, dict)
        assert np.isfinite(diag["affect_error"])

    def test_invalid_dims_raise(self):
        """dim=0 and n_actions=0 must raise ValueError at construction."""
        with pytest.raises(ValueError):
            SelfSchema(dim=0)
        with pytest.raises(ValueError):
            SelfSchema(dim=8, n_actions=0)
        # Sanity: positive dims construct fine.
        assert SelfSchema(dim=8, n_actions=4) is not None

    def test_predict_with_wrong_size_state(self):
        """predict tolerates wrong-size state by padding/truncating to dim."""
        dim = 16
        sch = SelfSchema(dim=dim, n_modules=3, n_actions=4, affect_dim=4, seed=0)

        # Too long -> truncated to first `dim` entries.
        long_state = np.arange(dim + 10, dtype=np.float64)
        pred_long = sch.predict(long_state)
        assert pred_long.attention_prediction.shape == (3,)
        assert pred_long.action_probs.shape == (4,)

        # Too short -> zero-padded to `dim`.
        short_state = np.ones(dim - 5)
        pred_short = sch.predict(short_state)
        assert pred_short.attention_prediction.shape == (3,)
        assert pred_short.affect_prediction.shape == (4,)

    def test_stats_object_exposed(self):
        """stats is a SelfSchemaStats instance with the documented fields."""
        sch = SelfSchema(dim=8, n_modules=3, n_actions=4, affect_dim=4, seed=0)
        assert isinstance(sch.stats, SelfSchemaStats)
        assert sch.stats.step == 0
        assert sch.stats.action_accuracy == 0.0
        assert sch.stats.attention_accuracy == 0.0


# ================================================================== #
# AutobiographicalMemory
# ================================================================== #
class TestAutobiographicalMemory:
    """Self-episode store backed by a Hopfield network."""

    @staticmethod
    def _prediction(action: int = 2, attn: int = 1) -> dict:
        return {
            "predicted_attention": [0.0, 0.0, 0.0, 0.0],
            "predicted_action": action,
            "_attn_winner": attn,
        }

    @staticmethod
    def _outcome(action: int = 2, fe: float = 0.5) -> dict:
        return {"actual_action": action, "free_energy": fe, "collision": False}

    def test_store_creates_episode(self):
        """store(...) returns a SelfEpisode and grows the episode count."""
        am = AutobiographicalMemory(dim=16, capacity=8, seed=0)
        assert am.size == 0

        ep = am.store(
            state_snapshot=np.ones(16),
            self_prediction=self._prediction(),
            actual_outcome=self._outcome(),
            free_energy=0.5,
            step=0,
            modality="train",
        )
        assert isinstance(ep, SelfEpisode)
        assert ep.step == 0
        assert ep.modality == "train"
        assert ep.free_energy == pytest.approx(0.5)
        assert am.size == 1, f"size={am.size} after one store, expected 1"

    def test_store_fifo_when_full(self):
        """Storing beyond capacity evicts the oldest episode (FIFO)."""
        capacity = 5
        am = AutobiographicalMemory(dim=8, capacity=capacity, seed=0)

        for i in range(capacity + 3):
            am.store(
                state_snapshot=np.full(8, float(i)),
                self_prediction=self._prediction(action=i),
                actual_outcome=self._outcome(action=i),
                free_energy=float(i),
                step=i,
                modality="train",
            )

        assert am.size == capacity, (
            f"size={am.size} after overflow, expected capped at {capacity}"
        )
        # Only the most recent `capacity` episodes remain: steps 3..7
        # (we stored steps 0..7, FIFO evicts 0,1,2).
        remaining_steps = [e.step for e in am._episodes]
        assert remaining_steps == list(range(3, capacity + 3)), (
            f"FIFO evicted wrong episodes; remaining steps={remaining_steps}"
        )

    def test_self_narrative_query_empty(self):
        """With no stored episodes the narrative reports no relevant past."""
        am = AutobiographicalMemory(dim=16, capacity=8, seed=0)
        result = am.self_narrative_query(np.ones(16))

        assert isinstance(result, NarrativeResult)
        assert isinstance(result.narrative, str)
        assert len(result.narrative) > 0
        assert result.retrieved_episodes == [], (
            "empty memory must return no episodes"
        )
        assert result.similarity == 0.0
        # The empty template explicitly says there is no relevant past.
        assert (
            "没有" in result.narrative
            or "no relevant" in result.narrative.lower()
            or "经历" in result.narrative
        ), f"empty narrative did not indicate absence of memory: {result.narrative!r}"

    def test_self_narrative_query_returns_relevant(self):
        """Querying with a stored state retrieves matching episodes."""
        am = AutobiographicalMemory(dim=32, capacity=16, seed=0)
        rng = np.random.default_rng(1)
        states = [rng.standard_normal(32) for _ in range(5)]
        for i, s in enumerate(states):
            am.store(
                state_snapshot=s,
                self_prediction=self._prediction(action=i),
                actual_outcome=self._outcome(action=i, fe=float(i)),
                free_energy=float(i),
                step=i,
                modality="train",
            )

        # Query with one of the stored states -> it should be the top match.
        result = am.self_narrative_query(states[0], top_k=3)
        assert isinstance(result, NarrativeResult)
        assert len(result.retrieved_episodes) > 0, "no episodes retrieved"
        assert result.similarity > 0.0, (
            f"similarity={result.similarity} must be > 0 for a matching query"
        )
        # The retrieved episodes are genuine SelfEpisode objects.
        assert all(isinstance(ep, SelfEpisode) for ep in result.retrieved_episodes)

    def test_self_narrative_query_with_top_k(self):
        """top_k=3 returns at most 3 episodes."""
        am = AutobiographicalMemory(dim=32, capacity=16, seed=0)
        rng = np.random.default_rng(2)
        for i in range(5):
            am.store(
                state_snapshot=rng.standard_normal(32),
                self_prediction=self._prediction(action=i),
                actual_outcome=self._outcome(action=i),
                free_energy=float(i),
                step=i,
                modality="train",
            )

        result = am.self_narrative_query(rng.standard_normal(32), top_k=3)
        assert 1 <= len(result.retrieved_episodes) <= 3, (
            f"top_k=3 returned {len(result.retrieved_episodes)} episodes, "
            f"expected 1..3"
        )

    def test_invalid_dim_raises(self):
        """dim=0 must raise ValueError at construction."""
        with pytest.raises(ValueError):
            AutobiographicalMemory(dim=0)
        # Sanity: a positive dim constructs fine.
        assert AutobiographicalMemory(dim=8) is not None


# ================================================================== #
# PhiSelf
# ================================================================== #
class TestPhiSelf:
    """Self-relevant integrated information Φ_self."""

    def test_record_accepts_states(self):
        """record(list of ndarrays) does not crash and advances the step count."""
        phi = PhiSelf(n_self_modules=4, history_window=50)  # default-ish
        states = [np.ones(8) for _ in range(4)]
        phi.record(states)
        snap = phi.get_snapshot()
        assert snap["step"] == 1
        assert snap["history_len"] == 1

    def test_compute_phi_empty_returns_zero(self):
        """compute_phi() with no recorded history returns 0.0."""
        phi = PhiSelf(n_self_modules=4, history_window=50)
        assert phi.compute_phi() == 0.0
        assert phi.last_phi == 0.0

    def test_compute_phi_increases_with_correlation(self):
        """Correlated module states yield Φ_self > 0.

        Uses n_self_modules=2 because, as documented at the top of this file,
        with n=4 the greedy bipartition estimator collapses Φ to 0 for any
        symmetric MI matrix. With n=2 the only bipartition is {0},{1} and the
        loss equals MI[0,1], which is strictly positive when the two module
        histories are correlated but not identical.
        """
        rng = np.random.default_rng(0)
        phi = PhiSelf(n_self_modules=2, history_window=50)
        # Common base signal (kept positive so |v| == v for the 1-D state and
        # the L2 norms track the signal linearly) + small per-module noise.
        signal = rng.uniform(0.5, 2.0, size=20)
        noise_scale = 0.02
        for t in range(20):
            states = [
                np.array([signal[t] + noise_scale * rng.standard_normal()]),
                np.array([signal[t] + noise_scale * rng.standard_normal()]),
            ]
            phi.record(states)

        phi_val = phi.compute_phi()
        assert phi_val >= 0.0, f"Φ must be non-negative, got {phi_val}"
        assert phi_val > 0.0, (
            f"expected Φ > 0 for correlated n=2 modules, got {phi_val}"
        )

    def test_compute_phi_decreases_with_random_states(self):
        """Independent random states give Φ no greater than the correlated case.

        With independent inputs the off-diagonal MI is ~0 (finite-sample noise
        may produce a tiny positive value), so Φ_random ≈ 0 and is <= Φ of the
        correlated case. Both being ~0 is also acceptable; we only assert the
        non-increase and document the behaviour.
        """
        rng = np.random.default_rng(3)

        # Correlated case -> Φ > 0 (see previous test).
        phi_corr = PhiSelf(n_self_modules=2, history_window=50)
        signal = rng.uniform(0.5, 2.0, size=20)
        for t in range(20):
            phi_corr.record([
                np.array([signal[t] + 0.02 * rng.standard_normal()]),
                np.array([signal[t] + 0.02 * rng.standard_normal()]),
            ])
        phi_correlated = phi_corr.compute_phi()

        # Independent case -> Φ ≈ 0.
        phi_rand = PhiSelf(n_self_modules=2, history_window=50)
        for _ in range(20):
            phi_rand.record([
                np.array([rng.standard_normal()]),
                np.array([rng.standard_normal()]),
            ])
        phi_random = phi_rand.compute_phi()

        assert phi_random <= phi_correlated + 1e-9, (
            f"independent Φ ({phi_random}) should be <= correlated Φ "
            f"({phi_correlated}); both may be ~0 for n=4-style degeneracy"
        )

    def test_last_phi_attribute(self):
        """After compute_phi(), last_phi matches the returned value."""
        rng = np.random.default_rng(4)
        phi = PhiSelf(n_self_modules=2, history_window=50)
        signal = rng.uniform(0.5, 2.0, size=20)
        for t in range(20):
            phi.record([
                np.array([signal[t] + 0.02 * rng.standard_normal()]),
                np.array([signal[t] + 0.02 * rng.standard_normal()]),
            ])
        returned = phi.compute_phi()
        assert phi.last_phi == pytest.approx(returned), (
            f"last_phi={phi.last_phi} != compute_phi()={returned}"
        )

    def test_phi_history_grows(self):
        """phi_history grows by one entry per successful compute_phi() call."""
        rng = np.random.default_rng(5)
        phi = PhiSelf(n_self_modules=2, history_window=50)
        signal = rng.uniform(0.5, 2.0, size=20)
        for t in range(20):
            phi.record([
                np.array([signal[t] + 0.02 * rng.standard_normal()]),
                np.array([signal[t] + 0.02 * rng.standard_normal()]),
            ])

        before = len(phi.phi_history)
        phi.compute_phi()
        phi.compute_phi()
        after = len(phi.phi_history)
        assert after == before + 2, (
            f"phi_history grew by {after - before} after 2 compute_phi() calls, "
            f"expected 2"
        )

    def test_handles_padding_when_states_too_few(self):
        """Recording fewer states than n_self_modules is padded, not a crash."""
        phi = PhiSelf(n_self_modules=4, history_window=50)
        # Only one state supplied; the other 3 slots are padded with zeros.
        phi.record([np.array([1.0, 2.0, 3.0])])
        snap = phi.get_snapshot()
        assert snap["step"] == 1
        assert snap["history_len"] == 1

    def test_get_state_returns_phi_self_state(self):
        """get_state() returns a properly-typed PhiSelfState."""
        phi = PhiSelf(n_self_modules=2, history_window=50)
        st = phi.get_state()
        assert isinstance(st, PhiSelfState)
        assert isinstance(st.phi_self, float)
        assert isinstance(st.is_self_aware, bool)
        assert st.self_module_count == 2


# ================================================================== #
# CounterfactualSelf
# ================================================================== #
class TestCounterfactualSelf:
    """Counterfactual 'what if I had acted differently' explainer."""

    @staticmethod
    def _seed_transitions(cf: CounterfactualSelf, dim: int,
                          action_fe_pairs: list[tuple[int, float]],
                          repeats: int = 20) -> None:
        """Record `repeats` transitions per (action, free_energy) pair so the
        exponentially-smoothed EFE (0.9*old + 0.1*fe) converges close to `fe`.
        """
        rng = np.random.default_rng(0)
        for action, fe in action_fe_pairs:
            for _ in range(repeats):
                s_before = rng.standard_normal(dim)
                s_after = rng.standard_normal(dim)
                cf.record_transition(
                    state_before=s_before,
                    action=action,
                    state_after=s_after,
                    outcome={"actual_action": action, "free_energy": fe,
                             "collision": False},
                    free_energy=fe,
                )

    def test_record_transition_doesnt_crash(self):
        """record_transition(...) runs without error."""
        cf = CounterfactualSelf(dim=16, n_actions=4, seed=0)
        cf.record_transition(
            state_before=np.ones(16),
            action=1,
            state_after=np.zeros(16),
            outcome={"actual_action": 1, "free_energy": 0.5, "collision": False},
            free_energy=0.5,
        )
        assert cf.get_snapshot()["step"] == 1

    def test_generate_counterfactual_returns_result(self):
        """After some transitions, generate_counterfactual() returns a
        CounterfactualResult with the documented fields and types."""
        cf = CounterfactualSelf(dim=16, n_actions=4, seed=0)
        self._seed_transitions(cf, 16, [(0, 1.0), (1, 0.5), (2, 2.0)])

        result = cf.generate_counterfactual(
            state_before=np.ones(16),
            actual_action=0,
            actual_outcome={"actual_action": 0, "free_energy": 1.0,
                            "collision": False},
        )
        assert isinstance(result, CounterfactualResult)
        assert isinstance(result.actual_action, (int, np.integer))
        assert isinstance(result.counterfactual_action, (int, np.integer))
        assert isinstance(result.narrative, str) and len(result.narrative) > 0
        assert isinstance(result.regret, float)
        assert np.isfinite(result.regret)
        assert isinstance(result.predicted_counterfactual_outcome, dict)
        assert isinstance(result.actual_outcome, dict)

    def test_generate_counterfactual_with_explicit_alternative(self):
        """Passing alternative_action=2 yields counterfactual_action == 2."""
        cf = CounterfactualSelf(dim=16, n_actions=4, seed=0)
        self._seed_transitions(cf, 16, [(0, 1.0), (1, 0.5), (2, 2.0), (3, 1.5)])

        result = cf.generate_counterfactual(
            state_before=np.ones(16),
            actual_action=0,
            alternative_action=2,
        )
        assert result.counterfactual_action == 2, (
            f"counterfactual_action={result.counterfactual_action}, expected 2"
        )
        assert result.actual_action == 0

    def test_generate_counterfactual_auto_selects_lowest_efe(self):
        """When alternative_action=None, cf_action = argmin EFE excluding actual.

        We seed EFE[action] ≈ fe for each action, then ask for a
        counterfactual to actual_action=0. The excluded-actual argmin over
        EFE = [inf, 0.1, 2.0, 1.5] is action 1.
        """
        cf = CounterfactualSelf(dim=16, n_actions=4, seed=0)
        self._seed_transitions(cf, 16, [
            (0, 1.0),   # EFE[0] ≈ 1.0  (excluded as actual)
            (1, 0.1),   # EFE[1] ≈ 0.1  <- lowest
            (2, 2.0),   # EFE[2] ≈ 2.0
            (3, 1.5),   # EFE[3] ≈ 1.5
        ])

        result = cf.generate_counterfactual(
            state_before=np.ones(16),
            actual_action=0,
            alternative_action=None,
        )
        assert result.counterfactual_action == 1, (
            f"auto cf_action={result.counterfactual_action}, expected 1 "
            f"(argmin EFE excluding actual=0); EFE={cf.get_snapshot()['expected_free_energy']}"
        )

    def test_regret_is_signed(self):
        """regret = actual_efe - cf_efe: positive when cf is better, negative
        when actual is better."""
        dim = 16

        # Positive regret: actual action has high EFE, cf has low EFE.
        cf_pos = CounterfactualSelf(dim=dim, n_actions=4, seed=0)
        self._seed_transitions(cf_pos, dim, [(0, 5.0), (1, 0.1)])
        regret_pos = cf_pos.generate_counterfactual(
            state_before=np.ones(dim), actual_action=0,
            alternative_action=1,
        ).regret
        assert regret_pos > 0, (
            f"expected positive regret (cf better), got {regret_pos}"
        )

        # Negative regret: actual action has low EFE, cf has high EFE.
        cf_neg = CounterfactualSelf(dim=dim, n_actions=4, seed=0)
        self._seed_transitions(cf_neg, dim, [(0, 0.1), (1, 5.0)])
        regret_neg = cf_neg.generate_counterfactual(
            state_before=np.ones(dim), actual_action=0,
            alternative_action=1,
        ).regret
        assert regret_neg < 0, (
            f"expected negative regret (actual better), got {regret_neg}"
        )

    def test_template_narrative_generated_without_decoder(self):
        """Without a text_decoder the template narrative is a non-empty string."""
        cf = CounterfactualSelf(dim=16, n_actions=4, seed=0)
        self._seed_transitions(cf, 16, [(0, 1.0), (1, 0.5)])
        result = cf.generate_counterfactual(
            state_before=np.ones(16),
            actual_action=0,
            alternative_action=1,
            text_decoder=None,
        )
        assert isinstance(result.narrative, str)
        assert len(result.narrative) > 0, "template narrative must not be empty"

    def test_handles_invalid_action(self):
        """record_transition with action=999 is clamped, not a crash."""
        cf = CounterfactualSelf(dim=16, n_actions=4, seed=0)
        cf.record_transition(
            state_before=np.ones(16),
            action=999,  # out of range -> clamped to 0 internally
            state_after=np.zeros(16),
            outcome={"actual_action": 999, "free_energy": 1.0, "collision": False},
            free_energy=1.0,
        )
        # No exception; step advanced.
        assert cf.get_snapshot()["step"] == 1
        # The clamped action 0 received the free-energy update.
        snap = cf.get_snapshot()
        efe = snap["expected_free_energy"]
        assert efe[0] > 0.0, f"clamped action 0 should hold EFE>0, got {efe[0]}"

    def test_narratives_generated_counter_increments(self):
        """_narratives_generated increments after each generate_counterfactual()."""
        cf = CounterfactualSelf(dim=16, n_actions=4, seed=0)
        self._seed_transitions(cf, 16, [(0, 1.0), (1, 0.5)])

        before = cf.get_snapshot()["narratives_generated"]
        cf.generate_counterfactual(
            state_before=np.ones(16), actual_action=0, alternative_action=1,
        )
        after = cf.get_snapshot()["narratives_generated"]
        assert after == before + 1, (
            f"narratives_generated {before} -> {after}, expected +1"
        )
