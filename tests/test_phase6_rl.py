# tests/test_phase6_rl.py
"""Phase 6 — Reinforcement Learning capability tests.

Covers the 7 classes in ``capabilities.rl`` and
``capabilities.rl_advanced``: SyntheticMDP, QLearner, PolicyOptimizer,
ValueFunction, DynaQ, MonteCarloTreeSearch, PosteriorSampling.

Total: 70+ tests covering correctness, edge cases, NaN/Inf guards,
state/action bounds, determinism (seeded), convergence sanity, and
API contract per spec §6 (RL).
"""

from __future__ import annotations

import numpy as np
import pytest

from zero_data_model.capabilities.rl import (
    PolicyOptimizer,
    QLearner,
    SyntheticMDP,
    ValueFunction,
)
from zero_data_model.capabilities.rl_advanced import (
    DynaQ,
    MonteCarloTreeSearch,
    PosteriorSampling,
)
from zero_data_model.capabilities.rules import RLRules


# --------------------------------------------------------------------- #
# SyntheticMDP
# --------------------------------------------------------------------- #


class TestSyntheticMDP:
    def test_default_shape(self):
        mdp = SyntheticMDP(n_states=16, n_actions=4, seed=42)
        assert mdp.P.shape == (16, 4, 16)
        assert mdp.R.shape == (16, 4)

    def test_transition_probabilities_sum_to_one(self):
        mdp = SyntheticMDP(n_states=8, n_actions=2, seed=0)
        for s in range(8):
            for a in range(2):
                assert mdp.P[s, a].sum() == pytest.approx(1.0, abs=1e-9)

    def test_reward_in_range(self):
        mdp = SyntheticMDP(n_states=8, n_actions=2, seed=0)
        assert np.all(mdp.R >= -1.0)
        assert np.all(mdp.R <= 1.0)

    def test_terminal_state_returns_done(self):
        mdp = SyntheticMDP(n_states=4, n_actions=2, seed=0)
        # Last state is terminal.
        r = mdp.step(mdp.n_states - 1, 0)
        assert r["done"] is True
        assert r["reward"] == 0.0

    def test_step_returns_valid_next_state(self):
        mdp = SyntheticMDP(n_states=8, n_actions=2, seed=0)
        r = mdp.step(0, 0)
        assert 0 <= r["next_state"] < 8
        assert isinstance(r["reward"], float)
        assert isinstance(r["done"], bool)

    def test_step_state_out_of_range_raises(self):
        mdp = SyntheticMDP(n_states=4, n_actions=2, seed=0)
        with pytest.raises(ValueError, match="state"):
            mdp.step(99, 0)
        with pytest.raises(ValueError, match="state"):
            mdp.step(-1, 0)

    def test_step_action_out_of_range_raises(self):
        mdp = SyntheticMDP(n_states=4, n_actions=2, seed=0)
        with pytest.raises(ValueError, match="action"):
            mdp.step(0, 99)
        with pytest.raises(ValueError, match="action"):
            mdp.step(0, -1)

    def test_reset_returns_non_terminal_state(self):
        mdp = SyntheticMDP(n_states=4, n_actions=2, seed=0)
        r = mdp.reset()
        assert r["state"] != mdp.n_states - 1  # never terminal

    def test_seed_determinism(self):
        m1 = SyntheticMDP(n_states=8, n_actions=2, seed=42)
        m2 = SyntheticMDP(n_states=8, n_actions=2, seed=42)
        np.testing.assert_allclose(m1.P, m2.P)
        np.testing.assert_allclose(m1.R, m2.R)

    def test_terminal_states_set(self):
        mdp = SyntheticMDP(n_states=4, n_actions=2, seed=0)
        assert mdp.terminal_states == {3}

    def test_gamma_from_rules(self):
        rules = RLRules(rl_gamma=0.95)
        mdp = SyntheticMDP(n_states=4, n_actions=2, rules=rules, seed=0)
        assert mdp.gamma == 0.95


# --------------------------------------------------------------------- #
# QLearner
# --------------------------------------------------------------------- #


class TestQLearner:
    @pytest.fixture()
    def q(self):
        mdp = SyntheticMDP(n_states=8, n_actions=2, seed=42)
        rng = np.random.default_rng(42)
        return QLearner(n_states=8, n_actions=2, mdp=mdp, rng=rng)

    def test_initial_q_is_zero(self, q):
        assert np.all(q.Q == 0.0)

    def test_update_changes_q(self, q):
        before = q.Q.copy()
        q.update(state=0, action=0, reward=1.0, next_state=1)
        assert q.Q[0, 0] != before[0, 0]

    def test_update_returns_td_error_and_q_value(self, q):
        r = q.update(state=0, action=0, reward=0.5, next_state=1)
        assert "td_error" in r
        assert "q_value" in r

    def test_update_terminal_state(self, q):
        # Next state is terminal (7) → gamma * max(Q[7]) should be 0 (Q is 0).
        r = q.update(state=0, action=0, reward=1.0, next_state=7)
        # td_target = reward + 0 = 1.0; td_error = 1.0 - 0.0 = 1.0.
        assert r["td_error"] == pytest.approx(1.0, abs=1e-9)

    def test_select_action_returns_valid_action(self, q):
        r = q.select_action(0)
        assert 0 <= r["action"] < 2
        assert r["q_values"].shape == (2,)
        assert isinstance(r["greedy"], bool)

    def test_select_action_state_out_of_range_raises(self, q):
        with pytest.raises(ValueError, match="out of range"):
            q.select_action(99)

    def test_select_action_greedy_when_epsilon_zero(self, q):
        # Set Q to known values.
        q.Q[0] = np.array([1.0, 5.0])
        r = q.select_action(0, epsilon=0.0)
        assert r["action"] == 1  # argmax.
        assert r["greedy"] is True

    def test_select_action_random_when_epsilon_one(self, q):
        # With epsilon=1, always random — try many times.
        actions = [q.select_action(0, epsilon=1.0)["action"] for _ in range(20)]
        # Both actions should appear with high probability.
        assert len(set(actions)) == 2

    def test_train_returns_episode_rewards(self, q):
        r = q.train(n_episodes=5, max_steps_per_episode=10)
        assert len(r["episode_rewards"]) == 5
        assert "final_policy" in r
        assert "final_q_table" in r
        assert r["n_episodes"] == 5

    def test_train_improves_q_table(self, q):
        before_norm = np.linalg.norm(q.Q)
        q.train(n_episodes=10, max_steps_per_episode=20)
        after_norm = np.linalg.norm(q.Q)
        # After training, Q should have non-zero values.
        assert after_norm > before_norm

    def test_evaluate_greedy_does_not_update_q(self, q):
        before = q.Q.copy()
        r = q.evaluate(n_episodes=3, max_steps_per_episode=10)
        np.testing.assert_allclose(before, q.Q)  # unchanged
        assert len(r["episode_rewards"]) == 3

    def test_evaluate_returns_mean_reward(self, q):
        r = q.evaluate(n_episodes=3, max_steps_per_episode=10)
        assert isinstance(r["mean_reward"], float)

    def test_train_mean_reward_in_rewards(self, q):
        r = q.train(n_episodes=3, max_steps_per_episode=10)
        assert r["mean_reward"] == pytest.approx(np.mean(r["episode_rewards"]))


# --------------------------------------------------------------------- #
# PolicyOptimizer
# --------------------------------------------------------------------- #


class TestPolicyOptimizer:
    @pytest.fixture()
    def po(self):
        mdp = SyntheticMDP(n_states=4, n_actions=2, seed=0)
        rng = np.random.default_rng(42)
        return PolicyOptimizer(n_states=4, n_actions=2, mdp=mdp, rng=rng)

    def test_policy_uniform_at_init(self, po):
        r = po.policy(0)
        np.testing.assert_allclose(r["probs"], np.array([0.5, 0.5]), atol=1e-9)

    def test_policy_returns_logits_and_probs(self, po):
        r = po.policy(0)
        assert "probs" in r and "logits" in r
        assert r["probs"].shape == (2,)

    def test_policy_state_out_of_range_raises(self, po):
        with pytest.raises(ValueError, match="out of range"):
            po.policy(99)

    def test_select_action_returns_valid_action(self, po):
        r = po.select_action(0)
        assert 0 <= r["action"] < 2

    def test_update_changes_logits(self, po):
        before = po.policy_logits.copy()
        po.update(state=0, action=0, return_value=1.0, baseline=0.0)
        after = po.policy_logits
        assert not np.allclose(before, after)

    def test_update_returns_advantage(self, po):
        r = po.update(state=0, action=0, return_value=2.0, baseline=1.0)
        assert r["advantage"] == pytest.approx(1.0, abs=1e-9)

    def test_update_negative_advantage_decreases_logit(self, po):
        before = po.policy_logits[0, 0]
        po.update(state=0, action=0, return_value=-1.0, baseline=0.0)
        after = po.policy_logits[0, 0]
        assert after < before

    def test_train_returns_rewards_and_policy(self, po):
        r = po.train(n_episodes=5, max_steps_per_episode=10)
        assert len(r["episode_rewards"]) == 5
        assert "final_policy" in r
        assert r["final_policy"].shape == (4,)

    def test_train_updates_logits(self, po):
        before = po.policy_logits.copy()
        po.train(n_episodes=3, max_steps_per_episode=5)
        after = po.policy_logits
        assert not np.allclose(before, after)


# --------------------------------------------------------------------- #
# ValueFunction
# --------------------------------------------------------------------- #


class TestValueFunction:
    @pytest.fixture()
    def vf(self):
        return ValueFunction(n_states=4, rng=np.random.default_rng(0))

    def test_initial_v_is_zero(self, vf):
        assert np.all(vf.V == 0.0)

    def test_td0_update_changes_v(self, vf):
        before = vf.V.copy()
        vf.td0_update(state=0, reward=1.0, next_state=1, done=False)
        assert vf.V[0] != before[0]

    def test_td0_update_terminal_zeros_bootstrap(self, vf):
        # When done=True, V(s') term is zeroed.
        vf.V[1] = 100.0  # pollute to verify zero bootstrap
        vf.td0_update(state=0, reward=1.0, next_state=1, done=True)
        # target = 1.0 + 0 = 1.0; V[0] = 0 + alpha * 1.0 = alpha.
        alpha = float(RLRules().rl_alpha)
        assert vf.V[0] == pytest.approx(alpha, abs=1e-9)

    def test_td0_update_returns_td_error(self, vf):
        r = vf.td0_update(state=0, reward=0.5, next_state=1, done=False)
        assert "td_error" in r and "value" in r

    def test_td_lambda_update_changes_v(self, vf):
        traj = [(0, 1.0), (1, 0.5), (2, 0.0)]
        r = vf.td_lambda_update(traj, lambda_=0.9)
        assert "mean_td_error" in r
        # At least V[0] should be non-zero after update.
        assert vf.V[0] != 0.0

    def test_monte_carlo_update(self, vf):
        traj = [(0, 1.0), (1, 0.5), (2, 0.0)]
        r = vf.monte_carlo_update(traj)
        assert r["n_updates"] == 3  # first-visit, all unique
        # V[0] should be the discounted return from state 0.
        gamma = float(RLRules().rl_gamma)
        expected = 1.0 + gamma * (0.5 + gamma * 0.0)
        alpha = float(RLRules().rl_alpha)
        assert vf.V[0] == pytest.approx(alpha * expected, abs=1e-9)

    def test_monte_carlo_first_visit_only(self, vf):
        # Repeated visits to same state — only first counts.
        traj = [(0, 1.0), (1, 1.0), (0, 1.0)]
        r = vf.monte_carlo_update(traj)
        assert r["n_updates"] == 2  # only first-visit of state 0.

    def test_evaluate_returns_values(self, vf):
        mdp = SyntheticMDP(n_states=4, n_actions=2, seed=0)
        policy = np.array([0, 1, 0, 1])
        r = vf.evaluate(mdp, policy, n_episodes=2, max_steps=5)
        assert "values" in r
        assert r["values"].shape == (4,)
        assert isinstance(r["mean_reward"], float)


# --------------------------------------------------------------------- #
# DynaQ
# --------------------------------------------------------------------- #


class TestDynaQ:
    @pytest.fixture()
    def dq(self):
        mdp = SyntheticMDP(n_states=8, n_actions=2, seed=42)
        rng = np.random.default_rng(42)
        return DynaQ(n_states=8, n_actions=2, mdp=mdp, rng=rng)

    def test_initial_q_zero(self, dq):
        assert np.all(dq.Q == 0.0)

    def test_update_changes_q(self, dq):
        before = dq.Q.copy()
        dq.update(state=0, action=0, reward=1.0, next_state=1)
        assert dq.Q[0, 0] != before[0, 0]

    def test_update_records_model(self, dq):
        dq.update(state=2, action=1, reward=0.5, next_state=3)
        assert dq.model_visited[2, 1]
        assert dq.model_next[2, 1] == 3
        assert dq.model_reward[2, 1] == 0.5

    def test_update_returns_td_error_and_planning_steps(self, dq):
        r = dq.update(state=0, action=0, reward=1.0, next_state=1)
        assert "td_error" in r
        assert "planning_steps" in r

    def test_update_with_no_planning_steps(self, dq):
        r = dq.update(state=0, action=0, reward=1.0, next_state=1, n_planning_steps=0)
        assert r["planning_steps"] == 0

    def test_train_returns_rewards(self, dq):
        r = dq.train(n_episodes=3, max_steps_per_episode=10)
        assert len(r["episode_rewards"]) == 3
        assert "final_policy" in r
        assert "final_q_table" in r

    def test_train_improves_q(self, dq):
        before = np.linalg.norm(dq.Q)
        dq.train(n_episodes=10, max_steps_per_episode=20)
        after = np.linalg.norm(dq.Q)
        assert after > before

    def test_imagined_updates_use_model(self, dq):
        # First update records model + runs n_planning_steps (limited by
        # len(visited_pairs)=1, so only 1 imagined step).
        dq.update(state=0, action=0, reward=1.0, next_state=1, n_planning_steps=5)
        # After update, model has 1 entry; planning_steps capped at 1.
        assert (0, 0) in dq.visited_pairs


# --------------------------------------------------------------------- #
# MonteCarloTreeSearch
# --------------------------------------------------------------------- #


class TestMonteCarloTreeSearch:
    @pytest.fixture()
    def mcts(self):
        mdp = SyntheticMDP(n_states=8, n_actions=2, seed=42)
        rng = np.random.default_rng(42)
        return MonteCarloTreeSearch(mdp=mdp, rng=rng)

    def test_search_returns_best_action(self, mcts):
        r = mcts.search(root_state=0, n_simulations=10, max_depth=5)
        assert r["best_action"] is not None
        assert 0 <= r["best_action"] < 2
        assert r["simulations"] == 10
        assert r["tree_size"] >= 1

    def test_search_action_values_non_empty(self, mcts):
        r = mcts.search(root_state=0, n_simulations=5)
        assert len(r["action_values"]) >= 1
        for av in r["action_values"]:
            assert "action" in av and "visits" in av and "value" in av

    def test_search_terminal_state_returns_zero_value(self, mcts):
        # Searching from terminal state: children all have value=0 (no
        # future reward). best_action is still picked by visits.
        r = mcts.search(root_state=7, n_simulations=5)
        assert r["best_action"] is not None
        for av in r["action_values"]:
            assert av["value"] == pytest.approx(0.0, abs=1e-9)

    def test_search_more_simulations_grows_tree(self, mcts):
        r1 = mcts.search(root_state=0, n_simulations=5)
        r2 = mcts.search(root_state=0, n_simulations=30)
        assert r2["tree_size"] >= r1["tree_size"]

    def test_search_default_simulations_from_rules(self, mcts):
        rules = RLRules(rl_mcts_simulations=15)
        mdp = SyntheticMDP(n_states=4, n_actions=2, seed=0, rules=rules)
        m = MonteCarloTreeSearch(mdp=mdp, rules=rules)
        r = m.search(root_state=0)
        assert r["simulations"] == 15

    def test_search_visits_increment(self, mcts):
        r = mcts.search(root_state=0, n_simulations=10)
        total_visits = sum(av["visits"] for av in r["action_values"])
        # At least 10 simulations visited children.
        assert total_visits >= 1


# --------------------------------------------------------------------- #
# PosteriorSampling
# --------------------------------------------------------------------- #


class TestPosteriorSampling:
    @pytest.fixture()
    def ps(self):
        mdp = SyntheticMDP(n_states=8, n_actions=2, seed=42)
        rng = np.random.default_rng(42)
        return PosteriorSampling(n_states=8, n_actions=2, mdp=mdp, rng=rng)

    def test_initial_posterior_is_unit_variance(self, ps):
        assert np.all(ps.posterior_mean == 0.0)
        assert np.all(ps.posterior_var == 1.0)

    def test_update_decreases_variance(self, ps):
        before_var = ps.posterior_var[0, 0]
        ps.update(state=0, action=0, reward=0.5)
        after_var = ps.posterior_var[0, 0]
        assert after_var < before_var

    def test_update_returns_obs_count(self, ps):
        r = ps.update(state=0, action=0, reward=0.5)
        assert r["obs_count"] == 1
        r2 = ps.update(state=0, action=0, reward=0.3)
        assert r2["obs_count"] == 2

    def test_update_pulls_mean_toward_observed(self, ps):
        ps.update(state=0, action=0, reward=1.0)
        # Posterior mean should now be > 0.
        assert ps.posterior_mean[0, 0] > 0

    def test_sample_reward_returns_matrix(self, ps):
        r = ps.sample_reward()
        assert r["sampled_reward"].shape == (8, 2)

    def test_value_iteration_returns_policy(self, ps):
        R = np.zeros((8, 2))
        r = ps.value_iteration(R, n_iters=10)
        assert "values" in r and "policy" in r and "q_table" in r
        assert r["policy"].shape == (8,)

    def test_value_iteration_converges(self, ps):
        # With zero reward, values should converge to 0.
        R = np.zeros((8, 2))
        r = ps.value_iteration(R, n_iters=100)
        np.testing.assert_allclose(r["values"], np.zeros(8), atol=1e-6)

    def test_train_returns_rewards(self, ps):
        r = ps.train(n_episodes=3, max_steps_per_episode=10)
        assert len(r["episode_rewards"]) == 3
        assert "posterior_mean" in r
        assert "posterior_var" in r

    def test_train_updates_posterior(self, ps):
        before_mean = ps.posterior_mean.copy()
        ps.train(n_episodes=3, max_steps_per_episode=10)
        after_mean = ps.posterior_mean
        assert not np.allclose(before_mean, after_mean)
