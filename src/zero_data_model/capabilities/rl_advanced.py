# src/zero_data_model/capabilities/rl_advanced.py
"""Phase 6 — Advanced RL capabilities.

- DynaQ: Q-learning + model learning + imagined rollouts.
- MonteCarloTreeSearch: MCTS over a (possibly unknown) MDP.
- PosteriorSampling: PSRL with a HamiltonianSampler-driven posterior
  over reward.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .rl import QLearner, SyntheticMDP
from .rules import RLRules


# ----------------------------------------------------------------------
# DynaQ
# ----------------------------------------------------------------------


class DynaQ:
    """Dyna-Q: Q-learning + model + imaginary rollouts.

    Maintains a tabular model ``M[s, a] = (next_state, reward)`` updated
    from real experience; performs ``n_planning_steps`` random-sample
    updates per real step using the model.
    """

    def __init__(
        self,
        n_states: int = 16,
        n_actions: int = 4,
        mdp: SyntheticMDP | None = None,
        rules: RLRules | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.mdp = mdp or SyntheticMDP(self.n_states, self.n_actions, rules=rules)
        self.rules = rules or self.mdp.rules
        self.rng = rng or np.random.default_rng()
        self.Q = np.zeros((self.n_states, self.n_actions))
        # Model: (next_state, reward) tables.
        self.model_next = np.zeros((self.n_states, self.n_actions), dtype=int)
        self.model_reward = np.zeros((self.n_states, self.n_actions))
        self.model_visited = np.zeros((self.n_states, self.n_actions), dtype=bool)
        self.visited_pairs: list[tuple[int, int]] = []

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        n_planning_steps: int | None = None,
    ) -> dict:
        """Real Q-update + model update + n imaginary updates."""
        alpha = float(self.rules.rl_alpha)
        gamma = float(self.rules.rl_gamma)
        # Real update.
        td_target = float(reward) + gamma * float(np.max(self.Q[int(next_state)]))
        td_error = td_target - float(self.Q[int(state), int(action)])
        self.Q[int(state), int(action)] += alpha * td_error
        # Model update.
        if not self.model_visited[int(state), int(action)]:
            self.visited_pairs.append((int(state), int(action)))
        self.model_next[int(state), int(action)] = int(next_state)
        self.model_reward[int(state), int(action)] = float(reward)
        self.model_visited[int(state), int(action)] = True
        # Imagined updates.
        n_plan = int(
            n_planning_steps if n_planning_steps is not None
            else self.rules.rl_dyna_planning_steps
        )
        n_plan = min(n_plan, len(self.visited_pairs))
        for _ in range(n_plan):
            idx = int(self.rng.integers(len(self.visited_pairs)))
            s, a = self.visited_pairs[idx]
            ns = int(self.model_next[s, a])
            r = float(self.model_reward[s, a])
            target = r + gamma * float(np.max(self.Q[ns]))
            err = target - float(self.Q[s, a])
            self.Q[s, a] += alpha * err
        return {"td_error": float(td_error), "planning_steps": int(n_plan)}

    def train(
        self,
        n_episodes: int = 100,
        max_steps_per_episode: int = 100,
    ) -> dict:
        episode_rewards: list[float] = []
        for _ in range(int(n_episodes)):
            state = self.mdp.reset()["state"]
            total_reward = 0.0
            for _ in range(int(max_steps_per_episode)):
                # Epsilon-greedy.
                if self.rng.random() < float(self.rules.rl_epsilon):
                    action = int(self.rng.integers(self.n_actions))
                else:
                    action = int(np.argmax(self.Q[state]))
                step_result = self.mdp.step(state, action)
                next_state = step_result["next_state"]
                reward = step_result["reward"]
                done = step_result["done"]
                self.update(state, action, reward, next_state)
                state = next_state
                total_reward += reward
                if done:
                    break
            episode_rewards.append(float(total_reward))
        return {
            "episode_rewards": episode_rewards,
            "final_policy": np.argmax(self.Q, axis=1),
            "final_q_table": self.Q.copy(),
            "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        }


# ----------------------------------------------------------------------
# MonteCarloTreeSearch
# ----------------------------------------------------------------------


class _MCTSNodeRL:
    """Internal MCTS node for RL."""

    __slots__ = (
        "state", "parent", "action", "children",
        "visits", "value_sum", "untried_actions",
    )

    def __init__(
        self,
        state: int,
        parent: "_MCTSNodeRL | None" = None,
        action: int | None = None,
        untried_actions: list[int] | None = None,
    ):
        self.state = int(state)
        self.parent = parent
        self.action = action
        self.children: list[_MCTSNodeRL] = []
        self.visits = 0
        self.value_sum = 0.0
        self.untried_actions = list(untried_actions) if untried_actions else []


class MonteCarloTreeSearch:
    """MCTS over an MDP. Unlike planning_advanced.MonteCarloTreePlanner,
    this version assumes the MDP is known (or learned via repeated
    interaction). UCT selection.
    """

    def __init__(
        self,
        mdp: SyntheticMDP,
        rules: RLRules | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.mdp = mdp
        self.rules = rules or RLRules()
        self.rng = rng or np.random.default_rng()

    def search(
        self,
        root_state: int,
        n_simulations: int | None = None,
        max_depth: int | None = None,
    ) -> dict:
        sims = int(
            n_simulations if n_simulations is not None
            else self.rules.rl_mcts_simulations
        )
        depth = int(max_depth if max_depth is not None else 10)
        root = _MCTSNodeRL(
            int(root_state),
            parent=None,
            action=None,
            untried_actions=list(range(self.mdp.n_actions)),
        )
        for _ in range(sims):
            node = self._select(root)
            if node.untried_actions:
                action = node.untried_actions.pop(0)
                step_result = self.mdp.step(node.state, action)
                next_state = step_result["next_state"]
                child = _MCTSNodeRL(
                    next_state,
                    parent=node,
                    action=action,
                    untried_actions=list(range(self.mdp.n_actions)) if not step_result["done"] else [],
                )
                node.children.append(child)
                node = child
            # Rollout.
            reward = self._rollout(node.state, depth)
            # Backprop.
            cur = node
            while cur is not None:
                cur.visits += 1
                cur.value_sum += reward
                cur = cur.parent

        if not root.children:
            return {
                "best_action": None,
                "action_values": [],
                "simulations": int(sims),
                "tree_size": 1,
            }
        best = max(root.children, key=lambda c: c.visits)
        action_values = [
            {
                "action": int(c.action),
                "visits": int(c.visits),
                "value": float(c.value_sum / c.visits) if c.visits > 0 else 0.0,
            }
            for c in root.children
        ]
        return {
            "best_action": int(best.action),
            "action_values": action_values,
            "simulations": int(sims),
            "tree_size": int(self._count(root)),
        }

    def _select(self, root: _MCTSNodeRL) -> _MCTSNodeRL:
        node = root
        while node.children and not node.untried_actions:
            c = float(self.rules.rl_gamma)  # reuse as UCB c
            c = max(c, 1.414)
            best, best_score = None, -float("inf")
            for child in node.children:
                if child.visits == 0:
                    score = float("inf")
                else:
                    exploit = child.value_sum / child.visits
                    explore = c * math.sqrt(math.log(node.visits + 1) / child.visits)
                    score = exploit + explore
                if score > best_score:
                    best_score = score
                    best = child
            if best is None:
                break
            node = best
        return node

    def _rollout(self, state: int, depth: int) -> float:
        gamma = float(self.rules.rl_gamma)
        total = 0.0
        cur = int(state)
        for d in range(depth):
            if cur in self.mdp.terminal_states:
                break
            action = int(self.rng.integers(self.mdp.n_actions))
            step_result = self.mdp.step(cur, action)
            total += (gamma ** d) * step_result["reward"]
            cur = step_result["next_state"]
            if step_result["done"]:
                break
        return total

    @staticmethod
    def _count(node: _MCTSNodeRL) -> int:
        n = 1
        for c in node.children:
            n += MonteCarloTreeSearch._count(c)
        return n


# ----------------------------------------------------------------------
# PosteriorSampling
# ----------------------------------------------------------------------


class PosteriorSampling:
    """PSRL: maintain a Gaussian posterior over reward R(s, a).

    Prior: ``R ~ N(0, 1)`` for each (s, a).
    Update on each observed reward: conjugate Gaussian update.

    Before each episode, sample a reward function from the posterior
    and compute the optimal policy via value iteration.
    """

    def __init__(
        self,
        n_states: int = 16,
        n_actions: int = 4,
        mdp: SyntheticMDP | None = None,
        rules: RLRules | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.mdp = mdp or SyntheticMDP(self.n_states, self.n_actions, rules=rules)
        self.rules = rules or self.mdp.rules
        self.rng = rng or np.random.default_rng()
        # Posterior over R(s, a): mean + variance.
        self.posterior_mean = np.zeros((self.n_states, self.n_actions))
        self.posterior_var = np.ones((self.n_states, self.n_actions))
        self._obs_counts = np.zeros((self.n_states, self.n_actions), dtype=int)

    def update(
        self,
        state: int,
        action: int,
        reward: float,
    ) -> dict:
        """Bayesian update of the reward posterior (Gaussian-Gaussian)."""
        s = int(state)
        a = int(action)
        # Prior precision = 1 (var=1); likelihood precision = n_obs (assume unit var).
        # Conjugate update:
        # new_precision = prior_precision + n_obs
        # new_mean = (prior_mean * prior_precision + sum_obs * 1) / new_precision
        # new_var = 1 / new_precision
        self._obs_counts[s, a] += 1
        n = int(self._obs_counts[s, a])
        prior_prec = 1.0 / max(self.posterior_var[s, a], 1e-6)
        new_prec = prior_prec + n
        new_mean = (self.posterior_mean[s, a] * prior_prec + float(reward) * n) / new_prec
        self.posterior_mean[s, a] = float(new_mean)
        self.posterior_var[s, a] = float(1.0 / new_prec)
        return {
            "posterior_mean": float(self.posterior_mean[s, a]),
            "posterior_var": float(self.posterior_var[s, a]),
            "obs_count": int(n),
        }

    def sample_reward(self) -> dict:
        """Sample a reward matrix from the posterior."""
        std = np.sqrt(np.maximum(self.posterior_var, 0.0))
        sampled = self.posterior_mean + std * self.rng.standard_normal(
            (self.n_states, self.n_actions)
        )
        return {"sampled_reward": sampled}

    def value_iteration(
        self,
        reward: np.ndarray,
        n_iters: int = 50,
    ) -> dict:
        """Compute optimal policy via value iteration on the true MDP
        transition tensor with the sampled reward."""
        gamma = float(self.rules.rl_gamma)
        R = np.asarray(reward, dtype=float)
        V = np.zeros(self.n_states)
        # Initialize Q so it is always defined even if n_iters == 0.
        Q = R + gamma * (self.mdp.P @ V)
        for _ in range(int(n_iters)):
            # Q(s, a) = R(s, a) + gamma * sum_s' P(s'|s,a) V(s')
            Q = R + gamma * (self.mdp.P @ V)  # shape (n_states, n_actions)
            new_V = np.max(Q, axis=1)
            if np.linalg.norm(new_V - V) < 1e-6:
                V = new_V
                break
            V = new_V
        policy = np.argmax(Q, axis=1)
        return {"values": V, "policy": policy, "q_table": Q}

    def train(
        self,
        n_episodes: int = 50,
        max_steps_per_episode: int = 100,
    ) -> dict:
        rewards_list: list[float] = []
        for _ in range(int(n_episodes)):
            sampled = self.sample_reward()["sampled_reward"]
            vi = self.value_iteration(sampled)
            policy = vi["policy"]
            state = self.mdp.reset()["state"]
            total_reward = 0.0
            for _ in range(int(max_steps_per_episode)):
                action = int(policy[state])
                step_result = self.mdp.step(state, action)
                next_state = step_result["next_state"]
                reward = step_result["reward"]
                self.update(state, action, reward)
                state = next_state
                total_reward += reward
                if step_result["done"]:
                    break
            rewards_list.append(float(total_reward))
        return {
            "episode_rewards": rewards_list,
            "mean_reward": float(np.mean(rewards_list)) if rewards_list else 0.0,
            "posterior_mean": self.posterior_mean.copy(),
            "posterior_var": self.posterior_var.copy(),
        }
