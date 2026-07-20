# src/zero_data_model/capabilities/rl.py
"""Phase 6 — Reinforcement Learning capabilities.

Tabular RL with synthetic MDPs. No external RL library required;
everything is numpy + standard library.

Classes
-------
SyntheticMDP
    Deterministic-seed random MDP: ``P(s' | s, a)``, ``R(s, a)``.

QLearner
    Tabular Q-learning with epsilon-greedy exploration.

PolicyOptimizer
    Softmax policy + REINFORCE-style gradient updates.

ValueFunction
    Tabular V-function with TD(0) / TD(λ) / Monte Carlo updates.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .rules import RLRules


# ----------------------------------------------------------------------
# SyntheticMDP
# ----------------------------------------------------------------------


class SyntheticMDP:
    """Random tabular MDP.

    Transition tensor ``P`` shape ``(n_states, n_actions, n_states)``
    normalized so that ``sum_s' P[s, a, s'] = 1`` for all ``(s, a)``.
    Reward matrix ``R`` shape ``(n_states, n_actions)``.
    """

    def __init__(
        self,
        n_states: int = 16,
        n_actions: int = 4,
        rules: RLRules | None = None,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.rules = rules or RLRules()
        self.gamma = float(self.rules.rl_gamma)
        # Per-instance RNG: caller may pass one in (facade uses _child_rngs[i]);
        # otherwise derive from `seed` for backward compatibility. Stored on
        # the instance so step()/reset() are deterministic and reproducible.
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        # Random transition tensor.
        P = self.rng.random((self.n_states, self.n_actions, self.n_states))
        self.P = P / P.sum(axis=2, keepdims=True)
        # Reward in [-1, 1].
        self.R = self.rng.uniform(-1.0, 1.0, size=(self.n_states, self.n_actions))
        # Terminal states: last state is terminal.
        self.terminal_states = {self.n_states - 1}

    def step(self, state: int, action: int) -> dict:
        """Take one step. Returns ``{'next_state', 'reward', 'done', 'info'}``."""
        s = int(state)
        a = int(action)
        if s < 0 or s >= self.n_states:
            raise ValueError(f"state {s} out of range [0, {self.n_states})")
        if a < 0 or a >= self.n_actions:
            raise ValueError(f"action {a} out of range [0, {self.n_actions})")
        if s in self.terminal_states:
            return {"next_state": s, "reward": 0.0, "done": True, "info": {"terminal": True}}
        next_state = int(self.rng.choice(self.n_states, p=self.P[s, a]))
        reward = float(self.R[s, a])
        done = next_state in self.terminal_states
        return {
            "next_state": next_state,
            "reward": reward,
            "done": bool(done),
            "info": {"terminal": bool(done)},
        }

    def reset(self) -> dict:
        """Reset to a random non-terminal start state."""
        non_terminal = [
            s for s in range(self.n_states) if s not in self.terminal_states
        ]
        if non_terminal:
            start = int(non_terminal[int(self.rng.integers(len(non_terminal)))])
        else:
            start = 0
        return {"state": start}


# ----------------------------------------------------------------------
# QLearner
# ----------------------------------------------------------------------


class QLearner:
    """Tabular Q-learning."""

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

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
    ) -> dict:
        """One TD update."""
        alpha = float(self.rules.rl_alpha)
        gamma = float(self.rules.rl_gamma)
        td_target = float(reward) + gamma * float(np.max(self.Q[int(next_state)]))
        td_error = td_target - float(self.Q[int(state), int(action)])
        self.Q[int(state), int(action)] += alpha * td_error
        return {"td_error": float(td_error), "q_value": float(self.Q[int(state), int(action)])}

    def select_action(self, state: int, epsilon: float | None = None) -> dict:
        """Epsilon-greedy."""
        eps = float(epsilon if epsilon is not None else self.rules.rl_epsilon)
        s = int(state)
        if s < 0 or s >= self.n_states:
            raise ValueError(f"state {s} out of range")
        if self.rng.random() < eps:
            action = int(self.rng.integers(self.n_actions))
            greedy = False
        else:
            action = int(np.argmax(self.Q[s]))
            greedy = True
        return {
            "action": action,
            "q_values": self.Q[s].copy(),
            "greedy": bool(greedy),
        }

    def train(
        self,
        n_episodes: int = 100,
        max_steps_per_episode: int = 100,
    ) -> dict:
        """Full training loop. Returns per-episode rewards + final policy."""
        episode_rewards: list[float] = []
        for _ in range(int(n_episodes)):
            state = self.mdp.reset()["state"]
            total_reward = 0.0
            for _ in range(int(max_steps_per_episode)):
                action_result = self.select_action(state)
                action = action_result["action"]
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
        final_policy = np.argmax(self.Q, axis=1)
        return {
            "episode_rewards": episode_rewards,
            "final_policy": final_policy,
            "final_q_table": self.Q.copy(),
            "n_episodes": int(n_episodes),
            "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        }

    def evaluate(
        self,
        n_episodes: int = 10,
        max_steps_per_episode: int = 100,
    ) -> dict:
        """Evaluate greedy policy without updating Q."""
        rewards: list[float] = []
        for _ in range(int(n_episodes)):
            state = self.mdp.reset()["state"]
            total = 0.0
            for _ in range(int(max_steps_per_episode)):
                action_result = self.select_action(state, epsilon=0.0)
                step_result = self.mdp.step(state, action_result["action"])
                state = step_result["next_state"]
                total += step_result["reward"]
                if step_result["done"]:
                    break
            rewards.append(float(total))
        return {
            "episode_rewards": rewards,
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "n_episodes": int(n_episodes),
        }


# ----------------------------------------------------------------------
# PolicyOptimizer
# ----------------------------------------------------------------------


class PolicyOptimizer:
    """Softmax policy + REINFORCE gradient updates."""

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
        # Policy logits: shape (n_states, n_actions).
        self.policy_logits = np.zeros((self.n_states, self.n_actions))

    def policy(self, state: int) -> dict:
        s = int(state)
        if s < 0 or s >= self.n_states:
            raise ValueError(f"state {s} out of range")
        logits = self.policy_logits[s].copy()
        logits = logits - logits.max()
        exp = np.exp(logits)
        probs = exp / max(exp.sum(), 1e-12)
        return {"probs": probs, "logits": logits}

    def select_action(self, state: int) -> dict:
        probs = self.policy(state)["probs"]
        action = int(self.rng.choice(self.n_actions, p=probs))
        return {"action": action, "probs": probs}

    def update(
        self,
        state: int,
        action: int,
        return_value: float,
        baseline: float = 0.0,
    ) -> dict:
        """REINFORCE update."""
        alpha = float(self.rules.rl_alpha)
        advantage = float(return_value) - float(baseline)
        probs = self.policy(state)["probs"]
        # Gradient of log-prob: (one_hot(action) - probs).
        one_hot = np.zeros(self.n_actions)
        one_hot[int(action)] = 1.0
        grad_logp = one_hot - probs
        self.policy_logits[int(state)] += alpha * advantage * grad_logp
        return {"advantage": float(advantage), "logits_norm": float(np.linalg.norm(self.policy_logits))}

    def train(
        self,
        n_episodes: int = 100,
        max_steps_per_episode: int = 100,
    ) -> dict:
        rewards_list: list[float] = []
        for _ in range(int(n_episodes)):
            state = self.mdp.reset()["state"]
            # Collect trajectory.
            trajectory: list[tuple[int, int, float]] = []
            total_reward = 0.0
            for _ in range(int(max_steps_per_episode)):
                action_result = self.select_action(state)
                action = action_result["action"]
                step_result = self.mdp.step(state, action)
                next_state = step_result["next_state"]
                reward = step_result["reward"]
                trajectory.append((state, action, reward))
                state = next_state
                total_reward += reward
                if step_result["done"]:
                    break
            # Compute returns + update.
            G = 0.0
            for s, a, r in reversed(trajectory):
                G = r + float(self.rules.rl_gamma) * G
                self.update(s, a, G, baseline=total_reward)
            rewards_list.append(float(total_reward))
        final_policy = np.argmax(self.policy_logits, axis=1)
        return {
            "episode_rewards": rewards_list,
            "final_policy": final_policy,
            "n_episodes": int(n_episodes),
            "mean_reward": float(np.mean(rewards_list)) if rewards_list else 0.0,
        }


# ----------------------------------------------------------------------
# ValueFunction
# ----------------------------------------------------------------------


class ValueFunction:
    """Tabular V[s] with TD(0) / TD(λ) / Monte Carlo updates."""

    def __init__(
        self,
        n_states: int = 16,
        rules: RLRules | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.n_states = int(n_states)
        self.rules = rules or RLRules()
        self.rng = rng or np.random.default_rng()
        self.V = np.zeros(self.n_states)

    def td0_update(
        self,
        state: int,
        reward: float,
        next_state: int,
        done: bool = False,
    ) -> dict:
        """TD(0) update: V(s) += alpha * (r + gamma * V(s') * (1 - done) - V(s))."""
        alpha = float(self.rules.rl_alpha)
        gamma = float(self.rules.rl_gamma)
        target = float(reward) + gamma * float(self.V[int(next_state)]) * (0.0 if done else 1.0)
        td_error = target - float(self.V[int(state)])
        self.V[int(state)] += alpha * td_error
        return {"td_error": float(td_error), "value": float(self.V[int(state)])}

    def td_lambda_update(
        self,
        trajectory: list[tuple[int, float]],
        lambda_: float = 0.9,
    ) -> dict:
        """TD(λ) update over a complete trajectory.

        ``trajectory`` is a list of ``(state, reward)`` tuples; the
        final state is assumed terminal.
        """
        alpha = float(self.rules.rl_alpha)
        gamma = float(self.rules.rl_gamma)
        lam = float(lambda_)
        eligibility = np.zeros(self.n_states)
        total_td = 0.0
        for i, (s, r) in enumerate(trajectory):
            s_next = trajectory[i + 1][0] if i + 1 < len(trajectory) else s
            done = i + 1 >= len(trajectory)
            target = float(r) + gamma * float(self.V[s_next]) * (0.0 if done else 1.0)
            td_error = target - float(self.V[s])
            eligibility *= gamma * lam
            eligibility[s] += 1.0
            self.V += alpha * td_error * eligibility
            total_td += abs(td_error)
        return {"mean_td_error": float(total_td / max(len(trajectory), 1))}

    def monte_carlo_update(
        self,
        trajectory: list[tuple[int, float]],
    ) -> dict:
        """First-visit MC update: V(s) = mean of returns from s."""
        gamma = float(self.rules.rl_gamma)
        alpha = float(self.rules.rl_alpha)
        # Compute returns.
        returns = np.zeros(len(trajectory))
        G = 0.0
        for i in range(len(trajectory) - 1, -1, -1):
            G = trajectory[i][1] + gamma * G
            returns[i] = G
        visited = set()
        updates = 0
        for i, (s, _) in enumerate(trajectory):
            if s in visited:
                continue
            visited.add(s)
            td_error = returns[i] - float(self.V[s])
            self.V[s] += alpha * td_error
            updates += 1
        return {"n_updates": int(updates)}

    def evaluate(
        self,
        mdp: SyntheticMDP,
        policy: np.ndarray,
        n_episodes: int = 10,
        max_steps: int = 100,
    ) -> dict:
        """Estimate V under a given policy via TD(0)."""
        rewards: list[float] = []
        for _ in range(int(n_episodes)):
            state = mdp.reset()["state"]
            total = 0.0
            for _ in range(int(max_steps)):
                action = int(policy[state])
                step_result = mdp.step(state, action)
                next_state = step_result["next_state"]
                reward = step_result["reward"]
                done = step_result["done"]
                self.td0_update(state, reward, next_state, done=done)
                state = next_state
                total += reward
                if done:
                    break
            rewards.append(float(total))
        return {
            "values": self.V.copy(),
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        }
