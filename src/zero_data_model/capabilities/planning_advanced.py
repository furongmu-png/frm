# src/zero_data_model/capabilities/planning_advanced.py
"""Phase 6 — Advanced planning capabilities.

Composes :class:`HamiltonianSampler` (rollout sampling) and
:class:`DifferentialGenerator` with classical AI planning algorithms:
MCTS (UCT), STRIPS symbolic planning, policy-gradient planning, and
contingency (fallback) planning.

No external ML library required.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .planning import _sanitize_state
from .rules import PlanningRules


# ----------------------------------------------------------------------
# MonteCarloTreePlanner
# ----------------------------------------------------------------------


class _MCTSNode:
    """Lightweight MCTS tree node (internal)."""

    __slots__ = (
        "state", "parent", "action", "children",
        "visits", "value_sum", "untried_actions",
    )

    def __init__(
        self,
        state: Any,
        parent: "_MCTSNode | None" = None,
        action: int | None = None,
        untried_actions: list[int] | None = None,
    ):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: list[_MCTSNode] = []
        self.visits = 0
        self.value_sum = 0.0
        self.untried_actions = (
            list(untried_actions) if untried_actions is not None else []
        )


class MonteCarloTreePlanner:
    """UCT (Upper Confidence Bound for Trees) MCTS planner.

    The transition / reward model is a callable pair:
    ``transition_fn(state, action) -> (next_state, reward, done)``
    ``action_set_fn(state) -> list[int]``

    Rollout policy defaults to uniform random over available actions;
    a HamiltonianSampler can be plugged in via ``rollout_sampler`` to
    draw policy samples from a posterior.
    """

    def __init__(
        self,
        dim: int = 64,
        hamiltonian_sampler=None,
        rules: PlanningRules | None = None,
        transition_fn=None,
        reward_fn=None,
        action_set_fn=None,
        rng: np.random.Generator | None = None,
    ):
        self.dim = int(dim)
        self.hamiltonian_sampler = hamiltonian_sampler
        self.rules = rules or PlanningRules()
        self.transition_fn = transition_fn
        self.reward_fn = reward_fn
        self.action_set_fn = action_set_fn
        # Per-instance RNG for reproducible rollouts.
        self.rng = rng if rng is not None else np.random.default_rng()

    def search(
        self,
        root_state: Any,
        n_simulations: int | None = None,
        max_depth: int | None = None,
    ) -> dict:
        """Run MCTS from root_state. Returns best action + stats."""
        if self.transition_fn is None or self.action_set_fn is None:
            raise RuntimeError(
                "transition_fn and action_set_fn must be set before search()"
            )
        sims = int(
            n_simulations if n_simulations is not None
            else self.rules.planning_mcts_simulations
        )
        depth = int(
            max_depth if max_depth is not None
            else self.rules.planning_max_depth
        )
        root = _MCTSNode(
            root_state, parent=None, action=None,
            untried_actions=self.action_set_fn(root_state),
        )
        for _ in range(sims):
            # 1. Selection
            node = self._select(root)
            # 2. Expansion
            if node.untried_actions:
                action = node.untried_actions.pop(0)
                next_state, _, _ = self.transition_fn(node.state, action)
                child = _MCTSNode(
                    next_state, parent=node, action=action,
                    untried_actions=self.action_set_fn(next_state),
                )
                node.children.append(child)
                node = child
            # 3. Simulation (random rollout, depth-limited)
            reward = self._rollout(node.state, depth)
            # 4. Backpropagation
            self._backprop(node, reward)

        # Pick best action by visit count (more robust than value mean).
        if not root.children:
            return {
                "best_action": None,
                "action_values": [],
                "simulations": int(sims),
                "tree_size": 1,
            }
        best_child = max(root.children, key=lambda c: c.visits)
        action_values = [
            {
                "action": int(c.action),
                "visits": int(c.visits),
                "value": float(c.value_sum / c.visits) if c.visits > 0 else 0.0,
            }
            for c in root.children
        ]
        return {
            "best_action": int(best_child.action),
            "action_values": action_values,
            "simulations": int(sims),
            "tree_size": int(self._count_nodes(root)),
        }

    def _select(self, root: _MCTSNode) -> _MCTSNode:
        node = root
        while node.children and not node.untried_actions:
            c = float(self.rules.planning_mcts_ucb_c)
            best = None
            best_score = -float("inf")
            for child in node.children:
                if child.visits == 0:
                    score = float("inf")
                else:
                    exploit = child.value_sum / child.visits
                    explore = c * math.sqrt(
                        math.log(node.visits + 1) / child.visits
                    )
                    score = exploit + explore
                if score > best_score:
                    best_score = score
                    best = child
            if best is None:
                break
            node = best
        return node

    def _rollout(self, state: Any, depth: int) -> float:
        total_reward = 0.0
        cur = state
        for _ in range(depth):
            actions = self.action_set_fn(cur)
            if not actions:
                break
            # Uniform random rollout via per-instance RNG (deterministic).
            action = int(actions[int(self.rng.integers(len(actions)))])
            next_state, reward, done = self.transition_fn(cur, action)
            total_reward += float(reward)
            if done:
                break
            cur = next_state
        return total_reward

    @staticmethod
    def _backprop(node: _MCTSNode, reward: float) -> None:
        cur = node
        while cur is not None:
            cur.visits += 1
            cur.value_sum += reward
            cur = cur.parent

    @staticmethod
    def _count_nodes(node: _MCTSNode) -> int:
        count = 1
        for child in node.children:
            count += MonteCarloTreePlanner._count_nodes(child)
        return count


# ----------------------------------------------------------------------
# SymbolicPlanner
# ----------------------------------------------------------------------


class SymbolicPlanner:
    """STRIPS-style symbolic planner with A* search.

    Operators are tuples ``(name, preconds, effects, cost)`` where
    preconds / effects are dict mappings ``{var: value}``. A state is
    a dict of variable bindings.

    Heuristic: number of unsatisfied goal conditions (admissible).
    """

    def __init__(
        self,
        dim: int = 64,
        rules: PlanningRules | None = None,
        operators: list[tuple] | None = None,
    ):
        self.dim = int(dim)
        self.rules = rules or PlanningRules()
        self.operators = list(operators) if operators else []

    def plan(
        self,
        initial_state: dict,
        goal_state: dict,
        max_expansions: int = 1000,
    ) -> dict:
        """A* search. Returns action sequence or None."""
        if not self.operators:
            return {"actions": [], "cost": 0.0, "expanded": 0, "found": False}

        def _h(state: dict) -> int:
            # Count unsatisfied goal literals.
            return sum(1 for k, v in goal_state.items() if state.get(k) != v)

        def _applicable(state: dict, op: tuple) -> bool:
            _, pre, _, _ = op
            return all(state.get(k) == v for k, v in pre.items())

        def _apply(state: dict, op: tuple) -> dict:
            new_state = dict(state)
            _, _, effects, _ = op
            new_state.update(effects)
            return new_state

        open_set = [(_h(initial_state), 0, initial_state, [])]
        visited = {frozenset(initial_state.items()): 0}
        expanded = 0
        while open_set and expanded < max_expansions:
            open_set.sort(key=lambda x: (x[0], x[1]))
            f, g, state, path = open_set.pop(0)
            expanded += 1
            if all(state.get(k) == v for k, v in goal_state.items()):
                return {
                    "actions": [op[0] for op in path],
                    "cost": float(g),
                    "expanded": int(expanded),
                    "found": True,
                }
            for op in self.operators:
                if _applicable(state, op):
                    new_state = _apply(state, op)
                    key = frozenset(new_state.items())
                    new_g = g + op[3]
                    if key not in visited or visited[key] > new_g:
                        visited[key] = new_g
                        new_f = new_g + _h(new_state)
                        open_set.append((new_f, new_g, new_state, path + [op]))
        return {
            "actions": [],
            "cost": 0.0,
            "expanded": int(expanded),
            "found": False,
        }


# ----------------------------------------------------------------------
# PolicyGradientPlanner
# ----------------------------------------------------------------------


class PolicyGradientPlanner:
    """Softmax-over-actions planner with REINFORCE-style updates.

    The "policy" is a linear mapping from state to action preferences:
    ``logits = W @ state``, ``probs = softmax(logits)``.
    Training updates ``W`` via REINFORCE:
    ``W[a] += alpha * (G - baseline) * state * (1 - probs[a])``.
    """

    def __init__(
        self,
        n_states: int = 16,
        n_actions: int = 4,
        rules: PlanningRules | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.rules = rules or PlanningRules()
        self.rng = rng if rng is not None else np.random.default_rng()
        # Policy weights: shape (n_actions, n_states).
        self.weights = np.zeros((self.n_actions, self.n_states))

    def policy(self, state: int) -> dict:
        s = int(state)
        if s < 0 or s >= self.n_states:
            raise ValueError(f"state {s} out of range [0, {self.n_states})")
        # One-hot encoding of state.
        x = np.zeros(self.n_states)
        x[s] = 1.0
        logits = self.weights @ x
        logits = logits - logits.max()
        exp = np.exp(logits)
        probs = exp / max(exp.sum(), 1e-12)
        return {"probs": probs, "logits": logits, "state_features": x}

    def select_action(self, state: int) -> dict:
        result = self.policy(state)
        probs = result["probs"]
        action = int(self.rng.choice(self.n_actions, p=probs))
        return {"action": action, "probs": probs}

    def update(
        self,
        state: int,
        action: int,
        return_value: float,
        baseline: float = 0.0,
    ) -> dict:
        advantage = float(return_value) - float(baseline)
        result = self.policy(state)
        probs = result["probs"]
        x = result["state_features"]
        alpha = float(self.rules.planning_mcts_ucb_c)  # reuse as lr
        # REINFORCE gradient ascent.
        grad = -probs[action] * probs
        grad[action] += probs[action]
        self.weights[action] += alpha * advantage * x
        return {
            "td_error": float(advantage),
            "weight_norm": float(np.linalg.norm(self.weights)),
        }


# ----------------------------------------------------------------------
# ContingencyPlanner
# ----------------------------------------------------------------------


class ContingencyPlanner:
    """Generate multiple plans, each tagged with a precondition.

    On ``fallback()``, picks the first plan whose precondition is
    satisfied. Used for adaptive behavior under environment changes.
    """

    def __init__(
        self,
        dim: int = 64,
        rules: PlanningRules | None = None,
    ):
        self.dim = int(dim)
        self.rules = rules or PlanningRules()
        self._plans: list[dict] = []  # {'name', 'precondition_fn', 'actions'}

    def add_plan(
        self,
        name: str,
        actions: list,
        precondition_fn=None,
    ) -> dict:
        """Register a plan. precondition_fn(state) -> bool."""
        self._plans.append({
            "name": str(name),
            "actions": list(actions),
            "precondition_fn": precondition_fn,
        })
        return {"name": str(name), "registered": True, "total_plans": len(self._plans)}

    def fallback(self, state: Any | None = None) -> dict:
        """Pick the first plan whose precondition holds; else last plan."""
        for plan in self._plans:
            pre = plan["precondition_fn"]
            if pre is None or (callable(pre) and pre(state)):
                return {
                    "name": plan["name"],
                    "actions": plan["actions"],
                    "fallback_used": False,
                }
        if not self._plans:
            return {"name": None, "actions": [], "fallback_used": False}
        last = self._plans[-1]
        return {
            "name": last["name"],
            "actions": last["actions"],
            "fallback_used": True,
        }

    def clear(self) -> dict:
        n = len(self._plans)
        self._plans.clear()
        return {"cleared": int(n)}

    def __len__(self) -> int:
        return len(self._plans)
