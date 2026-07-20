# src/zero_data_model/capabilities/planning.py
"""Phase 6 — Planning capabilities.

Composes :class:`causal_emergence.DifferentialGenerator` (damped
least-action trajectory) and :class:`causal_emergence.CausalInferenceEngine`
(causal DAG discovery + do-calculus) into domain planners.

Classes
-------
HierarchicalPlanner
    Decompose an abstract goal into a sequence of sub-goals using a
    causal-dependency graph from ``CausalInferenceEngine.discover()``.

GoalDecomposer
    AND/OR tree decomposition of a goal into atomic actions. Templates
    come from ``PlanningRules.goal_templates`` (default empty; callers
    may inject templates via the constructor).

TrajectoryPlanner
    Thin wrapper over ``DifferentialGenerator.generate()``. Accepts
    ``(start_state, goal_state, obstacles, n_steps)`` and returns the
    same dict as the engine plus a derived ``actions`` array (per-step
    velocity vectors).

ActionSequencer
    Topological sort of a DAG of actions into a linear sequence.
    Detects cycles and breaks them greedily by removing the lowest-
    weight edge.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .rules import PlanningRules


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _sanitize_state(state: np.ndarray) -> np.ndarray:
    arr = np.asarray(state, dtype=float).ravel()
    if not np.all(np.isfinite(arr)):
        raise ValueError("state must be finite (no NaN or Inf)")
    return arr


def _sanitize_obstacles(obstacles: np.ndarray | None, dim: int) -> np.ndarray | None:
    if obstacles is None:
        return None
    arr = np.asarray(obstacles, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != dim:
        raise ValueError(
            f"obstacles must be 2D with shape (K, {dim}), got {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("obstacles must be finite (no NaN or Inf)")
    return arr


# ----------------------------------------------------------------------
# TrajectoryPlanner
# ----------------------------------------------------------------------


class TrajectoryPlanner:
    """Wrap DifferentialGenerator into a planning facade.

    The output dict mirrors the engine's, plus an ``actions`` array of
    per-step velocity vectors ``q[k+1] - q[k]`` (shape ``(n_steps, dim)``).
    """

    def __init__(
        self,
        dim: int = 64,
        differential_generator=None,
        rules: PlanningRules | None = None,
    ):
        self.dim = int(dim)
        self.differential_generator = differential_generator
        self.rules = rules or PlanningRules()
        if self.differential_generator is None:
            # Lazy fallback: instantiate the real engine module.
            try:
                from ..causal_emergence.differential import DifferentialGenerator
                self.differential_generator = DifferentialGenerator(dim=self.dim)
            except ImportError:
                # Tests may stub the engine; we still construct.
                self.differential_generator = None

    def plan(
        self,
        start_state: np.ndarray,
        goal_state: np.ndarray,
        obstacles: np.ndarray | None = None,
        n_steps: int | None = None,
        margin: float | None = None,
    ) -> dict:
        """Generate a trajectory from start to goal.

        Returns the engine dict (``trajectory``, ``action``,
        ``converged``, ``iterations``, optional ``obstacle_violations``)
        plus ``actions`` (n_steps, dim) velocity vectors and
        ``action_cost`` (= total action).
        """
        start = _sanitize_state(start_state)
        goal = _sanitize_state(goal_state)
        if start.shape != goal.shape:
            raise ValueError(
                f"start_state shape {start.shape} != goal_state shape {goal.shape}"
            )
        if self.differential_generator is None:
            raise RuntimeError("differential_generator not configured")
        n = int(n_steps if n_steps is not None else self.rules.planning_horizon)
        constraints: dict | None = None
        if obstacles is not None:
            obs_arr = _sanitize_obstacles(obstacles, start.shape[0])
            constraints = {"obstacles": obs_arr}
            if margin is not None:
                constraints["margin"] = float(margin)
        result = self.differential_generator.generate(
            start, goal, n_steps=n, constraints=constraints,
        )
        traj = np.asarray(result.get("trajectory", []), dtype=float)
        if traj.size > 1 and traj.shape[0] >= 2:
            actions = np.diff(traj, axis=0)
        else:
            actions = np.zeros((0, start.shape[0]))
        result["actions"] = actions
        result["action_cost"] = float(result.get("action", 0.0))
        # goal_reached: True iff final point is within goal_tolerance.
        tol = float(self.rules.planning_goal_tolerance)
        if traj.size > 0:
            result["goal_reached"] = bool(
                np.linalg.norm(traj[-1] - goal) <= tol
            )
        else:
            result["goal_reached"] = False
        return result


# ----------------------------------------------------------------------
# GoalDecomposer
# ----------------------------------------------------------------------


class GoalDecomposer:
    """AND/OR tree decomposition of a goal into atomic actions.

    Templates are dict mappings ``{goal_name: [(sub_goal_or_action, type), ...]}``
    where ``type ∈ {'and', 'or', 'action'}``. Default templates are
    empty; callers inject via the constructor.
    """

    def __init__(
        self,
        dim: int = 64,
        rules: PlanningRules | None = None,
        templates: dict | None = None,
    ):
        self.dim = int(dim)
        self.rules = rules or PlanningRules()
        self.templates = templates or {}

    def decompose(self, goal: str, max_depth: int | None = None) -> dict:
        """Recursively decompose a goal into a tree.

        Returns ``{'tree': dict, 'actions': list[str], 'depth': int}``.
        Cycles are broken by stopping recursion when a goal is already
        on the active path (returns ``{'type': 'cycle', 'goal': goal}``).
        """
        depth = int(max_depth if max_depth is not None else self.rules.planning_max_depth)
        actions: list[str] = []

        def _expand(node: str, current_depth: int, path: set) -> dict:
            if current_depth >= depth:
                return {"type": "leaf", "goal": node}
            if node in path:
                return {"type": "cycle", "goal": node}
            if node not in self.templates:
                # Atomic action.
                actions.append(node)
                return {"type": "action", "name": node}
            children = self.templates[node]
            new_path = path | {node}
            expanded_children = []
            for child_name, child_type in children:
                child_tree = _expand(child_name, current_depth + 1, new_path)
                child_tree["name"] = child_name
                child_tree["combine"] = child_type
                expanded_children.append(child_tree)
            return {
                "type": "node",
                "goal": node,
                "children": expanded_children,
            }

        tree = _expand(goal, 0, set())
        return {
            "tree": tree,
            "actions": actions,
            "depth": int(depth),
        }


# ----------------------------------------------------------------------
# ActionSequencer
# ----------------------------------------------------------------------


class ActionSequencer:
    """Topological sort of a DAG of actions into a linear sequence.

    Input: ``adjacency`` (n x n binary ndarray) where ``adjacency[i, j] = 1``
    means action ``i`` must precede action ``j``. Cycles are broken
    greedily by removing the lowest-index edge in the cycle.

    Returns the linear sequence as a list of indices plus a list of
    edges removed for cycle-breaking.
    """

    def __init__(
        self,
        dim: int = 64,
        rules: PlanningRules | None = None,
    ):
        self.dim = int(dim)
        self.rules = rules or PlanningRules()

    def sequence(
        self,
        adjacency: np.ndarray,
        labels: list | None = None,
    ) -> dict:
        A = np.asarray(adjacency, dtype=int)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"adjacency must be square 2D, got {A.shape}")
        n = A.shape[0]
        if labels is None:
            labels = list(range(n))
        elif len(labels) != n:
            raise ValueError(
                f"labels length {len(labels)} != adjacency size {n}"
            )
        removed_edges: list[tuple[int, int]] = []
        # Greedy cycle breaking: while cycles exist, remove the lowest-
        # index edge in any cycle. Detect cycles by attempting a topo
        # sort with Kahn's algorithm; if it fails, find a back-edge.
        A_work = A.copy()
        max_breaks = n * n  # safety bound
        breaks = 0
        while breaks < max_breaks:
            order, cyclic = self._kahn_topo(A_work)
            if not cyclic:
                return {
                    "sequence": [labels[i] for i in order],
                    "sequence_indices": [int(i) for i in order],
                    "removed_edges": removed_edges,
                    "cycles_broken": int(breaks),
                }
            # Find a back edge: pick the first (i, j) where both i and j
            # remain in the residual graph and there's a path j -> i.
            found = False
            for i in range(n):
                for j in range(n):
                    if i == j or A_work[i, j] == 0:
                        continue
                    if self._has_path(A_work, j, i):
                        removed_edges.append((int(i), int(j)))
                        A_work[i, j] = 0
                        breaks += 1
                        found = True
                        break
                if found:
                    break
            if not found:
                # Should not happen if Kahn said cyclic, but break to
                # avoid infinite loop.
                break
        order, _ = self._kahn_topo(A_work)
        return {
            "sequence": [labels[i] for i in order],
            "sequence_indices": [int(i) for i in order],
            "removed_edges": removed_edges,
            "cycles_broken": int(breaks),
        }

    @staticmethod
    def _kahn_topo(A: np.ndarray) -> tuple[list[int], bool]:
        """Kahn's algorithm. Returns (order, cyclic). cyclic=True if a
        cycle remained (order is partial in that case)."""
        n = A.shape[0]
        in_degree = A.sum(axis=0).astype(int).tolist()
        queue = [i for i in range(n) if in_degree[i] == 0]
        order: list[int] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for j in range(n):
                if A[node, j] > 0:
                    in_degree[j] -= 1
                    if in_degree[j] == 0:
                        queue.append(j)
        return order, len(order) < n

    @staticmethod
    def _has_path(A: np.ndarray, src: int, dst: int) -> bool:
        """BFS path check from src to dst."""
        n = A.shape[0]
        if src == dst:
            return True
        visited = np.zeros(n, dtype=bool)
        queue = [src]
        visited[src] = True
        while queue:
            node = queue.pop(0)
            for j in range(n):
                if A[node, j] > 0 and not visited[j]:
                    if j == dst:
                        return True
                    visited[j] = True
                    queue.append(j)
        return False


# ----------------------------------------------------------------------
# HierarchicalPlanner
# ----------------------------------------------------------------------


class HierarchicalPlanner:
    """Decompose an abstract goal into a sub-goal sequence using a
    causal dependency graph.

    Uses ``CausalInferenceEngine.discover()`` to derive the action
    dependency DAG from observation data, then topologically sorts it
    with :class:`ActionSequencer`.
    """

    def __init__(
        self,
        dim: int = 64,
        causal_inference_engine=None,
        rules: PlanningRules | None = None,
    ):
        self.dim = int(dim)
        self.causal_inference_engine = causal_inference_engine
        self.rules = rules or PlanningRules()
        self.sequencer = ActionSequencer(dim=dim, rules=rules)

    def plan_hierarchy(
        self,
        observation: np.ndarray,
        var_names: list[str] | None = None,
        goal_vars: list[int] | None = None,
    ) -> dict:
        """Discover the causal DAG, then return a topological sequence.

        ``goal_vars`` (optional) restricts the plan to actions leading
        to the listed target variables.
        """
        if self.causal_inference_engine is None:
            try:
                from ..causal_emergence.causal_discovery import CausalInferenceEngine
                self.causal_inference_engine = CausalInferenceEngine(dim=self.dim)
            except ImportError as exc:
                raise RuntimeError(
                    "causal_inference_engine not configured and "
                    "causal_emergence.causal_discovery not importable"
                ) from exc
        data = np.asarray(observation, dtype=float)
        if data.ndim != 2:
            raise ValueError(f"observation must be 2D, got shape {data.shape}")
        if not np.all(np.isfinite(data)):
            raise ValueError("observation must be finite")
        discover = self.causal_inference_engine.discover(
            data, var_names=var_names,
        )
        adjacency = np.asarray(discover.get("adjacency", np.zeros((0, 0))), dtype=int)
        # Restrict to goal_vars if provided.
        if goal_vars and adjacency.size > 0:
            # Compute the set of ancestors of goal_vars via BFS.
            n = adjacency.shape[0]
            ancestors = set(int(g) for g in goal_vars if 0 <= int(g) < n)
            frontier = list(ancestors)
            while frontier:
                node = frontier.pop()
                for i in range(n):
                    if adjacency[i, node] > 0 and i not in ancestors:
                        ancestors.add(i)
                        frontier.append(i)
            keep = sorted(ancestors)
            if keep:
                sub_adj = adjacency[np.ix_(keep, keep)]
                sub_labels = [discover.get("var_names", list(range(n)))[i] for i in keep]
            else:
                sub_adj = np.zeros((0, 0), dtype=int)
                sub_labels = []
        else:
            sub_adj = adjacency
            sub_labels = discover.get("var_names", list(range(adjacency.shape[0])))
        seq_result = self.sequencer.sequence(sub_adj, labels=sub_labels)
        return {
            "sequence": seq_result["sequence"],
            "sequence_indices": seq_result["sequence_indices"],
            "removed_edges": seq_result["removed_edges"],
            "cycles_broken": seq_result["cycles_broken"],
            "adjacency": sub_adj,
            "var_names": sub_labels,
            "discover_method": discover.get("method", "unknown"),
            "n_edges": int(discover.get("n_edges", 0)),
        }
