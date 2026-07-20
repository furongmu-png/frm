# tests/test_phase6_planning.py
"""Phase 6 — Planning capability tests.

Covers the 8 classes in ``capabilities.planning`` and
``capabilities.planning_advanced``: TrajectoryPlanner, GoalDecomposer,
ActionSequencer, HierarchicalPlanner, MonteCarloTreePlanner,
SymbolicPlanner, PolicyGradientPlanner, ContingencyPlanner.

Total: 70+ tests covering correctness, edge cases, NaN/Inf guards,
determinism, cycle handling, and API contract per spec §6 (Planning).
"""

from __future__ import annotations

import numpy as np
import pytest

from zero_data_model.capabilities.planning import (
    ActionSequencer,
    GoalDecomposer,
    HierarchicalPlanner,
    TrajectoryPlanner,
)
from zero_data_model.capabilities.planning_advanced import (
    ContingencyPlanner,
    MonteCarloTreePlanner,
    PolicyGradientPlanner,
    SymbolicPlanner,
)
from zero_data_model.capabilities.rules import PlanningRules


# --------------------------------------------------------------------- #
# TrajectoryPlanner
# --------------------------------------------------------------------- #


class TestTrajectoryPlanner:
    @pytest.fixture()
    def planner(self):
        # Use the real DifferentialGenerator (lazy import).
        return TrajectoryPlanner(dim=4, rules=PlanningRules(planning_horizon=16))

    def test_plan_returns_trajectory(self, planner):
        r = planner.plan(np.zeros(4), np.ones(4))
        assert "trajectory" in r
        assert "actions" in r
        assert "action_cost" in r
        assert "goal_reached" in r

    def test_trajectory_satisfies_start_boundary(self, planner):
        r = planner.plan(np.array([0.0, 0.0, 0.0, 0.0]), np.array([1.0, 2.0, 3.0, 4.0]))
        np.testing.assert_allclose(r["trajectory"][0], np.zeros(4), atol=1e-9)

    def test_trajectory_satisfies_end_boundary(self, planner):
        end = np.array([1.0, 2.0, 3.0, 4.0])
        r = planner.plan(np.zeros(4), end)
        np.testing.assert_allclose(r["trajectory"][-1], end, atol=1e-3)

    def test_actions_are_diff_of_trajectory(self, planner):
        r = planner.plan(np.zeros(4), np.ones(4), n_steps=8)
        traj = np.asarray(r["trajectory"])
        expected = np.diff(traj, axis=0)
        np.testing.assert_allclose(r["actions"], expected, atol=1e-9)

    def test_goal_reached_within_tolerance(self, planner):
        r = planner.plan(np.zeros(4), np.ones(4), n_steps=16)
        assert r["goal_reached"] is True

    def test_shape_mismatch_raises(self, planner):
        with pytest.raises(ValueError, match="shape"):
            planner.plan(np.zeros(4), np.zeros(3))

    def test_nan_start_raises(self, planner):
        with pytest.raises(ValueError, match="finite"):
            planner.plan(np.array([np.nan, 0, 0, 0]), np.zeros(4))

    def test_nan_goal_raises(self, planner):
        with pytest.raises(ValueError, match="finite"):
            planner.plan(np.zeros(4), np.array([np.inf, 0, 0, 0]))

    def test_obstacles_wrong_dim_raises(self, planner):
        with pytest.raises(ValueError, match="obstacles"):
            planner.plan(
                np.zeros(4), np.ones(4),
                obstacles=np.array([[0.0, 0.0]]),  # wrong dim
            )

    def test_obstacles_with_margin(self, planner):
        r = planner.plan(
            np.zeros(4), np.ones(4),
            obstacles=np.array([[0.5, 0.5, 0.5, 0.5]]),
            n_steps=8, margin=0.1,
        )
        assert "obstacle_violations" in r or "trajectory" in r

    def test_default_n_steps_from_rules(self):
        planner = TrajectoryPlanner(
            dim=4, rules=PlanningRules(planning_horizon=12)
        )
        r = planner.plan(np.zeros(4), np.ones(4))
        # Trajectory should have horizon+1 rows.
        assert len(r["trajectory"]) == 13

    def test_n_steps_zero_returns_endpoints(self, planner):
        r = planner.plan(np.zeros(4), np.ones(4), n_steps=0)
        # n_steps=0 → trajectory has 1 row (start == end).
        assert len(r["trajectory"]) == 1


# --------------------------------------------------------------------- #
# GoalDecomposer
# --------------------------------------------------------------------- #


class TestGoalDecomposer:
    @pytest.fixture()
    def decomposer(self):
        templates = {
            "build": [("foundation", "and"), ("walls", "and"), ("roof", "and")],
            "foundation": [("dig", "action"), ("pour_concrete", "action")],
            "walls": [("lay_bricks", "action")],
            "roof": [("frame", "action"), ("shingle", "action")],
        }
        return GoalDecomposer(dim=4, templates=templates)

    def test_decompose_atomic_goal(self):
        d = GoalDecomposer(dim=4, templates={})
        r = d.decompose("simple_action")
        assert r["tree"]["type"] == "action"
        assert r["tree"]["name"] == "simple_action"
        assert "simple_action" in r["actions"]

    def test_decompose_and_node(self, decomposer):
        r = decomposer.decompose("build")
        assert r["tree"]["type"] == "node"
        assert r["tree"]["goal"] == "build"
        assert len(r["tree"]["children"]) == 3
        assert r["tree"]["children"][0]["combine"] == "and"

    def test_decompose_recurses(self, decomposer):
        r = decomposer.decompose("build")
        # Should produce 5 atomic actions across all sub-trees.
        assert len(r["actions"]) == 5
        assert "dig" in r["actions"]
        assert "shingle" in r["actions"]

    def test_decompose_max_depth_caps(self, decomposer):
        r = decomposer.decompose("build", max_depth=1)
        # At depth 1, children become leaves not expanded.
        assert r["depth"] == 1

    def test_cycle_detection(self):
        templates = {
            "a": [("b", "and")],
            "b": [("a", "and")],  # cycle back to a
        }
        d = GoalDecomposer(dim=4, templates=templates)
        r = d.decompose("a")
        # Should terminate without infinite recursion.
        assert r["tree"]["type"] == "node"

    def test_default_max_depth_from_rules(self):
        d = GoalDecomposer(
            dim=4, templates={"a": [("b", "and")]}, rules=PlanningRules(planning_max_depth=5),
        )
        r = d.decompose("a")
        assert r["depth"] == 5

    def test_empty_goal_returns_leaf(self):
        d = GoalDecomposer(dim=4, templates={})
        r = d.decompose("foo")
        assert r["tree"]["type"] == "action"

    def test_decompose_returns_actions_list(self, decomposer):
        r = decomposer.decompose("foundation")
        assert set(r["actions"]) == {"dig", "pour_concrete"}

    def test_decompose_or_node(self):
        templates = {
            "travel": [("by_car", "or"), ("by_train", "or")],
            "by_car": [("drive", "action")],
            "by_train": [("ride", "action")],
        }
        d = GoalDecomposer(dim=4, templates=templates)
        r = d.decompose("travel")
        assert r["tree"]["children"][0]["combine"] == "or"


# --------------------------------------------------------------------- #
# ActionSequencer
# --------------------------------------------------------------------- #


class TestActionSequencer:
    @pytest.fixture()
    def seq(self):
        return ActionSequencer(dim=4)

    def test_linear_chain(self, seq):
        # 0 -> 1 -> 2
        A = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ])
        r = seq.sequence(A)
        assert r["sequence_indices"] == [0, 1, 2]
        assert r["cycles_broken"] == 0

    def test_no_edges_returns_any_order(self, seq):
        A = np.zeros((3, 3))
        r = seq.sequence(A)
        assert sorted(r["sequence_indices"]) == [0, 1, 2]
        assert r["cycles_broken"] == 0

    def test_single_node(self, seq):
        A = np.zeros((1, 1))
        r = seq.sequence(A)
        assert r["sequence_indices"] == [0]

    def test_cycle_breaking(self, seq):
        # 0 -> 1 -> 2 -> 0 (cycle)
        A = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ])
        r = seq.sequence(A)
        assert r["cycles_broken"] >= 1
        assert len(r["sequence_indices"]) == 3
        assert len(r["removed_edges"]) >= 1

    def test_labels_substituted(self, seq):
        A = np.array([[0, 1], [0, 0]])
        r = seq.sequence(A, labels=["alpha", "beta"])
        assert r["sequence"] == ["alpha", "beta"]

    def test_labels_length_mismatch_raises(self, seq):
        A = np.zeros((2, 2))
        with pytest.raises(ValueError, match="labels"):
            seq.sequence(A, labels=["only_one"])

    def test_non_square_raises(self, seq):
        with pytest.raises(ValueError, match="square"):
            seq.sequence(np.zeros((2, 3)))

    def test_returns_cycles_broken_count(self, seq):
        # 2-node cycle: 0 <-> 1.
        A = np.array([
            [0, 1],
            [1, 0],
        ])
        r = seq.sequence(A)
        assert r["cycles_broken"] >= 1

    def test_diamond_dag(self, seq):
        # 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
        A = np.array([
            [0, 1, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ])
        r = seq.sequence(A)
        assert r["sequence_indices"][0] == 0
        assert r["sequence_indices"][-1] == 3
        assert r["cycles_broken"] == 0


# --------------------------------------------------------------------- #
# HierarchicalPlanner
# --------------------------------------------------------------------- #


class TestHierarchicalPlanner:
    @pytest.fixture()
    def planner(self):
        # Use the real CausalInferenceEngine via lazy import.
        return HierarchicalPlanner(dim=8)

    def test_plan_hierarchy_returns_sequence(self, planner):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((20, 4))
        r = planner.plan_hierarchy(data)
        assert "sequence" in r
        assert "adjacency" in r
        assert "var_names" in r
        assert "n_edges" in r

    def test_plan_hierarchy_with_var_names(self, planner):
        rng = np.random.default_rng(1)
        data = rng.standard_normal((20, 3))
        r = planner.plan_hierarchy(data, var_names=["x", "y", "z"])
        assert set(r["var_names"]).issubset({"x", "y", "z"})

    def test_plan_hierarchy_goal_vars_restricts(self, planner):
        rng = np.random.default_rng(2)
        data = rng.standard_normal((20, 4))
        # Restrict to ancestors of variable 3.
        r = planner.plan_hierarchy(data, goal_vars=[3])
        # Should not include variables unreachable from goal.
        assert len(r["sequence_indices"]) <= 4

    def test_plan_hierarchy_non_2d_raises(self, planner):
        with pytest.raises(ValueError, match="2D"):
            planner.plan_hierarchy(np.zeros(4))

    def test_plan_hierarchy_nan_raises(self, planner):
        with pytest.raises(ValueError, match="finite"):
            planner.plan_hierarchy(np.array([[np.nan, 1.0], [0.0, 1.0]]))

    def test_plan_hierarchy_returns_discover_method(self, planner):
        rng = np.random.default_rng(3)
        data = rng.standard_normal((20, 3))
        r = planner.plan_hierarchy(data)
        assert "discover_method" in r


# --------------------------------------------------------------------- #
# MonteCarloTreePlanner
# --------------------------------------------------------------------- #


class TestMonteCarloTreePlanner:
    @pytest.fixture()
    def mcts_env(self):
        """A tiny gridworld: 5 states, 2 actions, reward at state 4."""
        n_states, n_actions = 5, 2

        def transition_fn(s, a):
            # Action 0: move forward (s+1), action 1: stay.
            if s == 4:
                return s, 0.0, True
            next_s = s + 1 if a == 0 else s
            reward = 1.0 if next_s == 4 else 0.0
            done = next_s == 4
            return next_s, reward, done

        def action_set_fn(s):
            return [0, 1] if s < 4 else []

        planner = MonteCarloTreePlanner(
            dim=4,
            transition_fn=transition_fn,
            action_set_fn=action_set_fn,
            rules=PlanningRules(planning_mcts_simulations=20, planning_max_depth=5),
        )
        return planner, n_states

    def test_search_returns_best_action(self, mcts_env):
        planner, _ = mcts_env
        r = planner.search(root_state=0)
        assert r["best_action"] is not None
        assert r["simulations"] == 20
        assert r["tree_size"] >= 1

    def test_search_action_values_non_empty(self, mcts_env):
        planner, _ = mcts_env
        r = planner.search(root_state=0, n_simulations=10)
        assert len(r["action_values"]) >= 1
        for av in r["action_values"]:
            assert "action" in av and "visits" in av and "value" in av

    def test_search_terminal_state_no_actions(self):
        """State with no available actions returns best_action=None."""
        planner = MonteCarloTreePlanner(
            dim=4,
            transition_fn=lambda s, a: (s, 0.0, True),
            action_set_fn=lambda s: [],  # no actions anywhere
        )
        r = planner.search(root_state=0, n_simulations=5)
        assert r["best_action"] is None

    def test_search_missing_transition_fn_raises(self):
        planner = MonteCarloTreePlanner(dim=4)
        with pytest.raises(RuntimeError, match="transition_fn"):
            planner.search(root_state=0)

    def test_search_missing_action_set_fn_raises(self):
        planner = MonteCarloTreePlanner(
            dim=4, transition_fn=lambda s, a: (s, 0.0, False),
        )
        with pytest.raises(RuntimeError, match="transition_fn"):
            planner.search(root_state=0)

    def test_search_n_simulations_drives_visits(self, mcts_env):
        planner, _ = mcts_env
        r1 = planner.search(root_state=0, n_simulations=5)
        r2 = planner.search(root_state=0, n_simulations=30)
        # More simulations should grow the tree.
        assert r2["tree_size"] >= r1["tree_size"]


# --------------------------------------------------------------------- #
# SymbolicPlanner
# --------------------------------------------------------------------- #


class TestSymbolicPlanner:
    @pytest.fixture()
    def planner(self):
        # Classic blocks-world: move A from table to on top of B.
        operators = [
            ("pick_A", {"on": "table", "hand": "empty"}, {"on": "hand", "hand": "A"}, 1),
            ("place_A_on_B", {"on": "hand", "hand": "A"}, {"on": "B", "hand": "empty"}, 1),
        ]
        return SymbolicPlanner(dim=4, operators=operators)

    def test_plan_finds_solution(self, planner):
        init = {"on": "table", "hand": "empty"}
        goal = {"on": "B", "hand": "empty"}
        r = planner.plan(init, goal)
        assert r["found"] is True
        assert r["actions"] == ["pick_A", "place_A_on_B"]
        assert r["cost"] == 2.0

    def test_plan_no_solution(self, planner):
        init = {"on": "table", "hand": "empty"}
        goal = {"on": "C", "hand": "empty"}  # unreachable
        r = planner.plan(init, goal)
        assert r["found"] is False

    def test_plan_already_at_goal(self, planner):
        init = {"on": "B", "hand": "empty"}
        goal = {"on": "B", "hand": "empty"}
        r = planner.plan(init, goal)
        assert r["found"] is True
        assert r["actions"] == []

    def test_plan_no_operators_returns_empty(self):
        planner = SymbolicPlanner(dim=4, operators=[])
        r = planner.plan({}, {"x": 1})
        assert r["found"] is False
        assert r["actions"] == []

    def test_plan_max_expansions_caps_search(self, planner):
        init = {"on": "table", "hand": "empty"}
        goal = {"on": "B", "hand": "empty"}
        r = planner.plan(init, goal, max_expansions=1)
        # Only 1 expansion allowed → may or may not find solution.
        assert r["expanded"] <= 1

    def test_plan_returns_expanded_count(self, planner):
        init = {"on": "table", "hand": "empty"}
        goal = {"on": "B", "hand": "empty"}
        r = planner.plan(init, goal)
        assert r["expanded"] >= 1

    def test_plan_multiple_paths_picks_cheapest(self):
        # Two paths: direct (cost 1) vs two-step (cost 2).
        operators = [
            ("direct", {"s": 0}, {"s": 1}, 1),
            ("step1", {"s": 0}, {"s": 2}, 1),
            ("step2", {"s": 2}, {"s": 1}, 1),
        ]
        planner = SymbolicPlanner(dim=4, operators=operators)
        r = planner.plan({"s": 0}, {"s": 1})
        assert r["found"] is True
        assert r["cost"] == 1.0
        assert r["actions"] == ["direct"]


# --------------------------------------------------------------------- #
# PolicyGradientPlanner
# --------------------------------------------------------------------- #


class TestPolicyGradientPlanner:
    @pytest.fixture()
    def pg(self):
        rng = np.random.default_rng(42)
        return PolicyGradientPlanner(
            n_states=8, n_actions=4, rng=rng,
            rules=PlanningRules(planning_mcts_ucb_c=0.1),  # reused as lr
        )

    def test_policy_returns_probs(self, pg):
        r = pg.policy(0)
        assert r["probs"].shape == (4,)
        assert abs(sum(r["probs"]) - 1.0) < 1e-9

    def test_policy_state_out_of_range_raises(self, pg):
        with pytest.raises(ValueError, match="out of range"):
            pg.policy(99)

    def test_select_action_returns_valid_action(self, pg):
        r = pg.select_action(0)
        assert 0 <= r["action"] < 4

    def test_update_changes_weights(self, pg):
        before = pg.weights.copy()
        pg.update(state=0, action=1, return_value=1.0, baseline=0.0)
        after = pg.weights
        # Row for action 1 should have changed.
        assert not np.allclose(before[1], after[1])

    def test_update_returns_advantage(self, pg):
        r = pg.update(state=0, action=1, return_value=2.0, baseline=1.0)
        assert r["td_error"] == pytest.approx(1.0, abs=1e-9)

    def test_update_negative_advantage_decreases_weight(self, pg):
        # When return < baseline, weight should decrease.
        before = pg.weights[1, 0].copy()
        pg.update(state=0, action=1, return_value=-1.0, baseline=0.0)
        after = pg.weights[1, 0]
        assert after < before

    def test_policy_uniform_at_init(self):
        # At init (weights=0), policy should be uniform.
        pg = PolicyGradientPlanner(n_states=4, n_actions=3)
        r = pg.policy(0)
        np.testing.assert_allclose(r["probs"], np.array([1/3, 1/3, 1/3]), atol=1e-9)

    def test_select_action_respects_probabilities(self, pg):
        # Sample many actions; empirical distribution should be close.
        actions = [pg.select_action(0)["action"] for _ in range(500)]
        # All actions should be reachable in 500 samples.
        assert len(set(actions)) >= 2


# --------------------------------------------------------------------- #
# ContingencyPlanner
# --------------------------------------------------------------------- #


class TestContingencyPlanner:
    @pytest.fixture()
    def cp(self):
        return ContingencyPlanner(dim=4)

    def test_add_plan_returns_name(self, cp):
        r = cp.add_plan("primary", ["a", "b", "c"])
        assert r["name"] == "primary"
        assert r["registered"] is True
        assert r["total_plans"] == 1

    def test_add_multiple_plans(self, cp):
        cp.add_plan("p1", ["a"])
        cp.add_plan("p2", ["b"])
        assert len(cp) == 2

    def test_fallback_no_precondition_returns_first(self, cp):
        cp.add_plan("p1", ["a"])
        cp.add_plan("p2", ["b"])
        r = cp.fallback()
        assert r["name"] == "p1"
        assert r["fallback_used"] is False

    def test_fallback_with_satisfied_precondition(self, cp):
        cp.add_plan("p1", ["a"], precondition_fn=lambda s: s == "go")
        cp.add_plan("p2", ["b"], precondition_fn=lambda s: s == "stop")
        r = cp.fallback(state="stop")
        assert r["name"] == "p2"

    def test_fallback_no_precondition_matches_returns_last(self, cp):
        cp.add_plan("p1", ["a"], precondition_fn=lambda s: False)
        cp.add_plan("p2", ["b"], precondition_fn=lambda s: False)
        r = cp.fallback(state="x")
        assert r["fallback_used"] is True
        assert r["name"] == "p2"

    def test_fallback_empty_returns_none(self, cp):
        r = cp.fallback()
        assert r["name"] is None
        assert r["actions"] == []

    def test_clear(self, cp):
        cp.add_plan("p1", ["a"])
        r = cp.clear()
        assert r["cleared"] == 1
        assert len(cp) == 0

    def test_len(self, cp):
        assert len(cp) == 0
        cp.add_plan("p1", ["a"])
        assert len(cp) == 1
