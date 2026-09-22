# experiments/analyze_action_intent.py
"""Action-effect consistency: does the agent's action distribution
reflect an intent to control its body?

For each step in the cognitive snapshots we have:
- ``action`` (0=left, 1=right, 2=up, 3=no-op)
- ``sandbox_state`` (agent's cx, cy before the action)

We compute:
1. Conditional probability P(action | agent_x_bin): if the agent has
   learned that "left" actions move it leftward, the distribution
   should reflect intent — e.g. agent on the right edge should pick
   "left" more often than "right".
2. Mutual information between action and the resulting position
   change: high MI indicates the agent's actions are causally
   linked to its motion (vs random thrashing).
3. "Goal-directedness" score: fraction of steps where the action
   moves the agent AWAY from the nearest wall (a simple proxy for
   self-preservative intent).

Outputs:
  - experiments/output/eval/action_intent.json
  - experiments/output/eval/action_intent.png

Run with:
    python experiments/analyze_action_intent.py --snapshots output/replay_with/snapshots
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "eval"

WORLD_SIZE = 128
ACTION_NAMES = ["left", "right", "up", "no-op"]


# ------------------------------------------------------------------ #
# Loading
# ------------------------------------------------------------------ #
def load_snapshots(snapshots_dir: Path) -> dict:
    """Load steps, actions, sandbox states."""
    files = sorted(snapshots_dir.glob("snapshot_*.npz"))
    if not files:
        print(f"[error] no snapshots in {snapshots_dir}")
        sys.exit(1)
    steps, actions, sandbox_states = [], [], []
    for fp in files:
        with np.load(fp, allow_pickle=True) as data:
            steps.append(int(data["step"]))
            actions.append(int(data["action"]))
            state_str = str(data["sandbox_state"].item())
            try:
                state = json.loads(state_str)
            except Exception:
                state = {}
            sandbox_states.append(state)
    order = np.argsort(np.array(steps))
    return {
        "steps": np.array(steps)[order],
        "actions": np.array(actions)[order],
        "sandbox_states": [sandbox_states[i] for i in order],
    }


# ------------------------------------------------------------------ #
# Extract agent positions
# ------------------------------------------------------------------ #
def extract_agent_positions(sandbox_states: list[dict]) -> np.ndarray:
    """Return (n_steps, 2) array of (cx, cy) for the agent (body 0)."""
    rows = []
    for state in sandbox_states:
        bodies = state.get("bodies", [])
        if bodies:
            agent = bodies[0]
            rows.append([float(agent.get("cx", 0.0)),
                         float(agent.get("cy", 0.0))])
        else:
            rows.append([0.0, 0.0])
    return np.array(rows)


# ------------------------------------------------------------------ #
# Conditional probability P(action | x_bin)
# ------------------------------------------------------------------ #
def conditional_action_distribution(
    positions: np.ndarray, actions: np.ndarray, n_bins: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Compute P(action | x_bin) for the agent's x-position.

    Returns (bin_centres, prob_matrix) where prob_matrix[i, a] =
    P(action=a | x_bin=i).
    """
    cx = positions[:, 0]
    edges = np.linspace(0, WORLD_SIZE, n_bins + 1)
    bin_idx = np.digitize(cx, edges[1:-1])
    n_actions = 4
    prob = np.zeros((n_bins, n_actions))
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.any():
            for a in range(n_actions):
                prob[b, a] = float((actions[mask] == a).sum()) / mask.sum()
    bin_centres = 0.5 * (edges[:-1] + edges[1:])
    return bin_centres, prob


# ------------------------------------------------------------------ #
# Mutual information: action <-> resulting motion direction
# ------------------------------------------------------------------ #
def action_motion_mi(
    positions: np.ndarray, actions: np.ndarray, n_bins: int = 4
) -> float:
    """MI between the action and the sign of the resulting motion."""
    # Compute the position change between consecutive snapshots.
    # NOTE: snapshots are sampled at ``cognitive_snapshot_interval``
    # steps apart, so the motion is over MULTIPLE physics steps.
    # We use the SIGN of the motion (left/right, up/down) since the
    # magnitude is dominated by the sampling gap.
    if len(positions) < 2:
        return 0.0
    dx = np.diff(positions[:, 0])
    dy = np.diff(positions[:, 1])
    # Quantise motion direction into 4 bins (L/R/U/D).
    motion = np.where(
        np.abs(dx) > np.abs(dy),
        np.where(dx > 0, 1, 0),  # right=1, left=0
        np.where(dy > 0, 3, 2),  # down=3, up=2 (image coords)
    )
    # Align actions with the motion they produced.
    actions_aligned = actions[:-1]
    # Joint histogram.
    n_a, n_m = 4, 4
    joint = np.zeros((n_a, n_m))
    for a, m in zip(actions_aligned, motion):
        joint[a, m] += 1
    joint /= joint.sum() + 1e-12
    pa = joint.sum(axis=1, keepdims=True)
    pm = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        mi = np.sum(joint * np.log2(joint / (pa @ pm + 1e-12) + 1e-12))
    return float(mi)


# ------------------------------------------------------------------ #
# Goal-directedness: action moves AWAY from the nearest wall
# ------------------------------------------------------------------ #
def goal_directedness(
    positions: np.ndarray, actions: np.ndarray
) -> dict:
    """Fraction of actions that move the agent away from the nearest wall."""
    cx = positions[:, 0]
    cy = positions[:, 1]
    # Distance to each wall.
    d_left = cx
    d_right = WORLD_SIZE - cx
    d_top = cy
    d_bottom = WORLD_SIZE - cy
    # Nearest wall + which direction moves away from it.
    # If nearest wall is LEFT, the "away" action is RIGHT.
    n_steps = len(actions)
    away_count = 0
    valid_count = 0
    for i in range(n_steps):
        walls = np.array([d_left[i], d_right[i], d_top[i], d_bottom[i]])
        nearest = int(np.argmin(walls))
        # Action that moves away from each wall:
        # left wall (0) -> action 1 (right)
        # right wall (1) -> action 0 (left)
        # top wall (2) -> action 3 (down) — but our sandbox only has
        #   3 actions: 0=L, 1=R, 2=UP. There is no "down" action.
        #   So we count the agent's action as "away" only if there's
        #   a matching action.
        # Map: wall_idx -> "away" action.
        away_action = {
            0: 1,  # left wall -> right
            1: 0,  # right wall -> left
            2: None,  # top wall -> would need "down", unavailable
            3: 2,  # bottom wall -> up
        }[nearest]
        if away_action is None:
            continue
        valid_count += 1
        if actions[i] == away_action:
            away_count += 1
    return {
        "away_count": int(away_count),
        "valid_count": int(valid_count),
        "fraction_away": float(away_count / valid_count) if valid_count else 0.0,
        # Baseline: if the agent chose uniformly at random, fraction
        # would be 1/3 (3 valid actions, one of which is "away").
        "baseline_uniform": 1.0 / 3.0,
    }


# ------------------------------------------------------------------ #
# Main analysis
# ------------------------------------------------------------------ #
def run_analysis(
    snapshots_dir: Path,
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[action_intent] loading snapshots from {snapshots_dir}")
    data = load_snapshots(snapshots_dir)
    actions = data["actions"]
    positions = extract_agent_positions(data["sandbox_states"])
    n_steps = len(actions)
    print(f"[action_intent] {n_steps} snapshots")
    # Action distribution.
    action_dist = {ACTION_NAMES[a]: int((actions == a).sum())
                   for a in range(4)}
    print(f"[action_intent] action distribution: {action_dist}")
    # Conditional P(action | x_bin).
    bin_centres, prob = conditional_action_distribution(
        positions, actions, n_bins=4,
    )
    # MI between action and motion.
    mi = action_motion_mi(positions, actions)
    print(f"[action_intent] action-motion MI: {mi:.4f} bits")
    # Goal-directedness.
    gd = goal_directedness(positions, actions)
    print(f"[action_intent] goal-directedness: {gd['fraction_away']:.3f} "
          f"(baseline {gd['baseline_uniform']:.3f})")
    # Build result.
    result = {
        "n_snapshots": int(n_steps),
        "action_distribution": action_dist,
        "conditional_prob": {
            "bin_centres": bin_centres.tolist(),
            "prob_matrix": prob.tolist(),
            "bin_names": [f"x_bin_{i}" for i in range(len(bin_centres))],
            "action_names": ACTION_NAMES,
        },
        "action_motion_mi_bits": float(mi),
        "goal_directedness": gd,
    }
    # Save.
    with (output_dir / "action_intent.json").open("w") as f:
        json.dump(result, f, indent=2)
    # Plot.
    if _HAS_MPL:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        # Panel 1: action distribution.
        axes[0].bar(ACTION_NAMES, [action_dist[a] for a in ACTION_NAMES],
                    color="tab:blue", alpha=0.7)
        axes[0].set_title("Action distribution")
        axes[0].set_ylabel("count")
        # Panel 2: conditional P(action | x_bin).
        im = axes[1].imshow(prob.T, aspect="auto", cmap="Blues",
                             vmin=0, vmax=1)
        axes[1].set_title("P(action | x_bin)")
        axes[1].set_xlabel("x-position bin")
        axes[1].set_ylabel("action")
        axes[1].set_yticks(range(4))
        axes[1].set_yticklabels(ACTION_NAMES)
        axes[1].set_xticks(range(len(bin_centres)))
        axes[1].set_xticklabels([f"{c:.0f}" for c in bin_centres])
        plt.colorbar(im, ax=axes[1])
        # Panel 3: agent trajectory.
        sc = axes[2].scatter(positions[:, 0], positions[:, 1],
                              c=actions, cmap="tab10", s=8, alpha=0.6)
        axes[2].set_title("Agent trajectory, coloured by action")
        axes[2].set_xlabel("cx")
        axes[2].set_ylabel("cy")
        axes[2].set_xlim(0, WORLD_SIZE)
        axes[2].set_ylim(0, WORLD_SIZE)
        axes[2].set_aspect("equal")
        plt.colorbar(sc, ax=axes[2], ticks=range(4),
                     label="action")
        fig.tight_layout()
        png_path = output_dir / "action_intent.png"
        fig.savefig(png_path, dpi=110)
        plt.close(fig)
        print(f"[plot] saved {png_path}")
    print(f"[result] action-motion MI: {mi:.4f} bits  "
          f"goal-directedness: {gd['fraction_away']:.3f}")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshots", type=str, required=True)
    p.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    args = p.parse_args()
    run_analysis(
        snapshots_dir=Path(args.snapshots),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
