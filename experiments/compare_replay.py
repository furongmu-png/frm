# experiments/compare_replay.py
"""Compare free-energy trajectories of replay vs no-replay runs.

Reads the two ``loop_log.csv`` files produced by:

    python run_replay.py --steps 5000 --output_dir output/replay_with
    python run_replay.py --steps 5000 --no_replay --output_dir output/replay_without

Outputs:
  - experiments/output/replay/comparison.json
  - experiments/output/replay/fe_comparison.png
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

EXPERIMENTS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXPERIMENTS_DIR / "output" / "replay"


def load_fe_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load (steps, free_energies) from a run_replay loop_log.csv."""
    steps: list[int] = []
    fes: list[float] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            fes.append(float(row["free_energy"]))
    return np.array(steps), np.array(fes)


def smooth(x: np.ndarray, window: int = 50) -> np.ndarray:
    """Simple moving-average smoothing."""
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def main() -> None:
    with_csv = EXPERIMENTS_DIR / "output" / "replay_with" / "loop_log.csv"
    without_csv = EXPERIMENTS_DIR / "output" / "replay_without" / "loop_log.csv"
    if not with_csv.exists():
        print(f"[error] missing {with_csv}; run run_replay.py first")
        sys.exit(1)
    if not without_csv.exists():
        print(f"[error] missing {without_csv}; run run_replay.py --no_replay first")
        sys.exit(1)
    s_with, fe_with = load_fe_csv(with_csv)
    s_without, fe_without = load_fe_csv(without_csv)
    # Compute summary statistics.
    summary = {
        "with_replay": {
            "n_steps": int(len(s_with)),
            "mean_fe": float(fe_with.mean()),
            "std_fe": float(fe_with.std()),
            "final_fe": float(fe_with[-1]),
            "first_fe": float(fe_with[0]),
            "delta_fe": float(fe_with[-1] - fe_with[0]),
            "min_fe": float(fe_with.min()),
            "max_fe": float(fe_with.max()),
        },
        "without_replay": {
            "n_steps": int(len(s_without)),
            "mean_fe": float(fe_without.mean()),
            "std_fe": float(fe_without.std()),
            "final_fe": float(fe_without[-1]),
            "first_fe": float(fe_without[0]),
            "delta_fe": float(fe_without[-1] - fe_without[0]),
            "min_fe": float(fe_without.min()),
            "max_fe": float(fe_without.max()),
        },
    }
    # Smoothed stability: variance of the smoothed trajectory (lower =
    # more stable post-equilibrium).
    smooth_window = 100
    fe_with_smooth = smooth(fe_with, smooth_window)
    fe_without_smooth = smooth(fe_without, smooth_window)
    summary["with_replay"]["smoothed_std"] = float(fe_with_smooth.std())
    summary["without_replay"]["smoothed_std"] = float(fe_without_smooth.std())
    # Comparison verdict.
    delta_mean = summary["with_replay"]["mean_fe"] - \
                 summary["without_replay"]["mean_fe"]
    delta_smoothed_std = (summary["with_replay"]["smoothed_std"] -
                          summary["without_replay"]["smoothed_std"])
    summary["comparison"] = {
        "delta_mean_fe": float(delta_mean),
        "delta_smoothed_std": float(delta_smoothed_std),
        "verdict": (
            "replay MORE stable" if delta_smoothed_std < 0
            else "replay LESS stable"
        ),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "comparison.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    # Plot.
    if _HAS_MPL:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        # Left: raw trajectories (downsampled for plotting speed).
        ds = 10
        axes[0].plot(s_without[::ds], fe_without[::ds],
                     color="tab:gray", linewidth=0.6, alpha=0.7,
                     label="no replay")
        axes[0].plot(s_with[::ds], fe_with[::ds],
                     color="tab:blue", linewidth=0.6, alpha=0.7,
                     label="with replay")
        axes[0].set_title("Free energy (raw, downsampled 10x)")
        axes[0].set_xlabel("step")
        axes[0].set_ylabel("free energy")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        # Right: smoothed trajectories.
        s_with_s = s_with[smooth_window - 1:]
        s_without_s = s_without[smooth_window - 1:]
        axes[1].plot(s_without_s, fe_without_smooth,
                     color="tab:gray", linewidth=1.5,
                     label="no replay (smoothed)")
        axes[1].plot(s_with_s, fe_with_smooth,
                     color="tab:blue", linewidth=1.5,
                     label="with replay (smoothed)")
        axes[1].set_title(f"Free energy (moving avg, window={smooth_window})")
        axes[1].set_xlabel("step")
        axes[1].set_ylabel("free energy (smoothed)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        out_png = OUTPUT_DIR / "fe_comparison.png"
        fig.savefig(out_png, dpi=110)
        plt.close(fig)
        print(f"\nSaved plot: {out_png}")


if __name__ == "__main__":
    main()
