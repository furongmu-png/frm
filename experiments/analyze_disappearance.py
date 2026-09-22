# experiments/analyze_disappearance.py
"""Object-permanence evaluation: does FE spike when an object vanishes?

Reads a pre-trained model checkpoint (or trains one inline) and runs
the sandbox with a mid-run object removal. If the model has learned a
persistent representation of the object, its free energy should rise
sharply at the moment of removal and decay as it re-consolidates.

Two modes:
1. ``--snapshots DIR``: load cognitive snapshots from a previous run
   and analyse the FE time-series around user-specified removal points.
2. ``--inline`` (default): train a fresh model for ``--pre_steps``
   steps, then remove a non-agent object and continue for
   ``--post_steps`` steps, recording FE before/after.

Outputs:
  - experiments/output/eval/disappearance.json   (statistics)
  - experiments/output/eval/disappearance.png     (FE curve)

Run with:
    python experiments/analyze_disappearance.py --inline --pre_steps 500 --post_steps 500
    python experiments/analyze_disappearance.py --snapshots output/replay_with/snapshots
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from image_preprocessor import ImagePreprocessor        # noqa: E402
from physics_sandbox import PhysicsSandbox                # noqa: E402
from run_replay import ReplayClosedLoopRunner            # noqa: E402
from experience_buffer import ExperienceBuffer           # noqa: E402
from zero_data_model.model import ZeroDataModel         # noqa: E402

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "eval"


# ------------------------------------------------------------------ #
# Inline experiment: train + intervene + measure
# ------------------------------------------------------------------ #
def run_inline_experiment(
    pre_steps: int = 500,
    post_steps: int = 500,
    dim: int = 32,
    seed: int = 42,
    num_objects: int = 3,
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    """Run a fresh model + sandbox, remove an object mid-run, record FE.

    Steps:
    1. Build model + sandbox (with 3 objects so we can remove 1 and
       still have 2 left).
    2. Run ``pre_steps`` online cycles, recording FE.
    3. Remove body index 1 (the first non-agent) via
       ``sandbox.make_object_disappear(1)``.
    4. Run ``post_steps`` more cycles, recording FE.
    5. Compute statistical test (Welch's t-test via scipy if
       available, else a simple z-score on the difference of means).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[inline] training {pre_steps} steps, then removing object, "
          f"then {post_steps} steps")
    model = ZeroDataModel(dim=dim, seed=seed)
    sandbox = PhysicsSandbox(num_objects=num_objects, seed=seed)
    preprocessor = ImagePreprocessor(output_dim=dim, seed=seed)
    buffer = ExperienceBuffer(capacity=5000, seed=seed)
    runner = ReplayClosedLoopRunner(
        model=model, sandbox=sandbox, preprocessor=preprocessor,
        buffer=buffer, max_steps=pre_steps + post_steps,
        consolidation_interval=100, batch_size=16,
        snapshot_interval=10**9,  # disable PNG snapshots
        cognitive_snapshot_interval=10**9,  # disable .npz (we record inline)
        print_interval=max(100, pre_steps // 5),
        output_dir=output_dir,
        enable_consolidation=True,
    )
    # Custom loop: run ``pre_steps``, intervene, then continue.
    t0 = time.perf_counter()
    fe_pre: list[float] = []
    fe_post: list[float] = []
    frame = sandbox.frame.copy()
    for step in range(pre_steps + post_steps):
        if step == pre_steps:
            # INTERVENE: remove body index 1 (non-agent).
            print(f"\n[intervene] step {step}: removing body 1")
            print(f"  before: {len(sandbox.bodies)} bodies, "
                  f"state={sandbox.state_dict()['bodies']}")
            try:
                sandbox.make_object_disappear(1)
                print(f"  after:  {len(sandbox.bodies)} bodies")
            except Exception as exc:
                print(f"  [warn] removal failed: {exc}")
        try:
            record = runner._cycle(frame, step)
        except Exception as exc:
            print(f"[crash] step {step}: {exc}")
            continue
        if step < pre_steps:
            fe_pre.append(record.free_energy)
        else:
            fe_post.append(record.free_energy)
        if (step + 1) % max(100, (pre_steps + post_steps) // 10) == 0:
            phase = "pre" if step < pre_steps else "post"
            print(f"  step {step + 1:5d} ({phase})  fe={record.free_energy:.3f}")
        frame = runner._last_frame
    elapsed = time.perf_counter() - t0
    # Statistics.
    fe_pre_arr = np.array(fe_pre)
    fe_post_arr = np.array(fe_post)
    # Take the first 50 steps after removal as the "surprise" window —
    # later steps reflect re-consolidation, not the initial surprise.
    surprise_window = min(50, len(fe_post_arr))
    fe_surprise = fe_post_arr[:surprise_window]
    # Welch's t-test (scipy optional).
    p_value: float | None = None
    t_stat: float | None = None
    try:
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(
            fe_surprise, fe_pre_arr[-surprise_window:],
            equal_var=False, nan_policy="omit",
        )
        if np.isnan(t_stat):
            t_stat = None
        if np.isnan(p_value):
            p_value = None
    except ImportError:
        # Fallback: simple z-score on the difference of means.
        if len(fe_pre_arr) > 1 and len(fe_surprise) > 1:
            mu_pre = float(fe_pre_arr.mean())
            mu_post = float(fe_surprise.mean())
            sigma_pre = float(fe_pre_arr.std(ddof=1))
            sigma_post = float(fe_surprise.std(ddof=1))
            pooled_se = np.sqrt(
                sigma_pre**2 / len(fe_pre_arr)
                + sigma_post**2 / len(fe_surprise)
            )
            if pooled_se > 0:
                t_stat = (mu_post - mu_pre) / pooled_se
                # Rough two-tailed p via normal approx.
                from math import erf, sqrt
                p_value = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t_stat) / sqrt(2))))
    result = {
        "mode": "inline",
        "pre_steps": int(pre_steps),
        "post_steps": int(post_steps),
        "removal_step": int(pre_steps),
        "n_bodies_before": int(num_objects),
        "n_bodies_after": int(num_objects - 1),
        "fe_pre_mean": float(fe_pre_arr.mean()) if len(fe_pre_arr) else 0.0,
        "fe_pre_std": float(fe_pre_arr.std()) if len(fe_pre_arr) else 0.0,
        "fe_post_mean": float(fe_post_arr.mean()) if len(fe_post_arr) else 0.0,
        "fe_post_std": float(fe_post_arr.std()) if len(fe_post_arr) else 0.0,
        "fe_surprise_mean": float(fe_surprise.mean()) if len(fe_surprise) else 0.0,
        "fe_surprise_max": float(fe_surprise.max()) if len(fe_surprise) else 0.0,
        "delta_fe": (float(fe_surprise.mean() - fe_pre_arr.mean())
                     if len(fe_surprise) and len(fe_pre_arr) else 0.0),
        "t_stat": float(t_stat) if t_stat is not None else None,
        "p_value": float(p_value) if p_value is not None else None,
        "elapsed_s": float(elapsed),
    }
    result_path = output_dir / "disappearance.json"
    with result_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[result] delta_fe={result['delta_fe']:+.3f}  "
          f"t={result['t_stat']}  p={result['p_value']}")
    # Plot.
    if _HAS_MPL:
        fig, ax = plt.subplots(figsize=(10, 4))
        all_fe = np.concatenate([fe_pre_arr, fe_post_arr])
        steps = np.arange(len(all_fe))
        ax.plot(steps, all_fe, color="tab:blue", linewidth=0.7,
                label="free energy")
        # Mark the removal point.
        ax.axvline(pre_steps, color="tab:red", linestyle="--",
                   label=f"removal at step {pre_steps}")
        # Mark the means.
        ax.axhline(result["fe_pre_mean"], color="tab:green",
                   linestyle=":", alpha=0.6, label=f"pre mean={result['fe_pre_mean']:.2f}")
        ax.axhline(result["fe_surprise_mean"], color="tab:orange",
                   linestyle=":", alpha=0.6, label=f"surprise mean={result['fe_surprise_mean']:.2f}")
        ax.set_xlabel("step")
        ax.set_ylabel("free energy")
        ax.set_title("Object Disappearance Experiment")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        png_path = output_dir / "disappearance.png"
        fig.savefig(png_path, dpi=110)
        plt.close(fig)
        print(f"[plot] saved {png_path}")
    return result


# ------------------------------------------------------------------ #
# Snapshots mode: analyse existing .npz files
# ------------------------------------------------------------------ #
def run_snapshots_analysis(
    snapshots_dir: Path,
    removal_step: int,
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    """Analyse FE around a removal step from saved snapshots."""
    if not snapshots_dir.exists():
        print(f"[error] snapshots dir not found: {snapshots_dir}")
        sys.exit(1)
    files = sorted(snapshots_dir.glob("snapshot_*.npz"))
    if not files:
        print(f"[error] no snapshots in {snapshots_dir}")
        sys.exit(1)
    print(f"\n[snapshots] loading {len(files)} snapshots from {snapshots_dir}")
    steps: list[int] = []
    fes: list[float] = []
    for fp in files:
        with np.load(fp, allow_pickle=True) as data:
            steps.append(int(data["step"]))
            fes.append(float(data["free_energy"]))
    steps_arr = np.array(steps)
    fe_arr = np.array(fes)
    # Sort by step.
    order = np.argsort(steps_arr)
    steps_arr = steps_arr[order]
    fe_arr = fe_arr[order]
    # Split into pre/post windows.
    pre_mask = steps_arr < removal_step
    post_mask = steps_arr >= removal_step
    fe_pre = fe_arr[pre_mask]
    fe_post = fe_arr[post_mask]
    surprise_n = min(5, len(fe_post))  # snapshots are sparse
    fe_surprise = fe_post[:surprise_n]
    result = {
        "mode": "snapshots",
        "n_snapshots": int(len(files)),
        "removal_step": int(removal_step),
        "fe_pre_mean": float(fe_pre.mean()) if len(fe_pre) else 0.0,
        "fe_pre_std": float(fe_pre.std()) if len(fe_pre) else 0.0,
        "fe_post_mean": float(fe_post.mean()) if len(fe_post) else 0.0,
        "fe_post_std": float(fe_post.std()) if len(fe_post) else 0.0,
        "fe_surprise_mean": float(fe_surprise.mean()) if len(fe_surprise) else 0.0,
        "delta_fe": (float(fe_surprise.mean() - fe_pre.mean())
                     if len(fe_surprise) and len(fe_pre) else 0.0),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "disappearance.json").open("w") as f:
        json.dump(result, f, indent=2)
    print(f"[result] delta_fe={result['delta_fe']:+.3f}")
    if _HAS_MPL:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(steps_arr, fe_arr, color="tab:blue", linewidth=0.7,
                marker=".", markersize=3)
        ax.axvline(removal_step, color="tab:red", linestyle="--",
                   label=f"removal at step {removal_step}")
        ax.set_xlabel("step")
        ax.set_ylabel("free energy")
        ax.set_title("FE around object disappearance (snapshots)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        png_path = output_dir / "disappearance.png"
        fig.savefig(png_path, dpi=110)
        plt.close(fig)
        print(f"[plot] saved {png_path}")
    return result


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inline", action="store_true",
                   help="Run a fresh experiment instead of loading snapshots")
    p.add_argument("--snapshots", type=str, default=None,
                   help="Directory of .npz snapshots to analyse")
    p.add_argument("--removal_step", type=int, default=500,
                   help="Step at which the object was removed (snapshots mode)")
    p.add_argument("--pre_steps", type=int, default=500)
    p.add_argument("--post_steps", type=int, default=500)
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_objects", type=int, default=3)
    p.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    args = p.parse_args()
    if args.snapshots:
        run_snapshots_analysis(
            snapshots_dir=Path(args.snapshots),
            removal_step=args.removal_step,
            output_dir=Path(args.output_dir),
        )
    else:
        run_inline_experiment(
            pre_steps=args.pre_steps,
            post_steps=args.post_steps,
            dim=args.dim,
            seed=args.seed,
            num_objects=args.num_objects,
            output_dir=Path(args.output_dir),
        )


if __name__ == "__main__":
    main()
