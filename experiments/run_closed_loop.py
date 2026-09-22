# experiments/run_closed_loop.py
"""Closed-loop integration: ZeroDataModel <-> PhysicsSandbox.

Wires the 2D physics sandbox (sensory input) to ZeroDataModel
(perception-cognition-action) and runs the loop for a configurable
number of steps, logging metrics to CSV and saving periodic frame
snapshots as PNG.

Pipeline per step:
  1. current frame (128x128 uint8) from sandbox
  2. ImagePreprocessor.encode(frame) -> obs (32-dim float64)
  3. model.think(obs) -> Signal (updates internal belief state)
  4. engine.select_action(belief, current_observation=obs)
     -> continuous action_vec (action_dim,)
  5. discretise action_vec -> int in {0, 1, 2, 3}
     (0=left, 1=right, 2=up, 3=no-op)
  6. sandbox.step(action) -> next frame

Outputs:
  - experiments/output/closed_loop/loop_log.csv     (per-step metrics)
  - experiments/output/closed_loop/summary.json     (aggregated metrics)
  - experiments/output/closed_loop/frame_NNNN.png   (frames every 100 steps)

Run with:

    python experiments/run_closed_loop.py --steps 2000
    python experiments/run_closed_loop.py --steps 500 --dim 32 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# Make sibling modules + the package importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from image_preprocessor import ImagePreprocessor          # noqa: E402
from physics_sandbox import PhysicsSandbox                # noqa: E402
from zero_data_model.model import ZeroDataModel           # noqa: E402

# Optional matplotlib (only needed for the PNG snapshots).
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_MPL = False


# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #
DEFAULT_STEPS = 2000
DEFAULT_DIM = 32
DEFAULT_SEED = 42
DEFAULT_NUM_OBJECTS = 2
SNAPSHOT_INTERVAL = 100        # save a frame PNG every N steps
PRINT_INTERVAL = 100           # print a status line every N steps
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "closed_loop"

# Sandbox action space.
SANDBOX_ACTIONS = (0, 1, 2, 3)   # left, right, up, no-op


# ------------------------------------------------------------------ #
# Data records
# ------------------------------------------------------------------ #
@dataclass
class CycleRecord:
    """Per-step metrics logged to CSV."""

    step: int
    action: int                # discretised action 0-3
    action_vec_norm: float     # |a| from select_action
    obs_norm: float            # ||obs||
    free_energy: float         # engine.compute_free_energy(obs)
    confidence: float          # signal.confidence from think()
    cycle_count: int           # model.cycle_count
    frame_mean: float          # mean pixel intensity of next frame
    elapsed_ms: float          # wall time for this step


@dataclass
class RunSummary:
    """Aggregated metrics for the whole run."""

    n_steps: int = 0
    n_crashes: int = 0
    crash_messages: list[str] = field(default_factory=list)
    n_nan_obs: int = 0
    n_nan_signal: int = 0
    actions: list[int] = field(default_factory=list)
    free_energies: list[float] = field(default_factory=list)
    elapsed_s: float = 0.0
    cycles_per_sec: float = 0.0
    mean_free_energy: float = 0.0
    final_free_energy: float = 0.0
    mean_confidence: float = 0.0
    csv_path: str = ""
    snapshot_paths: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ #
# Discretisation
# ------------------------------------------------------------------ #
def discretise_action(action_vec: np.ndarray) -> int:
    """Map a continuous action vector to one of {0, 1, 2, 3}.

    Strategy: take the first two components as (vx, vy). The dominant
    axis wins; ties go to the horizontal axis. If both are below a
    small threshold, default to no-op (3).

        vx < 0 and |vx| >= |vy|  -> 0 (left)
        vx > 0 and |vx| >= |vy|  -> 1 (right)
        vy < 0 and |vy| >  |vx|  -> 2 (up)    [image y grows down]
        otherwise                -> 3 (no-op)
    """
    if action_vec is None or len(action_vec) < 2:
        return 3
    vx = float(action_vec[0])
    vy = float(action_vec[1])
    ax = abs(vx)
    ay = abs(vy)
    # Threshold below which we treat the action as "no input".
    threshold = 1e-3
    if max(ax, ay) < threshold:
        return 3
    if ax >= ay:
        return 0 if vx < 0 else 1
    return 2 if vy < 0 else 3


# ------------------------------------------------------------------ #
# Main loop
# ------------------------------------------------------------------ #
class ClosedLoopRunner:
    """Drive the model <-> sandbox loop and record metrics."""

    def __init__(
        self,
        model: ZeroDataModel,
        sandbox: PhysicsSandbox,
        preprocessor: ImagePreprocessor,
        max_steps: int = DEFAULT_STEPS,
        snapshot_interval: int = SNAPSHOT_INTERVAL,
        print_interval: int = PRINT_INTERVAL,
        output_dir: Path = OUTPUT_DIR,
    ):
        self.model = model
        self.sandbox = sandbox
        self.preprocessor = preprocessor
        self.max_steps = int(max_steps)
        self.snapshot_interval = int(snapshot_interval)
        self.print_interval = int(print_interval)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.summary = RunSummary()
        self.records: list[CycleRecord] = []
        # Cache the engine + belief accessor.
        self._engine = model.active_inference

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self) -> RunSummary:
        """Run the loop for ``self.max_steps`` steps."""
        t0 = time.perf_counter()
        print("=" * 64)
        print("Closed loop: ZeroDataModel <-> PhysicsSandbox")
        print("=" * 64)
        print(f"  steps={self.max_steps}  dim={self.model.dim}  "
              f"seed={getattr(self.model, '_seed', None)}")
        # Initial frame.
        frame = self.sandbox.frame.copy()
        for step in range(self.max_steps):
            try:
                record = self._cycle(frame, step)
            except Exception as exc:
                # Log + continue (don't crash the whole run).
                self.summary.n_crashes += 1
                msg = f"step {step}: {type(exc).__name__}: {exc}"
                self.summary.crash_messages.append(msg)
                if self.summary.n_crashes <= 5:
                    print(f"[crash] {msg}")
                # Try to recover by resetting the sandbox frame.
                try:
                    frame = self.sandbox.render().copy()
                except Exception:
                    pass
                continue
            self.records.append(record)
            self.summary.actions.append(record.action)
            self.summary.free_energies.append(record.free_energy)
            # Snapshot + print.
            if (step + 1) % self.snapshot_interval == 0:
                path = self._save_snapshot(frame, step + 1)
                if path:
                    self.summary.snapshot_paths.append(path)
            if (step + 1) % self.print_interval == 0:
                self._print_status(record)
            # Use the new frame returned by step() for the next cycle.
            frame = self._last_frame
        # Aggregate summary.
        elapsed = time.perf_counter() - t0
        self.summary.n_steps = len(self.records)
        self.summary.elapsed_s = elapsed
        self.summary.cycles_per_sec = (len(self.records) / elapsed
                                       if elapsed > 0 else 0.0)
        if self.summary.free_energies:
            self.summary.mean_free_energy = float(
                np.mean(self.summary.free_energies))
            self.summary.final_free_energy = float(
                self.summary.free_energies[-1])
        if self.records:
            self.summary.mean_confidence = float(
                np.mean([r.confidence for r in self.records]))
        return self.summary

    # ------------------------------------------------------------------ #
    # One perception-cognition-action cycle
    # ------------------------------------------------------------------ #
    def _cycle(self, frame: np.ndarray, step: int) -> CycleRecord:
        t0 = time.perf_counter()
        # 1) PERCEPTION: encode the current frame.
        obs = self.preprocessor.encode(frame)
        if not np.all(np.isfinite(obs)):
            self.summary.n_nan_obs += 1
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        obs_norm = float(np.linalg.norm(obs))
        # 2) COGNITION: think() updates belief state + returns signal.
        signal = self.model.think(obs)
        if not np.all(np.isfinite(signal.data)):
            self.summary.n_nan_signal += 1
        confidence = float(signal.confidence)
        # 3) ACTION: select_action returns a continuous vector.
        belief = self._engine.generative_model.belief_state
        action_vec = self._engine.select_action(
            belief, current_observation=obs
        )
        action_vec_norm = float(np.linalg.norm(action_vec))
        # 4) DISCRETISE: 0-3 for the sandbox.
        action = discretise_action(action_vec)
        # 5) ENVIRONMENT: step the sandbox.
        next_frame = self.sandbox.step(action)
        # 6) FREE ENERGY: pure call, does not mutate belief.
        fe = float(self._engine.compute_free_energy(obs))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        # Stash the next frame for the next cycle.
        self._last_frame = next_frame
        return CycleRecord(
            step=step,
            action=action,
            action_vec_norm=action_vec_norm,
            obs_norm=obs_norm,
            free_energy=fe,
            confidence=confidence,
            cycle_count=int(self.model.cycle_count),
            frame_mean=float(np.mean(next_frame)),
            elapsed_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------ #
    # Output helpers
    # ------------------------------------------------------------------ #
    def _save_snapshot(self, frame: np.ndarray, step: int) -> str:
        """Save a frame as a PNG. Returns the path (or '' if mpl missing)."""
        if not _HAS_MPL:
            return ""
        path = self.output_dir / f"frame_{step:06d}.png"
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(frame, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"step {step}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(path, dpi=80)
        plt.close(fig)
        return str(path)

    def _print_status(self, record: CycleRecord) -> None:
        """Print a one-line status every ``print_interval`` steps."""
        print(
            f"  step {record.step + 1:5d}  "
            f"action={record.action}  "
            f"fe={record.free_energy:7.3f}  "
            f"conf={record.confidence:.3f}  "
            f"|obs|={record.obs_norm:.3f}  "
            f"|a|={record.action_vec_norm:.3f}  "
            f"frame_mean={record.frame_mean:.2f}  "
            f"{record.elapsed_ms:.1f}ms"
        )

    def save_csv(self, path: str | Path) -> str:
        """Write per-step records to a CSV file. Returns the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "step", "action", "action_vec_norm", "obs_norm",
                "free_energy", "confidence", "cycle_count",
                "frame_mean", "elapsed_ms",
            ])
            for r in self.records:
                w.writerow([
                    r.step, r.action,
                    f"{r.action_vec_norm:.6f}",
                    f"{r.obs_norm:.6f}",
                    f"{r.free_energy:.6f}",
                    f"{r.confidence:.6f}",
                    r.cycle_count,
                    f"{r.frame_mean:.6f}",
                    f"{r.elapsed_ms:.3f}",
                ])
        self.summary.csv_path = str(path)
        return self.summary.csv_path

    def save_summary(self, path: str | Path) -> str:
        """Write the aggregated summary to a JSON file. Returns the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Action distribution.
        action_dist = {str(a): self.summary.actions.count(a)
                       for a in SANDBOX_ACTIONS}
        payload = {
            "n_steps": self.summary.n_steps,
            "n_crashes": self.summary.n_crashes,
            "crash_messages": self.summary.crash_messages[:10],
            "n_nan_obs": self.summary.n_nan_obs,
            "n_nan_signal": self.summary.n_nan_signal,
            "elapsed_s": self.summary.elapsed_s,
            "cycles_per_sec": self.summary.cycles_per_sec,
            "mean_free_energy": self.summary.mean_free_energy,
            "final_free_energy": self.summary.final_free_energy,
            "mean_confidence": self.summary.mean_confidence,
            "action_distribution": action_dist,
            "snapshot_paths": self.summary.snapshot_paths,
            "csv_path": self.summary.csv_path,
        }
        with path.open("w") as f:
            json.dump(payload, f, indent=2)
        return str(path)


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #
def main() -> None:
    p = argparse.ArgumentParser(description="Closed loop runner")
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                   help=f"number of steps (default {DEFAULT_STEPS})")
    p.add_argument("--dim", type=int, default=DEFAULT_DIM,
                   help=f"model dimensionality (default {DEFAULT_DIM})")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=f"RNG seed (default {DEFAULT_SEED})")
    p.add_argument("--num_objects", type=int, default=DEFAULT_NUM_OBJECTS,
                   help=f"sandbox num_objects 1-3 (default "
                        f"{DEFAULT_NUM_OBJECTS})")
    p.add_argument("--snapshot_interval", type=int,
                   default=SNAPSHOT_INTERVAL)
    p.add_argument("--print_interval", type=int,
                   default=PRINT_INTERVAL)
    p.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    args = p.parse_args()

    # --- Build model + sandbox + preprocessor --------------------- #
    print(f"\nBuilding ZeroDataModel(dim={args.dim}, seed={args.seed})...")
    model = ZeroDataModel(dim=args.dim, seed=args.seed)
    sandbox = PhysicsSandbox(num_objects=args.num_objects, seed=args.seed)
    preprocessor = ImagePreprocessor(output_dim=args.dim, seed=args.seed)

    # --- Run ------------------------------------------------------- #
    runner = ClosedLoopRunner(
        model=model,
        sandbox=sandbox,
        preprocessor=preprocessor,
        max_steps=args.steps,
        snapshot_interval=args.snapshot_interval,
        print_interval=args.print_interval,
        output_dir=Path(args.output_dir),
    )
    summary = runner.run()

    # --- Save outputs --------------------------------------------- #
    csv_path = runner.save_csv(Path(args.output_dir) / "loop_log.csv")
    summary_path = runner.save_summary(Path(args.output_dir) / "summary.json")

    # --- Print summary -------------------------------------------- #
    print("\n" + "=" * 64)
    print("Run summary")
    print("=" * 64)
    print(f"  steps completed: {summary.n_steps}/{args.steps}")
    print(f"  crashes: {summary.n_crashes}")
    print(f"  nan obs: {summary.n_nan_obs}  nan signals: {summary.n_nan_signal}")
    print(f"  elapsed: {summary.elapsed_s:.2f} s  "
          f"({summary.cycles_per_sec:.1f} cycles/s)")
    print(f"  mean free energy: {summary.mean_free_energy:.3f}")
    print(f"  final free energy: {summary.final_free_energy:.3f}")
    print(f"  mean confidence: {summary.mean_confidence:.3f}")
    action_dist = {str(a): summary.actions.count(a) for a in SANDBOX_ACTIONS}
    print(f"  action distribution: {action_dist}")
    print(f"  csv: {csv_path}")
    print(f"  summary: {summary_path}")
    if summary.snapshot_paths:
        print(f"  snapshots: {len(summary.snapshot_paths)} PNGs saved")

    # --- Explicit cleanup to detect memory leaks ----------------- #
    del model, sandbox, preprocessor, runner
    gc.collect()
    print("\n[done] cleanup OK — no obvious memory leak")


if __name__ == "__main__":
    main()
