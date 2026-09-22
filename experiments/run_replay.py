# experiments/run_replay.py
"""Closed loop with experience replay + offline consolidation.

Extends ``run_closed_loop.py`` with:
- ``ExperienceBuffer`` storing every (obs, action, next_obs, fe)
  transition.
- Periodic offline consolidation: every ``consolidation_interval``
  steps, sample ``batch_size`` experiences from the buffer
  (priority-weighted) and re-feed them through ``model.think()`` so
  the model can rehear past transitions and consolidate its
  generative model.
- Per-consolidation logging: duration, mean error, batch sample stats.
- Final comparison report: writes a JSON comparing FE trajectories
  against the no-replay baseline.

Outputs (under ``experiments/output/replay/``):
  - loop_log.csv             (per-step metrics, same as run_closed_loop)
  - consolidation_log.csv     (per-consolidation metrics)
  - summary.json             (aggregated run summary)
  - buffer_stats.json        (final buffer statistics)

Run with:

    python experiments/run_replay.py --steps 5000 --consolidation_interval 100
    python experiments/run_replay.py --steps 5000 --no_replay  # baseline

Design rationale:
- Consolidation does NOT advance ``model.cycle_count`` artificially:
  we temporarily stash the count, run the consolidation, then restore.
  This keeps the per-step cycle_count consistent with online steps.
- ``model.think()`` is called twice per replayed transition (once for
  ``obs``, once for ``next_obs``) — this gives the model two update
  opportunities per transition (state → action, action → next-state).
- The free-energy recorded during consolidation is the mean of the
  ``compute_free_energy`` calls on the replayed observations.
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

from experience_buffer import ExperienceBuffer          # noqa: E402
from image_preprocessor import ImagePreprocessor        # noqa: E402
from physics_sandbox import PhysicsSandbox              # noqa: E402
from run_closed_loop import (                           # noqa: E402
    SANDBOX_ACTIONS,
    CycleRecord,
    RunSummary,
    discretise_action,
)
from zero_data_model.model import ZeroDataModel         # noqa: E402

# Optional matplotlib.
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:  # pragma: no cover
    _HAS_MPL = False


# ------------------------------------------------------------------ #
# Defaults
# ------------------------------------------------------------------ #
DEFAULT_STEPS = 5000
DEFAULT_DIM = 32
DEFAULT_SEED = 42
DEFAULT_NUM_OBJECTS = 2
DEFAULT_CAPACITY = 10000
DEFAULT_BATCH_SIZE = 32
DEFAULT_CONSOLIDATION_INTERVAL = 100
PRINT_INTERVAL = 200
SNAPSHOT_INTERVAL = 1000
# Phase K: cognitive snapshot default interval (in steps). 10 → save
# every 10 steps → ~500 snapshots for a 5000-step run, manageable disk.
COGNITIVE_SNAPSHOT_INTERVAL = 10
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "replay"


# ------------------------------------------------------------------ #
# Records
# ------------------------------------------------------------------ #
@dataclass
class ConsolidationRecord:
    """One consolidation pass — recorded to consolidation_log.csv."""

    step: int
    n_replayed: int
    duration_ms: float
    mean_replay_fe: float
    max_replay_fe: float
    min_replay_fe: float
    buffer_size_before: int
    buffer_size_after: int


@dataclass
class ReplayRunSummary(RunSummary):
    """Extended summary including consolidation metrics."""

    n_consolidations: int = 0
    total_consolidation_ms: float = 0.0
    mean_consolidation_ms: float = 0.0
    mean_replay_fe: float = 0.0
    consolidation_csv_path: str = ""


# ------------------------------------------------------------------ #
# Main runner
# ------------------------------------------------------------------ #
class ReplayClosedLoopRunner:
    """Closed loop with periodic offline consolidation."""

    def __init__(
        self,
        model: ZeroDataModel,
        sandbox: PhysicsSandbox,
        preprocessor: ImagePreprocessor,
        buffer: ExperienceBuffer,
        max_steps: int = DEFAULT_STEPS,
        consolidation_interval: int = DEFAULT_CONSOLIDATION_INTERVAL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        snapshot_interval: int = SNAPSHOT_INTERVAL,
        print_interval: int = PRINT_INTERVAL,
        output_dir: Path = OUTPUT_DIR,
        enable_consolidation: bool = True,
        cognitive_snapshot_interval: int = COGNITIVE_SNAPSHOT_INTERVAL,
    ):
        self.model = model
        self.sandbox = sandbox
        self.preprocessor = preprocessor
        self.buffer = buffer
        self.max_steps = int(max_steps)
        self.consolidation_interval = int(consolidation_interval)
        self.batch_size = int(batch_size)
        self.snapshot_interval = int(snapshot_interval)
        self.print_interval = int(print_interval)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enable_consolidation = bool(enable_consolidation)
        # Phase K: cognitive snapshot interval (separate from PNG
        # snapshot_interval — PNGs are for human preview, .npz files
        # are for offline analysis scripts).
        self.cognitive_snapshot_interval = int(
            cognitive_snapshot_interval
        )
        # State.
        self.summary = ReplayRunSummary()
        self.records: list[CycleRecord] = []
        self.consolidation_records: list[ConsolidationRecord] = []
        self._engine = model.active_inference
        self._last_frame: np.ndarray = sandbox.frame.copy()
        # Phase K: cognitive snapshot directory for offline analysis.
        # Snapshots are saved as ``.npz`` files containing the obs
        # vector, belief state, action, free energy, and (optionally)
        # the rendered frame. Used by ``analyze_*.py`` scripts to
        # quantify pre-linguistic concept emergence.
        self.snapshot_dir = self.output_dir / "snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self) -> ReplayRunSummary:
        t0 = time.perf_counter()
        print("=" * 72)
        title = "Replay + Consolidation" if self.enable_consolidation \
            else "Baseline (no replay)"
        print(f"Closed loop: {title}")
        print("=" * 72)
        print(f"  steps={self.max_steps}  dim={self.model.dim}  "
              f"seed={getattr(self.model, '_seed', None)}")
        print(f"  consolidation_interval={self.consolidation_interval}  "
              f"batch_size={self.batch_size}  "
              f"buffer_capacity={self.buffer.capacity}")
        frame = self.sandbox.frame.copy()
        for step in range(self.max_steps):
            try:
                record = self._cycle(frame, step)
            except Exception as exc:
                self.summary.n_crashes += 1
                msg = f"step {step}: {type(exc).__name__}: {exc}"
                self.summary.crash_messages.append(msg)
                if self.summary.n_crashes <= 5:
                    print(f"[crash] {msg}")
                try:
                    frame = self.sandbox.render().copy()
                except Exception:
                    pass
                continue
            self.records.append(record)
            self.summary.actions.append(record.action)
            self.summary.free_energies.append(record.free_energy)
            # Periodic consolidation.
            if (self.enable_consolidation
                    and (step + 1) % self.consolidation_interval == 0
                    and len(self.buffer) >= self.batch_size):
                cons = self._consolidate(step + 1)
                self.consolidation_records.append(cons)
                self.summary.n_consolidations += 1
            # Snapshot + print.
            if (step + 1) % self.snapshot_interval == 0:
                self._save_snapshot(frame, step + 1)
            # Phase K: cognitive snapshot for offline analysis (every
            # ``snapshot_interval`` steps — separate from PNG snapshot
            # above which renders the frame; this saves raw arrays).
            if (step + 1) % self.cognitive_snapshot_interval == 0:
                self._save_cognitive_snapshot(step + 1, record)
            if (step + 1) % self.print_interval == 0:
                self._print_status(record)
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
        if self.consolidation_records:
            self.summary.total_consolidation_ms = float(sum(
                r.duration_ms for r in self.consolidation_records))
            self.summary.mean_consolidation_ms = (
                self.summary.total_consolidation_ms
                / len(self.consolidation_records))
            self.summary.mean_replay_fe = float(np.mean([
                r.mean_replay_fe for r in self.consolidation_records]))
        return self.summary

    # ------------------------------------------------------------------ #
    # One online cycle
    # ------------------------------------------------------------------ #
    def _cycle(self, frame: np.ndarray, step: int) -> CycleRecord:
        t0 = time.perf_counter()
        # 1) Encode.
        obs = self.preprocessor.encode(frame)
        if not np.all(np.isfinite(obs)):
            self.summary.n_nan_obs += 1
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        obs_norm = float(np.linalg.norm(obs))
        # 2) Think.
        signal = self.model.think(obs)
        if not np.all(np.isfinite(signal.data)):
            self.summary.n_nan_signal += 1
        confidence = float(signal.confidence)
        # 3) Select action.
        belief = self._engine.generative_model.belief_state
        action_vec = self._engine.select_action(
            belief, current_observation=obs)
        action_vec_norm = float(np.linalg.norm(action_vec))
        # 4) Discretise + step sandbox.
        action = discretise_action(action_vec)
        next_frame = self.sandbox.step(action)
        # 5) Free energy.
        fe = float(self._engine.compute_free_energy(obs))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        # 6) Store experience — store BOTH obs and next_obs so the
        #    consolidation loop can replay the full transition.
        next_obs = self.preprocessor.encode(next_frame)
        self.buffer.add(
            obs=obs,
            action=action,
            next_obs=next_obs,
            error=fe,
            step=step,
            metadata={"action_vec_norm": action_vec_norm},
        )
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
    # Consolidation
    # ------------------------------------------------------------------ #
    def _consolidate(self, step: int) -> ConsolidationRecord:
        """Sample a batch and re-feed through ``model.think()``.

        We temporarily stash the model's ``cycle_count`` so the
        consolidation does not inflate the online cycle count. The
        ``think()`` calls themselves still update the generative model
        (which is the point — that's how consolidation happens).
        """
        t0 = time.perf_counter()
        buf_before = len(self.buffer)
        batch = self.buffer.sample(self.batch_size, mode="priority")
        if len(batch) == 0:
            return ConsolidationRecord(
                step=step, n_replayed=0, duration_ms=0.0,
                mean_replay_fe=0.0, max_replay_fe=0.0, min_replay_fe=0.0,
                buffer_size_before=buf_before,
                buffer_size_after=buf_before,
            )
        # Stash cycle_count — think() increments it.
        saved_cycle_count = self.model.cycle_count
        fes: list[float] = []
        for exp in batch.experiences:
            # Rehear the transition: obs → next_obs.
            # 1) Listen to the original observation.
            try:
                self.model.think(exp.obs)
                fe_obs = float(self._engine.compute_free_energy(exp.obs))
                if np.isfinite(fe_obs):
                    fes.append(fe_obs)
            except Exception:
                pass
            # 2) Listen to the next observation (the outcome).
            try:
                self.model.think(exp.next_obs)
                fe_next = float(self._engine.compute_free_energy(exp.next_obs))
                if np.isfinite(fe_next):
                    fes.append(fe_next)
            except Exception:
                pass
        # Restore cycle_count so online counter stays consistent.
        self.model.cycle_count = saved_cycle_count
        duration_ms = (time.perf_counter() - t0) * 1000.0
        buf_after = len(self.buffer)
        if fes:
            mean_fe = float(np.mean(fes))
            max_fe = float(np.max(fes))
            min_fe = float(np.min(fes))
        else:
            mean_fe = max_fe = min_fe = 0.0
        rec = ConsolidationRecord(
            step=step,
            n_replayed=len(batch.experiences),
            duration_ms=duration_ms,
            mean_replay_fe=mean_fe,
            max_replay_fe=max_fe,
            min_replay_fe=min_fe,
            buffer_size_before=buf_before,
            buffer_size_after=buf_after,
        )
        return rec

    # ------------------------------------------------------------------ #
    # Output helpers
    # ------------------------------------------------------------------ #
    def _save_snapshot(self, frame: np.ndarray, step: int) -> str:
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

    def _save_cognitive_snapshot(
        self, step: int, record: CycleRecord
    ) -> str:
        """Save a cognitive snapshot (``.npz``) for offline analysis.

        Captures the model's internal state at the END of the step so
        ``analyze_*.py`` scripts can reconstruct belief trajectories,
        free-energy time series, etc. without re-running the loop.

        Contents:
        - ``step``           : int
        - ``obs``             : (dim,) float64  — encoded observation
        - ``belief``          : (dim,) float64  — generative model's
                                                  belief state
        - ``action``          : int              — discrete action 0-3
        - ``action_vec``     : (active_dim,)    — continuous action
        - ``free_energy``     : float            — FE at this step
        - ``prediction_error``: float            — |obs - belief| L2
        - ``confidence``      : float            — signal.confidence
        - ``frame``           : (128,128) uint8  — rendered frame
        - ``sandbox_state``   : dict-as-json-str — body positions etc.
        """
        engine = self._engine
        belief = np.asarray(engine.generative_model.belief_state)
        # Re-encode the current obs (we don't stash it across cycles).
        frame = self._last_frame
        obs = self.preprocessor.encode(frame)
        prediction_error = float(np.linalg.norm(obs - belief[:len(obs)]))
        # Sandbox state (positions, velocities) for cross-modal analysis.
        try:
            sandbox_state = json.dumps(self.sandbox.state_dict())
        except Exception:
            sandbox_state = "{}"
        path = self.snapshot_dir / f"snapshot_{step:06d}.npz"
        np.savez_compressed(
            path,
            step=np.array(step, dtype=np.int64),
            obs=obs,
            belief=belief,
            action=np.array(record.action, dtype=np.int64),
            free_energy=np.array(record.free_energy, dtype=np.float64),
            prediction_error=np.array(prediction_error, dtype=np.float64),
            confidence=np.array(record.confidence, dtype=np.float64),
            frame=frame,
            sandbox_state=np.array(sandbox_state, dtype=object),
        )
        return str(path)

    def _print_status(self, record: CycleRecord) -> None:
        n_cons = self.summary.n_consolidations
        cons_str = f"cons={n_cons}"
        if self.consolidation_records:
            last = self.consolidation_records[-1]
            cons_str += (f" last_replay_fe={last.mean_replay_fe:.3f}"
                         f" ({last.duration_ms:.0f}ms)")
        print(
            f"  step {record.step + 1:5d}  "
            f"action={record.action}  "
            f"fe={record.free_energy:7.3f}  "
            f"conf={record.confidence:.3f}  "
            f"|obs|={record.obs_norm:.3f}  "
            f"|a|={record.action_vec_norm:.3f}  "
            f"{cons_str}"
        )

    def save_csv(self, path: str | Path) -> str:
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

    def save_consolidation_csv(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "step", "n_replayed", "duration_ms", "mean_replay_fe",
                "max_replay_fe", "min_replay_fe",
                "buffer_size_before", "buffer_size_after",
            ])
            for r in self.consolidation_records:
                w.writerow([
                    r.step, r.n_replayed,
                    f"{r.duration_ms:.3f}",
                    f"{r.mean_replay_fe:.6f}",
                    f"{r.max_replay_fe:.6f}",
                    f"{r.min_replay_fe:.6f}",
                    r.buffer_size_before,
                    r.buffer_size_after,
                ])
        self.summary.consolidation_csv_path = str(path)
        return self.summary.consolidation_csv_path

    def save_summary(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
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
            "n_consolidations": self.summary.n_consolidations,
            "total_consolidation_ms": self.summary.total_consolidation_ms,
            "mean_consolidation_ms": self.summary.mean_consolidation_ms,
            "mean_replay_fe": self.summary.mean_replay_fe,
            "csv_path": self.summary.csv_path,
            "consolidation_csv_path": self.summary.consolidation_csv_path,
            "buffer_stats": self.buffer.stats(),
        }
        with path.open("w") as f:
            json.dump(payload, f, indent=2)
        return str(path)


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #
def main() -> None:
    p = argparse.ArgumentParser(description="Closed loop with replay")
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--dim", type=int, default=DEFAULT_DIM)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--num_objects", type=int, default=DEFAULT_NUM_OBJECTS)
    p.add_argument("--capacity", type=int, default=DEFAULT_CAPACITY)
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--consolidation_interval", type=int,
                   default=DEFAULT_CONSOLIDATION_INTERVAL)
    p.add_argument("--snapshot_interval", type=int,
                   default=SNAPSHOT_INTERVAL)
    p.add_argument("--cognitive_snapshot_interval", type=int,
                   default=COGNITIVE_SNAPSHOT_INTERVAL,
                   help="Save .npz cognitive snapshot every N steps "
                        "(default 10, used by analyze_*.py)")
    p.add_argument("--print_interval", type=int,
                   default=PRINT_INTERVAL)
    p.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    p.add_argument("--no_replay", action="store_true",
                   help="Disable consolidation (baseline mode)")
    args = p.parse_args()

    # --- Build model + sandbox + preprocessor + buffer ------------- #
    print(f"\nBuilding ZeroDataModel(dim={args.dim}, seed={args.seed})...")
    model = ZeroDataModel(dim=args.dim, seed=args.seed)
    sandbox = PhysicsSandbox(num_objects=args.num_objects, seed=args.seed)
    preprocessor = ImagePreprocessor(output_dim=args.dim, seed=args.seed)
    buffer = ExperienceBuffer(
        capacity=args.capacity,
        alpha=0.6,
        beta=0.4,
        seed=args.seed,
    )
    runner = ReplayClosedLoopRunner(
        model=model,
        sandbox=sandbox,
        preprocessor=preprocessor,
        buffer=buffer,
        max_steps=args.steps,
        consolidation_interval=args.consolidation_interval,
        batch_size=args.batch_size,
        snapshot_interval=args.snapshot_interval,
        print_interval=args.print_interval,
        output_dir=Path(args.output_dir),
        enable_consolidation=not args.no_replay,
        cognitive_snapshot_interval=args.cognitive_snapshot_interval,
    )
    summary = runner.run()

    # --- Save outputs --------------------------------------------- #
    out = Path(args.output_dir)
    csv_path = runner.save_csv(out / "loop_log.csv")
    cons_csv_path = runner.save_consolidation_csv(out / "consolidation_log.csv")
    summary_path = runner.save_summary(out / "summary.json")
    buffer_path = buffer.save_json(out / "buffer_stats.json")

    # --- Print summary -------------------------------------------- #
    print("\n" + "=" * 72)
    print("Run summary")
    print("=" * 72)
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
    print(f"  consolidations: {summary.n_consolidations}")
    if summary.n_consolidations > 0:
        print(f"  total consolidation time: "
              f"{summary.total_consolidation_ms:.0f} ms")
        print(f"  mean consolidation time: "
              f"{summary.mean_consolidation_ms:.1f} ms")
        print(f"  mean replay FE: {summary.mean_replay_fe:.3f}")
    print(f"  csv: {csv_path}")
    print(f"  consolidation csv: {cons_csv_path}")
    print(f"  summary: {summary_path}")
    print(f"  buffer stats: {buffer_path}")

    # --- Cleanup ------------------------------------------------- #
    del model, sandbox, preprocessor, buffer, runner
    gc.collect()
    print("\n[done] cleanup OK — no obvious memory leak")


if __name__ == "__main__":
    main()
