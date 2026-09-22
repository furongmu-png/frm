# experiments/run_sandbox_replay.py
"""Closed loop with cognitive-snapshot saving + disappearance intervention.

Extends the consolidation runner (Phase F) with:
  - Periodic cognitive snapshots (every ``snapshot_interval`` steps)
    saved as ``.pkl`` files containing the model's internal state.
  - A one-shot disappearance intervention at ``disappear_step``:
    a non-agent object is removed to test object-permanence.
  - A dense per-step log (CSV) of all metrics for offline analysis.

The snapshots are consumed by the ``analyze_*.py`` scripts to evaluate
whether the model has built internal representations of physical
concepts (object permanence, causal structure, category formation,
action-effect consistency).

Run with:  python experiments/run_sandbox_replay.py
"""

from __future__ import annotations

import csv
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from physics_sandbox import PhysicsSandbox  # noqa: E402
from run_sandbox_consolidation import ConsolidationRunSummary  # noqa: E402
from run_sandbox_curious import make_curious_model  # noqa: E402
from run_sandbox_closed_loop import SandboxRunner  # noqa: E402
from zero_data_model.model import ZeroDataModel  # noqa: E402


# --- Configuration ------------------------------------------------- #
DEFAULT_DIM = 64
DEFAULT_NUM_OBJECTS = 3              # need >=2 so disappearance is meaningful
DEFAULT_SEED = 42
DEFAULT_MAX_STEPS = 1000            # full eval uses 10000 via run_full_evaluation
DEFAULT_SNAPSHOT_INTERVAL = 50      # save a snapshot every N steps
DEFAULT_DISAPPEAR_STEP = 500        # remove object 1 at this step
DEFAULT_DISAPPEAR_OBJECT_INDEX = 1  # the second body (index 0 = agent)
DEFAULT_SLEEP_INTERVAL = 200
DEFAULT_SLEEP_ROUNDS = 3
DEFAULT_SLEEP_BATCH_SIZE = 8
DEFAULT_BUFFER_CAPACITY = 2000
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "replay_run"


@dataclass
class ReplayRunSummary(ConsolidationRunSummary):
    """Replay-loop results, including snapshot + intervention metadata."""

    snapshot_paths: list[str] = field(default_factory=list)
    snapshot_steps: list[int] = field(default_factory=list)
    disappear_step: int = -1
    disappear_object_index: int = -1
    disappear_free_energy_before: float = 0.0
    disappear_free_energy_after: float = 0.0


def collect_snapshot(model: ZeroDataModel, step: int, frame: np.ndarray) -> dict:
    """Collect a serialisable cognitive snapshot of the model's state.

    The snapshot is a flat dict of numpy arrays / lists / scalars, all
    picklable. It captures the KEY state from each cognitive module
    (see the exploration report in ``experiments/README.md`` for the
    full field map). Designed to be lightweight enough to save every
    50 steps without dominating runtime.

    Schema (v1.0):
        meta:   {schema_version, step, cycle_count, dim}
        frame:  (128, 128) uint8
        consciousness:  {self_state, self_confidence, workspace_sum,
                         workspace_attention, workspace_buffer_data}
        active_inference: {belief_state, emission, free_energy_history,
                           recent_actions, exploration_step, action_error_history}
        category_engine: {topos_classifier, n_categories, n_functors}
        math_universe:   {fractal_transforms}
        causal_emergence: {memory_patterns, memory_targets, emergence_score_last}
        # emergence_score_last is only set if emergence_cycle was run;
        # the runner does NOT call emergence_cycle every step (too slow),
        # so this is usually None. The analysis scripts run it offline.
    """
    consciousness = model.consciousness
    active_inf = model.active_inference
    category = model.category_engine
    math_uni = model.math_universe
    emergence = model.emergence

    # --- Consciousness core --- #
    ws = consciousness.workspace
    workspace_buffer_data = (
        np.stack([s.data for s in ws.buffer]) if len(ws.buffer) > 0
        else np.zeros((0, model.dim))
    )
    consciousness_snap = {
        "self_state": np.asarray(consciousness.self_model.state, dtype=float).copy(),
        "self_confidence": float(consciousness.self_model.confidence),
        "self_history_len": len(consciousness.self_model.history),
        "workspace_sum": np.asarray(ws._sum, dtype=float).copy(),
        "workspace_attention": np.asarray(ws.attention_weights, dtype=float).copy(),
        "workspace_buffer_data": workspace_buffer_data,
        "workspace_buffer_len": len(ws.buffer),
    }

    # --- Active inference --- #
    gen = active_inf.generative_model
    active_inf_snap = {
        "belief_state": np.asarray(gen.belief_state, dtype=float).copy(),
        "emission": np.asarray(gen.emission, dtype=float).copy(),
        "transition": np.asarray(gen.transition, dtype=float).copy(),
        "free_energy_history": np.asarray(list(active_inf.free_energy_history), dtype=float),
        "action_history": (
            np.stack(list(active_inf.action_history))
            if len(active_inf.action_history) > 0 else np.zeros((0, active_inf.blanket.active_dim))
        ),
        "recent_actions": (
            np.stack(list(active_inf._recent_actions))
            if len(active_inf._recent_actions) > 0 else np.zeros((0, active_inf.blanket.active_dim))
        ),
        "cached_sigma_q2": float(active_inf._cached_sigma_q2),
        "exploration_step": int(active_inf._exploration_step),
        "action_error_history": [
            list(bin_hist) for bin_hist in active_inf._action_error_history
        ],
        "blanket_active_weights": np.asarray(
            active_inf.blanket.active_weights, dtype=float
        ).copy(),
    }

    # --- Category engine --- #
    # Capture only summary counts + the topos classifier (the key
    # learnable "truth judgement" matrix). Full categories dict is
    # large; analysis scripts can reconstruct via the model pickle
    # if needed.
    category_snap = {
        "topos_classifier": np.asarray(category.topos.classifier, dtype=float).copy(),
        "topos_truth_values": np.asarray(category.topos.truth_values, dtype=float).copy(),
        "n_categories": len(category.categories),
        "category_names": list(category.categories.keys()),
        "n_functors": len(category.functors),
        "n_objects_per_category": {
            name: len(cat.objects) for name, cat in category.categories.items()
        },
        "last_process_output": (
            np.asarray(category._last_process_output, dtype=float).copy()
            if category._last_process_output is not None else None
        ),
    }

    # --- Math universe --- #
    # fractal.transforms is the only learnable state.
    fractal_transforms = [
        {
            "scale": np.asarray(t[0], dtype=float).copy(),
            "offset": np.asarray(t[1], dtype=float).copy(),
        }
        for t in math_uni.fractal.transforms
    ]
    math_snap = {
        "fractal_transforms": fractal_transforms,
        "last_process_output": (
            np.asarray(math_uni._last_process_output, dtype=float).copy()
            if math_uni._last_process_output is not None else None
        ),
    }

    # --- Causal emergence --- #
    memory = emergence.memory
    emergence_snap = {
        "memory_patterns": [
            np.asarray(p, dtype=float).copy() for p in memory._patterns
        ],
        "memory_labels": list(memory._labels),
        "memory_targets": [list(t) for t in memory._targets],
        "memory_capacity": len(memory._patterns),
        # emergence_score_last is filled lazily by the runner if it
        # runs emergence_cycle; defaults to None.
        "emergence_score_last": None,
    }

    return {
        "meta": {
            "schema_version": "1.0",
            "step": int(step),
            "cycle_count": int(model.cycle_count),
            "dim": int(model.dim),
        },
        "frame": np.asarray(frame, dtype=np.uint8).copy(),
        "consciousness": consciousness_snap,
        "active_inference": active_inf_snap,
        "category_engine": category_snap,
        "math_universe": math_snap,
        "causal_emergence": emergence_snap,
    }


def save_snapshot(snapshot: dict, path: Path) -> str:
    """Pickle a snapshot to ``path``. Returns the path as string."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)
    return str(path)


def load_snapshot(path: str | Path) -> dict:
    """Load a pickled snapshot. Returns the dict."""
    with Path(path).open("rb") as f:
        return pickle.load(f)


class ReplaySandboxRunner(SandboxRunner):
    """Closed-loop runner with snapshot saving + disappearance intervention.

    Combines:
      - Phase E curiosity (via ``make_curious_model``)
      - Phase F experience buffer + consolidation (inherited logic)
      - Phase G cognitive snapshots (every ``snapshot_interval`` steps)
      - Phase G disappearance intervention (at ``disappear_step``)

    The runner records a dense CSV log (one row per step) with all
    metrics needed for offline analysis, and saves pickled snapshots
    that the ``analyze_*.py`` scripts can load.
    """

    def __init__(
        self,
        model: ZeroDataModel,
        sandbox: PhysicsSandbox,
        max_steps: int = DEFAULT_MAX_STEPS,
        snapshot_interval: int = DEFAULT_SNAPSHOT_INTERVAL,
        disappear_step: int = DEFAULT_DISAPPEAR_STEP,
        disappear_object_index: int = DEFAULT_DISAPPEAR_OBJECT_INDEX,
        sleep_interval: int = DEFAULT_SLEEP_INTERVAL,
        sleep_rounds: int = DEFAULT_SLEEP_ROUNDS,
        sleep_batch_size: int = DEFAULT_SLEEP_BATCH_SIZE,
        buffer_capacity: int = DEFAULT_BUFFER_CAPACITY,
        output_dir: Path = OUTPUT_DIR,
        seed: int | None = None,
    ):
        super().__init__(model, sandbox, max_steps)
        self.summary = ReplayRunSummary()
        self.snapshot_interval = int(snapshot_interval)
        self.disappear_step = int(disappear_step)
        self.disappear_object_index = int(disappear_object_index)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Lazily import the consolidator + buffer (Phase F reuse).
        from experience_buffer import ExperienceBuffer, OfflineConsolidator
        self.buffer = ExperienceBuffer(
            capacity=buffer_capacity,
            surprise_weighted=True,
            seed=seed,
        )
        self.consolidator = OfflineConsolidator(
            batch_size=sleep_batch_size,
            sampling_mode="priority",
            seed=seed,
        )
        self.sleep_interval = int(sleep_interval)
        self.sleep_rounds = int(sleep_rounds)
        # Track the sandbox object positions BEFORE disappearance for
        # the analysis scripts to compare against.
        self._disappeared = False
        self._pre_disappear_fe: float | None = None

    # ------------------------------------------------------------------ #
    # Override the cycle to add snapshot saving + intervention
    # ------------------------------------------------------------------ #
    def _cycle(self, frame: np.ndarray, step: int) -> np.ndarray:
        """One cycle: perceive → think → select → act + snapshot + intervene."""
        # --- Disappearance intervention (BEFORE the step) --- #
        # Apply the intervention at the START of the target step so the
        # model's response is captured in THIS step's metrics.
        if step == self.disappear_step and not self._disappeared:
            # Record pre-disappearance free energy for delta computation.
            obs_pre = self.model.encode_image(frame)
            self._pre_disappear_fe = float(
                self.model.active_inference.compute_free_energy(obs_pre)
            )
            # Remove the object.
            n_before = len(self.sandbox.bodies)
            self.sandbox.make_object_disappear(self.disappear_object_index)
            n_after = len(self.sandbox.bodies)
            print(
                f"  [intervention] step {step}: removed object "
                f"{self.disappear_object_index} (bodies: {n_before} -> {n_after})"
            )
            self._disappeared = True
            # Re-render the frame so the model SEES the disappearance
            # immediately on this step.
            frame = self.sandbox.render()
            self.summary.disappear_step = step
            self.summary.disappear_object_index = self.disappear_object_index
            self.summary.disappear_free_energy_before = self._pre_disappear_fe

        # 1) PERCEPTION
        obs = self.model.encode_image(frame)
        if not np.all(np.isfinite(obs)):
            self.summary.n_nan_obs += 1
        obs_norm = float(np.linalg.norm(obs)) if np.all(np.isfinite(obs)) else float("nan")

        # 2) COGNITION
        signal = self.model.think(obs)
        if not np.all(np.isfinite(signal.data)):
            self.summary.n_nan_signal += 1

        # 3) ACTION SELECTION (curiosity)
        engine = self.model.active_inference
        beta_before = engine._compute_beta()
        belief = engine.generative_model.belief_state
        action_vec = engine.select_action(belief, current_observation=obs)
        info_gain = engine._compute_information_gain_proxy(action_vec)
        action_bin = engine._action_bin(action_vec)
        action = self._discretize_action(action_vec)

        # 4) ENVIRONMENT
        next_frame = self.sandbox.step(action)
        next_obs = self.model.encode_image(next_frame)

        # 5) STORE EXPERIENCE (Phase F)
        free_energy = float(engine.compute_free_energy(obs))
        self.buffer.add(
            obs=obs, action=action, next_obs=next_obs,
            error=free_energy, info_gain=info_gain, step=step,
        )

        # 6) PERIODIC SLEEP (Phase F)
        if (step + 1) % self.sleep_interval == 0 and len(self.buffer) > 0:
            results = self.consolidator.consolidate_many(
                self.model, self.buffer, n_rounds=self.sleep_rounds
            )
            self.summary.sleep_rounds.extend(results)
            self.summary.n_sleep_phases += 1
            self.summary.total_replayed += sum(r.n_replayed for r in results)

        # 7) PERIODIC SNAPSHOT (Phase G)
        if (step + 1) % self.snapshot_interval == 0:
            snap = collect_snapshot(self.model, step=step + 1, frame=next_frame)
            snap_path = self.output_dir / f"snapshot_step{step + 1:06d}.pkl"
            save_snapshot(snap, snap_path)
            self.summary.snapshot_paths.append(str(snap_path))
            self.summary.snapshot_steps.append(step + 1)

        # 8) RECORD post-disappearance free energy
        if step == self.disappear_step and self._disappeared:
            self.summary.disappear_free_energy_after = free_energy
            print(
                f"  [intervention] post-disappear FE: "
                f"before={self._pre_disappear_fe:.4f} after={free_energy:.4f} "
                f"delta={free_energy - self._pre_disappear_fe:+.4f}"
            )

        # 9) RECORD per-step metrics
        self.summary.records.append(_make_record(
            step=step, action=action, action_vec=action_vec,
            signal=signal, obs=obs, next_frame=next_frame,
            beta=beta_before, info_gain=info_gain, action_bin=action_bin,
            free_energy=free_energy,
        ))
        return next_frame

    # ------------------------------------------------------------------ #
    # Output: dense CSV log
    # ------------------------------------------------------------------ #
    def save_csv(self, path: str | Path) -> str:
        """Write the per-step CSV log (one row per step)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "step", "action", "action_bin", "action_vec_norm",
                "free_energy", "confidence", "obs_norm", "frame_mean",
                "beta", "info_gain",
                "agent_cx", "agent_cy", "agent_vx", "agent_vy",
                "n_bodies", "is_post_disappear",
            ])
            for r in self.summary.records:
                # Recover agent + body count from the sandbox state at
                # record time — we stored frame_mean but not bodies, so
                # we re-derive from the sandbox (assuming linear order).
                # NOTE: this is a simplification — the sandbox state at
                # CSV-write time is the FINAL state, not per-step. For
                # per-step body info, use the snapshots.
                agent = self.sandbox.bodies[0] if len(self.sandbox.bodies) > 0 else None
                w.writerow([
                    r.step, r.action, r.action_bin,
                    f"{r.action_vec_norm:.6f}",
                    f"{r.free_energy:.6f}",
                    f"{r.confidence:.6f}",
                    f"{r.obs_norm:.6f}",
                    f"{r.frame_mean:.6f}",
                    f"{r.beta:.6f}",
                    f"{r.info_gain:.6f}",
                    f"{agent.cx:.4f}" if agent else "",
                    f"{agent.cy:.4f}" if agent else "",
                    f"{agent.vx:.4f}" if agent else "",
                    f"{agent.vy:.4f}" if agent else "",
                    len(self.sandbox.bodies),
                    int(r.step >= self.disappear_step),
                ])
        self.summary.csv_path = str(path)
        return self.summary.csv_path


# ------------------------------------------------------------------ #
# Helper: build a CycleRecord-compatible dict
# ------------------------------------------------------------------ #
def _make_record(
    step: int, action: int, action_vec: np.ndarray, signal,
    obs: np.ndarray, next_frame: np.ndarray,
    beta: float, info_gain: float, action_bin: int,
    free_energy: float,
):
    """Build a per-step record (reuses CuriousCycleRecord's fields)."""
    # Import locally to avoid circular import at module load.
    from run_sandbox_curious import CuriousCycleRecord
    return CuriousCycleRecord(
        step=step,
        action=action,
        action_vec_norm=float(np.linalg.norm(action_vec)),
        free_energy=free_energy,
        confidence=float(signal.confidence),
        obs_norm=float(np.linalg.norm(obs)) if np.all(np.isfinite(obs)) else float("nan"),
        frame_mean=float(np.mean(next_frame)),
        rss_kb=0.0,
        beta=beta,
        info_gain=info_gain,
        action_bin=action_bin,
    )


# ------------------------------------------------------------------ #
# Main entry point
# ------------------------------------------------------------------ #
def main() -> None:
    """Run the replay loop with snapshots + disappearance intervention."""
    print("=" * 64)
    print("Replay Loop with Cognitive Snapshots + Disappearance")
    print("=" * 64)

    model = make_curious_model(
        dim=DEFAULT_DIM, seed=DEFAULT_SEED,
        beta_start=1.0, beta_min=0.01,
        decay_steps=DEFAULT_MAX_STEPS, decay_type="linear",
    )
    sandbox = PhysicsSandbox(
        num_objects=DEFAULT_NUM_OBJECTS, seed=DEFAULT_SEED,
    )
    runner = ReplaySandboxRunner(
        model, sandbox,
        max_steps=DEFAULT_MAX_STEPS,
        snapshot_interval=DEFAULT_SNAPSHOT_INTERVAL,
        disappear_step=DEFAULT_DISAPPEAR_STEP,
        disappear_object_index=DEFAULT_DISAPPEAR_OBJECT_INDEX,
        sleep_interval=DEFAULT_SLEEP_INTERVAL,
        sleep_rounds=DEFAULT_SLEEP_ROUNDS,
        output_dir=OUTPUT_DIR,
        seed=DEFAULT_SEED,
    )

    print(f"\nRunning {DEFAULT_MAX_STEPS} steps:")
    print(f"  snapshot every {DEFAULT_SNAPSHOT_INTERVAL} steps")
    print(f"  disappearance at step {DEFAULT_DISAPPEAR_STEP} (object {DEFAULT_DISAPPEAR_OBJECT_INDEX})")
    print(f"  sleep every {DEFAULT_SLEEP_INTERVAL} steps ({DEFAULT_SLEEP_ROUNDS} rounds)")
    runner.run()

    # Save CSV + manifest.
    csv_path = runner.save_csv(OUTPUT_DIR / "replay_log.csv")
    print(f"\nCSV: {csv_path}")
    print(f"Snapshots: {len(runner.summary.snapshot_paths)} files in {OUTPUT_DIR}")

    # Save a manifest JSON for the analysis scripts.
    import json
    manifest = {
        "snapshot_paths": runner.summary.snapshot_paths,
        "snapshot_steps": runner.summary.snapshot_steps,
        "disappear_step": runner.summary.disappear_step,
        "disappear_object_index": runner.summary.disappear_object_index,
        "disappear_free_energy_before": runner.summary.disappear_free_energy_before,
        "disappear_free_energy_after": runner.summary.disappear_free_energy_after,
        "max_steps": DEFAULT_MAX_STEPS,
        "dim": DEFAULT_DIM,
        "seed": DEFAULT_SEED,
        "num_objects": DEFAULT_NUM_OBJECTS,
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {manifest_path}")

    print("\n" + "=" * 64)
    print("Done. Run analyze_*.py to evaluate cognitive emergence.")
    print("=" * 64)


if __name__ == "__main__":
    main()
