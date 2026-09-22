# experiments/run_sandbox_consolidation.py
"""Closed loop with periodic offline consolidation (sleep cycles).

Extends ``run_sandbox_curious.CuriousSandboxRunner`` with:
  - An ``ExperienceBuffer`` that stores every online transition.
  - An ``OfflineConsolidator`` that runs a "sleep" phase every
    ``sleep_interval`` online steps, replaying ``sleep_rounds`` batches
    to consolidate high-surprise experiences.

Output: CSV (per-step + per-sleep-round) and a PNG showing free-energy
evolution across wake/sleep phases.

Run with:  python experiments/run_sandbox_consolidation.py
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from experience_buffer import (  # noqa: E402
    ConsolidationResult,
    ExperienceBuffer,
    OfflineConsolidator,
)
from physics_sandbox import PhysicsSandbox  # noqa: E402
from run_sandbox_closed_loop import SandboxRunner  # noqa: E402
from run_sandbox_curious import (  # noqa: E402
    CuriousCycleRecord,
    CuriousRunSummary,
    make_curious_model,
)

from zero_data_model.model import ZeroDataModel  # noqa: E402

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_MPL = False


# --- Configuration ------------------------------------------------- #
DEFAULT_DIM = 64
DEFAULT_NUM_OBJECTS = 2
DEFAULT_SEED = 42
DEFAULT_MAX_STEPS = 1000
# Sleep schedule: every ``sleep_interval`` online steps, run a sleep
# phase of ``sleep_rounds`` consolidation batches (each of size
# ``sleep_batch_size``). The defaults give ~5 sleep phases over 1000
# steps, each replaying 8*5=40 experiences — light enough not to
# dominate runtime, heavy enough to show a measurable free-energy drop.
DEFAULT_SLEEP_INTERVAL = 200
DEFAULT_SLEEP_ROUNDS = 5
DEFAULT_SLEEP_BATCH_SIZE = 8
DEFAULT_BUFFER_CAPACITY = 2000
DEFAULT_SAMPLING_MODE = "priority"  # surprise-weighted replay
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


@dataclass
class ConsolidationCycleRecord(CuriousCycleRecord):
    """Per-step record augmented with consolidation metadata."""

    # After a sleep phase, the NEXT ``sleep_interval`` records carry
    # the cumulative effect of the last consolidation round. These
    # fields let us correlate online free-energy changes with the
    # most recent sleep round.
    last_sleep_round: int = -1       # index of the last sleep round (-1 = none yet)
    last_sleep_delta_fe: float = 0.0  # delta_fe of the last sleep round
    last_sleep_replayed: int = 0     # experiences replayed in the last sleep


@dataclass
class ConsolidationRunSummary(CuriousRunSummary):
    """Aggregated results including consolidation rounds."""

    sleep_rounds: list[ConsolidationResult] = field(default_factory=list)
    n_sleep_phases: int = 0
    total_replayed: int = 0
    mean_sleep_delta_fe: float = 0.0
    consolidation_log_path: str = ""
    png_path: str = ""


class ConsolidatingSandboxRunner(SandboxRunner):
    """Closed-loop runner with periodic offline consolidation.

    Online phase: identical to ``CuriousSandboxRunner`` (perceive →
    think → select → act), but every transition is also stored in an
    ``ExperienceBuffer``.

    Sleep phase: every ``sleep_interval`` steps, pause the online
    loop and run ``sleep_rounds`` consolidation batches. Each batch
    samples from the buffer (default: surprise-weighted priority) and
    replays each experience through
    ``engine.generative_model.update_belief(obs) → engine.update(error)``
    to strengthen the generative model's predictions.

    The runner records both the online step metrics (free energy, beta,
    info gain) and the sleep-phase outcomes (delta_fe per round), so
    the effect of consolidation on subsequent online performance can
    be analysed.
    """

    def __init__(
        self,
        model: ZeroDataModel,
        sandbox: PhysicsSandbox,
        max_steps: int = DEFAULT_MAX_STEPS,
        sleep_interval: int = DEFAULT_SLEEP_INTERVAL,
        sleep_rounds: int = DEFAULT_SLEEP_ROUNDS,
        sleep_batch_size: int = DEFAULT_SLEEP_BATCH_SIZE,
        buffer_capacity: int = DEFAULT_BUFFER_CAPACITY,
        sampling_mode: str = DEFAULT_SAMPLING_MODE,
        seed: int | None = None,
    ):
        super().__init__(model, sandbox, max_steps)
        self.summary = ConsolidationRunSummary()
        self.sleep_interval = int(sleep_interval)
        self.sleep_rounds = int(sleep_rounds)
        self.sleep_batch_size = int(sleep_batch_size)
        self.buffer = ExperienceBuffer(
            capacity=buffer_capacity,
            surprise_weighted=(sampling_mode == "priority"),
            seed=seed,
        )
        self.consolidator = OfflineConsolidator(
            batch_size=sleep_batch_size,
            sampling_mode=sampling_mode,
            seed=seed,
        )
        # Track the most recent sleep round for correlation in records.
        self._last_sleep: ConsolidationResult | None = None

    # ------------------------------------------------------------------ #
    # Override the cycle to add buffer storage + sleep phase
    # ------------------------------------------------------------------ #
    def _cycle(self, frame: np.ndarray, step: int) -> np.ndarray:
        """One online cycle + periodic sleep phase."""
        # 1) PERCEPTION
        obs = self.model.encode_image(frame)
        if not np.all(np.isfinite(obs)):
            self.summary.n_nan_obs += 1
        obs_norm = float(np.linalg.norm(obs)) if np.all(np.isfinite(obs)) else float("nan")

        # 2) COGNITION
        signal = self.model.think(obs)
        if not np.all(np.isfinite(signal.data)):
            self.summary.n_nan_signal += 1

        # 3) ACTION SELECTION (with curiosity)
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

        # 5) STORE EXPERIENCE (for offline replay)
        free_energy = float(engine.compute_free_energy(obs))
        self.buffer.add(
            obs=obs,
            action=action,
            next_obs=next_obs,
            error=free_energy,  # use free energy as the surprise signal
            info_gain=info_gain,
            step=step,
        )

        # 6) PERIODIC SLEEP PHASE
        if (step + 1) % self.sleep_interval == 0 and len(self.buffer) > 0:
            self._run_sleep_phase()

        # 7) RECORD
        last_round = self._last_sleep
        self.summary.records.append(ConsolidationCycleRecord(
            step=step,
            action=action,
            action_vec_norm=float(np.linalg.norm(action_vec)),
            free_energy=free_energy,
            confidence=float(signal.confidence),
            obs_norm=obs_norm,
            frame_mean=float(np.mean(next_frame)),
            rss_kb=0.0,
            beta=beta_before,
            info_gain=info_gain,
            action_bin=action_bin,
            last_sleep_round=(self.summary.n_sleep_phases - 1) if last_round else -1,
            last_sleep_delta_fe=last_round.delta_free_energy if last_round else 0.0,
            last_sleep_replayed=last_round.n_replayed if last_round else 0,
        ))
        return next_frame

    def _run_sleep_phase(self) -> ConsolidationResult:
        """Run one sleep phase: ``sleep_rounds`` consolidation batches."""
        # Run sleep_rounds batches back-to-back. Each batch re-samples
        # from the buffer, so experiences may be replayed multiple
        # times across batches (but not within a batch).
        results = self.consolidator.consolidate_many(
            self.model, self.buffer, n_rounds=self.sleep_rounds
        )
        # The "representative" result for this sleep phase is the last
        # round (which reflects the cumulative effect of all rounds).
        last = results[-1]
        self._last_sleep = last
        self.summary.sleep_rounds.extend(results)
        self.summary.n_sleep_phases += 1
        self.summary.total_replayed += sum(r.n_replayed for r in results)
        # Print a brief log line so the user sees sleep happening.
        print(
            f"    [sleep #{self.summary.n_sleep_phases}] "
            f"replayed {sum(r.n_replayed for r in results)} exps, "
            f"ΔFE={last.delta_free_energy:+.4f} "
            f"(before={last.free_energy_before:.3f}, after={last.free_energy_after:.3f})"
        )
        return last

    # ------------------------------------------------------------------ #
    # Output helpers
    # ------------------------------------------------------------------ #
    def save_csv(self, path: str | Path) -> str:
        """Write per-step + per-sleep-round CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "step", "action", "action_bin", "action_vec_norm",
                "free_energy", "confidence", "obs_norm", "frame_mean",
                "beta", "info_gain",
                "last_sleep_round", "last_sleep_delta_fe", "last_sleep_replayed",
            ])
            for r in self.summary.records:
                w.writerow([
                    r.step, r.action, r.action_bin,
                    f"{r.action_vec_norm:.6f}",
                    f"{r.free_energy:.6f}",
                    f"{r.confidence:.6f}",
                    f"{r.obs_norm:.6f}",
                    f"{r.frame_mean:.6f}",
                    f"{r.beta:.6f}",
                    f"{r.info_gain:.6f}",
                    r.last_sleep_round,
                    f"{r.last_sleep_delta_fe:.6f}",
                    r.last_sleep_replayed,
                ])
        self.summary.csv_path = str(path)
        return self.summary.csv_path

    def save_consolidation_log(self, path: str | Path) -> str:
        """Write per-sleep-round consolidation log."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "sleep_phase", "round", "n_replayed", "mean_replay_error",
                "mean_replay_ig", "free_energy_before", "free_energy_after",
                "delta_free_energy", "elapsed_s",
            ])
            # Group rounds by sleep phase (each phase = sleep_rounds batches).
            rounds_per_phase = self.sleep_rounds
            for i, r in enumerate(self.summary.sleep_rounds):
                phase = i // rounds_per_phase
                round_in_phase = i % rounds_per_phase
                w.writerow([
                    phase, round_in_phase, r.n_replayed,
                    f"{r.mean_replay_error:.6f}",
                    f"{r.mean_replay_ig:.6f}",
                    f"{r.free_energy_before:.6f}",
                    f"{r.free_energy_after:.6f}",
                    f"{r.delta_free_energy:.6f}",
                    f"{r.elapsed_s:.6f}",
                ])
        self.summary.consolidation_log_path = str(path)
        return self.summary.consolidation_log_path

    def save_png(self, path: str | Path) -> str:
        """6-panel PNG: free-energy, beta, action dist, IG, sleep ΔFE, FE before/after."""
        if not _HAS_MPL:
            return ""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        recs = self.summary.records
        steps = [r.step for r in recs]
        fe = [r.free_energy for r in recs]
        betas = [r.beta for r in recs]
        igs = [r.info_gain for r in recs]
        sleep_steps = [r.step for r in recs if r.last_sleep_round >= 0]

        fig, axes = plt.subplots(3, 2, figsize=(12, 12))

        # Panel 1: free energy over time, with sleep phases marked.
        axes[0, 0].plot(steps, fe, color="tab:red", linewidth=0.8, label="online FE")
        for s in sleep_steps:
            axes[0, 0].axvline(s, color="tab:blue", alpha=0.15, linewidth=1.0)
        axes[0, 0].set_title("Online free energy (blue lines = sleep phases)")
        axes[0, 0].set_xlabel("step")
        axes[0, 0].set_ylabel("free energy")
        axes[0, 0].grid(True, alpha=0.3)

        # Panel 2: beta decay.
        axes[0, 1].plot(steps, betas, color="tab:blue", linewidth=0.8)
        axes[0, 1].set_title("Exploration weight β")
        axes[0, 1].set_xlabel("step")
        axes[0, 1].set_ylabel("β")
        axes[0, 1].grid(True, alpha=0.3)

        # Panel 3: action distribution.
        actions = [r.action for r in recs]
        counts = [actions.count(a) for a in range(4)]
        labels = ["L(0)", "R(1)", "Up(2)", "NOOP(3)"]
        bars = axes[1, 0].bar(labels, counts, color="tab:green", alpha=0.7)
        axes[1, 0].set_title("Action distribution")
        axes[1, 0].set_ylabel("count")
        for bar, c in zip(bars, counts, strict=False):
            axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                           str(c), ha="center", va="bottom", fontsize=9)

        # Panel 4: info gain over time.
        axes[1, 1].plot(steps, igs, color="tab:purple", linewidth=0.8)
        axes[1, 1].set_title("Info-gain proxy")
        axes[1, 1].set_xlabel("step")
        axes[1, 1].set_ylabel("IG")
        axes[1, 1].grid(True, alpha=0.3)

        # Panel 5: sleep ΔFE per sleep phase.
        if self.summary.sleep_rounds:
            # Aggregate by phase (mean ΔFE per phase).
            rounds_per_phase = self.sleep_rounds
            n_phases = (len(self.summary.sleep_rounds) + rounds_per_phase - 1) // rounds_per_phase
            phase_deltas = []
            for p in range(n_phases):
                start = p * rounds_per_phase
                end = min(start + rounds_per_phase, len(self.summary.sleep_rounds))
                phase_rounds = self.summary.sleep_rounds[start:end]
                phase_deltas.append(float(np.mean([r.delta_free_energy for r in phase_rounds])))
            axes[2, 0].bar(range(n_phases), phase_deltas, color="tab:orange", alpha=0.7)
            axes[2, 0].set_title("ΔFE per sleep phase (mean of rounds)")
            axes[2, 0].set_xlabel("sleep phase #")
            axes[2, 0].set_ylabel("ΔFE (after - before)")
            axes[2, 0].axhline(0.0, color="black", linewidth=0.5)
            axes[2, 0].grid(True, alpha=0.3)

        # Panel 6: FE before vs after per sleep round.
        if self.summary.sleep_rounds:
            fe_before = [r.free_energy_before for r in self.summary.sleep_rounds]
            fe_after = [r.free_energy_after for r in self.summary.sleep_rounds]
            x = range(len(fe_before))
            axes[2, 1].plot(x, fe_before, color="tab:red", linewidth=0.8, label="before")
            axes[2, 1].plot(x, fe_after, color="tab:green", linewidth=0.8, label="after")
            axes[2, 1].set_title("Free energy: before vs after each sleep round")
            axes[2, 1].set_xlabel("sleep round #")
            axes[2, 1].set_ylabel("free energy")
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(path, dpi=110)
        plt.close(fig)
        self.summary.png_path = str(path)
        return self.summary.png_path


def main() -> None:
    """Run the consolidation-enhanced loop, save CSV + log + PNG."""
    print("=" * 64)
    print("Consolidation-Enhanced Closed Loop (sleep cycles)")
    print("=" * 64)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build model + sandbox.
    model = make_curious_model(
        dim=DEFAULT_DIM,
        seed=DEFAULT_SEED,
        beta_start=1.0,
        beta_min=0.01,
        decay_steps=DEFAULT_MAX_STEPS,  # decay over the full run
        decay_type="linear",
    )
    sandbox = PhysicsSandbox(num_objects=DEFAULT_NUM_OBJECTS, seed=DEFAULT_SEED)
    runner = ConsolidatingSandboxRunner(
        model, sandbox,
        max_steps=DEFAULT_MAX_STEPS,
        sleep_interval=DEFAULT_SLEEP_INTERVAL,
        sleep_rounds=DEFAULT_SLEEP_ROUNDS,
        sleep_batch_size=DEFAULT_SLEEP_BATCH_SIZE,
        buffer_capacity=DEFAULT_BUFFER_CAPACITY,
        sampling_mode=DEFAULT_SAMPLING_MODE,
        seed=DEFAULT_SEED,
    )

    # Run.
    print(f"\nRunning {DEFAULT_MAX_STEPS} steps with sleep every "
          f"{DEFAULT_SLEEP_INTERVAL} steps ({DEFAULT_SLEEP_ROUNDS} rounds/batch, "
          f"batch={DEFAULT_SLEEP_BATCH_SIZE})...")
    runner.run()

    # Populate summary.
    s = runner.summary
    s.n_steps = len(s.records)
    s.actions = [r.action for r in s.records]
    s.betas = [r.beta for r in s.records]
    s.info_gains = [r.info_gain for r in s.records]
    s.free_energies = [r.free_energy for r in s.records]
    if s.sleep_rounds:
        s.mean_sleep_delta_fe = float(np.mean([r.delta_free_energy for r in s.sleep_rounds]))

    # Save outputs.
    csv_path = runner.save_csv(OUTPUT_DIR / "consolidation_run.csv")
    print(f"\nCSV saved: {csv_path}")
    log_path = runner.save_consolidation_log(OUTPUT_DIR / "consolidation_log.csv")
    print(f"Consolidation log saved: {log_path}")
    png_path = runner.save_png(OUTPUT_DIR / "consolidation_run.png")
    if png_path:
        print(f"PNG saved: {png_path}")

    # Summary.
    print("\n--- Summary ---")
    print(f"  online steps: {s.n_steps}  crashes: {s.n_crashes}")
    print(f"  sleep phases: {s.n_sleep_phases}  total replayed: {s.total_replayed}")
    action_counts = [s.actions.count(a) for a in range(4)]
    print(f"  action distribution [L,R,Up,NOOP]: {action_counts}")
    if s.betas:
        print(f"  beta: start={s.betas[0]:.4f}  end={s.betas[-1]:.4f}")
    if s.free_energies:
        n10 = max(1, len(s.free_energies) // 10)
        early = float(np.mean(s.free_energies[:n10]))
        late = float(np.mean(s.free_energies[-n10:]))
        print(f"  online FE: early={early:.4f}  late={late:.4f}  "
              f"delta={late-early:+.4f}")
    if s.sleep_rounds:
        deltas = [r.delta_free_energy for r in s.sleep_rounds]
        n_neg = sum(1 for d in deltas if d < 0.0)
        print(f"  sleep ΔFE: mean={np.mean(deltas):+.4f}  "
              f"improved {n_neg}/{len(deltas)} rounds")
        # Compare online FE right before vs right after each sleep.
        pre_post = []
        for i, r in enumerate(s.records):
            # First step AFTER a sleep phase: this is the step where
            # ``last_sleep_replayed > 0`` (set when a sleep just ran).
            if r.last_sleep_round >= 0 and r.last_sleep_replayed > 0 and i > 0:
                pre_post.append((s.records[i-1].free_energy, r.free_energy))
        if pre_post:
            pre_mean = float(np.mean([p for p, _ in pre_post]))
            post_mean = float(np.mean([a for _, a in pre_post]))
            print(f"  online FE pre-sleep={pre_mean:.4f}  post-sleep={post_mean:.4f}  "
                  f"delta={post_mean-pre_mean:+.4f}")

    print("\n" + "=" * 64)
    print("Done.")
    print("=" * 64)


if __name__ == "__main__":
    main()
