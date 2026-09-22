# experiments/run_crossmodal.py
"""Cross-modal training loop (Phase J, Task 3).

Trains a ``MultimodalBridge`` on an aligned (frame, text) dataset by
alternating between two regimes each step:

  - **Physics modality (50%)**: encode the frame, PREDICT the text
    hidden-state via ``W_pt``, compare to the actual text encoding,
    and update the bridge (Hebbian, error-driven).
  - **Text modality (50%)**: encode the text, PREDICT the physics
    hidden-state via ``W_tp``, compare to the actual frame encoding,
    and update the bridge.

The bridge also maintains an internal ``ZeroDataModel`` whose belief
state is updated by feeding it the FUSED hidden vector each step
(``model.think(h_fused)``). This couples the cross-modal alignment to
the model's generative core, so concept formation in either modality
propagates through the shared workspace.

Experience replay consolidation: every ``consolidation_interval`` steps,
sample ``batch_size`` experiences from the buffer and replay them
through ``bridge.update()`` — analogous to offline sleep replay.

Outputs (under ``experiments/output/crossmodal_run/``):
  - crossmodal_log.csv           per-step metrics
  - consolidation_log.csv         per-consolidation metrics
  - bridge.npz                   final bridge parameters
  - summary.json                 aggregated run summary

Run with:
    python experiments/run_crossmodal.py --steps 2000 --seed 42
    python experiments/run_crossmodal.py --dataset experiments/output/aligned_data/aligned_dataset.npz
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from aligned_data_generator import AlignedDataset  # noqa: E402
from experience_buffer import ExperienceBuffer  # noqa: E402
from multimodal_bridge import (  # noqa: E402
    DEFAULT_DIM,
    DEFAULT_LAMBDA,
    DEFAULT_LR,
    MultimodalBridge,
)
from zero_data_model.model import ZeroDataModel  # noqa: E402


# ------------------------------------------------------------------ #
# Defaults
# ------------------------------------------------------------------ #
DEFAULT_STEPS = 2000
DEFAULT_SEED = 42
DEFAULT_DIM = DEFAULT_DIM
DEFAULT_LR = DEFAULT_LR
DEFAULT_LAMBDA = DEFAULT_LAMBDA
DEFAULT_BUFFER_CAPACITY = 5000
DEFAULT_CONSOLIDATION_INTERVAL = 100
DEFAULT_CONSOLIDATION_BATCH_SIZE = 32
DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "output" / "aligned_data" / "aligned_dataset.npz"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "crossmodal_run"
PROGRESS_INTERVAL = 200


# ------------------------------------------------------------------ #
# Per-step + consolidation records
# ------------------------------------------------------------------ #
@dataclass
class CrossmodalStepRecord:
    """One training step's metrics."""
    step: int
    modality: str             # "physics" or "text"
    h_phys_norm: float
    h_text_norm: float
    h_fused_norm: float
    pred_error: float          # MSE of the cross-modal prediction
    bridge_W_pt_norm: float
    bridge_W_tp_norm: float
    belief_norm: float
    free_energy: float


@dataclass
class ConsolidationRecord:
    step: int
    n_replayed: int
    mean_pred_error_before: float
    mean_pred_error_after: float
    delta_pred_error: float
    duration_ms: float


@dataclass
class CrossmodalRunSummary:
    n_steps: int = 0
    n_consolidations: int = 0
    n_replayed_experiences: int = 0
    mean_pred_error: float = 0.0
    final_pred_error: float = 0.0
    mean_belief_norm: float = 0.0
    mean_free_energy: float = 0.0
    initial_W_pt_norm: float = 0.0
    final_W_pt_norm: float = 0.0
    initial_W_tp_norm: float = 0.0
    final_W_tp_norm: float = 0.0
    consolidation_records: list = field(default_factory=list)


# ------------------------------------------------------------------ #
# Runner
# ------------------------------------------------------------------ #
class CrossmodalRunner:
    """Cross-modal training loop with experience replay."""

    def __init__(
        self,
        bridge: MultimodalBridge,
        model: ZeroDataModel,
        dataset: AlignedDataset,
        max_steps: int = DEFAULT_STEPS,
        buffer_capacity: int = DEFAULT_BUFFER_CAPACITY,
        consolidation_interval: int = DEFAULT_CONSOLIDATION_INTERVAL,
        consolidation_batch_size: int = DEFAULT_CONSOLIDATION_BATCH_SIZE,
        lr: float = DEFAULT_LR,
        lam: float = DEFAULT_LAMBDA,
        seed: int = DEFAULT_SEED,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
    ):
        self.bridge = bridge
        self.model = model
        self.dataset = dataset
        self.max_steps = int(max_steps)
        self.consolidation_interval = int(consolidation_interval)
        self.consolidation_batch_size = int(consolidation_batch_size)
        self.lr = float(lr)
        self.lam = float(lam)
        self.seed = int(seed)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Experience buffer stores (h_phys, h_text) pairs as obs/next_obs.
        self.buffer = ExperienceBuffer(
            capacity=buffer_capacity, alpha=0.6, beta=0.4, seed=seed,
        )
        self._rng = np.random.default_rng(seed)
        self.records: list[CrossmodalStepRecord] = []
        self.consolidation_records: list[ConsolidationRecord] = []
        self.summary = CrossmodalRunSummary()

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self) -> CrossmodalRunSummary:
        t0 = time.perf_counter()
        print("=" * 72)
        print("Cross-Modal Training Loop")
        print("=" * 72)
        print(f"  steps                  = {self.max_steps}")
        print(f"  dim                    = {self.bridge.dim}")
        print(f"  seed                   = {self.seed}")
        print(f"  dataset size           = {len(self.dataset)}")
        print(f"  consolidation_interval = {self.consolidation_interval}")
        print(f"  consolidation_batch    = {self.consolidation_batch_size}")
        print(f"  lr                     = {self.lr}")
        print(f"  lambda (weight decay)  = {self.lam}")
        print()

        self.summary.initial_W_pt_norm = float(np.linalg.norm(self.bridge.W_pt))
        self.summary.initial_W_tp_norm = float(np.linalg.norm(self.bridge.W_tp))

        for step in range(self.max_steps):
            record = self._cycle(step)
            self.records.append(record)

            # Periodic consolidation.
            if (step + 1) % self.consolidation_interval == 0 and len(self.buffer) > 0:
                cons = self._consolidate(step + 1)
                self.consolidation_records.append(cons)
                self.summary.n_consolidations += 1

            if (step + 1) % PROGRESS_INTERVAL == 0:
                self._print_progress(step + 1, t0)

        self._finalise_summary(t0)
        self._save_outputs()
        return self.summary

    # ------------------------------------------------------------------ #
    # Single training cycle
    # ------------------------------------------------------------------ #
    def _cycle(self, step: int) -> CrossmodalStepRecord:
        # Sample a random (frame, text) pair from the dataset.
        idx = int(self._rng.integers(0, len(self.dataset)))
        frame = self.dataset.frames[idx]
        text = self.dataset.texts[idx]

        # Encode both modalities.
        h_phys = self.bridge.encode_physics(frame)
        h_text = self.bridge.encode_text(text)

        # Randomly pick which modality drives the prediction this step.
        # 50/50: physics->text or text->physics.
        if self._rng.random() < 0.5:
            modality = "physics"
            h_pred = self.bridge.predict_text_from_physics(h_phys)
            pred_error = self.bridge.prediction_error(h_text, h_pred)
        else:
            modality = "text"
            h_pred = self.bridge.predict_physics_from_text(h_text)
            pred_error = self.bridge.prediction_error(h_phys, h_pred)

        # Hebbian update (uses BOTH modalities regardless of which drove
        # the prediction — co-occurrence is the alignment signal).
        self.bridge.update(h_phys, h_text, lr=self.lr, lam=self.lam)

        # Fuse and feed to ZeroDataModel's generative core.
        h_fused = self.bridge.fuse(h_phys, h_text)
        signal = self.model.think(h_fused)

        # Belief decay (same Task G fix as run_text_replay.py): emulate
        # the missing KL gradient in update_belief.
        BELIEF_DECAY = 0.99
        gm = self.model.active_inference.generative_model
        gm.belief_state = gm.belief_state * BELIEF_DECAY

        # Pragmatic free energy: ||h_fused - belief @ emission||^2 / dim.
        belief = gm.belief_state
        predicted_obs = gm.predict_observation(belief)
        err = h_fused[:len(predicted_obs)] - predicted_obs[:len(h_fused)]
        if len(err) < gm.obs_dim:
            err = np.pad(err, (0, gm.obs_dim - len(err)))
        free_energy = float(np.dot(err, err)) / err.size

        # Store the experience for replay. We store (h_phys, h_text) as
        # (obs, next_obs) — they are the two aligned modalities, and the
        # consolidator re-runs bridge.update() on them.
        self.buffer.add(
            obs=h_phys, action=0, next_obs=h_text,
            error=pred_error, step=step,
            metadata={"text": text, "modality": modality},
        )

        return CrossmodalStepRecord(
            step=step,
            modality=modality,
            h_phys_norm=float(np.linalg.norm(h_phys)),
            h_text_norm=float(np.linalg.norm(h_text)),
            h_fused_norm=float(np.linalg.norm(h_fused)),
            pred_error=pred_error,
            bridge_W_pt_norm=float(np.linalg.norm(self.bridge.W_pt)),
            bridge_W_tp_norm=float(np.linalg.norm(self.bridge.W_tp)),
            belief_norm=float(np.linalg.norm(belief)),
            free_energy=free_energy,
        )

    # ------------------------------------------------------------------ #
    # Consolidation (offline replay)
    # ------------------------------------------------------------------ #
    def _consolidate(self, step: int) -> ConsolidationRecord:
        """Sample a batch from the buffer and replay through bridge.update()."""
        t0 = time.perf_counter()

        batch = self.buffer.sample(
            self.consolidation_batch_size, mode="priority"
        )
        if len(batch) == 0:
            return ConsolidationRecord(
                step=step, n_replayed=0,
                mean_pred_error_before=0.0, mean_pred_error_after=0.0,
                delta_pred_error=0.0, duration_ms=0.0,
            )

        errors_before: list[float] = []
        errors_after: list[float] = []
        for exp in batch.experiences:
            h_phys = exp.obs          # stored as obs
            h_text = exp.next_obs      # stored as next_obs
            # Error BEFORE update.
            h_text_pred = self.bridge.predict_text_from_physics(h_phys)
            err_before = self.bridge.prediction_error(h_text, h_text_pred)
            errors_before.append(err_before)
            # Update.
            self.bridge.update(h_phys, h_text, lr=self.lr, lam=self.lam)
            # Error AFTER update.
            h_text_pred_after = self.bridge.predict_text_from_physics(h_phys)
            err_after = self.bridge.prediction_error(h_text, h_text_pred_after)
            errors_after.append(err_after)
            self.summary.n_replayed_experiences += 1

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        mean_before = float(np.mean(errors_before)) if errors_before else 0.0
        mean_after = float(np.mean(errors_after)) if errors_after else 0.0
        return ConsolidationRecord(
            step=step,
            n_replayed=len(batch.experiences),
            mean_pred_error_before=mean_before,
            mean_pred_error_after=mean_after,
            delta_pred_error=mean_after - mean_before,
            duration_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------ #
    # Diagnostics + I/O
    # ------------------------------------------------------------------ #
    def _print_progress(self, step: int, t0: float) -> None:
        recent = self.records[-PROGRESS_INTERVAL:]
        mean_err = float(np.mean([r.pred_error for r in recent]))
        mean_belief = float(np.mean([r.belief_norm for r in recent]))
        elapsed = time.perf_counter() - t0
        print(
            f"  step {step:5d}/{self.max_steps}  "
            f"mean_err={mean_err:.6f}  "
            f"belief_norm={mean_belief:.4f}  "
            f"W_pt={float(np.linalg.norm(self.bridge.W_pt)):.4f}  "
            f"W_tp={float(np.linalg.norm(self.bridge.W_tp)):.4f}  "
            f"rate={step / max(elapsed, 1e-9):.1f} cyc/s"
        )

    def _finalise_summary(self, t0: float) -> None:
        s = self.summary
        s.n_steps = len(self.records)
        if self.records:
            errs = [r.pred_error for r in self.records]
            beliefs = [r.belief_norm for r in self.records]
            fes = [r.free_energy for r in self.records]
            s.mean_pred_error = float(np.mean(errs))
            s.final_pred_error = float(errs[-1])
            s.mean_belief_norm = float(np.mean(beliefs))
            s.mean_free_energy = float(np.mean(fes))
        s.final_W_pt_norm = float(np.linalg.norm(self.bridge.W_pt))
        s.final_W_tp_norm = float(np.linalg.norm(self.bridge.W_tp))
        s.consolidation_records = [asdict(c) for c in self.consolidation_records]
        elapsed = time.perf_counter() - t0
        print()
        print("=" * 72)
        print("Final summary")
        print("=" * 72)
        print(f"  steps                   = {s.n_steps}")
        print(f"  consolidations          = {s.n_consolidations}")
        print(f"  experiences replayed     = {s.n_replayed_experiences}")
        print(f"  mean pred error         = {s.mean_pred_error:.6f}")
        print(f"  final pred error        = {s.final_pred_error:.6f}")
        print(f"  mean belief norm        = {s.mean_belief_norm:.4f}")
        print(f"  mean free energy        = {s.mean_free_energy:.6f}")
        print(f"  W_pt norm  {s.initial_W_pt_norm:.4f} -> {s.final_W_pt_norm:.4f}")
        print(f"  W_tp norm  {s.initial_W_tp_norm:.4f} -> {s.final_W_tp_norm:.4f}")
        print(f"  elapsed                 = {elapsed:.1f}s")
        if self.consolidation_records:
            mean_delta = float(np.mean([
                c.delta_pred_error for c in self.consolidation_records
            ]))
            print(f"  mean consolidation Δerr = {mean_delta:.6f}")

    def _save_outputs(self) -> None:
        # CSV log.
        csv_path = self.output_dir / "crossmodal_log.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "step", "modality", "h_phys_norm", "h_text_norm",
                "h_fused_norm", "pred_error", "W_pt_norm", "W_tp_norm",
                "belief_norm", "free_energy",
            ])
            for r in self.records:
                w.writerow([
                    r.step, r.modality,
                    f"{r.h_phys_norm:.6f}", f"{r.h_text_norm:.6f}",
                    f"{r.h_fused_norm:.6f}", f"{r.pred_error:.6f}",
                    f"{r.bridge_W_pt_norm:.6f}", f"{r.bridge_W_tp_norm:.6f}",
                    f"{r.belief_norm:.6f}", f"{r.free_energy:.6f}",
                ])

        # Consolidation log.
        cons_path = self.output_dir / "consolidation_log.csv"
        with cons_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "step", "n_replayed", "mean_err_before", "mean_err_after",
                "delta_err", "duration_ms",
            ])
            for c in self.consolidation_records:
                w.writerow([
                    c.step, c.n_replayed,
                    f"{c.mean_pred_error_before:.6f}",
                    f"{c.mean_pred_error_after:.6f}",
                    f"{c.delta_pred_error:.6f}",
                    f"{c.duration_ms:.2f}",
                ])

        # Bridge parameters.
        bridge_path = self.output_dir / "bridge.npz"
        self.bridge.save(bridge_path)

        # Summary JSON.
        summary_path = self.output_dir / "summary.json"
        with summary_path.open("w") as f:
            json.dump({
                "n_steps": self.summary.n_steps,
                "n_consolidations": self.summary.n_consolidations,
                "n_replayed_experiences": self.summary.n_replayed_experiences,
                "mean_pred_error": self.summary.mean_pred_error,
                "final_pred_error": self.summary.final_pred_error,
                "mean_belief_norm": self.summary.mean_belief_norm,
                "mean_free_energy": self.summary.mean_free_energy,
                "initial_W_pt_norm": self.summary.initial_W_pt_norm,
                "final_W_pt_norm": self.summary.final_W_pt_norm,
                "initial_W_tp_norm": self.summary.initial_W_tp_norm,
                "final_W_tp_norm": self.summary.final_W_tp_norm,
                "dim": self.bridge.dim,
                "seed": self.seed,
                "lr": self.lr,
                "lambda": self.lam,
                "dataset_size": len(self.dataset),
                "consolidation_records": self.summary.consolidation_records,
            }, f, indent=2)

        print(f"\nSaved:")
        print(f"  {csv_path}")
        print(f"  {cons_path_path_str(cons_path)}")
        print(f"  {bridge_path}")
        print(f"  {summary_path}")


def cons_path_path_str(p: Path) -> str:
    return str(p)


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run cross-modal training on an aligned dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET_PATH))
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--dim", type=int, default=DEFAULT_DIM)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--lambda_", type=float, default=DEFAULT_LAMBDA)
    p.add_argument("--buffer_capacity", type=int, default=DEFAULT_BUFFER_CAPACITY)
    p.add_argument("--consolidation_interval", type=int, default=DEFAULT_CONSOLIDATION_INTERVAL)
    p.add_argument("--consolidation_batch_size", type=int, default=DEFAULT_CONSOLIDATION_BATCH_SIZE)
    p.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset = AlignedDataset.load(args.dataset)
    print(f"Loaded dataset: {len(dataset)} samples from {args.dataset}")

    bridge = MultimodalBridge(
        dim=args.dim, lr=args.lr, lam=args.lambda_, seed=args.seed,
    )
    model = ZeroDataModel(dim=args.dim, seed=args.seed)
    runner = CrossmodalRunner(
        bridge=bridge, model=model, dataset=dataset,
        max_steps=args.steps,
        buffer_capacity=args.buffer_capacity,
        consolidation_interval=args.consolidation_interval,
        consolidation_batch_size=args.consolidation_batch_size,
        lr=args.lr, lam=args.lambda_, seed=args.seed,
        output_dir=Path(args.output_dir),
    )
    runner.run()


if __name__ == "__main__":
    main()
