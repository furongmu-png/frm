# experiments/run_sandbox_curious.py
"""Curiosity-enhanced closed loop: ZeroDataModel (Phase E) <-> PhysicsSandbox.

Extends ``run_sandbox_closed_loop.SandboxRunner`` with per-step logging of:
  - the exploration weight beta (decaying over time)
  - the information-gain proxy of the chosen action
  - the angular bin the chosen action fell into

Outputs a CSV log and (optionally) a PNG with 4 panels:
  free-energy, beta, action distribution, and info-gain over time.

Run with:  python experiments/run_sandbox_curious.py
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Reuse the base runner's structure + discretiser.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from physics_sandbox import PhysicsSandbox  # noqa: E402
from run_sandbox_closed_loop import CycleRecord, SandboxRunner  # noqa: E402

from zero_data_model.model import ZeroDataModel  # noqa: E402

# Optional matplotlib (only needed for the PNG output).
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
# Curiosity defaults match ``ActiveInferenceEngine.__init__`` so a fresh
# model without explicit params uses the documented Phase-E defaults.
DEFAULT_BETA_START = 1.0
DEFAULT_BETA_MIN = 0.01
DEFAULT_DECAY_STEPS = 5000
DEFAULT_DECAY_TYPE = "linear"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


@dataclass
class CuriousCycleRecord(CycleRecord):
    """Per-step record augmented with curiosity metrics."""

    beta: float = 0.0           # exploration weight this step
    info_gain: float = 0.0      # IG proxy of the chosen action
    action_bin: int = 0         # angular bin of the chosen action


@dataclass
class CuriousRunSummary:
    """Aggregated curiosity-loop results."""

    n_steps: int = 0
    n_crashes: int = 0
    crash_messages: list[str] = field(default_factory=list)
    n_nan_obs: int = 0
    n_nan_signal: int = 0
    actions: list[int] = field(default_factory=list)
    elapsed_s: float = 0.0
    cycles_per_sec: float = 0.0
    betas: list[float] = field(default_factory=list)
    info_gains: list[float] = field(default_factory=list)
    free_energies: list[float] = field(default_factory=list)
    records: list[CuriousCycleRecord] = field(default_factory=list)
    csv_path: str = ""
    png_path: str = ""


class CuriousSandboxRunner(SandboxRunner):
    """Closed-loop runner with curiosity (Phase E) logging.

    Subclasses ``SandboxRunner`` to reuse the crash-recovery + memory
    tracking, but overrides ``_cycle`` to capture beta / info_gain /
    action_bin alongside the base metrics, and adds ``save_csv`` /
    ``save_png`` helpers for offline analysis.
    """

    def __init__(
        self,
        model: ZeroDataModel,
        sandbox: PhysicsSandbox,
        max_steps: int = DEFAULT_MAX_STEPS,
    ):
        super().__init__(model, sandbox, max_steps)
        # Replace the base summary with the curious variant so callers
        # get the extra fields for free.
        self.summary = CuriousRunSummary()

    def _cycle(self, frame: np.ndarray, step: int) -> np.ndarray:
        """One perception-cognition-action cycle with curiosity logging."""
        # 1) PERCEPTION
        obs = self.model.encode_image(frame)
        if not np.all(np.isfinite(obs)):
            self.summary.n_nan_obs += 1
        obs_norm = float(np.linalg.norm(obs)) if np.all(np.isfinite(obs)) else float("nan")

        # 2) COGNITION
        signal = self.model.think(obs)
        if not np.all(np.isfinite(signal.data)):
            self.summary.n_nan_signal += 1

        # 3) ACTION SELECTION — capture beta BEFORE the call (the call
        #    increments _exploration_step, so reading after would give
        #    the NEXT step's beta). We read the engine's beta directly
        #    via _compute_beta() which is a pure function of the current
        #    _exploration_step.
        engine = self.model.active_inference
        beta_before = engine._compute_beta()
        belief = engine.generative_model.belief_state
        action_vec = engine.select_action(belief, current_observation=obs)
        # The IG proxy for the chosen action — computed against the
        # action_vec returned (which is the chosen candidate). Note:
        # this reads the history AFTER the chosen action's error was
        # appended in select_action, so for bins with >=3 samples the
        # std includes the just-recorded sample. This is the "posterior"
        # IG (how unpredictable this action was, given the latest
        # outcome) — a reasonable proxy for the curiosity signal the
        # agent just experienced.
        info_gain = engine._compute_information_gain_proxy(action_vec)
        action_bin = engine._action_bin(action_vec)
        action = self._discretize_action(action_vec)

        # 4) ENVIRONMENT
        next_frame = self.sandbox.step(action)

        # 5) RECORD
        free_energy = float(engine.compute_free_energy(obs))
        self.summary.records.append(CuriousCycleRecord(
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
        ))
        return next_frame

    # ------------------------------------------------------------------ #
    # Output helpers
    # ------------------------------------------------------------------ #
    def save_csv(self, path: str | Path) -> str:
        """Write per-step records to a CSV file. Returns the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "step", "action", "action_bin", "action_vec_norm",
                "free_energy", "confidence", "obs_norm", "frame_mean",
                "beta", "info_gain",
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
                ])
        self.summary.csv_path = str(path)
        return self.summary.csv_path

    def save_png(self, path: str | Path) -> str:
        """Render a 4-panel summary PNG. Returns the path (or '' if mpl missing)."""
        if not _HAS_MPL:
            return ""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        steps = [r.step for r in self.summary.records]
        fe = [r.free_energy for r in self.summary.records]
        betas = [r.beta for r in self.summary.records]
        igs = [r.info_gain for r in self.summary.records]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Panel 1: free energy over time (should trend down as the model
        # learns to predict the sandbox dynamics).
        axes[0, 0].plot(steps, fe, color="tab:red", linewidth=0.8)
        axes[0, 0].set_title("Free Energy (pragmatic prediction error)")
        axes[0, 0].set_xlabel("step")
        axes[0, 0].set_ylabel("EFE pragmatic term")
        axes[0, 0].grid(True, alpha=0.3)

        # Panel 2: beta decay (should decrease from beta_start to beta_min).
        axes[0, 1].plot(steps, betas, color="tab:blue", linewidth=0.8)
        axes[0, 1].set_title("Exploration weight β (decay schedule)")
        axes[0, 1].set_xlabel("step")
        axes[0, 1].set_ylabel("β")
        axes[0, 1].grid(True, alpha=0.3)

        # Panel 3: action distribution (histogram of discretised actions).
        actions = [r.action for r in self.summary.records]
        counts = [actions.count(a) for a in range(4)]
        labels = ["L(0)", "R(1)", "Up(2)", "NOOP(3)"]
        bars = axes[1, 0].bar(labels, counts, color="tab:green", alpha=0.7)
        axes[1, 0].set_title("Discretised action distribution")
        axes[1, 0].set_ylabel("count")
        for bar, c in zip(bars, counts, strict=False):
            axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                           str(c), ha="center", va="bottom", fontsize=9)

        # Panel 4: info-gain proxy over time (should decline as bins
        # accumulate samples and the agent's predictions stabilise).
        axes[1, 1].plot(steps, igs, color="tab:purple", linewidth=0.8)
        axes[1, 1].set_title("Info-gain proxy (chosen action)")
        axes[1, 1].set_xlabel("step")
        axes[1, 1].set_ylabel("IG = std(prediction errors in bin)")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(path, dpi=110)
        plt.close(fig)
        self.summary.png_path = str(path)
        return self.summary.png_path


def make_curious_model(
    dim: int = DEFAULT_DIM,
    seed: int = DEFAULT_SEED,
    beta_start: float = DEFAULT_BETA_START,
    beta_min: float = DEFAULT_BETA_MIN,
    decay_steps: int = DEFAULT_DECAY_STEPS,
    decay_type: str = DEFAULT_DECAY_TYPE,
) -> ZeroDataModel:
    """Construct a ZeroDataModel with a curiosity-enhanced ActiveInferenceEngine.

    The engine is constructed with the Phase-E exploration params and
    then injected into the model in place of the default engine. This
    keeps the model's other 5 modules unchanged.
    """
    from zero_data_model.active_inference import ActiveInferenceEngine

    model = ZeroDataModel(dim=dim, seed=seed)
    # Replace the default engine with a curiosity-enabled one using
    # the SAME child rng (index 1) so seeded reproducibility is preserved.
    rng = model._rng  # the parent rng — but we need the child used by active_inference
    # Simpler: re-spawn the children the same way __init__ does and grab index 1.
    child_rngs = rng.spawn(7)
    new_engine = ActiveInferenceEngine(
        state_dim=dim,
        obs_dim=dim,
        action_dim=dim // 2,
        rng=child_rngs[1],
        exploration_beta_start=beta_start,
        exploration_beta_min=beta_min,
        exploration_decay_steps=decay_steps,
        exploration_decay_type=decay_type,
    )
    # Swap in the new engine and update the modules list.
    model.active_inference = new_engine
    model.modules[1] = new_engine  # active_inference is index 1 in modules
    return model


def main() -> None:
    """Run the curiosity-enhanced loop, save CSV + PNG, print summary."""
    print("=" * 64)
    print("Curiosity-Enhanced Closed Loop (Phase E)")
    print("=" * 64)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Build model + sandbox --------------------------------------- #
    model = make_curious_model(
        dim=DEFAULT_DIM,
        seed=DEFAULT_SEED,
        beta_start=DEFAULT_BETA_START,
        beta_min=DEFAULT_BETA_MIN,
        decay_steps=DEFAULT_DECAY_STEPS,
        decay_type=DEFAULT_DECAY_TYPE,
    )
    sandbox = PhysicsSandbox(num_objects=DEFAULT_NUM_OBJECTS, seed=DEFAULT_SEED)
    runner = CuriousSandboxRunner(model, sandbox, max_steps=DEFAULT_MAX_STEPS)

    # --- Run ---------------------------------------------------------- #
    print(f"\nRunning {DEFAULT_MAX_STEPS} steps "
          f"(beta_start={DEFAULT_BETA_START}, decay={DEFAULT_DECAY_TYPE}, "
          f"T={DEFAULT_DECAY_STEPS})...")
    runner.run()

    # --- Populate summary aggregates --------------------------------- #
    s = runner.summary
    s.n_steps = len(s.records)
    s.actions = [r.action for r in s.records]
    s.betas = [r.beta for r in s.records]
    s.info_gains = [r.info_gain for r in s.records]
    s.free_energies = [r.free_energy for r in s.records]

    # --- Save CSV + PNG ---------------------------------------------- #
    csv_path = runner.save_csv(OUTPUT_DIR / "curious_run.csv")
    print(f"\nCSV saved: {csv_path}")
    png_path = runner.save_png(OUTPUT_DIR / "curious_run.png")
    if png_path:
        print(f"PNG saved: {png_path}")
    else:
        print("PNG skipped (matplotlib not available)")

    # --- Summary ----------------------------------------------------- #
    print("\n--- Summary ---")
    print(f"  steps: {s.n_steps}  crashes: {s.n_crashes}  "
          f"nan_obs: {s.n_nan_obs}  nan_signal: {s.n_nan_signal}")
    action_counts = [s.actions.count(a) for a in range(4)]
    print(f"  action distribution [L,R,Up,NOOP]: {action_counts}")
    if s.betas:
        print(f"  beta: start={s.betas[0]:.4f}  end={s.betas[-1]:.4f}  "
              f"min={min(s.betas):.4f}  max={max(s.betas):.4f}")
    if s.info_gains:
        print(f"  info_gain: start={s.info_gains[0]:.4f}  end={s.info_gains[-1]:.4f}  "
              f"mean={np.mean(s.info_gains):.4f}")
    if s.free_energies:
        # Split into first and last 10% to show the trend.
        n10 = max(1, len(s.free_energies) // 10)
        early = float(np.mean(s.free_energies[:n10]))
        late = float(np.mean(s.free_energies[-n10:]))
        delta = late - early
        print(f"  free_energy: early={early:.4f}  late={late:.4f}  "
              f"delta={delta:+.4f} ({'down ✓' if delta < 0 else 'up ✗'})")

    print("\n" + "=" * 64)
    print("Done. Inspect the CSV/PNG for detailed trends.")
    print("=" * 64)


if __name__ == "__main__":
    main()
