# experiments/run_text_curious.py
"""Curiosity-driven text-reading closed loop (Phase H).

Extends the passive reading loop (``run_text_loop.py``) with ACTION-
DRIVEN NAVIGATION: the model's ``select_action`` output now controls
where the stream reads next, instead of being logged-and-discarded.

Architecture
------------
    TextStream.navigate(action)  ->  str   # action ∈ {0..4}
       ↓
    TextEncoder.encode(text)     ->  obs    # (output_dim,) vector
       ↓
    ZeroDataModel.think(obs)     ->  signal # cognitive cycle, updates belief
       ↓
    ActiveInference.select_action(belief, obs)
        ->  continuous action_vec + _last_selected_idx  (0..4)
       ↓
    stream.navigate(_last_selected_idx)    # FEEDS BACK to the stream
       ↓
    compute_free_energy(obs)     ->  fe
       ↓
    CSV log row + PNG plots (FE / β / action histogram / position heatmap)

Curiosity Mechanism (Phase E, reused from the sandbox loops)
-----------------------------------------------------------
- **Information-gain proxy (Plan A)**: each action maintains a
  sliding window (deque, maxlen=10) of recent pragmatic prediction
  errors. The std of this window is the IG proxy. Cold start = 1.0
  (UCB optimism).
- **β linear decay**: β goes from β_start=1.0 to β_min=0.01 over
  decay_steps=5000 calls to ``select_action``. This shifts the
  agent from exploration (high β → -β·IG dominates) to exploitation
  (low β → pragmatic + epistemic dominate).
- The EFE evaluated per candidate action is:
      EFE_total(a) = pragmatic(a) + homeostatic(a) + epistemic_bonus(a)
                   - β · IG(a)
  Lower EFE = preferred. So high-IG actions are preferred when β is
  high (exploration), and low-pragmatic-error actions are preferred
  when β is low (exploitation).

Action Mapping
--------------
The model has ``num_candidates=5`` (passed to ActiveInferenceEngine).
``select_action`` returns the index of the chosen candidate
(``_last_selected_idx``), which we map 1:1 to the 5 navigation
actions defined in ``text_stream.py``:

    idx  name             effect
    ---  ---------------  --------------------------------------
    0    forward          pos += block_size
    1    backward         pos -= block_size  (clamped at 0)
    2    fast_forward     pos += block_size * 4
    3    fast_backward    pos -= block_size * 4
    4    stay             pos unchanged (re-read block)

This direct mapping means the candidate index IS the navigation
action — no modulo needed.

Run with:
    python experiments/run_text_curious.py --steps 5000
    python experiments/run_text_curious.py --text_file wiki.txt --steps 5000
    python experiments/run_text_curious.py --beta_start 0.0   # disable curiosity (control)
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Reuse the passive-loop text components + ZeroDataModel factory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_sandbox_curious import make_curious_model  # noqa: E402
from text_encoder import TextEncoder  # noqa: E402
from text_stream import (  # noqa: E402
    NAV_ACTION_NAMES,
    NAV_BACKWARD,
    NAV_FAST_BACKWARD,
    NAV_FAST_FORWARD,
    NAV_FORWARD,
    NAV_STAY,
    N_NAV_ACTIONS,
    TextStream,
)
from run_text_loop import _make_preview, ensure_sample_text  # noqa: E402

from zero_data_model.active_inference import ActiveInferenceEngine  # noqa: E402
from zero_data_model.model import ZeroDataModel  # noqa: E402

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:  # pragma: no cover
    _HAS_MPL = False


# --- Configuration ------------------------------------------------- #
DEFAULT_DIM = 32
DEFAULT_BLOCK_SIZE = 128
DEFAULT_MAX_STEPS = 5000
DEFAULT_SEED = 42
PROGRESS_INTERVAL = 100
PREVIEW_CHARS = 20
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "text_curious_run"


# ------------------------------------------------------------------ #
# Model factory: curiosity-enhanced engine with 5 navigation candidates
# ------------------------------------------------------------------ #
def make_text_curious_model(
    dim: int = DEFAULT_DIM,
    seed: int = DEFAULT_SEED,
    beta_start: float = 1.0,
    beta_min: float = 0.01,
    decay_steps: int = DEFAULT_MAX_STEPS,
    decay_type: str = "linear",
    num_candidates: int = N_NAV_ACTIONS,
) -> ZeroDataModel:
    """Construct a ZeroDataModel with a curiosity-enhanced engine
    configured for text navigation.

    Differences from ``run_sandbox_curious.make_curious_model``:
      - ``num_candidates=5`` so ``_last_selected_idx`` ∈ [0, 5) maps
        1:1 to the 5 navigation actions.
      - All other curiosity params inherited from Phase E.
    """
    model = ZeroDataModel(dim=dim, seed=seed)
    # Spawn the same child RNG index (1 = active_inference) so seeded
    # reproducibility is preserved across the model swap.
    child_rngs = model._rng.spawn(7)
    new_engine = ActiveInferenceEngine(
        state_dim=dim,
        obs_dim=dim,
        action_dim=dim // 2,
        rng=child_rngs[1],
        # Phase E: curiosity params.
        exploration_beta_start=beta_start,
        exploration_beta_min=beta_min,
        exploration_decay_steps=decay_steps,
        exploration_decay_type=decay_type,
        # n_action_bins stays at 8 — the IG proxy bins by action
        # ANGLE, decoupled from the candidate count. With 5 candidates
        # sampled around ``mean_action``, all candidates fall in one or
        # two angular bins; the IG proxy then measures the
        # predictability of outcomes from "this direction of action
        # space" — exactly what we want.
        n_action_bins=8,
        action_error_window=10,
        # Phase H: 5 candidates → 5 navigation actions.
        num_candidates=num_candidates,
    )
    model.active_inference = new_engine
    model.modules[1] = new_engine  # active_inference is index 1
    return model


# ------------------------------------------------------------------ #
# Per-step record
# ------------------------------------------------------------------ #
@dataclass
class CuriousTextCycleRecord:
    """Per-step metrics captured during the curiosity-driven text loop."""

    step: int
    nav_action: int               # 0..4 navigation action actually executed
    nav_action_name: str          # human-readable action name
    action_vec_norm: float        # ||action_vec|| — model's signal magnitude
    free_energy: float            # model's free energy this cycle
    prediction_error: float       # |obs - belief| (surprise proxy)
    confidence: float             # integrated signal confidence
    obs_norm: float                # ||obs|| — sanity check
    pos_before: int               # stream pointer BEFORE navigate()
    pos_after: int                # stream pointer AFTER navigate()
    wrapped: bool                 # did the stream wrap this step?
    beta: float                   # current β (curiosity weight)
    info_gain: float              # IG proxy of the chosen action's bin
    text_preview: str             # first PREVIEW_CHARS chars of the block


@dataclass
class CuriousTextRunSummary:
    """Aggregated results from one ``run()`` call."""

    n_steps: int = 0
    n_crashes: int = 0
    crash_messages: list[str] = field(default_factory=list)
    n_nan_obs: int = 0
    n_nan_signal: int = 0
    n_wraps: int = 0
    elapsed_s: float = 0.0
    cycles_per_sec: float = 0.0
    mean_free_energy: float = 0.0
    final_free_energy: float = 0.0
    mean_prediction_error: float = 0.0
    mean_confidence: float = 0.0
    final_beta: float = 0.0
    initial_beta: float = 0.0
    action_counts: list[int] = field(default_factory=lambda: [0] * N_NAV_ACTIONS)
    pos_trajectory: list[int] = field(default_factory=list)
    records: list[CuriousTextCycleRecord] = field(default_factory=list)
    csv_path: str = ""
    png_path: str = ""


# ------------------------------------------------------------------ #
# Runner
# ------------------------------------------------------------------ #
class CuriousTextRunner:
    """Drives the curiosity-driven text-reading closed loop.

    The action selection loop is now CLOSED: ``select_action`` →
    ``_last_selected_idx`` → ``stream.navigate(idx)`` → next observation.
    The model's choice of action directly determines what it reads
    next, and the resulting prediction error feeds back into the
    IG proxy for that action's bin.
    """

    def __init__(
        self,
        model: ZeroDataModel,
        stream: TextStream,
        encoder: TextEncoder,
        max_steps: int = DEFAULT_MAX_STEPS,
    ):
        self.model = model
        self.stream = stream
        self.encoder = encoder
        self.max_steps = max_steps
        self.summary = CuriousTextRunSummary()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self) -> CuriousTextRunSummary:
        """Execute the curiosity-driven loop for ``self.max_steps`` steps."""
        t0 = time.perf_counter()
        # Capture initial β (before any select_action call) for the
        # summary so the β-decay curve can be normalised.
        self.summary.initial_beta = self._current_beta()
        for step in range(self.max_steps):
            try:
                self._cycle(step)
            except Exception as exc:  # noqa: BLE001 — intentional broad catch
                self.summary.n_crashes += 1
                msg = f"step {step}: {type(exc).__name__}: {exc}"
                self.summary.crash_messages.append(msg)
                # Continue — the stream and model are independent
                # enough that one bad step should not abort the run.

            if (step + 1) % PROGRESS_INTERVAL == 0:
                elapsed = time.perf_counter() - t0
                rate = (step + 1) / elapsed
                # Show recent (last 100 steps) FE for trend detection.
                recent_fe = (
                    float(np.mean([r.free_energy for r in self.summary.records[-PROGRESS_INTERVAL:]]))
                    if self.summary.records else 0.0
                )
                last = self.summary.records[-1] if self.summary.records else None
                if last is not None:
                    print(
                        f"  step {step + 1}/{self.max_steps}  "
                        f"crashes={self.summary.n_crashes}  "
                        f"rate={rate:.1f} cyc/s  "
                        f"recent_fe={recent_fe:.3f}  "
                        f"β={last.beta:.3f}  "
                        f"act={last.nav_action_name:13s}  "
                        f"pos={last.pos_after}  "
                        f"peek={last.text_preview[:15]!r}"
                    )
                else:
                    print(
                        f"  step {step + 1}/{self.max_steps}  "
                        f"crashes={self.summary.n_crashes}  "
                        f"rate={rate:.1f} cyc/s  recent_fe={recent_fe:.3f}"
                    )

        elapsed = time.perf_counter() - t0
        self.summary.n_steps = self.max_steps
        self.summary.elapsed_s = elapsed
        self.summary.cycles_per_sec = self.max_steps / elapsed if elapsed > 0 else 0.0
        if self.summary.records:
            fes = [r.free_energy for r in self.summary.records]
            pes = [r.prediction_error for r in self.summary.records]
            confs = [r.confidence for r in self.summary.records]
            self.summary.mean_free_energy = float(np.mean(fes))
            self.summary.final_free_energy = float(fes[-1])
            self.summary.mean_prediction_error = float(np.mean(pes))
            self.summary.mean_confidence = float(np.mean(confs))
            self.summary.final_beta = self.summary.records[-1].beta
            # Aggregate action counts.
            for r in self.summary.records:
                self.summary.action_counts[r.nav_action] += 1
            # Position trajectory for the heatmap.
            self.summary.pos_trajectory = [r.pos_after for r in self.summary.records]
        return self.summary

    # ------------------------------------------------------------------ #
    # One perception-cognition-action cycle (CLOSED LOOP)
    # ------------------------------------------------------------------ #
    def _cycle(self, step: int) -> None:
        """One closed-loop cycle: navigate → encode → think → select."""
        # --- Phase 1: NAVIGATE based on the model's last chosen action.#
        # NOTE: on step 0, the model has not yet produced an action. We
        # bootstrap by reading the first block via ``navigate(NAV_FORWARD)``
        # so the model has SOMETHING to think about. From step 1 on,
        # the action comes from the previous ``select_action`` call.
        engine = self.model.active_inference
        pos_before = self.stream.tell()
        wrapped_before = self.stream.wrapped
        if step == 0:
            nav_action = NAV_FORWARD
            text_block = self.stream.navigate(nav_action)
        else:
            # The model's chosen action is stored on the engine.
            # With num_candidates=5, _last_selected_idx ∈ [0, 5) maps
            # 1:1 to the 5 navigation actions.
            nav_action = int(engine._last_selected_idx)
            # Defensive: if for some reason the index is out of range
            # (shouldn't happen, but catch it), fall back to forward.
            if not 0 <= nav_action < N_NAV_ACTIONS:
                nav_action = NAV_FORWARD
            text_block = self.stream.navigate(nav_action)
        pos_after = self.stream.tell()
        wrapped_this_step = (not wrapped_before) and self.stream.wrapped
        if wrapped_this_step:
            self.summary.n_wraps += 1

        # --- Phase 2: ENCODE the text block into an observation. --- #
        obs = self.encoder.encode(text_block)
        if not np.all(np.isfinite(obs)):
            self.summary.n_nan_obs += 1
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        obs_norm = float(np.linalg.norm(obs))

        # --- Phase 3: COGNITION (think + select_action). --- #
        # ``think`` updates the belief; ``select_action`` evaluates the
        # 5 candidate actions and picks the one minimising EFE. The
        # chosen index is exposed via ``_last_selected_idx`` and will
        # be used in the NEXT cycle's navigate() call.
        signal = self.model.think(obs)
        if not np.all(np.isfinite(signal.data)):
            self.summary.n_nan_signal += 1
        belief = engine.generative_model.belief_state
        action_vec = engine.select_action(belief, current_observation=obs)
        action_vec_norm = float(np.linalg.norm(action_vec))
        # The chosen index (0..4) — this is what we'll navigate to
        # NEXT step. We DON'T use it this step (we already navigated
        # based on the previous step's choice). For step 0 we
        # bootstrapped with NAV_FORWARD above.

        # --- Phase 4: RECORD metrics. --- #
        free_energy = float(engine.compute_free_energy(obs))
        belief_after = engine.generative_model.belief_state
        b = belief_after[:obs.shape[0]]
        prediction_error = float(np.linalg.norm(obs - b))
        # β is captured AFTER select_action has incremented
        # _exploration_step. We want the β that WAS USED for this
        # selection, so we use the value just BEFORE the increment —
        # which is one step earlier. Approximate by reading the
        # current _compute_beta() (one step ahead, but the difference
        # over 5000 steps is negligible: 1/5000 = 0.0002 per step).
        beta = self._current_beta()
        # IG proxy of the chosen action's bin (information-gain value
        # that was used in this step's EFE computation).
        info_gain = self._current_info_gain(action_vec)

        preview = _make_preview(text_block, PREVIEW_CHARS)
        self.summary.records.append(CuriousTextCycleRecord(
            step=step,
            nav_action=nav_action,
            nav_action_name=NAV_ACTION_NAMES[nav_action],
            action_vec_norm=action_vec_norm,
            free_energy=free_energy,
            prediction_error=prediction_error,
            confidence=float(signal.confidence),
            obs_norm=obs_norm,
            pos_before=pos_before,
            pos_after=pos_after,
            wrapped=wrapped_this_step,
            beta=beta,
            info_gain=info_gain,
            text_preview=preview,
        ))

    # ------------------------------------------------------------------ #
    # Helpers: read the engine's current curiosity state.
    # ------------------------------------------------------------------ #
    def _current_beta(self) -> float:
        """Read the engine's current β (post-increment, ~one step ahead).

        ``select_action`` increments ``_exploration_step`` AFTER
        computing β. So calling ``_compute_beta()`` here returns the
        β that will be used on the NEXT call. The discrepancy is
        bounded by 1/decay_steps per step — negligible.
        """
        try:
            return float(self.model.active_inference._compute_beta())
        except Exception:
            return 0.0

    def _current_info_gain(self, action_vec: np.ndarray) -> float:
        """Read the engine's IG proxy for ``action_vec``'s angular bin."""
        try:
            engine = self.model.active_inference
            return float(engine._compute_information_gain_proxy(action_vec))
        except Exception:
            return 0.0


# ------------------------------------------------------------------ #
# Output: CSV + multi-panel PNG
# ------------------------------------------------------------------ #
def save_csv(summary: CuriousTextRunSummary, path: str | Path) -> str:
    """Write the per-step CSV log."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "step", "nav_action", "nav_action_name",
            "action_vec_norm", "free_energy", "prediction_error",
            "confidence", "obs_norm",
            "pos_before", "pos_after", "wrapped",
            "beta", "info_gain", "text_preview",
        ])
        for r in summary.records:
            w.writerow([
                r.step, r.nav_action, r.nav_action_name,
                f"{r.action_vec_norm:.6f}",
                f"{r.free_energy:.6f}",
                f"{r.prediction_error:.6f}",
                f"{r.confidence:.6f}",
                f"{r.obs_norm:.6f}",
                r.pos_before, r.pos_after, int(r.wrapped),
                f"{r.beta:.6f}", f"{r.info_gain:.6f}",
                r.text_preview,
            ])
    summary.csv_path = str(path)
    return summary.csv_path


def plot_curious_run(summary: CuriousTextRunSummary, path: str | Path) -> str:
    """Save a 6-panel PNG: FE, β decay, action histogram, IG, position heatmap, position trace."""
    if not _HAS_MPL or not summary.records:
        return ""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    steps = [r.step for r in summary.records]
    fes = [r.free_energy for r in summary.records]
    pes = [r.prediction_error for r in summary.records]
    betas = [r.beta for r in summary.records]
    igs = [r.info_gain for r in summary.records]
    pos = [r.pos_after for r in summary.records]
    actions = [r.nav_action for r in summary.records]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # Panel 1: free energy + prediction error (twin axis).
    ax = axes[0, 0]
    ax.plot(steps, fes, "-", color="#d62728", linewidth=1, label="free energy")
    ax.plot(steps, pes, "-", color="#1f77b4", linewidth=1, label="prediction error")
    ax.set_xlabel("step")
    ax.set_ylabel("value")
    ax.set_title(
        f"Curiosity-driven reading: FE & PE "
        f"(mean FE = {summary.mean_free_energy:.3f}, "
        f"final FE = {summary.final_free_energy:.3f})"
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Panel 2: β decay curve.
    ax = axes[0, 1]
    ax.plot(steps, betas, "-", color="#2ca02c", linewidth=1.5)
    ax.set_xlabel("step")
    ax.set_ylabel("β (curiosity weight)")
    ax.set_title(
        f"β linear decay: {summary.initial_beta:.3f} -> {summary.final_beta:.3f}"
    )
    ax.grid(True, alpha=0.3)

    # Panel 3: action distribution histogram.
    ax = axes[1, 0]
    counts = summary.action_counts
    total = max(1, sum(counts))
    fractions = [c / total for c in counts]
    x = np.arange(N_NAV_ACTIONS)
    bars = ax.bar(x, fractions, color="#9467bd", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(NAV_ACTION_NAMES, rotation=30, ha="right")
    ax.set_ylabel("fraction of steps")
    ax.set_title("Action distribution")
    ax.set_ylim(0, max(fractions) * 1.2 if fractions else 1.0)
    # Annotate each bar with its count.
    for bar, c in zip(bars, counts, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{c}", ha="center", va="bottom", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 4: information-gain proxy curve.
    ax = axes[1, 1]
    ax.plot(steps, igs, "-", color="#ff7f0e", linewidth=1)
    ax.set_xlabel("step")
    ax.set_ylabel("info-gain proxy (std of bin errors)")
    ax.set_title("Curiosity signal: IG proxy of chosen action")
    ax.grid(True, alpha=0.3)

    # Panel 5: position trajectory (line).
    ax = axes[2, 0]
    ax.plot(steps, pos, "-", color="#17becf", linewidth=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("stream position (chars)")
    ax.set_title("Reading position over time")
    ax.grid(True, alpha=0.3)

    # Panel 6: position × action heatmap (where each action was taken).
    ax = axes[2, 1]
    if pos and actions:
        n_bins = 20
        pos_min, pos_max = min(pos), max(pos)
        if pos_max > pos_min:
            pos_bins = np.linspace(pos_min, pos_max, n_bins + 1)
            # 2D histogram: rows = navigation actions, cols = position bins.
            heatmap = np.zeros((N_NAV_ACTIONS, n_bins))
            for p, a in zip(pos, actions, strict=True):
                bin_idx = min(n_bins - 1, int((p - pos_min) / (pos_max - pos_min) * n_bins))
                heatmap[a, bin_idx] += 1
            im = ax.imshow(heatmap, aspect="auto", cmap="viridis",
                           origin="lower", interpolation="nearest")
            ax.set_yticks(range(N_NAV_ACTIONS))
            ax.set_yticklabels(NAV_ACTION_NAMES)
            ax.set_xlabel("position bin (low → high)")
            ax.set_ylabel("navigation action")
            ax.set_title("Action-position heatmap (which actions go where)")
            fig.colorbar(im, ax=ax, label="count")
        else:
            ax.text(0.5, 0.5, "position is constant — no heatmap",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Action-position heatmap (degenerate)")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    summary.png_path = str(path)
    return summary.png_path


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the curiosity-driven text-reading closed loop.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--text_file", type=str, default="",
                   help="Path to the plain-text file. Empty = use sample.")
    p.add_argument("--steps", type=int, default=DEFAULT_MAX_STEPS)
    p.add_argument("--block_size", type=int, default=DEFAULT_BLOCK_SIZE)
    p.add_argument("--dim", type=int, default=DEFAULT_DIM)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--beta_start", type=float, default=1.0,
                   help="Initial curiosity weight. 0 = no curiosity (control).")
    p.add_argument("--beta_min", type=float, default=0.01)
    p.add_argument("--decay_steps", type=int, default=DEFAULT_MAX_STEPS,
                   help="Linear β decay horizon (steps to reach beta_min).")
    p.add_argument("--decay_type", type=str, default="linear",
                   choices=["linear", "exponential", "stage"])
    p.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.text_file:
        text_path = Path(args.text_file)
        if not text_path.exists():
            raise FileNotFoundError(f"text file not found: {text_path}")
    else:
        text_path = ensure_sample_text()
        print(f"[info] no --text_file given; using sample: {text_path}")

    print("=" * 72)
    print("Curiosity-Driven Text-Reading Loop (Phase H)")
    print("=" * 72)
    print(f"  text_file   = {text_path}")
    print(f"  block_size  = {args.block_size} chars")
    print(f"  steps       = {args.steps}")
    print(f"  dim         = {args.dim}")
    print(f"  seed        = {args.seed}")
    print(f"  β_start/min = {args.beta_start} / {args.beta_min}")
    print(f"  decay       = {args.decay_type} over {args.decay_steps} steps")
    print(f"  candidates  = {N_NAV_ACTIONS} (forward/backward/fast_forward/fast_backward/stay)")
    print(f"  output_dir  = {output_dir}")

    # --- Build model + stream + encoder ------------------------------ #
    model = make_text_curious_model(
        dim=args.dim, seed=args.seed,
        beta_start=args.beta_start, beta_min=args.beta_min,
        decay_steps=args.decay_steps, decay_type=args.decay_type,
        num_candidates=N_NAV_ACTIONS,
    )
    stream = TextStream(text_path, block_size=args.block_size, seed=args.seed)
    encoder = TextEncoder(
        block_size=args.block_size, output_dim=args.dim, seed=args.seed,
    )
    print(f"  stream len  = {len(stream)} chars, "
          f"{stream.n_blocks()} blocks of {args.block_size}")

    # --- Run the loop ------------------------------------------------- #
    runner = CuriousTextRunner(model, stream, encoder, max_steps=args.steps)
    print(f"\nRunning {args.steps} curiosity-driven steps...")
    summary = runner.run()

    # --- Save outputs ------------------------------------------------- #
    csv_path = save_csv(summary, output_dir / "text_curious_run.csv")
    print(f"\nCSV: {csv_path}")
    png_path = plot_curious_run(summary, output_dir / "text_curious_run.png")
    if png_path:
        print(f"PNG: {png_path}")

    # --- Print final summary ------------------------------------------ #
    print("\n" + "=" * 72)
    print("Final summary")
    print("=" * 72)
    print(f"  steps             = {summary.n_steps}")
    print(f"  crashes           = {summary.n_crashes}")
    print(f"  nan_obs           = {summary.n_nan_obs}")
    print(f"  nan_signal        = {summary.n_nan_signal}")
    print(f"  wraps             = {summary.n_wraps}")
    print(f"  elapsed           = {summary.elapsed_s:.1f}s")
    print(f"  rate              = {summary.cycles_per_sec:.1f} cyc/s")
    print(f"  initial β         = {summary.initial_beta:.4f}")
    print(f"  final β           = {summary.final_beta:.4f}")
    print(f"  mean FE           = {summary.mean_free_energy:.4f}")
    print(f"  final FE          = {summary.final_free_energy:.4f}")
    print(f"  mean pred error   = {summary.mean_prediction_error:.4f}")
    print(f"  mean confidence   = {summary.mean_confidence:.4f}")
    print(f"  action counts     = {dict(zip(NAV_ACTION_NAMES, summary.action_counts))}")
    if summary.crash_messages:
        print(f"  crashes (first 3) = {summary.crash_messages[:3]}")

    # Show first 5 + last 5 step log.
    print("\nFirst 5 steps:")
    for r in summary.records[:5]:
        print(f"  step {r.step}: FE={r.free_energy:.3f}  "
              f"β={r.beta:.3f}  act={r.nav_action_name:13s}  "
              f"pos={r.pos_before}->{r.pos_after}  "
              f"peek={r.text_preview[:20]!r}")
    print("Last 5 steps:")
    for r in summary.records[-5:]:
        print(f"  step {r.step}: FE={r.free_energy:.3f}  "
              f"β={r.beta:.3f}  act={r.nav_action_name:13s}  "
              f"pos={r.pos_before}->{r.pos_after}  "
              f"peek={r.text_preview[:20]!r}")
    print("\n" + "=" * 72)
    print("Done.")
    print("=" * 72)


if __name__ == "__main__":
    main()
