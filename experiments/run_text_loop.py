# experiments/run_text_loop.py
"""Closed-loop text-reading pipeline: ZeroDataModel <-> TextStream.

Phase H baseline: PASSIVE sequential reading. The model is fed a
stream of fixed-length character blocks from a plain-text file, with
no action influencing what is read next. The loop records free
energy, prediction error, confidence, and the first 20 chars of each
block for offline analysis.

The action channel is RESERVED for future curiosity-driven reading
(Phase H+): the model's ``select_action`` output is logged but not yet
fed back into the streamer. When Phase H+ lands, the discretised
action will map to stream-control operations (seek backward, seek
forward, re-read, skip).

Architecture:
    TextStream.step() -> str          # next block_size chars
       |
    TextEncoder.encode(text) -> obs   # (output_dim,) numpy vector
       |
    ZeroDataModel.think(obs) -> signal# cognitive cycle, updates belief
       |
    ActiveInference.compute_free_energy(obs) -> fe
       |
    CSV log row: step, action, action_vec_norm, free_energy, confidence,
                 obs_norm, text_preview

Run with:
    python experiments/run_text_loop.py --text_file path/to/wiki.txt --steps 5000
    python experiments/run_text_loop.py --steps 1000   # uses bundled sample
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Reuse the curiosity-enhanced model factory from Phase E so the
# text loop inherits the same exploration/curiosity defaults as the
# sandbox loops. This keeps the two sensory modalities comparable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from run_sandbox_curious import make_curious_model  # noqa: E402
from text_encoder import TextEncoder  # noqa: E402
from text_stream import TextStream  # noqa: E402

from zero_data_model.model import ZeroDataModel  # noqa: E402

# Optional matplotlib — only needed for the PNG FE curve.
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_MPL = False


# --- Configuration ------------------------------------------------- #
DEFAULT_DIM = 32              # smaller dim than the sandbox loop (64)
                              # because text observations are lower-
                              # entropy than pixel observations.
DEFAULT_BLOCK_SIZE = 128     # chars per block
DEFAULT_MAX_STEPS = 5000
DEFAULT_SEED = 42
PROGRESS_INTERVAL = 100      # print summary every N steps
PREVIEW_CHARS = 20           # how many chars of each block to log
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "text_run"

# Path to a bundled sample text file (used when --text_file is not
# given). We don't ship a Wiki dump with the repo, so the sample is a
# short essay on cognitive emergence synthesised at runtime.
SAMPLE_TEXT_PATH = Path(__file__).resolve().parent / "output" / "_sample_text.txt"


# ------------------------------------------------------------------ #
# Per-step record
# ------------------------------------------------------------------ #
@dataclass
class TextCycleRecord:
    """Per-step metrics captured during the text loop."""

    step: int
    action_discrete: int        # discretised action (placeholder; not used yet)
    action_vec_norm: float      # ||action_vec|| — model's signal magnitude
    free_energy: float          # model's free energy this cycle
    prediction_error: float     # |obs - belief| (proxy for surprise)
    confidence: float           # integrated signal confidence
    obs_norm: float              # ||obs|| — sanity check
    pos_before: int             # stream pointer before step()
    pos_after: int              # stream pointer after step()
    wrapped: bool               # did the stream wrap this step?
    text_preview: str           # first PREVIEW_CHARS chars of the block


@dataclass
class TextRunSummary:
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
    records: list[TextCycleRecord] = field(default_factory=list)
    csv_path: str = ""
    png_path: str = ""


# ------------------------------------------------------------------ #
# Runner
# ------------------------------------------------------------------ #
class TextRunner:
    """Drives the ZeroDataModel <-> TextStream closed loop.

    The runner is intentionally simple: it does NOT use the model's
    action output to control the stream. The action is recorded for
    offline analysis so we can later correlate "what the model wanted
    to do" with "what it was reading" — the foundation for Phase H+
    curiosity-driven reading.
    """

    # KL regularisation strength: after each think() cycle, pull
    # belief_state toward the N(0, I) prior by this factor. This
    # implements the KL gradient -d(0.5*||b||^2)/db = -b that
    # ``update_belief`` omits (it only does the pragmatic gradient).
    # Without this pull, ||belief|| drifts unboundedly across cycles
    # because each block's observation pushes belief in a different
    # direction with no restoring force. 0.99 = 1% pull per step.
    BELIEF_DECAY = 0.99

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
        self.summary = TextRunSummary()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self) -> TextRunSummary:
        """Execute the text-reading loop for ``self.max_steps`` steps.

        Resilient: a single step's exception is caught and recorded as
        a crash, and the loop continues. The stream is NOT rewound on
        a crash (the model's belief state may be corrupted, but the
        stream position is independent — we want to see if the model
        recovers on the next block).
        """
        t0 = time.perf_counter()
        for step in range(self.max_steps):
            try:
                self._cycle(step)
            except Exception as exc:  # noqa: BLE001 — intentional broad catch
                self.summary.n_crashes += 1
                msg = f"step {step}: {type(exc).__name__}: {exc}"
                self.summary.crash_messages.append(msg)
                # Continue to the next step — do NOT rewind the stream.

            if (step + 1) % PROGRESS_INTERVAL == 0:
                elapsed = time.perf_counter() - t0
                rate = (step + 1) / elapsed
                recent_fe = np.mean(
                    [r.free_energy for r in self.summary.records[-PROGRESS_INTERVAL:]]
                ) if self.summary.records else 0.0
                print(
                    f"  step {step + 1}/{self.max_steps}  "
                    f"crashes={self.summary.n_crashes}  "
                    f"rate={rate:.1f} cyc/s  "
                    f"recent_fe={recent_fe:.4f}"
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
        return self.summary

    # ------------------------------------------------------------------ #
    # One perception-cognition-action cycle
    # ------------------------------------------------------------------ #
    def _cycle(self, step: int) -> None:
        """Run one full cycle: read block -> encode -> think -> log."""
        # 1) READ: get the next block from the stream. ``step()`` is
        #    the ONLY mutation to the stream state. We snapshot ``pos``
        #    before and after for the log.
        pos_before = self.stream.tell()
        text_block = self.stream.step()
        pos_after = self.stream.tell()
        # Detect wrap by position drop: forward reading always increases
        # pos unless the stream wrapped around. (The stream's ``wrapped``
        # attribute is a sticky "has wrapped at least once" flag, so it
        # can't be used to count individual wraps.)
        wrapped_this_step = pos_after < pos_before
        if wrapped_this_step:
            self.summary.n_wraps += 1

        # 2) ENCODE: text -> observation vector.
        obs = self.encoder.encode(text_block)
        if not np.all(np.isfinite(obs)):
            self.summary.n_nan_obs += 1
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        obs_norm = float(np.linalg.norm(obs))

        # 3) COGNITION: think() runs the 6-module parallel process +
        #    predict + update cycle. This is where learning happens
        #    (emission/transition weights drift toward the obs).
        signal = self.model.think(obs)
        if not np.all(np.isfinite(signal.data)):
            self.summary.n_nan_signal += 1

        # 3b) KL REGULARISATION: pull belief_state toward the N(0, I)
        #     prior. ``update_belief`` implements only the pragmatic
        #     gradient (state += lr * error @ emission.T) but NOT the
        #     KL gradient (-state), so ||belief|| drifts unboundedly
        #     across cycles. This multiplicative decay implements the
        #     missing KL pull and keeps belief bounded.
        engine = self.model.active_inference
        gm = engine.generative_model
        gm.belief_state = gm.belief_state * self.BELIEF_DECAY

        # 4) ACTION SELECTION (recorded but not yet fed back to the
        #    streamer — Phase H+ will wire this up).
        belief = gm.belief_state
        try:
            action_vec = engine.select_action(belief, current_observation=obs)
            action_vec_norm = float(np.linalg.norm(action_vec))
            # Placeholder discretisation: pick the argmax axis. This is
            # NOT a meaningful action mapping yet; it just gives us a
            # categorical signal to log. Phase H+ will replace this with
            # stream-control operations (seek/re-read/skip).
            action_discrete = int(np.argmax(np.abs(action_vec)))
        except Exception:
            # select_action can raise on degenerate belief states
            # (e.g. all-NaN early in training). Fall back to no-op.
            action_vec = np.zeros(engine.blanket.active_dim)
            action_vec_norm = 0.0
            action_discrete = -1

        # 5) FREE ENERGY + PREDICTION ERROR.
        #
        #    We use the PRAGMATIC term (prediction error of the
        #    generative model) as the free-energy signal:
        #
        #        FE = ||obs - belief @ emission||^2 / dim
        #
        #    This is the expected negative log-likelihood under the
        #    variational posterior — the "accuracy" term of the
        #    variational free energy F = KL(q||p) + E_q[-log p(o|s)].
        #    We omit the KL/complexity term because:
        #
        #      a) The engine's ``compute_free_energy`` estimates
        #         sigma_q^2 from action-history variance, which is not
        #         a valid posterior-uncertainty proxy in passive reading
        #         (no real action feedback). It collapses to ~0.15,
        #         inflating ``-dim*log(sigma^2)`` to ~60 and masking
        #         the learning signal.
        #
        #      b) Fixing sigma_q^2=1 reduces KL to ``0.5*||belief||^2``,
        #         but ``update_belief`` lacks the KL gradient (-belief),
        #         so ||belief|| drifts unboundedly across cycles. The
        #         BELIEF_DECAY pull above bounds this drift, but the
        #         complexity term still rises during the transient,
        #         obscuring the downward pragmatic trend.
        #
        #    The pragmatic term faithfully tracks how well the model
        #    learns the text statistics: it decreases as the emission
        #    matrix converges to map belief -> observation.
        belief_after = gm.belief_state
        predicted_obs = gm.predict_observation(belief_after)
        err = obs[:len(predicted_obs)] - predicted_obs[:len(obs)]
        if len(err) < gm.obs_dim:
            err = np.pad(err, (0, gm.obs_dim - len(err)))
        free_energy = float(np.dot(err, err)) / err.size
        # prediction_error: L2 distance between obs and belief_state
        # (a simpler surprise proxy, independent of emission).
        b = belief_after[:obs.shape[0]]
        prediction_error = float(np.linalg.norm(obs - b))

        # 6) RECORD: capture metrics + text preview for offline analysis.
        #    We truncate the preview to printable chars to keep the CSV
        #    clean — some blocks may contain control chars from the
        #    UTF-8 replacement sequence.
        preview = _make_preview(text_block, PREVIEW_CHARS)
        self.summary.records.append(TextCycleRecord(
            step=step,
            action_discrete=action_discrete,
            action_vec_norm=action_vec_norm,
            free_energy=free_energy,
            prediction_error=prediction_error,
            confidence=float(signal.confidence),
            obs_norm=obs_norm,
            pos_before=pos_before,
            pos_after=pos_after,
            wrapped=wrapped_this_step,
            text_preview=preview,
        ))


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def _make_preview(text: str, n: int) -> str:
    """Return a CSV-safe preview of ``text`` (first ``n`` chars)."""
    if not text:
        return ""
    preview = text[:n]
    # Replace any char that would break the CSV (commas, newlines,
    # quotes). We keep alphanumerics + common punctuation, replace
    # everything else with a space.
    safe_chars = []
    for ch in preview:
        if ch.isalnum() or ch in " .,;:!?-'\"()":
            safe_chars.append(ch)
        else:
            safe_chars.append(" ")
    return "".join(safe_chars).strip()


def save_csv(summary: TextRunSummary, path: str | Path) -> str:
    """Write the per-step CSV log."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "step", "action_discrete", "action_vec_norm",
            "free_energy", "prediction_error", "confidence",
            "obs_norm", "pos_before", "pos_after", "wrapped",
            "text_preview",
        ])
        for r in summary.records:
            w.writerow([
                r.step, r.action_discrete,
                f"{r.action_vec_norm:.6f}",
                f"{r.free_energy:.6f}",
                f"{r.prediction_error:.6f}",
                f"{r.confidence:.6f}",
                f"{r.obs_norm:.6f}",
                r.pos_before, r.pos_after, int(r.wrapped),
                r.text_preview,
            ])
    summary.csv_path = str(path)
    return summary.csv_path


def plot_fe_curve(summary: TextRunSummary, path: str | Path) -> str:
    """Save a 3-panel PNG: FE curve, prediction error, confidence."""
    if not _HAS_MPL or not summary.records:
        return ""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    steps = [r.step for r in summary.records]
    fes = [r.free_energy for r in summary.records]
    pes = [r.prediction_error for r in summary.records]
    confs = [r.confidence for r in summary.records]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    ax = axes[0]
    ax.plot(steps, fes, "-", color="#d62728", linewidth=1)
    ax.set_ylabel("free energy")
    ax.set_title(
        f"Text-reading loop: {summary.n_steps} steps, "
        f"mean FE = {summary.mean_free_energy:.4f}, "
        f"final FE = {summary.final_free_energy:.4f}"
    )
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.plot(steps, pes, "-", color="#1f77b4", linewidth=1)
    ax.set_ylabel("prediction error")
    ax.grid(True, alpha=0.3)
    ax = axes[2]
    ax.plot(steps, confs, "-", color="#2ca02c", linewidth=1)
    ax.set_ylabel("confidence")
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    summary.png_path = str(path)
    return summary.png_path


# ------------------------------------------------------------------ #
# Sample text (used when --text_file is not provided)
# ------------------------------------------------------------------ #
def ensure_sample_text(path: Path = SAMPLE_TEXT_PATH, seed: int = 42) -> Path:
    """Generate a short sample text file if none exists.

    The sample is a multi-paragraph essay on cognitive emergence —
    deliberately repetitive in vocabulary so the model has a chance to
    learn word-boundary statistics within a few thousand steps. This
    is NOT a substitute for a real Wiki dump; it's a smoke-test
    fixture so the loop can run end-to-end without external data.
    """
    if path.exists() and path.stat().st_size > 1000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paragraphs = [
        "Cognitive emergence is the spontaneous formation of structure in "
        "a self-organising system. The ZeroDataModel project tests whether "
        "a free-energy-minimising agent can form pre-linguistic concepts "
        "from raw pixel or character streams. The model has no pretrained "
        "embeddings, no tokeniser, and no external language model. It "
        "must discover the structure of its sensory input from scratch.",
        "Active inference provides the mathematical foundation. The agent "
        "maintains a belief about the world and updates it to minimise "
        "free energy, which is the surprise of its observations given its "
        "belief. When the agent reads a stream of characters, the free "
        "energy reflects how predictable the current character block is "
        "given the belief state. As the model learns the statistics of "
        "the text, free energy should decrease.",
        "Curiosity is the drive to seek out informative observations. In "
        "Phase E, the model was extended with an information-gain term in "
        "its action selection. For text reading, curiosity could drive "
        "the model to seek unfamiliar passages or re-read passages that "
        "produced high prediction error. This phase lays the passive "
        "reading baseline; the active reading phase comes next.",
        "The text stream is a deterministic, position-tracked view over a "
        "plain-text file. The model reads fixed-length character blocks "
        "in order, with no action influencing what is read. The encoder "
        "maps each block to a fixed-dim observation vector using a random "
        "projection of one-hot character histograms. The same block "
        "always produces the same vector.",
        "Evaluation of the text loop focuses on the free-energy curve. If "
        "the curve decreases over time, the model is learning the "
        "statistics of the text. If it remains flat, the model is not "
        "learning. The prediction error, confidence, and action "
        "distribution provide additional diagnostic signals. A successful "
        "run produces a downward-trending free-energy curve with stable "
        "confidence and bounded prediction error.",
    ]
    # Repeat the paragraphs to give the streamer enough material for
    # the model to wrap around multiple times in a 2000-step run.
    # 2000 steps × 128 chars/block = 256K chars read. With an 8K-char
    # text, the model wraps ~32 times — enough repetition for the
    # emission matrix to converge while providing 64 distinct blocks
    # so learning spans more of the 2000-step window.
    text = " ".join(paragraphs)
    target_chars = 8_000  # ~62 blocks of 128 chars; wraps ~32x in 2000 steps
    repeats = max(1, target_chars // len(text) + 1)
    full_text = (text + " ") * repeats
    # Pure repetition (no permutation between cycles) so the model sees
    # the exact same block sequence each cycle. This gives the emission
    # matrix stable targets to converge toward.
    full_text = full_text[:target_chars]
    path.write_text(full_text, encoding="utf-8")
    return path


# ------------------------------------------------------------------ #
# Main entry point
# ------------------------------------------------------------------ #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the ZeroDataModel text-reading closed loop.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--text_file", type=str, default="",
        help="Path to the plain-text file to read. If empty, uses the "
             "bundled sample text (cognitive-emergence essay).",
    )
    p.add_argument(
        "--steps", type=int, default=DEFAULT_MAX_STEPS,
        help="Number of reading steps to run.",
    )
    p.add_argument(
        "--block_size", type=int, default=DEFAULT_BLOCK_SIZE,
        help="Characters per block.",
    )
    p.add_argument(
        "--dim", type=int, default=DEFAULT_DIM,
        help="Model + encoder output dimension.",
    )
    p.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="Random seed (for both model and encoder).",
    )
    p.add_argument(
        "--output_dir", type=str, default=str(OUTPUT_DIR),
        help="Output directory for CSV and PNG.",
    )
    p.add_argument(
        "--curious", action="store_true",
        help="Use the curiosity-enhanced engine (default: plain model).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the text file. If none given, use the bundled sample.
    if args.text_file:
        text_path = Path(args.text_file)
        if not text_path.exists():
            raise FileNotFoundError(f"text file not found: {text_path}")
    else:
        text_path = ensure_sample_text()
        print(f"[info] no --text_file given; using sample: {text_path}")

    print("=" * 64)
    print("ZeroDataModel Text-Reading Closed Loop (Phase H)")
    print("=" * 64)
    print(f"  text_file  = {text_path}")
    print(f"  block_size = {args.block_size} chars")
    print(f"  steps      = {args.steps}")
    print(f"  dim        = {args.dim}")
    print(f"  seed       = {args.seed}")
    print(f"  output_dir = {output_dir}")

    # --- Build model + stream + encoder ------------------------------ #
    # Default: plain ZeroDataModel (no curiosity) — a "basic" closed
    # loop where FE should decrease as the model learns text statistics.
    # Use --curious to opt into the curiosity-enhanced engine.
    if args.curious:
        model = make_curious_model(
            dim=args.dim, seed=args.seed,
            beta_start=1.0, beta_min=0.01,
            decay_steps=args.steps, decay_type="linear",
        )
        print(f"  model      = curious (beta_start=1.0, decay=linear)")
    else:
        model = ZeroDataModel(dim=args.dim, seed=args.seed)
        print(f"  model      = plain ZeroDataModel (no curiosity)")
    stream = TextStream(
        text_path, block_size=args.block_size, seed=args.seed,
    )
    encoder = TextEncoder(
        block_size=args.block_size, output_dim=args.dim, seed=args.seed,
    )
    print(f"  stream len = {len(stream)} chars, "
          f"{stream.n_blocks()} blocks of {args.block_size}")
    print(f"  encoder    = {encoder}")

    # --- Run the loop ------------------------------------------------- #
    runner = TextRunner(model, stream, encoder, max_steps=args.steps)
    print(f"\nRunning {args.steps} reading steps...")
    summary = runner.run()

    # --- Save outputs ------------------------------------------------- #
    csv_path = save_csv(summary, output_dir / "text_run.csv")
    print(f"\nCSV: {csv_path}")
    png_path = plot_fe_curve(summary, output_dir / "text_run.png")
    if png_path:
        print(f"PNG: {png_path}")

    # --- Print final summary ------------------------------------------ #
    print("\n" + "=" * 64)
    print("Final summary")
    print("=" * 64)
    print(f"  steps             = {summary.n_steps}")
    print(f"  crashes           = {summary.n_crashes}")
    print(f"  nan_obs           = {summary.n_nan_obs}")
    print(f"  nan_signal        = {summary.n_nan_signal}")
    print(f"  wraps             = {summary.n_wraps}")
    print(f"  elapsed           = {summary.elapsed_s:.1f}s")
    print(f"  rate              = {summary.cycles_per_sec:.1f} cyc/s")
    print(f"  mean FE           = {summary.mean_free_energy:.4f}")
    print(f"  final FE          = {summary.final_free_energy:.4f}")
    print(f"  mean pred error   = {summary.mean_prediction_error:.4f}")
    print(f"  mean confidence   = {summary.mean_confidence:.4f}")
    if summary.crash_messages:
        print(f"  crashes (first 3) = {summary.crash_messages[:3]}")

    # Show the first 5 rows of the log for human inspection.
    print("\nFirst 5 steps:")
    for r in summary.records[:5]:
        print(f"  step {r.step}: FE={r.free_energy:.4f}  "
              f"PE={r.prediction_error:.4f}  conf={r.confidence:.4f}  "
              f"pos={r.pos_before}->{r.pos_after}  "
              f"text={r.text_preview!r}")
    # Show the last 5 rows too — this is where we hope to see FE drop.
    print("Last 5 steps:")
    for r in summary.records[-5:]:
        print(f"  step {r.step}: FE={r.free_energy:.4f}  "
              f"PE={r.prediction_error:.4f}  conf={r.confidence:.4f}  "
              f"pos={r.pos_before}->{r.pos_after}  "
              f"text={r.text_preview!r}")
    print("\n" + "=" * 64)
    print("Done.")
    print("=" * 64)


if __name__ == "__main__":
    main()
