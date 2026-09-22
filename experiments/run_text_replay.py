# experiments/run_text_replay.py
"""Text-reading loop with experience replay + offline consolidation (Phase I).

Extends ``run_text_curious.py`` (Phase H curiosity-driven active reading)
with THREE new capabilities:

  1. **Experience replay** — every online step is stored in an
     ``ExperienceBuffer`` (priority-weighted by prediction error).
  2. **Periodic offline consolidation** — every ``consolidation_interval``
     steps, the loop pauses and replays ``consolidation_batch_size``
     high-surprise experiences through the model's
     ``think() → update_belief → update`` path. This "sleep phase"
     strengthens predictions for surprising transitions (rare words,
     syntactic boundary patterns), accelerating the formation of
     stable character-level intuition.
  3. **Cognitive snapshots** — every ``snapshot_interval`` steps, the
     full internal state of the model is pickled to disk. These
     snapshots are consumed by the ``analyze_*.py`` scripts to evaluate
     whether the model has begun forming text-structural concepts
     (word boundaries, topic clusters, factual associations).

Architecture
------------
    TextStream.navigate(action)  ->  str    # action ∈ {0..4}
       ↓
    TextEncoder.encode(text)     ->  obs    # (output_dim,) vector
       ↓
    ZeroDataModel.think(obs)     ->  signal
       ↓
    ActiveInference.select_action(belief, obs)  ->  action_vec + idx
       ↓
    stream.navigate(_last_selected_idx)  # FEEDS BACK to the stream
       ↓
    buffer.add(obs, action, next_obs, error=free_energy, step=step)
       ↓
    every consolidation_interval steps:
        _consolidate(model, buffer, batch_size)   # inline replay
       ↓
    every snapshot_interval steps:
        collect_snapshot(model, step, text, obs) -> .pkl
       ↓
    CSV log + PNG plots

CLI
---
    python experiments/run_text_replay.py --steps 2000
    python experiments/run_text_replay.py --consolidation_interval 50 \\
                                            --snapshot_interval 50

Determinism: with a fixed seed, the run produces the same snapshots,
CSV, and free-energy curve on every invocation.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from run_sandbox_curious import make_curious_model  # noqa: E402
from text_encoder import TextEncoder  # noqa: E402
from text_stream import (  # noqa: E402
    NAV_ACTION_NAMES,
    NAV_FORWARD,
    N_NAV_ACTIONS,
    TextStream,
)
from run_text_curious import (  # noqa: E402
    CuriousTextCycleRecord,
    CuriousTextRunner,
    CuriousTextRunSummary,
    make_text_curious_model,
    save_csv as save_curious_csv,
)
from run_text_loop import _make_preview, ensure_sample_text  # noqa: E402
from experience_buffer import ExperienceBuffer  # noqa: E402

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
DEFAULT_MAX_STEPS = 2000          # shorter than curious run (5K) — consolidation is expensive
DEFAULT_SEED = 42
DEFAULT_BUFFER_CAPACITY = 5000   # matches the spec; enough for 2000 steps
DEFAULT_CONSOLIDATION_INTERVAL = 100
DEFAULT_CONSOLIDATION_BATCH_SIZE = 32
DEFAULT_CONSOLIDATION_EPOCHS = 1  # number of passes over the sampled batch per round
DEFAULT_SNAPSHOT_INTERVAL = 50
PROGRESS_INTERVAL = 100
PREVIEW_CHARS = 20
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "text_replay_run"


# ------------------------------------------------------------------ #
# Per-step record (extends CuriousTextCycleRecord with replay fields)
# ------------------------------------------------------------------ #
@dataclass
class ReplayTextCycleRecord(CuriousTextCycleRecord):
    """Per-step record with replay/consolidation metadata.

    Inherits all fields from ``CuriousTextCycleRecord`` (FE, PE, β,
    action, position, etc.) and adds the experience-buffer index and
    a flag indicating whether this step triggered a consolidation
    round.
    """

    buffer_size_after: int = 0       # buffer length after add()
    in_consolidation: bool = False   # was this step a consolidation trigger?
    delta_fe_consolidation: float = 0.0  # FE change during the consolidation (neg = improvement)


@dataclass
class ConsolidationResult:
    """Result of one offline consolidation round."""

    n_replayed: int = 0
    delta_free_energy: float = 0.0
    free_energy_before: float = 0.0
    free_energy_after: float = 0.0
    mean_replay_error: float = 0.0
    mean_replay_ig: float = 0.0
    elapsed_s: float = 0.0


@dataclass
class ReplayTextRunSummary(CuriousTextRunSummary):
    """Extended run summary with consolidation + snapshot metadata."""

    n_consolidation_rounds: int = 0
    n_replayed_experiences: int = 0
    mean_delta_fe_consolidation: float = 0.0
    consolidation_rounds: list = field(default_factory=list)  # list of ConsolidationResult
    snapshot_paths: list[str] = field(default_factory=list)
    snapshot_steps: list[int] = field(default_factory=list)
    records: list[ReplayTextCycleRecord] = field(default_factory=list)  # type: ignore[assignment]


# ------------------------------------------------------------------ #
# Cognitive snapshot collector (text-domain)
# ------------------------------------------------------------------ #
def collect_snapshot(
    model: ZeroDataModel,
    step: int,
    text_block: str,
    obs: np.ndarray,
    pos: int,
    free_energy: float,
    prediction_error: float,
) -> dict:
    """Capture a serialisable cognitive snapshot for offline analysis.

    The schema is intentionally similar to ``run_sandbox_replay.collect_snapshot``
    (Phase G) so the same analysis scripts can be adapted easily, with
    text-specific additions (the raw text block, character position).

    Schema (v1.0):
        meta:        {schema_version, step, cycle_count, dim, pos, free_energy, prediction_error}
        text_block:  str                # the raw chars read this step
        obs:         (output_dim,) float64
        consciousness: {self_state, self_confidence, workspace_sum,
                        workspace_attention, workspace_buffer_data}
        active_inference: {belief_state, emission, free_energy_history,
                           exploration_step, action_error_history,
                           _last_selected_idx}
        category_engine: {topos_classifier, topos_truth_values,
                          n_categories, category_names, n_functors}
        causal_emergence: {memory_patterns, memory_labels, memory_targets}
    """
    consciousness = model.consciousness
    active_inf = model.active_inference
    category = model.category_engine
    emergence = model.emergence

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

    gen = active_inf.generative_model
    active_inf_snap = {
        "belief_state": np.asarray(gen.belief_state, dtype=float).copy(),
        "emission": np.asarray(gen.emission, dtype=float).copy(),
        "transition": np.asarray(gen.transition, dtype=float).copy(),
        "free_energy_history": np.asarray(list(active_inf.free_energy_history), dtype=float),
        "exploration_step": int(active_inf._exploration_step),
        "last_selected_idx": int(getattr(active_inf, "_last_selected_idx", 0)),
        "action_error_history": [
            list(bin_hist) for bin_hist in active_inf._action_error_history
        ],
        "blanket_active_weights": np.asarray(
            active_inf.blanket.active_weights, dtype=float
        ).copy(),
    }

    category_snap = {
        "topos_classifier": np.asarray(category.topos.classifier, dtype=float).copy(),
        "topos_truth_values": np.asarray(category.topos.truth_values, dtype=float).copy(),
        "n_categories": len(category.categories),
        "category_names": list(category.categories.keys()),
        "n_functors": len(category.functors),
        "n_objects_per_category": {
            name: len(cat.objects) for name, cat in category.categories.items()
        },
    }

    memory = emergence.memory
    emergence_snap = {
        "memory_patterns": [
            np.asarray(p, dtype=float).copy() for p in memory._patterns
        ],
        "memory_labels": list(memory._labels),
        "memory_targets": [list(t) for t in memory._targets],
        "memory_capacity": len(memory._patterns),
    }

    return {
        "meta": {
            "schema_version": "1.0-text",
            "step": int(step),
            "cycle_count": int(model.cycle_count),
            "dim": int(model.dim),
            "pos": int(pos),
            "free_energy": float(free_energy),
            "prediction_error": float(prediction_error),
        },
        "text_block": str(text_block),
        "obs": np.asarray(obs, dtype=float).copy(),
        "consciousness": consciousness_snap,
        "active_inference": active_inf_snap,
        "category_engine": category_snap,
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


# ------------------------------------------------------------------ #
# Runner with consolidation + snapshots
# ------------------------------------------------------------------ #
class ReplayTextRunner(CuriousTextRunner):
    """Curiosity-driven text reader + replay + snapshots.

    Subclasses ``CuriousTextRunner`` and overrides:
      - ``__init__`` to add buffer + consolidator + snapshot config
      - ``_cycle`` to store experiences + trigger consolidation + save snapshots
      - ``run`` to update the extended summary type

    The core perception-cognition-action loop is UNCHANGED — the model
    still reads text blocks, thinks about them, and chooses navigation
    actions. The replay/snapshot machinery is layered on top.

    FE definition (inherited from Task G fix in run_text_loop.py):
      We use the PRAGMATIC term ``||obs - belief @ emission||^2 / dim``
      as the free-energy signal, NOT ``engine.compute_free_energy()``.
      The engine's method estimates sigma_q^2 from action-history
      variance, which collapses to ~0.15 in passive reading and
      inflates the KL term ``-dim*log(sigma^2)`` to ~60, masking the
      learning signal. We also apply BELIEF_DECAY=0.99 each step to
      emulate the missing KL gradient (-state) in update_belief,
      preventing ||belief|| from drifting unboundedly.
    """

    BELIEF_DECAY = 0.99  # emulates missing KL gradient in update_belief

    def _pragmatic_fe(self, obs: np.ndarray) -> float:
        """Pragmatic free-energy term: ||obs - belief @ emission||^2 / dim.

        Replaces ``engine.compute_free_energy()`` which uses a broken
        sigma_q^2 proxy (action-variance-based) that collapses in
        passive reading. See run_text_loop.py Task G fix for details.
        """
        gm = self.model.active_inference.generative_model
        belief = gm.belief_state
        predicted = gm.predict_observation(belief)
        err = obs[:len(predicted)] - predicted[:len(obs)]
        if len(err) < gm.obs_dim:
            err = np.pad(err, (0, gm.obs_dim - len(err)))
        return float(np.dot(err, err)) / err.size

    def __init__(
        self,
        model: ZeroDataModel,
        stream: TextStream,
        encoder: TextEncoder,
        max_steps: int = DEFAULT_MAX_STEPS,
        buffer_capacity: int = DEFAULT_BUFFER_CAPACITY,
        consolidation_interval: int = DEFAULT_CONSOLIDATION_INTERVAL,
        consolidation_batch_size: int = DEFAULT_CONSOLIDATION_BATCH_SIZE,
        consolidation_epochs: int = DEFAULT_CONSOLIDATION_EPOCHS,
        snapshot_interval: int = DEFAULT_SNAPSHOT_INTERVAL,
        output_dir: Path = OUTPUT_DIR,
        seed: int | None = None,
    ):
        super().__init__(model, stream, encoder, max_steps)
        # Replace the parent summary with the extended one.
        self.summary = ReplayTextRunSummary()
        # Reuse the existing ExperienceBuffer (priority-weighted by
        # prediction error / free energy). alpha=0.6 gives proportional
        # sampling; beta=0.4 provides mild IS correction.
        self.buffer = ExperienceBuffer(
            capacity=buffer_capacity,
            alpha=0.6,
            beta=0.4,
            seed=seed,
        )
        self.consolidation_batch_size = int(consolidation_batch_size)
        self.consolidation_interval = int(consolidation_interval)
        self.consolidation_epochs = int(consolidation_epochs)
        self.snapshot_interval = int(snapshot_interval)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Track the last observation for FE-before-consolidation measurement.
        self._last_obs: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # Override the cycle: add experience + consolidate + snapshot
    # ------------------------------------------------------------------ #
    def _cycle(self, step: int) -> None:
        """One closed-loop cycle with replay + snapshot machinery."""
        # --- Phases 1-3: navigate → encode → think → select_action --- #
        # We replicate the parent's logic and add experience storage
        # + consolidation + snapshots. Duplicating the parent's body is
        # cleaner than monkey-patching here — the cycle is the hot path
        # and a super().cycle() call would still need us to recompute
        # the experience tuple from the side-effects.
        engine = self.model.active_inference
        pos_before = self.stream.tell()
        if step == 0:
            nav_action = NAV_FORWARD
        else:
            nav_action = int(engine._last_selected_idx)
            if not 0 <= nav_action < N_NAV_ACTIONS:
                nav_action = NAV_FORWARD
        text_block = self.stream.navigate(nav_action)
        pos_after = self.stream.tell()
        # Detect wrap by position drop (forward reading always increases
        # pos unless the stream wrapped). The stream's ``wrapped`` flag is
        # sticky and cannot count individual wraps.
        wrapped_this_step = pos_after < pos_before
        if wrapped_this_step:
            self.summary.n_wraps += 1

        obs = self.encoder.encode(text_block)
        if not np.all(np.isfinite(obs)):
            self.summary.n_nan_obs += 1
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        obs_norm = float(np.linalg.norm(obs))
        self._last_obs = obs  # for FE-before-consolidation

        signal = self.model.think(obs)
        if not np.all(np.isfinite(signal.data)):
            self.summary.n_nan_signal += 1
        # KL regularisation: pull belief_state toward N(0, I) prior.
        # update_belief implements only the pragmatic gradient, missing
        # the KL gradient (-state), so ||belief|| drifts unboundedly.
        gm = engine.generative_model
        gm.belief_state = gm.belief_state * self.BELIEF_DECAY
        belief = gm.belief_state
        action_vec = engine.select_action(belief, current_observation=obs)
        action_vec_norm = float(np.linalg.norm(action_vec))

        # Pragmatic FE (Task G fix): ||obs - belief @ emission||^2 / dim.
        # NOT engine.compute_free_energy() (sigma_q^2 proxy collapses).
        free_energy = self._pragmatic_fe(obs)
        belief_after = gm.belief_state
        b = belief_after[:obs.shape[0]]
        prediction_error = float(np.linalg.norm(obs - b))
        beta = self._current_beta()
        info_gain = self._current_info_gain(action_vec)

        # --- Phase I.1: STORE the experience ---------------------------- #
        # ``next_obs`` is approximated by obs itself — at inference time
        # we don't yet know the next observation (it depends on the
        # action we JUST chose, which will be executed next step). For
        # consolidation, what matters is the obs/belief pairing at THIS
        # step; the consolidator re-runs ``think(obs)`` which calls
        # ``update_belief(obs)`` + ``update(error)`` internally, so the
        # cached context is fresh after each replay.
        self.buffer.add(
            obs=obs, action=nav_action, next_obs=obs,
            error=free_energy, step=step,
        )
        buffer_size_after = len(self.buffer)

        # --- Phase I.2: PERIODIC CONSOLIDATION -------------------------- #
        delta_fe_consolidation = 0.0
        in_consolidation = False
        if (step + 1) % self.consolidation_interval == 0 and len(self.buffer) > 0:
            in_consolidation = True
            # Run ``epochs`` rounds of consolidation back-to-back.
            # Each round re-samples from the buffer and replays.
            for _ in range(max(1, self.consolidation_epochs)):
                result = self._consolidate()
                self.summary.consolidation_rounds.append(result)
                self.summary.n_consolidation_rounds += 1
                self.summary.n_replayed_experiences += result.n_replayed
                delta_fe_consolidation += result.delta_free_energy
            # The delta is summed across epochs; report the average.
            if self.consolidation_epochs > 0:
                delta_fe_consolidation /= self.consolidation_epochs

        # --- Phase I.3: PERIODIC SNAPSHOT ------------------------------- #
        if (step + 1) % self.snapshot_interval == 0:
            snap = collect_snapshot(
                self.model, step=step + 1, text_block=text_block,
                obs=obs, pos=pos_after,
                free_energy=free_energy, prediction_error=prediction_error,
            )
            snap_path = self.output_dir / f"snapshot_step{step + 1:06d}.pkl"
            save_snapshot(snap, snap_path)
            self.summary.snapshot_paths.append(str(snap_path))
            self.summary.snapshot_steps.append(step + 1)

        # --- Phase 4: RECORD metrics (extended record) ----------------- #
        preview = _make_preview(text_block, PREVIEW_CHARS)
        self.summary.records.append(ReplayTextCycleRecord(
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
            buffer_size_after=buffer_size_after,
            in_consolidation=in_consolidation,
            delta_fe_consolidation=delta_fe_consolidation,
        ))

    # ------------------------------------------------------------------ #
    # Inline consolidation: sample from buffer, replay through think()
    # ------------------------------------------------------------------ #
    def _consolidate(self) -> ConsolidationResult:
        """Sample a batch from the buffer and replay through ``think()``.

        For each sampled experience, we re-feed both ``obs`` and
        ``next_obs`` through ``model.think()``. This drives
        ``update_belief`` and ``update`` (gradient descent on emission),
        strengthening the model's predictions for surprising transitions.

        We temporarily stash ``cycle_count`` so the consolidation does
        not inflate the online cycle count. The ``think()`` calls still
        update the generative model — that is the point of consolidation.
        """
        import time as _time
        t0 = _time.perf_counter()
        engine = self.model.active_inference

        # Measure FE before consolidation (on the most recent obs).
        # Uses pragmatic FE (Task G fix), NOT engine.compute_free_energy().
        fe_before = self._pragmatic_fe(self._last_obs) \
            if self._last_obs is not None else 0.0

        batch = self.buffer.sample(
            self.consolidation_batch_size, mode="priority"
        )
        if len(batch) == 0:
            return ConsolidationResult(
                n_replayed=0, delta_free_energy=0.0,
                free_energy_before=fe_before, free_energy_after=fe_before,
                elapsed_s=0.0,
            )

        # Stash cycle_count — think() increments it.
        saved_cycle_count = self.model.cycle_count
        fes: list[float] = []
        for exp in batch.experiences:
            # Rehear the original observation.
            try:
                self.model.think(exp.obs)
                # Apply belief decay during consolidation too.
                gm = engine.generative_model
                gm.belief_state = gm.belief_state * self.BELIEF_DECAY
                fe_obs = self._pragmatic_fe(exp.obs)
                if np.isfinite(fe_obs):
                    fes.append(fe_obs)
            except Exception:
                pass
            # Rehear the next observation (the outcome).
            try:
                self.model.think(exp.next_obs)
                gm = engine.generative_model
                gm.belief_state = gm.belief_state * self.BELIEF_DECAY
                fe_next = self._pragmatic_fe(exp.next_obs)
                if np.isfinite(fe_next):
                    fes.append(fe_next)
            except Exception:
                pass
        # Restore cycle_count so online counter stays consistent.
        self.model.cycle_count = saved_cycle_count

        fe_after = float(np.mean(fes)) if fes else fe_before
        mean_replay_error = float(np.mean(fes)) if fes else 0.0
        elapsed = _time.perf_counter() - t0
        return ConsolidationResult(
            n_replayed=len(batch.experiences),
            delta_free_energy=fe_after - fe_before,
            free_energy_before=fe_before,
            free_energy_after=fe_after,
            mean_replay_error=mean_replay_error,
            mean_replay_ig=0.0,  # not tracked in inline consolidation
            elapsed_s=elapsed,
        )

    # ------------------------------------------------------------------ #
    # Override run() to compute the extended summary stats
    # ------------------------------------------------------------------ #
    def run(self) -> ReplayTextRunSummary:
        """Execute the loop and compute summary stats."""
        # Reuse the parent's run loop (it calls self._cycle which we
        # override). We need to replace the summary with our extended
        # type BEFORE calling super().run() so the records appended by
        # _cycle land in the right list. Parent's __init__ already set
        # self.summary to a ReplayTextRunSummary, but parent's run()
        # reads/writes CuriousTextRunSummary fields — those are all
        # inherited, so it works.
        summary = super().run()
        # The parent's run returns self.summary (same object), but typed
        # as CuriousTextRunSummary. Cast back to our type.
        assert isinstance(summary, ReplayTextRunSummary)
        # Compute consolidation-specific aggregates.
        if self.summary.consolidation_rounds:
            deltas = [r.delta_free_energy for r in self.summary.consolidation_rounds]
            self.summary.mean_delta_fe_consolidation = float(np.mean(deltas))
        return self.summary


# ------------------------------------------------------------------ #
# Output: CSV (extends curious CSV with replay fields) + PNG + manifest
# ------------------------------------------------------------------ #
def save_csv(summary: ReplayTextRunSummary, path: str | Path) -> str:
    """Write the per-step CSV log with replay fields."""
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
            "buffer_size_after", "in_consolidation", "delta_fe_consolidation",
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
                r.buffer_size_after, int(r.in_consolidation),
                f"{r.delta_fe_consolidation:.6f}",
            ])
    summary.csv_path = str(path)
    return summary.csv_path


def plot_replay_run(summary: ReplayTextRunSummary, path: str | Path) -> str:
    """Save a 6-panel PNG: FE, consolidation delta, β, action histogram, IG, position."""
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
    buf_sizes = [r.buffer_size_after for r in summary.records]
    cons_steps = [r.step for r in summary.records if r.in_consolidation]
    cons_deltas = [r.delta_fe_consolidation for r in summary.records if r.in_consolidation]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # Panel 1: free energy + prediction error
    ax = axes[0, 0]
    ax.plot(steps, fes, "-", color="#d62728", linewidth=1, label="free energy")
    ax.plot(steps, pes, "-", color="#1f77b4", linewidth=1, label="prediction error")
    # Mark consolidation steps.
    for s in cons_steps:
        ax.axvline(s, color="gray", alpha=0.15, linewidth=0.5)
    ax.set_xlabel("step")
    ax.set_ylabel("value")
    ax.set_title(
        f"Replay loop: FE & PE (mean FE = {summary.mean_free_energy:.3f}, "
        f"final = {summary.final_free_energy:.3f}, gray = consolidation)"
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Panel 2: consolidation delta FE
    ax = axes[0, 1]
    if cons_steps:
        ax.plot(cons_steps, cons_deltas, "o-", color="#9467bd", linewidth=1.5)
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("step (consolidation rounds)")
    ax.set_ylabel("Δ FE (after - before)")
    ax.set_title(
        f"Consolidation effect: mean ΔFE = {summary.mean_delta_fe_consolidation:.4f} "
        f"(negative = improvement)"
    )
    ax.grid(True, alpha=0.3)

    # Panel 3: β decay
    ax = axes[1, 0]
    ax.plot(steps, betas, "-", color="#2ca02c", linewidth=1.5)
    ax.set_xlabel("step")
    ax.set_ylabel("β")
    ax.set_title(f"β decay: {summary.initial_beta:.3f} -> {summary.final_beta:.3f}")
    ax.grid(True, alpha=0.3)

    # Panel 4: action distribution
    ax = axes[1, 1]
    counts = summary.action_counts
    total = max(1, sum(counts))
    fractions = [c / total for c in counts]
    x = np.arange(N_NAV_ACTIONS)
    bars = ax.bar(x, fractions, color="#ff7f0e", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(NAV_ACTION_NAMES, rotation=30, ha="right")
    ax.set_ylabel("fraction of steps")
    ax.set_title("Action distribution")
    for bar, c in zip(bars, counts, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{c}", ha="center", va="bottom", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 5: buffer size + IG
    ax = axes[2, 0]
    ax.plot(steps, buf_sizes, "-", color="#17becf", linewidth=1, label="buffer size")
    ax2 = ax.twinx()
    ax2.plot(steps, igs, "-", color="#ff7f0e", linewidth=0.8, alpha=0.7, label="IG proxy")
    ax.set_xlabel("step")
    ax.set_ylabel("buffer size", color="#17becf")
    ax2.set_ylabel("info gain proxy", color="#ff7f0e")
    ax.set_title("Buffer fill + IG proxy")
    ax.grid(True, alpha=0.3)

    # Panel 6: position trajectory
    ax = axes[2, 1]
    ax.plot(steps, pos, "-", color="#1f77b4", linewidth=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("stream position (chars)")
    ax.set_title("Reading position over time")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    summary.png_path = str(path)
    return summary.png_path


def save_manifest(summary: ReplayTextRunSummary, path: str | Path, args: argparse.Namespace) -> str:
    """Save a JSON manifest describing the run (consumed by analyze_*.py)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "1.0-text-replay",
        "max_steps": int(args.steps),
        "dim": int(args.dim),
        "seed": int(args.seed),
        "block_size": int(args.block_size),
        "buffer_capacity": int(args.buffer_capacity),
        "consolidation_interval": int(args.consolidation_interval),
        "consolidation_batch_size": int(args.consolidation_batch_size),
        "consolidation_epochs": int(args.consolidation_epochs),
        "snapshot_interval": int(args.snapshot_interval),
        "snapshot_paths": summary.snapshot_paths,
        "snapshot_steps": summary.snapshot_steps,
        "n_consolidation_rounds": summary.n_consolidation_rounds,
        "n_replayed_experiences": summary.n_replayed_experiences,
        "mean_delta_fe_consolidation": summary.mean_delta_fe_consolidation,
        "mean_free_energy": summary.mean_free_energy,
        "final_free_energy": summary.final_free_energy,
        "initial_beta": summary.initial_beta,
        "final_beta": summary.final_beta,
        "action_counts": dict(zip(NAV_ACTION_NAMES, summary.action_counts)),
    }
    with path.open("w") as f:
        json.dump(manifest, f, indent=2)
    return str(path)


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the text-reading loop with experience replay + snapshots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--text_file", type=str, default="",
                   help="Path to the plain-text file. Empty = use sample.")
    p.add_argument("--steps", type=int, default=DEFAULT_MAX_STEPS)
    p.add_argument("--block_size", type=int, default=DEFAULT_BLOCK_SIZE)
    p.add_argument("--dim", type=int, default=DEFAULT_DIM)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--beta_start", type=float, default=1.0)
    p.add_argument("--beta_min", type=float, default=0.01)
    p.add_argument("--decay_steps", type=int, default=DEFAULT_MAX_STEPS)
    p.add_argument("--decay_type", type=str, default="linear",
                   choices=["linear", "exponential", "stage"])
    p.add_argument("--buffer_capacity", type=int, default=DEFAULT_BUFFER_CAPACITY)
    p.add_argument("--consolidation_interval", type=int,
                   default=DEFAULT_CONSOLIDATION_INTERVAL,
                   help="Run consolidation every N steps.")
    p.add_argument("--consolidation_batch_size", type=int,
                   default=DEFAULT_CONSOLIDATION_BATCH_SIZE)
    p.add_argument("--consolidation_epochs", type=int,
                   default=DEFAULT_CONSOLIDATION_EPOCHS,
                   help="Number of replay passes per consolidation round.")
    p.add_argument("--snapshot_interval", type=int,
                   default=DEFAULT_SNAPSHOT_INTERVAL,
                   help="Save a cognitive snapshot every N steps.")
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
    print("Text-Reading Loop with Experience Replay + Snapshots (Phase I)")
    print("=" * 72)
    print(f"  text_file              = {text_path}")
    print(f"  block_size             = {args.block_size} chars")
    print(f"  steps                  = {args.steps}")
    print(f"  dim                    = {args.dim}")
    print(f"  seed                   = {args.seed}")
    print(f"  β_start/min            = {args.beta_start} / {args.beta_min}")
    print(f"  decay                  = {args.decay_type} over {args.decay_steps} steps")
    print(f"  buffer_capacity        = {args.buffer_capacity}")
    print(f"  consolidation_interval = {args.consolidation_interval}")
    print(f"  consolidation_batch    = {args.consolidation_batch_size}")
    print(f"  consolidation_epochs   = {args.consolidation_epochs}")
    print(f"  snapshot_interval      = {args.snapshot_interval}")
    print(f"  output_dir             = {output_dir}")

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
    print(f"  stream len             = {len(stream)} chars, "
          f"{stream.n_blocks()} blocks of {args.block_size}")

    # --- Run the loop ------------------------------------------------- #
    runner = ReplayTextRunner(
        model, stream, encoder, max_steps=args.steps,
        buffer_capacity=args.buffer_capacity,
        consolidation_interval=args.consolidation_interval,
        consolidation_batch_size=args.consolidation_batch_size,
        consolidation_epochs=args.consolidation_epochs,
        snapshot_interval=args.snapshot_interval,
        output_dir=output_dir, seed=args.seed,
    )
    print(f"\nRunning {args.steps} steps with consolidation every "
          f"{args.consolidation_interval} steps + snapshots every "
          f"{args.snapshot_interval} steps...")
    summary = runner.run()

    # --- Save outputs ------------------------------------------------- #
    csv_path = save_csv(summary, output_dir / "text_replay_run.csv")
    print(f"\nCSV: {csv_path}")
    png_path = plot_replay_run(summary, output_dir / "text_replay_run.png")
    if png_path:
        print(f"PNG: {png_path}")
    manifest_path = save_manifest(
        summary, output_dir / "manifest.json", args
    )
    print(f"Manifest: {manifest_path}")
    print(f"Snapshots: {len(summary.snapshot_paths)} files in {output_dir}")

    # --- Print final summary ----------------------------------------- #
    print("\n" + "=" * 72)
    print("Final summary")
    print("=" * 72)
    print(f"  steps                  = {summary.n_steps}")
    print(f"  crashes                = {summary.n_crashes}")
    print(f"  nan_obs / nan_signal   = {summary.n_nan_obs} / {summary.n_nan_signal}")
    print(f"  wraps                  = {summary.n_wraps}")
    print(f"  elapsed                = {summary.elapsed_s:.1f}s")
    print(f"  rate                   = {summary.cycles_per_sec:.1f} cyc/s")
    print(f"  initial β              = {summary.initial_beta:.4f}")
    print(f"  final β                = {summary.final_beta:.4f}")
    print(f"  mean FE                = {summary.mean_free_energy:.4f}")
    print(f"  final FE               = {summary.final_free_energy:.4f}")
    print(f"  mean pred error        = {summary.mean_prediction_error:.4f}")
    print(f"  mean confidence        = {summary.mean_confidence:.4f}")
    print(f"  consolidation rounds   = {summary.n_consolidation_rounds}")
    print(f"  experiences replayed   = {summary.n_replayed_experiences}")
    print(f"  mean ΔFE consolidation = {summary.mean_delta_fe_consolidation:.4f} "
          f"(negative = improvement)")
    print(f"  snapshots saved        = {len(summary.snapshot_paths)}")
    print(f"  action counts          = {dict(zip(NAV_ACTION_NAMES, summary.action_counts))}")
    if summary.crash_messages:
        print(f"  crashes (first 3)      = {summary.crash_messages[:3]}")

    # Show consolidation round details (first 5 + last 5).
    if summary.consolidation_rounds:
        print("\nConsolidation rounds (first 5):")
        for i, r in enumerate(summary.consolidation_rounds[:5]):
            print(f"  round {i}: replayed={r.n_replayed}  "
                  f"FE before={r.free_energy_before:.3f}  "
                  f"after={r.free_energy_after:.3f}  "
                  f"Δ={r.delta_free_energy:+.4f}")
        if len(summary.consolidation_rounds) > 5:
            print(f"  ... ({len(summary.consolidation_rounds) - 5} more rounds)")
            print("Consolidation rounds (last 5):")
            for i, r in enumerate(summary.consolidation_rounds[-5:],
                                  start=len(summary.consolidation_rounds) - 5):
                print(f"  round {i}: replayed={r.n_replayed}  "
                      f"FE before={r.free_energy_before:.3f}  "
                      f"after={r.free_energy_after:.3f}  "
                      f"Δ={r.delta_free_energy:+.4f}")

    print("\n" + "=" * 72)
    print("Done. Run analyze_*.py to evaluate cognitive emergence.")
    print("=" * 72)


if __name__ == "__main__":
    main()
