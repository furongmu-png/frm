# experiments/run_encyclopedia_learning.py
"""Long-term encyclopedia learning loop (Phase J, Task 2).

A session-based learning loop that drives the ZeroDataModel through
large-scale Wikipedia-like text corpora, with:

  - **6 navigation actions** (extended from Phase H's 5): forward,
    backward, fast-forward (paragraph skip), jump-within-article,
    follow-link (hyperlink surfing), and random-article (explore
    new territory).
  - **Session structure**: 1 session = N online reading steps +
    M consolidation steps. Each session ends with a checkpoint.
  - **Checkpoint/resume**: model state, reader position, buffer
    contents, knowledge graph, and metrics are saved to disk every
    session. The script can resume from the last checkpoint.
  - **Internal metrics** (computed every ``eval_interval`` steps):
    prediction-error slope, re-read preference, concept stability,
    internal-consistency (linked vs random article pairs).

Architecture
------------
    EncyclopediaReader.navigate(action)  ->  text_block
       ↓                                 # action ∈ {0..5}
    TextEncoder.encode(text)             ->  obs
       ↓
    ZeroDataModel.think(obs)             ->  signal
       ↓
    ActiveInference.select_action        ->  _last_selected_idx
       ↓                                  # 0..5 navigation action
    EncyclopediaReader.navigate(...)
       ↓
    buffer.add(obs, action, next_obs, error, info_gain)
       ↓
    every consolidation_interval:
        consolidator.consolidate(model, buffer)
       ↓
    every session_end:
        save_checkpoint(...)
       ↓
    every eval_interval:
        compute_internal_metrics(...)
       ↓
    CSV log + manifest + checkpoints + knowledge graph

CLI
---
    # Fresh run, 50K steps total, sessions of 500+100
    python experiments/run_encyclopedia_learning.py \\
        --corpus_dir my_wiki_extract \\
        --total_steps 50000 \\
        --session_steps 500 \\
        --consolidation_steps 100

    # Resume from last checkpoint
    python experiments/run_encyclopedia_learning.py --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from encyclopedia_reader import (  # noqa: E402
    DEFAULT_BLOCK_SIZE,
    EncyclopediaReader,
    make_synthetic_corpus,
)
from text_encoder import TextEncoder  # noqa: E402
from text_experience_buffer import (  # noqa: E402
    TextConsolidator,
    TextExperienceBuffer,
)
from run_text_curious import make_text_curious_model  # noqa: E402

from zero_data_model.model import ZeroDataModel  # noqa: E402

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:  # pragma: no cover
    _HAS_MPL = False


# ------------------------------------------------------------------ #
# Navigation action constants (6 actions for the encyclopedia domain)
# ------------------------------------------------------------------ #
# Phase H (basic text) used 5 actions (forward/backward/fast_forward/
# fast_backward/stay). Phase J extends this with ARTICLE-LEVEL
# navigation: the encyclopedia reader can jump to other articles via
# hyperlinks (action 4) or to a random article (action 5).
NAV_FORWARD          = 0  # +block_size  (next block, same article)
NAV_BACKWARD         = 1  # -block_size  (prev block, clamped at 0)
NAV_FAST_FORWARD     = 2  # +block_size * 4  (paragraph skip)
NAV_JUMP_WITHIN      = 3  # random position within current article
NAV_FOLLOW_LINK      = 4  # jump to a random linked article
NAV_RANDOM_ARTICLE   = 5  # jump to a completely random article
N_NAV_ACTIONS = 6
NAV_ACTION_NAMES = [
    "forward", "backward", "fast_forward",
    "jump_within", "follow_link", "random_article",
]


# ------------------------------------------------------------------ #
# Defaults
# ------------------------------------------------------------------ #
DEFAULT_DIM = 32
DEFAULT_SEED = 42
DEFAULT_BUFFER_CAPACITY = 10_000
DEFAULT_SESSION_STEPS = 500       # online reading steps per session
DEFAULT_CONSOLIDATION_STEPS = 100  # offline replay steps per session
DEFAULT_EVAL_INTERVAL = 1_000     # compute internal metrics every N steps
DEFAULT_BETA_START = 1.0
DEFAULT_BETA_MIN = 0.01
DEFAULT_DECAY_STEPS = 50_000      # β decay horizon (slow)
DEFAULT_DECAY_TYPE = "linear"
DEFAULT_MAX_STEPS = 50_000
PROGRESS_INTERVAL = 100
PREVIEW_CHARS = 30
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "encyclopedia_run"
DEFAULT_CHECKPOINT_DIR = DEFAULT_OUTPUT_DIR / "checkpoints"


# ------------------------------------------------------------------ #
# Per-step record
# ------------------------------------------------------------------ #
@dataclass
class EncRecord:
    """One step of the encyclopedia learning loop."""
    step: int
    session_id: int
    nav_action: int
    nav_action_name: str
    article_title: str
    pos_in_article: int
    action_vec_norm: float
    free_energy: float
    prediction_error: float
    confidence: float
    obs_norm: float
    beta: float
    info_gain: float
    text_preview: str
    in_consolidation: bool = False
    delta_fe_consolidation: float = 0.0


@dataclass
class EncRunSummary:
    """Aggregated run statistics."""
    total_steps: int = 0
    n_sessions: int = 0
    n_consolidation_rounds: int = 0
    n_replayed_experiences: int = 0
    elapsed_s: float = 0.0
    initial_beta: float = 0.0
    final_beta: float = 0.0
    mean_free_energy: float = 0.0
    final_free_energy: float = 0.0
    mean_prediction_error: float = 0.0
    mean_confidence: float = 0.0
    action_counts: list[int] = field(default_factory=lambda: [0] * N_NAV_ACTIONS)
    articles_visited: set[str] = field(default_factory=set)
    n_nan_obs: int = 0
    n_nan_signal: int = 0
    n_crashes: int = 0
    crash_messages: list[str] = field(default_factory=list)
    delta_fe_per_session: list[float] = field(default_factory=list)
    # Internal-metrics history (Task 4): one row per eval_interval.
    metrics_history: list[dict] = field(default_factory=list)
    records: list[EncRecord] = field(default_factory=list)
    csv_path: str = ""
    png_path: str = ""
    manifest_path: str = ""
    final_checkpoint_path: str = ""


# ------------------------------------------------------------------ #
# Encyclopedia navigator: applies a Phase J navigation action
# ------------------------------------------------------------------ #
class EncyclopediaNavigator:
    """Wraps ``EncyclopediaReader`` with the 6-action navigation API.

    Separated from ``EncyclopediaLearningLoop`` so the navigation
    logic is testable in isolation.
    """

    def __init__(self, reader: EncyclopediaReader, rng: np.random.Generator):
        self.reader = reader
        self._rng = rng
        # Start at a random article.
        self.reader.seek_to_random_article()

    def navigate(self, action: int) -> tuple[str, str]:
        """Apply ``action`` and return (block, article_title).

        Side effects: updates the reader's position / current article.
        """
        if action == NAV_FORWARD:
            # If we're at the end of the article, jump to a random one.
            if self.reader.at_article_end():
                self.reader.seek_to_random_article()
            block = self.reader.read_block()
        elif action == NAV_BACKWARD:
            # Move backward within the article (clamped at 0).
            _, pos = self.reader.tell()
            new_pos = max(0, pos - 2 * self.reader.block_size)
            article_idx, _ = self.reader.tell()
            self.reader.seek(article_idx, new_pos)
            block = self.reader.read_block()
        elif action == NAV_FAST_FORWARD:
            # Skip 4 blocks ahead.
            _, pos = self.reader.tell()
            article_idx, _ = self.reader.tell()
            new_pos = pos + 4 * self.reader.block_size
            self.reader.seek(article_idx, new_pos)
            if self.reader.at_article_end():
                self.reader.seek_to_random_article()
            block = self.reader.read_block()
        elif action == NAV_JUMP_WITHIN:
            # Jump to a random position within the current article.
            body = self.reader._get_current_body()
            article_idx, _ = self.reader.tell()
            if body and len(body) > self.reader.block_size:
                max_pos = max(0, len(body) - self.reader.block_size)
                new_pos = int(self._rng.integers(0, max_pos + 1))
                self.reader.seek(article_idx, new_pos)
            block = self.reader.read_block()
        elif action == NAV_FOLLOW_LINK:
            # Pick a random linked article and jump to it.
            current = self.reader.get_title()
            links = self.reader.get_article_links(current)
            if links:
                target = str(self._rng.choice(links))
                if not self.reader.seek_to_article(target):
                    # Link target not in corpus — fall back to random.
                    self.reader.seek_to_random_article()
            else:
                self.reader.seek_to_random_article()
            block = self.reader.read_block()
        elif action == NAV_RANDOM_ARTICLE:
            self.reader.seek_to_random_article()
            block = self.reader.read_block()
        else:
            raise ValueError(
                f"unknown navigation action {action!r}; "
                f"expected 0..{N_NAV_ACTIONS - 1}"
            )
        return block, self.reader.get_title()

    def tell(self) -> tuple[int | None, int]:
        return self.reader.tell()

    def get_title(self) -> str:
        return self.reader.get_title()


# ------------------------------------------------------------------ #
# Internal-metrics evaluator (Task 4)
# ------------------------------------------------------------------ #
def compute_internal_metrics(
    model: ZeroDataModel,
    reader: EncyclopediaReader,
    encoder: TextEncoder,
    records: list[EncRecord],
    eval_step: int,
) -> dict:
    """Compute the Phase J internal-metrics panel.

    Computes:
      1. prediction_error_slope — slope of FE over the last N records.
      2. re_read_preference — fraction of recent actions that were
         ``NAV_BACKWARD`` or ``NAV_JUMP_WITHIN`` (re-reading).
      3. internal_consistency — mean cosine similarity of belief_states
         for linked article pairs vs random pairs.
      4. concept_stability — TBD (placeholder; would require tracking
         belief_state drift for the same article over time).
    """
    n = len(records)
    if n < 2:
        return {"step": eval_step, "note": "not enough records"}

    # 1. FE slope (last min(100, n) records).
    window = min(100, n)
    fes = np.array([r.free_energy for r in records[-window:]])
    steps_arr = np.arange(window)
    if np.var(fes) > 0:
        slope = float(np.polyfit(steps_arr, fes, 1)[0])
    else:
        slope = 0.0

    # 2. Re-read preference (last 100 actions).
    recent_actions = [r.nav_action for r in records[-window:]]
    re_read_count = sum(1 for a in recent_actions
                       if a in (NAV_BACKWARD, NAV_JUMP_WITHIN))
    re_read_pref = re_read_count / max(1, len(recent_actions))

    # 3. Internal consistency: linked pairs vs random pairs.
    # Sample up to 20 linked pairs and 20 random pairs, encode their
    # first block, compute cosine similarity of belief_states.
    engine = model.active_inference
    titles = reader.get_article_list()
    linked_pairs = []
    random_pairs = []
    for title in titles[:50]:
        links = reader.get_article_links(title)
        for link in links:
            if link in reader.get_article_list():
                linked_pairs.append((title, link))
    # Random pairs (shuffle titles and pair adjacent).
    titles_shuffled = list(titles)
    np.random.default_rng(42).shuffle(titles_shuffled)
    for i in range(0, len(titles_shuffled) - 1, 2):
        random_pairs.append((titles_shuffled[i], titles_shuffled[i + 1]))
    linked_pairs = linked_pairs[:20]
    random_pairs = random_pairs[:20]

    def _belief_for_title(title: str) -> np.ndarray | None:
        if not reader.seek_to_article(title):
            return None
        block = reader.read_block()
        if not block:
            return None
        obs = encoder.encode(block)
        # We do NOT call model.think here (would mutate state). Instead,
        # we use the engine's pure ``infer_state`` to get a one-shot
        # belief without updating. We use the CURRENT belief_state and
        # add a one-step gradient.
        gen = engine.generative_model
        inferred_state, _ = gen.infer_state(obs)
        return inferred_state

    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _mean_cos_for_pairs(pairs: list[tuple[str, str]]) -> float:
        sims = []
        for a, b in pairs:
            ba = _belief_for_title(a)
            bb = _belief_for_title(b)
            if ba is not None and bb is not None:
                sims.append(_cosine(ba, bb))
        return float(np.mean(sims)) if sims else 0.0

    linked_sim = _mean_cos_for_pairs(linked_pairs)
    random_sim = _mean_cos_for_pairs(random_pairs)
    consistency = linked_sim - random_sim

    return {
        "step": eval_step,
        "n_records": n,
        "fe_slope_recent": slope,
        "re_read_preference": re_read_pref,
        "linked_pair_cos_sim": linked_sim,
        "random_pair_cos_sim": random_sim,
        "consistency_score": consistency,
        "n_linked_pairs_evaluated": len(linked_pairs),
        "n_random_pairs_evaluated": len(random_pairs),
    }


# ------------------------------------------------------------------ #
# Checkpoint / resume
# ------------------------------------------------------------------ #
def save_checkpoint(
    path: Path,
    model: ZeroDataModel,
    reader: EncyclopediaReader,
    navigator: EncyclopediaNavigator,
    buffer: TextExperienceBuffer,
    summary: EncRunSummary,
    step: int,
    session_id: int,
    rng_state: dict,
) -> str:
    """Save a full checkpoint to ``path`` (a directory).

    Contents:
      - model.pkl       (ZeroDataModel pickle — full state)
      - reader_state.json (current article + position)
      - buffer.pkl      (ExperienceBuffer pickle)
      - summary.pkl     (EncRunSummary pickle)
      - meta.json       (step, session_id, timestamp, rng_state)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(parents=True, exist_ok=True)
    # Model.
    with (path / "model.pkl").open("wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    # Reader state.
    article_idx, pos_in_article = navigator.tell()
    reader_state = {
        "article_idx": article_idx,
        "pos_in_article": pos_in_article,
        "current_title": navigator.get_title(),
    }
    with (path / "reader_state.json").open("w") as f:
        json.dump(reader_state, f, indent=2)
    # Buffer.
    with (path / "buffer.pkl").open("wb") as f:
        pickle.dump(buffer, f, protocol=pickle.HIGHEST_PROTOCOL)
    # Summary.
    with (path / "summary.pkl").open("wb") as f:
        pickle.dump(summary, f, protocol=pickle.HIGHEST_PROTOCOL)
    # Meta.
    meta = {
        "step": int(step),
        "session_id": int(session_id),
        "timestamp": time.time(),
        "rng_state": rng_state,
        "schema_version": "1.0-encyclopedia",
    }
    with (path / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2, default=str)
    return str(path)


def load_checkpoint(path: Path) -> dict:
    """Load a checkpoint directory. Returns a dict with all components."""
    with (path / "model.pkl").open("rb") as f:
        model = pickle.load(f)
    with (path / "reader_state.json").open() as f:
        reader_state = json.load(f)
    with (path / "buffer.pkl").open("rb") as f:
        buffer = pickle.load(f)
    with (path / "summary.pkl").open("rb") as f:
        summary = pickle.load(f)
    with (path / "meta.json").open() as f:
        meta = json.load(f)
    return {
        "model": model,
        "reader_state": reader_state,
        "buffer": buffer,
        "summary": summary,
        "meta": meta,
    }


def latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Return the path of the latest checkpoint, or None."""
    if not checkpoint_dir.exists():
        return None
    checkpoints = sorted(checkpoint_dir.glob("ckpt_step*"))
    if not checkpoints:
        return None
    return checkpoints[-1]


# ------------------------------------------------------------------ #
# Main learning loop
# ------------------------------------------------------------------ #
class EncyclopediaLearningLoop:
    """The main session-based learning loop."""

    def __init__(
        self,
        model: ZeroDataModel,
        reader: EncyclopediaReader,
        encoder: TextEncoder,
        total_steps: int,
        session_steps: int,
        consolidation_steps: int,
        buffer_capacity: int,
        eval_interval: int,
        output_dir: Path,
        checkpoint_dir: Path,
        seed: int = DEFAULT_SEED,
    ):
        self.model = model
        self.reader = reader
        self.encoder = encoder
        self.total_steps = int(total_steps)
        self.session_steps = int(session_steps)
        self.consolidation_steps = int(consolidation_steps)
        self.eval_interval = int(eval_interval)
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._rng = np.random.default_rng(seed)
        self.navigator = EncyclopediaNavigator(reader, self._rng)
        self.buffer = TextExperienceBuffer(
            capacity=buffer_capacity,
            surprise_weighted=True,
            seed=seed,
        )
        self.consolidator = TextConsolidator(
            batch_size=16,
            sampling_mode="priority",
            seed=seed,
        )
        self.summary = EncRunSummary()
        self.start_step = 0
        self.session_id = 0

    # ------------------------------------------------------------------ #
    # Resume from checkpoint
    # ------------------------------------------------------------------ #
    def resume(self, ckpt_path: Path) -> None:
        """Restore state from ``ckpt_path``."""
        ckpt = load_checkpoint(ckpt_path)
        self.model = ckpt["model"]
        self.buffer = ckpt["buffer"]
        self.summary = ckpt["summary"]
        self.start_step = int(ckpt["meta"]["step"])
        self.session_id = int(ckpt["meta"]["session_id"])
        # Restore reader position.
        rs = ckpt["reader_state"]
        if rs.get("article_idx") is not None:
            self.reader.seek(rs["article_idx"], rs.get("pos_in_article", 0))
        print(f"[resume] restored from {ckpt_path} at step {self.start_step}")

    # ------------------------------------------------------------------ #
    # One online step
    # ------------------------------------------------------------------ #
    def _online_step(self, step: int) -> EncRecord:
        engine = self.model.active_inference
        # Determine action: bootstrap on the first ever step, otherwise
        # use the model's last selected candidate index.
        if step == 0:
            nav_action = NAV_FORWARD
        else:
            nav_action = int(engine._last_selected_idx)
            if not 0 <= nav_action < N_NAV_ACTIONS:
                nav_action = NAV_FORWARD
        # Apply the navigation.
        text_block, article_title = self.navigator.navigate(nav_action)
        pos_in_article = self.navigator.tell()[1]
        # Encode + think.
        obs = self.encoder.encode(text_block)
        if not np.all(np.isfinite(obs)):
            self.summary.n_nan_obs += 1
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        obs_norm = float(np.linalg.norm(obs))
        signal = self.model.think(obs)
        if not np.all(np.isfinite(signal.data)):
            self.summary.n_nan_signal += 1
        belief = engine.generative_model.belief_state
        action_vec = engine.select_action(belief, current_observation=obs)
        action_vec_norm = float(np.linalg.norm(action_vec))
        free_energy = float(engine.compute_free_energy(obs))
        belief_after = engine.generative_model.belief_state
        b = belief_after[:obs.shape[0]]
        prediction_error = float(np.linalg.norm(obs - b))
        beta = self._current_beta()
        info_gain = self._current_info_gain(action_vec)
        # Store the experience.
        self.buffer.add(
            obs=obs, action=nav_action, next_obs=obs,
            error=free_energy, info_gain=info_gain, step=step,
        )
        self.summary.articles_visited.add(article_title)
        preview = text_block[:PREVIEW_CHARS].replace("\n", " ").replace("\r", "")
        return EncRecord(
            step=step,
            session_id=self.session_id,
            nav_action=nav_action,
            nav_action_name=NAV_ACTION_NAMES[nav_action],
            article_title=article_title,
            pos_in_article=pos_in_article,
            action_vec_norm=action_vec_norm,
            free_energy=free_energy,
            prediction_error=prediction_error,
            confidence=float(signal.confidence),
            obs_norm=obs_norm,
            beta=beta,
            info_gain=info_gain,
            text_preview=preview,
        )

    # ------------------------------------------------------------------ #
    # One consolidation step
    # ------------------------------------------------------------------ #
    def _consolidation_step(self, step: int, session_step: int) -> float:
        """Run one consolidation round. Returns the ΔFE."""
        if len(self.buffer) == 0:
            return 0.0
        result = self.consolidator.consolidate(self.model, self.buffer)
        self.summary.n_consolidation_rounds += 1
        self.summary.n_replayed_experiences += result.n_replayed
        return float(result.delta_free_energy)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _current_beta(self) -> float:
        return float(self.model.active_inference._compute_beta())

    def _current_info_gain(self, action_vec: np.ndarray) -> float:
        return float(self.model.active_inference._compute_information_gain_proxy(action_vec))

    # ------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------ #
    def run(self) -> EncRunSummary:
        engine = self.model.active_inference
        self.summary.initial_beta = self._current_beta()
        t0 = time.time()
        records: list[EncRecord] = self.summary.records
        # If resuming, records already has entries — don't duplicate.
        step = self.start_step
        next_eval_step = (step // self.eval_interval + 1) * self.eval_interval
        try:
            while step < self.total_steps:
                # Determine if we're in the online or consolidation phase.
                session_progress = step % (self.session_steps + self.consolidation_steps)
                in_consolidation = session_progress >= self.session_steps
                if not in_consolidation:
                    rec = self._online_step(step)
                    rec.session_id = self.session_id
                    records.append(rec)
                    self.summary.action_counts[rec.nav_action] += 1
                else:
                    # Consolidation phase.
                    delta = self._consolidation_step(step, session_progress)
                    # Record a placeholder entry for the consolidation step.
                    rec = EncRecord(
                        step=step,
                        session_id=self.session_id,
                        nav_action=-1,  # sentinel: consolidation
                        nav_action_name="consolidation",
                        article_title=self.navigator.get_title(),
                        pos_in_article=self.navigator.tell()[1],
                        action_vec_norm=0.0,
                        free_energy=float(engine.compute_free_energy(
                            np.zeros(self.model.dim)
                        )),
                        prediction_error=0.0,
                        confidence=0.0,
                        obs_norm=0.0,
                        beta=self._current_beta(),
                        info_gain=0.0,
                        text_preview="",
                        in_consolidation=True,
                        delta_fe_consolidation=delta,
                    )
                    records.append(rec)
                step += 1
                # Progress print.
                if step % PROGRESS_INTERVAL == 0:
                    last_rec = records[-1]
                    print(f"  step {step:6d} | session {self.session_id} | "
                          f"article={last_rec.article_title[:30]:30s} | "
                          f"action={last_rec.nav_action_name:14s} | "
                          f"FE={last_rec.free_energy:.3f} | β={last_rec.beta:.3f}")
                # Periodic eval.
                if step >= next_eval_step:
                    metrics = compute_internal_metrics(
                        self.model, self.reader, self.encoder, records, step,
                    )
                    self.summary.metrics_history.append(metrics)
                    print(f"  [eval @ step {step}] consistency={metrics.get('consistency_score', 0):.4f}  "
                          f"fe_slope={metrics.get('fe_slope_recent', 0):.4f}  "
                          f"re_read_pref={metrics.get('re_read_preference', 0):.3f}")
                    next_eval_step += self.eval_interval
                # End-of-session checkpoint.
                session_total = self.session_steps + self.consolidation_steps
                if step % session_total == 0:
                    self.session_id += 1
                    ckpt_path = self.checkpoint_dir / f"ckpt_step{step:06d}"
                    save_checkpoint(
                        ckpt_path, self.model, self.reader, self.navigator,
                        self.buffer, self.summary, step, self.session_id,
                        rng_state={},
                    )
                    self.summary.final_checkpoint_path = str(ckpt_path)
                    print(f"  [checkpoint] saved {ckpt_path}")
        except KeyboardInterrupt:
            print(f"\n[ctrl-c] saving final checkpoint at step {step}...")
            ckpt_path = self.checkpoint_dir / f"ckpt_step{step:06d}_interrupt"
            save_checkpoint(
                ckpt_path, self.model, self.reader, self.navigator,
                self.buffer, self.summary, step, self.session_id,
                rng_state={},
            )
            self.summary.final_checkpoint_path = str(ckpt_path)
            print(f"  [checkpoint] saved {ckpt_path}")
        # Final stats.
        self.summary.total_steps = step
        self.summary.n_sessions = self.session_id
        self.summary.elapsed_s = time.time() - t0
        self.summary.final_beta = self._current_beta()
        if records:
            fes = [r.free_energy for r in records if not r.in_consolidation]
            if fes:
                self.summary.mean_free_energy = float(np.mean(fes))
                self.summary.final_free_energy = float(fes[-1])
            pes = [r.prediction_error for r in records if not r.in_consolidation]
            if pes:
                self.summary.mean_prediction_error = float(np.mean(pes))
            confs = [r.confidence for r in records if not r.in_consolidation]
            if confs:
                self.summary.mean_confidence = float(np.mean(confs))
        return self.summary


# ------------------------------------------------------------------ #
# Output: CSV + PNG + manifest
# ------------------------------------------------------------------ #
def save_csv(summary: EncRunSummary, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "step", "session_id", "nav_action", "nav_action_name",
            "article_title", "pos_in_article",
            "action_vec_norm", "free_energy", "prediction_error",
            "confidence", "obs_norm", "beta", "info_gain",
            "text_preview", "in_consolidation", "delta_fe_consolidation",
        ])
        for r in summary.records:
            w.writerow([
                r.step, r.session_id, r.nav_action, r.nav_action_name,
                r.article_title, r.pos_in_article,
                f"{r.action_vec_norm:.6f}", f"{r.free_energy:.6f}",
                f"{r.prediction_error:.6f}", f"{r.confidence:.6f}",
                f"{r.obs_norm:.6f}", f"{r.beta:.6f}", f"{r.info_gain:.6f}",
                r.text_preview, int(r.in_consolidation),
                f"{r.delta_fe_consolidation:.6f}",
            ])
    summary.csv_path = str(path)
    return summary.csv_path


def save_metrics_csv(summary: EncRunSummary, path: Path) -> str:
    """Save the periodic internal-metrics history to a CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "step", "n_records", "fe_slope_recent",
            "re_read_preference", "linked_pair_cos_sim",
            "random_pair_cos_sim", "consistency_score",
            "n_linked_pairs_evaluated", "n_random_pairs_evaluated",
        ])
        for m in summary.metrics_history:
            w.writerow([
                m.get("step", ""), m.get("n_records", ""),
                f"{m.get('fe_slope_recent', 0):.6f}",
                f"{m.get('re_read_preference', 0):.6f}",
                f"{m.get('linked_pair_cos_sim', 0):.6f}",
                f"{m.get('random_pair_cos_sim', 0):.6f}",
                f"{m.get('consistency_score', 0):.6f}",
                m.get("n_linked_pairs_evaluated", 0),
                m.get("n_random_pairs_evaluated", 0),
            ])
    return str(path)


def plot_run(summary: EncRunSummary, path: Path) -> str:
    """6-panel PNG: FE, β, action dist, consistency, re-read, articles."""
    if not _HAS_MPL or not summary.records:
        return ""
    path.parent.mkdir(parents=True, exist_ok=True)
    online = [r for r in summary.records if not r.in_consolidation]
    cons = [r for r in summary.records if r.in_consolidation]
    steps = [r.step for r in online]
    fes = [r.free_energy for r in online]
    betas = [r.beta for r in online]
    cons_steps = [r.step for r in cons]
    cons_deltas = [r.delta_fe_consolidation for r in cons]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # Panel 1: FE + consolidation deltas
    ax = axes[0, 0]
    ax.plot(steps, fes, "-", color="#d62728", linewidth=0.8, label="free energy")
    for s in cons_steps:
        ax.axvline(s, color="gray", alpha=0.15, linewidth=0.5)
    ax.set_xlabel("step")
    ax.set_ylabel("FE")
    ax.set_title(f"Free energy (mean={summary.mean_free_energy:.3f}, "
                 f"final={summary.final_free_energy:.3f})")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Panel 2: β decay
    ax = axes[0, 1]
    ax.plot(steps, betas, "-", color="#2ca02c", linewidth=1)
    ax.set_xlabel("step")
    ax.set_ylabel("β")
    ax.set_title(f"β decay: {summary.initial_beta:.3f} -> {summary.final_beta:.3f}")
    ax.grid(True, alpha=0.3)

    # Panel 3: action distribution
    ax = axes[1, 0]
    counts = summary.action_counts
    total = max(1, sum(counts))
    fractions = [c / total for c in counts]
    x = np.arange(N_NAV_ACTIONS)
    bars = ax.bar(x, fractions, color="#ff7f0e", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(NAV_ACTION_NAMES, rotation=30, ha="right")
    ax.set_ylabel("fraction")
    ax.set_title("Action distribution")
    for bar, c in zip(bars, counts, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{c}", ha="center", va="bottom", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 4: internal consistency over time
    ax = axes[1, 1]
    if summary.metrics_history:
        m_steps = [m["step"] for m in summary.metrics_history]
        m_consistency = [m.get("consistency_score", 0) for m in summary.metrics_history]
        ax.plot(m_steps, m_consistency, "o-", color="#9467bd", linewidth=1.5)
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("step")
    ax.set_ylabel("consistency (linked - random)")
    ax.set_title("Internal consistency (positive = linked pairs more similar)")
    ax.grid(True, alpha=0.3)

    # Panel 5: re-read preference over time
    ax = axes[2, 0]
    if summary.metrics_history:
        m_steps = [m["step"] for m in summary.metrics_history]
        m_reread = [m.get("re_read_preference", 0) for m in summary.metrics_history]
        ax.plot(m_steps, m_reread, "o-", color="#1f77b4", linewidth=1.5)
    ax.set_xlabel("step")
    ax.set_ylabel("re-read fraction")
    ax.set_title("Re-read preference (NAV_BACKWARD + NAV_JUMP_WITHIN)")
    ax.grid(True, alpha=0.3)

    # Panel 6: consolidation ΔFE
    ax = axes[2, 1]
    if cons_steps:
        ax.plot(cons_steps, cons_deltas, "o", color="#9467bd", markersize=3)
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("step")
    ax.set_ylabel("ΔFE per consolidation round")
    ax.set_title("Consolidation effect (negative = improvement)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    summary.png_path = str(path)
    return summary.png_path


def save_manifest(summary: EncRunSummary, path: Path, args: argparse.Namespace) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "1.0-encyclopedia",
        "total_steps": summary.total_steps,
        "n_sessions": summary.n_sessions,
        "n_consolidation_rounds": summary.n_consolidation_rounds,
        "n_replayed_experiences": summary.n_replayed_experiences,
        "elapsed_s": summary.elapsed_s,
        "initial_beta": summary.initial_beta,
        "final_beta": summary.final_beta,
        "mean_free_energy": summary.mean_free_energy,
        "final_free_energy": summary.final_free_energy,
        "mean_prediction_error": summary.mean_prediction_error,
        "mean_confidence": summary.mean_confidence,
        "action_counts": dict(zip(NAV_ACTION_NAMES, summary.action_counts)),
        "n_articles_visited": len(summary.articles_visited),
        "articles_visited": sorted(summary.articles_visited),
        "n_nan_obs": summary.n_nan_obs,
        "n_nan_signal": summary.n_nan_signal,
        "n_crashes": summary.n_crashes,
        "metrics_history": summary.metrics_history,
        "final_checkpoint_path": summary.final_checkpoint_path,
        "args": {
            "corpus_dir": args.corpus_dir,
            "total_steps": args.total_steps,
            "session_steps": args.session_steps,
            "consolidation_steps": args.consolidation_steps,
            "buffer_capacity": args.buffer_capacity,
            "eval_interval": args.eval_interval,
            "dim": args.dim,
            "seed": args.seed,
        },
    }
    with path.open("w") as f:
        json.dump(manifest, f, indent=2, default=str)
    summary.manifest_path = str(path)
    return summary.manifest_path


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the long-term encyclopedia learning loop.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--corpus_dir", type=str, default="",
                   help="Directory containing wiki_* corpus files. Empty = synthetic.")
    p.add_argument("--total_steps", type=int, default=DEFAULT_MAX_STEPS)
    p.add_argument("--session_steps", type=int, default=DEFAULT_SESSION_STEPS)
    p.add_argument("--consolidation_steps", type=int, default=DEFAULT_CONSOLIDATION_STEPS)
    p.add_argument("--buffer_capacity", type=int, default=DEFAULT_BUFFER_CAPACITY)
    p.add_argument("--eval_interval", type=int, default=DEFAULT_EVAL_INTERVAL)
    p.add_argument("--block_size", type=int, default=DEFAULT_BLOCK_SIZE)
    p.add_argument("--dim", type=int, default=DEFAULT_DIM)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--beta_start", type=float, default=DEFAULT_BETA_START)
    p.add_argument("--beta_min", type=float, default=DEFAULT_BETA_MIN)
    p.add_argument("--decay_steps", type=int, default=DEFAULT_DECAY_STEPS)
    p.add_argument("--decay_type", type=str, default=DEFAULT_DECAY_TYPE,
                   choices=["linear", "exponential", "stage"])
    p.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--checkpoint_dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR))
    p.add_argument("--resume", action="store_true",
                   help="Resume from the latest checkpoint.")
    return p.parse_args()


def _build_corpus(corpus_dir: Path, seed: int) -> list[Path]:
    """Locate corpus files in ``corpus_dir`` or build a synthetic one."""
    if not corpus_dir.exists() or not any(corpus_dir.iterdir()):
        print(f"[info] corpus dir {corpus_dir} empty/missing; "
              f"generating synthetic corpus.")
        paths = make_synthetic_corpus(corpus_dir, n_files=4, articles_per_file=25, seed=seed)
        return paths
    # Look for wiki_* files.
    paths = sorted(corpus_dir.glob("wiki_*"))
    if not paths:
        # Fall back to any text file in the dir.
        paths = sorted(p for p in corpus_dir.iterdir()
                      if p.is_file() and p.suffix in {".txt", "", ".bz2", ".gz"})
    if not paths:
        paths = make_synthetic_corpus(corpus_dir, n_files=4, articles_per_file=25, seed=seed)
    return paths


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Long-term Encyclopedia Learning Loop (Phase J)")
    print("=" * 72)

    # Build or locate corpus.
    if args.corpus_dir:
        corpus_dir = Path(args.corpus_dir)
    else:
        corpus_dir = output_dir / "synthetic_corpus"
    corpus_paths = _build_corpus(corpus_dir, args.seed)
    print(f"  corpus: {len(corpus_paths)} files in {corpus_dir}")

    # Build reader.
    index_dir = corpus_dir / ".index"
    reader = EncyclopediaReader(
        corpus_paths, block_size=args.block_size,
        index_dir=index_dir, seed=args.seed,
    )
    print(f"  index: {reader.n_articles()} articles")

    # Build model + encoder.
    model = make_text_curious_model(
        dim=args.dim, seed=args.seed,
        beta_start=args.beta_start, beta_min=args.beta_min,
        decay_steps=args.decay_steps, decay_type=args.decay_type,
        num_candidates=N_NAV_ACTIONS,
    )
    encoder = TextEncoder(
        block_size=args.block_size, output_dim=args.dim, seed=args.seed,
    )
    print(f"  model dim: {args.dim}, num_candidates: {N_NAV_ACTIONS}")

    # Build the loop.
    loop = EncyclopediaLearningLoop(
        model, reader, encoder,
        total_steps=args.total_steps,
        session_steps=args.session_steps,
        consolidation_steps=args.consolidation_steps,
        buffer_capacity=args.buffer_capacity,
        eval_interval=args.eval_interval,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        seed=args.seed,
    )

    # Resume?
    if args.resume:
        ckpt = latest_checkpoint(checkpoint_dir)
        if ckpt:
            loop.resume(ckpt)
        else:
            print("[resume] no checkpoint found; starting fresh.")

    # Run.
    print(f"\nRunning {args.total_steps} steps "
          f"(sessions of {args.session_steps}+{args.consolidation_steps})...")
    summary = loop.run()

    # Save outputs.
    csv_path = save_csv(summary, output_dir / "encyclopedia_run.csv")
    print(f"\nCSV: {csv_path}")
    metrics_csv_path = save_metrics_csv(summary, output_dir / "encyclopedia_metrics.csv")
    print(f"Metrics CSV: {metrics_csv_path}")
    png_path = plot_run(summary, output_dir / "encyclopedia_run.png")
    if png_path:
        print(f"PNG: {png_path}")
    manifest_path = save_manifest(summary, output_dir / "manifest.json", args)
    print(f"Manifest: {manifest_path}")
    print(f"Final checkpoint: {summary.final_checkpoint_path}")

    # Print summary.
    print("\n" + "=" * 72)
    print("Final summary")
    print("=" * 72)
    print(f"  steps                  = {summary.total_steps}")
    print(f"  sessions               = {summary.n_sessions}")
    print(f"  consolidation rounds   = {summary.n_consolidation_rounds}")
    print(f"  experiences replayed   = {summary.n_replayed_experiences}")
    print(f"  elapsed                = {summary.elapsed_s:.1f}s")
    print(f"  cycles/sec             = {summary.total_steps / max(1, summary.elapsed_s):.1f}")
    print(f"  articles visited       = {len(summary.articles_visited)}")
    print(f"  initial β              = {summary.initial_beta:.4f}")
    print(f"  final β                = {summary.final_beta:.4f}")
    print(f"  mean FE                = {summary.mean_free_energy:.4f}")
    print(f"  final FE               = {summary.final_free_energy:.4f}")
    print(f"  mean pred error        = {summary.mean_prediction_error:.4f}")
    print(f"  mean confidence        = {summary.mean_confidence:.4f}")
    print(f"  action counts          = {dict(zip(NAV_ACTION_NAMES, summary.action_counts))}")
    if summary.metrics_history:
        last = summary.metrics_history[-1]
        print(f"  last consistency       = {last.get('consistency_score', 0):.4f}")
        print(f"  last re_read_pref      = {last.get('re_read_preference', 0):.3f}")
        print(f"  last fe_slope           = {last.get('fe_slope_recent', 0):.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
