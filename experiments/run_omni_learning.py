# experiments/run_omni_learning.py
"""Full omni-modal learning loop (Phase K, Task 4).

Integrates all Phase K components into one closed-loop experiment:

    OmniModalGenerator.sample()  -> (vec, modality, source)
        ↓
    ZeroDataModel.think(vec)     -> signal     # cognitive cycle
        ↓
    KnowledgeGraphBuilder.update_graph(...)
    TextDecoder.ingest(text, latent)         # only for text modality
        ↓
    Periodic: InquiryLoop.run(n_questions=5)   # active inquiry every N steps
    Periodic: InquiryLoop.propose_hypothesis() # hypothesis test every M steps
        ↓
    CSV log row + JSON metrics + knowledge graph snapshot

Modality mix (default, configurable via ``--ratios``):
    text    60 %    encyclopedia articles + (optional) text stream
    physics 20 %   physics-sandbox frames (encoded as images)
    image   10 %   image stream (synthetic if no real images provided)
    audio    5 %   audio stream (synthetic if no real audio provided)
    code     5 %   code stream (synthetic if no real code provided)

Smoke test (default):
    python experiments/run_omni_learning.py --steps 300

Production:
    python experiments/run_omni_learning.py --steps 50000 \\
        --corpus_dir path/to/wiki_dump --image_dir path/to/images
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

# Make sibling modules importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Phase H / J infrastructure.
from encyclopedia_reader import EncyclopediaReader, make_synthetic_corpus
from knowledge_graph_builder import KnowledgeGraphBuilder
from run_sandbox_curious import make_curious_model

# Phase K modules.
from audio_encoder import AudioEncoder
from code_encoder import CodeEncoder
from image_encoder import ImageEncoder
from omnimodal_data_generator import (
    DEFAULT_RATIOS,
    MOD_AUDIO,
    MOD_CODE,
    MOD_IMAGE,
    MOD_PHYSICS,
    MOD_TEXT,
    EncyclopediaStreamAdapter,
    OmniModalGenerator,
    PhysicsStreamAdapter,
)
from question_generator import QuestionGenerator
from text_decoder import TextDecoder
from text_encoder import TextEncoder
from active_inquiry import InquiryLoop


# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #
DEFAULT_DIM = 32
DEFAULT_SEED = 42
DEFAULT_STEPS = 300                 # smoke test
INQUIRY_INTERVAL = 1000
INQUIRY_N_QUESTIONS = 5
HYPOTHESIS_INTERVAL = 5000
HYPOTHESIS_N = 1
PROGRESS_INTERVAL = 50
PREVIEW_CHARS = 40
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "omni_run"


# ------------------------------------------------------------------ #
# Per-step record
# ------------------------------------------------------------------ #
@dataclass
class OmniCycleRecord:
    step: int
    modality: str
    source: str
    obs_norm: float
    confidence: float
    free_energy: float
    prediction_error: float
    cycle_count: int
    preview: str = ""


# ------------------------------------------------------------------ #
# Per-inquiry / per-hypothesis records
# ------------------------------------------------------------------ #
@dataclass
class OmniInquiryRecord:
    step: int
    n_questions: int
    n_answered: int
    qa_preview: list[dict] = field(default_factory=list)


@dataclass
class OmniHypothesisRecord:
    step: int
    n_hypotheses: int
    n_supported: int
    hypotheses_preview: list[dict] = field(default_factory=list)


# ------------------------------------------------------------------ #
# Run summary
# ------------------------------------------------------------------ #
@dataclass
class OmniRunSummary:
    n_steps: int = 0
    n_crashes: int = 0
    n_nan_obs: int = 0
    n_nan_signal: int = 0
    elapsed_s: float = 0.0
    cycles_per_sec: float = 0.0
    mean_free_energy: float = 0.0
    final_free_energy: float = 0.0
    mean_prediction_error: float = 0.0
    mean_confidence: float = 0.0
    modality_counts: dict[str, int] = field(default_factory=dict)
    n_inquiry_runs: int = 0
    n_questions_asked: int = 0
    n_questions_answered: int = 0
    n_hypothesis_runs: int = 0
    n_hypotheses_tested: int = 0
    n_hypotheses_supported: int = 0
    final_kg_nodes: int = 0
    final_kg_edges: int = 0
    final_decoder_pool_size: int = 0


# ------------------------------------------------------------------ #
# Physics adapter — wraps PhysicsSandbox in a stream interface
# ------------------------------------------------------------------ #
class SandboxStreamAdapter:
    """Adapt ``PhysicsSandbox`` to the ``.step() -> (vec, src)`` interface.

    Uses the supplied image encoder to convert each rendered frame
    into a latent. Random-action stepping keeps the sandbox producing
    fresh frames even when no model is driving it.
    """

    def __init__(self, sandbox, image_encoder, action_dim: int = 4, seed: int = 42):
        self.sandbox = sandbox
        self.encoder = image_encoder
        self._rng = np.random.default_rng(seed)
        self._step_count = 0

    def __len__(self) -> int:
        return 0  # unbounded

    def reset(self) -> None:
        self._step_count = 0
        try:
            self.sandbox.reset()
        except Exception:
            pass

    def step(self) -> tuple[np.ndarray, str]:
        # Random discrete action (PhysicsSandbox uses int actions 0..3).
        action = int(self._rng.integers(0, 4))
        try:
            frame = self.sandbox.step(action)
        except Exception:
            return np.zeros(self.encoder.output_dim, dtype=np.float64), "physics:error"
        self._step_count += 1
        # Encode the frame via the image encoder.
        vec = self.encoder.encode(np.asarray(frame))
        return vec, f"physics:step_{self._step_count}"


# ------------------------------------------------------------------ #
# The main learning loop
# ------------------------------------------------------------------ #
class OmniLearningLoop:
    """Drives the full omni-modal cognitive cycle.

    Lifecycle:
        loop = OmniLearningLoop(model, generator, kg_builder, decoder, ...)
        summary = loop.run()
        loop.save_outputs()

    FE definition (Task G fix): uses the pragmatic term
    ``||obs - belief @ emission||^2 / dim`` and applies BELIEF_DECAY=0.99
    after each ``think()`` call to emulate the missing KL gradient in
    ``update_belief`` (preventing ||belief|| from drifting unboundedly).
    """

    BELIEF_DECAY = 0.99  # emulates missing KL gradient in update_belief

    def __init__(
        self,
        model,
        generator: OmniModalGenerator,
        kg_builder: KnowledgeGraphBuilder,
        decoder: TextDecoder,
        text_encoder: TextEncoder,
        reader: EncyclopediaReader | None = None,
        max_steps: int = DEFAULT_STEPS,
        inquiry_interval: int = INQUIRY_INTERVAL,
        hypothesis_interval: int = HYPOTHESIS_INTERVAL,
        seed: int = DEFAULT_SEED,
        output_dir: Path = OUTPUT_DIR,
    ):
        self.model = model
        self.generator = generator
        self.kg_builder = kg_builder
        self.decoder = decoder
        self.text_encoder = text_encoder
        self.reader = reader
        self.max_steps = int(max_steps)
        self.inquiry_interval = int(inquiry_interval)
        self.hypothesis_interval = int(hypothesis_interval)
        self.seed = int(seed)
        self.output_dir = Path(output_dir)
        self.summary = OmniRunSummary()
        self.records: list[OmniCycleRecord] = []
        self.inquiry_records: list[OmniInquiryRecord] = []
        self.hypothesis_records: list[OmniHypothesisRecord] = []
        self._qa_log: list[dict] = []      # accumulated Q/A pairs
        self._h_log: list[dict] = []        # accumulated hypotheses
        self._inquiry_loop: InquiryLoop | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self) -> OmniRunSummary:
        t0 = time.perf_counter()
        # Pre-build the inquiry loop lazily (only when first needed).
        # NOTE: We deliberately pass ``question_generator=None`` so the
        # InquiryLoop can auto-build a QuestionGenerator with the
        # ``allowed_node_test`` filter derived from ``reader.get_article_list()``.
        # This ensures generated questions target only encyclopedia article
        # titles that actually exist in the corpus, so the search step can
        # find a matching article and the loop has a chance of answering.
        if self.reader is not None and self.inquiry_interval <= self.max_steps:
            self._inquiry_loop = InquiryLoop(
                model=self.model,
                text_encoder=self.text_encoder,
                decoder=self.decoder,
                kg_builder=self.kg_builder,
                reader=self.reader,
                question_generator=None,
                max_read_blocks=3,
                pred_err_threshold=0.0,
                seed=self.seed,
            )
        for step in range(self.max_steps):
            try:
                self._cycle(step)
            except Exception as exc:  # noqa: BLE001 — defensive
                self.summary.n_crashes += 1
                if self.summary.n_crashes <= 5:
                    print(f"[crash step={step}] {type(exc).__name__}: {exc}",
                          file=sys.stderr)
            # Periodic active inquiry.
            if (self._inquiry_loop is not None
                    and step > 0
                    and (step + 1) % self.inquiry_interval == 0):
                self._run_inquiry(step + 1)
            # Periodic hypothesis test.
            if (self._inquiry_loop is not None
                    and step > 0
                    and (step + 1) % self.hypothesis_interval == 0):
                self._run_hypothesis(step + 1)
            # Progress.
            if (step + 1) % PROGRESS_INTERVAL == 0:
                self._print_progress(step + 1, t0)
        elapsed = time.perf_counter() - t0
        self.summary.n_steps = self.max_steps
        self.summary.elapsed_s = elapsed
        self.summary.cycles_per_sec = self.max_steps / elapsed if elapsed > 0 else 0.0
        if self.records:
            fes = [r.free_energy for r in self.records]
            pes = [r.prediction_error for r in self.records]
            confs = [r.confidence for r in self.records]
            self.summary.mean_free_energy = float(np.mean(fes))
            self.summary.final_free_energy = float(fes[-1])
            self.summary.mean_prediction_error = float(np.mean(pes))
            self.summary.mean_confidence = float(np.mean(confs))
        self.summary.modality_counts = dict(self.generator.modality_counts)
        self.summary.n_inquiry_runs = len(self.inquiry_records)
        self.summary.n_hypothesis_runs = len(self.hypothesis_records)
        self.summary.final_kg_nodes = self.kg_builder.n_nodes()
        self.summary.final_kg_edges = self.kg_builder.n_edges()
        self.summary.final_decoder_pool_size = len(self.decoder.pool)
        return self.summary

    # ------------------------------------------------------------------ #
    # One perception-cognition cycle
    # ------------------------------------------------------------------ #
    def _cycle(self, step: int) -> None:
        # 1. SAMPLE: pick a modality + observation from the generator.
        sample = self.generator.sample()
        obs = sample.vector
        if obs.size == 0 or not np.all(np.isfinite(obs)):
            self.summary.n_nan_obs += 1
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        obs_norm = float(np.linalg.norm(obs))
        # 2. COGNITION: feed the observation to the model.
        signal = self.model.think(obs)
        # BELIEF_DECAY: emulate the missing KL gradient in update_belief.
        # Without this, ||belief_state|| drifts unboundedly across steps
        # (the pragmatic-only update_belief has no -state pull-back).
        try:
            gm = self.model.active_inference.generative_model
            gm.belief_state = gm.belief_state * self.BELIEF_DECAY
        except Exception:
            pass
        if not np.all(np.isfinite(signal.data)):
            self.summary.n_nan_signal += 1
        confidence = float(signal.confidence)
        cycle_count = int(signal.metadata.get("cycle", 0))
        # 3. POST-PROCESS: update KG + decoder pool based on modality.
        free_energy = self._read_free_energy(obs)
        pred_err = self._read_prediction_error(obs)
        # Build a clean node title for the KG. For wiki: titles, use
        # the title verbatim. For file paths, use the stem (basename
        # without extension). This makes the question generator's
        # ``_search_corpus`` matching work for non-text modalities too.
        node_title = self._clean_source_label(sample.source, sample.modality)
        # Update KG (treat every observation as a node keyed by source).
        # For text, we also ingest into the decoder's retrieval pool.
        self.kg_builder.update_graph(
            article_title=node_title,
            linked_titles=[],
            latent_state=obs,
            free_energy=free_energy,
            read_step=step,
        )
        if sample.modality == MOD_TEXT:
            # Recover the text block (heuristic: use the source label).
            # The encyclopedia adapter already produced a vector from
            # a text block; we re-encode the article title for the
            # decoder pool (the actual block text isn't returned by
            # the stream — but the decoder pool's retrieval still
            # works because we ingest a synthetic text snippet built
            # from the source label + modality).
            text = self._synthetic_text_for_source(sample.source, sample.modality)
            self.decoder.ingest(text, latent=obs, source=node_title,
                                free_energy=free_energy)
        # 4. RECORD.
        preview = sample.source[:PREVIEW_CHARS]
        self.records.append(OmniCycleRecord(
            step=step,
            modality=sample.modality,
            source=sample.source,
            obs_norm=obs_norm,
            confidence=confidence,
            free_energy=free_energy,
            prediction_error=pred_err,
            cycle_count=cycle_count,
            preview=preview,
        ))

    # ------------------------------------------------------------------ #
    # Periodic: active inquiry
    # ------------------------------------------------------------------ #
    def _run_inquiry(self, step: int) -> None:
        try:
            summary = self._inquiry_loop.run(n_questions=INQUIRY_N_QUESTIONS)
        except Exception as exc:  # noqa: BLE001 — defensive
            print(f"[inquiry crash step={step}] {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            return
        self.summary.n_questions_asked += summary["n_questions"]
        self.summary.n_questions_answered += summary["n_answered"]
        self.inquiry_records.append(OmniInquiryRecord(
            step=step,
            n_questions=summary["n_questions"],
            n_answered=summary["n_answered"],
            qa_preview=summary["qa"][:3],
        ))
        self._qa_log.extend(summary["qa"])
        print(f"[inquiry step={step}] {summary['n_answered']}/{summary['n_questions']} answered")

    # ------------------------------------------------------------------ #
    # Periodic: hypothesis
    # ------------------------------------------------------------------ #
    def _run_hypothesis(self, step: int) -> None:
        try:
            hyps = self._inquiry_loop.propose_hypothesis(max_hypotheses=HYPOTHESIS_N)
        except Exception as exc:  # noqa: BLE001 — defensive
            print(f"[hypothesis crash step={step}] {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            return
        n_supported = sum(1 for h in hyps if h.supported)
        self.summary.n_hypotheses_tested += len(hyps)
        self.summary.n_hypotheses_supported += n_supported
        self.hypothesis_records.append(OmniHypothesisRecord(
            step=step,
            n_hypotheses=len(hyps),
            n_supported=n_supported,
            hypotheses_preview=[asdict(h) for h in hyps[:3]],
        ))
        self._h_log.extend([asdict(h) for h in hyps])
        print(f"[hypothesis step={step}] {n_supported}/{len(hyps)} supported")

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #
    def save_outputs(self) -> dict[str, str]:
        """Save CSV log, JSON summary, KG, transcript."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, str] = {}
        # CSV log of per-step records.
        csv_path = self.output_dir / "omni_log.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "step", "modality", "source", "obs_norm", "confidence",
                "free_energy", "prediction_error", "cycle_count",
            ])
            for r in self.records:
                w.writerow([
                    r.step, r.modality, r.source, f"{r.obs_norm:.6f}",
                    f"{r.confidence:.6f}", f"{r.free_energy:.6f}",
                    f"{r.prediction_error:.6f}", r.cycle_count,
                ])
        paths["csv"] = str(csv_path)
        # JSON summary.
        summary_path = self.output_dir / "summary.json"
        with summary_path.open("w") as f:
            json.dump(asdict(self.summary), f, indent=2, default=str)
        paths["summary"] = str(summary_path)
        # Knowledge graph JSON.
        kg_path = self.output_dir / "knowledge_graph.json"
        self.kg_builder.save_json(kg_path)
        paths["kg"] = str(kg_path)
        # Compute latent edges before saving (so the graph has them).
        try:
            self.kg_builder.compute_latent_edges(top_k=3)
        except Exception:
            pass
        # Q/A transcript.
        if self._qa_log:
            qa_path = self.output_dir / "qa_transcript.json"
            with qa_path.open("w") as f:
                json.dump(self._qa_log, f, indent=2, default=str)
            paths["qa"] = str(qa_path)
        # Hypothesis transcript.
        if self._h_log:
            h_path = self.output_dir / "hypothesis_transcript.json"
            with h_path.open("w") as f:
                json.dump(self._h_log, f, indent=2, default=str)
            paths["hypotheses"] = str(h_path)
        return paths

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _read_free_energy(self, obs: np.ndarray) -> float:
        """Pragmatic free-energy term (Task G fix).

        Uses ``||obs - belief @ emission||^2 / dim`` instead of
        ``engine.compute_free_energy()``, which uses a sigma_q^2 proxy
        that collapses in passive reading and inflates the KL term.
        """
        engine = getattr(self.model, "active_inference", None)
        if engine is None:
            return 0.0
        try:
            gm = engine.generative_model
            belief = gm.belief_state
            predicted = gm.predict_observation(belief)
            err = np.asarray(obs)[:len(predicted)] - predicted[:len(obs)]
            if len(err) < gm.obs_dim:
                err = np.pad(err, (0, gm.obs_dim - len(err)))
            return float(np.dot(err, err)) / err.size
        except Exception:
            return 0.0

    def _read_prediction_error(self, obs: np.ndarray) -> float:
        engine = getattr(self.model, "active_inference", None)
        if engine is None:
            return 0.0
        try:
            belief = np.asarray(engine.generative_model.belief_state)
            b = belief[: len(obs)]
            return float(np.linalg.norm(np.asarray(obs) - b))
        except Exception:
            return 0.0

    @staticmethod
    def _synthetic_text_for_source(source: str, modality: str) -> str:
        """Build a small text snippet keyed by the source label.

        This is a fallback when the stream returns only a vector + label
        (not the raw text). For the encyclopedia adapter, the source
        label is ``wiki:ArticleTitle`` — we use that as the seed text.
        """
        # Extract the article title if it's a wiki: source.
        if source.startswith("wiki:"):
            title = source[len("wiki:"):]
            return (f"{title}. This article discusses concepts related to {title.lower()}. "
                    f"The study of {title.lower()} involves several related ideas.")
        # Otherwise, use the source label itself as a minimal seed.
        return f"{modality} observation from {source}."

    @staticmethod
    def _clean_source_label(source: str, modality: str) -> str:
        """Convert a raw source label into a clean KG node title.

        - ``wiki:Physics_0``  → ``Physics_0``
        - ``/tmp/foo/bar/fib_01.py`` → ``code:fib_01``
        - ``/tmp/foo/bar/sine_00_440hz.wav`` → ``audio:sine_00``
        - ``/tmp/foo/bar/circle_00.png`` → ``image:circle_00``
        - ``physics:step_42`` → ``physics:step_42``

        This ensures the KG node titles are short, descriptive, and
        the question generator's entity matching can find them in
        the encyclopedia's article list.
        """
        if source.startswith("wiki:"):
            return source[len("wiki:"):]
        if source.startswith("physics:"):
            return source
        # File path: extract the stem + prepend the modality.
        from pathlib import Path
        try:
            stem = Path(source).stem
            return f"{modality}:{stem}"
        except Exception:
            return source

    def _print_progress(self, step: int, t0: float) -> None:
        elapsed = time.perf_counter() - t0
        rate = step / elapsed if elapsed > 0 else 0.0
        recent = self.records[-PROGRESS_INTERVAL:]
        if recent:
            mean_fe = float(np.mean([r.free_energy for r in recent]))
            last = recent[-1]
            print(f"  step {step}/{self.max_steps}  "
                  f"rate={rate:.1f} cyc/s  "
                  f"recent_fe={mean_fe:.3f}  "
                  f"mod={last.modality:8s}  "
                  f"src={last.source[:30]:30s}")
        else:
            print(f"  step {step}/{self.max_steps}  rate={rate:.1f} cyc/s")


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #
def parse_ratios(s: str | None) -> dict[str, float] | None:
    """Parse a ratios string like 'text=0.6,physics=0.2,image=0.1'."""
    if not s:
        return None
    out: dict[str, float] = {}
    for part in s.split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        try:
            out[k.strip()] = float(v)
        except ValueError:
            pass
    return out or None


def main() -> None:
    p = argparse.ArgumentParser(description="Phase K omni-modal learning loop")
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--dim", type=int, default=DEFAULT_DIM)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--corpus_dir", type=str, default=None,
                   help="Directory of WikiExtractor-style files.")
    p.add_argument("--image_dir", type=str, default=None)
    p.add_argument("--audio_dir", type=str, default=None)
    p.add_argument("--code_dir", type=str, default=None)
    p.add_argument("--ratios", type=str, default=None,
                   help="e.g. 'text=0.6,physics=0.2,image=0.1,audio=0.05,code=0.05'")
    p.add_argument("--inquiry_interval", type=int, default=INQUIRY_INTERVAL)
    p.add_argument("--hypothesis_interval", type=int, default=HYPOTHESIS_INTERVAL)
    p.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    args = p.parse_args()

    ratios = parse_ratios(args.ratios) or dict(DEFAULT_RATIOS)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("Phase K — Omni-modal Learning Loop")
    print("=" * 64)
    print(f"  steps={args.steps}  dim={args.dim}  seed={args.seed}")
    print(f"  ratios={ratios}")
    print(f"  output_dir={output_dir}")

    # --- Build the model (curiosity-enhanced). ---------------------- #
    model = make_curious_model(dim=args.dim, seed=args.seed)

    # --- Build the encyclopedia reader (or synthetic). ------------- #
    if args.corpus_dir:
        corpus_files = sorted(Path(args.corpus_dir).iterdir())
        reader = EncyclopediaReader(corpus_files, index_dir=Path(args.corpus_dir) / ".index")
    else:
        import tempfile
        tmp_corpus = Path(tempfile.mkdtemp(prefix="omni_corpus_"))
        corpus_files = make_synthetic_corpus(tmp_corpus, n_files=4, articles_per_file=5, seed=args.seed)
        reader = EncyclopediaReader(corpus_files, index_dir=tmp_corpus / ".index")
    print(f"  corpus: {reader.n_articles()} articles")

    # --- Build the text encoder + decoder. -------------------------- #
    text_encoder = TextEncoder(block_size=512, output_dim=args.dim, seed=args.seed)
    decoder = TextDecoder(text_encoder=text_encoder, n=2, seed=args.seed, temperature=1.0)

    # --- Build the omni-modal generator. ---------------------------- #
    text_stream = EncyclopediaStreamAdapter(reader, encoder=text_encoder,
                                            output_dim=args.dim, seed=args.seed)
    # Physics stream: build a sandbox + image encoder.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
    from physics_sandbox import PhysicsSandbox
    sandbox = PhysicsSandbox(num_objects=3, seed=args.seed)
    image_encoder = ImageEncoder(output_dim=args.dim, seed=args.seed)
    physics_stream = SandboxStreamAdapter(sandbox, image_encoder, seed=args.seed)
    # Construct the generator with all streams + ratios.
    streams = {
        MOD_TEXT: text_stream,
        MOD_PHYSICS: physics_stream,
    }
    # The image / audio / code streams need their encoders. We use the
    # build_default_generator helper which handles synthetic fallbacks.
    # To honour the supplied ratios, we build streams explicitly.
    if args.image_dir:
        from image_encoder import ImageStream
        streams[MOD_IMAGE] = ImageStream(args.image_dir, image_encoder, seed=args.seed)
    if args.audio_dir:
        from audio_encoder import AudioStream
        audio_encoder = AudioEncoder(output_dim=args.dim, seed=args.seed)
        streams[MOD_AUDIO] = AudioStream(args.audio_dir, audio_encoder, seed=args.seed)
    if args.code_dir:
        from code_encoder import CodeStream
        code_encoder = CodeEncoder(output_dim=args.dim, seed=args.seed, text_encoder=text_encoder)
        streams[MOD_CODE] = CodeStream(args.code_dir, code_encoder, seed=args.seed)
    # Use build_default_generator to fill in any missing streams with
    # synthetic fallbacks (it picks up our pre-built streams + adds
    # synthetic image/audio/code dirs as needed).
    generator = OmniModalGenerator(streams=streams, ratios=ratios, seed=args.seed)
    print(f"  generator: {generator}")
    # If any modality is missing (no directory), add the synthetic fallback.
    missing = {"image", "audio", "code"} - set(generator.active_modalities)
    if missing:
        print(f"  [warn] no real {missing} directories; "
              "synthetic fallbacks will be used.")
        # Re-build with synthetic fallbacks for missing modalities.
        from omnimodal_data_generator import build_default_generator
        generator = build_default_generator(
            text_stream=text_stream,
            physics_stream=physics_stream,
            image_dir=args.image_dir,
            audio_dir=args.audio_dir,
            code_dir=args.code_dir,
            output_dim=args.dim,
            ratios=ratios,
            seed=args.seed,
        )
        print(f"  generator (rebuilt): {generator}")

    # --- Build the KG builder + loop. ------------------------------- #
    kg = KnowledgeGraphBuilder()
    loop = OmniLearningLoop(
        model=model,
        generator=generator,
        kg_builder=kg,
        decoder=decoder,
        text_encoder=text_encoder,
        reader=reader,
        max_steps=args.steps,
        inquiry_interval=args.inquiry_interval,
        hypothesis_interval=args.hypothesis_interval,
        seed=args.seed,
        output_dir=output_dir,
    )

    # --- Run. --------------------------------------------------------- #
    print("\n[starting loop]")
    summary = loop.run()

    # --- Print + save outputs. --------------------------------------- #
    print("\n" + "=" * 64)
    print("Summary")
    print("=" * 64)
    for k, v in asdict(summary).items():
        print(f"  {k}: {v}")

    paths = loop.save_outputs()
    print(f"\nOutputs written:")
    for name, path in paths.items():
        print(f"  {name:12s} -> {path}")


if __name__ == "__main__":
    main()
