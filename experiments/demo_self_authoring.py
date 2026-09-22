# experiments/demo_self_authoring.py
"""Phase K, Task 5: Self-authoring demonstration.

End-to-end demonstration of ZeroDataModel's autonomous cognition
capabilities. The script:

  1. Builds a curiosity-enhanced ZeroDataModel + a synthetic encyclopedia
     corpus + the omni-modal generator (text + physics + image + audio
     + code), and runs a brief learning prelude (~300 steps) so the
     knowledge graph and decoder pool contain REAL, model-derived state.
  2. Exercises FOUR self-authoring demonstrations:

     Demo 1 — Knowledge Summary:
         Pick a concept from the encyclopedia, encode its latent, and
         generate a self-authored explanation via the retrieval +
         Markov decoder.

     Demo 2 — Cross-Domain Analogy:
         Discover two KG nodes from different topic domains whose
         latent vectors share a moderate cosine similarity (0.3-0.7)
         and generate an analogy text bridging them.

     Demo 3 — Counterfactual Analysis:
         Read an article, encode the resulting belief, and generate a
         counterfactual narrative ("If <X> had changed, <Y> would
         change") using the article's structural neighbours in the KG.

     Demo 4 — Self Q&A:
         The InquiryLoop generates a question from KG gaps, searches
         the encyclopedia, reads a candidate article, and produces an
         answer with self-assessed confidence (delta prediction error).

  3. Writes a Markdown report to ``experiments/output/omni_run/
     self_authoring_demo.md`` with embedded KG stats and the four
     self-authored samples.

All questions and answers are internally generated — NO human-authored
Q/A pairs are used. The script is fully self-contained and runnable:

    python experiments/demo_self_authoring.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

# Make sibling modules importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

# Phase H / J infrastructure.
from encyclopedia_reader import EncyclopediaReader, make_synthetic_corpus
from knowledge_graph_builder import (
    EDGE_LATENT,
    EDGE_STRUCTURAL,
    KnowledgeGraphBuilder,
)
from run_sandbox_curious import make_curious_model

# Phase K modules.
from active_inquiry import InquiryLoop
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
    build_default_generator,
)
from physics_sandbox import PhysicsSandbox
from run_omni_learning import (
    OmniLearningLoop,
    SandboxStreamAdapter,
)
from text_decoder import TextDecoder
from text_encoder import TextEncoder


# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #
DEFAULT_DIM = 32
DEFAULT_SEED = 42
PRELUDE_STEPS = 300          # short learning prelude to populate KG + pool
INQUIRY_N_QUESTIONS = 5
ANALOGY_SIM_RANGE = (0.3, 0.7)
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "omni_run"


# ------------------------------------------------------------------ #
# Topic clustering helper
# ------------------------------------------------------------------ #
# Topic names used by ``make_synthetic_corpus`` (see encyclopedia_reader.py).
# Article titles look like ``"Physics_3"`` or ``"Computer Science_0"``.
def _topic_of(title: str) -> str:
    """Return the topic prefix of a synthetic article title.

    ``"Physics_3"`` -> ``"Physics"``
    ``"Computer Science_0"`` -> ``"Computer Science"``
    """
    # Strip the trailing ``_<digit>`` suffix.
    idx = title.rfind("_")
    if idx <= 0:
        return title
    return title[:idx]


# ------------------------------------------------------------------ #
# Demo runner
# ------------------------------------------------------------------ #
class SelfAuthoringDemo:
    """Run the four Phase-K self-authoring demonstrations.

    Construct once, then call ``run()`` to execute the prelude + the
    four demos and write a Markdown report.

    Parameters
    ----------
    dim : int
        Model dimensionality (matches encoder output_dim).
    seed : int
        RNG seed for reproducibility.
    prelude_steps : int
        Number of omni-modal learning cycles to run before the demos
        so the KG + decoder pool contain real, model-derived state.
    output_dir : Path
        Where to write the Markdown report.
    """

    def __init__(
        self,
        dim: int = DEFAULT_DIM,
        seed: int = DEFAULT_SEED,
        prelude_steps: int = PRELUDE_STEPS,
        output_dir: Path = OUTPUT_DIR,
    ):
        self.dim = int(dim)
        self.seed = int(seed)
        self.prelude_steps = int(prelude_steps)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Output accumulator — each entry is a (heading, body_lines) tuple.
        self._sections: list[tuple[str, list[str]]] = []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self) -> Path:
        """Execute the prelude + the four demos, return the report path."""
        t0 = time.perf_counter()
        self._setup()
        self._run_prelude()
        # The four demonstrations.
        demo1 = self._demo_knowledge_summary()
        demo2 = self._demo_cross_domain_analogy()
        demo3 = self._demo_counterfactual_analysis()
        demo4 = self._demo_self_qa()
        elapsed = time.perf_counter() - t0
        # Build the Markdown report.
        report_path = self._write_report(elapsed, [demo1, demo2, demo3, demo4])
        print(f"\n[done] report -> {report_path}")
        return report_path

    # ------------------------------------------------------------------ #
    # Setup: build model + corpus + generator + KG + decoder
    # ------------------------------------------------------------------ #
    def _setup(self) -> None:
        print("=" * 64)
        print("Phase K — Self-Authoring Demonstration")
        print("=" * 64)
        print(f"  dim={self.dim}  seed={self.seed}  prelude={self.prelude_steps} steps")

        # --- Model (curiosity-enhanced) ------------------------------ #
        self.model = make_curious_model(dim=self.dim, seed=self.seed)

        # --- Encyclopedia corpus (synthetic) ------------------------ #
        self._tmp_corpus = Path(tempfile.mkdtemp(prefix="omni_demo_corpus_"))
        self.corpus_files = make_synthetic_corpus(
            self._tmp_corpus,
            n_files=4,
            articles_per_file=5,
            seed=self.seed,
        )
        self.reader = EncyclopediaReader(
            self.corpus_files,
            index_dir=self._tmp_corpus / ".index",
            seed=self.seed,
        )
        print(f"  corpus: {self.reader.n_articles()} articles")

        # --- Encoders / decoder ------------------------------------- #
        self.text_encoder = TextEncoder(block_size=512, output_dim=self.dim,
                                        seed=self.seed)
        self.decoder = TextDecoder(text_encoder=self.text_encoder, n=2,
                                   seed=self.seed, temperature=1.0)

        # --- Omni-modal generator (text + physics + synth fallbacks) #
        text_stream = EncyclopediaStreamAdapter(
            self.reader, encoder=self.text_encoder,
            output_dim=self.dim, seed=self.seed,
        )
        self.sandbox = PhysicsSandbox(num_objects=3, seed=self.seed)
        self.image_encoder = ImageEncoder(output_dim=self.dim, seed=self.seed)
        self.audio_encoder = AudioEncoder(output_dim=self.dim, seed=self.seed)
        self.code_encoder = CodeEncoder(output_dim=self.dim, seed=self.seed,
                                        text_encoder=self.text_encoder)
        physics_stream = SandboxStreamAdapter(self.sandbox, self.image_encoder,
                                              seed=self.seed)
        # Use build_default_generator to get synthetic image/audio/code
        # streams when no real directories are provided.
        self.generator = build_default_generator(
            text_stream=text_stream,
            physics_stream=physics_stream,
            image_dir=None,
            audio_dir=None,
            code_dir=None,
            output_dim=self.dim,
            ratios=dict(DEFAULT_RATIOS),
            seed=self.seed,
        )
        print(f"  generator: {len(self.generator.active_modalities)} modalities")

        # --- KG ------------------------------------------------------ #
        self.kg_builder = KnowledgeGraphBuilder()

    # ------------------------------------------------------------------ #
    # Prelude: a short omni-modal learning loop to populate KG + pool
    # ------------------------------------------------------------------ #
    def _run_prelude(self) -> None:
        print(f"\n[prelude] running {self.prelude_steps} omni-modal cycles...")
        loop = OmniLearningLoop(
            model=self.model,
            generator=self.generator,
            kg_builder=self.kg_builder,
            decoder=self.decoder,
            text_encoder=self.text_encoder,
            reader=self.reader,
            max_steps=self.prelude_steps,
            inquiry_interval=self.prelude_steps + 1,    # disable inquiry
            hypothesis_interval=self.prelude_steps + 1, # disable hypothesis
            seed=self.seed,
            output_dir=self.output_dir,
        )
        summary = loop.run()
        # Refresh latent edges so demos can use them.
        try:
            self.kg_builder.compute_latent_edges(top_k=5)
        except Exception:
            pass
        print(f"[prelude] KG nodes={summary.final_kg_nodes} "
              f"edges={summary.final_kg_edges} "
              f"pool_size={summary.final_decoder_pool_size} "
              f"mean_fe={summary.mean_free_energy:.3f}")

    # ------------------------------------------------------------------ #
    # Demo 1 — Knowledge Summary
    # ------------------------------------------------------------------ #
    def _demo_knowledge_summary(self) -> dict:
        """Pick an article from the corpus, generate a self-authored summary."""
        print("\n[demo 1] knowledge summary")
        # Pick the article whose title has the lowest mean free energy in
        # the KG (i.e. the model is most "comfortable" with it). Fallback:
        # first article in the corpus.
        target_title = None
        target_latent = None
        target_fe = float("inf")
        for title in self.reader.get_article_list():
            node = self.kg_builder.nodes.get(title)
            if node is None:
                continue
            # KG nodes store the latest latent under "belief_state".
            belief = node.get("belief_state")
            if belief is None:
                continue
            fe = float(node.get("mean_free_energy", 1e9))
            if fe < target_fe:
                target_fe = fe
                target_title = title
                target_latent = np.asarray(belief, dtype=np.float64)
        if target_title is None:
            target_title = self.reader.get_article_list()[0]
            target_latent = self.text_encoder.encode(target_title)
        # Decode a summary from the latent.
        generated = self.decoder.decode(target_latent, max_length=80)
        # Retrieve the closest stored text for grounding.
        retrieved = self.decoder.pool.most_similar_text(target_latent) or ""
        # Read a few blocks of the article for an "objective" comparison.
        original_excerpt = self._read_article_excerpt(target_title, max_chars=240)
        print(f"  target: {target_title} (fe={target_fe:.3f})")
        print(f"  generated: {generated[:80]!r}")
        return {
            "title": target_title,
            "free_energy": target_fe,
            "latent_norm": float(np.linalg.norm(target_latent)),
            "generated_summary": generated,
            "retrieved_ground_truth": retrieved,
            "original_excerpt": original_excerpt,
        }

    # ------------------------------------------------------------------ #
    # Demo 2 — Cross-Domain Analogy
    # ------------------------------------------------------------------ #
    def _demo_cross_domain_analogy(self) -> dict:
        """Find two cross-domain nodes with moderate latent similarity."""
        print("\n[demo 2] cross-domain analogy")
        # Use latent edges to find candidate cross-domain pairs.
        # Restrict to ENCYCLOPEDIA article titles (no modality prefix
        # like ``image:``, ``audio:``, ``code:``, ``physics:``) so the
        # analogy is between two real-world concepts, not two streams
        # of the same modality.
        valid_titles = set(self.reader.get_article_list())
        lo, hi = ANALOGY_SIM_RANGE
        cross_pairs: list[tuple[str, str, float]] = []
        for (s, d), attrs in self.kg_builder.edges.items():
            if attrs.get("edge_type") != EDGE_LATENT:
                continue
            sim = float(attrs.get("weight", 0.0))
            if not (lo <= sim <= hi):
                continue
            # Both endpoints must be encyclopedia articles.
            if s not in valid_titles or d not in valid_titles:
                continue
            # Skip intra-topic pairs (same topic prefix).
            if _topic_of(s) == _topic_of(d):
                continue
            cross_pairs.append((s, d, sim))
        if not cross_pairs:
            # Fallback: pick two articles from different topics and
            # compute their latent similarity directly.
            titles = sorted(valid_titles)
            by_topic: dict[str, list[str]] = {}
            for t in titles:
                by_topic.setdefault(_topic_of(t), []).append(t)
            topics = list(by_topic.keys())
            if len(topics) >= 2:
                s_title = by_topic[topics[0]][0]
                d_title = by_topic[topics[1]][0]
                s_lat = self._latent_of(s_title)
                d_lat = self._latent_of(d_title)
                if s_lat is not None and d_lat is not None:
                    sim = float(np.dot(s_lat, d_lat) /
                                (np.linalg.norm(s_lat) *
                                 np.linalg.norm(d_lat) + 1e-12))
                    cross_pairs = [(s_title, d_title, sim)]
        if not cross_pairs:
            return {"error": "no cross-domain pair found"}
        # Pick the strongest pair.
        cross_pairs.sort(key=lambda x: -x[2])
        s_title, d_title, sim = cross_pairs[0]
        s_node = self.kg_builder.nodes[s_title]
        d_node = self.kg_builder.nodes[d_title]
        # KG nodes store the latest latent under "belief_state".
        s_lat = np.asarray(s_node["belief_state"], dtype=np.float64)
        d_lat = np.asarray(d_node["belief_state"], dtype=np.float64)
        # Combine the latents and decode an analogy.
        combined = (s_lat + d_lat) / 2.0
        # Use a templated decoder call to bias generation toward analogy.
        template = ("Looking at {a} and {b}, one might draw an analogy: "
                    "just as {a} involves certain properties, {b} exhibits "
                    "similar structural patterns.")
        substitutions = {"a": s_title, "b": d_title}
        analogy = self.decoder.decode_with_template(
            combined, template=template, substitutions=substitutions,
            max_length=120,
        )
        print(f"  pair: {s_title} <-> {d_title}  sim={sim:.3f}")
        print(f"  analogy: {analogy[:80]!r}")
        return {
            "source_a": s_title,
            "source_b": d_title,
            "topic_a": _topic_of(s_title),
            "topic_b": _topic_of(d_title),
            "cosine_similarity": sim,
            "free_energy_a": float(s_node.get("mean_free_energy", 0.0)),
            "free_energy_b": float(d_node.get("mean_free_energy", 0.0)),
            "analogy_text": analogy,
        }

    # ------------------------------------------------------------------ #
    # Demo 3 — Counterfactual Analysis
    # ------------------------------------------------------------------ #
    def _demo_counterfactual_analysis(self) -> dict:
        """Generate a counterfactual narrative from an article + KG context."""
        print("\n[demo 3] counterfactual analysis")
        # Prefer structural edges from the KG; if none exist, fall back
        # to the encyclopedia's article-link graph (which the reader
        # exposes via ``get_article_links(title)``).
        candidates = []
        for (s, d), attrs in self.kg_builder.edges.items():
            if attrs.get("edge_type") != EDGE_STRUCTURAL:
                continue
            candidates.append(s)
            candidates.append(d)
        # Pick the most-connected node that is a real article.
        target_title: str | None = None
        neighbours: list[str] = []
        if candidates:
            from collections import Counter
            counts = Counter(candidates)
            for title, _ in counts.most_common():
                if title in self.kg_builder.nodes:
                    target_title = title
                    break
            if target_title is not None:
                for (s, d), _attrs in self.kg_builder.edges.items():
                    if s == target_title and d not in neighbours:
                        neighbours.append(d)
                    elif d == target_title and s not in neighbours:
                        neighbours.append(s)
        # Fallback: use the encyclopedia's link graph directly.
        if target_title is None or not neighbours:
            valid_titles = set(self.reader.get_article_list())
            for title in valid_titles:
                try:
                    links = [l for l in self.reader.get_article_links(title)
                             if l in valid_titles]
                except Exception:
                    links = []
                if len(links) >= 2:
                    target_title = title
                    neighbours = links[:5]
                    break
        if target_title is None:
            return {"error": "no article with structural neighbours found"}
        # Fetch the latent.
        latent = self._latent_of(target_title)
        if latent is None:
            # The article wasn't visited during the prelude — encode its
            # title so we have something to seed the decoder with.
            latent = self.text_encoder.encode(target_title)
        node = self.kg_builder.nodes.get(target_title, {})
        # Build a counterfactual narrative via template + Markov.
        if len(neighbours) >= 2:
            template = ("If {target} had not been connected to {neighbour_a}, "
                        "then {neighbour_b} would have appeared in a different "
                        "context. The model's belief about {target} would shift, "
                        "and the article would discuss alternative ideas.")
            substitutions = {
                "target": target_title,
                "neighbour_a": neighbours[0],
                "neighbour_b": neighbours[1],
            }
        else:
            template = ("If {target} had been absent from the corpus, "
                        "the model's free energy would have been higher, "
                        "and related ideas would not have been linked.")
            substitutions = {"target": target_title}
        counterfactual = self.decoder.decode_with_template(
            latent, template=template, substitutions=substitutions,
            max_length=140,
        )
        print(f"  target: {target_title} (neighbours={len(neighbours)})")
        print(f"  counterfactual: {counterfactual[:80]!r}")
        return {
            "target_article": target_title,
            "n_neighbours": len(neighbours),
            "neighbours": neighbours[:5],
            "mean_free_energy": float(node.get("mean_free_energy", 0.0)),
            "counterfactual_text": counterfactual,
        }

    # ------------------------------------------------------------------ #
    # Demo 4 — Self Q&A
    # ------------------------------------------------------------------ #
    def _demo_self_qa(self) -> dict:
        """Let the InquiryLoop ask + answer a question autonomously."""
        print("\n[demo 4] self Q&A")
        loop = InquiryLoop(
            model=self.model,
            text_encoder=self.text_encoder,
            decoder=self.decoder,
            kg_builder=self.kg_builder,
            reader=self.reader,
            question_generator=None,    # auto-build with allowed_node_test
            max_read_blocks=3,
            pred_err_threshold=0.0,
            seed=self.seed,
        )
        result = loop.run(n_questions=INQUIRY_N_QUESTIONS)
        # Pick the first answered question, else the first question.
        qa_list = result.get("qa", [])
        if not qa_list:
            return {"error": "no questions generated"}
        chosen = None
        for entry in qa_list:
            if entry.get("answered"):
                chosen = entry
                break
        if chosen is None:
            chosen = qa_list[0]
        # Build the "self-Q&A" record.
        print(f"  questions asked: {result['n_questions']}")
        print(f"  answered: {result['n_answered']}")
        print(f"  chosen: {chosen['question']!r}")
        return {
            "n_questions_asked": result["n_questions"],
            "n_answered": result["n_answered"],
            "all_qa": qa_list,
            "chosen_qa": chosen,
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _latent_of(self, title: str) -> np.ndarray | None:
        """Return the latest belief_state of a KG node, or None."""
        node = self.kg_builder.nodes.get(title)
        if node is None:
            return None
        belief = node.get("belief_state")
        if belief is None:
            return None
        return np.asarray(belief, dtype=np.float64)

    def _read_article_excerpt(self, title: str, max_chars: int = 240) -> str:
        """Read up to ``max_chars`` characters of an article's body."""
        try:
            if not self.reader.seek_to_article(title):
                return ""
            buf: list[str] = []
            total = 0
            for _ in range(8):
                block = self.reader.read_block()
                if not block:
                    break
                buf.append(block)
                total += len(block)
                if total >= max_chars:
                    break
            excerpt = " ".join(buf).strip()
            return excerpt[:max_chars]
        except Exception:
            return ""

    # ------------------------------------------------------------------ #
    # Markdown report writer
    # ------------------------------------------------------------------ #
    def _write_report(self, elapsed: float, demos: list[dict]) -> Path:
        """Compose the final Markdown report and write it to disk."""
        path = self.output_dir / "self_authoring_demo.md"
        lines: list[str] = []
        # Header.
        lines.append("# ZeroDataModel — Self-Authoring Demonstration (Phase K, Task 5)")
        lines.append("")
        lines.append("This report was generated **autonomously** by the model. "
                     "No human-authored questions or answers were used — every "
                     "output below was internally driven by the model's own "
                     "knowledge graph, free-energy landscape, and retrieval + "
                     "Markov decoder.")
        lines.append("")
        lines.append("## Run configuration")
        lines.append("")
        lines.append(f"- Model dimensionality: **{self.dim}**")
        lines.append(f"- RNG seed: **{self.seed}**")
        lines.append(f"- Prelude learning steps: **{self.prelude_steps}**")
        lines.append(f"- Corpus articles: **{self.reader.n_articles()}**")
        lines.append(f"- Active modalities: "
                     f"**{', '.join(self.generator.active_modalities)}**")
        lines.append(f"- Wall time: **{elapsed:.2f} s**")
        lines.append("")

        # Knowledge graph snapshot.
        n_nodes = self.kg_builder.n_nodes()
        n_edges = self.kg_builder.n_edges()
        n_latent = sum(1 for attrs in self.kg_builder.edges.values()
                       if attrs.get("edge_type") == EDGE_LATENT)
        n_struct = sum(1 for attrs in self.kg_builder.edges.values()
                       if attrs.get("edge_type") == EDGE_STRUCTURAL)
        n_coread = sum(1 for attrs in self.kg_builder.edges.values()
                       if attrs.get("edge_type") == "co_read")
        # Cross-domain edges: count edges whose two endpoints have
        # different topic prefixes.
        n_cross = 0
        for (s, d) in self.kg_builder.edges:
            if _topic_of(s) != _topic_of(d):
                n_cross += 1
        lines.append("## Knowledge graph snapshot")
        lines.append("")
        lines.append(f"- Nodes: **{n_nodes}**")
        lines.append(f"- Edges (total): **{n_edges}**")
        lines.append(f"  - Structural: {n_struct}")
        lines.append(f"  - Co-read: {n_coread}")
        lines.append(f"  - Latent (cosine sim): {n_latent}")
        lines.append(f"- Cross-domain edges: **{n_cross}** "
                     f"(connect articles from different topic prefixes)")
        lines.append(f"- Decoder retrieval pool size: **{len(self.decoder.pool)}**")
        lines.append("")

        # Topic distribution.
        topic_counts: dict[str, int] = {}
        for title in self.kg_builder.nodes:
            topic = _topic_of(title)
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        if topic_counts:
            lines.append("### Topic distribution (KG nodes)")
            lines.append("")
            lines.append("| Topic | # nodes |")
            lines.append("|---|---|")
            for topic in sorted(topic_counts):
                lines.append(f"| {topic} | {topic_counts[topic]} |")
            lines.append("")

        # Free energy statistics.
        fes = [float(n.get("mean_free_energy", 0.0))
               for n in self.kg_builder.nodes.values()
               if n.get("n_free_energy_samples", 0) > 0]
        if fes:
            lines.append("### Free energy distribution")
            lines.append("")
            lines.append(f"- Mean: **{float(np.mean(fes)):.3f}**")
            lines.append(f"- Min: **{float(np.min(fes)):.3f}**")
            lines.append(f"- Max: **{float(np.max(fes)):.3f}**")
            lines.append(f"- Std: **{float(np.std(fes)):.3f}**")
            lines.append("")

        # Demos.
        demo1, demo2, demo3, demo4 = demos
        # --- Demo 1: Knowledge summary ------------------------------ #
        lines.append("## Demo 1 — Knowledge Summary")
        lines.append("")
        if "error" in demo1:
            lines.append(f"_{demo1['error']}_")
        else:
            lines.append(f"**Target concept:** `{demo1['title']}` "
                         f"(model free energy = {demo1['free_energy']:.3f})")
            lines.append("")
            lines.append("**Self-authored summary (decoder output):**")
            lines.append("")
            lines.append(f"> {demo1['generated_summary']}")
            lines.append("")
            lines.append("**Retrieved grounding (closest stored text):**")
            lines.append("")
            lines.append(f"> {demo1['retrieved_ground_truth']}")
            lines.append("")
            lines.append("**Original article excerpt (for comparison):**")
            lines.append("")
            lines.append(f"> {demo1['original_excerpt']}")
        lines.append("")

        # --- Demo 2: Cross-domain analogy --------------------------- #
        lines.append("## Demo 2 — Cross-Domain Analogy")
        lines.append("")
        if "error" in demo2:
            lines.append(f"_{demo2['error']}_")
        else:
            lines.append(f"**Source A:** `{demo2['source_a']}` "
                         f"(topic: *{demo2['topic_a']}*)")
            lines.append(f"**Source B:** `{demo2['source_b']}` "
                         f"(topic: *{demo2['topic_b']}*)")
            lines.append(f"**Latent cosine similarity:** {demo2['cosine_similarity']:.3f}")
            lines.append("")
            lines.append("**Self-authored analogy:**")
            lines.append("")
            lines.append(f"> {demo2['analogy_text']}")
        lines.append("")

        # --- Demo 3: Counterfactual analysis ----------------------- #
        lines.append("## Demo 3 — Counterfactual Analysis")
        lines.append("")
        if "error" in demo3:
            lines.append(f"_{demo3['error']}_")
        else:
            lines.append(f"**Target article:** `{demo3['target_article']}`")
            lines.append(f"**Structural neighbours:** "
                         f"{', '.join(demo3['neighbours']) or '(none)'}")
            lines.append(f"**Model free energy:** {demo3['mean_free_energy']:.3f}")
            lines.append("")
            lines.append("**Self-authored counterfactual:**")
            lines.append("")
            lines.append(f"> {demo3['counterfactual_text']}")
        lines.append("")

        # --- Demo 4: Self Q&A ------------------------------------- #
        lines.append("## Demo 4 — Self Q&A")
        lines.append("")
        if "error" in demo4:
            lines.append(f"_{demo4['error']}_")
        else:
            lines.append(f"**Questions asked:** {demo4['n_questions_asked']}  "
                         f"**Answered:** {demo4['n_answered']}")
            lines.append("")
            chosen = demo4["chosen_qa"]
            lines.append("### Chosen Q&A pair")
            lines.append("")
            lines.append(f"- **Question:** {chosen.get('question', '')}")
            lines.append(f"- **Gap type:** `{chosen.get('gap_type', '')}`")
            lines.append(f"- **Matched article:** "
                         f"`{chosen.get('matched_article', '') or '(none)'}`")
            lines.append(f"- **Answered:** "
                         f"{'yes' if chosen.get('answered') else 'no'}")
            lines.append(f"- **Answer:** {chosen.get('answer', '')}")
            lines.append(f"- **Prediction error before → after:** "
                         f"{chosen.get('pred_err_before', 0):.4f} → "
                         f"{chosen.get('pred_err_after', 0):.4f}  "
                         f"(Δ = {chosen.get('delta_pred_err', 0):+.4f})")
            lines.append(f"- **Confidence:** {chosen.get('confidence', 0):.4f}")
            lines.append("")
            # Show all Q/A pairs in a table for completeness.
            lines.append("### All Q/A pairs from this run")
            lines.append("")
            lines.append("| # | Question | Gap type | Answered | Δ pred err |")
            lines.append("|---|---|---|---|---|")
            for i, entry in enumerate(demo4["all_qa"], 1):
                q = entry.get("question", "").replace("|", "\\|")
                if len(q) > 80:
                    q = q[:77] + "..."
                answered = "yes" if entry.get("answered") else "no"
                delta = entry.get("delta_pred_err", 0.0)
                lines.append(f"| {i} | {q} | `{entry.get('gap_type', '')}` "
                             f"| {answered} | {delta:+.4f} |")
            lines.append("")

        # Footer.
        lines.append("---")
        lines.append("")
        lines.append("*Generated by `experiments/demo_self_authoring.py` — "
                     "Phase K capstone of the ZeroDataModel project.*")
        # Write.
        text = "\n".join(lines)
        path.write_text(text, encoding="utf-8")
        # Also dump the raw demos as JSON for downstream tooling.
        json_path = path.with_suffix(".json")
        with json_path.open("w") as f:
            json.dump({
                "config": {
                    "dim": self.dim,
                    "seed": self.seed,
                    "prelude_steps": self.prelude_steps,
                    "elapsed_s": elapsed,
                },
                "kg_stats": {
                    "n_nodes": n_nodes,
                    "n_edges": n_edges,
                    "n_latent_edges": n_latent,
                    "n_structural_edges": n_struct,
                    "n_co_read_edges": n_coread,
                    "n_cross_domain_edges": n_cross,
                    "decoder_pool_size": len(self.decoder.pool),
                },
                "demos": demos,
            }, f, indent=2, default=str)
        return path


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #
def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Phase K self-authoring demo")
    p.add_argument("--dim", type=int, default=DEFAULT_DIM)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--prelude_steps", type=int, default=PRELUDE_STEPS)
    p.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    args = p.parse_args()
    demo = SelfAuthoringDemo(
        dim=args.dim,
        seed=args.seed,
        prelude_steps=args.prelude_steps,
        output_dir=Path(args.output_dir),
    )
    demo.run()


if __name__ == "__main__":
    main()
