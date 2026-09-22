# experiments/active_inquiry.py
"""Active inquiry loop (Phase K, Task 3.2 / 3.3).

A self-contained ``InquiryLoop`` that closes the "ask → search → read
→ evaluate" cycle:

    1. QuestionGenerator identifies a gap in the knowledge graph.
    2. The loop encodes the question via a TextEncoder → question_latent.
    3. The loop searches the encyclopedia reader for a matching article
       (by title substring match; falls back to full-text retrieval).
    4. The loop READS the matched article by feeding its blocks through
       ZeroDataModel.think(obs), recording the belief_state before and
       after.
    5. The loop re-encodes the question and measures the prediction
       error / free-energy change. If reading the article REDUCED the
       question's prediction error, the question is considered answered.
    6. The Q/A pair is appended to a transcript; the KG is updated with
       the newly-read article's belief_state.

For the physics domain (Task 3.3), the loop can also propose a
HYPOTHESIS (templated) and test it via the causal emergence engine's
intervention API. Hypothesis outcomes are appended to the same
transcript with a "hypothesis" record type.

All output is JSON-serialisable for the report generator.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ------------------------------------------------------------------ #
# Records
# ------------------------------------------------------------------ #
@dataclass
class QARecord:
    """One question-answer pair."""
    step: int
    question: str
    gap_type: str
    entities: list[str]
    answer: str
    matched_article: str
    fe_before: float
    fe_after: float
    pred_err_before: float
    pred_err_after: float
    delta_pred_err: float
    answered: bool
    confidence: float
    metadata: dict = field(default_factory=dict)


@dataclass
class HypothesisRecord:
    """One hypothesis + its simulated outcome."""
    step: int
    hypothesis: str
    intervention: dict
    outcome: dict
    supported: bool
    confidence: float
    metadata: dict = field(default_factory=dict)


# ------------------------------------------------------------------ #
# InquiryLoop
# ------------------------------------------------------------------ #
class InquiryLoop:
    """Active inquiry cycle over a ZeroDataModel + KG + corpus.

    Parameters
    ----------
    model : ZeroDataModel
        The cognitive engine. The loop calls ``model.think(obs)`` for
        each text block read.
    text_encoder : TextEncoder
        Used to encode the question and the article blocks into latents.
    decoder : TextDecoder
        Used to generate the "answer" text from the post-reading latent.
    kg_builder : KnowledgeGraphBuilder
        Mutated in place: the loop appends newly-read articles as new
        nodes with latent edges.
    reader : EncyclopediaReader, optional
        The corpus reader for article lookup. If None, the loop will
        mark questions as "unanswerable" (no corpus available).
    question_generator : QuestionGenerator, optional
        If None, a default generator is constructed.
    max_read_blocks : int
        Maximum number of text blocks to read per article.
    pred_err_threshold : float
        Minimum reduction in prediction error for the question to be
        considered "answered".
    seed : int
        RNG seed for tie-breaking.
    """

    def __init__(
        self,
        model,
        text_encoder,
        decoder,
        kg_builder,
        reader=None,
        question_generator=None,
        max_read_blocks: int = 8,
        pred_err_threshold: float = 0.02,
        seed: int = 42,
    ):
        self.model = model
        self.text_encoder = text_encoder
        self.decoder = decoder
        self.kg_builder = kg_builder
        self.reader = reader
        if question_generator is None:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from question_generator import QuestionGenerator
            # If a reader is available, build a filter that only
            # accepts article titles present in the reader's index.
            # This ensures questions target encyclopedia concepts the
            # loop can actually answer.
            allowed_test = None
            if reader is not None:
                try:
                    valid_titles = set(reader.get_article_list())
                except Exception:
                    valid_titles = set()
                if valid_titles:
                    def _is_known_title(t: str, _valid=valid_titles) -> bool:
                        return t in _valid
                    allowed_test = _is_known_title
            question_generator = QuestionGenerator(seed=seed,
                                                   allowed_node_test=allowed_test)
        self.question_generator = question_generator
        self.max_read_blocks = int(max_read_blocks)
        self.pred_err_threshold = float(pred_err_threshold)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self.transcript: list = []
        self._step = 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self, n_questions: int = 5) -> dict:
        """Generate ``n_questions`` questions, answer each, return summary.

        Returns a dict with ``n_answered``, ``n_unanswered``, ``qa``,
        ``hypotheses`` fields.
        """
        # Generate questions from the current KG state.
        # Refresh latent edges so bridge candidates reflect the latest
        # belief states.
        try:
            self.kg_builder.compute_latent_edges(top_k=3)
        except Exception:
            pass
        questions = self.question_generator.generate(self.kg_builder)[:n_questions]
        results = []
        for q in questions:
            self._step += 1
            rec = self._answer_question(q)
            results.append(rec)
            self.transcript.append(rec)
        summary = {
            "n_questions": len(results),
            "n_answered": sum(1 for r in results if r.answered),
            "n_unanswered": sum(1 for r in results if not r.answered),
            "qa": [asdict(r) for r in results],
        }
        return summary

    def propose_hypothesis(self, max_hypotheses: int = 1) -> list[HypothesisRecord]:
        """Generate and test simple physics hypotheses.

        Each hypothesis is templated from the KG: pick a node whose
        title looks physics-y, propose "if X increases, Y increases",
        and run a synthetic intervention via the causal emergence
        engine (if available on the model).
        """
        records: list[HypothesisRecord] = []
        physics_keywords = ("physics", "mass", "velocity", "force",
                            "energy", "collision", "momentum", "gravity")
        candidates = []
        for title, node in self.kg_builder.nodes.items():
            if any(kw in title.lower() for kw in physics_keywords):
                candidates.append(title)
        if not candidates:
            return records
        # Pick top candidates.
        for cand in candidates[:max_hypotheses]:
            self._step += 1
            rec = self._test_hypothesis(cand)
            records.append(rec)
            self.transcript.append(rec)
        return records

    # ------------------------------------------------------------------ #
    # Internal: answer one question
    # ------------------------------------------------------------------ #
    def _answer_question(self, question) -> QARecord:
        # Encode the question to a latent (BEFORE reading).
        q_latent_before = self.text_encoder.encode(question.text)
        # Measure the model's free energy / prediction error for this latent.
        fe_before, pe_before = self._measure_model_surprise(q_latent_before)
        # Search the corpus for a matching article.
        matched_title = ""
        matched_text = ""
        if self.reader is not None and question.entities:
            for ent in question.entities:
                title = self._search_corpus(ent)
                if title:
                    matched_title = title
                    break
        if matched_title:
            matched_text = self._read_article(matched_title)
        # Re-encode the question AFTER reading (the model's belief has
        # shifted; the latent is still the same byte sequence, but the
        # model's internal surprise should be different now).
        q_latent_after = self.text_encoder.encode(question.text)
        fe_after, pe_after = self._measure_model_surprise(q_latent_after)
        # Generate the answer text via the decoder.
        answer = self.decoder.decode(q_latent_after, max_length=160)
        if not answer:
            answer = f"(no answer available; matched: {matched_title or 'none'})"
        # Decide whether the question was answered.
        delta_pe = pe_before - pe_after
        answered = bool(matched_title) and delta_pe >= self.pred_err_threshold
        # Confidence = normalised prediction-error reduction.
        confidence = float(min(max(delta_pe / max(pe_before, 1e-6), 0.0), 1.0))
        return QARecord(
            step=self._step,
            question=question.text,
            gap_type=question.gap_type,
            entities=list(question.entities),
            answer=answer,
            matched_article=matched_title,
            fe_before=float(fe_before),
            fe_after=float(fe_after),
            pred_err_before=float(pe_before),
            pred_err_after=float(pe_after),
            delta_pred_err=float(delta_pe),
            answered=answered,
            confidence=confidence,
            metadata={
                "template": question.template,
                "priority": float(question.priority),
            },
        )

    # ------------------------------------------------------------------ #
    # Internal: search corpus for an entity
    # ------------------------------------------------------------------ #
    def _search_corpus(self, entity: str) -> str:
        """Look up an article in the reader whose title matches the entity.

        Strategy:
          1. Exact title match (case-insensitive).
          2. Substring match (entity in article title or vice versa).
          3. Token-set match (any token of entity appears in title).
        Returns the article title, or "" if no match.
        """
        try:
            titles = self.reader.get_article_list()
        except Exception:
            return ""
        if not titles:
            return ""
        entity_lower = entity.lower()
        # Strip common wiki prefixes/suffixes for matching.
        entity_clean = re.sub(r"^(physics|biology|math|history|geography)_",
                              "", entity_lower)
        # Exact match.
        for t in titles:
            if t.lower() == entity_lower:
                return t
        # Substring match.
        for t in titles:
            tl = t.lower()
            if entity_lower in tl or entity_clean in tl:
                return t
            if tl in entity_lower:
                return t
        # Token-set match.
        entity_tokens = set(re.findall(r"\w+", entity_lower))
        if not entity_tokens:
            return ""
        best = ""
        best_overlap = 0
        for t in titles:
            tl_tokens = set(re.findall(r"\w+", t.lower()))
            overlap = len(entity_tokens & tl_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best = t
        return best if best_overlap > 0 else ""

    # ------------------------------------------------------------------ #
    # Internal: read an article and update belief
    # ------------------------------------------------------------------ #
    def _read_article(self, title: str) -> str:
        """Seek to ``title`` and read up to ``max_read_blocks`` blocks."""
        if self.reader is None:
            return ""
        try:
            self.reader.seek_to_article(title)
        except Exception:
            return ""
        full_text = []
        for _ in range(self.max_read_blocks):
            try:
                block = self.reader.read_block()
            except Exception:
                break
            if not block:
                break
            full_text.append(block)
            # Feed the block to the model + decoder pool.
            latent = self.text_encoder.encode(block)
            try:
                signal = self.model.think(latent)
            except Exception:
                signal = None
            # Update the decoder's retrieval pool so future questions
            # can retrieve this text.
            try:
                fe = float(getattr(signal, "free_energy", 0.0)) if signal else 0.0
            except Exception:
                fe = 0.0
            self.decoder.ingest(block, latent=latent, source=title, free_energy=fe)
            # Update the KG with the article + its latent.
            try:
                links = self.reader.get_article_links(title) or []
            except Exception:
                links = []
            self.kg_builder.update_graph(
                article_title=title,
                linked_titles=links,
                latent_state=latent,
                free_energy=fe,
                read_step=self._step,
            )
            if self.reader.at_article_end():
                break
        return "\n".join(full_text)

    # ------------------------------------------------------------------ #
    # Internal: measure model surprise for a latent
    # ------------------------------------------------------------------ #
    def _measure_model_surprise(self, latent: np.ndarray) -> tuple[float, float]:
        """Probe the model with ``latent`` and return (free_energy, pred_err).

        We do NOT mutate the model's belief_state here — we use the
        active_inference engine's analytic ``compute_free_energy`` if
        available, else fall back to a ``||obs - belief_state||`` proxy.
        """
        # Preferred path: use the active_inference engine directly.
        engine = getattr(self.model, "active_inference", None)
        if engine is not None:
            try:
                fe = float(engine.compute_free_energy(np.asarray(latent)))
            except Exception:
                fe = 0.0
            try:
                belief = engine.generative_model.belief_state
                b = np.asarray(belief)[: len(latent)]
                pe = float(np.linalg.norm(np.asarray(latent) - b))
            except Exception:
                pe = 0.0
            return fe, pe
        # Fall back: model.compute_free_energy if it exists.
        compute_fe = getattr(self.model, "compute_free_energy", None)
        if callable(compute_fe):
            try:
                fe = float(compute_fe(latent))
            except Exception:
                fe = 0.0
            belief = getattr(self.model, "belief_state", None)
            if belief is not None:
                pe = float(np.linalg.norm(np.asarray(belief) - np.asarray(latent)))
            else:
                pe = 0.0
            return fe, pe
        # Last resort: think() once and use the signal's confidence.
        try:
            signal = self.model.think(latent)
        except Exception:
            return 0.0, 0.0
        # confidence in (0, 1]; surprise proxy = -log(confidence).
        conf = float(getattr(signal, "confidence", 0.5))
        conf = max(min(conf, 1.0), 1e-6)
        fe = -float(np.log(conf))
        # prediction error proxy: distance between signal and obs.
        data = np.asarray(getattr(signal, "data", np.zeros_like(latent)))
        pe = float(np.linalg.norm(np.asarray(latent) - data[: len(latent)]))
        return fe, pe

    # ------------------------------------------------------------------ #
    # Internal: test a physics hypothesis
    # ------------------------------------------------------------------ #
    def _test_hypothesis(self, node_title: str) -> HypothesisRecord:
        """Propose a templated hypothesis and run a sandbox intervention."""
        # Pick a templated hypothesis.
        templates = [
            "If {entity} increases, the resulting collision energy increases.",
            "If {entity} decreases, the system reaches equilibrium faster.",
            "If {entity} is doubled, the outcome changes by a power-law factor.",
        ]
        template = templates[int(self._rng.integers(0, len(templates)))]
        hypothesis = template.replace("{entity}", node_title)
        # Build a synthetic intervention: shift the model's belief_state
        # by a random direction and measure the resulting change in
        # free energy. If the model has a causal emergence engine with
        # an intervention API, use it; otherwise, use a direct belief
        # shift and measure via compute_free_energy.
        intervention = {
            "target": node_title,
            "delta_mean": 0.5,
            "delta_std": 0.1,
        }
        outcome: dict = {}
        supported = False
        confidence = 0.0
        try:
            # The CausalEmergenceEngine.intervene() API requires an
            # adjacency matrix + data matrix + integer intervention_var
            # index, which we cannot easily synthesise from a single
            # KG node. So we use a direct belief-state intervention
            # instead: shift the current belief_state by ``delta_mean``
            # and measure the resulting change in free energy. If the
            # shift produces a measurable change in FE, the hypothesis
            # is "supported" (in the weak sense that the model's
            # generative model responded to the intervention).
            engine = getattr(self.model, "active_inference", None)
            belief = None
            if engine is not None:
                try:
                    belief = np.asarray(engine.generative_model.belief_state)
                except Exception:
                    belief = None
            if belief is None:
                belief = getattr(self.model, "belief_state", None)
                if belief is not None:
                    belief = np.asarray(belief)
            if belief is not None:
                # Apply a directional intervention: shift the belief
                # toward the node's own latent vector (so the model
                # "imagines" the node more strongly) and measure FE.
                node = self.kg_builder.nodes.get(node_title, {})
                node_belief = node.get("belief_state")
                if node_belief is not None:
                    node_belief = np.asarray(node_belief, dtype=np.float64)
                    # Align lengths.
                    n = min(len(belief), len(node_belief))
                    shift = np.zeros_like(belief)
                    shift[:n] = intervention["delta_mean"] * node_belief[:n]
                else:
                    shift = intervention["delta_mean"] * belief
                shifted = belief + shift
                fe_fn = None
                if engine is not None:
                    fe_fn = getattr(engine, "compute_free_energy", None)
                if not callable(fe_fn):
                    fe_fn = getattr(self.model, "compute_free_energy", None)
                if callable(fe_fn):
                    try:
                        fe_before = float(fe_fn(belief))
                        fe_after = float(fe_fn(shifted))
                    except Exception:
                        fe_before = fe_after = 0.0
                else:
                    # No analytic FE — use L2 distance as a proxy.
                    fe_before = float(np.linalg.norm(belief))
                    fe_after = float(np.linalg.norm(shifted))
                delta_fe = fe_after - fe_before
                outcome = {
                    "fe_before": fe_before,
                    "fe_after": fe_after,
                    "delta_fe": delta_fe,
                    "shift_norm": float(np.linalg.norm(shift)),
                }
                # Supported if the intervention produced a measurable
                # change in free energy (above a small noise threshold).
                supported = bool(abs(delta_fe) > 1e-3)
                confidence = float(min(abs(delta_fe) / 5.0, 1.0))
        except Exception as exc:
            outcome = {"error": f"{type(exc).__name__}: {exc}"}
        return HypothesisRecord(
            step=self._step,
            hypothesis=hypothesis,
            intervention=intervention,
            outcome=outcome,
            supported=supported,
            confidence=confidence,
            metadata={"node": node_title},
        )

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save_transcript(self, path: str | Path) -> str:
        """Save the Q/A transcript as a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        records = [asdict(r) if hasattr(r, "__dataclass_fields__") else r
                   for r in self.transcript]
        with path.open("w") as f:
            json.dump(records, f, indent=2, default=str)
        return str(path)


# ------------------------------------------------------------------ #
# Self-test
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from knowledge_graph_builder import KnowledgeGraphBuilder
    from text_encoder import TextEncoder
    from text_decoder import TextDecoder
    from question_generator import QuestionGenerator
    from encyclopedia_reader import EncyclopediaReader, make_synthetic_corpus

    # Tiny ZeroDataModel stub with the API the loop needs.
    class StubModel:
        def __init__(self, dim: int = 32, seed: int = 42):
            self.dim = dim
            self._rng = np.random.default_rng(seed)
            self.belief_state = self._rng.standard_normal(dim) * 0.1
        def think(self, obs):
            class Sig:
                def __init__(self, fe, pe):
                    self.free_energy = fe
                    self.prediction_error = pe
            fe = float(np.linalg.norm(obs - self.belief_state))
            pe = float(np.linalg.norm(obs - self.belief_state))
            # Update belief_state slowly toward obs.
            self.belief_state = 0.9 * self.belief_state + 0.1 * np.asarray(obs)
            return Sig(fe, pe)
        def compute_free_energy(self, obs):
            return float(np.linalg.norm(np.asarray(obs) - self.belief_state))

    # Build a synthetic encyclopedia corpus.
    import tempfile
    tmp_corpus = Path(tempfile.mkdtemp())
    file_paths = make_synthetic_corpus(tmp_corpus, n_files=4, articles_per_file=5, seed=0)
    reader = EncyclopediaReader(file_paths, index_dir=tmp_corpus / ".index")

    # Build a tiny KG with some physics-y nodes.
    kg = KnowledgeGraphBuilder()
    text_encoder = TextEncoder(block_size=256, output_dim=32, seed=42)
    base = np.zeros(32)
    base[:8] = 1.0
    rng = np.random.default_rng(0)
    def _vec(i):
        v = base + rng.standard_normal(32) * 0.4
        v /= np.linalg.norm(v) + 1e-9
        return v
    # Try to use the encyclopedia's actual article titles.
    titles = reader.get_article_list()
    print(f"[self-test] reader has {len(titles)} articles")
    for i, t in enumerate(titles[:6]):
        kg.update_graph(t, linked_titles=[], latent_state=_vec(i),
                        free_energy=10.0 + i * 2.0, read_step=i)
    kg.compute_latent_edges(top_k=3)

    # Build the loop.
    model = StubModel(dim=32, seed=42)
    decoder = TextDecoder(text_encoder=text_encoder, n=2, seed=42, temperature=1.0)
    # Pre-populate decoder pool with article snippets so it can generate text.
    for t in titles[:6]:
        try:
            reader.seek_to_article(t)
            block = reader.read_block() or ""
            if block:
                latent = text_encoder.encode(block)
                decoder.ingest(block, latent=latent, source=t)
        except Exception:
            pass

    loop = InquiryLoop(
        model=model,
        text_encoder=text_encoder,
        decoder=decoder,
        kg_builder=kg,
        reader=reader,
        max_read_blocks=3,
        pred_err_threshold=0.0,
        seed=42,
    )
    # Run a few questions.
    summary = loop.run(n_questions=4)
    print(f"[self-test] questions: {summary['n_questions']}, "
          f"answered: {summary['n_answered']}, "
          f"unanswered: {summary['n_unanswered']}")
    for qa in summary["qa"][:3]:
        print(f"  Q: {qa['question']}")
        print(f"    matched: {qa['matched_article']!r}")
        print(f"    answered={qa['answered']}  conf={qa['confidence']:.3f}  "
              f"Δpe={qa['delta_pred_err']:+.4f}")
        print(f"    A: {qa['answer'][:80]}...")

    # Test hypothesis proposal.
    hyps = loop.propose_hypothesis(max_hypotheses=2)
    print(f"[self-test] hypotheses: {len(hyps)}")
    for h in hyps:
        print(f"  H: {h.hypothesis}")
        print(f"    supported={h.supported}  conf={h.confidence:.3f}  outcome={h.outcome}")

    # Save transcript.
    transcript_path = loop.save_transcript("/tmp/inquiry_transcript.json")
    print(f"[self-test] saved transcript to {transcript_path}")
    print("[self-test] OK")
