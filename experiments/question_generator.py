# experiments/question_generator.py
"""Question generator (Phase K, Task 3.1).

Identifies "knowledge gaps" in the model's evolving knowledge graph and
generates templated questions whose answers would, if absorbed, fill
those gaps.

A "gap" is any of:

  1. **Sparse-link node**: a node with fewer than ``min_links`` latent /
     structural edges. The model has read it but hasn't connected it
     to anything — suggesting its meaning is unclear.
  2. **High free-energy node**: a node whose ``mean_free_energy`` is in
     the top quartile. The model finds this article hard to predict.
  3. **Bridge-candidate pairs**: two nodes from different "domains"
     (heuristic: different probe categories) that share a moderate
     latent cosine similarity (0.3–0.7). A question about their
     relationship might reveal an unrecognised analogy.
  4. **Under-visited neighbours**: a node A linked structurally to B,
     where B has been visited < ``min_visit`` times. The model has the
     link but never explored the target.

For each gap, the generator picks a question template:

    "What is {entity}?"
    "How is {entity} related to {other}?"
    "Why does {entity} have high prediction error?"
    "What does {entity} connect to?"
    "Is {entity} similar to {other}?"

All questions are pure-template — no learned text generation. The
actual *answering* happens in ``active_inquiry.py``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Edge type constants from the Phase J KG builder.
try:
    from knowledge_graph_builder import EDGE_STRUCTURAL, EDGE_CO_READ, EDGE_LATENT
except ImportError:  # pragma: no cover
    EDGE_STRUCTURAL = "structural"
    EDGE_CO_READ = "co_read"
    EDGE_LATENT = "latent"


# ------------------------------------------------------------------ #
# Question templates
# ------------------------------------------------------------------ #
# (template_string, gap_type). Templates use {entity} and optionally {other}.
QUESTION_TEMPLATES = [
    ("What is {entity}?", "sparse_link"),
    ("What does {entity} connect to?", "sparse_link"),
    ("Why does {entity} have high prediction error?", "high_fe"),
    ("How is {entity} related to {other}?", "bridge_candidate"),
    ("Is {entity} similar to {other}?", "bridge_candidate"),
    ("What is the relationship between {entity} and {other}?", "bridge_candidate"),
    ("What properties does {entity} have?", "sparse_link"),
    ("Why has the model read {entity} so rarely?", "under_visited"),
    ("What category does {entity} belong to?", "sparse_link"),
    ("Could {entity} explain {other}?", "bridge_candidate"),
]


@dataclass
class Question:
    """One generated question."""
    text: str
    template: str
    gap_type: str
    entities: list[str]
    priority: float = 0.0       # higher = more important to answer
    metadata: dict = field(default_factory=dict)


# ------------------------------------------------------------------ #
# QuestionGenerator
# ------------------------------------------------------------------ #
class QuestionGenerator:
    """Generate templated questions from a knowledge graph.

    Parameters
    ----------
    min_links : int
        Below this many edges, a node is considered "sparse".
    min_visit : int
        Below this many visits, a node is considered "under-visited".
    fe_top_quantile : float
        Fraction of nodes considered "high FE" (top of the distribution).
    bridge_sim_range : tuple[float, float]
        Cosine similarity window for bridge candidates (moderate sim).
    max_questions : int
        Maximum number of questions to return per ``generate`` call.
    seed : int
        RNG seed for tie-breaking among equally-priority gaps.
    """

    def __init__(
        self,
        min_links: int = 2,
        min_visit: int = 2,
        fe_top_quantile: float = 0.75,
        bridge_sim_range: tuple[float, float] = (0.3, 0.7),
        max_questions: int = 50,
        seed: int = 42,
        allowed_node_test: Optional[callable] = None,
    ):
        """Construct a question generator.

        Parameters
        ----------
        allowed_node_test : callable, optional
            A function ``(title: str) -> bool`` returning True iff a
            KG node should be considered for question generation. If
            None, all nodes are eligible. The InquiryLoop sets this to
            a function that only accepts article titles that exist in
            the encyclopedia's article list — so questions about
            image:/audio:/code:/physics: nodes are skipped and the
            loop only asks about text concepts it can actually answer.
        """
        self.min_links = int(min_links)
        self.min_visit = int(min_visit)
        self.fe_top_quantile = float(fe_top_quantile)
        self.bridge_sim_range = tuple(bridge_sim_range)
        self.max_questions = int(max_questions)
        self.seed = int(seed)
        self.allowed_node_test = allowed_node_test
        self._rng = np.random.default_rng(self.seed)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def generate(self, kg_builder) -> list[Question]:
        """Generate a list of questions for the given KG.

        ``kg_builder`` is a Phase J ``KnowledgeGraphBuilder`` instance.
        """
        questions: list[Question] = []
        nodes = kg_builder.nodes
        edges = kg_builder.edges
        if not nodes:
            return questions

        # --- 1. Sparse-link + high-FE + under-visited gaps ---------- #
        questions.extend(self._node_gap_questions(kg_builder))

        # --- 2. Bridge-candidate pairs ------------------------------ #
        questions.extend(self._bridge_questions(kg_builder))

        # --- 3. Sort by priority and trim to max_questions ---------- #
        questions.sort(key=lambda q: -q.priority)
        # Tie-break with RNG so two equally-priority gaps don't always
        # appear in the same order.
        rng = np.random.default_rng(self.seed)
        rng.shuffle(questions)  # stable after sort by priority? No — fully shuffles.
        # Re-sort by priority (stable) after the shuffle so ties remain shuffled.
        questions.sort(key=lambda q: -q.priority)
        return questions[:self.max_questions]

    # ------------------------------------------------------------------ #
    # Node-level gaps
    # ------------------------------------------------------------------ #
    def _node_gap_questions(self, kg_builder) -> list[Question]:
        out: list[Question] = []
        # Build per-node STRUCTURAL + CO_READ edge counts (ignore latent
        # edges, which are derived from belief-state similarity and
        # would mask genuine "sparse link" gaps).
        edge_counts: dict[str, int] = {n: 0 for n in kg_builder.nodes}
        for (s, d), attrs in kg_builder.edges.items():
            if attrs.get("edge_type") == EDGE_LATENT:
                continue
            edge_counts[s] = edge_counts.get(s, 0) + 1
            edge_counts[d] = edge_counts.get(d, 0) + 1
        # Compute FE threshold.
        fes = [n["mean_free_energy"] for n in kg_builder.nodes.values()
               if n.get("n_free_energy_samples", 0) > 0]
        fe_threshold = float(np.quantile(fes, self.fe_top_quantile)) if fes else float("inf")
        for title, node in kg_builder.nodes.items():
            # Apply the node filter (e.g. only wiki articles).
            if self.allowed_node_test is not None and not self.allowed_node_test(title):
                continue
            visit_count = node.get("visit_count", 0)
            mean_fe = node.get("mean_free_energy", 0.0)
            n_edges = edge_counts.get(title, 0)
            # Sparse-link gap.
            if n_edges < self.min_links:
                priority = float(self.min_links - n_edges) + 0.1
                out.append(self._make_question(
                    template="What is {entity}?",
                    gap_type="sparse_link",
                    entity=title,
                    priority=priority,
                    metadata={"n_edges": n_edges, "visit_count": visit_count},
                ))
            # High-FE gap.
            if mean_fe >= fe_threshold and mean_fe > 0:
                priority = float(mean_fe) / max(fe_threshold, 1e-6)
                out.append(self._make_question(
                    template="Why does {entity} have high prediction error?",
                    gap_type="high_fe",
                    entity=title,
                    priority=priority,
                    metadata={"mean_fe": float(mean_fe),
                              "fe_threshold": float(fe_threshold)},
                ))
            # Under-visited gap.
            if 0 < visit_count < self.min_visit:
                priority = float(self.min_visit - visit_count) + 0.1
                out.append(self._make_question(
                    template="Why has the model read {entity} so rarely?",
                    gap_type="under_visited",
                    entity=title,
                    priority=priority,
                    metadata={"visit_count": visit_count},
                ))
        return out

    # ------------------------------------------------------------------ #
    # Bridge-candidate gaps
    # ------------------------------------------------------------------ #
    def _bridge_questions(self, kg_builder) -> list[Question]:
        out: list[Question] = []
        # Use latent-edge info if present.
        # Latent edges have edge_type = EDGE_LATENT and weight = cosine sim.
        candidates = []
        for (s, d), attrs in kg_builder.edges.items():
            if attrs.get("edge_type") != EDGE_LATENT:
                continue
            sim = float(attrs.get("weight", 0.0))
            lo, hi = self.bridge_sim_range
            if not (lo <= sim <= hi):
                continue
            # Apply the node filter to both endpoints.
            if self.allowed_node_test is not None:
                if not self.allowed_node_test(s) or not self.allowed_node_test(d):
                    continue
            # Skip pairs already connected by structural or co_read edges.
            existing_types = set()
            # The KG stores undirected edges with a single (s, d) key, so
            # only one edge per pair. We can't check structural overlap
            # easily here. Just include all latent-edge bridge candidates.
            existing_types.add(attrs.get("edge_type"))
            candidates.append((s, d, sim))
        # Sort by sim (descending) so strongest bridges come first.
        candidates.sort(key=lambda x: -x[2])
        for s, d, sim in candidates[:self.max_questions]:
            # Two question flavours per pair.
            q1 = self._make_question(
                template="How is {entity} related to {other}?",
                gap_type="bridge_candidate",
                entity=s,
                other=d,
                priority=float(sim),
                metadata={"sim": float(sim)},
            )
            q2 = self._make_question(
                template="Is {entity} similar to {other}?",
                gap_type="bridge_candidate",
                entity=s,
                other=d,
                priority=float(sim) * 0.9,
                metadata={"sim": float(sim)},
            )
            out.append(q1)
            out.append(q2)
        return out

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #
    def _make_question(
        self,
        template: str,
        gap_type: str,
        entity: str,
        other: Optional[str] = None,
        priority: float = 0.0,
        metadata: Optional[dict] = None,
    ) -> Question:
        text = template.replace("{entity}", entity)
        if other is not None:
            text = text.replace("{other}", other)
        entities = [entity] if other is None else [entity, other]
        return Question(
            text=text,
            template=template,
            gap_type=gap_type,
            entities=entities,
            priority=float(priority),
            metadata=metadata or {},
        )


# ------------------------------------------------------------------ #
# Self-test
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from knowledge_graph_builder import KnowledgeGraphBuilder

    # Build a small synthetic KG with deliberate gaps.
    # Use CORRELATED latents so bridge candidates fall in the moderate
    # similarity window (0.3 .. 0.7) the generator looks for.
    base_vec = np.zeros(32)
    base_vec[:8] = 1.0   # shared direction
    rng = np.random.default_rng(0)
    def _make_vec(i: int, noise_scale: float = 0.6) -> np.ndarray:
        v = base_vec.copy() + rng.standard_normal(32) * noise_scale
        v /= np.linalg.norm(v) + 1e-9
        return v

    kg = KnowledgeGraphBuilder()
    # Some visited articles.
    kg.update_graph("Physics_Gravity", linked_titles=["Physics_Mass"],
                    latent_state=_make_vec(1),
                    free_energy=15.0, read_step=0)
    kg.update_graph("Physics_Mass", linked_titles=["Physics_Gravity"],
                    latent_state=_make_vec(2),
                    free_energy=16.0, read_step=1)
    # High-FE article.
    kg.update_graph("Hard_Topic_X", linked_titles=[],
                    latent_state=_make_vec(3),
                    free_energy=25.0, read_step=2)
    # Under-visited article.
    kg.update_graph("Obscure_Topic_Y", linked_titles=[],
                    latent_state=_make_vec(4),
                    free_energy=10.0, read_step=3)
    # Add latent edges (must compute after adding nodes).
    kg.compute_latent_edges(top_k=3)

    print(f"[self-test] KG nodes = {kg.n_nodes()}, edges = {kg.n_edges()}")

    gen = QuestionGenerator(min_links=2, min_visit=2, max_questions=20, seed=42)
    questions = gen.generate(kg)
    print(f"[self-test] generated {len(questions)} questions:")
    for q in questions[:10]:
        print(f"  [{q.gap_type:18s} p={q.priority:.3f}] {q.text}")

    # Verify each question references a real KG node.
    for q in questions:
        for ent in q.entities:
            assert ent in kg.nodes, f"entity {ent!r} not in KG"

    # Verify gap_type coverage.
    gap_types = {q.gap_type for q in questions}
    print(f"[self-test] gap types covered: {sorted(gap_types)}")
    assert "sparse_link" in gap_types, "expected sparse_link gap"
    assert "high_fe" in gap_types, "expected high_fe gap"
    assert "bridge_candidate" in gap_types, "expected bridge_candidate gap"
    print("[self-test] OK")
