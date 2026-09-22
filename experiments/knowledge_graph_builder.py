# experiments/knowledge_graph_builder.py
"""Dynamic knowledge-graph builder (Phase J, Task 3.1).

Builds a ``networkx`` graph capturing the model's READING HISTORY +
INTERNAL STATE, as opposed to the literal corpus link graph. The
nodes are articles the model has actually visited; edges record:

  - **structural** (weight=1): the article appears in the corpus's
    internal link list (``[[Target]]`` markers). These are the
    "ground-truth" links from the source.
  - **co-read** (weight = n co-reads): two articles visited in
    adjacent steps. Reflects the model's reading trajectory.
  - **latent** (weight = cosine sim of belief_states): two articles
    with similar internal representations. Reflects the model's
    learned semantic structure — independent of the corpus links.

The builder is INCREMENTAL: ``update_graph(article_title, ...)`` is
called every step (or every snapshot) and accumulates edge weights.
The graph is periodically saved to disk as both GraphML (for Gephi /
Cytoscape import) and JSON (for the visualiser).

Why networkx?
  - It's the de-facto Python graph library, ships with most distros.
  - GraphML export is supported natively (``write_graphml``).
  - If unavailable, the builder falls back to a pure-dict adjacency
    representation that still supports JSON export.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Optional: networkx for full graph library support.
try:
    import networkx as nx
    _HAS_NX = True
except ImportError:  # pragma: no cover
    _HAS_NX = False
    nx = None  # type: ignore[assignment]


# ------------------------------------------------------------------ #
# Edge types
# ------------------------------------------------------------------ #
EDGE_STRUCTURAL = "structural"  # from corpus [[Target]] links
EDGE_CO_READ    = "co_read"     # articles visited in adjacent steps
EDGE_LATENT     = "latent"      # cosine sim of belief_states


# ------------------------------------------------------------------ #
# KnowledgeGraph builder
# ------------------------------------------------------------------ #
@dataclass
class KnowledgeGraphBuilder:
    """Incremental knowledge-graph builder.

    Usage:
        builder = KnowledgeGraphBuilder()
        for step in steps:
            builder.update_graph(
                article_title="Apple",
                linked_titles=["Fruit", "Tree"],
                latent_state=belief_state,
                read_step=step,
            )
        builder.save("kg.json")
    """

    # Node attributes: title -> {visit_count, last_visit_step,
    #                            belief_state (latest), mean_free_energy}.
    nodes: dict[str, dict] = field(default_factory=dict)
    # Edge key: (src, dst) sorted tuple. Value: {weight, edge_type,
    #                                            last_updated_step}.
    edges: dict[tuple[str, str], dict] = field(default_factory=dict)
    # Track the previous article for co-read edge detection.
    _last_article: str | None = None
    # Track all belief_states per article for latent-edge computation.
    _belief_states: dict[str, list[np.ndarray]] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def update_graph(
        self,
        article_title: str,
        linked_titles: list[str] | None = None,
        latent_state: np.ndarray | None = None,
        free_energy: float | None = None,
        read_step: int = 0,
    ) -> None:
        """Record one read of ``article_title``.

        Updates:
          - The node's visit_count, last_visit_step, belief_state.
          - STRUCTURAL edges to ``linked_titles`` (each weight=1).
          - CO_READ edge from the previous article (weight += 1).
        """
        if not article_title:
            return
        # Node update.
        if article_title not in self.nodes:
            self.nodes[article_title] = {
                "visit_count": 0,
                "last_visit_step": read_step,
                "belief_state": None,
                "mean_free_energy": 0.0,
                "n_free_energy_samples": 0,
            }
        node = self.nodes[article_title]
        node["visit_count"] += 1
        node["last_visit_step"] = read_step
        if latent_state is not None:
            node["belief_state"] = np.asarray(latent_state, dtype=float).copy()
            # Track for latent-edge computation.
            if article_title not in self._belief_states:
                self._belief_states[article_title] = []
            self._belief_states[article_title].append(node["belief_state"])
            # Cap memory.
            if len(self._belief_states[article_title]) > 20:
                self._belief_states[article_title] = \
                    self._belief_states[article_title][-20:]
        if free_energy is not None:
            n = node["n_free_energy_samples"]
            node["mean_free_energy"] = (
                (node["mean_free_energy"] * n + free_energy) / (n + 1)
            )
            node["n_free_energy_samples"] = n + 1
        # Structural edges (corpus links).
        if linked_titles:
            for target in linked_titles:
                if target and target != article_title:
                    self._add_edge(article_title, target, EDGE_STRUCTURAL,
                                  weight=1.0, step=read_step)
        # Co-read edge from the previous article.
        if self._last_article is not None and self._last_article != article_title:
            self._add_edge(self._last_article, article_title, EDGE_CO_READ,
                          weight=1.0, step=read_step)
        self._last_article = article_title

    def compute_latent_edges(self, top_k: int = 5) -> int:
        """For each node, find the top-k most similar other nodes
        (by cosine sim of latest belief_state) and add LATENT edges.

        Returns the number of latent edges added.
        """
        # Get latest belief_state per node.
        nodes_with_belief = {
            t: n["belief_state"] for t, n in self.nodes.items()
            if n["belief_state"] is not None
        }
        if len(nodes_with_belief) < 2:
            return 0
        titles = list(nodes_with_belief.keys())
        # Build the (n, dim) matrix.
        dim = max(v.shape[0] for v in nodes_with_belief.values())
        mat = np.zeros((len(titles), dim))
        for i, t in enumerate(titles):
            v = nodes_with_belief[t]
            mat[i, :v.shape[0]] = v
        # Normalise.
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        mat_normed = mat / norms
        # Cosine sim matrix.
        cos_sim = mat_normed @ mat_normed.T
        # For each node, take the top-k highest sims (excluding self).
        n_added = 0
        for i, t in enumerate(titles):
            sims = cos_sim[i].copy()
            sims[i] = -1.0  # exclude self
            top_idx = np.argsort(sims)[::-1][:top_k]
            for j in top_idx:
                if sims[j] <= 0:
                    continue
                other = titles[j]
                self._add_edge(t, other, EDGE_LATENT,
                              weight=float(sims[j]), step=0)
                n_added += 1
        return n_added

    def get_graph(self):
        """Return a ``networkx.Graph`` if available, else None."""
        if not _HAS_NX:
            return None
        G = nx.Graph()
        for title, attrs in self.nodes.items():
            G.add_node(title, **_jsonable(attrs))
        for (src, dst), attrs in self.edges.items():
            G.add_edge(src, dst, **_jsonable(attrs))
        return G

    def n_nodes(self) -> int:
        return len(self.nodes)

    def n_edges(self) -> int:
        return len(self.edges)

    def top_articles(self, k: int = 10) -> list[tuple[str, int]]:
        """Return the top-k most-visited articles as (title, count)."""
        return sorted(
            ((t, n["visit_count"]) for t, n in self.nodes.items()),
            key=lambda x: -x[1],
        )[:k]

    def highest_fe_articles(self, k: int = 10) -> list[tuple[str, float]]:
        """Return the articles with the highest mean free energy."""
        return sorted(
            ((t, n["mean_free_energy"]) for t, n in self.nodes.items()
             if n["n_free_energy_samples"] > 0),
            key=lambda x: -x[1],
        )[:k]

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return {
            "schema_version": "1.0-kg",
            "nodes": {t: _jsonable(n) for t, n in self.nodes.items()},
            "edges": [
                {"src": s, "dst": d, **_jsonable(a)}
                for (s, d), a in self.edges.items()
            ],
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
        }

    def save_json(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return str(path)

    def save_graphml(self, path: str | Path) -> str | None:
        """Save as GraphML (requires networkx)."""
        if not _HAS_NX:
            return None
        G = self.get_graph()
        if G is None or G.number_of_nodes() == 0:
            return None
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # networkx write_graphml doesn't support nested dicts/lists in attrs.
        # Flatten node attrs.
        for n, attrs in G.nodes(data=True):
            if "belief_state" in attrs and attrs["belief_state"] is not None:
                attrs["belief_state"] = ""  # can't serialise ndarray to GraphML
        nx.write_graphml(G, str(path))
        return str(path)

    @classmethod
    def load_json(cls, path: str | Path) -> "KnowledgeGraphBuilder":
        with Path(path).open() as f:
            data = json.load(f)
        builder = cls()
        for title, attrs in data.get("nodes", {}).items():
            # Reconstruct belief_state from list if present.
            bs = attrs.get("belief_state")
            if isinstance(bs, list):
                attrs["belief_state"] = np.asarray(bs, dtype=float)
            builder.nodes[title] = attrs
        for edge in data.get("edges", []):
            src = edge.pop("src")
            dst = edge.pop("dst")
            builder.edges[(src, dst)] = edge
        return builder

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #
    def _add_edge(
        self,
        src: str,
        dst: str,
        edge_type: str,
        weight: float = 1.0,
        step: int = 0,
    ) -> None:
        # Use a canonical key (alphabetical order) so undirected edges
        # don't double up.
        key = (src, dst) if src <= dst else (dst, src)
        if key not in self.edges:
            self.edges[key] = {
                "weight": 0.0,
                "edge_type": edge_type,
                "last_updated_step": step,
                "n_updates": 0,
            }
        edge = self.edges[key]
        # For structural edges, weight stays at 1.0 (existence).
        # For co-read edges, weight accumulates (count).
        # For latent edges, weight is the latest cosine sim.
        if edge_type == EDGE_CO_READ:
            edge["weight"] += weight
        elif edge_type == EDGE_LATENT:
            edge["weight"] = weight  # overwrite with latest
        elif edge_type == EDGE_STRUCTURAL:
            edge["weight"] = max(edge["weight"], weight)  # keep = 1
        edge["n_updates"] += 1
        edge["last_updated_step"] = step


def _jsonable(d: dict) -> dict:
    """Convert numpy arrays / scalars in a dict to JSON-serialisable types."""
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = float(v) if isinstance(v, np.floating) else int(v)
        else:
            out[k] = v
    return out


# ------------------------------------------------------------------ #
# Build a KG from a run's records (offline analysis)
# ------------------------------------------------------------------ #
def build_from_records(
    records: list[dict],
    reader=None,
    model=None,
    encoder=None,
) -> KnowledgeGraphBuilder:
    """Build a knowledge graph from a list of run records.

    ``records`` is a list of dicts with at least:
      - ``step``
      - ``article_title``
      - ``nav_action`` (int)
      - ``free_energy`` (float)
      - (optional) ``belief_state`` (np.ndarray)

    If ``reader`` is provided, structural links are looked up from the
    corpus index. If ``model`` + ``encoder`` are provided, latent
    states are computed by encoding each article's first block and
    inferring the state (without mutating the model).

    Returns the populated builder.
    """
    builder = KnowledgeGraphBuilder()
    last_title = None
    for rec in records:
        title = rec.get("article_title", "")
        if not title:
            continue
        # Lookup structural links if a reader is provided.
        links = []
        if reader is not None:
            links = reader.get_article_links(title)
        # Get belief_state from the record if present, else compute.
        belief = rec.get("belief_state")
        if belief is None and model is not None and encoder is not None \
                and reader is not None:
            # Compute a one-shot belief (without mutating model state).
            if reader.seek_to_article(title):
                block = reader.read_block()
                if block:
                    obs = encoder.encode(block)
                    gen = model.active_inference.generative_model
                    belief, _ = gen.infer_state(obs)
        # Coerce numeric fields (CSV returns strings).
        try:
            free_energy = float(rec.get("free_energy", 0)) if rec.get("free_energy") else None
        except (ValueError, TypeError):
            free_energy = None
        try:
            step = int(rec.get("step", 0)) if rec.get("step") else 0
        except (ValueError, TypeError):
            step = 0
        builder.update_graph(
            article_title=title,
            linked_titles=links,
            latent_state=belief,
            free_energy=free_energy,
            read_step=step,
        )
        last_title = title
    # Compute latent edges from the accumulated belief_states.
    builder.compute_latent_edges(top_k=5)
    return builder


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a knowledge graph from a run's CSV log.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv", type=str, required=False, default="",
                   help="Path to encyclopedia_run.csv. If empty, uses default.")
    p.add_argument("--corpus_dir", type=str, default="",
                   help="Corpus dir (for structural link lookup).")
    p.add_argument("--output", type=str,
                   default=str(Path(__file__).resolve().parent
                              / "output" / "encyclopedia_run" / "knowledge_graph.json"),
                   help="Output JSON path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 64)
    print("Knowledge Graph Builder")
    print("=" * 64)
    csv_path = args.csv or str(
        Path(__file__).resolve().parent
        / "output" / "encyclopedia_run" / "encyclopedia_run.csv"
    )
    if not Path(csv_path).exists():
        print(f"CSV not found: {csv_path}")
        print("Run run_encyclopedia_learning.py first.")
        return
    import csv as csv_mod
    records = []
    with Path(csv_path).open() as f:
        for row in csv_mod.DictReader(f):
            records.append(row)
    print(f"  loaded {len(records)} records from {csv_path}")
    reader = None
    if args.corpus_dir:
        # Build a reader for structural link lookup.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from encyclopedia_reader import EncyclopediaReader, make_synthetic_corpus
        corpus_dir = Path(args.corpus_dir)
        if not corpus_dir.exists():
            corpus_dir = Path(args.output).parent / "synthetic_corpus"
        if not corpus_dir.exists() or not any(corpus_dir.iterdir()):
            make_synthetic_corpus(corpus_dir, n_files=2, articles_per_file=10, seed=42)
        paths = sorted(corpus_dir.glob("wiki_*"))
        reader = EncyclopediaReader(paths, block_size=256, index_dir=corpus_dir / ".index")
        print(f"  reader: {reader.n_articles()} articles")
    builder = build_from_records(records, reader=reader)
    builder.compute_latent_edges(top_k=5)
    json_path = builder.save_json(args.output)
    graphml_path = builder.save_graphml(Path(args.output).with_suffix(".graphml"))
    print(f"\n  n_nodes = {builder.n_nodes()}")
    print(f"  n_edges = {builder.n_edges()}")
    print(f"  top 5 articles: {builder.top_articles(5)}")
    print(f"  JSON  : {json_path}")
    if graphml_path:
        print(f"  GraphML: {graphml_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
