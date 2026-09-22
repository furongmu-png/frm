# experiments/concept_probes.py
"""Concept probes for semantic-emergence evaluation (Phase J, Task 3.3).

A "concept probe" is a PREDEFINED set of words we HOPE the model
eventually forms a stable internal representation for (e.g. ``gravity``,
``evolution``, ``war``, ``mathematics``). Each probe has:

  - A name (the concept)
  - A list of ``trigger_words`` that should appear together when the
    concept is being discussed (e.g. gravity → force, Newton, apple,
    fall).
  - A list of ``probe_articles`` (titles) we'll use to fetch belief
    states from the model. These are populated at run time by scanning
    the corpus for blocks that contain a trigger word.

The probes are NEVER used as training signal. They're a periodic
EVALUATION: at every ``eval_interval`` steps, we compute the belief
state for each probe's articles and measure:

  1. ``intra_concept_similarity``: mean cosine sim of belief_states
     WITHIN a concept (do different descriptions of "gravity" produce
     similar belief states?).
  2. ``inter_concept_similarity``: mean cosine sim BETWEEN different
     concepts (are "gravity" and "evolution" distinguishable?).
  3. ``silhouette_score``: standard clustering silhouette score
     (sklearn, optional) using the probe labels.

We expect the trajectory to look like:
  - early: intra_sim ≈ inter_sim ≈ baseline (no structure)
  - mid:   intra_sim rising, inter_sim flat (concepts forming)
  - late:  intra_sim >> inter_sim (clean separation)

All values are written to a CSV so we can plot the silhouette-over-time
curve in the final report.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from encyclopedia_reader import EncyclopediaReader  # noqa: E402
from text_encoder import TextEncoder  # noqa: E402

try:
    from sklearn.metrics import silhouette_score as _sk_silhouette
    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False


# ------------------------------------------------------------------ #
# Default probe set (NO external label tool)
# ------------------------------------------------------------------ #
# Each probe = one concept. ``trigger_words`` are the surface markers
# we use to find article blocks that discuss this concept. The
# ``articles`` list is populated at run time by scanning the corpus.
DEFAULT_PROBES = [
    {
        "name": "physics",
        "trigger_words": ["gravity", "force", "energy", "momentum", "newton", "particle"],
    },
    {
        "name": "biology",
        "trigger_words": ["evolution", "cell", "dna", "gene", "species", "organism"],
    },
    {
        "name": "mathematics",
        "trigger_words": ["algebra", "matrix", "vector", "function", "limit", "theorem"],
    },
    {
        "name": "history",
        "trigger_words": ["war", "revolution", "empire", "king", "dynasty", "ancient"],
    },
    {
        "name": "geography",
        "trigger_words": ["river", "mountain", "ocean", "continent", "climate", "desert"],
    },
    {
        "name": "computer_science",
        "trigger_words": ["algorithm", "data", "memory", "process", "code", "program"],
    },
]


# ------------------------------------------------------------------ #
# ProbeSet
# ------------------------------------------------------------------ #
@dataclass
class ProbeSet:
    """A set of concept probes with associated articles."""
    probes: list[dict] = field(default_factory=lambda: [dict(p) for p in DEFAULT_PROBES])

    def populate_articles(
        self,
        reader: EncyclopediaReader,
        min_articles_per_probe: int = 3,
        max_articles_per_probe: int = 10,
    ) -> dict[str, int]:
        """Scan the corpus and assign articles to each probe.

        Returns a dict {probe_name: n_articles_assigned}.
        """
        all_titles = reader.get_article_list()
        counts: dict[str, int] = {}
        for probe in self.probes:
            assigned = []
            triggers = [w.lower() for w in probe["trigger_words"]]
            for title in all_titles:
                # Look up the article body and check for trigger words.
                if not reader.seek_to_article(title):
                    continue
                body = reader._get_current_body()
                if body is None:
                    continue
                body_lower = body.lower()
                # Match if any trigger word appears in the body.
                if any(t in body_lower for t in triggers):
                    assigned.append(title)
                    if len(assigned) >= max_articles_per_probe:
                        break
            # If we couldn't find enough articles with triggers, fall
            # back to articles whose TITLE contains the probe name.
            if len(assigned) < min_articles_per_probe:
                for title in all_titles:
                    if probe["name"].lower() in title.lower() and title not in assigned:
                        assigned.append(title)
                        if len(assigned) >= min_articles_per_probe:
                            break
            probe["articles"] = assigned
            counts[probe["name"]] = len(assigned)
        return counts

    def get_articles_for_probe(self, probe_name: str) -> list[str]:
        for p in self.probes:
            if p["name"] == probe_name:
                return list(p.get("articles", []))
        return []

    def to_dict(self) -> dict:
        return {"probes": [dict(p) for p in self.probes]}

    def save_json(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return str(path)


# ------------------------------------------------------------------ #
# Probe evaluation
# ------------------------------------------------------------------ #
def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _belief_for_article(
    title: str,
    reader: EncyclopediaReader,
    encoder: TextEncoder,
    model,
) -> np.ndarray | None:
    """Get a one-shot belief_state for ``title`` without mutating the model."""
    if not reader.seek_to_article(title):
        return None
    block = reader.read_block()
    if not block:
        return None
    obs = encoder.encode(block)
    gen = model.active_inference.generative_model
    belief, _ = gen.infer_state(obs)
    return belief


def evaluate_probes(
    probe_set: ProbeSet,
    reader: EncyclopediaReader,
    encoder: TextEncoder,
    model,
) -> dict:
    """Compute probe similarity metrics. Returns a metrics dict.

    Computes:
      - per-probe intra-concept similarity (mean cosine within probe)
      - mean inter-concept similarity (mean cosine across probes)
      - silhouette score (if sklearn available, and >=2 probes with >=2 articles)
    """
    # Collect belief_states for every probe article.
    beliefs_per_probe: dict[str, list[np.ndarray]] = {}
    for probe in probe_set.probes:
        beliefs = []
        for title in probe.get("articles", []):
            b = _belief_for_article(title, reader, encoder, model)
            if b is not None:
                beliefs.append(b)
        beliefs_per_probe[probe["name"]] = beliefs

    # Intra-concept similarity: mean pairwise cosine within each probe.
    intra_sims = {}
    for name, beliefs in beliefs_per_probe.items():
        if len(beliefs) < 2:
            intra_sims[name] = None
            continue
        sims = []
        for i in range(len(beliefs)):
            for j in range(i + 1, len(beliefs)):
                sims.append(_cosine_sim(beliefs[i], beliefs[j]))
        intra_sims[name] = float(np.mean(sims))

    # Inter-concept similarity: mean cosine across pairs of probes.
    inter_sims = []
    probe_names = list(beliefs_per_probe.keys())
    for i in range(len(probe_names)):
        for j in range(i + 1, len(probe_names)):
            beliefs_a = beliefs_per_probe[probe_names[i]]
            beliefs_b = beliefs_per_probe[probe_names[j]]
            if not beliefs_a or not beliefs_b:
                continue
            # Mean cosine across all (a, b) pairs.
            pair_sims = [
                _cosine_sim(a, b) for a in beliefs_a for b in beliefs_b
            ]
            inter_sims.append(float(np.mean(pair_sims)) if pair_sims else 0.0)
    inter_sim = float(np.mean(inter_sims)) if inter_sims else None

    # Silhouette score.
    silhouette = None
    silhouette_note = ""
    valid_probes = [n for n, b in beliefs_per_probe.items() if len(b) >= 2]
    if len(valid_probes) < 2:
        silhouette_note = f"need >=2 probes with >=2 articles; got {valid_probes}"
    elif not _HAS_SKLEARN:
        silhouette_note = "sklearn not available"
    else:
        # Stack all belief_states and build label array.
        all_beliefs = []
        all_labels = []
        for name in valid_probes:
            for b in beliefs_per_probe[name]:
                all_beliefs.append(b)
                all_labels.append(name)
        if len(all_beliefs) >= 4 and len(set(all_labels)) >= 2:
            try:
                # Pad to same length (in case different beliefs have different dims).
                max_dim = max(b.shape[0] for b in all_beliefs)
                mat = np.zeros((len(all_beliefs), max_dim))
                for i, b in enumerate(all_beliefs):
                    mat[i, :b.shape[0]] = b
                silhouette = float(_sk_silhouette(mat, all_labels))
                silhouette_note = f"computed on {len(all_beliefs)} samples, {len(set(all_labels))} probes"
            except Exception as exc:
                silhouette_note = f"silhouette failed: {exc}"

    return {
        "n_probes": len(probe_set.probes),
        "n_probes_with_articles": sum(1 for b in beliefs_per_probe.values() if b),
        "intra_concept_similarity": intra_sims,
        "mean_intra_similarity": (
            float(np.mean([v for v in intra_sims.values() if v is not None]))
            if any(v is not None for v in intra_sims.values()) else None
        ),
        "inter_concept_similarity": inter_sim,
        "silhouette_score": silhouette,
        "silhouette_note": silhouette_note,
        "articles_per_probe": {
            name: len(b) for name, b in beliefs_per_probe.items()
        },
    }


# ------------------------------------------------------------------ #
# Probe-history tracker (records metrics over time)
# ------------------------------------------------------------------ #
class ProbeHistory:
    """Records probe metrics at each eval point. Saves to CSV."""

    def __init__(self):
        self.history: list[dict] = []

    def record(self, step: int, metrics: dict) -> None:
        metrics["step"] = step
        self.history.append(metrics)

    def save_csv(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.history:
            return ""
        # Flatten intra_concept_similarity dict into columns.
        probe_names = sorted(self.history[0].get("intra_concept_similarity", {}).keys())
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            header = ["step", "mean_intra_similarity", "inter_concept_similarity",
                     "silhouette_score", "n_probes_with_articles"]
            header += [f"intra_{n}" for n in probe_names]
            header += [f"n_articles_{n}" for n in probe_names]
            w.writerow(header)
            for m in self.history:
                row = [
                    m.get("step", ""),
                    _fmt(m.get("mean_intra_similarity")),
                    _fmt(m.get("inter_concept_similarity")),
                    _fmt(m.get("silhouette_score")),
                    m.get("n_probes_with_articles", 0),
                ]
                intra = m.get("intra_concept_similarity", {})
                arts = m.get("articles_per_probe", {})
                row += [_fmt(intra.get(n)) for n in probe_names]
                row += [arts.get(n, 0) for n in probe_names]
                w.writerow(row)
        return str(path)


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


# ------------------------------------------------------------------ #
# Plot silhouette-over-time
# ------------------------------------------------------------------ #
def plot_probe_history(
    history_csv: str | Path,
    output_png: str | Path,
) -> str | None:
    """Read the probe history CSV and plot the silhouette curve."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    history_csv = Path(history_csv)
    if not history_csv.exists():
        return None
    # Read CSV manually (no pandas dep).
    rows = []
    with history_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return None
    steps = [int(r["step"]) for r in rows]
    intra = [float(r["mean_intra_similarity"]) if r["mean_intra_similarity"] else None
             for r in rows]
    inter = [float(r["inter_concept_similarity"]) if r["inter_concept_similarity"] else None
             for r in rows]
    silh = [float(r["silhouette_score"]) if r["silhouette_score"] else None
            for r in rows]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, intra, "o-", color="#2ca02c", label="intra-concept sim", linewidth=1.5)
    ax.plot(steps, inter, "o-", color="#d62728", label="inter-concept sim", linewidth=1.5)
    ax.plot(steps, silh, "o-", color="#1f77b4", label="silhouette score", linewidth=1.5)
    ax.set_xlabel("step")
    ax.set_ylabel("similarity / silhouette")
    ax.set_title("Concept probe evolution (intra vs inter concept similarity)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=120)
    plt.close(fig)
    return str(output_png)


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate concept probes against a model + corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--corpus_dir", type=str, default="",
                   help="Corpus directory (for article lookup).")
    p.add_argument("--checkpoint", type=str, default="",
                   help="Model checkpoint dir (with model.pkl). Empty = use fresh model.")
    p.add_argument("--output_dir", type=str,
                   default=str(Path(__file__).resolve().parent
                              / "output" / "encyclopedia_run" / "probes"))
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--block_size", type=int, default=256)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 64)
    print("Concept Probes Evaluation")
    print("=" * 64)
    import pickle
    # Load or create model.
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        with (ckpt_path / "model.pkl").open("rb") as f:
            model = pickle.load(f)
        print(f"  loaded model from {ckpt_path}")
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from run_text_curious import make_text_curious_model
        model = make_text_curious_model(
            dim=args.dim, seed=args.seed,
            beta_start=0.0, beta_min=0.0,  # disable curiosity for eval
            decay_steps=1, decay_type="linear",
            num_candidates=6,
        )
        print(f"  using fresh model (no checkpoint given)")
    # Locate corpus.
    if args.corpus_dir:
        corpus_dir = Path(args.corpus_dir)
    else:
        corpus_dir = Path(args.output_dir).parent / "synthetic_corpus"
    if not corpus_dir.exists() or not any(corpus_dir.glob("wiki_*")):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from encyclopedia_reader import make_synthetic_corpus
        make_synthetic_corpus(corpus_dir, n_files=4, articles_per_file=25, seed=args.seed)
    paths = sorted(corpus_dir.glob("wiki_*"))
    reader = EncyclopediaReader(paths, block_size=args.block_size,
                               index_dir=corpus_dir / ".index", seed=args.seed)
    print(f"  corpus: {reader.n_articles()} articles")
    encoder = TextEncoder(block_size=args.block_size, output_dim=args.dim, seed=args.seed)
    # Build probes + populate articles.
    probe_set = ProbeSet()
    counts = probe_set.populate_articles(reader, min_articles_per_probe=2, max_articles_per_probe=10)
    print(f"  probe articles: {counts}")
    # Evaluate.
    metrics = evaluate_probes(probe_set, reader, encoder, model)
    print("\nResults:")
    print(f"  n_probes_with_articles = {metrics['n_probes_with_articles']}")
    print(f"  mean intra-concept sim = {metrics.get('mean_intra_similarity')}")
    print(f"  inter-concept sim       = {metrics.get('inter_concept_similarity')}")
    print(f"  silhouette score        = {metrics.get('silhouette_score')}")
    print(f"  silhouette note         = {metrics.get('silhouette_note')}")
    # Save.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probe_set.save_json(output_dir / "probes.json")
    with (output_dir / "probe_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2, default=str)
    # Save a single-row history CSV (for plot_probe_history).
    history = ProbeHistory()
    history.record(0, metrics)
    csv_path = history.save_csv(output_dir / "probe_history.csv")
    png_path = plot_probe_history(csv_path, output_dir / "probe_history.png")
    if png_path:
        print(f"\nOutputs:")
        print(f"  {output_dir / 'probes.json'}")
        print(f"  {output_dir / 'probe_metrics.json'}")
        print(f"  {csv_path}")
        print(f"  {png_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
