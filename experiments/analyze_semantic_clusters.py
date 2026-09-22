# experiments/analyze_semantic_clusters.py
"""Semantic cluster analysis of model hidden states (Phase I, Task 2.3).

Hypothesis: if the model has captured topic-level statistics, the
hidden-state vectors (belief_state) for blocks from the SAME topic
should be MORE similar to each other than to blocks from DIFFERENT
topics — even though the model never received a topic label.

Method:
  1. Load all snapshots from ``experiments/output/text_replay_run/``.
  2. For each snapshot, extract:
       - text_block (raw chars)
       - belief_state (model's internal representation)
  3. Heuristic topic labelling (NO external label tool):
       - Scan the text_block for keyword presence.
       - Assign one of: 'cognitive', 'physics', 'language', 'math',
         'history', 'other'.
       - This is a SIMPLE bag-of-keywords classifier — the goal is to
         evaluate whether the model's belief_state clusters by these
         surface markers, NOT to build a perfect classifier.
  4. Stack all belief_states into a (n_snapshots, dim) matrix.
  5. Reduce to 2D via PCA (or t-SNE if sklearn is available).
  6. Compute silhouette score using the heuristic labels (sklearn).
  7. Plot: 2D scatter coloured by topic + silhouette score annotation.
  8. Save: JSON (silhouette score, topic counts), CSV (per-snapshot
     label + 2D coords), PNG (scatter).

Limitations:
  - Heuristic labels are noisy (a block may contain multiple topics).
  - Silhouette score is meaningful only if each topic has >=3 samples.
  - The model has NOT been trained for topic separation; any
    structure is emergent.

Run with:  python experiments/analyze_semantic_clusters.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_text_replay import load_snapshot  # noqa: E402

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:  # pragma: no cover
    _HAS_MPL = False

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False


DEFAULT_INPUT_DIR = Path(__file__).resolve().parent / "output" / "text_replay_run"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR


# ------------------------------------------------------------------ #
# Heuristic topic labeller (bag of keywords)
# ------------------------------------------------------------------ #
TOPIC_KEYWORDS = {
    "cognitive": [
        "cognitive", "emergence", "consciousness", "perception",
        "self-organising", "self-organizing", "free energy",
        "active inference", "belief", "concept", "mind",
    ],
    "physics": [
        "physics", "particle", "energy", "force", "velocity",
        "momentum", "sandbox", "collision", "trajectory",
    ],
    "language": [
        "language", "text", "character", "word", "sentence",
        "linguistic", "syntax", "grammatical", "phoneme",
    ],
    "math": [
        "math", "matrix", "vector", "tensor", "equation",
        "fractal", "topology", "category", "functor", "algebra",
    ],
    "history": [
        "history", "century", "ancient", "war", "revolution",
        "empire", "kingdom", "dynasty",
    ],
}


def label_block(text: str) -> str:
    """Assign a topic label by bag-of-keywords majority vote.

    Returns 'other' if no keyword matches. Lowercases the text first.
    """
    text_lower = text.lower()
    scores = {topic: 0 for topic in TOPIC_KEYWORDS}
    for topic, keywords in TOPIC_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[topic] += 1
    best_topic = max(scores, key=scores.get)
    if scores[best_topic] == 0:
        return "other"
    return best_topic


# ------------------------------------------------------------------ #
# Main analysis
# ------------------------------------------------------------------ #
def analyze_snapshots(input_dir: Path) -> dict:
    """Load snapshots, extract belief_states + labels, return summary."""
    snapshot_paths = sorted(input_dir.glob("snapshot_step*.pkl"))
    if not snapshot_paths:
        return {"error": f"no snapshots found in {input_dir}"}

    records = []
    belief_states = []
    labels = []
    for p in snapshot_paths:
        snap = load_snapshot(p)
        block = snap["text_block"]
        belief = np.asarray(snap["active_inference"]["belief_state"], dtype=float)
        label = label_block(block)
        records.append({
            "step": int(snap["meta"]["step"]),
            "pos": int(snap["meta"]["pos"]),
            "label": label,
            "text_preview": block[:30],
            "belief_norm": float(np.linalg.norm(belief)),
            "free_energy": float(snap["meta"]["free_energy"]),
        })
        belief_states.append(belief)
        labels.append(label)

    belief_matrix = np.stack(belief_states) if belief_states else np.zeros((0, 0))
    labels_arr = np.array(labels)

    # Topic counts.
    unique, counts = np.unique(labels_arr, return_counts=True)
    topic_counts = dict(zip(unique.tolist(), counts.tolist()))

    summary = {
        "n_snapshots": len(records),
        "n_topics": len(topic_counts),
        "topic_counts": topic_counts,
        "belief_matrix_shape": list(belief_matrix.shape),
    }

    # Need at least 2 topics with >=2 samples each for silhouette.
    valid_topics = [t for t, c in topic_counts.items() if c >= 2]
    if len(valid_topics) < 2:
        summary["silhouette_score"] = None
        summary["silhouette_note"] = (
            "Need >=2 topics with >=2 samples each; got " + str(topic_counts)
        )
    elif _HAS_SKLEARN and belief_matrix.shape[0] >= 4:
        # Dimensionality reduction: PCA to 2D (or 10D then t-SNE to 2D).
        n_components = min(2, belief_matrix.shape[1], belief_matrix.shape[0] - 1)
        if n_components < 2:
            summary["silhouette_score"] = None
            summary["silhouette_note"] = "belief_dim too small for PCA"
        else:
            try:
                pca = PCA(n_components=n_components, random_state=42)
                coords_2d = pca.fit_transform(belief_matrix)
                summary["pca_explained_variance"] = pca.explained_variance_ratio_.tolist()
            except Exception as exc:
                summary["pca_error"] = str(exc)
                # Fallback: just take the first 2 columns.
                coords_2d = belief_matrix[:, :2]

            # Silhouette score (on the full belief_state, not the 2D coords).
            # Only compute on samples whose topic is in valid_topics.
            mask = np.array([l in valid_topics for l in labels])
            if mask.sum() >= 4:
                try:
                    score = float(silhouette_score(
                        belief_matrix[mask], labels_arr[mask]
                    ))
                    summary["silhouette_score"] = score
                    summary["silhouette_note"] = (
                        f"computed on {int(mask.sum())} samples, {len(valid_topics)} topics"
                    )
                except Exception as exc:
                    summary["silhouette_score"] = None
                    summary["silhouette_note"] = f"silhouette failed: {exc}"
            else:
                summary["silhouette_score"] = None
                summary["silhouette_note"] = "not enough samples in valid_topics"

            # Also try t-SNE for the plot (more cluster-friendly).
            if _HAS_SKLEARN and belief_matrix.shape[0] >= 5:
                try:
                    # t-SNE on the full belief_state, then to 2D.
                    tsne = TSNE(
                        n_components=2, random_state=42, perplexity=min(
                            30, max(2, belief_matrix.shape[0] // 3)
                        ),
                        init="pca", learning_rate="auto",
                    )
                    coords_tsne = tsne.fit_transform(belief_matrix)
                    summary["has_tsne"] = True
                except Exception as exc:
                    summary["has_tsne"] = False
                    summary["tsne_error"] = str(exc)
                    coords_tsne = coords_2d  # fallback
            else:
                coords_tsne = coords_2d
                summary["has_tsne"] = False
                summary["tsne_note"] = "sklearn not available or too few samples"

            # Attach coords to records for CSV.
            for i, r in enumerate(records):
                r["pca_x"] = float(coords_2d[i, 0])
                r["pca_y"] = float(coords_2d[i, 1])
                r["tsne_x"] = float(coords_tsne[i, 0])
                r["tsne_y"] = float(coords_tsne[i, 1])
            summary["records"] = records
            summary["_coords_2d"] = coords_2d
            summary["_coords_tsne"] = coords_tsne
            summary["_labels"] = labels_arr
    else:
        summary["silhouette_score"] = None
        summary["silhouette_note"] = (
            "sklearn not available or too few samples for silhouette"
        )
        summary["records"] = records

    return summary


def save_outputs(summary: dict, output_dir: Path) -> dict[str, str]:
    """Save CSV + PNG + JSON. Returns dict of saved paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    # CSV: per-snapshot records.
    csv_path = output_dir / "semantic_clusters.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "step", "pos", "label", "text_preview",
            "belief_norm", "free_energy",
            "pca_x", "pca_y", "tsne_x", "tsne_y",
        ])
        for r in summary.get("records", []):
            w.writerow([
                r["step"], r["pos"], r["label"], r["text_preview"],
                f"{r['belief_norm']:.6f}", f"{r['free_energy']:.6f}",
                f"{r.get('pca_x', 0):.6f}", f"{r.get('pca_y', 0):.6f}",
                f"{r.get('tsne_x', 0):.6f}", f"{r.get('tsne_y', 0):.6f}",
            ])
    paths["csv"] = str(csv_path)

    # JSON summary.
    json_path = output_dir / "semantic_clusters_summary.json"
    summary_no_records = {k: v for k, v in summary.items()
                         if not k.startswith("_") and k != "records"}
    with json_path.open("w") as f:
        json.dump(summary_no_records, f, indent=2, default=str)
    paths["json"] = str(json_path)

    # PNG: PCA scatter coloured by topic.
    if _HAS_MPL and "_coords_2d" in summary and summary.get("records"):
        png_path = output_dir / "semantic_clusters.png"
        coords = summary["_coords_2d"]
        labels = summary["_labels"]
        records = summary["records"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        # Distinct colours per topic.
        unique_topics = sorted(set(labels.tolist()))
        cmap = plt.colormaps.get_cmap("tab10")
        for i, topic in enumerate(unique_topics):
            mask = labels == topic
            axes[0].scatter(coords[mask, 0], coords[mask, 1],
                            c=[cmap(i)], label=topic, s=40, alpha=0.7,
                            edgecolors="black", linewidths=0.5)
        axes[0].set_xlabel("PCA 1")
        axes[0].set_ylabel("PCA 2")
        score = summary.get("silhouette_score")
        score_str = f"{score:.3f}" if score is not None else "N/A"
        axes[0].set_title(
            f"PCA of belief_state coloured by heuristic topic "
            f"(silhouette = {score_str})"
        )
        axes[0].legend(loc="best", fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # Right: t-SNE scatter.
        if "_coords_tsne" in summary:
            coords_tsne = summary["_coords_tsne"]
            for i, topic in enumerate(unique_topics):
                mask = labels == topic
                axes[1].scatter(coords_tsne[mask, 0], coords_tsne[mask, 1],
                                c=[cmap(i)], label=topic, s=40, alpha=0.7,
                                edgecolors="black", linewidths=0.5)
            axes[1].set_xlabel("t-SNE 1")
            axes[1].set_ylabel("t-SNE 2")
            axes[1].set_title("t-SNE of belief_state coloured by heuristic topic")
            axes[1].legend(loc="best", fontsize=8)
            axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        paths["png"] = str(png_path)

    return paths


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyse semantic clustering in model belief_states.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input_dir", type=str, default=str(DEFAULT_INPUT_DIR))
    p.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    print("=" * 64)
    print("Semantic Cluster Analysis")
    print("=" * 64)
    print(f"  input_dir  = {input_dir}")
    print(f"  output_dir = {output_dir}")

    summary = analyze_snapshots(input_dir)
    if "error" in summary:
        print(f"\nERROR: {summary['error']}")
        return

    paths = save_outputs(summary, output_dir)
    print("\nResults:")
    print(f"  n_snapshots       = {summary['n_snapshots']}")
    print(f"  n_topics          = {summary['n_topics']}")
    print(f"  topic_counts      = {summary['topic_counts']}")
    print(f"  belief_matrix     = {summary['belief_matrix_shape']}")
    if summary.get("silhouette_score") is not None:
        print(f"  silhouette_score  = {summary['silhouette_score']:.4f}")
    else:
        print(f"  silhouette_score  = N/A ({summary.get('silhouette_note', '')})")
    if "pca_explained_variance" in summary:
        ev = summary["pca_explained_variance"]
        print(f"  PCA explained var = {[f'{v:.3f}' for v in ev]}")

    print(f"\nOutputs:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    print("=" * 64)


if __name__ == "__main__":
    main()
