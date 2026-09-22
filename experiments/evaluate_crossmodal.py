# experiments/evaluate_crossmodal.py
"""Evaluate cross-modal alignment via Top-1 retrieval (Phase J, Task 4).

Given a trained ``MultimodalBridge`` and a held-out aligned (frame, text)
dataset, we test whether the cross-modal predictors have learned the
alignment by asking:

  **Physics → Text retrieval**: For each test sample i, encode the frame
  to h_phys_i, predict h_text_i^* = W_pt @ h_phys_i, and find the j in
  the corpus whose ACTUAL h_text_j is most similar (cosine) to h_text_i^*.
  Retrieval is correct iff j == i.

  **Text → Physics retrieval**: Symmetric — predict h_phys from h_text,
  find the closest actual h_phys in the corpus.

We compare against two random baselines:
  - **Uniform random**: 1/N expected accuracy.
  - **Modality-mean baseline**: always retrieve the corpus item whose
    hidden state is closest to the corpus MEAN hidden state (a constant
    predictor — if the bridge does worse than this, it has learned
    nothing useful).

Outputs (under ``experiments/output/crossmodal_run/evaluation/``):
  - retrieval_results.csv      per-sample retrieval ranks
  - retrieval_summary.json    aggregated Top-1/Top-5 accuracy
  - retrieval_summary.md       human-readable Markdown summary

Run with:
    python experiments/evaluate_crossmodal.py \\
        --dataset experiments/output/aligned_data/aligned_dataset.npz \\
        --bridge experiments/output/crossmodal_run/bridge.npz \\
        --output_dir experiments/output/crossmodal_run/evaluation
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from aligned_data_generator import AlignedDataset  # noqa: E402
from multimodal_bridge import DEFAULT_DIM, MultimodalBridge  # noqa: E402


# ------------------------------------------------------------------ #
# Defaults
# ------------------------------------------------------------------ #
DEFAULT_DIM = DEFAULT_DIM
DEFAULT_SEED = 42
DEFAULT_N_TEST = 200            # number of held-out samples to evaluate
DEFAULT_N_CORPUS = 500          # corpus size for retrieval
DEFAULT_BRIDGE_PATH = Path(__file__).resolve().parent / "output" / "crossmodal_run" / "bridge.npz"
DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "output" / "aligned_data" / "aligned_dataset.npz"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "crossmodal_run" / "evaluation"


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosine similarity between every row of A and every row of B.

    Returns (n_A, n_B) matrix.
    """
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A_norm @ B_norm.T


def topk_accuracy(sim_matrix: np.ndarray, k: int = 1) -> float:
    """Top-k accuracy: diagonal is the correct match."""
    n = sim_matrix.shape[0]
    if n == 0:
        return 0.0
    # For each query i, get the indices of the k most similar corpus items.
    topk = np.argsort(-sim_matrix, axis=1)[:, :k]
    correct = sum(1 for i in range(n) if i in topk[i])
    return correct / n


def mean_rank(sim_matrix: np.ndarray) -> float:
    """Mean rank of the correct match (1 = best)."""
    n = sim_matrix.shape[0]
    if n == 0:
        return 0.0
    # For each query i, find the rank of corpus item i.
    ranks = []
    for i in range(n):
        order = np.argsort(-sim_matrix[i])
        rank = int(np.where(order == i)[0][0]) + 1  # 1-indexed
        ranks.append(rank)
    return float(np.mean(ranks))


# ------------------------------------------------------------------ #
# Evaluation
# ------------------------------------------------------------------ #
def evaluate(
    bridge: MultimodalBridge,
    dataset: AlignedDataset,
    n_test: int = DEFAULT_N_TEST,
    n_corpus: int = DEFAULT_N_CORPUS,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Run Top-1 retrieval evaluation in both directions.

    Two retrieval criteria:
      - **Exact match** (strict): retrieved corpus index == test index.
        With highly-repetitive text (only ~12 unique descriptions), many
        corpus items share the same h_text, so exact Top-1 is harsh.
      - **Event-category match** (semantic): retrieved corpus item has
        the same EVENT LABEL as the query. This tests whether the
        cross-modal predictor has learned event-level semantics.
    """
    rng = np.random.default_rng(seed)
    n_total = len(dataset)
    n_eval = min(n_test + n_corpus, n_total)
    if n_test + n_corpus > n_total:
        n_corpus = max(1, n_total - n_test)
        if n_corpus < 1:
            n_test = n_total // 2
            n_corpus = n_total - n_test

    # Random split: test set + corpus (disjoint).
    indices = rng.permutation(n_total)
    test_idx = indices[:n_test]
    corpus_idx = indices[n_test:n_test + n_corpus]

    # Encode corpus.
    print(f"Encoding corpus ({len(corpus_idx)} samples)...")
    H_phys_corpus = np.stack([
        bridge.encode_physics(dataset.frames[i]) for i in corpus_idx
    ])
    H_text_corpus = np.stack([
        bridge.encode_text(dataset.texts[i]) for i in corpus_idx
    ])

    # Encode test set.
    print(f"Encoding test set ({len(test_idx)} samples)...")
    H_phys_test = np.stack([
        bridge.encode_physics(dataset.frames[i]) for i in test_idx
    ])
    H_text_test = np.stack([
        bridge.encode_text(dataset.texts[i]) for i in test_idx
    ])

    # Event labels for category-level retrieval.
    test_events = [dataset.events[i] for i in test_idx]
    corpus_events = [dataset.events[i] for i in corpus_idx]
    # Unique event labels in the test set.
    unique_events = sorted(set(test_events) | set(corpus_events))
    event_to_idx = {e: i for i, e in enumerate(unique_events)}
    test_event_labels = np.array([event_to_idx[e] for e in test_events])
    corpus_event_labels = np.array([event_to_idx[e] for e in corpus_events])

    # --- Physics -> Text retrieval ---------------------------------- #
    H_text_pred_from_phys = np.stack([
        bridge.predict_text_from_physics(h) for h in H_phys_test
    ])
    sim_pt = cosine_similarity_matrix(H_text_pred_from_phys, H_text_corpus)
    top1_pt_exact = topk_accuracy(sim_pt, k=1)
    top5_pt_exact = topk_accuracy(sim_pt, k=5)
    rank_pt = mean_rank(sim_pt)

    # Event-category Top-1: does the retrieved item share the event label?
    top1_indices_pt = np.argmax(sim_pt, axis=1)
    top1_pt_category = float(np.mean(
        corpus_event_labels[top1_indices_pt] == test_event_labels
    ))
    # Top-5 category: any of the top-5 retrieved share the label?
    top5_indices_pt = np.argsort(-sim_pt, axis=1)[:, :5]
    top5_pt_category = float(np.mean([
        test_event_labels[i] in corpus_event_labels[top5_indices_pt[i]]
        for i in range(len(test_idx))
    ]))

    # --- Text -> Physics retrieval ---------------------------------- #
    H_phys_pred_from_text = np.stack([
        bridge.predict_physics_from_text(h) for h in H_text_test
    ])
    sim_tp = cosine_similarity_matrix(H_phys_pred_from_text, H_phys_corpus)
    top1_tp_exact = topk_accuracy(sim_tp, k=1)
    top5_tp_exact = topk_accuracy(sim_tp, k=5)
    rank_tp = mean_rank(sim_tp)

    top1_indices_tp = np.argmax(sim_tp, axis=1)
    top1_tp_category = float(np.mean(
        corpus_event_labels[top1_indices_tp] == test_event_labels
    ))
    top5_indices_tp = np.argsort(-sim_tp, axis=1)[:, :5]
    top5_tp_category = float(np.mean([
        test_event_labels[i] in corpus_event_labels[top5_indices_tp[i]]
        for i in range(len(test_idx))
    ]))

    # --- Random baselines ------------------------------------------- #
    n_corpus_actual = len(corpus_idx)
    random_top1_exact = 1.0 / n_corpus_actual
    random_top5_exact = min(1.0, 5.0 / n_corpus_actual)
    random_rank = (n_corpus_actual + 1) / 2.0

    # Random category baseline: P(random corpus item has same label as query).
    # = sum over labels of (n_label_corpus / n_corpus) * (n_label_test / n_test)
    label_counts_corpus = np.bincount(corpus_event_labels, minlength=len(unique_events))
    label_counts_test = np.bincount(test_event_labels, minlength=len(unique_events))
    p_corpus = label_counts_corpus / max(1, n_corpus_actual)
    p_test = label_counts_test / max(1, len(test_idx))
    random_top1_category = float(np.sum(p_corpus * p_test))
    # Top-5 random category (inclusion-exclusion approx): higher than Top-1.
    random_top5_category = float(min(1.0, 5.0 * random_top1_category))

    # --- Modality-mean baseline (exact) ----------------------------- #
    text_corpus_mean = H_text_corpus.mean(axis=0, keepdims=True)
    sim_mean = cosine_similarity_matrix(text_corpus_mean, H_text_corpus)[0]
    closest_to_mean = int(np.argmax(sim_mean))
    mean_baseline_top1 = sum(
        1 for i in range(len(test_idx)) if closest_to_mean == i
    ) / max(1, len(test_idx))

    return {
        "n_test": len(test_idx),
        "n_corpus": len(corpus_idx),
        "n_unique_events": len(unique_events),
        "event_labels": unique_events,
        "physics_to_text": {
            "top1_exact": top1_pt_exact,
            "top5_exact": top5_pt_exact,
            "top1_category": top1_pt_category,
            "top5_category": top5_pt_category,
            "mean_rank": rank_pt,
        },
        "text_to_physics": {
            "top1_exact": top1_tp_exact,
            "top5_exact": top5_tp_exact,
            "top1_category": top1_tp_category,
            "top5_category": top5_tp_category,
            "mean_rank": rank_tp,
        },
        "random_baseline": {
            "top1_exact": random_top1_exact,
            "top5_exact": random_top5_exact,
            "top1_category": random_top1_category,
            "top5_category": random_top5_category,
            "mean_rank": random_rank,
        },
        "mean_baseline_top1": mean_baseline_top1,
        "test_indices": test_idx.tolist(),
        "corpus_indices": corpus_idx.tolist(),
    }


# ------------------------------------------------------------------ #
# Output rendering
# ------------------------------------------------------------------ #
def save_outputs(
    results: dict,
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    # JSON summary.
    json_path = output_dir / "retrieval_summary.json"
    with json_path.open("w") as f:
        json.dump(results, f, indent=2)
    paths["json"] = str(json_path)

    # Markdown summary.
    md_path = output_dir / "retrieval_summary.md"
    with md_path.open("w") as f:
        f.write("# Cross-Modal Retrieval Evaluation\n\n")
        f.write(f"- Test samples: {results['n_test']}\n")
        f.write(f"- Corpus size: {results['n_corpus']}\n")
        f.write(f"- Unique event labels: {results['n_unique_events']} "
                f"({', '.join(results['event_labels'])})\n\n")

        f.write("## Exact Match (Top-1 / Top-5)\n\n")
        f.write("Retrieved corpus index must equal the test index.\n\n")
        f.write("| Direction | Top-1 | Top-5 | Mean Rank |\n")
        f.write("|---|---|---|---|\n")
        f.write(
            f"| Physics → Text | {results['physics_to_text']['top1_exact']:.4f} | "
            f"{results['physics_to_text']['top5_exact']:.4f} | "
            f"{results['physics_to_text']['mean_rank']:.1f} |\n"
        )
        f.write(
            f"| Text → Physics | {results['text_to_physics']['top1_exact']:.4f} | "
            f"{results['text_to_physics']['top5_exact']:.4f} | "
            f"{results['text_to_physics']['mean_rank']:.1f} |\n"
        )
        f.write(
            f"| Random baseline | {results['random_baseline']['top1_exact']:.4f} | "
            f"{results['random_baseline']['top5_exact']:.4f} | "
            f"{results['random_baseline']['mean_rank']:.1f} |\n"
        )
        f.write(
            f"| Modality-mean baseline | {results['mean_baseline_top1']:.4f} | "
            f"— | — |\n\n"
        )

        f.write("## Event-Category Match (Top-1 / Top-5)\n\n")
        f.write("Retrieved corpus item must share the same EVENT LABEL as the "
                "query (collision / movement / wall_hit / action / stationary).\n")
        f.write("This is the semantic test — does the bridge know what KIND of "
                "event it is seeing?\n\n")
        f.write("| Direction | Top-1 | Top-5 |\n")
        f.write("|---|---|---|\n")
        f.write(
            f"| Physics → Text | {results['physics_to_text']['top1_category']:.4f} | "
            f"{results['physics_to_text']['top5_category']:.4f} |\n"
        )
        f.write(
            f"| Text → Physics | {results['text_to_physics']['top1_category']:.4f} | "
            f"{results['text_to_physics']['top5_category']:.4f} |\n"
        )
        f.write(
            f"| Random baseline | {results['random_baseline']['top1_category']:.4f} | "
            f"{results['random_baseline']['top5_category']:.4f} |\n\n"
        )

        # Verdict — based on event-category Top-1 (the meaningful signal).
        pt = results["physics_to_text"]["top1_category"]
        tp = results["text_to_physics"]["top1_category"]
        rand = results["random_baseline"]["top1_category"]
        mean_b = results["mean_baseline_top1"]
        best = max(pt, tp)
        ratio = best / rand if rand > 0 else float("inf")

        if best > 2.0 * rand and best > 0.5:
            verdict = "STRONG ALIGNMENT"
        elif best > 1.5 * rand and best > 0.35:
            verdict = "MODERATE ALIGNMENT"
        elif best > 1.1 * rand and best > 0.25:
            verdict = "WEAK ALIGNMENT"
        else:
            verdict = "NO ALIGNMENT"

        f.write(f"## Verdict: {verdict}\n\n")
        f.write(
            f"- Best event-category Top-1 = {best:.4f} vs random {rand:.4f} "
            f"(ratio = {ratio:.2f}x)\n"
        )
        f.write(
            f"- Best exact Top-1 vs modality-mean baseline = {mean_b:.4f}\n"
        )
        if "STRONG" in verdict:
            f.write(
                "\nThe bridge has learned a strong cross-modal mapping at the "
                "event-category level: retrieval substantially exceeds the "
                "random baseline, indicating genuine concept alignment.\n"
            )
        elif "MODERATE" in verdict:
            f.write(
                "\nThe bridge has learned a meaningful cross-modal mapping at "
                "the event-category level — retrieval clearly exceeds chance.\n"
            )
        elif "WEAK" in verdict:
            f.write(
                "\nThe bridge shows weak cross-modal alignment at the "
                "event-category level — retrieval is slightly above chance. "
                "Longer training or richer text may help.\n"
            )
        else:
            f.write(
                "\nThe bridge has NOT learned cross-modal alignment. "
                "Event-category retrieval is at or below chance.\n"
            )
    paths["md"] = str(md_path)
    return paths


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate cross-modal retrieval accuracy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET_PATH))
    p.add_argument("--bridge", type=str, default=str(DEFAULT_BRIDGE_PATH))
    p.add_argument("--dim", type=int, default=DEFAULT_DIM)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n_test", type=int, default=DEFAULT_N_TEST)
    p.add_argument("--n_corpus", type=int, default=DEFAULT_N_CORPUS)
    p.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 64)
    print("Cross-Modal Retrieval Evaluation")
    print("=" * 64)
    print(f"  dataset   = {args.dataset}")
    print(f"  bridge    = {args.bridge}")
    print(f"  dim       = {args.dim}")
    print(f"  n_test    = {args.n_test}")
    print(f"  n_corpus  = {args.n_corpus}")
    print()

    dataset = AlignedDataset.load(args.dataset)
    print(f"Loaded dataset: {len(dataset)} samples")

    bridge = MultimodalBridge(dim=args.dim, seed=args.seed)
    bridge.load(args.bridge)
    print(f"Loaded bridge: dim={bridge.dim}, updates={bridge._n_updates}")
    print()

    results = evaluate(
        bridge=bridge, dataset=dataset,
        n_test=args.n_test, n_corpus=args.n_corpus,
        seed=args.seed,
    )

    paths = save_outputs(results, Path(args.output_dir))
    print("\nResults (exact match):")
    print(f"  Physics -> Text: Top-1 = {results['physics_to_text']['top1_exact']:.4f}  "
          f"Top-5 = {results['physics_to_text']['top5_exact']:.4f}  "
          f"mean_rank = {results['physics_to_text']['mean_rank']:.1f}")
    print(f"  Text    -> Physics: Top-1 = {results['text_to_physics']['top1_exact']:.4f}  "
          f"Top-5 = {results['text_to_physics']['top5_exact']:.4f}  "
          f"mean_rank = {results['text_to_physics']['mean_rank']:.1f}")
    print(f"  Random baseline:  Top-1 = {results['random_baseline']['top1_exact']:.4f}  "
          f"Top-5 = {results['random_baseline']['top5_exact']:.4f}")
    print()
    print("Results (event-category match):")
    print(f"  Physics -> Text: Top-1 = {results['physics_to_text']['top1_category']:.4f}  "
          f"Top-5 = {results['physics_to_text']['top5_category']:.4f}")
    print(f"  Text    -> Physics: Top-1 = {results['text_to_physics']['top1_category']:.4f}  "
          f"Top-5 = {results['text_to_physics']['top5_category']:.4f}")
    print(f"  Random baseline:  Top-1 = {results['random_baseline']['top1_category']:.4f}  "
          f"Top-5 = {results['random_baseline']['top5_category']:.4f}")
    print(f"  Modality-mean baseline (exact): Top-1 = {results['mean_baseline_top1']:.4f}")
    print()
    print(f"Saved: {paths['json']}")
    print(f"       {paths['md']}")


if __name__ == "__main__":
    main()
