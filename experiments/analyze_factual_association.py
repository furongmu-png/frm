# experiments/analyze_factual_association.py
"""Factual-association analysis via hidden-state cosine similarity (Phase I, Task 2.4).

Hypothesis: if the model has internalised co-occurrence statistics, the
belief_state vectors for text blocks containing RELATED entities
(e.g. ``"Paris"`` and ``"France"``) should be MORE cosine-similar than
blocks containing UNRELATED entities (e.g. ``"Paris"`` and ``"Tokyo"``).

Method:
  1. Load all snapshots from ``experiments/output/text_replay_run/``.
  2. For each (entity_a, entity_b) pair in ``FACT_PAIRS``:
       - Find snapshots whose text_block contains entity_a.
       - Find snapshots whose text_block contains entity_b.
       - Compute the mean cosine similarity BETWEEN the two groups
         (entity_a vs entity_b blocks).
       - Compute the WITHIN-group cosine similarity for each group
         (sanity baseline).
       - Also compute the similarity between entity_a blocks and
         UNRELATED entity blocks (e.g. Paris vs Tokyo) — this is
         the negative control.
  3. Compare: cos(related_pair) > cos(unrelated_pair)?
  4. Save: JSON (per-pair mean cos sims), CSV (per-snapshot label +
     cos sim to each entity centroid), PNG (bar chart of pair sims).

Fallback: if too few snapshots contain any of the target entities
(quite likely with a short sample text), the script reports
"insufficient co-occurrence data" without crashing.

Run with:  python experiments/analyze_factual_association.py
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


DEFAULT_INPUT_DIR = Path(__file__).resolve().parent / "output" / "text_replay_run"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR


# ------------------------------------------------------------------ #
# Entity pairs to test.
# Each pair is (entity_a, entity_b, related_bool). related=True means
# we EXPECT high cosine similarity (the entities co-occur in real text).
# related=False is the negative control (entities are unrelated).
# ------------------------------------------------------------------ #
FACT_PAIRS = [
    # Related pairs (positive controls).
    ("Paris", "France", True),
    ("Tokyo", "Japan", True),
    ("London", "England", True),
    ("energy", "free", True),       # "free energy" co-occurs
    ("active", "inference", True),  # "active inference" co-occurs
    ("cognitive", "emergence", True),
    # Unrelated pairs (negative controls).
    ("Paris", "Tokyo", False),
    ("France", "Japan", False),
    ("energy", "history", False),
    ("inference", "particle", False),
]


# ------------------------------------------------------------------ #
# Cosine similarity utilities
# ------------------------------------------------------------------ #
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (returns 0 if either is zero)."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def mean_cosine_between_groups(
    group_a: list[np.ndarray], group_b: list[np.ndarray]
) -> float:
    """Mean cosine similarity across all (a, b) pairs.

    Returns 0.0 if either group is empty.
    """
    if not group_a or not group_b:
        return 0.0
    sims = []
    for a in group_a:
        for b in group_b:
            sims.append(cosine_sim(a, b))
    return float(np.mean(sims))


def mean_cosine_within_group(group: list[np.ndarray]) -> float:
    """Mean cosine similarity WITHIN a group (excluding self-pairs)."""
    if len(group) < 2:
        return 0.0
    sims = []
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            sims.append(cosine_sim(group[i], group[j]))
    return float(np.mean(sims)) if sims else 0.0


# ------------------------------------------------------------------ #
# Main analysis
# ------------------------------------------------------------------ #
def analyze_snapshots(input_dir: Path) -> dict:
    """Load snapshots, group by entity presence, compute cos sims."""
    snapshot_paths = sorted(input_dir.glob("snapshot_step*.pkl"))
    if not snapshot_paths:
        return {"error": f"no snapshots found in {input_dir}"}

    # Load all snapshots once.
    snapshots = []
    for p in snapshot_paths:
        snap = load_snapshot(p)
        snapshots.append({
            "step": int(snap["meta"]["step"]),
            "pos": int(snap["meta"]["pos"]),
            "text_block": snap["text_block"],
            "belief_state": np.asarray(
                snap["active_inference"]["belief_state"], dtype=float
            ),
        })

    # Collect the set of entities that appear in at least 1 snapshot.
    all_entities = sorted({e for pair in FACT_PAIRS for e in pair[:2]})
    entity_groups: dict[str, list[dict]] = {e: [] for e in all_entities}
    for snap in snapshots:
        text_lower = snap["text_block"].lower()
        for entity in all_entities:
            if entity.lower() in text_lower:
                entity_groups[entity].append(snap)

    # Build per-pair results.
    pair_results = []
    for entity_a, entity_b, related in FACT_PAIRS:
        group_a = entity_groups.get(entity_a, [])
        group_b = entity_groups.get(entity_b, [])
        beliefs_a = [s["belief_state"] for s in group_a]
        beliefs_b = [s["belief_state"] for s in group_b]
        sim_between = mean_cosine_between_groups(beliefs_a, beliefs_b)
        sim_within_a = mean_cosine_within_group(beliefs_a)
        sim_within_b = mean_cosine_within_group(beliefs_b)
        pair_results.append({
            "entity_a": entity_a,
            "entity_b": entity_b,
            "related": bool(related),
            "n_a": len(group_a),
            "n_b": len(group_b),
            "cos_sim_between": sim_between,
            "cos_sim_within_a": sim_within_a,
            "cos_sim_within_b": sim_within_b,
        })

    # Aggregate: related vs unrelated mean similarity.
    related_sims = [r["cos_sim_between"] for r in pair_results
                    if r["related"] and r["n_a"] > 0 and r["n_b"] > 0]
    unrelated_sims = [r["cos_sim_between"] for r in pair_results
                      if not r["related"] and r["n_a"] > 0 and r["n_b"] > 0]
    summary = {
        "n_snapshots": len(snapshots),
        "n_entities": len(all_entities),
        "entity_counts": {e: len(g) for e, g in entity_groups.items()},
        "pair_results": pair_results,
        "mean_cos_related": float(np.mean(related_sims)) if related_sims else None,
        "mean_cos_unrelated": float(np.mean(unrelated_sims)) if unrelated_sims else None,
        "n_related_pairs_with_data": len(related_sims),
        "n_unrelated_pairs_with_data": len(unrelated_sims),
    }
    # The hypothesis test: related > unrelated?
    if related_sims and unrelated_sims:
        summary["related_minus_unrelated"] = (
            float(np.mean(related_sims)) - float(np.mean(unrelated_sims))
        )
        summary["hypothesis_supported"] = bool(
            summary["related_minus_unrelated"] > 0
        )
    else:
        summary["hypothesis_supported"] = None
        summary["note"] = (
            "Insufficient co-occurrence data to test hypothesis. "
            "Try a longer text or different entity pairs."
        )
    return summary


def save_outputs(summary: dict, output_dir: Path) -> dict[str, str]:
    """Save CSV + PNG + JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    # CSV: per-pair results.
    csv_path = output_dir / "factual_association.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "entity_a", "entity_b", "related",
            "n_a", "n_b",
            "cos_sim_between", "cos_sim_within_a", "cos_sim_within_b",
        ])
        for r in summary.get("pair_results", []):
            w.writerow([
                r["entity_a"], r["entity_b"], int(r["related"]),
                r["n_a"], r["n_b"],
                f"{r['cos_sim_between']:.6f}",
                f"{r['cos_sim_within_a']:.6f}",
                f"{r['cos_sim_within_b']:.6f}",
            ])
    paths["csv"] = str(csv_path)

    # JSON summary.
    json_path = output_dir / "factual_association_summary.json"
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    paths["json"] = str(json_path)

    # PNG: bar chart of pair sims.
    if _HAS_MPL and summary.get("pair_results"):
        png_path = output_dir / "factual_association.png"
        fig, ax = plt.subplots(figsize=(12, 6))
        pairs = summary["pair_results"]
        labels = [f"{r['entity_a']}\nvs\n{r['entity_b']}" for r in pairs]
        sims = [r["cos_sim_between"] for r in pairs]
        colors = ["#2ca02c" if r["related"] else "#d62728" for r in pairs]
        # Only plot pairs that have data (n_a > 0 and n_b > 0).
        valid_mask = [r["n_a"] > 0 and r["n_b"] > 0 for r in pairs]
        if any(valid_mask):
            valid_labels = [l for l, v in zip(labels, valid_mask) if v]
            valid_sims = [s for s, v in zip(sims, valid_mask) if v]
            valid_colors = [c for c, v in zip(colors, valid_mask) if v]
            x = np.arange(len(valid_labels))
            ax.bar(x, valid_sims, color=valid_colors, alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(valid_labels, fontsize=8)
            ax.set_ylabel("mean cosine similarity (belief_state)")
            ax.set_title(
                "Factual association: green=related (expected high), "
                "red=unrelated (expected low)"
            )
            # Mean lines.
            if summary.get("mean_cos_related") is not None:
                ax.axhline(summary["mean_cos_related"], color="#2ca02c",
                           linestyle="--", alpha=0.7,
                           label=f"mean related = {summary['mean_cos_related']:.3f}")
            if summary.get("mean_cos_unrelated") is not None:
                ax.axhline(summary["mean_cos_unrelated"], color="#d62728",
                           linestyle="--", alpha=0.7,
                           label=f"mean unrelated = {summary['mean_cos_unrelated']:.3f}")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3, axis="y")
        else:
            ax.text(0.5, 0.5, "no entity pairs with co-occurrence data",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Factual association (insufficient data)")
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
        description="Analyse factual associations via hidden-state cosine similarity.",
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
    print("Factual Association Analysis")
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
    print(f"  n_entities        = {summary['n_entities']}")
    print(f"  entity_counts     = {summary['entity_counts']}")
    print(f"  pairs with data   = {summary['n_related_pairs_with_data']} related, "
          f"{summary['n_unrelated_pairs_with_data']} unrelated")
    if summary.get("mean_cos_related") is not None:
        print(f"  mean cos related  = {summary['mean_cos_related']:.4f}")
    if summary.get("mean_cos_unrelated") is not None:
        print(f"  mean cos unrel    = {summary['mean_cos_unrelated']:.4f}")
    if summary.get("hypothesis_supported") is not None:
        verdict = "SUPPORTED" if summary["hypothesis_supported"] else "NOT supported"
        print(f"  hypothesis        = {verdict} "
              f"(related - unrelated = {summary['related_minus_unrelated']:+.4f})")
    elif summary.get("note"):
        print(f"  hypothesis        = N/A ({summary['note']})")

    print(f"\nOutputs:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    print("=" * 64)


if __name__ == "__main__":
    main()
