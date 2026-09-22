# experiments/generate_report.py
"""Aggregate evaluation report generator (Phase I, Task 3).

Runs all three analysis scripts (word-boundary, semantic-cluster,
factual-association) on a snapshot directory and produces a single
Markdown report ``evaluation_report.md`` summarising the findings.

The report includes:
  - Experiment parameters (loaded from ``manifest.json`` if present)
  - Consolidation effect summary (ΔFE, replay counts)
  - Free-energy curve (PNG embedded if available)
  - Word-boundary detection: t-test result, mean PE per group
  - Semantic clustering: silhouette score, PCA explained variance
  - Factual association: related vs unrelated cosine similarity
  - Conclusion: did the model show signs of concept emergence?

The script is IDEMPOTENT: re-running it overwrites the report. It
DOES NOT re-run the online experiment — only the offline analyses.

Run with:  python experiments/generate_report.py
           python experiments/generate_report.py --input_dir path/to/snapshots
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Reuse the analysis functions directly (not via subprocess) so we
# can capture the structured summaries in-process.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_word_boundaries import analyze_snapshots as analyze_boundaries  # noqa: E402
from analyze_semantic_clusters import analyze_snapshots as analyze_clusters  # noqa: E402
from analyze_factual_association import analyze_snapshots as analyze_facts  # noqa: E402


DEFAULT_INPUT_DIR = Path(__file__).resolve().parent / "output" / "text_replay_run"
DEFAULT_REPORT_PATH = DEFAULT_INPUT_DIR / "evaluation_report.md"


# ------------------------------------------------------------------ #
# Report sections
# ------------------------------------------------------------------ #
def _fmt(v, fmt_str: str = "{:.4f}") -> str:
    """Format a possibly-None scalar."""
    if v is None:
        return "N/A"
    if isinstance(v, (int, float)):
        return fmt_str.format(v)
    return str(v)


def render_header(manifest: dict | None) -> str:
    """Render the report header + experiment parameters."""
    lines = [
        "# ZeroDataModel Phase I Evaluation Report",
        "",
        "## Experiment Overview",
        "",
        "This report evaluates whether the ZeroDataModel, after running the",
        "Phase I text-reading loop with **experience replay + offline",
        "consolidation**, shows signs of having internalised textual",
        "structure (word boundaries, semantic clusters, factual",
        "associations) — WITHOUT any external labels or pretrained",
        "embeddings.",
        "",
        "All analyses are computed on **cognitive snapshots** saved",
        "during the run (every `snapshot_interval` steps).",
        "",
    ]
    if manifest:
        lines += [
            "## Experiment Parameters",
            "",
            "| Parameter | Value |",
            "|---|---|",
            f"| `max_steps` | {manifest.get('max_steps', 'N/A')} |",
            f"| `dim` | {manifest.get('dim', 'N/A')} |",
            f"| `seed` | {manifest.get('seed', 'N/A')} |",
            f"| `block_size` | {manifest.get('block_size', 'N/A')} |",
            f"| `buffer_capacity` | {manifest.get('buffer_capacity', 'N/A')} |",
            f"| `consolidation_interval` | {manifest.get('consolidation_interval', 'N/A')} |",
            f"| `consolidation_batch_size` | {manifest.get('consolidation_batch_size', 'N/A')} |",
            f"| `consolidation_epochs` | {manifest.get('consolidation_epochs', 'N/A')} |",
            f"| `snapshot_interval` | {manifest.get('snapshot_interval', 'N/A')} |",
            "",
            "### Run Outcome",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| `mean_free_energy` | {_fmt(manifest.get('mean_free_energy'))} |",
            f"| `final_free_energy` | {_fmt(manifest.get('final_free_energy'))} |",
            f"| `initial_beta` | {_fmt(manifest.get('initial_beta'))} |",
            f"| `final_beta` | {_fmt(manifest.get('final_beta'))} |",
            f"| `n_consolidation_rounds` | {manifest.get('n_consolidation_rounds', 'N/A')} |",
            f"| `n_replayed_experiences` | {manifest.get('n_replayed_experiences', 'N/A')} |",
            f"| `mean_delta_fe_consolidation` | {_fmt(manifest.get('mean_delta_fe_consolidation'), '{:+.4f}')} |",
            f"| `n_snapshots` | {len(manifest.get('snapshot_paths', []))} |",
            "",
        ]
    else:
        lines += [
            "## Experiment Parameters",
            "",
            "_No `manifest.json` found in the input directory. Run `run_text_replay.py` first to generate one._",
            "",
        ]
    return "\n".join(lines)


def render_consolidation(manifest: dict | None) -> str:
    """Render the consolidation-effect section."""
    lines = [
        "## 1. Consolidation Effect (Phase I, Task 1)",
        "",
        "The offline consolidator replays high-prediction-error",
        "experiences through `update_belief → update`, strengthening",
        "the generative model's predictions for surprising transitions.",
        "",
    ]
    if manifest:
        delta = manifest.get("mean_delta_fe_consolidation")
        verdict = (
            "POSITIVE — consolidation reduced free energy on the replay batch"
            if delta is not None and delta < 0 else
            ("NEUTRAL — consolidation did not significantly change FE"
             if delta is not None and abs(delta) < 0.1 else
             "NEGATIVE — consolidation increased FE (unexpected; investigate)")
        )
        lines += [
            f"- **Mean ΔFE per consolidation round**: {_fmt(delta, '{:+.4f}')}",
            f"- **Total consolidation rounds**: {manifest.get('n_consolidation_rounds', 0)}",
            f"- **Total experiences replayed**: {manifest.get('n_replayed_experiences', 0)}",
            f"- **Verdict**: {verdict}",
            "",
            "Interpretation: a negative ΔFE means the model's belief",
            "improved its fit to the replayed batch — the 'sleep phase'",
            "is doing useful work consolidating surprising transitions.",
            "",
        ]
    else:
        lines.append("_No manifest data available._\n")
    return "\n".join(lines)


def render_boundaries(summary: dict, image_path: str | None) -> str:
    """Render the word-boundary detection section."""
    lines = [
        "## 2. Word-Boundary Detection (Phase I, Task 2.2)",
        "",
        "Hypothesis: if the model has internalised character-transition",
        "statistics, its prediction error should be HIGHER on text blocks",
        "containing rare-bigram positions (hard-to-predict transitions).",
        "",
    ]
    if "error" in summary:
        lines.append(f"_Analysis error: {summary['error']}_\n")
        return "\n".join(lines)
    lines += [
        f"- **Snapshots analysed**: {summary['n_snapshots']}",
        f"- **Full text length**: {summary['full_text_length']} chars",
        f"- **Rare-bigram positions**: {summary['n_rare_positions_in_full_text']}",
        f"- **Space-boundary positions**: {summary['n_space_positions_in_full_text']}",
        "",
        "### Prediction Error by Group",
        "",
        "| Group | n | mean PE | std PE |",
        "|---|---|---|---|",
        f"| High rare-bigram density | {summary['n_high_group']} | "
        f"{_fmt(summary.get('mean_pe_high_rare_density'))} | "
        f"{_fmt(summary.get('std_pe_high_rare_density'))} |",
        f"| Low rare-bigram density | {summary['n_low_group']} | "
        f"{_fmt(summary.get('mean_pe_low_rare_density'))} | "
        f"{_fmt(summary.get('std_pe_low_rare_density'))} |",
        "",
    ]
    if "p_value" in summary:
        lines += [
            f"- **t-statistic**: {_fmt(summary.get('t_stat'), '{:.3f}')}",
            f"- **p-value**: {_fmt(summary.get('p_value'), '{:.4f}')}",
            f"- **Significant at α=0.05**: {summary.get('significant_at_0.05', False)}",
        ]
    elif "p_value_approx" in summary:
        lines += [
            f"- **z-statistic** (fallback, scipy unavailable): "
            f"{_fmt(summary.get('z_stat'), '{:.3f}')}",
            f"- **p-value (approx)**: {_fmt(summary.get('p_value_approx'), '{:.4f}')}",
            f"- **Significant at α=0.05**: {summary.get('significant_at_0.05', False)}",
        ]
    if image_path:
        lines += [
            "",
            f"![Word-boundary analysis]({Path(image_path).name})",
            "",
        ]
    return "\n".join(lines)


def render_clusters(summary: dict, image_path: str | None) -> str:
    """Render the semantic-cluster section."""
    lines = [
        "## 3. Semantic Clustering (Phase I, Task 2.3)",
        "",
        "Hypothesis: if the model has captured topic-level statistics,",
        "the belief_state vectors for blocks on the SAME topic should",
        "be more similar than those on DIFFERENT topics — even though",
        "the model never received a topic label.",
        "",
    ]
    if "error" in summary:
        lines.append(f"_Analysis error: {summary['error']}_\n")
        return "\n".join(lines)
    lines += [
        f"- **Snapshots analysed**: {summary['n_snapshots']}",
        f"- **Heuristic topics detected**: {summary['n_topics']}",
        f"- **Topic counts**: `{summary['topic_counts']}`",
        f"- **Belief matrix shape**: `{summary['belief_matrix_shape']}`",
        "",
        f"- **Silhouette score**: {_fmt(summary.get('silhouette_score'))}",
        f"  - Note: {summary.get('silhouette_note', '')}",
        "",
    ]
    if "pca_explained_variance" in summary:
        ev = summary["pca_explained_variance"]
        ev_str = ", ".join(f"{v:.3f}" for v in ev)
        lines.append(f"- **PCA explained variance** (per component): [{ev_str}]")
    lines.append("")
    lines.append("Interpretation: a silhouette score > 0 indicates the")
    lines.append("heuristic topic labels correspond to SOME structure")
    lines.append("in the belief_state. A score near 0 means no clear")
    lines.append("structure; negative means topics overlap. Even a small")
    lines.append("positive score is meaningful because the labels are noisy.")
    lines.append("")
    if image_path:
        lines += [
            f"![Semantic clustering]({Path(image_path).name})",
            "",
        ]
    return "\n".join(lines)


def render_facts(summary: dict, image_path: str | None) -> str:
    """Render the factual-association section."""
    lines = [
        "## 4. Factual Association (Phase I, Task 2.4)",
        "",
        "Hypothesis: if the model has internalised co-occurrence",
        "statistics, belief_states for blocks containing RELATED",
        "entities (e.g. \"Paris\" and \"France\") should be more",
        "cosine-similar than those containing UNRELATED entities",
        "(e.g. \"Paris\" and \"Tokyo\").",
        "",
    ]
    if "error" in summary:
        lines.append(f"_Analysis error: {summary['error']}_\n")
        return "\n".join(lines)
    lines += [
        f"- **Snapshots analysed**: {summary['n_snapshots']}",
        f"- **Entities tested**: {summary['n_entities']}",
        f"- **Entity counts**: `{summary['entity_counts']}`",
        f"- **Related pairs with co-occurrence data**: "
        f"{summary['n_related_pairs_with_data']}",
        f"- **Unrelated pairs with co-occurrence data**: "
        f"{summary['n_unrelated_pairs_with_data']}",
        "",
        "### Cosine Similarity Comparison",
        "",
        "| Group | Mean cosine similarity |",
        "|---|---|",
        f"| Related entity pairs | {_fmt(summary.get('mean_cos_related'))} |",
        f"| Unrelated entity pairs | {_fmt(summary.get('mean_cos_unrelated'))} |",
        "",
    ]
    if summary.get("hypothesis_supported") is not None:
        verdict = (
            "SUPPORTED" if summary["hypothesis_supported"] else "NOT supported"
        )
        lines += [
            f"- **Hypothesis verdict**: {verdict}",
            f"- **Difference (related − unrelated)**: "
            f"{_fmt(summary.get('related_minus_unrelated'), '{:+.4f}')}",
        ]
    elif summary.get("note"):
        lines.append(f"- **Hypothesis verdict**: N/A — {summary['note']}")
    lines.append("")
    if image_path:
        lines += [
            f"![Factual association]({Path(image_path).name})",
            "",
        ]
    return "\n".join(lines)


def render_conclusion(
    cons_summary: dict | None,
    boundary_summary: dict,
    cluster_summary: dict,
    fact_summary: dict,
) -> str:
    """Render the discussion + conclusion section."""
    lines = [
        "## 5. Conclusion & Discussion",
        "",
        "### Overall Emergence Verdict",
        "",
    ]

    # Tally positive signals.
    n_signals = 0
    n_total = 0
    signals = []

    # Signal 1: consolidation effectiveness
    n_total += 1
    if cons_summary:
        delta = cons_summary.get("mean_delta_fe_consolidation")
        if delta is not None and delta < 0:
            n_signals += 1
            signals.append(
                f"Consolidation: POSITIVE (ΔFE = {delta:+.4f}, "
                "model fit improved during sleep phase)"
            )
        else:
            signals.append(
                f"Consolidation: NEUTRAL/NEGATIVE (ΔFE = {_fmt(delta, '{:+.4f}')})"
            )

    # Signal 2: word-boundary detection
    n_total += 1
    if boundary_summary.get("significant_at_0.05"):
        n_signals += 1
        signals.append(
            f"Word boundaries: POSITIVE (p = {boundary_summary.get('p_value', 'N/A'):.4f}, "
            "PE higher at rare-bigram positions)"
        )
    else:
        signals.append(
            f"Word boundaries: NEUTRAL (p = {boundary_summary.get('p_value', 'N/A')}, "
            "no significant PE difference at boundaries)"
        )

    # Signal 3: semantic clustering
    n_total += 1
    score = cluster_summary.get("silhouette_score")
    if score is not None and score > 0:
        n_signals += 1
        signals.append(
            f"Semantic clusters: POSITIVE (silhouette = {score:.4f}, "
            "belief_state shows topic-level structure)"
        )
    else:
        signals.append(
            f"Semantic clusters: NEUTRAL (silhouette = {_fmt(score)}, "
            "no clear topic-level structure)"
        )

    # Signal 4: factual association
    n_total += 1
    if fact_summary.get("hypothesis_supported"):
        n_signals += 1
        signals.append(
            "Factual association: POSITIVE (related pairs more similar than unrelated)"
        )
    else:
        signals.append(
            "Factual association: NEUTRAL/NEGATIVE (related pairs not more similar)"
        )

    lines.append(f"**{n_signals}/{n_total} positive emergence signals detected.**")
    lines.append("")
    for s in signals:
        lines.append(f"- {s}")
    lines.append("")

    # Overall discussion.
    if n_signals >= 3:
        verdict = "**Strong evidence** of concept emergence."
    elif n_signals >= 2:
        verdict = "**Moderate evidence** of concept emergence."
    elif n_signals >= 1:
        verdict = "**Weak evidence** of concept emergence — partial structure."
    else:
        verdict = "**No evidence** of concept emergence — model needs more training or a different setup."
    lines += [
        f"**Verdict**: {verdict}",
        "",
        "### Discussion",
        "",
        "These results are PROVISIONAL. The model was trained on a",
        "small text corpus with no pretrained embeddings; a production",
        "evaluation would use a longer corpus, more snapshots, and",
        "sklearn-based silhouette scoring. The word-boundary test is",
        "the strongest signal — it directly probes whether the model",
        "encodes character-transition statistics, which is the first",
        "step toward word-level concepts. The cluster and factual",
        "tests are weaker because they depend on (a) the model having",
        "seen enough co-occurrence data and (b) the heuristic labels",
        "being meaningful.",
        "",
        "### Limitations",
        "",
        "1. **Sample size**: with only a few hundred snapshots, the",
        "   statistical power of the t-test is limited.",
        "2. **Heuristic labels**: the topic labeller is a bag-of-keywords",
        "   classifier, which is noisy. A single block may contain",
        "   multiple topics, or none.",
        "3. **Co-occurrence data**: factual association requires the",
        "   model to have seen the entity pair in nearby blocks. With a",
        "   short text, many pairs have insufficient data.",
        "4. **No external evaluation**: there is no ground-truth label",
        "   set, so all metrics are PROXIES for concept emergence.",
        "",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------ #
# Main report generator
# ------------------------------------------------------------------ #
def generate_report(input_dir: Path, report_path: Path) -> str:
    """Run all three analyses and write the Markdown report."""
    # Load the manifest (experiment parameters + run summary).
    manifest_path = input_dir / "manifest.json"
    manifest = None
    if manifest_path.exists():
        with manifest_path.open() as f:
            manifest = json.load(f)

    # Run the three analyses.
    print("[1/3] Running word-boundary analysis...")
    boundary_summary = analyze_boundaries(input_dir)
    print("[2/3] Running semantic-cluster analysis...")
    cluster_summary = analyze_clusters(input_dir)
    print("[3/3] Running factual-association analysis...")
    fact_summary = analyze_facts(input_dir)

    # Build the report.
    sections = [
        render_header(manifest),
        render_consolidation(manifest),
        render_boundaries(boundary_summary, input_dir / "word_boundaries.png"),
        render_clusters(cluster_summary, input_dir / "semantic_clusters.png"),
        render_facts(fact_summary, input_dir / "factual_association.png"),
        render_conclusion(manifest, boundary_summary, cluster_summary, fact_summary),
    ]
    report = "\n".join(sections)

    # Write the report.
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return str(report_path)


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate the Phase I evaluation report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input_dir", type=str, default=str(DEFAULT_INPUT_DIR),
                   help="Directory with snapshots + manifest.json.")
    p.add_argument("--report_path", type=str, default=str(DEFAULT_REPORT_PATH),
                   help="Output Markdown report path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    report_path = Path(args.report_path)
    print("=" * 64)
    print("Generate Evaluation Report")
    print("=" * 64)
    print(f"  input_dir   = {input_dir}")
    print(f"  report_path = {report_path}")

    if not input_dir.exists():
        print(f"\nERROR: input_dir does not exist: {input_dir}")
        print("Run `python experiments/run_text_replay.py` first.")
        return

    out = generate_report(input_dir, report_path)
    print(f"\nReport written to: {out}")
    print("=" * 64)


if __name__ == "__main__":
    main()
