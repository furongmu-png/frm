# experiments/generate_encyclopedia_report.py
"""Final report generator for the encyclopedia learning run (Phase J, Task 5).

Aggregates the run manifest, knowledge graph, concept probes, and
internal-metrics CSV into a single Markdown report
``encyclopedia_report.md``. Optionally also generates the KG
visualization PNG and the probe history PNG by invoking their
respective scripts.

The report includes:
  - Experiment parameters + run summary (from ``manifest.json``)
  - Free-energy curve + β decay (PNG from run)
  - Action distribution
  - Internal-consistency curve + re-read preference
  - Knowledge graph visualization
  - Concept-probe silhouette + intra/inter similarity
  - Top-visited articles + highest-FE articles ("hardest concepts")
  - Final verdict on whether concept emergence was observed

The script is IDEMPOTENT: re-running overwrites the report. It does
NOT re-run the learning loop — only the offline analyses (KG build +
probe eval + visualisations).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


DEFAULT_RUN_DIR = Path(__file__).resolve().parent / "output" / "encyclopedia_run"
DEFAULT_REPORT_PATH = DEFAULT_RUN_DIR / "encyclopedia_report.md"


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def _fmt(v, fmt_str: str = "{:.4f}") -> str:
    if v is None or v == "":
        return "N/A"
    if isinstance(v, (int, float)):
        try:
            return fmt_str.format(v)
        except (ValueError, TypeError):
            return str(v)
    return str(v)


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _load_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def _run_script(script: str, args: list[str]) -> bool:
    """Run a sibling script with args. Returns True on success."""
    script_path = Path(__file__).resolve().parent / script
    if not script_path.exists():
        return False
    cmd = [sys.executable, str(script_path)] + args
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


# ------------------------------------------------------------------ #
# Report sections
# ------------------------------------------------------------------ #
def render_header(manifest: dict | None) -> str:
    lines = [
        "# ZeroDataModel Encyclopedia Learning Report (Phase J)",
        "",
        "## Experiment Overview",
        "",
        "This report evaluates whether the ZeroDataModel, after running",
        "the Phase J **long-term encyclopedia learning loop** (with",
        "experience replay, offline consolidation, and 6-action",
        "navigation including hyperlink-following), shows signs of",
        "having internalised a structured knowledge graph from a large",
        "text corpus — without any external labels or pretrained",
        "embeddings.",
        "",
        "The evaluation combines:",
        "1. Run-level metrics (FE, β, action distribution)",
        "2. Internal-consistency metrics (linked vs random article pairs)",
        "3. Knowledge graph topology (nodes, edges, top articles)",
        "4. Concept-probe silhouette (intra vs inter concept similarity)",
        "",
    ]
    if manifest:
        args = manifest.get("args", {})
        lines += [
            "## Experiment Parameters",
            "",
            "| Parameter | Value |",
            "|---|---|",
            f"| `corpus_dir` | {args.get('corpus_dir', 'synthetic')} |",
            f"| `total_steps` | {manifest.get('total_steps', 'N/A')} |",
            f"| `session_steps` | {args.get('session_steps', 'N/A')} |",
            f"| `consolidation_steps` | {args.get('consolidation_steps', 'N/A')} |",
            f"| `buffer_capacity` | {args.get('buffer_capacity', 'N/A')} |",
            f"| `eval_interval` | {args.get('eval_interval', 'N/A')} |",
            f"| `dim` | {args.get('dim', 'N/A')} |",
            f"| `seed` | {args.get('seed', 'N/A')} |",
            "",
            "### Run Outcome",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| `elapsed_s` | {_fmt(manifest.get('elapsed_s'))} |",
            f"| `n_sessions` | {manifest.get('n_sessions', 'N/A')} |",
            f"| `n_consolidation_rounds` | {manifest.get('n_consolidation_rounds', 'N/A')} |",
            f"| `n_replayed_experiences` | {manifest.get('n_replayed_experiences', 'N/A')} |",
            f"| `initial_beta` | {_fmt(manifest.get('initial_beta'))} |",
            f"| `final_beta` | {_fmt(manifest.get('final_beta'))} |",
            f"| `mean_free_energy` | {_fmt(manifest.get('mean_free_energy'))} |",
            f"| `final_free_energy` | {_fmt(manifest.get('final_free_energy'))} |",
            f"| `mean_prediction_error` | {_fmt(manifest.get('mean_prediction_error'))} |",
            f"| `mean_confidence` | {_fmt(manifest.get('mean_confidence'))} |",
            f"| `n_articles_visited` | {manifest.get('n_articles_visited', 'N/A')} |",
            f"| `n_nan_obs` | {manifest.get('n_nan_obs', 'N/A')} |",
            f"| `n_crashes` | {manifest.get('n_crashes', 'N/A')} |",
            "",
            "### Action Distribution",
            "",
            "| Action | Count |",
            "|---|---|",
        ]
        for action, count in manifest.get("action_counts", {}).items():
            lines.append(f"| `{action}` | {count} |")
        lines.append("")
    else:
        lines += [
            "## Experiment Parameters",
            "",
            "_No `manifest.json` found. Run `run_encyclopedia_learning.py` first._",
            "",
        ]
    return "\n".join(lines)


def render_internal_metrics(metrics_csv: Path, png_path: Path) -> str:
    rows = _load_csv_rows(metrics_csv)
    if not rows:
        return "## 1. Internal Metrics (Phase J, Task 4)\n\n_No metrics CSV found._\n"
    lines = [
        "## 1. Internal Metrics (Phase J, Task 4)",
        "",
        "Periodic (every ``eval_interval`` steps) measurements of:",
        "1. **FE slope** — recent slope of free-energy curve (negative = improving)",
        "2. **Re-read preference** — fraction of recent actions that were `backward` or `jump_within`",
        "3. **Internal consistency** — cosine sim of belief_states for linked article pairs minus random pairs",
        "",
        "| step | fe_slope | re_read_pref | linked_sim | random_sim | consistency |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows[-10:]:  # last 10 eval points
        lines.append(
            f"| {r.get('step', '')} | {_fmt(r.get('fe_slope_recent'), '{:+.4f}')} | "
            f"{_fmt(r.get('re_read_preference'), '{:.3f}')} | "
            f"{_fmt(r.get('linked_pair_cos_sim'))} | "
            f"{_fmt(r.get('random_pair_cos_sim'))} | "
            f"{_fmt(r.get('consistency_score'), '{:+.4f}')} |"
        )
    lines.append("")
    if png_path.exists():
        lines += [
            f"![Run summary]({Path(png_path).name})",
            "",
        ]
    return "\n".join(lines)


def render_knowledge_graph(kg_json: Path, kg_png: Path) -> str:
    kg = _load_json(kg_json)
    if not kg:
        return "## 2. Knowledge Graph (Phase J, Task 3)\n\n_No knowledge graph JSON found._\n"
    lines = [
        "## 2. Knowledge Graph (Phase J, Task 3)",
        "",
        f"- **Nodes**: {kg.get('n_nodes', 0)} articles",
        f"- **Edges**: {kg.get('n_edges', 0)} (structural + co-read + latent)",
        "",
    ]
    # Top articles by visit count.
    nodes = kg.get("nodes", {})
    top_visited = sorted(
        nodes.items(), key=lambda kv: -kv[1].get("visit_count", 0),
    )[:5]
    lines += [
        "### Top-5 Most-Read Articles",
        "",
        "| Rank | Article | Visit count | Mean FE |",
        "|---|---|---|---|",
    ]
    for i, (title, attrs) in enumerate(top_visited, 1):
        lines.append(
            f"| {i} | `{title}` | {attrs.get('visit_count', 0)} | "
            f"{_fmt(attrs.get('mean_free_energy'))} |"
        )
    # Highest-FE articles ("hardest concepts").
    top_fe = sorted(
        nodes.items(),
        key=lambda kv: -kv[1].get("mean_free_energy", 0)
        if kv[1].get("n_free_energy_samples", 0) > 0 else 0,
    )[:5]
    lines += [
        "",
        "### Top-5 \"Hardest Concepts\" (highest mean FE)",
        "",
        "| Rank | Article | Mean FE | Visit count |",
        "|---|---|---|---|",
    ]
    for i, (title, attrs) in enumerate(top_fe, 1):
        if attrs.get("n_free_energy_samples", 0) == 0:
            continue
        lines.append(
            f"| {i} | `{title}` | {_fmt(attrs.get('mean_free_energy'))} | "
            f"{attrs.get('visit_count', 0)} |"
        )
    lines.append("")
    if kg_png.exists():
        lines += [
            f"![Knowledge graph]({Path(kg_png).name})",
            "",
        ]
    return "\n".join(lines)


def render_probes(probe_metrics: dict, probe_history_png: Path) -> str:
    if not probe_metrics:
        return "## 3. Concept Probes (Phase J, Task 3.3)\n\n_No probe metrics found._\n"
    lines = [
        "## 3. Concept Probes (Phase J, Task 3.3)",
        "",
        "Predefined set of concepts (physics, biology, math, history,",
        "geography, computer science). For each concept we found articles",
        "containing trigger words and computed:",
        "  - **intra-concept similarity**: mean cosine sim of belief_states",
        "    WITHIN a concept (should rise over time as concepts form)",
        "  - **inter-concept similarity**: mean cosine sim BETWEEN concepts",
        "    (should stay flat or fall as concepts separate)",
        "  - **silhouette score**: standard clustering score (sklearn)",
        "",
        f"- **n_probes**: {probe_metrics.get('n_probes', 'N/A')}",
        f"- **n_probes_with_articles**: {probe_metrics.get('n_probes_with_articles', 'N/A')}",
        f"- **Mean intra-concept similarity**: {_fmt(probe_metrics.get('mean_intra_similarity'))}",
        f"- **Inter-concept similarity**: {_fmt(probe_metrics.get('inter_concept_similarity'))}",
        f"- **Silhouette score**: {_fmt(probe_metrics.get('silhouette_score'))}",
        f"  - Note: {probe_metrics.get('silhouette_note', '')}",
        "",
        "### Articles per probe",
        "",
        "| Probe | n articles |",
        "|---|---|",
    ]
    for probe, n in probe_metrics.get("articles_per_probe", {}).items():
        lines.append(f"| `{probe}` | {n} |")
    lines.append("")
    if probe_history_png.exists():
        lines += [
            f"![Probe history]({Path(probe_history_png).name})",
            "",
        ]
    return "\n".join(lines)


def render_conclusion(
    manifest: dict | None,
    metrics_rows: list[dict],
    kg: dict | None,
    probe_metrics: dict | None,
) -> str:
    lines = [
        "## 4. Conclusion & Discussion",
        "",
        "### Overall Emergence Verdict",
        "",
    ]
    n_signals = 0
    n_total = 0
    signals = []

    # Signal 1: FE trend (negative slope = improving).
    n_total += 1
    if metrics_rows:
        last = metrics_rows[-1]
        slope = float(last.get("fe_slope_recent", 0))
        if slope < 0:
            n_signals += 1
            signals.append(
                f"FE trend: POSITIVE (slope = {slope:+.4f}, model is improving predictions)"
            )
        else:
            signals.append(
                f"FE trend: NEUTRAL/NEGATIVE (slope = {slope:+.4f}, no improvement)"
            )
    else:
        signals.append("FE trend: N/A (no metrics data)")

    # Signal 2: internal consistency (positive = linked pairs more similar).
    n_total += 1
    if metrics_rows:
        last = metrics_rows[-1]
        cons = float(last.get("consistency_score", 0))
        if cons > 0:
            n_signals += 1
            signals.append(
                f"Internal consistency: POSITIVE (Δ = {cons:+.4f}, "
                "linked pairs are more similar than random)"
            )
        else:
            signals.append(
                f"Internal consistency: NEUTRAL/NEGATIVE (Δ = {cons:+.4f})"
            )
    else:
        signals.append("Internal consistency: N/A (no metrics data)")

    # Signal 3: knowledge graph has at least N nodes (visited enough articles).
    n_total += 1
    if kg:
        n_nodes = kg.get("n_nodes", 0)
        if n_nodes >= 5:
            n_signals += 1
            signals.append(
                f"Knowledge graph: POSITIVE ({n_nodes} nodes, "
                "model accumulated article-level structure)"
            )
        else:
            signals.append(
                f"Knowledge graph: NEUTRAL ({n_nodes} nodes, too few to assess)"
            )

    # Signal 4: probe silhouette > 0 (concepts are separable).
    n_total += 1
    if probe_metrics:
        silh = probe_metrics.get("silhouette_score")
        if silh is not None and silh > 0:
            n_signals += 1
            signals.append(
                f"Probe silhouette: POSITIVE (silhouette = {silh:.3f}, "
                "concept-level structure in belief_state)"
            )
        else:
            signals.append(
                f"Probe silhouette: NEUTRAL/NEGATIVE (silhouette = {silh})"
            )

    lines.append(f"**{n_signals}/{n_total} positive emergence signals detected.**")
    lines.append("")
    for s in signals:
        lines.append(f"- {s}")
    lines.append("")

    if n_signals >= 3:
        verdict = "**Strong evidence** of knowledge-graph emergence."
    elif n_signals >= 2:
        verdict = "**Moderate evidence** of knowledge-graph emergence."
    elif n_signals >= 1:
        verdict = "**Weak evidence** — partial structure forming."
    else:
        verdict = "**No evidence** — model needs more training, larger corpus, or longer run."
    lines += [
        f"**Verdict**: {verdict}",
        "",
        "### Discussion",
        "",
        "These results are PROVISIONAL. The Phase J loop is designed for",
        "long-term (50K+ steps) absorption of large corpora. A short",
        "smoke-test run on a synthetic corpus will not produce",
        "meaningful emergence — the model needs sufficient exposure to",
        "real Wikipedia-scale text for the belief_state to develop",
        "stable article-level representations.",
        "",
        "The most informative signals are:",
        "1. **FE trend** — directly probes whether the generative model",
        "   is learning to predict character-level statistics.",
        "2. **Internal consistency** — directly probes whether the",
        "   belief_state encodes the corpus's hyperlink structure.",
        "3. **Probe silhouette** — directly probes whether concept-level",
        "   structure exists in the belief_state.",
        "",
        "A production evaluation would use:",
        "- A real Simple Wikipedia dump (~200K articles)",
        "- 50,000+ steps of training (≈3 hours)",
        "- sklearn for silhouette scoring",
        "- A larger model dimension (64+) for richer representations",
        "",
        "### Limitations",
        "",
        "1. **Synthetic corpus fallback**: if no real Wikipedia dump is",
        "   available, the loop generates a tiny synthetic corpus (10-100",
        "   articles) which is too small for meaningful emergence.",
        "2. **No external evaluation**: there is no held-out test set",
        "   or ground-truth knowledge graph, so all metrics are PROXIES",
        "   for emergence.",
        "3. **Memory**: the experience buffer is capped at 10K entries,",
        "   so very long runs may evict useful early experiences.",
        "4. **Single-pass**: the loop does ONE pass over the corpus. A",
        "   production system would benefit from multiple passes with",
        "   curriculum-style article ordering.",
        "",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def generate_report(run_dir: Path, report_path: Path) -> str:
    """Generate the full report."""
    manifest = _load_json(run_dir / "manifest.json")
    metrics_csv = run_dir / "encyclopedia_metrics.csv"
    metrics_rows = _load_csv_rows(metrics_csv)

    # Build the KG if it doesn't exist.
    kg_json = run_dir / "knowledge_graph.json"
    if not kg_json.exists():
        print("[kg] building knowledge graph from CSV...")
        csv_path = run_dir / "encyclopedia_run.csv"
        if csv_path.exists():
            _run_script("knowledge_graph_builder.py",
                       ["--csv", str(csv_path), "--output", str(kg_json)])
    kg = _load_json(kg_json)

    # Render the KG PNG.
    kg_png = kg_json.with_suffix(".png")
    if kg and not kg_png.exists():
        print("[kg] rendering knowledge graph PNG...")
        _run_script("visualize_knowledge.py",
                   ["--input", str(kg_json), "--output", str(kg_png)])

    # Run the probes if they don't exist.
    probe_metrics_path = run_dir / "probes" / "probe_metrics.json"
    if not probe_metrics_path.exists():
        print("[probes] running concept probes evaluation...")
        ckpt_dir = run_dir / "checkpoints"
        # Use the latest checkpoint if available.
        ckpts = sorted(ckpt_dir.glob("ckpt_step*")) if ckpt_dir.exists() else []
        ckpt_arg = str(ckpts[-1]) if ckpts else ""
        _run_script("concept_probes.py",
                   ["--checkpoint", ckpt_arg, "--output_dir",
                    str(run_dir / "probes")])
    probe_metrics = _load_json(probe_metrics_path)
    probe_history_png = run_dir / "probes" / "probe_history.png"

    # Build the report.
    sections = [
        render_header(manifest),
        render_internal_metrics(metrics_csv, run_dir / "encyclopedia_run.png"),
        render_knowledge_graph(kg_json, kg_png),
        render_probes(probe_metrics, probe_history_png),
        render_conclusion(manifest, metrics_rows, kg, probe_metrics),
    ]
    report = "\n".join(sections)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return str(report_path)


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate the Phase J encyclopedia learning report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run_dir", type=str, default=str(DEFAULT_RUN_DIR),
                   help="Directory containing manifest.json + CSV + checkpoints.")
    p.add_argument("--report_path", type=str, default=str(DEFAULT_REPORT_PATH),
                   help="Output Markdown report path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    report_path = Path(args.report_path)
    print("=" * 64)
    print("Generate Encyclopedia Report")
    print("=" * 64)
    print(f"  run_dir     = {run_dir}")
    print(f"  report_path = {report_path}")
    if not run_dir.exists():
        print(f"\nERROR: run_dir does not exist: {run_dir}")
        print("Run `python experiments/run_encyclopedia_learning.py` first.")
        return
    out = generate_report(run_dir, report_path)
    print(f"\nReport written to: {out}")
    print("=" * 64)


if __name__ == "__main__":
    main()
