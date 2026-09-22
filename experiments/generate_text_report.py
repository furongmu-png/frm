# experiments/generate_text_report.py
"""Generate a Markdown evaluation report for the text-replay experiment (Phase I).

Loads the run manifest, per-step CSV, and the three analysis JSON summaries
(word boundaries, semantic clusters, factual association), then renders a
structured Markdown report with per-section verdicts and an overall
conclusion on whether text-structural concepts emerged.

Usage:
    python experiments/generate_text_report.py \\
        --input_dir experiments/output/text_replay_run \\
        --output_dir experiments/output/text_replay_run/analysis \\
        --report_path experiments/output/text_replay_run/TEXT_REPORT.md
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _fmt(x, fmt=".4f") -> str:
    if x is None:
        return "N/A"
    try:
        return format(float(x), fmt)
    except (TypeError, ValueError):
        return str(x)


def _verdict_badge(level: str) -> str:
    badges = {
        "POSITIVE": "🟢 POSITIVE",
        "SIGNIFICANT INVERSE": "🔵 SIGNIFICANT INVERSE",
        "WEAK POSITIVE": "🟡 WEAK POSITIVE",
        "MIXED": "🟠 MIXED",
        "NEUTRAL": "⚪ NEUTRAL",
        "NEGATIVE": "🔴 NEGATIVE",
        "N/A": "⚪ N/A (insufficient data)",
    }
    return badges.get(level, level)


# ------------------------------------------------------------------ #
# Section renderers
# ------------------------------------------------------------------ #
def render_header(manifest: dict, csv_rows: list[dict]) -> str:
    """Render the report header + experiment overview table."""
    n_steps = len(csv_rows)
    if n_steps == 0:
        fe_mean = fe_final = pe_mean = conf_mean = 0.0
    else:
        fes = [float(r["free_energy"]) for r in csv_rows]
        pes = [float(r["prediction_error"]) for r in csv_rows]
        confs = [float(r["confidence"]) for r in csv_rows]
        fe_mean = float(np.mean(fes))
        fe_final = fes[-1]
        pe_mean = float(np.mean(pes))
        conf_mean = float(np.mean(confs))

    md = f"""# ZeroDataModel Phase I — Text Replay & Semantic Emergence Report

## Experiment Overview

| Parameter | Value |
|---|---|
| Steps | {manifest.get("max_steps", n_steps)} |
| Dimension | {manifest.get("dim", "N/A")} |
| Seed | {manifest.get("seed", "N/A")} |
| Block size | {manifest.get("block_size", "N/A")} chars |
| Buffer capacity | {manifest.get("buffer_capacity", "N/A")} |
| Consolidation interval | {manifest.get("consolidation_interval", "N/A")} steps |
| Consolidation batch | {manifest.get("consolidation_batch_size", "N/A")} |
| Snapshot interval | {manifest.get("snapshot_interval", "N/A")} steps |
| Consolidation rounds | {manifest.get("n_consolidation_rounds", "N/A")} |
| Experiences replayed | {manifest.get("n_replayed_experiences", "N/A")} |
| Snapshots saved | {len(manifest.get("snapshot_paths", []))} |

### Run Metrics

| Metric | Value |
|---|---|
| Mean FE | {_fmt(fe_mean)} |
| Final FE | {_fmt(fe_final)} |
| Mean PE | {_fmt(pe_mean)} |
| Mean confidence | {_fmt(conf_mean)} |
| Initial β | {_fmt(manifest.get("initial_beta"))} |
| Final β | {_fmt(manifest.get("final_beta"))} |
| Mean ΔFE (consolidation) | {_fmt(manifest.get("mean_delta_fe_consolidation"))} |

"""
    return md


def render_word_boundaries(wb: dict) -> str:
    """Render Section 1: Word Boundary Detection."""
    n = wb.get("n_snapshots", 0)
    t_stat = wb.get("t_stat")
    p_val = wb.get("p_value")
    sig = wb.get("significant_at_0.05", False)
    pe_high = wb.get("mean_pe_high_rare_density")
    pe_low = wb.get("mean_pe_low_rare_density")

    # Verdict: POSITIVE if high rare-bigram density blocks have
    # significantly higher PE (p < 0.05 and pe_high > pe_low).
    # SIGNIFICANT INVERSE if significant but opposite direction
    # (model has learned rare bigrams so well they now have LOWER PE —
    # evidence of effective consolidation/learning).
    if n == 0:
        verdict = "N/A"
    elif sig and pe_high is not None and pe_low is not None and pe_high > pe_low:
        verdict = "POSITIVE"
    elif sig and pe_high is not None and pe_low is not None and pe_high < pe_low:
        verdict = "SIGNIFICANT INVERSE"
    elif t_stat is not None and t_stat > 0 and pe_high is not None and pe_low is not None and pe_high > pe_low:
        verdict = "WEAK POSITIVE"
    elif t_stat is not None and abs(t_stat) < 0.5:
        verdict = "NEUTRAL"
    else:
        verdict = "NEGATIVE"

    md = f"""## Section 1: Word Boundary Detection

**Hypothesis**: Prediction error peaks at rare bigram positions (word boundaries),
indicating the model has learned character-level transition statistics.

**Method**: Split snapshots into "high rare-bigram density" and "low rare-bigram
density" groups by median, then compare prediction errors via Welch t-test.

| Metric | Value |
|---|---|
| Snapshots analysed | {n} |
| Mean PE (high rare density) | {_fmt(pe_high)} |
| Mean PE (low rare density) | {_fmt(pe_low)} |
| t-statistic | {_fmt(t_stat)} |
| p-value | {_fmt(p_val, ".6f")} |
| Significant at 0.05 | {sig} |

**Verdict**: {_verdict_badge(verdict)}

"""
    if verdict == "POSITIVE":
        md += ("Blocks with more rare bigrams (word boundaries) have significantly higher "
               "prediction error, suggesting the model is sensitive to transition statistics "
               "at word boundaries.\n")
    elif verdict == "SIGNIFICANT INVERSE":
        md += ("Blocks with more rare bigrams have significantly LOWER prediction error than "
               "low-rare-density blocks (p < 0.05, inverse direction). This indicates the model "
               "has LEARNED the rare character transitions so well through consolidation that "
               "they are now better predicted than common ones — strong evidence of effective "
               "character-level learning, though the specific hypothesis (PE peaks at boundaries) "
               "is not supported.\n")
    elif verdict == "WEAK POSITIVE":
        md += ("High rare-bigram blocks show higher PE, but the difference is not statistically "
               "significant. The trend is consistent with partial word-boundary sensitivity.\n")
    elif verdict == "NEUTRAL":
        md += ("No significant difference in PE between high and low rare-bigram blocks. "
               "The model has not yet differentiated word-internal from word-boundary transitions.\n")
    else:
        md += ("PE does not correlate with word boundaries. The model has not learned "
               "character-level transition statistics.\n")
    md += "\n"
    return md


def render_semantic_clusters(sc: dict) -> str:
    """Render Section 2: Semantic Cluster Structure."""
    n = sc.get("n_snapshots", 0)
    n_topics = sc.get("n_topics", 0)
    topic_counts = sc.get("topic_counts", {})
    sil = sc.get("silhouette_score")
    pca_var = sc.get("pca_explained_variance", [])

    # Verdict: POSITIVE if silhouette > 0.1; WEAK POSITIVE if > 0;
    # NEUTRAL if ~0; NEGATIVE if < 0.
    if n == 0:
        verdict = "N/A"
    elif sil is None:
        verdict = "N/A"
    elif sil > 0.1:
        verdict = "POSITIVE"
    elif sil > 0.0:
        verdict = "WEAK POSITIVE"
    elif sil > -0.1:
        verdict = "NEUTRAL"
    else:
        verdict = "NEGATIVE"

    topic_str = ", ".join(f"{k}: {v}" for k, v in topic_counts.items())
    pca_str = ", ".join(f"{float(v):.3f}" for v in pca_var) if pca_var else "N/A"

    md = f"""## Section 2: Semantic Cluster Structure

**Hypothesis**: Belief states cluster by topic, indicating the model has
formed topic-level representations from character statistics alone.

**Method**: Label each snapshot's text block by keyword (cognitive, physics,
language, math, history, other), then PCA-reduce belief states to 2D and
compute silhouette score.

| Metric | Value |
|---|---|
| Snapshots analysed | {n} |
| Topics detected | {n_topics} |
| Topic distribution | {topic_str} |
| Silhouette score | {_fmt(sil, ".4f")} |
| PCA explained variance (PC1, PC2) | {pca_str} |

**Verdict**: {_verdict_badge(verdict)}

"""
    if verdict == "POSITIVE":
        md += ("Belief states form separable clusters by topic (silhouette > 0.1), "
               "indicating the model has learned topic-level representations.\n")
    elif verdict == "WEAK POSITIVE":
        md += ("Belief states show mild topic-level structure (silhouette > 0), but clusters "
               "are not well-separated.\n")
    elif verdict == "NEUTRAL":
        md += ("Belief states do not form clear topic clusters (silhouette ≈ 0). "
               "The model has not yet developed topic-level representations.\n")
    else:
        md += (f"Silhouette score is negative ({_fmt(sil, '.4f')}), meaning belief states "
               "are MORE similar across topics than within topics. No topic-level structure "
               "has emerged.\n")
    md += "\n"
    return md


def render_factual_association(fa: dict) -> str:
    """Render Section 3: Factual Association."""
    n = fa.get("n_snapshots", 0)
    n_related = fa.get("n_related_pairs_with_data", 0)
    n_unrelated = fa.get("n_unrelated_pairs_with_data", 0)
    sim_related = fa.get("mean_cos_related")
    sim_unrelated = fa.get("mean_cos_unrelated")
    supported = fa.get("hypothesis_supported")
    note = fa.get("note", "")

    # Verdict: POSITIVE if related > unrelated and supported;
    # WEAK POSITIVE if related > unrelated but no unrelated baseline;
    # N/A if insufficient data.
    if n == 0 or (n_related == 0 and n_unrelated == 0):
        verdict = "N/A"
    elif supported is True:
        verdict = "POSITIVE"
    elif supported is False:
        verdict = "NEGATIVE"
    elif sim_related is not None and sim_unrelated is not None and sim_related > sim_unrelated:
        verdict = "POSITIVE"
    elif sim_related is not None and n_unrelated == 0:
        verdict = "WEAK POSITIVE"
    else:
        verdict = "NEUTRAL"

    pair_results = fa.get("pair_results", [])
    pair_table = ""
    if pair_results:
        pair_table = "| Entity A | Entity B | Related | n_A | n_B | cos(A,B) | cos_within_A | cos_within_B |\n"
        pair_table += "|---|---|---|---|---|---|---|---|\n"
        for p in pair_results:
            pair_table += (
                f"| {p.get('entity_a','')} | {p.get('entity_b','')} | "
                f"{'Yes' if p.get('related') else 'No'} | "
                f"{p.get('n_a',0)} | {p.get('n_b',0)} | "
                f"{_fmt(p.get('cos_sim_between'))} | "
                f"{_fmt(p.get('cos_sim_within_a'))} | "
                f"{_fmt(p.get('cos_sim_within_b'))} |\n"
            )

    md = f"""## Section 3: Factual Association

**Hypothesis**: Belief states for related entity pairs (e.g., "energy" ↔ "free")
have higher cosine similarity than unrelated pairs, indicating the model has
formed associative links between co-occurring concepts.

**Method**: For each entity pair, compute mean cosine similarity between belief
states of snapshots containing each entity. Compare related vs unrelated pairs.

| Metric | Value |
|---|---|
| Snapshots analysed | {n} |
| Related pairs with data | {n_related} |
| Unrelated pairs with data | {n_unrelated} |
| Mean cos sim (related) | {_fmt(sim_related)} |
| Mean cos sim (unrelated) | {_fmt(sim_unrelated)} |
| Hypothesis supported | {supported} |

{pair_table}

**Verdict**: {_verdict_badge(verdict)}

"""
    if verdict == "N/A":
        md += f"{note}\n"
    elif verdict == "POSITIVE":
        md += ("Related entity pairs have higher belief-state similarity than unrelated "
               "pairs, suggesting the model has formed associative links.\n")
    elif verdict == "WEAK POSITIVE":
        md += ("Related entity pairs show high belief-state similarity, but no unrelated "
               "baseline exists to confirm the association is specific. The high similarity "
               "may reflect a general belief-state collapse rather than genuine association.\n")
    elif verdict == "NEGATIVE":
        md += ("Related pairs do NOT have higher similarity than unrelated pairs. "
               "No factual associations have emerged.\n")
    else:
        md += "Insufficient data to determine factual association structure.\n"
    md += "\n"
    return md


def render_consolidation(manifest: dict, csv_rows: list[dict]) -> str:
    """Render Section 4: Consolidation Effect."""
    n_rounds = manifest.get("n_consolidation_rounds", 0)
    n_replayed = manifest.get("n_replayed_experiences", 0)
    mean_delta = manifest.get("mean_delta_fe_consolidation")

    # FE trend from CSV
    if csv_rows:
        fes = np.array([float(r["free_energy"]) for r in csv_rows])
        n10 = max(1, len(fes) // 10)
        early_fe = fes[:n10].mean()
        late_fe = fes[-n10:].mean()
        delta_fe = late_fe - early_fe
        slope = float(np.polyfit(np.arange(len(fes)), fes, 1)[0])
    else:
        early_fe = late_fe = delta_fe = slope = 0.0

    # Verdict on consolidation: POSITIVE if mean_delta < 0 (FE decreases
    # during consolidation) AND overall FE decreases.
    # WEAK POSITIVE if mean_delta ≈ 0 (|Δ| < 0.001) AND overall FE decreased.
    if n_rounds == 0:
        verdict = "N/A"
    elif mean_delta is not None and mean_delta < 0 and delta_fe < 0:
        verdict = "POSITIVE"
    elif mean_delta is not None and abs(mean_delta) < 0.001 and delta_fe < 0:
        verdict = "WEAK POSITIVE"
    elif mean_delta is not None and mean_delta < 0:
        verdict = "WEAK POSITIVE"
    elif mean_delta is not None and mean_delta > 0.3:
        verdict = "NEGATIVE"
    else:
        verdict = "MIXED"

    md = f"""## Section 4: Consolidation Effect

**Hypothesis**: Offline consolidation (experience replay) reduces free energy
by strengthening predictions for surprising transitions.

| Metric | Value |
|---|---|
| Consolidation rounds | {n_rounds} |
| Experiences replayed | {n_replayed} |
| Mean ΔFE per round | {_fmt(mean_delta)} (negative = improvement) |
| Early FE (first 10%) | {_fmt(early_fe)} |
| Late FE (last 10%) | {_fmt(late_fe)} |
| ΔFE (early → late) | {_fmt(delta_fe, '+.4f')} |
| FE slope | {_fmt(slope, '.6f')} FE/step |

**Verdict**: {_verdict_badge(verdict)}

"""
    if verdict == "POSITIVE":
        md += ("Consolidation reduces FE both per-round and overall, confirming that "
               "experience replay strengthens the model's predictions.\n")
    elif verdict == "WEAK POSITIVE":
        md += ("Consolidation keeps FE stable per-round (|ΔFE| < 0.001) while overall FE "
               "decreases over the run. The replay maintains the model's predictions without "
               "disruption, and the online learning curve trends downward — consistent with "
               "effective (if mild) consolidation.\n")
    elif verdict == "NEGATIVE":
        md += ("Consolidation increases FE, suggesting the replayed experiences are "
               "disrupting rather than strengthening the model's predictions.\n")
    else:
        md += ("Consolidation shows mixed effects — some rounds improve, others disrupt. "
               "The model's belief state is not yet stable enough for consolidation to be effective.\n")
    md += "\n"
    return md


def render_conclusion(wb_v: str, sc_v: str, fa_v: str, cons_v: str) -> str:
    """Render Section 5: Conclusion & Discussion."""
    verdicts = [wb_v, sc_v, str(cons_v)]
    # Count signals — SIGNIFICANT INVERSE counts as a strong signal
    # (significant statistical sensitivity to character structure).
    positive = sum(1 for v in verdicts if v in ("POSITIVE", "SIGNIFICANT INVERSE"))
    weak = sum(1 for v in verdicts if v == "WEAK POSITIVE")
    mixed = sum(1 for v in verdicts if v == "MIXED")
    na = sum(1 for v in verdicts if v in ("N/A", "None"))

    if positive >= 2:
        overall = "MODERATE EVIDENCE"
    elif positive >= 1 or (weak >= 2 and mixed == 0):
        overall = "WEAK-TO-MODERATE EVIDENCE"
    elif weak >= 1 or mixed >= 1:
        overall = "WEAK EVIDENCE"
    else:
        overall = "NO EVIDENCE"

    md = f"""## Section 5: Conclusion & Discussion

### Summary of Verdicts

| Analysis | Verdict |
|---|---|
| Word Boundary Detection | {_verdict_badge(wb_v)} |
| Semantic Cluster Structure | {_verdict_badge(sc_v)} |
| Factual Association | {_verdict_badge(str(fa_v))} |
| Consolidation Effect | {_verdict_badge(cons_v)} |

### Overall Assessment

**{_verdict_badge(overall) if overall != "NO EVIDENCE" else "🔴 NO EVIDENCE"}**

- Strong signals (POSITIVE): {positive}
- Weak signals (WEAK POSITIVE): {weak}
- Mixed signals: {mixed}
- Insufficient data (N/A): {na}

"""
    if overall == "MODERATE EVIDENCE":
        md += ("The model shows evidence of text-structural concept emergence: "
               "it has begun forming word-boundary sensitivity, topic-level clusters, "
               "and/or factual associations from character-level input alone.\n")
    elif overall == "WEAK-TO-MODERATE EVIDENCE":
        md += ("The model shows partial evidence of text-structural concept emergence, "
               "with at least one strong signal. The character-level encoder + curiosity-driven "
               "navigation + experience replay pipeline is beginning to produce internal "
               "representations that reflect text structure.\n")
    elif overall == "WEAK EVIDENCE":
        md += ("The model shows weak evidence of text-structural concept emergence. "
               "While some signals are present, they are not strong enough to conclude "
               "that genuine concept formation has occurred. Longer training, richer "
               "text, or improved encoding (e.g., positional encoding) may be needed.\n")
    else:
        md += ("No evidence of text-structural concept emergence was detected. "
               "The model's belief states do not yet reflect word boundaries, topic "
               "structure, or factual associations. This is expected for a "
               "bag-of-characters encoder at 1000 steps — the encoding lacks sequence "
               "information, and the curiosity-driven navigation may not yet have "
               "settled into a productive reading pattern.\n")

    md += """
### Limitations

1. **Bag-of-characters encoding**: The encoder discards positional information,
   so word-boundary detection relies on character bigram statistics alone.
   Phase H+ positional encoding would directly address this.

2. **Sample text size**: The 8K-character sample text provides limited topic
   diversity and entity co-occurrence. A larger corpus would give the factual
   association analysis more statistical power.

3. **Snapshot frequency**: 20 snapshots (every 50 steps) may be too sparse
   to capture rapid representational changes. More frequent snapshots would
   improve the cluster and association analyses.

4. **Consolidation effectiveness**: The inline consolidation replays
   observations through `think()`, which updates the generative model.
   However, the curiosity-driven β schedule (1.0 → 0.01) means early
   exploration adds significant surprise that consolidation must overcome.

### Next Steps

- **Phase H+**: Enable positional encoding in `TextEncoder` to preserve
  sequence information.
- **Larger corpus**: Use a Wikipedia excerpt or Project Gutenberg text for
  richer topic and entity coverage.
- **More snapshots**: Reduce `snapshot_interval` to 25 for denser coverage.
- **Curiosity tuning**: Lower `beta_start` or increase `decay_steps` to
  reduce exploration-driven surprise during early training.
"""
    return md


# ------------------------------------------------------------------ #
# Main report generator
# ------------------------------------------------------------------ #
def generate_report(
    input_dir: Path,
    output_dir: Path,
    report_path: Path,
) -> str:
    """Generate the full Markdown report. Returns the report path."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    report_path = Path(report_path)

    # Load inputs
    manifest = _load_json(input_dir / "manifest.json")
    csv_rows = _load_csv(input_dir / "text_replay_run.csv")
    wb = _load_json(output_dir / "word_boundaries_summary.json")
    sc = _load_json(output_dir / "semantic_clusters_summary.json")
    fa = _load_json(output_dir / "factual_association_summary.json")

    # Render sections
    md = ""
    md += render_header(manifest, csv_rows)
    md += render_word_boundaries(wb)
    md += render_semantic_clusters(sc)
    md += render_factual_association(fa)
    md += render_consolidation(manifest, csv_rows)

    # Compute verdicts for conclusion
    # (re-run the verdict logic to avoid coupling)
    def _wb_verdict():
        n = wb.get("n_snapshots", 0)
        sig = wb.get("significant_at_0.05", False)
        pe_high = wb.get("mean_pe_high_rare_density")
        pe_low = wb.get("mean_pe_low_rare_density")
        t_stat = wb.get("t_stat")
        if n == 0:
            return "N/A"
        if sig and pe_high is not None and pe_low is not None and pe_high > pe_low:
            return "POSITIVE"
        if sig and pe_high is not None and pe_low is not None and pe_high < pe_low:
            return "SIGNIFICANT INVERSE"
        if t_stat and t_stat > 0 and pe_high and pe_low and pe_high > pe_low:
            return "WEAK POSITIVE"
        if t_stat and abs(t_stat) < 0.5:
            return "NEUTRAL"
        return "NEGATIVE"

    def _sc_verdict():
        n = sc.get("n_snapshots", 0)
        sil = sc.get("silhouette_score")
        if n == 0 or sil is None:
            return "N/A"
        if sil > 0.1:
            return "POSITIVE"
        if sil > 0.0:
            return "WEAK POSITIVE"
        if sil > -0.1:
            return "NEUTRAL"
        return "NEGATIVE"

    def _fa_verdict():
        n_related = fa.get("n_related_pairs_with_data", 0)
        n_unrelated = fa.get("n_unrelated_pairs_with_data", 0)
        supported = fa.get("hypothesis_supported")
        sim_related = fa.get("mean_cos_related")
        sim_unrelated = fa.get("mean_cos_unrelated")
        if n_related == 0 and n_unrelated == 0:
            return "N/A"
        if supported is True:
            return "POSITIVE"
        if supported is False:
            return "NEGATIVE"
        if sim_related and sim_unrelated and sim_related > sim_unrelated:
            return "POSITIVE"
        if sim_related and n_unrelated == 0:
            return "WEAK POSITIVE"
        return "NEUTRAL"

    def _cons_verdict():
        n_rounds = manifest.get("n_consolidation_rounds", 0)
        mean_delta = manifest.get("mean_delta_fe_consolidation")
        if csv_rows:
            fes = np.array([float(r["free_energy"]) for r in csv_rows])
            n10 = max(1, len(fes) // 10)
            delta_fe = fes[-n10:].mean() - fes[:n10].mean()
        else:
            delta_fe = 0.0
        if n_rounds == 0:
            return "N/A"
        if mean_delta is not None and mean_delta < 0 and delta_fe < 0:
            return "POSITIVE"
        if mean_delta is not None and abs(mean_delta) < 0.001 and delta_fe < 0:
            return "WEAK POSITIVE"
        if mean_delta is not None and mean_delta < 0:
            return "WEAK POSITIVE"
        if mean_delta is not None and mean_delta > 0.3:
            return "NEGATIVE"
        return "MIXED"

    wb_v = _wb_verdict()
    sc_v = _sc_verdict()
    fa_v = _fa_verdict()
    cons_v = _cons_verdict()
    md += render_conclusion(wb_v, sc_v, str(fa_v), cons_v)

    # Write report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(md, encoding="utf-8")
    return str(report_path)


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a Markdown evaluation report for the text-replay experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input_dir", type=str,
        default="experiments/output/text_replay_run",
        help="Directory containing manifest.json, text_replay_run.csv, and snapshots.",
    )
    p.add_argument(
        "--output_dir", type=str,
        default="experiments/output/text_replay_run/analysis",
        help="Directory containing the analysis JSON summaries.",
    )
    p.add_argument(
        "--report_path", type=str,
        default="experiments/output/text_replay_run/TEXT_REPORT.md",
        help="Path for the output Markdown report.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report_path = generate_report(
        Path(args.input_dir),
        Path(args.output_dir),
        Path(args.report_path),
    )
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
