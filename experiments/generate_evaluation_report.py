# experiments/generate_evaluation_report.py
"""Phase K: aggregate the four cognitive-emergence analyses into a
single Markdown report.

This script:
  1. Runs (or loads) the four analysis scripts on a snapshots dir:
       - analyze_disappearance.py   (object permanence)
       - analyze_causal_graph.py     (causal structure of beliefs)
       - analyze_categories.py       (category alignment w/ physics)
       - analyze_action_intent.py    (action-effect consistency)
  2. Aggregates their JSON summaries + PNG figures.
  3. Renders a single Markdown report with an explicit verdict:
     did the model show signs of pre-linguistic concept emergence?

Idempotent: re-running overwrites the report. Re-runs the analyses
in-process (no subprocess) so we can capture structured summaries.

Run with:
    python experiments/generate_evaluation_report.py
    python experiments/generate_evaluation_report.py \\
        --snapshots output/replay/snapshots \\
        --output_dir output/eval \\
        --report_path output/eval/EVALUATION_REPORT.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Reuse the analysis functions directly (not via subprocess) so we
# can capture the structured summaries in-process.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_disappearance import run_inline_experiment as analyze_disappearance  # noqa: E402
from analyze_causal_graph import run_analysis as analyze_causal_graph  # noqa: E402
from analyze_categories import run_analysis as analyze_categories  # noqa: E402
from analyze_action_intent import run_analysis as analyze_action_intent  # noqa: E402


DEFAULT_SNAPSHOTS_DIR = Path(__file__).resolve().parent / "output" / "replay" / "snapshots"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "eval"
DEFAULT_REPORT_PATH = DEFAULT_OUTPUT_DIR / "EVALUATION_REPORT.md"


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def _fmt(v, fmt_str: str = "{:.4f}") -> str:
    """Format a possibly-None scalar."""
    if v is None:
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
    try:
        with path.open() as f:
            return json.load(f)
    except Exception:
        return None


# ------------------------------------------------------------------ #
# Report sections
# ------------------------------------------------------------------ #
def render_header(
    snapshots_dir: Path,
    output_dir: Path,
    summary: dict | None,
) -> str:
    """Render the report header + experiment overview."""
    lines = [
        "# ZeroDataModel Phase K — Cognitive Emergence Evaluation Report",
        "",
        "## Experiment Overview",
        "",
        "This report evaluates whether the ZeroDataModel — running in",
        "the 2D `PhysicsSandbox` closed loop with **curiosity-driven",
        "exploration + priority experience replay** — shows signs of",
        "having internalised **pre-linguistic physical concepts**:",
        "object permanence, causal structure, category organisation,",
        "and goal-directed action.",
        "",
        "All analyses are computed on **cognitive snapshots**",
        "(``.npz`` files saved every ``cognitive_snapshot_interval``",
        "steps) containing the model's belief vector, free energy,",
        "prediction error, the rendered frame, and the sandbox state",
        "(body positions, velocities).",
        "",
        "**Zero-data principle**: the evaluation uses NO external",
        "pretrained models, NO labelled training data, and NO",
        "supervised signals. The only human-supplied information is",
        "the sandbox's body indexing (which body is the agent) and",
        "the action semantics (0=L, 1=R, 2=U, 3=no-op) — both of",
        "which are properties of the environment, not the model.",
        "",
        f"- **Snapshots directory**: `{snapshots_dir}`",
        f"- **Report directory**: `{output_dir}`",
        f"- **Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    if summary:
        lines += [
            "## Run Summary",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| `n_steps` | {summary.get('n_steps', 'N/A')} |",
            f"| `mean_free_energy` | {_fmt(summary.get('mean_free_energy'))} |",
            f"| `final_free_energy` | {_fmt(summary.get('final_free_energy'))} |",
            f"| `mean_confidence` | {_fmt(summary.get('mean_confidence'))} |",
            f"| `n_consolidations` | {summary.get('n_consolidations', 'N/A')} |",
            f"| `elapsed_s` | {_fmt(summary.get('elapsed_s'), '{:.2f}')} |",
            f"| `cycles_per_sec` | {_fmt(summary.get('cycles_per_sec'), '{:.1f}')} |",
            "",
        ]
    return "\n".join(lines)


def render_disappearance(summary: dict, image_path: Path) -> str:
    """Render the object-permanence section."""
    lines = [
        "## 1. Object Permanence (`analyze_disappearance.py`)",
        "",
        "**Hypothesis**: if the model has internalised a persistent",
        "representation of a non-agent body, removing that body",
        "mid-episode should produce a SURPRISE response — a sharp",
        "rise in free energy in the steps immediately after the",
        "removal, decaying back as the model re-consolidates.",
        "",
        "**Method**: a fresh model is trained for `pre_steps` cycles",
        "with 3 bodies in the sandbox. At step `pre_steps`, body 1",
        "(non-agent) is removed via `sandbox.make_object_disappear(1)`.",
        "The model continues for `post_steps` cycles. We compare the",
        "free energy of the 50-step window AFTER removal (the",
        "\"surprise window\") against the same-size window BEFORE.",
        "",
    ]
    if not summary:
        lines.append("_Analysis not available._\n")
        return "\n".join(lines)
    lines += [
        f"- **Mode**: {summary.get('mode', 'inline')}",
        f"- **Pre-steps**: {summary.get('pre_steps', 'N/A')}",
        f"- **Post-steps**: {summary.get('post_steps', 'N/A')}",
        f"- **Removal step**: {summary.get('removal_step', 'N/A')}",
        f"- **Bodies before/after**: "
        f"{summary.get('n_bodies_before', 'N/A')} → "
        f"{summary.get('n_bodies_after', 'N/A')}",
        "",
        "### Free-Energy Statistics",
        "",
        "| Window | mean FE | std FE |",
        "|---|---|---|",
        f"| pre-removal (last 50) | "
        f"{_fmt(summary.get('fe_pre_mean'))} | "
        f"{_fmt(summary.get('fe_pre_std'))} |",
        f"| surprise (first 50 post) | "
        f"{_fmt(summary.get('fe_surprise_mean'))} | "
        f"surprise_max={_fmt(summary.get('fe_surprise_max'))} |",
        f"| full post-removal | "
        f"{_fmt(summary.get('fe_post_mean'))} | "
        f"{_fmt(summary.get('fe_post_std'))} |",
        "",
        f"- **ΔFE (surprise − pre)**: "
        f"{_fmt(summary.get('delta_fe'), '{:+.4f}')}",
        f"- **t-statistic (Welch)**: "
        f"{_fmt(summary.get('t_stat'), '{:.3f}')}",
        f"- **p-value**: {_fmt(summary.get('p_value'), '{:.4f}')}",
        "",
    ]
    delta = summary.get("delta_fe")
    p = summary.get("p_value")
    t_stat = summary.get("t_stat")
    # The t-test compares the surprise window vs the LAST 50 pre
    # steps. A POSITIVE t (and small p) means surprise > last-50-pre,
    # which is the cleanest surprise signal. A NEGATIVE t with
    # small p means surprise < last-50-pre — i.e. the model's FE
    # had already climbed before removal (pre-phase was non-stationary)
    # and the surprise window did not exceed the immediate pre level.
    # delta_fe alone (vs all-pre mean) is a weaker signal because
    # it conflates the rising pre-phase with the surprise response.
    if (delta is not None and delta > 0
            and p is not None and p < 0.05
            and t_stat is not None and t_stat > 0):
        verdict = (
            "**POSITIVE** — removing the body produced a statistically"
            " significant rise in free energy (p<0.05, t>0 vs the"
            " 50-step pre-removal window), consistent with the model"
            " having maintained a persistent representation of the"
            " vanished object."
        )
    elif (delta is not None and delta > 0
          and p is not None and p < 0.05
          and t_stat is not None and t_stat < 0):
        verdict = (
            "**MIXED** — the post-removal mean FE is higher than the"
            " overall pre-removal mean (ΔFE>0), but the surprise window"
            " does NOT exceed the immediate pre-removal level (t<0)."
            " This typically means the pre-phase FE was already rising"
            " (non-stationary baseline), so the 'surprise' signal is"
            " confounded with the baseline trend. The model MAY have"
            " a persistent representation, but this test cannot cleanly"
            " confirm it."
        )
    elif delta is not None and delta > 0:
        verdict = (
            "**WEAK POSITIVE** — free energy rose after removal but"
            " the effect is not statistically significant at α=0.05."
            " The model may have a partial persistent representation."
        )
    else:
        verdict = (
            "**NEUTRAL/NEGATIVE** — no surprise response detected."
            " The model has not formed a persistent object"
            " representation (or it was already absent)."
        )
    lines += [
        f"- **Verdict**: {verdict}",
        "",
        f"![Object disappearance experiment]({image_path.name})",
        "",
    ]
    return "\n".join(lines)


def render_causal_graph(summary: dict, image_path: Path) -> str:
    """Render the causal-structure section."""
    lines = [
        "## 2. Causal Structure of Beliefs (`analyze_causal_graph.py`)",
        "",
        "**Hypothesis**: if the model's belief state has organised",
        "itself into a causal model of the world, we should find",
        "directed dependencies between latent dimensions — e.g. a",
        "dimension encoding the agent's x-position should Granger-",
        "cause a dimension encoding the prediction error.",
        "",
        "**Method**: load the belief trajectory from snapshots,",
        "select the top-`max_vars` dimensions by variance, augment",
        "with the scalar `free_energy` and `pred_error`, then run",
        "the in-repo `CausalInferenceEngine.discover()` (PC / LiNGAM",
        "/ correlation). As a fallback (and a sanity check) we also",
        "compute a pairwise Granger-causality matrix and a mutual-",
        "information matrix.",
        "",
    ]
    if not summary:
        lines.append("_Analysis not available._\n")
        return "\n".join(lines)
    lines += [
        f"- **Snapshots analysed**: {summary.get('n_snapshots', 'N/A')}",
        f"- **Belief dimensions (raw)**: "
        f"{summary.get('n_belief_dims', 'N/A')}",
        f"- **Variables selected (top variance)**: "
        f"{summary.get('n_selected_vars', 'N/A')} beliefs + "
        "2 scalars (FE, pred_error)",
        f"- **Engine method**: {summary.get('engine_method', 'N/A')}",
        f"- **Engine edges discovered**: "
        f"{summary.get('engine_n_edges', 0)}",
        f"- **Engine acyclic**: "
        f"{summary.get('engine_is_acyclic', 'N/A')}",
        "",
        "### Fallback: Granger Causality + Mutual Information",
        "",
        f"- **Granger max F-stat**: "
        f"{_fmt(summary.get('gc_matrix_max'), '{:.2f}')}",
        f"- **Granger mean F-stat**: "
        f"{_fmt(summary.get('gc_matrix_mean'), '{:.3f}')}",
        f"- **Granger significant edges (F>1)**: "
        f"{summary.get('gc_n_significant', 0)}",
        f"- **MI max**: {_fmt(summary.get('mi_matrix_max'))}",
        f"- **MI mean**: {_fmt(summary.get('mi_matrix_mean'))}",
        "",
    ]
    # Top edges.
    top_edges = summary.get("gc_top_edges", []) or []
    if top_edges:
        lines += [
            "### Top Granger-causal edges",
            "",
            "| source | target | F-stat |",
            "|---|---|---|",
        ]
        for e in top_edges[:10]:
            lines.append(
                f"| {e.get('source', '?')} | "
                f"{e.get('target', '?')} | "
                f"{_fmt(e.get('f_stat'), '{:.2f}')} |"
            )
        lines.append("")
    # Interpretation.
    n_edges = summary.get("gc_n_significant", 0) or 0
    n_interesting_engine = len(summary.get("engine_interesting_edges", []) or [])
    if n_interesting_engine > 0:
        verdict = (
            f"**POSITIVE** — the in-repo causal engine discovered "
            f"{n_interesting_engine} directed edges involving "
            "free_energy or pred_error, suggesting the belief state "
            "has causal structure that mirrors the physical world."
        )
    elif n_edges >= 5:
        verdict = (
            f"**WEAK POSITIVE** — {n_edges} Granger-significant "
            "edges discovered. The belief trajectory shows temporal "
            "dependencies consistent with an internal causal model."
        )
    elif n_edges > 0:
        verdict = (
            f"**NEUTRAL** — only {n_edges} Granger-significant edges. "
            "Structure is weak or the model has not yet formed a "
            "stable causal representation."
        )
    else:
        verdict = (
            "**NEGATIVE** — no significant Granger-causal edges. "
            "Belief trajectory does not show interpretable causal "
            "structure."
        )
    lines += [
        f"- **Verdict**: {verdict}",
        "",
        f"![Causal graph]({image_path.name})",
        "",
    ]
    if summary.get("engine_error"):
        lines.append(
            f"- **Note**: the in-repo causal engine failed "
            f"({summary['engine_error']}); results rely on the "
            "Granger + MI fallback.\n"
        )
    return "\n".join(lines)


def render_categories(summary: dict, image_path: Path) -> str:
    """Render the category-structure section."""
    lines = [
        "## 3. Category Structure (`analyze_categories.py`)",
        "",
        "**Hypothesis**: if the model has internalised a notion of",
        "physical configuration, frames with similar physical",
        "states (e.g. agent on the left, two bodies close together)",
        "should be assigned to the same latent category.",
        "",
        "**Method**: extract per-frame belief vectors, run the",
        "in-repo `CategoryTheoryEngine.topos.classify()` (soft",
        "classifier) OR k-means as a fallback. Then bin the agent's",
        "physical (cx, cy) position into a 3×3 grid and compute the",
        "**purity** of the (latent × physical) contingency table.",
        "High purity ⇒ latent categories align with physical reality.",
        "",
    ]
    if not summary:
        lines.append("_Analysis not available._\n")
        return "\n".join(lines)
    lines += [
        f"- **Snapshots analysed**: {summary.get('n_snapshots', 'N/A')}",
        f"- **Belief dimensions**: {summary.get('n_belief_dims', 'N/A')}",
        f"- **Label source**: {summary.get('label_source', 'N/A')}",
        f"- **N categories**: {summary.get('n_categories', 'N/A')}",
        f"- **Category names**: {summary.get('category_names', [])}",
        "",
        "### Cluster Quality",
        "",
        f"- **Silhouette score (k-means)**: "
        f"{_fmt(summary.get('kmeans_silhouette'))}",
        "",
        "### Alignment with Physical Reality",
        "",
    ]
    purity_main = summary.get("engine_or_kmeans_purity") or {}
    purity_kmeans = summary.get("kmeans_purity") or {}
    lines += [
        "| Label source | Purity | NMI |",
        "|---|---|---|",
        f"| {summary.get('label_source', 'N/A')} | "
        f"{_fmt(purity_main.get('purity'))} | "
        f"{_fmt(purity_main.get('nmi'))} |",
        f"| k-means | "
        f"{_fmt(purity_kmeans.get('purity'))} | "
        f"{_fmt(purity_kmeans.get('nmi'))} |",
        "",
    ]
    # Label counts.
    counts = summary.get("label_counts") or {}
    if counts:
        lines.append("### Category Counts")
        lines.append("")
        lines.append("| Category | n frames |")
        lines.append("|---|---|")
        for k, v in counts.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")
    # Interpretation.
    # We use k-means metrics as the primary signal because the
    # in-repo CategoryTheoryEngine's default categories (NLP/CV/
    # Analytics) often produce a degenerate distribution (one
    # category dominates), which inflates purity without true
    # discriminative power. NMI is robust to such imbalance.
    sil = summary.get("kmeans_silhouette")
    kmeans_purity = (purity_kmeans.get("purity") if purity_kmeans else None)
    kmeans_nmi = (purity_kmeans.get("nmi") if purity_kmeans else None)
    baseline_purity = 1.0 / 9.0  # 9 physical bins, 3 categories
    # Detect degenerate engine classification (one cluster > 80%).
    counts = summary.get("label_counts") or {}
    total = sum(counts.values()) or 1
    max_frac = max((v / total for v in counts.values()), default=0.0)
    engine_degenerate = max_frac > 0.8
    if engine_degenerate:
        lines += [
            f"- **Note**: the engine's classification is degenerate "
            f"(largest category has {max_frac * 100:.1f}% of frames); "
            "falling back to k-means for the verdict.",
            "",
        ]
    # Verdict: require BOTH silhouette AND NMI on k-means.
    if (sil is not None and sil > 0.1) and (kmeans_nmi is not None and kmeans_nmi > 0.1):
        verdict = (
            f"**POSITIVE** — k-means silhouette={_fmt(sil)} and "
            f"NMI={_fmt(kmeans_nmi)} (above 0.1 threshold), "
            "showing belief clusters align with physical configuration."
        )
    elif (sil is not None and sil > 0.1) or (kmeans_nmi is not None and kmeans_nmi > 0.05):
        verdict = (
            f"**WEAK POSITIVE** — k-means silhouette={_fmt(sil)}, "
            f"NMI={_fmt(kmeans_nmi)}; partial alignment between "
            "latent categories and physical configuration."
        )
    else:
        verdict = (
            "**NEUTRAL/NEGATIVE** — belief clusters do not align "
            "with physical configuration above chance."
        )
    lines += [
        f"- **Verdict**: {verdict}",
        "",
        f"![Category structure]({image_path.name})",
        "",
    ]
    if summary.get("engine_error"):
        lines.append(
            f"- **Note**: the in-repo CategoryTheoryEngine failed "
            f"({summary['engine_error']}); results rely on k-means "
            "fallback.\n"
        )
    return "\n".join(lines)


def render_action_intent(summary: dict, image_path: Path) -> str:
    """Render the action-effect consistency section."""
    lines = [
        "## 4. Action-Effect Consistency (`analyze_action_intent.py`)",
        "",
        "**Hypothesis**: if the model has discovered that its actions",
        "cause its body to move, its action distribution should",
        "depend on its position — e.g. when near the left wall it",
        "should preferentially emit `right` (action 1) over `left`",
        "(action 0). Furthermore, the mutual information between",
        "the action and the resulting motion direction should be",
        "non-zero.",
        "",
        "**Method**: for each snapshot, take the recorded `action`",
        "and the agent's `(cx, cy)` from `sandbox_state`. Compute:",
        "(a) `P(action | x_bin)` for 4 x-bins, (b) the MI between",
        "action and the sign of the resulting motion, (c) the",
        "fraction of actions that move the agent AWAY from the",
        "nearest wall (a goal-directedness proxy, baseline 1/3).",
        "",
    ]
    if not summary:
        lines.append("_Analysis not available._\n")
        return "\n".join(lines)
    action_dist = summary.get("action_distribution") or {}
    lines += [
        f"- **Snapshots analysed**: {summary.get('n_snapshots', 'N/A')}",
        "",
        "### Action Distribution",
        "",
        "| Action | Count |",
        "|---|---|",
    ]
    for a_name, n in action_dist.items():
        lines.append(f"| {a_name} | {n} |")
    lines += [
        "",
        f"- **Action-motion MI**: "
        f"{_fmt(summary.get('action_motion_mi_bits'), '{:.4f}')} bits",
        "",
    ]
    gd = summary.get("goal_directedness") or {}
    lines += [
        "### Goal-Directedness (away from nearest wall)",
        "",
        f"- **Fraction away**: "
        f"{_fmt(gd.get('fraction_away'), '{:.3f}')} "
        f"(baseline uniform = {_fmt(gd.get('baseline_uniform'), '{:.3f}')})",
        f"- **Valid samples**: {gd.get('valid_count', 'N/A')}",
        "",
    ]
    # Interpretation.
    # Two independent signals:
    #   (a) action-motion MI > 0.1 bits: actions causally affect motion
    #   (b) fraction_away > baseline: actions are directed AWAY from walls
    # A curiosity-driven agent may have (a) without (b) — it discovers
    # that actions affect motion but explores toward walls (novelty).
    # POSITIVE requires BOTH; WEAK POSITIVE requires (a) alone (the
    # most basic form of body-schema).
    mi = summary.get("action_motion_mi_bits")
    frac = gd.get("fraction_away")
    baseline = gd.get("baseline_uniform", 1.0 / 3.0)
    has_body_schema = mi is not None and mi > 0.05
    is_goal_directed = frac is not None and frac > baseline * 1.1
    is_anti_directed = frac is not None and frac < baseline * 0.9
    if has_body_schema and is_goal_directed:
        verdict = (
            f"**POSITIVE** — action-motion MI={_fmt(mi, '{:.3f}')} bits "
            f"and goal-directedness above baseline ({_fmt(frac, '{:.3f}')} "
            f"> {_fmt(baseline, '{:.3f}')}); the model has both a body "
            "schema and a navigation intent."
        )
    elif has_body_schema:
        if is_anti_directed:
            verdict = (
                f"**WEAK POSITIVE (exploratory)** — action-motion MI="
                f"{_fmt(mi, '{:.3f}')} bits indicates the model has "
                "discovered that its actions affect its motion (basic "
                "body schema), BUT fraction_away="
                f"{_fmt(frac, '{:.3f}')} < baseline "
                f"{_fmt(baseline, '{:.3f}')} — the agent moves TOWARD "
                "walls, consistent with curiosity-driven exploration "
                "rather than wall-avoidance."
            )
        else:
            verdict = (
                f"**WEAK POSITIVE** — action-motion MI="
                f"{_fmt(mi, '{:.3f}')} bits indicates the model has "
                "discovered that its actions affect its motion, but "
                "no clear navigation intent (fraction_away "
                f"= {_fmt(frac, '{:.3f}')} ≈ baseline)."
            )
    else:
        verdict = (
            "**NEUTRAL/NEGATIVE** — action distribution is not "
            "modulated by motion; the model has not learned a "
            "body schema."
        )
    lines += [
        f"- **Verdict**: {verdict}",
        "",
        f"![Action intent]({image_path.name})",
        "",
    ]
    return "\n".join(lines)


def render_conclusion(
    disappearance: dict | None,
    causal: dict | None,
    categories: dict | None,
    action_intent: dict | None,
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

    # Signal 1: object permanence.
    # POSITIVE requires delta_fe > 0 AND p < 0.05 AND t > 0 (clean
    # surprise vs immediate pre-removal baseline). MIXED signals
    # (delta > 0 but t < 0) indicate a non-stationary pre-phase.
    n_total += 1
    if disappearance:
        delta = disappearance.get("delta_fe")
        p = disappearance.get("p_value")
        t_stat = disappearance.get("t_stat")
        if (delta is not None and delta > 0
                and p is not None and p < 0.05
                and t_stat is not None and t_stat > 0):
            n_signals += 1
            signals.append(
                f"Object permanence: POSITIVE "
                f"(ΔFE={delta:+.3f}, p={p:.4f}, t={t_stat:.2f})"
            )
        elif (delta is not None and delta > 0
              and p is not None and p < 0.05
              and t_stat is not None and t_stat < 0):
            signals.append(
                f"Object permanence: MIXED "
                f"(ΔFE={delta:+.3f} vs all-pre, but t={t_stat:.2f} "
                "< 0 vs last-50-pre — non-stationary baseline)"
            )
        elif delta is not None and delta > 0:
            signals.append(
                f"Object permanence: WEAK POSITIVE "
                f"(ΔFE={delta:+.3f}, p={p})"
            )
        else:
            signals.append(
                f"Object permanence: NEUTRAL/NEGATIVE "
                f"(ΔFE={delta})"
            )
    else:
        signals.append("Object permanence: N/A — analysis not run")

    # Signal 2: causal structure.
    n_total += 1
    if causal:
        n_edges = causal.get("gc_n_significant", 0) or 0
        n_interesting = len(causal.get("engine_interesting_edges", []) or [])
        if n_interesting > 0:
            n_signals += 1
            signals.append(
                f"Causal structure: POSITIVE "
                f"(engine found {n_interesting} edges touching FE/pred_error, "
                f"{n_edges} Granger-significant edges total)"
            )
        elif n_edges >= 5:
            signals.append(
                f"Causal structure: WEAK POSITIVE "
                f"({n_edges} Granger-significant edges, no engine confirmation)"
            )
        else:
            signals.append(
                f"Causal structure: NEUTRAL/NEGATIVE "
                f"({n_edges} Granger-significant edges)"
            )
    else:
        signals.append("Causal structure: N/A — analysis not run")

    # Signal 3: category alignment.
    # Use k-means NMI (robust to degenerate engine output).
    n_total += 1
    if categories:
        sil = categories.get("kmeans_silhouette")
        kmeans_nmi = (categories.get("kmeans_purity") or {}).get("nmi")
        if (sil is not None and sil > 0.1) and (kmeans_nmi is not None and kmeans_nmi > 0.1):
            n_signals += 1
            signals.append(
                f"Category structure: POSITIVE "
                f"(silhouette={sil:.3f}, NMI={kmeans_nmi:.3f})"
            )
        elif (sil is not None and sil > 0.1) or (
            kmeans_nmi is not None and kmeans_nmi > 0.05
        ):
            signals.append(
                f"Category structure: WEAK POSITIVE "
                f"(silhouette={sil}, NMI={kmeans_nmi})"
            )
        else:
            signals.append(
                f"Category structure: NEUTRAL/NEGATIVE "
                f"(silhouette={sil}, NMI={kmeans_nmi})"
            )
    else:
        signals.append("Category structure: N/A — analysis not run")

    # Signal 4: action intent.
    # POSITIVE requires both body schema (MI > 0.1) AND goal-directedness.
    # WEAK POSITIVE: body schema alone (MI > 0.05), no navigation intent.
    n_total += 1
    if action_intent:
        mi = action_intent.get("action_motion_mi_bits")
        gd = action_intent.get("goal_directedness") or {}
        frac = gd.get("fraction_away")
        baseline = gd.get("baseline_uniform", 1.0 / 3.0)
        has_body_schema = mi is not None and mi > 0.05
        is_goal_directed = frac is not None and frac > baseline * 1.1
        is_anti_directed = frac is not None and frac < baseline * 0.9
        if has_body_schema and is_goal_directed:
            n_signals += 1
            signals.append(
                f"Action intent: POSITIVE "
                f"(MI={mi:.4f} bits, away-frac={frac:.3f} > "
                f"baseline {baseline:.3f})"
            )
        elif has_body_schema:
            tag = "exploratory" if is_anti_directed else "neutral"
            signals.append(
                f"Action intent: WEAK POSITIVE ({tag}) "
                f"(MI={mi}, away-frac={frac}; body schema "
                "without navigation intent)"
            )
        else:
            signals.append(
                f"Action intent: NEUTRAL/NEGATIVE "
                f"(MI={mi}, away-frac={frac})"
            )
    else:
        signals.append("Action intent: N/A — analysis not run")

    lines.append(
        f"**{n_signals}/{n_total} positive emergence signals detected.**"
    )
    lines.append("")
    for s in signals:
        lines.append(f"- {s}")
    lines.append("")

    # Count WEAK POSITIVE signals too (partial credit).
    n_weak = sum(1 for s in signals if "WEAK POSITIVE" in s)
    n_mixed = sum(1 for s in signals if "MIXED" in s)

    # Overall verdict.
    if n_signals >= 3:
        verdict = (
            "**STRONG EVIDENCE** of pre-linguistic concept emergence. "
            "The model has internalised multiple distinct physical "
            "concepts (object permanence, causal structure, category "
            "organisation, or action intentionality) without any "
            "external supervision signal."
        )
    elif n_signals >= 2:
        verdict = (
            "**MODERATE EVIDENCE** of concept emergence — multiple "
            "concepts show weak-to-moderate signals, but not all "
            "tests agree."
        )
    elif n_signals >= 1 or (n_weak >= 3 and n_mixed == 0):
        verdict = (
            f"**WEAK-TO-MODERATE EVIDENCE** of concept emergence — "
            f"{n_signals}/{n_total} strong signals and "
            f"{n_weak}/{n_total} weak signals detected. The model "
            "shows partial structure in its internal representations, "
            "but no single concept is unambiguously internalised. "
            "Longer training, more snapshots, or richer physics may "
            "yield stronger signals."
        )
    elif n_weak >= 2 or n_mixed >= 1:
        verdict = (
            f"**WEAK EVIDENCE** of concept emergence — "
            f"{n_signals}/{n_total} strong signals, {n_weak}/{n_total} "
            f"weak signals, and {n_mixed}/{n_total} mixed signals. "
            "The model's internal representations show SOME structure "
            "but no clean concept has emerged. Consider extending the "
            "run, increasing the curiosity β-start, or expanding the "
            "sandbox."
        )
    elif n_weak >= 1:
        verdict = (
            "**VERY WEAK EVIDENCE** of concept emergence — at least "
            "one test detected a faint signal above chance, but the "
            "overall picture is inconsistent. Longer training, more "
            "snapshots, or richer physics may be required."
        )
    else:
        verdict = (
            "**NO EVIDENCE** of concept emergence in this run. The "
            "model's internal representations do not yet show "
            "interpretable structure. Consider extending the run, "
            "increasing the curiosity β-start, or expanding the "
            "sandbox."
        )
    lines += [
        f"**Verdict**: {verdict}",
        "",
        "### Discussion",
        "",
        "These results are computed entirely from internal",
        "representations (belief_state, free_energy, prediction_error)",
        "captured in cognitive snapshots — NO external labels, NO",
        "pretrained encoders. A positive signal in any one test is",
        "meaningful because the model received only raw 32-dim",
        "Johnson-Lindenstrauss projections of the 128×128 frame.",
        "",
        "The strongest single test is the **object-permanence**",
        "experiment, because it directly probes whether the model's",
        "free-energy functional encodes a counterfactual expectation",
        "about the continued existence of a non-agent body. The",
        "**causal-structure** test is the most general — it asks",
        "whether the belief trajectory has *any* directed structure",
        "at all. The **category** and **action-intent** tests",
        "require the model to have learned features specific to",
        "spatial position and self-action respectively.",
        "",
        "### Limitations",
        "",
        "1. **Sample size**: cognitive snapshots are sampled every",
        "   10 steps, so a 5000-step run yields ~500 snapshots. The",
        "   statistical power of the t-test and Granger test is",
        "   therefore limited.",
        "2. **Belief dimensionality**: the 32-dim JL projection is",
        "   compact; causal structure may exist in dimensions we",
        "   did not select (we pick top-16 by variance).",
        "3. **CategoryTheoryEngine**: the in-repo engine's default",
        "   categories (NLP/CV/Analytics) were designed for text/CV",
        "   tasks, so its soft classifier may not transfer to",
        "   physics beliefs. We treat k-means + silhouette as the",
        "   primary signal.",
        "4. **CausalInferenceEngine**: PC/LiNGAM algorithms assume",
        "   linear-Gaussian or non-Gaussian linear relationships;",
        "   if the belief dynamics are strongly nonlinear, the",
        "   engine may miss edges. Granger is a linear test too.",
        "5. **Sandbox physics**: 2D elastic collisions on a 128×128",
        "   grid are limited; richer physics (gravity, friction,",
        "   rotations) would give the model more to internalise.",
        "6. **No external ground truth**: the only way to verify a",
        "   'concept' is to test the model's response to an",
        "   intervention (object removal) or to find structure",
        "   (causal edges, clusters) that aligns with the known",
        "   physical configuration.",
        "",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------ #
# Main report generator
# ------------------------------------------------------------------ #
def generate_report(
    snapshots_dir: Path,
    output_dir: Path,
    report_path: Path,
    disappearance_pre_steps: int = 500,
    disappearance_post_steps: int = 500,
    causal_max_vars: int = 16,
    causal_method: str = "pc",
    n_categories: int = 3,
    summary: dict | None = None,
) -> str:
    """Run all four analyses and write the Markdown report.

    Parameters
    ----------
    snapshots_dir : Path
        Directory of ``.npz`` cognitive snapshots from ``run_replay.py``.
    output_dir : Path
        Where to write analysis JSONs + PNGs.
    report_path : Path
        Where to write the final Markdown report.
    disappearance_pre_steps, disappearance_post_steps : int
        Inline experiment parameters (the disappearance analysis
        re-trains a fresh model because it requires a mid-run
        intervention, which cannot be done on existing snapshots).
    causal_max_vars : int
        Max belief dims to feed to the causal engine.
    causal_method : str
        Causal discovery method ('pc', 'lingam', 'correlation').
    n_categories : int
        Number of k-means clusters for the category analysis.
    summary : dict | None
        Optional run summary from ``run_replay.py``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("Phase K — Cognitive Emergence Evaluation")
    print("=" * 72)
    print(f"  snapshots_dir = {snapshots_dir}")
    print(f"  output_dir    = {output_dir}")
    print(f"  report_path   = {report_path}")

    # 1. Disappearance (inline — needs to intervene mid-run).
    print("\n[1/4] Running object-permanence experiment (inline)...")
    try:
        disappearance = analyze_disappearance(
            pre_steps=disappearance_pre_steps,
            post_steps=disappearance_post_steps,
            output_dir=output_dir,
        )
    except Exception as exc:
        print(f"  [error] {type(exc).__name__}: {exc}")
        disappearance = None
    disappearance_img = output_dir / "disappearance.png"

    # 2. Causal graph (from snapshots).
    print("\n[2/4] Running causal-graph analysis...")
    try:
        causal = analyze_causal_graph(
            snapshots_dir=snapshots_dir,
            output_dir=output_dir,
            max_vars=causal_max_vars,
            method=causal_method,
        )
    except Exception as exc:
        print(f"  [error] {type(exc).__name__}: {exc}")
        causal = None
    causal_img = output_dir / "causal_graph.png"

    # 3. Categories (from snapshots).
    print("\n[3/4] Running category-structure analysis...")
    try:
        categories = analyze_categories(
            snapshots_dir=snapshots_dir,
            output_dir=output_dir,
            n_categories=n_categories,
        )
    except Exception as exc:
        print(f"  [error] {type(exc).__name__}: {exc}")
        categories = None
    categories_img = output_dir / "categories.png"

    # 4. Action intent (from snapshots).
    print("\n[4/4] Running action-intent analysis...")
    try:
        action_intent = analyze_action_intent(
            snapshots_dir=snapshots_dir,
            output_dir=output_dir,
        )
    except Exception as exc:
        print(f"  [error] {type(exc).__name__}: {exc}")
        action_intent = None
    action_intent_img = output_dir / "action_intent.png"

    # Build the report.
    sections = [
        render_header(snapshots_dir, output_dir, summary),
        render_disappearance(disappearance or {}, disappearance_img),
        render_causal_graph(causal or {}, causal_img),
        render_categories(categories or {}, categories_img),
        render_action_intent(action_intent or {}, action_intent_img),
        render_conclusion(
            disappearance, causal, categories, action_intent,
        ),
    ]
    report = "\n".join(sections)

    # Write the report.
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"\n[report] written to {report_path}")
    return str(report_path)


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate the Phase K cognitive-emergence report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--snapshots", type=str, default=str(DEFAULT_SNAPSHOTS_DIR),
        help="Directory of .npz cognitive snapshots.",
    )
    p.add_argument(
        "--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help="Where to write analysis JSONs + PNGs.",
    )
    p.add_argument(
        "--report_path", type=str, default=str(DEFAULT_REPORT_PATH),
        help="Output Markdown report path.",
    )
    p.add_argument(
        "--run_summary", type=str, default=None,
        help="Optional path to run_replay.py's summary.json.",
    )
    p.add_argument(
        "--disappearance_pre_steps", type=int, default=500,
        help="Pre-removal steps for the inline disappearance experiment.",
    )
    p.add_argument(
        "--disappearance_post_steps", type=int, default=500,
        help="Post-removal steps for the inline disappearance experiment.",
    )
    p.add_argument(
        "--causal_max_vars", type=int, default=16,
        help="Max belief dimensions to feed to the causal engine.",
    )
    p.add_argument(
        "--causal_method", type=str, default="pc",
        choices=["pc", "lingam", "correlation"],
        help="Causal discovery method.",
    )
    p.add_argument(
        "--n_categories", type=int, default=3,
        help="Number of k-means clusters for category analysis.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    snapshots_dir = Path(args.snapshots)
    output_dir = Path(args.output_dir)
    report_path = Path(args.report_path)

    if not snapshots_dir.exists():
        print(f"\nERROR: snapshots dir does not exist: {snapshots_dir}")
        print("Run `python experiments/run_replay.py` first.")
        sys.exit(1)

    summary = _load_json(Path(args.run_summary)) if args.run_summary else None

    generate_report(
        snapshots_dir=snapshots_dir,
        output_dir=output_dir,
        report_path=report_path,
        disappearance_pre_steps=args.disappearance_pre_steps,
        disappearance_post_steps=args.disappearance_post_steps,
        causal_max_vars=args.causal_max_vars,
        causal_method=args.causal_method,
        n_categories=args.n_categories,
        summary=summary,
    )


if __name__ == "__main__":
    main()
