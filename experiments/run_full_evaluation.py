# experiments/run_full_evaluation.py
"""Main orchestrator: run a full cognitive-emergence evaluation.

Pipeline:
  1. Run ``run_sandbox_replay.ReplaySandboxRunner`` for ``max_steps``
     steps, saving snapshots + a CSV log + manifest.
  2. Run each ``analyze_*.py`` script in turn, producing per-analysis
     JSON summaries + PNG figures in ``experiments/output/replay_run/``.
  3. Aggregate all summaries into a final Markdown report at
     ``experiments/output/replay_run/FULL_REPORT.md``.

Defaults (overridable via CLI):
  - max_steps           = 1000   (quick mode; --long for 10000)
  - snapshot_interval   = 50
  - disappear_step      = 500    (must be < max_steps and > window)
  - disappear_object    = 1
  - sleep_interval      = 200
  - sleep_rounds        = 3
  - buffer_capacity     = 2000
  - seed                = 42

The default ``max_steps=1000`` is a SMOKE run that takes ~30s and lets
you sanity-check the pipeline end-to-end. For real evaluation, run
``--long`` which sets ``max_steps=10000, snapshot_interval=50,
disappear_step=5000``.

Run with:
    python experiments/run_full_evaluation.py             # smoke (1000 steps)
    python experiments/run_full_evaluation.py --long      # full  (10000 steps)
    python experiments/run_full_evaluation.py --max-steps 5000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# Re-use the runner + analyzers as importable modules so we don't
# have to spawn subprocesses (faster, easier to debug).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from physics_sandbox import PhysicsSandbox  # noqa: E402
from run_sandbox_curious import make_curious_model  # noqa: E402
from run_sandbox_replay import (  # noqa: E402
    OUTPUT_DIR,
    ReplaySandboxRunner,
)


# ------------------------------------------------------------------ #
# Argument parsing
# ------------------------------------------------------------------ #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a full cognitive-emergence evaluation pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--max-steps", type=int, default=1000,
        help="Total steps to run the closed loop.",
    )
    p.add_argument(
        "--snapshot-interval", type=int, default=50,
        help="Save a snapshot every N steps.",
    )
    p.add_argument(
        "--disappear-step", type=int, default=500,
        help="Step at which to remove an object.",
    )
    p.add_argument(
        "--disappear-object-index", type=int, default=1,
        help="Index of the body to remove (0 = agent).",
    )
    p.add_argument(
        "--sleep-interval", type=int, default=200,
        help="Online steps between sleep phases.",
    )
    p.add_argument(
        "--sleep-rounds", type=int, default=3,
        help="Consolidation batches per sleep phase.",
    )
    p.add_argument(
        "--buffer-capacity", type=int, default=2000,
        help="Experience buffer capacity.",
    )
    p.add_argument("--dim", type=int, default=64, help="Model dimension.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--num-objects", type=int, default=3,
        help="Initial sandbox body count (incl. agent).",
    )
    p.add_argument(
        "--long", action="store_true",
        help="Convenience: set max-steps=10000, snapshot-interval=50, "
             "disappear-step=5000. Overrides the explicit args.",
    )
    p.add_argument(
        "--output-dir", type=str, default=str(OUTPUT_DIR),
        help="Output directory (default: experiments/output/replay_run/).",
    )
    return p.parse_args()


def resolve_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Apply --long overrides and validate arguments."""
    if args.long:
        args.max_steps = 10000
        args.snapshot_interval = 50
        args.disappear_step = 5000
    if args.disappear_step >= args.max_steps:
        raise ValueError(
            f"--disappear-step ({args.disappear_step}) must be < "
            f"--max-steps ({args.max_steps})"
        )
    if args.disappear_object_index == 0:
        raise ValueError("cannot remove the agent (index 0)")
    if args.disappear_object_index >= args.num_objects:
        raise ValueError(
            f"--disappear-object-index ({args.disappear_object_index}) "
            f"must be < --num-objects ({args.num_objects})"
        )
    return args


# ------------------------------------------------------------------ #
# Stage 1: Run the replay loop
# ------------------------------------------------------------------ #
def run_replay(args: argparse.Namespace) -> dict:
    """Run the closed-loop replay with snapshots + disappearance."""
    print("\n" + "=" * 64)
    print("STAGE 1: Closed-loop replay (cognitive snapshot collection)")
    print("=" * 64)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = make_curious_model(
        dim=args.dim, seed=args.seed,
        beta_start=1.0, beta_min=0.01,
        decay_steps=args.max_steps, decay_type="linear",
    )
    sandbox = PhysicsSandbox(num_objects=args.num_objects, seed=args.seed)
    runner = ReplaySandboxRunner(
        model, sandbox,
        max_steps=args.max_steps,
        snapshot_interval=args.snapshot_interval,
        disappear_step=args.disappear_step,
        disappear_object_index=args.disappear_object_index,
        sleep_interval=args.sleep_interval,
        sleep_rounds=args.sleep_rounds,
        sleep_batch_size=8,
        buffer_capacity=args.buffer_capacity,
        output_dir=out_dir,
        seed=args.seed,
    )
    t0 = time.time()
    runner.run()
    elapsed = time.time() - t0
    print(f"\n  loop finished in {elapsed:.1f}s "
          f"({args.max_steps / max(elapsed, 1e-3):.1f} steps/s)")
    csv_path = runner.save_csv(out_dir / "replay_log.csv")

    # Save the manifest.
    manifest = {
        "snapshot_paths": runner.summary.snapshot_paths,
        "snapshot_steps": runner.summary.snapshot_steps,
        "disappear_step": runner.summary.disappear_step,
        "disappear_object_index": runner.summary.disappear_object_index,
        "disappear_free_energy_before": runner.summary.disappear_free_energy_before,
        "disappear_free_energy_after": runner.summary.disappear_free_energy_after,
        "max_steps": args.max_steps,
        "dim": args.dim,
        "seed": args.seed,
        "num_objects": args.num_objects,
        "snapshot_interval": args.snapshot_interval,
        "sleep_interval": args.sleep_interval,
        "sleep_rounds": args.sleep_rounds,
        "elapsed_s": elapsed,
        "csv_path": str(csv_path),
    }
    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  manifest: {manifest_path}")
    print(f"  CSV:      {csv_path}")
    print(f"  snapshots: {len(runner.summary.snapshot_paths)} files in {out_dir}")
    return manifest


# ------------------------------------------------------------------ #
# Stage 2: Run all analysis scripts
# ------------------------------------------------------------------ #
def run_analyses(output_dir: Path) -> dict:
    """Run each analyze_*.py script and return its summary dict.

    The scripts are run as SUBPROCESSES (not imports) so that each
    gets a fresh Python process and a clean module-level state. This
    avoids any side effects from one analyzer mutating another's
    state — important because they all read from the same snapshot
    dir but might create/destroy matplotlib figures.
    """
    print("\n" + "=" * 64)
    print("STAGE 2: Running analysis scripts")
    print("=" * 64)
    analyzers = [
        "analyze_disappearance.py",
        "analyze_causal_graph.py",
        "analyze_categories.py",
        "analyze_action_intent.py",
    ]
    summaries = {}
    for name in analyzers:
        script_path = Path(__file__).resolve().parent / name
        print(f"\n>>> {name}")
        print("-" * 40)
        t0 = time.time()
        # Use the SAME python interpreter so the path setup matches.
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,  # stream to console for live progress
            cwd=str(Path(__file__).resolve().parent.parent),
            env={**__import__("os").environ, "PYTHONPATH": ":".join([
                str(Path(__file__).resolve().parent.parent / "examples"),
                str(Path(__file__).resolve().parent),
                str(Path(__file__).resolve().parent.parent / "src"),
            ])},
        )
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"  [WARN] {name} exited with code {result.returncode} "
                  f"({elapsed:.1f}s) — continuing with other analyzers")
            summaries[name] = {"error": f"exit code {result.returncode}",
                                "elapsed_s": elapsed}
            continue
        print(f"  done in {elapsed:.1f}s")
        # Load the summary JSON if the analyzer wrote one.
        # Convention: analyze_X.py writes X_summary.json (drop "analyze_").
        summary_name = name.replace("analyze_", "").replace(".py", "") + "_summary.json"
        summary_path = output_dir / summary_name
        if summary_path.exists():
            with summary_path.open() as f:
                summaries[name] = json.load(f)
        else:
            summaries[name] = {"note": "no summary written", "elapsed_s": elapsed}
    return summaries


# ------------------------------------------------------------------ #
# Stage 3: Aggregate into a Markdown report
# ------------------------------------------------------------------ #
def write_markdown_report(
    manifest: dict, summaries: dict, output_dir: Path,
) -> Path:
    """Write the final Markdown report aggregating all summaries."""
    print("\n" + "=" * 64)
    print("STAGE 3: Writing final Markdown report")
    print("=" * 64)
    report_path = output_dir / "FULL_REPORT.md"

    disappear_before = manifest.get("disappear_free_energy_before", 0.0)
    disappear_after = manifest.get("disappear_free_energy_after", 0.0)
    fe_delta = disappear_after - disappear_before

    # Extract per-analyzer interpretation strings.
    def _interp(key: str) -> str:
        s = summaries.get(key, {})
        return s.get("interpretation", "_No interpretation available._")

    # Extract disappearance stats.
    disp_summary = summaries.get("analyze_disappearance.py", {})
    disp_fe = disp_summary.get("fe_curve_stats", {}) if isinstance(disp_summary, dict) else {}

    # Extract causal graph stats.
    cg_summary = summaries.get("analyze_causal_graph.py", {})
    cg_snaps = cg_summary.get("snapshot_features", {}) if isinstance(cg_summary, dict) else {}
    cg_csv = cg_summary.get("csv_features", {}) if isinstance(cg_summary, dict) else {}

    # Extract category stats.
    cat_summary = summaries.get("analyze_categories.py", {})
    cat_drift = cat_summary.get("classifier_drift", {}) if isinstance(cat_summary, dict) else {}
    cat_obj = cat_summary.get("object_counts", {}) if isinstance(cat_summary, dict) else {}

    # Extract action intent stats.
    ai_summary = summaries.get("analyze_action_intent.py", {})
    ai_mi = ai_summary.get("mutual_information_nats", "n/a") if isinstance(ai_summary, dict) else "n/a"
    ai_chi2 = ai_summary.get("chi_squared", {}) if isinstance(ai_summary, dict) else {}
    ai_vel = ai_summary.get("velocity_action_correlation", {}) if isinstance(ai_summary, dict) else {}
    ai_bdy = ai_summary.get("boundary_intent", {}) if isinstance(ai_summary, dict) else {}

    # Build the Markdown.
    md = []
    md.append("# Cognitive Emergence Evaluation — Final Report\n")
    md.append(f"_Generated by `run_full_evaluation.py` on "
              f"{time.strftime('%Y-%m-%d %H:%M:%S %Z')}_.\n")
    md.append("\n## 1. Experiment Configuration\n")
    md.append("| Parameter | Value |\n|---|---|")
    md.append(f"| Model dim | {manifest.get('dim', 'n/a')} |")
    md.append(f"| Random seed | {manifest.get('seed', 'n/a')} |")
    md.append(f"| Initial sandbox bodies | {manifest.get('num_objects', 'n/a')} |")
    md.append(f"| Total steps | {manifest.get('max_steps', 'n/a')} |")
    md.append(f"| Snapshot interval | every {manifest.get('snapshot_interval', 'n/a')} steps |")
    md.append(f"| Total snapshots saved | {len(manifest.get('snapshot_paths', []))} |")
    md.append(f"| Disappearance step | {manifest.get('disappear_step', 'n/a')} |")
    md.append(f"| Disappearance object index | {manifest.get('disappear_object_index', 'n/a')} |")
    md.append(f"| Sleep interval | every {manifest.get('sleep_interval', 'n/a')} steps |")
    md.append(f"| Sleep rounds per phase | {manifest.get('sleep_rounds', 'n/a')} |")
    md.append(f"| Total elapsed | {manifest.get('elapsed_s', 0):.1f}s |\n")

    md.append("## 2. Disappearance Intervention (Object Permanence)\n")
    md.append(f"![FE curve](disappearance_fe_curve.png)\n")
    md.append(f"![State drift](disappearance_state_drift.png)\n")
    md.append("| Metric | Value |\n|---|---|")
    md.append(f"| FE baseline (pre-event) | {disp_fe.get('fe_baseline', 'n/a'):.4f} |" if isinstance(disp_fe.get('fe_baseline'), (int, float)) else f"| FE baseline | {disp_fe.get('fe_baseline', 'n/a')} |")
    md.append(f"| FE at event | {disp_fe.get('fe_spike', 'n/a'):.4f} |" if isinstance(disp_fe.get('fe_spike'), (int, float)) else f"| FE at event | {disp_fe.get('fe_spike', 'n/a')} |")
    md.append(f"| FE Δ at event | {disp_fe.get('fe_delta_at_event', 'n/a'):+.4f} |" if isinstance(disp_fe.get('fe_delta_at_event'), (int, float)) else f"| FE Δ at event | {disp_fe.get('fe_delta_at_event', 'n/a')} |")
    md.append(f"| FE recovery slope (per step) | {disp_fe.get('fe_recovery_slope', 'n/a'):+.5f} |" if isinstance(disp_fe.get('fe_recovery_slope'), (int, float)) else f"| FE recovery slope | {disp_fe.get('fe_recovery_slope', 'n/a')} |")
    md.append(f"| Manifest ΔFE (before→after) | {disappear_before:.4f} → {disappear_after:.4f} ({fe_delta:+.4f}) |")
    md.append(f"\n**Interpretation:** {_interp('analyze_disappearance.py')}\n")

    md.append("## 3. Causal Graph (Internal State Time Series)\n")
    md.append(f"![Snapshot DAG](causal_graph_snapshots.png)\n")
    md.append(f"![CSV DAG](causal_graph_csv.png)\n")
    md.append("| Feature set | Variables | Edges discovered | Method | Acyclic? |\n|---|---|---|---|---|")
    md.append(f"| Snapshots | {cg_snaps.get('n_vars', 'n/a')} | {cg_snaps.get('n_edges', 'n/a')} | {cg_snaps.get('method', 'n/a')} | {cg_snaps.get('is_acyclic', 'n/a')} |")
    if cg_csv:
        md.append(f"| CSV (dense) | {cg_csv.get('n_vars', 'n/a')} | {cg_csv.get('n_edges', 'n/a')} | {cg_csv.get('method', 'n/a')} | {cg_csv.get('is_acyclic', 'n/a')} |")
    if cg_snaps.get("edges"):
        md.append("\nSnapshot-feature edges:")
        md.append("| Cause | Effect | Weight |")
        md.append("|---|---|---|")
        for e in cg_snaps["edges"][:20]:  # cap at 20 rows
            md.append(f"| {e['src']} | {e['tgt']} | {e['weight']:+.4f} |")
    md.append(f"\n**Interpretation:** {_interp('analyze_causal_graph.py')}\n")

    md.append("## 4. Category Structure (Topos + Object Counts)\n")
    md.append(f"![Classifier drift](categories_classifier_drift.png)\n")
    md.append(f"![Truth values](categories_truth_values.png)\n")
    md.append(f"![Object counts](categories_object_counts.png)\n")
    md.append("| Metric | Value |\n|---|---|")
    md.append(f"| Mean step-to-step classifier drift | {cat_drift.get('mean_step_drift', 'n/a')} |")
    md.append(f"| Final cumulative classifier drift | {cat_drift.get('final_cumulative_drift', 'n/a')} |")
    md.append(f"| Total category objects (pre-event mean) | {cat_obj.get('total_pre_mean', 'n/a')} |")
    md.append(f"| Total category objects (post-event mean) | {cat_obj.get('total_post_mean', 'n/a')} |")
    md.append(f"| Total category objects (Δ at event) | {cat_obj.get('total_delta_at_disappear', 'n/a')} |")
    md.append(f"\n**Interpretation:** {_interp('analyze_categories.py')}\n")

    md.append("## 5. Action-Effect Intent\n")
    md.append(f"![Action distribution](action_intent_distribution.png)\n")
    md.append("| Metric | Value |\n|---|---|")
    md.append(f"| Mutual information I(action; position) | {ai_mi if isinstance(ai_mi, str) else f'{ai_mi:.5f} nats'} |")
    md.append(f"| Chi-squared statistic | {ai_chi2.get('statistic', 'n/a')} |")
    md.append(f"| Chi-squared p-value | {ai_chi2.get('p_value', 'n/a')} |")
    md.append(f"| Velocity-action correlation r | {ai_vel.get('pearson_r', 'n/a')} |")
    md.append(f"| Boundary intent score | {ai_bdy.get('boundary_intent_score', 'n/a')} |")
    md.append(f"\n**Interpretation:** {_interp('analyze_action_intent.py')}\n")

    md.append("## 6. Overall Conclusions\n")
    md.append(_overall_conclusions(manifest, summaries))
    md.append("\n## 7. Files Produced\n")
    md.append("- `manifest.json` — run configuration + paths.\n")
    md.append("- `replay_log.csv` — dense per-step log.\n")
    md.append("- `snapshot_step*.pkl` — pickled cognitive snapshots.\n")
    md.append("- `disappearance_fe_curve.png` / `disappearance_state_drift.png`\n")
    md.append("- `disappearance_summary.json`\n")
    md.append("- `causal_graph_snapshots.png` / `causal_graph_csv.png`\n")
    md.append("- `causal_graph_summary.json`\n")
    md.append("- `categories_classifier_drift.png` / `categories_truth_values.png` / `categories_object_counts.png`\n")
    md.append("- `categories_summary.json`\n")
    md.append("- `action_intent_distribution.png`\n")
    md.append("- `action_intent_summary.json`\n")
    md.append("- `FULL_REPORT.md` — this file.\n")

    with report_path.open("w") as f:
        f.write("\n".join(md))
    print(f"\nFinal report: {report_path}")
    return report_path


def _overall_conclusions(manifest: dict, summaries: dict) -> str:
    """Synthesise a high-level conclusion paragraph from all findings.

    Decision rules (each signal independently):
      - disappearance: FE spike >= 5% of baseline  => "+1 surprise signal"
      - causal graph:  at least 1 'meaningful' snapshot edge => "+1 causal structure"
      - categories:    total Δ at disappear < 0 OR cumulative drift > 3x mean => "+1 category coupling"
      - action intent:  MI > 0.02 nats OR boundary score > 0.05 => "+1 intent signal"
    Tally: 0-1 signals = "no emergence"; 2 = "weak"; 3 = "moderate"; 4 = "strong".
    """
    signals = []

    # Disappearance.
    disp = summaries.get("analyze_disappearance.py", {})
    fe = disp.get("fe_curve_stats", {}) if isinstance(disp, dict) else {}
    delta = fe.get("fe_delta_at_event")
    baseline = fe.get("fe_baseline")
    if isinstance(delta, (int, float)) and isinstance(baseline, (int, float)) and abs(baseline) > 1e-6:
        ratio = delta / baseline
        if abs(ratio) >= 0.05:
            signals.append(
                f"surprise signal: FE Δ/baseline = {ratio*100:+.1f}% "
                f"(threshold ±5%)"
            )

    # Causal graph.
    cg = summaries.get("analyze_causal_graph.py", {})
    cg_snaps = cg.get("snapshot_features", {}) if isinstance(cg, dict) else {}
    if cg_snaps.get("n_edges", 0) > 0:
        signals.append(
            f"causal structure: {cg_snaps.get('n_edges', 0)} edges in "
            f"snapshot-feature DAG"
        )

    # Categories.
    cat = summaries.get("analyze_categories.py", {})
    if isinstance(cat, dict):
        obj_delta = cat.get("object_counts", {}).get("total_delta_at_disappear", 0)
        drift_final = cat.get("classifier_drift", {}).get(
            "final_cumulative_drift", 0.0
        )
        drift_mean = cat.get("classifier_drift", {}).get(
            "mean_step_drift", 0.0
        )
        if obj_delta < 0:
            signals.append(
                f"category coupling: object counts decreased by {-obj_delta} "
                f"at disappearance"
            )
        elif drift_mean > 0 and drift_final > 3 * drift_mean:
            signals.append(
                f"category learning: cumulative classifier drift "
                f"({drift_final:.4f}) > 3x mean step drift "
                f"({drift_mean:.4f})"
            )

    # Action intent.
    ai = summaries.get("analyze_action_intent.py", {})
    if isinstance(ai, dict):
        mi = ai.get("mutual_information_nats", 0.0)
        bd = ai.get("boundary_intent", {}).get("boundary_intent_score", 0.0)
        if isinstance(mi, (int, float)) and mi > 0.02:
            signals.append(
                f"action intent: I(action; pos) = {mi:.4f} nats "
                f"(>0.02 threshold)"
            )
        elif isinstance(bd, (int, float)) and bd > 0.05:
            signals.append(
                f"action intent: boundary score = {bd:+.4f} (>0.05 threshold)"
            )

    n = len(signals)
    if n == 0:
        verdict = "**No emergence signals detected.** The model has not yet "
        "formed measurable internal representations of physical concepts "
        "in this run. Run longer (--long) or with a smaller dim for a "
        "stronger learning signal."
    elif n == 1:
        verdict = (
            "**Weak emergence signal (1/4).** One measurable "
            "signature found. Inconclusive — needs corroborating signals."
        )
    elif n == 2:
        verdict = (
            "**Moderate emergence signal (2/4).** Two independent "
            "signatures found — tentative evidence of pre-linguistic "
            "physical concepts."
        )
    elif n == 3:
        verdict = (
            "**Strong emergence signal (3/4).** Three independent "
            "signatures found — strong evidence of pre-linguistic "
            "physical concepts."
        )
    else:
        verdict = (
            "**Very strong emergence signal (4/4).** All four "
            "evaluation axes show measurable signatures — the model "
            "has demonstrably formed internal representations of "
            "physical objects, causality, categories, and intent."
        )
    bullets = "\n".join(f"  - {s}" for s in signals) if signals else "  - (none)"
    return f"{verdict}\n\nSignals detected:\n{bullets}\n"


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main() -> None:
    args = parse_args()
    args = resolve_defaults(args)
    output_dir = Path(args.output_dir)

    print("=" * 64)
    print("ZeroDataModel — Full Cognitive Emergence Evaluation")
    print("=" * 64)
    print(f"  output_dir  = {output_dir}")
    print(f"  max_steps   = {args.max_steps}")
    print(f"  snapshot_int= {args.snapshot_interval}")
    print(f"  disappear   = step {args.disappear_step}, obj {args.disappear_object_index}")
    print(f"  sleep       = every {args.sleep_interval} steps, "
          f"{args.sleep_rounds} rounds")
    print(f"  dim/seed    = {args.dim}/{args.seed}")

    t_total = time.time()
    manifest = run_replay(args)
    summaries = run_analyses(output_dir)
    report_path = write_markdown_report(manifest, summaries, output_dir)
    t_total = time.time() - t_total

    print("\n" + "=" * 64)
    print(f"DONE. Total time: {t_total:.1f}s")
    print(f"Final report: {report_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
