# experiments/analyze_word_boundaries.py
"""Word-boundary detection analysis (Phase I, Task 2.2).

Hypothesis: if the model has internalised character-transition
statistics, its prediction error should peak at TRUE word boundaries
(positions where the next character is statistically hard to predict
given the recent context).

Method:
  1. Load all snapshots from ``experiments/output/text_replay_run/``.
  2. For each snapshot, extract:
       - the raw ``text_block`` (the chars the model read this step)
       - the ``prediction_error`` recorded in ``meta``
  3. Independently compute, on the FULL text the model read:
       - bigram transition probabilities
       - per-position "bigram entropy" H(c_{t+1} | c_t) — high at
         word-internal positions where multiple chars follow, LOW at
         word boundaries where a single delimiter (space) is the
         only likely continuation.
       - We INVERT this: positions where the bigram entropy is HIGH
         are "non-boundary" (predictable interior) and positions
         where the entropy is LOW (space after content) ARE the
         boundaries. We use a different proxy: positions where the
         bigram count is RARE (low transition probability) are the
         hard-to-predict "boundary-like" positions. The model's PE
         should be HIGHER at these positions.
  4. Build two groups of snapshots: those whose text_block contains
     a rare-bigram position vs those that don't. Compare the mean
     prediction_error of the two groups with a t-test.
  5. Save: JSON summary, CSV (per-snapshot PE + boundary indicator),
     PNG (PE over time coloured by boundary status).

Limitations:
  - The model's prediction_error is a SINGLE scalar per text_block
    (block_size=128 chars). We can't localise the error to a specific
    char within the block. So we treat the whole block as "boundary-
    containing" if ANY of its char positions is a rare-bigram position.
  - The bigram stats are computed on the OBSERVED text only (the
    blocks the model actually visited), not on the full corpus —
    so the rare/common split is relative to what the model saw.

Run with:  python experiments/analyze_word_boundaries.py
           python experiments/analyze_word_boundaries.py --input_dir other/dir
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
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
    from scipy import stats as _scipy_stats
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False


DEFAULT_INPUT_DIR = Path(__file__).resolve().parent / "output" / "text_replay_run"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR  # write alongside snapshots


# ------------------------------------------------------------------ #
# Bigram statistics
# ------------------------------------------------------------------ #
def compute_bigram_stats(text: str) -> dict[str, int]:
    """Return a Counter of bigram counts over ``text``."""
    return Counter(text[i:i + 2] for i in range(len(text) - 1))


def find_rare_bigram_positions(text: str, percentile: float = 25.0) -> set[int]:
    """Find character positions where the bigram starting at that
    position is in the bottom ``percentile``% by count.

    These are the "hard-to-predict" positions — the model should have
    higher prediction error on blocks containing them.
    """
    if len(text) < 3:
        return set()
    bigrams = compute_bigram_stats(text)
    counts = np.array(list(bigrams.values()), dtype=float)
    if len(counts) == 0:
        return set()
    threshold = float(np.percentile(counts, percentile))
    rare_positions = set()
    for i in range(len(text) - 1):
        bg = text[i:i + 2]
        if bigrams[bg] <= threshold:
            rare_positions.add(i)
    return rare_positions


def find_space_after_content_positions(text: str) -> set[int]:
    """Find positions where a space follows a non-space character
    (the canonical word-boundary signal).

    These are the most common word boundaries in English text. The
    model is expected to have higher PE at blocks that contain these.
    """
    positions = set()
    for i in range(len(text) - 1):
        if text[i] != " " and text[i + 1] == " ":
            positions.add(i)
    return positions


# ------------------------------------------------------------------ #
# Main analysis
# ------------------------------------------------------------------ #
def analyze_snapshots(input_dir: Path) -> dict:
    """Load all snapshots, compute PE vs boundary indicator, return summary."""
    snapshot_paths = sorted(input_dir.glob("snapshot_step*.pkl"))
    if not snapshot_paths:
        return {"error": f"no snapshots found in {input_dir}"}

    # First pass: collect all text blocks (concatenated) so the
    # bigram stats reflect the model's actual reading history.
    all_text_parts = []
    snapshots = []
    for p in snapshot_paths:
        snap = load_snapshot(p)
        all_text_parts.append(snap["text_block"])
        snapshots.append(snap)
    full_text = "".join(all_text_parts)

    # Compute rare-bigram positions on the FULL text.
    rare_positions = find_rare_bigram_positions(full_text, percentile=25.0)
    space_positions = find_space_after_content_positions(full_text)

    # Second pass: for each snapshot, check whether its text_block
    # contains any rare-bigram position (mapped into the full_text
    # coordinate system by step-order). We compute a simpler proxy:
    # for each snapshot, check whether its text_block has above-median
    # rare-bigram density.
    # text_block rare-bigram density = fraction of positions in
    # text_block that are in rare_positions (using the block-local
    # bigram stats — easier and equally valid).
    records = []
    for snap in snapshots:
        block = snap["text_block"]
        pe = float(snap["meta"]["prediction_error"])
        step = int(snap["meta"]["step"])
        pos = int(snap["meta"]["pos"])
        fe = float(snap["meta"]["free_energy"])
        # Block-local rare-bigram density.
        block_rare = find_rare_bigram_positions(block, percentile=25.0)
        block_space = find_space_after_content_positions(block)
        n_rare = len(block_rare)
        n_space = len(block_space)
        density_rare = n_rare / max(1, len(block) - 1)
        density_space = n_space / max(1, len(block) - 1)
        records.append({
            "step": step,
            "pos": pos,
            "free_energy": fe,
            "prediction_error": pe,
            "n_rare_bigrams": n_rare,
            "n_space_boundaries": n_space,
            "density_rare": density_rare,
            "density_space": density_space,
        })

    # Split into "high rare-bigram density" vs "low" using the median.
    densities = np.array([r["density_rare"] for r in records])
    if len(densities) == 0:
        return {"error": "no records", "n_snapshots": 0}
    median_density = float(np.median(densities))
    high_group = [r for r in records if r["density_rare"] > median_density]
    low_group = [r for r in records if r["density_rare"] <= median_density]

    pes_high = np.array([r["prediction_error"] for r in high_group])
    pes_low = np.array([r["prediction_error"] for r in low_group])

    summary = {
        "n_snapshots": len(records),
        "full_text_length": len(full_text),
        "n_rare_positions_in_full_text": len(rare_positions),
        "n_space_positions_in_full_text": len(space_positions),
        "median_rare_density": median_density,
        "n_high_group": len(high_group),
        "n_low_group": len(low_group),
        "mean_pe_high_rare_density": float(np.mean(pes_high)) if len(pes_high) else None,
        "mean_pe_low_rare_density": float(np.mean(pes_low)) if len(pes_low) else None,
        "std_pe_high_rare_density": float(np.std(pes_high)) if len(pes_high) else None,
        "std_pe_low_rare_density": float(np.std(pes_low)) if len(pes_low) else None,
    }

    # t-test: is the PE difference between the two groups significant?
    if len(pes_high) >= 2 and len(pes_low) >= 2:
        if _HAS_SCIPY:
            t_stat, p_value = _scipy_stats.ttest_ind(pes_high, pes_low, equal_var=False)
            summary["t_stat"] = float(t_stat)
            summary["p_value"] = float(p_value)
            summary["significant_at_0.05"] = bool(p_value < 0.05)
        else:
            # Fallback: simple z-test on the difference of means.
            pooled_std = float(np.sqrt(
                (np.var(pes_high, ddof=1) / len(pes_high)) +
                (np.var(pes_low, ddof=1) / len(pes_low))
            ))
            if pooled_std > 0:
                z = (float(np.mean(pes_high)) - float(np.mean(pes_low))) / pooled_std
                # Approximate two-sided p-value via the normal CDF.
                from math import erf
                p = 2 * (1 - 0.5 * (1 + erf(abs(z) / 1.4142135623730951)))
                summary["z_stat"] = float(z)
                summary["p_value_approx"] = float(p)
                summary["significant_at_0.05"] = bool(p < 0.05)
    summary["records"] = records
    return summary


def save_outputs(summary: dict, output_dir: Path) -> dict[str, str]:
    """Save CSV + PNG + JSON. Returns dict of saved paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    # CSV: per-snapshot records.
    csv_path = output_dir / "word_boundaries.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "step", "pos", "free_energy", "prediction_error",
            "n_rare_bigrams", "n_space_boundaries",
            "density_rare", "density_space",
        ])
        for r in summary.get("records", []):
            w.writerow([
                r["step"], r["pos"],
                f"{r['free_energy']:.6f}",
                f"{r['prediction_error']:.6f}",
                r["n_rare_bigrams"], r["n_space_boundaries"],
                f"{r['density_rare']:.6f}",
                f"{r['density_space']:.6f}",
            ])
    paths["csv"] = str(csv_path)

    # JSON summary (without records — they're in CSV).
    json_path = output_dir / "word_boundaries_summary.json"
    summary_no_records = {k: v for k, v in summary.items() if k != "records"}
    with json_path.open("w") as f:
        json.dump(summary_no_records, f, indent=2, default=str)
    paths["json"] = str(json_path)

    # PNG: PE over time, coloured by rare-bigram-density group.
    if _HAS_MPL and summary.get("records"):
        png_path = output_dir / "word_boundaries.png"
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        steps = [r["step"] for r in summary["records"]]
        pes = [r["prediction_error"] for r in summary["records"]]
        densities = [r["density_rare"] for r in summary["records"]]
        median_d = summary["median_rare_density"]
        # Top: PE over time, colour by density.
        colors = ["#d62728" if d > median_d else "#1f77b4" for d in densities]
        axes[0].scatter(steps, pes, c=colors, s=20, alpha=0.7)
        axes[0].set_xlabel("step")
        axes[0].set_ylabel("prediction error")
        axes[0].set_title(
            f"PE over time (red = high rare-bigram density, "
            f"blue = low). Mean PE: red={summary.get('mean_pe_high_rare_density')}, "
            f"blue={summary.get('mean_pe_low_rare_density')}"
        )
        axes[0].grid(True, alpha=0.3)
        # Bottom: rare-bigram density over time.
        axes[1].plot(steps, densities, "-", color="#9467bd", linewidth=1)
        axes[1].axhline(median_d, color="black", linestyle="--", alpha=0.5,
                       label=f"median = {median_d:.3f}")
        axes[1].set_xlabel("step")
        axes[1].set_ylabel("rare-bigram density")
        axes[1].set_title("Rare-bigram density per snapshot")
        axes[1].legend(loc="best")
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
        description="Analyze word-boundary detection in model snapshots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input_dir", type=str, default=str(DEFAULT_INPUT_DIR),
                   help="Directory containing snapshot_step*.pkl files.")
    p.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                   help="Directory for output files.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    print("=" * 64)
    print("Word-Boundary Detection Analysis")
    print("=" * 64)
    print(f"  input_dir  = {input_dir}")
    print(f"  output_dir = {output_dir}")

    summary = analyze_snapshots(input_dir)
    if "error" in summary:
        print(f"\nERROR: {summary['error']}")
        return

    paths = save_outputs(summary, output_dir)
    print("\nResults:")
    print(f"  n_snapshots                = {summary['n_snapshots']}")
    print(f"  full_text_length           = {summary['full_text_length']}")
    print(f"  n_rare_positions           = {summary['n_rare_positions_in_full_text']}")
    print(f"  n_space_positions          = {summary['n_space_positions_in_full_text']}")
    print(f"  median rare density        = {summary['median_rare_density']:.4f}")
    print(f"  n_high_group               = {summary['n_high_group']}")
    print(f"  n_low_group                = {summary['n_low_group']}")
    print(f"  mean PE (high rare density)= {summary['mean_pe_high_rare_density']}")
    print(f"  mean PE (low rare density) = {summary['mean_pe_low_rare_density']}")
    if "p_value" in summary:
        print(f"  t-stat                     = {summary.get('t_stat')}")
        print(f"  p-value                    = {summary['p_value']}")
        print(f"  significant @ 0.05         = {summary['significant_at_0.05']}")
    elif "p_value_approx" in summary:
        print(f"  z-stat                     = {summary.get('z_stat')}")
        print(f"  p-value (approx)           = {summary['p_value_approx']}")
        print(f"  significant @ 0.05         = {summary['significant_at_0.05']}")

    print(f"\nOutputs:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    print("=" * 64)


if __name__ == "__main__":
    main()
