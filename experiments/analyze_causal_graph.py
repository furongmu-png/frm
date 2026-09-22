# experiments/analyze_causal_graph.py
"""Discover causal structure in the model's belief-state trajectory.

Loads cognitive snapshots (``.npz`` from ``run_replay.py``), stacks
the per-step belief vectors into a ``(n_steps, n_dim)`` array, and
runs the in-repo ``CausalInferenceEngine.discover()`` (PC / LiNGAM /
correlation fallback) on the resulting time series.

If the engine finds non-trivial directed edges (e.g. a "position"
latent dim → "prediction error" dim), that is evidence the model's
internal representation has organised itself into a causal graph
mirroring the physical world.

Falls back to Granger causality + mutual information if the engine
is unavailable.

Outputs:
  - experiments/output/eval/causal_graph.json    (edges + stats)
  - experiments/output/eval/causal_graph.png      (adjacency heatmap)

Run with:
    python experiments/analyze_causal_graph.py --snapshots output/replay_with/snapshots
    python experiments/analyze_causal_graph.py --snapshots output/replay_with/snapshots --max_vars 16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "eval"


# ------------------------------------------------------------------ #
# Snapshot loading
# ------------------------------------------------------------------ #
def load_belief_trajectory(snapshots_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (steps, beliefs) from .npz snapshots."""
    files = sorted(snapshots_dir.glob("snapshot_*.npz"))
    if not files:
        print(f"[error] no snapshots in {snapshots_dir}")
        sys.exit(1)
    steps: list[int] = []
    beliefs: list[np.ndarray] = []
    for fp in files:
        with np.load(fp, allow_pickle=True) as data:
            steps.append(int(data["step"]))
            beliefs.append(np.asarray(data["belief"], dtype=float))
    steps_arr = np.array(steps)
    beliefs_arr = np.array(beliefs)
    # Sort by step.
    order = np.argsort(steps_arr)
    return steps_arr[order], beliefs_arr[order]


def load_full_trajectory(snapshots_dir: Path) -> dict:
    """Load steps + belief + obs + free_energy + prediction_error."""
    files = sorted(snapshots_dir.glob("snapshot_*.npz"))
    steps, beliefs, obss, fes, pes = [], [], [], [], []
    for fp in files:
        with np.load(fp, allow_pickle=True) as data:
            steps.append(int(data["step"]))
            beliefs.append(np.asarray(data["belief"], dtype=float))
            obss.append(np.asarray(data["obs"], dtype=float))
            fes.append(float(data["free_energy"]))
            pes.append(float(data["prediction_error"]))
    order = np.argsort(np.array(steps))
    return {
        "steps": np.array(steps)[order],
        "beliefs": np.array(beliefs)[order],
        "obs": np.array(obss)[order],
        "free_energies": np.array(fes)[order],
        "prediction_errors": np.array(pes)[order],
    }


# ------------------------------------------------------------------ #
# Causal discovery
# ------------------------------------------------------------------ #
def discover_with_engine(
    data: np.ndarray, var_names: list[str], method: str = "pc"
) -> dict:
    """Use the in-repo ``CausalInferenceEngine``.

    Returns dict with ``adjacency``, ``edges``, ``n_edges``, ``method``,
    ``var_names``, ``is_acyclic``.
    """
    from zero_data_model.causal_emergence.causal_discovery import (
        CausalInferenceEngine,
    )
    engine = CausalInferenceEngine(dim=data.shape[1])
    return engine.discover(data, var_names=var_names, method=method)


def granger_causality_matrix(data: np.ndarray, max_lag: int = 3) -> np.ndarray:
    """Fallback: pairwise Granger causality.

    For each pair (i, j), compute the reduction in residual sum of
    squares when adding lagged values of series i to an AR(max_lag)
    model of series j. Returns a (n_vars, n_vars) matrix where entry
    (i, j) is the Granger causality statistic of i → j.
    """
    n_samples, n_vars = data.shape
    gc_matrix = np.zeros((n_vars, n_vars))
    for j in range(n_vars):
        y = data[max_lag:, j]
        # AR(max_lag) baseline model: y_t = const + sum_lag a_l y_{t-l}
        X_base = np.column_stack(
            [data[max_lag - lag - 1: -lag - 1, j] for lag in range(max_lag)]
        ) if max_lag > 0 else np.ones((len(y), 1))
        X_base = np.column_stack([np.ones(len(y)), X_base])
        # Least squares for restricted model.
        coef_r, _, _, _ = np.linalg.lstsq(X_base, y, rcond=None)
        resid_r = y - X_base @ coef_r
        rss_r = float(np.sum(resid_r ** 2))
        for i in range(n_vars):
            if i == j:
                continue
            # Add lagged series i.
            X_full = np.column_stack([
                X_base,
                np.array([data[max_lag - lag - 1: -lag - 1, i]
                          for lag in range(max_lag)]).T
            ]) if max_lag > 0 else X_base
            coef_f, _, _, _ = np.linalg.lstsq(X_full, y, rcond=None)
            resid_f = y - X_full @ coef_f
            rss_f = float(np.sum(resid_f ** 2))
            # F-statistic.
            if rss_f > 0 and rss_r > rss_f:
                df_num = max_lag
                df_den = len(y) - X_full.shape[1]
                if df_den > 0 and rss_f > 0:
                    f_stat = ((rss_r - rss_f) / df_num) / (rss_f / df_den)
                    gc_matrix[i, j] = float(f_stat)
    return gc_matrix


def mutual_information_matrix(data: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """Pairwise mutual information (fallback)."""
    n_samples, n_vars = data.shape
    mi_matrix = np.zeros((n_vars, n_vars))
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            # Bin both series.
            xi = np.digitize(data[:, i], np.linspace(
                data[:, i].min(), data[:, i].max(), n_bins))
            xj = np.digitize(data[:, j], np.linspace(
                data[:, j].min(), data[:, j].max(), n_bins))
            # Joint histogram.
            joint = np.histogram2d(xi, xj, bins=n_bins)[0]
            joint /= joint.sum() + 1e-12
            pi = joint.sum(axis=1, keepdims=True)
            pj = joint.sum(axis=0, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                mi = np.sum(joint * np.log2(
                    joint / (pi @ pj + 1e-12) + 1e-12
                ))
            mi_matrix[i, j] = float(mi)
            mi_matrix[j, i] = float(mi)
    return mi_matrix


# ------------------------------------------------------------------ #
# Main analysis
# ------------------------------------------------------------------ #
def run_analysis(
    snapshots_dir: Path,
    output_dir: Path = OUTPUT_DIR,
    max_vars: int = 16,
    method: str = "pc",
) -> dict:
    """Run causal discovery on belief trajectories."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[causal] loading snapshots from {snapshots_dir}")
    data = load_full_trajectory(snapshots_dir)
    beliefs = data["beliefs"]
    n_steps, n_dim = beliefs.shape
    print(f"[causal] {n_steps} snapshots × {n_dim} belief dims")
    # Variable selection: pick the top ``max_vars`` dimensions by
    # variance — the most "active" latent variables are the most
    # likely to carry causal signal.
    if n_dim > max_vars:
        variances = np.var(beliefs, axis=0)
        top_idx = np.argsort(variances)[::-1][:max_vars]
        top_idx = np.sort(top_idx)  # keep order
        selected = beliefs[:, top_idx]
        var_names = [f"belief_{i}" for i in top_idx]
        print(f"[causal] selected top {max_vars} belief dims by variance")
    else:
        selected = beliefs
        top_idx = np.arange(n_dim)
        var_names = [f"belief_{i}" for i in range(n_dim)]
    # Augment with the scalar quantities (FE, prediction_error) so we
    # can test "belief → prediction_error" causal edges.
    augmented = np.column_stack([
        selected,
        data["free_energies"].reshape(-1, 1),
        data["prediction_errors"].reshape(-1, 1),
    ])
    aug_names = var_names + ["free_energy", "pred_error"]
    print(f"[causal] augmented data shape: {augmented.shape}")
    # Try the in-repo engine first.
    engine_result: dict | None = None
    engine_error: str | None = None
    try:
        engine_result = discover_with_engine(
            augmented, var_names=aug_names, method=method,
        )
        print(f"[causal] engine method: {engine_result.get('method')}")
        print(f"[causal] engine edges: {engine_result.get('n_edges')}")
    except Exception as exc:
        engine_error = f"{type(exc).__name__}: {exc}"
        print(f"[causal] engine failed: {engine_error}")
    # Fallback: Granger + MI.
    print("[causal] computing Granger causality fallback...")
    gc_matrix = granger_causality_matrix(augmented, max_lag=3)
    print("[causal] computing mutual information fallback...")
    mi_matrix = mutual_information_matrix(augmented, n_bins=8)
    # Build the result payload.
    result = {
        "n_snapshots": int(n_steps),
        "n_belief_dims": int(n_dim),
        "n_selected_vars": int(len(var_names)),
        "max_vars": int(max_vars),
        "selected_belief_dims": top_idx.tolist(),
        "var_names": aug_names,
        "engine_method": (engine_result.get("method") if engine_result
                          else None),
        "engine_n_edges": (engine_result.get("n_edges") if engine_result
                           else 0),
        "engine_is_acyclic": (engine_result.get("is_acyclic") if engine_result
                              else None),
        "engine_error": engine_error,
        "gc_matrix_shape": list(gc_matrix.shape),
        "gc_matrix_max": float(gc_matrix.max()),
        "gc_matrix_mean": float(gc_matrix.mean()),
        "mi_matrix_max": float(mi_matrix.max()),
        "mi_matrix_mean": float(mi_matrix.mean()),
    }
    # Extract top-edges from the engine (if available).
    if engine_result:
        edges = engine_result.get("edges", [])
        # Filter to edges involving FE or pred_error (most interpretable).
        interesting_edges = [
            e for e in edges
            if isinstance(e, dict)
            and ("free_energy" in str(e.get("source", ""))
                 or "free_energy" in str(e.get("target", ""))
                 or "pred_error" in str(e.get("source", ""))
                 or "pred_error" in str(e.get("target", "")))
        ]
        result["engine_edges_total"] = int(len(edges))
        result["engine_interesting_edges"] = interesting_edges[:20]
    # Top Granger edges.
    gc_top = []
    n_aug = gc_matrix.shape[0]
    for i in range(n_aug):
        for j in range(n_aug):
            if i != j and gc_matrix[i, j] > 1.0:  # F > 1
                gc_top.append({
                    "source": aug_names[i],
                    "target": aug_names[j],
                    "f_stat": float(gc_matrix[i, j]),
                })
    gc_top.sort(key=lambda e: e["f_stat"], reverse=True)
    result["gc_top_edges"] = gc_top[:15]
    result["gc_n_significant"] = int(len(gc_top))
    # Save.
    with (output_dir / "causal_graph.json").open("w") as f:
        json.dump(result, f, indent=2)
    # Save matrices.
    np.savez_compressed(
        output_dir / "causal_matrices.npz",
        gc=gc_matrix, mi=mi_matrix,
        var_names=np.array(aug_names, dtype=object),
    )
    # Plot.
    if _HAS_MPL:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # Cap the colour range for visibility.
        gc_cap = np.percentile(gc_matrix[gc_matrix > 0], 95) if (gc_matrix > 0).any() else 1.0
        mi_cap = np.percentile(mi_matrix[mi_matrix > 0], 95) if (mi_matrix > 0).any() else 1.0
        im0 = axes[0].imshow(np.clip(gc_matrix, 0, gc_cap), cmap="viridis",
                              aspect="auto")
        axes[0].set_title("Granger causality (capped at 95th pct)")
        axes[0].set_xticks(range(n_aug))
        axes[0].set_yticks(range(n_aug))
        axes[0].set_xticklabels(aug_names, rotation=90, fontsize=7)
        axes[0].set_yticklabels(aug_names, fontsize=7)
        plt.colorbar(im0, ax=axes[0])
        im1 = axes[1].imshow(np.clip(mi_matrix, 0, mi_cap), cmap="magma",
                              aspect="auto")
        axes[1].set_title("Mutual information (capped at 95th pct)")
        axes[1].set_xticks(range(n_aug))
        axes[1].set_yticks(range(n_aug))
        axes[1].set_xticklabels(aug_names, rotation=90, fontsize=7)
        axes[1].set_yticklabels(aug_names, fontsize=7)
        plt.colorbar(im1, ax=axes[1])
        fig.tight_layout()
        png_path = output_dir / "causal_graph.png"
        fig.savefig(png_path, dpi=110)
        plt.close(fig)
        print(f"[plot] saved {png_path}")
    print(f"[result] engine edges: {result['engine_n_edges']}  "
          f"GC significant: {result['gc_n_significant']}  "
          f"MI max: {result['mi_matrix_max']:.3f}")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshots", type=str, required=True,
                   help="Directory of .npz snapshots")
    p.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    p.add_argument("--max_vars", type=int, default=16)
    p.add_argument("--method", type=str, default="pc",
                   choices=["pc", "lingam", "correlation"])
    args = p.parse_args()
    run_analysis(
        snapshots_dir=Path(args.snapshots),
        output_dir=Path(args.output_dir),
        max_vars=args.max_vars,
        method=args.method,
    )


if __name__ == "__main__":
    main()
