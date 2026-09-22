# experiments/analyze_categories.py
"""Category-structure analysis: do similar physical configurations
cluster into the same latent category?

Loads cognitive snapshots, extracts per-frame belief vectors, and:

1. Uses ``CategoryTheoryEngine.topos.classify()`` to assign each frame
   to one of the engine's default categories (NLP / CV / Analytics).
2. As a fallback, performs k-means + silhouette analysis on the belief
   vectors directly.
3. Cross-tabulates category assignments with the physical
   configuration of the sandbox at each step (extracted from the
   ``sandbox_state`` field of the snapshot). High purity = the model
   has discovered structure that aligns with physical reality.

Outputs:
  - experiments/output/eval/categories.json   (assignments + stats)
  - experiments/output/eval/categories.png   (PCA scatter, 2 panels)

Run with:
    python experiments/analyze_categories.py --snapshots output/replay_with/snapshots
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
# Loading
# ------------------------------------------------------------------ #
def load_snapshots(snapshots_dir: Path) -> dict:
    """Load belief vectors + sandbox states from snapshots."""
    files = sorted(snapshots_dir.glob("snapshot_*.npz"))
    if not files:
        print(f"[error] no snapshots in {snapshots_dir}")
        sys.exit(1)
    steps, beliefs, sandbox_states, actions = [], [], [], []
    for fp in files:
        with np.load(fp, allow_pickle=True) as data:
            steps.append(int(data["step"]))
            beliefs.append(np.asarray(data["belief"], dtype=float))
            actions.append(int(data["action"]))
            # sandbox_state is stored as a JSON string.
            state_str = str(data["sandbox_state"].item())
            try:
                state = json.loads(state_str)
            except Exception:
                state = {}
            sandbox_states.append(state)
    order = np.argsort(np.array(steps))
    return {
        "steps": np.array(steps)[order],
        "beliefs": np.array(beliefs)[order],
        "actions": np.array(actions)[order],
        "sandbox_states": [sandbox_states[i] for i in order],
    }


# ------------------------------------------------------------------ #
# Physical-feature extraction from sandbox states
# ------------------------------------------------------------------ #
def extract_physical_features(sandbox_states: list[dict]) -> np.ndarray:
    """Extract (n_steps, 4) physical features.

    Features:
    - agent_cx, agent_cy (the first body's centre)
    - mean_distance_between_bodies
    - n_bodies
    """
    rows = []
    for state in sandbox_states:
        bodies = state.get("bodies", [])
        if not bodies:
            rows.append([0.0, 0.0, 0.0, 0.0])
            continue
        agent = bodies[0]
        cx = float(agent.get("cx", 0.0))
        cy = float(agent.get("cy", 0.0))
        n = len(bodies)
        if n >= 2:
            dists = []
            for i in range(n):
                for j in range(i + 1, n):
                    dx = bodies[i]["cx"] - bodies[j]["cx"]
                    dy = bodies[i]["cy"] - bodies[j]["cy"]
                    dists.append(float(np.hypot(dx, dy)))
            mean_d = float(np.mean(dists))
        else:
            mean_d = 0.0
        rows.append([cx, cy, mean_d, float(n)])
    return np.array(rows)


# ------------------------------------------------------------------ #
# Categorisation: try the CategoryTheoryEngine first, fall back to k-means
# ------------------------------------------------------------------ #
def categorise_with_engine(
    beliefs: np.ndarray, dim: int
) -> tuple[np.ndarray, list[str]]:
    """Use ``CategoryTheoryEngine.topos.classify()``.

    Returns (labels, category_names).
    """
    from zero_data_model.category_engine import CategoryTheoryEngine
    engine = CategoryTheoryEngine(dim=dim)
    # The default categories are NLP, CV, Analytics.
    category_names = ["NLP", "CV", "Analytics"]
    # ``classify`` returns a (dim,) vector — interpret the argmax
    # over the first 3 dims as the category index (the engine's
    # default _init_default_categories creates 3 categories, but the
    # classifier output is a soft assignment over ``dim`` dims).
    # We take the first 3 outputs as a 3-way classifier.
    labels = np.zeros(len(beliefs), dtype=int)
    for i, b in enumerate(beliefs):
        out = engine.topos.classify(b)
        # Take argmax over the first 3 dims.
        if len(out) >= 3:
            labels[i] = int(np.argmax(out[:3]))
    return labels, category_names


def categorise_kmeans(beliefs: np.ndarray, k: int = 3,
                       seed: int = 42) -> tuple[np.ndarray, list[str]]:
    """Fallback: simple k-means on the belief vectors."""
    # Standardise.
    mu = beliefs.mean(axis=0)
    sigma = beliefs.std(axis=0) + 1e-8
    X = (beliefs - mu) / sigma
    # Init: random points from X.
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=k, replace=False)
    centroids = X[idx].copy()
    for _ in range(50):
        # Assign.
        dists = np.linalg.norm(
            X[:, None, :] - centroids[None, :, :], axis=2
        )
        labels = np.argmin(dists, axis=1)
        # Update.
        for c in range(k):
            mask = labels == c
            if mask.any():
                centroids[c] = X[mask].mean(axis=0)
    return labels, [f"cluster_{i}" for i in range(k)]


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """Simple silhouette score (no sklearn)."""
    n = len(X)
    if n < 4:
        return 0.0
    # Pairwise distances.
    diff = X[:, None, :] - X[None, :, :]
    D = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(D, 0)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0
    sils = []
    for i in range(n):
        li = labels[i]
        # a(i): mean distance to same-cluster points.
        same = D[i, labels == li]
        a = same.mean() if len(same) > 1 else 0.0
        # b(i): min mean distance to other-cluster points.
        b_vals = []
        for lj in unique_labels:
            if lj == li:
                continue
            other = D[i, labels == lj]
            if len(other) > 0:
                b_vals.append(other.mean())
        b = min(b_vals) if b_vals else 0.0
        s = (b - a) / max(a, b, 1e-8)
        sils.append(s)
    return float(np.mean(sils))


# ------------------------------------------------------------------ #
# Purity: cross-tabulate category labels with physical bins
# ------------------------------------------------------------------ #
def purity_score(labels: np.ndarray, physical: np.ndarray,
                  n_bins: int = 3) -> dict:
    """Bin the physical features and compute purity.

    Purity = (1/N) * sum_c max_k |{i : label_i=c and phys_i=k}|
    """
    n = len(labels)
    if n == 0:
        return {"purity": 0.0}
    # Bin agent position into a n_bins x n_bins grid.
    cx = physical[:, 0]
    cy = physical[:, 1]
    # Use percentile-based bin edges for balanced bins.
    cx_edges = np.percentile(cx, np.linspace(0, 100, n_bins + 1))
    cy_edges = np.percentile(cy, np.linspace(0, 100, n_bins + 1))
    cx_bin = np.digitize(cx, cx_edges[1:-1])
    cy_bin = np.digitize(cy, cy_edges[1:-1])
    phys_label = cx_bin * n_bins + cy_bin
    # Build the contingency table.
    n_cats = int(labels.max()) + 1
    n_phys = int(phys_label.max()) + 1
    table = np.zeros((n_cats, n_phys), dtype=int)
    for c, k in zip(labels, phys_label):
        table[c, k] += 1
    # Purity.
    purity = float(table.max(axis=0).sum() / n)
    # Normalised mutual information (rough).
    row_sums = table.sum(axis=1, keepdims=True)
    col_sums = table.sum(axis=0, keepdims=True)
    total = table.sum()
    nmi = 0.0
    for c in range(n_cats):
        for k in range(n_phys):
            if table[c, k] > 0:
                p_ck = table[c, k] / total
                p_c = row_sums[c, 0] / total
                p_k = col_sums[0, k] / total
                nmi += p_ck * np.log2(p_ck / (p_c * p_k))
    # Normalise.
    h_c = -np.sum((row_sums / total) * np.log2(row_sums / total + 1e-12))
    h_k = -np.sum((col_sums / total) * np.log2(col_sums / total + 1e-12))
    nmi = float(nmi / max(np.sqrt(h_c * h_k), 1e-12))
    return {
        "purity": purity,
        "nmi": nmi,
        "contingency_table": table.tolist(),
        "n_bins": int(n_bins),
        "n_categories": int(n_cats),
    }


# ------------------------------------------------------------------ #
# Main analysis
# ------------------------------------------------------------------ #
def run_analysis(
    snapshots_dir: Path,
    output_dir: Path = OUTPUT_DIR,
    n_categories: int = 3,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[categories] loading snapshots from {snapshots_dir}")
    data = load_snapshots(snapshots_dir)
    beliefs = data["beliefs"]
    n_steps, n_dim = beliefs.shape
    print(f"[categories] {n_steps} snapshots × {n_dim} belief dims")
    # Extract physical features.
    physical = extract_physical_features(data["sandbox_states"])
    print(f"[categories] physical features shape: {physical.shape}")
    # Try the engine.
    engine_labels: np.ndarray | None = None
    engine_error: str | None = None
    category_names: list[str] = []
    try:
        engine_labels, category_names = categorise_with_engine(
            beliefs, dim=n_dim,
        )
        print(f"[categories] engine assigned "
              f"{len(np.unique(engine_labels))} categories")
    except Exception as exc:
        engine_error = f"{type(exc).__name__}: {exc}"
        print(f"[categories] engine failed: {engine_error}")
    # Fallback: k-means.
    kmeans_labels, kmeans_names = categorise_kmeans(
        beliefs, k=n_categories, seed=42,
    )
    sil = silhouette_score(beliefs, kmeans_labels)
    print(f"[categories] k-means silhouette: {sil:.3f}")
    # Pick the labels we'll use for purity analysis.
    if engine_labels is not None:
        labels = engine_labels
        names = category_names
        label_source = "engine"
    else:
        labels = kmeans_labels
        names = kmeans_names
        label_source = "kmeans"
    # Purity vs physical position.
    purity = purity_score(labels, physical, n_bins=3)
    # For comparison: k-means purity.
    kmeans_purity = purity_score(kmeans_labels, physical, n_bins=3)
    # Build result.
    result = {
        "n_snapshots": int(n_steps),
        "n_belief_dims": int(n_dim),
        "label_source": label_source,
        "n_categories": int(len(names)),
        "category_names": names,
        "engine_error": engine_error,
        "kmeans_silhouette": sil,
        "kmeans_purity": kmeans_purity,
        "engine_or_kmeans_purity": purity,
        "label_counts": {
            names[i]: int((labels == i).sum())
            for i in range(len(names))
        },
    }
    # Save.
    with (output_dir / "categories.json").open("w") as f:
        json.dump(result, f, indent=2)
    # Save arrays.
    np.savez_compressed(
        output_dir / "categories.npz",
        beliefs=beliefs, labels=labels, physical=physical,
        kmeans_labels=kmeans_labels,
        steps=data["steps"], actions=data["actions"],
    )
    # Plot: PCA scatter coloured by (a) labels, (b) physical position.
    if _HAS_MPL:
        # Simple PCA via SVD.
        X = beliefs - beliefs.mean(axis=0)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        pc1 = U[:, 0] * S[0]
        pc2 = U[:, 1] * S[1]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # Panel 1: by label.
        for i in range(len(names)):
            mask = labels == i
            axes[0].scatter(pc1[mask], pc2[mask], s=10, alpha=0.6,
                            label=names[i])
        axes[0].set_title(f"Belief PCA, coloured by {label_source} label")
        axes[0].set_xlabel(f"PC1 ({S[0]**2 / (S**2).sum() * 100:.1f}%)")
        axes[0].set_ylabel(f"PC2 ({S[1]**2 / (S**2).sum() * 100:.1f}%)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        # Panel 2: by agent x-position.
        sc = axes[1].scatter(pc1, pc2, c=physical[:, 0], s=10,
                              cmap="viridis", alpha=0.6)
        plt.colorbar(sc, ax=axes[1], label="agent cx")
        axes[1].set_title("Belief PCA, coloured by agent x-position")
        axes[1].set_xlabel(f"PC1 ({S[0]**2 / (S**2).sum() * 100:.1f}%)")
        axes[1].set_ylabel(f"PC2 ({S[1]**2 / (S**2).sum() * 100:.1f}%)")
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        png_path = output_dir / "categories.png"
        fig.savefig(png_path, dpi=110)
        plt.close(fig)
        print(f"[plot] saved {png_path}")
    print(f"[result] label source: {label_source}  "
          f"silhouette: {sil:.3f}  purity: {purity['purity']:.3f}  "
          f"NMI: {purity['nmi']:.3f}")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshots", type=str, required=True)
    p.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    p.add_argument("--n_categories", type=int, default=3)
    args = p.parse_args()
    run_analysis(
        snapshots_dir=Path(args.snapshots),
        output_dir=Path(args.output_dir),
        n_categories=args.n_categories,
    )


if __name__ == "__main__":
    main()
