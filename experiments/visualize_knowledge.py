# experiments/visualize_knowledge.py
"""Knowledge-graph visualiser (Phase J, Task 3.2).

Reads a knowledge graph JSON produced by ``knowledge_graph_builder.py``
and renders a static matplotlib network diagram. For interactive HTML
output (hover, click), the script optionally generates a ``pyvis`` HTML
file if pyvis is installed; otherwise it falls back to matplotlib only.

Visual encoding:
  - Node SIZE ∝ log(visit_count + 1)
  - Node COLOR ∝ mean_free_energy (red = high surprise, blue = low)
  - Edge WIDTH ∝ log(weight + 1)
  - Edge COLOR by edge_type:
      - structural (corpus links):  gray, solid
      - co_read (adjacent reads):   green, dashed
      - latent (cosine sim):        purple, dotted

Time-slice support:
  - The script can take a LIST of KG JSONs (e.g., one per checkpoint)
    and produce a multi-panel "evolution" image showing how the graph
    grew over the learning run.

CLI
---
    python experiments/visualize_knowledge.py
    python experiments/visualize_knowledge.py --input kg.json --output kg.png
    python experiments/visualize_knowledge.py --input_dir checkpoints/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:  # pragma: no cover
    _HAS_MPL = False

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:  # pragma: no cover
    _HAS_NX = False


DEFAULT_INPUT = Path(__file__).resolve().parent / "output" / "encyclopedia_run" / "knowledge_graph.json"
DEFAULT_OUTPUT = DEFAULT_INPUT.with_suffix(".png")

# Edge type -> (color, linestyle).
EDGE_STYLE = {
    "structural": ("gray", "solid"),
    "co_read":    ("#2ca02c", "dashed"),
    "latent":     ("#9467bd", "dotted"),
}


# ------------------------------------------------------------------ #
# Layout
# ------------------------------------------------------------------ #
def _compute_layout(G, seed: int = 42) -> dict:
    """Compute 2D node positions using the best available layout."""
    if G is None or G.number_of_nodes() == 0:
        return {}
    # spring_layout is the most universally available; it works on
    # small/medium graphs (up to a few thousand nodes).
    try:
        return nx.spring_layout(G, seed=seed, k=0.5, iterations=50)
    except Exception:
        # Fallback: circular layout.
        try:
            return nx.circular_layout(G)
        except Exception:
            # Last resort: random.
            rng = np.random.default_rng(seed)
            return {n: rng.standard_normal(2) for n in G.nodes()}


# ------------------------------------------------------------------ #
# Single-graph rendering
# ------------------------------------------------------------------ #
def render_graph(
    kg_json_path: str | Path,
    output_path: str | Path,
    title: str = "Knowledge Graph",
    top_k_nodes: int = 50,
    top_k_edges: int = 100,
) -> str | None:
    """Render one knowledge graph to a PNG. Returns the path or None on failure."""
    if not _HAS_MPL or not _HAS_NX:
        return None
    kg_json_path = Path(kg_json_path)
    if not kg_json_path.exists():
        return None
    with kg_json_path.open() as f:
        data = json.load(f)
    nodes_data = data.get("nodes", {})
    edges_data = data.get("edges", [])
    if not nodes_data:
        return None
    # Build a networkx graph (only top-k nodes by visit_count to keep
    # the plot readable).
    sorted_nodes = sorted(
        nodes_data.items(),
        key=lambda kv: -kv[1].get("visit_count", 0),
    )[:top_k_nodes]
    G = nx.Graph()
    for title, attrs in sorted_nodes:
        G.add_node(title, **attrs)
    # Edges: only those between nodes in our top-k.
    node_set = {t for t, _ in sorted_nodes}
    edges_added = []
    for e in edges_data:
        if e["src"] in node_set and e["dst"] in node_set:
            edges_added.append(e)
    edges_added = sorted(edges_added, key=lambda e: -e.get("weight", 0))[:top_k_edges]
    for e in edges_added:
        G.add_edge(e["src"], e["dst"],
                  weight=e.get("weight", 1.0),
                  edge_type=e.get("edge_type", "structural"))
    if G.number_of_nodes() == 0:
        return None
    # Layout.
    pos = _compute_layout(G, seed=42)
    # Render.
    fig, ax = plt.subplots(figsize=(12, 10))
    # Node sizes (log of visit_count + 1).
    node_sizes = [
        50 + 200 * np.log1p(G.nodes[n].get("visit_count", 0))
        for n in G.nodes()
    ]
    # Node colors by mean_free_energy.
    fes = [G.nodes[n].get("mean_free_energy", 0) for n in G.nodes()]
    if max(fes) > min(fes):
        fes_norm = [(f - min(fes)) / (max(fes) - min(fes)) for f in fes]
    else:
        fes_norm = [0.5] * len(fes)
    # Draw edges first (so nodes overlay them).
    for etype, (color, ls) in EDGE_STYLE.items():
        edge_list = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get("edge_type") == etype
        ]
        if not edge_list:
            continue
        widths = [
            0.5 + 2 * np.log1p(G[u][v].get("weight", 1.0))
            for u, v in edge_list
        ]
        nx.draw_networkx_edges(
            G, pos, edgelist=edge_list, ax=ax,
            edge_color=color, style=ls, width=widths, alpha=0.5,
        )
    # Draw nodes.
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=node_sizes,
        node_color=fes_norm, cmap="coolwarm",
        edgecolors="black", linewidths=0.5, alpha=0.85,
    )
    # Labels (only for high-visit-count nodes to avoid clutter).
    labels_to_show = {
        n: n for n in G.nodes()
        if G.nodes[n].get("visit_count", 0) >= 2
    }
    nx.draw_networkx_labels(
        G, pos, labels=labels_to_show, ax=ax, font_size=7, font_color="black",
    )
    ax.set_title(
        f"{title}\n"
        f"nodes={G.number_of_nodes()} edges={G.number_of_edges()}  "
        f"(node size = visit count, color = mean FE)"
    )
    ax.axis("off")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return str(output_path)


# ------------------------------------------------------------------ #
# Time-slice evolution rendering
# ------------------------------------------------------------------ #
def render_evolution(
    kg_json_paths: list[Path],
    output_path: Path,
    title: str = "Knowledge Graph Evolution",
) -> str | None:
    """Render a multi-panel image showing KG growth over checkpoints."""
    if not _HAS_MPL or not _HAS_NX or not kg_json_paths:
        return None
    n = len(kg_json_paths)
    if n == 0:
        return None
    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    for i, p in enumerate(kg_json_paths):
        ax = axes[i // n_cols, i % n_cols]
        if not p.exists():
            ax.set_title(f"{p.name}\n(missing)")
            ax.axis("off")
            continue
        with p.open() as f:
            data = json.load(f)
        nodes = data.get("nodes", {})
        edges = data.get("edges", [])
        G = nx.Graph()
        for t, a in nodes.items():
            G.add_node(t, **a)
        for e in edges:
            G.add_edge(e["src"], e["dst"], **{k: v for k, v in e.items()
                                              if k not in ("src", "dst")})
        pos = _compute_layout(G, seed=42)
        # Reuse the same layout for all panels? No — each checkpoint
        # may have different nodes. We compute a fresh layout per panel.
        sizes = [50 + 100 * np.log1p(G.nodes[n].get("visit_count", 0))
                 for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes,
                              node_color="#1f77b4", alpha=0.7)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="gray", alpha=0.3, width=0.5)
        ax.set_title(f"{p.stem}\nnodes={G.number_of_nodes()} edges={G.number_of_edges()}")
        ax.axis("off")
    # Hide unused axes.
    for i in range(len(kg_json_paths), n_rows * n_cols):
        axes[i // n_cols, i % n_cols].axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return str(output_path)


# ------------------------------------------------------------------ #
# pyvis interactive HTML (optional)
# ------------------------------------------------------------------ #
def render_interactive_html(
    kg_json_path: str | Path,
    output_path: str | Path,
) -> str | None:
    """Render an interactive HTML network (requires pyvis)."""
    try:
        from pyvis.network import Network
    except ImportError:
        return None
    kg_json_path = Path(kg_json_path)
    if not kg_json_path.exists():
        return None
    with kg_json_path.open() as f:
        data = json.load(f)
    net = Network(height="600px", width="100%", bgcolor="#ffffff",
                  font_color="black", notebook=False)
    for title, attrs in data.get("nodes", {}).items():
        vc = attrs.get("visit_count", 0)
        size = 10 + 5 * np.log1p(vc)
        net.add_node(title, label=title, size=size,
                    title=f"visits={vc}, FE={attrs.get('mean_free_energy', 0):.2f}")
    for e in data.get("edges", []):
        net.add_edge(e["src"], e["dst"],
                    value=e.get("weight", 1.0),
                    title=e.get("edge_type", "structural"))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(output_path))
    return str(output_path)


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualise a knowledge graph JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", type=str, default=str(DEFAULT_INPUT),
                   help="Path to knowledge_graph.json.")
    p.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                   help="Output PNG path.")
    p.add_argument("--input_dir", type=str, default="",
                   help="Directory of KG JSON files (time-slice evolution mode).")
    p.add_argument("--interactive", action="store_true",
                   help="Also save interactive HTML (requires pyvis).")
    p.add_argument("--top_k_nodes", type=int, default=50)
    p.add_argument("--top_k_edges", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 64)
    print("Knowledge Graph Visualiser")
    print("=" * 64)
    if args.input_dir:
        # Evolution mode.
        kg_paths = sorted(Path(args.input_dir).glob("*.json"))
        print(f"  evolution mode: {len(kg_paths)} KG files")
        out = render_evolution(kg_paths, Path(args.output), title="KG Evolution")
        if out:
            print(f"  PNG: {out}")
        else:
            print("  [skip] matplotlib or networkx unavailable, or no input files")
    else:
        print(f"  input:  {args.input}")
        print(f"  output: {args.output}")
        out = render_graph(
            args.input, args.output,
            top_k_nodes=args.top_k_nodes, top_k_edges=args.top_k_edges,
        )
        if out:
            print(f"  PNG: {out}")
        else:
            print("  [skip] matplotlib or networkx unavailable, or empty graph")
        if args.interactive:
            html_path = Path(args.output).with_suffix(".html")
            html_out = render_interactive_html(args.input, html_path)
            if html_out:
                print(f"  HTML: {html_out}")
            else:
                print("  [skip] pyvis not available")
    print("=" * 64)


if __name__ == "__main__":
    main()
