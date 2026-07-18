# src/zero_data_model/capabilities/graph_advanced.py
"""Advanced Graph capabilities for the zero-data cognitive model.

Like the base ``graph`` module, every operator here is a deterministic,
rule-based prior composed with the existing core cognitive modules. No
external graph libraries (no networkx / scipy.sparse) and no learned
weights are required -- isomorphism detection, dynamic graph tracking
and spanning-tree extraction are all derived from rule priors.
"""

from __future__ import annotations

import numpy as np

from .graph import CommunityDetector, _validate_adjacency
from .rules import GraphRules


class GraphIsomorphismDetector:
    """Weisfeiler-Lehman style isomorphism check (rule-based, approximate).

    The Weisfeiler-Lehman (WL) algorithm iteratively relabels each node by
    hashing its own label together with the sorted labels of its neighbors.
    After ``max_iter`` rounds, the sorted multiset of node labels forms a
    graph hash. If two graphs have the same hash, they are likely (but not
    guaranteed) isomorphic. WL is a necessary but not sufficient condition
    for isomorphism.
    """

    def __init__(self, dim: int = 64, rules: GraphRules | None = None):
        self.dim = dim
        self.rules = rules or GraphRules()

    def _wl_hash(self, adjacency: np.ndarray, max_iter: int) -> tuple[str, np.ndarray]:
        """Compute the Weisfeiler-Lehman hash of a graph.

        Returns ``(hash_string, final_labels)``.
        """
        adj = _validate_adjacency(adjacency)
        n = adj.shape[0]
        if n == 0:
            return ("", np.zeros(0, dtype=int))
        # Initialize all nodes with the same label.
        labels = np.ones(n, dtype=int)
        for _ in range(max_iter):
            new_labels = []
            for i in range(n):
                # Collect neighbor labels (sorted).
                neighbor_labels = []
                for j in range(n):
                    if adj[i, j] > 0 or adj[j, i] > 0:
                        neighbor_labels.append(int(labels[j]))
                neighbor_labels.sort()
                # Hash: combine own label with sorted neighbor labels.
                combined = (int(labels[i]), tuple(neighbor_labels))
                new_labels.append(hash(combined))
            # Relabel: map raw hashes to consecutive integers.
            unique = sorted(set(new_labels))
            label_map = {h: i + 1 for i, h in enumerate(unique)}
            labels = np.array([label_map[h] for h in new_labels], dtype=int)
        # Final hash: sorted multiset of labels as a string.
        hash_str = ",".join(str(x) for x in sorted(labels))
        return (hash_str, labels)

    def check(self, adj_a: np.ndarray, adj_b: np.ndarray) -> dict:
        """Check if two graphs are likely isomorphic via WL hash.

        Returns ``{'isomorphic', 'confidence', 'wl_hash_a', 'wl_hash_b'}``.
        """
        a = _validate_adjacency(adj_a)
        b = _validate_adjacency(adj_b)
        if a.shape[0] != b.shape[0]:
            return {
                "isomorphic": False,
                "confidence": 1.0,
                "wl_hash_a": "",
                "wl_hash_b": "",
            }
        max_iter = self.rules.isomorphism_max_iter
        hash_a, labels_a = self._wl_hash(a, max_iter)
        hash_b, labels_b = self._wl_hash(b, max_iter)
        isomorphic = (hash_a == hash_b) and a.shape[0] == b.shape[0]
        # WL is necessary but not sufficient; confidence is high but not 1.0.
        confidence = 0.95 if isomorphic else 1.0
        return {
            "isomorphic": bool(isomorphic),
            "confidence": float(confidence),
            "wl_hash_a": hash_a,
            "wl_hash_b": hash_b,
        }


class DynamicGraphTracker:
    """Track community drift in temporal graphs (snapshot diff).

    Given a sequence of graph snapshots (adjacency matrices of the same
    size), this class runs community detection on each, matches communities
    across snapshots by Jaccard overlap, and reports the drift (community
    membership change) over time.
    """

    def __init__(self, dim: int = 64, rules: GraphRules | None = None):
        self.dim = dim
        self.rules = rules or GraphRules()
        self.community_detector = CommunityDetector(dim=dim, rules=rules)

    def track(self, snapshots: list[np.ndarray]) -> dict:
        """Track community drift across ``snapshots``.

        Returns ``{'community_drift', 'node_migrations', 'stability'}``.
        """
        if not snapshots:
            return {
                "community_drift": [],
                "node_migrations": [],
                "stability": 0.0,
            }
        # Validate all snapshots have the same shape.
        adj_list = [_validate_adjacency(s) for s in snapshots]
        if len(adj_list) == 1:
            return {
                "community_drift": [0.0],
                "node_migrations": [],
                "stability": 1.0,
            }
        n = adj_list[0].shape[0]
        # Run community detection on each snapshot.
        results = [self.community_detector.detect(adj) for adj in adj_list]
        # Build node -> community_id maps per snapshot.
        node_comm = []
        for res in results:
            nc = np.full(n, -1, dtype=int)
            for cid, members in enumerate(res["communities"]):
                for node in members:
                    nc[node] = cid
            node_comm.append(nc)
        # Track drift and migrations between consecutive snapshots.
        drift = []
        migrations = []
        for t in range(1, len(adj_list)):
            prev_nc = node_comm[t - 1]
            curr_nc = node_comm[t]
            # Jaccard overlap per community pair.
            prev_comms = set(prev_nc.tolist())
            curr_comms = set(curr_nc.tolist())
            total_overlap = 0.0
            total_union = 0.0
            for pc in prev_comms:
                if pc < 0:
                    continue
                prev_members = set(np.where(prev_nc == pc)[0].tolist())
                for cc in curr_comms:
                    if cc < 0:
                        continue
                    curr_members = set(np.where(curr_nc == cc)[0].tolist())
                    intersection = len(prev_members & curr_members)
                    union = len(prev_members | curr_members)
                    if union > 0:
                        overlap = intersection / union
                        if overlap > 0.5:
                            total_overlap += intersection
                            total_union += union
            drift_t = 1.0 - (total_overlap / total_union if total_union > 0 else 0.0)
            drift.append(float(drift_t))
            # Track individual node migrations.
            for node in range(n):
                if prev_nc[node] != curr_nc[node]:
                    migrations.append({
                        "node": int(node),
                        "from": int(prev_nc[node]),
                        "to": int(curr_nc[node]),
                        "snapshot": t,
                    })
        # Stability = 1 - average drift.
        avg_drift = float(np.mean(drift)) if drift else 0.0
        stability = max(0.0, 1.0 - avg_drift)
        return {
            "community_drift": drift,
            "node_migrations": migrations,
            "stability": float(stability),
        }


class SpanningTreeExtractor:
    """Extract minimum spanning tree via Kruskal (rule-based tie-breaking).

    Kruskal's algorithm: sort edges by weight, then greedily add edges that
    don't create a cycle (checked via union-find). Ties are broken by lower
    node index first (deterministic, rule-based).
    """

    def __init__(self, dim: int = 64, rules: GraphRules | None = None):
        self.dim = dim
        self.rules = rules or GraphRules()

    def extract(self, adjacency: np.ndarray) -> dict:
        """Extract the MST from ``adjacency`` (undirected, weighted).

        Returns ``{'mst_edges', 'total_weight', 'mst_adjacency'}``.
        """
        adj = _validate_adjacency(adjacency)
        n = adj.shape[0]
        if n == 0:
            return {
                "mst_edges": [],
                "total_weight": 0.0,
                "mst_adjacency": np.zeros((0, 0), dtype=float),
            }
        if n == 1:
            return {
                "mst_edges": [],
                "total_weight": 0.0,
                "mst_adjacency": np.zeros((1, 1), dtype=float),
            }
        # Collect edges (upper triangle only, undirected).
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] > 0:
                    edges.append((float(adj[i, j]), i, j))
        # Sort by weight, then by (i, j) for deterministic tie-breaking.
        edges.sort(key=lambda e: (e[0], e[1], e[2]))
        # Union-find.
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            parent[px] = py
            return True

        # Kruskal.
        mst_edges = []
        total_weight = 0.0
        for weight, i, j in edges:
            if union(i, j):
                mst_edges.append((int(i), int(j)))
                total_weight += weight
                if len(mst_edges) == n - 1:
                    break
        # Build MST adjacency matrix.
        mst_adj = np.zeros((n, n), dtype=float)
        for i, j in mst_edges:
            mst_adj[i, j] = adj[i, j]
            mst_adj[j, i] = adj[i, j]
        return {
            "mst_edges": mst_edges,
            "total_weight": float(total_weight),
            "mst_adjacency": mst_adj,
        }
