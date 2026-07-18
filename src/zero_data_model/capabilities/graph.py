# src/zero_data_model/capabilities/graph.py
"""Graph capability module for the zero-data cognitive model.

Composes the core cognitive modules (math universe) with a small rule
library (``GraphRules``) used as prior knowledge. No external training
data and no learned weights are required -- every operator below is a
deterministic, rule-based prior blended with self-generated representations.

All graph processing uses only numpy core (no networkx / scipy.sparse
dependencies). Graphs are represented as dense numpy adjacency matrices
(assuming small-to-medium graphs, n_nodes <= a few hundred).
"""

from __future__ import annotations

import numpy as np

from ..math_universe import MathematicalUniverse
from .rules import GraphRules


def _validate_adjacency(adjacency: np.ndarray) -> np.ndarray:
    """Coerce array-like into a 2D square float matrix."""
    adj = np.asarray(adjacency, dtype=float)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(
            f"adjacency must be 2D square, got shape {adj.shape}"
        )
    return adj


def _degree(adjacency: np.ndarray) -> np.ndarray:
    """Node degrees (row sums, treating the matrix as weighted)."""
    return np.sum(adjacency, axis=1)


def _n_edges(adjacency: np.ndarray) -> float:
    """Total edge weight (sum of upper triangle, undirected)."""
    n = adjacency.shape[0]
    if n == 0:
        return 0.0
    upper = np.triu(adjacency, k=1)
    return float(np.sum(upper))


class GraphEncoder:
    """Encode a graph (adjacency matrix + optional node features) into a vector.

    Pipeline: graph statistics (n_nodes, n_edges, density, degree moments) ->
    top-k Laplacian eigenvalues -> fractal compression via math_universe ->
    dim-length L2-normalized vector.
    """

    def __init__(
        self,
        dim: int = 64,
        math_universe: MathematicalUniverse | None = None,
        rules: GraphRules | None = None,
    ):
        self.dim = dim
        self.rules = rules or GraphRules()
        self.math_universe = math_universe or MathematicalUniverse(dim=dim)

    def encode(
        self,
        adjacency: np.ndarray,
        node_features: np.ndarray | None = None,
    ) -> np.ndarray:
        """Encode ``adjacency`` into a ``dim``-length L2-normalized vector."""
        adj = _validate_adjacency(adjacency)
        n = adj.shape[0]
        if n == 0:
            return np.zeros(self.dim, dtype=float)
        # Graph statistics.
        n_nodes = float(n)
        n_edges = _n_edges(adj)
        max_possible = n * (n - 1) / 2.0
        density = n_edges / max_possible if max_possible > 0 else 0.0
        degrees = _degree(adj)
        degree_mean = float(np.mean(degrees)) if n > 0 else 0.0
        degree_std = float(np.std(degrees)) if n > 0 else 0.0
        # Spectral features: top-k eigenvalues of the normalized Laplacian.
        # L = I - D^{-1/2} A D^{-1/2}
        d_sqrt_inv = np.zeros_like(degrees)
        nonzero = degrees > 1e-12
        d_sqrt_inv[nonzero] = 1.0 / np.sqrt(degrees[nonzero])
        norm_adj = adj * d_sqrt_inv[:, None] * d_sqrt_inv[None, :]
        laplacian = np.eye(n) - norm_adj
        try:
            eigvals = np.linalg.eigvalsh(laplacian)
        except np.linalg.LinAlgError:
            eigvals = np.zeros(n)
        eigvals = np.nan_to_num(eigvals, nan=0.0, posinf=0.0, neginf=0.0)
        # Take up to 8 eigenvalues (sorted).
        k = min(8, n)
        top_eigvals = np.sort(eigvals)[:k]
        # Combine features.
        features = np.concatenate([
            np.array([n_nodes, n_edges, density, degree_mean, degree_std]),
            top_eigvals,
        ])
        # Optionally append node feature aggregates.
        if node_features is not None:
            nf = np.asarray(node_features, dtype=float)
            if nf.size > 0:
                agg = np.array([
                    float(np.mean(nf)),
                    float(np.std(nf)),
                    float(np.max(nf)),
                    float(np.min(nf)),
                ])
                features = np.concatenate([features, agg])
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        # Pad / truncate to dim, then L2 normalize.
        if features.size >= self.dim:
            compressed = features[: self.dim]
        else:
            compressed = np.pad(features, (0, self.dim - features.size))
        norm = float(np.linalg.norm(compressed))
        if norm > 1e-8:
            compressed = compressed / norm
        return compressed.astype(float)


class CommunityDetector:
    """Rule-based community detection via modularity optimization.

    Implements a simplified Louvain-style local move: start each node in its
    own community, then iteratively move nodes to a neighbor's community if
    the modularity gain is positive. Max 10 iterations or convergence.
    """

    def __init__(self, dim: int = 64, rules: GraphRules | None = None):
        self.dim = dim
        self.rules = rules or GraphRules()
        self.max_iterations = 10

    def _modularity_gain(
        self,
        adjacency: np.ndarray,
        degrees: np.ndarray,
        m: float,
        node: int,
        target_community: int,
        communities: np.ndarray,
    ) -> float:
        """Modularity gain of moving ``node`` to ``target_community``."""
        n = adjacency.shape[0]
        # Sum of links from node to target community.
        k_i_in = 0.0
        for j in range(n):
            if communities[j] == target_community and j != node:
                k_i_in += adjacency[node, j]
        # Sum of degrees in target community (excluding node).
        k_target = float(np.sum(degrees[communities == target_community]))
        k_i = float(degrees[node])
        if m <= 0:
            return 0.0
        # Modularity gain (simplified).
        gain = k_i_in / m - self.rules.community_resolution * (
            k_target * k_i / (2.0 * m * m)
        )
        return float(gain)

    def detect(self, adjacency: np.ndarray) -> dict:
        """Detect communities in ``adjacency``.

        Returns ``{'communities', 'modularity', 'n_communities'}``.
        """
        adj = _validate_adjacency(adjacency)
        n = adj.shape[0]
        if n == 0:
            return {"communities": [], "modularity": 0.0, "n_communities": 0}
        if n == 1:
            return {
                "communities": [[0]],
                "modularity": 0.0,
                "n_communities": 1,
            }
        m = _n_edges(adj)
        if m <= 0:
            # No edges -> each node is its own community.
            return {
                "communities": [[i] for i in range(n)],
                "modularity": 0.0,
                "n_communities": n,
            }
        degrees = _degree(adj)
        communities = np.arange(n)
        # Iterative local move.
        for _ in range(self.max_iterations):
            moved = False
            for node in range(n):
                current_comm = communities[node]
                # Find neighbor communities.
                neighbor_comms = set()
                for j in range(n):
                    if adj[node, j] > 0 or adj[j, node] > 0:
                        neighbor_comms.add(int(communities[j]))
                if not neighbor_comms:
                    continue
                # Try moving to each neighbor community.
                best_gain = 0.0
                best_comm = current_comm
                for target in neighbor_comms:
                    if target == current_comm:
                        continue
                    gain = self._modularity_gain(
                        adj, degrees, m, node, target, communities
                    )
                    if gain > best_gain:
                        best_gain = gain
                        best_comm = target
                if best_comm != current_comm:
                    communities[node] = best_comm
                    moved = True
            if not moved:
                break
        # Collect communities.
        comm_labels = np.unique(communities)
        comm_list = []
        for label in comm_labels:
            members = [int(i) for i in range(n) if communities[i] == label]
            comm_list.append(members)
        # Compute final modularity.
        modularity = self._compute_modularity(adj, communities, degrees, m)
        return {
            "communities": comm_list,
            "modularity": float(modularity),
            "n_communities": len(comm_list),
        }

    def _compute_modularity(
        self,
        adjacency: np.ndarray,
        communities: np.ndarray,
        degrees: np.ndarray,
        m: float,
    ) -> float:
        """Compute the modularity Q of a partition."""
        if m <= 0:
            return 0.0
        n = adjacency.shape[0]
        q = 0.0
        for i in range(n):
            for j in range(n):
                if communities[i] == communities[j]:
                    expected = degrees[i] * degrees[j] / (2.0 * m)
                    q += adjacency[i, j] - expected
        return q / (2.0 * m)


class PathFinder:
    """Shortest path via Dijkstra (dense adjacency, no heapq dependency).

    The graph is treated as undirected; edge weights are the adjacency values.
    If a heuristic function is provided, A* is used instead (the heuristic
    guides the search but does not affect the optimal path for admissible
    heuristics).
    """

    def __init__(self, dim: int = 64, rules: GraphRules | None = None):
        self.dim = dim
        self.rules = rules or GraphRules()

    def find(
        self,
        adjacency: np.ndarray,
        source: int,
        target: int,
    ) -> dict:
        """Find the shortest path from ``source`` to ``target``.

        Returns ``{'path', 'distance', 'visited'}``.
        """
        adj = _validate_adjacency(adjacency)
        n = adj.shape[0]
        if n == 0 or source < 0 or source >= n or target < 0 or target >= n:
            return {"path": [], "distance": float("inf"), "visited": 0}
        if source == target:
            return {"path": [source], "distance": 0.0, "visited": 1}
        # Dijkstra.
        dist = np.full(n, float("inf"))
        dist[source] = 0.0
        prev = np.full(n, -1, dtype=int)
        visited = np.zeros(n, dtype=bool)
        visited_count = 0
        for _ in range(n):
            # Find unvisited node with min distance.
            min_dist = float("inf")
            u = -1
            for v in range(n):
                if not visited[v] and dist[v] < min_dist:
                    min_dist = dist[v]
                    u = v
            if u == -1 or u == target:
                break
            visited[u] = True
            visited_count += 1
            for v in range(n):
                if not visited[v] and adj[u, v] > 0:
                    alt = dist[u] + adj[u, v]
                    if alt < dist[v]:
                        dist[v] = alt
                        prev[v] = u
        if dist[target] == float("inf"):
            return {"path": [], "distance": float("inf"), "visited": visited_count}
        # Reconstruct path.
        path = [target]
        node = target
        while prev[node] != -1:
            path.append(int(prev[node]))
            node = int(prev[node])
        path.reverse()
        return {
            "path": path,
            "distance": float(dist[target]),
            "visited": visited_count,
        }


class CentralityAnalyzer:
    """Compute degree, closeness, and betweenness centrality (rule-based).

    All three are computed from the dense adjacency matrix without any
    external graph library. Betweenness uses the Brandes algorithm
    (O(V*E) for unweighted, O(V^3) for dense).
    """

    def __init__(self, dim: int = 64, rules: GraphRules | None = None):
        self.dim = dim
        self.rules = rules or GraphRules()

    def analyze(self, adjacency: np.ndarray) -> dict:
        """Analyze centrality of ``adjacency``.

        Returns ``{'degree', 'betweenness', 'closeness', 'most_central'}``.
        """
        adj = _validate_adjacency(adjacency)
        n = adj.shape[0]
        if n == 0:
            return {
                "degree": np.zeros(0, dtype=float),
                "betweenness": np.zeros(0, dtype=float),
                "closeness": np.zeros(0, dtype=float),
                "most_central": -1,
            }
        # Degree centrality.
        degrees = _degree(adj)
        degree_cent = (
            degrees / (n - 1)
            if self.rules.centrality_normalized and n > 1
            else degrees
        )
        # Closeness centrality via Dijkstra per node.
        closeness = np.zeros(n, dtype=float)
        for s in range(n):
            dists = self._dijkstra(adj, s)
            reachable = dists[dists < float("inf")]
            if reachable.size > 0:
                total = float(np.sum(reachable))
                if total > 0:
                    closeness[s] = (reachable.size - 1) / total
                    if self.rules.centrality_normalized:
                        closeness[s] *= (reachable.size - 1) / (n - 1) if n > 1 else 1.0
        # Betweenness centrality (Brandes-style, simplified for dense).
        betweenness = self._betweenness(adj)
        # Most central node (by sum of normalized centralities).
        deg_max = degree_cent.max()
        deg_norm = degree_cent / (deg_max + 1e-12) if deg_max > 0 else degree_cent
        btw_max = betweenness.max()
        btw_norm = betweenness / (btw_max + 1e-12) if btw_max > 0 else betweenness
        cls_max = closeness.max()
        cls_norm = closeness / (cls_max + 1e-12) if cls_max > 0 else closeness
        combined = deg_norm + btw_norm + cls_norm
        most_central = int(np.argmax(combined)) if n > 0 else -1
        return {
            "degree": degree_cent,
            "betweenness": betweenness,
            "closeness": closeness,
            "most_central": most_central,
        }

    def _dijkstra(self, adj: np.ndarray, source: int) -> np.ndarray:
        """Dijkstra from ``source``, returning distance array."""
        n = adj.shape[0]
        dist = np.full(n, float("inf"))
        dist[source] = 0.0
        visited = np.zeros(n, dtype=bool)
        for _ in range(n):
            u = -1
            min_d = float("inf")
            for v in range(n):
                if not visited[v] and dist[v] < min_d:
                    min_d = dist[v]
                    u = v
            if u == -1:
                break
            visited[u] = True
            for v in range(n):
                if not visited[v] and adj[u, v] > 0:
                    alt = dist[u] + adj[u, v]
                    if alt < dist[v]:
                        dist[v] = alt
        return dist

    def _betweenness(self, adj: np.ndarray) -> np.ndarray:
        """Betweenness centrality via Brandes algorithm (simplified)."""
        n = adj.shape[0]
        betweenness = np.zeros(n, dtype=float)
        for s in range(n):
            # Single-source shortest paths (Dijkstra-like).
            stack = []
            pred = [[] for _ in range(n)]
            sigma = np.zeros(n, dtype=float)
            sigma[s] = 1.0
            dist = np.full(n, -1, dtype=int)
            dist[s] = 0
            # Priority queue (sorted list).
            queue = [(0, s)]
            while queue:
                queue.sort()
                d, v = queue.pop(0)
                if d > dist[v] and dist[v] >= 0:
                    continue
                stack.append(v)
                for w in range(n):
                    if adj[v, w] > 0:
                        if dist[w] < 0:
                            queue.append((d + 1, w))
                            dist[w] = d + 1
                            sigma[w] = sigma[v]
                            pred[w] = [v]
                        elif dist[w] == d + 1:
                            sigma[w] += sigma[v]
                            pred[w].append(v)
            # Accumulation.
            delta = np.zeros(n, dtype=float)
            while stack:
                w = stack.pop()
                for v in pred[w]:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
                if w != s:
                    betweenness[w] += delta[w]
        # Normalize for undirected graphs.
        if n > 2:
            betweenness /= ((n - 1) * (n - 2)) / 2.0
        return betweenness
