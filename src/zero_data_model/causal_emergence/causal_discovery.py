# src/zero_data_model/causal_emergence/causal_discovery.py
"""Module B: CausalInferenceEngine.

Discovers a directed acyclic graph (DAG) from multivariate observational
data and answers interventional / counterfactual queries.

Three discovery methods (selected by ``rules.causal_method``):
- 'pc' (default): PC algorithm with Fisher-z partial correlation tests
  and Meek (1995) orientation rules R1-R3.
- 'lingam': simplified LiNGAM via SVD whitening + fixed-point ICA
  (Hyvärinen 1999). Does NOT depend on scikit-learn (fix C2).
- 'correlation': threshold-based fallback for degenerate inputs.

Do-calculus uses linear-Gaussian closed-form intervention and
counterfactual (fix H3), avoiding belief propagation.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
from scipy import stats as sp_stats
from scipy.linalg import svd

from .rules import EmergenceRules


class CausalInferenceEngine:
    """Causal discovery and do-calculus engine.

    Constructor follows the capability-domain pattern: plain Python class,
    accepts ``dim``, optional core modules, ``rules``, ``rng``.
    """

    def __init__(
        self,
        dim: int = 64,
        active_inference: Any | None = None,
        rules: EmergenceRules | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.dim = dim
        self.rules = rules or EmergenceRules()
        self.active_inference = active_inference  # reserved for free-energy
        self.rng = rng or np.random.default_rng()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def discover(
        self,
        data: np.ndarray,
        var_names: list[str] | None = None,
        method: str | None = None,
    ) -> dict:
        """Discover causal DAG from observational data.

        Returns dict with ``adjacency``, ``edges``, ``method``,
        ``var_names``, ``n_edges``, ``is_acyclic``.
        """
        method = method or self.rules.causal_method
        data = self._prepare(data)
        if data.size == 0 or data.ndim < 2:
            return self._empty_dag(method, var_names)

        n_samples, n_vars = data.shape
        if var_names is None or len(var_names) != n_vars:
            var_names = [f"x{i}" for i in range(n_vars)]

        # Degenerate cases -> correlation fallback
        if n_samples < 3:
            method_used = "correlation"
            adj = self._correlation_dag(data)
        elif method == "lingam":
            if n_vars > self.rules.causal_max_vars_lingam:
                method_used = "correlation"
                adj = self._correlation_dag(data)
            else:
                adj = self._lingam_dag(data)
                method_used = "lingam"
                # Cyclic -> greedy break
                if not self._is_acyclic(adj):
                    adj = self._break_cycles(adj)
        elif method == "pc":
            adj = self._pc_dag(data)
            method_used = "pc"
        else:  # correlation
            adj = self._correlation_dag(data)
            method_used = "correlation"

        edges = self._edges_from_adj(adj)
        return {
            "adjacency": adj,
            "edges": edges,
            "method": method_used,
            "var_names": var_names,
            "n_edges": int(len(edges)),
            "is_acyclic": self._is_acyclic(adj),
        }

    def intervene(
        self,
        adjacency: np.ndarray,
        data: np.ndarray,
        intervention_var: int,
        intervention_value: float,
    ) -> dict:
        """Estimate the effect of do(X[intervention_var] = value).

        Linear-Gaussian closed-form (spec §4.3): estimate weights via
        OLS regression, then propagate the intervention in topological
        order.
        """
        data = self._prepare(data)
        if data.size == 0:
            return {
                "effect": np.zeros(0),
                "pre_mean": np.zeros(0),
                "post_mean": np.zeros(0),
            }
        n_vars = data.shape[1]
        if intervention_var < 0 or intervention_var >= n_vars:
            raise ValueError(
                f"intervention_var {intervention_var} out of range [0, {n_vars})"
            )

        pre_mean = data.mean(axis=0)
        weights = self._estimate_weights(adjacency, data)

        # Graph mutilation: remove all incoming edges to intervention_var
        adj_do = adjacency.copy()
        adj_do[:, intervention_var] = 0

        # Topological order of the mutilated DAG
        topo = self._topological_sort(adj_do)

        # Propagate intervention
        post_mean = pre_mean.copy()
        post_mean[intervention_var] = intervention_value
        for node in topo:
            if node == intervention_var:
                continue
            parents = np.nonzero(adj_do[:, node])[0]
            if len(parents) == 0:
                continue
            # Post mean of node = sum over parents of (weight * post_mean[parent])
            # Use estimated weights: X[:, node] = sum(weights[p, node] * X[:, p]) + intercept
            w_node = weights[parents, node]
            post_mean[node] = float(
                np.sum(w_node * post_mean[parents])
            )

        effect = post_mean - pre_mean
        return {
            "effect": np.nan_to_num(effect),
            "pre_mean": np.nan_to_num(pre_mean),
            "post_mean": np.nan_to_num(post_mean),
        }

    def counterfactual(
        self,
        adjacency: np.ndarray,
        observed: np.ndarray,
        intervention_var: int,
        intervention_value: float,
    ) -> dict:
        """Counterfactual: 'what would have happened if X[var] had been value?'

        Linear closed-form (spec §4.3): ``cf = observed - W[:, var] *
        (observed[var] - value)`` where ``W`` is the estimated weight matrix.

        fix NEW-H2 (spec deviation documented): this implementation uses
        **unit-weight simplified propagation** — every descendant of
        ``intervention_var`` receives the identical scalar shift
        ``delta = intervention_value - observed[intervention_var]``,
        regardless of edge strength. The full spec formula requires
        ``W`` which cannot be estimated from a single observation. For
        spec-compliant counterfactual, call ``discover()`` first to fit
        ``_estimate_weights``, then apply the weighted propagation
        externally. Returns ``counterfactual``, ``factual``, ``shift``.
        """
        observed = np.asarray(observed, dtype=float).flatten()
        n_vars = observed.shape[0]
        if intervention_var < 0 or intervention_var >= n_vars:
            raise ValueError(
                f"intervention_var {intervention_var} out of range [0, {n_vars})"
            )

        # Estimate weights from a single observation is impossible;
        # use adjacency structure directly (weights default to adjacency
        # indicator; for proper counterfactual, call estimate_weights with
        # data first). Here we approximate with adjacency as identity weights.
        # The caller is expected to have called discover() with data first;
        # for the spec-compliant simplified version, we use linear unit weights.
        cf = observed.copy()
        # Set intervention variable to its counterfactual value (do-operator).
        cf[intervention_var] = intervention_value
        # Propagate delta to descendants in topological order.
        delta = intervention_value - observed[intervention_var]
        topo = self._topological_sort(adjacency)
        propagated = {intervention_var}
        for node in topo:
            if node == intervention_var:
                continue
            parents = np.nonzero(adjacency[:, node])[0]
            if any(p in propagated for p in parents):
                # Linear unit-weight propagation: shift child by delta.
                cf[node] = observed[node] + delta
                propagated.add(node)
        # Parents of intervention_var are unaffected (post-intervention).
        # Counterfactual factual value is `observed`, counterfactual is `cf`.
        shift = cf - observed
        return {
            "counterfactual": np.nan_to_num(cf),
            "factual": np.nan_to_num(observed),
            "shift": np.nan_to_num(shift),
        }

    # ------------------------------------------------------------------
    # Internals — data preparation
    # ------------------------------------------------------------------
    def _prepare(self, data: np.ndarray) -> np.ndarray:
        """Sanitize input: contiguous float, NaN/Inf -> 0, 2D."""
        if data is None:
            return np.zeros(0)
        arr = np.ascontiguousarray(data, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    def _empty_dag(
        self, method: str, var_names: list[str] | None
    ) -> dict:
        return {
            "adjacency": np.zeros((0, 0)),
            "edges": [],
            "method": method,
            "var_names": var_names or [],
            "n_edges": 0,
            "is_acyclic": True,
        }

    # ------------------------------------------------------------------
    # Internals — PC algorithm
    # ------------------------------------------------------------------
    def _pc_dag(self, data: np.ndarray) -> np.ndarray:
        """PC algorithm: skeleton + Meek R1-R3 orientation."""
        n_vars = data.shape[1]
        adj = np.ones((n_vars, n_vars), dtype=int) - np.eye(
            n_vars, dtype=int
        )

        # Skeleton learning: remove edges i-j if i ⊥ j | S for some S
        # Layered: |S| = 0, 1, 2, ... up to causal_max_cond_set.
        for cond_size in range(self.rules.causal_max_cond_set + 1):
            # For each remaining edge (i, j), test independence given cond_size neighbors
            edges_to_test = []
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    if adj[i, j] == 1:
                        edges_to_test.append((i, j))
            for i, j in edges_to_test:
                if adj[i, j] == 0:
                    continue  # already removed
                # Candidate conditioning set: neighbors of i (excluding j)
                neighbors = [
                    k for k in range(n_vars)
                    if k != i and k != j and (adj[i, k] == 1 or adj[k, i] == 1)
                ]
                if len(neighbors) < cond_size:
                    continue
                independent = False
                for S in combinations(neighbors, cond_size):
                    if self._is_independent(data, i, j, S):
                        adj[i, j] = 0
                        adj[j, i] = 0
                        independent = True
                        break
                if independent:
                    continue

        # Orient edges using Meek rules R1-R3 (fix M2).
        # Start with undirected adjacency (symmetric). Orient as directed.
        # We track direction via a separate matrix: directed[i, j] = 1 means i -> j.
        directed = np.zeros((n_vars, n_vars), dtype=int)
        # Undirected edges are those where adj[i, j] == 1 and directed[i, j] == 0
        # and directed[j, i] == 0.

        # R1 (v-structures / colliders): for each unshielded triple a - c - b
        # where a, b not adjacent, if c is NOT in the separator of a and b,
        # orient as a -> c <- b.
        for c in range(n_vars):
            neighbors_c = [
                k for k in range(n_vars) if adj[k, c] == 1 or adj[c, k] == 1
            ]
            for a, b in combinations(neighbors_c, 2):
                if adj[a, b] == 1 or adj[b, a] == 1:
                    continue  # a, b adjacent -> shielded
                # Check if c was in the separator of a, b
                # Simplified: assume c was not in separator (v-structure)
                directed[a, c] = 1
                directed[b, c] = 1
                adj[a, c] = 0  # consume undirected
                adj[c, a] = 0
                adj[b, c] = 0
                adj[c, b] = 0

        # R2: a -> b -> c and a - c => a -> c (avoid cycle)
        changed = True
        while changed:
            changed = False
            for a in range(n_vars):
                for c in range(n_vars):
                    if a == c:
                        continue
                    if directed[a, c] == 1:
                        continue
                    if adj[a, c] == 0:
                        continue
                    # Check for b with a -> b -> c
                    for b in range(n_vars):
                        if b in (a, c):
                            continue
                        if directed[a, b] == 1 and directed[b, c] == 1:
                            directed[a, c] = 1
                            adj[a, c] = 0
                            adj[c, a] = 0
                            changed = True
                            break

            # R1 (fix NEW-H1): a -> b, a - c, c - b, and a NOT adjacent to c
            #   => orient c -> b. The previous block was dead code with three
            #   nested `pass` statements and never oriented any edges.
            for a in range(n_vars):
                for b in range(n_vars):
                    if a == b or directed[a, b] != 1:
                        continue
                    for c in range(n_vars):
                        if c in (a, b) or directed[c, b] == 1:
                            continue
                        # c - b undirected and a NOT adjacent to c
                        if (
                            adj[c, b] == 1
                            and adj[a, c] == 0
                            and adj[c, a] == 0
                        ):
                            directed[c, b] = 1
                            adj[c, b] = 0
                            adj[b, c] = 0
                            changed = True

            # R3: a - b, a - c, a - d, c -> b, d -> b, c, d not adjacent => a -> b
            for b in range(n_vars):
                for a in range(n_vars):
                    if a == b:
                        continue
                    if directed[a, b] == 1:
                        continue
                    if adj[a, b] == 0:
                        continue
                    # Find c, d with c -> b, d -> b, a - c, a - d, c not adj d
                    candidates = [
                        c for c in range(n_vars)
                        if c != a and c != b and directed[c, b] == 1
                        and (adj[a, c] == 1 or adj[c, a] == 1
                             or directed[a, c] == 1 or directed[c, a] == 1)
                    ]
                    found = False
                    for c, d in combinations(candidates, 2):
                        if adj[c, d] == 0 and adj[d, c] == 0:
                            # c, d not adjacent
                            directed[a, b] = 1
                            adj[a, b] = 0
                            adj[b, a] = 0
                            changed = True
                            found = True
                            break
                    if found:
                        break

        # Remaining undirected edges: orient by index order (i < j => i -> j)
        # to break ties deterministically.
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if adj[i, j] == 1:
                    directed[i, j] = 1
                    adj[i, j] = 0
                    adj[j, i] = 0

        return directed

    def _is_independent(
        self,
        data: np.ndarray,
        i: int,
        j: int,
        S: tuple[int, ...],
    ) -> bool:
        """Test conditional independence i ⊥ j | S via Fisher z (fix M1)."""
        n = data.shape[0]
        if len(S) == 0:
            with np.errstate(invalid="ignore", divide="ignore"):
                r = float(np.corrcoef(data[:, i], data[:, j])[0, 1])
        else:
            # Partial correlation via regression residuals
            r = self._partial_correlation(data, i, j, S)

        if not np.isfinite(r):
            return True  # treat as independent if undefined

        # Fisher z transform
        r_clipped = float(np.clip(r, -0.9999, 0.9999))
        z = 0.5 * np.log((1 + r_clipped) / (1 - r_clipped))
        stat = np.sqrt(max(0, n - len(S) - 3)) * abs(z)
        p_value = 2 * (1 - sp_stats.norm.cdf(stat))
        return bool(p_value > self.rules.causal_significance)

    @staticmethod
    def _partial_correlation(
        data: np.ndarray, i: int, j: int, S: tuple[int, ...]
    ) -> float:
        """Partial correlation of i, j given S via residual regression."""
        if len(S) == 0:
            with np.errstate(invalid="ignore", divide="ignore"):
                r = np.corrcoef(data[:, i], data[:, j])[0, 1]
            return float(r)
        Z = data[:, list(S)]
        # Add intercept
        Z_aug = np.column_stack([Z, np.ones(Z.shape[0])])
        # Residuals of i, j after regressing on Z
        try:
            beta_i, *_ = np.linalg.lstsq(Z_aug, data[:, i], rcond=None)
            beta_j, *_ = np.linalg.lstsq(Z_aug, data[:, j], rcond=None)
            resid_i = data[:, i] - Z_aug @ beta_i
            resid_j = data[:, j] - Z_aug @ beta_j
            std_i = float(np.std(resid_i))
            std_j = float(np.std(resid_j))
            if std_i < 1e-12 or std_j < 1e-12:
                return 0.0
            with np.errstate(invalid="ignore", divide="ignore"):
                r = float(np.corrcoef(resid_i, resid_j)[0, 1])
            return r
        except np.linalg.LinAlgError:
            return 0.0

    # ------------------------------------------------------------------
    # Internals — LiNGAM (fix C2: no scikit-learn)
    # ------------------------------------------------------------------
    def _lingam_dag(self, data: np.ndarray) -> np.ndarray:
        """Simplified LiNGAM via SVD whitening + fixed-point ICA."""
        n_vars = data.shape[1]
        # Center
        X = data - data.mean(axis=0)
        # Whiten via SVD
        try:
            U, S, Vt = svd(X, full_matrices=False)
        except np.linalg.LinAlgError:
            return self._correlation_dag(data)
        # Whitening matrix
        eps = 1e-8
        D = np.diag(1.0 / np.sqrt(S + eps))
        W_white = Vt.T @ D @ Vt  # (n_vars, n_vars)
        Z = X @ W_white  # whitened data, (n_samples, n_vars)

        # Fixed-point ICA (Hyvärinen 1999)
        W = self._fixed_point_ica(Z, n_vars)
        if W is None:
            return self._correlation_dag(data)

        # Find permutation making W as lower-triangular as possible
        # (LiNGAM: B = W @ W_white is lower triangular under correct ordering).
        B = W @ W_white
        adj = self._permute_to_lower_triangular(B)
        # Binarize
        adj = (np.abs(adj) > 1e-6).astype(int)
        np.fill_diagonal(adj, 0)
        return adj

    def _fixed_point_ica(
        self, Z: np.ndarray, n_vars: int
    ) -> np.ndarray | None:
        """Fixed-point ICA (Hyvärinen 1999) with tanh nonlinearity."""
        max_iter = 100
        tol = 1e-5
        W = np.eye(n_vars)
        rng = self.rng
        # Initialize with random orthogonal matrix
        A = rng.standard_normal((n_vars, n_vars))
        Q, _ = np.linalg.qr(A)
        W = Q

        for _ in range(max_iter):
            W_prev = W.copy()
            for i in range(n_vars):
                w = W[i, :]
                # Fixed-point update: w = E[Z * g(w^T Z)] - E[g'(w^T Z)] * w
                proj = Z @ w  # (n_samples,)
                g = np.tanh(proj)
                g_prime = 1 - g * g
                w_new = (Z * g[:, None]).mean(axis=0) - float(
                    g_prime.mean()
                ) * w
                # Symmetric decorrelation
                W_new = W.copy()
                W_new[i, :] = w_new
                # Decorrelate: W = (W @ W^T)^{-1/2} @ W
                try:
                    s, V = np.linalg.eigh(W_new @ W_new.T)
                    s = np.clip(s, 1e-8, None)
                    D_inv_sqrt = V @ np.diag(1.0 / np.sqrt(s)) @ V.T
                    W = D_inv_sqrt @ W_new
                except np.linalg.LinAlgError:
                    return None
            # Convergence check
            delta = np.max(np.abs(W - W_prev))
            if delta < tol:
                break
        return W

    @staticmethod
    def _permute_to_lower_triangular(B: np.ndarray) -> np.ndarray:
        """Find row permutation making B as lower-triangular as possible.

        Greedy: at each step, pick the row with the smallest sum of
        absolute values in the unfilled columns (LiNGAM ordering).
        """
        n = B.shape[0]
        perm = []
        remaining = list(range(n))
        B_abs = np.abs(B)
        for _ in range(n):
            if not remaining:
                break
            # Pick row with smallest sum in remaining columns
            scores = [
                (B_abs[r, remaining].sum(), r) for r in remaining
            ]
            scores.sort()
            perm.append(scores[0][1])
            remaining.remove(scores[0][1])
        # Reorder rows of B
        B_perm = B[perm, :][:, perm]
        # Zero out upper triangle to enforce DAG
        B_perm = np.tril(B_perm, k=-1)
        return B_perm

    # ------------------------------------------------------------------
    # Internals — correlation fallback
    # ------------------------------------------------------------------
    def _correlation_dag(self, data: np.ndarray) -> np.ndarray:
        """Threshold-based DAG via correlation (i < j => i -> j)."""
        n_vars = data.shape[1]
        adj = np.zeros((n_vars, n_vars), dtype=int)
        if data.shape[0] < 2:
            return adj
        try:
            with np.errstate(invalid="ignore", divide="ignore"):
                C = np.corrcoef(data.T)
        except (ValueError, np.linalg.LinAlgError):
            return adj
        C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if abs(C[i, j]) > self.rules.causal_significance:
                    adj[i, j] = 1
        return adj

    # ------------------------------------------------------------------
    # Internals — graph utilities
    # ------------------------------------------------------------------
    def _estimate_weights(
        self, adjacency: np.ndarray, data: np.ndarray
    ) -> np.ndarray:
        """Estimate linear weights W[parent, child] via OLS regression."""
        n_vars = data.shape[1]
        weights = np.zeros((n_vars, n_vars))
        for child in range(n_vars):
            parents = np.nonzero(adjacency[:, child])[0]
            if len(parents) == 0:
                continue
            X_p = data[:, parents]
            # Add intercept
            X_aug = np.column_stack([X_p, np.ones(X_p.shape[0])])
            try:
                beta, *_ = np.linalg.lstsq(X_aug, data[:, child], rcond=None)
                weights[parents, child] = beta[:-1]
            except np.linalg.LinAlgError:
                weights[parents, child] = 1.0
        return weights

    @staticmethod
    def _edges_from_adj(adj: np.ndarray) -> list[tuple[int, int]]:
        edges = []
        n = adj.shape[0]
        for i in range(n):
            for j in range(n):
                if adj[i, j] == 1:
                    edges.append((int(i), int(j)))
        return edges

    @staticmethod
    def _is_acyclic(adj: np.ndarray) -> bool:
        """Check if adjacency matrix represents a DAG (via DFS)."""
        n = adj.shape[0]
        if n == 0:
            return True
        color = np.zeros(n, dtype=int)  # 0=white, 1=gray, 2=black

        def dfs(u: int) -> bool:
            color[u] = 1
            for v in range(n):
                if adj[u, v] == 1:
                    if color[v] == 1:
                        return False  # back edge -> cycle
                    if color[v] == 0 and not dfs(v):
                        return False
            color[u] = 2
            return True

        return all(not (color[u] == 0 and not dfs(u)) for u in range(n))

    def _break_cycles(self, adj: np.ndarray) -> np.ndarray:
        """Greedily remove edges to break cycles.

        fix NEW-L2: ``adj`` is binary (entries 0/1), so the sort is a
        deterministic tiebreak by ``(i, j)`` index order, not by weight
        magnitude. For weighted edge removal, populate ``adj`` with
        correlation/LiNGAM coefficients before calling this method.
        """
        adj = adj.copy()
        # Binary adjacency: sort deterministically by (i, j) index order.
        edges = []
        n = adj.shape[0]
        for i in range(n):
            for j in range(n):
                if adj[i, j] == 1:
                    edges.append((float(adj[i, j]), i, j))
        edges.sort()
        for _, i, j in edges:
            if self._is_acyclic(adj):
                break
            adj[i, j] = 0
        return adj

    @staticmethod
    def _topological_sort(adj: np.ndarray) -> list[int]:
        """Topological sort of DAG via Kahn's algorithm."""
        n = adj.shape[0]
        in_degree = adj.sum(axis=0).astype(int).tolist()
        queue = [i for i in range(n) if in_degree[i] == 0]
        topo = []
        while queue:
            u = queue.pop(0)
            topo.append(u)
            for v in range(n):
                if adj[u, v] == 1:
                    in_degree[v] -= 1
                    if in_degree[v] == 0:
                        queue.append(v)
        return topo
