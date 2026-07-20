# src/zero_data_model/causal_emergence/topology.py
"""Module A: PersistentHomologyPerceiver.

Perceives arbitrary high-dimensional data as a topological point cloud and
computes its persistent homology via Vietoris-Rips filtration. Output is
invariant under rotation, translation, and non-degenerate deformation.

Algorithm: layered degradation strategy (spec §3.2). For n_points ≤ 16
(default), full boundary matrix column reduction over GF(2) computes
betti_0, betti_1, betti_2 and a persistence diagram. Larger inputs are
deterministically subsampled to ``topology_max_points`` via the rng.

Phase D complexity guard: the boundary-matrix column reduction is
O(n_cols^3 / 64) word ops, where n_cols = sum_{d=0}^{max_dim+1} C(n_points,
d+1). For the default ``max_dim=1`` this is O(n^6) -- n=16 finishes in
~20ms, n=32 in ~25ms. For ``max_dim >= 2`` it is O(n^9) -- n=16 finishes
in ~300ms, n=24 in ~9s, n=32 in ~100s, n=40 in ~10min.

To prevent the silent O(n^9) hang on medium-sized inputs (the user-facing
symptom of "feature exists but is unusable"), two hard caps are enforced
in ``perceive`` / ``_persistent_homology``:

- ``topology_max_points_high_dim`` (default 16): hard cap on n_points
  when ``max_dim >= 2``. Raise this rule ONLY if you understand the
  O(n^9) cost.
- ``topology_max_simplices`` (default 5000): hard cap on total simplex
  count regardless of dim. The ultimate guard against combinatorial
  explosion. n_cols = 5000 ~= 2e9 reduction ops ~= 30s.

Both caps raise ``ValueError`` with a message pointing at the rule
fields to override, so the silent 100s hang becomes a loud, actionable
error.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

from .rules import EmergenceRules


class PersistentHomologyPerceiver:
    """Persistent homology perceiver (Vietoris-Rips filtration).

    Constructor matches the capability-domain pattern: plain Python class,
    accepts ``dim``, optional core modules, ``rules``, ``rng``. Not a
    ``CognitiveModule`` subclass; not in ``self.modules``; not counted in
    ``_N_COGNITIVE_MODULES``.
    """

    def __init__(
        self,
        dim: int = 64,
        math_universe: Any | None = None,
        rules: EmergenceRules | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.dim = dim
        self.rules = rules or EmergenceRules()
        self.math_universe = math_universe  # reserved for fractal compression
        self.rng = rng or np.random.default_rng()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def perceive(self, data: np.ndarray, max_dim: int | None = None) -> dict:
        """Compute persistent homology of the input point cloud.

        Parameters
        ----------
        data
            Input array. 1D arrays are treated as a single-point cloud
            (degenerate case). 2D arrays are interpreted as
            ``(n_points, n_features)``.
        max_dim
            Maximum homology dimension to compute. If ``None`` (default),
            uses ``rules.topology_max_dim``. Method parameter takes
            precedence over rules (fix L1).

        Returns
        -------
        dict with keys ``betti_numbers``, ``persistence_diagram``,
        ``persistence_entropy``, ``euler_characteristic``, ``n_points``,
        ``max_eps``.
        """
        if max_dim is None:
            max_dim = self.rules.topology_max_dim

        # Empty input
        if data is None:
            return self._empty_result()
        arr = np.ascontiguousarray(data, dtype=float)
        if arr.size == 0:
            return self._empty_result()

        # NaN/Inf guard
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        # Reshape to 2D
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)

        n_points = arr.shape[0]

        # Single point: degenerate
        if n_points == 1:
            return {
                "betti_numbers": [1, 0, 0],
                "persistence_diagram": [],
                "persistence_entropy": 0.0,
                "euler_characteristic": 1,
                "n_points": 1,
                "max_eps": 0.0,
            }

        # Subsample if too large (fix C1)
        if n_points > self.rules.topology_max_points:
            k = self.rules.topology_max_points
            idx = self.rng.choice(n_points, size=k, replace=False)
            idx.sort()  # deterministic order
            arr = arr[idx]
            n_points = k

        # Phase D complexity guard: hard cap on n_points when max_dim >= 2.
        # The column reduction is O(n_cols^3 / 64) where n_cols grows like
        # O(n^(max_dim+2)); for max_dim=2 this is O(n^9). Without this cap
        # a 24-point cloud at max_dim=2 hangs for ~9s, n=32 for ~100s.
        # Convert the silent hang into a loud, actionable ValueError.
        if max_dim >= 2 and n_points > self.rules.topology_max_points_high_dim:
            raise ValueError(
                f"PersistentHomologyPerceiver: max_dim={max_dim} with "
                f"n_points={n_points} exceeds the hard cap "
                f"topology_max_points_high_dim="
                f"{self.rules.topology_max_points_high_dim}. "
                f"The boundary-matrix column reduction is O(n^9) at "
                f"max_dim=2: n=16 is ~0.3s, n=24 is ~9s, n=32 is ~100s. "
                f"Either pass max_dim=1 (default; computes betti_0 + "
                f"betti_1 only) or raise "
                f"EmergenceRules.topology_max_points_high_dim after "
                f"understanding the O(n^9) cost."
            )

        # Pairwise distance matrix
        diff = arr[:, None, :] - arr[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1))
        max_eps = float(dist.max())

        # Compute persistent homology via boundary matrix reduction
        betti, diagram = self._persistent_homology(dist, max_dim)

        # Persistence entropy
        entropy = self._persistence_entropy(diagram)

        # Euler characteristic
        euler = sum((-1) ** d * betti[d] for d in range(len(betti)))

        return {
            "betti_numbers": [int(b) for b in betti],
            "persistence_diagram": diagram,
            "persistence_entropy": float(entropy),
            "euler_characteristic": int(euler),
            "n_points": int(n_points),
            "max_eps": float(max_eps),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _empty_result(self) -> dict:
        return {
            "betti_numbers": [0, 0, 0],
            "persistence_diagram": [],
            "persistence_entropy": 0.0,
            "euler_characteristic": 0,
            "n_points": 0,
            "max_eps": 0.0,
        }

    def _persistent_homology(
        self, dist: np.ndarray, max_dim: int
    ) -> tuple[list[int], list[tuple[int, float, float]]]:
        """Compute persistent homology via Vietoris-Rips filtration.

        Builds the full simplicial complex up to dimension ``max_dim + 1``
        and reduces the boundary matrix over GF(2) to find birth/death
        pairs. Returns ``(betti_numbers, persistence_diagram)`` where the
        diagram is a list of ``(dim, birth, death)`` tuples (death may be
        ``inf`` for essential classes).
        """
        n = dist.shape[0]
        if n < 2:
            return [1, 0, 0][: max_dim + 1], []

        # Build simplices with their filtration values (birth eps).
        # simplex_tuple is a sorted tuple of vertex indices.
        # filtration_value = max distance over all pairs in the simplex.
        simplices: list[tuple[tuple[int, ...], float]] = []
        for d in range(min(max_dim + 2, n)):  # dimensions 0 .. max_dim+1
            for combo in combinations(range(n), d + 1):
                # Filtration value = max pairwise distance in the simplex
                if d == 0:
                    val = 0.0
                else:
                    max_d = 0.0
                    for i, j in combinations(combo, 2):
                        dij = float(dist[i, j])
                        if dij > max_d:
                            max_d = dij
                    val = max_d
                simplices.append((combo, val))

        # Sort by filtration value, then by simplex dimension (lower first),
        # then by vertex order — deterministic ordering.
        simplices.sort(key=lambda s: (s[1], len(s[0]) - 1, s[0]))

        # Map simplex -> column index
        index: dict[tuple[int, ...], int] = {
            sxm[0]: i for i, sxm in enumerate(simplices)
        }
        n_cols = len(simplices)

        # Phase D complexity guard: hard cap on total simplex count.
        # The column reduction below is O(n_cols^3 / 64) word ops; n_cols
        # = 5000 ~= 2e9 ops ~= 30s on commodity hardware. This is the
        # ultimate guard against combinatorial explosion regardless of
        # which (n_points, max_dim) combination triggered it -- the
        # n_points cap above only catches max_dim >= 2, but a user could
        # raise both caps and still hit a uselessly-slow regime.
        max_simplices = getattr(self.rules, "topology_max_simplices", 5000)
        if n_cols > max_simplices:
            raise ValueError(
                f"PersistentHomologyPerceiver: built {n_cols} simplices "
                f"from n={n} points at max_dim={max_dim}, exceeding the "
                f"hard cap topology_max_simplices={max_simplices}. "
                f"The column reduction is O(n_cols^3 / 64): "
                f"{n_cols} cols ~= {n_cols**3 / 64:.2e} ops. "
                f"Either lower max_dim, lower n_points (via "
                f"topology_max_points), or raise "
                f"EmergenceRules.topology_max_simplices after "
                f"understanding the cost."
            )

        # Build boundary matrix over GF(2) (as bool ndarray).
        # boundary[i, j] = 1 if simplex i is a face of simplex j (codim 1).
        boundary = np.zeros((n_cols, n_cols), dtype=bool)
        for j, (sxm, _) in enumerate(simplices):
            d = len(sxm) - 1
            if d < 1:
                continue
            # Each face removes one vertex
            for k in range(d + 1):
                face = sxm[:k] + sxm[k + 1:]
                if face in index:
                    i = index[face]
                    boundary[i, j] = True

        # Column reduction over GF(2) (standard left-to-right).
        # Find the lowest row index with a 1 in each column, then reduce.
        low = [-1] * n_cols  # low[j] = row index of lowest 1, or -1
        # Work on a copy of boundary columns (as Python lists for speed).
        cols = [boundary[:, j].copy() for j in range(n_cols)]

        for j in range(n_cols):
            while True:
                low_j = self._low(cols[j])
                if low_j == -1:
                    break
                # Look for another column k < j with the same low
                conflict = -1
                for k in range(j):
                    if low[k] == low_j:
                        conflict = k
                        break
                if conflict == -1:
                    low[j] = low_j
                    break
                # Reduce column j by XOR with column conflict
                cols[j] = cols[j] ^ cols[conflict]
            # Recompute low after reduction
            low[j] = self._low(cols[j])

        # Extract persistence pairs.
        # birth = filtration value of simplex j (creator).
        # death = filtration value of simplex i (destroyer).
        # For each column j with low[j] != -1, the pair is (low[j], j):
        #   simplex low[j] (lower dim) is destroyed by simplex j (higher dim).
        diagram: list[tuple[int, float, float]] = []
        # Count essential classes (no destroyer).
        # For columns with low[j] != -1: pair (low[j], j).
        # For columns with low[j] == -1 AND no column k has low[k] == j:
        #   essential class born at simplex j's filtration value.
        destroyed = set()
        pairs: list[tuple[int, int]] = []
        for j in range(n_cols):
            if low[j] != -1:
                pairs.append((low[j], j))
                destroyed.add(j)
                destroyed.add(low[j])

        # Determine the persistence threshold for "long-lived" features.
        # Finite point clouds sampled from continuous shapes have their
        # topological features eventually die when the VR complex becomes
        # a full simplex (all pairwise distances connect). To recover the
        # underlying shape's Betti numbers, we count both essential classes
        # (death = inf) AND finite pairs whose persistence exceeds half
        # the maximal filtration value. This is the standard "persistent
        # betti number" at eps = 0.5 * max_eps.
        max_filtration = max((s[1] for s in simplices), default=0.0)
        persistence_threshold = 0.5 * max_filtration

        betti = [0] * (max_dim + 1)
        # Betti_d = number of essential d-simplices (not destroyed).
        for j, (sxm, val) in enumerate(simplices):
            d = len(sxm) - 1
            if d > max_dim:
                continue
            if j not in destroyed:
                betti[d] += 1
                diagram.append((d, float(val), float("inf")))

        # Add finite pairs (birth, death) to diagram. Long-lived pairs
        # (persistence >= threshold) also count toward betti_d (this
        # captures the topology of the underlying continuous shape).
        for i, j in pairs:
            d_birth = len(simplices[i][0]) - 1
            # Pair invariant: d_death == d_birth + 1 (not recorded; only
            # d_birth is used to bucket the persistence pair).
            if d_birth <= max_dim:
                birth = float(simplices[i][1])
                death = float(simplices[j][1])
                if death > birth:  # only count non-trivial persistence
                    diagram.append((d_birth, birth, death))
                    if death - birth >= persistence_threshold:
                        betti[d_birth] += 1

        # Pad betti to max_dim + 1
        while len(betti) < max_dim + 1:
            betti.append(0)

        return betti[: max_dim + 1], diagram

    @staticmethod
    def _low(col: np.ndarray) -> int:
        """Return the lowest row index with a 1 in column, or -1."""
        nonzero = np.nonzero(col)[0]
        if nonzero.size == 0:
            return -1
        return int(nonzero[-1])

    @staticmethod
    def _persistence_entropy(
        diagram: list[tuple[int, float, float]]
    ) -> float:
        """Compute persistence entropy.

        ``H = -sum(p_i * log(p_i))`` where ``p_i = (death_i - birth_i) /
        total_persistence``. Essential classes (death = inf) are excluded.
        """
        finite = [
            (d, b, death)
            for d, b, death in diagram
            if death != float("inf") and death > b
        ]
        if not finite:
            return 0.0
        persistences = np.array([death - b for _, b, death in finite])
        total = float(persistences.sum())
        if total < 1e-12:
            return 0.0
        p = persistences / total
        # Shannon entropy, normalized: divide by log(n) so result in [0, 1].
        n = len(p)
        h = -float(np.sum(p * np.log(p + 1e-12)))
        if n > 1:
            h = h / np.log(n)
        return float(np.clip(h, 0.0, 1.0))
