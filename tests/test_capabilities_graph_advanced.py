"""Tests for the Advanced Graph capability domain."""

import numpy as np
import pytest

from zero_data_model.capabilities import (
    DynamicGraphTracker,
    GraphIsomorphismDetector,
    SpanningTreeExtractor,
)


@pytest.fixture(autouse=True)
def _deterministic_rng():
    np.random.seed(42)


def _make_chain(n: int = 6) -> np.ndarray:
    adj = np.zeros((n, n))
    for i in range(n - 1):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0
    return adj


def _make_complete(n: int = 4) -> np.ndarray:
    return np.ones((n, n)) - np.eye(n)


# --------------------------------------------------------------------------- #
# GraphIsomorphismDetector
# --------------------------------------------------------------------------- #


class TestGraphIsomorphismDetector:
    def test_check_returns_expected_dict_keys(self):
        iso = GraphIsomorphismDetector()
        adj = _make_chain(4)
        result = iso.check(adj, adj)
        assert set(result.keys()) == {
            "isomorphic", "confidence", "wl_hash_a", "wl_hash_b"
        }

    def test_check_same_graph_isomorphic(self):
        iso = GraphIsomorphismDetector()
        adj = _make_chain(4)
        result = iso.check(adj, adj)
        assert result["isomorphic"] is True
        assert result["wl_hash_a"] == result["wl_hash_b"]

    def test_check_different_sizes_not_isomorphic(self):
        iso = GraphIsomorphismDetector()
        result = iso.check(_make_chain(4), _make_chain(5))
        assert result["isomorphic"] is False

    def test_check_permuted_graph_isomorphic(self):
        """A permuted adjacency matrix of the same graph is isomorphic."""
        iso = GraphIsomorphismDetector()
        adj = _make_complete(4)
        # Permute: swap nodes 0 and 1.
        perm = np.array([1, 0, 2, 3])
        adj_perm = adj[perm][:, perm]
        result = iso.check(adj, adj_perm)
        assert result["isomorphic"] is True

    def test_check_chain_vs_complete_not_isomorphic(self):
        """A chain and a complete graph of the same size are not isomorphic."""
        iso = GraphIsomorphismDetector()
        result = iso.check(_make_chain(4), _make_complete(4))
        assert result["isomorphic"] is False

    def test_check_confidence_in_unit_interval(self):
        iso = GraphIsomorphismDetector()
        result = iso.check(_make_chain(4), _make_chain(4))
        assert 0.0 <= result["confidence"] <= 1.0


# --------------------------------------------------------------------------- #
# DynamicGraphTracker
# --------------------------------------------------------------------------- #


class TestDynamicGraphTracker:
    def test_track_returns_expected_dict_keys(self):
        dt = DynamicGraphTracker()
        snapshots = [_make_chain(4), _make_chain(4)]
        result = dt.track(snapshots)
        assert set(result.keys()) == {
            "community_drift", "node_migrations", "stability"
        }

    def test_track_identical_snapshots_zero_drift(self):
        """Identical snapshots should show zero drift and stability 1.0."""
        dt = DynamicGraphTracker()
        adj = _make_chain(4)
        result = dt.track([adj, adj, adj])
        assert all(d == 0.0 for d in result["community_drift"])
        assert result["stability"] == 1.0

    def test_track_empty_snapshots(self):
        dt = DynamicGraphTracker()
        result = dt.track([])
        assert result["stability"] == 0.0

    def test_track_single_snapshot(self):
        dt = DynamicGraphTracker()
        result = dt.track([_make_chain(4)])
        assert result["stability"] == 1.0

    def test_track_stability_in_unit_interval(self):
        dt = DynamicGraphTracker()
        adj1 = _make_chain(4)
        adj2 = _make_complete(4)
        result = dt.track([adj1, adj2])
        assert 0.0 <= result["stability"] <= 1.0

    def test_track_drift_length_is_n_snapshots_minus_1(self):
        dt = DynamicGraphTracker()
        adj = _make_chain(4)
        result = dt.track([adj, adj, adj, adj])
        assert len(result["community_drift"]) == 3


# --------------------------------------------------------------------------- #
# SpanningTreeExtractor
# --------------------------------------------------------------------------- #


class TestSpanningTreeExtractor:
    def test_extract_returns_expected_dict_keys(self):
        st = SpanningTreeExtractor()
        result = st.extract(_make_chain(4))
        assert set(result.keys()) == {
            "mst_edges", "total_weight", "mst_adjacency"
        }

    def test_extract_chain_mst(self):
        """A chain graph's MST is the chain itself."""
        st = SpanningTreeExtractor()
        result = st.extract(_make_chain(4))
        assert len(result["mst_edges"]) == 3  # n-1 edges
        assert result["total_weight"] == 3.0

    def test_extract_complete_graph_mst(self):
        """A complete graph's MST has n-1 edges with total weight n-1."""
        st = SpanningTreeExtractor()
        result = st.extract(_make_complete(4))
        assert len(result["mst_edges"]) == 3  # n-1 = 3
        assert result["total_weight"] == 3.0  # unit weights

    def test_extract_empty_graph(self):
        st = SpanningTreeExtractor()
        result = st.extract(np.zeros((0, 0)))
        assert result["mst_edges"] == []
        assert result["total_weight"] == 0.0

    def test_extract_single_node(self):
        st = SpanningTreeExtractor()
        result = st.extract(np.zeros((1, 1)))
        assert result["mst_edges"] == []
        assert result["total_weight"] == 0.0

    def test_extract_mst_adjacency_correct_shape(self):
        st = SpanningTreeExtractor()
        adj = _make_chain(5)
        result = st.extract(adj)
        n = adj.shape[0]
        assert result["mst_adjacency"].shape == (n, n)

    def test_extract_weighted_mst_picks_lowest(self):
        """MST should prefer the lowest-weight edges."""
        st = SpanningTreeExtractor()
        adj = np.array([
            [0, 1.0, 5.0],
            [1.0, 0, 1.0],
            [5.0, 1.0, 0],
        ])
        result = st.extract(adj)
        # MST should pick edges (0,1) and (1,2) with total weight 2.0.
        assert result["total_weight"] == 2.0
        assert (0, 1) in result["mst_edges"] or (1, 0) in result["mst_edges"]
        assert (1, 2) in result["mst_edges"] or (2, 1) in result["mst_edges"]

    def test_extract_mst_no_cycles(self):
        """The MST must not contain cycles (n-1 edges for n nodes)."""
        st = SpanningTreeExtractor()
        adj = _make_complete(5)
        result = st.extract(adj)
        n = adj.shape[0]
        assert len(result["mst_edges"]) == n - 1
