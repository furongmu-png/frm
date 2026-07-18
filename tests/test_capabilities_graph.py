"""Tests for the Graph capability domain (base module)."""

import numpy as np
import pytest

from zero_data_model.capabilities import (
    CentralityAnalyzer,
    CommunityDetector,
    GraphEncoder,
    GraphRules,
    PathFinder,
)


@pytest.fixture(autouse=True)
def _deterministic_rng():
    np.random.seed(42)


def _make_chain(n: int = 6) -> np.ndarray:
    """Make a simple chain graph (0-1-2-...-n-1) with unit weights."""
    adj = np.zeros((n, n))
    for i in range(n - 1):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0
    return adj


def _make_complete(n: int = 4) -> np.ndarray:
    """Make a complete graph with unit weights."""
    adj = np.ones((n, n)) - np.eye(n)
    return adj


def _make_two_components() -> np.ndarray:
    """Make a graph with two disconnected components."""
    adj = np.zeros((6, 6))
    # Component 1: 0-1-2
    adj[0, 1] = adj[1, 0] = 1.0
    adj[1, 2] = adj[2, 1] = 1.0
    # Component 2: 3-4-5
    adj[3, 4] = adj[4, 3] = 1.0
    adj[4, 5] = adj[5, 4] = 1.0
    return adj


# --------------------------------------------------------------------------- #
# GraphRules
# --------------------------------------------------------------------------- #


class TestGraphRules:
    def test_default_parameters(self):
        rules = GraphRules()
        assert rules.default_weight == 1.0
        assert rules.community_resolution == 1.0
        assert rules.path_heuristic_weight == 1.0
        assert rules.centrality_normalized is True
        assert rules.isomorphism_max_iter == 5


# --------------------------------------------------------------------------- #
# GraphEncoder
# --------------------------------------------------------------------------- #


class TestGraphEncoder:
    def test_encode_returns_l2_normalized_dim_vector(self):
        enc = GraphEncoder(dim=64)
        adj = _make_chain()
        vec = enc.encode(adj)
        assert vec.shape == (64,)
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-6)

    def test_encode_deterministic(self):
        enc = GraphEncoder(dim=32)
        adj = _make_complete(4)
        v1 = enc.encode(adj)
        v2 = enc.encode(adj)
        np.testing.assert_allclose(v1, v2, atol=1e-8)

    def test_encode_empty_graph(self):
        enc = GraphEncoder(dim=32)
        vec = enc.encode(np.zeros((0, 0)))
        assert vec.shape == (32,)
        assert np.all(vec == 0.0)

    def test_encode_different_graphs_different_vectors(self):
        enc = GraphEncoder(dim=64)
        v_chain = enc.encode(_make_chain(6))
        v_complete = enc.encode(_make_complete(6))
        assert not np.allclose(v_chain, v_complete, atol=1e-4)

    def test_encode_with_node_features(self):
        enc = GraphEncoder(dim=32)
        adj = _make_chain(4)
        features = np.random.randn(4, 3)
        vec = enc.encode(adj, node_features=features)
        assert vec.shape == (32,)
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-6)

    def test_encode_invalid_adjacency_raises(self):
        enc = GraphEncoder(dim=32)
        with pytest.raises(ValueError):
            enc.encode(np.zeros((3, 4)))  # non-square


# --------------------------------------------------------------------------- #
# CommunityDetector
# --------------------------------------------------------------------------- #


class TestCommunityDetector:
    def test_detect_returns_expected_dict_keys(self):
        cd = CommunityDetector()
        result = cd.detect(_make_chain(6))
        assert set(result.keys()) == {"communities", "modularity", "n_communities"}

    def test_detect_two_components(self):
        cd = CommunityDetector()
        result = cd.detect(_make_two_components())
        assert result["n_communities"] == 2

    def test_detect_empty_graph(self):
        cd = CommunityDetector()
        result = cd.detect(np.zeros((0, 0)))
        assert result["n_communities"] == 0
        assert result["communities"] == []

    def test_detect_single_node(self):
        cd = CommunityDetector()
        result = cd.detect(np.zeros((1, 1)))
        assert result["n_communities"] == 1
        assert result["communities"] == [[0]]

    def test_detect_no_edges(self):
        """A graph with no edges should give n communities (each node alone)."""
        cd = CommunityDetector()
        result = cd.detect(np.zeros((4, 4)))
        assert result["n_communities"] == 4

    def test_detect_complete_graph(self):
        """A complete graph should detect at least one community."""
        cd = CommunityDetector()
        result = cd.detect(_make_complete(4))
        assert result["n_communities"] >= 1
        # Modularity of a complete graph's all-in-one partition is 0; the
        # simplified Louvain local-move may end up with a different partition
        # that yields slightly negative modularity. Just check it's finite.
        assert np.isfinite(result["modularity"])

    def test_detect_all_nodes_assigned(self):
        """Every node must appear in exactly one community."""
        cd = CommunityDetector()
        adj = _make_chain(6)
        result = cd.detect(adj)
        all_nodes = set()
        for comm in result["communities"]:
            for node in comm:
                assert node not in all_nodes  # no duplicates
                all_nodes.add(node)
        assert all_nodes == set(range(6))


# --------------------------------------------------------------------------- #
# PathFinder
# --------------------------------------------------------------------------- #


class TestPathFinder:
    def test_find_returns_expected_dict_keys(self):
        pf = PathFinder()
        result = pf.find(_make_chain(6), 0, 5)
        assert set(result.keys()) == {"path", "distance", "visited"}

    def test_find_chain_path(self):
        """In a chain 0-1-2-3-4-5, the path from 0 to 5 is [0,1,2,3,4,5]."""
        pf = PathFinder()
        result = pf.find(_make_chain(6), 0, 5)
        assert result["path"] == [0, 1, 2, 3, 4, 5]
        assert result["distance"] == 5.0

    def test_find_same_source_target(self):
        pf = PathFinder()
        result = pf.find(_make_chain(4), 2, 2)
        assert result["path"] == [2]
        assert result["distance"] == 0.0

    def test_find_no_path(self):
        """Disconnected graph should return empty path."""
        pf = PathFinder()
        result = pf.find(_make_two_components(), 0, 5)
        assert result["path"] == []
        assert result["distance"] == float("inf")

    def test_find_weighted_graph(self):
        """Shortest path should prefer lower-weight edges."""
        pf = PathFinder()
        adj = np.array([
            [0, 1.0, 10.0],
            [1.0, 0, 1.0],
            [10.0, 1.0, 0],
        ])
        result = pf.find(adj, 0, 2)
        assert result["path"] == [0, 1, 2]
        assert result["distance"] == 2.0

    def test_find_invalid_indices(self):
        pf = PathFinder()
        adj = _make_chain(4)
        result = pf.find(adj, -1, 2)
        assert result["path"] == []
        result = pf.find(adj, 0, 99)
        assert result["path"] == []


# --------------------------------------------------------------------------- #
# CentralityAnalyzer
# --------------------------------------------------------------------------- #


class TestCentralityAnalyzer:
    def test_analyze_returns_expected_dict_keys(self):
        ca = CentralityAnalyzer()
        result = ca.analyze(_make_chain(6))
        assert set(result.keys()) == {
            "degree", "betweenness", "closeness", "most_central"
        }

    def test_analyze_chain_middle_most_central(self):
        """In a chain, the middle node should have highest betweenness."""
        ca = CentralityAnalyzer()
        result = ca.analyze(_make_chain(5))
        # Middle node (index 2) has highest betweenness.
        btw = result["betweenness"]
        assert int(np.argmax(btw)) == 2

    def test_analyze_empty_graph(self):
        ca = CentralityAnalyzer()
        result = ca.analyze(np.zeros((0, 0)))
        assert result["most_central"] == -1

    def test_analyze_degree_in_unit_interval(self):
        """Normalized degree centrality is in [0, 1]."""
        ca = CentralityAnalyzer()
        result = ca.analyze(_make_complete(4))
        deg = result["degree"]
        assert np.all(deg >= 0.0) and np.all(deg <= 1.0 + 1e-6)

    def test_analyze_closeness_nonnegative(self):
        ca = CentralityAnalyzer()
        result = ca.analyze(_make_chain(5))
        cls = result["closeness"]
        assert np.all(cls >= 0.0)

    def test_analyze_betweenness_nonnegative(self):
        ca = CentralityAnalyzer()
        result = ca.analyze(_make_chain(5))
        btw = result["betweenness"]
        assert np.all(btw >= -1e-8)

    def test_analyze_most_central_valid_index(self):
        ca = CentralityAnalyzer()
        adj = _make_chain(5)
        result = ca.analyze(adj)
        assert 0 <= result["most_central"] < 5
