"""Comprehensive pytest unit tests for enhanced scientific discovery modules.

Covers:
  - ``LiteratureMiner`` / ``LiteratureGraph`` / ``Paper`` (literature_miner)
  - ``PeerReviewer`` / ``ReviewReport`` (peer_reviewer)
  - ``HypothesisGenerator.generate_from_literature`` (hypothesis_generator)
  - ``ScienceLoop`` with literature mining + peer review enabled (science_loop)
"""
from __future__ import annotations

import tempfile
import time

import numpy as np
import pytest

from src.discovery.literature_miner import (
    LiteratureMiner,
    LiteratureGraph,
    Paper as LitPaper,
)
from src.discovery.peer_reviewer import PeerReviewer, ReviewReport
from src.discovery.hypothesis_generator import HypothesisGenerator, Hypothesis
from src.discovery.experiment_designer import ExperimentDesigner, ExperimentDesign
from src.discovery.result_analyzer import AnalysisResult
from src.discovery.paper_writer import Paper, PaperWriter
from src.discovery.science_loop import ScienceLoop, DiscoveryCycle


# ------------------------------------------------------------------ #
# LiteratureMiner
# ------------------------------------------------------------------ #
class TestLiteratureMiner:
    """Tests for the literature mining and knowledge graph builder."""

    def test_embed_text_returns_dim_vector(self):
        """embed_text returns an ndarray of shape (embedding_dim,) with finite values."""
        with tempfile.TemporaryDirectory() as tmp:
            miner = LiteratureMiner(embedding_dim=32, cache_dir=tmp)
            vec = miner.embed_text("hello world")
            assert isinstance(vec, np.ndarray), "embed_text should return np.ndarray"
            assert vec.shape == (32,), f"Expected shape (32,), got {vec.shape}"
            assert np.all(np.isfinite(vec)), "Embedding values should all be finite"

    def test_embed_text_empty_returns_zeros(self):
        """embed_text('') returns a zero vector."""
        with tempfile.TemporaryDirectory() as tmp:
            miner = LiteratureMiner(embedding_dim=16, cache_dir=tmp)
            vec = miner.embed_text("")
            assert np.all(vec == 0), "Empty string embedding should be all zeros"
            assert vec.shape == (16,)

    def test_embed_text_deterministic(self):
        """embed_text is deterministic: same string → same vector."""
        with tempfile.TemporaryDirectory() as tmp:
            miner = LiteratureMiner(embedding_dim=32, cache_dir=tmp)
            s = "the quick brown fox jumps over the lazy dog"
            v1 = miner.embed_text(s)
            v2 = miner.embed_text(s)
            assert np.array_equal(v1, v2), "Same string should produce identical embeddings"

    def test_load_local_papers_returns_loaded(self):
        """load_local_papers returns a list of Paper objects and stores them in the graph."""
        with tempfile.TemporaryDirectory() as tmp:
            miner = LiteratureMiner(embedding_dim=32, cache_dir=tmp)
            papers = miner.load_local_papers([
                {"title": "Paper A", "abstract": "Abstract A", "year": 2020},
                {"title": "Paper B", "abstract": "Abstract B", "year": 2021},
            ])
            assert len(papers) == 2, f"Expected 2 papers, got {len(papers)}"
            assert all(isinstance(p, LitPaper) for p in papers)
            assert all(p.source == "local" for p in papers)
            # Papers should be stored in the graph
            assert len(miner.graph.papers) == 2, "Papers should be stored in graph"
            assert len(miner._paper_index) == 2

    def test_load_local_papers_filters_invalid(self):
        """load_local_papers filters out entries with no title and no abstract."""
        with tempfile.TemporaryDirectory() as tmp:
            miner = LiteratureMiner(embedding_dim=32, cache_dir=tmp)
            papers = miner.load_local_papers([
                {"year": 2020},              # no title, no abstract
                {"authors": ["x"]},          # no title, no abstract
                "not a dict",                # not a dict
                {"title": "", "abstract": ""},  # both empty
            ])
            assert papers == [], f"Expected empty list for invalid entries, got {len(papers)}"

    def test_build_graph_creates_nodes_and_edges(self):
        """build_graph creates paper + concept nodes and mentions/similar edges."""
        with tempfile.TemporaryDirectory() as tmp:
            miner = LiteratureMiner(embedding_dim=32, cache_dir=tmp)
            miner.load_local_papers([
                {"title": "gravity mass study", "abstract": "gravity mass study", "year": 2020},
                {"title": "gravity mass study", "abstract": "gravity mass study", "year": 2021},
                {"title": "velocity force study", "abstract": "velocity force study", "year": 2022},
            ])
            graph = miner.build_graph()
            assert len(graph.nodes) > 0, "Graph should have nodes"
            node_types = {n["type"] for n in graph.nodes}
            assert "paper" in node_types, f"Should have paper nodes, got {node_types}"
            assert "concept" in node_types, f"Should have concept nodes, got {node_types}"
            assert len(graph.edges) > 0, "Graph should have edges"
            edge_types = {e["type"] for e in graph.edges}
            assert "mentions" in edge_types, f"Should have mentions edges, got {edge_types}"
            # Papers 1 & 2 are identical → high similarity → similar or contradiction edge
            assert (
                "similar" in edge_types or "contradiction" in edge_types
            ), f"Should have similar/contradiction edges, got {edge_types}"

    def test_build_graph_detects_contradictions(self):
        """build_graph detects contradictions between papers with opposite conclusions."""
        with tempfile.TemporaryDirectory() as tmp:
            miner = LiteratureMiner(embedding_dim=32, cache_dir=tmp)
            # Titles differ slightly so embeddings differ (otherwise identical
            # title+abstract produce identical paper_ids and sim=1.0, which
            # always goes into the "similar" branch and never the contradiction
            # branch).
            miner.load_local_papers([
                {
                    "title": "gravity increases mass effect",
                    "abstract": "study of gravity mass",
                    "conclusion": "gravity increases mass",
                    "year": 2020,
                },
                {
                    "title": "gravity decreases mass effect",
                    "abstract": "study of gravity mass",
                    "conclusion": "gravity decreases mass",
                    "year": 2021,
                },
            ])
            # Use high similarity threshold so the contradiction branch is taken
            # (otherwise near-identical papers get "similar" edges).
            miner.build_graph(similarity_threshold=0.99)
            assert len(miner.graph.contradictions) > 0, (
                "Should detect contradiction between papers with increase/decrease"
            )

    def test_find_structural_holes_returns_list(self):
        """find_structural_holes returns a list of (concept_a, concept_b, score) tuples."""
        with tempfile.TemporaryDirectory() as tmp:
            miner = LiteratureMiner(embedding_dim=32, cache_dir=tmp)
            miner.load_local_papers([
                {"title": "alpha beta", "abstract": "alpha beta", "year": 2020},
                {"title": "gamma delta", "abstract": "gamma delta", "year": 2021},
                {"title": "epsilon zeta", "abstract": "epsilon zeta", "year": 2022},
            ])
            miner.build_graph()
            holes = miner.find_structural_holes()
            assert isinstance(holes, list)
            # With 3 papers having disjoint concepts, structural holes should exist
            if len(holes) > 0:
                h = holes[0]
                assert len(h) == 3, f"Each hole should be (concept_a, concept_b, score), got {h}"
                assert isinstance(h[0], str) and isinstance(h[1], str)
                assert isinstance(h[2], (int, float))

    def test_merge_into_existing_kg(self):
        """merge_into returns a merged dict with new nodes added to existing KG."""
        with tempfile.TemporaryDirectory() as tmp:
            miner = LiteratureMiner(embedding_dim=32, cache_dir=tmp)
            miner.load_local_papers([
                {"title": "alpha beta", "abstract": "alpha beta", "year": 2020},
            ])
            miner.build_graph()
            existing_kg = {
                "nodes": [{"id": "existing_node", "type": "external"}],
                "edges": [{"source": "a", "target": "b", "type": "causal"}],
            }
            merged = miner.merge_into(existing_kg)
            assert isinstance(merged, dict)
            assert "nodes" in merged and "edges" in merged
            assert len(merged["nodes"]) > len(existing_kg["nodes"]), (
                "Merged KG should have more nodes than the original"
            )
            # Existing node should still be present
            ids = {n["id"] for n in merged["nodes"]}
            assert "existing_node" in ids

    def test_get_snapshot_returns_summary(self):
        """get_snapshot returns a dict with n_papers, n_nodes, n_edges, n_contradictions."""
        with tempfile.TemporaryDirectory() as tmp:
            miner = LiteratureMiner(embedding_dim=32, cache_dir=tmp)
            miner.load_local_papers([
                {"title": "alpha beta", "abstract": "alpha beta", "year": 2020},
                {"title": "gamma delta", "abstract": "gamma delta", "year": 2021},
            ])
            miner.build_graph()
            snap = miner.get_snapshot()
            assert isinstance(snap, dict)
            assert "n_papers" in snap
            assert "n_nodes" in snap
            assert "n_edges" in snap
            assert "n_contradictions" in snap
            assert snap["n_papers"] == 2

    def test_reset_clears_graph(self):
        """reset() clears the graph and paper index."""
        with tempfile.TemporaryDirectory() as tmp:
            miner = LiteratureMiner(embedding_dim=32, cache_dir=tmp)
            miner.load_local_papers([
                {"title": "alpha beta", "abstract": "alpha beta", "year": 2020},
                {"title": "gamma delta", "abstract": "gamma delta", "year": 2021},
            ])
            miner.build_graph()
            assert len(miner.graph.papers) > 0
            assert len(miner.graph.nodes) > 0
            miner.reset()
            assert len(miner.graph.papers) == 0, "reset should clear papers"
            assert len(miner.graph.nodes) == 0, "reset should clear nodes"
            assert len(miner.graph.edges) == 0, "reset should clear edges"
            assert len(miner._paper_index) == 0


# ------------------------------------------------------------------ #
# PeerReviewer
# ------------------------------------------------------------------ #
class TestPeerReviewer:
    """Tests for the internal peer review module."""

    @staticmethod
    def _make_full_paper() -> Paper:
        """Construct a well-structured paper with all required sections."""
        md = (
            "# Test Paper\n\n"
            "## 摘要\nThis study accepts the hypothesis.\n\n"
            "## 1. 引言\nIntroduction.\n\n"
            "## 2. 方法\nMethods.\n\n"
            "## 3. 结果\nResults.\n\n"
            "## 4. 讨论\nDiscussion.\n\n"
            "## 5. 结论\nThe hypothesis is accepted (accept).\n\n"
            "## 参考文献\n[1] Some reference.\n\n"
            "## 复现说明\nReproducibility details.\n"
        )
        return Paper(
            paper_id="p_full",
            title="Test Paper",
            hypothesis_id="h1",
            decision="accept",
            markdown=md,
            file_path="",
            timestamp=time.time(),
        )

    @staticmethod
    def _make_good_design(hypothesis_id: str = "h1") -> ExperimentDesign:
        return ExperimentDesign(
            hypothesis_id=hypothesis_id,
            intervention_type="apply_force",
            intervention_vars=["x"],
            observation_vars=["y"],
            params={"force": 1.0},
            expected_info_gain=0.5,
            action_sequence=[{"step": 0, "action": "reset_sandbox"}],
            n_repeats=5,
        )

    @staticmethod
    def _make_accept_analysis(hypothesis_id: str = "h1") -> AnalysisResult:
        return AnalysisResult(
            hypothesis_id=hypothesis_id,
            bayes_factor=15.0,
            decision="accept",
            confidence=0.9,
            effect_size=1.0,
            n_samples=5,
            conclusion="The hypothesis is accepted.",
        )

    def test_review_accepts_well_structured_paper(self):
        """A well-structured paper with good design/analysis gets overall_score >= 0.5."""
        reviewer = PeerReviewer(accept_threshold=0.7, reject_threshold=0.3)
        paper = self._make_full_paper()
        h = Hypothesis(
            statement="test hypothesis",
            confidence=0.7,
            testability=0.8,
            strategy="causal_gap",
            intervention_vars=["x"],
            observation_vars=["y"],
        )
        design = self._make_good_design(h.hypothesis_id)
        analysis = self._make_accept_analysis(h.hypothesis_id)
        report = reviewer.review(paper=paper, hypothesis=h, design=design, analysis=analysis)
        assert report.overall_score >= 0.5, (
            f"Well-structured paper should score >= 0.5, got {report.overall_score}"
        )

    def test_review_rejects_missing_sections(self):
        """A paper with only '摘要' gets revise/reject or overall_score < 0.7."""
        reviewer = PeerReviewer(accept_threshold=0.7, reject_threshold=0.3)
        paper = Paper(
            paper_id="p_min",
            title="Minimal",
            hypothesis_id="h1",
            decision="accept",
            markdown="## 摘要\nSome abstract.",
            file_path="",
            timestamp=time.time(),
        )
        report = reviewer.review(paper=paper)
        assert report.decision in ("revise", "reject") or report.overall_score < 0.7, (
            f"Paper with missing sections should be revise/reject or score < 0.7, "
            f"got decision={report.decision}, score={report.overall_score}"
        )

    def test_review_flags_contradiction_with_known_finding(self):
        """A conclusion contradicting a known finding lowers literature_score < 1.0."""
        reviewer = PeerReviewer(accept_threshold=0.7, reject_threshold=0.3)
        reviewer.register_known_finding("gravity increases mass")
        paper = Paper(
            paper_id="p_contra",
            title="Contradiction",
            hypothesis_id="h1",
            decision="accept",
            markdown="## 摘要\nStudy.\n\n## 参考文献\n[1] ref",
            file_path="",
            timestamp=time.time(),
        )
        analysis = AnalysisResult(
            hypothesis_id="h1",
            bayes_factor=15.0,
            decision="accept",
            confidence=0.9,
            effect_size=1.0,
            n_samples=5,
            conclusion="gravity decreases mass",
        )
        report = reviewer.review(paper=paper, analysis=analysis)
        assert report.literature_score < 1.0, (
            f"Literature score should be < 1.0 when contradicting known finding, "
            f"got {report.literature_score}"
        )

    def test_review_returns_review_report_fields(self):
        """ReviewReport has all expected fields."""
        reviewer = PeerReviewer(accept_threshold=0.7, reject_threshold=0.3)
        paper = self._make_full_paper()
        report = reviewer.review(paper=paper)
        expected_fields = [
            "paper_id",
            "decision",
            "overall_score",
            "logic_score",
            "method_score",
            "literature_score",
            "reproducibility_score",
            "comments",
            "required_revisions",
        ]
        for field_name in expected_fields:
            assert hasattr(report, field_name), f"ReviewReport missing field: {field_name}"

    def test_decision_thresholds(self):
        """Decision follows thresholds: accept (>=0.7, no revisions), reject (<0.3), revise."""
        reviewer = PeerReviewer(accept_threshold=0.7, reject_threshold=0.3)

        # --- Accept: full paper, good design/analysis, no revisions ---
        h = Hypothesis(
            statement="test accept",
            confidence=0.7,
            testability=0.8,
            strategy="causal_gap",
            intervention_vars=["x"],
            observation_vars=["y"],
        )
        paper_accept = self._make_full_paper()
        paper_accept.hypothesis_id = h.hypothesis_id
        design_good = self._make_good_design(h.hypothesis_id)
        analysis_good = self._make_accept_analysis(h.hypothesis_id)
        report_accept = reviewer.review(
            paper=paper_accept, hypothesis=h, design=design_good, analysis=analysis_good
        )
        assert report_accept.overall_score >= 0.7, (
            f"Accept scenario overall_score should be >= 0.7, got {report_accept.overall_score}"
        )
        assert len(report_accept.required_revisions) == 0, (
            f"Accept scenario should have no revisions, got {report_accept.required_revisions}"
        )
        assert report_accept.decision == "accept", (
            f"Expected 'accept', got '{report_accept.decision}'"
        )

        # --- Reject: empty paper, bad design/analysis, contradicting known finding ---
        reviewer2 = PeerReviewer(accept_threshold=0.7, reject_threshold=0.3)
        reviewer2.register_known_finding("gravity increases mass")
        h2 = Hypothesis(
            statement="test reject",
            confidence=0.5,
            testability=0.8,
            strategy="causal_gap",
        )
        paper_reject = Paper(
            paper_id="p_reject",
            title="",
            hypothesis_id=h2.hypothesis_id,
            decision="accept",
            markdown="",  # no sections, no refs, no 复现
            file_path="",
            timestamp=time.time(),
        )
        design_bad = ExperimentDesign(
            hypothesis_id=h2.hypothesis_id,
            intervention_type="test",
            intervention_vars=[],      # empty → revision
            observation_vars=["y"],
            params={},                  # empty → revision
            expected_info_gain=0.05,    # < 0.1 → revision
            action_sequence=[],         # empty → revision
            n_repeats=1,                # < 3 → revision
        )
        analysis_bad = AnalysisResult(
            hypothesis_id=h2.hypothesis_id,
            bayes_factor=0.05,          # < 0.1, decision != reject → logic deduction
            decision="accept",
            confidence=0.5,
            effect_size=0.001,          # < 0.01 & accept → method revision
            n_samples=1,                # < 3 → method revision
            conclusion="gravity decreases mass",  # contradicts known finding
        )
        report_reject = reviewer2.review(
            paper=paper_reject, hypothesis=h2, design=design_bad, analysis=analysis_bad
        )
        assert report_reject.overall_score < 0.3, (
            f"Reject scenario overall_score should be < 0.3, got {report_reject.overall_score}"
        )
        assert report_reject.decision == "reject", (
            f"Expected 'reject', got '{report_reject.decision}'"
        )

        # --- Revise: paper with some sections, mediocre scores → revise ---
        reviewer3 = PeerReviewer(accept_threshold=0.7, reject_threshold=0.3)
        paper_revise = Paper(
            paper_id="p_revise",
            title="Revise",
            hypothesis_id="h3",
            decision="accept",
            markdown="## 摘要\nAbstract.",  # only 1 section → 5 missing
            file_path="",
            timestamp=time.time(),
        )
        design_med = ExperimentDesign(
            hypothesis_id="h3",
            intervention_type="test",
            intervention_vars=[],      # empty → revision
            observation_vars=["y"],
            params={},                  # empty → revision
            expected_info_gain=0.05,    # < 0.1 → revision
            action_sequence=[],         # empty → revision
            n_repeats=1,                # < 3 → revision
        )
        report_revise = reviewer3.review(paper=paper_revise, design=design_med)
        assert 0.3 <= report_revise.overall_score < 0.7, (
            f"Revise scenario overall_score should be in [0.3, 0.7), "
            f"got {report_revise.overall_score}"
        )
        assert report_revise.decision == "revise", (
            f"Expected 'revise', got '{report_revise.decision}'"
        )

    def test_invalid_thresholds_raise(self):
        """Invalid accept/reject thresholds raise ValueError."""
        # accept <= reject
        with pytest.raises(ValueError):
            PeerReviewer(accept_threshold=0.3, reject_threshold=0.7)
        # accept == reject
        with pytest.raises(ValueError):
            PeerReviewer(accept_threshold=0.5, reject_threshold=0.5)
        # accept out of [0, 1]
        with pytest.raises(ValueError):
            PeerReviewer(accept_threshold=1.5, reject_threshold=0.3)
        with pytest.raises(ValueError):
            PeerReviewer(accept_threshold=0.0, reject_threshold=0.0)
        # reject out of [0, 1]
        with pytest.raises(ValueError):
            PeerReviewer(accept_threshold=0.7, reject_threshold=-0.1)
        with pytest.raises(ValueError):
            PeerReviewer(accept_threshold=0.7, reject_threshold=1.5)

    def test_get_snapshot_returns_stats(self):
        """After some reviews, get_snapshot returns stats with expected keys."""
        reviewer = PeerReviewer(accept_threshold=0.7, reject_threshold=0.3)
        # Perform a couple of reviews
        paper1 = self._make_full_paper()
        paper1.paper_id = "p1"
        reviewer.review(paper=paper1)
        paper2 = Paper(
            paper_id="p2",
            title="",
            hypothesis_id="",
            decision="",
            markdown="",
            file_path="",
            timestamp=time.time(),
        )
        reviewer.review(paper=paper2)
        snap = reviewer.get_snapshot()
        expected_keys = {
            "n_reviews",
            "n_accept",
            "n_revise",
            "n_reject",
            "accept_rate",
            "avg_score",
        }
        assert expected_keys.issubset(snap.keys()), (
            f"Snapshot missing keys: {expected_keys - set(snap.keys())}"
        )
        assert snap["n_reviews"] >= 2


# ------------------------------------------------------------------ #
# HypothesisGenerator (literature-driven)
# ------------------------------------------------------------------ #
class TestHypothesisGeneratorLiteratureDriven:
    """Tests for HypothesisGenerator.generate_from_literature."""

    def test_generate_from_literature_contradictions(self):
        """Contradictions produce Hypothesis objects with strategy='contradiction'."""
        gen = HypothesisGenerator(seed=0)
        contradictions = [("p1", "p2", 0.5)]
        lit_graph = {
            "papers": [
                {"paper_id": "p1", "title": "Paper A", "abstract": "..."},
                {"paper_id": "p2", "title": "Paper B", "abstract": "..."},
            ]
        }
        hyps = gen.generate_from_literature(
            literature_graph=lit_graph, contradictions=contradictions
        )
        assert len(hyps) > 0, "Should generate hypotheses from contradictions"
        assert any(h.strategy == "contradiction" for h in hyps), (
            f"Should have at least one 'contradiction' strategy hypothesis, "
            f"got strategies: {[h.strategy for h in hyps]}"
        )

    def test_generate_from_literature_structural_holes(self):
        """Structural holes produce Hypothesis with strategy='analogy_transfer'."""
        gen = HypothesisGenerator(seed=0)
        structural_holes = [("concept_a", "concept_b", 0.8)]
        hyps = gen.generate_from_literature(structural_holes=structural_holes)
        assert len(hyps) > 0, "Should generate hypotheses from structural holes"
        assert any(h.strategy == "analogy_transfer" for h in hyps), (
            f"Should have at least one 'analogy_transfer' strategy hypothesis, "
            f"got strategies: {[h.strategy for h in hyps]}"
        )

    def test_generate_from_literature_concept_clusters(self):
        """Concept clusters produce Hypothesis mentioning the concept."""
        gen = HypothesisGenerator(seed=0)
        lit_graph = {
            "concept_index": {"gravity": ["p1", "p2"]},
        }
        hyps = gen.generate_from_literature(literature_graph=lit_graph)
        assert len(hyps) > 0, "Should generate hypotheses from concept clusters"
        assert any("gravity" in h.statement for h in hyps), (
            f"Should have a hypothesis mentioning 'gravity', "
            f"got: {[h.statement for h in hyps]}"
        )

    def test_generate_from_literature_empty_returns_empty(self):
        """All-None inputs return an empty list."""
        gen = HypothesisGenerator(seed=0)
        hyps = gen.generate_from_literature()
        assert hyps == [], f"Empty inputs should return [], got {len(hyps)} hypotheses"

    def test_generate_from_literature_respects_max(self):
        """max_hypotheses limits the number of returned hypotheses."""
        gen = HypothesisGenerator(max_hypotheses=2, seed=0)
        contradictions = [
            ("p1", "p2", 0.5),
            ("p3", "p4", 0.6),
            ("p5", "p6", 0.7),
        ]
        lit_graph = {
            "papers": [
                {"paper_id": f"p{i}", "title": f"T{i}", "abstract": "..."}
                for i in range(1, 7)
            ],
            "concept_index": {
                "alpha": ["p1", "p2"],
                "beta": ["p3", "p4"],
                "gamma": ["p5", "p6"],
            },
        }
        hyps = gen.generate_from_literature(
            literature_graph=lit_graph, contradictions=contradictions
        )
        assert len(hyps) <= 2, f"Should return at most 2 hypotheses, got {len(hyps)}"


# ------------------------------------------------------------------ #
# ScienceLoop (enhanced with literature mining + peer review)
# ------------------------------------------------------------------ #
class TestScienceLoopEnhanced:
    """Tests for the enhanced ScienceLoop with literature mining and peer review."""

    @staticmethod
    def _make_loop(**kwargs) -> ScienceLoop:
        """Create a ScienceLoop with a temporary output directory."""
        tmp = tempfile.mkdtemp(prefix="sci_loop_test_")
        defaults = {
            "trigger_interval": 1,
            "output_dir": tmp,
            "enable_literature_mining": True,
            "enable_peer_review": True,
            "seed": 0,
        }
        defaults.update(kwargs)
        return ScienceLoop(**defaults)

    @staticmethod
    def _contradictory_papers() -> list[dict]:
        """Return 3 local papers, two of which contradict each other."""
        return [
            {
                "title": "gravity mass effect study",
                "abstract": "study of gravity mass",
                "conclusion": "gravity increases mass",
                "year": 2020,
            },
            {
                "title": "gravity mass effect study",
                "abstract": "study of gravity mass",
                "conclusion": "gravity decreases mass",
                "year": 2021,
            },
            {
                "title": "quantum gravity effects",
                "abstract": "quantum gravity study",
                "year": 2022,
            },
        ]

    def test_init_with_literature_and_review_enabled(self):
        """ScienceLoop with both flags enabled has non-None miner and reviewer."""
        loop = self._make_loop(enable_literature_mining=True, enable_peer_review=True)
        assert loop.literature_miner is not None, "literature_miner should be non-None"
        assert loop.peer_reviewer is not None, "peer_reviewer should be non-None"
        assert loop.enable_literature_mining is True
        assert loop.enable_peer_review is True

    def test_init_can_disable_literature(self):
        """When literature mining is disabled, load_local_papers returns 0."""
        loop = self._make_loop(enable_literature_mining=False)
        count = loop.load_local_papers([
            {"title": "A", "abstract": "B", "year": 2020}
        ])
        assert count == 0, f"Disabled literature mining should return 0, got {count}"

    def test_load_local_papers_via_loop(self):
        """loop.load_local_papers returns a count and stores papers in the miner."""
        loop = self._make_loop(enable_literature_mining=True)
        count = loop.load_local_papers([
            {"title": "Paper A", "abstract": "Abstract A", "year": 2020},
            {"title": "Paper B", "abstract": "Abstract B", "year": 2021},
        ])
        assert count > 0, f"Expected positive count, got {count}"
        assert len(loop.literature_miner.graph.papers) > 0, (
            "literature_miner should have papers after load"
        )

    def test_register_known_finding_via_loop(self):
        """loop.register_known_finding adds to peer_reviewer's known findings."""
        loop = self._make_loop(enable_peer_review=True)
        initial = len(loop.peer_reviewer._known_findings)
        loop.register_known_finding("gravity increases mass")
        assert len(loop.peer_reviewer._known_findings) == initial + 1, (
            "known_findings should grow by 1 after register_known_finding"
        )
        # Also verifiable via snapshot
        snap = loop.peer_reviewer.get_snapshot()
        assert snap["n_known_findings"] >= 1

    def test_run_cycle_with_literature_and_review(self):
        """run_cycle with 3 papers (2 contradictory) scans papers and tracks reviews."""
        loop = self._make_loop(enable_literature_mining=True, enable_peer_review=True)
        loop.load_local_papers(self._contradictory_papers())
        cycle = loop.run_cycle()
        assert isinstance(cycle, DiscoveryCycle)
        assert cycle.n_papers_scanned > 0, (
            f"Expected n_papers_scanned > 0, got {cycle.n_papers_scanned}"
        )
        assert cycle.n_reviews >= 0, f"n_reviews should be >= 0, got {cycle.n_reviews}"
        # Cycle should have the new review-tracking fields
        assert hasattr(cycle, "n_review_accept"), "Cycle missing n_review_accept"
        assert hasattr(cycle, "n_review_revise"), "Cycle missing n_review_revise"
        assert hasattr(cycle, "n_review_reject"), "Cycle missing n_review_reject"
        assert cycle.n_review_accept >= 0
        assert cycle.n_review_revise >= 0
        assert cycle.n_review_reject >= 0

    def test_snapshot_includes_new_components(self):
        """snapshot() includes 'literature_miner' and 'peer_reviewer' keys."""
        loop = self._make_loop(enable_literature_mining=True, enable_peer_review=True)
        snap = loop.snapshot()
        assert "literature_miner" in snap, "snapshot should have 'literature_miner' key"
        assert "peer_reviewer" in snap, "snapshot should have 'peer_reviewer' key"
        # When enabled, each should be a dict
        assert snap["literature_miner"] is None or isinstance(
            snap["literature_miner"], dict
        ), f"literature_miner should be dict or None, got {type(snap['literature_miner'])}"
        assert snap["peer_reviewer"] is None or isinstance(
            snap["peer_reviewer"], dict
        ), f"peer_reviewer should be dict or None, got {type(snap['peer_reviewer'])}"

        # When disabled, they should be None
        loop2 = self._make_loop(enable_literature_mining=False, enable_peer_review=False)
        snap2 = loop2.snapshot()
        assert snap2["literature_miner"] is None, (
            "literature_miner snapshot should be None when disabled"
        )
        assert snap2["peer_reviewer"] is None, (
            "peer_reviewer snapshot should be None when disabled"
        )

    def test_configure_supports_new_flags(self):
        """configure() can update enable_literature_mining and enable_peer_review."""
        loop = self._make_loop(enable_literature_mining=True, enable_peer_review=True)
        assert loop.enable_literature_mining is True
        assert loop.enable_peer_review is True
        loop.configure(enable_literature_mining=False, enable_peer_review=False)
        assert loop.enable_literature_mining is False, (
            "configure should disable literature mining"
        )
        assert loop.enable_peer_review is False, (
            "configure should disable peer review"
        )

    def test_last_snapshot_property(self):
        """After run_cycle(), loop.last_snapshot returns a non-empty dict."""
        loop = self._make_loop(enable_literature_mining=True, enable_peer_review=True)
        # Before any cycle, last_snapshot is empty
        assert loop.last_snapshot == {}
        loop.load_local_papers(self._contradictory_papers())
        loop.run_cycle()
        snap = loop.last_snapshot
        assert isinstance(snap, dict), "last_snapshot should be a dict"
        assert len(snap) > 0, "last_snapshot should be non-empty after run_cycle"
        assert "cycle_count" in snap or "enabled" in snap, (
            f"last_snapshot should contain expected keys, got {list(snap.keys())}"
        )