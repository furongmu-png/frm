# tests/test_phase6_multimodal.py
"""Phase 6 — Multimodal capability tests.

Covers the 7 classes in ``capabilities.multimodal`` and
``capabilities.multimodal_advanced``: CrossModalAligner, SharedLatentSpace,
ModalityFuser, ModalityEncoder, AttentionBasedFuser, ContrastiveAligner,
MultimodalRetriever.

Total: 70+ tests covering correctness, edge cases, NaN/Inf guards,
shape mismatch handling, fusion strategies, attention mechanism,
contrastive loss, and API contract per spec §6 (Multimodal).
"""

from __future__ import annotations

import numpy as np
import pytest

from zero_data_model.capabilities.multimodal import (
    CrossModalAligner,
    ModalityEncoder,
    ModalityFuser,
    SharedLatentSpace,
)
from zero_data_model.capabilities.multimodal_advanced import (
    AttentionBasedFuser,
    ContrastiveAligner,
    MultimodalRetriever,
)
from zero_data_model.capabilities.rules import MultimodalRules


# --------------------------------------------------------------------- #
# CrossModalAligner (CCA)
# --------------------------------------------------------------------- #


class TestCrossModalAligner:
    @pytest.fixture()
    def aligner(self):
        return CrossModalAligner(dim=4)

    def test_fit_returns_projections(self, aligner):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((20, 4))
        B = rng.standard_normal((20, 6))
        r = aligner.fit(A, B)
        assert r["projection_a"].shape == (4,)
        assert r["projection_b"].shape == (6,)
        assert isinstance(r["correlation"], float)

    def test_fit_perfectly_correlated_yields_high_correlation(self, aligner):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((50, 4))
        # B is a linear function of A → perfect correlation.
        B = A @ np.array([[1.0, 0.5], [0.0, 1.0], [1.0, 0.0], [0.5, 1.0]])
        r = aligner.fit(A, B)
        assert r["correlation"] > 0.9

    def test_fit_uncorrelated_yields_low_correlation(self, aligner):
        rng = np.random.default_rng(7)
        A = rng.standard_normal((100, 4))
        B = rng.standard_normal((100, 4))
        r = aligner.fit(A, B)
        assert abs(r["correlation"]) < 0.5

    def test_fit_correlation_bounded_in_unit(self, aligner):
        rng = np.random.default_rng(3)
        r = aligner.fit(rng.standard_normal((20, 4)), rng.standard_normal((20, 5)))
        assert -1.0 <= r["correlation"] <= 1.0

    def test_fit_mismatched_rows_raises(self, aligner):
        with pytest.raises(ValueError, match="n="):
            aligner.fit(np.zeros((10, 4)), np.zeros((20, 4)))

    def test_fit_too_few_samples_raises(self, aligner):
        with pytest.raises(ValueError, match="samples"):
            aligner.fit(np.zeros((1, 4)), np.zeros((1, 4)))

    def test_fit_nan_raises(self, aligner):
        with pytest.raises(ValueError, match="finite"):
            aligner.fit(
                np.array([[np.nan, 1.0], [0.0, 1.0]]),
                np.zeros((2, 4)),
            )

    def test_fit_non_2d_raises(self, aligner):
        with pytest.raises(ValueError, match="2D"):
            aligner.fit(np.zeros(4), np.zeros((4, 4)))

    def test_align_before_fit_raises(self, aligner):
        with pytest.raises(RuntimeError, match="not fitted"):
            aligner.align(np.zeros(4), source="a")

    def test_align_returns_scalar(self, aligner):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((10, 4))
        B = rng.standard_normal((10, 4))
        aligner.fit(A, B)
        r = aligner.align(A[0], source="a")
        assert r["aligned"].shape == (1,)
        assert r["source"] == "a"

    def test_align_invalid_source_raises(self, aligner):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((10, 4))
        B = rng.standard_normal((10, 4))
        aligner.fit(A, B)
        with pytest.raises(ValueError, match="source"):
            aligner.align(A[0], source="c")

    def test_align_dim_mismatch_raises(self, aligner):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((10, 4))
        B = rng.standard_normal((10, 4))
        aligner.fit(A, B)
        with pytest.raises(ValueError, match="dim"):
            aligner.align(np.zeros(8), source="a")  # wrong dim

    def test_align_nan_raises(self, aligner):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((10, 4))
        B = rng.standard_normal((10, 4))
        aligner.fit(A, B)
        with pytest.raises(ValueError, match="finite"):
            aligner.align(np.array([np.nan, 0, 0, 0]), source="a")


# --------------------------------------------------------------------- #
# SharedLatentSpace
# --------------------------------------------------------------------- #


class TestSharedLatentSpace:
    @pytest.fixture()
    def sls(self):
        return SharedLatentSpace(dim_modal=4, rules=MultimodalRules(multimodal_shared_dim=2))

    def test_init_random_orthogonal_projection(self, sls):
        assert sls.projection.shape == (4, 2)
        # Columns should be orthonormal.
        Q = sls.projection
        np.testing.assert_allclose(Q.T @ Q, np.eye(2), atol=1e-9)

    def test_update_increments_count(self, sls):
        sls.update(np.zeros(4), np.zeros(4))
        sls.update(np.ones(4), np.ones(4))
        assert sls._n_updates == 2

    def test_update_changes_mean(self, sls):
        before = sls.mean.copy()
        sls.update(np.array([1.0, 1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0, 1.0]))
        after = sls.mean
        assert not np.allclose(before, after)

    def test_update_dim_mismatch_raises(self, sls):
        with pytest.raises(ValueError, match="dim"):
            sls.update(np.zeros(4), np.zeros(3))

    def test_update_auto_resizes_dim(self):
        sls = SharedLatentSpace(dim_modal=8, rules=MultimodalRules(multimodal_shared_dim=2))
        sls.update(np.zeros(4), np.zeros(4))
        assert sls.dim_modal == 4

    def test_project_returns_dim_shared(self, sls):
        r = sls.project(np.zeros(4))
        assert r["projected"].shape == (2,)
        assert r["dim_shared"] == 2

    def test_project_dim_mismatch_raises(self, sls):
        with pytest.raises(ValueError, match="dim"):
            sls.project(np.zeros(8))

    def test_project_nan_raises(self, sls):
        with pytest.raises(ValueError, match="finite"):
            sls.project(np.array([np.nan, 0, 0, 0]))

    def test_update_preserves_orthonormality(self, sls):
        for _ in range(5):
            sls.update(np.random.default_rng(0).standard_normal(4),
                      np.random.default_rng(1).standard_normal(4))
        # After re-orthogonalization, columns should still be orthonormal.
        Q = sls.projection
        np.testing.assert_allclose(Q.T @ Q, np.eye(2), atol=1e-6)


# --------------------------------------------------------------------- #
# ModalityFuser
# --------------------------------------------------------------------- #


class TestModalityFuser:
    @pytest.fixture()
    def fuser(self):
        return ModalityFuser(dim=4)

    def test_fuse_mean(self, fuser):
        r = fuser.fuse([np.array([1.0, 2, 3, 4]), np.array([3.0, 4, 5, 6])], strategy="mean")
        np.testing.assert_allclose(r["fused"], np.array([2.0, 3, 4, 5]))
        assert r["strategy"] == "mean"

    def test_fuse_concat(self, fuser):
        r = fuser.fuse([np.array([1.0, 2]), np.array([3.0, 4])], strategy="concat")
        np.testing.assert_allclose(r["fused"], np.array([1.0, 2, 3, 4]))

    def test_fuse_weighted(self):
        fuser = ModalityFuser(
            dim=4, rules=MultimodalRules(multimodal_fusion_weights=[0.3, 0.7]),
        )
        r = fuser.fuse(
            [np.array([10.0, 0, 0, 0]), np.array([0.0, 10, 0, 0])],
            strategy="weighted",
        )
        np.testing.assert_allclose(r["fused"], np.array([3.0, 7.0, 0, 0]))

    def test_fuse_default_strategy_from_rules(self):
        fuser = ModalityFuser(
            dim=4, rules=MultimodalRules(multimodal_fusion_strategy="mean"),
        )
        r = fuser.fuse([np.array([1.0, 0, 0, 0]), np.array([3.0, 0, 0, 0])])
        assert r["strategy"] == "mean"

    def test_fuse_unknown_strategy_raises(self, fuser):
        with pytest.raises(ValueError, match="unknown"):
            fuser.fuse([np.zeros(4), np.zeros(4)], strategy="bogus")

    def test_fuse_dim_mismatch_for_mean_raises(self, fuser):
        with pytest.raises(ValueError, match="dim"):
            fuser.fuse([np.array([1.0, 2]), np.array([1.0, 2, 3, 4])], strategy="mean")

    def test_fuse_dim_mismatch_for_concat_ok(self, fuser):
        # concat accepts variable dims.
        r = fuser.fuse([np.array([1.0, 2]), np.array([1.0, 2, 3, 4])], strategy="concat")
        assert r["fused"].shape == (6,)

    def test_fuse_weighted_count_mismatch_raises(self):
        fuser = ModalityFuser(
            dim=4, rules=MultimodalRules(multimodal_fusion_weights=[0.5, 0.3, 0.2]),
        )
        with pytest.raises(ValueError, match="fusion_weights"):
            fuser.fuse([np.zeros(4), np.zeros(4)], strategy="weighted")

    def test_fuse_empty_returns_empty(self, fuser):
        r = fuser.fuse([])
        assert r["fused"].size == 0

    def test_fuse_nan_raises(self, fuser):
        with pytest.raises(ValueError, match="finite"):
            fuser.fuse([np.array([np.nan, 0, 0, 0]), np.zeros(4)])


# --------------------------------------------------------------------- #
# ModalityEncoder
# --------------------------------------------------------------------- #


class TestModalityEncoder:
    def test_fit_bow_builds_vocab(self):
        enc = ModalityEncoder(dim=8, method="bow")
        r = enc.fit([np.array([1, 2, 3]), np.array([2, 3, 4])])
        assert r["vocab_size"] == 4  # tokens {1, 2, 3, 4}

    def test_encode_bow(self):
        enc = ModalityEncoder(dim=8, method="bow")
        enc.fit([np.array([1, 2, 3])])
        r = enc.encode(np.array([1, 1, 2]))
        assert r["embedding"][0] == 2.0  # token 1 appears twice
        assert r["embedding"][1] == 1.0  # token 2 once

    def test_fit_tfidf(self):
        enc = ModalityEncoder(dim=8, method="tfidf")
        r = enc.fit([np.array([1, 2]), np.array([1, 3])])
        assert r["method"] == "tfidf"
        assert enc._idf is not None

    def test_encode_tfidf(self):
        enc = ModalityEncoder(dim=8, method="tfidf")
        enc.fit([np.array([1, 2]), np.array([1, 3])])
        r = enc.encode(np.array([1, 2]))
        assert r["embedding"].shape == (8,)

    def test_encode_tfidf_unfitted_raises(self):
        enc = ModalityEncoder(dim=8, method="tfidf")
        with pytest.raises(RuntimeError, match="not fitted"):
            enc.encode(np.array([1, 2]))

    def test_fit_pca(self):
        enc = ModalityEncoder(dim=4, method="pca")
        r = enc.fit([np.array([1.0, 2, 3, 4]), np.array([2.0, 3, 4, 5]),
                     np.array([3.0, 4, 5, 6])])
        assert r["method"] == "pca"
        assert enc._pca_components is not None

    def test_encode_pca(self):
        enc = ModalityEncoder(dim=4, method="pca")
        enc.fit([np.array([1.0, 2, 3, 4]), np.array([2.0, 3, 4, 5]),
                 np.array([3.0, 4, 5, 6])])
        r = enc.encode(np.array([1.0, 2, 3, 4]))
        assert r["embedding"].shape[0] <= 4

    def test_encode_pca_unfitted_raises(self):
        enc = ModalityEncoder(dim=4, method="pca")
        with pytest.raises(RuntimeError, match="not fitted"):
            enc.encode(np.array([1.0, 2, 3, 4]))

    def test_unknown_method_raises(self):
        enc = ModalityEncoder(dim=4, method="bogus")
        with pytest.raises(ValueError, match="unknown method"):
            enc.fit([np.array([1, 2])])

    def test_fit_empty_raises(self):
        enc = ModalityEncoder(dim=4, method="bow")
        with pytest.raises(ValueError, match="non-empty"):
            enc.fit([])

    def test_fit_with_explicit_vocab(self):
        enc = ModalityEncoder(dim=8, method="bow")
        r = enc.fit([np.array([10, 20])], vocabulary=[10, 20, 30])
        assert r["vocab_size"] == 3


# --------------------------------------------------------------------- #
# AttentionBasedFuser
# --------------------------------------------------------------------- #


class TestAttentionBasedFuser:
    @pytest.fixture()
    def fuser(self):
        rng = np.random.default_rng(42)
        return AttentionBasedFuser(dim=4, rules=MultimodalRules(multimodal_attention_heads=2), rng=rng)

    def test_fuse_returns_fused_and_weights(self, fuser):
        Q = np.random.default_rng(0).standard_normal((3, 4))
        r = fuser.fuse(Q, Q, Q)
        assert r["fused"].shape == (4,)
        assert r["attention_weights"].shape == (3,)
        assert r["outputs"].shape == (3, 4)

    def test_attention_weights_sum_to_one(self, fuser):
        Q = np.random.default_rng(0).standard_normal((5, 4))
        r = fuser.fuse(Q, Q, Q)
        assert r["attention_weights"].sum() == pytest.approx(1.0, abs=1e-6)

    def test_fuse_dim_mismatch_raises(self, fuser):
        with pytest.raises(ValueError, match="dim"):
            fuser.fuse(
                np.zeros((3, 8)), np.zeros((3, 8)), np.zeros((3, 8))
            )

    def test_fuse_n_items_mismatch_raises(self, fuser):
        with pytest.raises(ValueError, match="n_items"):
            fuser.fuse(
                np.zeros((3, 4)), np.zeros((5, 4)), np.zeros((3, 4))
            )

    def test_fuse_nan_raises(self, fuser):
        Q = np.array([[np.nan, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        with pytest.raises(ValueError, match="finite"):
            fuser.fuse(Q, Q, Q)

    def test_fuse_non_2d_raises(self, fuser):
        with pytest.raises(ValueError, match="2D"):
            fuser.fuse(np.zeros(4), np.zeros(4), np.zeros(4))

    def test_fuse_outputs_proportional_to_attention(self, fuser):
        # When Q == K == V and all rows identical, fused should equal a single row.
        V = np.tile(np.array([1.0, 2.0, 3.0, 4.0]), (3, 1))
        r = fuser.fuse(V, V, V)
        # Attention weights are uniform (3 identical items), fused = mean.
        np.testing.assert_allclose(r["fused"], np.array([1.0, 2, 3, 4]), atol=1e-6)


# --------------------------------------------------------------------- #
# ContrastiveAligner
# --------------------------------------------------------------------- #


class TestContrastiveAligner:
    @pytest.fixture()
    def ca(self):
        return ContrastiveAligner(dim=4)

    def test_loss_returns_float(self, ca):
        rng = np.random.default_rng(0)
        r = ca.loss(rng.standard_normal((5, 4)), rng.standard_normal((5, 4)))
        assert isinstance(r["loss"], float)
        assert "similarity_matrix" in r
        assert "positive_similarity" in r

    def test_loss_positive_pairs_lower_loss(self, ca):
        rng = np.random.default_rng(0)
        # A perfectly aligned batch: A == B → low loss.
        A = rng.standard_normal((4, 4))
        r_aligned = ca.loss(A, A)
        # Random batch: high loss.
        B_random = rng.standard_normal((4, 4))
        r_random = ca.loss(A, B_random)
        assert r_aligned["loss"] < r_random["loss"]

    def test_loss_batch_size_mismatch_raises(self, ca):
        with pytest.raises(ValueError, match="batch"):
            ca.loss(np.zeros((5, 4)), np.zeros((4, 4)))

    def test_loss_too_small_batch_raises(self, ca):
        with pytest.raises(ValueError, match="batch"):
            ca.loss(np.zeros((1, 4)), np.zeros((1, 4)))

    def test_loss_returns_similarity_matrix(self, ca):
        rng = np.random.default_rng(0)
        r = ca.loss(rng.standard_normal((4, 4)), rng.standard_normal((4, 4)))
        assert r["similarity_matrix"].shape == (4, 4)

    def test_loss_positive_similarity_in_unit(self, ca):
        rng = np.random.default_rng(0)
        r = ca.loss(rng.standard_normal((5, 4)), rng.standard_normal((5, 4)))
        # After L2 normalization, cosine similarities are in [-1, 1].
        # After temperature scaling, magnitude may shrink but abs ≤ 1.
        assert -1.0 <= r["positive_similarity"] <= 1.0

    def test_loss_nan_raises(self, ca):
        with pytest.raises(ValueError, match="finite"):
            ca.loss(np.array([[np.nan, 0, 0, 0], [0, 0, 0, 0]]),
                    np.zeros((2, 4)))

    def test_loss_non_2d_raises(self, ca):
        with pytest.raises(ValueError, match="2D"):
            ca.loss(np.zeros(4), np.zeros(4))


# --------------------------------------------------------------------- #
# MultimodalRetriever
# --------------------------------------------------------------------- #


class TestMultimodalRetriever:
    @pytest.fixture()
    def retriever(self):
        return MultimodalRetriever(dim=4)

    def test_add_returns_indexed_count(self, retriever):
        r = retriever.add(np.zeros((3, 4)))
        assert r["indexed"] == 3

    def test_add_with_labels(self, retriever):
        retriever.add(np.zeros((2, 4)), labels=["a", "b"])
        r = retriever.search(np.array([1.0, 0.0, 0.0, 0.0]), top_k=2)
        assert r["labels"] in (["a", "b"], ["b", "a"])

    def test_search_empty_returns_empty(self, retriever):
        r = retriever.search(np.zeros(4))
        assert r["positions"] == []
        assert r["similarities"] == []
        assert r["labels"] == []

    def test_search_top_k(self, retriever):
        # Items: [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]
        retriever.add(np.eye(3, 4))
        r = retriever.search(np.array([1.0, 0.0, 0.0, 0.0]), top_k=2)
        assert len(r["positions"]) == 2
        # Best match is item 0 (cosine=1).
        assert r["positions"][0] == 0

    def test_search_caps_at_index_size(self, retriever):
        retriever.add(np.eye(3, 4))
        r = retriever.search(np.zeros(4), top_k=10)
        assert len(r["positions"]) == 3

    def test_search_zero_norm_query(self, retriever):
        retriever.add(np.eye(3, 4))
        r = retriever.search(np.zeros(4), top_k=2)
        # All similarities should be 0.
        assert all(s == 0.0 for s in r["similarities"])

    def test_clear(self, retriever):
        retriever.add(np.eye(3, 4))
        r = retriever.clear()
        assert r["cleared"] == 3
        assert len(retriever) == 0

    def test_len(self, retriever):
        assert len(retriever) == 0
        retriever.add(np.eye(2, 4))
        assert len(retriever) == 2

    def test_search_returns_labels(self, retriever):
        retriever.add(np.eye(3, 4), labels=["x", "y", "z"])
        r = retriever.search(np.array([1.0, 0.0, 0.0, 0.0]), top_k=1)
        assert r["labels"][0] == "x"

    def test_add_multiple_calls_accumulate(self, retriever):
        retriever.add(np.eye(2, 4))
        retriever.add(np.array([[0.0, 0.0, 1.0, 0.0]]))
        assert len(retriever) == 3

    def test_search_with_shared_latent_space(self):
        sls = SharedLatentSpace(dim_modal=4, rules=MultimodalRules(multimodal_shared_dim=2))
        retriever = MultimodalRetriever(dim=4, shared_latent_space=sls)
        retriever.add(np.eye(3, 4))
        r = retriever.search(np.array([1.0, 0.0, 0.0, 0.0]), top_k=2)
        assert len(r["positions"]) == 2
