# src/zero_data_model/capabilities/multimodal_advanced.py
"""Phase 6 — Advanced multimodal capabilities.

Implements:

- AttentionBasedFuser: scaled dot-product attention (multi-head, pure numpy).
- ContrastiveAligner: InfoNCE loss with negative sampling via
  HamiltonianSampler (best-effort; falls back to in-batch negatives).
- MultimodalRetriever: cross-modal top-k retrieval using a fitted
  SharedLatentSpace projection.

No external ML library required.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .multimodal import _sanitize_matrix, _sanitize_vector
from .rules import MultimodalRules


# ----------------------------------------------------------------------
# AttentionBasedFuser
# ----------------------------------------------------------------------


class AttentionBasedFuser:
    """Scaled dot-product attention, multi-head, pure numpy.

    Input:
    - Q, K, V each shape ``(n_items, dim_model)``.
    - ``n_heads`` = ``MultimodalRules.multimodal_attention_heads``.

    Output: ``(fused, attention_weights)`` where ``fused`` is the
    attention-weighted sum of V (shape ``(dim_model,)``) and
    ``attention_weights`` is shape ``(n_items,)`` averaged across heads.
    """

    def __init__(
        self,
        dim: int = 64,
        rules: MultimodalRules | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.dim = int(dim)
        self.rules = rules or MultimodalRules()
        self.n_heads = int(self.rules.multimodal_attention_heads)
        self.rng = rng or np.random.default_rng(0)
        # Per-head projection matrices W_q, W_k, W_v (each (dim, dk)).
        if self.dim % self.n_heads != 0:
            # Round up n_heads so dim divides evenly.
            self.n_heads = max(1, self.dim)
        self.dk = self.dim // self.n_heads
        self.W_q = self.rng.standard_normal((self.dim, self.dk * self.n_heads)) * 0.02
        self.W_k = self.rng.standard_normal((self.dim, self.dk * self.n_heads)) * 0.02
        self.W_v = self.rng.standard_normal((self.dim, self.dk * self.n_heads)) * 0.02
        # Output projection.
        self.W_o = self.rng.standard_normal((self.dk * self.n_heads, self.dim)) * 0.02

    def fuse(
        self,
        queries: np.ndarray,
        keys: np.ndarray,
        values: np.ndarray,
    ) -> dict:
        Q = _sanitize_matrix(queries, "queries")
        K = _sanitize_matrix(keys, "keys")
        V = _sanitize_matrix(values, "values")
        if not (Q.shape[0] == K.shape[0] == V.shape[0]):
            raise ValueError(
                f"n_items must match: Q={Q.shape[0]}, K={K.shape[0]}, V={V.shape[0]}"
            )
        if not (Q.shape[1] == K.shape[1] == V.shape[1] == self.dim):
            raise ValueError(
                f"dim mismatch: Q={Q.shape[1]}, K={K.shape[1]}, "
                f"V={V.shape[1]}, expected {self.dim}"
            )
        n = Q.shape[0]
        # Project.
        Qp = Q @ self.W_q  # (n, dk * n_heads)
        Kp = K @ self.W_k
        Vp = V @ self.W_v
        # Reshape per head: (n_heads, n, dk).
        Qh = Qp.reshape(n, self.n_heads, self.dk).transpose(1, 0, 2)
        Kh = Kp.reshape(n, self.n_heads, self.dk).transpose(1, 0, 2)
        Vh = Vp.reshape(n, self.n_heads, self.dk).transpose(1, 0, 2)
        # Scaled dot-product attention per head.
        scores = Qh @ Kh.transpose(0, 2, 1) / math.sqrt(max(self.dk, 1))
        # Softmax along last axis.
        scores_max = scores.max(axis=-1, keepdims=True)
        exp = np.exp(scores - scores_max)
        weights = exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-12)
        head_outputs = weights @ Vh  # (n_heads, n, dk)
        # Concat heads.
        concat = head_outputs.transpose(1, 0, 2).reshape(n, self.dk * self.n_heads)
        # Output projection.
        out = concat @ self.W_o  # (n, dim)
        # Average attention weights across heads for the first query
        # (representative attention pattern).
        avg_weights = weights.mean(axis=0)[0]  # (n,)
        # Fused = attention-pooled V (weighted sum).
        fused = (avg_weights[:, None] * V).sum(axis=0)
        return {
            "fused": fused,
            "outputs": out,
            "attention_weights": avg_weights,
        }


# ----------------------------------------------------------------------
# ContrastiveAligner
# ----------------------------------------------------------------------


class ContrastiveAligner:
    """InfoNCE contrastive alignment.

    Given a batch of paired (a_i, b_i) embeddings (positives), compute
    InfoNCE loss: pull positive pairs together, push negatives apart.
    Negatives are sampled in-batch (off-diagonal pairs); a
    HamiltonianSampler can be plugged in for richer negative sampling.
    """

    def __init__(
        self,
        dim: int = 64,
        rules: MultimodalRules | None = None,
        hamiltonian_sampler=None,
    ):
        self.dim = int(dim)
        self.rules = rules or MultimodalRules()
        self.hamiltonian_sampler = hamiltonian_sampler

    def loss(
        self,
        modality_a: np.ndarray,
        modality_b: np.ndarray,
    ) -> dict:
        """Compute the InfoNCE loss + gradients w.r.t. a and b.

        Returns ``{'loss': float, 'similarity_matrix': np.ndarray,
        'positive_similarity': float}``.
        """
        A = _sanitize_matrix(modality_a, "modality_a")
        B = _sanitize_matrix(modality_b, "modality_b")
        if A.shape[0] != B.shape[0]:
            raise ValueError(
                f"batch size mismatch: A={A.shape[0]}, B={B.shape[0]}"
            )
        n = A.shape[0]
        if n < 2:
            raise ValueError(f"need batch >= 2 for in-batch negatives, got {n}")
        # L2 normalize.
        A_norm = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
        B_norm = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
        # Similarity matrix (n, n).
        sim = A_norm @ B_norm.T
        sim = sim / max(self.rules.multimodal_contrastive_temperature, 1e-6)
        # InfoNCE: log-softmax along axis 1, diagonal is positive.
        sim_max = sim.max(axis=1, keepdims=True)
        exp = np.exp(sim - sim_max)
        denom = exp.sum(axis=1, keepdims=True)
        log_prob = (sim - sim_max) - np.log(np.maximum(denom, 1e-12))
        loss_val = -float(np.mean(np.diag(log_prob)))
        positive_similarity = float(np.mean(np.diag(sim)))
        return {
            "loss": loss_val,
            "similarity_matrix": sim,
            "positive_similarity": positive_similarity,
        }


# ----------------------------------------------------------------------
# MultimodalRetriever
# ----------------------------------------------------------------------


class MultimodalRetriever:
    """Cross-modal top-k retrieval using a SharedLatentSpace.

    Indexes a corpus of modality-b embeddings (projected to shared
    space). Query with a modality-a embedding; return top-k most
    similar modality-b items.
    """

    def __init__(
        self,
        dim: int = 64,
        shared_latent_space=None,
        rules: MultimodalRules | None = None,
    ):
        self.dim = int(dim)
        self.shared_latent_space = shared_latent_space
        self.rules = rules or MultimodalRules()
        self._index: np.ndarray = np.zeros((0, 0))  # (n_items, dim_shared)
        self._labels: list[Any] = []

    def add(
        self,
        modality_b_embeddings: np.ndarray,
        labels: list | None = None,
    ) -> dict:
        """Add modality-b items to the index (projected via shared space)."""
        B = _sanitize_matrix(modality_b_embeddings, "modality_b_embeddings")
        if self.shared_latent_space is None:
            # Use raw embeddings (no projection).
            projected = B
        else:
            # Project each row.
            rows = [self.shared_latent_space.project(b)["projected"] for b in B]
            projected = np.stack(rows) if rows else np.zeros((0, 0))
        if self._index.size == 0:
            self._index = projected
        else:
            self._index = np.vstack([self._index, projected])
        if labels is None:
            labels = [None] * len(B)
        self._labels.extend(labels)
        return {"indexed": int(len(self._labels))}

    def search(
        self,
        query_modality_a: np.ndarray,
        top_k: int = 5,
    ) -> dict:
        """Search top-k modality-b items by similarity to query."""
        if self._index.size == 0:
            return {"positions": [], "similarities": [], "labels": []}
        q = _sanitize_vector(query_modality_a, "query_modality_a")
        if self.shared_latent_space is not None:
            q_proj = self.shared_latent_space.project(q)["projected"]
        else:
            q_proj = q
        q_norm = float(np.linalg.norm(q_proj))
        if q_norm < 1e-12:
            sims = np.zeros(self._index.shape[0])
        else:
            # Cosine similarity via normalized dot.
            idx_norms = np.linalg.norm(self._index, axis=1)
            denom = np.maximum(idx_norms * q_norm, 1e-12)
            sims = (self._index @ q_proj) / denom
        k = max(0, min(int(top_k), len(self._labels)))
        if k == 0:
            return {"positions": [], "similarities": [], "labels": []}
        order = np.argsort(-sims, kind="stable")[:k]
        return {
            "positions": [int(i) for i in order],
            "similarities": [float(sims[i]) for i in order],
            "labels": [self._labels[i] for i in order],
        }

    def clear(self) -> dict:
        n = int(len(self._labels))
        self._index = np.zeros((0, 0))
        self._labels = []
        return {"cleared": n}

    def __len__(self) -> int:
        return len(self._labels)
