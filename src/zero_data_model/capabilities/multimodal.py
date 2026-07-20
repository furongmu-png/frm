# src/zero_data_model/capabilities/multimodal.py
"""Phase 6 — Multimodal capabilities.

Cross-modal alignment and fusion using pure numpy. Implements:

- Canonical Correlation Analysis (CCA) for projecting two modalities
  into a shared latent subspace.
- A shared latent space wrapper with incremental updates.
- Fusion strategies: mean / concat / weighted.
- A simple modality encoder (BoW / TF-IDF / PCA).

No external ML library required.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .rules import MultimodalRules


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _sanitize_matrix(arr: np.ndarray, name: str = "matrix") -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite (no NaN or Inf)")
    return a


def _sanitize_vector(vec: np.ndarray, name: str = "vector") -> np.ndarray:
    v = np.asarray(vec, dtype=float).ravel()
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{name} must be finite (no NaN or Inf)")
    return v


# ----------------------------------------------------------------------
# CrossModalAligner (CCA)
# ----------------------------------------------------------------------


class CrossModalAligner:
    """Canonical Correlation Analysis (CCA).

    Given two matrices ``A (n, p)`` and ``B (n, q)`` of paired
    observations, find projection vectors ``w_a`` and ``w_b`` such that
    ``corr(A @ w_a, B @ w_b)`` is maximized. We solve this via the
    generalized eigenvalue problem on the covariance blocks.
    """

    def __init__(
        self,
        dim: int = 64,
        rules: MultimodalRules | None = None,
    ):
        self.dim = int(dim)
        self.rules = rules or MultimodalRules()
        self.projection_a: np.ndarray | None = None
        self.projection_b: np.ndarray | None = None
        self.correlation: float = 0.0
        self._mean_a: np.ndarray | None = None
        self._mean_b: np.ndarray | None = None
        self._fitted = False

    def fit(self, modality_a: np.ndarray, modality_b: np.ndarray) -> dict:
        """Fit CCA. Returns projection vectors + canonical correlation."""
        A = _sanitize_matrix(modality_a, "modality_a")
        B = _sanitize_matrix(modality_b, "modality_b")
        if A.shape[0] != B.shape[0]:
            raise ValueError(
                f"modality_a n={A.shape[0]} != modality_b n={B.shape[0]}"
            )
        n = A.shape[0]
        if n < 2:
            raise ValueError(f"need at least 2 paired samples, got {n}")
        # Center.
        mean_a = A.mean(axis=0)
        mean_b = B.mean(axis=0)
        A_c = A - mean_a
        B_c = B - mean_b
        # Covariance blocks.
        S_aa = (A_c.T @ A_c) / max(n - 1, 1)
        S_bb = (B_c.T @ B_c) / max(n - 1, 1)
        S_ab = (A_c.T @ B_c) / max(n - 1, 1)
        # Regularize diagonals for numerical stability.
        eps = 1e-6
        S_aa_reg = S_aa + eps * np.eye(S_aa.shape[0])
        S_bb_reg = S_bb + eps * np.eye(S_bb.shape[0])
        # Whitening.
        L_aa = np.linalg.cholesky(S_aa_reg)
        L_bb = np.linalg.cholesky(S_bb_reg)
        # M = L_aa^-1 @ S_ab @ L_bb^-T
        L_aa_inv = np.linalg.solve(L_aa, np.eye(L_aa.shape[0]))
        L_bb_inv_T = np.linalg.solve(L_bb.T, np.eye(L_bb.shape[0]))
        M = L_aa_inv @ S_ab @ L_bb_inv_T
        # SVD of M gives canonical correlations and directions.
        with np.errstate(divide="ignore", invalid="ignore"):
            U, S, Vt = np.linalg.svd(M, full_matrices=False)
        S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
        # Top-1 canonical direction.
        if S.size == 0 or S[0] < 1e-12:
            # Degenerate; use first unit vector.
            w_a_canonical = np.zeros(A.shape[1])
            w_a_canonical[0] = 1.0
            w_b_canonical = np.zeros(B.shape[1])
            w_b_canonical[0] = 1.0
            correlation = 0.0
        else:
            w_a_canonical = L_aa_inv.T @ U[:, 0]
            w_b_canonical = L_bb_inv_T.T @ Vt[0]
            correlation = float(min(S[0], 1.0))
        self.projection_a = w_a_canonical / max(np.linalg.norm(w_a_canonical), 1e-12)
        self.projection_b = w_b_canonical / max(np.linalg.norm(w_b_canonical), 1e-12)
        self.correlation = correlation
        self._mean_a = mean_a
        self._mean_b = mean_b
        self._fitted = True
        return {
            "projection_a": self.projection_a.copy(),
            "projection_b": self.projection_b.copy(),
            "correlation": float(correlation),
        }

    def align(self, embedding: np.ndarray, source: str) -> dict:
        """Project an embedding into the shared subspace.

        source ∈ {'a', 'b'}.
        """
        if not self._fitted:
            raise RuntimeError("CrossModalAligner not fitted; call fit() first")
        if source == "a":
            proj = self.projection_a
            mean = self._mean_a
        elif source == "b":
            proj = self.projection_b
            mean = self._mean_b
        else:
            raise ValueError(f"source must be 'a' or 'b', got {source!r}")
        e = _sanitize_vector(embedding)
        if proj is None or mean is None or e.size != proj.size:
            raise ValueError(
                f"embedding dim {e.size} != projection dim "
                f"{proj.size if proj is not None else 'unknown'}"
            )
        aligned = float(np.dot(e - mean, proj))
        return {"aligned": np.array([aligned]), "source": source}


# ----------------------------------------------------------------------
# SharedLatentSpace
# ----------------------------------------------------------------------


class SharedLatentSpace:
    """Maintains a (dim_modal, dim_shared) projection matrix.

    Supports incremental update via online CCA (a simple stochastic
    approximation: ``W += alpha * (x - mean) * (y - mean).T``).
    """

    def __init__(
        self,
        dim_modal: int = 64,
        rules: MultimodalRules | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.dim_modal = int(dim_modal)
        self.rules = rules or MultimodalRules()
        self.dim_shared = int(self.rules.multimodal_shared_dim)
        # Per-instance RNG so the facade (_child_rngs[i]) can drive reproducibility.
        self.rng = rng if rng is not None else np.random.default_rng()
        # Random orthogonal initialization.
        Q, _ = np.linalg.qr(self.rng.standard_normal((self.dim_modal, self.dim_shared)))
        self.projection = Q
        self.mean = np.zeros(self.dim_modal)
        self._n_updates = 0

    def update(
        self,
        modality_a: np.ndarray,
        modality_b: np.ndarray,
        alpha: float | None = None,
    ) -> dict:
        """Incremental update from paired observations."""
        a = _sanitize_vector(modality_a, "modality_a")
        b = _sanitize_vector(modality_b, "modality_b")
        if a.size != b.size:
            raise ValueError(f"modality_a dim {a.size} != modality_b dim {b.size}")
        if a.size != self.dim_modal:
            # Auto-resize.
            self.dim_modal = int(a.size)
            Q, _ = np.linalg.qr(
                self.rng.standard_normal((self.dim_modal, self.dim_shared))
            )
            self.projection = Q
            self.mean = np.zeros(self.dim_modal)
        lr = float(alpha) if alpha is not None else 0.01
        # Update running mean.
        self._n_updates += 1
        self.mean += (a - self.mean) / self._n_updates
        # Hebbian-style update: outer product of centered vectors.
        centered_a = a - self.mean
        centered_b = b - self.mean
        update = np.outer(centered_a, centered_b) @ self.projection
        self.projection = self.projection + lr * update
        # Re-orthogonalize (cheap QR on small matrix).
        Q, _ = np.linalg.qr(self.projection)
        self.projection = Q
        return {"n_updates": int(self._n_updates), "dim_shared": self.dim_shared}

    def project(self, embedding: np.ndarray) -> dict:
        """Project embedding into the shared latent space."""
        e = _sanitize_vector(embedding, "embedding")
        if e.size != self.dim_modal:
            raise ValueError(
                f"embedding dim {e.size} != projection dim {self.dim_modal}"
            )
        projected = (e - self.mean) @ self.projection
        return {"projected": projected, "dim_shared": int(projected.size)}


# ----------------------------------------------------------------------
# ModalityFuser
# ----------------------------------------------------------------------


class ModalityFuser:
    """Combine multiple modality embeddings into a single fused vector.

    Strategies:
    - 'mean': arithmetic mean.
    - 'concat': concatenation along axis 1 (assumes 2D input).
    - 'weighted': weighted sum using ``MultimodalRules.fusion_weights``.
    """

    def __init__(
        self,
        dim: int = 64,
        rules: MultimodalRules | None = None,
    ):
        self.dim = int(dim)
        self.rules = rules or MultimodalRules()

    def fuse(self, embeddings: list[np.ndarray], strategy: str | None = None) -> dict:
        """Fuse a list of modality embeddings."""
        if not embeddings:
            return {"fused": np.zeros(0), "strategy": strategy or self.rules.multimodal_fusion_strategy}
        strat = str(strategy or self.rules.multimodal_fusion_strategy).lower()
        cleaned = []
        for emb in embeddings:
            e = np.asarray(emb, dtype=float).ravel()
            if not np.all(np.isfinite(e)):
                raise ValueError("embeddings must be finite")
            cleaned.append(e)
        # All same dim?
        dims = set(e.size for e in cleaned)
        if len(dims) > 1 and strat != "concat":
            raise ValueError(
                f"all embeddings must share dim for {strat} fusion, got dims {dims}"
            )
        if strat == "mean":
            stacked = np.stack(cleaned)
            fused = stacked.mean(axis=0)
        elif strat == "concat":
            fused = np.concatenate(cleaned)
        elif strat == "weighted":
            weights = self.rules.multimodal_fusion_weights
            if len(weights) != len(cleaned):
                raise ValueError(
                    f"fusion_weights length {len(weights)} != "
                    f"embeddings count {len(cleaned)}"
                )
            stacked = np.stack(cleaned)
            fused = (stacked.T * np.asarray(weights)).T.sum(axis=0)
        else:
            raise ValueError(f"unknown fusion strategy: {strat!r}")
        return {"fused": fused, "strategy": strat}


# ----------------------------------------------------------------------
# ModalityEncoder
# ----------------------------------------------------------------------


class ModalityEncoder:
    """Encode raw modality inputs to fixed-dim embeddings.

    Supports three encoding modes selected by ``method``:
    - 'bow': bag-of-words over token id sequences.
    - 'tfidf': TF-IDF weighted bow.
    - 'pca': first k principal components (PCA via SVD).
    """

    def __init__(
        self,
        dim: int = 64,
        rules: MultimodalRules | None = None,
        method: str = "bow",
    ):
        self.dim = int(dim)
        self.rules = rules or MultimodalRules()
        self.method = str(method).lower()
        self._vocabulary: dict[int, int] = {}
        self._idf: np.ndarray | None = None
        self._pca_components: np.ndarray | None = None
        self._pca_mean: np.ndarray | None = None

    def fit(
        self,
        sequences: list[np.ndarray] | np.ndarray,
        vocabulary: list[int] | None = None,
    ) -> dict:
        """Fit the encoder on training sequences.

        For 'bow' / 'tfidf': builds vocabulary.
        For 'pca': computes mean + top-k SVD components.
        """
        if isinstance(sequences, np.ndarray):
            seqs = [np.asarray(s, dtype=int).ravel() for s in sequences]
        else:
            seqs = [np.asarray(s, dtype=int).ravel() for s in sequences]
        if not seqs:
            raise ValueError("sequences must be non-empty")
        if self.method in ("bow", "tfidf"):
            if vocabulary is not None:
                self._vocabulary = {int(tok): i for i, tok in enumerate(vocabulary)}
            else:
                # Build vocabulary from observed tokens.
                unique = sorted(set(int(t) for s in seqs for t in s))
                self._vocabulary = {tok: i for i, tok in enumerate(unique)}
            if self.method == "tfidf":
                n_docs = len(seqs)
                df = np.zeros(len(self._vocabulary))
                for s in seqs:
                    seen = set(int(t) for t in s)
                    for t in seen:
                        if t in self._vocabulary:
                            df[self._vocabulary[t]] += 1
                # idf = log(N / (1 + df)).
                self._idf = np.log((n_docs + 1.0) / (df + 1.0) + 1.0)
        elif self.method == "pca":
            # Stack sequences as 2D matrix; pad/truncate to dim.
            matrix = np.zeros((len(seqs), self.dim))
            for i, s in enumerate(seqs):
                v = np.asarray(s, dtype=float).ravel()
                if v.size > self.dim:
                    matrix[i] = v[: self.dim]
                else:
                    matrix[i, : v.size] = v
            self._pca_mean = matrix.mean(axis=0)
            centered = matrix - self._pca_mean
            with np.errstate(divide="ignore", invalid="ignore"):
                U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            # Keep top-k where k = min(dim, n_components).
            k = min(self.dim, Vt.shape[0])
            self._pca_components = Vt[:k].T  # (dim, k)
        else:
            raise ValueError(f"unknown method: {self.method!r}")
        return {"vocab_size": len(self._vocabulary), "method": self.method}

    def encode(self, sequence: np.ndarray) -> dict:
        """Encode a single sequence into a fixed-dim vector."""
        s = np.asarray(sequence, dtype=float).ravel()
        if self.method == "bow":
            vec = np.zeros(self.dim)
            for tok in s.astype(int):
                if tok in self._vocabulary:
                    idx = self._vocabulary[tok]
                    if idx < self.dim:
                        vec[idx] += 1.0
            return {"embedding": vec, "method": "bow"}
        if self.method == "tfidf":
            if self._idf is None:
                raise RuntimeError("tfidf encoder not fitted; call fit() first")
            vec = np.zeros(self.dim)
            counts: dict[int, int] = {}
            for tok in s.astype(int):
                if tok in self._vocabulary:
                    idx = self._vocabulary[tok]
                    counts[idx] = counts.get(idx, 0) + 1
            for idx, count in counts.items():
                if idx < self.dim:
                    vec[idx] = count * self._idf[idx]
            return {"embedding": vec, "method": "tfidf"}
        if self.method == "pca":
            if self._pca_components is None or self._pca_mean is None:
                raise RuntimeError("pca encoder not fitted; call fit() first")
            v = np.zeros(self.dim)
            if s.size > self.dim:
                v[:] = s[: self.dim]
            else:
                v[: s.size] = s
            centered = v - self._pca_mean
            projected = centered @ self._pca_components
            return {"embedding": projected, "method": "pca"}
        raise ValueError(f"unknown method: {self.method!r}")
