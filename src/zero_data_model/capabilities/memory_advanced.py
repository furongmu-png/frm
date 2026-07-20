# src/zero_data_model/capabilities/memory_advanced.py
"""Phase 6 — Advanced memory capabilities.

Composes :mod:`memory` with stronger retrieval / consolidation
strategies: hierarchical short/medium/long-term memory, spreading
activation (Collins & Loftus 1975), Ebbinghaus forgetting curve, and a
Faiss-lite-style inverted index for fast top-k retrieval.

No external ML library required.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .memory import (
    EpisodicMemory,
    WorkingMemory,
    _cosine_similarity,
    _sanitize_vector,
)
from .rules import MemoryRules


# ----------------------------------------------------------------------
# HierarchicalMemory
# ----------------------------------------------------------------------


class HierarchicalMemory:
    """Three-tier memory: short / medium / long.

    Items are promoted from short → medium → long based on
    ``memory_hierarchy_consolidation_times`` (seconds since insertion).
    Each tier is an :class:`EpisodicMemory` with a separate capacity:
    short = capacity, medium = 2 * capacity, long = 4 * capacity.
    """

    def __init__(
        self,
        dim: int = 64,
        chaotic_memory=None,
        rules: MemoryRules | None = None,
    ):
        self.dim = int(dim)
        self.rules = rules or MemoryRules()
        capacity = int(self.rules.memory_capacity)
        # Each tier has growing capacity: short term has the tightest.
        self.short = EpisodicMemory(
            dim=dim, chaotic_memory=chaotic_memory,
            rules=MemoryRules(memory_capacity=capacity),
        )
        self.medium = EpisodicMemory(
            dim=dim, chaotic_memory=None,
            rules=MemoryRules(memory_capacity=2 * capacity),
        )
        self.long = EpisodicMemory(
            dim=dim, chaotic_memory=None,
            rules=MemoryRules(memory_capacity=4 * capacity),
        )

    def encode(
        self,
        observation: np.ndarray,
        label: str | int | None = None,
    ) -> dict:
        """Insert into short-term memory."""
        return self.short.encode(observation, label=label)

    def consolidate(self) -> dict:
        """Promote items whose age exceeds the tier thresholds.

        Returns counts: ``{'short_to_medium': int, 'medium_to_long': int}``.
        """
        times = self.rules.memory_hierarchy_consolidation_times
        now = float(time.time())
        short_threshold = float(times[0]) if len(times) > 0 else 60.0
        medium_threshold = float(times[1]) if len(times) > 1 else 3600.0

        # Short → medium
        short_to_medium = 0
        keep_short_obs: list[np.ndarray] = []
        keep_short_labels: list[Any] = []
        keep_short_times: list[float] = []
        for obs, lbl, ts in zip(
            self.short._observations,
            self.short._labels,
            self.short._timestamps,
        ):
            age = now - ts
            if age >= short_threshold:
                self.medium.encode(obs, label=lbl)
                short_to_medium += 1
            else:
                keep_short_obs.append(obs)
                keep_short_labels.append(lbl)
                keep_short_times.append(ts)
        self.short._observations = keep_short_obs
        self.short._labels = keep_short_labels
        self.short._timestamps = keep_short_times

        # Medium → long
        medium_to_long = 0
        keep_medium_obs: list[np.ndarray] = []
        keep_medium_labels: list[Any] = []
        keep_medium_times: list[float] = []
        for obs, lbl, ts in zip(
            self.medium._observations,
            self.medium._labels,
            self.medium._timestamps,
        ):
            age = now - ts
            if age >= medium_threshold:
                self.long.encode(obs, label=lbl)
                medium_to_long += 1
            else:
                keep_medium_obs.append(obs)
                keep_medium_labels.append(lbl)
                keep_medium_times.append(ts)
        self.medium._observations = keep_medium_obs
        self.medium._labels = keep_medium_labels
        self.medium._timestamps = keep_medium_times

        return {
            "short_to_medium": int(short_to_medium),
            "medium_to_long": int(medium_to_long),
        }

    def retrieve(self, query: np.ndarray, top_k: int | None = None) -> dict:
        """Top-k retrieval across all three tiers, merged by similarity."""
        results = {"items": [], "similarities": []}
        for tier_name, tier in (("short", self.short), ("medium", self.medium), ("long", self.long)):
            r = tier.retrieve(query, top_k=top_k)
            for item, sim in zip(r["items"], r["similarities"]):
                item_with_tier = dict(item)
                item_with_tier["tier"] = tier_name
                results["items"].append(item_with_tier)
                results["similarities"].append(sim)
        # Sort all by descending similarity and truncate to top_k.
        order = np.argsort(-np.asarray(results["similarities"]), kind="stable")
        k = int(top_k if top_k is not None else self.rules.memory_top_k)
        k = max(0, min(k, len(results["items"])))
        results["items"] = [results["items"][i] for i in order[:k]]
        results["similarities"] = [float(results["similarities"][i]) for i in order[:k]]
        return results

    def __len__(self) -> int:
        return len(self.short) + len(self.medium) + len(self.long)


# ----------------------------------------------------------------------
# SpreadingActivationMemory
# ----------------------------------------------------------------------


class SpreadingActivationMemory:
    """Collins & Loftus (1975) spreading activation retrieval.

    Items form a similarity graph; retrieval diffuses activation along
    edges up to ``memory_spreading_depth`` hops. Returns items ranked
    by final activation level.
    """

    def __init__(
        self,
        dim: int = 64,
        rules: MemoryRules | None = None,
    ):
        self.dim = int(dim)
        self.rules = rules or MemoryRules()
        self._observations: list[np.ndarray] = []
        self._labels: list[Any] = []
        self._timestamps: list[float] = []

    def encode(
        self,
        observation: np.ndarray,
        label: str | int | None = None,
    ) -> dict:
        vec = _sanitize_vector(observation)
        capacity = int(self.rules.memory_capacity)
        if len(self._observations) >= capacity:
            self._observations.pop(0)
            self._labels.pop(0)
            self._timestamps.pop(0)
        ts = float(time.time())
        self._observations.append(vec.copy())
        self._labels.append(label)
        self._timestamps.append(ts)
        return {"position": len(self._observations) - 1, "timestamp": ts}

    def retrieve(
        self,
        query: np.ndarray,
        top_k: int | None = None,
    ) -> dict:
        """Spreading-activation retrieval.

        Steps:
        1. Seed activation = cosine similarity to query.
        2. For each hop up to ``memory_spreading_depth``, propagate
           activation along edges weighted by inter-item similarity
           (decay factor = ``memory_decay``).
        3. Return top-k items by final activation.
        """
        q = _sanitize_vector(query)
        if not self._observations:
            return {"activations": np.array([]), "items": [], "similarities": []}
        n = len(self._observations)
        sims_to_query = np.array([_cosine_similarity(q, o) for o in self._observations])
        # Build similarity matrix (n x n) with zero diagonal.
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                s = _cosine_similarity(self._observations[i], self._observations[j])
                sim_matrix[i, j] = s
                sim_matrix[j, i] = s
        # Iterative spreading.
        activations = sims_to_query.copy()
        decay = float(self.rules.memory_decay)
        for _ in range(int(self.rules.memory_spreading_depth)):
            new_activations = decay * (sim_matrix @ activations) + sims_to_query
            activations = 0.5 * activations + 0.5 * new_activations
        # Normalize.
        max_act = float(activations.max()) if activations.size > 0 else 0.0
        if max_act > 1e-12:
            activations = activations / max_act
        k = int(top_k if top_k is not None else self.rules.memory_top_k)
        k = max(0, min(k, n))
        order = np.argsort(-activations, kind="stable")[:k]
        items = []
        similarities = []
        for idx in order:
            items.append({
                "id": int(idx),
                "label": self._labels[idx],
                "timestamp": float(self._timestamps[idx]),
            })
            similarities.append(float(activations[idx]))
        return {
            "activations": activations,
            "items": items,
            "similarities": similarities,
        }

    def __len__(self) -> int:
        return len(self._observations)


# ----------------------------------------------------------------------
# ForgetfulMemory
# ----------------------------------------------------------------------


class ForgetfulMemory:
    """Ebbinghaus forgetting curve: ``R = exp(-t / S)``.

    Items are not deleted; their retrieval weight decays over time
    according to the Ebbinghaus curve. The effective similarity is
    ``sim * R(t)`` where ``R(t) = exp(-t / S)`` and ``S`` is the
    stability constant.
    """

    def __init__(
        self,
        dim: int = 64,
        rules: MemoryRules | None = None,
    ):
        self.dim = int(dim)
        self.rules = rules or MemoryRules()
        self._observations: list[np.ndarray] = []
        self._labels: list[Any] = []
        self._timestamps: list[float] = []

    def encode(
        self,
        observation: np.ndarray,
        label: str | int | None = None,
    ) -> dict:
        vec = _sanitize_vector(observation)
        capacity = int(self.rules.memory_capacity)
        if len(self._observations) >= capacity:
            self._observations.pop(0)
            self._labels.pop(0)
            self._timestamps.pop(0)
        ts = float(time.time())
        self._observations.append(vec.copy())
        self._labels.append(label)
        self._timestamps.append(ts)
        return {"position": len(self._observations) - 1, "timestamp": ts}

    def retrieve(
        self,
        query: np.ndarray,
        top_k: int | None = None,
        at_time: float | None = None,
    ) -> dict:
        """Top-k retrieval with Ebbinghaus decay.

        ``at_time`` allows retrospective retrieval (e.g. simulate
        retrieval 1 hour ago). Defaults to now.
        """
        q = _sanitize_vector(query)
        if not self._observations:
            return {"items": [], "similarities": [], "retention": []}
        now = float(at_time if at_time is not None else time.time())
        S = float(self.rules.memory_ebbinghaus_S)
        S = max(S, 1e-6)  # guard against zero / negative
        retentions = []
        effective_sims = []
        for obs, ts in zip(self._observations, self._timestamps):
            age = max(0.0, now - ts)
            R = float(np.exp(-age / S))
            sim = _cosine_similarity(q, obs)
            retentions.append(R)
            effective_sims.append(sim * R)
        effective_sims_arr = np.asarray(effective_sims)
        k = int(top_k if top_k is not None else self.rules.memory_top_k)
        k = max(0, min(k, len(self._observations)))
        if k == 0:
            return {"items": [], "similarities": [], "retention": []}
        order = np.argsort(-effective_sims_arr, kind="stable")[:k]
        items = []
        similarities = []
        retention_out = []
        for idx in order:
            items.append({
                "id": int(idx),
                "label": self._labels[idx],
                "timestamp": float(self._timestamps[idx]),
            })
            similarities.append(float(effective_sims[idx]))
            retention_out.append(float(retentions[idx]))
        return {
            "items": items,
            "similarities": similarities,
            "retention": retention_out,
        }

    def review(self, ids: list[int]) -> dict:
        """Reset the timestamp of reviewed items (Ebbinghaus 'spaced
        repetition' reset). Returns count reset.
        """
        if not ids:
            return {"reviewed": 0}
        now = float(time.time())
        reviewed = 0
        for i in ids:
            if 0 <= int(i) < len(self._timestamps):
                self._timestamps[int(i)] = now
                reviewed += 1
        return {"reviewed": int(reviewed)}

    def __len__(self) -> int:
        return len(self._observations)


# ----------------------------------------------------------------------
# MemoryIndexer
# ----------------------------------------------------------------------


class MemoryIndexer:
    """Faiss-lite-style inverted index over episodic memory.

    Maintains a matrix of observations and an L2-normalized view for
    fast cosine similarity via matrix multiply. Supports incremental
    update (add / remove) without rebuilding from scratch.
    """

    def __init__(
        self,
        dim: int = 64,
        rules: MemoryRules | None = None,
    ):
        self.dim = int(dim)
        self.rules = rules or MemoryRules()
        self._matrix: np.ndarray = np.zeros((0, dim), dtype=float)
        self._normalized: np.ndarray = np.zeros((0, dim), dtype=float)
        self._labels: list[Any] = []

    def add(
        self,
        observation: np.ndarray,
        label: str | int | None = None,
    ) -> dict:
        """Add an observation to the index. Returns position/label."""
        vec = _sanitize_vector(observation)
        if vec.size != self.dim:
            # Auto-resize the index if the first observation has a
            # different dim (allows lazy dim inference).
            if len(self._matrix) == 0:
                self.dim = int(vec.size)
                self._matrix = np.zeros((0, self.dim), dtype=float)
                self._normalized = np.zeros((0, self.dim), dtype=float)
            else:
                raise ValueError(
                    f"observation dim {vec.size} != index dim {self.dim}"
                )
        self._matrix = np.vstack([self._matrix, vec.reshape(1, -1)])
        norm = float(np.linalg.norm(vec))
        if norm < 1e-12:
            self._normalized = np.vstack([self._normalized, np.zeros_like(vec)])
        else:
            self._normalized = np.vstack([
                self._normalized,
                (vec / norm).reshape(1, -1),
            ])
        self._labels.append(label)
        return {"position": int(len(self._matrix) - 1), "label": label}

    def search(
        self,
        query: np.ndarray,
        top_k: int | None = None,
    ) -> dict:
        """Top-k search via normalized matrix multiply. O(n*d) per query."""
        q = _sanitize_vector(query)
        if q.size != self.dim:
            raise ValueError(
                f"query dim {q.size} != index dim {self.dim}"
            )
        if len(self._matrix) == 0:
            return {"positions": [], "similarities": [], "labels": []}
        q_norm = float(np.linalg.norm(q))
        if q_norm < 1e-12:
            q_normalized = np.zeros_like(q)
        else:
            q_normalized = q / q_norm
        sims = self._normalized @ q_normalized  # (n,)
        k = int(top_k if top_k is not None else self.rules.memory_top_k)
        k = max(0, min(k, len(self._matrix)))
        if k == 0:
            return {"positions": [], "similarities": [], "labels": []}
        order = np.argsort(-sims, kind="stable")[:k]
        return {
            "positions": [int(i) for i in order],
            "similarities": [float(sims[i]) for i in order],
            "labels": [self._labels[i] for i in order],
        }

    def remove(self, positions: list[int]) -> dict:
        """Remove items by position. Returns count removed."""
        if not positions or len(self._matrix) == 0:
            return {"removed": 0}
        keep_mask = np.ones(len(self._matrix), dtype=bool)
        for p in positions:
            if 0 <= int(p) < len(self._matrix):
                keep_mask[int(p)] = False
        self._matrix = self._matrix[keep_mask]
        self._normalized = self._normalized[keep_mask]
        self._labels = [l for l, k in zip(self._labels, keep_mask) if k]
        return {"removed": int((~keep_mask).sum())}

    def build_from(
        self,
        observations: np.ndarray,
        labels: list | None = None,
    ) -> dict:
        """Rebuild the entire index from a 2D array. Returns count.

        Optional ``labels`` (length = n_observations) is attached to each
        indexed row; ``None`` fills with ``[None] * n``.
        """
        arr = np.asarray(observations, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"observations must be 2D, got shape {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("observations must be finite")
        n = int(len(arr))
        if labels is not None:
            labels_list = list(labels)
            if len(labels_list) != n:
                raise ValueError(
                    f"labels length {len(labels_list)} != observations {n}"
                )
        else:
            labels_list = [None] * n
        self.dim = int(arr.shape[1])
        self._matrix = arr.copy()
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms_safe = np.where(norms < 1e-12, 1.0, norms)
        self._normalized = arr / norms_safe
        self._labels = labels_list
        return {"indexed": int(n)}

    def __len__(self) -> int:
        return int(len(self._matrix))
