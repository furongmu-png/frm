# src/zero_data_model/capabilities/memory.py
"""Phase 6 — Memory capabilities.

Composes the chaotic associative memory from ``causal_emergence`` with
domain-specific episodic / working / context memory abstractions. No
external ML library required; everything is numpy + the rules library.

Classes
-------
EpisodicMemory
    Store ``(observation, label, timestamp)`` triples and retrieve the
    top-k most similar entries. Optionally uses
    ``ChaoticAssociativeMemory.recall()`` to boost queries with a
    Lorenz-basin attractor before retrieval.

WorkingMemory
    Sliding window of the last ``capacity`` observations with attention
    weights relative to a focus vector. Supports push / pop / peek.

ContextMemory
    Conversation-style memory of ``(user_msg, agent_msg, turn_id)``
    triples with a summary method that returns the mean embedding of
    the most recent ``top_k`` turns.

MemoryConsolidator
    Promote high-weight WorkingMemory items into EpisodicMemory and
    drop near-duplicates above ``memory_similarity_threshold``.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .rules import MemoryRules


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _sanitize_vector(observation: np.ndarray) -> np.ndarray:
    """Flatten / float-cast / NaN-guard an observation vector."""
    arr = np.asarray(observation, dtype=float).ravel()
    if arr.size == 0:
        return arr
    if not np.all(np.isfinite(arr)):
        raise ValueError("observation must be finite (no NaN or Inf)")
    return arr


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity, NaN-safe. Returns 0.0 for zero-norm inputs."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ----------------------------------------------------------------------
# EpisodicMemory
# ----------------------------------------------------------------------


class EpisodicMemory:
    """Episodic store with optional chaotic-memory query boost."""

    def __init__(
        self,
        dim: int = 64,
        chaotic_memory=None,
        rules: MemoryRules | None = None,
    ):
        self.dim = int(dim)
        self.chaotic_memory = chaotic_memory  # optional ChaoticAssociativeMemory
        self.rules = rules or MemoryRules()
        self._observations: list[np.ndarray] = []
        self._labels: list[Any] = []
        self._timestamps: list[float] = []
        self._next_id: int = 0

    # ------------------------------------------------------------------
    def encode(
        self,
        observation: np.ndarray,
        label: str | int | None = None,
    ) -> dict:
        """Store an observation. Returns id/label/timestamp."""
        vec = _sanitize_vector(observation)
        capacity = int(self.rules.memory_capacity)
        if len(self._observations) >= capacity:
            # Evict oldest (FIFO).
            self._observations.pop(0)
            self._labels.pop(0)
            self._timestamps.pop(0)
        ts = float(time.time())
        self._observations.append(vec.copy())
        self._labels.append(label)
        self._timestamps.append(ts)
        item_id = int(self._next_id)
        self._next_id += 1
        return {"id": item_id, "label": label, "timestamp": ts}

    def retrieve(
        self,
        query: np.ndarray,
        top_k: int | None = None,
    ) -> dict:
        """Top-k retrieval by cosine similarity.

        Returns ``{'items': list[dict], 'similarities': list[float]}``.
        Items with similarity > ``memory_similarity_threshold`` may be
        boosted by ``ChaoticAssociativeMemory.recall()``.
        """
        q = _sanitize_vector(query)
        if not self._observations:
            return {"items": [], "similarities": []}

        # Optional chaotic-memory boost: use the query to recall a Lorenz
        # attractor trajectory and average it back into the query. This
        # is a soft prior — only fires when chaotic_memory is set and
        # produces a non-empty trajectory.
        if self.chaotic_memory is not None:
            try:
                recall = self.chaotic_memory.recall(q, n_steps=4)
                traj = np.asarray(recall.get("trajectory", []), dtype=float)
                if traj.size > 0 and np.all(np.isfinite(traj)):
                    # Recenter the query toward the mean of the
                    # recalled trajectory (gentle pull toward the
                    # attractor basin).
                    q = 0.5 * q + 0.5 * traj.mean(axis=0)
            except Exception:
                # Chaotic memory boost is best-effort; never fail retrieval.
                pass

        sims = np.array(
            [_cosine_similarity(q, obs) for obs in self._observations]
        )
        k = int(top_k if top_k is not None else self.rules.memory_top_k)
        k = max(0, min(k, len(self._observations)))
        if k == 0:
            return {"items": [], "similarities": []}
        # Descending sort, stable on insertion order for ties.
        order = np.argsort(-sims, kind="stable")[:k]
        items = []
        similarities = []
        for idx in order:
            items.append({
                "id": int(idx),  # positional id (not the encode() id)
                "label": self._labels[idx],
                "timestamp": float(self._timestamps[idx]),
                "observation": self._observations[idx].copy(),
            })
            similarities.append(float(sims[idx]))
        return {"items": items, "similarities": similarities}

    def forget(self, ids: list[int]) -> dict:
        """Remove items by positional id. Returns count forgotten."""
        if not ids:
            return {"forgotten": 0}
        keep_mask = np.ones(len(self._observations), dtype=bool)
        for i in ids:
            if 0 <= int(i) < len(self._observations):
                keep_mask[int(i)] = False
        self._observations = [o for o, k in zip(self._observations, keep_mask) if k]
        self._labels = [l for l, k in zip(self._labels, keep_mask) if k]
        self._timestamps = [t for t, k in zip(self._timestamps, keep_mask) if k]
        return {"forgotten": int((~keep_mask).sum())}

    def clear(self) -> dict:
        """Clear all. Returns count cleared."""
        n = len(self._observations)
        self._observations.clear()
        self._labels.clear()
        self._timestamps.clear()
        self._next_id = 0
        return {"cleared": int(n)}

    def __len__(self) -> int:
        return len(self._observations)


# ----------------------------------------------------------------------
# WorkingMemory
# ----------------------------------------------------------------------


class WorkingMemory:
    """Sliding window of recent observations with attention weights."""

    def __init__(
        self,
        dim: int = 64,
        rules: MemoryRules | None = None,
    ):
        self.dim = int(dim)
        self.rules = rules or MemoryRules()
        self._observations: list[np.ndarray] = []
        self._timestamps: list[float] = []
        self._focus: np.ndarray | None = None

    # ------------------------------------------------------------------
    def push(self, observation: np.ndarray) -> dict:
        """Push an observation; evicts oldest if at capacity."""
        vec = _sanitize_vector(observation)
        capacity = int(self.rules.memory_capacity)
        if len(self._observations) >= capacity:
            self._observations.pop(0)
            self._timestamps.pop(0)
        ts = float(time.time())
        self._observations.append(vec.copy())
        self._timestamps.append(ts)
        return {"position": len(self._observations) - 1, "timestamp": ts}

    def pop(self) -> dict:
        """Pop the most recent observation."""
        if not self._observations:
            return {"observation": None, "timestamp": None}
        obs = self._observations.pop()
        ts = self._timestamps.pop()
        return {"observation": obs, "timestamp": ts}

    def peek(self) -> dict:
        """Peek the most recent observation without removing it."""
        if not self._observations:
            return {"observation": None, "timestamp": None}
        return {
            "observation": self._observations[-1].copy(),
            "timestamp": float(self._timestamps[-1]),
        }

    def set_focus(self, focus: np.ndarray) -> dict:
        """Set the focus vector for attention weight computation."""
        self._focus = _sanitize_vector(focus)
        return {"dim": int(self._focus.size)}

    def attention_weights(self) -> dict:
        """Compute attention weights = softmax(similarity to focus)."""
        if not self._observations:
            return {"weights": np.array([]), "focus_set": self._focus is not None}
        if self._focus is None:
            # Uniform weights when no focus is set.
            n = len(self._observations)
            w = np.full(n, 1.0 / n)
        else:
            sims = np.array([
                _cosine_similarity(self._focus, o)
                for o in self._observations
            ])
            # Softmax with numerical stability.
            sims = sims - sims.max()
            exp = np.exp(sims)
            w = exp / max(exp.sum(), 1e-12)
        return {"weights": w, "focus_set": self._focus is not None}

    def summary(self) -> dict:
        """Attention-weighted mean of observations."""
        if not self._observations:
            return {
                "summary": np.zeros(0),
                "n_items": 0,
                "focus_set": self._focus is not None,
            }
        weights = self.attention_weights()["weights"]
        stack = np.stack(self._observations)
        summary = (stack * weights[:, None]).sum(axis=0)
        return {
            "summary": summary,
            "n_items": int(len(self._observations)),
            "focus_set": self._focus is not None,
        }

    def clear(self) -> dict:
        n = len(self._observations)
        self._observations.clear()
        self._timestamps.clear()
        self._focus = None
        return {"cleared": int(n)}

    def __len__(self) -> int:
        return len(self._observations)


# ----------------------------------------------------------------------
# ContextMemory
# ----------------------------------------------------------------------


class ContextMemory:
    """Conversation-style memory of (user_msg, agent_msg, turn_id)."""

    def __init__(
        self,
        dim: int = 64,
        sentence_encoder=None,
        rules: MemoryRules | None = None,
    ):
        self.dim = int(dim)
        self.sentence_encoder = sentence_encoder  # optional nlp.SentenceEncoder
        self.rules = rules or MemoryRules()
        self._turns: list[dict] = []  # {'user', 'agent', 'turn_id', 'embedding'}

    def add_turn(
        self,
        user_msg: str,
        agent_msg: str,
        embedding: np.ndarray | None = None,
    ) -> dict:
        """Add a turn. If embedding is None and sentence_encoder is set,
        compute it from ``user_msg``; otherwise expect embedding to be set.
        Returns turn_id / status.
        """
        if embedding is None and self.sentence_encoder is not None:
            try:
                embedding = self.sentence_encoder.encode(user_msg)
            except Exception:
                embedding = None
        if embedding is not None:
            embedding = _sanitize_vector(embedding)
        turn_id = len(self._turns)
        self._turns.append({
            "user": str(user_msg),
            "agent": str(agent_msg),
            "turn_id": int(turn_id),
            "embedding": embedding.copy() if embedding is not None else None,
        })
        return {"turn_id": int(turn_id), "status": "stored"}

    def summary(self, top_k: int | None = None) -> dict:
        """Mean embedding of the most recent ``top_k`` turns."""
        if not self._turns:
            return {"summary": np.zeros(0), "n_turns": 0}
        k = int(top_k if top_k is not None else self.rules.memory_top_k)
        k = max(1, min(k, len(self._turns)))
        recent = self._turns[-k:]
        embeddings = [t["embedding"] for t in recent if t["embedding"] is not None]
        if not embeddings:
            return {"summary": np.zeros(0), "n_turns": int(k)}
        summary = np.mean(np.stack(embeddings), axis=0)
        return {"summary": summary, "n_turns": int(k)}

    def recent(self, top_k: int | None = None) -> dict:
        """Return the most recent ``top_k`` turns as dicts."""
        if not self._turns:
            return {"turns": []}
        k = int(top_k if top_k is not None else self.rules.memory_top_k)
        k = max(1, min(k, len(self._turns)))
        return {
            "turns": [
                {"user": t["user"], "agent": t["agent"], "turn_id": t["turn_id"]}
                for t in self._turns[-k:]
            ]
        }

    def clear(self) -> dict:
        n = len(self._turns)
        self._turns.clear()
        return {"cleared": int(n)}

    def __len__(self) -> int:
        return len(self._turns)


# ----------------------------------------------------------------------
# MemoryConsolidator
# ----------------------------------------------------------------------


class MemoryConsolidator:
    """Promote high-weight working-memory items into episodic memory.

    Also drops near-duplicates in episodic memory whose pairwise cosine
    similarity exceeds ``memory_similarity_threshold``.
    """

    def __init__(
        self,
        dim: int = 64,
        rules: MemoryRules | None = None,
    ):
        self.dim = int(dim)
        self.rules = rules or MemoryRules()

    def consolidate(
        self,
        working_memory: WorkingMemory,
        episodic_memory: EpisodicMemory,
    ) -> dict:
        """Move high-attention items from working → episodic.

        Returns counts: ``{'promoted': int, 'deduplicated': int}``.
        """
        if len(working_memory) == 0:
            return {"promoted": 0, "deduplicated": 0}
        weights = working_memory.attention_weights()["weights"]
        threshold = float(self.rules.memory_consolidation_weight)
        promoted = 0
        for obs, w in zip(working_memory._observations, weights):
            if float(w) >= threshold:
                episodic_memory.encode(obs, label=f"consolidated:{float(w):.3f}")
                promoted += 1

        # Deduplicate episodic store.
        deduped = self._deduplicate(episodic_memory)
        return {"promoted": int(promoted), "deduplicated": int(deduped)}

    def _deduplicate(self, episodic_memory: EpisodicMemory) -> int:
        """Remove items whose cosine similarity to an earlier kept item
        exceeds ``memory_similarity_threshold``. Returns count removed.
        """
        if len(episodic_memory) < 2:
            return 0
        threshold = float(self.rules.memory_similarity_threshold)
        obs_list = episodic_memory._observations
        keep_mask = np.ones(len(obs_list), dtype=bool)
        for i in range(len(obs_list)):
            if not keep_mask[i]:
                continue
            for j in range(i + 1, len(obs_list)):
                if not keep_mask[j]:
                    continue
                if _cosine_similarity(obs_list[i], obs_list[j]) > threshold:
                    keep_mask[j] = False
        removed = int((~keep_mask).sum())
        episodic_memory._observations = [
            o for o, k in zip(obs_list, keep_mask) if k
        ]
        episodic_memory._labels = [
            l for l, k in zip(episodic_memory._labels, keep_mask) if k
        ]
        episodic_memory._timestamps = [
            t for t, k in zip(episodic_memory._timestamps, keep_mask) if k
        ]
        return removed
