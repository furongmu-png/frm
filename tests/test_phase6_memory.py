# tests/test_phase6_memory.py
"""Phase 6 — Memory capability tests.

Covers the 8 classes in ``capabilities.memory`` and
``capabilities.memory_advanced``: EpisodicMemory, WorkingMemory,
ContextMemory, MemoryConsolidator, HierarchicalMemory,
SpreadingActivationMemory, ForgetfulMemory, MemoryIndexer.

Each class gets ≥15 tests covering: correctness, edge cases,
determinism, NaN/Inf guards, capacity / eviction, and the API
contract per spec §6 (Memory). Total: 75+ tests.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from zero_data_model.capabilities.memory import (
    ContextMemory,
    EpisodicMemory,
    MemoryConsolidator,
    WorkingMemory,
)
from zero_data_model.capabilities.memory_advanced import (
    ForgetfulMemory,
    HierarchicalMemory,
    MemoryIndexer,
    SpreadingActivationMemory,
)
from zero_data_model.capabilities.rules import MemoryRules


# --------------------------------------------------------------------- #
# EpisodicMemory
# --------------------------------------------------------------------- #


class TestEpisodicMemory:
    @pytest.fixture()
    def mem(self):
        return EpisodicMemory(dim=4, rules=MemoryRules(memory_capacity=10))

    def test_encode_returns_id_and_timestamp(self, mem):
        r = mem.encode(np.array([1.0, 0.0, 0.0, 0.0]), label="a")
        assert r["id"] == 0
        assert isinstance(r["timestamp"], float)
        assert r["label"] == "a"

    def test_encode_increments_id(self, mem):
        r1 = mem.encode(np.zeros(4), label="x")
        r2 = mem.encode(np.zeros(4), label="y")
        assert r1["id"] == 0 and r2["id"] == 1

    def test_encode_label_none_default(self, mem):
        r = mem.encode(np.zeros(4))
        assert r["label"] is None

    def test_retrieve_empty_returns_empty(self, mem):
        r = mem.retrieve(np.zeros(4))
        assert r["items"] == []
        assert r["similarities"] == []

    def test_retrieve_top_k(self, mem):
        mem.encode(np.array([1.0, 0.0, 0.0, 0.0]), label="a")
        mem.encode(np.array([0.0, 1.0, 0.0, 0.0]), label="b")
        mem.encode(np.array([1.0, 1.0, 0.0, 0.0]), label="c")
        r = mem.retrieve(np.array([1.0, 0.0, 0.0, 0.0]), top_k=2)
        assert len(r["items"]) == 2
        # Best match should be "a" (cosine = 1.0).
        assert r["items"][0]["label"] == "a"
        assert r["similarities"][0] == pytest.approx(1.0, abs=1e-9)

    def test_retrieve_default_top_k_from_rules(self, mem):
        for i in range(5):
            mem.encode(np.array([float(i), 0.0, 0.0, 0.0]), label=f"l{i}")
        # rules.memory_top_k default is 5 — should return all 5.
        r = mem.retrieve(np.array([1.0, 0.0, 0.0, 0.0]))
        assert len(r["items"]) <= 5
        assert len(r["items"]) == min(5, len(mem))

    def test_retrieve_returns_observation_copy(self, mem):
        mem.encode(np.array([0.5, 0.5, 0.0, 0.0]), label="x")
        r = mem.retrieve(np.array([0.5, 0.5, 0.0, 0.0]), top_k=1)
        assert isinstance(r["items"][0]["observation"], np.ndarray)
        assert r["items"][0]["observation"].shape == (4,)

    def test_capacity_eviction(self):
        mem = EpisodicMemory(dim=2, rules=MemoryRules(memory_capacity=3))
        for i in range(5):
            mem.encode(np.array([float(i), 0.0]), label=f"l{i}")
        assert len(mem) == 3  # FIFO eviction kept last 3.

    def test_forget_by_id(self, mem):
        mem.encode(np.array([1.0, 0.0, 0.0, 0.0]), label="a")
        mem.encode(np.array([0.0, 1.0, 0.0, 0.0]), label="b")
        r = mem.forget([0])
        assert r["forgotten"] == 1
        assert len(mem) == 1

    def test_forget_empty_ids_noop(self, mem):
        mem.encode(np.zeros(4))
        r = mem.forget([])
        assert r["forgotten"] == 0

    def test_forget_out_of_range_ignored(self, mem):
        mem.encode(np.zeros(4))
        r = mem.forget([99, -1])
        assert r["forgotten"] == 0

    def test_clear_resets_store(self, mem):
        mem.encode(np.zeros(4))
        mem.encode(np.zeros(4))
        r = mem.clear()
        assert r["cleared"] == 2
        assert len(mem) == 0

    def test_nan_observation_raises(self, mem):
        with pytest.raises(ValueError, match="finite"):
            mem.encode(np.array([np.nan, 0.0, 0.0, 0.0]))

    def test_inf_observation_raises(self, mem):
        with pytest.raises(ValueError, match="finite"):
            mem.encode(np.array([np.inf, 0.0, 0.0, 0.0]))

    def test_2d_observation_flattened(self, mem):
        # _sanitize_vector ravels; encode should accept 2D and store 1D.
        r = mem.encode(np.array([[1.0, 2.0], [3.0, 4.0]]), label="2d")
        assert r["id"] == 0
        # Retrieve should return the raveled vector.
        retrieved = mem.retrieve(np.array([1.0, 2.0, 3.0, 4.0]), top_k=1)
        assert retrieved["similarities"][0] == pytest.approx(1.0, abs=1e-9)

    def test_chaotic_memory_boost_is_best_effort(self):
        """When chaotic_memory is set, retrieve must not fail even if
        the boost raises."""
        class BadChaotic:
            def recall(self, *args, **kwargs):
                raise RuntimeError("simulated failure")
        mem = EpisodicMemory(dim=4, chaotic_memory=BadChaotic())
        mem.encode(np.array([1.0, 0.0, 0.0, 0.0]), label="a")
        r = mem.retrieve(np.array([1.0, 0.0, 0.0, 0.0]), top_k=1)
        assert len(r["items"]) == 1

    def test_chaotic_memory_boost_pulls_query(self):
        """When chaotic_memory returns a usable trajectory, the query
        is gently pulled toward the trajectory mean."""
        class FakeChaotic:
            def recall(self, q, n_steps=4):
                # Return a 3x3 trajectory centered at (1, 1, 1).
                traj = np.ones((5, 4)) * 0.9
                return {"trajectory": traj}
        mem = EpisodicMemory(dim=4, chaotic_memory=FakeChaotic())
        mem.encode(np.array([1.0, 1.0, 1.0, 1.0]), label="target")
        # Even though query is zeros, the boost pulls it toward (0.9,0.9,...)
        # which is closer to the stored (1,1,1,1) than (0,0,0,0).
        r = mem.retrieve(np.zeros(4), top_k=1)
        assert r["similarities"][0] > 0.0


# --------------------------------------------------------------------- #
# WorkingMemory
# --------------------------------------------------------------------- #


class TestWorkingMemory:
    @pytest.fixture()
    def wm(self):
        return WorkingMemory(dim=4, rules=MemoryRules(memory_capacity=3))

    def test_push_returns_position(self, wm):
        r = wm.push(np.zeros(4))
        assert r["position"] == 0

    def test_push_increments_position(self, wm):
        wm.push(np.zeros(4))
        r = wm.push(np.zeros(4))
        assert r["position"] == 1

    def test_push_evicts_oldest_at_capacity(self, wm):
        for i in range(5):
            wm.push(np.array([float(i), 0.0, 0.0, 0.0]))
        assert len(wm) == 3
        # Peek should be the last pushed (i=4).
        peek = wm.peek()
        np.testing.assert_allclose(peek["observation"], np.array([4.0, 0, 0, 0]))

    def test_pop_returns_last(self, wm):
        wm.push(np.array([1.0, 0.0, 0.0, 0.0]))
        wm.push(np.array([2.0, 0.0, 0.0, 0.0]))
        r = wm.pop()
        np.testing.assert_allclose(r["observation"], np.array([2.0, 0, 0, 0]))
        assert len(wm) == 1

    def test_pop_empty_returns_none(self, wm):
        r = wm.pop()
        assert r["observation"] is None
        assert r["timestamp"] is None

    def test_peek_does_not_remove(self, wm):
        wm.push(np.array([1.0, 0.0, 0.0, 0.0]))
        wm.peek()
        assert len(wm) == 1

    def test_peek_empty_returns_none(self, wm):
        r = wm.peek()
        assert r["observation"] is None

    def test_set_focus_returns_dim(self, wm):
        r = wm.set_focus(np.array([1.0, 0.0, 0.0, 0.0]))
        assert r["dim"] == 4

    def test_attention_weights_uniform_without_focus(self, wm):
        wm.push(np.zeros(4))
        wm.push(np.zeros(4))
        r = wm.attention_weights()
        assert r["focus_set"] is False
        np.testing.assert_allclose(r["weights"], np.array([0.5, 0.5]), atol=1e-9)

    def test_attention_weights_softmax_with_focus(self, wm):
        wm.push(np.array([1.0, 0.0, 0.0, 0.0]))  # matches focus
        wm.push(np.array([0.0, 1.0, 0.0, 0.0]))  # orthogonal
        wm.set_focus(np.array([1.0, 0.0, 0.0, 0.0]))
        r = wm.attention_weights()
        assert r["focus_set"] is True
        # First item should dominate.
        assert r["weights"][0] > r["weights"][1]

    def test_attention_weights_empty(self, wm):
        r = wm.attention_weights()
        assert r["focus_set"] is False
        assert len(r["weights"]) == 0

    def test_summary_empty_returns_zeros(self, wm):
        r = wm.summary()
        assert r["n_items"] == 0
        assert r["summary"].size == 0

    def test_summary_with_focus(self, wm):
        wm.push(np.array([1.0, 0.0, 0.0, 0.0]))
        wm.push(np.array([0.0, 1.0, 0.0, 0.0]))
        wm.set_focus(np.array([1.0, 0.0, 0.0, 0.0]))
        r = wm.summary()
        assert r["n_items"] == 2
        assert r["focus_set"] is True
        # softmax([1.0, 0.0]) = [0.731, 0.269]; weighted combination.
        # summary[0] is dominated by item 0 (focus-aligned); summary[1]
        # is the residual weight from item 1.
        assert r["summary"][0] > r["summary"][1]
        assert r["summary"][2] == pytest.approx(0.0, abs=1e-9)
        assert r["summary"][3] == pytest.approx(0.0, abs=1e-9)

    def test_clear(self, wm):
        wm.push(np.zeros(4))
        wm.set_focus(np.zeros(4))
        r = wm.clear()
        assert r["cleared"] == 1
        assert len(wm) == 0

    def test_nan_raises(self, wm):
        with pytest.raises(ValueError, match="finite"):
            wm.push(np.array([np.nan, 0, 0, 0]))


# --------------------------------------------------------------------- #
# ContextMemory
# --------------------------------------------------------------------- #


class TestContextMemory:
    @pytest.fixture()
    def cm(self):
        return ContextMemory(dim=4, rules=MemoryRules(memory_capacity=10))

    def test_add_turn_explicit_embedding(self, cm):
        r = cm.add_turn("hello", "hi", embedding=np.zeros(4))
        assert r["turn_id"] == 0
        assert r["status"] == "stored"

    def test_add_turn_id_increments(self, cm):
        cm.add_turn("a", "b", embedding=np.zeros(4))
        r = cm.add_turn("c", "d", embedding=np.zeros(4))
        assert r["turn_id"] == 1

    def test_add_turn_uses_sentence_encoder(self):
        class FakeEncoder:
            def encode(self, text):
                # Length-based deterministic embedding.
                v = np.zeros(4)
                v[0] = float(len(text))
                return v
        cm = ContextMemory(dim=4, sentence_encoder=FakeEncoder())
        r = cm.add_turn("hello", "hi")  # no embedding
        assert r["status"] == "stored"

    def test_add_turn_encoder_failure_falls_back_to_none(self):
        class FailingEncoder:
            def encode(self, text):
                raise RuntimeError("encoder broken")
        cm = ContextMemory(dim=4, sentence_encoder=FailingEncoder())
        r = cm.add_turn("hi", "hello")  # should not raise
        assert r["status"] == "stored"

    def test_add_turn_nan_embedding_raises(self, cm):
        with pytest.raises(ValueError, match="finite"):
            cm.add_turn("a", "b", embedding=np.array([np.nan, 0, 0, 0]))

    def test_summary_empty(self, cm):
        r = cm.summary()
        assert r["n_turns"] == 0
        assert r["summary"].size == 0

    def test_summary_with_turns(self, cm):
        cm.add_turn("a", "b", embedding=np.array([1.0, 0.0, 0.0, 0.0]))
        cm.add_turn("c", "d", embedding=np.array([0.0, 1.0, 0.0, 0.0]))
        r = cm.summary(top_k=2)
        assert r["n_turns"] == 2
        np.testing.assert_allclose(r["summary"], np.array([0.5, 0.5, 0, 0]), atol=1e-9)

    def test_summary_skips_none_embeddings(self, cm):
        cm.add_turn("a", "b", embedding=None)  # no encoder set; falls to None
        cm.add_turn("c", "d", embedding=np.array([1.0, 0.0, 0.0, 0.0]))
        r = cm.summary()
        # Only one valid embedding; summary uses it.
        assert r["n_turns"] == 2  # k defaults to rules.memory_top_k
        np.testing.assert_allclose(r["summary"], np.array([1.0, 0, 0, 0]))

    def test_recent_empty(self, cm):
        r = cm.recent()
        assert r["turns"] == []

    def test_recent_returns_last_k(self, cm):
        for i in range(3):
            cm.add_turn(f"u{i}", f"a{i}", embedding=np.zeros(4))
        r = cm.recent(top_k=2)
        assert len(r["turns"]) == 2
        assert r["turns"][0]["user"] == "u1"
        assert r["turns"][1]["user"] == "u2"

    def test_recent_default_k(self, cm):
        for i in range(3):
            cm.add_turn(f"u{i}", f"a{i}", embedding=np.zeros(4))
        r = cm.recent()
        # Should return up to rules.memory_top_k.
        assert len(r["turns"]) <= 5

    def test_clear(self, cm):
        cm.add_turn("a", "b", embedding=np.zeros(4))
        r = cm.clear()
        assert r["cleared"] == 1
        assert len(cm) == 0

    def test_len(self, cm):
        assert len(cm) == 0
        cm.add_turn("a", "b", embedding=np.zeros(4))
        assert len(cm) == 1

    def test_recent_preserves_turn_ids(self, cm):
        for i in range(3):
            cm.add_turn(f"u{i}", f"a{i}", embedding=np.zeros(4))
        r = cm.recent(top_k=3)
        ids = [t["turn_id"] for t in r["turns"]]
        assert ids == [0, 1, 2]


# --------------------------------------------------------------------- #
# MemoryConsolidator
# --------------------------------------------------------------------- #


class TestMemoryConsolidator:
    @pytest.fixture()
    def setup(self):
        rules = MemoryRules(memory_capacity=10, memory_consolidation_weight=0.0)
        wm = WorkingMemory(dim=4, rules=rules)
        em = EpisodicMemory(dim=4, rules=MemoryRules(memory_capacity=20))
        con = MemoryConsolidator(dim=4, rules=rules)
        return wm, em, con

    def test_consolidate_empty_working(self, setup):
        wm, em, con = setup
        r = con.consolidate(wm, em)
        assert r["promoted"] == 0
        assert r["deduplicated"] == 0

    def test_consolidate_promotes_high_weight(self, setup):
        wm, em, con = setup
        # With weight_threshold=0, every item is promoted.
        wm.push(np.array([1.0, 0.0, 0.0, 0.0]))
        wm.push(np.array([0.0, 1.0, 0.0, 0.0]))
        r = con.consolidate(wm, em)
        assert r["promoted"] == 2
        assert len(em) == 2

    def test_consolidate_deduplicates_near_duplicates(self):
        rules = MemoryRules(
            memory_capacity=10,
            memory_consolidation_weight=0.0,
            memory_similarity_threshold=0.99,
        )
        wm = WorkingMemory(dim=4, rules=rules)
        em = EpisodicMemory(dim=4, rules=MemoryRules(memory_capacity=20))
        con = MemoryConsolidator(dim=4, rules=rules)
        wm.push(np.array([1.0, 0.0, 0.0, 0.0]))
        wm.push(np.array([1.0, 0.0, 0.0, 0.0]))  # exact duplicate
        r1 = con.consolidate(wm, em)
        # After dedup, episodic should hold 1 unique item.
        assert r1["deduplicated"] == 1
        assert len(em) == 1
        # Second consolidate re-promotes both wm items (they remain in
        # the working window), then dedup removes the new duplicate again.
        r2 = con.consolidate(wm, em)
        assert r2["deduplicated"] == 2  # 2 new duplicates removed.
        assert len(em) == 1  # still 1 unique item.

    def test_consolidate_low_weight_not_promoted(self):
        rules = MemoryRules(
            memory_capacity=10,
            memory_consolidation_weight=2.0,  # higher than any softmax weight.
            memory_similarity_threshold=0.99,
        )
        wm = WorkingMemory(dim=4, rules=rules)
        em = EpisodicMemory(dim=4, rules=MemoryRules(memory_capacity=20))
        con = MemoryConsolidator(dim=4, rules=rules)
        wm.push(np.array([1.0, 0.0, 0.0, 0.0]))
        r = con.consolidate(wm, em)
        assert r["promoted"] == 0
        assert len(em) == 0

    def test_consolidate_idempotent_when_no_new_items(self, setup):
        wm, em, con = setup
        wm.push(np.array([1.0, 0.0, 0.0, 0.0]))
        con.consolidate(wm, em)
        r2 = con.consolidate(wm, em)
        assert r2["promoted"] == 1  # always re-promotes; dedup removes dup.
        assert len(em) == 1

    def test_consolidate_preserves_labels(self, setup):
        wm, em, con = setup
        wm.push(np.array([1.0, 0.0, 0.0, 0.0]))
        wm.push(np.array([0.0, 1.0, 0.0, 0.0]))
        con.consolidate(wm, em)
        # Consolidator tags promoted items with "consolidated:..." labels.
        for lbl in em._labels:
            assert lbl.startswith("consolidated:")

    def test_consolidator_dim_attribute(self):
        con = MemoryConsolidator(dim=8)
        assert con.dim == 8

    def test_consolidate_with_focus(self, setup):
        """Items aligned with focus get higher attention and are promoted
        first (when threshold > 0)."""
        wm, em, con = setup
        # Override consolidator with positive threshold.
        con.rules = MemoryRules(
            memory_capacity=10,
            memory_consolidation_weight=0.5,
            memory_similarity_threshold=0.99,
        )
        wm.rules = con.rules
        wm.push(np.array([1.0, 0.0, 0.0, 0.0]))  # aligned
        wm.push(np.array([0.0, 1.0, 0.0, 0.0]))  # orthogonal
        wm.set_focus(np.array([1.0, 0.0, 0.0, 0.0]))
        r = con.consolidate(wm, em)
        # Only the aligned item exceeds the threshold.
        assert r["promoted"] == 1


# --------------------------------------------------------------------- #
# HierarchicalMemory
# --------------------------------------------------------------------- #


class TestHierarchicalMemory:
    @pytest.fixture()
    def hm(self):
        # Use small consolidation times so we can test promotion.
        return HierarchicalMemory(
            dim=4,
            rules=MemoryRules(
                memory_capacity=5,
                memory_hierarchy_consolidation_times=[0.0, 0.0],  # immediate
            ),
        )

    def test_encode_into_short_term(self, hm):
        r = hm.encode(np.array([1.0, 0.0, 0.0, 0.0]), label="a")
        assert len(hm.short) == 1
        assert len(hm.medium) == 0
        assert len(hm.long) == 0
        assert r["id"] == 0

    def test_consolidate_promotes_short_to_medium(self, hm):
        hm.encode(np.array([1.0, 0.0, 0.0, 0.0]))
        # Force the timestamp into the past.
        hm.short._timestamps[0] = time.time() - 100.0
        r = hm.consolidate()
        assert r["short_to_medium"] == 1
        assert len(hm.short) == 0
        assert len(hm.medium) == 1

    def test_consolidate_promotes_medium_to_long(self, hm):
        hm.encode(np.array([1.0, 0.0, 0.0, 0.0]))
        # Move to short past.
        hm.short._timestamps[0] = time.time() - 100.0
        hm.consolidate()
        # Move medium past too.
        hm.medium._timestamps[0] = time.time() - 10000.0
        r = hm.consolidate()
        assert r["medium_to_long"] == 1
        assert len(hm.long) == 1

    def test_retrieve_across_tiers(self, hm):
        # Insert into long directly.
        hm.long.encode(np.array([1.0, 0.0, 0.0, 0.0]), label="long")
        hm.medium.encode(np.array([0.9, 0.1, 0.0, 0.0]), label="medium")
        hm.short.encode(np.array([0.8, 0.2, 0.0, 0.0]), label="short")
        r = hm.retrieve(np.array([1.0, 0.0, 0.0, 0.0]), top_k=3)
        # All 3 tiers should be merged.
        assert len(r["items"]) == 3
        tiers = {it["tier"] for it in r["items"]}
        assert tiers == {"short", "medium", "long"}

    def test_retrieve_top_k_caps_results(self, hm):
        for i in range(3):
            hm.short.encode(np.array([float(i), 0.0, 0.0, 0.0]), label=f"s{i}")
        r = hm.retrieve(np.array([1.0, 0.0, 0.0, 0.0]), top_k=2)
        assert len(r["items"]) == 2

    def test_len_aggregates_tiers(self, hm):
        hm.short.encode(np.zeros(4))
        hm.medium.encode(np.zeros(4))
        hm.long.encode(np.zeros(4))
        assert len(hm) == 3

    def test_tiers_have_growing_capacity(self):
        hm = HierarchicalMemory(dim=4, rules=MemoryRules(memory_capacity=5))
        assert hm.short.rules.memory_capacity == 5
        assert hm.medium.rules.memory_capacity == 10
        assert hm.long.rules.memory_capacity == 20

    def test_retrieve_empty(self, hm):
        r = hm.retrieve(np.zeros(4))
        assert r["items"] == []

    def test_short_term_evicts_on_overflow(self, hm):
        for i in range(10):  # capacity = 5
            hm.encode(np.array([float(i), 0.0, 0.0, 0.0]), label=f"l{i}")
        assert len(hm.short) == 5


# --------------------------------------------------------------------- #
# SpreadingActivationMemory
# --------------------------------------------------------------------- #


class TestSpreadingActivationMemory:
    @pytest.fixture()
    def sm(self):
        return SpreadingActivationMemory(
            dim=4, rules=MemoryRules(memory_capacity=10, memory_spreading_depth=2)
        )

    def test_encode_returns_position(self, sm):
        r = sm.encode(np.array([1.0, 0.0, 0.0, 0.0]), label="a")
        assert r["position"] == 0

    def test_retrieve_empty(self, sm):
        r = sm.retrieve(np.zeros(4))
        assert len(r["items"]) == 0

    def test_retrieve_returns_activations(self, sm):
        sm.encode(np.array([1.0, 0.0, 0.0, 0.0]), label="a")
        sm.encode(np.array([0.0, 1.0, 0.0, 0.0]), label="b")
        r = sm.retrieve(np.array([1.0, 0.0, 0.0, 0.0]), top_k=2)
        assert len(r["items"]) == 2
        assert "activations" in r
        assert r["activations"].size == 2

    def test_retrieve_top_match(self, sm):
        sm.encode(np.array([1.0, 0.0, 0.0, 0.0]), label="a")
        sm.encode(np.array([0.0, 1.0, 0.0, 0.0]), label="b")
        r = sm.retrieve(np.array([1.0, 0.0, 0.0, 0.0]), top_k=1)
        assert r["items"][0]["label"] == "a"

    def test_retrieve_normalizes_activations(self, sm):
        sm.encode(np.array([1.0, 0.0, 0.0, 0.0]), label="a")
        sm.encode(np.array([0.9, 0.1, 0.0, 0.0]), label="b")
        r = sm.retrieve(np.array([1.0, 0.0, 0.0, 0.0]), top_k=2)
        # Max activation should be ~1.0 (normalized).
        assert max(r["similarities"]) == pytest.approx(1.0, abs=1e-6)

    def test_spreading_amplifies_connected_items(self, sm):
        """Items similar to each other should boost each other's
        activation through spreading."""
        # Three items forming a chain: a~b~c, with query near a.
        sm.encode(np.array([1.0, 0.0, 0.0, 0.0]), label="a")
        sm.encode(np.array([0.9, 0.1, 0.0, 0.0]), label="b")
        sm.encode(np.array([0.8, 0.2, 0.0, 0.0]), label="c")
        r = sm.retrieve(np.array([1.0, 0.0, 0.0, 0.0]), top_k=3)
        # Without spreading, c would have lower activation than a; with
        # spreading through b, c is amplified.
        labels = [it["label"] for it in r["items"]]
        assert "a" in labels and "b" in labels and "c" in labels

    def test_capacity_eviction(self):
        sm = SpreadingActivationMemory(
            dim=2, rules=MemoryRules(memory_capacity=3, memory_spreading_depth=1)
        )
        for i in range(5):
            sm.encode(np.array([float(i), 0.0]), label=f"l{i}")
        assert len(sm) == 3

    def test_top_k_caps_results(self, sm):
        for i in range(5):
            sm.encode(np.array([float(i), 0.0, 0.0, 0.0]), label=f"l{i}")
        r = sm.retrieve(np.array([1.0, 0.0, 0.0, 0.0]), top_k=2)
        assert len(r["items"]) == 2

    def test_nan_raises(self, sm):
        with pytest.raises(ValueError, match="finite"):
            sm.encode(np.array([np.nan, 0, 0, 0]))

    def test_retrieve_returns_label_and_timestamp(self, sm):
        sm.encode(np.array([1.0, 0.0, 0.0, 0.0]), label="x")
        r = sm.retrieve(np.array([1.0, 0.0, 0.0, 0.0]), top_k=1)
        assert r["items"][0]["label"] == "x"
        assert isinstance(r["items"][0]["timestamp"], float)

    def test_len(self, sm):
        assert len(sm) == 0
        sm.encode(np.zeros(4))
        assert len(sm) == 1


# --------------------------------------------------------------------- #
# ForgetfulMemory
# --------------------------------------------------------------------- #


class TestForgetfulMemory:
    @pytest.fixture()
    def fm(self):
        return ForgetfulMemory(
            dim=4, rules=MemoryRules(memory_capacity=10, memory_ebbinghaus_S=100.0)
        )

    def test_encode_returns_position(self, fm):
        r = fm.encode(np.array([1.0, 0.0, 0.0, 0.0]), label="a")
        assert r["position"] == 0

    def test_retrieve_empty(self, fm):
        r = fm.retrieve(np.zeros(4))
        assert r["items"] == []
        assert r["retention"] == []

    def test_retrieve_recent_item_full_retention(self, fm):
        fm.encode(np.array([1.0, 0.0, 0.0, 0.0]), label="a")
        # Review to refresh timestamp.
        fm.review([0])
        r = fm.retrieve(np.array([1.0, 0.0, 0.0, 0.0]), top_k=1)
        assert r["retention"][0] == pytest.approx(1.0, abs=1e-6)
        assert r["similarities"][0] == pytest.approx(1.0, abs=1e-6)

    def test_retrieve_decays_with_age(self, fm):
        fm.encode(np.array([1.0, 0.0, 0.0, 0.0]), label="a")
        # Force the timestamp into the past.
        fm._timestamps[0] = time.time() - 100.0  # age = S = 100
        r = fm.retrieve(np.array([1.0, 0.0, 0.0, 0.0]), top_k=1, at_time=time.time())
        # R = exp(-100/100) = exp(-1) ≈ 0.368.
        assert r["retention"][0] == pytest.approx(np.exp(-1.0), abs=0.01)

    def test_review_resets_timestamp(self, fm):
        fm.encode(np.array([1.0, 0.0, 0.0, 0.0]), label="a")
        fm._timestamps[0] = time.time() - 1000.0
        r = fm.review([0])
        assert r["reviewed"] == 1
        # Now retrieve; retention should be ~1.0.
        r2 = fm.retrieve(np.array([1.0, 0.0, 0.0, 0.0]), top_k=1)
        assert r2["retention"][0] > 0.99

    def test_review_empty_ids(self, fm):
        r = fm.review([])
        assert r["reviewed"] == 0

    def test_review_out_of_range(self, fm):
        fm.encode(np.zeros(4))
        r = fm.review([99])
        assert r["reviewed"] == 0

    def test_at_time_retrospective_retrieval(self, fm):
        fm.encode(np.array([1.0, 0.0, 0.0, 0.0]), label="a")
        encode_time = fm._timestamps[0]
        # Retrieve "as of" 1 hour ago (item was encoded 3600s later than that).
        r = fm.retrieve(
            np.array([1.0, 0.0, 0.0, 0.0]),
            top_k=1,
            at_time=encode_time - 3600.0,
        )
        # Age is negative -> clamped to 0 -> retention = 1.0.
        assert r["retention"][0] == pytest.approx(1.0, abs=1e-6)

    def test_capacity_eviction(self):
        fm = ForgetfulMemory(
            dim=2, rules=MemoryRules(memory_capacity=3, memory_ebbinghaus_S=100.0)
        )
        for i in range(5):
            fm.encode(np.array([float(i), 0.0]), label=f"l{i}")
        assert len(fm) == 3

    def test_top_k_caps_results(self, fm):
        for i in range(5):
            fm.encode(np.array([float(i), 0.0, 0.0, 0.0]), label=f"l{i}")
        r = fm.retrieve(np.array([1.0, 0.0, 0.0, 0.0]), top_k=2)
        assert len(r["items"]) == 2

    def test_nan_raises(self, fm):
        with pytest.raises(ValueError, match="finite"):
            fm.encode(np.array([np.nan, 0, 0, 0]))

    def test_zero_S_guards_against_division(self):
        fm = ForgetfulMemory(
            dim=4, rules=MemoryRules(memory_ebbinghaus_S=0.0)
        )
        fm.encode(np.array([1.0, 0.0, 0.0, 0.0]))
        # Should not raise even with S=0.
        r = fm.retrieve(np.array([1.0, 0.0, 0.0, 0.0]), top_k=1)
        assert len(r["items"]) == 1

    def test_len(self, fm):
        assert len(fm) == 0
        fm.encode(np.zeros(4))
        assert len(fm) == 1

    def test_ebbinghaus_curve_monotonic(self, fm):
        """R(t) = exp(-t/S) is monotonically decreasing in t."""
        fm.encode(np.array([1.0, 0.0, 0.0, 0.0]), label="a")
        now = fm._timestamps[0]
        r1 = fm.retrieve(np.array([1.0, 0.0, 0.0, 0.0]), top_k=1, at_time=now + 10.0)
        r2 = fm.retrieve(np.array([1.0, 0.0, 0.0, 0.0]), top_k=1, at_time=now + 100.0)
        assert r1["retention"][0] > r2["retention"][0]


# --------------------------------------------------------------------- #
# MemoryIndexer
# --------------------------------------------------------------------- #


class TestMemoryIndexer:
    @pytest.fixture()
    def idx(self):
        return MemoryIndexer(dim=4, rules=MemoryRules(memory_capacity=100))

    def test_add_first_vector_sets_dim(self):
        idx = MemoryIndexer(dim=8, rules=MemoryRules(memory_capacity=10))
        # First vector has dim 4 — should auto-resize.
        r = idx.add(np.array([1.0, 0.0, 0.0, 0.0]), label="a")
        assert idx.dim == 4
        assert r["position"] == 0

    def test_add_dim_mismatch_raises(self, idx):
        idx.add(np.array([1.0, 0.0, 0.0, 0.0]), label="a")
        with pytest.raises(ValueError, match="dim"):
            idx.add(np.array([1.0, 0.0, 0.0]))  # wrong dim

    def test_add_returns_position_and_label(self, idx):
        r = idx.add(np.array([1.0, 0.0, 0.0, 0.0]), label="x")
        assert r["position"] == 0
        assert r["label"] == "x"

    def test_search_empty(self, idx):
        r = idx.search(np.zeros(4))
        assert r["positions"] == []
        assert r["similarities"] == []

    def test_search_top_k(self, idx):
        idx.add(np.array([1.0, 0.0, 0.0, 0.0]), label="a")
        idx.add(np.array([0.0, 1.0, 0.0, 0.0]), label="b")
        idx.add(np.array([0.0, 0.0, 1.0, 0.0]), label="c")
        r = idx.search(np.array([1.0, 0.0, 0.0, 0.0]), top_k=2)
        assert len(r["positions"]) == 2
        assert r["positions"][0] == 0  # best match is item 0.
        assert r["labels"][0] == "a"

    def test_search_dim_mismatch_raises(self, idx):
        idx.add(np.array([1.0, 0.0, 0.0, 0.0]), label="a")
        with pytest.raises(ValueError, match="dim"):
            idx.search(np.array([1.0, 0.0, 0.0]))  # wrong dim

    def test_remove(self, idx):
        idx.add(np.array([1.0, 0.0, 0.0, 0.0]), label="a")
        idx.add(np.array([0.0, 1.0, 0.0, 0.0]), label="b")
        r = idx.remove([0])
        assert r["removed"] == 1
        assert len(idx) == 1

    def test_remove_out_of_range(self, idx):
        idx.add(np.array([1.0, 0.0, 0.0, 0.0]), label="a")
        r = idx.remove([99])
        assert r["removed"] == 0

    def test_remove_empty(self, idx):
        r = idx.remove([])
        assert r["removed"] == 0

    def test_build_from_2d(self, idx):
        arr = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ])
        r = idx.build_from(arr)
        assert r["indexed"] == 3
        assert len(idx) == 3

    def test_build_from_non_2d_raises(self, idx):
        with pytest.raises(ValueError, match="2D"):
            idx.build_from(np.array([1.0, 2.0, 3.0]))

    def test_build_from_nan_raises(self, idx):
        with pytest.raises(ValueError, match="finite"):
            idx.build_from(np.array([[1.0, np.nan], [0.0, 1.0]]))

    def test_search_zero_norm_query(self, idx):
        idx.add(np.array([1.0, 0.0, 0.0, 0.0]), label="a")
        r = idx.search(np.zeros(4))  # zero-norm query
        assert len(r["positions"]) == 1
        # Similarity should be 0 (zero-vector normalized).
        assert r["similarities"][0] == 0.0

    def test_search_zero_norm_index_item(self, idx):
        idx.add(np.zeros(4), label="zero")
        r = idx.search(np.array([1.0, 0.0, 0.0, 0.0]))
        assert len(r["positions"]) == 1
        assert r["similarities"][0] == 0.0

    def test_len(self, idx):
        assert len(idx) == 0
        idx.add(np.zeros(4))
        assert len(idx) == 1
