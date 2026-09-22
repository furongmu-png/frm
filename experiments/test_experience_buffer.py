# experiments/test_experience_buffer.py
"""Verification tests for the ExperienceBuffer + OfflineConsolidator."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experience_buffer import (
    ExperienceBuffer,
    OfflineConsolidator,
)

from zero_data_model.active_inference import ActiveInferenceEngine


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #
@pytest.fixture
def buffer():
    return ExperienceBuffer(capacity=100, seed=42)


@pytest.fixture
def filled_buffer(buffer):
    """A buffer pre-filled with 50 experiences of varying surprise."""
    rng = np.random.default_rng(0)
    for i in range(50):
        obs = rng.standard_normal(8) * 0.1
        next_obs = obs + rng.standard_normal(8) * 0.05
        # Vary the error: most are small, a few are large (surprising).
        error = abs(rng.standard_normal()) * (10.0 if i % 10 == 0 else 0.1)
        buffer.add(
            obs=obs, action=i % 4, next_obs=next_obs,
            error=error, info_gain=float(rng.random()), step=i,
        )
    return buffer


@pytest.fixture
def engine():
    return ActiveInferenceEngine(
        state_dim=8, obs_dim=8, action_dim=4,
        rng=np.random.default_rng(42),
    )


class _MockModel:
    """Minimal model stub exposing ``active_inference``."""
    def __init__(self, engine):
        self.active_inference = engine


@pytest.fixture
def model(engine):
    return _MockModel(engine)


# ------------------------------------------------------------------ #
# ExperienceBuffer tests
# ------------------------------------------------------------------ #
class TestExperienceBuffer:
    def test_add_and_len(self, buffer):
        assert len(buffer) == 0
        buffer.add(np.zeros(8), 0, np.zeros(8), 0.5, 0.1, step=0)
        assert len(buffer) == 1
        for i in range(10):
            buffer.add(np.zeros(8), 0, np.zeros(8), 0.5, 0.1, step=i)
        assert len(buffer) == 11

    def test_capacity_bounded(self, buffer):
        """Buffer must respect its capacity (ring-buffer eviction)."""
        for i in range(150):  # capacity=100
            buffer.add(np.zeros(8), 0, np.zeros(8), 0.5, 0.1, step=i)
        assert len(buffer) == 100  # evicted the oldest 50

    def test_sample_empty_returns_empty(self, buffer):
        assert buffer.sample(10) == []

    def test_sample_uniform(self, filled_buffer):
        batch = filled_buffer.sample(10, mode="uniform")
        assert len(batch) == 10
        # All distinct (no replacement).
        assert len({id(e) for e in batch}) == 10

    def test_sample_priority(self, filled_buffer):
        batch = filled_buffer.sample(10, mode="priority")
        assert len(batch) == 10
        # Priority sampling should include high-error experiences more
        # often. The buffer has 5 high-error experiences (every 10th).
        # Run multiple times and verify high-error experiences appear
        # disproportionately often.
        counts = dict.fromkeys(range(50), 0)
        for _ in range(50):
            for e in filled_buffer.sample(20, mode="priority"):
                # Match by step (unique id).
                counts[e.step] += 1
        # The 5 high-error experiences (step 0, 10, 20, 30, 40) should
        # have higher counts than the average low-error experience.
        high_err_counts = [counts[i] for i in range(0, 50, 10)]
        low_err_counts = [counts[i] for i in range(50) if i % 10 != 0]
        high_mean = float(np.mean(high_err_counts))
        low_mean = float(np.mean(low_err_counts))
        assert high_mean > low_mean, (
            f"priority sampling should favour high-error: "
            f"high={high_mean}, low={low_mean}"
        )

    def test_sample_recent(self, filled_buffer):
        batch = filled_buffer.sample(10, mode="recent")
        assert len(batch) == 10
        # Recent sampling should bias toward high step indices.
        steps = [e.step for e in batch]
        # Mean sampled step should be > middle (25).
        mean_step = float(np.mean(steps))
        assert mean_step > 25.0, (
            f"recent sampling should favour later steps, got mean={mean_step}"
        )

    def test_sample_batch_larger_than_buffer(self, filled_buffer):
        """If batch_size > buffer size, return all available."""
        batch = filled_buffer.sample(1000)
        assert len(batch) == 50  # buffer has 50

    def test_invalid_mode_raises(self, filled_buffer):
        with pytest.raises(ValueError, match="unknown mode"):
            filled_buffer.sample(5, mode="invalid")

    def test_priority_computation(self):
        """High-error experiences should have higher priority."""
        buf = ExperienceBuffer(capacity=10, surprise_weighted=True, seed=0)
        buf.add(np.zeros(4), 0, np.zeros(4), error=0.01, info_gain=0.0, step=0)
        buf.add(np.zeros(4), 0, np.zeros(4), error=5.0, info_gain=0.0, step=1)
        priorities = [e.priority for e in buf._buffer]
        assert priorities[1] > priorities[0], (
            f"high-error should have higher priority: {priorities}"
        )

    def test_surprise_weighted_disabled(self):
        """When surprise_weighted=False, all priorities should be 1.0."""
        buf = ExperienceBuffer(capacity=10, surprise_weighted=False, seed=0)
        buf.add(np.zeros(4), 0, np.zeros(4), error=10.0, info_gain=0.0, step=0)
        buf.add(np.zeros(4), 0, np.zeros(4), error=0.01, info_gain=0.0, step=1)
        assert all(e.priority == 1.0 for e in buf._buffer)

    def test_arrays_copied(self, buffer):
        """Buffer must copy arrays — caller mutations must not leak."""
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        buffer.add(obs, 0, np.zeros(4), 0.5, 0.1, step=0)
        obs[0] = 999.0  # mutate caller's array
        assert buffer._buffer[0].obs[0] == 1.0, "buffer should not alias caller arrays"

    def test_clear(self, filled_buffer):
        n = filled_buffer.clear()
        assert n == 50
        assert len(filled_buffer) == 0

    def test_deterministic_sampling(self):
        """Same seed → same sample order."""
        # Pre-generate a fixed list of experiences so both buffers
        # receive IDENTICAL data in the SAME order (the previous
        # version drew random errors/igs from two separate rng calls,
        # which produced different priorities for the two buffers).
        rng = np.random.default_rng(0)
        experiences = []
        for i in range(50):
            obs = rng.standard_normal(4)
            err = float(rng.random())
            ig = float(rng.random())
            experiences.append((obs, i % 4, obs.copy(), err, ig, i))
        buf1 = ExperienceBuffer(capacity=50, seed=42)
        buf2 = ExperienceBuffer(capacity=50, seed=42)
        for obs, a, next_obs, err, ig, step in experiences:
            buf1.add(obs, a, next_obs, err, ig, step=step)
            buf2.add(obs.copy(), a, next_obs.copy(), err, ig, step=step)
        # Same seed + same data → same sample (deterministic).
        s1 = [e.step for e in buf1.sample(10, mode="priority")]
        s2 = [e.step for e in buf2.sample(10, mode="priority")]
        assert s1 == s2, f"same seed should give same sample: {s1} vs {s2}"


# ------------------------------------------------------------------ #
# OfflineConsolidator tests
# ------------------------------------------------------------------ #
class TestOfflineConsolidator:
    def test_consolidate_empty_buffer_returns_zero(self, model):
        """Consolidating an empty buffer should be a no-op."""
        buf = ExperienceBuffer(capacity=10)
        cons = OfflineConsolidator(batch_size=5)
        result = cons.consolidate(model, buf)
        assert result.n_replayed == 0
        assert result.delta_free_energy == 0.0

    def test_consolidate_replays_experiences(self, model, filled_buffer):
        """Consolidation should replay exactly batch_size experiences."""
        cons = OfflineConsolidator(batch_size=10, seed=0)
        result = cons.consolidate(model, filled_buffer)
        assert result.n_replayed == 10
        assert result.mean_replay_error > 0.0
        assert result.elapsed_s > 0.0

    def test_consolidate_reduces_free_energy(self, model, filled_buffer):
        """After replay, the model should predict the batch better.

        This is the core hypothesis: replaying high-surprise experiences
        strengthens the generative model's predictions for them, so
        the average free energy on the SAME batch should decrease.
        We use a larger batch and multiple rounds to make the effect
        measurable above the stochastic noise floor.
        """
        cons = OfflineConsolidator(batch_size=32, seed=0)
        # Run multiple rounds to make the effect measurable.
        results = cons.consolidate_many(model, filled_buffer, n_rounds=5)
        # At least one round should show improvement (delta_fe < 0).
        # We can't guarantee EVERY round improves (stochastic), but
        # the LAST round should show improvement after 5 rounds of
        # consolidation on the same data.
        last = results[-1]
        assert last.delta_free_energy < 0.0, (
            f"consolidation should reduce FE: delta={last.delta_free_energy}"
        )

    def test_consolidate_log_appended(self, model, filled_buffer):
        """Each consolidate() call should append to the log."""
        cons = OfflineConsolidator(batch_size=5, seed=0)
        assert cons.log.n_rounds == 0
        cons.consolidate(model, filled_buffer)
        assert cons.log.n_rounds == 1
        cons.consolidate(model, filled_buffer)
        assert cons.log.n_rounds == 2
        summary = cons.log.summary()
        assert summary["n_rounds"] == 2
        assert summary["total_replayed"] == 10

    def test_consolidate_many(self, model, filled_buffer):
        """consolidate_many should run n_rounds rounds."""
        cons = OfflineConsolidator(batch_size=5, seed=0)
        results = cons.consolidate_many(model, filled_buffer, n_rounds=3)
        assert len(results) == 3
        assert all(r.n_replayed == 5 for r in results)

    def test_invalid_batch_size(self):
        with pytest.raises(ValueError, match="batch_size"):
            OfflineConsolidator(batch_size=0)

    def test_invalid_sampling_mode(self):
        with pytest.raises(ValueError, match="sampling_mode"):
            OfflineConsolidator(sampling_mode="invalid")


# ------------------------------------------------------------------ #
# Integration test
# ------------------------------------------------------------------ #
class TestIntegration:
    def test_full_cycle_buffer_then_consolidate(self, model):
        """End-to-end: add experiences online, then consolidate offline."""
        buf = ExperienceBuffer(capacity=100, seed=42)
        rng = np.random.default_rng(0)
        # Online phase: simulate 20 steps of interaction.
        for i in range(20):
            obs = rng.standard_normal(8) * 0.1
            action = i % 4
            next_obs = obs + rng.standard_normal(8) * 0.05
            error = float(abs(rng.standard_normal()))
            ig = float(rng.random())
            buf.add(obs, action, next_obs, error, ig, step=i)
        # Offline phase: consolidate 2 rounds.
        cons = OfflineConsolidator(batch_size=8, seed=0)
        results = cons.consolidate_many(model, buf, n_rounds=2)
        assert len(results) == 2
        assert all(r.n_replayed == 8 for r in results)
        assert cons.log.summary()["total_replayed"] == 16


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
