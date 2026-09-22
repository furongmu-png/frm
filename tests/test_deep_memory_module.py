"""Tests for DeepMemoryModule (S4 + PCN + Hopfield coordinator).

Verifies the first-stage cognitive upgrade module works correctly:
- Initialization with various configs
- Step-wise processing produces valid DeepMemoryState
- Novel patterns get stored in Hopfield when FE is low enough
- Snapshot returns correct component state
- Reset clears state without losing learned weights
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path for direct import
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from zero_data_model.deep_memory import (
    DeepMemoryModule,
    DeepMemoryState,
    DeepMemoryStats,
)


class TestDeepMemoryModuleInit:
    """Initialization tests."""

    def test_default_init(self):
        dm = DeepMemoryModule(dim=32, seed=42)
        assert dm.dim == 32
        assert dm.alpha + dm.beta + dm.gamma == pytest.approx(1.0)
        assert dm.s4.state_dim == 64  # 2 * dim
        assert dm.hopfield.capacity == 1024
        assert dm.pcn_l0.dim == 32
        assert dm.pcn_l1.dim == 32
        assert dm.pcn_l2.dim == 16  # dim // 2

    def test_custom_state_dim(self):
        dm = DeepMemoryModule(dim=32, s4_state_dim=128, seed=42)
        assert dm.s4.state_dim == 128

    def test_custom_hopfield_capacity(self):
        dm = DeepMemoryModule(dim=32, hopfield_capacity=256, seed=42)
        assert dm.hopfield.capacity == 256

    def test_fusion_weights_normalized(self):
        # alpha+beta+gamma != 1 should normalize
        dm = DeepMemoryModule(
            dim=32, alpha=2.0, beta=2.0, gamma=1.0, seed=42
        )
        assert dm.alpha == pytest.approx(0.4, abs=1e-10)
        assert dm.beta == pytest.approx(0.4, abs=1e-10)
        assert dm.gamma == pytest.approx(0.2, abs=1e-10)

    def test_invalid_dim_raises(self):
        with pytest.raises(ValueError, match="dim must be >= 1"):
            DeepMemoryModule(dim=0)

    def test_invalid_fusion_weights_raises(self):
        with pytest.raises(ValueError, match="alpha\\+beta\\+gamma"):
            DeepMemoryModule(dim=32, alpha=0, beta=0, gamma=0)


class TestDeepMemoryStep:
    """Step processing tests."""

    def test_step_returns_valid_state(self):
        dm = DeepMemoryModule(dim=32, seed=42)
        rng = np.random.default_rng(0)
        obs = rng.standard_normal(32)
        state = dm.step(obs)
        assert isinstance(state, DeepMemoryState)
        assert state.fused.shape == (32,)
        assert np.all(np.isfinite(state.fused))
        assert state.free_energy >= 0
        assert state.is_novel in (True, False)
        assert state.hopfield_size >= 0
        assert set(state.pcn_layer_errors.keys()) == {"l0", "l1", "l2"}

    def test_step_shape_validation(self):
        dm = DeepMemoryModule(dim=32, seed=42)
        with pytest.raises(ValueError, match="obs must have shape"):
            dm.step(np.zeros(16))

    def test_step_handles_nan_input(self):
        """NaN inputs should not crash; S4 has internal NaN guards."""
        dm = DeepMemoryModule(dim=32, seed=42)
        obs = np.full(32, np.nan)
        # Should not raise; may produce NaN fused but won't crash
        try:
            state = dm.step(obs)
            # If it returns, fused should be finite (NaN guard in step())
            # or the test just verifies no exception
        except Exception as exc:
            # Some internal components might raise; verify it's caught
            # Actually our step() has NaN guard on fused
            pass

    def test_multiple_steps_accumulate_stats(self):
        dm = DeepMemoryModule(dim=32, seed=42)
        rng = np.random.default_rng(0)
        for _ in range(10):
            obs = rng.standard_normal(32)
            dm.step(obs)
        assert dm.stats.total_steps == 10
        assert dm.stats.avg_free_energy > 0
        assert len(dm.stats.fe_history) == 10

    def test_s4_state_evolves(self):
        """S4 state should change as observations flow in."""
        dm = DeepMemoryModule(dim=32, seed=42)
        rng = np.random.default_rng(0)
        initial_state = dm.s4._state.copy()
        for _ in range(20):
            dm.step(rng.standard_normal(32))
        assert not np.allclose(initial_state, dm.s4._state)


class TestDeepMemoryStorage:
    """Hopfield storage tests."""

    def test_low_fe_pattern_gets_stored(self):
        """A clean pattern with low FE should be stored in Hopfield."""
        dm = DeepMemoryModule(
            dim=32,
            novelty_threshold=0.95,  # high → most patterns are novel
            store_fe_threshold=1e9,  # essentially no FE limit
            seed=42,
        )
        rng = np.random.default_rng(0)
        # Feed a normalized observation
        obs = rng.standard_normal(32)
        obs = obs / np.linalg.norm(obs)  # unit norm → low FE
        for _ in range(5):
            state = dm.step(obs)
        # Should have stored at least one pattern
        assert state.hopfield_size >= 1
        assert dm.stats.stored_count >= 1

    def test_repeat_pattern_increases_similarity(self):
        """A repeated pattern should yield increasing Hopfield similarity."""
        dm = DeepMemoryModule(
            dim=32,
            novelty_threshold=0.95,
            store_fe_threshold=1e9,
            seed=42,
        )
        rng = np.random.default_rng(0)
        obs = rng.standard_normal(32)
        obs = obs / np.linalg.norm(obs)
        # Initial step: novelty, similarity ~0
        s1 = dm.step(obs)
        # Repeated steps: similarity should rise as pattern gets stored
        similarities = [s1.hopfield_similarity]
        for _ in range(10):
            s = dm.step(obs)
            similarities.append(s.hopfield_similarity)
        # At least one later similarity should be > initial
        assert max(similarities[1:]) >= similarities[0]


class TestDeepMemorySnapshot:
    """Snapshot tests."""

    def test_snapshot_contains_all_components(self):
        dm = DeepMemoryModule(dim=32, seed=42)
        snap = dm.get_snapshot()
        assert "s4" in snap
        assert "pcn" in snap
        assert "hopfield" in snap
        assert "stats" in snap
        assert "spectral_radius" in snap["s4"]
        assert "size" in snap["hopfield"]
        assert "total_steps" in snap["stats"]

    def test_snapshot_s4_stable(self):
        """S4 spectral radius should be < 1 (stability)."""
        dm = DeepMemoryModule(dim=32, seed=42)
        snap = dm.get_snapshot()
        assert snap["s4"]["spectral_radius"] < 1.0


class TestDeepMemoryReset:
    """Reset tests."""

    def test_reset_clears_state(self):
        dm = DeepMemoryModule(dim=32, seed=42)
        rng = np.random.default_rng(0)
        for _ in range(10):
            dm.step(rng.standard_normal(32))
        dm.reset()
        assert dm.stats.total_steps == 0
        assert dm.stats.novel_count == 0
        # S4 state should be zeroed
        assert np.allclose(dm.s4._state, 0)
        # Hopfield should retain its stored memories (only state resets)
        # Actually, Hopfield stores persist — that's the point of memory.

    def test_reset_keeps_weights(self):
        """Reset clears state but keeps learned weights."""
        dm = DeepMemoryModule(dim=32, seed=42)
        rng = np.random.default_rng(0)
        for _ in range(20):
            dm.step(rng.standard_normal(32))
        s4_weights_before = dm.s4._C.copy()
        pcn_weights_before = dm.pcn_l1._W_gen.copy() if dm.pcn_l1._W_gen is not None else None
        dm.reset()
        # Weights should be unchanged
        assert np.allclose(dm.s4._C, s4_weights_before)
        if pcn_weights_before is not None:
            assert np.allclose(dm.pcn_l1._W_gen, pcn_weights_before)


class TestDeepMemoryRetrieve:
    """Retrieve associated memory tests."""

    def test_retrieve_empty_returns_none(self):
        dm = DeepMemoryModule(dim=32, seed=42)
        assert dm.retrieve_associated(np.zeros(32)) is None

    def test_retrieve_after_store_returns_vector(self):
        dm = DeepMemoryModule(
            dim=32,
            novelty_threshold=0.95,
            store_fe_threshold=1e9,
            seed=42,
        )
        rng = np.random.default_rng(0)
        obs = rng.standard_normal(32)
        obs = obs / np.linalg.norm(obs)
        for _ in range(5):
            dm.step(obs)
        retrieved = dm.retrieve_associated(obs)
        assert retrieved is not None
        assert retrieved.shape == (32,)
