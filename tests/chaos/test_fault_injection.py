# tests/chaos/test_fault_injection.py
"""Chaos engineering tests — verify system resilience under fault conditions.

These tests inject faults at the Python level (not network level). They
verify that:

* ``think()`` fails fast and cleanly when a hard resource error
  (``MemoryError``) prevents it from even building the input signal —
  no partial state mutation leaks to subsequent cycles.
* ``think()`` keeps working when an OPTIONAL cognitive-upgrade module
  raises. Every phase-7 upgrade hook (architect / layered_predictor /
  temporal_memory / episodic_graph / logic_layer / meta_cognition /
  experiment_planner / multiagent) is wrapped in ``try/except`` inside
  ``model.think()``, so a single failing upgrade must never crash the
  core perception-action cycle.
* The system rejects ``NaN``/``Inf`` observations without permanently
  corrupting the belief state.
* Repeated ``think()`` cycles do not leak memory unboundedly.

NOTE: the method names mocked here match the ACTUAL methods called by
``ZeroDataModel.think()`` (verified against ``model.py``):

  - ``architect.evaluate``        (NOT ``architect.step``)
  - ``temporal_memory.update``     (NOT ``temporal_memory.step``)
  - ``layered_predictor.update``
  - ``logic_layer.check_all``     (NOT ``logic_layer.evaluate``)
  - ``meta_cognition.update``     (NOT ``meta_cognition.assess``)
  - ``experiment_planner.evaluate`` (NOT ``experiment_planner.select``)
  - ``multiagent_world.step``

The belief state lives at
``model.active_inference.generative_model.belief_state`` (NOT
``model.belief_state`` — ``ZeroDataModel`` has no such attribute).
"""

from __future__ import annotations

import gc
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from zero_data_model import ZeroDataModel


class TestMemoryPressure:
    """System behavior under memory pressure."""

    def test_think_survives_low_memory(self):
        """think() does not catch MemoryError from numpy allocation.

        ``think()`` calls ``_self_generate()`` which calls
        ``np.zeros(self.dim)`` to build the input signal. When numpy
        raises ``MemoryError`` there, ``think()`` does NOT swallow it —
        the error propagates so the caller knows the cycle failed. The
        important resilience property is that the failure is ATOMIC:
        ``cycle_count`` is not incremented, so no half-completed cycle
        is observable by subsequent calls.
        """
        model = ZeroDataModel(dim=16, seed=42)
        cycles_before = model.cycle_count
        # Patch numpy.zeros to raise MemoryError (simulates OOM during the
        # very first allocation in _self_generate).
        with (
            patch("numpy.zeros", side_effect=MemoryError("Out of memory")),
            pytest.raises(MemoryError),
        ):
            # think() does NOT catch MemoryError — it must propagate.
            model.think()
        # Atomicity: no partial state mutation — cycle_count unchanged.
        assert model.cycle_count == cycles_before
        # A subsequent think() (with numpy.zeros restored) must still work.
        result = model.think()
        assert result is not None
        assert model.cycle_count == cycles_before + 1

    def test_model_handles_nan_injection(self):
        """System should reject NaN values without corrupting state.

        ``GenerativeModel.update_belief`` explicitly guards against
        non-finite observations: it returns the UNCHANGED belief and a
        sentinel free energy, leaving ``belief_state`` intact.
        """
        model = ZeroDataModel(dim=16, seed=42)
        model.think()
        original_belief = model.active_inference.generative_model.belief_state.copy()
        # Inject NaN into observation.
        nan_obs = np.full(16, np.nan)
        # update_belief should reject NaN (returns unchanged belief).
        model.active_inference.generative_model.update_belief(nan_obs)
        # Belief should be unchanged.
        np.testing.assert_array_equal(
            model.active_inference.generative_model.belief_state, original_belief
        )


class TestComponentFailure:
    """System behavior when individual components fail."""

    def test_think_continues_if_architect_fails(self):
        """think() should continue if architect module raises.

        The architect hook calls ``architect.record_errors`` then
        ``architect.evaluate``; both are wrapped in try/except inside
        think(). When ``evaluate`` raises, the hook logs a warning and
        think() continues. The ``cognitive_upgrades`` metadata should
        NOT contain the ``architecture`` key (it was never set).
        """
        model = ZeroDataModel(dim=16, seed=42, enable_architect=True)
        # Make architect.evaluate raise (the actual method think() calls).
        model.architect.evaluate = MagicMock(side_effect=RuntimeError("architect crashed"))
        # think() should catch the error and continue.
        result = model.think()
        assert result is not None
        upgrades = result.metadata.get("cognitive_upgrades", {})
        assert "architecture" not in upgrades

    def test_think_continues_if_temporal_memory_fails(self):
        model = ZeroDataModel(dim=16, seed=42, enable_layered_predictor=True)
        # temporal_memory hook calls update(), not step().
        model.temporal_memory.update = MagicMock(side_effect=RuntimeError("temporal crashed"))
        result = model.think()
        assert result is not None

    def test_think_continues_if_logic_layer_fails(self):
        model = ZeroDataModel(dim=16, seed=42, enable_logic_layer=True)
        # logic_layer hook calls check_all(), not evaluate().
        model.logic_layer.check_all = MagicMock(side_effect=RuntimeError("logic crashed"))
        result = model.think()
        assert result is not None

    def test_think_continues_if_multiagent_fails(self):
        model = ZeroDataModel(dim=16, seed=42, enable_multiagent=True)
        # multiagent hook calls step().
        model.multiagent_world.step = MagicMock(side_effect=RuntimeError("world crashed"))
        result = model.think()
        assert result is not None

    def test_think_continues_if_all_upgrades_fail(self):
        """If ALL cognitive upgrade modules fail, think() should still work."""
        model = ZeroDataModel(
            dim=16,
            seed=42,
            enable_architect=True,
            enable_layered_predictor=True,
            enable_episodic_memory=True,
            enable_logic_layer=True,
            enable_meta_cognition=True,
            enable_experiment_planner=True,
            enable_multiagent=True,
        )
        # Make all upgrade components fail using the ACTUAL method names
        # called by think()'s upgrade hooks (see module docstring).
        if hasattr(model, "architect") and model.architect is not None:
            model.architect.evaluate = MagicMock(side_effect=RuntimeError())
        if hasattr(model, "temporal_memory") and model.temporal_memory is not None:
            model.temporal_memory.update = MagicMock(side_effect=RuntimeError())
        if hasattr(model, "layered_predictor") and model.layered_predictor is not None:
            model.layered_predictor.update = MagicMock(side_effect=RuntimeError())
        if hasattr(model, "logic_layer") and model.logic_layer is not None:
            model.logic_layer.check_all = MagicMock(side_effect=RuntimeError())
        if hasattr(model, "meta_cognition") and model.meta_cognition is not None:
            model.meta_cognition.update = MagicMock(side_effect=RuntimeError())
        if hasattr(model, "experiment_planner") and model.experiment_planner is not None:
            model.experiment_planner.evaluate = MagicMock(side_effect=RuntimeError())
        if hasattr(model, "multiagent_world") and model.multiagent_world is not None:
            model.multiagent_world.step = MagicMock(side_effect=RuntimeError())

        # think() should still complete (all hooks are wrapped in try/except).
        result = model.think()
        assert result is not None
        # Signal is a dataclass with .data/.metadata/.confidence — verify the
        # core cycle produced a usable result.
        assert result.data is not None
        assert result.metadata is not None


class TestResourceExhaustion:
    """System behavior under resource exhaustion."""

    def test_think_after_garbage_collection(self):
        """System should work after forced GC (simulates memory pressure)."""
        model = ZeroDataModel(dim=16, seed=42)
        gc.collect()
        result = model.think()
        assert result is not None

    def test_repeated_think_no_memory_leak(self):
        """Running think() 100 times should not significantly increase memory."""
        import tracemalloc

        model = ZeroDataModel(dim=16, seed=42)
        tracemalloc.start()
        # Warm up.
        for _ in range(3):
            model.think()
        gc.collect()
        current_before, _ = tracemalloc.get_traced_memory()

        # Run 100 cycles.
        for _ in range(100):
            model.think()
        gc.collect()
        current_after, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        growth = current_after - current_before
        # Allow some growth but flag if > 10MB.
        assert growth < 10 * 1024 * 1024, (
            f"Memory grew by {growth / 1024 / 1024:.1f}MB in 100 cycles"
        )


class TestStateCorruption:
    """System behavior when state is corrupted."""

    def test_think_rejects_corrupted_belief(self):
        """If belief_state is corrupted with inf, think() should recover.

        The belief state lives at
        ``model.active_inference.generative_model.belief_state``. When it
        contains ``inf``, the per-module uncertainty computation in
        think() explicitly rejects non-finite values (``finite_mask``)
        so a single corrupted module cannot poison the integrated output.
        think() either completes with a finite confidence, or raises a
        numerical error (both acceptable).
        """
        model = ZeroDataModel(dim=16, seed=42)
        model.think()
        # Corrupt belief_state with inf (the actual attribute path).
        model.active_inference.generative_model.belief_state[0] = np.inf
        try:
            result = model.think()
            # If it completes, confidence should be finite.
            assert np.isfinite(result.confidence), "Confidence should be finite"
        except (ValueError, FloatingPointError, OverflowError):
            # Acceptable: system rejects corrupted state numerically.
            pass

    def test_model_state_recovery_after_nan(self):
        """After NaN injection and rejection, subsequent think() should work."""
        model = ZeroDataModel(dim=16, seed=42)
        model.think()
        original = model.active_inference.generative_model.belief_state.copy()
        # Inject NaN — update_belief rejects it (returns unchanged belief).
        model.active_inference.generative_model.update_belief(np.full(16, np.nan))
        # State should be preserved.
        np.testing.assert_array_equal(
            model.active_inference.generative_model.belief_state, original
        )
        # Subsequent think() should work normally.
        result = model.think()
        assert result is not None
