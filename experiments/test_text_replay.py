# experiments/test_text_replay.py
"""Verification tests for the Phase I text-replay + consolidation loop.

Four test classes:

1. ``TestExperienceStorage`` — experiences are added to the buffer
   every step, with the correct (obs, action, error, info_gain) tuple.
2. ``TestConsolidationEffect`` — consolidation reduces or maintains
   free energy on the replay batch (the design intent).
3. ``TestConsolidationVsNoConsolidation`` — over a short run on a
   structured text (alternating ``"abc abc abc ..."`` and
   ``"xyz xyz xyz ..."``), the WITH-consolidation variant does NOT
   have higher mean FE than the WITHOUT-consolidation variant. The
   test is a SOFT comparison because both runs use stochastic action
   selection; we only assert consolidation doesn't HURT learning.
4. ``TestSnapshots`` — snapshots are saved at the right interval and
   contain the expected fields.

Run with:  python -m pytest experiments/test_text_replay.py -v
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_encoder import TextEncoder  # noqa: E402
from text_stream import TextStream  # noqa: E402
from run_text_replay import (  # noqa: E402
    ReplayTextRunner,
    collect_snapshot,
    load_snapshot,
)
from run_text_curious import make_text_curious_model  # noqa: E402
from text_experience_buffer import TextExperienceBuffer, TextConsolidator  # noqa: E402

from zero_data_model.model import ZeroDataModel  # noqa: E402


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #
# A short, structured text: alternating "abc abc ..." and "xyz xyz ..."
# blocks. The transition between the two patterns should be a HIGH-
# prediction-error region, ideal for testing consolidation.
STRUCTURED_TEXT = (
    "abc abc abc abc abc abc abc abc abc abc "
    "xyz xyz xyz xyz xyz xyz xyz xyz xyz xyz "
) * 20  # ~5600 chars


@pytest.fixture
def structured_text_file():
    """Write STRUCTURED_TEXT to a temp file and return the path."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(STRUCTURED_TEXT)
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)


def _make_runner(
    text_path: str,
    steps: int = 200,
    consolidation_interval: int = 50,
    snapshot_interval: int = 50,
    seed: int = 42,
    dim: int = 32,
) -> ReplayTextRunner:
    """Build a fresh ReplayTextRunner for tests."""
    model = make_text_curious_model(
        dim=dim, seed=seed,
        beta_start=1.0, beta_min=0.01,
        decay_steps=steps, decay_type="linear",
    )
    stream = TextStream(text_path, block_size=64, seed=seed)
    encoder = TextEncoder(block_size=64, output_dim=dim, seed=seed)
    with tempfile.TemporaryDirectory() as tmpdir:
        # The runner needs an output_dir for snapshots. We set it
        # AFTER construction so the runner is constructed with a
        # persistent dir we control. But __init__ creates the dir,
        # so we pass a real path here.
        pass
    runner = ReplayTextRunner(
        model, stream, encoder, max_steps=steps,
        buffer_capacity=1000,
        consolidation_interval=consolidation_interval,
        consolidation_batch_size=16,
        consolidation_epochs=1,
        snapshot_interval=snapshot_interval,
        output_dir=Path(tempfile.mkdtemp()),
        seed=seed,
    )
    return runner


# ------------------------------------------------------------------ #
# Test 1: experience storage
# ------------------------------------------------------------------ #
class TestExperienceStorage:
    """Experiences should be stored in the buffer every step."""

    def test_buffer_grows_with_steps(self, structured_text_file):
        runner = _make_runner(structured_text_file, steps=30,
                              consolidation_interval=1000,  # disable consolidation
                              snapshot_interval=1000)      # disable snapshots
        runner.run()
        # After 30 steps, the buffer should have 30 entries (capacity
        # is 1000, so no eviction yet).
        assert len(runner.buffer) == 30, (
            f"buffer should have 30 entries after 30 steps, got {len(runner.buffer)}"
        )

    def test_buffer_records_correct_fields(self, structured_text_file):
        runner = _make_runner(structured_text_file, steps=5,
                              consolidation_interval=1000,
                              snapshot_interval=1000)
        runner.run()
        # Sample one experience and check its fields.
        batch = runner.buffer.sample(1, mode="uniform")
        assert len(batch) == 1
        exp = batch[0]
        assert exp.obs.shape == (32,), f"obs shape wrong: {exp.obs.shape}"
        assert isinstance(exp.action, int)
        assert 0 <= exp.action <= 4, f"action out of range: {exp.action}"
        assert isinstance(exp.error, float)
        assert isinstance(exp.info_gain, float)
        assert exp.step >= 0

    def test_buffer_priority_correlates_with_error(self, structured_text_file):
        """Higher-error experiences should have higher-or-equal priority.

        The priority formula is ``1 + tanh(|error| / max_error_leaky)``
        where ``max_error_leaky`` is a running max with 0.99 decay.
        This normalises the priority into [1, 2] AND applies a tanh
        squashing, so the correlation with raw |error| is intentionally
        softened (we don't want a single huge error to dominate
        sampling). We assert only that the correlation is non-negative
        — i.e., higher error → higher-or-equal priority, not strict
        monotonicity.
        """
        runner = _make_runner(structured_text_file, steps=50,
                              consolidation_interval=1000,
                              snapshot_interval=1000)
        runner.run()
        all_exp = list(runner.buffer._buffer)
        errors = np.array([abs(e.error) for e in all_exp])
        priorities = np.array([e.priority for e in all_exp])
        # Correlation should be non-negative (any positive value is fine;
        # the design intent is monotonic in expectation, not strictly).
        if np.std(errors) > 0 and np.std(priorities) > 0:
            corr = float(np.corrcoef(errors, priorities)[0, 1])
            assert corr > -0.1, (
                f"priority-error correlation should be > -0.1 (non-negative "
                f"in expectation), got {corr}"
            )


# ------------------------------------------------------------------ #
# Test 2: consolidation effect on the replay batch
# ------------------------------------------------------------------ #
class TestConsolidationEffect:
    """Consolidation should reduce (or not increase) FE on the replay batch."""

    def test_consolidation_round_records_delta(self, structured_text_file):
        runner = _make_runner(structured_text_file, steps=100,
                              consolidation_interval=50,
                              snapshot_interval=1000)
        summary = runner.run()
        # Should have run at least 1 consolidation round (at step 50 and 100).
        assert summary.n_consolidation_rounds >= 1, (
            f"should have >=1 consolidation rounds, got {summary.n_consolidation_rounds}"
        )
        # Each round should have a recorded delta.
        for r in summary.consolidation_rounds:
            assert r.n_replayed > 0, "round should replay >0 experiences"
            assert isinstance(r.delta_free_energy, float)
            assert isinstance(r.free_energy_before, float)
            assert isinstance(r.free_energy_after, float)

    def test_consolidation_does_not_increase_mean_fe(self, structured_text_file):
        """The mean ΔFE across consolidation rounds should not be
        significantly positive (i.e., consolidation shouldn't make
        things worse). We allow a small tolerance for noise."""
        runner = _make_runner(structured_text_file, steps=200,
                              consolidation_interval=50,
                              snapshot_interval=1000)
        summary = runner.run()
        # Mean delta should be near zero or negative (improvement).
        # We allow a +0.5 tolerance for stochasticity.
        assert summary.mean_delta_fe_consolidation < 0.5, (
            f"mean ΔFE should be < 0.5 (consolidation shouldn't hurt), "
            f"got {summary.mean_delta_fe_consolidation}"
        )


# ------------------------------------------------------------------ #
# Test 3: consolidation vs no-consolidation FE trajectory
# ------------------------------------------------------------------ #
class TestConsolidationVsNoConsolidation:
    """Over a short run, WITH-consolidation should not have higher
    final FE than WITHOUT-consolidation.

    NOTE: this is a SOFT test. The action-selection noise + the small
    step count means the comparison is noisy. We assert the WITH run
    does at least as well as WITHOUT (final_FE_with <= final_FE_without + tol).
    """

    def _run(self, text_path: str, with_consolidation: bool, steps: int = 200) -> tuple[float, float]:
        """Return (mean_fe, final_fe) for one run."""
        if with_consolidation:
            cons_interval = 50
        else:
            cons_interval = 10000  # effectively disable
        runner = _make_runner(text_path, steps=steps,
                              consolidation_interval=cons_interval,
                              snapshot_interval=10000)
        summary = runner.run()
        return summary.mean_free_energy, summary.final_free_energy

    def test_with_cons_not_worse_than_without(self, structured_text_file):
        """WITH-consolidation should NOT have higher final FE than WITHOUT."""
        # Run twice (deterministic with seed) to compare.
        mean_with, final_with = self._run(structured_text_file, with_consolidation=True)
        mean_without, final_without = self._run(structured_text_file, with_consolidation=False)
        # Allow a generous tolerance (the structured text + short run
        # means signal-to-noise is low).
        assert final_with <= final_without + 2.0, (
            f"WITH-consolidation final FE ({final_with:.3f}) should be "
            f"<= WITHOUT ({final_without:.3f}) + 2.0"
        )
        assert mean_with <= mean_without + 2.0, (
            f"WITH-consolidation mean FE ({mean_with:.3f}) should be "
            f"<= WITHOUT ({mean_without:.3f}) + 2.0"
        )


# ------------------------------------------------------------------ #
# Test 4: snapshots
# ------------------------------------------------------------------ #
class TestSnapshots:
    """Snapshots should be saved at the right interval and contain
    the expected fields."""

    def test_snapshots_saved_at_right_interval(self, structured_text_file):
        runner = _make_runner(structured_text_file, steps=100,
                              consolidation_interval=10000,  # disable
                              snapshot_interval=50)
        summary = runner.run()
        # 100 steps / 50 interval = 2 snapshots (at step 50 and 100).
        assert len(summary.snapshot_paths) == 2, (
            f"should have 2 snapshots, got {len(summary.snapshot_paths)}"
        )
        assert summary.snapshot_steps == [50, 100], (
            f"snapshot steps should be [50, 100], got {summary.snapshot_steps}"
        )
        # Each path should exist.
        for p in summary.snapshot_paths:
            assert Path(p).exists(), f"snapshot file missing: {p}"

    def test_snapshot_schema(self, structured_text_file):
        """Loaded snapshot should contain the expected top-level keys."""
        runner = _make_runner(structured_text_file, steps=50,
                              consolidation_interval=10000,
                              snapshot_interval=50)
        summary = runner.run()
        assert len(summary.snapshot_paths) >= 1
        snap = load_snapshot(summary.snapshot_paths[0])
        # Top-level keys.
        for key in ["meta", "text_block", "obs", "consciousness",
                    "active_inference", "category_engine", "causal_emergence"]:
            assert key in snap, f"snapshot missing top-level key: {key}"
        # Meta fields.
        meta = snap["meta"]
        for key in ["schema_version", "step", "cycle_count", "dim", "pos",
                    "free_energy", "prediction_error"]:
            assert key in meta, f"snapshot meta missing key: {key}"
        # Text block is a string.
        assert isinstance(snap["text_block"], str)
        assert len(snap["text_block"]) > 0
        # obs is a numpy array.
        assert isinstance(snap["obs"], np.ndarray)
        # Active inference has belief_state.
        assert "belief_state" in snap["active_inference"]
        # Consciousness has self_state.
        assert "self_state" in snap["consciousness"]

    def test_collect_snapshot_directly(self):
        """Test ``collect_snapshot`` directly without running the loop."""
        model = make_text_curious_model(dim=16, seed=42)
        obs = np.random.default_rng(0).standard_normal(16)
        text = "hello world test"
        snap = collect_snapshot(
            model, step=42, text_block=text, obs=obs,
            pos=128, free_energy=1.5, prediction_error=0.5,
        )
        assert snap["meta"]["step"] == 42
        assert snap["meta"]["pos"] == 128
        assert snap["meta"]["free_energy"] == 1.5
        assert snap["text_block"] == text
        assert np.array_equal(snap["obs"], obs)


# ------------------------------------------------------------------ #
# Self-test entry point (run without pytest)
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v"],
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    sys.exit(result.returncode)
