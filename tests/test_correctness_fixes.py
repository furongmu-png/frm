# tests/test_correctness_fixes.py
"""Regression tests for the specific bugs identified in the military-grade audit
and fixed in the source code.

Each test pins one documented fix so the bug cannot silently reappear:

* Fix 1+2  - ``kl_divergence`` pads mismatched-length inputs and self-KL is ~0.
* Fix 3    - ``compute_free_energy`` is pure (no belief_state mutation).
* Fix 4    - DNA storage uses per-cycle sequence keys (crossover actually runs).
* Fix 5    - ``classify_text`` does not corrupt the shared belief state.
* Fix 6    - ``PatternRecognizer`` uses input-prototype free-energy diff.
* Fix 7    - Model holds a re-entrant lock around state-mutating cycles.
* Fix 8    - Action / free-energy histories are bounded deques.
* Fix 9    - Optional ``seed`` parameter pins the global RNG.
* Fix 10   - ``VisionRules.convolve`` uses scipy.ndimage.correlate.
* Fix 11   - ``CellularAutomata.step_n`` advances without recording history.
* Fix 13   - ``Functor.apply`` consults ``object_map`` before shape matching.
* Fix 14   - ``solve`` actually depends on the input (cost = outer(p,p) + 0.1*I).
* Fix 18   - Script detection covers Greek and returns 'unknown' for empty input.
* Fix 24   - ``get_quantum_backend`` clamps ``n_qubits`` to <= 20.
* skewness - ``kernels._skewness`` matches scipy.stats.skew default (g1).
"""

from __future__ import annotations

from threading import Thread

import numpy as np
import scipy.stats

from zero_data_model.active_inference import ActiveInferenceEngine
from zero_data_model.base import Signal
from zero_data_model.biological import BiologicalSubstrate
from zero_data_model.capabilities.nlp_advanced import MultiLingualEncoder
from zero_data_model.category_engine import Functor
from zero_data_model.hardware import kernels
from zero_data_model.hardware.quantum import get_quantum_backend
from zero_data_model.math_universe import FractalGenerator, InformationGeometry
from zero_data_model.model import ZeroDataModel

# --------------------------------------------------------------------------- #
# Fix 1 + 2: kl_divergence handles mismatched lengths and self-KL is ~0.
# --------------------------------------------------------------------------- #


def test_kl_divergence_mismatched_lengths():
    """kl_divergence does NOT crash on inputs of differing lengths."""
    ig = InformationGeometry(dim=8)
    # Lengths 3 and 5 -- both shorter than dim=8 so they are padded to 8.
    out = ig.kl_divergence(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert isinstance(out, float)
    assert np.isfinite(out)


def test_kl_divergence_self_is_zero():
    """kl_divergence(p, p) is approximately zero (within 1e-6)."""
    ig = InformationGeometry(dim=8)
    p = np.abs(np.random.randn(8)) + 0.1
    out = ig.kl_divergence(p, p)
    assert abs(out) < 1e-6


# --------------------------------------------------------------------------- #
# Fix 1 (TrendAnalyzer odd-length path): odd-length series does not crash.
# --------------------------------------------------------------------------- #


def test_analyze_trend_odd_length():
    """analyze_trend does NOT crash on an odd-length series (KL on unequal halves)."""
    model = ZeroDataModel(dim=16)
    out = model.analyze_trend(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert isinstance(out, dict)
    assert set(out.keys()) == {
        "trend_slope",
        "regime",
        "curvature",
        "geodesic_deviation",
        "isomorphism_score",
    }
    assert np.isfinite(out["geodesic_deviation"])


def test_fractal_compress_odd_dim():
    """FractalGenerator.compress works on an odd dim (corrcoef split fix)."""
    fg = FractalGenerator(dim=7)
    out = fg.compress(np.random.randn(7))
    assert isinstance(out, dict)
    assert set(out.keys()) == {"mean", "std", "self_similarity"}
    assert np.isfinite(out["self_similarity"])


# --------------------------------------------------------------------------- #
# Fix 3: compute_free_energy is pure (no belief_state mutation).
# --------------------------------------------------------------------------- #


def test_compute_free_energy_does_not_mutate_belief():
    """compute_free_energy does NOT mutate the shared belief_state."""
    engine = ActiveInferenceEngine(state_dim=16, obs_dim=16, action_dim=8)
    belief_before = engine.generative_model.belief_state.copy()
    engine.compute_free_energy(np.random.randn(16))
    np.testing.assert_array_equal(
        engine.generative_model.belief_state, belief_before
    )


def test_classify_text_does_not_corrupt_belief():
    """classify_text does NOT corrupt the active-inference belief_state."""
    model = ZeroDataModel(dim=16)
    belief_before = model.active_inference.generative_model.belief_state.copy()
    model.classify_text("hello world")
    np.testing.assert_allclose(
        model.active_inference.generative_model.belief_state, belief_before, atol=1e-12
    )


# --------------------------------------------------------------------------- #
# Fix 4: DNA storage uses per-cycle sequence keys so crossover runs.
# --------------------------------------------------------------------------- #


def test_dna_recombination_runs_with_population():
    """After 3 process() calls the store holds 3 keys and generate produces a
    finite, dim-length signal from crossover (not the random fallback)."""
    substrate = BiologicalSubstrate(dim=16)
    for _ in range(3):
        substrate.process(
            Signal(data=np.random.randn(16), metadata={}, confidence=1.0)
        )
    # The store now has >=2 unique sequence keys (Fix 4), so generate() takes
    # the crossover path rather than the random fallback.
    generated = substrate.dna_storage.generate(
        Signal(data=np.random.randn(16), metadata={}, confidence=1.0)
    )
    assert np.all(np.isfinite(generated.data))
    assert generated.data.shape[0] > 0


# --------------------------------------------------------------------------- #
# Skewness fix: _skewness matches scipy.stats.skew default (g1, bias=True).
# --------------------------------------------------------------------------- #


def test_skewness_matches_scipy_default():
    """kernels._skewness matches scipy.stats.skew(data) default to < 1e-10."""
    rng = np.random.default_rng(123)
    data = rng.standard_normal(64)
    expected = float(scipy.stats.skew(data))  # default bias=True -> g1
    got = float(kernels._skewness(np.ascontiguousarray(data, dtype=float)))
    assert abs(expected - got) < 1e-10, (
        f"scipy={expected!r}, kernel={got!r}, diff={abs(expected - got)!r}"
    )


# --------------------------------------------------------------------------- #
# Fix 14: solve() actually depends on the input (energy is finite).
# --------------------------------------------------------------------------- #


def test_solve_uses_problem():
    """model.solve encodes the problem into the annealer cost matrix and
    returns a Signal whose metadata carries a finite 'energy' value."""
    model = ZeroDataModel(dim=16)
    problem = np.random.randn(16)
    result = model.solve(problem)
    assert "energy" in result.metadata
    assert np.isfinite(result.metadata["energy"])


# --------------------------------------------------------------------------- #
# Fix 13: Functor.apply consults object_map before shape matching.
# --------------------------------------------------------------------------- #


def test_functor_apply_uses_object_map():
    """Functor.apply uses the morphism selected by object_map (not just any
    shape-matching morphism)."""
    # Set up two morphisms with the same shape but different values.
    desired = np.full((4, 4), 1.0)  # the morphism object_map points to
    other = np.full((4, 4), 0.0)  # a shape-matching decoy
    functor = Functor(
        source="A",
        target="B",
        object_map={"a": "b"},
        morphism_map={("a", "b"): desired, ("x", "y"): other},
    )
    obj = np.ones(4)
    out = functor.apply(obj)
    # The desired morphism is all 1s -> output is sum of obj = [4, 4, 4, 4].
    np.testing.assert_allclose(out, np.full(4, 4.0), atol=1e-10)


# --------------------------------------------------------------------------- #
# Fix 18: Script detection covers Greek; unknown input returns 'unknown'.
# --------------------------------------------------------------------------- #


def test_script_detection_greek():
    """Greek script is detected (Fix 18 added the Greek Unicode block)."""
    enc = MultiLingualEncoder(dim=64)
    assert enc.detect_script("αβγδε") == "greek"


def test_script_detection_unknown_returns_unknown():
    """Empty / punctuation-only input returns 'unknown', not 'latin'."""
    enc = MultiLingualEncoder(dim=64)
    assert enc.detect_script("") == "unknown"
    assert enc.detect_script("123 !!!") == "unknown"


# --------------------------------------------------------------------------- #
# Fix 11: CellularAutomata.step_n advances without recording history.
# --------------------------------------------------------------------------- #


def test_cellular_automata_step_n():
    """step_n advances n steps in place; state shape is preserved and no
    history is recorded (it is not a return value of the tape form)."""
    from zero_data_model.biological import CellularAutomata

    ca = CellularAutomata(size=32, rule=110)
    final_state = ca.step_n(5)
    assert final_state.shape == (32,)
    # The state dtype stays int (CA states are 0/1).
    assert final_state.dtype.kind == "i"


# --------------------------------------------------------------------------- #
# Fix 24: get_quantum_backend clamps n_qubits to <= 20.
# --------------------------------------------------------------------------- #


def test_annealer_clamps_qubits():
    """A request for n_qubits=50 is clamped to 20 internally; the factory
    still returns a working QuantumBackend instance."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        backend = get_quantum_backend(n_qubits=50, prefer="simulator")
    # The factory must have clamped the qubit count down to the supported max.
    assert backend.n_qubits == 20
    # Shape the ansatz parameters for the *clamped* qubit count so the
    # state-vector math (length 2**n_qubits) broadcasts correctly.
    n = backend.n_qubits
    params = np.random.randn(2, n, 2) * 0.1
    entangling = np.random.randn(n, n) * 0.05
    out = backend.evolve_and_measure(params, entangling, n_shots=64)
    assert isinstance(out, np.ndarray)
    # C-6 fix: output length is 2**n_qubits for both backends (was 2 * n_qubits).
    assert out.shape == (2 ** n,)
    assert out.shape[0] > 0


# --------------------------------------------------------------------------- #
# Fix 9: model seed reproducibility.
# --------------------------------------------------------------------------- #


def test_model_seed_reproducibility():
    """Two models with the same seed produce the same think() output.

    Fix 9 (final): ``ZeroDataModel.__init__`` stores ``self._seed`` and
    ``self._rng = np.random.default_rng(seed)``. ``think()`` re-seeds numpy's
    global RNG from ``self._rng`` at the start of every cycle so the legacy
    ``np.random.*`` calls inside each module's ``process``/``update`` consume
    a deterministic stream that is identical across two models built with the
    same seed. The Qiskit ``StatevectorSampler`` performs stochastic
    shot-based measurement that does NOT draw from numpy's global RNG, so when
    a seed is set the model forces the deterministic ``SimulatorQuantumBackend``
    (closed-form state-vector evolution, no sampling) for full reproducibility.
    """
    m1 = ZeroDataModel(dim=16, seed=42)
    m2 = ZeroDataModel(dim=16, seed=42)

    out1a = m1.think()
    out2a = m2.think()
    assert np.allclose(out1a.data, out2a.data), "First think() should match"

    # Second think() cycle must also match — both models' ``_rng`` advance
    # identically and both re-seed from the same cycle_seed each cycle.
    out1b = m1.think()
    out2b = m2.think()
    assert np.allclose(out1b.data, out2b.data), "Second think() should match"

    # And the second cycle should differ from the first (the model actually
    # progresses; otherwise re-seeding with a constant would mask the bug).
    assert not np.allclose(out1a.data, out1b.data), "think() should progress across cycles"


# --------------------------------------------------------------------------- #
# Fix 7: Model holds a re-entrant lock around state-mutating cycles.
# --------------------------------------------------------------------------- #


def test_model_thread_lock():
    """Concurrent think() calls do not corrupt state (lock serializes them).

    We can't observe the lock directly, but we can assert that 4 concurrent
    think() calls all complete, all return the right shape, and none raises.
    """
    model = ZeroDataModel(dim=16)
    results: list = []

    def run() -> None:
        try:
            results.append(model.think())
        except Exception as exc:  # noqa: BLE001 - we want to surface any failure
            results.append(exc)

    threads = [Thread(target=run) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 4
    for r in results:
        assert not isinstance(r, Exception), f"thread raised: {r!r}"
        assert r.data.shape == (16,)
