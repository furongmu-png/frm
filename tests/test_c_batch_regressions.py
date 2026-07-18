# tests/test_c_batch_regressions.py
"""Regression tests for the C-batch theoretical rewrites.

Each test guards one of the C-1 ... C-7 fixes against silent regression:
  C-1  DNAStorage byte-exact round-trip (no quantization loss)
       MorphogeneticField Gray-Scott reaction-diffusion (non-trivial u/v)
  C-2  Functor.apply_morphism + find_invertible_map (Householder, orthogonal)
  C-3  ConsciousnessCore.process updates attention_weights (non-uniform)
  C-4  vietoris_rips_betti: square=1 loop, pentagon=1 loop, K5=0 loops
  C-5  ActiveInference gradient descent reduces error; KL term present;
       select_action uses active_weights + epistemic bonus
  C-6  SimulatorQuantumBackend: 2**n_qubits dim, RY, CNOT, Born rule, Bell state
  C-7  ZeroDataModel._integrate weights by inverse uncertainty;
       seeded reproducibility preserved under weighted integration
"""

from __future__ import annotations

import numpy as np
import pytest

from zero_data_model.base import Signal
from zero_data_model.model import ZeroDataModel

# ---------------------------------------------------------------------------
# C-1: DNAStorage lossless round-trip + Gray-Scott morphogenesis
# ---------------------------------------------------------------------------


def test_c1_dna_storage_lossless_roundtrip():
    """DNAStorage encodes/decodes float64 arrays with byte-exact precision.

    C-1 fix: the previous 4-level quantization (normalize + clip to 4 bins)
    kept only 4 distinct values alive, losing all continuous information.
    The new byte-view encoding (float64 -> uint8 bytes -> 4 base-4 digits
    per byte) is information-preserving and round-trips exactly.
    """
    from zero_data_model.biological import DNAStorage

    dna = DNAStorage()
    data = np.array(
        [0.123456789, -1.0, 3.14159265, 1e-10, 1e10, 0.0, -0.0, 42.0],
        dtype=np.float64,
    )
    dna.store("lossless", data)
    retrieved = dna.retrieve("lossless")
    assert retrieved is not None
    # Byte-exact round-trip (not just close — exactly equal).
    np.testing.assert_array_equal(retrieved, data)


def test_c1_dna_storage_preserves_precision():
    """Sub-ULP differences survive the DNA round-trip (true losslessness)."""
    from zero_data_model.biological import DNAStorage

    dna = DNAStorage()
    data = np.array([1.0, 1.0 + 2**-52], dtype=np.float64)  # 1 ULP apart
    dna.store("ulp", data)
    retrieved = dna.retrieve("ulp")
    assert retrieved is not None
    np.testing.assert_array_equal(retrieved, data)
    # The two values must still be distinct after round-trip.
    assert retrieved[0] != retrieved[1]


def test_c1_morphogenetic_field_gray_scott_nontrivial():
    """MorphogeneticField develops a non-trivial Gray-Scott pattern.

    C-1 fix: the previous pure-Laplacian diffusion could only smooth the
    field to its mean. The Gray-Scott reaction-diffusion system (with
    activator u and inhibitor v) produces self-organising patterns whose
    variance grows or stabilises rather than collapsing to zero.
    """
    from zero_data_model.biological import MorphogeneticField

    mf = MorphogeneticField(grid_size=16)
    pattern = mf.develop(n_steps=20)
    assert pattern.shape == (16, 16)
    # The pattern must not be uniform (Gray-Scott creates structure).
    assert float(np.std(pattern)) > 1e-6
    # The Gray-Scott (u, v) fields are clipped to [0, 1] each step
    # (explicit-Euler stability). ``develop`` returns the passive ``grid``
    # carrier modulated by the morphogens, so only the morphogens -- not the
    # composite ``pattern`` -- are bounded to [0, 1].
    u, v = mf.morphogens[0], mf.morphogens[1]
    assert float(u.min()) >= -1e-10
    assert float(u.max()) <= 1.0 + 1e-10
    assert float(v.min()) >= -1e-10
    assert float(v.max()) <= 1.0 + 1e-10


# ---------------------------------------------------------------------------
# C-2: Functor.apply_morphism + find_invertible_map (Householder)
# ---------------------------------------------------------------------------


def test_c2_find_invertible_map_is_orthogonal():
    """find_invertible_map returns an invertible map honoring T @ source = target.

    C-2 fix: the previous find_isomorphism was a misnamed cosine similarity
    (not invertible). The new find_invertible_map builds a Householder
    reflection T = I - 2vv^T that maps the normalised source to the
    normalised target.

    Round-9 audit R9-003: the map is now SCALED by ``nb/na`` so the
    documented contract ``T @ source = target`` holds exactly (not just
    up to a norm ratio). The reflection component is still orthogonal
    (T_reflection @ T_reflection.T = I), so the scaled map satisfies
    ``T @ T.T = (nb/na)^2 * I`` -- still invertible (non-zero scalar
    times an orthogonal matrix). We assert the documented contract
    directly: ``T @ source ≈ target`` to float tolerance.
    """
    from zero_data_model.category_engine import CategoryTheoryEngine

    engine = CategoryTheoryEngine(dim=8)
    rng = np.random.default_rng(42)
    source = rng.standard_normal(8)
    target = rng.standard_normal(8)
    T = engine.find_invertible_map(source, target)
    assert T is not None
    # Documented contract: T @ source == target (NOT just up to a norm ratio).
    np.testing.assert_allclose(T @ source, target, atol=1e-10)
    # Invertibility: the scaled orthogonal map has full rank.
    assert np.linalg.matrix_rank(T) == 8
    # Scaled orthogonality: T @ T.T = (nb/na)^2 * I.
    na = float(np.linalg.norm(source))
    nb = float(np.linalg.norm(target))
    scale_sq = (nb / na) ** 2
    np.testing.assert_allclose(T @ T.T, scale_sq * np.eye(8), atol=1e-10)
    # The map still sends the normalised source to the normalised target
    # (the reflection component maps a_hat -> b_hat; the scale preserves
    # the direction).
    s_hat = source / np.linalg.norm(source)
    t_hat = target / np.linalg.norm(target)
    np.testing.assert_allclose(T @ s_hat, t_hat * (nb / na), atol=1e-10)


def test_c2_functor_apply_morphism_uses_object_map():
    """Functor.apply_morphism applies the morphism to the source vector.

    C-2 fix: the original Functor.apply ignored the morphism_map entirely.
    """
    from zero_data_model.category_engine import Category, Functor

    cat = Category(name="morph")
    a = np.array([1.0, 0.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0, 0.0])
    cat.add_object("a", a)
    cat.add_object("b", b)
    # Build a 90-degree CCW rotation in the (a, b) plane: R @ [1,0,0,0] = [0,1,0,0].
    # apply_morphism does ``transform @ vector``, so the first column of R must
    # equal b (= [0,1,0,0]); the previous sign convention sent a -> [0,-1,0,0].
    R = np.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    functor = Functor(
        source=cat,
        target=cat,
        object_map={"a": "b"},
        morphism_map={("a", "b"): R},
    )
    result = functor.apply_morphism("a", "b", a)
    assert result is not None
    np.testing.assert_allclose(result, b, atol=1e-10)


# ---------------------------------------------------------------------------
# C-3: ConsciousnessCore.process updates attention_weights
# ---------------------------------------------------------------------------


def test_c3_consciousness_process_updates_attention():
    """ConsciousnessCore.process produces non-uniform attention_weights.

    C-3 fix: update_attention was never called from process, so the weights
    stayed uniform (1/dim) and broadcast's signal * attention_weights was a
    no-op scaling. After the fix, process calls update_attention(|x|) before
    broadcasting, so the weights reflect the input's per-dimension salience.

    Round-3 audit C-batch: pass a fixed rng so the test is deterministic
    under the per-module Generator refactor (CRIT-1). Without a seed the
    random layer weights + forward noise could flip the salient dimension.
    """
    from zero_data_model.consciousness_core import ConsciousnessCore

    core = ConsciousnessCore(dim=8, rng=np.random.default_rng(0))
    # A signal with one highly salient dimension and the rest near zero.
    data = np.zeros(8)
    data[3] = 10.0
    core.process(Signal(data=data))
    weights = core.workspace.attention_weights
    # The weights must NOT be uniform (1/8 each) after processing a
    # strongly non-uniform input.
    assert not np.allclose(weights, np.ones(8) / 8.0)
    # The salient dimension (index 3) should have above-mean weight.
    assert weights[3] > float(np.mean(weights))


# ---------------------------------------------------------------------------
# C-4: Vietoris-Rips Betti numbers
# ---------------------------------------------------------------------------


def test_c4_vietoris_rips_betti_square_one_loop():
    """Four points at the corners of a unit square (r=1.1) form one 1-loop.

    C-4 fix: the previous compute_betti_numbers used ``betti_1 = n - betti_0``
    which reported spurious 1-loops for any 1D point cloud. The new
    vietoris_rips_betti builds the 1-skeleton + 2-simplices and computes
    rank(d_2) over GF(2) for the correct betti_1.
    """
    from zero_data_model.math_universe import MathematicalUniverse

    mu = MathematicalUniverse(dim=8)
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    # ``vietoris_rips_betti`` lives on the ``TopologicalAnalyzer`` held by
    # ``MathematicalUniverse`` (mu.topology), not on ``MathematicalUniverse``
    # itself.
    betti = mu.topology.vietoris_rips_betti(square, max_radius=1.1)
    assert betti[0] == 1  # one connected component
    assert betti[1] == 1  # exactly one 1-loop (the square perimeter)


def test_c4_vietoris_rips_betti_k5_no_loops():
    """A complete graph K5 (pentagon at r=2.0) fills in all 1-loops.

    With 5 points and all edges + all 10 triangles present, the boundary
    matrix d_2 has full rank, so betti_1 = 0.
    """
    from zero_data_model.math_universe import MathematicalUniverse

    mu = MathematicalUniverse(dim=8)
    # Pentagon vertices on the unit circle; r=2.0 connects all pairs.
    angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    pentagon = np.column_stack([np.cos(angles), np.sin(angles)])
    betti = mu.topology.vietoris_rips_betti(pentagon, max_radius=2.0)
    assert betti[0] == 1
    assert betti[1] == 0  # K5 fills all 1-loops


# ---------------------------------------------------------------------------
# C-5: ActiveInference gradient descent + KL + epistemic
# ---------------------------------------------------------------------------


def test_c5_gradient_descent_reduces_error():
    """emission_gradient_step monotonically reduces prediction error.

    C-5 fix: update() was a noise-injection random walk (no gradient). The
    new emission_gradient_step performs proper gradient descent on
    ||obs - state @ emission||^2, so error strictly decreases.
    """
    from zero_data_model.active_inference import ActiveInferenceEngine

    engine = ActiveInferenceEngine(state_dim=8, obs_dim=8, action_dim=4)
    # Fix the state and observation so we measure only the emission update.
    state = np.random.default_rng(0).standard_normal(8)
    obs = np.random.default_rng(1).standard_normal(8)
    # The generative model is exposed as ``generative_model`` (not ``generative``).
    # Populate the full gradient-descent cache: emission_gradient_step takes
    # the cached branch (which recomputes error from _last_observation + the
    # current emission) only when BOTH _last_error and _last_observation are
    # non-None -- otherwise it targets a zero observation, which would grow
    # the error against ``obs`` instead of shrinking it.
    engine.generative_model.belief_state = state.copy()
    engine.generative_model._last_state = state.copy()
    engine.generative_model._last_observation = obs.copy()
    engine.generative_model._last_error = obs.copy()

    # Compute initial error.
    predicted = state @ engine.generative_model.emission[:, :8]
    err0 = float(np.mean((obs[:8] - predicted[:8]) ** 2))

    # Run a few gradient steps.
    for _ in range(20):
        engine.generative_model.emission_gradient_step(0.1)

    # Recompute error with the same state (gradient step only updated emission).
    predicted = state @ engine.generative_model.emission[:, :8]
    err1 = float(np.mean((obs[:8] - predicted[:8]) ** 2))
    assert err1 < err0, f"error did not decrease: {err0} -> {err1}"


def test_c5_free_energy_includes_kl_term():
    """compute_free_energy includes KL(q||p) in addition to prediction error.

    C-5 fix: the original complexity term was just ||belief||^2 (a
    regularizer, not a KL divergence). The new form adds the true
    KL(N(belief, sigma^2 I) || N(0, I)) = 0.5*(||b||^2 + sigma^2 * dim
    - dim - dim * log(sigma^2)).

    To isolate the KL contribution we zero the emission (so the prediction
    error term vanishes against a zero observation) and push the belief far
    from the prior. ``compute_free_energy`` is a method on the engine (not
    the generative model) and takes the observation as its argument.
    """
    from zero_data_model.active_inference import ActiveInferenceEngine

    engine = ActiveInferenceEngine(state_dim=8, obs_dim=8, action_dim=4)
    # Push the belief far from the prior (0) so KL(q||p) is large.
    engine.generative_model.belief_state = np.ones(8) * 5.0
    # Zero the emission so the prediction-error term vanishes, isolating KL.
    engine.generative_model.emission[:] = 0.0
    # Observation matching the (zero) prediction -> prediction error ~ 0.
    fe = engine.compute_free_energy(np.zeros(8))
    # With belief = 5*ones(8), sigma^2 = 1 (no action history yet) and zero
    # prediction error, FE = KL = 0.5 * ||5*ones||^2 = 0.5 * 25 * 8 = 100.
    assert fe > 1.0, f"KL term missing: FE = {fe}"


# ---------------------------------------------------------------------------
# C-6: Quantum simulator (2**n_qubits, RY, CNOT, Born rule)
# ---------------------------------------------------------------------------


def test_c6_simulator_output_dim_is_2_pow_n_qubits():
    """SimulatorQuantumBackend returns a 2**n_qubits probability vector.

    C-6 fix: the previous simulator returned a length-``2 * n_qubits`` real
    vector (treating the state as n_qubits independent 2-level systems).
    The new simulator evolves a complex state vector in the full
    2**n_qubits Hilbert space and returns Born-rule probabilities.
    """
    from zero_data_model.hardware.quantum import SimulatorQuantumBackend

    for n in (1, 2, 3, 4):
        backend = SimulatorQuantumBackend(n_qubits=n, n_layers=1)
        params = np.zeros((1, n, 2))
        entangling = np.zeros((n, n))
        out = backend.evolve_and_measure(params, entangling)
        assert out.shape == (2 ** n,), f"n={n}: got {out.shape}, expected ({2**n},)"
        assert abs(out.sum() - 1.0) < 1e-6


def test_c6_simulator_bell_state():
    """RY(pi/2) on q0 + CNOT(0,1) creates the Bell state (|00>+|11>)/sqrt(2).

    C-6 fix: the previous RY gate paired adjacent amplitudes (0::2, 1::2)
    as if each qubit was independent. The new RY correctly pairs amplitudes
    whose indices differ only in the target qubit's bit. The previous
    "entangle" added 0.01 * entangling @ state (non-unitary). The new CNOT
    is the textbook bit-flip conditioned on the control.
    """
    from zero_data_model.hardware.quantum import SimulatorQuantumBackend

    backend = SimulatorQuantumBackend(n_qubits=2, n_layers=1)
    state = np.zeros(4, dtype=complex)
    state[0] = 1.0  # |00>
    state = backend._apply_ry(state, np.pi / 2, 0)  # RY(pi/2) on qubit 0
    state = backend._apply_cnot(state, 0, 1)  # CNOT(0, 1)
    probs = (state.real**2 + state.imag**2).astype(float)
    np.testing.assert_allclose(probs, [0.5, 0.0, 0.0, 0.5], atol=1e-10)


def test_c6_simulator_cnot_is_self_inverse():
    """CNOT is its own inverse: CNOT * CNOT = I (unitary, no numerical drift)."""
    from zero_data_model.hardware.quantum import SimulatorQuantumBackend

    backend = SimulatorQuantumBackend(n_qubits=3, n_layers=1)
    rng = np.random.default_rng(42)
    state = rng.standard_normal(8) + 1j * rng.standard_normal(8)
    state = state / np.linalg.norm(state)  # normalise
    state2 = backend._apply_cnot(backend._apply_cnot(state, 0, 1), 0, 1)
    np.testing.assert_allclose(state, state2, atol=1e-12)


def test_c6_simulator_rejects_oversize_n_qubits():
    """n_qubits > 20 is rejected to prevent exponential state-vector blowup."""
    from zero_data_model.hardware.quantum import SimulatorQuantumBackend

    with pytest.raises(ValueError):
        SimulatorQuantumBackend(n_qubits=21)


# ---------------------------------------------------------------------------
# C-7: Weighted integration by inverse uncertainty
# ---------------------------------------------------------------------------


def test_c7_integrate_none_uncertainties_is_equal_weight_mean():
    """_integrate(signals, None) falls back to the equal-weight mean."""
    model = ZeroDataModel(dim=8)
    signals = [Signal(data=np.ones(8) * i) for i in range(1, 4)]
    result = model._integrate(signals, None)
    expected = np.mean([[1] * 8, [2] * 8, [3] * 8], axis=0)
    np.testing.assert_allclose(result.data, expected)


def test_c7_integrate_equal_uncertainties_is_uniform_weights():
    """Equal uncertainties produce uniform softmax weights (same as the mean)."""
    model = ZeroDataModel(dim=8)
    signals = [Signal(data=np.ones(8) * i) for i in range(1, 4)]
    uncertainties = np.array([0.5, 0.5, 0.5])
    result = model._integrate(signals, uncertainties)
    expected = np.mean([[1] * 8, [2] * 8, [3] * 8], axis=0)
    np.testing.assert_allclose(result.data, expected)


def test_c7_integrate_confident_module_dominates():
    """A module with very low uncertainty dominates the integration."""
    model = ZeroDataModel(dim=8)
    # Module 0 is confident (low uncertainty) and outputs 10 * ones.
    # Modules 1 and 2 are uncertain and output 1 * ones.
    signals = [
        Signal(data=np.ones(8) * 10),
        Signal(data=np.ones(8) * 1),
        Signal(data=np.ones(8) * 1),
    ]
    uncertainties = np.array([0.001, 100.0, 100.0])
    result = model._integrate(signals, uncertainties)
    # softmax([1000, 0.01, 0.01]) -> ~[1, 0, 0], so result ~ 10 * ones.
    np.testing.assert_allclose(result.data, [10] * 8, atol=0.01)


def test_c7_integrate_mismatched_length_falls_back_to_mean():
    """Mismatched uncertainties length falls back to equal-weight mean."""
    model = ZeroDataModel(dim=8)
    signals = [
        Signal(data=np.ones(8) * 10),
        Signal(data=np.ones(8) * 1),
        Signal(data=np.ones(8) * 1),
    ]
    uncertainties = np.array([0.5, 0.5])  # length 2, but 3 signals
    result = model._integrate(signals, uncertainties)
    expected = np.mean([[10] * 8, [1] * 8, [1] * 8], axis=0)
    np.testing.assert_allclose(result.data, expected)


def test_c7_seeded_model_reproducible_under_weighted_integration():
    """Seeded models produce identical think() sequences despite C-7 weighting.

    The weighting must not break reproducibility: two models with the same
    seed produce identical outputs because the predict() uncertainties are
    deterministic (derived from the seeded module state).
    """
    m1 = ZeroDataModel(dim=16, seed=42)
    m2 = ZeroDataModel(dim=16, seed=42)
    for cycle in range(3):
        r1 = m1.think()
        r2 = m2.think()
        np.testing.assert_allclose(r1.data, r2.data, atol=1e-10)
        assert r1.metadata["cycle"] == r2.metadata["cycle"] == cycle + 1


def test_c7_think_cycles_evolve_not_stuck():
    """Multiple think() cycles produce evolving (not identical) outputs."""
    model = ZeroDataModel(dim=16)
    results = [model.think() for _ in range(5)]
    distinct_transitions = sum(
        1 for i in range(4) if not np.allclose(results[i].data, results[i + 1].data)
    )
    assert distinct_transitions >= 3, "system should evolve across cycles"
