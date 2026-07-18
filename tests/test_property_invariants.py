# tests/test_property_invariants.py
"""Property-based invariant tests (Round-3 audit C-batch).

Uses Hypothesis to verify HIGH-priority mathematical invariants hold across
the full input space, not just the hand-picked examples in the unit tests.
Each test below corresponds to a HIGH-severity finding from the Round-3
audit and guards the corresponding fix against silent regression.

Invariants covered (11):
  P1   Householder reflection is orthogonal: T @ T.T == I
  P2   Householder reflection is an involution: T @ T == I
  P3   Householder maps source direction to target direction
  P4   Quantum Born rule: probabilities sum to 1
  P5   Quantum Born rule: probabilities are non-negative
  P6   Quantum output dimension: len(p) == 2**n_qubits
  P7   Gray-Scott morphogen u stays in [0, 1] after step
  P8   Gray-Scott morphogen v stays in [0, 1] after step
  P9   Cellular automaton state is binary {0, 1}
  P10  Vietoris-Rips betti_0 >= 1 for any non-empty point cloud
  P11  KL(N(b, sigma^2 I) || N(0, I)) >= 0 (variational free energy)
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from zero_data_model.biological import (
    CellularAutomata,
    MorphogeneticField,
)
from zero_data_model.category_engine import CategoryTheoryEngine
from zero_data_model.hardware.quantum import SimulatorQuantumBackend
from zero_data_model.math_universe import MathematicalUniverse

# --------------------------------------------------------------------------- #
# Shared strategies
# --------------------------------------------------------------------------- #

# Finite, non-NaN, non-Inf floats in a sane magnitude range. Most module
# maths break down on Inf/NaN; Hypothesis would otherwise find those
# counter-examples, but they are out of scope (the audit's NaN guards in
# the A-batch handle them).
_FINITE_FLOAT = st.floats(
    min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False
)


def _finite_array(shape: int | tuple[int, ...]) -> st.SearchStrategy[np.ndarray]:
    """Strategy returning a finite float64 numpy array of the given shape."""
    return arrays(dtype=np.float64, shape=shape, elements=_FINITE_FLOAT)


# --------------------------------------------------------------------------- #
# P1, P2, P3: Householder reflection invariants (category_engine)
# --------------------------------------------------------------------------- #


@given(
    source=_finite_array((8,)),
    target=_finite_array((8,)),
)
@settings(max_examples=50, deadline=None)
def test_p1_householder_reflection_is_orthogonal(source, target):
    """P1: ``T = find_invertible_map(s, t)`` is a SCALED orthogonal map.

    Round-9 audit R9-003: the map is now ``T = (nb/na) * (I - 2vv^T)``
    so the documented contract ``T @ source = target`` holds exactly
    (not just up to a norm ratio). The reflection component
    ``(I - 2vv^T)`` is orthogonal, so the scaled map satisfies
    ``T @ T.T = (nb/na)^2 * I`` -- still invertible (non-zero scalar
    times an orthogonal matrix is non-singular).
    """
    engine = CategoryTheoryEngine(dim=8, rng=np.random.default_rng(0))
    T = engine.find_invertible_map(source, target)
    if T is None:
        # Degenerate inputs (zero norm) -- invariant is vacuously satisfied.
        return
    n = T.shape[0]
    assert T.shape == (n, n)
    na = float(np.linalg.norm(source))
    nb = float(np.linalg.norm(target))
    # Scaled orthogonality: T @ T.T == (nb/na)^2 * I.
    scale_sq = (nb / na) ** 2
    np.testing.assert_allclose(T @ T.T, scale_sq * np.eye(n), atol=1e-10, rtol=1e-10)


@given(
    source=_finite_array((8,)),
    target=_finite_array((8,)),
)
@settings(max_examples=50, deadline=None)
def test_p2_householder_reflection_is_involution(source, target):
    """P2: the SCALED Householder map is invertible (full rank).

    Round-9 audit R9-003: previously the map was a pure reflection
    (an involution: ``T @ T == I``). After scaling by ``nb/na``, the
    map is no longer an involution but is still invertible (a non-zero
    scalar times an orthogonal matrix is non-singular). We assert
    full rank, which is the property ``transfer_solution`` relies on.
    """
    engine = CategoryTheoryEngine(dim=8, rng=np.random.default_rng(0))
    T = engine.find_invertible_map(source, target)
    if T is None:
        return
    n = T.shape[0]
    assert np.linalg.matrix_rank(T) == n, (
        f"scaled Householder map is rank-deficient (rank={np.linalg.matrix_rank(T)}, "
        f"n={n}) -- the map is no longer invertible."
    )


@given(
    source=_finite_array((8,)),
    target=_finite_array((8,)),
)
@settings(max_examples=50, deadline=None)
def test_p3_householder_maps_source_to_target_direction(source, target):
    """P3: ``T @ source == target`` (the documented contract).

    Round-9 audit R9-003: previously the map only matched unit vectors
    (``T @ s_hat == t_hat``). The map is now scaled by ``nb/na`` so
    ``T @ source = (nb/na) * (|source| * t_hat) = |target| * t_hat = target``
    exactly. If this invariant breaks, ``transfer_solution`` silently
    produces wrong-domain answers.
    """
    engine = CategoryTheoryEngine(dim=8, rng=np.random.default_rng(0))
    T = engine.find_invertible_map(source, target)
    if T is None:
        return
    na = float(np.linalg.norm(source))
    nb = float(np.linalg.norm(target))
    if na < 1e-12 or nb < 1e-12:
        return
    # Documented contract: T @ source == target (with magnitude).
    mapped = T @ source
    np.testing.assert_allclose(mapped, target, atol=1e-10, rtol=1e-10)


# --------------------------------------------------------------------------- #
# P4, P5, P6: Quantum Born-rule invariants (SimulatorQuantumBackend)
# --------------------------------------------------------------------------- #


# Small n_qubits to keep each test fast; the invariant is independent of size.
_QUBITS = st.integers(min_value=1, max_value=4)
_LAYERS = st.integers(min_value=1, max_value=3)


@given(
    n_qubits=_QUBITS,
    n_layers=_LAYERS,
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=30, deadline=None)
def test_p4_quantum_probabilities_sum_to_one(n_qubits, n_layers, seed):
    """P4: ``sum(evolve_and_measure(params, entangling)) ≈ 1.0``.

    The Born rule ``p(k) = |amplitude_k|^2`` always yields a valid
    probability distribution. A drift away from 1 indicates the state
    vector was not properly normalised (a common bug in hand-rolled
    simulators).
    """
    rng = np.random.default_rng(seed)
    backend = SimulatorQuantumBackend(n_qubits=n_qubits, n_layers=n_layers)
    params = rng.standard_normal((n_layers, n_qubits, 2)) * 0.1
    entangling = rng.standard_normal((n_qubits, n_qubits)) * 0.05
    probs = backend.evolve_and_measure(params, entangling, n_shots=512)
    total = float(np.sum(probs))
    # Allow small numerical drift from the explicit-Euler state evolution.
    assert abs(total - 1.0) < 1e-6, f"probabilities sum to {total}, not 1.0"


@given(
    n_qubits=_QUBITS,
    n_layers=_LAYERS,
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=30, deadline=None)
def test_p5_quantum_probabilities_non_negative(n_qubits, n_layers, seed):
    """P5: ``all(p >= -tol)`` for the Born-rule output.

    Probabilities are squared magnitudes, so they cannot be negative
    (modulo float rounding). Negative entries indicate a bug in the
    ``|amplitude|^2`` computation.
    """
    rng = np.random.default_rng(seed)
    backend = SimulatorQuantumBackend(n_qubits=n_qubits, n_layers=n_layers)
    params = rng.standard_normal((n_layers, n_qubits, 2)) * 0.1
    entangling = rng.standard_normal((n_qubits, n_qubits)) * 0.05
    probs = backend.evolve_and_measure(params, entangling, n_shots=512)
    assert float(np.min(probs)) >= -1e-10, (
        f"min probability {float(np.min(probs))} is negative"
    )


@given(n_qubits=_QUBITS, n_layers=_LAYERS)
@settings(max_examples=20, deadline=None)
def test_p6_quantum_output_dimension_is_2_pow_n_qubits(n_qubits, n_layers):
    """P6: ``len(evolve_and_measure(...)) == 2**n_qubits``.

    CRIT-2 (Round-3 audit): the IBM backend previously returned
    ``2*n_qubits`` entries (a per-qubit marginal), dimensionally
    inconsistent with the simulator's ``2**n_qubits`` basis-state vector.
    This invariant guards the dimension contract across all backends.
    """
    rng = np.random.default_rng(0)
    backend = SimulatorQuantumBackend(n_qubits=n_qubits, n_layers=n_layers)
    params = rng.standard_normal((n_layers, n_qubits, 2)) * 0.1
    entangling = rng.standard_normal((n_qubits, n_qubits)) * 0.05
    probs = backend.evolve_and_measure(params, entangling, n_shots=256)
    assert probs.shape == (1 << n_qubits,), (
        f"expected shape ({1 << n_qubits},), got {probs.shape}"
    )


# --------------------------------------------------------------------------- #
# P7, P8: Gray-Scott reaction-diffusion bounds (MorphogeneticField)
# --------------------------------------------------------------------------- #


_GRID_SIZE = st.integers(min_value=4, max_value=24)
_N_STEPS = st.integers(min_value=1, max_value=20)


@given(grid_size=_GRID_SIZE, n_steps=_N_STEPS, seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=30, deadline=None)
def test_p7_gray_scott_morphogen_u_in_unit_interval(grid_size, n_steps, seed):
    """P7: After ``step``, Gray-Scott activator ``u`` stays in ``[0, 1]``.

    The explicit-Euler update clips ``u`` to ``[0, 1]`` for stability
    (the Gray-Scott model is only well-defined on that domain). A drift
    outside indicates the clip was removed or bypassed.
    """
    mf = MorphogeneticField(grid_size=grid_size, rng=np.random.default_rng(seed))
    for _ in range(n_steps):
        mf.step()
    u = mf.morphogens[0]
    assert float(u.min()) >= -1e-10, f"u min {float(u.min())} < 0"
    assert float(u.max()) <= 1.0 + 1e-10, f"u max {float(u.max())} > 1"


@given(grid_size=_GRID_SIZE, n_steps=_N_STEPS, seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=30, deadline=None)
def test_p8_gray_scott_morphogen_v_in_unit_interval(grid_size, n_steps, seed):
    """P8: After ``step``, Gray-Scott inhibitor ``v`` stays in ``[0, 1]``.

    Same as P7 for the second morphogen. Both must stay bounded for the
    reaction-diffusion system to remain physically meaningful.
    """
    mf = MorphogeneticField(grid_size=grid_size, rng=np.random.default_rng(seed))
    for _ in range(n_steps):
        mf.step()
    v = mf.morphogens[1]
    assert float(v.min()) >= -1e-10, f"v min {float(v.min())} < 0"
    assert float(v.max()) <= 1.0 + 1e-10, f"v max {float(v.max())} > 1"


# --------------------------------------------------------------------------- #
# P9: Cellular automaton binary state
# --------------------------------------------------------------------------- #


_RULE = st.integers(min_value=0, max_value=255)
_SIZE = st.integers(min_value=8, max_value=128)


@given(rule=_RULE, size=_SIZE, n_steps=_N_STEPS)
@settings(max_examples=40, deadline=None)
def test_p9_cellular_automaton_state_is_binary(rule, size, n_steps):
    """P9: `` CellularAutomata.state`` only contains ``{0, 1}`` after any number of steps.

    A Wolfram elementary CA operates on binary cells; any value outside
    ``{0, 1}`` corrupts the rule-table lookup ``(rule >> index) & 1``.
    """
    ca = CellularAutomata(size=size, rule=rule, rng=np.random.default_rng(0))
    for _ in range(n_steps):
        ca.step()
    state = ca.state
    assert set(np.unique(state)).issubset({0, 1}), (
        f"state has non-binary values: {set(np.unique(state))}"
    )


# --------------------------------------------------------------------------- #
# P10: Vietoris-Rips betti_0 >= 1 (MathematicalUniverse)
# --------------------------------------------------------------------------- #


_N_POINTS = st.integers(min_value=1, max_value=10)
_DIM = st.integers(min_value=2, max_value=4)


@given(
    n_points=_N_POINTS,
    dim=_DIM,
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=40, deadline=None)
def test_p10_vietoris_rips_betti0_at_least_one_component(n_points, dim, seed):
    """P10: For any non-empty point cloud, ``betti_0 >= 1``.

    The 0-th Betti number counts connected components; a non-empty set
    always has at least one. ``betti_0 == 0`` would be a serious bug in
    the simplex-tree construction.
    """
    rng = np.random.default_rng(seed)
    mu = MathematicalUniverse(dim=8, rng=rng)
    points = rng.standard_normal((n_points, dim))
    betti = mu.topology.vietoris_rips_betti(points, max_radius=10.0)
    assert betti[0] >= 1, f"betti_0 = {betti[0]} < 1 for {n_points} points"


# --------------------------------------------------------------------------- #
# P11: KL divergence non-negative (ActiveInferenceEngine.compute_free_energy)
# --------------------------------------------------------------------------- #


_OBS_DIM = st.integers(min_value=4, max_value=16)


@given(
    obs_dim=_OBS_DIM,
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=40, deadline=None)
def test_p11_kl_divergence_non_negative(obs_dim, seed):
    """P11: ``KL(N(b, sigma^2 I) || N(0, I)) >= 0`` (variational free energy).

    The KL divergence between any two distributions is non-negative
    (Gibbs' inequality). The active-inference free energy uses this term
    as the complexity penalty; a negative KL would mean the model is
    "better than the prior" in a way that violates information theory.
    """
    from zero_data_model.active_inference import ActiveInferenceEngine

    rng = np.random.default_rng(seed)
    # state_dim must match obs_dim for the diagonal KL formula in
    # ``compute_free_energy`` to align with the engine's internal dims.
    engine = ActiveInferenceEngine(
        state_dim=obs_dim, obs_dim=obs_dim, action_dim=obs_dim // 2, rng=rng
    )
    # Drive the engine through a few cycles so action_history populates
    # (this gives sigma_q^2 a non-trivial value to test the general case).
    for _ in range(5):
        obs = rng.standard_normal(obs_dim)
        engine.generative_model.update_belief(obs)
        action = engine.select_action(engine.generative_model.belief_state)
        engine.action_history.append(action)
    # Now compute free energy on a fresh observation.
    obs = rng.standard_normal(obs_dim)
    fe = engine.compute_free_energy(obs)
    assert np.isfinite(fe), f"free energy {fe} is not finite"
    # The pragmatic prediction-error term is always >= 0, and the KL term
    # is >= 0 by Gibbs' inequality, so the sum is >= 0.
    assert fe >= -1e-9, f"free energy {fe} < 0 (KL divergence must be >= 0)"
