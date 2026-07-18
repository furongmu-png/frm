# Causal Emergence Engine — Design Specification

> **Status:** Approved (2026-07-19)
> **Author:** Agent
> **Implementation plan:** follow-up document in `docs/superpowers/plans/`

---

## 1. Motivation and Axioms

This specification defines a **Causal Emergence Engine** (CEE): a self-contained, zero-data reasoning system that composes persistent homology, causal discovery, variational PDE solving, Hamiltonian Monte Carlo, and chaotic-attractor memory into a single recursive loop. The design is grounded in three axioms (per the originating brief):

- **Axiom I (Physical Realism):** The world's substance is action and symmetry, not pixels or words. The system's primary task is to compute conserved quantities, not to classify or predict next tokens.
- **Axiom II (Topology First):** Semantic information lives in topological invariants (connected components, loops, voids). Deformations, noise, and modality changes do not alter topological essence.
- **Axiom III (Causality over Correlation):** True intelligence must answer counterfactual queries ("what if?") via structural causal models, not merely output probability distributions.

The engine is **zero-data**: no pretrained weights, no external datasets, no heavy ML dependencies (gudhi / pymc / causal-learn / ripser are explicitly excluded). All algorithms are implemented in pure `numpy` + `scipy` + Python stdlib, consistent with the existing `zero_data_model` codebase philosophy.

---

## 2. Architecture Overview

### 2.1 Package Location

```
src/zero_data_model/causal_emergence/
├── __init__.py            # Public exports + CausalEmergenceEngine re-export
├── rules.py               # EmergenceRules dataclass (configurable priors)
├── topology.py            # Module A: PersistentHomologyPerceiver
├── causal_discovery.py    # Module B: CausalInferenceEngine (PC + LiNGAM + do-calculus)
├── differential.py       # Module C: DifferentialGenerator (Euler-Lagrange PDE solver)
├── hmc.py                 # Module D: HamiltonianSampler (leapfrog HMC)
├── chaotic_memory.py      # Module E: ChaoticAssociativeMemory (Lorenz attractor basins)
└── engine.py              # CausalEmergenceEngine (recursive-loop orchestrator)
```

### 2.2 Module Pattern

Each module is a **plain Python class** (not a `CognitiveModule` subclass, not in `self.modules`, not counted in `_N_COGNITIVE_MODULES`). This matches the existing `capabilities/` pattern:

- Constructor: `(dim=64, <core_module>=None, rules=None)`. Core-module arguments default to `None`; each constructor lazily instantiates a fallback (`if x is None: x = CoreModule(...)`).
- Methods return plain dicts / ndarrays; all numeric outputs are `np.nan_to_num`-guarded.
- L2-normalized vector outputs where applicable.
- Determinism: every class accepts an optional `rng: np.random.Generator | None`.

### 2.3 Integration with ZeroDataModel

The engine is wired into `ZeroDataModel` as a single instance attribute + facade methods:

```python
# in ZeroDataModel.__init__
self.emergence_rules = EmergenceRules()
self.emergence = CausalEmergenceEngine(
    dim=dim,
    active_inference=self.active_inference,
    math_universe=self.math_universe,
    rules=self.emergence_rules,
    rng=_child_rngs[N],
)

# facade methods
def perceive_topology(self, data: np.ndarray) -> dict: ...
def discover_causal_dynamics(self, data: np.ndarray, var_names=None) -> dict: ...
def generate_trajectory(self, boundary: dict) -> dict: ...
def sample_posterior(self, log_prob_fn, initial_position, n_samples=100, **kwargs) -> dict: ...
def recall_memory(self, query: np.ndarray) -> dict: ...
def emergence_cycle(self, observation: np.ndarray) -> dict: ...
```

The engine instance is **not** added to `self.modules` and does not change `_N_COGNITIVE_MODULES`. This preserves the existing cognitive-module count contract.

---

## 3. Module A: PersistentHomologyPerceiver (`topology.py`)

### 3.1 Purpose

Perceive arbitrary high-dimensional data as a topological point cloud and compute its persistent homology. Output is invariant under rotation, translation, and non-degenerate deformation.

### 3.2 Algorithm

1. **Input shaping:** flatten input to a 2D point cloud `(n_points, n_features)`. If input is 1D, treat as a 1-point cloud (degenerate case: returns trivial invariants).
2. **Pairwise distance matrix:** Euclidean `D[i, j] = ||x_i - x_j||`. Symmetric, zero diagonal.
3. **Vietoris-Rips filtration:** for each threshold `eps` in increasing order, build a simplicial complex where simplices appear when all pairwise distances ≤ `eps`.
4. **Boundary matrix reduction:** standard left-to-right column reduction over GF(2) (for low-dimensional betti numbers, this is tractable for n ≤ 32 points). For larger inputs, fall back to the existing `hardware/kernels._betti_numbers` JIT kernel if available, else cap n_points at 64 by random subsampling (deterministic with the rng).
5. **Persistence diagram:** for each homology dimension `d` (0, 1, 2), record `(birth, death)` pairs where features appear and disappear.
6. **Persistence entropy:** `H = -sum(p_i * log(p_i))` where `p_i = (death_i - birth_i) / total_persistence`.
7. **Euler characteristic:** `chi = sum_d (-1)^d * betti_d`.

### 3.3 API

```python
class PersistentHomologyPerceiver:
    def __init__(
        self,
        dim: int = 64,
        math_universe=None,
        rules: EmergenceRules | None = None,
        rng: np.random.Generator | None = None,
    ): ...

    def perceive(self, data: np.ndarray, max_dim: int = 2) -> dict:
        """Compute persistent homology of the input point cloud.

        Returns:
            {
                'betti_numbers': list[int],      # [betti_0, betti_1, betti_2]
                'persistence_diagram': list[tuple],  # [(dim, birth, death), ...]
                'persistence_entropy': float,
                'euler_characteristic': int,
                'n_points': int,
                'max_eps': float,
            }
        """
```

### 3.4 Edge Cases

- Empty input: return `{betti_numbers: [0, 0, 0], persistence_diagram: [], persistence_entropy: 0.0, euler_characteristic: 0, n_points: 0, max_eps: 0.0}`.
- Single point: `betti_numbers = [1, 0, 0]`, empty persistence diagram, entropy 0.
- `NaN`/`Inf` in input: `np.nan_to_num` guard before distance computation.
- Large input (> 64 points): deterministic subsampling via `rng.choice`.

---

## 4. Module B: CausalInferenceEngine (`causal_discovery.py`)

### 4.1 Purpose

Discover a directed acyclic graph (DAG) from multivariate observational data and answer interventional / counterfactual queries.

### 4.2 Algorithm

Three discovery methods, selected by `rules.causal_method`:

1. **PC algorithm (default):** Start with complete undirected graph; for each pair `(i, j)`, remove edge if `i ⊥ j | S` for some conditioning set `S` (partial correlation test against a significance threshold `causal_significance`). Orient edges using collider detection (v-structures) and acyclicity constraints. Deterministic ordering of edge tests for reproducibility.

2. **LiNGAM (non-Gaussian ICA):** Run FastICA on the data to obtain independent components; the mixing matrix `W` reveals causal directions (non-zero entries in `W` indicate causal edges). Convert `W` to a lower-triangular DAG after permutation.

3. **Correlation fallback:** when n_samples < 3 or the data is degenerate (zero variance in any column), fall back to the existing `CausalGraphBuilder` correlation-threshold method.

### 4.3 Do-Calculus

Given the discovered DAG, `intervene(var, value)` estimates the intervention effect:
- Set `data[:, var] = value` (mutilation).
- Recompute marginal means.
- Effect = post - pre.

Counterfactual: for a single observation, compute the posterior mean under intervention via belief propagation on the DAG.

### 4.4 API

```python
class CausalInferenceEngine:
    def __init__(
        self,
        dim: int = 64,
        active_inference=None,
        rules: EmergenceRules | None = None,
        rng: np.random.Generator | None = None,
    ): ...

    def discover(
        self,
        data: np.ndarray,
        var_names: list[str] | None = None,
        method: str | None = None,  # 'pc' | 'lingam' | 'correlation'; default rules.causal_method
    ) -> dict:
        """Discover causal DAG from observational data.

        Returns:
            {
                'adjacency': np.ndarray,        # (n_vars, n_vars) binary
                'edges': list[tuple[int, int]],
                'method': str,
                'var_names': list[str],
                'n_edges': int,
                'is_acyclic': bool,
            }
        """

    def intervene(
        self,
        adjacency: np.ndarray,
        data: np.ndarray,
        intervention_var: int,
        intervention_value: float,
    ) -> dict:
        """Estimate the effect of do(X[intervention_var] = value)."""

    def counterfactual(
        self,
        adjacency: np.ndarray,
        observed: np.ndarray,
        intervention_var: int,
        intervention_value: float,
    ) -> dict:
        """Counterfactual: 'what would have happened if X[var] had been value?'."""
```

### 4.5 Edge Cases

- `n_samples < 3`: fall back to correlation method.
- Constant column: skip the column in causal tests.
- Non-convergent ICA: fall back to correlation method.
- Cyclic adjacency from LiNGAM (rare): prune edges to break cycles greedily.

---

## 5. Module C: DifferentialGenerator (`differential.py`)

### 5.1 Purpose

Generate continuous, physically-realistic trajectories by solving a discretized Euler-Lagrange boundary value problem.

### 5.2 Algorithm

Given boundary conditions `(start_state, end_state, n_steps)`:

1. Initialize a linear interpolation trajectory `q(t)` between start and end.
2. Define a Lagrangian `L(q, q_dot) = T(q_dot) - V(q)` where:
   - `T(q_dot) = 0.5 * ||q_dot||^2` (kinetic energy)
   - `V(q) = 0.5 * ||q - target||^2` (quadratic potential toward target)
3. Discretize on a uniform grid `t = 0, 1, ..., n_steps` with `dt = 1 / n_steps`.
4. Solve the discretized Euler-Lagrange equation:
   `d/dt(dL/dq_dot) - dL/dq = 0`
   In discretized form:
   `(q[k+1] - 2*q[k] + q[k-1]) / dt^2 = -dV/dq = -(q[k] - target)`
5. Iterate via Gauss-Seidel relaxation until `max ||delta|| < tol` or `max_iter` reached.
6. Compute per-step Lagrangian, total action `S = sum(L * dt)`.

### 5.3 API

```python
class DifferentialGenerator:
    def __init__(
        self,
        dim: int = 64,
        rules: EmergenceRules | None = None,
        rng: np.random.Generator | None = None,
    ): ...

    def generate(
        self,
        start_state: np.ndarray,
        end_state: np.ndarray,
        n_steps: int = 32,
        constraints: dict | None = None,  # optional obstacle avoidance
    ) -> dict:
        """Solve the boundary value problem.

        Returns:
            {
                'trajectory': np.ndarray,       # (n_steps+1, dim)
                'lagrangian': np.ndarray,        # (n_steps,) per-step
                'action': float,
                'converged': bool,
                'iterations': int,
            }
        """
```

### 5.4 Edge Cases

- `start == end`: return constant trajectory.
- `n_steps == 0`: return single-point trajectory.
- Non-convergence within `max_iter`: return the last iterate with `converged=False`.
- NaN guard on all outputs.

---

## 6. Module D: HamiltonianSampler (`hmc.py`)

### 6.1 Purpose

Sample from a posterior distribution using Hamiltonian Monte Carlo with leapfrog integration. Provides calibrated uncertainty estimates.

### 6.2 Algorithm

1. **Inputs:** `log_prob_fn(position) -> float` (the log posterior), `initial_position`, `n_samples`, `step_size`, `n_leapfrog`.
2. **Potential energy:** `U(q) = -log_prob_fn(q)`. Gradient via finite differences (or analytic gradient if provided).
3. **Momentum resampling:** `p ~ N(0, M)` where `M` is the identity mass matrix.
4. **Leapfrog integration:**
   ```
   p_half = p - (step_size / 2) * grad_U(q)
   q_new = q + step_size * p_half
   p_new = p_half - (step_size / 2) * grad_U(q_new)
   ```
   Repeat for `n_leapfrog` steps.
5. **Metropolis accept/reject:** accept with probability `min(1, exp(H_old - H_new))` where `H = U + 0.5 * ||p||^2`.
6. Repeat for `n_samples` to collect samples.
7. Compute mean, std, ESS (effective sample size), and `accept_rate`.

### 6.3 API

```python
class HamiltonianSampler:
    def __init__(
        self,
        dim: int = 64,
        rules: EmergenceRules | None = None,
        rng: np.random.Generator | None = None,
    ): ...

    def sample(
        self,
        log_prob_fn,                       # callable: np.ndarray -> float
        initial_position: np.ndarray,
        n_samples: int = 100,
        step_size: float = 0.1,
        n_leapfrog: int = 10,
        grad_fn=None,                      # optional analytic gradient
    ) -> dict:
        """Sample from the posterior defined by log_prob_fn.

        Returns:
            {
                'samples': np.ndarray,     # (n_samples, dim)
                'mean': np.ndarray,
                'std': np.ndarray,
                'accept_rate': float,
                'ess': float,             # effective sample size
                'converged': bool,         # True if accept_rate in [0.2, 0.9]
            }
        """
```

### 6.4 Edge Cases

- `log_prob_fn` returns `-inf` or NaN: reject the proposal.
- `initial_position` non-finite: raise `ValueError` at the boundary.
- `n_samples == 0`: return empty arrays.
- Accept rate < 0.2 or > 0.9: flag `converged=False` (tuning recommended).

---

## 7. Module E: ChaoticAssociativeMemory (`chaotic_memory.py`)

### 7.1 Purpose

Store patterns as stable attractor basins of a Lorenz-like dynamical system. Recall converges to the nearest stored pattern; queries on basin boundaries produce chaotic wandering (defined as "insight" or "emergence").

### 7.2 Algorithm

1. **Storage:** each pattern is encoded as a target equilibrium point of a Lorenz-like system:
   ```
   dx/dt = sigma * (y - x)
   dy/dt = x * (rho - z) - y - alpha * (y - target_y)
   dz/dt = x * y - beta * z - alpha * (z - target_z)
   ```
   where `(target_y, target_z)` are derived from the pattern. The `alpha` term pulls the trajectory toward the stored pattern's basin.
2. **Recall:** integrate the Lorenz system with the query as initial condition for `n_steps` using RK4. After convergence (or `n_steps`), find the nearest stored pattern by L2 distance.
3. **Emergence detection:** if the trajectory's Lyapunov exponent stays positive (chaotic wandering) for the full `n_steps` without converging, flag `emerged=True`.
4. **Capacity:** no theoretical limit (each pattern is an independent attractor), but practical memory is bounded by `rules.chaotic_memory_capacity` (default 32).

### 7.3 API

```python
class ChaoticAssociativeMemory:
    def __init__(
        self,
        dim: int = 64,
        rules: EmergenceRules | None = None,
        rng: np.random.Generator | None = None,
    ): ...

    def store(self, pattern: np.ndarray, label: int | str) -> dict:
        """Store a pattern as a new attractor basin.

        Returns: {'label': ..., 'n_stored': int, 'capacity': int}
        """

    def recall(self, query: np.ndarray, n_steps: int = 100) -> dict:
        """Recall the nearest stored pattern.

        Returns:
            {
                'label': int | str | None,
                'similarity': float,
                'emerged': bool,
                'trajectory': np.ndarray,    # (n_steps, 3) Lorenz state
                'converged': bool,
            }
        """

    def clear(self) -> None:
        """Erase all stored patterns."""
```

### 7.4 Edge Cases

- Empty memory: `recall` returns `label=None, similarity=0.0, emerged=True` (chaotic wandering with no basin).
- Pattern capacity exceeded: oldest pattern is evicted (FIFO).
- Query is NaN: return `label=None, emerged=False, converged=False`.

---

## 8. CausalEmergenceEngine (`engine.py`)

### 8.1 Purpose

Orchestrate the five modules into the recursive loop described in the brief: perception → causal anchoring → counterfactual simulation → uncertainty assessment → memory interaction → emergence.

### 8.2 API

```python
class CausalEmergenceEngine:
    def __init__(
        self,
        dim: int = 64,
        active_inference=None,
        math_universe=None,
        rules: EmergenceRules | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.topology = PersistentHomologyPerceiver(dim=dim, ...)
        self.causal = CausalInferenceEngine(dim=dim, ...)
        self.differential = DifferentialGenerator(dim=dim, ...)
        self.hmc = HamiltonianSampler(dim=dim, ...)
        self.memory = ChaoticAssociativeMemory(dim=dim, ...)
        # ... store core modules

    def perceive_topology(self, data: np.ndarray) -> dict:
        """Module A facade."""

    def discover_causal_dynamics(self, data: np.ndarray, var_names=None) -> dict:
        """Module B facade."""

    def generate_trajectory(self, boundary: dict) -> dict:
        """Module C facade."""

    def sample_posterior(self, log_prob_fn, initial_position, **kwargs) -> dict:
        """Module D facade."""

    def recall_memory(self, query: np.ndarray) -> dict:
        """Module E facade."""

    def emergence_cycle(self, observation: np.ndarray) -> dict:
        """Run the full recursive loop on an observation.

        Steps:
        1. perception = perceive_topology(observation)
        2. causal_graph = discover_causal_dynamics(observation)
        3. counterfactual = generate_trajectory({
               'start_state': observation,
               'end_state': observation + delta,  # perturbation
           })
        4. posterior = sample_posterior(
               log_prob_fn=lambda x: -0.5 * ||x - observation||^2,
               initial_position=observation,
               n_samples=rules.hmc_samples,
           )
        5. memory_response = recall_memory(observation)
        6. emergence_score = compute_emergence_score(
               perception, causal_graph, counterfactual, posterior, memory_response
           )

        Returns: {
            'perception': dict,
            'causal_graph': dict,
            'counterfactual': dict,
            'posterior': dict,
            'memory': dict,
            'emergence_score': float,    # in [0, 1]
        }
        """

    def compute_emergence_score(self, *outputs) -> float:
        """Heuristic emergence score: high persistence entropy +
        high causal graph density + high memory emergence + low
        posterior variance -> high emergence score."""
```

### 8.3 Emergence Score

The emergence score is a heuristic in `[0, 1]` combining:
- `0.3 * persistence_entropy` (topological complexity)
- `0.2 * (n_edges / max_edges)` (causal graph density)
- `0.2 * memory.emerged` (1.0 if chaotic wandering occurred)
- `0.15 * (1 - posterior.std / dim)` (low uncertainty → high score)
- `0.15 * (counterfactual.action / reference_action)` (action magnitude)

All terms are normalized to `[0, 1]` before combining.

---

## 9. EmergenceRules (`rules.py`)

```python
@dataclass
class EmergenceRules(DomainRules):
    """Priors for the causal emergence engine."""

    # Module A: topology
    topology_max_points: int = 64          # cap on point cloud size
    topology_max_dim: int = 2              # max homology dimension
    topology_eps_steps: int = 50          # filtration resolution

    # Module B: causal discovery
    causal_method: str = "pc"              # 'pc' | 'lingam' | 'correlation'
    causal_significance: float = 0.05      # independence test threshold
    causal_max_cond_set: int = 3          # max conditioning set size in PC

    # Module C: differential
    differential_max_iter: int = 100       # Gauss-Seidel iterations
    differential_tol: float = 1e-6         # convergence tolerance
    differential_dt: float = 0.01          # time step

    # Module D: HMC
    hmc_step_size: float = 0.1
    hmc_n_leapfrog: int = 10
    hmc_samples: int = 100
    hmc_target_accept: float = 0.65        # optimal per Beskos et al.

    # Module E: chaotic memory
    chaotic_memory_capacity: int = 32
    chaotic_lorenz_sigma: float = 10.0
    chaotic_lorenz_rho: float = 28.0
    chaotic_lorenz_beta: float = 8.0 / 3.0
    chaotic_lyapunov_threshold: float = 0.9  # emergence detection

    # Engine
    emergence_cycle_perturbation: float = 0.1  # delta for counterfactual
    seed: int | None = None
```

---

## 10. Testing Strategy

### 10.1 Per-module tests

For each module, a `tests/test_causal_emergence_<module>.py` file:

- **Correctness:** known-answer tests (e.g., betti numbers of a circle = [1, 1, 0]; HMC recovers the mean of a Gaussian).
- **Edge cases:** empty input, single point, NaN, degenerate (zero variance), high-dim.
- **Determinism:** seeded runs produce identical outputs.
- **API contract:** return dict has the expected keys.

### 10.2 Engine integration tests

`tests/test_causal_emergence_engine.py`:
- Full `emergence_cycle` on a synthetic observation.
- Each facade method delegates correctly to the underlying module.
- Thread safety: concurrent `emergence_cycle` calls under `self._lock`.

### 10.3 Test files

```
tests/
├── test_causal_emergence_topology.py
├── test_causal_emergence_causal_discovery.py
├── test_causal_emergence_differential.py
├── test_causal_emergence_hmc.py
├── test_causal_emergence_chaotic_memory.py
└── test_causal_emergence_engine.py
```

### 10.4 Determinism fixture

Every test file includes:

```python
@pytest.fixture(autouse=True)
def _deterministic_rng():
    np.random.seed(42)
```

---

## 11. Military-Grade Review Scope (15 Dimensions)

After implementation, a comprehensive review covering:

1. **Mathematical correctness:** algorithms compared against reference literature (Edelsbrunner for persistent homology; Spirtes-Glymour-Scheines for PC; Shimizu et al. for LiNGAM; Betancourt for HMC; Lorenz 1963 for the attractor).
2. **Numerical stability:** NaN/Inf guards, condition numbers, convergence thresholds, finite-difference step choices.
3. **Edge cases:** empty input, single point, degenerate distributions, all-zero data, extreme magnitudes.
4. **Thread safety:** `_lock` consistency on `ZeroDataModel` facades; no shared mutable state in modules.
5. **API contract consistency:** return-dict keys match the spec; types match annotations; `Optional` returns are documented.
6. **Performance:** algorithmic complexity (e.g., persistent homology is O(n^3) in the worst case for boundary matrix reduction); degradation paths for large inputs (subsampling, JIT kernel fallback).
7. **Test coverage gaps:** branch coverage, mutation testing, parameter boundary tests.
8. **Integration correctness:** engine composes modules without hidden coupling; `ZeroDataModel` facade delegates correctly; `_N_COGNITIVE_MODULES` unchanged.
9. **zero-data philosophy adherence:** no pretrained weights, no external datasets, no heavy ML deps; all priors are encoded in `EmergenceRules`.
10. **Documentation accuracy:** docstrings match implementation; examples in docstrings are runnable.
11. **Determinism:** seeded runs are reproducible; no global state mutation.
12. **Memory bounds:** bounded deques, no unbounded caches, subsampling caps memory usage.
13. **Type annotations:** all public APIs are annotated; `from __future__ import annotations` consistently used.
14. **Error handling at boundaries:** `ValueError` for non-finite inputs at the public API; `try/except` for optional JIT kernel imports.
15. **Adversarial inputs:** NaN, Inf, all-zero, all-same, extremely large/small magnitudes, non-contiguous arrays, wrong dtype.

The review produces a written report with severity-tagged findings (CRITICAL / HIGH / MEDIUM / LOW / INFO) and concrete fix recommendations.

---

## 12. Implementation Phases

The implementation follows the four phases from the originating brief, executed sequentially as a single batch:

1. **Phase 1 (Foundation):** Module A (topology) + Module B (causal_discovery) + `EmergenceRules` + partial engine (perceive + discover facades only).
2. **Phase 2 (Action):** Module C (differential) + generate_trajectory facade.
3. **Phase 3 (Cognition):** Module D (hmc) + Module E (chaotic_memory) + sample_posterior + recall_memory facades.
4. **Phase 4 (Closed loop):** `emergence_cycle` orchestration + emergence_score computation + ZeroDataModel integration.

Each phase ends with a commit and a focused test pass.

---

## 13. Acceptance Criteria

- All 5 modules + engine implemented per this spec.
- All 6 test files pass (≥ 30 tests per module, ≥ 15 tests for engine).
- Existing capability tests still pass (no regressions).
- `ruff check` clean on all new files.
- Military-grade review completed and committed as a separate report document.
- `ZeroDataModel` integration: `emergence_cycle` runs end-to-end on a synthetic observation without errors.

---

## 14. Out of Scope

- Production-grade persistent homology (no gudhi / ripser).
- GPU acceleration (no cupy / numba CUDA kernels).
- Distributed / parallel sampling (no MPI / Dask).
- Web API endpoints (no FastAPI routes).
- Pretrained models or external datasets.
- Real-time performance guarantees.
- Persistence layer (no save/load of the engine state).
