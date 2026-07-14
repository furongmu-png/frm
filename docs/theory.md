# Zero-Data Model — Theoretical Foundations

This document explains the science behind each of the six core modules and the
zero-data principle that ties them together. For every theory we give a short
exposition and then map it to the concrete classes and methods in
`src/zero_data_model/`. A consolidated reference list is at the end.

---

## Contents

1. [Global Workspace Theory](#1-global-workspace-theory)
2. [Predictive Processing](#2-predictive-processing)
3. [Free Energy Principle / Active Inference](#3-free-energy-principle--active-inference)
4. [Category Theory / Topos Theory](#4-category-theory--topos-theory)
5. [Variational Quantum Circuits & Quantum Annealing](#5-variational-quantum-circuits--quantum-annealing)
6. [Biological Computation](#6-biological-computation)
7. [Mathematical Universe](#7-mathematical-universe)
8. [The Zero-Data Principle](#8-the-zero-data-principle)
9. [References](#9-references)

---

## 1. Global Workspace Theory

**Theory.** Bernard Baars's Global Workspace Theory (GWT) models consciousness
as a "theater" in which many parallel, specialist processes compete for access
to a shared **global workspace** — a limited-capacity blackboard. Whichever
process wins the competition is **broadcast** back to all specialists, making
that information globally available. The blackboard metaphor is central: there
is no single homunculus, only distributed specialists plus a broadcast bus that
momentarily integrates their outputs. GWT explains how a brain made of parallel
modules can behave as a single, coherent system.

**Mapping to the code.** The class `GlobalWorkspace` in
`src/zero_data_model/consciousness_core.py` is a direct implementation of this
blackboard. It holds:

- `attention_weights` — a per-dimension gate that selects which parts of a
  signal win the "competition".
- `buffer` — the limited-capacity workspace (`capacity=16` by default). When it
  overflows the oldest entry is dropped (`buffer.pop(0)`).
- `broadcast(signal)` — multiplies the signal by the attention weights,
  normalizes, appends to the buffer, and returns the **mean of the buffered
  signals** as the globally integrated output.

`ConsciousnessCore.process` calls `self.workspace.broadcast(...)` after running
its predictive hierarchy, so the workspace is the integration point that turns
per-layer predictions into a single broadcast signal. At the model level,
`ZeroDataModel._integrate` performs an analogous broadcast by averaging the
outputs of all six core modules — the workspace pattern is replicated at the
whole-system scale.

The metacognitive companion to GWT here is `SelfModel`
(`consciousness_core.py`), which keeps an exponentially-smoothed running state
and a confidence that grows with history length; `SelfModel.reflect()` is what
`ConsciousnessCore.reflect()` returns so the system can "think about its own
state".

---

## 2. Predictive Processing

**Theory.** Predictive Processing (Clark 2013; Friston 2010) holds that the
brain is a hierarchical prediction machine: each layer constantly predicts the
activity of the layer below, and propagates **prediction errors** upward. What
flows down the hierarchy is not raw sensory data but predictions; what flows up
is the residual error. Perception is the process of finding the latent causes
that minimize prediction error, and learning is the slow updating of the
generative weights that produce the predictions. The hierarchy is
recurrent — top-down predictions meet bottom-up errors at every level.

**Mapping to the code.** `PredictiveLayer` (`consciousness_core.py`) is one
level of such a hierarchy: it owns `weights` and `bias`, computes a prediction
via `predict(x) -> tanh(x @ W + b)`, and exposes
`prediction_error(actual, predicted)` as the mean-squared residual.
`ConsciousnessCore` stacks `n_layers=3` of these by default.

The predictive loop is realized in `ConsciousnessCore.process`: for each layer
it computes `prediction = layer.predict(x)`, then `error =
layer.prediction_error(x, prediction)`, then injects a small noise term scaled
by the error back into the state (`x = prediction + randn * error * 0.01`). The
slow weight update happens in `ConsciousnessCore.update(prediction_error)`,
which nudges every layer's weights by `prediction_error * 0.001`-scaled noise —
the closest a weight-agnostic system can get to error-modulated plasticity.

The same predict/error/update triad is enforced as an abstract contract by
`CognitiveModule` in `base.py` (`process`, `predict`, `update`), so **every**
core module — not just the consciousness core — is a predictive-processing unit.
At the system level, `ZeroDataModel.think()` computes the mean prediction
uncertainty across all six modules and feeds it back into each module's
`update`, closing the hierarchical-error loop at the global scale.

---

## 3. Free Energy Principle / Active Inference

**Theory.** Friston's Free Energy Principle (FEP) reframes survival as
minimization of **variational free energy** — an upper bound on the surprisal
of sensory observations. Variational free energy decomposes as
`F = complexity − accuracy`: the cost of holding a belief that is too far from
the prior (complexity) minus how well that belief predicts the data (accuracy).
A self-organizing system must therefore stay within a **Markov blanket**: the
statistical boundary separating its internal states from external states, with
sensory and active states mediating the two. **Active inference** generalizes
this to action selection: an agent chooses the action that minimizes *expected*
free energy, balancing **pragmatic** value (reaching preferred states) with
**epistemic** value (reducing uncertainty / gathering information — "epistemic
foraging"). Homeostasis is the special case where preferred states are fixed
set-points.

**Mapping to the code.** `ActiveInferenceEngine` (`active_inference.py`)
implements all of the above:

- `MarkovBlanket.create(sensory_dim, active_dim, internal_dim)` builds the
  blanket with `sensory_weights` and `active_weights` mediating internal↔external.
- `GenerativeModel` is the agent's internal model: `transition` (state→next
  state), `emission` (state→observation), `belief_state`. Its methods
  `infer_state(observation)` (perception via prediction-error gradient ascent
  on the belief), `predict_observation(state)`, and
  `predict_next_state(state, action)` are the standard active-inference
  primitives.
- `compute_free_energy(observation)` returns exactly
  `pred_error + 0.01 * ||belief_state||²` — i.e. **accuracy (the prediction
  error) plus complexity (the L2-norm of the belief)**. This is the literal
  `F = complexity − accuracy` decomposition (with signs chosen so minimizing F
  minimizes surprisal).
- `select_action(belief)` samples 8 candidate actions, predicts the resulting
  state and observation, and picks the candidate minimizing
  `expected_free_energy + 0.1 * homeostatic_deviation` — pragmatic + epistemic
  value.
- `epistemic_foraging(belief)` returns an exploration `Signal` only when
  `var(belief) > 0.1`, directly implementing information-seeking under
  uncertainty.
- `HomeostaticController.regulate(state)` pulls the state toward `target` with a
  proportional correction; `deviation(state)` is its scalar distance metric.

The engine is reused across domains: `ZeroShotClassifier.classify` and
`PatternRecognizer.recognize` both call `compute_free_energy` as a surprisal
penalty, and `TimeSeriesForecaster.forecast` rolls the `GenerativeModel`
forward to produce forecasts.

---

## 4. Category Theory / Topos Theory

**Theory.** Category theory studies structure via **objects** and
**morphisms** (arrows) between them, subject to composition and identity laws.
A **functor** is a structure-preserving map between two categories: it sends
objects to objects and morphisms to morphisms while respecting composition.
This is the formal language of "same structure, different content" — exactly
what cross-domain transfer requires: a solution in domain *A* can be carried
to domain *B* via a functor without solving the problem from scratch. A
**topos** is a category rich enough to model its own internal logic; central to
it is the **subobject classifier** Ω, an object that represents truth values
and lets one classify subobjects (in the classical case Ω = {0, 1}; in a
general topos truth is graded). **Isomorphism** — an invertible morphism —
identifies when two objects are "the same up to relabeling".

**Mapping to the code.** `CategoryTheoryEngine` (`category_engine.py`) is the
operational core, supported by three helpers:

- `Category` — a named collection of `objects` (dict[str, ndarray]) and
  `morphisms` (dict[(src, tgt), ndarray]). `add_object` / `add_morphism` build
  it; `compose(source, intermediate, target)` returns `m2 @ m1`, realizing
  morphism composition as matrix multiplication.
- `Functor` — a structure-preserving map with `source`, `target`,
  `object_map`, `morphism_map`. `Functor.apply(obj)` applies the first
  shape-compatible morphism transform to the object. `CategoryTheoryEngine`
  constructs a default `NLP → CV` functor in `_init_default_categories`.
- `ToposEngine` — owns `truth_values = linspace(0, 1, dim)` (graded truth!) and
  a `classifier` matrix. `classify(signal)` returns `1 / (1 + exp(-s @ Ω))`,
  i.e. a logistic readout — the subobject-classifier analogue that turns a
  signal into truth-value degrees.

The engine's user-facing methods are:

- `find_isomorphism(problem_a, problem_b)` — cosine similarity in `[-1, 1]`,
  used as a structural-equivalence score (and reused by
  `SemanticComparator.similarity`, `TrendAnalyzer.analyze`,
  `ZeroDataModel.find_analogies`).
- `transfer_solution(source_cat, target_cat, solution)` — applies the matching
  functor to move a solution between domains.
- `process(signal)` — classifies via the topos, and if any stored object scores
  > 0.8 isomorphism with the signal, transfers it through a functor and
  returns the transferred representation.

---

## 5. Variational Quantum Circuits & Quantum Annealing

**Theory.** A **variational quantum circuit (VQC)** is a parameterized quantum
circuit whose gates depend on classical parameters θ that are optimized to
minimize a cost — a hybrid quantum-classical scheme. A standard ansatz is the
**RY ansatz**: layers of single-qubit `RY(θ)` rotations (which move population
between |0⟩ and |1⟩) interleaved with **CNOT** entangling gates between
neighbors. The entanglement is what gives the ansatz its expressivity beyond
classical product states; measurement of the final state yields a probability
distribution over bitstrings. **Quantum annealing** solves combinatorial
optimization by slowly evolving a system from a trivial ground state to the
ground state of a problem Hamiltonian — typically an Ising / QUBO model with a
transverse field. In the absence of real quantum hardware, **simulated
annealing** approximates this with a classical temperature schedule
`T(t) = 1 / log(1 + t)` and Metropolis acceptance of bit-flip moves.

**Mapping to the code.** The hardware/quantum split lives in
`src/zero_data_model/hardware/quantum.py` and the orchestration in
`src/zero_data_model/quantum_hybrid.py`.

- `QiskitQuantumBackend` (`hardware/quantum.py`) builds a **real** Qiskit
  circuit in `_build_circuit`: for each layer it applies `qc.ry(theta, q)` to
  every qubit, then `qc.cx(i, i+1)` entanglements (gated by the magnitude of
  the `entangling` coupling matrix), then another `RY` layer and another round
  of CNOTs, finishing with `qc.measure`. It samples via
  `qiskit.primitives.StatevectorSampler` and collapses the bitstring histogram
  into a probability vector of length `2 * n_qubits`.
- `SimulatorQuantumBackend` is the hand-rolled fallback: it applies RY gates
  (`_ry_gate`) and a linear entanglement (`_entangle`) on a state vector,
  without Qiskit. The factory `get_quantum_backend(prefer=...)` picks one.
- `VariationalQuantumCircuit` (`quantum_hybrid.py`) owns the ansatz parameters
  (`params` of shape `(n_layers, n_qubits, 2)` — two RY angles per qubit per
  layer) and the symmetric `entangling` coupling matrix. Its `evolve(input)`
  biases the first-layer RY angles by the (scaled) input signal so external
  data steers the circuit, then delegates to `backend.evolve_and_measure`.
- `QuantumAnnealer` solves a binary quadratic problem: it minimizes
  `state @ cost_matrix @ state` over `state ∈ {-1, +1}^n` using the
  Metropolis schedule `temperature = 1 / log(1 + t)` and single-bit flips. The
  inner loop `_anneal_inner` is `@njit(cache=True)`-compiled by numba when
  available (pure-NumPy fallback otherwise); `.optimize(n_iterations)` warms up
  the JIT cache and returns `(best_state, best_energy)`. This is the simulated
  annealing approximation to a transverse-field Ising annealer.

`QuantumClassicalHybrid` composes both: `process(signal)` mixes the VQC output
(weight 0.3) with a classical `tanh(x @ W)` (weight 0.7), and
`solve_optimization()` exposes the annealer. `TimeSeriesForecaster` holds a
reference to the hybrid module so forecasts can lean on quantum exploration.

---

## 6. Biological Computation

**Theory.** Three biological metaphors underpin this module. **(a) DNA
storage** encodes information in a quaternary alphabet {A, C, G, T} rather than
binary, achieving high density and — crucially for a generative system —
enabling **crossover recombination** as a knowledge-generation operator: two
parent sequences are spliced at a random point to produce a child that inherits
structure from both. **(b) Morphogenesis** (Turing 1952) explains how
spatially homogeneous tissues self-organize into patterns through
**reaction-diffusion**: chemicals that diffuse and react can spontaneously form
stripes, spots and gradients. The mathematical engine is the discrete
**Laplacian** (the diffusion operator). **(c) Cellular automata** (Wolfram
2002) are discrete grids where each cell's next state is a fixed function of
its neighborhood; the 256 elementary **Wolfram rules** (one for each 8-bit
lookup table over {left, center, right}) generate a remarkable range of
behavior from trivial to Turing-complete — Rule 30 is famously chaotic and
used for pseudo-randomness.

**Mapping to the code.** `BiologicalSubstrate` (`biological.py`) owns all
three:

- `DNAStorage(KnowledgeStore)` uses the quaternary encoding
  `A=0, T=1, C=2, G=3`. `_encode` min-max-scales the data to `[0, 3]` and
  rounds; `_decode` inverts. `generate(query)` is the knowledge generator: if
  at least two sequences are stored, it picks two parents at random, picks a
  `crossover` point, and concatenates `parent1[:crossover] + parent2[crossover:]`
  — literal DNA crossover. This is one of the system's two self-generation
  engines (the other is the fractal generator).
- `MorphogeneticField.step` computes the discrete 5-point Laplacian
  (`roll`-based neighbors minus `4 * grid`) and advances the grid and each
  morphogen by `diffusion_rate * laplacian`. `develop(n_steps)` iterates and
  returns `grid * (1 + 0.1 * Σ morphogens)` — a Turing-style self-organized
  pattern. `BiologicalSubstrate.update` even adapts `diffusion_rate` to the
  prediction error, so the "morphology" can speed up or slow down its
  self-organization.
- `CellularAutomata(rule=30)` implements the Wolfram rule lookup via
  `_apply_rule(left, center, right)`: the index is `(L << 2) | (C << 1) | R`
  and the next state is bit `index` of `rule`. `evolve(n_steps)` returns the
  full history. Default rule 30 is the chaotic Wolfram rule.

`BiologicalSubstrate.process` blends 40 % DNA-crossover, 30 % morphogenetic
pattern, 30 % CA state into a single `dim`-length signal. Downstream,
`FeatureExtractor.extract` uses the CA for image texture, `TextGenerator`
uses `DNAStorage.generate` to recombine seed encodings, and `PatternMiner`
sweeps all 256 Wolfram rules to find the one whose evolution best matches a
binarized series.

---

## 7. Mathematical Universe

**Theory.** Three branches of mathematics meet here. **(a) Information
geometry** (Amari) treats probability distributions as points on a Riemannian
manifold whose metric is the **Fisher information matrix**; the natural
distance between distributions is the **KL divergence**, and the shortest path
between two distributions (a **geodesic**) is the curve that respects this
metric. On the probability simplex the Fisher metric reduces to `1/p_i`, and
the geodesic between `p` and `q` is the Hellinger/Bhattacharyya interpolation
`((1−t)√p + t√q)²`. **(b) Topological data analysis** (TDA) extracts shape
from point clouds via **persistent homology**: as a scale parameter grows,
connected components merge and holes appear/disappear; the **Betti numbers**
`β_0, β_1, β_2, ...` count components, loops, voids. **(c) Fractal geometry**
studies **self-similarity** — structures that look similar at every scale. An
**iterated function system (IFS)** is a finite set of contraction maps whose
attractor is a fractal; iterating the maps on any starting point produces the
fractal, and the same machinery gives **fractal compression** (representing
data as the parameters of the IFS that reproduces it).

**Mapping to the code.** `MathematicalUniverse` (`math_universe.py`) bundles
three engines:

- `InformationGeometry`:
  - `fisher_metric(distribution)` returns `1 / p` (the diagonal Fisher metric
    on the simplex).
  - `kl_divergence(p, q)` returns `Σ p log(p/q)` — the canonical KL.
  - `geodesic(p, q, t)` returns `((1−t)√p̂ + t√q̂)²` normalized — the
    Bhattacharyya geodesic on the simplex.
  Used by `SemanticComparator.similarity` (KL → similarity via `exp(−KL)`) and
  `TrendAnalyzer.analyze` (`geodesic_deviation` between series halves).
- `TopologicalAnalyzer`:
  - `compute_betti_numbers(data, max_radius)` — a simplified persistent-homology
    pass: sorts the values, counts `β_0` as the number of gaps exceeding
    `max_radius / n_points`, and sets `β_1 = max(0, n_points − β_0)`.
  - `topological_features(data)` packs `[β_0, β_1, mean, std, skew]` into a
    feature vector. Used by `ImageEncoder.encode` and `PatternMiner.mine`.
- `FractalGenerator`:
  - `_init_transforms` builds four affine contraction maps `(scale, offset)`.
  - `generate(initial, n_iterations)` iterates `x = tanh(scale @ x + offset)`
    cycling through the transforms — an IFS attractor iteration.
  - `compress(data)` returns `{mean, std, self_similarity}`, where
    `self_similarity` is the correlation between the first and second halves —
    a cheap self-similarity descriptor. Used by `ImageEncoder.encode`,
  `PatternMiner.mine`, and `TextGenerator.generate` (fractal refinement of the
  DNA-crossover vector).

`MathematicalUniverse.process` returns `0.5 * topo_features + 0.5 * fractal`
— the system's "mathematical view" of any signal.

---

## 8. The Zero-Data Principle

The defining claim of the system is that a useful cognitive model can be built
**without any external training data**. This is not magic: it rests on three
concrete substitutes for learned weights.

### 8.1 Self-generation

Where a trained model has weights fit to data, this system has **generative
operators** that produce novel structure from internal seeds:

- `DNAStorage.generate` recombines stored sequences via crossover (§6).
- `FractalGenerator.generate` iterates an IFS to produce self-similar
  attractors (§7).
- `MorphogeneticField.develop` runs Laplacian diffusion to self-organize
  patterns (§6).
- `CellularAutomata.evolve` applies Wolfram rules to grow structure from a seed
  (§6).

`ZeroDataModel._self_generate()` blends a DNA-crossover signal with a
fractal-iterated signal 50/50 — this is literally the input to `think()` when
no external data is supplied, and to `generate_knowledge()`. So the system can
always "think" because its generators never run out of material.

### 8.2 World-knowledge-as-rules

Where a trained model has implicit knowledge baked into its weights, this
system has **explicit rule priors** in `src/zero_data_model/capabilities/rules.py`:

- `NLPRules` — stop-words, sentiment lexicon, and topic keywords
  (`tech`, `nature`, `emotion`, `science`). These encode linguistic universals
  (which words are function words, which words signal which topic) that any
  human-annotated corpus would also surface. `TextEncoder.encode` hashes
  topic-keyword matches into the embedding, which is why two "tech" sentences
  land close together even with no shared surface vocabulary.
- `VisionRules` — Sobel x/y kernels, a 3×3 Gaussian kernel, and the shape name
  set (`circle`, `square`, `triangle`, `line`, `blob`). `FeatureExtractor`
  convolves these to get edges; `PatternRecognizer` synthesizes a prototype
  image per shape name.
- `AnalyticsRules` — moving-average window (5), z-score anomaly threshold (2.0),
  seasonal lags (4, 7, 12, 24), trend slope threshold (0.05). These are the
  statistical priors an analyst would bring to any series.

These rules are **inductive priors**, not data: they are short, hand-written,
deterministic, and consumed without any fitting. They are what let the
capabilities be useful on day one.

### 8.3 Compositionality

The third substitute for data is **structure reuse**. Because every
`CognitiveModule` shares the same `process / predict / update` contract, a
capability can be assembled by composition: `TimeSeriesForecaster` =
`ActiveInferenceEngine` + `QuantumClassicalHybrid` + `AnalyticsRules`;
`PatternRecognizer` = `ImageEncoder` + `ActiveInferenceEngine` +
`ConsciousnessCore` + `VisionRules`. Each composition multiplies the effective
behaviors of its parts without new parameters. Combined with cross-domain
functors (§4), a solution found in one domain can be transferred to another
without retraining.

### 8.4 Putting it together

`ZeroDataModel.think(None)` is the proof. With no input at all it:

1. Self-generates a signal (`_self_generate` — §8.1).
2. Fans it out to six theory-driven modules (§1–§7).
3. Integrates their outputs (Global-Workspace-style broadcast — §1).
4. Reflects on its own state (`SelfModel.reflect` — §1).
5. Predicts and updates from the mean prediction error (§2, §3).

The result is a coherent, metacognitively-tagged output produced from pure
structure plus three short rule files — no dataset in sight.

---

## 9. References

Key works behind each section. Citations are given in plain text; DOIs/links
are included where they are stable and freely resolvable.

- **Global Workspace Theory**
  - Baars, B. J. (1988). *A Cognitive Theory of Consciousness*. Cambridge
    University Press.
  - Baars, B. J., Franklin, S., & Ramsoy, T. Z. (2013). Global Workspace
    Dynamics: Cortical Integration and Conscious Experience. *Trends in
    Cognitive Sciences*, 17(10), 493–502.

- **Predictive Processing**
  - Clark, A. (2013). Whatever next? Predictive brains, situated agents, and
    the future of cognitive science. *Behavioral and Brain Sciences*, 36(3),
    181–204.
  - Friston, K. (2010). The free-energy principle: a unified brain theory?
    *Nature Reviews Neuroscience*, 11, 127–138.

- **Free Energy Principle / Active Inference**
  - Friston, K. (2010). The free-energy principle: a unified brain theory?
    *Nature Reviews Neuroscience*, 11, 127–138.
  - Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P., & Pezzulo, G.
    (2017). Active Inference: A Process Theory. *Neural Computation*, 29(1),
    1–49.
  - Parr, T., Pezzulo, G., & Friston, K. (2022). *Active Inference: The Free
    Energy Principle in Mind, Brain, and Behavior*. MIT Press.

- **Category Theory / Topos Theory**
  - Spivak, D. I. (2014). *Category Theory for the Sciences*. MIT Press.
    (Freely available: https://math.mit.edu/~dspivak/CT4S.pdf)
  - Mac Lane, S., & Moerdijk, I. (1992). *Sheaves in Geometry and Logic: A
    First Introduction to Topos Theory*. Springer.
  - Fong, B., & Spivak, D. I. (2019). *An Invitation to Applied Category
    Theory: Seven Sketches in Compositionality*. Cambridge University Press.

- **Variational Quantum Circuits / Quantum Annealing**
  - McClean, J. R., Romero, J., Babbush, R., & Aspuru-Guzik, A. (2016). The
    theory of variational hybrid quantum-classical algorithms. *New Journal of
    Physics*, 18, 023023. (https://doi.org/10.1088/1367-2630/18/2/023023)
  - Farhi, E., Goldstone, J., & Gutmann, S. (2000). Quantum annealing: A
    numerical study. arXiv:quant-ph/0007071.
  - Kadowaki, T., & Nishimori, H. (1998). Quantum annealing in the
    transverse Ising model. *Physical Review E*, 58, 5355.

- **Biological Computation**
  - Turing, A. M. (1952). The chemical basis of morphogenesis.
    *Philosophical Transactions of the Royal Society of London B*, 237, 37–72.
  - Wolfram, S. (2002). *A New Kind of Science*. Wolfram Media.
    (https://www.wolframscience.com/nks/)
  - Church, G. M., Gao, Y., & Kosuri, S. (2012). Next-Generation Digital
    Information Storage in DNA. *Science*, 337(6102), 1628.

- **Mathematical Universe**
  - Amari, S. (2016). *Information Geometry and Its Applications*. Springer.
  - Amari, S., & Nagaoka, H. (2000). *Methods of Information Geometry*. AMS.
  - Carlsson, G. (2009). Topology and Data. *Bulletin of the American
    Mathematical Society*, 46(2), 255–308.
  - Edelsbrunner, H., & Harer, J. (2010). *Computational Topology: An
    Introduction*. AMS.
  - Barnsley, M. F. (1988). *Fractals Everywhere*. Academic Press.

- **Zero-Data / Inductive Priors**
  - Mitchell, T. M. (1980). The need for biases in learning generalizations.
    Technical Report CBM-TR-117, Rutgers University.
  - Baxter, J. (2000). A model of inductive bias learning. *Journal of
    Artificial Intelligence Research*, 12, 149–198.
