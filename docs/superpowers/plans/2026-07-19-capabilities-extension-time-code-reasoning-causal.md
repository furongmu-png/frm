# Capabilities Extension Round 2: Time + Code + Reasoning + Causal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four new capability domains (Time, Code, Reasoning, Causal) to the zero-data cognitive model, each with a base module (4 classes) and an advanced module (3 classes), totaling 28 new classes. All follow the established capability-domain pattern (plain classes, core-module composition, rule priors, L2-normalized outputs where applicable, zero external dependencies beyond numpy/scipy/stdlib `ast`).

**Architecture:** Each domain follows the existing Audio/Graph/Robotics pattern:
- `capabilities/<domain>.py` — 4 base classes
- `capabilities/<domain>_advanced.py` — 3 advanced classes
- `capabilities/rules.py` extended with `<Domain>Rules` dataclass
- `capabilities/__init__.py` — export new classes
- `model.py` — instantiate as `self.<domain>_<role>` attributes (NOT in `self.modules`, NOT changing `_N_COGNITIVE_MODULES`)
- `model.py` — add facade methods (`with self._lock: return self.<capability>.<method>(...)`)
- `tests/test_capabilities_<domain>.py` + `tests/test_capabilities_<domain>_advanced.py`

**Tech Stack:** numpy (FFT, linear algebra), scipy.ndimage (convolution), stdlib `ast` (Code domain), existing core modules (math_universe, biological, active_inference, category_engine, consciousness_core, quantum_hybrid). No new external dependencies.

**Design Principles:**
1. Zero-data: no pretrained weights, no external datasets
2. Compositional: each capability composes core modules, not rewrite logic
3. Rule-augmented: small `DomainRules` priors provide domain bias
4. TDD: tests written alongside implementation; existing tests must pass
5. Deterministic: `@pytest.fixture(autouse=True) def _deterministic_rng(): np.random.seed(42)`
6. NaN defense: `np.nan_to_num()` guards on all numeric outputs
7. CA save/restore: try/finally when borrowing `biological.automata` or `biological.morphogenetic` (R9-002/R9-010)

---

## File Structure

### New source files (8)
- `src/zero_data_model/capabilities/time.py` — TimeSeriesEncoder, SeasonalityDetector, FrequencyAnalyzer, EventTimestampAnalyzer
- `src/zero_data_model/capabilities/time_advanced.py` — AnomalyTimingDetector, CyclePhaseTracker, ForecastabilityScorer
- `src/zero_data_model/capabilities/code.py` — CodeEncoder, ASTAnalyzer, CodeSimilarityChecker, DefectPatternDetector
- `src/zero_data_model/capabilities/code_advanced.py` — ControlFlowAnalyzer, CodeStyleAnalyzer, DependencyGraphBuilder
- `src/zero_data_model/capabilities/reasoning.py` — PropositionalLogicEngine, DeductiveReasoner, InductiveReasoner, AnalogicalReasoner
- `src/zero_data_model/capabilities/reasoning_advanced.py` — AbductiveReasoner, DefeasibleReasoner, CausalChainReasoner
- `src/zero_data_model/capabilities/causal.py` — DecisionTreeBuilder, GameTheoryAnalyzer, CounterfactualReasoner, MultiArmedBandit
- `src/zero_data_model/capabilities/causal_advanced.py` — POMDPApproximator, CausalGraphBuilder, InterventionAnalyzer

### Modified source files (3)
- `src/zero_data_model/capabilities/rules.py` — add TimeRules, CodeRules, ReasoningRules, CausalRules dataclasses
- `src/zero_data_model/capabilities/__init__.py` — export 28 new classes + 4 Rules
- `src/zero_data_model/model.py` — 28 instance attributes + ~30 facade methods

### New test files (8)
- `tests/test_capabilities_time.py`
- `tests/test_capabilities_time_advanced.py`
- `tests/test_capabilities_code.py`
- `tests/test_capabilities_code_advanced.py`
- `tests/test_capabilities_reasoning.py`
- `tests/test_capabilities_reasoning_advanced.py`
- `tests/test_capabilities_causal.py`
- `tests/test_capabilities_causal_advanced.py`

---

## Domain D: Time (7 classes)

### TimeRules (in rules.py)
```python
@dataclass
class TimeRules(DomainRules):
    sample_rate: float = 1.0           # samples per time unit
    fft_window_size: int = 64         # window for STFT
    fft_hop_size: int = 32
    seasonality_max_lag: int = 128    # max autocorr lag to scan
    seasonality_threshold: float = 0.3  # min autocorr peak
    event_threshold_std: float = 2.0  # |z| > threshold = event
    forecast_min_samples: int = 8     # below this, return zeros

    def __post_init__(self):
        super().__init__()
        self.rules = {...}
```

### time.py

#### 1. TimeSeriesEncoder
```python
class TimeSeriesEncoder:
    """Encode a 1D time series into a dim-length L2-normalized vector.

    Pipeline: z-score normalize -> STFT magnitude -> log compression ->
    fractal compression (math_universe) -> dim vector.
    """
    def __init__(self, dim=64, math_universe=None, rules=None): ...
    def encode(self, series: np.ndarray) -> np.ndarray: ...
```
Algorithm:
- z-score: `(x - mean) / (std + 1e-8)`
- STFT: Hann window of `fft_window_size`, hop `fft_hop_size`, `np.fft.rfft`
- Log magnitude: `log(|stft| + 1e-8)`
- Fractal compress via `math_universe.fractal.compress`
- Pad/truncate to `dim`, L2 normalize

#### 2. SeasonalityDetector
```python
class SeasonalityDetector:
    """Detect seasonal periods via autocorrelation peaks."""
    def __init__(self, dim=64, rules=None): ...
    def detect(self, series: np.ndarray) -> dict:
        # Returns: {'periods': list[int], 'strengths': list[float], 'dominant_period': int}
```
Algorithm:
- Compute autocorrelation for lags 1..`seasonality_max_lag`
- Find peaks where `autocorr[lag] > seasonality_threshold` and is local max
- Rank by strength

#### 3. FrequencyAnalyzer
```python
class FrequencyAnalyzer:
    """Spectral analysis via FFT: dominant frequencies, power spectrum."""
    def __init__(self, dim=64, math_universe=None, rules=None): ...
    def analyze(self, series: np.ndarray) -> dict:
        # Returns: {'frequencies': np.ndarray, 'power': np.ndarray, 'dominant_freq': float, 'spectral_entropy': float}
```
Algorithm:
- `np.fft.rfft(series)` -> power = `|X|^2`
- Frequencies via `np.fft.rfftfreq(n, d=1/sample_rate)`
- Dominant = argmax of power (excluding DC)
- Spectral entropy: normalize power -> p, `-sum(p * log(p))`

#### 4. EventTimestampAnalyzer
```python
class EventTimestampAnalyzer:
    """Extract event timing: inter-arrival times, rate, burstiness."""
    def __init__(self, dim=64, rules=None): ...
    def analyze(self, timestamps: np.ndarray) -> dict:
        # Returns: {'inter_arrival': np.ndarray, 'rate': float, 'burstiness': float, 'total_events': int}
```
Algorithm:
- Sort timestamps, compute `diff = t[1:] - t[:-1]`
- Rate = `n / (t[-1] - t[0])`
- Burstiness = `std(diff) / (mean(diff) + std(diff))` (in [-1, 1])

### time_advanced.py

#### 5. AnomalyTimingDetector
```python
class AnomalyTimingDetector:
    """Detect anomalous inter-arrival gaps (rare timing patterns)."""
    def __init__(self, dim=64, active_inference=None, rules=None): ...
    def detect(self, timestamps: np.ndarray) -> dict:
        # Returns: {'anomalies': np.ndarray[bool], 'scores': np.ndarray, 'threshold': float}
```
Algorithm:
- Compute inter-arrival times
- z-score; |z| > event_threshold_std = anomaly
- Score = |z|

#### 6. CyclePhaseTracker
```python
class CyclePhaseTracker:
    """Track phase within a periodic cycle."""
    def __init__(self, dim=64, biological=None, rules=None): ...
    def track(self, series: np.ndarray, period: int | None = None) -> dict:
        # Returns: {'phases': np.ndarray, 'period': int, 'phase_coherence': float}
```
Algorithm:
- Detect period via SeasonalityDetector if not given
- Phase = `(t mod period) / period` in [0, 1)
- Phase coherence = `|mean(exp(i * 2π * phase))|`

#### 7. ForecastabilityScorer
```python
class ForecastabilityScorer:
    """Score how forecastable a series is (entropy + stationarity)."""
    def __init__(self, dim=64, math_universe=None, rules=None): ...
    def score(self, series: np.ndarray) -> dict:
        # Returns: {'forecastability': float, 'entropy': float, 'stationarity': float, 'autocorr_strength': float}
```
Algorithm:
- Entropy: histogram-based Shannon entropy (normalized to [0, 1])
- Stationarity: 1 - |mean(first_half) - mean(second_half)| / (std + 1e-8)
- Autocorr strength: max autocorrelation at non-zero lag
- Forecastability = 0.4*(1-entropy) + 0.3*stationarity + 0.3*autocorr_strength

---

## Domain E: Code (7 classes)

### CodeRules (in rules.py)
```python
@dataclass
class CodeRules(DomainRules):
    max_line_length: int = 100
    indent_size: int = 4
    max_function_complexity: int = 10  # cyclomatic threshold
    max_function_lines: int = 50
    naming_convention: str = "snake_case"
    defect_patterns: tuple = (
        "mutable_default",  # def f(x=[])
        "bare_except",      # except:
        "unused_import",    # placeholder (heuristic)
        "equals_none",      # if x == None
    )

    def __post_init__(self): ...
```

### code.py

#### 1. CodeEncoder
```python
class CodeEncoder:
    """Encode source code into a dim-length L2-normalized vector.

    Pipeline: tokenize -> char + token frequency features -> fractal
    compression (math_universe) -> dim vector.
    """
    def __init__(self, dim=64, math_universe=None, rules=None): ...
    def encode(self, source: str) -> np.ndarray: ...
```
Algorithm:
- Tokenize via `str.split()` after basic punctuation handling
- Feature vector: token count, char count, line count, indent depth, keyword density
- Fractal compress -> dim, L2 normalize

#### 2. ASTAnalyzer
```python
class ASTAnalyzer:
    """Parse Python AST and extract structural features."""
    def __init__(self, dim=64, rules=None): ...
    def analyze(self, source: str) -> dict:
        # Returns: {'functions': list[str], 'classes': list[str],
        #           'imports': list[str], 'complexity': int, 'max_depth': int,
        #           'node_count': int}
```
Algorithm:
- `ast.parse(source)`
- Walk tree: count FunctionDef, ClassDef, Import, ImportFrom
- Cyclomatic complexity: count branches (If, For, While, ExceptHandler, BoolOp) + 1
- Max depth: deepest nesting level

#### 3. CodeSimilarityChecker
```python
class CodeSimilarityChecker:
    """Compare two code snippets via token + structure similarity."""
    def __init__(self, dim=64, category_engine=None, rules=None): ...
    def compare(self, source_a: str, source_b: str) -> dict:
        # Returns: {'similarity': float, 'token_overlap': float, 'structure_similarity': float}
```
Algorithm:
- Tokenize both (split on whitespace + punctuation)
- Token overlap: Jaccard of token sets
- Structure similarity: compare AST node type sequences (cosine of count vectors)
- Overall = 0.5 * token_overlap + 0.5 * structure_similarity

#### 4. DefectPatternDetector
```python
class DefectPatternDetector:
    """Detect rule-based defect patterns via AST walking."""
    def __init__(self, dim=64, rules=None): ...
    def detect(self, source: str) -> dict:
        # Returns: {'defects': list[dict], 'total': int}
        # Each defect: {'pattern': str, 'line': int, 'col': int, 'snippet': str}
```
Patterns:
- `mutable default`: `ast.FunctionDef` with `args.defaults` containing `List`/`Dict`/`Set`
- `bare_except`: `ast.ExceptHandler` with `type=None`
- `equals_none`: `ast.Compare` with `Is` operator and `Constant(None)` (note: `== None` is `Eq`, `is None` is `Is`; flag `Eq` with None)

### code_advanced.py

#### 5. ControlFlowAnalyzer
```python
class ControlFlowAnalyzer:
    """Extract basic blocks and compute cyclomatic complexity per function."""
    def __init__(self, dim=64, rules=None): ...
    def analyze(self, source: str) -> dict:
        # Returns: {'functions': list[dict], 'avg_complexity': float, 'max_complexity': int}
        # Each function: {'name': str, 'complexity': int, 'lines': int, 'basic_blocks': int}
```
Algorithm:
- Walk FunctionDef nodes
- Per function: cyclomatic complexity = decision points + 1
- Basic blocks: decision points + 1 (approximation)

#### 6. CodeStyleAnalyzer
```python
class CodeStyleAnalyzer:
    """Rule-based style checks: line length, naming, indentation."""
    def __init__(self, dim=64, rules=None): ...
    def analyze(self, source: str) -> dict:
        # Returns: {'violations': list[dict], 'total': int, 'score': float}
```
Checks:
- Line length > max_line_length
- Mixed tabs/spaces
- Function name not matching naming convention (regex)
- Trailing whitespace

#### 7. DependencyGraphBuilder
```python
class DependencyGraphBuilder:
    """Build a module-level import dependency graph."""
    def __init__(self, dim=64, rules=None): ...
    def build(self, source: str) -> dict:
        # Returns: {'nodes': list[str], 'edges': list[tuple], 'adjacency': np.ndarray, 'n_modules': int}
```
Algorithm:
- Walk Import + ImportFrom nodes -> extract module names
- Build adjacency matrix (each import is an edge from current to imported)

---

## Domain F: Reasoning (7 classes)

### ReasoningRules (in rules.py)
```python
@dataclass
class ReasoningRules(DomainRules):
    max_inference_depth: int = 10
    consistency_check: bool = True
    default_confidence: float = 0.5
    contradiction_threshold: float = 0.5
    abduction_max_hypotheses: int = 5

    def __post_init__(self): ...
```

### reasoning.py

#### 1. PropositionalLogicEngine
```python
class PropositionalLogicEngine:
    """Rule-based propositional logic: modus ponens, resolution."""
    def __init__(self, dim=64, rules=None): ...
    def add_fact(self, proposition: str, value: bool) -> None: ...
    def add_rule(self, antecedent: str, consequent: str) -> None: ...
    def infer(self) -> dict:
        # Returns: {'facts': dict[str, bool], 'inferences': list[tuple], 'contradictions': list}
```
Algorithm:
- Facts as `dict[str, bool]`
- Rules as `dict[antecedent, list[consequents]]`
- Modus ponens: if antecedent is True, set consequent True
- Iterate until fixpoint or max_inference_depth

#### 2. DeductiveReasoner
```python
class DeductiveReasoner:
    """Term logic: syllogisms (Barbara, Celarent, etc.)."""
    def __init__(self, dim=64, category_engine=None, rules=None): ...
    def syllogism(self, major: tuple, minor: tuple) -> dict:
        # major = ("All M are P", ) or ("No M are P", ) ...
        # minor = ("S is M", )
        # Returns: {'conclusion': str | None, 'valid': bool, 'form': str}
```
Algorithm:
- Parse "All X are Y" / "No X are Y" / "Some X are Y" / "Some X are not Y"
- Apply Barbara/Celarent/Darii/Ferio rules
- Return conclusion or None

#### 3. InductiveReasoner
```python
class InductiveReasoner:
    """Generalize from examples: rule extraction."""
    def __init__(self, dim=64, category_engine=None, rules=None): ...
    def generalize(self, examples: list[dict], labels: list[bool]) -> dict:
        # Returns: {'rule': str, 'confidence': float, 'support': int, 'coverage': float}
```
Algorithm:
- For each (key, value) pair in positive examples, find values that appear only in positive
- Rule = "key == value"
- Confidence = support / total

#### 4. AnalogicalReasoner
```python
class AnalogicalReasoner:
    """Analogy via structural similarity (category theory)."""
    def __init__(self, dim=64, category_engine=None, math_universe=None, rules=None): ...
    def analogize(self, source: dict, target: dict) -> dict:
        # Returns: {'mapping': dict, 'similarity': float, 'transfer': dict}
```
Algorithm:
- Structural similarity via `category_engine.structural_similarity`
- Map source attrs to target attrs by closest match
- Transfer: apply source's "relation" to target

### reasoning_advanced.py

#### 5. AbductiveReasoner
```python
class AbductiveReasoner:
    """Best-explanation inference: given observation, find most plausible hypothesis."""
    def __init__(self, dim=64, active_inference=None, rules=None): ...
    def explain(self, observation: str, hypotheses: list[str], priors: list[float] | None = None) -> dict:
        # Returns: {'best': str, 'scores': list[float], 'confidence': float}
```
Algorithm:
- Score each hypothesis by prior * likelihood (rule-based keyword overlap)
- Best = argmax score
- Confidence = best_score / sum(scores)

#### 6. DefeasibleReasoner
```python
class DefeasibleReasoner:
    """Default logic with exceptions: 'All X are Y, unless Z'."""
    def __init__(self, dim=64, rules=None): ...
    def add_default(self, rule: tuple, exception: tuple | None = None) -> None: ...
    def conclude(self, facts: dict) -> dict:
        # Returns: {'conclusions': dict, 'defeated': list, 'ambiguous': list}
```
Algorithm:
- Apply rule if antecedent is in facts AND exception is NOT in facts
- Track defeated rules (exception triggered)
- Track ambiguous (rule conflict)

#### 7. CausalChainReasoner
```python
class CausalChainReasoner:
    """Chain causal implications: A -> B -> C."""
    def __init__(self, dim=64, rules=None): ...
    def add_causal(self, cause: str, effect: str) -> None: ...
    def trace(self, start: str, max_depth: int = 5) -> dict:
        # Returns: {'chain': list[str], 'effects': list[str], 'depth': int, 'cycles': bool}
```
Algorithm:
- Build causal graph as adjacency list
- DFS from start; track visited for cycle detection
- Return chain of effects

---

## Domain G: Causal/Decision (7 classes)

### CausalRules (in rules.py)
```python
@dataclass
class CausalRules(DomainRules):
    max_tree_depth: int = 5
    min_samples_split: int = 2
    nash_max_iter: int = 100
    bandit_epsilon: float = 0.1
    pomdp_horizon: int = 10
    causal_significance: float = 0.05

    def __post_init__(self): ...
```

### causal.py

#### 1. DecisionTreeBuilder
```python
class DecisionTreeBuilder:
    """Rule-based decision tree (ID3-style, entropy-based splits)."""
    def __init__(self, dim=64, rules=None): ...
    def fit(self, features: np.ndarray, labels: np.ndarray) -> dict:
        # Returns: {'tree': dict, 'depth': int, 'n_leaves': int, 'features_used': list[int]}
```
Algorithm:
- Recursive ID3: pick feature with max information gain
- Stop at max_tree_depth, pure node, or < min_samples_split
- Tree = nested dict

#### 2. GameTheoryAnalyzer
```python
class GameTheoryAnalyzer:
    """2-player normal-form game: payoff matrix, Nash equilibrium."""
    def __init__(self, dim=64, rules=None): ...
    def analyze(self, payoff_a: np.ndarray, payoff_b: np.ndarray | None = None) -> dict:
        # Returns: {'nash_equilibria': list[tuple], 'value_a': float, 'value_b': float, 'is_zero_sum': bool}
```
Algorithm:
- If payoff_b is None, assume zero-sum (payoff_b = -payoff_a)
- Find pure-strategy Nash: cell where a is row max AND b is col max
- Value = payoff at equilibrium

#### 3. CounterfactualReasoner
```python
class CounterfactualReasoner:
    """'What if' alternate scenarios: perturb input, recompute output."""
    def __init__(self, dim=64, active_inference=None, rules=None): ...
    def counterfactual(self, observed: np.ndarray, intervention: dict, model_fn=None) -> dict:
        # intervention: {'index': int, 'value': float}
        # Returns: {'factual': float, 'counterfactual': float, 'effect': float, 'surprisal': float}
```
Algorithm:
- factual = mean(observed) (or model_fn(observed) if provided)
- Build counterfactual input by replacing `observed[intervention['index']] = intervention['value']`
- counterfactual = mean(modified) (or model_fn(modified))
- effect = counterfactual - factual
- surprisal = active_inference.compute_free_energy(diff)

#### 4. MultiArmedBandit
```python
class MultiArmedBandit:
    """epsilon-greedy + UCB1 multi-armed bandit."""
    def __init__(self, dim=64, rules=None): ...
    def select(self, rewards_history: list[list[float]]) -> dict:
        # Returns: {'arm': int, 'method': str, 'expected_values': np.ndarray, 'confidence_bounds': np.ndarray}
```
Algorithm:
- Compute per-arm mean reward
- With prob epsilon, pick random (explore); else pick argmax (exploit)
- UCB1: `mean + sqrt(2 * ln(N) / n_i)`

### causal_advanced.py

#### 5. POMDPApproximator
```python
class POMDPApproximator:
    """Belief-state MDP approximation (point-based value iteration)."""
    def __init__(self, dim=64, active_inference=None, rules=None): ...
    def solve(self, transitions: np.ndarray, observations: np.ndarray, rewards: np.ndarray) -> dict:
        # Returns: {'policy': np.ndarray, 'value': np.ndarray, 'iterations': int, 'converged': bool}
```
Algorithm:
- transitions: (S, A, S) tensor
- observations: (S, O) matrix
- rewards: (S,) vector
- Value iteration: V[s] = max_a (R + gamma * sum_s' T[s,a,s'] * V[s'])
- Policy = argmax_a
- Converged if max |V - V_prev| < 1e-6

#### 6. CausalGraphBuilder
```python
class CausalGraphBuilder:
    """Build DAG from observational data (simplified PC algorithm)."""
    def __init__(self, dim=64, rules=None): ...
    def discover(self, data: np.ndarray, var_names: list[str] | None = None) -> dict:
        # Returns: {'adjacency': np.ndarray, 'edges': list[tuple], 'n_edges': int}
```
Algorithm:
- Compute correlation matrix
- For each pair (i, j) with |corr| > causal_significance, add edge i -> j (i < j as orientation heuristic)
- Return DAG

#### 7. InterventionAnalyzer
```python
class InterventionAnalyzer:
    """do-calculus approximation: estimate intervention effects from data."""
    def __init__(self, dim=64, active_inference=None, rules=None): ...
    def intervene(self, data: np.ndarray, intervention_var: int, intervention_value: float) -> dict:
        # Returns: {'pre_intervention_mean': np.ndarray, 'post_intervention_mean': np.ndarray, 'effect': np.ndarray, 'surprisal': float}
```
Algorithm:
- pre = mean(data, axis=0)
- post: for each row in data, set data[i, intervention_var] = intervention_value, compute new mean (this is a marginalization)
- effect = post - pre
- surprisal = active_inference.compute_free_energy(effect)

---

## Task Decomposition

### Batch D: Time Domain (Tasks D1-D7)
- D1: Add `TimeRules` to `rules.py`
- D2: Create `time.py` (4 classes)
- D3: Create `time_advanced.py` (3 classes)
- D4: Update `__init__.py` (time exports)
- D5: Update `model.py` (7 time attributes + 7 facade methods)
- D6: Create `test_capabilities_time.py`
- D7: Create `test_capabilities_time_advanced.py` + ruff + commit

### Batch E: Code Domain (Tasks E1-E7)
- E1: Add `CodeRules` to `rules.py`
- E2: Create `code.py` (4 classes)
- E3: Create `code_advanced.py` (3 classes)
- E4: Update `__init__.py` (code exports)
- E5: Update `model.py` (7 code attributes + 7 facade methods)
- E6: Create `test_capabilities_code.py`
- E7: Create `test_capabilities_code_advanced.py` + ruff + commit

### Batch F: Reasoning Domain (Tasks F1-F7)
- F1: Add `ReasoningRules` to `rules.py`
- F2: Create `reasoning.py` (4 classes)
- F3: Create `reasoning_advanced.py` (3 classes)
- F4: Update `__init__.py` (reasoning exports)
- F5: Update `model.py` (7 reasoning attributes + 7 facade methods)
- F6: Create `test_capabilities_reasoning.py`
- F7: Create `test_capabilities_reasoning_advanced.py` + ruff + commit

### Batch G: Causal Domain (Tasks G1-G7)
- G1: Add `CausalRules` to `rules.py`
- G2: Create `causal.py` (4 classes)
- G3: Create `causal_advanced.py` (3 classes)
- G4: Update `__init__.py` (causal exports)
- G5: Update `model.py` (7 causal attributes + 7 facade methods)
- G6: Create `test_capabilities_causal.py`
- G7: Create `test_capabilities_causal_advanced.py` + ruff + commit

### Final Verification
- Full ruff check on all src + tests
- Full pytest run
- Final summary commit (if any cleanup needed)

---

## Self-Review

### Spec Coverage
- All 28 classes (4 domains × 7 classes) have explicit tasks
- Each batch follows the proven Audio/Graph/Robotics pattern
- Rules, source files, init exports, model facades, tests — all covered

### Placeholder Scan
- No TBD/TODO markers
- All algorithms specified concretely
- Method signatures match across plan sections

### Type Consistency
- `TimeRules`, `CodeRules`, `ReasoningRules`, `CausalRules` — all extend `DomainRules`
- All encoders return `np.ndarray` (dim-length, L2-normalized)
- All analyzers return `dict` with documented keys
- Facade methods follow `with self._lock: return self.<cap>.<method>(...)` pattern
- Instance attributes follow `self.<domain>_<role>` naming (e.g., `self.time_encoder`, `self.code_encoder`)

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-19-capabilities-extension-time-code-reasoning-causal.md`. Will execute inline (4 batches) following the established Audio/Graph/Robotics pattern.
