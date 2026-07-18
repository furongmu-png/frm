# Capabilities Extension: Audio + Graph + Robotics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three new capability domains (Audio, Graph, Robotics) to the zero-data cognitive model, each with a base module (4 classes) and an advanced module (3 classes), totaling 21 new classes. All follow the existing capability-domain pattern (plain classes, core-module composition, rule priors, L2-normalized outputs, zero external dependencies beyond numpy/scipy).

**Architecture:** Each domain follows the existing NLP/CV/Analytics pattern:
- `capabilities/<domain>.py` — 4 base classes (encoder + 3 analyzers)
- `capabilities/<domain>_advanced.py` — 3 advanced classes
- `capabilities/rules.py` extended with `<Domain>Rules` dataclass
- `capabilities/__init__.py` + `__init__.pyi` — export new classes
- `model.py` — instantiate as `self.<domain>_<role>` attributes (NOT in `self.modules`, NOT changing `_N_COGNITIVE_MODULES`)
- `model.py` — add facade methods (`with self._lock: return self.<capability>.<method>(...)`)
- `tests/test_capabilities_<domain>.py` + `tests/test_capabilities_<domain>_advanced.py` — shape/norm/determinism/domain-semantics/boundary tests

**Tech Stack:** numpy (FFT, linear algebra), scipy.ndimage (optional, for convolution), math (stdlib), existing core modules (math_universe, biological, active_inference, category_engine, consciousness_core, quantum_hybrid). No new external dependencies.

**Design Principles (from existing capabilities-extension plan):**
1. Zero-data: no pretrained weights, no external datasets
2. Compositional: each capability composes core modules, not rewrite logic
3. Rule-augmented: small `DomainRules` priors provide domain bias
4. TDD: tests written alongside implementation; existing tests must pass

---

## File Structure

### New source files (6)
- `src/zero_data_model/capabilities/audio.py` — AudioEncoder, OnsetDetector, PitchDetector, AudioClassifier
- `src/zero_data_model/capabilities/audio_advanced.py` — SpeechSegmenter, MusicAnalyzer, SpeakerRecognizer
- `src/zero_data_model/capabilities/graph.py` — GraphEncoder, CommunityDetector, PathFinder, CentralityAnalyzer
- `src/zero_data_model/capabilities/graph_advanced.py` — GraphIsomorphismDetector, DynamicGraphTracker, SpanningTreeExtractor
- `src/zero_data_model/capabilities/robotics.py` — MotionPlanner, KinematicsSolver, SensorFuser, GaitGenerator
- `src/zero_data_model/capabilities/robotics_advanced.py` — TrajectoryOptimizer, CollisionChecker, MPCController

### Modified source files (5)
- `src/zero_data_model/capabilities/rules.py` — add AudioRules, GraphRules, RoboticsRules dataclasses
- `src/zero_data_model/capabilities/__init__.py` — export 21 new classes + 3 Rules
- `src/zero_data_model/capabilities/__init__.pyi` — type stubs for new exports
- `src/zero_data_model/model.py` — 21 instance attributes + 21 facade methods
- `src/zero_data_model/model.pyi` — type stubs for new facade methods (if applicable)

### New test files (6)
- `tests/test_capabilities_audio.py`
- `tests/test_capabilities_audio_advanced.py`
- `tests/test_capabilities_graph.py`
- `tests/test_capabilities_graph_advanced.py`
- `tests/test_capabilities_robotics.py`
- `tests/test_capabilities_robotics_advanced.py`

---

## Domain 1: Audio (7 classes)

### AudioRules (in rules.py)
```python
@dataclass
class AudioRules(DomainRules):
    sample_rate: int = 16000
    frame_size: int = 1024      # STFT window
    hop_size: int = 512         # STFT hop
    n_mels: int = 26            # mel filter bank count
    onset_threshold: float = 0.3  # spectral flux threshold
    pitch_min_hz: float = 80.0   # min detectable pitch
    pitch_max_hz: float = 500.0  # max detectable pitch
    texture_labels: tuple = ("speech", "music", "noise", "silence")
    # Mel scale helper
    def hz_to_mel(self, hz: float) -> float: ...
    def mel_to_hz(self, mel: float) -> float: ...
```

### audio.py

#### 1. AudioEncoder
```python
class AudioEncoder:
    """Encode a 1D audio signal into a dim-length L2-normalized vector.

    Pipeline: Hann-windowed STFT -> mel-scale magnitude binning ->
    log compression -> fractal compression (via math_universe) -> dim vector.
    """
    def __init__(self, dim: int = 64, math_universe=None, rules: AudioRules | None = None): ...
    def encode(self, signal: np.ndarray, sample_rate: int = 16000) -> np.ndarray: ...
```
Key algorithm:
- Hann window: `0.5 - 0.5 * cos(2*pi*n/(frame_size-1))`
- STFT: `np.fft.rfft(windowed_frames, axis=1)` -> magnitude
- Mel filter bank: triangular filters spaced on mel scale, `n_mels` bins
- Log compression: `log(mel_mag + 1e-8)`
- Fractal compression: `math_universe.fractal.generate(mel_vec, n_iterations=3)`
- L2 normalize to dim

#### 2. OnsetDetector
```python
class OnsetDetector:
    """Detect note/onset events via spectral flux (positive magnitude diff)."""
    def __init__(self, dim: int = 64, rules: AudioRules | None = None): ...
    def detect(self, signal: np.ndarray, sample_rate: int = 16000) -> dict:
        # Returns: {'onset_frames': list[int], 'onset_times': list[float],
        #           'spectral_flux': np.ndarray, 'mean_flux': float}
```
Key algorithm:
- STFT magnitude per frame
- Spectral flux: `sum(max(0, mag[t] - mag[t-1]))` per frame
- Peak picking: flux > mean + threshold * std
- Convert frame indices to times: `frame * hop / sample_rate`

#### 3. PitchDetector
```python
class PitchDetector:
    """Detect fundamental frequency via autocorrelation + parabolic interpolation."""
    def __init__(self, dim: int = 64, rules: AudioRules | None = None): ...
    def detect(self, signal: np.ndarray, sample_rate: int = 16000) -> dict:
        # Returns: {'pitch_hz': float, 'confidence': float,
        #           'f0_candidates': np.ndarray}
```
Key algorithm:
- Autocorrelation: `np.correlate(frame, frame, mode='full')[len-1:]`
- Find peaks in lag range `[sample_rate/pitch_max, sample_rate/pitch_min]`
- Parabolic interpolation: `offset = 0.5*(y[t-1]-y[t+1])/(y[t-1]-2*y[t]+y[t+1])`
- Confidence: normalized peak height

#### 4. AudioClassifier
```python
class AudioClassifier:
    """Rule-based audio texture classification (speech/music/noise/silence)."""
    def __init__(self, dim: int = 64, rules: AudioRules | None = None): ...
    def classify(self, signal: np.ndarray, sample_rate: int = 16000) -> dict:
        # Returns: {'label': str, 'confidence': float, 'features': dict}
```
Key algorithm:
- Features: spectral centroid, zero-crossing rate, RMS energy, spectral rolloff
- Rules:
  - silence: energy < 1e-6
  - speech: centroid > 500Hz & ZCR > 0.05 & flatness < 0.3
  - music: flatness > 0.1 & onset regularity > 0.5
  - noise: flatness > 0.4 (default fallback)

### audio_advanced.py

#### 5. SpeechSegmenter
```python
class SpeechSegmenter:
    """Voice activity detection (VAD) via energy + zero-crossing rate."""
    def __init__(self, dim: int = 64, biological=None, rules: AudioRules | None = None): ...
    def segment(self, signal: np.ndarray, sample_rate: int = 16000) -> dict:
        # Returns: {'segments': list[{'start': float, 'end': float, 'label': str}],
        #           'total_duration': float, 'speech_ratio': float}
```
Key algorithm:
- Frame energy + zero-crossing rate
- Rule: speech if energy > threshold AND zcr < zcr_max
- Merge adjacent speech frames; fill gaps < 200ms
- Temporal encoding via biological.automata (with save/restore)

#### 6. MusicAnalyzer
```python
class MusicAnalyzer:
    """Beat tracking + tempo estimation via onset autocorrelation."""
    def __init__(self, dim: int = 64, biological=None, rules: AudioRules | None = None): ...
    def analyze(self, signal: np.ndarray, sample_rate: int = 16000) -> dict:
        # Returns: {'tempo_bpm': float, 'beat_frames': list[int],
        #           'beat_times': list[float], 'onset_envelope': np.ndarray}
```
Key algorithm:
- Onset envelope from spectral flux (reuse OnsetDetector internally)
- Autocorrelation of onset envelope
- Peak in lag range `[60..180 BPM]` -> tempo
- Beat tracking: dynamic programming peak picking

#### 7. SpeakerRecognizer
```python
class SpeakerRecognizer:
    """Extract speaker-discriminative features via LPC + formant estimation."""
    def __init__(self, dim: int = 64, math_universe=None, rules: AudioRules | None = None): ...
    def extract_features(self, signal: np.ndarray, sample_rate: int = 16000) -> np.ndarray: ...
    def compare(self, features_a: np.ndarray, features_b: np.ndarray) -> float: ...
```
Key algorithm:
- LPC (order 12) via Levinson-Durbin recursion (pure numpy)
- Formant estimation: roots of LPC polynomial -> Hz
- Feature vector: formants + spectral centroid + pitch
- L2 normalize to dim
- Compare: cosine similarity via math_universe

---

## Domain 2: Graph (7 classes)

### GraphRules (in rules.py)
```python
@dataclass
class GraphRules(DomainRules):
    default_weight: float = 1.0
    community_resolution: float = 1.0  # modularity threshold
    path_heuristic_weight: float = 1.0  # A* g vs h balance
    centrality_normalized: bool = True
    isomorphism_max_iter: int = 5  # Weisfeiler-Lehman iterations
```

### graph.py

#### 1. GraphEncoder
```python
class GraphEncoder:
    """Encode a graph (adjacency matrix + node features) into a dim vector."""
    def __init__(self, dim: int = 64, math_universe=None, rules: GraphRules | None = None): ...
    def encode(self, adjacency: np.ndarray, node_features: np.ndarray | None = None) -> np.ndarray: ...
```
Key algorithm:
- Graph statistics: n_nodes, n_edges, density, degree mean/std
- Spectral features: top-k eigenvalues of normalized Laplacian
- If node_features: aggregate (mean/std/max) along feature dim
- Fractal compress via math_universe -> dim L2 normalized

#### 2. CommunityDetector
```python
class CommunityDetector:
    """Rule-based community detection via modularity optimization (Louvain-style)."""
    def __init__(self, dim: int = 64, rules: GraphRules | None = None): ...
    def detect(self, adjacency: np.ndarray) -> dict:
        # Returns: {'communities': list[list[int]], 'modularity': float,
        #           'n_communities': int}
```
Key algorithm:
- Start each node in own community
- Iterative: move node to neighbor's community if modularity gain > 0
- Modularity: `Q = (1/2m) * sum(A_ij - k_i*k_j/2m) * delta(c_i, c_j)`
- Max 10 iterations or convergence

#### 3. PathFinder
```python
class PathFinder:
    """Shortest path via Dijkstra (unweighted) or A* (with heuristic)."""
    def __init__(self, dim: int = 64, rules: GraphRules | None = None): ...
    def find(self, adjacency: np.ndarray, source: int, target: int) -> dict:
        # Returns: {'path': list[int], 'distance': float, 'visited': int}
```
Key algorithm:
- Dijkstra with priority queue (sorted list, no heapq dependency beyond stdlib)
- A* variant with Euclidean heuristic (if node coordinates provided)
- Early termination when target reached

#### 4. CentralityAnalyzer
```python
class CentralityAnalyzer:
    """Compute degree, betweenness, closeness centrality (rule-based)."""
    def __init__(self, dim: int = 64, rules: GraphRules | None = None): ...
    def analyze(self, adjacency: np.ndarray) -> dict:
        # Returns: {'degree': np.ndarray, 'betweenness': np.ndarray,
        #           'closeness': np.ndarray, 'most_central': int}
```
Key algorithm:
- Degree: row sums
- Closeness: `1 / sum(shortest_path_lengths)` via Dijkstra per node
- Betweenness: Brandes algorithm (O(V*E))

### graph_advanced.py

#### 5. GraphIsomorphismDetector
```python
class GraphIsomorphismDetector:
    """Weisfeiler-Lehman style isomorphism check (rule-based, approximate)."""
    def __init__(self, dim: int = 64, rules: GraphRules | None = None): ...
    def check(self, adj_a: np.ndarray, adj_b: np.ndarray) -> dict:
        # Returns: {'isomorphic': bool, 'confidence': float, 'wl_hash_a': str, 'wl_hash_b': str}
```
Key algorithm:
- Weisfeiler-Lehman: iteratively relabel nodes by sorted neighbor labels
- If WL hashes match -> isomorphic (with high confidence)
- `max_iter` rounds (default 5)

#### 6. DynamicGraphTracker
```python
class DynamicGraphTracker:
    """Track community drift in temporal graphs (snapshot diff)."""
    def __init__(self, dim: int = 64, rules: GraphRules | None = None): ...
    def track(self, snapshots: list[np.ndarray]) -> dict:
        # Returns: {'community_drift': list[float], 'node_migrations': list[dict],
        #           'stability': float}
```
Key algorithm:
- Run CommunityDetector on each snapshot
- Match communities across snapshots by Jaccard overlap
- Track node migrations: which nodes changed community

#### 7. SpanningTreeExtractor
```python
class SpanningTreeExtractor:
    """Extract minimum spanning tree via Kruskal (with rule-based tie-breaking)."""
    def __init__(self, dim: int = 64, rules: GraphRules | None = None): ...
    def extract(self, adjacency: np.ndarray) -> dict:
        # Returns: {'mst_edges': list[tuple[int, int]], 'total_weight': float,
        #           'mst_adjacency': np.ndarray}
```
Key algorithm:
- Kruskal: sort edges by weight, union-find to reject cycles
- Tie-breaking: lower node index first (rule-based, deterministic)

---

## Domain 3: Robotics (7 classes)

### RoboticsRules (in rules.py)
```python
@dataclass
class RoboticsRules(DomainRules):
    dt: float = 0.01              # control timestep
    max_velocity: float = 1.0     # m/s
    max_acceleration: float = 5.0 # m/s^2
    arm_segments: int = 3         # DOF for planar arm
    arm_length: float = 1.0      # total reach (m)
    safety_margin: float = 0.05   # collision margin (m)
    mpc_horizon: int = 10         # MPC lookahead steps
```

### robotics.py

#### 1. MotionPlanner
```python
class MotionPlanner:
    """Generate smooth trajectories via cubic splines (rule-based)."""
    def __init__(self, dim: int = 64, math_universe=None, rules: RoboticsRules | None = None): ...
    def plan(self, waypoints: np.ndarray, n_steps: int = 100) -> dict:
        # Returns: {'trajectory': np.ndarray (n_steps, n_dof), 'velocities': np.ndarray,
        #           'accelerations': np.ndarray, 'total_time': float}
```
Key algorithm:
- Cubic spline interpolation between waypoints
- Velocity = 1st derivative, acceleration = 2nd derivative
- Clamp velocity/acceleration to rules limits
- Time parametrization: equal time per segment

#### 2. KinematicsSolver
```python
class KinematicsSolver:
    """Forward + inverse kinematics for planar N-DOF arm (rule-based)."""
    def __init__(self, dim: int = 64, rules: RoboticsRules | None = None): ...
    def forward(self, joint_angles: np.ndarray) -> np.ndarray:
        # Returns end-effector position (x, y)
    def inverse(self, target: np.ndarray, seed: np.ndarray | None = None) -> dict:
        # Returns: {'joint_angles': np.ndarray, 'success': bool, 'iterations': int}
```
Key algorithm:
- Forward: cumulative rotation matrices, end = sum of segment vectors
- Inverse: Jacobian pseudo-inverse iteration (damped least squares)
- `J = d(forward)/d(angles)` via finite differences
- `delta_angles = J^+ * (target - current)`, iterate until convergence

#### 3. SensorFuser
```python
class SensorFuser:
    """Kalman-style weighted sensor fusion (rule-based, no control theory libs)."""
    def __init__(self, dim: int = 64, rules: RoboticsRules | None = None): ...
    def fuse(self, measurements: list[np.ndarray], variances: list[float]) -> np.ndarray: ...
    def update(self, prior: np.ndarray, prior_var: float,
               measurement: np.ndarray, meas_var: float) -> dict:
        # Returns: {'estimate': np.ndarray, 'variance': float}
```
Key algorithm:
- Weighted average: `x_hat = sum(x_i / var_i) / sum(1/var_i)`
- Combined variance: `1 / sum(1/var_i)`
- Sequential update: `K = prior_var / (prior_var + meas_var)`

#### 4. GaitGenerator
```python
class GaitGenerator:
    """Generate periodic gait patterns (rule-based, for legged locomotion)."""
    def __init__(self, dim: int = 64, biological=None, rules: RoboticsRules | None = None): ...
    def generate(self, n_steps: int = 100, gait_type: str = "walk") -> dict:
        # Returns: {'joint_angles': np.ndarray (n_steps, n_legs*n_dof),
        #           'foot_contacts': np.ndarray (n_steps, n_legs), 'period': float}
```
Key algorithm:
- Sinusoidal patterns per leg, phase-offset by gait type
- walk: 4 legs, phase = [0, 0.5, 0.5, 0] (diagonal pairs)
- trot: phase = [0, 0.5, 0, 0.5] (diagonal sync)
- Contact pattern derived from phase (stance > 50% of cycle)
- Uses biological.automata for temporal smoothing (with save/restore)

### robotics_advanced.py

#### 5. TrajectoryOptimizer
```python
class TrajectoryOptimizer:
    """Minimize jerk (5th-order derivative) for smooth motion."""
    def __init__(self, dim: int = 64, math_universe=None, rules: RoboticsRules | None = None): ...
    def optimize(self, trajectory: np.ndarray, n_iter: int = 10) -> dict:
        # Returns: {'optimized': np.ndarray, 'jerk': float, 'improvement': float}
```
Key algorithm:
- Compute jerk = 3rd derivative
- Gradient descent: `traj -= lr * d(jerk^2)/d(traj)` via finite differences
- Constrained: clamp to velocity/acceleration limits
- Report improvement ratio

#### 6. CollisionChecker
```python
class CollisionChecker:
    """Bounding-box / sphere collision checking (rule-based)."""
    def __init__(self, dim: int = 64, rules: RoboticsRules | None = None): ...
    def check(self, obstacles: list[dict], position: np.ndarray, radius: float = 0.1) -> dict:
        # Returns: {'collision': bool, 'nearest_obstacle': dict | None,
        #           'distance': float}
    def check_path(self, obstacles: list[dict], path: np.ndarray,
                   radius: float = 0.1) -> dict:
        # Returns: {'collisions': list[int], 'safe': bool}
```
Key algorithm:
- Obstacle: `{'center': np.ndarray, 'radius': float}` (sphere) or `{'bbox': np.ndarray}` (box)
- Distance: Euclidean (sphere) or AABB (box)
- Safety margin from rules.safety_margin
- Path check: sample points, check each

#### 7. MPCController
```python
class MPCController:
    """Simplified model predictive control (rule-based, no optimizer libs)."""
    def __init__(self, dim: int = 64, active_inference=None, rules: RoboticsRules | None = None): ...
    def control(self, current_state: np.ndarray, target_state: np.ndarray,
                obstacles: list[dict] | None = None) -> dict:
        # Returns: {'action': np.ndarray, 'predicted_trajectory': np.ndarray,
        #           'cost': float}
```
Key algorithm:
- Predict `horizon` steps ahead using simple dynamics (velocity integrator)
- Cost = distance to target + obstacle penalty + control effort
- Rule-based optimization: try 8 candidate actions (cardinal + diagonal), pick min cost
- If active_inference provided: use compute_free_energy as cost component

---

## Integration into ZeroDataModel

### model.py changes (in `__init__`, after existing capability instantiation ~line 214)

```python
# Audio capabilities
self.audio_rules = AudioRules()
self.audio_encoder = AudioEncoder(dim=dim, math_universe=self.math_universe, rules=self.audio_rules)
self.audio_onset = OnsetDetector(dim=dim, rules=self.audio_rules)
self.audio_pitch = PitchDetector(dim=dim, rules=self.audio_rules)
self.audio_classifier = AudioClassifier(dim=dim, rules=self.audio_rules)
self.audio_segmenter = SpeechSegmenter(dim=dim, biological=self.biological, rules=self.audio_rules)
self.audio_music = MusicAnalyzer(dim=dim, biological=self.biological, rules=self.audio_rules)
self.audio_speaker = SpeakerRecognizer(dim=dim, math_universe=self.math_universe, rules=self.audio_rules)

# Graph capabilities
self.graph_rules = GraphRules()
self.graph_encoder = GraphEncoder(dim=dim, math_universe=self.math_universe, rules=self.graph_rules)
self.graph_community = CommunityDetector(dim=dim, rules=self.graph_rules)
self.graph_path = PathFinder(dim=dim, rules=self.graph_rules)
self.graph_centrality = CentralityAnalyzer(dim=dim, rules=self.graph_rules)
self.graph_isomorphism = GraphIsomorphismDetector(dim=dim, rules=self.graph_rules)
self.graph_dynamic = DynamicGraphTracker(dim=dim, rules=self.graph_rules)
self.graph_spanning = SpanningTreeExtractor(dim=dim, rules=self.graph_rules)

# Robotics capabilities
self.robotics_rules = RoboticsRules()
self.robotics_motion = MotionPlanner(dim=dim, math_universe=self.math_universe, rules=self.robotics_rules)
self.robotics_kinematics = KinematicsSolver(dim=dim, rules=self.robotics_rules)
self.robotics_sensor = SensorFuser(dim=dim, rules=self.robotics_rules)
self.robotics_gait = GaitGenerator(dim=dim, biological=self.biological, rules=self.robotics_rules)
self.robotics_trajectory = TrajectoryOptimizer(dim=dim, math_universe=self.math_universe, rules=self.robotics_rules)
self.robotics_collision = CollisionChecker(dim=dim, rules=self.robotics_rules)
self.robotics_mpc = MPCController(dim=dim, active_inference=self.active_inference, rules=self.robotics_rules)
```

### model.py facade methods (after existing facade block ~line 644)

```python
# Audio facades
def encode_audio(self, signal, sample_rate=16000): with self._lock: return self.audio_encoder.encode(signal, sample_rate)
def detect_onsets(self, signal, sample_rate=16000): with self._lock: return self.audio_onset.detect(signal, sample_rate)
def detect_pitch(self, signal, sample_rate=16000): with self._lock: return self.audio_pitch.detect(signal, sample_rate)
def classify_audio(self, signal, sample_rate=16000): with self._lock: return self.audio_classifier.classify(signal, sample_rate)
def segment_speech(self, signal, sample_rate=16000): with self._lock: return self.audio_segmenter.segment(signal, sample_rate)
def analyze_music(self, signal, sample_rate=16000): with self._lock: return self.audio_music.analyze(signal, sample_rate)
def extract_speaker_features(self, signal, sample_rate=16000): with self._lock: return self.audio_speaker.extract_features(signal, sample_rate)

# Graph facades
def encode_graph(self, adjacency, node_features=None): ...
def detect_communities(self, adjacency): ...
def find_path(self, adjacency, source, target): ...
def analyze_centrality(self, adjacency): ...
def check_isomorphism(self, adj_a, adj_b): ...
def track_dynamic_graph(self, snapshots): ...
def extract_spanning_tree(self, adjacency): ...

# Robotics facades
def plan_motion(self, waypoints, n_steps=100): ...
def solve_forward_kinematics(self, joint_angles): ...
def solve_inverse_kinematics(self, target, seed=None): ...
def fuse_sensors(self, measurements, variances): ...
def generate_gait(self, n_steps=100, gait_type="walk"): ...
def optimize_trajectory(self, trajectory, n_iter=10): ...
def check_collision(self, obstacles, position, radius=0.1): ...
def mpc_control(self, current_state, target_state, obstacles=None): ...
```

---

## Test Pattern (per domain)

Each test file follows the existing `test_capabilities_*.py` pattern:

```python
import numpy as np
import pytest

@pytest.fixture(autouse=True)
def _deterministic_rng():
    np.random.seed(42)

class TestAudioEncoder:
    def test_encode_returns_l2_normalized_dim_vector(self): ...
    def test_encode_deterministic(self): ...
    def test_encode_silence_returns_near_zero(self): ...
    def test_encode_different_signals_different_vectors(self): ...
    def test_encode_empty_signal_returns_zero_vector(self): ...
    def test_encode_stereo_downmixes_to_mono(self): ...

class TestOnsetDetector:
    def test_detect_returns_expected_dict_keys(self): ...
    def test_detect_silence_no_onsets(self): ...
    def test_detect_click_detects_onset(self): ...

# ... etc for each class
```

Test categories per class:
1. **Shape/norm/type contract** — return type, array shapes, L2 norm ≈ 1.0
2. **Determinism** — same input twice gives same output
3. **Domain semantics** — silence/speech/music distinguishable, shortest path correct
4. **Boundary cases** — empty input, single element, degenerate geometry
5. **Core module composition** — verify shared state restored (if borrowing biological.automata)

---

## Task Breakdown (3 batches)

### Batch A: Audio (7 classes + rules + facade + 2 test files)
- [ ] A1: Add AudioRules to rules.py
- [ ] A2: Create audio.py (AudioEncoder, OnsetDetector, PitchDetector, AudioClassifier)
- [ ] A3: Create audio_advanced.py (SpeechSegmenter, MusicAnalyzer, SpeakerRecognizer)
- [ ] A4: Export in capabilities/__init__.py + __init__.pyi
- [ ] A5: Integrate into model.py (7 instance attrs + 7 facade methods)
- [ ] A6: Write tests/test_capabilities_audio.py
- [ ] A7: Write tests/test_capabilities_audio_advanced.py
- [ ] A8: Run tests + ruff + commit

### Batch B: Graph (7 classes + rules + facade + 2 test files)
- [ ] B1: Add GraphRules to rules.py
- [ ] B2: Create graph.py (GraphEncoder, CommunityDetector, PathFinder, CentralityAnalyzer)
- [ ] B3: Create graph_advanced.py (GraphIsomorphismDetector, DynamicGraphTracker, SpanningTreeExtractor)
- [ ] B4: Export in capabilities/__init__.py + __init__.pyi
- [ ] B5: Integrate into model.py (7 instance attrs + 7 facade methods)
- [ ] B6: Write tests/test_capabilities_graph.py
- [ ] B7: Write tests/test_capabilities_graph_advanced.py
- [ ] B8: Run tests + ruff + commit

### Batch C: Robotics (7 classes + rules + facade + 2 test files)
- [ ] C1: Add RoboticsRules to rules.py
- [ ] C2: Create robotics.py (MotionPlanner, KinematicsSolver, SensorFuser, GaitGenerator)
- [ ] C3: Create robotics_advanced.py (TrajectoryOptimizer, CollisionChecker, MPCController)
- [ ] C4: Export in capabilities/__init__.py + __init__.pyi
- [ ] C5: Integrate into model.py (7 instance attrs + 7 facade methods)
- [ ] C6: Write tests/test_capabilities_robotics.py
- [ ] C7: Write tests/test_capabilities_robotics_advanced.py
- [ ] C8: Run tests + ruff + commit

### Final verification
- [ ] F1: Run full test suite `pytest tests/ -q`
- [ ] F2: Run `ruff check src/ tests/`
- [ ] F3: Verify demo.py still works (if it imports capabilities)

---

## Acceptance Criteria

1. All 21 new classes implemented with working algorithms (no stubs)
2. All 21 new facade methods on ZeroDataModel
3. 6 new test files with 5+ tests per class (~35+ tests per domain, ~105+ total)
4. `ruff check src/ tests/` passes
5. `pytest tests/ -q` — all existing tests pass + all new tests pass (excluding pre-existing httpx2 environment errors)
6. No new external dependencies beyond numpy/scipy
7. Each batch committed separately for easy rollback
