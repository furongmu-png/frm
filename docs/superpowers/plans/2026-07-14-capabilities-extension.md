# Zero-Data Model — Capabilities Extension Plan

## Goal
Extend the zero-data model with concrete NLP, CV, and Analytics capabilities
built on top of the existing 6 core modules (Consciousness, Active Inference,
Category Theory, Quantum Hybrid, Biological, Math Universe). Domain rule
libraries are allowed as prior knowledge.

## Design Principles
1. **Zero-data**: No pretrained weights, no external datasets. All capabilities
   derive from internal mechanisms (self-generation, cross-domain analogy,
   active inference, free-energy minimization).
2. **Compositional**: Each capability composes existing core modules rather
   than reimplementing logic.
3. **Rule-augmented**: A small `DomainRules` prior (stop-words, Sobel kernel,
   seasonal decomposition, etc.) provides domain inductive bias.
4. **TDD**: Tests written alongside implementation; all 38 existing tests must
   continue to pass.

## File Layout
```
src/zero_data_model/capabilities/
    __init__.py          # Public exports
    rules.py             # DomainRules base + NLPRules / VisionRules / AnalyticsRules
    nlp.py               # TextEncoder, SemanticComparator, ZeroShotClassifier, TextGenerator
    vision.py            # ImageEncoder, FeatureExtractor, PatternRecognizer, ShapeAnalyzer
    analytics.py         # TimeSeriesForecaster, AnomalyDetector, PatternMiner, TrendAnalyzer
tests/
    test_capabilities_nlp.py
    test_capabilities_vision.py
    test_capabilities_analytics.py
src/zero_data_model/model.py  # Extended with .nlp / .vision / .analytics
demo.py                        # Extended demo
```

## Module Specs

### NLP (`nlp.py`)
- `TextEncoder(dim=64)`: char n-gram hashing + Unicode features + LinguisticRules.
  `encode(text: str) -> np.ndarray`.
- `SemanticComparator(encoder, math_universe, category_engine)`:
  `similarity(a, b) -> float` (cosine + KL divergence blend).
- `ZeroShotClassifier(dim, active_inference, category_engine)`:
  Prototypes self-generated from DNA storage; classes defined by rules.
  `classify(text: str) -> tuple[str, float]`.
- `TextGenerator(dim, biological, math_universe)`:
  Char-level generation via DNA crossover + fractal iteration + rule priors.
  `generate(seed: str, length: int) -> str`.

### CV (`vision.py`)
- `ImageEncoder(dim=64, math_universe)`:
  Topological features + fractal compression of flattened image.
  `encode(image: np.ndarray) -> np.ndarray`.
- `FeatureExtractor(dim, biological)`:
  Morphogenetic Laplacian as edge map + cellular automata texture descriptor.
  `extract(image: np.ndarray) -> dict[str, np.ndarray]`.
- `PatternRecognizer(dim, active_inference, consciousness)`:
  Free-energy-based pattern matching against self-generated prototypes.
  `recognize(image: np.ndarray) -> tuple[str, float]`.
- `ShapeAnalyzer(dim, category_engine)`:
  Isomorphism-based shape comparison + geometric rule priors.
  `analyze(image: np.ndarray) -> dict`.

### Analytics (`analytics.py`)
- `TimeSeriesForecaster(dim, active_inference, quantum_hybrid)`:
  Generative-model next-state prediction + annealer parameter tuning.
  `forecast(series: np.ndarray, horizon: int) -> np.ndarray`.
- `AnomalyDetector(dim, active_inference)`:
  High free-energy = anomaly. `detect(series: np.ndarray) -> np.ndarray[bool]`.
- `PatternMiner(dim, biological, math_universe)`:
  Cellular automata rule mining + fractal self-similarity scoring.
  `mine(series: np.ndarray) -> dict`.
- `TrendAnalyzer(dim, math_universe, category_engine)`:
  Information-geometry geodesic interpolation + regime classification.
  `analyze(series: np.ndarray) -> dict`.

## Integration into ZeroDataModel
```python
self.nlp = NLPCapabilities(dim=dim, ...)
self.vision = VisionCapabilities(dim=dim, ...)
self.analytics = AnalyticsCapabilities(dim=dim, ...)
```
Helper methods on `ZeroDataModel`:
- `encode_text`, `classify_text`, `generate_text`
- `encode_image`, `recognize_pattern`
- `forecast`, `detect_anomalies`

## Tasks
1. Create `capabilities/__init__.py`, `capabilities/rules.py` (all rule libs).
2. Implement `nlp.py` + `test_capabilities_nlp.py` (TDD).
3. Implement `vision.py` + `test_capabilities_vision.py` (TDD).
4. Implement `analytics.py` + `test_capabilities_analytics.py` (TDD).
5. Integrate into `model.py`; extend `demo.py`.
6. Final validation: all tests pass, demo runs end-to-end.

## Acceptance
- New tests pass; existing 38 tests still pass.
- demo.py demonstrates NLP/CV/Analytics without external data.
- Zero external dependencies beyond numpy/scipy.
