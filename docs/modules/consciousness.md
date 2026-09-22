# Consciousness Core

`ConsciousnessCore` (`src/zero_data_model/consciousness_core.py`) is the
Global-Workspace-style integration layer. It owns a `GlobalWorkspace`, a
`SelfModel`, and a `PredictiveLayer`.

## Theory basis

- **Global Workspace Theory (GWT)** — independent specialist modules compete
  for access to a shared workspace; the winner is broadcast back to all modules.
  `GlobalWorkspace` integrates per-module outputs into a broadcast signal.
- **Self-model** — `SelfModel` maintains a `self_confidence` scalar and a
  rolling history length. `consciousness.reflect()` returns a `Signal` whose
  `metadata` carries these — the system's metacognitive readout.
- **Predictive layer** — `PredictiveLayer` predicts the next signal; its
  prediction error feeds the model-wide `mean_err` that nudges every module's
  parameters via `update()`.

## Role in `think()`

After `_integrate()` produces the broadcast signal, `think()` calls
`consciousness.reflect()`. The returned `Signal.metadata["self_reflection"]`
contains:

```python
{
    "self_confidence": float,    # SelfModel confidence in [0, 1]
    "history_len": int,          # rolling history length
}
```

## Auto-generated reference

::: zero_data_model.consciousness_core.ConsciousnessCore

::: zero_data_model.consciousness_core.GlobalWorkspace

::: zero_data_model.consciousness_core.SelfModel

::: zero_data_model.consciousness_core.PredictiveLayer
