# Layered Time

Phase 2 replaces the single-timescale predictor with a hierarchy of fast/slow
layers (`LayeredPredictor`) and adds a recurrent temporal memory
(`TemporalMemory`) so the system carries state across cycles.

## Theory basis

- **Hierarchical predictive coding** — multi-timescale prediction: L0 fast /
  L1 medium / L2 slow. Each layer predicts the layer below it; prediction
  errors propagate upward and predictions propagate downward.
- **Echo State Network (ESN)** — `TemporalMemory` is a fixed random reservoir
  with a trained readout. The reservoir state
  `x_t = tanh(W_in·u_t + W_res·x_{t-1})` carries temporal context without
  backprop-through-time (BPTT). Sub-dims are floored at 1 (`max(dim//2, 1)`)
  so `dim=1` does not produce degenerate 0-width layers.

## Feature flag

`enable_layered_predictor` (default `False`).

## `think()` integration

Two hooks (each wrapped in its own `try/except`):

1. `layered_predictor.update(signal.data, cycle)` →
   `upgrade_meta["layered_predictor"]`.
2. `temporal_memory.update(signal.data, signal.data, lr=0.01)` →
   `upgrade_meta["temporal_memory_error"]`.

## Thread safety

`LayeredPredictor` and `TemporalMemory` each carry their own `threading.RLock`,
taken around `update` (in-place weight / reservoir updates).

## Auto-generated reference

::: zero_data_model.cogtime.layered_predictor.LayeredPredictor

::: zero_data_model.cogtime.layered_predictor.Layer

::: zero_data_model.cogtime.temporal_memory.TemporalMemory
