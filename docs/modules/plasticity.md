# Architecture Plasticity

`ArchitectureOptimizer` (`src/zero_data_model/plasticity/architect.py`) is the
**Phase 1** cognitive upgrade. Let the module graph adapt: a module whose
prediction error stays high gets *split*; one whose error stays low gets
*pruned*. The system stops being a fixed 6-module architecture and becomes a
self-tuning graph.

## Theory basis

Predictive-processing error budgeting: each module is allocated representation
capacity proportional to the free-energy reduction it delivers. A persistently
high-error module is evidence that its single generative model is under-fitting
two phenomena → split. A near-zero-error module is redundant → prune. The split
decision mirrors the active-inference free-energy (FE) decomposition:
ΔFE = accuracy − complexity, applied at the *architectural* level rather than
the parameter level.

## Feature flag

`enable_architect` (default `False`). When set, the architect is lazily imported
and constructed as `self.architect`; otherwise `None`.

## `think()` integration

End-of-cycle hook (the **only write-adjacent hook**):

1. `record_errors(module_names, errors)` — append this cycle's per-module
   prediction errors to the architect's history.
2. `evaluate(copy(self.modules), cycle)` — decide split / prune. Run **under
   `self._lock`** and passed a **defensive copy** of the modules list so a
   regression in `architect.py` cannot corrupt the live list.
3. Result lands in `upgrade_meta["architecture"]`.

## REST endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/architect/stats` | Error history + pending split/prune plan. |
| `GET` | `/architect/dormant` | List dormant (pruned) modules. |
| `POST` | `/architect/reactivate` | Reactivate a dormant module by name. |

## Thread safety

`ArchitectureOptimizer` carries its own `threading.RLock`, taken around
`record_errors` and `evaluate` (the error history and pending plan are read by
`think()`'s architect hook, so they must not be torn).

## Auto-generated reference

::: zero_data_model.plasticity.architect.ArchitectureOptimizer
