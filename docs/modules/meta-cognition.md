# Meta-Cognition

Phase 5 adds second-order beliefs: the model forms beliefs *about* its own
prediction error and parameter drift, enabling "I am uncertain about being
uncertain" monitoring.

## Theory basis

Meta-cognition as a higher-order predictive layer: the meta-module predicts the
first-order prediction error and the norm of the parameter update, and tracks
the divergence as a meta-uncertainty signal (a second-order free-energy term).
This is the active-inference FE decomposition applied one level up.

## Feature flag

`enable_meta_cognition` (default `False`).

## `think()` integration

`meta_cognition.update(prediction_error=fe, param_update_norm=||transition||)`
→ `upgrade_meta["meta_cognition"]`.

## Thread safety

`MetaCognition` carries its own `threading.RLock`, taken around `update`.
NaN guards (`_last_valid_context`) and length-capped histories are applied
under the same lock so a concurrent read never observes a NaN that a writer is
in the middle of purging.

## REST endpoints (via MCP / direct call)

The meta-cognition readouts are exposed as MCP tools:

| Tool | Returns |
| --- | --- |
| `phase7_meta_cognition_confidence` | Second-order confidence scalar. |
| `phase7_meta_cognition_uncertainty` | Meta-uncertainty scalar. |
| `phase7_meta_cognition_should_seek_info` | Boolean: should the model seek more information? |
| `phase7_meta_cognition_stats` | Snapshot of internal state. |

## Auto-generated reference

::: zero_data_model.metacog.meta_cognition.MetaCognition
