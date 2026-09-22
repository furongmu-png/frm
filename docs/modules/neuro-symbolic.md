# Neuro-Symbolic Fusion

Phase 4 layers symbolic rule constraints on top of the sub-symbolic core
(`LogicLayer`) and exposes the inferred causal transition structure
(`CausalInference`).

## Theory basis

- **Gödel t-norm fuzzy logic** — `LogicLayer` evaluates rules under the Gödel
  t-norm: conjunction `min(a,b)`, so a rule's satisfaction is bounded by its
  weakest premise. This is a hard symbolic constraint projected onto continuous
  activations.
- **Causal transition inference** — `CausalInference` maintains a transition
  matrix and surfaces its dimensionality (the "hot" causal structure the model
  currently believes).

## Feature flag

`enable_logic_layer` (default `False`). Note: the import is aliased
(`CausalInference as _KnowledgeCausalInference`) inside `model.py` to avoid
shadowing the top-level `CausalInference` from `capabilities.analytics_advanced`.

## `think()` integration

1. `logic_layer.check_all()` → `upgrade_meta["logic_violations"]` — only
   attached when `n_violations > 0`.
2. `causal_inference` surfaces `transition_matrix.shape[0]` →
   `upgrade_meta["causal_dim"]` (guarded against `None`).

## Thread safety

- `LogicLayer` — lock around `add_rule` / `check_all`.
- `CausalInference` — lock around the transition-matrix update.

## REST endpoints

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/logic/rule` | Add a symbolic rule to the `LogicLayer`. |
| `POST` | `/logic/predicate` | Add a logical predicate. |
| `POST` | `/causal/transition` | Update the causal transition matrix. |
| `POST` | `/causal/do` | Apply a do-intervention. |
| `POST` | `/causal/counterfactual` | Counterfactual query. |
| `POST` | `/causal/confounders` | List confounders. |

## Auto-generated reference

::: zero_data_model.knowledge.logic_layer.LogicLayer

::: zero_data_model.knowledge.logic_layer.LogicRule

::: zero_data_model.knowledge.causal_inference.CausalInference
