# Structured Memory

Phase 3 stores experiences as a graph of episodes (`EpisodicGraph`) and indexes
belief vectors for similarity retrieval (`SemanticIndex`), replacing the flat
rolling buffer.

## Theory basis

- **Episodic memory as a labelled transition graph** — state → action →
  next-state with a free-energy tag. Enables trajectory replay and
  counterfactual inspection.
- **Vector retrieval index** — `SemanticIndex` is a vector retrieval index over
  belief vectors (cosine / inner-product neighbourhood queries) — the substrate
  for "have I been in a similar belief before?".

## Feature flag

`enable_episodic_memory` (default `False`).

## `think()` integration

1. `episodic_graph.insert(state=belief, action, next_state=None,
   free_energy=fe, step=cycle)` → `upgrade_meta["episodic_nodes"]`.
2. `semantic_index.add(signal.data, node_id=cycle, metadata)` →
   `upgrade_meta["semantic_index_size"]`.

## Thread safety

- `EpisodicGraph` — lock around `insert` and `_next_id`. The monotonically
  increasing episode id is the **critical race** fixed in this phase: the id
  counter is read-then-incremented, so without the lock two concurrent `insert`
  calls could hand out the same id.
- `SemanticIndex` — lock around `add` / `search`.

## REST endpoints

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/episodic/plan` | Plan a trajectory through the episodic graph. |
| `POST` | `/semantic/query` | Semantic similarity retrieval over belief vectors. |
| `POST` | `/memory/encode` | Encode an observation into the working/episodic store. |
| `POST` | `/memory/retrieve` | Retrieve the top-k nearest memories. |
| `POST` | `/memory/consolidate` | Consolidate short-term traces into long-term memory. |

## Auto-generated reference

::: zero_data_model.cogmem.episodic_graph.EpisodicGraph

::: zero_data_model.cogmem.episodic_graph.Episode

::: zero_data_model.cogmem.semantic_index.SemanticIndex
