# Multi-Agent

Phase 7 moves from a single-agent model to a world of agents that communicate
and propagate culture across generations.

## Theory basis

- **Multi-agent active inference** — each agent minimises its own free energy
  over a shared Markov blanket that includes other agents.
- **Cultural transmission** — `CulturePropagation` models cultural transmission
  across `Generation`s (vertical/horizontal meme flow) as a biased-copy process
  with mutation.

## Status

The three modules ship as **library code** with per-method `threading.RLock`
protection. The `ZeroDataModel.__init__` feature flag (`enable_multiagent`) and
`think()` hook are being staged so that phases 1–6 ship with stable,
fully-verified integration first.

A planned `world.step()` hook will contribute `["multiagent"]` to
`Signal.metadata["cognitive_upgrades"]`, plus `CommunicationChannel` /
`CulturePropagation` readouts.

## Thread safety

- `MultiAgentWorld` — `threading.RLock` around `step` (protects `agents`,
  `_collaboration_events`, `communication_log`).
- `CommunicationChannel` — `threading.RLock` around `encode_message` /
  `decode_observation` and the usage-count / correlation updates.
- `CulturePropagation` — `threading.RLock` around `record_generation` and the
  bounded `generations` deque.

## Programmatic usage

The three modules ship as library code today — they are constructed lazily by
`ZeroDataModel` only when the `enable_multiagent` flag lands, but you can use
them directly:

```python
from zero_data_model.multiagent.world import MultiAgentWorld
from zero_data_model.multiagent.communication import CommunicationChannel
from zero_data_model.multiagent.culture import CulturePropagation, Generation

# Each agent owns its own ZeroDataModel(dim, seed=seed+i).
world = MultiAgentWorld(n_agents=2, dim=64, seed=42)
channel = CommunicationChannel(vocab_size=10, embed_dim=8, seed=42)
culture = CulturePropagation(max_generations=256, seed=42)

# Run one world step: every agent thinks; collaboration events are detected
# under world._lock. Returns one Signal per agent.
signals = world.step()
print(len(signals), world.step_count, world.agent_count)

# Record a generation snapshot for the cultural-acceleration curve.
culture.record_generation(
    knowledge_graph_size=128,
    mean_free_energy=0.42,
    n_steps=100,
)
print(culture.get_knowledge_curve())      # list[(gen_id, knowledge_size), ...]
```

## Auto-generated reference

::: zero_data_model.multiagent.world.MultiAgentWorld

::: zero_data_model.multiagent.world.AgentState

::: zero_data_model.multiagent.communication.CommunicationChannel

::: zero_data_model.multiagent.culture.CulturePropagation

::: zero_data_model.multiagent.culture.Generation
