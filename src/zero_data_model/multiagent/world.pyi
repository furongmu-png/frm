from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

@dataclass
class AgentState:
    """单个智能体在多智能体世界中的运行时状态。"""

    agent_id: int
    model: Any
    last_signal: Any = None
    last_action: np.ndarray | None = None
    communication_log: deque[dict[str, Any]] = field(default_factory=deque)
    reward: float = 0.0


class MultiAgentWorld:
    """多智能体环境：管理 N 个 ZeroDataModel 实例的并行思考与协作。"""

    dim: int
    seed: int
    shared_env: bool
    agents: list[AgentState]
    shared_observation: np.ndarray | None
    _step_count: int
    _collaboration_events: deque[dict[str, Any]]
    _rng: np.random.Generator
    _lock: threading.RLock

    def __init__(
        self,
        n_agents: int = 2,
        dim: int = 64,
        seed: int = 42,
        shared_env: bool = True,
        max_events: int = 1024,
        max_log: int = 4096,
    ) -> None: ...

    def step(
        self, observations: list[np.ndarray] | None = None
    ) -> list[Any]: ...

    def exchange_observations(self) -> None: ...

    def record_collaboration(
        self, agent_a: int, agent_b: int, event_type: str
    ) -> None: ...

    def get_collaboration_stats(self) -> dict[str, Any]: ...

    @property
    def agent_count(self) -> int: ...

    @property
    def step_count(self) -> int: ...

    # Phase 3 扩展
    def step_all(
        self, observations: list[np.ndarray] | None = None
    ) -> list[dict[str, Any]]: ...

    @classmethod
    def from_config(
        cls, config: dict[str, Any] | str
    ) -> MultiAgentWorld: ...

    def get_snapshot(self) -> dict[str, Any]: ...


__all__: list[str] = ["AgentState", "MultiAgentWorld"]
