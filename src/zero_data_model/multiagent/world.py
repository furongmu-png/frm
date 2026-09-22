"""多智能体世界：承载多个 ZeroDataModel 实例的共享环境。"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# 顶层导入 ZeroDataModel：经核查 model.py（及其依赖链）不反向导入
# multiagent.world，因此无循环依赖；原先的延迟导入仅为避免拉起重型依赖，
# 但顶层导入更清晰且符合“显式优于隐式”。
from zero_data_model.model import ZeroDataModel

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """单个智能体在多智能体世界中的运行时状态。"""

    agent_id: int
    model: Any
    last_signal: Any = None
    last_action: np.ndarray | None = None
    # 通信日志：默认空 deque；MultiAgentWorld 会用带 maxlen 的 deque 覆盖，
    # 以限制每智能体日志的无界增长。
    communication_log: deque[dict] = field(default_factory=deque)
    reward: float = 0.0


class MultiAgentWorld:
    """多智能体环境：管理 N 个 ZeroDataModel 实例的并行思考与协作。"""

    def __init__(
        self,
        n_agents: int = 2,
        dim: int = 64,
        seed: int = 42,
        shared_env: bool = True,
        max_events: int = 1024,
        max_log: int = 4096,
    ) -> None:
        self.dim = dim
        self.seed = seed
        self.shared_env = shared_env
        # 每个智能体使用独立种子（seed + i），保证可复现且彼此不同。
        self.agents: list[AgentState] = [
            AgentState(
                agent_id=i,
                model=ZeroDataModel(dim=dim, seed=seed + i),
            )
            for i in range(n_agents)
        ]
        # 用带 maxlen 的 deque 覆盖默认日志，限制每智能体通信日志增长。
        for agent in self.agents:
            agent.communication_log = deque(maxlen=max_log)
        # 共享环境观测（shared_env=True 时由世界统一生成）。
        self.shared_observation: np.ndarray | None = None
        self._step_count: int = 0
        # 协作事件使用有界 deque 防止无界增长。
        self._collaboration_events: deque[dict] = deque(maxlen=max_events)
        # 局部 RNG，用于在没有外部观测时合成共享观测。
        self._rng = np.random.default_rng(seed)
        # 可重入锁：保护 agents / _collaboration_events / communication_log。
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # 步进
    # ------------------------------------------------------------------
    def step(
        self, observations: list[np.ndarray] | None = None
    ) -> list[Any]:
        """推进一个世界步：让每个智能体思考并记录信号，随后检测协作。

        Args:
            observations: 可选的逐智能体观测列表。若为 ``None``，则由世界
                自行合成观测（共享环境时所有智能体看到同一份；否则各
                自让模型自生成）。

        Returns:
            每个智能体本步产生的信号列表（顺序与 ``self.agents`` 一致）。
        """
        with self._lock:
            # 决定每个智能体本轮看到的观测。
            per_agent_obs: list[np.ndarray | None] = []
            if observations is not None:
                # 外部提供观测：按位置分发；数量不足的智能体回退到自生成。
                # 复制以避免调用方与模型间共享同一 ndarray 引用（防别名）。
                for i, _agent in enumerate(self.agents):
                    if i < len(observations) and observations[i] is not None:
                        per_agent_obs.append(np.asarray(observations[i]).copy())
                    else:
                        per_agent_obs.append(None)
            elif self.shared_env:
                # 共享环境：世界合成一份观测，所有智能体共用。
                shared = self._rng.standard_normal(self.dim)
                self.shared_observation = shared
                # 每个智能体获得独立副本，避免模型原地修改互相串扰。
                per_agent_obs = [shared.copy() for _ in self.agents]
            else:
                # 非共享：每个智能体传 None，由模型内部自生成。
                per_agent_obs = [None for _ in self.agents]

            signals: list[Any] = []
            for i, (agent, obs) in enumerate(zip(self.agents, per_agent_obs, strict=False)):
                try:
                    signal = agent.model.think(obs)
                except (
                    ValueError,
                    KeyError,
                    IndexError,
                    FloatingPointError,
                ) as exc:
                    # 防御性：单个智能体出错不应拖垮整个世界。
                    logger.warning(
                        "step.think failed for agent %d: %s: %s",
                        i,
                        type(exc).__name__,
                        exc,
                    )
                    signal = None
                agent.last_signal = signal
                # 用信号的 data 字段作为该智能体本轮“动作”的几何表示。
                if signal is not None and hasattr(signal, "data"):
                    try:
                        agent.last_action = np.asarray(signal.data, dtype=float)
                    except (
                        ValueError,
                        KeyError,
                        IndexError,
                        FloatingPointError,
                    ) as exc:
                        logger.warning(
                            "step.action cast failed for agent %d: %s: %s",
                            i,
                            type(exc).__name__,
                            exc,
                        )
                        agent.last_action = None
                signals.append(signal)

            self._step_count += 1
            self._detect_collaboration()
            return signals

    # ------------------------------------------------------------------
    # 通信
    # ------------------------------------------------------------------
    def exchange_observations(self) -> None:
        """让每个智能体看到其他智能体上一动作的均值摘要。

        将“其他智能体 last_action 的均值”作为广播上下文写入各智能体的
        ``communication_log``。这是最朴素的观测共享协议。
        """
        with self._lock:
            actions = [a.last_action for a in self.agents]
            for i, agent in enumerate(self.agents):
                others = [
                    acts
                    for j, acts in enumerate(actions)
                    if j != i and acts is not None
                ]
                if others:
                    try:
                        summary = np.mean(np.stack(others), axis=0)
                    except (
                        ValueError,
                        KeyError,
                        IndexError,
                        FloatingPointError,
                    ) as exc:
                        logger.warning(
                            "exchange_observations summary failed: %s: %s",
                            type(exc).__name__,
                            exc,
                        )
                        summary = None
                else:
                    summary = None
                agent.communication_log.append(
                    {
                        "step": self._step_count,
                        "from": "world",
                        "kind": "observation_summary",
                        "summary": summary,
                    }
                )

    # ------------------------------------------------------------------
    # 协作
    # ------------------------------------------------------------------
    def record_collaboration(
        self, agent_a: int, agent_b: int, event_type: str
    ) -> None:
        """记录一次协作事件。"""
        with self._lock:
            self._collaboration_events.append(
                {
                    "step": self._step_count,
                    "agents": [agent_a, agent_b],
                    "type": event_type,
                }
            )

    def get_collaboration_stats(self) -> dict:
        """返回协作统计信息。"""
        with self._lock:
            return {
                "n_events": len(self._collaboration_events),
                "n_agents": len(self.agents),
                "total_steps": self._step_count,
            }

    def _detect_collaboration(self, threshold: float = 0.8) -> None:
        """检测本轮是否存在动作方向相近的智能体对，若存在则记录协作事件。

        注：由 :meth:`step` 在持有 ``_lock`` 时调用，故此处不再单独加锁；
        :meth:`record_collaboration` 的锁为可重入锁，重复获取安全。
        """
        n = len(self.agents)
        for i in range(n):
            for j in range(i + 1, n):
                a_i = self.agents[i].last_action
                a_j = self.agents[j].last_action
                if a_i is None or a_j is None:
                    continue
                try:
                    sim = self._cosine_similarity(a_i, a_j)
                except (
                    ValueError,
                    KeyError,
                    IndexError,
                    FloatingPointError,
                ) as exc:
                    logger.warning(
                        "_detect_collaboration similarity failed: %s: %s",
                        type(exc).__name__,
                        exc,
                    )
                    continue
                if sim is not None and sim >= threshold:
                    self.record_collaboration(i, j, "aligned_action")

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float | None:
        """计算两个向量的余弦相似度，对零向量或非有限值返回 ``None``。"""
        a = np.asarray(a, dtype=float).ravel()
        b = np.asarray(b, dtype=float).ravel()
        if a.shape != b.shape or a.size == 0:
            return None
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return None
        dot = float(np.dot(a, b))
        sim = dot / (na * nb)
        if not np.isfinite(sim):
            return None
        return sim

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------
    @property
    def agent_count(self) -> int:
        """当前世界中的智能体数量。"""
        return len(self.agents)

    @property
    def step_count(self) -> int:
        """世界已推进的步数。"""
        return self._step_count

    # ------------------------------------------------------------------
    # Phase 3 扩展：统一步进接口、配置加载、文化快照
    # ------------------------------------------------------------------
    def step_all(
        self, observations: list[np.ndarray] | None = None
    ) -> list[dict[str, Any]]:
        """统一步进接口：返回所有智能体的观测、动作、自由能等结构化信息。

        与 :meth:`step` 的区别：``step`` 返回原始 signal 列表，
        ``step_all`` 返回结构化字典列表，便于前端面板和审计日志消费。

        Returns
        -------
        list[dict]
            每个智能体一个字典，含：
            - ``agent_id``: 智能体 ID
            - ``observation``: 本步观测（截断到 dim）
            - ``action``: 动作向量
            - ``free_energy``: 自由能（从 signal.metadata 提取）
            - ``prediction_error``: 预测误差
        """
        signals = self.step(observations)
        results: list[dict[str, Any]] = []
        for i, (agent, sig) in enumerate(zip(self.agents, signals, strict=False)):
            obs = (
                observations[i] if observations and i < len(observations)
                else self.shared_observation
            )
            fe = 0.0
            pe = 0.0
            if sig is not None and hasattr(sig, "metadata"):
                _m = sig.metadata or {}
                fe = float(_m.get("free_energy", 0.0)) if isinstance(_m, dict) else 0.0
                pe = float(_m.get("prediction_error", 0.0)) if isinstance(_m, dict) else 0.0
            results.append({
                "agent_id": agent.agent_id,
                "observation": (
                    np.asarray(obs, dtype=float).flatten()[: self.dim]
                    if obs is not None else None
                ),
                "action": (
                    agent.last_action.copy()
                    if agent.last_action is not None else None
                ),
                "free_energy": fe,
                "prediction_error": pe,
                "reward": agent.reward,
            })
        return results

    @classmethod
    def from_config(
        cls, config: dict[str, Any] | str
    ) -> MultiAgentWorld:
        """从配置字典或 YAML 文件路径创建世界。

        配置格式（YAML 或 dict）::

            n_agents: 3
            dim: 64
            seed: 42
            shared_env: true
            scenario: collaborative  # collaborative|competitive|mixed

        Parameters
        ----------
        config:
            配置字典或 YAML 文件路径字符串。
        """
        if isinstance(config, str):
            try:
                import yaml  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(
                    "从 YAML 文件加载需要 PyYAML：pip install pyyaml"
                ) from exc
            with open(config, encoding="utf-8") as f:
                config = yaml.safe_load(f)
        if not isinstance(config, dict):
            raise TypeError(f"config 必须为 dict 或 YAML 路径，收到 {type(config)!r}")
        return cls(
            n_agents=int(config.get("n_agents", 2)),
            dim=int(config.get("dim", 64)),
            seed=int(config.get("seed", 42)),
            shared_env=bool(config.get("shared_env", True)),
            max_events=int(config.get("max_events", 1024)),
            max_log=int(config.get("max_log", 4096)),
        )

    def get_snapshot(self) -> dict[str, Any]:
        """返回世界当前状态的快照（供前端面板和文化传承使用）。"""
        with self._lock:
            return {
                "n_agents": len(self.agents),
                "step_count": self._step_count,
                "dim": self.dim,
                "shared_env": self.shared_env,
                "agent_states": [
                    {
                        "agent_id": a.agent_id,
                        "has_action": a.last_action is not None,
                        "action_norm": (
                            float(np.linalg.norm(a.last_action))
                            if a.last_action is not None else 0.0
                        ),
                        "reward": a.reward,
                    }
                    for a in self.agents
                ],
                "collaboration_stats": self.get_collaboration_stats(),
            }
