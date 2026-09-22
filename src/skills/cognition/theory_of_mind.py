"""心理理论技能。

维护一个"他人模型"，预测其他智能体的信念、目标和未来动作。
在多智能体环境中，启用心理理论的智能体可以预测伙伴行为，
实现更优协作或策略欺骗。
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np

from ..base import SkillBase, SkillContext, SkillResult

logger = logging.getLogger(__name__)


class AgentModel:
    """对单个其他智能体的内部模型。

    维护该智能体的：
    - 信念状态估计
    - 目标估计
    - 动作历史
    - 动作预测模型（简单线性回归）
    """

    def __init__(self, agent_id: int, belief_dim: int = 16) -> None:
        self.agent_id = agent_id
        self.belief_dim = belief_dim
        self.estimated_belief: np.ndarray = np.zeros(belief_dim)
        self.estimated_goal: str = "unknown"
        self.action_history: deque = deque(maxlen=50)
        self.observation_history: deque = deque(maxlen=50)

        # 简单线性动作预测模型：belief → action
        self._pred_W = np.eye(belief_dim, dtype=np.float64) * 0.1
        self._pred_b = np.zeros(belief_dim, dtype=np.float64)
        self._last_prediction: np.ndarray | None = None
        self._last_prediction_error: float = 0.0

    def observe(
        self,
        observation: np.ndarray | None,
        action: np.ndarray | None,
    ) -> None:
        """记录其他智能体的一步观测和动作。"""
        if observation is not None:
            obs = np.asarray(observation, dtype=np.float64).flatten()
            obs = obs[: self.belief_dim]
            if obs.size < self.belief_dim:
                obs = np.pad(obs, (0, self.belief_dim - obs.size))
            self.observation_history.append(obs)
            # 更新信念估计：指数移动平均
            alpha = 0.1
            self.estimated_belief = (
                (1 - alpha) * self.estimated_belief + alpha * obs
            )

        if action is not None:
            act = np.asarray(action, dtype=np.float64).flatten()
            act = act[: self.belief_dim]
            if act.size < self.belief_dim:
                act = np.pad(act, (0, self.belief_dim - act.size))
            self.action_history.append(act)

            # 如果有上次预测，计算预测误差并更新模型
            if self._last_prediction is not None:
                error = act - self._last_prediction
                self._last_prediction_error = float(np.linalg.norm(error))
                # 简单梯度更新
                lr = 0.01
                last_obs = (
                    self.observation_history[-1]
                    if self.observation_history
                    else np.zeros(self.belief_dim)
                )
                self._pred_W += lr * np.outer(error, last_obs)
                self._pred_b += lr * error

    def predict_next_action(self) -> np.ndarray:
        """预测该智能体的下一步动作。"""
        prediction = self.estimated_belief @ self._pred_W + self._pred_b
        self._last_prediction = prediction.copy()
        return prediction

    def infer_goal(self) -> str:
        """根据动作历史推断目标。"""
        if len(self.action_history) < 3:
            return "unknown"
        actions = np.array(list(self.action_history))
        # 动作一致性
        action_std = float(np.std(actions, axis=0).mean())
        # 动作趋势
        if len(actions) >= 2:
            trend = actions[-1] - actions[0]
            if np.linalg.norm(trend) < 0.1:
                return "stationary"
            if action_std < 0.15:
                return "focused"
            return "exploring"
        return "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "estimated_belief": self.estimated_belief.tolist(),
            "estimated_goal": self.infer_goal(),
            "n_observations": len(self.observation_history),
            "n_actions": len(self.action_history),
            "prediction_error": self._last_prediction_error,
        }


class TheoryOfMindModule(SkillBase):
    """心理理论模块：建模其他智能体的信念、目标和动作。

    在多智能体环境中，为每个其他智能体维护一个 AgentModel，
    预测其行为以实现更好的协作或竞争策略。
    """

    name = "theory_of_mind"
    dimension = "cognition"

    def __init__(
        self,
        *,
        belief_dim: int = 16,
        max_agents: int = 10,
        enabled: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(enabled=enabled, **kwargs)
        self.belief_dim = belief_dim
        self.max_agents = max_agents
        self._agent_models: dict[int, AgentModel] = {}

    def get_or_create_model(self, agent_id: int) -> AgentModel:
        if agent_id not in self._agent_models:
            if len(self._agent_models) >= self.max_agents:
                # 移除最旧的
                oldest = min(self._agent_models)
                del self._agent_models[oldest]
            self._agent_models[agent_id] = AgentModel(
                agent_id=agent_id, belief_dim=self.belief_dim
            )
        return self._agent_models[agent_id]

    def observe_agent(
        self,
        agent_id: int,
        observation: np.ndarray | None = None,
        action: np.ndarray | None = None,
    ) -> None:
        model = self.get_or_create_model(agent_id)
        model.observe(observation, action)

    def predict_agent_action(self, agent_id: int) -> np.ndarray:
        model = self.get_or_create_model(agent_id)
        return model.predict_next_action()

    def process(self, ctx: SkillContext) -> SkillResult:
        # 从模型中提取多智能体信息
        # ctx.model 可能是 MultiAgentWorld 或 HierarchicalZeroDataModel
        other_agents_data: list[dict[str, Any]] = []
        predictions: dict[str, Any] = {}

        world = None
        if ctx.model is not None:
            # 尝试获取多智能体世界
            if hasattr(ctx.model, "agents"):
                world = ctx.model
            elif hasattr(ctx.model, "_model") and hasattr(
                ctx.model._model, "agents"
            ):
                world = ctx.model._model

        if world is not None and hasattr(world, "agents"):
            for agent_state in world.agents:
                aid = agent_state.agent_id
                if aid == 0:  # 跳过自己
                    continue
                model = self.get_or_create_model(aid)
                # 观测该智能体
                obs = getattr(agent_state, "last_signal", None)
                if obs is not None and hasattr(obs, "data"):
                    obs_vec = obs.data
                else:
                    obs_vec = None
                action = getattr(agent_state, "last_action", None)
                model.observe(obs_vec, action)

                # 预测下一步
                pred = model.predict_next_action()
                predictions[f"agent_{aid}"] = {
                    "predicted_action": pred.tolist()[:8],  # 截断
                    "estimated_goal": model.infer_goal(),
                    "prediction_error": model._last_prediction_error,
                    "n_observations": len(model.observation_history),
                }
                other_agents_data.append(model.to_dict())
        else:
            # 单智能体模式：模拟一个虚拟其他智能体用于演示
            if ctx.belief is not None:
                model = self.get_or_create_model(1)
                # 将自身信念作为"其他智能体"的观测
                obs_vec = ctx.belief[: self.belief_dim]
                if obs_vec.size < self.belief_dim:
                    obs_vec = np.pad(obs_vec, (0, self.belief_dim - obs_vec.size))
                model.observe(obs_vec, ctx.belief[: self.belief_dim])
                pred = model.predict_next_action()
                predictions["simulated_other"] = {
                    "predicted_action": pred.tolist()[:8],
                    "estimated_goal": model.infer_goal(),
                    "prediction_error": model._last_prediction_error,
                }
                other_agents_data.append(model.to_dict())

        return SkillResult(
            name=self.name,
            data={
                "n_modeled_agents": len(self._agent_models),
                "agent_models": [m.to_dict() for m in self._agent_models.values()],
                "predictions": predictions,
            },
        )
