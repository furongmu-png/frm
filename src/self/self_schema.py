"""SelfSchema — predictive model of the agent's own self.

理论基础
========
统一自我模型 (Integrated World-Self Model, IWSM) 认为自我意识的核心是一个
"自我图式"——一组关于"我自己"的预测模型。智能体不仅预测外部世界，还预测
自己下一时刻会注意什么、做什么动作、处于什么情绪状态。

本模块实现三个预测器：
  - **注意预测器**：基于 GWT 竞争历史，预测自己下一步会关注哪个模块的输出
  - **动作预测器**：预测自己下一步会选择什么动作
  - **情感预测器**：预测自己下一时刻的"情绪状态"（自由能分布模式）

训练信号：实际发生的注意选择、实际动作、实际自由能模式 vs 预测值的误差。
误差驱动 Hebbian 更新，使自我图式逐步校准。

自由能映射：自我预测误差 = 自我模型自洽性的自由能分量。当自我预测准确时，
系统对自己的认知高度自洽（高自我意识）；预测失误时触发自我图式校准。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


# ------------------------------------------------------------------ #
# 数据结构
# ------------------------------------------------------------------ #
@dataclass
class SelfPrediction:
    """自我图式单步预测结果。"""
    attention_prediction: np.ndarray   # 预测的注意分布 (n_modules,)
    action_prediction: int             # 预测的动作 ID
    action_probs: np.ndarray           # 预测的动作分布 (n_actions,)
    affect_prediction: np.ndarray      # 预测的情绪状态 (affect_dim,)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SelfSchemaStats:
    """自我图式累计统计。"""
    step: int = 0
    attention_accuracy: float = 0.0     # 注意预测准确率
    action_accuracy: float = 0.0        # 动作预测准确率
    affect_error: float = 0.0           # 情感预测误差
    self_consistency: float = 1.0       # 自我一致性（综合）


# ------------------------------------------------------------------ #
# 主类
# ------------------------------------------------------------------ #
class SelfSchema:
    """自我图式：关于"我自己"的预测模型。

    Parameters
    ----------
    dim : int
        隐空间维度（与信念状态一致）
    n_modules : int
        可被注意的模块数（GWT 候选数）
    n_actions : int
        动作空间大小
    affect_dim : int
        情绪状态维度（默认 4：valence, arousal, dominance, novelty）
    lr : float
        学习率
    seed : int | None
    """

    def __init__(
        self,
        dim: int = 64,
        n_modules: int = 4,
        n_actions: int = 14,
        affect_dim: int = 4,
        lr: float = 0.01,
        seed: Optional[int] = 42,
    ):
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if n_modules < 1:
            raise ValueError(f"n_modules must be >= 1, got {n_modules}")
        if n_actions < 1:
            raise ValueError(f"n_actions must be >= 1, got {n_actions}")
        if affect_dim < 1:
            raise ValueError(f"affect_dim must be >= 1, got {affect_dim}")

        self.dim = int(dim)
        self.n_modules = int(n_modules)
        self.n_actions = int(n_actions)
        self.affect_dim = int(affect_dim)
        self.lr = float(lr)
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._lock = threading.RLock()

        # 注意预测器：dim → n_modules
        # 输入是广播后的整合状态，输出是各模块的注意权重
        self._W_attn = self._rng.standard_normal(
            (n_modules, dim)
        ) * np.sqrt(2.0 / (dim + n_modules))

        # 动作预测器：dim → n_actions
        self._W_action = self._rng.standard_normal(
            (n_actions, dim)
        ) * np.sqrt(2.0 / (dim + n_actions))

        # 情感预测器：dim → affect_dim
        self._W_affect = self._rng.standard_normal(
            (affect_dim, dim)
        ) * np.sqrt(2.0 / (dim + affect_dim))

        # 上一时刻的整合状态（用于预测"下一步"）
        self._last_integrated = np.zeros(dim, dtype=np.float64)

        # 统计
        self.stats = SelfSchemaStats()
        self._attn_correct = 0
        self._action_correct = 0
        self._affect_error_sum = 0.0

    # ---------------------------------------------------------------- #
    # 预测
    # ---------------------------------------------------------------- #
    def predict(self, integrated_state: np.ndarray) -> SelfPrediction:
        """预测自己下一步的注意/动作/情感。

        Parameters
        ----------
        integrated_state : ndarray, shape (dim,)
            GWT 广播后的整合状态

        Returns
        -------
        SelfPrediction
        """
        state = np.asarray(integrated_state, dtype=np.float64).flatten()
        if state.shape != (self.dim,):
            s = np.zeros(self.dim)
            n = min(len(state), self.dim)
            s[:n] = state[:n]
            state = s
        if not np.all(np.isfinite(state)):
            state = np.zeros(self.dim)

        with self._lock:
            # 注意预测：softmax
            attn_logits = self._W_attn @ state
            attn_logits -= attn_logits.max()
            exp_logits = np.exp(attn_logits)
            attn_pred = exp_logits / exp_logits.sum()

            # 动作预测：softmax
            action_logits = self._W_action @ state
            action_logits -= action_logits.max()
            exp_a = np.exp(action_logits)
            action_probs = exp_a / exp_a.sum()
            action_pred = int(np.argmax(action_probs))

            # 情感预测：线性
            affect_pred = self._W_affect @ state
            if not np.all(np.isfinite(affect_pred)):
                affect_pred = np.zeros(self.affect_dim)

            return SelfPrediction(
                attention_prediction=attn_pred,
                action_prediction=action_pred,
                action_probs=action_probs,
                affect_prediction=affect_pred,
                metadata={"step": self.stats.step},
            )

    # ---------------------------------------------------------------- #
    # 更新（误差驱动 Hebbian）
    # ---------------------------------------------------------------- #
    def update(
        self,
        integrated_state: np.ndarray,
        actual_attention: np.ndarray,
        actual_action: int,
        actual_affect: np.ndarray,
    ) -> dict[str, Any]:
        """根据实际发生的注意/动作/情感更新自我图式。

        Parameters
        ----------
        integrated_state : ndarray, shape (dim,)
            GWT 广播后的整合状态（用于预测时的输入）
        actual_attention : ndarray, shape (n_modules,)
            实际的注意分布（GWT 胜者的 one-hot 或概率分布）
        actual_action : int
            实际选择的动作
        actual_affect : ndarray, shape (affect_dim,)
            实际的情绪状态

        Returns
        -------
        dict[str, Any]
            自我图式更新诊断
        """
        state = np.asarray(integrated_state, dtype=np.float64).flatten()
        if state.shape != (self.dim,):
            s = np.zeros(self.dim)
            n = min(len(state), self.dim)
            s[:n] = state[:n]
            state = s
        if not np.all(np.isfinite(state)):
            state = np.zeros(self.dim)

        actual_attn = np.asarray(actual_attention, dtype=np.float64).flatten()
        if actual_attn.shape != (self.n_modules,):
            a = np.zeros(self.n_modules)
            n = min(len(actual_attn), self.n_modules)
            a[:n] = actual_attn[:n]
            actual_attn = a
            s = a.sum()
            if s > 0:
                actual_attn = actual_attn / s
            else:
                actual_attn = np.ones(self.n_modules) / self.n_modules

        actual_aff = np.asarray(actual_affect, dtype=np.float64).flatten()
        if actual_aff.shape != (self.affect_dim,):
            af = np.zeros(self.affect_dim)
            n = min(len(actual_aff), self.affect_dim)
            af[:n] = actual_aff[:n]
            actual_aff = af
        if not np.all(np.isfinite(actual_aff)):
            actual_aff = np.zeros(self.affect_dim)

        if not (0 <= actual_action < self.n_actions):
            actual_action = 0

        with self._lock:
            # 用上一时刻状态做预测（"预测下一步"）
            pred = self.predict(self._last_integrated)

            # 注意误差
            attn_error = pred.attention_prediction - actual_attn
            attn_error_norm = float(np.linalg.norm(attn_error))

            # 动作误差
            action_target = np.zeros(self.n_actions)
            action_target[actual_action] = 1.0
            action_error = pred.action_probs - action_target
            action_error_norm = float(np.linalg.norm(action_error))
            action_correct = int(pred.action_prediction == actual_action)

            # 情感误差
            affect_error = pred.affect_prediction - actual_aff
            affect_error_norm = float(np.linalg.norm(affect_error))

            # Hebbian 更新（用上一时刻状态作为输入）
            # ΔW_attn = -lr * outer(attn_error, last_state)
            delta_attn = -self.lr * np.outer(attn_error, self._last_integrated)
            norm = np.linalg.norm(delta_attn)
            if norm > 1.0:
                delta_attn = delta_attn * (1.0 / norm)
            self._W_attn += delta_attn

            delta_action = -self.lr * np.outer(
                action_error, self._last_integrated
            )
            norm = np.linalg.norm(delta_action)
            if norm > 1.0:
                delta_action = delta_action * (1.0 / norm)
            self._W_action += delta_action

            delta_affect = -self.lr * np.outer(
                affect_error, self._last_integrated
            )
            norm = np.linalg.norm(delta_affect)
            if norm > 1.0:
                delta_affect = delta_affect * (1.0 / norm)
            self._W_affect += delta_affect

            # 更新状态
            self._last_integrated = state.copy()

            # 统计
            self.stats.step += 1
            n = self.stats.step
            # 注意准确率：argmax 匹配
            attn_correct = int(np.argmax(pred.attention_prediction) == np.argmax(actual_attn))
            self._attn_correct += attn_correct
            self.stats.attention_accuracy = self._attn_correct / n
            self._action_correct += action_correct
            self.stats.action_accuracy = self._action_correct / n
            self._affect_error_sum += affect_error_norm
            self.stats.affect_error = self._affect_error_sum / n
            # 自我一致性 = 1 - 归一化的综合误差
            combined_error = (
                attn_error_norm / max(np.sqrt(2), 1)
                + action_error_norm / max(np.sqrt(2), 1)
                + affect_error_norm / 10.0
            ) / 3.0
            self.stats.self_consistency = float(
                np.exp(-combined_error)
            )

            return {
                "attn_error": attn_error_norm,
                "action_error": action_error_norm,
                "affect_error": affect_error_norm,
                "attn_correct": bool(attn_correct),
                "action_correct": bool(action_correct),
                "self_consistency": self.stats.self_consistency,
                "step": self.stats.step,
            }

    # ---------------------------------------------------------------- #
    # 查询
    # ---------------------------------------------------------------- #
    def get_snapshot(self) -> dict[str, Any]:
        """返回自我图式快照（用于前端"自我模型面板"）。"""
        with self._lock:
            pred = self.predict(self._last_integrated)
            return {
                "step": self.stats.step,
                "attention_accuracy": self.stats.attention_accuracy,
                "action_accuracy": self.stats.action_accuracy,
                "affect_error": self.stats.affect_error,
                "self_consistency": self.stats.self_consistency,
                "predicted_attention": pred.attention_prediction.tolist(),
                "predicted_action": pred.action_prediction,
                "predicted_affect": pred.affect_prediction.tolist(),
                "w_attn_norm": float(np.linalg.norm(self._W_attn)),
                "w_action_norm": float(np.linalg.norm(self._W_action)),
                "w_affect_norm": float(np.linalg.norm(self._W_affect)),
            }

    def reset(self) -> None:
        with self._lock:
            self._last_integrated = np.zeros(self.dim, dtype=np.float64)
            self.stats = SelfSchemaStats()
            self._attn_correct = 0
            self._action_correct = 0
            self._affect_error_sum = 0.0
