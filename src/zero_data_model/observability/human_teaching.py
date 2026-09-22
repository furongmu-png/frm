"""Phase 3 §3.3 人类教学接口。

允许人类在前端对模型的预测或动作给予反馈（👍/👎 或更精细的"正确
方向"提示），反馈转化为额外的预测误差信号：👍 降低该方向的误差权重，
👎 增加。反馈强度随时间衰减，避免过度依赖人类。

教学模式：
- "演示模式"（DEMONSTRATION）：人类手动控制智能体几步，模型观察并
  调整其世界模型以匹配演示。
- "纠正模式"（CORRECTION）：当模型做出错误预测时，人类给出正确标签，
  模型立即更新信念。
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np


# ------------------------------------------------------------------ #
# TeachingMode enum
# ------------------------------------------------------------------ #
class TeachingMode(Enum):
    """教学模式枚举。"""

    NONE = auto()  # 无教学模式
    DEMONSTRATION = auto()  # 演示模式
    CORRECTION = auto()  # 纠正模式


# ------------------------------------------------------------------ #
# FeedbackType enum
# ------------------------------------------------------------------ #
class FeedbackType(Enum):
    """反馈类型：👍 正向 / 👎 负向 / 方向提示。"""

    POSITIVE = auto()  # 👍
    NEGATIVE = auto()  # 👎
    DIRECTION_HINT = auto()  # 正确方向提示


# ------------------------------------------------------------------ #
# FeedbackRecord
# ------------------------------------------------------------------ #
@dataclass
class FeedbackRecord:
    """单次人类反馈记录。"""

    step: int
    feedback_type: FeedbackType
    prediction: Any = None
    actual: Any = None
    error_signal: float = 0.0
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return {
            "step": self.step,
            "feedback_type": self.feedback_type.name,
            "prediction": self.prediction,
            "actual": self.actual,
            "error_signal": self.error_signal,
            "weight": self.weight,
            "metadata": self.metadata,
        }


# ------------------------------------------------------------------ #
# HumanFeedback
# ------------------------------------------------------------------ #
class HumanFeedback:
    """人类教学接口：将人类反馈转化为预测误差信号。

    反馈强度随时间衰减：每次反馈后 ``current_weight`` 按
    ``decay_rate`` 递减，避免模型过度依赖人类。当 ``total_feedback``
    达到 ``max_feedback`` 时自动停止接受反馈。

    Parameters
    ----------
    decay_rate:
        每次反馈后权重的衰减率（0.0–1.0）。例如 0.95 表示每次反馈后
        权重变为原来的 95%。
    positive_strength:
        👍 反馈降低的误差权重幅度。
    negative_strength:
        👎 反馈增加的误差权重幅度。
    max_feedback:
        最多接受的反馈次数（超过后 ``is_active`` 为 False）。
    history_limit:
        反馈历史记录上限。
    """

    def __init__(
        self,
        decay_rate: float = 0.95,
        positive_strength: float = 0.1,
        negative_strength: float = 0.5,
        max_feedback: int = 10000,
        history_limit: int = 5000,
    ) -> None:
        if not 0.0 <= decay_rate <= 1.0:
            raise ValueError("decay_rate 必须在 [0.0, 1.0] 区间内")
        self.decay_rate = decay_rate
        self.positive_strength = positive_strength
        self.negative_strength = negative_strength
        self.max_feedback = max_feedback
        self._current_weight: float = 1.0
        self._mode: TeachingMode = TeachingMode.NONE
        self._total_feedback: int = 0
        self._history: deque[FeedbackRecord] = deque(maxlen=history_limit)
        self._demonstrations: deque[dict[str, Any]] = deque(maxlen=history_limit)
        self._corrections: deque[dict[str, Any]] = deque(maxlen=history_limit)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # 模式管理
    # ------------------------------------------------------------------
    @property
    def mode(self) -> TeachingMode:
        """当前教学模式。"""
        with self._lock:
            return self._mode

    def set_mode(self, mode: TeachingMode) -> None:
        """设置教学模式。"""
        with self._lock:
            self._mode = mode

    @property
    def is_active(self) -> bool:
        """教学接口是否仍接受反馈。"""
        with self._lock:
            return self._total_feedback < self.max_feedback

    @property
    def current_weight(self) -> float:
        """当前反馈权重（随反馈次数衰减）。"""
        with self._lock:
            return self._current_weight

    # ------------------------------------------------------------------
    # 反馈接口
    # ------------------------------------------------------------------
    def give_feedback(
        self,
        step: int,
        feedback_type: FeedbackType,
        prediction: Any = None,
        actual: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> FeedbackRecord:
        """记录一次人类反馈并生成对应的误差信号。

        - POSITIVE (👍): 误差信号为负值，降低该方向的误差权重。
        - NEGATIVE (👎): 误差信号为正值，增加该方向的误差权重。
        - DIRECTION_HINT: 根据 prediction 与 actual 的差异生成方向信号。

        返回 :class:`FeedbackRecord`，其 ``error_signal`` 字段可叠加
        到模型的预测误差上。
        """
        with self._lock:
            if not self.is_active:
                # 已达上限，返回零信号
                return FeedbackRecord(
                    step=step,
                    feedback_type=feedback_type,
                    prediction=prediction,
                    actual=actual,
                    error_signal=0.0,
                    weight=0.0,
                    metadata={"exhausted": True},
                )

            # 计算误差信号
            if feedback_type == FeedbackType.POSITIVE:
                error_signal = -self.positive_strength * self._current_weight
            elif feedback_type == FeedbackType.NEGATIVE:
                error_signal = self.negative_strength * self._current_weight
            elif feedback_type == FeedbackType.DIRECTION_HINT:
                # 方向提示：基于 prediction 与 actual 的差异
                error_signal = self._compute_direction_signal(
                    prediction, actual
                )
            else:
                error_signal = 0.0

            record = FeedbackRecord(
                step=step,
                feedback_type=feedback_type,
                prediction=prediction,
                actual=actual,
                error_signal=error_signal,
                weight=self._current_weight,
                metadata=metadata or {},
            )
            self._history.append(record)
            self._total_feedback += 1
            # 衰减权重
            self._current_weight *= self.decay_rate
            return record

    def _compute_direction_signal(
        self, prediction: Any, actual: Any
    ) -> float:
        """根据 prediction 与 actual 的差异生成方向误差信号。"""
        # 尝试转为 numpy 数组计算差异
        try:
            pred_arr = np.asarray(prediction, dtype=float)
            act_arr = np.asarray(actual, dtype=float)
            if pred_arr.shape == act_arr.shape and pred_arr.size > 0:
                # L2 范数差异，乘以当前权重
                diff = float(np.linalg.norm(pred_arr - act_arr))
                return diff * self._current_weight
        except (ValueError, TypeError):
            pass
        # 无法计算差异，返回中性信号
        return 0.0

    # ------------------------------------------------------------------
    # 演示模式
    # ------------------------------------------------------------------
    def demonstrate(
        self,
        step: int,
        observation: Any,
        action: Any,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """演示模式：记录人类演示的 (observation, action) 对。

        模型可观察这些演示数据，调整其世界模型以匹配演示行为。
        """
        with self._lock:
            demo: dict[str, Any] = {
                "step": step,
                "observation": observation,
                "action": action,
                "metadata": metadata or {},
            }
            self._demonstrations.append(demo)
            return demo

    def get_demonstrations(self, n: int = 20) -> list[dict[str, Any]]:
        """获取最近 N 条演示数据。"""
        with self._lock:
            return list(self._demonstrations)[-n:]

    # ------------------------------------------------------------------
    # 纠正模式
    # ------------------------------------------------------------------
    def correct(
        self,
        step: int,
        prediction: Any,
        correct_label: Any,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """纠正模式：人类给出正确标签，触发模型立即更新信念。

        返回纠正记录，含 prediction、correct_label 和生成的误差信号。
        """
        with self._lock:
            correction: dict[str, Any] = {
                "step": step,
                "prediction": prediction,
                "correct_label": correct_label,
                "metadata": metadata or {},
            }
            # 纠正模式生成方向误差信号
            error_signal = self._compute_direction_signal(
                prediction, correct_label
            )
            correction["error_signal"] = error_signal
            self._corrections.append(correction)
            self._total_feedback += 1
            self._current_weight *= self.decay_rate
            return correction

    def get_corrections(self, n: int = 20) -> list[dict[str, Any]]:
        """获取最近 N 条纠正记录。"""
        with self._lock:
            return list(self._corrections)[-n:]

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------
    @property
    def total_feedback(self) -> int:
        """已接受的反馈总数。"""
        with self._lock:
            return self._total_feedback

    @property
    def n_demonstrations(self) -> int:
        """演示记录数。"""
        with self._lock:
            return len(self._demonstrations)

    @property
    def n_corrections(self) -> int:
        """纠正记录数。"""
        with self._lock:
            return len(self._corrections)

    def get_history(self, n: int = 20) -> list[dict[str, Any]]:
        """获取最近 N 条反馈历史。"""
        with self._lock:
            return [r.to_dict() for r in list(self._history)[-n:]]

    def reset_weight(self) -> None:
        """重置反馈权重为 1.0（重置衰减）。

        在需要"重新开始教学"时调用。
        """
        with self._lock:
            self._current_weight = 1.0

    @property
    def stats(self) -> dict[str, Any]:
        """人类教学接口汇总统计。"""
        with self._lock:
            history = list(self._history)
            n_pos = sum(
                1 for r in history if r.feedback_type == FeedbackType.POSITIVE
            )
            n_neg = sum(
                1 for r in history if r.feedback_type == FeedbackType.NEGATIVE
            )
            n_hint = sum(
                1
                for r in history
                if r.feedback_type == FeedbackType.DIRECTION_HINT
            )
            return {
                "total_feedback": self._total_feedback,
                "n_positive": n_pos,
                "n_negative": n_neg,
                "n_direction_hint": n_hint,
                "n_demonstrations": len(self._demonstrations),
                "n_corrections": len(self._corrections),
                "current_weight": round(self._current_weight, 4),
                "is_active": self.is_active,
                "mode": self._mode.name,
            }


__all__: list[str] = [
    "FeedbackRecord",
    "FeedbackType",
    "HumanFeedback",
    "TeachingMode",
]
