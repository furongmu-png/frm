# src/zero_data_model/metacog/meta_trigger.py
"""元认知触发机制（MetaTrigger）。

第二阶段 §2.2：当不确定度超过动态阈值（历史均值 + 2σ）时，自动
启动信息寻求行为：

1. **主动信息寻求**：向实验规划器发送"需要更多关于 X 的信息"信号。
2. **降低学习率**：临时提高精度权重（precision），减少探索幅度，
   避免灾难性错误。
3. **自我提问**：生成自然语言问题（如"What is X?"），送入百科
   阅读器搜索答案。

仅依赖 ``numpy``，独立、自包含、可插拔。
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Any

import numpy as np

# 学习率缩放的下限：避免完全冻结学习。
MIN_LR_SCALE = 0.1


class MetaTrigger:
    """元认知触发机制：不确定度超阈值时启动信息寻求。

    Parameters
    ----------
    threshold_sigma:
        动态阈值的 σ 倍数（默认 2.0，即均值 + 2σ）。
    history_window:
        不确定度历史窗口大小（用于计算均值和标准差）。
    min_history:
        触发所需的最小历史样本数（避免早期误触发）。
    lr_scale_factor:
        触发时学习率缩放因子（默认 0.5，即学习率减半）。
    seed:
        随机种子。

    Attributes
    ----------
    triggered:
        当前是否处于触发状态。
    learning_rate_scale:
        当前学习率缩放因子（1.0=正常，<1.0=谨慎）。
    """

    def __init__(
        self,
        threshold_sigma: float = 2.0,
        history_window: int = 100,
        min_history: int = 5,
        lr_scale_factor: float = 0.5,
        seed: int = 42,
    ) -> None:
        """初始化元认知触发器。"""
        self._threshold_sigma = float(threshold_sigma)
        self._min_history = int(min_history)
        self._lr_scale_factor = max(
            MIN_LR_SCALE, min(1.0, float(lr_scale_factor))
        )
        self._rng = np.random.default_rng(seed)
        self._lock = threading.RLock()
        # 不确定度历史（用于动态阈值计算）。
        self._uncertainty_history: deque[float] = deque(
            maxlen=history_window
        )
        # 当前状态。
        self._triggered: bool = False
        self._learning_rate_scale: float = 1.0
        self._current_threshold: float = 1.0
        self._step_count: int = 0
        # 待处理的信息寻求请求与自我提问。
        self._pending_info_requests: list[str] = []
        self._pending_questions: list[str] = []
        # 触发事件历史（供故事模式）。
        self._trigger_events: deque[dict[str, Any]] = deque(maxlen=200)

    # ------------------------------------------------------------------ #
    # 核心更新
    # ------------------------------------------------------------------ #
    def update(self, uncertainty: float, topic: str = "") -> dict[str, Any]:
        """更新不确定度历史并检查触发条件。

        Parameters
        ----------
        uncertainty:
            当前不确定度（标量，``[0, 1]``）。
        topic:
            当前不确定的主题（用于生成自我提问）。

        Returns
        -------
        dict
            ``{"triggered": bool, "threshold": float,
            "learning_rate_scale": float, "mode": str}``。
        """
        unc = float(uncertainty) if np.isfinite(uncertainty) else 1.0
        unc = max(0.0, min(1.0, unc))
        with self._lock:
            self._uncertainty_history.append(unc)
            self._step_count += 1

            # 计算动态阈值：历史均值 + threshold_sigma * 历史标准差。
            if len(self._uncertainty_history) >= self._min_history:
                hist = list(self._uncertainty_history)
                mean_u = float(np.mean(hist))
                std_u = float(np.std(hist))
                self._current_threshold = mean_u + self._threshold_sigma * std_u
                # 阈值不应低于 0.5（避免过度触发）。
                self._current_threshold = max(0.5, self._current_threshold)
            else:
                # 历史不足，使用保守阈值。
                self._current_threshold = 0.8

            # 检查是否触发。
            should_trigger = unc > self._current_threshold
            if should_trigger and not self._triggered:
                # 新触发事件。
                self._trigger(unc, topic)
            elif not should_trigger and self._triggered:
                # 解除触发。
                self._untrigger()

            return {
                "triggered": self._triggered,
                "threshold": self._current_threshold,
                "learning_rate_scale": self._learning_rate_scale,
                "mode": "cautious" if self._triggered else "normal",
                "uncertainty": unc,
            }

    def _trigger(self, uncertainty: float, topic: str) -> None:
        """执行触发：降低学习率、生成信息寻求请求与自我提问。"""
        self._triggered = True
        self._learning_rate_scale = self._lr_scale_factor
        # 主动信息寻求请求。
        if topic:
            request = f"need_more_info_about_{topic}"
            self._pending_info_requests.append(request)
            # 自我提问。
            question = self.generate_question(topic)
            self._pending_questions.append(question)
        # 记录触发事件（供故事模式）。
        event: dict[str, Any] = {
            "type": "meta_trigger",
            "event": "uncertainty_exceeded",
            "step": self._step_count,
            "uncertainty": float(uncertainty),
            "threshold": float(self._current_threshold),
            "topic": topic,
            "mode": "cautious",
        }
        self._trigger_events.append(event)

    def _untrigger(self) -> None:
        """解除触发：恢复正常学习率。"""
        self._triggered = False
        self._learning_rate_scale = 1.0

    # ------------------------------------------------------------------ #
    # 自我提问
    # ------------------------------------------------------------------ #
    def generate_question(self, topic: str) -> str:
        """生成关于不确定主题的自然语言问题。

        Parameters
        ----------
        topic:
            不确定的主题（如 ``"collision_dynamics"``）。

        Returns
        -------
        str
            自然语言问题（如 ``"What is collision_dynamics?"``）。
        """
        # 将下划线/驼峰转为可读形式。
        readable = topic.replace("_", " ").replace("-", " ").strip()
        if not readable:
            readable = "this topic"
        return f"What is {readable}?"

    # ------------------------------------------------------------------ #
    # 查询接口
    # ------------------------------------------------------------------ #
    def should_seek_info(self) -> bool:
        """是否应触发主动信息寻求。"""
        with self._lock:
            return self._triggered

    def get_pending_info_requests(self) -> list[str]:
        """获取并清空待处理的信息寻求请求。"""
        with self._lock:
            reqs = list(self._pending_info_requests)
            self._pending_info_requests.clear()
            return reqs

    def get_pending_questions(self) -> list[str]:
        """获取并清空待处理的自我提问。"""
        with self._lock:
            qs = list(self._pending_questions)
            self._pending_questions.clear()
            return qs

    def get_trigger_events(self) -> list[dict[str, Any]]:
        """返回触发事件历史（供故事模式）。"""
        with self._lock:
            return list(self._trigger_events)

    # ------------------------------------------------------------------ #
    # 属性
    # ------------------------------------------------------------------ #
    @property
    def triggered(self) -> bool:
        """当前是否处于触发状态。"""
        with self._lock:
            return self._triggered

    @property
    def learning_rate_scale(self) -> float:
        """当前学习率缩放因子（1.0=正常，<1.0=谨慎）。"""
        with self._lock:
            return self._learning_rate_scale

    @property
    def current_threshold(self) -> float:
        """当前动态阈值。"""
        with self._lock:
            return self._current_threshold

    @property
    def stats(self) -> dict[str, Any]:
        """统计信息。"""
        with self._lock:
            hist = list(self._uncertainty_history)
            return {
                "step_count": self._step_count,
                "triggered": self._triggered,
                "learning_rate_scale": self._learning_rate_scale,
                "current_threshold": self._current_threshold,
                "mean_uncertainty": float(np.mean(hist)) if hist else 0.0,
                "std_uncertainty": float(np.std(hist)) if hist else 0.0,
                "n_trigger_events": len(self._trigger_events),
                "n_pending_requests": len(self._pending_info_requests),
                "n_pending_questions": len(self._pending_questions),
            }


__all__: list[str] = ["MetaTrigger"]
