"""GWT Workspace Broadcaster — 广播胜者并同步模块。

广播内容包含选中隐向量 + 置信度标记。接收模块按置信度调整自身：
高置信→更信赖广播、低置信→保留自身信念。所有模块共享全局时间步。
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .attention_selector import AttentionSelector, BroadcastCandidate, CompetitionResult


class WorkspaceBroadcaster:
    """工作空间广播与同步。

    Parameters
    ----------
    selector : AttentionSelector
        竞争选择器。
    alignment_lr : float, default 0.1
        接收模块对齐广播的学习率（高置信时）。
    """

    def __init__(
        self,
        selector: AttentionSelector,
        alignment_lr: float = 0.1,
    ) -> None:
        if alignment_lr <= 0 or alignment_lr > 1:
            raise ValueError(f"alignment_lr must be in (0, 1], got {alignment_lr}")
        self.selector = selector
        self.alignment_lr = float(alignment_lr)

        #: 已注册的接收模块：name → update_callback(broadcast, confidence)
        self._receivers: dict[str, Callable[[np.ndarray, float], None]] = {}
        #: 最近一次广播
        self._last_broadcast: np.ndarray | None = None
        self._last_confidence: float = 0.0
        self._last_winner: str | None = None
        self._global_timestamp: int = 0

    # ------------------------------------------------------------------ #
    # 接收模块注册
    # ------------------------------------------------------------------ #
    def register(
        self, module_name: str, callback: Callable[[np.ndarray, float], None]
    ) -> None:
        """注册接收模块。

        callback(broadcast_vector, confidence) → None
        模块应在 callback 中按置信度调整自身状态。
        """
        self._receivers[module_name] = callback

    def unregister(self, module_name: str) -> None:
        self._receivers.pop(module_name, None)

    @property
    def registered_modules(self) -> list[str]:
        return list(self._receivers.keys())

    # ------------------------------------------------------------------ #
    # 广播主循环
    # ------------------------------------------------------------------ #
    def broadcast(self, candidates: list[BroadcastCandidate]) -> CompetitionResult:
        """执行竞争选择并广播给所有注册模块。

        Returns
        -------
        CompetitionResult
            竞争结果（含胜者与广播向量）。
        """
        result = self.selector.compete(candidates)
        if result.winner is None:
            return result

        self._last_broadcast = result.broadcast
        self._last_confidence = result.broadcast_confidence
        self._last_winner = result.winner.module_name
        self._global_timestamp = result.timestamp

        # 广播给所有注册模块（胜者也收到，确认自身胜出）
        for name, callback in self._receivers.items():
            try:
                # 接收模块按置信度调整：高置信→强对齐，低置信→弱更新
                effective_lr = self.alignment_lr * result.broadcast_confidence
                # 抑制中的模块保留更多自身信念
                if self.selector.get_inhibition(name) > 0:
                    effective_lr *= 0.1
                callback(result.broadcast, effective_lr)
            except Exception:
                # 单模块接收失败不影响其他模块
                continue

        return result

    # ------------------------------------------------------------------ #
    # 访问器
    # ------------------------------------------------------------------ #
    @property
    def last_broadcast(self) -> np.ndarray | None:
        return self._last_broadcast

    @property
    def last_confidence(self) -> float:
        return self._last_confidence

    @property
    def last_winner(self) -> str | None:
        return self._last_winner

    @property
    def global_timestamp(self) -> int:
        """全局意识时间步（所有模块共享）。"""
        return self._global_timestamp

    def snapshot(self) -> dict:
        return {
            "alignment_lr": self.alignment_lr,
            "n_receivers": len(self._receivers),
            "registered_modules": list(self._receivers.keys()),
            "last_winner": self._last_winner,
            "last_confidence": round(self._last_confidence, 6),
            "global_timestamp": self._global_timestamp,
            "last_broadcast_3d": (
                [round(float(x), 6) for x in self._last_broadcast[:3]]
                if self._last_broadcast is not None
                else [0.0, 0.0, 0.0]
            ),
            "selector": self.selector.snapshot(),
        }
