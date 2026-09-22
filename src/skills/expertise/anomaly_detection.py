"""异常检测技能。

对传感器流（物理沙盒变量）持续计算预测误差，当误差超过动态阈值
（历史均值 + 3σ）时触发告警。结合元认知置信度输出异常解释。
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..base import SkillBase, SkillContext, SkillResult

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# 滑动窗口统计
# ------------------------------------------------------------------ #


@dataclass
class _AnomalyAlert:
    """单条告警。"""
    timestamp: float
    metric: str
    value: float
    threshold: float
    severity: float
    explanation: str


class WindowStats:
    """Welford 在线均值/方差 + 滑动窗口。

    维护最近 ``window_size`` 个观测，同时维护全局均值/方差。
    用于动态阈值计算。
    """

    def __init__(self, window_size: int = 128) -> None:
        self.window_size = window_size
        self._buf: deque[float] = deque(maxlen=window_size)
        self._count: int = 0
        self._mean: float = 0.0
        self._m2: float = 0.0  # Welford 累积平方差

    def push(self, x: float) -> None:
        """更新统计量。"""
        self._buf.append(x)
        # Welford 在线均值/方差（不受窗口影响的全局估计）
        self._count += 1
        delta = x - self._mean
        self._mean += delta / self._count
        delta2 = x - self._mean
        self._m2 += delta * delta2

    @property
    def mean(self) -> float:
        if self._count == 0:
            return 0.0
        return self._mean

    @property
    def std(self) -> float:
        """总体标准差（ddof=0）。

        使用总体方差（除以 n）以便与 ``window_std``（也是 ddof=0）口径一致，
        并保证常量序列返回 0。
        """
        if self._count < 1:
            return 0.0
        var = self._m2 / self._count
        return float(np.sqrt(max(var, 0.0)))

    @property
    def window_mean(self) -> float:
        if not self._buf:
            return 0.0
        return float(np.mean(self._buf))

    @property
    def window_std(self) -> float:
        if len(self._buf) < 2:
            return 0.0
        return float(np.std(self._buf))

    def percentile(self, q: float) -> float:
        if not self._buf:
            return 0.0
        return float(np.percentile(list(self._buf), q * 100))

    @property
    def count(self) -> int:
        return self._count


# ------------------------------------------------------------------ #
# 解释器
# ------------------------------------------------------------------ #


#: 元素方向 → 自然语言短语
_DIRECTION_PHRASES = {
    "velocity": "物体速度",
    "position": "物体位置",
    "acceleration": "物体加速度",
    "error": "预测误差",
    "energy": "系统能量",
    "temperature": "环境温度",
    "pressure": "接触压力",
}


def _explain(
    metric: str,
    value: float,
    threshold: float,
    mean: float,
    confidence: float,
) -> str:
    """生成异常解释文本。"""
    name = _DIRECTION_PHRASES.get(metric, metric)
    direction = "增加" if value > mean else "下降"
    sigma_ratio = (
        abs(value - mean) / max(abs(threshold - mean), 1e-6)
        if threshold != mean
        else 0.0
    )
    conf_phrase = "高置信" if confidence > 0.7 else "中置信" if confidence > 0.4 else "低置信"
    return (
        f"{name}异常{direction}（值={value:.3g}, "
        f"阈值={threshold:.3g}, σ偏差={sigma_ratio:.2f}, {conf_phrase}）"
    )


# ------------------------------------------------------------------ #
# 主技能类
# ------------------------------------------------------------------ #


class AnomalyDetector(SkillBase):
    """异常检测模块。

    用法：
    - 每步 ``update(metric, value)`` 推送新观测。
    - 当 ``value > mean + sigma * std`` 时触发告警。
    - ``process`` 从 ``SkillContext.prediction_error`` 取当前误差，
      作为 ``error`` 度量进行检测。
    """

    name = "anomaly_detection"
    dimension = "expertise"

    def __init__(
        self,
        *,
        sigma_threshold: float = 3.0,
        min_samples: int = 10,
        window_size: int = 128,
        cooldown: int = 5,
        enabled: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(enabled=enabled, **kwargs)
        self.sigma_threshold = sigma_threshold
        self.min_samples = min_samples
        self.cooldown = cooldown  # 同度量连续触发告警的最小间隔（push 次数）
        self._stats: dict[str, WindowStats] = {}
        self._window_size = window_size
        self._alerts: deque[_AnomalyAlert] = deque(maxlen=200)
        self._last_alert_step: dict[str, int] = {}
        self._step: int = 0

    # ------------------------------------------------------------------ #
    # 对外 API
    # ------------------------------------------------------------------ #

    def update(self, metric: str, value: float) -> _AnomalyAlert | None:
        """推送新观测，返回触发的告警（若有）。

        阈值计算使用 ``push`` 之前的窗口（即不含当前值），避免当前
        异常值拉偏自身的判定阈值。
        """
        self._step += 1
        if metric not in self._stats:
            self._stats[metric] = WindowStats(self._window_size)
        stats = self._stats[metric]

        # 在 push 之前取出窗口统计（不含当前值）
        pre_count = stats.count
        pre_window_mean = stats.window_mean
        pre_window_std = stats.window_std

        # 仍要 push 以更新统计
        stats.push(value)

        # 仅当历史样本足够时才判定（不计入当前值）
        if pre_count < self.min_samples:
            return None

        mean = pre_window_mean
        std = pre_window_std
        if std < 1e-9:
            return None  # 无变化
        threshold_high = mean + self.sigma_threshold * std
        threshold_low = mean - self.sigma_threshold * std

        if value > threshold_high or value < threshold_low:
            # 冷却检查
            last = self._last_alert_step.get(metric, -self.cooldown - 1)
            if self._step - last < self.cooldown:
                return None
            self._last_alert_step[metric] = self._step
            threshold = threshold_high if value > threshold_high else threshold_low
            severity = float(
                min(1.0, abs(value - mean) / max(self.sigma_threshold * std, 1e-9))
            )
            alert = _AnomalyAlert(
                timestamp=time.time(),
                metric=metric,
                value=float(value),
                threshold=float(threshold),
                severity=severity,
                explanation=_explain(
                    metric, value, threshold, mean, confidence=0.8
                ),
            )
            self._alerts.append(alert)
            return alert
        return None

    def recent_alerts(self, n: int = 10) -> list[dict[str, Any]]:
        """返回最近 n 条告警。"""
        items = list(self._alerts)[-n:]
        return [
            {
                "timestamp": a.timestamp,
                "metric": a.metric,
                "value": a.value,
                "threshold": a.threshold,
                "severity": a.severity,
                "explanation": a.explanation,
            }
            for a in items
        ]

    def stats_snapshot(self) -> dict[str, Any]:
        return {
            m: {
                "count": s.count,
                "mean": s.mean,
                "std": s.std,
                "window_mean": s.window_mean,
                "window_std": s.window_std,
            }
            for m, s in self._stats.items()
        }

    # ------------------------------------------------------------------ #
    # SkillBase 接口
    # ------------------------------------------------------------------ #

    def process(self, ctx: SkillContext) -> SkillResult:
        """从 ``ctx.prediction_error`` 推送误差作为度量。"""
        error = float(ctx.prediction_error)
        alert = self.update("error", error)
        return SkillResult(
            name=self.name,
            data={
                "current_error": error,
                "alert": (
                    {
                        "metric": alert.metric,
                        "value": alert.value,
                        "threshold": alert.threshold,
                        "severity": alert.severity,
                        "explanation": alert.explanation,
                    }
                    if alert
                    else None
                ),
                "n_alerts": len(self._alerts),
                "recent_alerts": self.recent_alerts(5),
                "stats": self.stats_snapshot(),
            },
        )

    def snapshot(self) -> dict[str, Any]:
        base = super().snapshot()
        base["data"] = base.get("data", {}) | {
            "n_metrics": len(self._stats),
            "n_alerts": len(self._alerts),
        }
        return base
