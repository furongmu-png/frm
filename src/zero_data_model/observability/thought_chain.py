"""Phase 3 §3.1 思维链可视化。

在模型 ``think()`` 中记录每个模块的内部状态变化序列（信念前/后、
预测误差、更新量），在多层级 PCN 架构中记录自底向上和自顶向下的
信息流。

前端"思维链视图"水平时间轴展示最近 N 步的思维过程，每个节点可
展开显示详细状态，高亮异常步骤（高惊讶度、元认知触发、工具调用
决策点），支持步进回放。
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any


# ------------------------------------------------------------------ #
# ThoughtRecord dataclass
# ------------------------------------------------------------------ #
@dataclass
class ThoughtRecord:
    """单步思维记录：一个认知周期内的模块状态变化。"""

    step: int
    timestamp: float = 0.0
    # 信念状态变化：前/后快照（截断到前 N 维）
    belief_before: list[float] = field(default_factory=list)
    belief_after: list[float] = field(default_factory=list)
    # 预测误差（各层）
    prediction_errors: dict[str, float] = field(default_factory=dict)
    # 更新量（参数变化的 Frobenius 范数）
    update_magnitude: float = 0.0
    # 元认知状态
    confidence: float = 1.0
    meta_triggered: bool = False
    # 工具调用（若有）
    tool_called: str = ""
    # 自由能
    free_energy: float = 0.0
    # 异常标记
    is_anomaly: bool = False
    anomaly_reason: str = ""
    # 自定义元数据
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return {
            "step": self.step,
            "timestamp": self.timestamp,
            "belief_before": self.belief_before,
            "belief_after": self.belief_after,
            "prediction_errors": self.prediction_errors,
            "update_magnitude": self.update_magnitude,
            "confidence": self.confidence,
            "meta_triggered": self.meta_triggered,
            "tool_called": self.tool_called,
            "free_energy": self.free_energy,
            "is_anomaly": self.is_anomaly,
            "anomaly_reason": self.anomaly_reason,
            "metadata": self.metadata,
        }


# ------------------------------------------------------------------ #
# ThoughtChain
# ------------------------------------------------------------------ #
class ThoughtChain:
    """思维链：记录和回放模型的认知过程。

    维护一个有界的 :class:`ThoughtRecord` 序列，支持：
    - 追加记录
    - 异常检测（高惊讶度、元认知触发、工具调用）
    - 步进回放
    - 查询最近的异常步骤

    Parameters
    ----------
    max_records:
        最大记录数（超出后丢弃最旧）。
    anomaly_pe_threshold:
        预测误差超过此值标记为异常。
    anomaly_confidence_threshold:
        置信度低于此值标记为异常。
    """

    def __init__(
        self,
        max_records: int = 5000,
        anomaly_pe_threshold: float = 2.0,
        anomaly_confidence_threshold: float = 0.3,
    ) -> None:
        self.max_records = max_records
        self.anomaly_pe_threshold = anomaly_pe_threshold
        self.anomaly_confidence_threshold = anomaly_confidence_threshold
        self._records: deque[ThoughtRecord] = deque(maxlen=max_records)
        self._lock = threading.RLock()

    def record(
        self,
        step: int,
        belief_before: list[float] | None = None,
        belief_after: list[float] | None = None,
        prediction_errors: dict[str, float] | None = None,
        update_magnitude: float = 0.0,
        confidence: float = 1.0,
        meta_triggered: bool = False,
        tool_called: str = "",
        free_energy: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> ThoughtRecord:
        """记录一步思维过程。

        自动检测异常并标记 :attr:`ThoughtRecord.is_anomaly`。
        """
        with self._lock:
            # 异常检测
            pe_values = list((prediction_errors or {}).values())
            max_pe = max(pe_values) if pe_values else 0.0
            is_anomaly = (
                max_pe > self.anomaly_pe_threshold
                or confidence < self.anomaly_confidence_threshold
                or meta_triggered
                or bool(tool_called)
            )
            reason = ""
            if max_pe > self.anomaly_pe_threshold:
                reason = f"高预测误差 {max_pe:.4f}"
            elif confidence < self.anomaly_confidence_threshold:
                reason = f"低置信度 {confidence:.4f}"
            elif meta_triggered:
                reason = "元认知触发"
            elif tool_called:
                reason = f"工具调用: {tool_called}"

            rec = ThoughtRecord(
                step=step,
                belief_before=belief_before or [],
                belief_after=belief_after or [],
                prediction_errors=prediction_errors or {},
                update_magnitude=update_magnitude,
                confidence=confidence,
                meta_triggered=meta_triggered,
                tool_called=tool_called,
                free_energy=free_energy,
                is_anomaly=is_anomaly,
                anomaly_reason=reason,
                metadata=metadata or {},
            )
            self._records.append(rec)
            return rec

    def get_recent(self, n: int = 20) -> list[dict[str, Any]]:
        """获取最近 N 步记录。"""
        with self._lock:
            records = list(self._records)
            return [r.to_dict() for r in records[-n:]]

    def get_anomalies(self, n: int = 20) -> list[dict[str, Any]]:
        """获取最近 N 个异常步骤。"""
        with self._lock:
            anomalies = [r for r in self._records if r.is_anomaly]
            return [r.to_dict() for r in anomalies[-n:]]

    def get_at_step(self, step: int) -> dict[str, Any] | None:
        """获取指定步的记录。"""
        with self._lock:
            for r in self._records:
                if r.step == step:
                    return r.to_dict()
            return None

    def get_range(
        self, start_step: int, end_step: int
    ) -> list[dict[str, Any]]:
        """获取步范围内的记录（含两端）。"""
        with self._lock:
            return [
                r.to_dict()
                for r in self._records
                if start_step <= r.step <= end_step
            ]

    def replay(self) -> list[dict[str, Any]]:
        """返回全部记录用于回放。"""
        with self._lock:
            return [r.to_dict() for r in self._records]

    def clear(self) -> None:
        """清空记录。"""
        with self._lock:
            self._records.clear()

    @property
    def length(self) -> int:
        """当前记录数。"""
        with self._lock:
            return len(self._records)

    @property
    def n_anomalies(self) -> int:
        """异常步骤数。"""
        with self._lock:
            return sum(1 for r in self._records if r.is_anomaly)

    @property
    def stats(self) -> dict[str, Any]:
        """思维链统计。"""
        with self._lock:
            records = list(self._records)
            if not records:
                return {
                    "n_records": 0,
                    "n_anomalies": 0,
                    "mean_confidence": 0.0,
                    "mean_pe": 0.0,
                }
            mean_conf = sum(r.confidence for r in records) / len(records)
            all_pes = [
                pe for r in records for pe in r.prediction_errors.values()
            ]
            mean_pe = sum(all_pes) / len(all_pes) if all_pes else 0.0
            return {
                "n_records": len(records),
                "n_anomalies": sum(1 for r in records if r.is_anomaly),
                "mean_confidence": round(mean_conf, 4),
                "mean_pe": round(mean_pe, 4),
                "n_tool_calls": sum(1 for r in records if r.tool_called),
            }


__all__: list[str] = ["ThoughtRecord", "ThoughtChain"]
