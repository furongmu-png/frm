"""Phase 3 §3.4 审计与安全日志。

记录所有关键决策（动作选择、工具调用、元认知触发、人类干预）的时间戳、
上下文和结果，以结构化 JSON 格式存储，支持查询和可视化导出。

伦理约束：``ValueVector`` 提供可配置的偏好状态（如"避免伤害"、"追求
真相"），在自由能计算中作为额外先验项，使决策偏向符合价值观的行为。
"""
from __future__ import annotations

import csv
import io
import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


# ------------------------------------------------------------------ #
# AuditEventType enum
# ------------------------------------------------------------------ #
class AuditEventType(Enum):
    """审计事件类型。"""

    ACTION_SELECTION = auto()  # 动作选择
    TOOL_CALL = auto()  # 工具调用
    METACOGNITION_TRIGGER = auto()  # 元认知触发
    HUMAN_INTERVENTION = auto()  # 人类干预
    COMMUNICATION = auto()  # 通信事件
    CULTURE_TRANSFER = auto()  # 文化传递
    VALUE_VIOLATION = auto()  # 价值观违反
    EMERGENCY_STOP = auto()  # 紧急停止
    OTHER = auto()  # 其他


# ------------------------------------------------------------------ #
# AuditEntry
# ------------------------------------------------------------------ #
@dataclass
class AuditEntry:
    """单条审计日志记录。"""

    timestamp: float
    event_type: AuditEventType
    module: str
    context: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    entry_id: int = 0

    def to_dict(self) -> dict[str, Any]:
        """序列化为可 JSON 化的字典。"""
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type.name,
            "module": self.module,
            "context": self.context,
            "result": self.result,
            "metadata": self.metadata,
        }


# ------------------------------------------------------------------ #
# ValueVector
# ------------------------------------------------------------------ #
@dataclass
class ValueDimension:
    """单个价值观维度。"""

    name: str
    description: str
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "weight": self.weight,
        }


class ValueVector:
    """价值观向量：可配置的偏好状态，作为自由能计算的额外先验项。

    每个维度代表一项价值观（如"避免伤害"、"追求真相"），其 ``weight``
    越大，对应的违反行为受到的自由能惩罚越高。

    通过 :meth:`compute_value_penalty` 计算某行为对各价值观的违反程度，
    返回一个非负惩罚值，叠加到该行为的预期自由能上。
    """

    def __init__(self, dimensions: list[ValueDimension] | None = None) -> None:
        self._dimensions: dict[str, ValueDimension] = {}
        self._lock = threading.RLock()
        for dim in dimensions or []:
            self._dimensions[dim.name] = dim

    def add_dimension(
        self, name: str, description: str, weight: float = 1.0
    ) -> ValueDimension:
        """添加一个价值观维度。"""
        with self._lock:
            dim = ValueDimension(
                name=name, description=description, weight=weight
            )
            self._dimensions[name] = dim
            return dim

    def remove_dimension(self, name: str) -> bool:
        """移除一个价值观维度。"""
        with self._lock:
            if name in self._dimensions:
                del self._dimensions[name]
                return True
            return False

    def get_dimension(self, name: str) -> ValueDimension | None:
        """获取指定价值观维度。"""
        with self._lock:
            return self._dimensions.get(name)

    def set_weight(self, name: str, weight: float) -> bool:
        """调整某维度的权重。"""
        with self._lock:
            dim = self._dimensions.get(name)
            if dim is None:
                return False
            dim.weight = weight
            return True

    def compute_value_penalty(
        self, action_violations: dict[str, float]
    ) -> float:
        """计算行为对价值观的违反惩罚。

        Parameters
        ----------
        action_violations:
            ``{dimension_name: violation_score}``，violation_score ∈ [0, ∞)
            表示该行为对该价值观的违反程度（0 = 完全符合）。

        Returns
        -------
        float
            非负惩罚值，``Σ violation_score * weight``，叠加到自由能。
        """
        with self._lock:
            penalty = 0.0
            for name, violation in action_violations.items():
                dim = self._dimensions.get(name)
                if dim is not None:
                    penalty += float(violation) * dim.weight
            return penalty

    def list_dimensions(self) -> list[ValueDimension]:
        """列出所有价值观维度。"""
        with self._lock:
            return list(self._dimensions.values())

    @property
    def dimension_names(self) -> list[str]:
        """所有维度名列表。"""
        with self._lock:
            return list(self._dimensions.keys())

    @property
    def stats(self) -> dict[str, Any]:
        """价值观向量汇总。"""
        with self._lock:
            return {
                "n_dimensions": len(self._dimensions),
                "dimensions": [
                    d.to_dict() for d in self._dimensions.values()
                ],
            }

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        with self._lock:
            return {
                "dimensions": [
                    d.to_dict() for d in self._dimensions.values()
                ],
            }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValueVector:
        """从字典恢复。"""
        dims = [
            ValueDimension(
                name=d["name"],
                description=d["description"],
                weight=float(d.get("weight", 1.0)),
            )
            for d in data.get("dimensions", [])
        ]
        return cls(dimensions=dims)


# ------------------------------------------------------------------ #
# AuditLogger
# ------------------------------------------------------------------ #
class AuditLogger:
    """审计日志记录器。

    记录所有关键决策（动作选择、工具调用、元认知触发、人类干预）的
    时间戳、上下文和结果，以结构化 JSON 格式存储，支持按时间、模块、
    事件类型过滤查询，并支持导出为 CSV。

    Parameters
    ----------
    max_entries:
        日志条目上限（超出丢弃最旧）。
    value_vector:
        可选的 :class:`ValueVector`，用于在日志中记录价值观违反情况。
    """

    def __init__(
        self,
        max_entries: int = 10000,
        value_vector: ValueVector | None = None,
    ) -> None:
        self._entries: deque[AuditEntry] = deque(maxlen=max_entries)
        self._value_vector = value_vector
        self._next_id: int = 1
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # 日志记录
    # ------------------------------------------------------------------
    def log(
        self,
        event_type: AuditEventType,
        module: str,
        context: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """记录一条审计日志。"""
        with self._lock:
            entry = AuditEntry(
                timestamp=time.time(),
                event_type=event_type,
                module=module,
                context=context or {},
                result=result or {},
                metadata=metadata or {},
                entry_id=self._next_id,
            )
            self._next_id += 1
            self._entries.append(entry)
            return entry

    def log_value_violation(
        self,
        module: str,
        action_violations: dict[str, float],
        context: dict[str, Any] | None = None,
    ) -> AuditEntry | None:
        """记录价值观违反事件。

        如果未配置 :class:`ValueVector`，返回 None。
        """
        with self._lock:
            if self._value_vector is None:
                return None
            penalty = self._value_vector.compute_value_penalty(
                action_violations
            )
            entry = self.log(
                event_type=AuditEventType.VALUE_VIOLATION,
                module=module,
                context=context or {},
                result={
                    "violations": action_violations,
                    "penalty": penalty,
                },
            )
            return entry

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def query(
        self,
        event_type: AuditEventType | None = None,
        module: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """按条件查询审计日志。

        所有过滤参数均为可选，None 表示不过滤该维度。
        """
        with self._lock:
            entries = list(self._entries)
        results: list[dict[str, Any]] = []
        for entry in entries:
            if event_type is not None and entry.event_type != event_type:
                continue
            if module is not None and entry.module != module:
                continue
            if start_time is not None and entry.timestamp < start_time:
                continue
            if end_time is not None and entry.timestamp > end_time:
                continue
            results.append(entry.to_dict())
            if limit is not None and len(results) >= limit:
                break
        return results

    def get_recent(self, n: int = 20) -> list[dict[str, Any]]:
        """获取最近 N 条日志。"""
        with self._lock:
            return [e.to_dict() for e in list(self._entries)[-n:]]

    def get_by_event_type(
        self, event_type: AuditEventType, n: int = 20
    ) -> list[dict[str, Any]]:
        """获取指定事件类型的最近 N 条日志。"""
        with self._lock:
            filtered = [
                e for e in self._entries if e.event_type == event_type
            ]
            return [e.to_dict() for e in filtered[-n:]]

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------
    def export_json(self) -> str:
        """导出为 JSON 字符串。"""
        with self._lock:
            entries = [e.to_dict() for e in self._entries]
        return json.dumps(entries, ensure_ascii=False, default=str)

    def export_csv(self) -> str:
        """导出为 CSV 字符串。

        列：entry_id, timestamp, event_type, module, context, result, metadata
        （context/result/metadata 以 JSON 字符串形式存储）。
        """
        with self._lock:
            entries = list(self._entries)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            [
                "entry_id",
                "timestamp",
                "event_type",
                "module",
                "context",
                "result",
                "metadata",
            ]
        )
        for entry in entries:
            writer.writerow(
                [
                    entry.entry_id,
                    f"{entry.timestamp:.6f}",
                    entry.event_type.name,
                    entry.module,
                    json.dumps(entry.context, ensure_ascii=False, default=str),
                    json.dumps(entry.result, ensure_ascii=False, default=str),
                    json.dumps(entry.metadata, ensure_ascii=False, default=str),
                ]
            )
        return output.getvalue()

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------
    @property
    def n_entries(self) -> int:
        """日志总数。"""
        with self._lock:
            return len(self._entries)

    @property
    def value_vector(self) -> ValueVector | None:
        """当前关联的价值观向量。"""
        return self._value_vector

    def set_value_vector(self, value_vector: ValueVector) -> None:
        """设置/替换价值观向量。"""
        with self._lock:
            self._value_vector = value_vector

    @property
    def stats(self) -> dict[str, Any]:
        """审计日志汇总统计。"""
        with self._lock:
            entries = list(self._entries)
            if not entries:
                return {
                    "n_entries": 0,
                    "by_event_type": {},
                    "by_module": {},
                }
            by_type: dict[str, int] = {}
            by_module: dict[str, int] = {}
            for e in entries:
                type_name = e.event_type.name
                by_type[type_name] = by_type.get(type_name, 0) + 1
                by_module[e.module] = by_module.get(e.module, 0) + 1
            return {
                "n_entries": len(entries),
                "by_event_type": by_type,
                "by_module": by_module,
                "n_value_violations": by_type.get("VALUE_VIOLATION", 0),
            }

    def clear(self) -> None:
        """清空日志。"""
        with self._lock:
            self._entries.clear()
            self._next_id = 1


__all__: list[str] = [
    "AuditEntry",
    "AuditEventType",
    "AuditLogger",
    "ValueDimension",
    "ValueVector",
]
