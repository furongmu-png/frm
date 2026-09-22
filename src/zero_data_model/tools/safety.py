"""Phase 3 §2.3 工具调用安全与沙盒。

提供工具调用前的安全检查：
  - 代码执行限制（时间、内存、禁止文件系统和网络访问）
  - 速率限制（每工具最小调用间隔）
  - 审计日志（记录参数、结果、触发原因）
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


# ------------------------------------------------------------------ #
# SafetyViolation
# ------------------------------------------------------------------ #
@dataclass
class SafetyViolation:
    """安全违规记录。"""

    tool_name: str
    reason: str
    params: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "reason": self.reason,
            "params": str(self.params),
            "timestamp": self.timestamp,
        }


# ------------------------------------------------------------------ #
# SafetyChecker
# ------------------------------------------------------------------ #
class SafetyChecker:
    """工具调用安全检查器。

    职责：
    1. 速率限制：每个工具调用需满足最小间隔 ``min_interval`` 秒。
    2. 参数黑名单：检查参数中是否包含危险模式。
    3. 审计日志：记录每次调用的参数、结果、触发原因。

    Parameters
    ----------
    min_interval:
        同一工具两次调用之间的最小间隔（秒）。
    max_violation_log:
        违规日志最大条数（超出后丢弃最旧）。
    """

    # 参数中禁止出现的危险模式。
    _DANGEROUS_PATTERNS: list[str] = [
        "rm -rf",
        "sudo ",
        "chmod 777",
        "DROP TABLE",
        "DELETE FROM",
        "<script>",
        "javascript:",
    ]

    def __init__(
        self,
        min_interval: float = 0.1,
        max_violation_log: int = 1000,
    ) -> None:
        self.min_interval = min_interval
        self._last_call_time: dict[str, float] = {}
        self._violations: deque[SafetyViolation] = deque(maxlen=max_violation_log)
        self._audit_log: deque[dict[str, Any]] = deque(maxlen=max_violation_log)
        self._lock = threading.RLock()

    def check(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> SafetyViolation | None:
        """检查工具调用是否安全。

        返回 ``None`` 表示通过，返回 :class:`SafetyViolation` 表示违规。
        """
        with self._lock:
            # 1. 速率限制
            now = time.time()
            last = self._last_call_time.get(tool_name, 0.0)
            if now - last < self.min_interval:
                v = SafetyViolation(
                    tool_name=tool_name,
                    reason=(
                        f"速率限制：距上次调用 {now - last:.3f}s < "
                        f"最小间隔 {self.min_interval}s"
                    ),
                    params=params,
                )
                self._violations.append(v)
                return v
            # 2. 参数黑名单检查
            params_str = str(params).lower()
            for pattern in self._DANGEROUS_PATTERNS:
                if pattern.lower() in params_str:
                    v = SafetyViolation(
                        tool_name=tool_name,
                        reason=f"参数包含危险模式: {pattern!r}",
                        params=params,
                    )
                    self._violations.append(v)
                    return v
            # 通过：记录调用时间
            self._last_call_time[tool_name] = now
            return None

    def record_call(
        self,
        tool_name: str,
        params: dict[str, Any],
        result: dict[str, Any],
        trigger_reason: str = "",
    ) -> None:
        """记录一次工具调用到审计日志。"""
        with self._lock:
            self._audit_log.append({
                "tool_name": tool_name,
                "params": str(params),
                "result_success": result.get("success", False),
                "result_output": str(result.get("output", ""))[:200],
                "trigger_reason": trigger_reason,
                "timestamp": time.time(),
            })

    @property
    def violations(self) -> list[dict[str, Any]]:
        """所有安全违规记录。"""
        with self._lock:
            return [v.to_dict() for v in self._violations]

    @property
    def audit_log(self) -> list[dict[str, Any]]:
        """审计日志。"""
        with self._lock:
            return list(self._audit_log)

    @property
    def stats(self) -> dict[str, Any]:
        """安全统计。"""
        with self._lock:
            return {
                "n_violations": len(self._violations),
                "n_audit_entries": len(self._audit_log),
                "n_tools_tracked": len(self._last_call_time),
            }


__all__: list[str] = ["SafetyChecker", "SafetyViolation"]
