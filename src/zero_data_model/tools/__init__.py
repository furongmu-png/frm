"""Phase 3 §2 工具使用包。

提供模型自主调用外部 API 的能力，通过自由能最小化学习何时使用工具。
"""
from __future__ import annotations

from .safety import SafetyChecker, SafetyViolation
from .tool_policy import ToolPolicy
from .tool_registry import (
    CalculatorTool,
    CodeExecutorTool,
    DatabaseQueryTool,
    SearchTool,
    Tool,
    ToolRegistry,
    ToolResult,
)

__all__: list[str] = [
    "CalculatorTool",
    "CodeExecutorTool",
    "DatabaseQueryTool",
    "SearchTool",
    "SafetyChecker",
    "SafetyViolation",
    "Tool",
    "ToolPolicy",
    "ToolRegistry",
    "ToolResult",
]
