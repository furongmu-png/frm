from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

@dataclass
class ToolResult:
    success: bool
    output: str
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]: ...


class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def input_schema(self) -> dict[str, Any]: ...

    @abstractmethod
    def execute(self, params: dict[str, Any]) -> ToolResult: ...


class CalculatorTool(Tool):
    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    @property
    def input_schema(self) -> dict[str, Any]: ...
    def execute(self, params: dict[str, Any]) -> ToolResult: ...


class SearchTool(Tool):
    _local_index: dict[str, str]
    _lock: threading.RLock

    def __init__(self, local_index: dict[str, str] | None = None) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    @property
    def input_schema(self) -> dict[str, Any]: ...
    def add_to_index(self, keyword: str, content: str) -> None: ...
    def execute(self, params: dict[str, Any]) -> ToolResult: ...


class CodeExecutorTool(Tool):
    max_output_chars: int
    max_lines: int

    def __init__(
        self, max_output_chars: int = 4096, max_lines: int = 100
    ) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    @property
    def input_schema(self) -> dict[str, Any]: ...
    def execute(self, params: dict[str, Any]) -> ToolResult: ...


class DatabaseQueryTool(Tool):
    _triples: list[tuple[str, str, str]]
    _db_path: str | None
    _lock: threading.RLock

    def __init__(
        self,
        triples: list[tuple[str, str, str]] | None = None,
        db_path: str | None = None,
    ) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    @property
    def input_schema(self) -> dict[str, Any]: ...
    def add_triple(self, subject: str, relation: str, obj: str) -> None: ...
    def execute(self, params: dict[str, Any]) -> ToolResult: ...


class ToolRegistry:
    _tools: dict[str, Tool]
    _lock: threading.RLock

    def __init__(self) -> None: ...
    def register(self, tool: Tool) -> None: ...
    def unregister(self, name: str) -> bool: ...
    def get(self, name: str) -> Tool | None: ...
    def list_tools(self) -> list[dict[str, Any]]: ...
    @property
    def size(self) -> int: ...
    def execute(self, name: str, params: dict[str, Any]) -> ToolResult: ...


__all__: list[str] = [
    "Tool",
    "ToolResult",
    "ToolRegistry",
    "CalculatorTool",
    "SearchTool",
    "CodeExecutorTool",
    "DatabaseQueryTool",
]
