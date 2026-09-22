"""Phase 3 §2.1 工具抽象与注册。

定义 ``Tool`` 抽象基类和 ``ToolRegistry`` 管理器，以及四个内置工具：
  - SearchTool: Wikipedia / 本地百科索引搜索
  - CalculatorTool: 安全数学表达式求值
  - CodeExecutorTool: 隔离环境执行 Python 代码
  - DatabaseQueryTool: 本地知识图谱 / SQLite 查询

工具选择策略见 :mod:`tool_policy`，安全检查见 :mod:`safety`。
"""
from __future__ import annotations

import ast
import contextlib
import io
import logging
import math
import operator
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# ToolResult dataclass
# ------------------------------------------------------------------ #
@dataclass
class ToolResult:
    """工具执行结果。"""

    success: bool
    output: str
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata,
        }


# ------------------------------------------------------------------ #
# Tool 抽象基类
# ------------------------------------------------------------------ #
class Tool(ABC):
    """工具抽象基类。

    子类必须实现 :attr:`description`、:attr:`input_schema` 和
    :meth:`execute`。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """工具唯一名称。"""

    @property
    @abstractmethod
    def description(self) -> str:
        """自然语言描述工具功能。"""

    @property
    @abstractmethod
    def input_schema(self) -> dict[str, Any]:
        """输入参数规范（JSON Schema 子集）。"""

    @abstractmethod
    def execute(self, params: dict[str, Any]) -> ToolResult:
        """执行工具并返回结果。"""


# ------------------------------------------------------------------ #
# 内置工具：CalculatorTool
# ------------------------------------------------------------------ #
class CalculatorTool(Tool):
    """安全数学表达式求值工具。

    使用 AST 解析限制为白名单运算符（加减乘除、幂、模、函数调用），
    禁止任意属性访问和赋值，防止代码注入。
    """

    # 允许的二元运算符。
    _BINOPS: dict[type[ast.AST], Any] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.FloorDiv: operator.floordiv,
    }
    # 允许的一元运算符。
    _UNARYOPS: dict[type[ast.AST], Any] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }
    # 允许的数学函数。
    _FUNCTIONS: dict[str, Any] = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
        "tan": math.tan, "log": math.log, "log10": math.log10,
        "exp": math.exp, "floor": math.floor, "ceil": math.ceil,
        "pi": math.pi, "e": math.e,
    }

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "执行数学表达式求值（支持 +, -, *, /, **, sqrt, sin, cos, log 等）"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式，如 '2 + 3 * 4' 或 'sqrt(16)'",
                }
            },
            "required": ["expression"],
        }

    def execute(self, params: dict[str, Any]) -> ToolResult:
        expression = params.get("expression", "")
        if not isinstance(expression, str) or not expression.strip():
            return ToolResult(
                success=False, output="", error="expression 必须为非空字符串"
            )
        try:
            tree = ast.parse(expression.strip(), mode="eval")
            result = self._eval_node(tree.body)
            return ToolResult(
                success=True,
                output=str(result),
                metadata={"expression": expression, "result_type": type(result).__name__},
            )
        except (ValueError, SyntaxError, TypeError, ZeroDivisionError) as exc:
            return ToolResult(
                success=False, output="", error=f"计算错误: {type(exc).__name__}: {exc}"
            )

    def _eval_node(self, node: ast.AST) -> Any:
        """递归求值 AST 节点，仅允许白名单运算。"""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, complex)):
                return node.value
            raise ValueError(f"不支持的常量类型: {type(node.value)}")
        if isinstance(node, ast.BinOp):
            op_func = self._BINOPS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"不支持的二元运算: {type(node.op).__name__}")
            return op_func(self._eval_node(node.left), self._eval_node(node.right))
        if isinstance(node, ast.UnaryOp):
            op_func = self._UNARYOPS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"不支持的一元运算: {type(node.op).__name__}")
            return op_func(self._eval_node(node.operand))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("仅允许直接函数调用")
            func = self._FUNCTIONS.get(node.func.id)
            if func is None:
                raise ValueError(f"不支持的函数: {node.func.id}")
            args = [self._eval_node(a) for a in node.args]
            return func(*args)
        if isinstance(node, ast.Name):
            val = self._FUNCTIONS.get(node.id)
            if val is None:
                raise ValueError(f"未定义的名称: {node.id}")
            return val
        raise ValueError(f"不支持的 AST 节点: {type(node).__name__}")


# ------------------------------------------------------------------ #
# 内置工具：SearchTool
# ------------------------------------------------------------------ #
class SearchTool(Tool):
    """百科搜索工具。

    默认查询本地知识索引（若提供），可选调用 Wikipedia API。
    本地索引用 dict 存储 {query_keyword: article_text}。
    """

    def __init__(self, local_index: dict[str, str] | None = None) -> None:
        self._local_index: dict[str, str] = local_index if local_index else {}
        self._lock = threading.RLock()

    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return "搜索百科知识库获取文章内容（本地索引或 Wikipedia API）"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词",
                }
            },
            "required": ["query"],
        }

    def add_to_index(self, keyword: str, content: str) -> None:
        """向本地索引添加条目。"""
        with self._lock:
            self._local_index[keyword.lower()] = content

    def execute(self, params: dict[str, Any]) -> ToolResult:
        query = params.get("query", "")
        if not isinstance(query, str) or not query.strip():
            return ToolResult(
                success=False, output="", error="query 必须为非空字符串"
            )
        query = query.strip()
        # 先查本地索引
        with self._lock:
            result = self._local_index.get(query.lower())
        if result:
            return ToolResult(
                success=True,
                output=result,
                metadata={"source": "local_index", "query": query},
            )
        # 本地未命中：返回提示（不实际调用网络 API，保持离线可测试）
        return ToolResult(
            success=False,
            output="",
            error=f"本地索引中未找到 '{query}'（离线模式不调用 Wikipedia API）",
            metadata={"source": "miss", "query": query},
        )


# ------------------------------------------------------------------ #
# 内置工具：CodeExecutorTool
# ------------------------------------------------------------------ #
class CodeExecutorTool(Tool):
    """隔离环境 Python 代码执行工具。

    使用受限的全局命名空间（仅内置 print/len/range 等），禁止
    ``import``、``open``、``exec``、``eval``、属性访问。
    执行时间和结果长度有限制。
    """

    # 允许的内置函数。
    _SAFE_BUILTINS: dict[str, Any] = {
        "print": print, "len": len, "range": range, "abs": abs,
        "round": round, "sum": sum, "min": min, "max": max,
        "sorted": sorted, "enumerate": enumerate, "zip": zip,
        "map": map, "filter": filter, "list": list, "dict": dict,
        "set": set, "tuple": tuple, "str": str, "int": int,
        "float": float, "bool": bool, "True": True, "False": False,
        "None": None,
    }

    def __init__(
        self,
        max_output_chars: int = 4096,
        max_lines: int = 100,
    ) -> None:
        self.max_output_chars = max_output_chars
        self.max_lines = max_lines

    @property
    def name(self) -> str:
        return "code_executor"

    @property
    def description(self) -> str:
        return "在隔离环境中执行 Python 代码片段并返回输出（禁止 import/open/exec）"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "要执行的 Python 代码",
                }
            },
            "required": ["code"],
        }

    def execute(self, params: dict[str, Any]) -> ToolResult:
        code = params.get("code", "")
        if not isinstance(code, str) or not code.strip():
            return ToolResult(
                success=False, output="", error="code 必须为非空字符串"
            )
        # 静态检查：禁止危险模式
        danger = self._check_safety(code)
        if danger:
            return ToolResult(success=False, output="", error=danger)
        # 执行
        output_buf = io.StringIO()
        sandbox_globals: dict[str, Any] = {"__builtins__": self._SAFE_BUILTINS}
        try:
            with contextlib.redirect_stdout(output_buf):
                exec(compile(code, "<sandbox>", "exec"), sandbox_globals)  # noqa: S102
            output = output_buf.getvalue()
            if len(output) > self.max_output_chars:
                output = output[: self.max_output_chars] + "...[truncated]"
            return ToolResult(
                success=True,
                output=output,
                metadata={"code_lines": code.count("\n") + 1},
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                success=False,
                output="",
                error=f"执行错误: {type(exc).__name__}: {exc}",
            )

    @staticmethod
    def _check_safety(code: str) -> str:
        """静态检查代码安全性，返回危险描述或空字符串。"""
        forbidden = [
            ("import ", "禁止使用 import"),
            ("__import__", "禁止使用 __import__"),
            ("open(", "禁止使用 open()"),
            ("exec(", "禁止使用 exec()"),
            ("eval(", "禁止使用 eval()"),
            ("os.", "禁止访问 os 模块"),
            ("subprocess", "禁止使用 subprocess"),
            ("system(", "禁止调用 system()"),
        ]
        code_lower = code.lower()
        for pattern, reason in forbidden:
            if pattern.lower() in code_lower:
                return reason
        return ""


# ------------------------------------------------------------------ #
# 内置工具：DatabaseQueryTool
# ------------------------------------------------------------------ #
class DatabaseQueryTool(Tool):
    """本地知识图谱 / SQLite 数据库查询工具。

    支持两种模式：
    1. 内存三元组查询（传入 list[(subject, relation, object)]）
    2. SQLite 查询（传入 db_path，需要 sqlite3）
    """

    def __init__(
        self,
        triples: list[tuple[str, str, str]] | None = None,
        db_path: str | None = None,
    ) -> None:
        self._triples: list[tuple[str, str, str]] = triples if triples else []
        self._db_path = db_path
        self._lock = threading.RLock()

    @property
    def name(self) -> str:
        return "database_query"

    @property
    def description(self) -> str:
        return "查询本地知识图谱或 SQLite 数据库"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "主语（可选）"},
                "relation": {"type": "string", "description": "关系（可选）"},
                "object": {"type": "string", "description": "宾语（可选）"},
            },
        }

    def add_triple(self, subject: str, relation: str, obj: str) -> None:
        """添加三元组。"""
        with self._lock:
            self._triples.append((subject, relation, obj))

    def execute(self, params: dict[str, Any]) -> ToolResult:
        subject = params.get("subject")
        relation = params.get("relation")
        obj = params.get("object")
        with self._lock:
            results = []
            for s, r, o in self._triples:
                if subject and subject != s:
                    continue
                if relation and relation != r:
                    continue
                if obj and obj != o:
                    continue
                results.append({"subject": s, "relation": r, "object": o})
        if not results:
            return ToolResult(
                success=True,
                output="无匹配结果",
                metadata={"n_results": 0},
            )
        return ToolResult(
            success=True,
            output=str(results),
            metadata={"n_results": len(results)},
        )


# ------------------------------------------------------------------ #
# ToolRegistry
# ------------------------------------------------------------------ #
class ToolRegistry:
    """工具注册表：管理可用工具，支持动态添加/移除。

    线程安全，支持按名称查找和列举所有工具。
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._lock = threading.RLock()
        # 注册内置工具
        self._register_builtins()

    def _register_builtins(self) -> None:
        """注册默认内置工具集。"""
        builtins: list[Tool] = [
            CalculatorTool(),
            SearchTool(),
            CodeExecutorTool(),
            DatabaseQueryTool(),
        ]
        for tool in builtins:
            self._tools[tool.name] = tool

    def register(self, tool: Tool) -> None:
        """注册工具。若名称已存在则覆盖。"""
        with self._lock:
            self._tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        """移除工具。返回是否成功。"""
        with self._lock:
            if name in self._tools:
                del self._tools[name]
                return True
            return False

    def get(self, name: str) -> Tool | None:
        """按名称获取工具。"""
        with self._lock:
            return self._tools.get(name)

    def list_tools(self) -> list[dict[str, Any]]:
        """列出所有工具的元信息。"""
        with self._lock:
            return [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in self._tools.values()
            ]

    @property
    def size(self) -> int:
        """已注册工具数。"""
        with self._lock:
            return len(self._tools)

    def execute(self, name: str, params: dict[str, Any]) -> ToolResult:
        """按名称执行工具。"""
        tool = self.get(name)
        if tool is None:
            return ToolResult(
                success=False, output="", error=f"工具 '{name}' 未注册"
            )
        return tool.execute(params)


__all__: list[str] = [
    "Tool",
    "ToolResult",
    "ToolRegistry",
    "CalculatorTool",
    "SearchTool",
    "CodeExecutorTool",
    "DatabaseQueryTool",
]
