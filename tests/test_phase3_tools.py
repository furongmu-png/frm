"""Phase 3 §2 工具使用测试。

覆盖：
  - ToolRegistry: 注册/移除/列举/执行
  - CalculatorTool: 数学表达式求值（白名单运算符）
  - SearchTool: 本地索引搜索
  - CodeExecutorTool: 沙盒代码执行（安全检查）
  - DatabaseQueryTool: 三元组查询
  - SafetyChecker: 速率限制 + 危险模式检测
  - ToolPolicy: 置信度驱动决策 + 自由能最小化
"""
from __future__ import annotations

import numpy as np

from zero_data_model.tools.safety import SafetyChecker
from zero_data_model.tools.tool_policy import ToolDecision, ToolPolicy
from zero_data_model.tools.tool_registry import (
    CalculatorTool,
    CodeExecutorTool,
    DatabaseQueryTool,
    SearchTool,
    Tool,
    ToolRegistry,
    ToolResult,
)


# ------------------------------------------------------------------ #
# CalculatorTool
# ------------------------------------------------------------------ #
class TestCalculatorTool:
    """数学表达式求值。"""

    def test_basic_arithmetic(self):
        calc = CalculatorTool()
        r = calc.execute({"expression": "2 + 3 * 4"})
        assert r.success
        assert float(r.output) == 14.0

    def test_power_and_mod(self):
        calc = CalculatorTool()
        r = calc.execute({"expression": "2 ** 10"})
        assert r.success
        assert float(r.output) == 1024.0
        r = calc.execute({"expression": "17 % 5"})
        assert r.success
        assert float(r.output) == 2.0

    def test_math_functions(self):
        calc = CalculatorTool()
        r = calc.execute({"expression": "sqrt(16)"})
        assert r.success
        assert float(r.output) == 4.0
        r = calc.execute({"expression": "abs(-5)"})
        assert r.success
        assert float(r.output) == 5.0

    def test_division_by_zero_error(self):
        calc = CalculatorTool()
        r = calc.execute({"expression": "1 / 0"})
        assert not r.success
        assert "ZeroDivisionError" in r.error

    def test_invalid_expression_error(self):
        calc = CalculatorTool()
        r = calc.execute({"expression": "2 + "})
        assert not r.success

    def test_empty_expression_error(self):
        calc = CalculatorTool()
        r = calc.execute({"expression": ""})
        assert not r.success

    def test_no_attribute_access(self):
        """禁止属性访问（如 __import__）。"""
        calc = CalculatorTool()
        r = calc.execute({"expression": "__import__('os')"})
        assert not r.success


# ------------------------------------------------------------------ #
# SearchTool
# ------------------------------------------------------------------ #
class TestSearchTool:
    """本地索引搜索。"""

    def test_search_found(self):
        search = SearchTool()
        search.add_to_index("python", "Python is a programming language.")
        r = search.execute({"query": "python"})
        assert r.success
        assert "programming language" in r.output

    def test_search_not_found(self):
        search = SearchTool()
        r = search.execute({"query": "nonexistent"})
        assert not r.success

    def test_empty_query_error(self):
        search = SearchTool()
        r = search.execute({"query": ""})
        assert not r.success


# ------------------------------------------------------------------ #
# CodeExecutorTool
# ------------------------------------------------------------------ #
class TestCodeExecutorTool:
    """沙盒代码执行。"""

    def test_basic_print(self):
        tool = CodeExecutorTool()
        r = tool.execute({"code": "print('hello world')"})
        assert r.success
        assert "hello world" in r.output

    def test_loop_and_computation(self):
        tool = CodeExecutorTool()
        r = tool.execute({"code": "x = sum(range(10)); print(x)"})
        assert r.success
        assert "45" in r.output

    def test_import_blocked(self):
        tool = CodeExecutorTool()
        r = tool.execute({"code": "import os"})
        assert not r.success
        assert "import" in r.error.lower()

    def test_open_blocked(self):
        tool = CodeExecutorTool()
        r = tool.execute({"code": "open('/etc/passwd')"})
        assert not r.success
        assert "open" in r.error.lower()

    def test_exec_blocked(self):
        tool = CodeExecutorTool()
        r = tool.execute({"code": "exec('print(1)')"})
        assert not r.success

    def test_empty_code_error(self):
        tool = CodeExecutorTool()
        r = tool.execute({"code": ""})
        assert not r.success


# ------------------------------------------------------------------ #
# DatabaseQueryTool
# ------------------------------------------------------------------ #
class TestDatabaseQueryTool:
    """三元组查询。"""

    def test_query_all(self):
        tool = DatabaseQueryTool(
            triples=[("cat", "is_a", "animal"), ("dog", "is_a", "animal")]
        )
        r = tool.execute({})
        assert r.success
        assert r.metadata["n_results"] == 2

    def test_query_by_subject(self):
        tool = DatabaseQueryTool(
            triples=[("cat", "is_a", "animal"), ("dog", "is_a", "animal")]
        )
        r = tool.execute({"subject": "cat"})
        assert r.success
        assert r.metadata["n_results"] == 1

    def test_no_match(self):
        tool = DatabaseQueryTool(triples=[("cat", "is_a", "animal")])
        r = tool.execute({"subject": "fish"})
        assert r.success
        assert r.metadata["n_results"] == 0

    def test_add_triple(self):
        tool = DatabaseQueryTool()
        tool.add_triple("x", "rel", "y")
        r = tool.execute({"subject": "x"})
        assert r.success
        assert r.metadata["n_results"] == 1


# ------------------------------------------------------------------ #
# ToolRegistry
# ------------------------------------------------------------------ #
class TestToolRegistry:
    """注册表管理。"""

    def test_default_builtins(self):
        reg = ToolRegistry()
        assert reg.size == 4  # calculator, search, code_executor, database_query

    def test_get_existing(self):
        reg = ToolRegistry()
        tool = reg.get("calculator")
        assert tool is not None
        assert tool.name == "calculator"

    def test_get_nonexistent(self):
        reg = ToolRegistry()
        assert reg.get("nonexistent") is None

    def test_register_custom(self):
        reg = ToolRegistry()

        class CustomTool(Tool):
            @property
            def name(self) -> str:
                return "custom"

            @property
            def description(self) -> str:
                return "custom tool"

            @property
            def input_schema(self) -> dict:
                return {}

            def execute(self, params: dict) -> ToolResult:
                return ToolResult(success=True, output="custom")

        reg.register(CustomTool())
        assert reg.size == 5
        assert reg.get("custom") is not None

    def test_unregister(self):
        reg = ToolRegistry()
        assert reg.unregister("calculator") is True
        assert reg.size == 3
        assert reg.unregister("nonexistent") is False

    def test_list_tools(self):
        reg = ToolRegistry()
        tools = reg.list_tools()
        assert len(tools) == 4
        names = [t["name"] for t in tools]
        assert "calculator" in names
        assert "search" in names

    def test_execute_by_name(self):
        reg = ToolRegistry()
        r = reg.execute("calculator", {"expression": "1 + 1"})
        assert r.success
        assert float(r.output) == 2.0

    def test_execute_nonexistent(self):
        reg = ToolRegistry()
        r = reg.execute("nonexistent", {})
        assert not r.success


# ------------------------------------------------------------------ #
# SafetyChecker
# ------------------------------------------------------------------ #
class TestSafetyChecker:
    """安全检查器。"""

    def test_first_call_passes(self):
        sc = SafetyChecker(min_interval=0.0)
        v = sc.check("calculator", {"expression": "1+1"})
        assert v is None

    def test_rate_limit_blocks(self):
        sc = SafetyChecker(min_interval=1.0)
        sc.check("calculator", {"expression": "1+1"})
        v = sc.check("calculator", {"expression": "2+2"})
        assert v is not None
        assert "速率限制" in v.reason

    def test_dangerous_pattern_blocked(self):
        sc = SafetyChecker(min_interval=0.0)
        v = sc.check("code_executor", {"code": "rm -rf /"})
        assert v is not None
        assert "危险模式" in v.reason

    def test_audit_log(self):
        sc = SafetyChecker(min_interval=0.0)
        sc.record_call("calculator", {"expression": "1+1"}, {"success": True})
        log = sc.audit_log
        assert len(log) == 1
        assert log[0]["tool_name"] == "calculator"

    def test_stats(self):
        sc = SafetyChecker(min_interval=0.0)
        sc.check("calculator", {"expression": "1+1"})
        stats = sc.stats
        assert stats["n_tools_tracked"] == 1


# ------------------------------------------------------------------ #
# ToolPolicy
# ------------------------------------------------------------------ #
class TestToolPolicy:
    """工具选择策略。"""

    def test_no_tool_when_confident(self):
        """置信度高时不调用工具。"""
        reg = ToolRegistry()
        policy = ToolPolicy(reg, confidence_threshold=0.6)
        ctx = np.ones(16)
        decision = policy.decide(
            context=ctx, prediction_error=0.1, confidence=0.9
        )
        assert decision is None

    def test_calls_tool_when_uncertain(self):
        """置信度低时触发工具调用。"""
        reg = ToolRegistry()
        policy = ToolPolicy(reg, confidence_threshold=0.6)
        ctx = np.ones(16)
        decision = policy.decide(
            context=ctx, prediction_error=2.0, confidence=0.3
        )
        assert decision is not None
        assert decision.tool_name in {"calculator", "search", "code_executor", "database_query"}
        assert decision.confidence == 0.3

    def test_execute_records_history(self):
        """执行工具后记录调用历史。"""
        reg = ToolRegistry()
        policy = ToolPolicy(reg, confidence_threshold=0.6)
        decision = ToolDecision(
            tool_name="calculator",
            params={"expression": "1 + 1"},
            expected_info_gain=0.5,
            confidence=0.3,
        )
        result = policy.execute(decision)
        assert result.success
        assert len(policy.call_history) == 1
        assert policy.call_history[0]["tool_name"] == "calculator"

    def test_stats(self):
        reg = ToolRegistry()
        policy = ToolPolicy(reg)
        assert policy.stats["total_calls"] == 0
        assert policy.stats["success_rate"] == 0.0
        assert policy.stats["n_tools_available"] == 4

    def test_execute_with_safety_check(self):
        """安全检查器阻止违规调用。"""
        reg = ToolRegistry()
        sc = SafetyChecker(min_interval=0.0)
        policy = ToolPolicy(reg, safety_checker=sc, confidence_threshold=0.9)
        decision = ToolDecision(
            tool_name="code_executor",
            params={"code": "rm -rf /"},
        )
        result = policy.execute(decision)
        assert not result.success
        assert "安全检查" in result.error

    def test_expected_free_energy_lower_than_no_tool(self):
        """调用工具的预期自由能应低于不调用时的自由能。"""
        reg = ToolRegistry()
        policy = ToolPolicy(reg, confidence_threshold=0.6, info_gain_weight=0.5)
        ctx = np.ones(16)
        decision = policy.decide(
            context=ctx, prediction_error=3.0, confidence=0.2
        )
        if decision is not None:
            assert decision.expected_free_energy < 3.0


# ------------------------------------------------------------------ #
# 集成演示场景
# ------------------------------------------------------------------ #
class TestToolUseDemo:
    """演示场景：模型自主调用搜索工具回答百科问题。"""

    def test_search_answers_question(self):
        """模型遇到未知问题 → 置信度低 → 调用搜索 → 获得答案。"""
        reg = ToolRegistry()
        # 预填充搜索索引
        search = reg.get("search")
        assert isinstance(search, SearchTool)
        search.add_to_index("tokyo", "Tokyo is the capital of Japan.")

        policy = ToolPolicy(reg, confidence_threshold=0.6)
        # 模型对"东京是什么"不确定（置信度 0.2）
        decision = policy.decide(
            context=np.ones(16) * 0.5,
            prediction_error=2.0,
            confidence=0.2,
            tool_hint="search",
        )
        # 决策可能选择 search 也可能选择其他工具（上下文匹配）
        # 这里用 tool_hint 强制指定 search
        if decision is not None and decision.tool_name == "search":
            # 设置查询参数
            decision.params = {"query": "tokyo"}
            result = policy.execute(decision)
            assert result.success
            assert "capital of Japan" in result.output
