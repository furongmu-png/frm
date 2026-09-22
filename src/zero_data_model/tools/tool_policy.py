"""Phase 3 §2.2 工具选择策略。

模型通过自由能最小化决定是否调用工具、调用哪个工具：
  - 将"调用工具"作为特殊动作插入候选动作列表
  - 每个工具调用动作的预期自由能 = 预测误差 - 预期信息增益
  - 当元认知置信度低于阈值时，工具调用动作的期望效用提高

执行流程：
  1. 模型选择"调用工具"动作
  2. 根据上下文选择工具（工具描述与上下文的余弦相似度）
  3. 执行工具，获取结果
  4. 结果作为额外观测输入，触发信念更新
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .safety import SafetyChecker
from .tool_registry import Tool, ToolRegistry, ToolResult


# ------------------------------------------------------------------ #
# ToolDecision dataclass
# ------------------------------------------------------------------ #
@dataclass
class ToolDecision:
    """工具调用决策。"""

    tool_name: str
    params: dict[str, Any] = field(default_factory=dict)
    expected_info_gain: float = 0.0
    expected_free_energy: float = 0.0
    confidence: float = 1.0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "params": self.params,
            "expected_info_gain": self.expected_info_gain,
            "expected_free_energy": self.expected_free_energy,
            "confidence": self.confidence,
            "reason": self.reason,
        }


# ------------------------------------------------------------------ #
# ToolPolicy
# ------------------------------------------------------------------ #
class ToolPolicy:
    """工具选择策略：基于自由能最小化选择工具。

    策略：
    1. 当元认知置信度低于 ``confidence_threshold`` 时，工具调用效用提高。
    2. 通过工具描述与当前上下文的余弦相似度选择最合适的工具。
    3. 预期自由能 = 当前预测误差 - 预期信息增益。
    4. 只有当预期自由能 < 不调用工具时的自由能时，才调用工具。

    Parameters
    ----------
    registry:
        工具注册表。
    safety_checker:
        安全检查器（可选；为 None 时跳过安全检查）。
    confidence_threshold:
        置信度阈值，低于此值时工具调用效用提高。
    info_gain_weight:
        信息增益在预期自由能中的权重。
    """

    def __init__(
        self,
        registry: ToolRegistry,
        safety_checker: SafetyChecker | None = None,
        confidence_threshold: float = 0.6,
        info_gain_weight: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.registry = registry
        self.safety_checker = safety_checker
        self.confidence_threshold = confidence_threshold
        self.info_gain_weight = info_gain_weight
        self._rng = np.random.default_rng(seed)
        self._lock = threading.RLock()
        # 工具调用历史：用于学习哪些工具有效。
        self._call_history: list[dict[str, Any]] = []

    def decide(
        self,
        context: np.ndarray,
        prediction_error: float,
        confidence: float,
        tool_hint: str = "",
    ) -> ToolDecision | None:
        """决定是否调用工具，以及调用哪个工具。

        Parameters
        ----------
        context:
            当前上下文向量（用于匹配工具描述）。
        prediction_error:
            当前预测误差（自由能代理）。
        confidence:
            元认知置信度（0-1）。
        tool_hint:
            可选的工具名称提示（如用户指定或上下文推断）。

        Returns
        -------
        ToolDecision | None
            返回 ``None`` 表示不调用工具。
        """
        with self._lock:
            # 1. 置信度低 → 工具调用效用提高
            if confidence >= self.confidence_threshold:
                # 置信度足够高，不需要工具
                return None

            # 2. 选择工具
            if tool_hint:
                tool = self.registry.get(tool_hint)
            else:
                tool = self._select_tool_by_context(context)
            if tool is None:
                return None

            # 3. 估计预期信息增益（简化：基于历史调用成功率）
            success_rate = self._estimate_success_rate(tool.name)
            expected_gain = (
                prediction_error * self.info_gain_weight * success_rate
            )

            # 4. 预期自由能
            expected_fe = prediction_error - expected_gain
            # 只当调用工具能降低自由能时才调用
            if expected_fe >= prediction_error:
                return None

            # 5. 生成参数（简化：用上下文的统计量）
            params = self._generate_params(tool, context)

            return ToolDecision(
                tool_name=tool.name,
                params=params,
                expected_info_gain=expected_gain,
                expected_free_energy=expected_fe,
                confidence=confidence,
                reason=(
                    f"置信度 {confidence:.2f} < 阈值 {self.confidence_threshold:.2f}"
                ),
            )

    def execute(
        self,
        decision: ToolDecision,
        trigger_reason: str = "",
    ) -> ToolResult:
        """执行工具调用决策。

        执行前通过安全检查器检查，执行后记录到审计日志和调用历史。
        """
        with self._lock:
            # 安全检查
            if self.safety_checker is not None:
                violation = self.safety_checker.check(
                    decision.tool_name, decision.params
                )
                if violation is not None:
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"安全检查失败: {violation.reason}",
                    )
            # 执行
            result = self.registry.execute(
                decision.tool_name, decision.params
            )
            # 审计日志
            if self.safety_checker is not None:
                self.safety_checker.record_call(
                    decision.tool_name,
                    decision.params,
                    result.to_dict(),
                    trigger_reason or decision.reason,
                )
            # 调用历史
            self._call_history.append({
                "tool_name": decision.tool_name,
                "params": str(decision.params),
                "success": result.success,
                "expected_gain": decision.expected_info_gain,
                "timestamp": len(self._call_history),
            })
            return result

    def _select_tool_by_context(self, context: np.ndarray) -> Tool | None:
        """通过工具描述与上下文的余弦相似度选择工具。

        将上下文向量和工具描述分别嵌入为向量，计算余弦相似度。
        这里使用简化的嵌入：对工具描述取字符哈希为向量。
        """
        tools = self.registry.list_tools()
        if not tools:
            return None
        # 上下文嵌入：取前 N 维的统计量
        ctx = np.asarray(context, dtype=np.float64).flatten()
        ctx_norm = np.linalg.norm(ctx)
        if ctx_norm < 1e-12:
            # 上下文为零向量：随机选一个工具
            idx = int(self._rng.integers(0, len(tools)))
            return self.registry.get(tools[idx]["name"])
        # 工具描述嵌入：简化为描述长度的哈希向量
        best_tool = None
        best_sim = -1.0
        for t in tools:
            desc_vec = self._embed_description(t["description"], ctx.shape[0])
            sim = float(np.dot(ctx, desc_vec) / (
                ctx_norm * np.linalg.norm(desc_vec) + 1e-12
            ))
            if sim > best_sim:
                best_sim = sim
                best_tool = self.registry.get(t["name"])
        return best_tool

    def _embed_description(self, description: str, dim: int) -> np.ndarray:
        """将工具描述嵌入为指定维度的向量（简化的字符哈希）。"""
        vec = np.zeros(dim)
        for i, ch in enumerate(description):
            vec[i % dim] += ord(ch) / 256.0
        return vec

    def _estimate_success_rate(self, tool_name: str) -> float:
        """基于历史调用估计工具成功率。"""
        calls = [c for c in self._call_history if c["tool_name"] == tool_name]
        if not calls:
            return 0.5  # 先验
        successes = sum(1 for c in calls if c["success"])
        return successes / len(calls)

    def _generate_params(
        self, tool: Tool, context: np.ndarray,
    ) -> dict[str, Any]:
        """根据上下文生成工具调用参数（简化）。"""
        ctx = np.asarray(context, dtype=np.float64).flatten()
        # 根据 schema 生成参数
        schema = tool.input_schema
        params: dict[str, Any] = {}
        for prop_name, prop_schema in schema.get("properties", {}).items():
            prop_type = prop_schema.get("type", "string")
            if prop_type == "string":
                # 用上下文的均值作为"查询"
                params[prop_name] = f"context_{float(np.mean(ctx)):.4f}"
            elif prop_type == "number":
                params[prop_name] = float(np.mean(ctx))
            elif prop_type == "object":
                params[prop_name] = {}
        return params

    @property
    def call_history(self) -> list[dict[str, Any]]:
        """工具调用历史。"""
        with self._lock:
            return list(self._call_history)

    @property
    def stats(self) -> dict[str, Any]:
        """策略统计。"""
        with self._lock:
            total = len(self._call_history)
            successes = sum(1 for c in self._call_history if c["success"])
            return {
                "total_calls": total,
                "successful_calls": successes,
                "success_rate": (successes / total) if total > 0 else 0.0,
                "n_tools_available": self.registry.size,
            }


__all__: list[str] = ["ToolDecision", "ToolPolicy"]
