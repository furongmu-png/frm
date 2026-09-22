"""Phase 3 §3.2 反事实解释引擎。

使用世界模型模拟"如果当时采取不同动作，结果会怎样"，比较事实轨迹
与反事实轨迹的自由能差异，生成自然语言解释。

解释生成模板示例：
    "模型选择动作 A 而非 B，因为 A 的预期自由能更低（1.2 vs 3.4）。
     若选择 B，预测误差将增加 45%，可能导致碰撞。"
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol


# ------------------------------------------------------------------ #
# World model protocol：用于反事实模拟的前向预测接口
# ------------------------------------------------------------------ #
class _WorldModel(Protocol):
    """世界模型接口：给定上下文与动作，返回预测的下一状态。"""

    def __call__(self, context: dict[str, Any], action: Any) -> dict[str, Any]:
        """模拟在 ``context`` 下执行 ``action`` 的结果。

        返回字典至少应包含：
        - ``free_energy``: float
        - ``prediction_error``: float
        - ``outcome``: str（结果的自然语言描述）
        """
        ...


# ------------------------------------------------------------------ #
# CounterfactualResult
# ------------------------------------------------------------------ #
@dataclass
class CounterfactualResult:
    """单次反事实解释的结果。"""

    factual_action: Any
    alternative_action: Any
    factual_free_energy: float
    counterfactual_free_energy: float
    factual_prediction_error: float
    counterfactual_prediction_error: float
    factual_outcome: str
    counterfactual_outcome: str
    free_energy_delta: float
    prediction_error_change_pct: float
    explanation: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return {
            "factual_action": self.factual_action,
            "alternative_action": self.alternative_action,
            "factual_free_energy": self.factual_free_energy,
            "counterfactual_free_energy": self.counterfactual_free_energy,
            "factual_prediction_error": self.factual_prediction_error,
            "counterfactual_prediction_error": self.counterfactual_prediction_error,
            "factual_outcome": self.factual_outcome,
            "counterfactual_outcome": self.counterfactual_outcome,
            "free_energy_delta": self.free_energy_delta,
            "prediction_error_change_pct": self.prediction_error_change_pct,
            "explanation": self.explanation,
            "metadata": self.metadata,
        }


# ------------------------------------------------------------------ #
# CounterfactualExplainer
# ------------------------------------------------------------------ #
class CounterfactualExplainer:
    """反事实解释引擎。

    通过世界模型前向模拟，比较"事实动作"与"反事实动作"在相同上下文
    下的预期自由能差异，生成人类可读的解释。

    Parameters
    ----------
    world_model:
        可调用对象，签名 ``(context, action) -> dict``，返回含
        ``free_energy``、``prediction_error``、``outcome`` 的预测结果。
    history_limit:
        保存的反事实解释历史上限（超出丢弃最旧）。
    """

    def __init__(
        self,
        world_model: _WorldModel | None = None,
        history_limit: int = 1000,
    ) -> None:
        self._world_model = world_model
        self._history: deque[CounterfactualResult] = deque(maxlen=history_limit)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # 设置/替换世界模型
    # ------------------------------------------------------------------
    def set_world_model(self, world_model: _WorldModel) -> None:
        """替换当前世界模型。"""
        with self._lock:
            self._world_model = world_model

    # ------------------------------------------------------------------
    # 反事实模拟
    # ------------------------------------------------------------------
    def _simulate(
        self, context: dict[str, Any], action: Any
    ) -> dict[str, Any]:
        """调用世界模型模拟指定动作。

        如果未设置世界模型，返回一个"未知"占位结果，使解释仍能生成
        （但会标注"无法模拟"）。
        """
        if self._world_model is None:
            return {
                "free_energy": float("nan"),
                "prediction_error": float("nan"),
                "outcome": "（世界模型未设置，无法模拟）",
            }
        result = self._world_model(context, action)
        # 填充默认值，确保下游字段存在
        result.setdefault("free_energy", float("nan"))
        result.setdefault("prediction_error", float("nan"))
        result.setdefault("outcome", "未知结果")
        return result

    # ------------------------------------------------------------------
    # 生成自然语言解释
    # ------------------------------------------------------------------
    def _generate_explanation(
        self,
        factual: dict[str, Any],
        counterfactual: dict[str, Any],
        factual_action: Any,
        alternative_action: Any,
    ) -> str:
        """根据事实与反事实指标生成自然语言解释。"""
        fe_f = factual["free_energy"]
        fe_c = counterfactual["free_energy"]
        pe_f = factual["prediction_error"]
        pe_c = counterfactual["prediction_error"]

        # 自由能比较
        if fe_f == fe_f and fe_c == fe_c:  # 均非 NaN
            fe_lower = "事实" if fe_f <= fe_c else "反事实"
            fe_f_str = f"{fe_f:.2f}"
            fe_c_str = f"{fe_c:.2f}"
        else:
            fe_lower = "未知"
            fe_f_str = "N/A"
            fe_c_str = "N/A"

        # 预测误差变化百分比
        pe_change_str = "N/A"
        if pe_f == pe_f and pe_c == pe_c and pe_f > 0:
            pct = (pe_c - pe_f) / pe_f * 100
            pe_change_str = f"{pct:+.1f}%"

        # 结果描述
        out_f = factual.get("outcome", "")
        out_c = counterfactual.get("outcome", "")

        explanation = (
            f"模型选择动作 {factual_action} 而非 {alternative_action}，"
            f"因为 {fe_lower} 动作的预期自由能更低"
            f"（{fe_f_str} vs {fe_c_str}）。"
            f"若选择 {alternative_action}，预测误差将变化 {pe_change_str}"
        )
        if out_c and out_c != "未知结果":
            explanation += f"，可能导致 {out_c}"
        explanation += "。"
        if out_f and out_f != "未知结果":
            explanation += f"（实际选择 {factual_action} 的结果：{out_f}。）"
        return explanation

    # ------------------------------------------------------------------
    # 公共 API：生成反事实解释
    # ------------------------------------------------------------------
    def explain(
        self,
        factual_state: dict[str, Any],
        alternative_action: Any,
    ) -> CounterfactualResult:
        """生成反事实解释。

        Parameters
        ----------
        factual_state:
            事实状态字典，应包含：
            - ``action``: 实际采取的动作
            - ``free_energy``: 实际自由能
            - ``prediction_error``: 实际预测误差
            - ``outcome``: 实际结果描述
            - ``context``: 当时的上下文（传给世界模型）
        alternative_action:
            要模拟的反事实动作。

        Returns
        -------
        CounterfactualResult
            含事实/反事实对比与自然语言解释。
        """
        with self._lock:
            factual_action = factual_state.get("action", "A")
            fe_f = float(factual_state.get("free_energy", float("nan")))
            pe_f = float(factual_state.get("prediction_error", float("nan")))
            out_f = str(factual_state.get("outcome", ""))
            context = factual_state.get("context", {})

            # 反事实模拟
            counterfactual = self._simulate(context, alternative_action)
            fe_c = float(counterfactual["free_energy"])
            pe_c = float(counterfactual["prediction_error"])
            out_c = str(counterfactual["outcome"])

            # 计算差异
            fe_delta = (
                float(fe_c - fe_f)
                if (fe_f == fe_f and fe_c == fe_c)
                else float("nan")
            )
            pe_change_pct = (
                float((pe_c - pe_f) / pe_f * 100)
                if (pe_f == pe_f and pe_c == pe_c and pe_f > 0)
                else float("nan")
            )

            explanation = self._generate_explanation(
                factual={
                    "free_energy": fe_f,
                    "prediction_error": pe_f,
                    "outcome": out_f,
                },
                counterfactual={
                    "free_energy": fe_c,
                    "prediction_error": pe_c,
                    "outcome": out_c,
                },
                factual_action=factual_action,
                alternative_action=alternative_action,
            )

            result = CounterfactualResult(
                factual_action=factual_action,
                alternative_action=alternative_action,
                factual_free_energy=fe_f,
                counterfactual_free_energy=fe_c,
                factual_prediction_error=pe_f,
                counterfactual_prediction_error=pe_c,
                factual_outcome=out_f,
                counterfactual_outcome=out_c,
                free_energy_delta=fe_delta,
                prediction_error_change_pct=pe_change_pct,
                explanation=explanation,
                metadata={"context": context},
            )
            self._history.append(result)
            return result

    # ------------------------------------------------------------------
    # 历史/统计
    # ------------------------------------------------------------------
    def get_history(self, n: int = 20) -> list[dict[str, Any]]:
        """获取最近 N 条反事实解释历史。"""
        with self._lock:
            return [r.to_dict() for r in list(self._history)[-n:]]

    def clear_history(self) -> None:
        """清空历史。"""
        with self._lock:
            self._history.clear()

    @property
    def n_explanations(self) -> int:
        """已生成的解释数。"""
        with self._lock:
            return len(self._history)

    @property
    def stats(self) -> dict[str, Any]:
        """反事实解释汇总统计。"""
        with self._lock:
            history = list(self._history)
            if not history:
                return {
                    "n_explanations": 0,
                    "mean_fe_delta": 0.0,
                    "mean_pe_change_pct": 0.0,
                }
            fe_deltas = [
                r.free_energy_delta
                for r in history
                if r.free_energy_delta == r.free_energy_delta
            ]
            pe_changes = [
                r.prediction_error_change_pct
                for r in history
                if r.prediction_error_change_pct
                == r.prediction_error_change_pct
            ]
            return {
                "n_explanations": len(history),
                "mean_fe_delta": round(
                    sum(fe_deltas) / len(fe_deltas), 4
                ) if fe_deltas else 0.0,
                "mean_pe_change_pct": round(
                    sum(pe_changes) / len(pe_changes), 4
                ) if pe_changes else 0.0,
            }


__all__: list[str] = ["CounterfactualExplainer", "CounterfactualResult"]
