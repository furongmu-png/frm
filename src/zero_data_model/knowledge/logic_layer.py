# src/zero_data_model/knowledge/logic_layer.py
"""模糊逻辑约束层（LogicLayer）。

在认知架构中叠加一层“软”逻辑约束：用户注册形如“若前件集合成立则后件
成立”的规则（``LogicRule``），系统在当前谓词真值（``predicate_values``）
上用 Gödel t-范数/t-余范数进行模糊推理，并对“后件实际真值低于推断真值”
的规则产生惩罚信号。该惩罚可作为预测误差的附加项注入 ``think()``。

仅依赖 ``numpy``，独立、自包含、可插拔。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class LogicRule:
    """一条模糊逻辑规则。

    Attributes
    ----------
    name:
        规则名（唯一标识，便于报告）。
    antecedents:
        前件谓词名列表；其真值经 t-范数聚合得到推断真值。
    consequent:
        后件谓词名。
    weight:
        规则权重，缩放违反惩罚。
    description:
        人类可读说明。
    """

    name: str
    antecedents: list[str]
    consequent: str
    weight: float = 1.0
    description: str = ""


class LogicLayer:
    """模糊逻辑约束层。

    Attributes
    ----------
    rules:
        已注册的规则列表。
    predicate_values:
        当前谓词真值表，取值被钳制到 ``[0, 1]``。
    """

    def __init__(self) -> None:
        """初始化空的逻辑层。"""
        self.rules: list[LogicRule] = []
        self.predicate_values: dict[str, float] = {}
        # 可重入锁：保护 rules / predicate_values 的读写，t_norm/t_conorm
        # 为纯函数（仅依赖入参）无需加锁。
        self._lock = threading.RLock()

    def add_rule(
        self,
        name: str,
        antecedents: list[str],
        consequent: str,
        weight: float = 1.0,
        description: str = "",
    ) -> None:
        """注册一条逻辑规则。"""
        with self._lock:
            self.rules.append(
                LogicRule(
                    name=name,
                    antecedents=list(antecedents),
                    consequent=consequent,
                    weight=float(weight),
                    description=description,
                )
            )

    def set_predicate(self, name: str, value: float) -> None:
        """设置谓词真值，钳制到 ``[0, 1]``。

        校验：``name`` 必须为字符串；``value`` 必须为实数（int/float），
        ``None`` 抛 :class:`TypeError`。超出 ``[0, 1]`` 的值被钳制到
        ``[0, 1]``（选择钳制而非抛错以提升鲁棒性）。
        """
        if not isinstance(name, str):
            raise TypeError(f"name 必须为 str，收到 {type(name)!r}")
        if value is None:
            raise TypeError("value 不能为 None，期望 int/float")
        # bool 是 int 的子类，此处不视作合法数值。
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"value 必须为数值（int/float），收到 {type(value)!r}"
            )
        val = float(value)
        if not np.isfinite(val):
            raise ValueError(f"value 必须为有限实数，收到 {val}")
        # 钳制到 [0, 1]（鲁棒性优先，不抛 ValueError）。
        self.predicate_values[name] = max(0.0, min(1.0, val))

    def t_norm(self, a: float, b: float) -> float:
        """Gödel t-范数：``min(a, b)``。纯函数，无需加锁。"""
        return min(float(a), float(b))

    def t_conorm(self, a: float, b: float) -> float:
        """Gödel t-余范数：``max(a, b)``。纯函数，无需加锁。"""
        return max(float(a), float(b))

    def evaluate_rule(self, rule: LogicRule) -> tuple[float, float]:
        """评估单条规则。

        前件真值为所有前件谓词真值的 Gödel t-范数；缺失谓词按 0 处理。
        无前件时按“空真”（vacuous truth）处理：推断真值取 1.0（t-范数单位元），
        且不产生违反惩罚（空前件的规则无法被“违反”）。

        缺失后件谓词值统一按 0.0 处理（视为规则被违反），与
        :meth:`check_all` 保持一致。若后件实际真值低于推断真值，则产生
        违反惩罚 ``((inferred - actual) * weight)``。

        Returns
        -------
        tuple[float, float]
            ``(inferred_consequent_truth, violation_penalty)``。
        """
        with self._lock:
            # 空前件：空真（vacuous truth），推断为 1.0，无惩罚。
            if not rule.antecedents:
                return (1.0, 0.0)

            # 前件聚合：缺失默认 0，t-范数从单位元 1.0 起步
            inferred: float = 1.0
            for ant in rule.antecedents:
                inferred = self.t_norm(
                    inferred, self.predicate_values.get(ant, 0.0)
                )

            # 后件实际值：缺失统一按 0.0 处理（规则被违反），与 check_all 对齐。
            actual = float(self.predicate_values.get(rule.consequent, 0.0))
            penalty = (inferred - actual) * rule.weight if actual < inferred else 0.0
            return (float(inferred), float(penalty))

    def check_all(self) -> dict[str, Any]:
        """评估全部规则，汇总违反情况。

        缺失后件谓词值按 0.0 处理（与 :meth:`evaluate_rule` 一致）。

        Returns
        -------
        dict
            ``{"violations": [...], "total_penalty": float, "n_violations": int}``，
            其中每个违反项含 ``rule``、``penalty``、``inferred``、``actual``。
        """
        with self._lock:
            violations: list[dict[str, Any]] = []
            total_penalty: float = 0.0
            for rule in self.rules:
                inferred, penalty = self.evaluate_rule(rule)
                if penalty > 0.0:
                    actual = float(
                        self.predicate_values.get(rule.consequent, 0.0)
                    )
                    violations.append(
                        {
                            "rule": rule.name,
                            "penalty": float(penalty),
                            "inferred": float(inferred),
                            "actual": float(actual),
                        }
                    )
                    total_penalty += penalty
            return {
                "violations": violations,
                "total_penalty": float(total_penalty),
                "n_violations": len(violations),
            }

    def get_penalty_signal(self) -> np.ndarray:
        """返回每条规则惩罚值组成的一维数组，便于叠加到预测误差。"""
        with self._lock:
            penalties = [self.evaluate_rule(rule)[1] for rule in self.rules]
            return np.array(penalties, dtype=float)


__all__: list[str] = ["LogicRule", "LogicLayer"]
