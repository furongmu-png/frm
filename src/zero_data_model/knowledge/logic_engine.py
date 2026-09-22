# src/zero_data_model/knowledge/logic_engine.py
"""神经符号融合：逻辑规则引擎（LogicEngine）。

第二阶段 §1.1：在模糊逻辑约束层（``LogicLayer``）之上叠加**可微模糊推理
与矛盾检测**。与 ``LogicLayer`` 的区别：

- **字典式规则声明**：``{"if": [...], "then": "...", "weight": float}``，
  便于从自然语言/百科文本中自动提炼规则。
- **可学习谓词嵌入**：每个谓词维护一个嵌入向量，可用于类比与迁移。
- **标量矛盾度**：``check_consistency`` 返回 ``[0, 1]`` 的矛盾度，
  便于注入预测误差与元认知触发。
- **Hebbian 权重调整**：有效规则（推断≈实际）增强，无效规则（推断≫实际）
  减弱，使规则库随经验自适应。

模糊逻辑采用 Gödel t-范数：

    t(A, B) = min(A, B)
    impl(A, B) = B   if A > B   (规则被违反，真值降为 B)
               = 1   if A <= B   (规则被满足)

仅依赖 ``numpy``，独立、自包含、可插拔。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import numpy as np


# ------------------------------------------------------------------ #
# 规则
# ------------------------------------------------------------------ #
@dataclass
class LogicRuleV2:
    """一条字典式模糊逻辑规则。

    Attributes
    ----------
    antecedents:
        前件谓词名列表；真值经 Gödel t-范数聚合得到推断真值。
    consequent:
        后件谓词名。
    weight:
        规则权重，缩放违反惩罚与 Hebbian 更新幅度。钳制到 ``[0, 2]``。
    name:
        规则名（可选，便于报告）。
    description:
        人类可读说明（可选）。
    """

    antecedents: list[str]
    consequent: str
    weight: float = 1.0
    name: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        """后处理：钳制权重、规范化字段类型。"""
        self.weight = max(0.0, min(2.0, float(self.weight)))
        self.antecedents = list(self.antecedents)
        self.consequent = str(self.consequent)
        self.name = str(self.name)
        self.description = str(self.description)


# ------------------------------------------------------------------ #
# 逻辑引擎
# ------------------------------------------------------------------ #
class LogicEngine:
    """可微模糊推理与矛盾检测引擎。

    Parameters
    ----------
    predicate_dim:
        谓词嵌入向量维度（默认 16）。
    lr:
        Hebbian 权重调整学习率（默认 0.05）。
    contradiction_gain:
        ``resolve_contradiction`` 中矛盾度对误差的放大系数（默认 1.0）。
    seed:
        随机种子，用于可复现的谓词嵌入初始化。

    Attributes
    ----------
    rules:
        已注册的规则列表。
    predicate_embeddings:
        谓词名到嵌入向量的映射。
    """

    def __init__(
        self,
        predicate_dim: int = 16,
        lr: float = 0.05,
        contradiction_gain: float = 1.0,
        seed: int = 42,
    ) -> None:
        """初始化空的逻辑引擎。"""
        if predicate_dim <= 0:
            raise ValueError(
                f"predicate_dim 必须为正整数，收到 {predicate_dim}"
            )
        self._predicate_dim = int(predicate_dim)
        self._lr = float(lr)
        self._contradiction_gain = float(contradiction_gain)
        self._rng = np.random.default_rng(seed)
        # 可重入锁：保护 rules / embeddings / usage 统计的读写。
        self._lock = threading.RLock()
        self.rules: list[LogicRuleV2] = []
        self.predicate_embeddings: dict[str, np.ndarray] = {}
        # 每条规则的使用统计：index -> {"fired": int, "satisfied": int}
        self._rule_stats: list[dict[str, int]] = []
        # 最近一次矛盾度，便于外部查询。
        self._last_contradiction: float = 0.0

    # ------------------------------------------------------------------ #
    # 规则管理
    # ------------------------------------------------------------------ #
    def add_rule(self, rule: dict[str, Any]) -> None:
        """添加一条逻辑规则到知识库。

        Parameters
        ----------
        rule:
            字典格式 ``{"if": ["pred1", "pred2"], "then": "pred3",
            "weight": 0.8}``。``"weight"`` 可选（默认 1.0）；
            ``"name"`` / ``"description"`` 可选。

        Raises
        ------
        KeyError
            如果 ``rule`` 缺少 ``"if"`` 或 ``"then"`` 键。
        TypeError
            如果 ``rule`` 不是 dict，或 ``"if"`` 不是列表。
        ValueError
            如果 ``"if"`` 为空列表或 ``"then"`` 为空字符串。
        """
        if not isinstance(rule, dict):
            raise TypeError(f"rule 必须为 dict，收到 {type(rule)!r}")
        if "if" not in rule or "then" not in rule:
            raise KeyError("rule 必须包含 'if' 和 'then' 键")
        antecedents = rule["if"]
        if not isinstance(antecedents, list):
            raise TypeError(f"rule['if'] 必须为 list，收到 {type(antecedents)!r}")
        if not antecedents:
            raise ValueError("rule['if'] 不能为空列表")
        consequent = rule["then"]
        if not isinstance(consequent, str) or not consequent:
            raise ValueError("rule['then'] 必须为非空字符串")
        weight = float(rule.get("weight", 1.0))
        name = str(rule.get("name", ""))
        description = str(rule.get("description", ""))

        with self._lock:
            r = LogicRuleV2(
                antecedents=antecedents,
                consequent=consequent,
                weight=weight,
                name=name,
                description=description,
            )
            self.rules.append(r)
            self._rule_stats.append({"fired": 0, "satisfied": 0})
            # 确保所有谓词都有嵌入。
            for pred in antecedents:
                self._ensure_predicate(pred)
            self._ensure_predicate(consequent)

    def _ensure_predicate(self, name: str) -> np.ndarray:
        """确保谓词有嵌入向量，若无则随机初始化。"""
        with self._lock:
            if name not in self.predicate_embeddings:
                self.predicate_embeddings[name] = (
                    self._rng.standard_normal(self._predicate_dim) * 0.1
                )
            return self.predicate_embeddings[name]

    # ------------------------------------------------------------------ #
    # 模糊逻辑算子（Gödel t-范数）
    # ------------------------------------------------------------------ #
    @staticmethod
    def t_norm(a: float, b: float) -> float:
        """Gödel t-范数：``min(a, b)``。纯函数。"""
        return min(float(a), float(b))

    @staticmethod
    def implication(a: float, b: float) -> float:
        """Gödel 蕴含：``b if a > b else 1``。

        当 ``a > b``（前件比后件更真）时规则被违反，蕴含真值降为 ``b``；
        当 ``a <= b``（后件至少与前件一样真）时规则被满足，蕴含真值为 1。
        """
        a_f = float(a)
        b_f = float(b)
        return b_f if a_f > b_f else 1.0

    # ------------------------------------------------------------------ #
    # 一致性检查
    # ------------------------------------------------------------------ #
    def check_consistency(self, belief_state: dict[str, float]) -> float:
        """检查当前信念是否违反规则，返回矛盾度（0-1）。

        对每条规则：
        1. 前件真值 = 所有前件谓词值的 Gödel t-范数（缺失按 0）。
        2. 后件实际真值 = ``belief_state.get(consequent, 0)``。
        3. 违反量 = ``max(0, inferred - actual) * weight``。
        4. 蕴含真值 = ``implication(inferred, actual)``。

        矛盾度 = 总违反量 / (总违反量 + 1)，归一化到 ``[0, 1]``。
        同时执行 Hebbian 权重调整（有效规则增强，无效规则减弱）。

        Parameters
        ----------
        belief_state:
            谓词名到真值（``[0, 1]``）的映射。

        Returns
        -------
        float
            矛盾度，范围 ``[0, 1]``。0 = 完全一致，1 = 极度矛盾。
        """
        if not isinstance(belief_state, dict):
            raise TypeError(
                f"belief_state 必须为 dict，收到 {type(belief_state)!r}"
            )
        with self._lock:
            if not self.rules:
                self._last_contradiction = 0.0
                return 0.0

            total_violation = 0.0
            total_weight = 0.0
            for idx, rule in enumerate(self.rules):
                # 前件聚合：Gödel t-范数，从单位元 1.0 起步。
                inferred = 1.0
                for ant in rule.antecedents:
                    val = self._clamp_truth(belief_state.get(ant, 0.0))
                    inferred = self.t_norm(inferred, val)
                # 后件实际值：缺失按 0（规则被违反）。
                actual = self._clamp_truth(belief_state.get(rule.consequent, 0.0))
                # 违反量。
                violation = max(0.0, inferred - actual) * rule.weight
                total_violation += violation
                total_weight += rule.weight
                # Hebbian：记录使用与满足情况。
                self._rule_stats[idx]["fired"] += 1
                if actual >= inferred - 1e-6:
                    # 规则被满足（有效）→ 增强。
                    self._rule_stats[idx]["satisfied"] += 1
                    self._hebbian_adjust(idx, effective=True)
                else:
                    # 规则被违反（无效）→ 减弱。
                    self._hebbian_adjust(idx, effective=False)

            # 归一化矛盾度：使用 sigmoid-like 变换。
            # contradiction = violation / (violation + 1)，保证 [0, 1)。
            contradiction = total_violation / (total_violation + 1.0)
            self._last_contradiction = float(
                max(0.0, min(1.0, contradiction))
            )
            return self._last_contradiction

    def resolve_contradiction(
        self,
        belief_state: dict[str, float],
        error: np.ndarray,
    ) -> np.ndarray:
        """基于矛盾度产生额外预测误差，驱动模型修正信念。

        策略：``extra = error * contradiction * gain``。
        矛盾越高，额外误差越大；调用方将其叠加到原预测误差上
        （``total = error + extra``），驱动模型更激进地修正违反规则的信念。

        Parameters
        ----------
        belief_state:
            当前信念状态（谓词 → 真值）。
        error:
            当前预测误差向量（任意维度）。

        Returns
        -------
        np.ndarray
            额外预测误差向量（与 ``error`` 同形状），调用方叠加到原误差上。
        """
        err = np.asarray(error, dtype=np.float64).flatten()
        with self._lock:
            contradiction = self._last_contradiction
            if contradiction == 0.0:
                # 缓存可能过期，重新计算。
                contradiction = self.check_consistency(belief_state)
            # 额外误差 = error * contradiction * gain
            result: np.ndarray = err * contradiction * self._contradiction_gain
            return result

    # ------------------------------------------------------------------ #
    # Hebbian 权重调整
    # ------------------------------------------------------------------ #
    def _hebbian_adjust(self, rule_idx: int, effective: bool) -> None:
        """Hebbian 权重调整：有效规则增强，无效规则减弱。

        使用使用频率（fired 次数）调制学习率，避免早期噪声主导。
        """
        with self._lock:
            rule = self.rules[rule_idx]
            stats = self._rule_stats[rule_idx]
            # 学习率随使用次数衰减（避免无限漂移）。
            decay = 1.0 / (1.0 + stats["fired"] * 0.01)
            lr_eff = self._lr * decay
            if effective:
                rule.weight = min(2.0, rule.weight + lr_eff)
            else:
                rule.weight = max(0.0, rule.weight - lr_eff)

    # ------------------------------------------------------------------ #
    # 查询接口
    # ------------------------------------------------------------------ #
    def get_violations(self, belief_state: dict[str, float]) -> list[dict[str, Any]]:
        """返回当前被违反的规则详情列表。"""
        if not isinstance(belief_state, dict):
            return []
        with self._lock:
            violations: list[dict[str, Any]] = []
            for rule in self.rules:
                inferred = 1.0
                for ant in rule.antecedents:
                    inferred = self.t_norm(
                        inferred, self._clamp_truth(belief_state.get(ant, 0.0))
                    )
                actual = self._clamp_truth(
                    belief_state.get(rule.consequent, 0.0)
                )
                if actual < inferred - 1e-6:
                    violations.append({
                        "rule": rule.name or f"if {rule.antecedents} then {rule.consequent}",
                        "inferred": float(inferred),
                        "actual": float(actual),
                        "violation": float(inferred - actual),
                        "weight": float(rule.weight),
                    })
            return violations

    def get_predicate_embedding(self, name: str) -> np.ndarray:
        """获取谓词嵌入向量（不存在则自动创建）。"""
        with self._lock:
            return self._ensure_predicate(name).copy()

    @property
    def last_contradiction(self) -> float:
        """最近一次 ``check_consistency`` 的矛盾度。"""
        with self._lock:
            return self._last_contradiction

    @property
    def stats(self) -> dict[str, Any]:
        """引擎统计信息。"""
        with self._lock:
            total_fired = sum(s["fired"] for s in self._rule_stats)
            total_satisfied = sum(s["satisfied"] for s in self._rule_stats)
            return {
                "n_rules": len(self.rules),
                "n_predicates": len(self.predicate_embeddings),
                "total_fired": total_fired,
                "total_satisfied": total_satisfied,
                "last_contradiction": self._last_contradiction,
                "rule_weights": [float(r.weight) for r in self.rules],
            }

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    @staticmethod
    def _clamp_truth(value: float) -> float:
        """将真值钳制到 ``[0, 1]``。"""
        try:
            v = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(v):
            return 0.0
        return max(0.0, min(1.0, v))


__all__: list[str] = ["LogicRuleV2", "LogicEngine"]
