"""Discovery Result Analyzer — 贝叶斯因子分析实验结果。

收集实验数据，计算贝叶斯因子 BF = P(data|H1) / P(data|H0)，
BF > 10 接受假设，BF < 1/10 拒绝假设，更新知识图谱。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .hypothesis_generator import Hypothesis, _stable_hash_id


@dataclass
class AnalysisResult:
    """实验结果分析。"""

    hypothesis_id: str
    bayes_factor: float  # BF = P(D|H1)/P(D|H0)
    decision: str  # "accept" | "reject" | "inconclusive"
    confidence: float  # 0-1
    effect_size: float  # 干预效应大小
    n_samples: int
    #: 自然语言结论
    conclusion: str
    #: 更新后的知识图谱增量
    kg_delta: dict[str, Any] = field(default_factory=dict)
    analysis_id: str = ""

    def __post_init__(self) -> None:
        if not self.analysis_id:
            self.analysis_id = _stable_hash_id(self.hypothesis_id, "A")


class ResultAnalyzer:
    """实验结果分析器。

    Parameters
    ----------
    accept_threshold : float, default 10.0
        BF > 此值接受假设。
    reject_threshold : float, default 0.1
        BF < 此值拒绝假设（即 1/10）。
    seed : int, default 42
    """

    def __init__(
        self,
        accept_threshold: float = 10.0,
        reject_threshold: float = 0.1,
        seed: int = 42,
    ) -> None:
        if accept_threshold <= 1.0:
            raise ValueError(f"accept_threshold must be > 1, got {accept_threshold}")
        if not 0.0 < reject_threshold < 1.0:
            raise ValueError(
                f"reject_threshold must be in (0, 1), got {reject_threshold}"
            )
        if reject_threshold > 1.0 / accept_threshold:
            raise ValueError("reject_threshold too high relative to accept_threshold")

        self.accept_threshold = float(accept_threshold)
        self.reject_threshold = float(reject_threshold)
        self._rng = np.random.default_rng(seed)
        self._analyses: list[AnalysisResult] = []
        self._max_history = 100

    # ------------------------------------------------------------------ #
    # 分析
    # ------------------------------------------------------------------ #
    def analyze(
        self, hypothesis: Hypothesis, experiment_data: dict[str, Any]
    ) -> AnalysisResult:
        """分析实验数据，计算贝叶斯因子。

        Parameters
        ----------
        hypothesis : Hypothesis
            被测试的假设。
        experiment_data : dict
            实验执行返回的数据，含 results 列表（每次重复的 before/after）。

        Returns
        -------
        AnalysisResult
            含贝叶斯因子、决策、结论。
        """
        results = experiment_data.get("results", [])
        if not results:
            return self._inconclusive(hypothesis, "无实验数据")

        # 提取 before/after 状态向量
        # M3 修复：ragged 列表（各 result 的 before/after 长度不一致）
        # 会让 np.array(..., dtype=np.float64) 抛 ValueError 并生成 object
        # 数组，后续 diff = after - before 会广播错位或抛错。
        # 军事级健壮性：逐条校验并填充到统一长度，失败则返回 inconclusive。
        before_list = []
        after_list = []
        max_len = 0
        for r in results:
            b = r.get("before", [])
            a = r.get("after", [])
            # 容忍标量/None：转为 0 长度向量
            if b is None:
                b = []
            if a is None:
                a = []
            if np.isscalar(b):
                b = [float(b)]
            if np.isscalar(a):
                a = [float(a)]
            try:
                b_arr = np.asarray(b, dtype=np.float64).flatten()
                a_arr = np.asarray(a, dtype=np.float64).flatten()
            except (TypeError, ValueError):
                # 无法解析的数据，用零向量占位
                b_arr = np.zeros(1)
                a_arr = np.zeros(1)
            before_list.append(b_arr)
            after_list.append(a_arr)
            max_len = max(max_len, b_arr.shape[0], a_arr.shape[0])

        # M3 修复：统一填充到 max_len，确保 np.array 生成规整 2D 数组
        if max_len == 0:
            return self._inconclusive(hypothesis, "实验数据为空")
        before_states = np.zeros((len(results), max_len), dtype=np.float64)
        after_states = np.zeros((len(results), max_len), dtype=np.float64)
        for i, (b_arr, a_arr) in enumerate(zip(before_list, after_list)):
            before_states[i, : b_arr.shape[0]] = b_arr
            after_states[i, : a_arr.shape[0]] = a_arr

        # NaN/inf 防护（沙盒可能返回污染数据）
        if not np.all(np.isfinite(before_states)):
            before_states = np.nan_to_num(before_states, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.all(np.isfinite(after_states)):
            after_states = np.nan_to_num(after_states, nan=0.0, posinf=0.0, neginf=0.0)

        # 效应大小：before 与 after 的差异
        diff = after_states - before_states
        effect_size = float(np.mean(np.linalg.norm(diff, axis=1)))

        # 贝叶斯因子计算
        bf = self._compute_bayes_factor(before_states, after_states, hypothesis)

        # 决策
        if bf > self.accept_threshold:
            decision = "accept"
        elif bf < self.reject_threshold:
            decision = "reject"
        else:
            decision = "inconclusive"

        # 置信度
        confidence = self._bf_to_confidence(bf)

        # 结论文本
        conclusion = self._generate_conclusion(hypothesis, bf, decision, effect_size)

        # 知识图谱增量
        kg_delta = self._compute_kg_delta(hypothesis, decision, effect_size)

        result = AnalysisResult(
            hypothesis_id=hypothesis.hypothesis_id,
            bayes_factor=round(bf, 6),
            decision=decision,
            confidence=round(confidence, 4),
            effect_size=round(effect_size, 6),
            n_samples=len(results),
            conclusion=conclusion,
            kg_delta=kg_delta,
        )
        self._analyses.append(result)
        if len(self._analyses) > self._max_history:
            self._analyses = self._analyses[-self._max_history :]
        return result

    def _compute_bayes_factor(
        self,
        before: np.ndarray,
        after: np.ndarray,
        hypothesis: Hypothesis,
    ) -> float:
        """计算贝叶斯因子 BF = P(D|H1) / P(D|H0)。

        H1: 干预有效应（before != after）
        H0: 干预无效应（before == after）
        简化：用均值差异的 t 统计量近似。
        """
        if before.size == 0 or after.size == 0:
            return 1.0

        # M4 修复：n < 2 时方差无定义（单样本无离散度）。
        # 原代码 np.var(n=1) 返回 0，使 pooled_std ≈ 1e-8，
        # t_stat → 巨大值，BF 恒为 accept——统计学上无效。
        # 军事级健壮性：样本不足时返回中性 BF=1.0（inconclusive），
        # 避免基于伪统计量做出错误科学决策。
        n_before = before.shape[0] if before.ndim > 1 else 1
        n_after = after.shape[0] if after.ndim > 1 else 1
        if n_before < 2 or n_after < 2:
            return 1.0

        # 逐维度均值差异
        mean_diff = np.mean(after, axis=0) - np.mean(before, axis=0)
        # 合并标准差
        pooled_std = np.sqrt(
            (np.var(before, axis=0) + np.var(after, axis=0)) / 2 + 1e-8
        )
        # t 统计量
        t_stat = mean_diff / pooled_std
        # 取最大维度的 |t| 作为效应强度
        max_t = float(np.max(np.abs(t_stat)))
        # F9 修复：若 t_stat 含 NaN（调用方传入未清洗数据），
        # max_t 为 NaN，后续 min(NaN, 50.0) 行为不确定。
        if not np.isfinite(max_t):
            return 1.0

        # BF 近似：|t| 越大，BF 越大
        # BF ≈ exp(|t| - 1) 当 |t| > 1，否则接近 1
        # 裁剪 max_t 防止 np.exp 溢出（exp(709) 接近 float64 上限）
        max_t = min(max_t, 50.0)
        if max_t > 1.0:
            bf = float(np.exp(max_t - 1.0))
        else:
            bf = float(max_t)  # 0-1 之间
        # 限制范围
        return max(0.01, min(bf, 1000.0))

    def _bf_to_confidence(self, bf: float) -> float:
        """贝叶斯因子转置信度。"""
        if bf > self.accept_threshold:
            return min(1.0, 0.5 + 0.5 * (1 - 1.0 / bf))
        if bf < self.reject_threshold:
            return min(1.0, 0.5 + 0.5 * (1 - bf))
        return 0.3  # 不确定

    def _generate_conclusion(
        self, h: Hypothesis, bf: float, decision: str, effect_size: float
    ) -> str:
        """生成自然语言结论。"""
        strength = "强证据" if bf > 50 else "中等证据" if bf > 10 else "弱证据"
        if decision == "accept":
            return (
                f"实验结果支持假设（{strength}，BF={bf:.2f}）。"
                f"观测到干预效应大小为 {effect_size:.3f}。"
                f"假设「{h.statement}」被接受。"
            )
        if decision == "reject":
            return (
                f"实验结果拒绝假设（BF={bf:.4f} < {self.reject_threshold}）。"
                f"未观测到显著干预效应。"
                f"假设「{h.statement}」被拒绝。"
            )
        return (
            f"实验结果不确定（BF={bf:.2f}）。"
            f"需更多实验验证。假设「{h.statement}」暂未定论。"
        )

    def _compute_kg_delta(
        self, h: Hypothesis, decision: str, effect_size: float
    ) -> dict[str, Any]:
        """根据决策计算知识图谱增量。"""
        if decision == "accept":
            return {
                "new_nodes": h.intervention_vars + h.observation_vars,
                "new_edges": [
                    {
                        "source": v,
                        "target": h.observation_vars[0] if h.observation_vars else "outcome",
                        "weight": round(effect_size, 4),
                        "type": "causal",
                    }
                    for v in h.intervention_vars
                ],
            }
        if decision == "reject":
            return {
                "new_nodes": [],
                "new_edges": [
                    {
                        "source": v,
                        "target": h.observation_vars[0] if h.observation_vars else "outcome",
                        "weight": 0.0,
                        "type": "independent",
                    }
                    for v in h.intervention_vars
                ],
            }
        return {"new_nodes": [], "new_edges": []}

    def _inconclusive(self, h: Hypothesis, reason: str) -> AnalysisResult:
        return AnalysisResult(
            hypothesis_id=h.hypothesis_id,
            bayes_factor=1.0,
            decision="inconclusive",
            confidence=0.0,
            effect_size=0.0,
            n_samples=0,
            conclusion=f"分析无法完成：{reason}",
        )

    # ------------------------------------------------------------------ #
    # 访问器
    # ------------------------------------------------------------------ #
    @property
    def analyses(self) -> list[AnalysisResult]:
        return list(self._analyses)

    @property
    def accepted_count(self) -> int:
        return sum(1 for a in self._analyses if a.decision == "accept")

    @property
    def rejected_count(self) -> int:
        return sum(1 for a in self._analyses if a.decision == "reject")

    def snapshot(self) -> dict:
        return {
            "accept_threshold": self.accept_threshold,
            "reject_threshold": self.reject_threshold,
            "total_analyses": len(self._analyses),
            "accepted": self.accepted_count,
            "rejected": self.rejected_count,
            "inconclusive": len(self._analyses)
            - self.accepted_count
            - self.rejected_count,
        }
