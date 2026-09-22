"""PeerReviewer — internal peer review of generated papers.

理论基础
========
科学共同体的核心机制是同行评审。本模块实现一个"内部审稿人"，对自动
生成的论文进行逻辑一致性、方法正确性、与现有文献一致性的评审。

评审维度：
  1. 逻辑一致性：假设→方法→结论的推理链是否自洽
  2. 方法正确性：统计检验是否恰当、样本量是否充分
  3. 文献一致性：结论是否与已知文献矛盾
  4. 可复现性：实验设计是否包含足够细节

输出评审意见：accept / revise / reject，附带具体修改建议。

自由能映射：评审 = 对论文模型的"预测误差"评估。论文作为对世界的模型，
其自由能 = 与证据的偏差 + 与文献的矛盾。评审降低论文的自由能。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .paper_writer import Paper
from .hypothesis_generator import Hypothesis
from .experiment_designer import ExperimentDesign
from .result_analyzer import AnalysisResult


# ------------------------------------------------------------------ #
# 数据结构
# ------------------------------------------------------------------ #
@dataclass
class ReviewReport:
    """评审报告。"""
    paper_id: str
    decision: str                    # "accept" | "revise" | "reject"
    overall_score: float             # 0-1 综合评分
    logic_score: float               # 逻辑一致性评分
    method_score: float              # 方法正确性评分
    literature_score: float          # 文献一致性评分
    reproducibility_score: float     # 可复现性评分
    comments: list[str]              # 评审意见
    required_revisions: list[str]    # 必须修改项
    metadata: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------ #
# 主类
# ------------------------------------------------------------------ #
class PeerReviewer:
    """内部同行评审器。

    Parameters
    ----------
    accept_threshold : float
        综合评分高于此值 → accept（默认 0.7）
    reject_threshold : float
        综合评分低于此值 → reject（默认 0.3）
    seed : int | None
    """

    def __init__(
        self,
        accept_threshold: float = 0.7,
        reject_threshold: float = 0.3,
        seed: Optional[int] = 42,
    ):
        if not (0 < reject_threshold < accept_threshold < 1):
            raise ValueError(
                f"need 0 < reject({reject_threshold}) < "
                f"accept({accept_threshold}) < 1"
            )
        self.accept_threshold = float(accept_threshold)
        self.reject_threshold = float(reject_threshold)
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._lock = threading.RLock()

        # 评审历史
        self._reviews: list[ReviewReport] = []
        self._max_history = 100
        # 已知文献结论（用于一致性检查）
        self._known_findings: list[str] = []

    # ---------------------------------------------------------------- #
    # 注册已知文献结论
    # ---------------------------------------------------------------- #
    def register_known_finding(self, finding: str) -> None:
        """注册一个已知文献结论，用于一致性检查。"""
        with self._lock:
            if finding and isinstance(finding, str):
                self._known_findings.append(finding)

    # ---------------------------------------------------------------- #
    # 评审
    # ---------------------------------------------------------------- #
    def review(
        self,
        paper: Paper,
        hypothesis: Optional[Hypothesis] = None,
        design: Optional[ExperimentDesign] = None,
        analysis: Optional[AnalysisResult] = None,
    ) -> ReviewReport:
        """评审一篇论文。

        Parameters
        ----------
        paper : Paper
            待评审论文
        hypothesis : Hypothesis | None
            关联假设（若有）
        design : ExperimentDesign | None
            关联实验设计
        analysis : AnalysisResult | None
            关联分析结果
        """
        with self._lock:
            comments: list[str] = []
            revisions: list[str] = []

            # 1. 逻辑一致性评分
            logic_score, logic_comments, logic_revisions = self._check_logic(
                paper, hypothesis, design, analysis
            )
            comments.extend(logic_comments)
            revisions.extend(logic_revisions)

            # 2. 方法正确性评分
            method_score, method_comments, method_revisions = (
                self._check_methods(paper, design, analysis)
            )
            comments.extend(method_comments)
            revisions.extend(method_revisions)

            # 3. 文献一致性评分
            lit_score, lit_comments, lit_revisions = self._check_literature(
                paper, analysis
            )
            comments.extend(lit_comments)
            revisions.extend(lit_revisions)

            # 4. 可复现性评分
            rep_score, rep_comments, rep_revisions = (
                self._check_reproducibility(paper, design, analysis)
            )
            comments.extend(rep_comments)
            revisions.extend(rep_revisions)

            # 综合评分（加权平均）
            overall = (
                0.3 * logic_score
                + 0.3 * method_score
                + 0.2 * lit_score
                + 0.2 * rep_score
            )

            # 决策
            if overall >= self.accept_threshold and not revisions:
                decision = "accept"
            elif overall < self.reject_threshold:
                decision = "reject"
            else:
                decision = "revise"

            report = ReviewReport(
                paper_id=paper.paper_id,
                decision=decision,
                overall_score=float(overall),
                logic_score=float(logic_score),
                method_score=float(method_score),
                literature_score=float(lit_score),
                reproducibility_score=float(rep_score),
                comments=comments,
                required_revisions=revisions,
                metadata={
                    "n_comments": len(comments),
                    "n_revisions": len(revisions),
                    "hypothesis_id": (
                        hypothesis.hypothesis_id if hypothesis else None
                    ),
                },
            )

            self._reviews.append(report)
            if len(self._reviews) > self._max_history:
                self._reviews = self._reviews[-self._max_history:]

            return report

    # ---------------------------------------------------------------- #
    # 检查维度
    # ---------------------------------------------------------------- #
    def _check_logic(
        self,
        paper: Paper,
        hypothesis: Optional[Hypothesis],
        design: Optional[ExperimentDesign],
        analysis: Optional[AnalysisResult],
    ) -> tuple[float, list[str], list[str]]:
        """检查逻辑一致性。"""
        score = 1.0
        comments: list[str] = []
        revisions: list[str] = []

        # 检查论文是否包含必要章节
        md = paper.markdown.lower()
        required_sections = ["摘要", "引言", "方法", "结果", "讨论", "结论"]
        # 兼容英文
        required_sections_en = ["abstract", "introduction", "method", "result", "discussion", "conclusion"]
        for cn, en in zip(required_sections, required_sections_en):
            if cn not in md and en not in md:
                score -= 0.15
                revisions.append(f"缺少「{cn}」章节")

        # 检查假设-结论一致性
        if hypothesis and analysis:
            if analysis.decision == "accept":
                if "接受" not in paper.markdown and "accept" not in md.lower():
                    score -= 0.2
                    revisions.append("结论与假设接受状态不一致")
            elif analysis.decision == "reject":
                if "拒绝" not in paper.markdown and "reject" not in md.lower():
                    score -= 0.2
                    revisions.append("结论与假设拒绝状态不一致")

        # 检查贝叶斯因子与决策的一致性
        if analysis:
            if analysis.bayes_factor > 10 and analysis.decision != "accept":
                score -= 0.1
                comments.append("贝叶斯因子高但未接受，需说明理由")
            if analysis.bayes_factor < 0.1 and analysis.decision != "reject":
                score -= 0.1
                comments.append("贝叶斯因子极低但未拒绝，需说明理由")

        if score < 0:
            score = 0.0
        if score >= 0.8:
            comments.append("逻辑结构完整，假设-方法-结论链自洽。")
        return score, comments, revisions

    def _check_methods(
        self,
        paper: Paper,
        design: Optional[ExperimentDesign],
        analysis: Optional[AnalysisResult],
    ) -> tuple[float, list[str], list[str]]:
        """检查方法正确性。"""
        score = 1.0
        comments: list[str] = []
        revisions: list[str] = []

        if design:
            # 样本量检查
            if design.n_repeats < 3:
                score -= 0.2
                revisions.append(
                    f"重复次数过少（{design.n_repeats}），建议≥5"
                )
            elif design.n_repeats < 5:
                score -= 0.1
                comments.append("重复次数偏少，统计功效可能不足。")

            # EIG 检查
            if design.expected_info_gain < 0.1:
                score -= 0.15
                revisions.append("预期信息增益过低，实验价值有限。")

            # 干预参数检查
            if not design.params:
                score -= 0.1
                revisions.append("实验设计缺少具体干预参数。")

        if analysis:
            # 样本量检查
            if analysis.n_samples < 3:
                score -= 0.2
                revisions.append("分析样本量过少，结果不可靠。")

            # 效应量检查
            if analysis.effect_size < 0.01 and analysis.decision == "accept":
                score -= 0.15
                revisions.append("效应量极小却接受假设，需谨慎。")

        if score < 0:
            score = 0.0
        if score >= 0.8 and not revisions:
            comments.append("方法设计合理，统计检验恰当。")
        return score, comments, revisions

    def _check_literature(
        self,
        paper: Paper,
        analysis: Optional[AnalysisResult],
    ) -> tuple[float, list[str], list[str]]:
        """检查文献一致性。"""
        score = 1.0
        comments: list[str] = []
        revisions: list[str] = []

        # 检查是否引用文献
        md = paper.markdown.lower()
        if "参考文献" not in md and "references" not in md and "[" not in md:
            score -= 0.2
            revisions.append("论文缺少参考文献引用。")

        # 检查与已知结论的矛盾
        if analysis and analysis.conclusion:
            conclusion_lower = analysis.conclusion.lower()
            for finding in self._known_findings:
                finding_lower = finding.lower()
                # 简化：检测对立词
                if (
                    ("增加" in conclusion_lower and "减少" in finding_lower)
                    or ("减少" in conclusion_lower and "增加" in finding_lower)
                    or ("increase" in conclusion_lower and "decrease" in finding_lower)
                    or ("decrease" in conclusion_lower and "increase" in finding_lower)
                ):
                    score -= 0.3
                    revisions.append(
                        f"结论与已知发现矛盾：{finding[:60]}"
                    )

        if score < 0:
            score = 0.0
        if score >= 0.8 and not revisions:
            comments.append("结论与现有文献一致。")
        return score, comments, revisions

    def _check_reproducibility(
        self,
        paper: Paper,
        design: Optional[ExperimentDesign],
        analysis: Optional[AnalysisResult],
    ) -> tuple[float, list[str], list[str]]:
        """检查可复现性。"""
        score = 1.0
        comments: list[str] = []
        revisions: list[str] = []

        if design:
            # 检查动作序列是否记录
            if not design.action_sequence:
                score -= 0.2
                revisions.append("缺少实验动作序列，无法复现。")

            # 检查干预变量是否明确
            if not design.intervention_vars:
                score -= 0.15
                revisions.append("干预变量未明确指定。")

        # 检查论文是否包含复现说明
        md = paper.markdown.lower()
        if "复现" not in md and "reproduc" not in md:
            score -= 0.1
            comments.append("建议添加复现说明章节。")

        if score < 0:
            score = 0.0
        if score >= 0.8 and not revisions:
            comments.append("实验细节充分，可复现。")
        return score, comments, revisions

    # ---------------------------------------------------------------- #
    # 查询
    # ---------------------------------------------------------------- #
    def get_snapshot(self) -> dict[str, Any]:
        with self._lock:
            n_accept = sum(1 for r in self._reviews if r.decision == "accept")
            n_revise = sum(1 for r in self._reviews if r.decision == "revise")
            n_reject = sum(1 for r in self._reviews if r.decision == "reject")
            n_total = len(self._reviews)
            return {
                "n_reviews": n_total,
                "n_accept": n_accept,
                "n_revise": n_revise,
                "n_reject": n_reject,
                "accept_rate": (n_accept / n_total) if n_total > 0 else 0.0,
                "avg_score": (
                    float(np.mean([r.overall_score for r in self._reviews]))
                    if self._reviews else 0.0
                ),
                "n_known_findings": len(self._known_findings),
                "accept_threshold": self.accept_threshold,
                "reject_threshold": self.reject_threshold,
            }

    @property
    def reviews(self) -> list[ReviewReport]:
        return list(self._reviews)

    def reset(self) -> None:
        with self._lock:
            self._reviews.clear()
            self._known_findings.clear()
