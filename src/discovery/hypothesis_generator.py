"""Discovery Hypothesis Generator — 从知识图谱/因果图生成候选假设。

三种策略:
  - 因果缺口法：因果图中强度接近 0 的边 → 零假设
  - 类比迁移法：知识图谱远程类比 → 迁移假设
  - 矛盾驱动法：逻辑引擎矛盾 → 解决矛盾假设
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _stable_hash_id(text: str, prefix: str, mod: int = 100000) -> str:
    """确定性 hash ID（跨进程可复现，不依赖 PYTHONHASHSEED）。

    Python 内置 ``hash()`` 在每次进程启动时随机化（PYTHONHASHSEED），
    导致相同假设在不同运行中获得不同 ID，破坏可复现性。
    使用 hashlib.md5 取代，保证确定性。
    """
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return f"{prefix}_{int(digest[:8], 16) % mod:05d}"


@dataclass
class Hypothesis:
    """候选假设。"""

    statement: str
    confidence: float  # 0-1，初始置信度（贝叶斯先验）
    testability: float  # 0-1，可测试性评分
    strategy: str  # "causal_gap" | "analogy_transfer" | "contradiction"
    #: 相关变量（用于实验设计）
    intervention_vars: list[str] = field(default_factory=list)
    observation_vars: list[str] = field(default_factory=list)
    #: 假设的零假设陈述（用于贝叶斯因子比较）
    null_statement: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    #: 唯一 ID
    hypothesis_id: str = ""

    def __post_init__(self) -> None:
        if not self.hypothesis_id:
            # 确定性 hash（不依赖 PYTHONHASHSEED，跨进程可复现）
            self.hypothesis_id = _stable_hash_id(self.statement, "H")


class HypothesisGenerator:
    """假设生成器。

    Parameters
    ----------
    max_hypotheses : int, default 10
        每次生成最多假设数。
    min_confidence : float, default 0.1
        最低初始置信度阈值。
    seed : int, default 42
    """

    def __init__(
        self,
        max_hypotheses: int = 10,
        min_confidence: float = 0.1,
        seed: int = 42,
    ) -> None:
        if max_hypotheses <= 0:
            raise ValueError(f"max_hypotheses must be positive, got {max_hypotheses}")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(f"min_confidence must be in [0,1], got {min_confidence}")

        self.max_hypotheses = int(max_hypotheses)
        self.min_confidence = float(min_confidence)
        self._rng = np.random.default_rng(seed)
        self._generated_count = 0

    # ------------------------------------------------------------------ #
    # 主生成接口
    # ------------------------------------------------------------------ #
    def generate(
        self,
        causal_graph: dict | None = None,
        knowledge_graph: dict | None = None,
        prediction_errors: dict[str, float] | None = None,
        logic_contradictions: list[dict] | None = None,
    ) -> list[Hypothesis]:
        """生成候选假设列表。

        Parameters
        ----------
        causal_graph : dict
            形如 {"nodes": [{"id","label"}], "edges": [{"source","target","strength"}]}
        knowledge_graph : dict
            形如 {"nodes": [...], "edges": [...]}
        prediction_errors : dict
            近期预测误差统计，{"L0": float, "L1": float, "L2": float}
        logic_contradictions : list[dict]
            逻辑引擎检测到的矛盾列表

        Returns
        -------
        list[Hypothesis]
            候选假设，按 testability * confidence 排序。
        """
        hypotheses: list[Hypothesis] = []

        # 策略 1：因果缺口法
        if causal_graph:
            hypotheses.extend(self._from_causal_gaps(causal_graph))

        # 策略 2：类比迁移法
        if knowledge_graph:
            hypotheses.extend(self._from_analogy_transfer(knowledge_graph))

        # 策略 3：矛盾驱动法
        if logic_contradictions:
            hypotheses.extend(self._from_contradictions(logic_contradictions))

        # 额外：高预测误差驱动假设
        if prediction_errors:
            hypotheses.extend(self._from_high_error(prediction_errors))

        # 过滤与排序
        hypotheses = [h for h in hypotheses if h.confidence >= self.min_confidence]
        hypotheses.sort(key=lambda h: h.testability * h.confidence, reverse=True)
        hypotheses = hypotheses[: self.max_hypotheses]

        self._generated_count += len(hypotheses)
        return hypotheses

    # ------------------------------------------------------------------ #
    # 策略实现
    # ------------------------------------------------------------------ #
    def _from_causal_gaps(self, causal_graph: dict) -> list[Hypothesis]:
        """因果缺口法：强度接近 0 的边 → 零假设。"""
        hypotheses: list[Hypothesis] = []
        edges = causal_graph.get("edges", [])
        nodes = {n.get("id", i): n.get("label", f"var_{n.get('id', i)}") for i, n in enumerate(causal_graph.get("nodes", []))}

        for edge in edges:
            strength = abs(float(edge.get("strength", 0.0)))
            if strength < 0.1:  # 弱因果边
                src = nodes.get(edge.get("source"), str(edge.get("source")))
                tgt = nodes.get(edge.get("target"), str(edge.get("target")))
                hyp = Hypothesis(
                    statement=f"{src} 与 {tgt} 之间不存在显著的因果关系",
                    confidence=max(0.3, 1.0 - strength),
                    testability=0.8,
                    strategy="causal_gap",
                    intervention_vars=[str(src)],
                    observation_vars=[str(tgt)],
                    null_statement=f"{src} 与 {tgt} 相互独立",
                )
                hypotheses.append(hyp)
        return hypotheses

    def _from_analogy_transfer(self, knowledge_graph: dict) -> list[Hypothesis]:
        """类比迁移法：从知识图谱远程节点提取迁移假设。"""
        hypotheses: list[Hypothesis] = []
        # 兼容两种 schema：标准 {"nodes":[...], "edges":[...]} 和
        # 增量 {"new_nodes":[...], "new_edges":[...]}。节点可以是
        # dict（{"id","label"}）或裸字符串。
        nodes = knowledge_graph.get("nodes", knowledge_graph.get("new_nodes", []))
        edges = knowledge_graph.get("edges", knowledge_graph.get("new_edges", []))

        if len(nodes) < 2:
            return hypotheses

        # 归一化节点：dict → (id, label)，str → (str, str)
        node_ids: list[str] = []
        for n in nodes:
            if isinstance(n, dict):
                nid = str(n.get("id", n.get("label", "")))
            else:
                nid = str(n)
            node_ids.append(nid)

        # 找度数最低的节点（远程/孤立概念）
        node_degree: dict[str, int] = {nid: 0 for nid in node_ids}
        for e in edges:
            s = str(e.get("source", e.get("src", "")))
            t = str(e.get("target", e.get("dst", "")))
            if s in node_degree:
                node_degree[s] += 1
            if t in node_degree:
                node_degree[t] += 1

        # 取连接最少的两个节点做类比
        sorted_nodes = sorted(node_degree.items(), key=lambda x: x[1])
        if len(sorted_nodes) >= 2:
            a, _ = sorted_nodes[0]
            b, _ = sorted_nodes[1]
            hyp = Hypothesis(
                statement=f"{a} 的行为规律可迁移适用于 {b}",
                confidence=0.2,
                testability=0.5,
                strategy="analogy_transfer",
                intervention_vars=[a],
                observation_vars=[b],
                null_statement=f"{a} 与 {b} 的行为规律无关",
            )
            hypotheses.append(hyp)
        return hypotheses

    def _from_contradictions(self, contradictions: list[dict]) -> list[Hypothesis]:
        """矛盾驱动法：检测矛盾生成解决假设。"""
        hypotheses: list[Hypothesis] = []
        for c in contradictions[:3]:  # 限制数量
            rule = c.get("rule", "unknown")
            inferred = c.get("inferred", "?")
            actual = c.get("actual", "?")
            hyp = Hypothesis(
                statement=f"规则 {rule} 的推断 ({inferred}) 与观测 ({actual}) 矛盾，存在未建模的中间变量",
                confidence=0.6,
                testability=0.7,
                strategy="contradiction",
                intervention_vars=[str(rule)],
                observation_vars=[str(inferred), str(actual)],
                null_statement=f"规则 {rule} 正确且无中间变量",
            )
            hypotheses.append(hyp)
        return hypotheses

    def _from_high_error(self, errors: dict[str, float]) -> list[Hypothesis]:
        """高预测误差驱动假设。"""
        hypotheses: list[Hypothesis] = []
        for layer, err in errors.items():
            if float(err) > 1.0:  # 高误差
                hyp = Hypothesis(
                    statement=f"层 {layer} 的高预测误差 ({err:.3f}) 暗示存在未建模的关键特征",
                    confidence=0.4,
                    testability=0.6,
                    strategy="causal_gap",
                    intervention_vars=[f"{layer}_features"],
                    observation_vars=[f"{layer}_error"],
                    null_statement=f"层 {layer} 的预测误差源于随机噪声",
                )
                hypotheses.append(hyp)
        return hypotheses

    # ------------------------------------------------------------------ #
    # 文献驱动假设生成（升级 3.2）
    # ------------------------------------------------------------------ #
    def generate_from_literature(
        self,
        literature_graph: dict | None = None,
        contradictions: list[tuple] | None = None,
        structural_holes: list[tuple] | None = None,
    ) -> list[Hypothesis]:
        """文献驱动的假设生成。

        Parameters
        ----------
        literature_graph : dict | None
            LiteratureMiner.build_graph() 的快照（含 papers/nodes/edges）。
        contradictions : list[tuple] | None
            文献矛盾对列表 (paper_a_id, paper_b_id, score)。
        structural_holes : list[tuple] | None
            知识图谱结构空洞 (concept_a, concept_b, opportunity_score)。

        Returns
        -------
        list[Hypothesis]
            文献驱动的假设列表。
        """
        hypotheses: list[Hypothesis] = []

        # 策略 A：矛盾解决假设
        if contradictions:
            hypotheses.extend(self._from_literature_contradictions(
                contradictions, literature_graph
            ))

        # 策略 B：跨学科链接假设（结构空洞）
        if structural_holes:
            hypotheses.extend(self._from_structural_holes(structural_holes))

        # 策略 C：高被引概念的新颖应用
        if literature_graph:
            hypotheses.extend(self._from_concept_clusters(literature_graph))

        # 过滤与排序
        hypotheses = [h for h in hypotheses if h.confidence >= self.min_confidence]
        hypotheses.sort(key=lambda h: h.testability * h.confidence, reverse=True)
        hypotheses = hypotheses[: self.max_hypotheses]

        self._generated_count += len(hypotheses)
        return hypotheses

    def _from_literature_contradictions(
        self,
        contradictions: list[tuple],
        literature_graph: dict | None,
    ) -> list[Hypothesis]:
        """从文献矛盾生成"解决矛盾"假设。"""
        hypotheses: list[Hypothesis] = []
        # 构建 paper_id → title 索引
        paper_titles: dict[str, str] = {}
        if literature_graph:
            for p in literature_graph.get("papers", []):
                pid = p.get("paper_id", "") if isinstance(p, dict) else ""
                title = p.get("title", "") if isinstance(p, dict) else ""
                if pid:
                    paper_titles[pid] = title

        for c in contradictions[:3]:
            if len(c) < 3:
                continue
            pid_a, pid_b, score = c[0], c[1], float(c[2])
            title_a = paper_titles.get(pid_a, pid_a)[:40]
            title_b = paper_titles.get(pid_b, pid_b)[:40]
            hyp = Hypothesis(
                statement=(
                    f"文献 '{title_a}' 与 '{title_b}' 存在矛盾（强度 {score:.2f}），"
                    f"存在未考虑的调节变量可同时解释两者"
                ),
                confidence=max(0.4, min(0.8, score)),
                testability=0.7,
                strategy="contradiction",
                intervention_vars=["moderator_variable"],
                observation_vars=[title_a, title_b],
                null_statement="两文献矛盾源于测量噪声，无中间变量",
            )
            hypotheses.append(hyp)
        return hypotheses

    def _from_structural_holes(
        self, structural_holes: list[tuple]
    ) -> list[Hypothesis]:
        """从知识图谱结构空洞发现跨学科链接。"""
        hypotheses: list[Hypothesis] = []
        for hole in structural_holes[:3]:
            if len(hole) < 3:
                continue
            concept_a, concept_b, opportunity = hole[0], hole[1], float(hole[2])
            hyp = Hypothesis(
                statement=(
                    f"概念 '{concept_a}' 与 '{concept_b}' 之间存在未探索的"
                    f"跨学科联系（机会评分 {opportunity:.2f}）"
                ),
                confidence=0.3,
                testability=0.5,
                strategy="analogy_transfer",
                intervention_vars=[concept_a],
                observation_vars=[concept_b],
                null_statement=f"{concept_a} 与 {concept_b} 无跨学科关联",
            )
            hypotheses.append(hyp)
        return hypotheses

    def _from_concept_clusters(
        self, literature_graph: dict
    ) -> list[Hypothesis]:
        """从概念聚类中提取可推广假设。"""
        hypotheses: list[Hypothesis] = []
        concept_index = literature_graph.get("concept_index", {})
        # 取出现次数最多的概念
        sorted_concepts = sorted(
            concept_index.items(), key=lambda x: -len(x[1])
        )
        for concept, paper_ids in sorted_concepts[:2]:
            if len(paper_ids) < 2:
                continue
            hyp = Hypothesis(
                statement=(
                    f"概念 '{concept}' 在 {len(paper_ids)} 篇文献中反复出现，"
                    f"暗示其可作为普适解释变量"
                ),
                confidence=0.4,
                testability=0.6,
                strategy="causal_gap",
                intervention_vars=[concept],
                observation_vars=["outcome_variable"],
                null_statement=f"{concept} 无普适解释力",
            )
            hypotheses.append(hyp)
        return hypotheses

    # ------------------------------------------------------------------ #
    # 访问器
    # ------------------------------------------------------------------ #
    @property
    def generated_count(self) -> int:
        return self._generated_count

    def snapshot(self) -> dict:
        return {
            "max_hypotheses": self.max_hypotheses,
            "min_confidence": self.min_confidence,
            "generated_count": self._generated_count,
        }
