# src/zero_data_model/knowledge/reasoning_graph.py
"""知识图谱增强推理（ReasoningGraph）。

第二阶段 §1.2：在知识图谱之上叠加三种推理能力：

1. **传递性推理**：若 ``is_a(A,B)`` 和 ``is_a(B,C)`` 都在图中，
   自动推断 ``is_a(A,C)``（传递闭包）。
2. **类比推理**：给定实体 X，在图谱中找到结构相似（邻域重合度高）
   的实体 Y，输出类比关系。
3. **缺省推理**：若鸟会飞（通则），但企鹅是鸟且不会飞（例外），
   自动创建例外规则。

推理结果作为"虚拟观测"输入模型，驱动信念更新。

仅依赖 ``numpy``，独立、自包含、可插拔。与
``experiments/knowledge_graph_builder.py``（阅读历史图谱）正交：
本模块关注**逻辑/分类学推理**，而非共读/潜语义边。
"""
from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any

import numpy as np


# ------------------------------------------------------------------ #
# 虚拟观测
# ------------------------------------------------------------------ #
class VirtualObservation:
    """推理产生的虚拟观测，可注入模型驱动信念更新。

    Attributes
    ----------
    subject:
        主语实体。
    relation:
        关系类型（如 ``"is_a"``）。
    obj:
        宾语实体。
    confidence:
        推理置信度 ``[0, 1]``。
    source:
        推理来源（``"transitive"`` / ``"analogical"`` / ``"default"``）。
    """

    __slots__ = ("subject", "relation", "obj", "confidence", "source")

    def __init__(
        self,
        subject: str,
        relation: str,
        obj: str,
        confidence: float = 1.0,
        source: str = "inference",
    ) -> None:
        """初始化虚拟观测。"""
        self.subject = subject
        self.relation = relation
        self.obj = obj
        self.confidence = max(0.0, min(1.0, float(confidence)))
        self.source = source

    def to_dict(self) -> dict[str, Any]:
        """转为字典（便于序列化与快照）。"""
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.obj,
            "confidence": self.confidence,
            "source": self.source,
        }

    def __repr__(self) -> str:
        return (
            f"VirtualObservation({self.subject} -[{self.relation}]-> "
            f"{self.obj}, conf={self.confidence:.3f}, src={self.source})"
        )


# ------------------------------------------------------------------ #
# 推理图谱
# ------------------------------------------------------------------ #
class ReasoningGraph:
    """知识图谱增强推理引擎。

    维护一个三元组图 ``(subject, relation, object)``，支持传递性、
    类比、缺省推理，并输出虚拟观测。

    Attributes
    ----------
    triples:
        已存储的三元组集合，``{(s, r, o): weight}``。
    exceptions:
        缺省推理产生的例外规则，``{(general_rule_key): [exception, ...]}``。
    """

    def __init__(self, seed: int = 42) -> None:
        """初始化空推理图谱。"""
        self._rng = np.random.default_rng(seed)
        self._lock = threading.RLock()
        # 三元组存储：(subject, relation, object) -> weight
        self.triples: dict[tuple[str, str, str], float] = {}
        # 邻接表（按关系分组）：relation -> {subject -> set(objects)}
        self._adj: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        # 反向邻接表：relation -> {object -> set(subjects)}
        self._radj: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        # 缺省推理的例外规则
        self.exceptions: list[dict[str, Any]] = []
        # 缓存的虚拟观测队列
        self._virtual_obs: list[VirtualObservation] = []

    # ------------------------------------------------------------------ #
    # 事实管理
    # ------------------------------------------------------------------ #
    def add_fact(
        self,
        subject: str,
        relation: str,
        obj: str,
        weight: float = 1.0,
    ) -> None:
        """添加一个三元组事实到图谱。

        Parameters
        ----------
        subject:
            主语实体。
        relation:
            关系类型（如 ``"is_a"``、``"part_of"``）。
        obj:
            宾语实体。
        weight:
            边权重（``[0, 1]``，默认 1.0 = 确定）。
        """
        if not subject or not relation or not obj:
            return
        w = max(0.0, min(1.0, float(weight)))
        with self._lock:
            key = (subject, relation, obj)
            self.triples[key] = w
            self._adj[relation][subject].add(obj)
            self._radj[relation][obj].add(subject)

    def add_facts(self, triples: list[tuple[str, str, str]]) -> int:
        """批量添加三元组。返回添加数量。"""
        n = 0
        for t in triples:
            if len(t) >= 3:
                w = float(t[3]) if len(t) > 3 else 1.0
                self.add_fact(t[0], t[1], t[2], w)
                n += 1
        return n

    def has_fact(self, subject: str, relation: str, obj: str) -> bool:
        """查询三元组是否存在。"""
        with self._lock:
            return (subject, relation, obj) in self.triples

    # ------------------------------------------------------------------ #
    # 1. 传递性推理
    # ------------------------------------------------------------------ #
    def transitive_inference(
        self,
        relation: str = "is_a",
        max_depth: int = 10,
    ) -> list[VirtualObservation]:
        """传递性推理：若 ``is_a(A,B)`` 且 ``is_a(B,C)``，推断 ``is_a(A,C)``。

        使用 BFS 遍历传递闭包，跳过已存在的三元组。
        推理结果存入虚拟观测队列。

        Parameters
        ----------
        relation:
            要做传递推理的关系类型（默认 ``"is_a"``）。
        max_depth:
            最大传递深度（防环）。

        Returns
        -------
        list[VirtualObservation]
            新推断的虚拟观测列表。
        """
        with self._lock:
            adj = self._adj.get(relation, {})
            if not adj:
                return []
            new_obs: list[VirtualObservation] = []
            # 对每个起始节点做 BFS，收集可达的传递后代。
            for start in list(adj.keys()):
                visited: set[str] = {start}
                # BFS 队列：(当前节点, 累积置信度, 深度)
                queue: list[tuple[str, float, int]] = [
                    (start, 1.0, 0)
                ]
                while queue:
                    node, conf, depth = queue.pop(0)
                    if depth >= max_depth:
                        continue
                    for neighbor in adj.get(node, set()):
                        if neighbor in visited:
                            continue
                        visited.add(neighbor)
                        # 传递置信度 = 路径置信度的乘积（模糊逻辑 AND）。
                        edge_w = self.triples.get(
                            (node, relation, neighbor), 1.0
                        )
                        new_conf = conf * edge_w * 0.9  # 每跳衰减 10%
                        # 跳过已存在的直接边。
                        if not self.has_fact(start, relation, neighbor):
                            obs = VirtualObservation(
                                subject=start,
                                relation=relation,
                                obj=neighbor,
                                confidence=new_conf,
                                source="transitive",
                            )
                            new_obs.append(obs)
                            self._virtual_obs.append(obs)
                        # 继续扩展。
                        queue.append((neighbor, new_conf, depth + 1))
            return new_obs

    # ------------------------------------------------------------------ #
    # 2. 类比推理
    # ------------------------------------------------------------------ #
    def analogical_reasoning(
        self,
        entity: str,
        top_k: int = 3,
        relation: str | None = None,
    ) -> list[tuple[str, float]]:
        """类比推理：找到与给定实体结构相似的实体。

        结构相似度 = 邻域 Jaccard 相似度（共同邻居比例）。
        若 ``relation`` 指定，只考虑该关系的邻域；否则考虑所有关系。

        Parameters
        ----------
        entity:
            目标实体名。
        top_k:
            返回最相似的 top-k 个实体。
        relation:
            限定关系类型（``None`` = 全部关系）。

        Returns
        -------
        list[tuple[str, float]]
            ``(entity_name, similarity)`` 列表，按相似度降序。
        """
        with self._lock:
            # 收集目标实体的邻域签名。
            target_nbrs = self._get_neighborhood(entity, relation)
            if not target_nbrs:
                return []
            # 收集所有其他实体。
            all_entities = self._get_all_entities(relation)
            similarities: list[tuple[str, float]] = []
            for other in all_entities:
                if other == entity:
                    continue
                other_nbrs = self._get_neighborhood(other, relation)
                if not other_nbrs:
                    continue
                # Jaccard 相似度。
                sim = self._jaccard(target_nbrs, other_nbrs)
                if sim > 0.0:
                    similarities.append((other, sim))
            similarities.sort(key=lambda x: -x[1])
            return similarities[:top_k]

    def _get_neighborhood(
        self, entity: str, relation: str | None
    ) -> set[tuple[str, str]]:
        """获取实体的邻域签名：``{(relation, neighbor)}``。"""
        nbrs: set[tuple[str, str]] = set()
        relations = [relation] if relation else list(self._adj.keys())
        for rel in relations:
            # 出边
            for obj in self._adj.get(rel, {}).get(entity, set()):
                nbrs.add((rel, obj))
            # 入边
            for subj in self._radj.get(rel, {}).get(entity, set()):
                nbrs.add((f"~{rel}", subj))
        return nbrs

    def _get_all_entities(self, relation: str | None) -> set[str]:
        """获取图中所有实体名。"""
        entities: set[str] = set()
        relations = [relation] if relation else list(self._adj.keys())
        for rel in relations:
            for subj, objs in self._adj.get(rel, {}).items():
                entities.add(subj)
                entities.update(objs)
        return entities

    @staticmethod
    def _jaccard(a: set[Any], b: set[Any]) -> float:
        """Jaccard 相似度。"""
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union > 0 else 0.0

    # ------------------------------------------------------------------ #
    # 3. 缺省推理
    # ------------------------------------------------------------------ #
    def default_reasoning(
        self,
        general_subject: str,
        general_relation: str,
        general_obj: str,
        exception_subject: str,
        exception_obj: str,
    ) -> dict[str, Any] | None:
        """缺省推理：通则 + 例外 → 创建例外规则。

        示例：鸟会飞（通则），企鹅是鸟且不会飞（例外）。
        自动创建例外规则：``if penguin then not_flies``。

        Parameters
        ----------
        general_subject:
            通则的主语（如 ``"bird"``）。
        general_relation:
            通则的关系（如 ``"can"``）。
        general_obj:
            通则的宾语（如 ``"fly"``）。
        exception_subject:
            例外实体的主语（如 ``"penguin"``）。
        exception_obj:
            例外实体的宾语（如 ``"not_fly"``）。

        Returns
        -------
        dict | None
            创建的例外规则，或 ``None``（若通则不存在）。
        """
        with self._lock:
            # 检查通则是否存在。
            if not self.has_fact(general_subject, general_relation, general_obj):
                return None
            # 检查例外实体是否属于通则主语的子类（通过 is_a 关系）。
            is_subclass = self.has_fact(
                exception_subject, "is_a", general_subject
            )
            # 创建例外规则。
            exception_rule: dict[str, Any] = {
                "general": f"{general_subject} {general_relation} {general_obj}",
                "exception_subject": exception_subject,
                "exception_obj": exception_obj,
                "is_subclass_of_general": is_subclass,
                "rule": f"if {exception_subject} then {exception_obj} (overrides general)",
            }
            self.exceptions.append(exception_rule)
            # 添加例外事实到图谱。
            self.triples[
                (exception_subject, general_relation, exception_obj)
            ] = 1.0
            self._adj[general_relation][exception_subject].add(exception_obj)
            self._radj[general_relation][exception_obj].add(
                exception_subject
            )
            # 产生虚拟观测。
            obs = VirtualObservation(
                subject=exception_subject,
                relation=general_relation,
                obj=exception_obj,
                confidence=1.0,
                source="default",
            )
            self._virtual_obs.append(obs)
            return exception_rule

    # ------------------------------------------------------------------ #
    # 虚拟观测
    # ------------------------------------------------------------------ #
    def get_virtual_observations(self) -> list[VirtualObservation]:
        """获取并清空虚拟观测队列。"""
        with self._lock:
            obs = list(self._virtual_obs)
            self._virtual_obs.clear()
            return obs

    def peek_virtual_observations(self) -> list[VirtualObservation]:
        """查看虚拟观测队列（不清空）。"""
        with self._lock:
            return list(self._virtual_obs)

    # ------------------------------------------------------------------ #
    # 查询接口
    # ------------------------------------------------------------------ #
    def get_neighbors(
        self, entity: str, relation: str
    ) -> list[str]:
        """获取实体在指定关系下的所有出边邻居。"""
        with self._lock:
            return list(self._adj.get(relation, {}).get(entity, set()))

    def get_ancestors(
        self, entity: str, relation: str = "is_a", max_depth: int = 10
    ) -> list[str]:
        """获取实体的所有祖先（传递闭包）。

        Parameters
        ----------
        entity:
            起始实体。
        relation:
            传递关系（默认 ``"is_a"``）。
        max_depth:
            最大深度。

        Returns
        -------
        list[str]
            祖先列表（不含自身）。
        """
        with self._lock:
            adj = self._adj.get(relation, {})
            ancestors: set[str] = set()
            queue: list[tuple[str, int]] = [(entity, 0)]
            visited: set[str] = {entity}
            while queue:
                node, depth = queue.pop(0)
                if depth >= max_depth:
                    continue
                for parent in adj.get(node, set()):
                    if parent not in visited:
                        visited.add(parent)
                        ancestors.add(parent)
                        queue.append((parent, depth + 1))
            return list(ancestors)

    @property
    def stats(self) -> dict[str, Any]:
        """图谱统计信息。"""
        with self._lock:
            relations = {r for (_, r, _) in self.triples}
            entities = self._get_all_entities(None)
            return {
                "n_triples": len(self.triples),
                "n_entities": len(entities),
                "n_relations": len(relations),
                "n_exceptions": len(self.exceptions),
                "n_pending_virtual_obs": len(self._virtual_obs),
                "relations": sorted(relations),
            }


__all__: list[str] = ["VirtualObservation", "ReasoningGraph"]
