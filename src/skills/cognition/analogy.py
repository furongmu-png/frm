"""类比推理与隐喻生成技能。

利用 Hopfield 记忆检索 + 知识图谱结构匹配：
- 对给定概念 A，检索图谱中具有相似关系结构的概念 B
- 生成自然语言类比："A 之于 C 就像 B 之于 D"
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..base import SkillBase, SkillContext, SkillResult

logger = logging.getLogger(__name__)


class ConceptGraph:
    """简化的概念关系图，用于类比匹配。

    内部存储 (subject, relation, object) 三元组，
    支持结构匹配（同构子图检索）。
    """

    def __init__(self) -> None:
        self._triples: list[tuple[str, str, str]] = []
        self._relations: dict[str, list[tuple[str, str]]] = {}

    def add(self, subject: str, relation: str, obj: str) -> None:
        triple = (subject, relation, obj)
        if triple not in self._triples:
            self._triples.append(triple)
        key = relation
        if key not in self._relations:
            self._relations[key] = []
        self._relations[key].append((subject, obj))

    def get_relations_of(self, concept: str) -> list[tuple[str, str]]:
        """获取概念的所有 (relation, object) 关系。"""
        result = []
        for s, r, o in self._triples:
            if s == concept:
                result.append((r, o))
        return result

    def find_structural_matches(
        self,
        source_concept: str,
        source_relations: list[tuple[str, str]],
        exclude: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """找到与源概念具有相似关系结构的目标概念。

        Returns
        -------
        list of (concept, similarity_score)
        """
        if exclude is None:
            exclude = {source_concept}

        # 收集所有候选概念
        candidates: set[str] = set()
        for s, _, o in self._triples:
            if s not in exclude:
                candidates.add(s)
            if o not in exclude:
                candidates.add(o)

        results = []
        source_rel_set = {(r, o) for r, o in source_relations}

        for candidate in candidates:
            cand_relations = self.get_relations_of(candidate)
            cand_rel_set = {(r, o) for r, o in cand_relations}

            # 关系类型匹配
            source_rel_types = {r for r, _ in source_relations}
            cand_rel_types = {r for r, _ in cand_relations}
            common_types = source_rel_types & cand_rel_types

            if not common_types:
                continue

            # 结构相似度：共同关系类型数 / 总关系类型数
            sim = len(common_types) / max(
                len(source_rel_types | cand_rel_types), 1
            )
            results.append((candidate, float(sim)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:10]


class AnalogyEngine(SkillBase):
    """类比推理引擎。

    利用概念图的结构匹配 + Hopfield 记忆检索，
    生成跨领域类比。

    流程：
    1. 接收源概念 A 和目标关系 C
    2. 在概念图中检索与 A 结构相似的概念 B
    3. 找到 B 的对应关系 D
    4. 生成类比句："A 之于 C 就像 B 之于 D"
    """

    name = "analogy"
    dimension = "cognition"

    def __init__(
        self,
        *,
        latent_dim: int = 16,
        enabled: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(enabled=enabled, **kwargs)
        self._graph = ConceptGraph()
        self._latent_dim = latent_dim
        self._concept_latents: dict[str, np.ndarray] = {}
        self._rng = np.random.default_rng(654)

        # 预填充一些基础概念关系
        self._init_base_concepts()

        self._last_analogy: dict[str, Any] | None = None

    def _init_base_concepts(self) -> None:
        """初始化基础概念图。"""
        base_triples = [
            # 自然
            ("sun", "illuminates", "earth"),
            ("earth", "orbits", "sun"),
            ("moon", "orbits", "earth"),
            ("heart", "pumps", "blood"),
            ("brain", "controls", "body"),
            ("root", "anchors", "tree"),
            ("branch", "extends", "tree"),
            ("leaf", "absorbs", "light"),
            # 社会
            ("teacher", "guides", "student"),
            ("parent", "nurtures", "child"),
            ("leader", "directs", "team"),
            ("engine", "powers", "car"),
            ("foundation", "supports", "building"),
            # 抽象
            ("logic", "structures", "argument"),
            ("rhythm", "structures", "music"),
            ("grammar", "structures", "language"),
            ("algorithm", "structures", "computation"),
            ("memory", "stores", "knowledge"),
            ("library", "stores", "books"),
            ("database", "stores", "data"),
        ]
        for s, r, o in base_triples:
            self._graph.add(s, r, o)
            # 为每个概念分配 latent
            if s not in self._concept_latents:
                self._concept_latents[s] = self._rng.standard_normal(
                    self._latent_dim
                )
                self._concept_latents[s] /= np.linalg.norm(
                    self._concept_latents[s]
                )
            if o not in self._concept_latents:
                self._concept_latents[o] = self._rng.standard_normal(
                    self._latent_dim
                )
                self._concept_latents[o] /= np.linalg.norm(
                    self._concept_latents[o]
                )

    def add_concept(self, subject: str, relation: str, obj: str) -> None:
        """添加概念关系。"""
        self._graph.add(subject, relation, obj)
        for c in (subject, obj):
            if c not in self._concept_latents:
                v = self._rng.standard_normal(self._latent_dim)
                self._concept_latents[c] = v / np.linalg.norm(v)

    def find_analogy(
        self,
        source: str,
        source_relation: str,
        source_object: str,
    ) -> dict[str, Any] | None:
        """为源概念寻找类比。

        生成 "source 之于 source_object 就像 B 之于 D" 的类比。

        Returns
        -------
        dict with keys: source, source_object, analog, analog_object, similarity, sentence
        or None if no match found.
        """
        source_relations = self._graph.get_relations_of(source)
        if not source_relations:
            source_relations = [(source_relation, source_object)]

        matches = self._graph.find_structural_matches(
            source, source_relations, exclude={source, source_object}
        )

        if not matches:
            return None

        best_match, similarity = matches[0]

        # 找到最佳匹配概念的对应关系
        match_relations = self._graph.get_relations_of(best_match)

        # 找到与 source_relation 相同类型的关系
        analog_object = None
        for rel, obj in match_relations:
            if rel == source_relation:
                analog_object = obj
                break

        if analog_object is None and match_relations:
            analog_object = match_relations[0][1]

        if analog_object is None:
            return None

        # 生成类比句
        sentence = (
            f"{source} 之于 {source_object} 就像 "
            f"{best_match} 之于 {analog_object}"
        )

        # Hopfield 检索：找到与源概念 latent 最相似的记忆概念
        hopfield_match = None
        if source in self._concept_latents:
            source_latent = self._concept_latents[source]
            best_sim = -1.0
            for concept, latent in self._concept_latents.items():
                if concept == source:
                    continue
                sim = float(np.dot(source_latent, latent))
                if sim > best_sim:
                    best_sim = sim
                    hopfield_match = concept

        result = {
            "source": source,
            "source_object": source_object,
            "analog": best_match,
            "analog_object": analog_object,
            "similarity": similarity,
            "sentence": sentence,
            "hopfield_match": hopfield_match,
            "hopfield_similarity": best_sim if hopfield_match else 0.0,
        }
        self._last_analogy = result
        return result

    def process(self, ctx: SkillContext) -> SkillResult:
        # 从 context 提取源概念
        source = None
        source_relation = "structures"
        source_object = "system"

        if ctx.raw_observation is not None:
            if isinstance(ctx.raw_observation, dict):
                source = ctx.raw_observation.get("concept")
                source_relation = ctx.raw_observation.get(
                    "relation", source_relation
                )
                source_object = ctx.raw_observation.get(
                    "object", source_object
                )
            elif isinstance(ctx.raw_observation, str):
                # 从文本中提取关键词
                source = ctx.raw_observation.strip().lower().split()[0] if ctx.raw_observation.strip() else None

        if source is None:
            # 从 belief 提取：找到 belief 中最大的维度对应的预设概念
            if ctx.belief is not None and len(ctx.belief) > 0:
                concepts = ["logic", "memory", "rhythm", "heart", "sun", "engine"]
                idx = int(np.argmax(np.abs(ctx.belief[: len(concepts)])))
                source = concepts[idx]
            else:
                source = "logic"

        analogy = self.find_analogy(source, source_relation, source_object)

        if analogy is None:
            return SkillResult(
                name=self.name,
                data={"found": False, "source": source},
            )

        return SkillResult(
            name=self.name,
            data={"found": True, **analogy},
        )
