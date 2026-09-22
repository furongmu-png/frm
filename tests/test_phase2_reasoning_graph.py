# tests/test_phase2_reasoning_graph.py
"""第二阶段 §1.2 知识图谱增强推理单元测试。

验证：
- 传递性推理（北京-中国-亚洲层级关系）
- 类比推理（结构相似实体）
- 缺省推理（鸟会飞，企鹅不会飞）
- 虚拟观测生成
"""
from __future__ import annotations

from zero_data_model.knowledge.reasoning_graph import (
    ReasoningGraph,
    VirtualObservation,
)


# ------------------------------------------------------------------ #
# 事实管理
# ------------------------------------------------------------------ #
class TestFactManagement:
    """测试三元组事实管理。"""

    def test_add_fact(self) -> None:
        rg = ReasoningGraph(seed=42)
        rg.add_fact("cat", "is_a", "mammal")
        assert rg.has_fact("cat", "is_a", "mammal")
        assert not rg.has_fact("cat", "is_a", "fish")

    def test_add_facts_batch(self) -> None:
        rg = ReasoningGraph(seed=42)
        n = rg.add_facts([
            ("a", "is_a", "b"),
            ("b", "is_a", "c"),
        ])
        assert n == 2
        assert rg.has_fact("a", "is_a", "b")

    def test_empty_subject_ignored(self) -> None:
        rg = ReasoningGraph(seed=42)
        rg.add_fact("", "is_a", "x")
        assert len(rg.triples) == 0

    def test_weight_clamped(self) -> None:
        rg = ReasoningGraph(seed=42)
        rg.add_fact("a", "r", "b", weight=5.0)
        assert rg.triples[("a", "r", "b")] == 1.0


# ------------------------------------------------------------------ #
# 传递性推理
# ------------------------------------------------------------------ #
class TestTransitiveInference:
    """测试传递性推理。"""

    def test_beijing_china_asia(self) -> None:
        """验证：北京-中国-亚洲层级关系。"""
        rg = ReasoningGraph(seed=42)
        rg.add_fact("Beijing", "is_a", "China")
        rg.add_fact("China", "is_a", "Asia")
        rg.add_fact("Asia", "is_a", "Earth")

        obs = rg.transitive_inference("is_a")
        inferred = {(o.subject, o.obj) for o in obs}
        # Beijing -> Asia, Beijing -> Earth, China -> Earth
        assert ("Beijing", "Asia") in inferred
        assert ("Beijing", "Earth") in inferred
        assert ("China", "Earth") in inferred

    def test_transitive_confidence_decreases(self) -> None:
        rg = ReasoningGraph(seed=42)
        rg.add_fact("a", "is_a", "b")
        rg.add_fact("b", "is_a", "c")
        rg.add_fact("c", "is_a", "d")
        obs = rg.transitive_inference("is_a")
        # a->c should have higher confidence than a->d
        a_c = [o for o in obs if o.subject == "a" and o.obj == "c"]
        a_d = [o for o in obs if o.subject == "a" and o.obj == "d"]
        assert len(a_c) == 1 and len(a_d) == 1
        assert a_c[0].confidence > a_d[0].confidence

    def test_no_existing_duplicates(self) -> None:
        rg = ReasoningGraph(seed=42)
        rg.add_fact("a", "is_a", "b")
        rg.add_fact("b", "is_a", "c")
        rg.add_fact("a", "is_a", "c")  # already exists
        obs = rg.transitive_inference("is_a")
        # Should NOT re-infer a->c
        assert not any(o.subject == "a" and o.obj == "c" for o in obs)

    def test_cycle_safe(self) -> None:
        rg = ReasoningGraph(seed=42)
        rg.add_fact("a", "is_a", "b")
        rg.add_fact("b", "is_a", "a")  # cycle
        # Should not hang
        obs = rg.transitive_inference("is_a", max_depth=5)
        assert isinstance(obs, list)

    def test_empty_graph(self) -> None:
        rg = ReasoningGraph(seed=42)
        obs = rg.transitive_inference("is_a")
        assert obs == []

    def test_get_ancestors(self) -> None:
        rg = ReasoningGraph(seed=42)
        rg.add_fact("Beijing", "is_a", "China")
        rg.add_fact("China", "is_a", "Asia")
        ancestors = rg.get_ancestors("Beijing", "is_a")
        assert "China" in ancestors
        assert "Asia" in ancestors
        assert "Beijing" not in ancestors


# ------------------------------------------------------------------ #
# 类比推理
# ------------------------------------------------------------------ #
class TestAnalogicalReasoning:
    """测试类比推理。"""

    def test_similar_entities_found(self) -> None:
        rg = ReasoningGraph(seed=42)
        rg.add_fact("cat", "is_a", "mammal")
        rg.add_fact("cat", "eats", "meat")
        rg.add_fact("dog", "is_a", "mammal")
        rg.add_fact("dog", "eats", "meat")
        rg.add_fact("fish", "is_a", "animal")
        rg.add_fact("fish", "eats", "plankton")
        sim = rg.analogical_reasoning("cat", top_k=2)
        assert len(sim) > 0
        # dog shares both relations → highest similarity
        assert sim[0][0] == "dog"
        assert sim[0][1] > 0.5

    def test_dissimilar_entity_low_score(self) -> None:
        rg = ReasoningGraph(seed=42)
        rg.add_fact("cat", "is_a", "mammal")
        rg.add_fact("rock", "made_of", "stone")
        sim = rg.analogical_reasoning("cat", top_k=5)
        # rock shares no neighborhood with cat
        rock_scores = [s for s in sim if s[0] == "rock"]
        assert len(rock_scores) == 0

    def test_top_k_limit(self) -> None:
        rg = ReasoningGraph(seed=42)
        for i in range(10):
            rg.add_fact(f"entity_{i}", "is_a", "thing")
        sim = rg.analogical_reasoning("entity_0", top_k=3)
        assert len(sim) <= 3

    def test_no_neighbors_returns_empty(self) -> None:
        rg = ReasoningGraph(seed=42)
        sim = rg.analogical_reasoning("nonexistent")
        assert sim == []


# ------------------------------------------------------------------ #
# 缺省推理
# ------------------------------------------------------------------ #
class TestDefaultReasoning:
    """测试缺省推理。"""

    def test_penguin_bird_exception(self) -> None:
        """鸟会飞，企鹅是鸟但不会飞 → 创建例外。"""
        rg = ReasoningGraph(seed=42)
        rg.add_fact("bird", "can", "fly")
        rg.add_fact("penguin", "is_a", "bird")
        exc = rg.default_reasoning(
            "bird", "can", "fly", "penguin", "not_fly"
        )
        assert exc is not None
        assert exc["exception_subject"] == "penguin"
        assert exc["is_subclass_of_general"]
        assert rg.has_fact("penguin", "can", "not_fly")

    def test_no_general_rule_returns_none(self) -> None:
        rg = ReasoningGraph(seed=42)
        # No "bird can fly" fact
        result = rg.default_reasoning(
            "bird", "can", "fly", "penguin", "not_fly"
        )
        assert result is None

    def test_non_subclass_still_creates_exception(self) -> None:
        rg = ReasoningGraph(seed=42)
        rg.add_fact("bird", "can", "fly")
        # robot is NOT a bird, but we still create exception
        result = rg.default_reasoning(
            "bird", "can", "fly", "robot", "not_fly"
        )
        assert result is not None
        assert not result["is_subclass_of_general"]


# ------------------------------------------------------------------ #
# 虚拟观测
# ------------------------------------------------------------------ #
class TestVirtualObservations:
    """测试虚拟观测生成。"""

    def test_transitive_produces_virtual_obs(self) -> None:
        rg = ReasoningGraph(seed=42)
        rg.add_fact("a", "is_a", "b")
        rg.add_fact("b", "is_a", "c")
        rg.transitive_inference("is_a")
        obs = rg.get_virtual_observations()
        assert len(obs) > 0
        assert all(o.source == "transitive" for o in obs)

    def test_default_produces_virtual_obs(self) -> None:
        rg = ReasoningGraph(seed=42)
        rg.add_fact("bird", "can", "fly")
        rg.add_fact("penguin", "is_a", "bird")
        rg.default_reasoning("bird", "can", "fly", "penguin", "not_fly")
        obs = rg.get_virtual_observations()
        default_obs = [o for o in obs if o.source == "default"]
        assert len(default_obs) == 1

    def test_get_clears_queue(self) -> None:
        rg = ReasoningGraph(seed=42)
        rg.add_fact("a", "is_a", "b")
        rg.add_fact("b", "is_a", "c")
        rg.transitive_inference("is_a")
        first = rg.get_virtual_observations()
        second = rg.get_virtual_observations()
        assert len(first) > 0
        assert len(second) == 0

    def test_peek_does_not_clear(self) -> None:
        rg = ReasoningGraph(seed=42)
        rg.add_fact("a", "is_a", "b")
        rg.add_fact("b", "is_a", "c")
        rg.transitive_inference("is_a")
        peeked = rg.peek_virtual_observations()
        assert len(peeked) > 0
        # Queue still has items
        assert len(rg.peek_virtual_observations()) == len(peeked)

    def test_virtual_observation_to_dict(self) -> None:
        vo = VirtualObservation("a", "is_a", "b", confidence=0.9, source="t")
        d = vo.to_dict()
        assert d["subject"] == "a"
        assert d["relation"] == "is_a"
        assert d["object"] == "b"
        assert d["confidence"] == 0.9
        assert d["source"] == "t"


# ------------------------------------------------------------------ #
# 统计与查询
# ------------------------------------------------------------------ #
class TestStats:
    """测试统计接口。"""

    def test_stats(self) -> None:
        rg = ReasoningGraph(seed=42)
        rg.add_fact("a", "is_a", "b")
        rg.add_fact("b", "part_of", "c")
        s = rg.stats
        assert s["n_triples"] == 2
        assert s["n_entities"] == 3
        assert "is_a" in s["relations"]
        assert "part_of" in s["relations"]

    def test_get_neighbors(self) -> None:
        rg = ReasoningGraph(seed=42)
        rg.add_fact("a", "is_a", "b")
        rg.add_fact("a", "is_a", "c")
        nbrs = rg.get_neighbors("a", "is_a")
        assert set(nbrs) == {"b", "c"}
