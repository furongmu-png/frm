from __future__ import annotations

from threading import Thread

import numpy as np
import pytest

from zero_data_model.cogmem.episodic_graph import EpisodicGraph, Episode


def _make_state(rng: np.random.Generator, dim: int = 8) -> np.ndarray:
    return rng.normal(0.0, 1.0, dim)


# --------------------------------------------------------------------------- #
# 构造与只读属性
# --------------------------------------------------------------------------- #


def test_construct_default():
    g = EpisodicGraph()
    assert g.node_count == 0
    assert g.edge_count == 0
    assert g.max_nodes == 5000
    assert g.similarity_threshold == 0.95


def test_construct_with_args():
    g = EpisodicGraph(similarity_threshold=0.5, max_nodes=10)
    assert g.similarity_threshold == 0.5
    assert g.max_nodes == 10


# --------------------------------------------------------------------------- #
# insert / plan / query / get_recent
# --------------------------------------------------------------------------- #


def test_insert_creates_node():
    g = EpisodicGraph(similarity_threshold=0.99, max_nodes=100)
    rng = np.random.default_rng(42)
    state = _make_state(rng)
    action = _make_state(rng, dim=4)
    next_state = _make_state(rng)
    sid, nid = g.insert(state, action, next_state, free_energy=1.0, step=0)
    assert sid == 0
    assert nid == 1
    assert g.node_count == 2
    assert g.edge_count == 1


def test_insert_dedup_similar_state():
    """相似度高于阈值时复用既有节点（不创建新节点）。"""
    g = EpisodicGraph(similarity_threshold=0.5, max_nodes=100)
    rng = np.random.default_rng(42)
    state = _make_state(rng)
    # next_state 仅微小扰动 → 应被去重为同一节点
    next_state = state + 1e-6
    sid, nid = g.insert(state, None, next_state, free_energy=0.5, step=0)
    # next_state 与 state 相似 → 复用 sid
    assert nid == sid
    assert g.node_count == 1


def test_insert_none_next_state():
    g = EpisodicGraph()
    rng = np.random.default_rng(7)
    state = _make_state(rng)
    sid, nid = g.insert(state, None, None, free_energy=0.1, step=0)
    assert nid is None
    assert g.edge_count == 0
    assert g.node_count == 1


def test_plan_finds_path():
    """Dijkstra 在三节点链上找到唯一路径。"""
    g = EpisodicGraph(similarity_threshold=0.99, max_nodes=100)
    rng = np.random.default_rng(42)
    s0 = _make_state(rng)
    s1 = _make_state(rng)
    s2 = _make_state(rng)
    # 构造链 s0 -> s1 -> s2
    g.insert(s0, None, s1, 1.0, 0)
    g.insert(s1, None, s2, 1.0, 1)
    path = g.plan(s0, s2, horizon=10)
    assert path is not None
    assert len(path) >= 2
    # 起终点应位于路径两端
    assert path[0] == 0
    assert path[-1] == 2


def test_plan_unreachable_returns_none():
    g = EpisodicGraph(similarity_threshold=0.99, max_nodes=100)
    rng = np.random.default_rng(42)
    s0 = _make_state(rng)
    s1 = _make_state(rng)
    # 仅插入 s0，goal_state 不可匹配 → None
    g.insert(s0, None, None, 0.0, 0)
    path = g.plan(s0, s1)
    assert path is None


def test_plan_horizon_exceeded_returns_none():
    """路径跳数超过 horizon 时返回 None。"""
    g = EpisodicGraph(similarity_threshold=0.99, max_nodes=100)
    rng = np.random.default_rng(42)
    states = [_make_state(rng) for _ in range(5)]
    for i in range(4):
        g.insert(states[i], None, states[i + 1], 1.0, i)
    # horizon=1 但链长 4 跳 → None
    path = g.plan(states[0], states[4], horizon=1)
    assert path is None


def test_query_topk_semantic():
    g = EpisodicGraph(similarity_threshold=0.99, max_nodes=100)
    rng = np.random.default_rng(42)
    # 插入 3 个带有不同语义向量的节点
    sems = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0])]
    for i, sem in enumerate(sems):
        g.insert(
            _make_state(rng), None, None, 0.1, i, semantic_vec=sem
        )
    # 查询应能命中与 [1,0,0] 最相似的节点
    res = g.query(np.array([1.0, 0.0, 0.0]), k=3)
    assert len(res) == 3
    top_id, top_sim = res[0]
    assert top_sim > 0.99


def test_get_recent_returns_latest_step():
    g = EpisodicGraph(similarity_threshold=0.99, max_nodes=100)
    rng = np.random.default_rng(42)
    for step in range(5):
        g.insert(_make_state(rng), None, None, 0.1, step)
    recent = g.get_recent(n=2)
    assert len(recent) == 2
    assert recent[0].step >= recent[1].step
    # 最大 step 应为 4
    assert recent[0].step == 4


# --------------------------------------------------------------------------- #
# 军事级修复点
# --------------------------------------------------------------------------- #


def test_insert_with_nan_semantic_vec_does_not_crash():
    """NaN 防护：含 NaN 的 semantic_vec 不应导致崩溃或污染节点。"""
    g = EpisodicGraph(similarity_threshold=0.99, max_nodes=100)
    rng = np.random.default_rng(42)
    state = _make_state(rng)
    bad_vec = np.array([1.0, np.nan, 0.0])
    # 不应抛异常
    sid, _ = g.insert(state, None, None, 0.1, 0, semantic_vec=bad_vec)
    assert sid == 0
    # 节点存在且后续查询不应崩溃
    res = g.query(np.array([1.0, 0.0, 0.0]), k=5)
    assert isinstance(res, list)


def test_plan_handles_negative_edge_weights():
    """Dijkstra 负边权被截断到 0：路径仍可返回且距离有限。"""
    g = EpisodicGraph(similarity_threshold=0.99, max_nodes=100)
    rng = np.random.default_rng(42)
    s0 = _make_state(rng)
    s1 = _make_state(rng)
    s2 = _make_state(rng)
    # 插入时 free_energy 为负 → avg_free_energy 也为负
    g.insert(s0, None, s1, -5.0, 0)
    g.insert(s1, None, s2, -3.0, 1)
    path = g.plan(s0, s2, horizon=10)
    assert path is not None
    assert path[0] == 0
    assert path[-1] == 2


def test_plan_skips_nan_edge_weights():
    """NaN 边权被视为不可达：构造一个 NaN 边路径应失败。"""
    g = EpisodicGraph(similarity_threshold=0.99, max_nodes=100)
    rng = np.random.default_rng(42)
    s0 = _make_state(rng)
    s1 = _make_state(rng)
    s2 = _make_state(rng)
    # 手动建立结构后注入 NaN 边权
    g.insert(s0, None, s1, 1.0, 0)
    g.insert(s1, None, s2, 1.0, 1)
    # 把 (s1->s2) 边权改为 NaN
    for k, edge in g.edges.items():
        edge.avg_free_energy = float("nan")
    # 所有边都 NaN → 不可达 → None
    path = g.plan(s0, s2, horizon=10)
    # 不可达或异常路径都接受，关键是 not crash
    assert path is None or path == []


def test_concurrent_insert_no_id_collision():
    """_next_id 原子性：并发 insert 不产生 id 碰撞。"""
    g = EpisodicGraph(similarity_threshold=0.999, max_nodes=10000)
    rng = np.random.default_rng(42)
    states = [rng.normal(0.0, 1.0, 16) for _ in range(200)]

    def worker(start: int, end: int) -> None:
        for i in range(start, end):
            g.insert(states[i % len(states)], None, None, 0.1, i)

    threads = [
        Thread(target=worker, args=(0, 100)),
        Thread(target=worker, args=(100, 200)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # 收集所有节点 id，确认无重复
    ids = list(g.nodes.keys())
    assert len(ids) == len(set(ids))
    # _next_id 应等于节点数（每个 insert 创建一个独立节点）
    assert g._next_id == len(ids)


def test_concurrent_insert_does_not_crash():
    """线程安全：多线程并发 insert 不抛异常。"""
    g = EpisodicGraph(similarity_threshold=0.99, max_nodes=5000)
    rng = np.random.default_rng(0)
    states = [rng.normal(0.0, 1.0, 8) for _ in range(500)]

    def worker() -> None:
        for s in states:
            g.insert(s, None, None, 0.1, 0)

    threads = [Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # 至少要有些节点被插入
    assert g.node_count > 0


def test_eviction_when_max_nodes_exceeded():
    """max_nodes 上限触发淘汰最旧节点。"""
    g = EpisodicGraph(similarity_threshold=0.999, max_nodes=3)
    rng = np.random.default_rng(42)
    # 插入 5 个不同节点，应淘汰 2 个最旧的
    for step in range(5):
        # 每个状态相互正交以避免去重
        s = np.zeros(8)
        s[step] = 1.0
        g.insert(s, None, None, 0.1, step)
    assert g.node_count <= 3
    # 最旧的 step=0,1 应被淘汰
    steps = [ep.step for ep in g.nodes.values()]
    assert 0 not in steps
    assert 1 not in steps
