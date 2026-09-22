from __future__ import annotations

from threading import Thread

import numpy as np
import pytest

from zero_data_model.cogmem.semantic_index import SemanticIndex


# --------------------------------------------------------------------------- #
# 构造与属性
# --------------------------------------------------------------------------- #


def test_construct_default():
    idx = SemanticIndex()
    assert idx.dim == 64
    assert idx.max_size == 10000
    assert idx.size == 0


def test_construct_with_args():
    idx = SemanticIndex(dim=32, max_size=5)
    assert idx.dim == 32
    assert idx.max_size == 5


# --------------------------------------------------------------------------- #
# add / query / query_by_tag
# --------------------------------------------------------------------------- #


def test_add_increases_size():
    idx = SemanticIndex(dim=4, max_size=10)
    rng = np.random.default_rng(42)
    idx.add(rng.normal(0, 1, 4), node_id=0)
    assert idx.size == 1


def test_query_returns_sorted_topk():
    idx = SemanticIndex(dim=4, max_size=10)
    # 3 个正交基向量
    idx.add(np.array([1.0, 0.0, 0.0, 0.0]), 0)
    idx.add(np.array([0.0, 1.0, 0.0, 0.0]), 1)
    idx.add(np.array([0.0, 0.0, 1.0, 0.0]), 2)
    res = idx.query(np.array([1.0, 0.0, 0.0, 0.0]), k=2)
    assert len(res) == 2
    # 应返回 (node_id, sim, metadata)
    nid, sim, _meta = res[0]
    assert nid == 0
    assert sim > 0.99
    # 应降序排列
    assert res[0][1] >= res[1][1]


def test_query_empty_index_returns_empty():
    idx = SemanticIndex(dim=4, max_size=10)
    res = idx.query(np.array([1.0, 0.0, 0.0, 0.0]), k=5)
    assert res == []


def test_query_by_tag_filters():
    idx = SemanticIndex(dim=4, max_size=10)
    idx.add(np.array([1.0, 0.0, 0.0, 0.0]), 0, metadata={"tags": ["alpha"]})
    idx.add(np.array([0.0, 1.0, 0.0, 0.0]), 1, metadata={"tag": "beta"})
    idx.add(np.array([0.0, 0.0, 1.0, 0.0]), 2, metadata={"tags": ["alpha"]})
    res = idx.query_by_tag("alpha", k=5)
    assert len(res) == 2
    nids = {nid for nid, _meta in res}
    assert nids == {0, 2}


def test_query_by_tag_with_metadata_copy():
    """metadata 浅拷贝存储：add() 时 dict(metadata) 拷贝顶层 dict，
    调用方新增顶层键不影响内部。"""
    idx = SemanticIndex(dim=4, max_size=10)
    orig_meta = {"tags": ["alpha"]}
    idx.add(np.array([1.0, 0.0, 0.0, 0.0]), 0, metadata=orig_meta)
    # 在原 dict 上加新键（不影响内部）
    orig_meta["new_key"] = "leak"
    # 再次查询不应包含新键
    res = idx.query_by_tag("alpha", k=1)
    assert len(res) == 1
    _nid, meta = res[0]
    assert "new_key" not in meta
    assert "tags" in meta


def test_add_copies_input_metadata():
    """metadata 顶层 dict 被浅拷贝存储，新增顶层键不影响内部。"""
    idx = SemanticIndex(dim=4, max_size=10)
    meta = {"tags": ["x"]}
    idx.add(np.array([1.0, 0.0, 0.0, 0.0]), 0, metadata=meta)
    meta["injected"] = "leak"
    res = idx.query_by_tag("x", k=5)
    assert len(res) == 1
    _nid, stored_meta = res[0]
    assert "injected" not in stored_meta


# --------------------------------------------------------------------------- #
# 军事级修复点
# --------------------------------------------------------------------------- #


def test_add_copies_input_array():
    """add 后修改原数组不影响内部存储（别名隔离）。"""
    idx = SemanticIndex(dim=4, max_size=10)
    arr = np.array([1.0, 0.0, 0.0, 0.0])
    idx.add(arr, node_id=0)
    # 修改原数组
    arr[0] = 999.0
    # 内部存储应保持 [1,0,0,0]，故仍与 [1,0,0,0] 最相似
    res = idx.query(np.array([1.0, 0.0, 0.0, 0.0]), k=1)
    assert len(res) == 1
    _nid, sim, _meta = res[0]
    assert sim > 0.99  # 几乎 1.0，未被污染


def test_query_k_zero_returns_empty():
    """off-by-one 修复：k=0 必须返回空列表。"""
    idx = SemanticIndex(dim=4, max_size=10)
    idx.add(np.array([1.0, 0.0, 0.0, 0.0]), 0)
    res = idx.query(np.array([1.0, 0.0, 0.0, 0.0]), k=0)
    assert res == []


def test_query_by_tag_k_zero_returns_empty():
    """query_by_tag k=0 也必须返回空列表（off-by-one 修复点）。"""
    idx = SemanticIndex(dim=4, max_size=10)
    idx.add(np.array([1.0, 0.0, 0.0, 0.0]), 0, metadata={"tags": ["a"]})
    res = idx.query_by_tag("a", k=0)
    assert res == []


def test_query_with_nan_input_returns_zero_similarity():
    """NaN 防护：含 NaN 的查询向量不应崩溃，相似度按 0 处理。"""
    idx = SemanticIndex(dim=4, max_size=10)
    idx.add(np.array([1.0, 0.0, 0.0, 0.0]), 0)
    res = idx.query(np.array([np.nan, 0.0, 0.0, 0.0]), k=1)
    assert len(res) == 1
    _nid, sim, _meta = res[0]
    assert sim == 0.0


def test_add_with_nan_vec_does_not_crash():
    """NaN 防护：含 NaN 的语义向量 add 不应崩溃。"""
    idx = SemanticIndex(dim=4, max_size=10)
    idx.add(np.array([np.nan, 0.0, 0.0, 0.0]), 0)
    assert idx.size == 1
    # 后续查询仍应工作
    res = idx.query(np.array([1.0, 0.0, 0.0, 0.0]), k=5)
    assert len(res) == 1


def test_deque_maxlen_fifo_eviction():
    """deque maxlen 限长：超过 max_size 后丢弃最旧条目。"""
    idx = SemanticIndex(dim=4, max_size=3)
    for i in range(10):
        v = np.zeros(4)
        v[i % 4] = 1.0
        idx.add(v, node_id=i)
    assert idx.size == 3
    # 应保留最后 3 个：node_id 7,8,9
    res = idx.query(np.zeros(4), k=10)
    nids = {nid for nid, _sim, _meta in res}
    assert nids == {7, 8, 9}


def test_concurrent_add_query_does_not_crash():
    """线程安全：并发 add 与 query 不抛异常。"""
    idx = SemanticIndex(dim=4, max_size=1000)
    rng = np.random.default_rng(42)
    base = rng.normal(0, 1, 4)

    def adder() -> None:
        for i in range(100):
            idx.add(base, node_id=i)

    def querier() -> None:
        for _ in range(50):
            idx.query(base, k=5)

    threads = [Thread(target=adder) for _ in range(3)]
    threads.append(Thread(target=querier))
    threads.append(Thread(target=querier))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # 3 个 adder 各加 100 个 → 共 300；max_size=1000 故全部保留
    assert idx.size == 300
