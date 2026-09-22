from __future__ import annotations

from threading import Thread

import numpy as np
import pytest

from zero_data_model.plasticity.architect import ArchitectureOptimizer


# --------------------------------------------------------------------------- #
# 构造
# --------------------------------------------------------------------------- #


def test_construct_default():
    ao = ArchitectureOptimizer()
    assert ao.dim == 64
    assert ao.eval_interval == 100
    assert ao.split_threshold_steps == 200
    assert ao.prune_threshold_steps == 500
    assert ao.split_error_ratio == 0.3
    assert ao.prune_error_ratio == 0.01
    # 初始统计
    s = ao.stats
    assert s["active_count"] == 0
    assert s["dormant_count"] == 0
    assert s["total_splits"] == 0
    assert s["total_prunes"] == 0


def test_construct_with_args():
    ao = ArchitectureOptimizer(
        dim=8, seed=7, eval_interval=1,
        split_threshold_steps=2, prune_threshold_steps=3,
    )
    assert ao.dim == 8
    assert ao.eval_interval == 1
    assert ao.split_threshold_steps == 2


# --------------------------------------------------------------------------- #
# record_errors / evaluate / reactivate
# --------------------------------------------------------------------------- #


class _DummyModule:
    """简单 mock 模块用于 evaluate。"""
    def __init__(self, name: str) -> None:
        self.name = name


def test_record_errors_creates_history():
    ao = ArchitectureOptimizer()
    ao.record_errors(["A", "B"], [0.1, 0.2])
    # 每个模块的 deque 已被创建
    assert "A" in ao._error_history
    assert "B" in ao._error_history
    assert len(ao._error_history["A"]) == 1


def test_evaluate_returns_expected_keys():
    ao = ArchitectureOptimizer(eval_interval=1)
    modules = [_DummyModule("A"), _DummyModule("B")]
    result = ao.evaluate(modules, cycle=0)
    assert "actions" in result
    assert "splits" in result
    assert "prunes" in result
    assert "dormant_count" in result
    assert "module_count" in result
    assert result["module_count"] == 2


def test_evaluate_skips_off_interval():
    """非评估周期：返回空动作但保持 module_count。"""
    ao = ArchitectureOptimizer(eval_interval=100)
    modules = [_DummyModule("A")]
    # 第 1 次调用 → step_count=1，1%100!=0 → 跳过评估
    r = ao.evaluate(modules, cycle=0)
    assert r["actions"] == []
    assert r["splits"] == []
    assert r["prunes"] == []
    assert r["module_count"] == 1


def test_reactivate_returns_false_for_missing():
    ao = ArchitectureOptimizer()
    modules = []
    assert ao.reactivate(modules, "nonexistent") is False


# --------------------------------------------------------------------------- #
# 军事级修复点
# --------------------------------------------------------------------------- #


def test_evaluate_does_not_mutate_modules_list():
    """核心修复点：evaluate 不变更输入 modules 列表长度。"""
    ao = ArchitectureOptimizer(eval_interval=1, split_threshold_steps=1,
                                prune_threshold_steps=1)
    modules = [_DummyModule("A"), _DummyModule("B"), _DummyModule("C")]
    original_len = len(modules)
    # 让 A 占满误差，B/C 接近 0，触发 split/prune
    for _ in range(5):
        ao.record_errors(["A", "B", "C"], [1.0, 0.0, 0.0])
    ao.evaluate(modules, cycle=0)
    ao.evaluate(modules, cycle=1)
    # 即使触发了 split/prune，输入列表长度不变
    assert len(modules) == original_len


def test_evaluate_handles_nan_error_ratios():
    """NaN 误差比率处理：NaN 不应导致崩溃。"""
    ao = ArchitectureOptimizer(eval_interval=1)
    modules = [_DummyModule("A")]
    # 直接注入 NaN 到 error_history
    ao._error_history["A"] = __import__("collections").deque([float("nan")], maxlen=500)
    result = ao.evaluate(modules, cycle=0)
    # 不应抛异常，total 被钳为 0 → 返回安全默认
    assert "module_count" in result
    assert result["module_count"] == 1


def test_evaluate_returns_actions_splits_prunes_dormant():
    """返回值含 actions/splits/prunes/dormant_count 字段。"""
    ao = ArchitectureOptimizer(eval_interval=1, split_threshold_steps=1,
                                prune_threshold_steps=1)
    modules = [_DummyModule("A"), _DummyModule("B")]
    for _ in range(3):
        ao.record_errors(["A", "B"], [1.0, 0.0])
    # 触发评估（split_threshold=1 → 高 streak 1 即触发）
    r1 = ao.evaluate(modules, cycle=0)
    r2 = ao.evaluate(modules, cycle=1)
    # 累计的 splits + prunes > 0（A 高误差 → split；B 低误差 → prune）
    assert isinstance(r1["actions"], list)
    assert isinstance(r2["actions"], list)
    # 至少有一个动作产生
    total_actions = len(r1["actions"]) + len(r2["actions"])
    assert total_actions > 0


def test_concurrent_evaluate_does_not_crash():
    """线程安全：并发 evaluate 不抛异常。"""
    ao = ArchitectureOptimizer(eval_interval=1, split_threshold_steps=100,
                                prune_threshold_steps=100)
    modules = [_DummyModule(f"M{i}") for i in range(5)]

    def worker() -> None:
        for i in range(50):
            ao.record_errors([f"M{i % 5}"], [0.1 * (i % 5)])
            ao.evaluate(modules, cycle=i)

    threads = [Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # 不抛异常即通过


def test_concurrent_record_errors_does_not_crash():
    """线程安全：并发 record_errors 不抛异常。"""
    ao = ArchitectureOptimizer()

    def worker(idx: int) -> None:
        for i in range(100):
            ao.record_errors([f"agent{idx}"], [float(i)])

    threads = [Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # 每个 agent 应有 100 条记录
    for i in range(4):
        assert len(ao._error_history[f"agent{i}"]) == 100


def test_dormant_names_returns_list():
    ao = ArchitectureOptimizer()
    assert ao.dormant_names() == []
    # 手动注入休眠模块
    ao._dormant_modules["X"] = _DummyModule("X")
    assert ao.dormant_names() == ["X"]


def test_error_history_deque_maxlen():
    """deque maxlen 限长：error_history 上限 500。"""
    ao = ArchitectureOptimizer()
    for i in range(1000):
        ao.record_errors(["A"], [float(i)])
    assert len(ao._error_history["A"]) == 500


def test_reactivate_pops_from_dormant():
    """reactivate 把休眠模块移回活跃列表。"""
    ao = ArchitectureOptimizer()
    mod = _DummyModule("X")
    ao._dormant_modules["X"] = mod
    modules = []
    assert ao.reactivate(modules, "X") is True
    assert modules == [mod]
    # 休眠池清空
    assert ao.dormant_names() == []
