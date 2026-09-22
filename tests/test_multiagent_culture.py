from __future__ import annotations

from threading import Thread

import numpy as np
import pytest

from zero_data_model.multiagent.culture import CulturePropagation, Generation


# --------------------------------------------------------------------------- #
# 构造
# --------------------------------------------------------------------------- #


def test_construct_default():
    cp = CulturePropagation()
    assert len(cp.generations) == 0
    assert cp._current_gen == 0


def test_construct_with_args():
    cp = CulturePropagation(max_generations=5, seed=42)
    assert cp.generations.maxlen == 5


# --------------------------------------------------------------------------- #
# record_generation / inherit_weights / compute_cultural_acceleration / get_knowledge_curve
# --------------------------------------------------------------------------- #


def test_record_generation_appends_generation():
    cp = CulturePropagation(seed=42)
    gen = cp.record_generation(
        knowledge_graph_size=10, mean_free_energy=0.5, n_steps=100
    )
    assert isinstance(gen, Generation)
    assert gen.gen_id == 0
    assert gen.knowledge_graph_size == 10
    assert gen.mean_free_energy == 0.5
    assert gen.n_steps == 100
    assert len(cp.generations) == 1
    assert cp._current_gen == 1


def test_record_generation_with_weights():
    cp = CulturePropagation(seed=42)
    weights = {"w": np.array([1.0, 2.0])}
    gen = cp.record_generation(
        knowledge_graph_size=5, mean_free_energy=0.1, n_steps=10,
        weights=weights,
    )
    assert gen.weights_snapshot is weights


def test_compute_cultural_acceleration_single_gen_returns_one():
    """不足两代时返回 1.0（中性基线）。"""
    cp = CulturePropagation(seed=42)
    cp.record_generation(knowledge_graph_size=10, mean_free_energy=0.5, n_steps=100)
    assert cp.compute_cultural_acceleration() == 1.0


def test_compute_cultural_acceleration_empty_returns_one():
    cp = CulturePropagation(seed=42)
    assert cp.compute_cultural_acceleration() == 1.0


def test_compute_cultural_acceleration_two_gens():
    """二代且后代学习更快 → 加速度 > 1。"""
    cp = CulturePropagation(seed=42)
    # 第一代：100 步学到 10 个知识 → rate = 0.1
    cp.record_generation(knowledge_graph_size=10, mean_free_energy=0.5, n_steps=100)
    # 第二代：50 步学到 10 个知识 → rate = 0.2
    cp.record_generation(knowledge_graph_size=10, mean_free_energy=0.4, n_steps=50)
    acc = cp.compute_cultural_acceleration()
    # ratio = 0.2 / 0.1 = 2.0
    assert acc == pytest.approx(2.0)


def test_compute_cultural_acceleration_zero_n_steps_skipped():
    """n_steps <= 0 的代际被跳过。"""
    cp = CulturePropagation(seed=42)
    cp.record_generation(knowledge_graph_size=10, mean_free_energy=0.5, n_steps=0)
    cp.record_generation(knowledge_graph_size=20, mean_free_energy=0.4, n_steps=10)
    # 第一代 n_steps=0 → 跳过 → ratios 空 → 返回 1.0
    assert cp.compute_cultural_acceleration() == 1.0


def test_get_knowledge_curve_returns_tuples():
    cp = CulturePropagation(seed=42)
    cp.record_generation(knowledge_graph_size=10, mean_free_energy=0.5, n_steps=100)
    cp.record_generation(knowledge_graph_size=20, mean_free_energy=0.4, n_steps=50)
    curve = cp.get_knowledge_curve()
    assert len(curve) == 2
    assert curve[0] == (0, 10)
    assert curve[1] == (1, 20)


# --------------------------------------------------------------------------- #
# 军事级修复点：inherit_weights 深拷贝
# --------------------------------------------------------------------------- #


def test_inherit_weights_returns_independent_dict():
    """核心修复点：inherit_weights 深拷贝——修改子代权重不影响父代。"""
    cp = CulturePropagation(seed=42)
    parent = {"w": np.array([1.0, 2.0, 3.0])}
    child = cp.inherit_weights(parent, mutation_rate=0.0)
    # 子代应是与父代不同的对象
    assert child is not parent
    assert child["w"] is not parent["w"]
    # 子代的数组已被深拷贝
    # （mutation_rate=0 时理论上是父代的 float 拷贝）
    # 修改子代不应影响父代
    child["w"][0] = 999.0
    assert parent["w"][0] == 1.0


def test_inherit_weights_modifies_only_child_dict():
    cp = CulturePropagation(seed=42)
    parent = {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])}
    child = cp.inherit_weights(parent, mutation_rate=0.0)
    # 在 child 上加新键不应影响 parent
    child["new_key"] = "leak"
    assert "new_key" not in parent


def test_inherit_weights_int_dtype_array_no_truncation():
    """核心修复点：int-dtype 数组加噪声不截断（先转 float 再加）。

    原实现 ``arr += noise`` 在 int-dtype 上会静默截断浮点噪声为 0，
    导致变异实际未生效。修复后先 ``astype(float, copy=True)`` 再加。
    """
    cp = CulturePropagation(seed=42)
    parent = {"w": np.array([0, 0, 0], dtype=int)}
    # mutation_rate=1.0 → 噪声标准差 1.0，足以产生明显变化
    child = cp.inherit_weights(parent, mutation_rate=1.0)
    # 子代类型应是 float（不再是 int）
    assert child["w"].dtype == np.float64
    # 数值应有变化（不再被截断为 0）
    assert not np.all(child["w"] == 0)


def test_inherit_weights_float_dtype_preserved():
    cp = CulturePropagation(seed=42)
    parent = {"w": np.array([1.0, 2.0, 3.0])}
    child = cp.inherit_weights(parent, mutation_rate=0.5)
    assert child["w"].dtype == np.float64
    # 子代应与父代不同（噪声被加入）
    assert not np.allclose(child["w"], parent["w"])


def test_inherit_weights_non_array_values_deep_copied():
    """非 ndarray 值用 copy.deepcopy 拷贝（容器类不与父代共享内部引用）。"""
    cp = CulturePropagation(seed=42)
    parent = {"list_val": [1, 2, 3], "dict_val": {"x": 1}}
    child = cp.inherit_weights(parent, mutation_rate=0.0)
    # 子代的 list 应是独立拷贝
    assert child["list_val"] == parent["list_val"]
    assert child["list_val"] is not parent["list_val"]
    # 修改子代不影响父代
    child["list_val"].append(999)
    assert parent["list_val"] == [1, 2, 3]
    child["dict_val"]["y"] = 2
    assert "y" not in parent["dict_val"]


# --------------------------------------------------------------------------- #
# 军事级修复点：deque maxlen 限长
# --------------------------------------------------------------------------- #


def test_generations_deque_maxlen():
    """deque maxlen 限长：generations 上限 max_generations。"""
    cp = CulturePropagation(max_generations=3, seed=42)
    for i in range(10):
        cp.record_generation(
            knowledge_graph_size=i, mean_free_energy=0.1, n_steps=10,
        )
    assert len(cp.generations) == 3
    # FIFO：保留最后 3 个，gen_id 为 7,8,9
    gen_ids = [g.gen_id for g in cp.generations]
    assert gen_ids == [7, 8, 9]


# --------------------------------------------------------------------------- #
# 线程安全
# --------------------------------------------------------------------------- #


def test_concurrent_record_generation_does_not_crash():
    cp = CulturePropagation(max_generations=10000, seed=42)

    def worker(idx: int) -> None:
        for i in range(20):
            cp.record_generation(
                knowledge_graph_size=idx * 20 + i,
                mean_free_energy=0.1, n_steps=10,
            )

    threads = [Thread(target=worker, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # 至少有些代际被记录
    assert len(cp.generations) > 0


def test_concurrent_inherit_weights_does_not_crash():
    cp = CulturePropagation(seed=42)
    parent = {"w": np.array([1.0, 2.0, 3.0])}

    def worker() -> None:
        for _ in range(20):
            cp.inherit_weights(parent, mutation_rate=0.1)

    threads = [Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # 父代未被修改
    assert np.allclose(parent["w"], [1.0, 2.0, 3.0])
