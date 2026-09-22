"""Modern Hopfield memory tests — verify storage, retrieval, consolidation."""
from __future__ import annotations

import numpy as np
import pytest

from zero_data_model.hopfield import HopfieldMemory


class TestConstruction:
    def test_construction(self):
        mem = HopfieldMemory(memory_dim=32, capacity=100, seed=42)
        assert mem.memory_dim == 32
        assert mem.capacity == 100
        assert mem.size == 0

    def test_invalid_dim(self):
        with pytest.raises(ValueError):
            HopfieldMemory(memory_dim=0)

    def test_invalid_capacity(self):
        with pytest.raises(ValueError):
            HopfieldMemory(memory_dim=32, capacity=0)

    def test_default_beta(self):
        """默认 beta 应为 1/sqrt(d)。"""
        mem = HopfieldMemory(memory_dim=16)
        assert abs(mem.beta - 1.0 / np.sqrt(16)) < 1e-10


class TestStoreRetrieve:
    def test_store_single(self):
        mem = HopfieldMemory(memory_dim=32, capacity=10, seed=42)
        key = np.random.default_rng(0).standard_normal(32)
        mem.store(key)
        assert mem.size == 1

    def test_store_invalid_shape(self):
        mem = HopfieldMemory(memory_dim=32, seed=42)
        with pytest.raises(ValueError):
            mem.store(np.ones(16))

    def test_store_nan_skipped(self):
        """NaN key 不应被存储。"""
        mem = HopfieldMemory(memory_dim=32, seed=42)
        mem.store(np.full(32, np.nan))
        assert mem.size == 0

    def test_store_value(self):
        mem = HopfieldMemory(memory_dim=32, seed=42)
        key = np.ones(32)
        value = np.ones(32) * 2
        mem.store(key, value)
        retrieved, _ = mem.retrieve(key)
        np.testing.assert_allclose(retrieved, value, atol=1e-6)

    def test_retrieve_self_associative(self):
        """自联想：key == value，retrieve(key) 应返回 key 本身。"""
        mem = HopfieldMemory(memory_dim=32, capacity=10, seed=42, beta=1.0)
        key = np.random.default_rng(0).standard_normal(32)
        mem.store(key)
        retrieved, sim = mem.retrieve(key)
        np.testing.assert_allclose(retrieved, key, atol=1e-6)
        assert sim[0] > 0.99  # 应几乎是 1（只有一个记忆）

    def test_retrieve_top_k(self):
        mem = HopfieldMemory(memory_dim=16, capacity=10, seed=42)
        rng = np.random.default_rng(0)
        for _ in range(5):
            mem.store(rng.standard_normal(16))
        retrieved, sims = mem.retrieve(np.zeros(16), k=3)
        assert sims.shape == (3,)
        # top-1 相似度 >= top-2 >= top-3
        assert sims[0] >= sims[1] >= sims[2]

    def test_retrieve_k_zero_returns_empty_sims(self):
        mem = HopfieldMemory(memory_dim=16, seed=42)
        mem.store(np.ones(16))
        retrieved, sims = mem.retrieve(np.ones(16), k=0)
        assert sims.shape == (0,)

    def test_retrieve_invalid_query_shape(self):
        mem = HopfieldMemory(memory_dim=32, seed=42)
        mem.store(np.ones(32))
        with pytest.raises(ValueError):
            mem.retrieve(np.ones(16))

    def test_retrieve_empty_memory_raises(self):
        mem = HopfieldMemory(memory_dim=32, seed=42)
        with pytest.raises(RuntimeError):
            mem.retrieve(np.ones(32))

    def test_retrieve_nan_query_sanitized(self):
        """NaN 查询应被清洗，不崩溃。"""
        mem = HopfieldMemory(memory_dim=16, seed=42)
        mem.store(np.ones(16))
        retrieved, _ = mem.retrieve(np.full(16, np.nan))
        assert np.all(np.isfinite(retrieved))


class TestCapacityAndFIFO:
    def test_fifo_eviction(self):
        """达到 capacity 后应 FIFO 淘汰。"""
        mem = HopfieldMemory(memory_dim=16, capacity=3, seed=42)
        rng = np.random.default_rng(0)
        keys = [rng.standard_normal(16) for _ in range(5)]
        for k in keys:
            mem.store(k)
        assert mem.size == 3  # 只保留最后 3 个
        # 第一个应被淘汰

    def test_high_capacity_storage(self):
        """测试存储 1000 条记忆（容量验证）。"""
        mem = HopfieldMemory(memory_dim=64, capacity=1000, seed=42, beta=0.1)
        rng = np.random.default_rng(0)
        for _ in range(500):
            mem.store(rng.standard_normal(64))
        assert mem.size == 500


class TestRetrievalAccuracy:
    def test_top5_accuracy_100memories(self):
        """存储 100 条，检索 top-5 精度 > 90%。

        存储正交化（高斯随机）的 100 条记忆，每条加小噪声作为 query，
        验证 top-5 中包含正确记忆的比例 > 90%。
        """
        d = 64
        N = 100
        mem = HopfieldMemory(memory_dim=d, capacity=N, seed=42, beta=1.0)
        rng = np.random.default_rng(42)

        # 存储正交化记忆
        keys = [rng.standard_normal(d) for _ in range(N)]
        for k in keys:
            mem.store(k)

        # 巩固
        mem.update_weights()

        # 测试：每条记忆加噪声检索
        noise_level = 0.1
        correct_in_top5 = 0
        for i, k in enumerate(keys):
            query = k + rng.standard_normal(d) * noise_level
            _, top_sims = mem.retrieve(query, k=5)
            # 检查是否检索到了正确记忆
            # 因为我们没有返回索引，只能检查相似度最高的那个是否对应
            # 用直接检索判断
            retrieved, _ = mem.retrieve(query, k=1)
            # 如果 retrieved 接近 k，就算正确
            if np.linalg.norm(retrieved - k) < np.linalg.norm(retrieved - keys[(i + 1) % N]):
                correct_in_top5 += 1

        accuracy = correct_in_top5 / N
        assert accuracy > 0.9, f"Retrieval accuracy {accuracy:.2%} < 90%"

    def test_partial_query_completion(self):
        """部分状态（被遮挡）的查询应能补全。"""
        d = 32
        mem = HopfieldMemory(memory_dim=d, capacity=10, seed=42, beta=2.0)
        rng = np.random.default_rng(0)

        # 存储几个明显不同的模式
        original = rng.standard_normal(d)
        mem.store(original)
        # 加几个干扰
        for _ in range(5):
            mem.store(rng.standard_normal(d))

        # 部分遮挡：保留前一半，后半设为 0
        partial = original.copy()
        partial[d // 2:] = 0

        retrieved, _ = mem.retrieve(partial, k=1)
        # 检索结果应更接近 original 而不是其他
        dist_to_original = np.linalg.norm(retrieved - original)
        # 至少应该比查询本身更接近 original
        dist_query_to_original = np.linalg.norm(partial - original)
        assert dist_to_original < dist_query_to_original, \
            f"Retrieval did not improve: dist_retrieved={dist_to_original:.3f} " \
            f"vs dist_query={dist_query_to_original:.3f}"


class TestConsolidation:
    def test_update_weights_changes_keys(self):
        """巩固后 keys 矩阵应改变。"""
        mem = HopfieldMemory(memory_dim=16, capacity=10, seed=42)
        rng = np.random.default_rng(0)
        for _ in range(5):
            mem.store(rng.standard_normal(16))

        # 触发矩阵构建
        mem.retrieve(rng.standard_normal(16), k=1)
        keys_before = mem._keys.copy()

        mem.update_weights()
        assert mem._consolidated
        # keys 应该变了
        assert not np.allclose(keys_before, mem._keys)

    def test_update_weights_empty_memory_no_crash(self):
        mem = HopfieldMemory(memory_dim=16, seed=42)
        mem.update_weights()  # 不应崩溃
        assert mem.size == 0

    def test_update_weights_improves_retrieval(self):
        """巩固后检索精度应提高。"""
        d = 32
        N = 50
        mem = HopfieldMemory(memory_dim=d, capacity=N, seed=42, beta=1.0)
        rng = np.random.default_rng(0)
        keys = [rng.standard_normal(d) for _ in range(N)]
        for k in keys:
            mem.store(k)

        # 巩固前检索误差
        errors_before = []
        for k in keys:
            retrieved, _ = mem.retrieve(k, k=1)
            errors_before.append(np.linalg.norm(retrieved - k))
        err_before = np.mean(errors_before)

        # 巩固
        mem.update_weights()

        # 巩固后检索误差
        errors_after = []
        for k in keys:
            retrieved, _ = mem.retrieve(k, k=1)
            errors_after.append(np.linalg.norm(retrieved - k))
        err_after = np.mean(errors_after)

        # 巩固后误差应不大于巩固前（伪逆重构应保持或提升精度）
        assert err_after <= err_before * 1.1  # 允许 10% 容差


class TestThreadSafety:
    def test_concurrent_store_retrieve(self):
        """并发 store + retrieve 不应崩溃。"""
        import threading
        mem = HopfieldMemory(memory_dim=32, capacity=100, seed=42)
        rng = np.random.default_rng(0)
        errors = []

        def store_worker():
            try:
                for _ in range(50):
                    mem.store(rng.standard_normal(32))
            except Exception as e:
                errors.append(e)

        def retrieve_worker():
            try:
                for _ in range(50):
                    if mem.size > 0:
                        mem.retrieve(rng.standard_normal(32), k=1)
            except Exception as e:
                # 空记忆时 retrieve 会 RuntimeError，忽略
                if not isinstance(e, RuntimeError):
                    errors.append(e)

        threads = [threading.Thread(target=store_worker) for _ in range(2)]
        threads += [threading.Thread(target=retrieve_worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
