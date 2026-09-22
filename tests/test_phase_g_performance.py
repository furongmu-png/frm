"""Phase G (五.3): 性能基准测试。

对应 spec §五.3 的三项硬性性能要求：
  - S4 单步推理延迟 < 1ms（dim=64）
  - Hopfield 记忆检索延迟 < 2ms（1000条记忆）
  - 整体 think() 延迟增加不超过 20%

性能测试与单元测试分开标记（performance marker），避免在常规
CI 流水线上每次都跑（耗时较长且对系统负载敏感）。

用法：
    pytest tests/test_phase_g_performance.py -v             # 跑全部
    pytest tests/test_phase_g_performance.py -m perf -v      # 仅 perf
    pytest tests/test_phase_g_performance.py -m "not perf"   # 跳过 perf
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from zero_data_model.s4 import S4Layer  # noqa: E402
from zero_data_model.hopfield import HopfieldMemory  # noqa: E402
from zero_data_model.model import ZeroDataModel  # noqa: E402


# --------------------------------------------------------------------------- #
# 5.3.1  S4 单步推理延迟 < 1ms（dim=64）
# --------------------------------------------------------------------------- #


@pytest.mark.perf
class TestS4StepLatency:
    """S4 layer.step() 在 dim=64 时的单步延迟应 < 1ms。"""

    def test_s4_step_latency_under_1ms_dim64(self):
        """dim=64, state_dim=64, input_dim=64, output_dim=32."""
        layer = S4Layer(state_dim=64, input_dim=64, output_dim=32, seed=42)
        u = np.random.default_rng(0).standard_normal(64)
        # Warm up (JIT/cache effects)
        for _ in range(50):
            layer.step(u)
        # Measure 200 steps
        n = 200
        t0 = time.perf_counter()
        for _ in range(n):
            layer.step(u)
        elapsed = time.perf_counter() - t0
        per_step_ms = (elapsed / n) * 1000.0
        assert per_step_ms < 1.0, (
            f"S4 step latency {per_step_ms:.3f}ms >= 1ms (dim=64)"
        )

    def test_s4_step_latency_under_1ms_smaller_dim(self):
        """Smaller dim (32) should be even faster."""
        layer = S4Layer(state_dim=32, input_dim=32, output_dim=16, seed=42)
        u = np.random.default_rng(0).standard_normal(32)
        for _ in range(50):
            layer.step(u)
        n = 200
        t0 = time.perf_counter()
        for _ in range(n):
            layer.step(u)
        elapsed = time.perf_counter() - t0
        per_step_ms = (elapsed / n) * 1000.0
        assert per_step_ms < 1.0, (
            f"S4 step latency {per_step_ms:.3f}ms >= 1ms (dim=32)"
        )


# --------------------------------------------------------------------------- #
# 5.3.2  Hopfield 记忆检索延迟 < 2ms（1000条记忆）
# --------------------------------------------------------------------------- #


@pytest.mark.perf
class TestHopfieldRetrieveLatency:
    """Hopfield retrieve() 在 1000 条记忆时的延迟应 < 2ms。"""

    def test_retrieve_latency_under_2ms_1000_memories(self):
        """存储 1000 条 dim=64 记忆，retrieve 单查询延迟 < 2ms。"""
        d = 64
        N = 1000
        mem = HopfieldMemory(memory_dim=d, capacity=N, seed=42, beta=1.0)
        rng = np.random.default_rng(42)
        # Store 1000 random memories
        for _ in range(N):
            k = rng.standard_normal(d)
            k = k / (np.linalg.norm(k) + 1e-12)
            mem.store(k)
        mem.update_weights()
        # Query
        query = rng.standard_normal(d)
        query = query / (np.linalg.norm(query) + 1e-12)
        # Warm up
        for _ in range(20):
            mem.retrieve(query, k=1)
        # Measure 100 retrievals
        n = 100
        t0 = time.perf_counter()
        for _ in range(n):
            mem.retrieve(query, k=1)
        elapsed = time.perf_counter() - t0
        per_retrieve_ms = (elapsed / n) * 1000.0
        assert per_retrieve_ms < 2.0, (
            f"Hopfield retrieve latency {per_retrieve_ms:.3f}ms >= 2ms "
            f"(1000 memories)"
        )

    def test_retrieve_topk_latency_under_2ms_1000_memories(self):
        """retrieve_topk (k=5) 在 1000 条记忆时延迟也应 < 2ms。"""
        d = 64
        N = 1000
        mem = HopfieldMemory(memory_dim=d, capacity=N, seed=42, beta=1.0)
        rng = np.random.default_rng(42)
        for _ in range(N):
            k = rng.standard_normal(d)
            k = k / (np.linalg.norm(k) + 1e-12)
            mem.store(k)
        mem.update_weights()
        query = rng.standard_normal(d)
        query = query / (np.linalg.norm(query) + 1e-12)
        for _ in range(20):
            mem.retrieve_topk(query, k=5)
        n = 100
        t0 = time.perf_counter()
        for _ in range(n):
            mem.retrieve_topk(query, k=5)
        elapsed = time.perf_counter() - t0
        per_call_ms = (elapsed / n) * 1000.0
        assert per_call_ms < 2.0, (
            f"Hopfield retrieve_topk latency {per_call_ms:.3f}ms >= 2ms "
            f"(1000 memories, k=5)"
        )


# --------------------------------------------------------------------------- #
# 5.3.3  整体 think() 延迟增加 ≤ 20%
# --------------------------------------------------------------------------- #


@pytest.mark.perf
class TestThinkLatencyOverhead:
    """启用所有三项升级后，think() 延迟增加不应超过 20%。"""

    @staticmethod
    def _measure_think_latency(model, n_steps: int = 100) -> float:
        """Measure mean think() latency in ms."""
        rng = np.random.default_rng(0)
        obs = rng.standard_normal(model.dim)
        # Warm up
        for _ in range(10):
            model.think(obs)
        # Measure
        t0 = time.perf_counter()
        for _ in range(n_steps):
            model.think(obs)
        elapsed = time.perf_counter() - t0
        return (elapsed / n_steps) * 1000.0

    def test_think_latency_overhead_under_20_percent(self):
        """启用 use_s4 + use_pcn + use_hopfield 后延迟增加 ≤ 20%。"""
        dim = 64
        seed = 42
        # Baseline
        m_base = ZeroDataModel(dim=dim, seed=seed)
        base_latency = self._measure_think_latency(m_base, n_steps=100)
        del m_base
        # Phase G (all upgrades on)
        m_pg = ZeroDataModel(dim=dim, seed=seed, use_s4=True, use_pcn=True, use_hopfield=True)
        # Pre-populate Hopfield memory so the retrieve hook actually runs
        rng = np.random.default_rng(seed)
        for _ in range(5):
            v = rng.standard_normal(dim)
            v = v / (np.linalg.norm(v) + 1e-12)
            m_pg.hopfield_memory.store(v, v)
        pg_latency = self._measure_think_latency(m_pg, n_steps=100)
        overhead_pct = ((pg_latency - base_latency) / base_latency) * 100.0
        # Allow generous margin: 20% is the spec threshold
        assert overhead_pct <= 20.0, (
            f"think() latency overhead {overhead_pct:.1f}% > 20% "
            f"(base={base_latency:.2f}ms, phase_g={pg_latency:.2f}ms)"
        )

    def test_think_latency_overhead_per_upgrade_isolated(self):
        """单独启用每项升级时，延迟增加均应 < 20%。

        用于隔离瓶颈 —— 如果整体超标，此测试能定位是哪个升级导致。
        """
        dim = 64
        seed = 42
        m_base = ZeroDataModel(dim=dim, seed=seed)
        base_latency = self._measure_think_latency(m_base, n_steps=50)
        del m_base

        for flag_name, kwargs in [
            ("use_s4", {"use_s4": True}),
            ("use_pcn", {"use_pcn": True}),
            ("use_hopfield", {"use_hopfield": True}),
        ]:
            m = ZeroDataModel(dim=dim, seed=seed, **kwargs)
            if flag_name == "use_hopfield":
                rng = np.random.default_rng(seed)
                for _ in range(5):
                    v = rng.standard_normal(dim)
                    v = v / (np.linalg.norm(v) + 1e-12)
                    m.hopfield_memory.store(v, v)
            latency = self._measure_think_latency(m, n_steps=50)
            overhead = ((latency - base_latency) / base_latency) * 100.0
            assert overhead <= 20.0, (
                f"{flag_name} alone: latency overhead {overhead:.1f}% > 20% "
                f"(base={base_latency:.2f}ms, {flag_name}={latency:.2f}ms)"
            )
            del m


# --------------------------------------------------------------------------- #
# 5.3.4  非 perf 标记的快速冒烟测试（CI 默认运行）
# --------------------------------------------------------------------------- #


class TestPerformanceSmoke:
    """快速性能冒烟（不带 perf marker），CI 默认运行。

    这些不是严格的 spec §五.3 性能测试，而是确认升级未引入明显
    延迟回归的快速检查。
    """

    def test_s4_step_under_5ms_smoke(self):
        """S4 单步 < 5ms（更宽松的冒烟阈值）。"""
        layer = S4Layer(state_dim=64, input_dim=64, output_dim=32, seed=42)
        u = np.random.default_rng(0).standard_normal(64)
        for _ in range(10):
            layer.step(u)
        n = 50
        t0 = time.perf_counter()
        for _ in range(n):
            layer.step(u)
        elapsed = time.perf_counter() - t0
        per_step_ms = (elapsed / n) * 1000.0
        assert per_step_ms < 5.0, (
            f"S4 step {per_step_ms:.3f}ms >= 5ms (smoke threshold)"
        )

    def test_hopfield_retrieve_under_10ms_smoke(self):
        """Hopfield retrieve < 10ms（更宽松的冒烟阈值）。"""
        d = 32
        N = 100
        mem = HopfieldMemory(memory_dim=d, capacity=N, seed=42, beta=1.0)
        rng = np.random.default_rng(42)
        for _ in range(N):
            k = rng.standard_normal(d)
            k = k / (np.linalg.norm(k) + 1e-12)
            mem.store(k)
        mem.update_weights()
        query = rng.standard_normal(d)
        query = query / (np.linalg.norm(query) + 1e-12)
        for _ in range(5):
            mem.retrieve(query, k=1)
        n = 50
        t0 = time.perf_counter()
        for _ in range(n):
            mem.retrieve(query, k=1)
        elapsed = time.perf_counter() - t0
        per_call_ms = (elapsed / n) * 1000.0
        assert per_call_ms < 10.0, (
            f"Hopfield retrieve {per_call_ms:.3f}ms >= 10ms (smoke)"
        )

    def test_think_overhead_under_50_percent_smoke(self):
        """think() 整体延迟增加 < 50%（更宽松的冒烟阈值）。"""
        dim = 32
        seed = 42
        m_base = ZeroDataModel(dim=dim, seed=seed)
        rng = np.random.default_rng(0)
        obs = rng.standard_normal(dim)
        for _ in range(5):
            m_base.think(obs)
        n = 20
        t0 = time.perf_counter()
        for _ in range(n):
            m_base.think(obs)
        base_latency = (time.perf_counter() - t0) / n * 1000.0
        del m_base

        m_pg = ZeroDataModel(dim=dim, seed=seed,
                              use_s4=True, use_pcn=True, use_hopfield=True)
        for _ in range(5):
            v = rng.standard_normal(dim)
            v = v / (np.linalg.norm(v) + 1e-12)
            m_pg.hopfield_memory.store(v, v)
        for _ in range(5):
            m_pg.think(obs)
        t0 = time.perf_counter()
        for _ in range(n):
            m_pg.think(obs)
        pg_latency = (time.perf_counter() - t0) / n * 1000.0
        overhead = ((pg_latency - base_latency) / base_latency) * 100.0
        assert overhead < 50.0, (
            f"think() overhead {overhead:.1f}% >= 50% (smoke threshold)"
        )
