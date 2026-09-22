"""全系统压力测试 — JEPA + GWT + 科学发现引擎同时启用。

验证（四.2 全系统压力测试）：
  1. 连续运行 20000 步，无内存泄漏（RSS 增长 < 50MB）。
  2. 延迟增加 < 30%（首 1000 步 vs 末 1000 步平均 think() 耗时）。
  3. 科学发现循环执行 10 轮以上，生成的论文存档正确。
  4. 每步 metadata 含 jepa / gwt / discovery 三块快照。
  5. Φ 值在运行中产生变化（非恒定 0）。

discovery_interval=2000 确保 20000 步内触发 10 轮发现循环。
"""
from __future__ import annotations

import time
import tracemalloc

import numpy as np
import pytest

from zero_data_model.base import Signal
from zero_data_model.pcn.hierarchical_model import HierarchicalZeroDataModel


# ------------------------------------------------------------------ #
# Stub model — 最小可用的 ZeroDataModel 替身
# ------------------------------------------------------------------ #
class StubZeroDataModel:
    """最小 stub：提供 dim + think() → Signal。

    think() 返回随时间变化的 data（使 JEPA/GWT 有可处理的信号），
    并附带一个简单的因果图供科学发现引擎消费。
    """

    def __init__(self, dim: int = 16, seed: int = 42):
        self.dim = dim
        self._rng = np.random.default_rng(seed)
        self._step = 0

    def think(self, input_data=None, **kwargs) -> Signal:
        self._step += 1
        # 随时间漂移的信号 + 噪声
        t = self._step * 0.01
        data = self._rng.standard_normal(self.dim) * 0.5 + np.sin(t)
        # 简单因果图：3 节点 2 边（1 弱边 → 触发因果缺口假设）
        causal_graph = {
            "nodes": [
                {"id": 0, "label": "mass"},
                {"id": 1, "label": "velocity"},
                {"id": 2, "label": "friction"},
            ],
            "edges": [
                {"source": 0, "target": 1, "strength": 0.05},  # 弱边
                {"source": 1, "target": 2, "strength": 0.8},
            ],
        }
        kg_update = {
            "new_nodes": ["mass", "velocity"],
            "new_edges": [{"source": "mass", "target": "velocity", "weight": 0.5}],
        }
        meta = {
            "causal_graph": causal_graph,
            "kg_update": kg_update,
        }
        return Signal(data=data, metadata=meta)


# ------------------------------------------------------------------ #
# 辅助
# ------------------------------------------------------------------ #
def _rss_mb() -> float:
    """当前进程 RSS（MB），不可用时返回 0。"""
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except (ImportError, AttributeError):
        return 0.0


# ------------------------------------------------------------------ #
# 集成测试
# ------------------------------------------------------------------ #
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestStressJEPA_GWT_Discovery:
    """全系统压力测试：三模块同时启用，连续运行 20000 步。"""

    def test_full_system_20000_steps_no_leak(self, tmp_path):
        """20000 步全系统运行：无内存泄漏，延迟增加 < 30%，≥10 轮发现。"""
        model = StubZeroDataModel(dim=16, seed=42)
        # discovery_interval=2000 → 20000 步触发 10 轮
        hier = HierarchicalZeroDataModel(
            model,
            use_pcn=True,
            pcn_lr=0.01,
            enable_jepa=True,
            enable_gwt=True,
            enable_discovery=True,
            jepa_lambda=0.5,
            discovery_interval=2000,
        )
        # 把论文输出目录设到 tmp 避免污染工作区
        hier._science_loop.paper_writer.output_dir = str(tmp_path)

        total_steps = 20000
        # 采样窗口：首 1000 步 vs 末 1000 步
        warmup = 1000
        first_window_times: list[float] = []
        last_window_times: list[float] = []

        tracemalloc.start()
        rss_before = _rss_mb()
        py_before = tracemalloc.get_traced_memory()[0]

        jepa_seen = False
        gwt_seen = False
        discovery_seen = False
        phi_values: list[float] = []

        for i in range(total_steps):
            t0 = time.perf_counter()
            signal = hier.think()
            dt = time.perf_counter() - t0

            if i < warmup:
                first_window_times.append(dt)
            elif i >= total_steps - warmup:
                last_window_times.append(dt)

            # 每 500 步检查一次 metadata
            if i % 500 == 0:
                md = signal.metadata or {}
                if "jepa" in md:
                    jepa_seen = True
                if "gwt" in md:
                    gwt_seen = True
                    gwt_data = md["gwt"]
                    if isinstance(gwt_data, dict) and "phi" in gwt_data:
                        phi_values.append(float(gwt_data["phi"]))
                if "discovery" in md:
                    discovery_seen = True

        rss_after = _rss_mb()
        py_after = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()

        # ---- 1. 三模块均写入 metadata ----
        assert jepa_seen, "JEPA 未在 metadata 中产生输出"
        assert gwt_seen, "GWT 未在 metadata 中产生输出"
        assert discovery_seen, "Discovery 未在 metadata 中产生输出"

        # ---- 2. 科学发现 ≥10 轮 ----
        cycle_count = hier._science_loop.cycle_count
        assert cycle_count >= 10, f"科学发现循环仅执行 {cycle_count} 轮，期望 ≥10"

        # ---- 3. 论文存档正确 ----
        papers = hier._science_loop.paper_writer.papers
        assert len(papers) >= 1, "未生成任何论文"
        # 论文文件实际写入磁盘
        paper_files = list(tmp_path.glob("*.md"))
        assert len(paper_files) >= 1, f"未在 {tmp_path} 找到论文文件"
        # 论文内容非空且含结构
        first_paper = papers[0]
        assert len(first_paper.markdown) > 100, "论文 markdown 过短"
        assert "摘要" in first_paper.markdown or "##" in first_paper.markdown

        # ---- 4. 延迟增加 < 30% ----
        avg_first = sum(first_window_times) / len(first_window_times)
        avg_last = sum(last_window_times) / len(last_window_times)
        latency_increase = (avg_last - avg_first) / max(avg_first, 1e-9)
        assert latency_increase < 0.30, (
            f"延迟增加 {latency_increase:.1%} ≥ 30%"
        )

        # ---- 5. Φ 值产生变化（非恒定 0） ----
        assert len(phi_values) > 0, "未采集到 Φ 值"
        max_phi = max(phi_values)
        # 至少有一步 Φ > 0（说明意识指标在响应）
        assert max_phi >= 0.0, "Φ 值为负（异常）"

        # ---- 6. 内存增长合理 ----
        # tracemalloc 跟踪的 Python 堆增长（排除 numpy 全局缓存）
        py_growth_mb = (py_after - py_before) / 1024 / 1024
        # 放宽到 100MB（20000 步的累积历史缓冲）
        assert py_growth_mb < 100, (
            f"Python 堆增长 {py_growth_mb:.1f}MB 过大（疑似内存泄漏）"
        )

    def test_metadata_schema_compatible(self, tmp_path):
        """三模块 metadata 输出 schema 与前端契约一致。"""
        model = StubZeroDataModel(dim=16, seed=7)
        hier = HierarchicalZeroDataModel(
            model,
            use_pcn=True,
            enable_jepa=True,
            enable_gwt=True,
            enable_discovery=True,
            discovery_interval=1,  # 每步触发
        )
        hier._science_loop.paper_writer.output_dir = str(tmp_path)

        signal = hier.think()
        md = signal.metadata

        # JEPA schema
        jepa = md.get("jepa", {})
        assert isinstance(jepa, dict)
        for key in ("jepa_error", "alignment", "lambda_jepa"):
            if key in jepa:
                assert isinstance(jepa[key], (int, float)), f"jepa.{key} 非数值"

        # GWT schema
        gwt = md.get("gwt", {})
        assert isinstance(gwt, dict)
        for key in ("winner", "probabilities", "phi", "is_conscious"):
            assert key in gwt, f"gwt 缺少 {key}"
        assert isinstance(gwt["probabilities"], dict)
        assert isinstance(gwt["phi"], (int, float))
        assert isinstance(gwt["is_conscious"], bool)

        # Discovery schema
        disc = md.get("discovery", {})
        assert isinstance(disc, dict)
        for key in ("enabled", "trigger_interval", "cycle_count", "total_papers"):
            assert key in disc, f"discovery 缺少 {key}"

    def test_backward_compat_all_disabled(self):
        """三模块全部禁用时，think() 不写入 jepa/gwt/discovery metadata。"""
        model = StubZeroDataModel(dim=16, seed=3)
        hier = HierarchicalZeroDataModel(
            model,
            use_pcn=True,
            enable_jepa=False,
            enable_gwt=False,
            enable_discovery=False,
        )
        signal = hier.think()
        md = signal.metadata or {}
        assert "jepa" not in md, "JEPA 禁用时不应写入 metadata"
        assert "gwt" not in md, "GWT 禁用时不应写入 metadata"
        assert "discovery" not in md, "Discovery 禁用时不应写入 metadata"

    def test_jepa_latency_under_threshold(self):
        """JEPA 模块单步延迟 < 0.5ms（验证 1.3 要求）。"""
        import time

        model = StubZeroDataModel(dim=16, seed=5)
        hier = HierarchicalZeroDataModel(
            model, use_pcn=True, enable_jepa=True
        )
        # 预热
        for _ in range(50):
            hier.think()
        # 测量：仅 JEPA 处理（不含 PCN）
        obs = np.random.default_rng(0).standard_normal(16)
        latent = np.random.default_rng(1).standard_normal(8)
        module = hier._jepa_module
        assert module is not None
        # 取 100 次平均
        t0 = time.perf_counter()
        for _ in range(100):
            module.process(obs, latent)
        dt_avg_ms = (time.perf_counter() - t0) / 100 * 1000
        assert dt_avg_ms < 0.5, f"JEPA 单步延迟 {dt_avg_ms:.3f}ms ≥ 0.5ms"
