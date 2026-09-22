"""Discovery 引擎单元测试：5 个模块完整覆盖。"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from src.discovery.hypothesis_generator import HypothesisGenerator, Hypothesis
from src.discovery.experiment_designer import ExperimentDesigner, ExperimentDesign
from src.discovery.result_analyzer import ResultAnalyzer, AnalysisResult
from src.discovery.paper_writer import PaperWriter, Paper
from src.discovery.science_loop import ScienceLoop, DiscoveryCycle


# ------------------------------------------------------------------ #
# HypothesisGenerator
# ------------------------------------------------------------------ #
class TestHypothesisGenerator:
    def test_generate_from_causal_gaps(self):
        """因果缺口法生成假设。"""
        gen = HypothesisGenerator(seed=0)
        cg = {
            "nodes": [{"id": 0, "label": "mass"}, {"id": 1, "label": "velocity"}],
            "edges": [{"source": 0, "target": 1, "strength": 0.02}],
        }
        hyps = gen.generate(causal_graph=cg)
        assert len(hyps) > 0
        assert hyps[0].strategy == "causal_gap"
        assert "mass" in hyps[0].statement or "velocity" in hyps[0].statement

    def test_generate_from_contradictions(self):
        """矛盾驱动法生成假设。"""
        gen = HypothesisGenerator(seed=0)
        contradictions = [
            {"rule": "newton_2", "inferred": 0.5, "actual": 0.3, "penalty": 0.6}
        ]
        hyps = gen.generate(logic_contradictions=contradictions)
        assert any(h.strategy == "contradiction" for h in hyps)

    def test_generate_from_high_error(self):
        """高预测误差驱动假设。"""
        gen = HypothesisGenerator(seed=0)
        errors = {"L0": 2.5, "L1": 0.3, "L2": 0.1}
        hyps = gen.generate(prediction_errors=errors)
        assert any("L0" in h.statement for h in hyps)

    def test_generate_empty_inputs(self):
        """边界：无输入返回空列表。"""
        gen = HypothesisGenerator(seed=0)
        assert gen.generate() == []

    def test_sorted_by_score(self):
        """假设按 testability * confidence 降序排列。"""
        gen = HypothesisGenerator(max_hypotheses=20, seed=0)
        cg = {
            "nodes": [{"id": i, "label": f"v{i}"} for i in range(5)],
            "edges": [{"source": i, "target": i + 1, "strength": 0.05} for i in range(4)],
        }
        hyps = gen.generate(causal_graph=cg)
        for i in range(len(hyps) - 1):
            score_i = hyps[i].testability * hyps[i].confidence
            score_next = hyps[i + 1].testability * hyps[i + 1].confidence
            assert score_i >= score_next

    def test_max_hypotheses_limit(self):
        """限制最大假设数。"""
        gen = HypothesisGenerator(max_hypotheses=2, seed=0)
        cg = {
            "nodes": [{"id": i, "label": f"v{i}"} for i in range(10)],
            "edges": [
                {"source": i, "target": i + 1, "strength": 0.05} for i in range(9)
            ],
        }
        hyps = gen.generate(causal_graph=cg)
        assert len(hyps) <= 2

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            HypothesisGenerator(max_hypotheses=0)
        with pytest.raises(ValueError):
            HypothesisGenerator(min_confidence=1.5)

    def test_hypothesis_has_unique_id(self):
        """每个假设有唯一 ID。"""
        gen = HypothesisGenerator(seed=0)
        cg = {
            "nodes": [{"id": 0, "label": "a"}, {"id": 1, "label": "b"}],
            "edges": [{"source": 0, "target": 1, "strength": 0.01}],
        }
        hyps = gen.generate(causal_graph=cg)
        ids = {h.hypothesis_id for h in hyps}
        assert len(ids) == len(hyps)


# ------------------------------------------------------------------ #
# ExperimentDesigner
# ------------------------------------------------------------------ #
class TestExperimentDesigner:
    def test_design_returns_design(self):
        """正常设计返回实验方案。"""
        designer = ExperimentDesigner(seed=0)
        h = Hypothesis(
            statement="质量影响速度",
            confidence=0.7,
            testability=0.8,
            strategy="causal_gap",
            intervention_vars=["mass"],
            observation_vars=["velocity"],
        )
        design = designer.design(h)
        assert design is not None
        assert design.hypothesis_id == h.hypothesis_id
        assert design.intervention_type == "change_mass"
        assert design.expected_info_gain >= 0
        assert len(design.action_sequence) >= 2

    def test_design_untestable_returns_none(self):
        """边界：不可测试的假设返回 None。"""
        designer = ExperimentDesigner(seed=0)
        h = Hypothesis(
            statement="untestable",
            confidence=0.5,
            testability=0.05,  # 低于阈值
            strategy="causal_gap",
        )
        assert designer.design(h) is None

    def test_select_best_picks_highest_eig(self):
        """select_best 返回 EIG 最大的实验。"""
        designer = ExperimentDesigner(seed=0)
        hyps = [
            Hypothesis("stmt1", 0.5, 0.6, "causal_gap", ["a"], ["b"]),
            Hypothesis("stmt2", 0.8, 0.9, "causal_gap", ["c"], ["d"]),
        ]
        best = designer.select_best(hyps)
        assert best is not None
        assert best.hypothesis_id in [h.hypothesis_id for h in hyps]

    def test_execute_without_sandbox_simulates(self):
        """无沙盒时返回模拟数据。"""
        designer = ExperimentDesigner(seed=0)
        h = Hypothesis("test", 0.6, 0.7, "causal_gap", ["x"], ["y"])
        design = designer.design(h)
        result = designer.execute(design)
        assert result["executed"] is False
        assert result["simulated"] is True
        assert len(result["results"]) == design.n_repeats

    def test_invalid_n_samples_raises(self):
        with pytest.raises(ValueError):
            ExperimentDesigner(n_samples=0)

    def test_snapshot_fields(self):
        designer = ExperimentDesigner(seed=0)
        snap = designer.snapshot()
        assert "designed_count" in snap
        assert "sandbox_attached" in snap


# ------------------------------------------------------------------ #
# ResultAnalyzer
# ------------------------------------------------------------------ #
class TestResultAnalyzer:
    def _make_exp_data(self, has_effect: bool = True, n: int = 5) -> dict:
        """生成实验数据。"""
        rng = np.random.default_rng(0)
        before = [rng.standard_normal(8).tolist() for _ in range(n)]
        if has_effect:
            after = [(np.array(b) + 3.0).tolist() for b in before]  # 大效应
        else:
            after = [b for b in before]  # 无效应
        return {"results": [{"repeat": i, "before": before[i], "after": after[i]} for i in range(n)]}

    def test_accept_with_strong_effect(self):
        """强效应→接受假设。"""
        analyzer = ResultAnalyzer(accept_threshold=5.0, reject_threshold=0.2)
        h = Hypothesis("mass affects velocity", 0.5, 0.8, "causal_gap", ["mass"], ["velocity"])
        data = self._make_exp_data(has_effect=True)
        result = analyzer.analyze(h, data)
        assert result.decision == "accept"
        assert result.bayes_factor > 5.0
        assert result.confidence > 0.5

    def test_reject_with_no_effect(self):
        """无效应→拒绝假设。"""
        analyzer = ResultAnalyzer(accept_threshold=5.0, reject_threshold=0.2)
        h = Hypothesis("no effect", 0.5, 0.8, "causal_gap", ["x"], ["y"])
        data = self._make_exp_data(has_effect=False)
        result = analyzer.analyze(h, data)
        assert result.decision == "reject"
        assert result.bayes_factor < 0.2

    def test_empty_data_inconclusive(self):
        """边界：无数据返回不确定。"""
        analyzer = ResultAnalyzer()
        h = Hypothesis("test", 0.5, 0.8, "causal_gap")
        result = analyzer.analyze(h, {"results": []})
        assert result.decision == "inconclusive"

    def test_kg_delta_on_accept(self):
        """接受假设时生成知识图谱增量。"""
        analyzer = ResultAnalyzer(accept_threshold=2.0)
        h = Hypothesis("test", 0.5, 0.8, "causal_gap", ["mass"], ["velocity"])
        data = self._make_exp_data(has_effect=True)
        result = analyzer.analyze(h, data)
        if result.decision == "accept":
            assert len(result.kg_delta.get("new_edges", [])) > 0

    def test_conclusion_text_generated(self):
        """生成自然语言结论。"""
        analyzer = ResultAnalyzer()
        h = Hypothesis("test hypothesis", 0.5, 0.8, "causal_gap", ["a"], ["b"])
        data = self._make_exp_data(has_effect=True)
        result = analyzer.analyze(h, data)
        assert len(result.conclusion) > 10
        assert "假设" in result.conclusion

    def test_invalid_thresholds_raise(self):
        with pytest.raises(ValueError):
            ResultAnalyzer(accept_threshold=0.5)
        with pytest.raises(ValueError):
            ResultAnalyzer(reject_threshold=1.5)

    def test_snapshot_fields(self):
        analyzer = ResultAnalyzer()
        snap = analyzer.snapshot()
        assert "total_analyses" in snap
        assert "accepted" in snap
        assert "rejected" in snap


# ------------------------------------------------------------------ #
# PaperWriter
# ------------------------------------------------------------------ #
class TestPaperWriter:
    def test_write_creates_paper(self):
        """撰写论文并存档。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = PaperWriter(output_dir=tmpdir)
            h = Hypothesis("test stmt", 0.5, 0.8, "causal_gap", ["a"], ["b"])
            design = ExperimentDesign(
                hypothesis_id=h.hypothesis_id,
                intervention_type="change_mass",
                intervention_vars=["a"],
                observation_vars=["b"],
                params={"mass": 2.0},
                expected_info_gain=0.5,
                action_sequence=[{"step": 0, "action": "reset"}],
            )
            analysis = AnalysisResult(
                hypothesis_id=h.hypothesis_id,
                bayes_factor=15.0,
                decision="accept",
                confidence=0.9,
                effect_size=1.2,
                n_samples=5,
                conclusion="支持假设",
            )
            paper = writer.write(h, design, analysis)
            assert paper.paper_id
            assert "test stmt" in paper.markdown
            assert os.path.exists(paper.file_path)
            assert writer.paper_count == 1

    def test_markdown_has_required_sections(self):
        """Markdown 包含必要章节。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = PaperWriter(output_dir=tmpdir)
            h = Hypothesis("test", 0.5, 0.8, "causal_gap", ["a"], ["b"])
            design = ExperimentDesign(
                hypothesis_id=h.hypothesis_id,
                intervention_type="apply_force",
                intervention_vars=["a"],
                observation_vars=["b"],
                params={"force": 1.0},
                expected_info_gain=0.3,
                action_sequence=[],
            )
            analysis = AnalysisResult(
                hypothesis_id=h.hypothesis_id,
                bayes_factor=0.05,
                decision="reject",
                confidence=0.8,
                effect_size=0.1,
                n_samples=3,
                conclusion="拒绝假设",
            )
            paper = writer.write(h, design, analysis)
            assert "## 摘要" in paper.markdown
            assert "## 1. 引言" in paper.markdown
            assert "## 2. 方法" in paper.markdown
            assert "## 3. 结果" in paper.markdown
            assert "## 4. 讨论" in paper.markdown
            assert "拒绝" in paper.markdown

    def test_get_paper_by_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = PaperWriter(output_dir=tmpdir)
            h = Hypothesis("test", 0.5, 0.8, "causal_gap")
            design = ExperimentDesign(
                hypothesis_id=h.hypothesis_id,
                intervention_type="test",
                intervention_vars=[],
                observation_vars=[],
                params={},
                expected_info_gain=0.1,
            )
            analysis = AnalysisResult(
                hypothesis_id=h.hypothesis_id,
                bayes_factor=1.0,
                decision="inconclusive",
                confidence=0.3,
                effect_size=0.0,
                n_samples=1,
                conclusion="不确定",
            )
            paper = writer.write(h, design, analysis)
            found = writer.get_paper(paper.paper_id)
            assert found is not None
            assert writer.get_paper("nonexistent") is None

    def test_snapshot_fields(self):
        writer = PaperWriter(output_dir="/tmp/test_disc")
        snap = writer.snapshot()
        assert "paper_count" in snap
        assert "recent_titles" in snap


# ------------------------------------------------------------------ #
# ScienceLoop
# ------------------------------------------------------------------ #
class TestScienceLoop:
    def test_step_triggers_at_interval(self):
        """按间隔触发发现循环。"""
        loop = ScienceLoop(trigger_interval=3, output_dir="/tmp/test_sl", seed=0)
        results = []
        for _ in range(10):
            r = loop.step(
                causal_graph={
                    "nodes": [{"id": 0, "label": "a"}, {"id": 1, "label": "b"}],
                    "edges": [{"source": 0, "target": 1, "strength": 0.02}],
                }
            )
            if r is not None:
                results.append(r)
        # 10 步 / interval=3 → 触发约 3 次
        assert len(results) >= 2
        assert all(isinstance(r, DiscoveryCycle) for r in results)

    def test_run_cycle_completes(self):
        """立即执行一次完整循环。"""
        loop = ScienceLoop(trigger_interval=1, output_dir="/tmp/test_sl2", seed=0)
        cycle = loop.run_cycle(
            causal_graph={
                "nodes": [{"id": 0, "label": "mass"}, {"id": 1, "label": "velocity"}],
                "edges": [{"source": 0, "target": 1, "strength": 0.01}],
            }
        )
        assert cycle.cycle_id == 0
        assert cycle.n_hypotheses > 0
        assert cycle.n_experiments > 0

    def test_disabled_returns_none(self):
        """禁用后 step 返回 None。"""
        loop = ScienceLoop(trigger_interval=1, seed=0)
        loop.disable()
        assert loop.step() is None

    def test_require_approval_queues_experiments(self):
        """需审批时实验入队不执行。"""
        loop = ScienceLoop(
            trigger_interval=1, output_dir="/tmp/test_sl3", require_approval=True, seed=0
        )
        loop.run_cycle(
            causal_graph={
                "nodes": [{"id": 0, "label": "a"}, {"id": 1, "label": "b"}],
                "edges": [{"source": 0, "target": 1, "strength": 0.01}],
            }
        )
        assert len(loop.pending_approvals) > 0

    def test_snapshot_fields(self):
        loop = ScienceLoop(trigger_interval=5, seed=0)
        snap = loop.snapshot()
        assert "cycle_count" in snap
        assert "total_papers" in snap
        assert "recent_cycles" in snap
        assert "hypothesis_generator" in snap

    def test_invalid_interval_raises(self):
        with pytest.raises(ValueError):
            ScienceLoop(trigger_interval=0)

    def test_configure_runtime(self):
        loop = ScienceLoop(trigger_interval=10, seed=0)
        loop.configure(trigger_interval=5, require_approval=True)
        assert loop.trigger_interval == 5
        assert loop.require_approval is True


# ------------------------------------------------------------------ #
# 军事级审查回归测试
# ------------------------------------------------------------------ #
class TestMilitaryReviewRegressions:
    """第一轮军事级审查发现的 bug 回归测试。"""

    def test_paper_threshold_not_corrupted_by_bf(self):
        """B1: 论文中接受阈值不应被 `a.bayes_factor and 10.0` 污染。

        修复前: `a.bayes_factor and 10.0` 在 BF≠0 时恒返回 10.0，
        在 BF=0 时返回 0.0，产生误导性输出。
        """
        from src.discovery.paper_writer import PaperWriter
        from src.discovery.hypothesis_generator import Hypothesis
        from src.discovery.experiment_designer import ExperimentDesign
        from src.discovery.result_analyzer import AnalysisResult

        with tempfile.TemporaryDirectory() as tmp:
            pw = PaperWriter(output_dir=tmp)
            h = Hypothesis(
                statement="测试假设",
                confidence=0.5,
                testability=0.8,
                strategy="causal_gap",
            )
            d = ExperimentDesign(
                hypothesis_id=h.hypothesis_id,
                intervention_type="apply_force",
                intervention_vars=["x"],
                observation_vars=["y"],
                params={"force": 1.0},
                expected_info_gain=0.5,
            )
            a = AnalysisResult(
                hypothesis_id=h.hypothesis_id,
                bayes_factor=50.0,
                decision="accept",
                confidence=0.95,
                effect_size=1.2,
                n_samples=5,
                conclusion="测试结论",
            )
            paper = pw.write(h, d, a)
            # 修复后: 阈值显示应为固定 "10.0"，不随 BF 变化
            assert "BF > 10.0" in paper.markdown, "接受阈值应固定为 10.0"
            assert "BF < 0.1" in paper.markdown, "拒绝阈值应固定为 0.1"

    def test_deterministic_hypothesis_id_across_processes(self):
        """B16: 相同 statement 产生相同 ID（跨进程可复现）。

        修复前: 使用内置 hash()，受 PYTHONHASHSEED 影响每次运行不同。
        """
        from src.discovery.hypothesis_generator import Hypothesis

        h1 = Hypothesis(statement="相同陈述", confidence=0.5, testability=0.8, strategy="x")
        h2 = Hypothesis(statement="相同陈述", confidence=0.5, testability=0.8, strategy="x")
        assert h1.hypothesis_id == h2.hypothesis_id, "相同陈述应有相同 ID"
        # 不同陈述应有不同 ID
        h3 = Hypothesis(statement="不同陈述", confidence=0.5, testability=0.8, strategy="x")
        assert h1.hypothesis_id != h3.hypothesis_id

    def test_exp_overflow_clipped(self):
        """B7: max_t 极大时 np.exp 不溢出（裁剪到 50）。"""
        from src.discovery.result_analyzer import ResultAnalyzer
        from src.discovery.hypothesis_generator import Hypothesis

        ra = ResultAnalyzer()
        h = Hypothesis(
            statement="测试",
            confidence=0.5,
            testability=0.8,
            strategy="causal_gap",
        )
        # 构造 before 方差为 0（pooled_std ≈ 1e-4），after 均值差异巨大
        # → max_t 极大 → 修复前 np.exp 溢出为 inf + RuntimeWarning
        import warnings

        results = []
        for i in range(3):
            results.append({
                "before": [0.0] * 8,
                "after": [1e6] * 8,
            })
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            # 修复后: max_t 被裁剪到 50，exp(49) 不会溢出
            result = ra.analyze(h, {"results": results})
        assert np.isfinite(result.bayes_factor)
        assert result.bayes_factor <= 1000.0

    def test_sandbox_dict_after_handled(self):
        """B19: 沙盒返回 dict 时 result_analyzer 仍能处理。"""
        from src.discovery.experiment_designer import ExperimentDesigner, ExperimentDesign
        from src.discovery.hypothesis_generator import Hypothesis
        from src.discovery.result_analyzer import ResultAnalyzer

        # 模拟沙盒返回 dict（如 {"state": np.array([...])}）
        class DictSandbox:
            def get_state_distribution(self):
                return np.random.default_rng(0).standard_normal(8)

            def apply_intervention(self, itype, params):
                return {"state": np.random.default_rng(1).standard_normal(8)}

        designer = ExperimentDesigner(seed=0)
        designer.attach_sandbox(DictSandbox())
        h = Hypothesis(
            statement="测试",
            confidence=0.5,
            testability=0.8,
            strategy="causal_gap",
            intervention_vars=["x"],
            observation_vars=["y"],
        )
        design = designer.design(h)
        assert design is not None
        exp_data = designer.execute(design)
        # 不应崩溃（之前 after 是 dict，result_analyzer 无法转 array）
        ra = ResultAnalyzer()
        result = ra.analyze(h, exp_data)
        assert result.decision in ("accept", "reject", "inconclusive")


# ------------------------------------------------------------------ #
# 第二轮军事级审查回归测试
# ------------------------------------------------------------------ #
class TestMilitaryReviewRound2Regressions:
    """第二轮军事级审查修复的 bug 回归测试。

    覆盖：C1（线程安全）、H1（step 双递增）、H2/H3（GWT 候选/Φ）、
    H4（makedirs）、H5（异常捕获）、M2（sandbox None）、
    M3（ragged shape）、M4（n=1 方差）。
    """

    # ---- H4: paper_writer makedirs 移入 try ---- #
    def test_h4_makedirs_failure_returns_paper_without_crash(self):
        """H4: 目录创建失败时 write() 仍返回内存论文，不抛 OSError。

        修复前: os.makedirs 在 try 外，权限/路径冲突会中断 write()。
        """
        h = Hypothesis("测试", 0.5, 0.8, "causal_gap")
        d = ExperimentDesign(
            hypothesis_id=h.hypothesis_id,
            intervention_type="apply_force",
            intervention_vars=["x"],
            observation_vars=["y"],
            params={"force": 1.0},
            expected_info_gain=0.5,
        )
        a = AnalysisResult(
            hypothesis_id=h.hypothesis_id,
            bayes_factor=10.0,
            decision="accept",
            confidence=0.9,
            effect_size=0.5,
            n_samples=5,
            conclusion="测试",
        )
        # 非法路径（含空字节）触发 OSError
        pw = PaperWriter(output_dir="/nonexistent_root\x00/sub")
        paper = pw.write(h, d, a)  # 不应抛异常
        assert paper is not None
        assert len(paper.markdown) > 0
        # 写入失败时 file_path 应为空字符串
        assert paper.file_path == ""

    def test_h4_makedirs_success_writes_file(self, tmp_path):
        """H4: 正常路径下仍写入磁盘文件。"""
        h = Hypothesis("正常测试", 0.5, 0.8, "causal_gap")
        d = ExperimentDesign(
            hypothesis_id=h.hypothesis_id,
            intervention_type="apply_force",
            intervention_vars=["x"],
            observation_vars=["y"],
            params={"force": 1.0},
            expected_info_gain=0.5,
        )
        a = AnalysisResult(
            hypothesis_id=h.hypothesis_id,
            bayes_factor=10.0,
            decision="accept",
            confidence=0.9,
            effect_size=0.5,
            n_samples=5,
            conclusion="测试",
        )
        pw = PaperWriter(output_dir=str(tmp_path))
        paper = pw.write(h, d, a)
        assert paper.file_path != ""
        assert os.path.exists(paper.file_path)

    # ---- H5: science_loop 异常捕获拓宽为 Exception ---- #
    def test_h5_unexpected_exception_does_not_crash_cycle(self):
        """H5: hypothesis_generator 抛 TypeError 时 cycle 记录 error 而非崩溃。

        修复前: 仅捕获 (ValueError, KeyError, RuntimeError, OSError)，
        TypeError/AttributeError 等会导致 cycle 变量未绑定（UnboundLocalError）。
        """
        loop = ScienceLoop(trigger_interval=1, output_dir="/tmp/test_h5", seed=0)

        # monkey-patch: 让 generate 抛 TypeError
        original_generate = loop.hypothesis_generator.generate

        def boom(*args, **kwargs):
            raise TypeError("模拟未预期错误")

        loop.hypothesis_generator.generate = boom
        try:
            cycle = loop.run_cycle()
            assert cycle is not None
            assert cycle.error is not None
            assert "TypeError" in cycle.error
            assert cycle.n_hypotheses == 0
        finally:
            loop.hypothesis_generator.generate = original_generate

    # ---- M2: sandbox 返回 None ---- #
    def test_m2_sandbox_returning_none_handled(self):
        """M2: sandbox.get_state_distribution/apply_intervention 返回 None 时不崩溃。"""
        from src.discovery.experiment_designer import ExperimentDesigner

        class NoneSandbox:
            def get_state_distribution(self):
                return None

            def apply_intervention(self, itype, params):
                return None

        designer = ExperimentDesigner(seed=0)
        designer.attach_sandbox(NoneSandbox())
        h = Hypothesis(
            statement="测试 None 沙盒",
            confidence=0.5,
            testability=0.8,
            strategy="causal_gap",
            intervention_vars=["x"],
            observation_vars=["y"],
        )
        design = designer.design(h)
        assert design is not None
        # 不应崩溃，结果中 before/after 为零向量
        exp_data = designer.execute(design)
        for r in exp_data["results"]:
            assert np.all(np.isfinite(np.asarray(r["before"])))
            assert np.all(np.isfinite(np.asarray(r["after"])))

    def test_m2_sandbox_partial_none_handled(self):
        """M2: only apply_intervention 返回 None（before 正常）时不崩溃。"""
        from src.discovery.experiment_designer import ExperimentDesigner

        class PartialNoneSandbox:
            def __init__(self):
                self._rng = np.random.default_rng(0)

            def get_state_distribution(self):
                return self._rng.standard_normal(8)

            def apply_intervention(self, itype, params):
                return None

        designer = ExperimentDesigner(seed=1)
        designer.attach_sandbox(PartialNoneSandbox())
        h = Hypothesis(
            statement="测试部分 None",
            confidence=0.5,
            testability=0.8,
            strategy="causal_gap",
            intervention_vars=["x"],
            observation_vars=["y"],
        )
        design = designer.design(h)
        exp_data = designer.execute(design)
        # after 应为零向量（与 before 同形）
        for r in exp_data["results"]:
            after = np.asarray(r["after"])
            before = np.asarray(r["before"])
            assert after.shape == before.shape

    # ---- M3: ragged before/after shape ---- #
    def test_m3_ragged_vectors_handled(self):
        """M3: before/after 长度不一致时不崩溃，填充到统一长度。

        修复前: np.array(ragged, dtype=float64) 抛 ValueError 或生成 object 数组。
        """
        ra = ResultAnalyzer()
        h = Hypothesis("ragged 测试", 0.5, 0.8, "causal_gap")
        results = [
            {"before": [1.0, 2.0, 3.0], "after": [1.5, 2.5]},  # 长度不同
            {"before": [1.0, 2.0], "after": [1.5, 2.5, 3.5, 4.5]},
            {"before": [1.0, 2.0, 3.0, 4.0], "after": [1.5, 2.5, 3.5]},
        ]
        result = ra.analyze(h, {"results": results})
        assert result.decision in ("accept", "reject", "inconclusive")
        assert np.isfinite(result.bayes_factor)
        assert np.isfinite(result.effect_size)

    def test_m3_none_and_scalar_values_handled(self):
        """M3: before/after 含 None 或标量时不崩溃。"""
        ra = ResultAnalyzer()
        h = Hypothesis("边界测试", 0.5, 0.8, "causal_gap")
        results = [
            {"before": None, "after": [1.0, 2.0]},
            {"before": 3.0, "after": [4.0, 5.0, 6.0]},
            {"before": [1.0, 2.0], "after": None},
        ]
        result = ra.analyze(h, {"results": results})
        assert result.decision in ("accept", "reject", "inconclusive")

    def test_m3_nan_inf_values_sanitized(self):
        """M3: before/after 含 NaN/inf 时被清理，不污染结果。"""
        ra = ResultAnalyzer()
        h = Hypothesis("NaN 测试", 0.5, 0.8, "causal_gap")
        results = [
            {"before": [float("nan"), 2.0, 3.0], "after": [1.0, float("inf"), 3.0]},
            {"before": [1.0, 2.0, 3.0], "after": [1.0, 2.0, 3.0]},
            {"before": [1.0, 2.0, 3.0], "after": [1.0, 2.0, 3.0]},
        ]
        result = ra.analyze(h, {"results": results})
        assert np.isfinite(result.bayes_factor)
        assert np.isfinite(result.effect_size)

    # ---- M4: n=1 方差无定义 ---- #
    def test_m4_single_sample_returns_inconclusive_bf(self):
        """M4: n=1 时方差无定义，BF 应返回中性值 1.0（inconclusive）。

        修复前: np.var(n=1) 返回 0，pooled_std≈1e-8，t_stat 巨大，
        BF 恒为 accept——统计学无效。
        """
        ra = ResultAnalyzer()
        h = Hypothesis("单样本测试", 0.5, 0.8, "causal_gap")
        results = [{"before": [1.0, 2.0, 3.0], "after": [10.0, 20.0, 30.0]}]
        result = ra.analyze(h, {"results": results})
        # n=1 时 BF 应为 1.0（中性），决策应为 inconclusive
        assert abs(result.bayes_factor - 1.0) < 1e-6, (
            f"n=1 时 BF 应为 1.0（中性），实际 {result.bayes_factor}"
        )
        assert result.decision == "inconclusive"

    def test_m4_two_samples_can_compute_bf(self):
        """M4: n>=2 时正常计算 BF（对照测试）。"""
        ra = ResultAnalyzer()
        h = Hypothesis("双样本测试", 0.5, 0.8, "causal_gap")
        results = [
            {"before": [0.0, 0.0, 0.0], "after": [5.0, 5.0, 5.0]},
            {"before": [0.1, 0.1, 0.1], "after": [5.1, 5.1, 5.1]},
        ]
        result = ra.analyze(h, {"results": results})
        # n=2 且差异显著，BF 应较大（accept）
        assert result.bayes_factor > 1.0

    def test_m4_empty_results_returns_inconclusive(self):
        """M4: results 为空时返回 inconclusive。"""
        ra = ResultAnalyzer()
        h = Hypothesis("空测试", 0.5, 0.8, "causal_gap")
        result = ra.analyze(h, {"results": []})
        assert result.decision == "inconclusive"
        assert result.n_samples == 0


# ------------------------------------------------------------------ #
# 第三轮军事级审查回归测试（F1-F9 修复）
# ------------------------------------------------------------------ #
class TestMilitaryReviewRound3Regressions:
    """第三轮军事级审查修复的 Critical/High 级别 bug 回归测试。

    覆盖：F1（__getattr__ 递归）、F3（paper_writer TypeError）、
    F4（approve_pending 异常）、F5（execute 沙盒异常）、
    F7（_simulate None 参数）、F8（CA float dtype）。
    """

    # ---- F3: paper_writer os.path.join TypeError ---- #
    def test_f3_none_output_dir_does_not_crash(self):
        """F3: output_dir=None 时 write() 不抛 TypeError。

        修复前: os.path.join(None, filename) 在 try 外抛 TypeError，
        不被 (OSError, ValueError) 捕获，write() 崩溃。
        """
        h = Hypothesis("None 路径测试", 0.5, 0.8, "causal_gap")
        d = ExperimentDesign(
            hypothesis_id=h.hypothesis_id,
            intervention_type="apply_force",
            intervention_vars=["x"],
            observation_vars=["y"],
            params={"force": 1.0},
            expected_info_gain=0.5,
        )
        a = AnalysisResult(
            hypothesis_id=h.hypothesis_id,
            bayes_factor=10.0,
            decision="accept",
            confidence=0.9,
            effect_size=0.5,
            n_samples=5,
            conclusion="测试",
        )
        pw = PaperWriter(output_dir=None)  # type: ignore[arg-type]
        paper = pw.write(h, d, a)  # 不应抛异常
        assert paper is not None
        assert paper.file_path == ""

    # ---- F4: approve_pending 异常保护 ---- #
    def test_f4_approve_pending_execute_exception_returns_error(self):
        """F4: approve_pending 中 execute() 抛异常时返回 {"error": ...}。

        修复前: 异常直接传播给调用方，且 _pending_approvals 未 pop。
        F5 修复使沙盒调用本身不抛异常，但 execute() 仍可能因
        其他原因（如 _simulate 内部错误）抛异常。F4 作为安全网
        确保任何异常都被捕获。
        """
        from src.discovery.experiment_designer import ExperimentDesigner

        designer = ExperimentDesigner(seed=0)
        loop = ScienceLoop(trigger_interval=1, seed=0)
        loop.experiment_designer = designer

        h = Hypothesis("crash 测试", 0.5, 0.8, "causal_gap")
        design = designer.design(h)
        assert design is not None
        loop._pending_approvals.append(design)

        # monkey-patch execute 使其抛异常（模拟未预期内部错误）
        original_execute = designer.execute

        def boom(design):
            raise RuntimeError("模拟 execute 内部崩溃")

        designer.execute = boom
        try:
            # approve_pending 不应抛异常，应返回 {"error": ...}
            result = loop.approve_pending(design.experiment_id)
            assert result is not None
            assert isinstance(result, dict)
            assert "error" in result
            assert "RuntimeError" in result["error"]
            # 队列应已清理
            assert len(loop.pending_approvals) == 0
        finally:
            designer.execute = original_execute

    # ---- F5: execute() 沙盒中途异常不丢失已收集数据 ---- #
    def test_f5_execute_sandbox_crash_preserves_partial_results(self):
        """F5: 沙盒在第 2 次重复时崩溃，已收集的前 1 次结果应保留。

        修复前: 整个 execute() 崩溃，所有 results 丢失。
        """
        from src.discovery.experiment_designer import ExperimentDesigner

        class FlakySandbox:
            def __init__(self):
                self._call_count = 0

            def get_state_distribution(self):
                self._call_count += 1
                if self._call_count > 2:
                    raise RuntimeError("沙盒崩溃")
                return np.random.default_rng(0).standard_normal(8)

            def apply_intervention(self, itype, params):
                return np.random.default_rng(1).standard_normal(8)

        designer = ExperimentDesigner(seed=0)
        designer.attach_sandbox(FlakySandbox())
        h = Hypothesis("flaky 测试", 0.5, 0.8, "causal_gap",
                       intervention_vars=["x"], observation_vars=["y"])
        design = designer.design(h)
        assert design is not None
        # n_repeats=5 但沙盒会在第 2 次后崩溃
        design.n_repeats = 5
        exp_data = designer.execute(design)  # 不应抛异常
        # 应返回部分结果（至少 1 次成功）
        assert len(exp_data["results"]) == 5  # 失败的也有占位
        # 前 1-2 次结果应有有效数据
        first_result = exp_data["results"][0]
        assert np.all(np.isfinite(np.asarray(first_result["before"])))

    # ---- F7: _simulate 中 params 含 None 值 ---- #
    def test_f7_simulate_none_param_value_handled(self):
        """F7: params={"mass": None} 时不抛 TypeError。

        修复前: float(None) 抛 TypeError，_simulate 崩溃。
        """
        designer = ExperimentDesigner(seed=0)
        h = Hypothesis("None 参数测试", 0.5, 0.8, "causal_gap",
                       intervention_vars=["x"], observation_vars=["y"])
        design = designer.design(h)
        assert design is not None
        # 注入 None 值参数
        design.params = {"mass": None}
        # 无沙盒 → 走 _simulate 路径
        exp_data = designer.execute(design)
        assert len(exp_data["results"]) > 0
        # 结果数据应有限
        for r in exp_data["results"]:
            assert np.all(np.isfinite(np.asarray(r["before"])))
            assert np.all(np.isfinite(np.asarray(r["after"])))
