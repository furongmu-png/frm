# tests/test_phase2_experiment_logger.py
"""第二阶段 §3.3 实验日志器单元测试。

验证：
- log() 记录实验并推送到前端回调
- log_from_dict() 便捷接口
- 里程碑检测（first_experiment / first_supported / first_rejected / strong_evidence）
- 多里程碑同时触发不被覆盖
- 里程碑去重（同一类型只触发一次）
- get_records / get_milestones / get_pending_milestones
- push_callback 异常不影响主流程
- deque maxlen 限长
- stats 统计正确
"""
from __future__ import annotations

from threading import Thread

from zero_data_model.experiment.experiment_logger import (
    ExperimentLogger,
    ExperimentRecord,
)


# --------------------------------------------------------------------------- #
# 初始化
# --------------------------------------------------------------------------- #
class TestInit:
    """测试初始化。"""

    def test_default_init(self) -> None:
        lg = ExperimentLogger()
        assert len(lg.records) == 0
        assert len(lg.milestones) == 0

    def test_with_push_callback(self) -> None:
        calls: list[dict] = []

        def cb(d: dict) -> None:
            calls.append(d)

        lg = ExperimentLogger(push_callback=cb)
        assert lg.stats["has_push_callback"] is True


# --------------------------------------------------------------------------- #
# 基础记录
# --------------------------------------------------------------------------- #
class TestLog:
    """测试 log() 方法。"""

    def test_log_creates_record(self) -> None:
        lg = ExperimentLogger()
        rec = lg.log(step=100, hypothesis="A causes B", intervention="fix A")
        assert isinstance(rec, ExperimentRecord)
        assert rec.step == 100
        assert rec.hypothesis == "A causes B"
        assert len(lg.records) == 1

    def test_log_default_values(self) -> None:
        """默认值：bayes_factor=1.0, conclusion='inconclusive'。"""
        lg = ExperimentLogger()
        rec = lg.log(step=0)
        assert rec.bayes_factor == 1.0
        assert rec.conclusion == "inconclusive"
        assert rec.info_gain == 0.0

    def test_log_returns_record_with_timestamp(self) -> None:
        lg = ExperimentLogger()
        rec = lg.log(step=1)
        assert rec.timestamp > 0.0

    def test_to_dict_roundtrip(self) -> None:
        rec = ExperimentRecord(
            step=42, hypothesis="H", intervention="I",
            observation="O", bayes_factor=5.5,
            conclusion="supported", info_gain=0.3,
        )
        d = rec.to_dict()
        assert d["step"] == 42
        assert d["hypothesis"] == "H"
        assert d["bayes_factor"] == 5.5
        assert d["conclusion"] == "supported"
        assert "timestamp" in d


class TestLogFromDict:
    """测试 log_from_dict() 便捷接口。"""

    def test_log_from_dict_basic(self) -> None:
        lg = ExperimentLogger()
        rec = lg.log_from_dict({
            "step": 10, "hypothesis": "H1",
            "intervention": "do_x",
            "bayes_factor": 3.0,
            "conclusion": "supported",
            "info_gain": 0.5,
        })
        assert rec.step == 10
        assert rec.intervention == "do_x"
        assert rec.bayes_factor == 3.0

    def test_log_from_dict_uses_name_as_intervention_fallback(self) -> None:
        """无 intervention 时回退到 name 字段。"""
        lg = ExperimentLogger()
        rec = lg.log_from_dict({"step": 1, "name": "exp1"})
        assert rec.intervention == "exp1"

    def test_log_from_dict_uses_actual_gain_as_info_gain_fallback(self) -> None:
        """无 info_gain 时回退到 actual_gain 字段。"""
        lg = ExperimentLogger()
        rec = lg.log_from_dict({"step": 1, "actual_gain": 0.42})
        assert rec.info_gain == 0.42

    def test_log_from_dict_defaults(self) -> None:
        """空 dict 应使用默认值。"""
        lg = ExperimentLogger()
        rec = lg.log_from_dict({})
        assert rec.step == 0
        assert rec.bayes_factor == 1.0
        assert rec.conclusion == "inconclusive"


# --------------------------------------------------------------------------- #
# 里程碑
# --------------------------------------------------------------------------- #
class TestMilestones:
    """测试里程碑检测。"""

    def test_first_experiment_milestone(self) -> None:
        lg = ExperimentLogger()
        lg.log(step=10, hypothesis="H")
        ms = lg.get_milestones()
        assert any(m["event"] == "first_experiment" for m in ms)

    def test_first_supported_milestone(self) -> None:
        lg = ExperimentLogger()
        lg.log(step=10, hypothesis="H", conclusion="supported",
               bayes_factor=3.0)
        ms = lg.get_milestones()
        # 首次实验 + 首次支持假设 → 2 个里程碑。
        assert any(m["event"] == "first_experiment" for m in ms)
        assert any(m["event"] == "first_supported_hypothesis" for m in ms)

    def test_first_rejected_milestone(self) -> None:
        lg = ExperimentLogger()
        lg.log(step=5, hypothesis="H", conclusion="rejected")
        ms = lg.get_milestones()
        assert any(m["event"] == "first_rejected_hypothesis" for m in ms)

    def test_strong_evidence_milestone(self) -> None:
        """BF > 10 触发强证据里程碑。"""
        lg = ExperimentLogger()
        lg.log(step=100, hypothesis="H", bayes_factor=15.0,
               conclusion="supported")
        ms = lg.get_milestones()
        assert any(m["event"] == "strong_evidence" for m in ms)

    def test_no_strong_evidence_below_threshold(self) -> None:
        """BF <= 10 不触发强证据。"""
        lg = ExperimentLogger()
        lg.log(step=1, hypothesis="H", bayes_factor=10.0,  # 边界值，不触发
               conclusion="supported")
        ms = lg.get_milestones()
        assert not any(m["event"] == "strong_evidence" for m in ms)

    def test_milestone_deduplication(self) -> None:
        """同一类型里程碑只触发一次。"""
        lg = ExperimentLogger()
        lg.log(step=1, hypothesis="H1", conclusion="supported")
        lg.log(step=2, hypothesis="H2", conclusion="supported")
        lg.log(step=3, hypothesis="H3", conclusion="supported")
        ms = lg.get_milestones()
        supported = [m for m in ms if m["event"] == "first_supported_hypothesis"]
        assert len(supported) == 1

    def test_multiple_milestones_in_single_log(self) -> None:
        """单次 log 同时触发多个里程碑（关键修复点）。

        首次实验 + 首次支持假设 + 强证据（BF>10）应同时记录，
        而不是被覆盖。
        """
        lg = ExperimentLogger()
        lg.log(step=1, hypothesis="H", conclusion="supported",
               bayes_factor=20.0)
        ms = lg.get_milestones()
        events = {m["event"] for m in ms}
        # 三个里程碑都应存在。
        assert "first_experiment" in events
        assert "first_supported_hypothesis" in events
        assert "strong_evidence" in events

    def test_milestone_includes_step_and_description(self) -> None:
        lg = ExperimentLogger()
        lg.log(step=42, hypothesis="momentum conservation",
               conclusion="supported", bayes_factor=5.0)
        ms = lg.get_milestones()
        first_exp = next(m for m in ms if m["event"] == "first_experiment")
        assert first_exp["step"] == 42
        assert "momentum conservation" in first_exp["description"]
        assert first_exp["type"] == "experiment_milestone"


# --------------------------------------------------------------------------- #
# 推送回调
# --------------------------------------------------------------------------- #
class TestPushCallback:
    """测试 WebSocket 推送回调。"""

    def test_callback_invoked_on_log(self) -> None:
        calls: list[dict] = []

        def cb(d: dict) -> None:
            calls.append(d)

        lg = ExperimentLogger(push_callback=cb)
        lg.log(step=1, hypothesis="H")
        assert len(calls) == 1
        assert calls[0]["step"] == 1
        assert calls[0]["hypothesis"] == "H"

    def test_callback_exception_does_not_break(self) -> None:
        """回调抛异常不应影响主流程。"""

        def bad_cb(d: dict) -> None:
            raise RuntimeError("simulated WS failure")

        lg = ExperimentLogger(push_callback=bad_cb)
        # 不应抛异常。
        rec = lg.log(step=1, hypothesis="H")
        assert rec.step == 1
        assert len(lg.records) == 1

    def test_no_callback_does_not_error(self) -> None:
        """未设置回调时不报错。"""
        lg = ExperimentLogger()
        rec = lg.log(step=1, hypothesis="H")
        assert rec.step == 1


# --------------------------------------------------------------------------- #
# 查询接口
# --------------------------------------------------------------------------- #
class TestQueries:
    """测试查询接口。"""

    def test_get_records_limit(self) -> None:
        lg = ExperimentLogger()
        for i in range(10):
            lg.log(step=i, hypothesis=f"H{i}")
        recs = lg.get_records(limit=3)
        assert len(recs) == 3
        # 返回最近的 3 条（step 7, 8, 9）。
        assert recs[0]["step"] == 7
        assert recs[-1]["step"] == 9

    def test_get_records_returns_dicts(self) -> None:
        lg = ExperimentLogger()
        lg.log(step=1, hypothesis="H")
        recs = lg.get_records()
        assert isinstance(recs, list)
        assert isinstance(recs[0], dict)

    def test_get_milestones_returns_list(self) -> None:
        lg = ExperimentLogger()
        lg.log(step=1, hypothesis="H")
        ms = lg.get_milestones()
        assert isinstance(ms, list)

    def test_get_pending_milestones_clears_queue(self) -> None:
        """get_pending_milestones 清空后再次调用应返回空。"""
        lg = ExperimentLogger()
        lg.log(step=1, hypothesis="H")
        first = lg.get_pending_milestones()
        assert len(first) >= 1
        second = lg.get_pending_milestones()
        assert second == []

    def test_get_milestones_does_not_clear(self) -> None:
        """get_milestones 不清空队列。"""
        lg = ExperimentLogger()
        lg.log(step=1, hypothesis="H")
        first = lg.get_milestones()
        second = lg.get_milestones()
        assert len(first) == len(second) >= 1


# --------------------------------------------------------------------------- #
# 限长
# --------------------------------------------------------------------------- #
class TestMaxLen:
    """测试 deque 限长。"""

    def test_records_maxlen(self) -> None:
        lg = ExperimentLogger(max_records=5)
        for i in range(10):
            lg.log(step=i, hypothesis=f"H{i}")
        assert len(lg.records) == 5
        # 保留最后 5 条（step 5-9）。
        assert lg.records[0].step == 5

    def test_milestones_maxlen(self) -> None:
        """里程碑 deque 上限 200。"""
        lg = ExperimentLogger()
        # 触发 250 次 first_experiment 不会发生（去重），改用 supported。
        # 但 supported 也去重。这里只验证 milestones 容器存在且有上限。
        assert lg.milestones.maxlen == 200


# --------------------------------------------------------------------------- #
# 统计
# --------------------------------------------------------------------------- #
class TestStats:
    """测试 stats 属性。"""

    def test_stats_empty(self) -> None:
        lg = ExperimentLogger()
        s = lg.stats
        assert s["n_records"] == 0
        assert s["n_supported"] == 0
        assert s["n_rejected"] == 0
        assert s["n_milestones"] == 0
        assert s["mean_bayes_factor"] == 0.0
        assert s["has_push_callback"] is False

    def test_stats_after_logging(self) -> None:
        lg = ExperimentLogger()
        lg.log(step=1, hypothesis="H1", bayes_factor=3.0,
               conclusion="supported")
        lg.log(step=2, hypothesis="H2", bayes_factor=0.2,
               conclusion="rejected")
        lg.log(step=3, hypothesis="H3", bayes_factor=1.0,
               conclusion="inconclusive")
        s = lg.stats
        assert s["n_records"] == 3
        assert s["n_supported"] == 1
        assert s["n_rejected"] == 1
        assert s["n_milestones"] >= 2
        # mean BF = (3.0 + 0.2 + 1.0) / 3 ≈ 1.4
        assert abs(s["mean_bayes_factor"] - (3.0 + 0.2 + 1.0) / 3.0) < 1e-6


# --------------------------------------------------------------------------- #
# 线程安全
# --------------------------------------------------------------------------- #
class TestThreadSafety:
    """测试并发访问。"""

    def test_concurrent_log_does_not_crash(self) -> None:
        lg = ExperimentLogger(max_records=10000)

        def worker(tid: int) -> None:
            for i in range(20):
                lg.log(step=i, hypothesis=f"H{tid}_{i}")

        threads = [Thread(target=worker, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert lg.stats["n_records"] == 80

    def test_concurrent_log_with_callback(self) -> None:
        calls: list[dict] = []

        def cb(d: dict) -> None:
            calls.append(d)

        lg = ExperimentLogger(max_records=10000, push_callback=cb)

        def worker(tid: int) -> None:
            for i in range(10):
                lg.log(step=i, hypothesis=f"H{tid}_{i}")

        threads = [Thread(target=worker, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(calls) == 40
