"""Phase 3 §3 全栈可观测性测试。

覆盖：
  - ThoughtChain: 记录/异常检测/查询/回放/统计
  - CounterfactualExplainer: 世界模型模拟/解释生成/历史统计
  - HumanFeedback: 反馈信号/权重衰减/演示模式/纠正模式
  - AuditLogger: 日志记录/查询/导出/统计
  - ValueVector: 维度管理/价值观惩罚计算
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from zero_data_model.observability.audit import (
    AuditEntry,
    AuditEventType,
    AuditLogger,
    ValueVector,
)
from zero_data_model.observability.counterfactual_explainer import (
    CounterfactualExplainer,
    CounterfactualResult,
)
from zero_data_model.observability.human_teaching import (
    FeedbackRecord,
    FeedbackType,
    HumanFeedback,
    TeachingMode,
)
from zero_data_model.observability.thought_chain import (
    ThoughtChain,
    ThoughtRecord,
)


# ------------------------------------------------------------------ #
# ThoughtChain
# ------------------------------------------------------------------ #
class TestThoughtChain:
    """思维链记录与异常检测。"""

    def test_record_basic(self):
        """基本记录：返回 ThoughtRecord 并存储。"""
        tc = ThoughtChain()
        rec = tc.record(
            step=0,
            belief_before=[0.1, 0.2],
            belief_after=[0.3, 0.4],
            prediction_errors={"layer1": 0.5},
            confidence=0.8,
            free_energy=1.2,
        )
        assert isinstance(rec, ThoughtRecord)
        assert rec.step == 0
        assert tc.length == 1

    def test_record_anomaly_high_pe(self):
        """高预测误差触发异常标记。"""
        tc = ThoughtChain(anomaly_pe_threshold=2.0)
        rec = tc.record(
            step=0,
            prediction_errors={"layer1": 3.0},
            confidence=0.9,
        )
        assert rec.is_anomaly
        assert "高预测误差" in rec.anomaly_reason

    def test_record_anomaly_low_confidence(self):
        """低置信度触发异常标记。"""
        tc = ThoughtChain(anomaly_confidence_threshold=0.3)
        rec = tc.record(
            step=0,
            prediction_errors={"layer1": 0.1},
            confidence=0.2,
        )
        assert rec.is_anomaly
        assert "低置信度" in rec.anomaly_reason

    def test_record_anomaly_meta_triggered(self):
        """元认知触发标记异常。"""
        tc = ThoughtChain()
        rec = tc.record(
            step=0,
            prediction_errors={"layer1": 0.1},
            confidence=0.9,
            meta_triggered=True,
        )
        assert rec.is_anomaly
        assert "元认知" in rec.anomaly_reason

    def test_record_anomaly_tool_called(self):
        """工具调用标记异常。"""
        tc = ThoughtChain()
        rec = tc.record(
            step=0,
            prediction_errors={"layer1": 0.1},
            confidence=0.9,
            tool_called="SearchTool",
        )
        assert rec.is_anomaly
        assert "工具调用" in rec.anomaly_reason

    def test_get_recent(self):
        """获取最近 N 步。"""
        tc = ThoughtChain()
        for i in range(10):
            tc.record(step=i, confidence=0.9)
        recent = tc.get_recent(n=3)
        assert len(recent) == 3
        assert recent[-1]["step"] == 9
        assert recent[0]["step"] == 7

    def test_get_anomalies(self):
        """获取异常步骤。"""
        tc = ThoughtChain(anomaly_pe_threshold=2.0)
        tc.record(step=0, prediction_errors={"l1": 0.1}, confidence=0.9)
        tc.record(step=1, prediction_errors={"l1": 3.0}, confidence=0.9)
        tc.record(step=2, prediction_errors={"l1": 0.1}, confidence=0.9)
        tc.record(step=3, prediction_errors={"l1": 5.0}, confidence=0.9)
        anomalies = tc.get_anomalies(n=5)
        assert len(anomalies) == 2
        assert anomalies[0]["step"] == 1
        assert anomalies[1]["step"] == 3

    def test_get_at_step(self):
        """按步号查询。"""
        tc = ThoughtChain()
        tc.record(step=5, confidence=0.9)
        rec = tc.get_at_step(5)
        assert rec is not None
        assert rec["step"] == 5
        assert tc.get_at_step(999) is None

    def test_get_range(self):
        """按范围查询。"""
        tc = ThoughtChain()
        for i in range(10):
            tc.record(step=i, confidence=0.9)
        records = tc.get_range(2, 5)
        assert len(records) == 4
        assert records[0]["step"] == 2
        assert records[-1]["step"] == 5

    def test_replay(self):
        """回放全部记录。"""
        tc = ThoughtChain()
        for i in range(5):
            tc.record(step=i, confidence=0.9)
        replay = tc.replay()
        assert len(replay) == 5

    def test_clear(self):
        """清空记录。"""
        tc = ThoughtChain()
        tc.record(step=0, confidence=0.9)
        tc.clear()
        assert tc.length == 0
        assert tc.n_anomalies == 0

    def test_max_records_bounded(self):
        """有界 deque 防止无界增长。"""
        tc = ThoughtChain(max_records=5)
        for i in range(20):
            tc.record(step=i, confidence=0.9)
        assert tc.length == 5

    def test_stats(self):
        """统计信息。"""
        tc = ThoughtChain(anomaly_pe_threshold=2.0)
        tc.record(step=0, prediction_errors={"l1": 0.5}, confidence=0.9)
        tc.record(
            step=1,
            prediction_errors={"l1": 3.0},
            confidence=0.2,
            tool_called="SearchTool",
        )
        stats = tc.stats
        assert stats["n_records"] == 2
        assert stats["n_anomalies"] == 1
        assert stats["mean_confidence"] < 1.0
        assert stats["n_tool_calls"] == 1

    def test_to_dict_serializable(self):
        """记录可序列化为 JSON。"""
        tc = ThoughtChain()
        tc.record(
            step=0,
            belief_before=[0.1],
            belief_after=[0.2],
            prediction_errors={"l1": 0.5},
            confidence=0.9,
            metadata={"key": "value"},
        )
        recent = tc.get_recent(1)
        # 必须可 JSON 序列化
        json_str = json.dumps(recent[0], ensure_ascii=False)
        assert json.loads(json_str)["step"] == 0


# ------------------------------------------------------------------ #
# CounterfactualExplainer
# ------------------------------------------------------------------ #
class TestCounterfactualExplainer:
    """反事实解释引擎。"""

    @staticmethod
    def _make_world_model(fe_map: dict, pe_map: dict, outcome_map: dict):
        """构造简单的世界模型 stub。"""

        def model(context: dict, action):
            return {
                "free_energy": fe_map.get(action, 1.0),
                "prediction_error": pe_map.get(action, 0.1),
                "outcome": outcome_map.get(action, "未知"),
            }

        return model

    def test_explain_with_world_model(self):
        """带世界模型：生成完整反事实解释。"""
        model = self._make_world_model(
            fe_map={"A": 1.2, "B": 3.4},
            pe_map={"A": 0.5, "B": 0.725},
            outcome_map={"A": "安全通过", "B": "碰撞"},
        )
        explainer = CounterfactualExplainer(world_model=model)
        result = explainer.explain(
            factual_state={
                "action": "A",
                "free_energy": 1.2,
                "prediction_error": 0.5,
                "outcome": "安全通过",
                "context": {"pos": [0, 0]},
            },
            alternative_action="B",
        )
        assert isinstance(result, CounterfactualResult)
        assert result.factual_action == "A"
        assert result.alternative_action == "B"
        assert result.factual_free_energy == 1.2
        assert result.counterfactual_free_energy == 3.4
        assert result.free_energy_delta == pytest.approx(2.2)
        # 预测误差增加 (0.725 - 0.5) / 0.5 * 100 = 45%
        assert result.prediction_error_change_pct == pytest.approx(45.0)
        assert "碰撞" in result.explanation
        assert "A" in result.explanation
        assert "B" in result.explanation

    def test_explain_without_world_model(self):
        """无世界模型：仍生成解释（标注无法模拟）。"""
        explainer = CounterfactualExplainer(world_model=None)
        result = explainer.explain(
            factual_state={
                "action": "A",
                "free_energy": 1.0,
                "prediction_error": 0.3,
                "context": {},
            },
            alternative_action="B",
        )
        assert result.counterfactual_free_energy != result.counterfactual_free_energy  # NaN
        assert "无法模拟" in result.counterfactual_outcome
        assert result.explanation  # 仍生成解释字符串

    def test_set_world_model(self):
        """动态设置世界模型。"""
        explainer = CounterfactualExplainer(world_model=None)
        model = self._make_world_model(
            fe_map={"A": 1.0, "B": 2.0},
            pe_map={"A": 0.1, "B": 0.2},
            outcome_map={"A": "ok", "B": "bad"},
        )
        explainer.set_world_model(model)
        result = explainer.explain(
            factual_state={
                "action": "A",
                "free_energy": 1.0,
                "prediction_error": 0.1,
                "context": {},
            },
            alternative_action="B",
        )
        assert result.counterfactual_free_energy == 2.0

    def test_history_and_stats(self):
        """历史记录与统计。"""
        model = self._make_world_model(
            fe_map={"A": 1.0, "B": 2.0},
            pe_map={"A": 0.5, "B": 0.6},
            outcome_map={"A": "ok", "B": "bad"},
        )
        explainer = CounterfactualExplainer(world_model=model)
        for _ in range(3):
            explainer.explain(
                factual_state={
                    "action": "A",
                    "free_energy": 1.0,
                    "prediction_error": 0.5,
                    "context": {},
                },
                alternative_action="B",
            )
        assert explainer.n_explanations == 3
        history = explainer.get_history(n=2)
        assert len(history) == 2
        stats = explainer.stats
        assert stats["n_explanations"] == 3
        assert stats["mean_fe_delta"] == pytest.approx(1.0)

    def test_clear_history(self):
        """清空历史。"""
        model = self._make_world_model(
            fe_map={"A": 1.0, "B": 2.0},
            pe_map={"A": 0.1, "B": 0.2},
            outcome_map={"A": "ok", "B": "bad"},
        )
        explainer = CounterfactualExplainer(world_model=model)
        explainer.explain(
            factual_state={"action": "A", "free_energy": 1.0, "context": {}},
            alternative_action="B",
        )
        explainer.clear_history()
        assert explainer.n_explanations == 0


# ------------------------------------------------------------------ #
# HumanFeedback
# ------------------------------------------------------------------ #
class TestHumanFeedback:
    """人类教学接口。"""

    def test_positive_feedback_signal(self):
        """👍 反馈生成负误差信号。"""
        hf = HumanFeedback(positive_strength=0.1)
        rec = hf.give_feedback(
            step=0,
            feedback_type=FeedbackType.POSITIVE,
        )
        assert isinstance(rec, FeedbackRecord)
        assert rec.error_signal < 0  # 降低误差权重

    def test_negative_feedback_signal(self):
        """👎 反馈生成正误差信号。"""
        hf = HumanFeedback(negative_strength=0.5)
        rec = hf.give_feedback(
            step=0,
            feedback_type=FeedbackType.NEGATIVE,
        )
        assert rec.error_signal > 0  # 增加误差权重

    def test_direction_hint_signal(self):
        """方向提示：基于 prediction 与 actual 差异生成信号。"""
        hf = HumanFeedback()
        rec = hf.give_feedback(
            step=0,
            feedback_type=FeedbackType.DIRECTION_HINT,
            prediction=[1.0, 0.0],
            actual=[0.0, 1.0],
        )
        # L2 范数差异 = sqrt(2) ≈ 1.414
        assert rec.error_signal == pytest.approx(np.sqrt(2.0), rel=1e-4)

    def test_weight_decay(self):
        """反馈权重随反馈次数衰减。"""
        hf = HumanFeedback(decay_rate=0.9, positive_strength=0.1)
        rec1 = hf.give_feedback(step=0, feedback_type=FeedbackType.POSITIVE)
        rec2 = hf.give_feedback(step=1, feedback_type=FeedbackType.POSITIVE)
        # 第二次权重 = 1.0 * 0.9
        assert rec2.weight == pytest.approx(0.9)
        # 信号也按比例衰减
        assert abs(rec2.error_signal) < abs(rec1.error_signal)

    def test_reset_weight(self):
        """重置权重为 1.0。"""
        hf = HumanFeedback(decay_rate=0.5)
        hf.give_feedback(step=0, feedback_type=FeedbackType.POSITIVE)
        hf.give_feedback(step=1, feedback_type=FeedbackType.POSITIVE)
        assert hf.current_weight < 1.0
        hf.reset_weight()
        assert hf.current_weight == 1.0

    def test_max_feedback_exhaustion(self):
        """达到上限后停止接受反馈。"""
        hf = HumanFeedback(max_feedback=2)
        hf.give_feedback(step=0, feedback_type=FeedbackType.POSITIVE)
        hf.give_feedback(step=1, feedback_type=FeedbackType.POSITIVE)
        assert not hf.is_active
        rec = hf.give_feedback(step=2, feedback_type=FeedbackType.POSITIVE)
        assert rec.error_signal == 0.0
        assert rec.weight == 0.0

    def test_teaching_mode(self):
        """教学模式设置与查询。"""
        hf = HumanFeedback()
        assert hf.mode == TeachingMode.NONE
        hf.set_mode(TeachingMode.DEMONSTRATION)
        assert hf.mode == TeachingMode.DEMONSTRATION
        hf.set_mode(TeachingMode.CORRECTION)
        assert hf.mode == TeachingMode.CORRECTION

    def test_demonstration_mode(self):
        """演示模式：记录 (observation, action) 对。"""
        hf = HumanFeedback()
        hf.set_mode(TeachingMode.DEMONSTRATION)
        demo = hf.demonstrate(
            step=0,
            observation=[0.1, 0.2],
            action=[1.0, 0.0],
        )
        assert demo["step"] == 0
        assert demo["observation"] == [0.1, 0.2]
        assert demo["action"] == [1.0, 0.0]
        assert hf.n_demonstrations == 1
        demos = hf.get_demonstrations()
        assert len(demos) == 1

    def test_correction_mode(self):
        """纠正模式：生成误差信号。"""
        hf = HumanFeedback()
        hf.set_mode(TeachingMode.CORRECTION)
        correction = hf.correct(
            step=0,
            prediction=[1.0, 0.0],
            correct_label=[0.5, 0.5],
        )
        assert correction["step"] == 0
        assert correction["prediction"] == [1.0, 0.0]
        assert correction["correct_label"] == [0.5, 0.5]
        assert "error_signal" in correction
        assert hf.n_corrections == 1

    def test_invalid_decay_rate(self):
        """非法 decay_rate 抛异常。"""
        with pytest.raises(ValueError, match="decay_rate"):
            HumanFeedback(decay_rate=1.5)

    def test_stats(self):
        """统计信息。"""
        hf = HumanFeedback()
        hf.give_feedback(step=0, feedback_type=FeedbackType.POSITIVE)
        hf.give_feedback(step=1, feedback_type=FeedbackType.NEGATIVE)
        hf.give_feedback(
            step=2,
            feedback_type=FeedbackType.DIRECTION_HINT,
            prediction=[0.0],
            actual=[1.0],
        )
        hf.demonstrate(step=3, observation=[0.0], action=[1.0])
        stats = hf.stats
        assert stats["total_feedback"] >= 3
        assert stats["n_positive"] == 1
        assert stats["n_negative"] == 1
        assert stats["n_direction_hint"] == 1
        assert stats["n_demonstrations"] == 1


# ------------------------------------------------------------------ #
# ValueVector
# ------------------------------------------------------------------ #
class TestValueVector:
    """价值观向量。"""

    def test_add_and_get_dimension(self):
        """添加和获取维度。"""
        vv = ValueVector()
        dim = vv.add_dimension(
            name="avoid_harm",
            description="避免伤害",
            weight=2.0,
        )
        assert dim.name == "avoid_harm"
        assert dim.weight == 2.0
        got = vv.get_dimension("avoid_harm")
        assert got is not None
        assert got.description == "避免伤害"

    def test_remove_dimension(self):
        """移除维度。"""
        vv = ValueVector()
        vv.add_dimension("test", "测试", 1.0)
        assert vv.remove_dimension("test") is True
        assert vv.remove_dimension("nonexistent") is False

    def test_set_weight(self):
        """调整权重。"""
        vv = ValueVector()
        vv.add_dimension("test", "测试", 1.0)
        assert vv.set_weight("test", 5.0) is True
        assert vv.get_dimension("test").weight == 5.0
        assert vv.set_weight("nonexistent", 1.0) is False

    def test_compute_value_penalty(self):
        """计算价值观惩罚。"""
        vv = ValueVector()
        vv.add_dimension("avoid_harm", "避免伤害", weight=2.0)
        vv.add_dimension("pursue_truth", "追求真相", weight=1.0)
        penalty = vv.compute_value_penalty(
            {"avoid_harm": 0.5, "pursue_truth": 0.3}
        )
        # 0.5 * 2.0 + 0.3 * 1.0 = 1.3
        assert penalty == pytest.approx(1.3)

    def test_compute_value_penalty_unknown_dim(self):
        """未知维度不计入惩罚。"""
        vv = ValueVector()
        vv.add_dimension("avoid_harm", "避免伤害", 1.0)
        penalty = vv.compute_value_penalty(
            {"avoid_harm": 1.0, "unknown_dim": 5.0}
        )
        assert penalty == pytest.approx(1.0)

    def test_to_dict_and_from_dict(self):
        """序列化与反序列化。"""
        vv = ValueVector()
        vv.add_dimension("avoid_harm", "避免伤害", 2.0)
        vv.add_dimension("pursue_truth", "追求真相", 1.0)
        data = vv.to_dict()
        vv2 = ValueVector.from_dict(data)
        assert len(vv2.dimension_names) == 2
        assert vv2.get_dimension("avoid_harm").weight == 2.0

    def test_stats(self):
        """统计信息。"""
        vv = ValueVector()
        vv.add_dimension("a", "维度A", 1.0)
        stats = vv.stats
        assert stats["n_dimensions"] == 1
        assert len(stats["dimensions"]) == 1


# ------------------------------------------------------------------ #
# AuditLogger
# ------------------------------------------------------------------ #
class TestAuditLogger:
    """审计日志记录器。"""

    def test_log_basic(self):
        """基本日志记录。"""
        logger = AuditLogger()
        entry = logger.log(
            event_type=AuditEventType.ACTION_SELECTION,
            module="hierarchical_model",
            context={"step": 0},
            result={"action": "move_left"},
        )
        assert isinstance(entry, AuditEntry)
        assert entry.event_type == AuditEventType.ACTION_SELECTION
        assert entry.module == "hierarchical_model"
        assert logger.n_entries == 1

    def test_log_auto_increment_id(self):
        """entry_id 自动递增。"""
        logger = AuditLogger()
        e1 = logger.log(AuditEventType.OTHER, "m1")
        e2 = logger.log(AuditEventType.OTHER, "m1")
        assert e2.entry_id == e1.entry_id + 1

    def test_query_by_event_type(self):
        """按事件类型查询。"""
        logger = AuditLogger()
        logger.log(AuditEventType.ACTION_SELECTION, "m1")
        logger.log(AuditEventType.TOOL_CALL, "m1")
        logger.log(AuditEventType.ACTION_SELECTION, "m1")
        results = logger.query(event_type=AuditEventType.ACTION_SELECTION)
        assert len(results) == 2

    def test_query_by_module(self):
        """按模块查询。"""
        logger = AuditLogger()
        logger.log(AuditEventType.OTHER, "module_a")
        logger.log(AuditEventType.OTHER, "module_b")
        results = logger.query(module="module_a")
        assert len(results) == 1
        assert results[0]["module"] == "module_a"

    def test_query_by_time_range(self):
        """按时间范围查询。"""
        import time as time_mod

        logger = AuditLogger()
        logger.log(AuditEventType.OTHER, "m1")
        mid_time = time_mod.time()
        logger.log(AuditEventType.OTHER, "m1")
        logger.log(AuditEventType.OTHER, "m1")
        results = logger.query(start_time=mid_time)
        assert len(results) == 2

    def test_query_limit(self):
        """查询结果限制。"""
        logger = AuditLogger()
        for _ in range(10):
            logger.log(AuditEventType.OTHER, "m1")
        results = logger.query(limit=3)
        assert len(results) == 3

    def test_get_recent(self):
        """获取最近 N 条。"""
        logger = AuditLogger()
        for i in range(5):
            logger.log(AuditEventType.OTHER, "m1", context={"i": i})
        recent = logger.get_recent(2)
        assert len(recent) == 2
        assert recent[-1]["context"]["i"] == 4

    def test_get_by_event_type(self):
        """按事件类型获取最近 N 条。"""
        logger = AuditLogger()
        logger.log(AuditEventType.TOOL_CALL, "m1")
        logger.log(AuditEventType.ACTION_SELECTION, "m1")
        logger.log(AuditEventType.TOOL_CALL, "m1")
        results = logger.get_by_event_type(AuditEventType.TOOL_CALL, n=5)
        assert len(results) == 2

    def test_export_json(self):
        """JSON 导出。"""
        logger = AuditLogger()
        logger.log(AuditEventType.ACTION_SELECTION, "m1", context={"s": 0})
        json_str = logger.export_json()
        data = json.loads(json_str)
        assert len(data) == 1
        assert data[0]["event_type"] == "ACTION_SELECTION"

    def test_export_csv(self):
        """CSV 导出。"""
        logger = AuditLogger()
        logger.log(AuditEventType.ACTION_SELECTION, "m1")
        csv_str = logger.export_csv()
        lines = csv_str.strip().split("\n")
        assert len(lines) == 2  # header + 1 row
        assert "entry_id" in lines[0]

    def test_stats(self):
        """统计信息。"""
        logger = AuditLogger()
        logger.log(AuditEventType.ACTION_SELECTION, "module_a")
        logger.log(AuditEventType.TOOL_CALL, "module_b")
        logger.log(AuditEventType.ACTION_SELECTION, "module_a")
        stats = logger.stats
        assert stats["n_entries"] == 3
        assert stats["by_event_type"]["ACTION_SELECTION"] == 2
        assert stats["by_event_type"]["TOOL_CALL"] == 1
        assert stats["by_module"]["module_a"] == 2

    def test_max_entries_bounded(self):
        """有界 deque 防止无界增长。"""
        logger = AuditLogger(max_entries=5)
        for _ in range(20):
            logger.log(AuditEventType.OTHER, "m1")
        assert logger.n_entries == 5

    def test_clear(self):
        """清空日志。"""
        logger = AuditLogger()
        logger.log(AuditEventType.OTHER, "m1")
        logger.clear()
        assert logger.n_entries == 0

    def test_log_value_violation_with_vector(self):
        """带价值观向量的违规记录。"""
        vv = ValueVector()
        vv.add_dimension("avoid_harm", "避免伤害", weight=2.0)
        logger = AuditLogger(value_vector=vv)
        entry = logger.log_value_violation(
            module="hierarchical_model",
            action_violations={"avoid_harm": 0.5},
            context={"step": 1},
        )
        assert entry is not None
        assert entry.event_type == AuditEventType.VALUE_VIOLATION
        assert entry.result["penalty"] == pytest.approx(1.0)

    def test_log_value_violation_without_vector(self):
        """无价值观向量时返回 None。"""
        logger = AuditLogger(value_vector=None)
        entry = logger.log_value_violation(
            module="m1",
            action_violations={"x": 1.0},
        )
        assert entry is None

    def test_set_value_vector(self):
        """动态设置价值观向量。"""
        logger = AuditLogger()
        assert logger.value_vector is None
        vv = ValueVector()
        logger.set_value_vector(vv)
        assert logger.value_vector is vv


# ------------------------------------------------------------------ #
# 集成演示
# ------------------------------------------------------------------ #
class TestObservabilityIntegration:
    """可观测性模块集成演示。"""

    def test_full_pipeline(self):
        """完整管线：思维链记录 → 审计日志 → 反事实解释。"""
        # 1. 思维链
        tc = ThoughtChain(anomaly_pe_threshold=2.0)
        tc.record(
            step=0,
            prediction_errors={"layer1": 0.3},
            confidence=0.9,
            tool_called="",
        )
        # 碰撞事件：高预测误差
        tc.record(
            step=1,
            prediction_errors={"layer1": 3.5},
            confidence=0.2,
            meta_triggered=True,
        )
        assert tc.n_anomalies == 1

        # 2. 审计日志
        logger = AuditLogger()
        logger.log(
            event_type=AuditEventType.ACTION_SELECTION,
            module="hierarchical_model",
            context={"step": 1},
            result={"action": "turn_right"},
        )

        # 3. 反事实解释
        def world_model(context, action):
            if action == "turn_right":
                return {
                    "free_energy": 1.2,
                    "prediction_error": 0.3,
                    "outcome": "安全通过",
                }
            return {
                "free_energy": 4.5,
                "prediction_error": 3.5,
                "outcome": "碰撞",
            }

        explainer = CounterfactualExplainer(world_model=world_model)
        result = explainer.explain(
            factual_state={
                "action": "turn_right",
                "free_energy": 1.2,
                "prediction_error": 0.3,
                "outcome": "安全通过",
                "context": {"step": 1},
            },
            alternative_action="turn_left",
        )
        assert "碰撞" in result.explanation
        assert result.free_energy_delta > 0  # 反事实更差
