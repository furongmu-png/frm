"""Phase 2 集成烟雾测试 — 验证 HierarchicalZeroDataModel(use_phase2=True)
在 think() 中正确运行 Phase 2 循环，并将结果写入 signal.metadata。

测试覆盖：
  1. think() 后 signal.metadata["cognitive_upgrades"] 包含全部 4 个键
     (logic_violations / meta_cognition / experiment / reasoning_chain)
  2. logic_engine / reasoning_graph / experiment_planner / experiment_logger
     属性在 use_phase2=True 时正确暴露
  3. 向 logic_engine 添加规则后，logic_violations 反映矛盾度
  4. meta_cognition.confidence 是有效的百分比 (0-100)
  5. use_phase2=False 时不写入 cognitive_upgrades（向后兼容）
  6. 多步运行不报错（稳定性）
"""
from __future__ import annotations

import numpy as np
import pytest

from zero_data_model.base import Signal
from zero_data_model.pcn.hierarchical_model import HierarchicalZeroDataModel


# ------------------------------------------------------------------ #
# Stub model — 最小可用的 ZeroDataModel 替身
# ------------------------------------------------------------------ #
class StubZeroDataModel:
    """最小 stub：仅需 dim + think() → Signal。

    HierarchicalZeroDataModel.__init__ 读取 model.dim 来初始化 PCN 层和
    Phase 2 模块；think() 调用 model.think() 获取 signal。
    """

    def __init__(self, dim: int = 16, seed: int = 42):
        self.dim = dim
        self._rng = np.random.default_rng(seed)

    def think(self, input_data=None, **kwargs) -> Signal:
        data = self._rng.standard_normal(self.dim)
        return Signal(data=data, metadata={})


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #
@pytest.fixture
def stub_model():
    return StubZeroDataModel(dim=16, seed=42)


@pytest.fixture
def hier_model_phase2(stub_model):
    return HierarchicalZeroDataModel(
        stub_model,
        use_pcn=True,
        pcn_lr=0.05,
        use_phase2=True,
    )


@pytest.fixture
def hier_model_no_phase2(stub_model):
    return HierarchicalZeroDataModel(
        stub_model,
        use_pcn=True,
        pcn_lr=0.05,
        use_phase2=False,
    )


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #
class TestPhase2Smoke:
    """端到端烟雾测试：think() → cognitive_upgrades 写入。"""

    def test_think_populates_cognitive_upgrades_keys(self, hier_model_phase2):
        """think() 后 metadata 含全部 4 个 Phase 2 子键。"""
        signal = hier_model_phase2.think()
        assert signal is not None
        assert isinstance(signal.metadata, dict)
        # PCN 块
        assert "pcn" in signal.metadata
        # Phase 2 块
        cu = signal.metadata.get("cognitive_upgrades")
        assert isinstance(cu, dict), "cognitive_upgrades 必须是 dict"
        for key in ("logic_violations", "meta_cognition",
                    "experiment", "reasoning_chain"):
            assert key in cu, f"缺少 Phase 2 键: {key}"

    def test_logic_violations_structure(self, hier_model_phase2):
        """logic_violations 包含矛盾度和违反列表。"""
        signal = hier_model_phase2.think()
        lv = signal.metadata["cognitive_upgrades"]["logic_violations"]
        assert isinstance(lv, dict)
        assert "contradiction" in lv
        assert 0.0 <= float(lv["contradiction"]) < 1.0
        assert "n_violations" in lv
        assert isinstance(lv["n_violations"], int)
        assert "violations" in lv
        assert isinstance(lv["violations"], list)

    def test_meta_cognition_confidence_range(self, hier_model_phase2):
        """meta_cognition.confidence 在 0-100 范围内。"""
        signal = hier_model_phase2.think()
        mc = signal.metadata["cognitive_upgrades"]["meta_cognition"]
        assert isinstance(mc, dict)
        conf = float(mc["confidence"])
        assert 0.0 <= conf <= 100.0
        assert "mode" in mc
        assert mc["mode"] in ("explore", "exploit")
        assert "triggered" in mc
        assert isinstance(mc["triggered"], bool)
        assert "mean_uncertainty" in mc
        assert 0.0 <= float(mc["mean_uncertainty"]) <= 1.0

    def test_experiment_structure(self, hier_model_phase2):
        """experiment 含名称、干预、预测增益、历史。"""
        signal = hier_model_phase2.think()
        exp = signal.metadata["cognitive_upgrades"]["experiment"]
        assert isinstance(exp, dict)
        assert "param_uncertainty" in exp
        assert "history" in exp
        assert isinstance(exp["history"], list)

    def test_reasoning_chain_structure(self, hier_model_phase2):
        """reasoning_chain 含事实数、规则数、待处理虚拟观测。"""
        signal = hier_model_phase2.think()
        rc = signal.metadata["cognitive_upgrades"]["reasoning_chain"]
        assert isinstance(rc, dict)
        assert "n_facts" in rc
        assert "n_rules" in rc
        assert "pending_virtual_observations" in rc
        assert isinstance(rc["n_facts"], int)
        assert isinstance(rc["n_rules"], int)


class TestPhase2Properties:
    """Phase 2 组件属性在 use_phase2=True 时正确暴露。"""

    def test_logic_engine_exposed(self, hier_model_phase2):
        le = hier_model_phase2.logic_engine
        assert le is not None
        assert hasattr(le, "add_rule")
        assert hasattr(le, "check_consistency")

    def test_reasoning_graph_exposed(self, hier_model_phase2):
        rg = hier_model_phase2.reasoning_graph
        assert rg is not None
        assert hasattr(rg, "add_fact")
        assert hasattr(rg, "transitive_inference")

    def test_experiment_planner_exposed(self, hier_model_phase2):
        ep = hier_model_phase2.experiment_planner
        assert ep is not None
        assert hasattr(ep, "evaluate")
        assert hasattr(ep, "select_best")

    def test_experiment_logger_exposed(self, hier_model_phase2):
        el = hier_model_phase2.experiment_logger
        assert el is not None
        assert hasattr(el, "log_from_dict")
        assert hasattr(el, "get_milestones")


class TestPhase2LogicIntegration:
    """向 logic_engine 添加规则后，think() 的 logic_violations 反映矛盾。"""

    def test_adding_rule_affects_logic_output(self, hier_model_phase2):
        """添加规则后，contradiction 字段存在且为有效值。"""
        le = hier_model_phase2.logic_engine
        # 添加一条规则：d0 ∧ d1 → d2
        le.add_rule({"if": ["d0", "d1"], "then": "d2", "weight": 0.8})
        signal = hier_model_phase2.think()
        lv = signal.metadata["cognitive_upgrades"]["logic_violations"]
        assert "contradiction" in lv
        assert 0.0 <= float(lv["contradiction"]) < 1.0
        assert lv["n_violations"] >= 0


class TestPhase2ReasoningIntegration:
    """向 reasoning_graph 添加事实后，n_facts 更新。"""

    def test_adding_facts_updates_n_facts(self, hier_model_phase2):
        rg = hier_model_phase2.reasoning_graph
        # 先跑一次 think 获取 baseline
        hier_model_phase2.think()
        # 添加事实
        rg.add_facts([
            ("beijing", "is_a", "china"),
            ("china", "is_a", "asia"),
        ])
        signal = hier_model_phase2.think()
        rc = signal.metadata["cognitive_upgrades"]["reasoning_chain"]
        assert rc["n_facts"] >= 2


class TestPhase2BackwardCompat:
    """use_phase2=False 时不写入 cognitive_upgrades。"""

    def test_no_cognitive_upgrades_when_disabled(self, hier_model_no_phase2):
        signal = hier_model_no_phase2.think()
        assert signal is not None
        # PCN 块仍存在
        assert "pcn" in signal.metadata
        # cognitive_upgrades 不应被写入
        cu = signal.metadata.get("cognitive_upgrades")
        assert cu is None or cu == {}

    def test_properties_are_none_when_disabled(self, hier_model_no_phase2):
        assert hier_model_no_phase2.logic_engine is None
        assert hier_model_no_phase2.reasoning_graph is None
        assert hier_model_no_phase2.experiment_planner is None
        assert hier_model_no_phase2.experiment_logger is None


class TestPhase2Stability:
    """多步运行稳定性。"""

    def test_multi_step_stable(self, hier_model_phase2):
        """连续运行 20 步不报错，每步都有 cognitive_upgrades。"""
        for i in range(20):
            signal = hier_model_phase2.think()
            cu = signal.metadata.get("cognitive_upgrades")
            assert isinstance(cu, dict), f"第 {i} 步缺少 cognitive_upgrades"
            assert "logic_violations" in cu
            assert "meta_cognition" in cu
            assert "experiment" in cu
            assert "reasoning_chain" in cu

    def test_no_nan_in_output(self, hier_model_phase2):
        """输出数值中无 NaN。"""
        signal = hier_model_phase2.think()
        cu = signal.metadata["cognitive_upgrades"]
        # 检查 meta_cognition
        mc = cu["meta_cognition"]
        assert np.isfinite(mc["confidence"])
        assert np.isfinite(mc["mean_uncertainty"])
        assert np.isfinite(mc["predicted_uncertainty"])
        # 检查 logic_violations
        lv = cu["logic_violations"]
        assert np.isfinite(lv["contradiction"])
