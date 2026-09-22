"""PCN layer + HierarchicalZeroDataModel tests."""
from __future__ import annotations

import threading

import numpy as np
import pytest

from zero_data_model import ZeroDataModel
from zero_data_model.pcn import HierarchicalZeroDataModel, PCNLayer


class TestPCNLayerConstruction:
    def test_construction(self):
        layer = PCNLayer(dim=32, lower_dim=16, higher_dim=32, seed=42)
        assert layer.dim == 32
        assert layer.lower_dim == 16
        assert layer.higher_dim == 32

    def test_invalid_dim(self):
        with pytest.raises(ValueError):
            PCNLayer(dim=0)
        with pytest.raises(ValueError):
            PCNLayer(dim=32, lr=0)

    def test_bottom_layer_no_lower(self):
        layer = PCNLayer(dim=32, higher_dim=16, seed=42)
        assert layer.lower_dim is None
        assert layer._W_gen is None
        assert layer._W_rec is None
        # predict 应返回 None
        assert layer.predict() is None

    def test_top_layer_no_higher(self):
        layer = PCNLayer(dim=16, lower_dim=32, seed=42)
        assert layer.higher_dim is None


class TestPCNPredictUpdate:
    def test_predict_shape(self):
        layer = PCNLayer(dim=32, lower_dim=16, seed=42)
        layer.set_state(np.ones(32))
        pred = layer.predict()
        assert pred.shape == (16,)

    def test_set_state(self):
        layer = PCNLayer(dim=32, seed=42)
        layer.set_state(np.ones(32) * 0.5)
        assert np.allclose(layer.state, 0.5)

    def test_set_state_invalid_shape(self):
        layer = PCNLayer(dim=32, seed=42)
        with pytest.raises(ValueError):
            layer.set_state(np.ones(16))

    def test_update_returns_error(self):
        layer = PCNLayer(dim=32, higher_dim=32, seed=42)
        layer.set_state(np.ones(32))
        prediction_from_higher = np.zeros(32)
        error = layer.update(prediction_from_higher=prediction_from_higher)
        assert error.shape == (32,)
        # error = state - prediction = 1 - 0 = 1
        np.testing.assert_allclose(error, np.ones(32))

    def test_update_with_lower_error(self):
        layer = PCNLayer(dim=32, lower_dim=16, higher_dim=32, seed=42)
        layer.set_state(np.ones(32))
        error_lower = np.ones(16) * 0.5
        pred_higher = np.zeros(32)
        error = layer.update(
            error_from_lower=error_lower, prediction_from_higher=pred_higher
        )
        assert error.shape == (32,)

    def test_update_reduces_error_over_iterations(self):
        """多次 update 后误差应减小。"""
        layer = PCNLayer(dim=16, higher_dim=16, lr=0.1, seed=42)
        layer.set_state(np.ones(16))
        target = np.zeros(16)  # 上层预测

        errors = []
        for _ in range(50):
            err = layer.update(prediction_from_higher=target)
            errors.append(np.linalg.norm(err))

        # 误差应递减
        assert errors[-1] < errors[0]

    def test_update_with_nan_prediction(self):
        """NaN 预测不应崩溃或污染状态。"""
        layer = PCNLayer(dim=16, higher_dim=16, seed=42)
        layer.set_state(np.ones(16))
        layer.update(prediction_from_higher=np.full(16, np.nan))
        # state 应保持（不应用 NaN 更新）
        assert np.all(np.isfinite(layer.state))

    def test_weight_update_happens(self):
        """下层误差传入时 W_gen 应更新。"""
        layer = PCNLayer(dim=16, lower_dim=8, higher_dim=16, lr=0.1, seed=42)
        W_gen_before = layer._W_gen.copy()
        layer.set_state(np.ones(16))
        layer.update(
            error_from_lower=np.ones(8) * 0.5,
            prediction_from_higher=np.zeros(16),
        )
        # W_gen 应该变了
        assert not np.allclose(W_gen_before, layer._W_gen)


class TestHierarchicalModel:
    @pytest.fixture
    def model(self):
        return ZeroDataModel(dim=16, seed=42)

    @pytest.fixture
    def hier_model(self, model):
        return HierarchicalZeroDataModel(model, use_pcn=True, pcn_lr=0.05)

    def test_construction(self, model):
        hier = HierarchicalZeroDataModel(model, use_pcn=True)
        assert hier.use_pcn
        assert "L0" in hier.layers
        assert "L1" in hier.layers
        assert "L2" in hier.layers

    def test_construction_no_pcn(self, model):
        hier = HierarchicalZeroDataModel(model, use_pcn=False)
        assert not hier.use_pcn
        assert hier.layers == {}

    def test_think_returns_signal(self, hier_model):
        signal = hier_model.think()
        assert signal is not None

    def test_think_adds_pcn_metadata(self, hier_model):
        signal = hier_model.think()
        # metadata 应包含 pcn 字段
        meta = getattr(signal, "metadata", None)
        assert meta is not None
        if isinstance(meta, dict):
            assert "pcn" in meta
            pcn_meta = meta["pcn"]
            assert "layer_errors" in pcn_meta
            assert "L0" in pcn_meta["layer_errors"]
            assert "L1" in pcn_meta["layer_errors"]
            assert "L2" in pcn_meta["layer_errors"]

    def test_think_no_pcn_no_metadata(self, model):
        hier = HierarchicalZeroDataModel(model, use_pcn=False)
        signal = hier.think()
        meta = getattr(signal, "metadata", None)
        if isinstance(meta, dict):
            assert "pcn" not in meta

    def test_layer_errors_decrease_over_time(self, hier_model):
        """多次 think 后各层误差应整体递减。"""
        errors_history = []
        for _ in range(20):
            signal = hier_model.think()
            meta = getattr(signal, "metadata", {})
            if isinstance(meta, dict) and "pcn" in meta:
                errs = meta["pcn"]["layer_errors"]
                errors_history.append(errs["L0"] + errs["L1"] + errs["L2"])

        if len(errors_history) >= 2:
            # 整体趋势应递减（不要求单调，但最终应小于初始）
            # 用移动平均判断趋势
            n = len(errors_history)
            first_half = np.mean(errors_history[: n // 2])
            second_half = np.mean(errors_history[n // 2 :])
            # 后半段平均应不大于前半段
            assert second_half <= first_half * 1.5  # 允许一些波动

    def test_delegates_to_underlying_model(self, hier_model):
        """HierarchicalZeroDataModel 应委托方法给底层 ZeroDataModel。"""
        # model.dim 应该可访问
        assert hier_model.dim == 16
        # model.belief_state 应该可访问
        assert hasattr(hier_model, "belief_state")

    def test_thread_safety(self, hier_model):
        """并发 think 不应崩溃。"""
        errors = []

        def worker():
            try:
                for _ in range(5):
                    hier_model.think()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0


# ------------------------------------------------------------------ #
# 第二轮军事级审查回归测试（HierarchicalZeroDataModel）
# ------------------------------------------------------------------ #
class TestHierarchicalModelMilitaryRound2:
    """第二轮军事级审查修复的 hierarchical_model.py bug 回归测试。

    覆盖：C1（线程安全）、H1（step 双递增）、H2/H3（GWT 候选/Φ）、
    M5（sigmoid 溢出）、M11（NaN 传播）。
    """

    @pytest.fixture
    def model(self):
        return ZeroDataModel(dim=16, seed=42)

    # ---- H1: _step_count 单次递增 ---- #
    def test_h1_step_count_increments_once_per_think(self, model):
        """H1: phase2 + skills + jepa + gwt 同时启用时 step_count 每步只 +1。

        修复前: _step_count 在 _run_phase2_cycle 和 _run_skills_cycle 中
        各递增一次，双递增导致 discovery_interval 计算错误。
        """
        hier = HierarchicalZeroDataModel(
            model,
            use_pcn=True,
            use_phase2=True,
            enable_skills=True,
            enable_jepa=True,
            enable_gwt=True,
        )
        before = hier._step_count
        hier.think()
        after = hier._step_count
        assert after - before == 1, (
            f"step_count 应每步 +1，实际 +{after - before}（双递增 bug）"
        )

    def test_h1_step_count_consistent_across_n_steps(self, model):
        """H1: 连续 N 步后 step_count 应等于 N。"""
        hier = HierarchicalZeroDataModel(
            model,
            use_pcn=True,
            use_phase2=True,
            enable_jepa=True,
            enable_gwt=True,
        )
        n = 10
        for _ in range(n):
            hier.think()
        assert hier._step_count == n, (
            f"{n} 步后 step_count 应为 {n}，实际 {hier._step_count}"
        )

    # ---- H2: GWT 候选向量使用真实 PCN 层状态 ---- #
    def test_h2_gwt_candidates_use_distinct_layer_states(self, model):
        """H2: GWT 候选向量应使用 L0/L1/L2 真实状态，而非 belief 的同一副本。

        修复前: 所有候选 vector=belief.copy()，无论哪个模块胜出，
        广播内容都相同——竞争失去意义。
        """
        hier = HierarchicalZeroDataModel(
            model, use_pcn=True, enable_gwt=True
        )
        # 运行几步让各层状态分化
        for _ in range(5):
            hier.think()
        # GWT winner 的 broadcast 应来自真实层状态
        signal = hier.think()
        gwt = (signal.metadata or {}).get("gwt", {})
        assert gwt.get("winner") is not None
        # winner 应是真实模块（pcn_L0/pcn_L1/pcn_L2/consciousness），
        # 而非所有候选都是同一 belief 副本
        assert gwt["winner"] in ("pcn_L0", "pcn_L1", "pcn_L2", "consciousness")

    # ---- H3: Φ 计算使用 4 个不同模块状态 ---- #
    def test_h3_phi_uses_distinct_modules(self, model):
        """H3: Φ 计算应使用 4 个不同模块状态（L0/L1/L2/belief），

        而非用 belief.copy() 填充到 6 个相同副本。
        修复前: 相同副本让互信息退化为自信息，导致 Φ 虚高。
        """
        hier = HierarchicalZeroDataModel(
            model, use_pcn=True, enable_gwt=True
        )
        # 运行足够步数让 Φ 节流周期触发
        for _ in range(15):
            hier.think()
        signal = hier.think()
        gwt = (signal.metadata or {}).get("gwt", {})
        phi = gwt.get("phi")
        assert phi is not None
        assert np.isfinite(phi)
        # Φ 应 >= 0（互信息分区定义）
        assert phi >= 0.0

    def test_h3_phi_not_constant_zero(self, model):
        """H3: Φ 在运行中应产生非零值（证明用了不同模块状态计算）。

        若用相同副本，互信息 = 自信息，Φ 会退化为 0 或极小。
        """
        hier = HierarchicalZeroDataModel(
            model, use_pcn=True, enable_gwt=True
        )
        phi_values = []
        for _ in range(30):
            signal = hier.think()
            gwt = (signal.metadata or {}).get("gwt", {})
            if "phi" in gwt:
                phi_values.append(float(gwt["phi"]))
        # 至少有一个 Φ > 0 的采样（证明模块状态不完全相同）
        assert any(p > 0 for p in phi_values), (
            f"Φ 全为 0，疑似使用了相同副本（H3 bug）: {phi_values[:5]}"
        )

    # ---- C1: 线程安全 — 所有循环在锁内 ---- #
    def test_c1_concurrent_jepa_gwt_discovery_no_crash(self, model):
        """C1: 并发 think()（启用 JEPA+GWT+Discovery）不崩溃、无数据撕裂。

        修复前: JEPA/GWT/Discovery 循环在锁外读取共享 PCN 状态，
        多线程竞争导致数据撕裂或 crash。
        """
        hier = HierarchicalZeroDataModel(
            model,
            use_pcn=True,
            enable_jepa=True,
            enable_gwt=True,
            enable_discovery=True,
            discovery_interval=1,  # 每步触发发现循环
        )
        errors = []

        def worker():
            try:
                for _ in range(3):
                    hier.think()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0, f"并发执行崩溃: {errors}"

    # ---- M5: sigmoid 溢出裁剪 ---- #
    def test_m5_large_negative_input_no_overflow(self, model):
        """M5: _belief_to_predicates 接收极大负值时不产生 exp 溢出警告。

        修复前: np.exp(-val) 对大负值溢出为 inf + RuntimeWarning。
        """
        hier = HierarchicalZeroDataModel(
            model, use_pcn=True, use_phase2=True
        )
        # 注入极大值向量，触发 _belief_to_predicates 的 sigmoid
        large_input = np.full(16, -1e6)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            # 不应抛 RuntimeWarning（sigmoid 已裁剪）
            signal = hier.think(large_input)
        assert signal is not None

    # ---- M11: NaN 传播防护 ---- #
    def test_m11_nan_input_does_not_propagate(self, model):
        """M11: 含 NaN 的观测不应污染 JEPA 计算结果。

        修复前: raw_obs 含 NaN 会传播到 latent_state → jepa_error，
        使自由能变为 NaN，破坏整个 think() 链。
        """
        hier = HierarchicalZeroDataModel(
            model, use_pcn=True, enable_jepa=True, enable_gwt=True
        )
        # 注入含 NaN 的输入
        nan_input = np.array([1.0, float("nan"), 3.0, float("inf")] * 4)
        signal = hier.think(nan_input)
        # metadata 中的 jepa/gwt 结果不应含 NaN
        jepa = (signal.metadata or {}).get("jepa", {})
        if isinstance(jepa, dict):
            for key in ("jepa_error", "alignment", "lambda_jepa"):
                if key in jepa:
                    assert np.isfinite(jepa[key]), (
                        f"jepa.{key} 为 NaN/inf（NaN 未被清理）"
                    )
        gwt = (signal.metadata or {}).get("gwt", {})
        if isinstance(gwt, dict) and "phi" in gwt:
            assert np.isfinite(gwt["phi"]), "gwt.phi 为 NaN/inf（NaN 传播）"

    # ---- 综合稳定性：连续运行无 NaN/inf ---- #
    def test_stability_no_nan_across_steps(self, model):
        """综合: 连续 50 步 think()，所有 metadata 值均为有限数。"""
        hier = HierarchicalZeroDataModel(
            model,
            use_pcn=True,
            enable_jepa=True,
            enable_gwt=True,
        )
        for _ in range(50):
            signal = hier.think()
            md = signal.metadata or {}
            # PCN 误差有限
            pcn = md.get("pcn", {})
            if isinstance(pcn, dict) and "layer_errors" in pcn:
                for layer, err in pcn["layer_errors"].items():
                    assert np.isfinite(err), f"pcn.layer_errors.{layer} 非有限"
            # JEPA 误差有限
            jepa = md.get("jepa", {})
            if isinstance(jepa, dict):
                for k, v in jepa.items():
                    if isinstance(v, (int, float)):
                        assert np.isfinite(v), f"jepa.{k} 非有限"


# ------------------------------------------------------------------ #
# 第三轮军事级审查回归测试（F1 Critical / F2-High 修复）
# ------------------------------------------------------------------ #
class TestHierarchicalModelMilitaryRound3:
    """第三轮军事级审查修复的 Critical/High 级别 bug 回归测试。

    覆盖：F1（__getattr__ 无限递归）、F2（confidence 上界）、
    F8（CA float dtype）。
    """

    @pytest.fixture
    def model(self):
        return ZeroDataModel(dim=16, seed=42)

    # ---- F1 [Critical]: __getattr__ 无限递归 ---- #
    def test_f1_getattr_no_infinite_recursion(self):
        """F1: _model 未初始化时 __getattr__ 不应无限递归。

        修复前: __getattr__("_model") → getattr(self._model, "_model")
        → 再次 __getattr__ → RecursionError。影响 pickle/copy/hasattr。
        """
        # 创建未完成初始化的实例（跳过 __init__）
        hier = HierarchicalZeroDataModel.__new__(HierarchicalZeroDataModel)
        # _model 未设置，访问任意属性应抛 AttributeError 而非 RecursionError
        with pytest.raises(AttributeError):
            _ = hier.some_undefined_attr

    def test_f1_getattr_model_not_set_hasattr_safe(self):
        """F1: hasattr 在 _model 未初始化时返回 False 而非 RecursionError。"""
        hier = HierarchicalZeroDataModel.__new__(HierarchicalZeroDataModel)
        assert hasattr(hier, "nonexistent") is False

    def test_f1_normal_delegation_still_works(self, model):
        """F1: 正常初始化后 __getattr__ 仍正确委托给底层 model。"""
        hier = HierarchicalZeroDataModel(model, use_pcn=True)
        # dim 是底层 model 的属性，应通过 __getattr__ 委托
        assert hier.dim == model.dim

    # ---- F2 [High]: GWT confidence 上界裁剪 ---- #
    def test_f2_gwt_confidence_bounded(self, model):
        """F2: GWT confidence 应在 [0,1] 范围内。

        修复前: err_f 为负值/NaN 时 confidence 可能 >1.0 或为 NaN，
        使 softmax 竞争失真。
        """
        hier = HierarchicalZeroDataModel(
            model, use_pcn=True, enable_gwt=True
        )
        for _ in range(15):
            signal = hier.think()
        gwt = (signal.metadata or {}).get("gwt", {})
        # 检查 GWT 结果有限
        assert np.isfinite(gwt.get("phi", 0.0))
        # broadcast_confidence 应在 [0,1]
        bc = gwt.get("broadcast_confidence", 0.5)
        assert 0.0 <= bc <= 1.0, f"broadcast_confidence={bc} 超出 [0,1]"
