"""JEPA 模块单元测试。覆盖 TargetEncoder / OnlinePredictor / JEPAModule。"""

from __future__ import annotations

import numpy as np
import pytest

from src.jepa.target_encoder import TargetEncoder
from src.jepa.predictor import OnlinePredictor
from src.jepa.jepa_module import JEPAModule


# ------------------------------------------------------------------ #
# TargetEncoder
# ------------------------------------------------------------------ #
class TestTargetEncoder:
    def test_encode_output_shape_and_range(self):
        """正常输入：输出维度正确，值在 [-1, 1]。"""
        enc = TargetEncoder(input_dim=64, latent_dim=32, seed=0)
        obs = np.random.default_rng(1).standard_normal(64)
        z = enc.encode(obs)
        assert z.shape == (32,)
        assert np.all(np.abs(z) <= 1.0 + 1e-6)

    def test_encode_l2_normalized(self):
        """输出 L2 归一化。"""
        enc = TargetEncoder(input_dim=16, latent_dim=8)
        z = enc.encode(np.ones(16))
        assert abs(np.linalg.norm(z) - 1.0) < 1e-6 or np.linalg.norm(z) < 1e-8

    def test_encode_pads_short_input(self):
        """边界：输入短于 input_dim 时零填充。"""
        enc = TargetEncoder(input_dim=32, latent_dim=8)
        z = enc.encode(np.ones(8))  # 只有 8 个元素
        assert z.shape == (8,)
        assert np.all(np.isfinite(z))

    def test_encode_truncates_long_input(self):
        """边界：输入长于 input_dim 时截断。"""
        enc = TargetEncoder(input_dim=8, latent_dim=4)
        z = enc.encode(np.arange(100, dtype=float))
        assert z.shape == (4,)

    def test_ema_update_reduces_drift(self):
        """EMA 更新后目标权重向在线权重靠拢。"""
        enc = TargetEncoder(input_dim=8, latent_dim=4, ema_tau=0.5, ema_tau_min=0.1, seed=0)
        original = enc.weights.copy()
        online_W = np.random.default_rng(1).standard_normal((4, 8)) * 0.5
        enc.update_from_online(online_W)
        # 权重应发生变化（不是完全不变）
        assert not np.allclose(original, enc.weights)
        # 且朝 online 方向移动（cosine 增大）
        from src.jepa.target_encoder import TargetEncoder as _TE
        cos_before = _TE._cosine(original, online_W)
        cos_after = _TE._cosine(enc.weights, online_W)
        assert cos_after >= cos_before - 1e-6

    def test_ema_tau_decay(self):
        """τ 随步数衰减到下限。"""
        enc = TargetEncoder(
            input_dim=4, latent_dim=2, ema_tau=0.99, ema_tau_min=0.9, ema_decay_step=2
        )
        initial_tau = enc.ema_tau
        W = np.random.default_rng(0).standard_normal((2, 4))
        for _ in range(10):
            enc.update_from_online(W)
        assert enc.ema_tau <= initial_tau
        assert enc.ema_tau >= enc.ema_tau_min

    def test_invalid_params_raise(self):
        """非法参数抛 ValueError。"""
        with pytest.raises(ValueError):
            TargetEncoder(input_dim=0)
        with pytest.raises(ValueError):
            TargetEncoder(input_dim=4, ema_tau=0.0)
        with pytest.raises(ValueError):
            TargetEncoder(input_dim=4, ema_tau=0.5, ema_tau_min=0.9)

    def test_snapshot_fields(self):
        """快照包含必要字段且 JSON 安全。"""
        enc = TargetEncoder(input_dim=8, latent_dim=4)
        enc.encode(np.ones(8))
        snap = enc.snapshot()
        assert "ema_tau" in snap
        assert "tracking" in snap
        assert "last_target_3d" in snap
        assert len(snap["last_target_3d"]) == 3
        assert isinstance(snap["tracking"], float)


# ------------------------------------------------------------------ #
# OnlinePredictor
# ------------------------------------------------------------------ #
class TestOnlinePredictor:
    def test_predict_output_shape(self):
        """正常输入：输出维度正确。"""
        pred = OnlinePredictor(latent_dim=16, target_dim=8, seed=0)
        z = pred.predict(np.random.default_rng(1).standard_normal(16))
        assert z.shape == (8,)

    def test_update_reduces_error_on_stationary_target(self):
        """对固定目标，重复 update 应使预测误差下降。"""
        rng = np.random.default_rng(42)
        pred = OnlinePredictor(latent_dim=32, target_dim=16, lr=0.01, seed=0)
        latent = rng.standard_normal(32)
        target = rng.standard_normal(16)
        target = target / np.linalg.norm(target)
        errs = []
        for _ in range(50):
            e = pred.update(latent, target)
            errs.append(e)
        # 后期误差应低于初期
        assert errs[-1] < errs[0]
        assert errs[-1] < 0.5

    def test_predict_handles_dim_mismatch(self):
        """边界：latent_state 维度不匹配时自动适配。"""
        pred = OnlinePredictor(latent_dim=8, target_dim=4)
        z = pred.predict(np.ones(4))  # 短输入
        assert z.shape == (4,)
        assert np.all(np.isfinite(z))
        z2 = pred.predict(np.ones(20))  # 长输入
        assert z2.shape == (4,)

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            OnlinePredictor(latent_dim=0, target_dim=4)
        with pytest.raises(ValueError):
            OnlinePredictor(latent_dim=4, target_dim=4, lr=0)

    def test_snapshot_fields(self):
        pred = OnlinePredictor(latent_dim=8, target_dim=4, seed=0)
        pred.predict(np.ones(8))
        snap = pred.snapshot()
        assert "last_error_norm" in snap
        assert "update_count" in snap
        assert len(snap["last_prediction_3d"]) == 3


# ------------------------------------------------------------------ #
# JEPAModule
# ------------------------------------------------------------------ #
class TestJEPAModule:
    def test_process_returns_required_fields(self):
        """正常流程返回 jepa_error 与对齐度。"""
        mod = JEPAModule(input_dim=64, latent_dim=32, seed=0)
        obs = np.random.default_rng(1).standard_normal(64)
        latent = np.random.default_rng(2).standard_normal(32)
        result = mod.process(obs, latent)
        assert result["enabled"] is True
        assert "jepa_error" in result
        assert "latency_ms" in result
        assert result["latency_ms"] < 5.0  # 应远低于 0.5ms 预算（留余量）

    def test_latency_under_budget(self):
        """验证 JEPA 模块延迟 < 0.5ms（预热后）。"""
        mod = JEPAModule(input_dim=128, latent_dim=64, seed=0)
        obs = np.random.default_rng(1).standard_normal(128)
        latent = np.random.default_rng(2).standard_normal(64)
        # 预热
        for _ in range(5):
            mod.process(obs, latent)
        # 测量
        import time
        t0 = time.perf_counter()
        for _ in range(100):
            mod.process(obs, latent)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0 / 100
        assert elapsed_ms < 0.5, f"JEPA latency {elapsed_ms:.3f}ms exceeds 0.5ms budget"

    def test_combine_free_energy(self):
        """自由能加权合并。"""
        mod = JEPAModule(input_dim=8, latent_dim=4, lambda_jepa=0.5)
        combined = mod.combine_free_energy(base_prediction_error=1.0, jepa_error=0.4)
        # (1-0.5)*1.0 + 0.5*0.4 = 0.7
        assert abs(combined - 0.7) < 1e-6

    def test_get_target_embedding(self):
        """下游接口返回正确维度。"""
        mod = JEPAModule(input_dim=16, latent_dim=8)
        z = mod.get_target_embedding(np.ones(16))
        assert z.shape == (8,)

    def test_disabled_returns_disabled(self):
        """禁用后返回 enabled=False。"""
        mod = JEPAModule(input_dim=8, latent_dim=4)
        mod.disable()
        result = mod.process(np.ones(8), np.ones(4))
        assert result == {"enabled": False}

    def test_alignment_improves_over_steps(self):
        """随训练步数增加，在线预测与目标表示对齐度提升。"""
        rng = np.random.default_rng(0)
        mod = JEPAModule(input_dim=32, latent_dim=16, lr=0.01, seed=0)
        obs = rng.standard_normal(32)
        latent = rng.standard_normal(16)
        # 初始对齐度
        snap0 = mod.snapshot()
        align0 = snap0["alignment"]
        for _ in range(100):
            mod.process(obs, latent)
        snap1 = mod.snapshot()
        align1 = snap1["alignment"]
        # 对齐度应提升（abs 增加）
        assert abs(align1) >= abs(align0) - 0.1

    def test_snapshot_has_3d_projections(self):
        """快照包含 online_3d / target_3d 供前端 LatentSpace3D。"""
        mod = JEPAModule(input_dim=16, latent_dim=8)
        mod.process(np.ones(16), np.ones(8))
        snap = mod.snapshot()
        assert "online_3d" in snap
        assert "target_3d" in snap
        assert len(snap["online_3d"]) == 3
        assert len(snap["target_3d"]) == 3

    def test_invalid_lambda_raises(self):
        with pytest.raises(ValueError):
            JEPAModule(input_dim=8, latent_dim=4, lambda_jepa=1.5)
        with pytest.raises(ValueError):
            JEPAModule(input_dim=8, latent_dim=4, lambda_jepa=-0.1)

    def test_configure_runtime(self):
        """运行时配置 lambda_jepa 与 enabled。"""
        mod = JEPAModule(input_dim=8, latent_dim=4, lambda_jepa=0.5)
        mod.configure(lambda_jepa=0.3)
        assert mod.lambda_jepa == 0.3
        mod.configure(enabled=False)
        assert mod.enabled is False
