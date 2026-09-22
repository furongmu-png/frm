"""S4 layer tests — verify HiPPO stability, prediction, Hebbian update."""
from __future__ import annotations

import numpy as np
import pytest

from zero_data_model.s4 import S4Layer


class TestS4Construction:
    def test_construction_default(self):
        layer = S4Layer(state_dim=32, input_dim=8, output_dim=8, seed=42)
        assert layer.state_dim == 32
        assert layer.input_dim == 8
        assert layer.output_dim == 8

    def test_construction_invalid_dims(self):
        with pytest.raises(ValueError):
            S4Layer(state_dim=0, input_dim=8, output_dim=8)
        with pytest.raises(ValueError):
            S4Layer(state_dim=32, input_dim=0, output_dim=8)
        with pytest.raises(ValueError):
            S4Layer(state_dim=32, input_dim=8, output_dim=0)

    def test_construction_invalid_dt(self):
        with pytest.raises(ValueError):
            S4Layer(state_dim=32, input_dim=8, output_dim=8, dt=0)
        with pytest.raises(ValueError):
            S4Layer(state_dim=32, input_dim=8, output_dim=8, dt=-0.1)

    def test_initial_state_zero(self):
        layer = S4Layer(state_dim=32, input_dim=8, output_dim=8, seed=42)
        assert np.allclose(layer.state, 0)

    def test_spectral_radius_le_one(self):
        """HiPPO 对角初始化的谱半径应 < 1（稳定性）。"""
        layer = S4Layer(state_dim=64, input_dim=16, output_dim=16, dt=0.1, seed=42)
        assert layer.spectral_radius < 1.0, f"spectral_radius={layer.spectral_radius}"


class TestS4StepForward:
    def test_step_output_shape(self):
        layer = S4Layer(state_dim=32, input_dim=8, output_dim=4, seed=42)
        u = np.ones(8)
        y = layer.step(u)
        assert y.shape == (4,)

    def test_step_invalid_input_shape(self):
        layer = S4Layer(state_dim=32, input_dim=8, output_dim=4, seed=42)
        with pytest.raises(ValueError):
            layer.step(np.ones(7))

    def test_step_updates_state(self):
        """step 应更新隐状态（非零输入后 state 非零）。"""
        layer = S4Layer(state_dim=32, input_dim=8, output_dim=4, seed=42)
        u = np.ones(8) * 1.0
        layer.step(u)
        assert np.linalg.norm(layer.state) > 0

    def test_forward_output_shape(self):
        layer = S4Layer(state_dim=32, input_dim=8, output_dim=4, seed=42)
        u = np.ones((20, 8))
        y = layer.forward(u)
        assert y.shape == (20, 4)

    def test_forward_invalid_shape(self):
        layer = S4Layer(state_dim=32, input_dim=8, output_dim=4, seed=42)
        with pytest.raises(ValueError):
            layer.forward(np.ones((20, 7)))

    def test_forward_does_not_mutate_state(self):
        """forward 是纯查询，不应污染在线 state。"""
        layer = S4Layer(state_dim=32, input_dim=8, output_dim=4, seed=42)
        # 先 step 一次建立非零 state
        layer.step(np.ones(8))
        state_before = layer.state.copy()
        # forward 不应改变 state
        layer.forward(np.ones((10, 8)))
        state_after = layer.state.copy()
        np.testing.assert_array_equal(state_before, state_after)

    def test_step_sequence_consistent_with_forward(self):
        """逐步 step 的结果应与 forward 一致（递归模式）。"""
        layer = S4Layer(state_dim=16, input_dim=4, output_dim=2, seed=42, dt=0.05)
        u = np.random.default_rng(0).standard_normal((15, 4))
        # forward
        y_forward = layer.forward(u)
        # 逐步 step（需要新 layer）
        layer2 = S4Layer(state_dim=16, input_dim=4, output_dim=2, seed=42, dt=0.05)
        y_step = np.array([layer2.step(u[t]) for t in range(15)])
        np.testing.assert_allclose(y_forward, y_step, atol=1e-12)


class TestS4HebbianUpdate:
    def test_update_reduces_error(self):
        """Hebbian 更新应减少相同输入下的预测误差。"""
        layer = S4Layer(state_dim=32, input_dim=8, output_dim=4, seed=42, lr=0.1)
        u = np.ones(8)
        target = np.array([1.0, 0.0, 0.0, 0.0])
        # warm up state
        for _ in range(5):
            layer.step(u)
        # 初始误差
        y0 = layer.step(u)
        err0 = y0 - target
        # 更新若干次
        for _ in range(20):
            y = layer.step(u)
            layer.update(y - target, u)
        # 最终误差应减小
        y_final = layer.step(u)
        err_final = y_final - target
        assert np.linalg.norm(err_final) < np.linalg.norm(err0)

    def test_update_invalid_error_shape(self):
        layer = S4Layer(state_dim=32, input_dim=8, output_dim=4, seed=42)
        with pytest.raises(ValueError):
            layer.update(np.ones(3))

    def test_update_with_nan_error_no_crash(self):
        """NaN 误差不应崩溃或污染权重。"""
        layer = S4Layer(state_dim=32, input_dim=8, output_dim=4, seed=42)
        C_before = layer._C.copy()
        layer.update(np.array([np.nan, 0, 0, 0]))
        np.testing.assert_array_equal(layer._C, C_before)

    def test_update_with_none_u_skips_B(self):
        """u=None 时只更新 C，不动 B。"""
        layer = S4Layer(state_dim=32, input_dim=8, output_dim=4, seed=42, lr=0.1)
        layer.step(np.ones(8))  # warm up
        B_before = layer._B_bar.copy()
        layer.update(np.ones(4) * 0.1, u=None)
        np.testing.assert_array_equal(layer._B_bar, B_before)
        # C 应该变了
        # (need to step to populate state, then update)


class TestS4Stability:
    def test_long_sequence_no_divergence(self):
        """1000 步序列后状态不应发散。"""
        layer = S4Layer(state_dim=64, input_dim=16, output_dim=8, seed=42, dt=0.05)
        rng = np.random.default_rng(0)
        for _ in range(1000):
            u = rng.standard_normal(16) * 0.1
            layer.step(u)
        assert np.all(np.isfinite(layer.state))
        assert np.linalg.norm(layer.state) < 100  # 不应爆炸

    def test_nan_input_does_not_corrupt_state(self):
        """NaN 输入不应污染隐状态。"""
        layer = S4Layer(state_dim=32, input_dim=8, output_dim=4, seed=42)
        # 建立 state
        layer.step(np.ones(8))
        state_before = layer.state.copy()
        # NaN 输入
        layer.step(np.full(8, np.nan))
        # state 应保持（或被 last_valid 恢复）
        assert np.all(np.isfinite(layer.state))


class TestS4SineWavePrediction:
    """验证：S4 对正弦波的预测误差应显著低于固定矩阵。"""

    def test_s4_beats_fixed_matrix_on_sine(self):
        """生成正弦波序列，对比 S4 vs 固定矩阵的预测 MSE。"""
        # 生成正弦波
        T = 200
        t = np.linspace(0, 8 * np.pi, T)
        signal = np.sin(t)
        # 构造输入：当前帧，目标：下一帧
        inputs = np.stack([signal[:-1]], axis=1)  # (T-1, 1)
        targets = signal[1:]  # (T-1,)

        # S4 模型
        s4 = S4Layer(state_dim=32, input_dim=1, output_dim=1, seed=42, dt=0.1, lr=0.05)
        # 在线预测 + 更新
        s4_preds = []
        for k in range(T - 1):
            u = inputs[k]
            y = s4.step(u)
            s4.update(np.array([y[0] - targets[k]]), u)
            s4_preds.append(y[0])
        s4_mse = np.mean((np.array(s4_preds) - targets) ** 2)

        # 固定转移矩阵基线：y = W x, W 固定随机
        rng = np.random.default_rng(42)
        W_fixed = rng.standard_normal((1, 1)) * 0.1
        x_fixed = np.zeros(1)
        fixed_preds = []
        for k in range(T - 1):
            u = inputs[k]
            y = W_fixed @ x_fixed
            x_fixed = u  # 无学习
            fixed_preds.append(y[0])
        fixed_mse = np.mean((np.array(fixed_preds) - targets) ** 2)

        # S4 应显著优于固定矩阵
        assert s4_mse < fixed_mse, f"S4 MSE={s4_mse:.4f} vs Fixed MSE={fixed_mse:.4f}"
