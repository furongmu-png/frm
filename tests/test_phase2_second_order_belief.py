# tests/test_phase2_second_order_belief.py
"""第二阶段 §2.1 二阶信念系统单元测试。

验证：
- 高斯分布维护（均值=信念，方差=不确定度）
- 不确定度三项来源（预测误差、参数更新、记忆相似度）
- predicted_uncertainty 输出
- 已知环境 vs 未知环境的置信度差异
- 高斯采样
"""
from __future__ import annotations

import numpy as np
import pytest

from zero_data_model.metacog.second_order_belief import SecondOrderBelief


# ------------------------------------------------------------------ #
# 初始化
# ------------------------------------------------------------------ #
class TestInit:
    """测试初始化。"""

    def test_default_init(self) -> None:
        sob = SecondOrderBelief(dim=16, seed=42)
        assert sob.belief_mean.shape == (16,)
        assert sob.uncertainty_var.shape == (16,)
        assert 0.0 <= sob.predicted_uncertainty <= 1.0

    def test_invalid_dim_raises(self) -> None:
        with pytest.raises(ValueError):
            SecondOrderBelief(dim=0)

    def test_custom_weights(self) -> None:
        sob = SecondOrderBelief(
            dim=8, error_weight=0.6, param_weight=0.3, memory_weight=0.1
        )
        result = sob.update(np.zeros(8), 0.5, 0.5, 0.5)
        assert "decomposition" in result


# ------------------------------------------------------------------ #
# 不确定度更新
# ------------------------------------------------------------------ #
class TestUncertaintyUpdate:
    """测试不确定度更新机制。"""

    def test_high_error_increases_uncertainty(self) -> None:
        sob = SecondOrderBelief(dim=8, ema_alpha=0.3, seed=42)
        # Low error first
        for _ in range(20):
            sob.update(np.zeros(8), 0.01, 0.0, 1.0)
        unc_low = sob.predicted_uncertainty
        # High error
        for _ in range(20):
            sob.update(np.zeros(8), 10.0, 0.0, 1.0)
        unc_high = sob.predicted_uncertainty
        assert unc_high > unc_low

    def test_low_memory_similarity_increases_uncertainty(self) -> None:
        sob = SecondOrderBelief(dim=8, ema_alpha=0.3, seed=42)
        # High memory similarity (found match)
        for _ in range(20):
            sob.update(np.zeros(8), 0.1, 0.0, 0.95)
        unc_matched = sob.predicted_uncertainty
        # Low memory similarity (no match)
        for _ in range(20):
            sob.update(np.zeros(8), 0.1, 0.0, 0.05)
        unc_unmatched = sob.predicted_uncertainty
        assert unc_unmatched > unc_matched

    def test_large_param_update_increases_uncertainty(self) -> None:
        sob = SecondOrderBelief(dim=8, ema_alpha=0.3, seed=42)
        for _ in range(20):
            sob.update(np.zeros(8), 0.1, 0.01, 1.0)
        unc_stable = sob.predicted_uncertainty
        for _ in range(20):
            sob.update(np.zeros(8), 0.1, 5.0, 1.0)
        unc_unstable = sob.predicted_uncertainty
        assert unc_unstable > unc_stable

    def test_uncertainty_in_range(self) -> None:
        sob = SecondOrderBelief(dim=8, ema_alpha=0.5, seed=42)
        rng = np.random.default_rng(0)
        for _ in range(100):
            err = float(rng.uniform(0, 100))
            mag = float(rng.uniform(0, 10))
            sim = float(rng.uniform(0, 1))
            sob.update(rng.standard_normal(8), err, mag, sim)
            assert 0.0 <= sob.predicted_uncertainty <= 1.0
            assert 0.0 <= sob.get_confidence() <= 1.0


# ------------------------------------------------------------------ #
# 已知 vs 未知环境
# ------------------------------------------------------------------ #
class TestKnownVsUnknownEnv:
    """验证：模型进入全新环境时置信度显著降低。"""

    def test_confidence_drops_in_new_env(self) -> None:
        sob = SecondOrderBelief(dim=16, ema_alpha=0.2, seed=42)
        rng = np.random.default_rng(42)
        # Known environment: low error, high memory match
        for _ in range(50):
            sob.update(
                rng.standard_normal(16) * 0.1,
                prediction_error=0.1,
                param_update_norm=0.01,
                memory_similarity=0.9,
            )
        conf_known = sob.get_confidence()
        # New environment: high error, no memory match
        for _ in range(10):
            sob.update(
                rng.standard_normal(16) * 2.0,
                prediction_error=5.0,
                param_update_norm=2.0,
                memory_similarity=0.1,
            )
        conf_new = sob.get_confidence()
        assert conf_new < conf_known
        assert conf_new < 0.7  # significant drop

    def test_confidence_recovers_with_learning(self) -> None:
        sob = SecondOrderBelief(dim=16, ema_alpha=0.2, seed=42)
        rng = np.random.default_rng(42)
        # New environment
        for _ in range(10):
            sob.update(rng.standard_normal(16), 5.0, 2.0, 0.1)
        conf_new = sob.get_confidence()
        # Learn the new environment
        for _ in range(50):
            sob.update(rng.standard_normal(16) * 0.3, 0.3, 0.1, 0.8)
        conf_recovered = sob.get_confidence()
        assert conf_recovered > conf_new


# ------------------------------------------------------------------ #
# predicted_uncertainty 输出
# ------------------------------------------------------------------ #
class TestPredictedUncertainty:
    """测试 predicted_uncertainty 输出。"""

    def test_predicted_uncertainty_property(self) -> None:
        sob = SecondOrderBelief(dim=8, seed=42)
        assert hasattr(sob, "predicted_uncertainty")
        assert isinstance(sob.predicted_uncertainty, float)

    def test_update_returns_predicted_uncertainty(self) -> None:
        sob = SecondOrderBelief(dim=8, seed=42)
        result = sob.update(np.zeros(8), 0.5, 0.5, 0.5)
        assert "predicted_uncertainty" in result
        assert "confidence" in result
        assert "mean_uncertainty" in result


# ------------------------------------------------------------------ #
# 高斯采样
# ------------------------------------------------------------------ #
class TestGaussianSampling:
    """测试高斯采样。"""

    def test_sample_shape(self) -> None:
        sob = SecondOrderBelief(dim=16, seed=42)
        samples = sob.sample_belief(n_samples=10)
        assert samples.shape == (10, 16)

    def test_sample_mean_close_to_belief(self) -> None:
        sob = SecondOrderBelief(dim=16, seed=42)
        belief = np.ones(16) * 0.5
        for _ in range(100):
            sob.update(belief, 0.01, 0.0, 1.0)
        # With low uncertainty, samples should cluster around belief
        samples = sob.sample_belief(n_samples=1000)
        sample_mean = samples.mean(axis=0)
        assert np.allclose(sample_mean, belief, atol=0.1)


# ------------------------------------------------------------------ #
# 分解与历史
# ------------------------------------------------------------------ #
class TestDecomposition:
    """测试不确定度来源分解。"""

    def test_decomposition_keys(self) -> None:
        sob = SecondOrderBelief(dim=8, seed=42)
        sob.update(np.zeros(8), 0.5, 0.5, 0.5)
        decomp = sob.get_decomposition()
        assert "error_term" in decomp
        assert "param_term" in decomp
        assert "memory_term" in decomp

    def test_confidence_history(self) -> None:
        sob = SecondOrderBelief(dim=8, seed=42)
        for _ in range(10):
            sob.update(np.zeros(8), 0.5, 0.5, 0.5)
        hist = sob.get_confidence_history()
        assert len(hist) == 10

    def test_stats(self) -> None:
        sob = SecondOrderBelief(dim=8, seed=42)
        sob.update(np.zeros(8), 0.5, 0.5, 0.5)
        s = sob.stats
        assert s["dim"] == 8
        assert s["step_count"] == 1
        assert "predicted_uncertainty" in s
        assert "confidence" in s
