from __future__ import annotations

from threading import Thread

import numpy as np
import pytest

from zero_data_model.cogtime.layered_predictor import LayeredPredictor


# --------------------------------------------------------------------------- #
# 构造
# --------------------------------------------------------------------------- #


def test_construct_default():
    lp = LayeredPredictor()
    assert lp.dim_l0 == 64
    assert lp.dim_l1 == 32
    assert lp.dim_l2 == 16
    assert lp.l1_interval == 10
    assert lp.l2_interval == 100
    assert lp.l0.dim == 64
    assert lp.l1.dim == 32
    assert lp.l2.dim == 16
    # 三个层独立 rng
    assert lp.l0.rng is not lp.l1.rng
    assert lp.l1.rng is not lp.l2.rng


def test_construct_with_args():
    lp = LayeredPredictor(
        dim_l0=8, dim_l1=4, dim_l2=2, seed=42, l1_interval=2, l2_interval=4
    )
    assert lp.dim_l0 == 8
    assert lp.dim_l1 == 4
    assert lp.dim_l2 == 2
    assert lp.l1_interval == 2
    assert lp.l2_interval == 4


# --------------------------------------------------------------------------- #
# update / get_context / predict_rhythm
# --------------------------------------------------------------------------- #


def test_update_returns_dict_with_expected_keys():
    lp = LayeredPredictor(dim_l0=8, dim_l1=4, dim_l2=2, seed=42,
                          l1_interval=2, l2_interval=4)
    rng = np.random.default_rng(0)
    obs = rng.normal(0, 1, 8)
    result = lp.update(obs, step=0)
    assert "l0_error" in result
    assert "l1_error" in result
    assert "l2_error" in result
    assert "l2_belief_norm" in result
    # step=0 → 触发 L1 与 L2 初始化脉冲
    assert result["l1_error"] is not None
    assert result["l2_error"] is not None


def test_update_l1_only_on_interval():
    lp = LayeredPredictor(dim_l0=8, dim_l1=4, dim_l2=2, seed=42,
                          l1_interval=2, l2_interval=100)
    rng = np.random.default_rng(0)
    # step=1：不触发 L1
    r = lp.update(rng.normal(0, 1, 8), step=1)
    assert r["l1_error"] is None
    assert r["l2_error"] is None
    # step=2：触发 L1
    r = lp.update(rng.normal(0, 1, 8), step=2)
    assert r["l1_error"] is not None


def test_get_context_shape():
    lp = LayeredPredictor(dim_l0=8, dim_l1=4, dim_l2=2, seed=42)
    rng = np.random.default_rng(0)
    lp.update(rng.normal(0, 1, 8), step=0)
    ctx = lp.get_context()
    # 长度 = dim_l0 + dim_l1 + dim_l2
    assert ctx.shape == (8 + 4 + 2,)


def test_get_context_finite_after_updates():
    """多次 update 后 get_context 全部有限。"""
    lp = LayeredPredictor(dim_l0=8, dim_l1=4, dim_l2=2, seed=42,
                          l1_interval=2, l2_interval=4)
    rng = np.random.default_rng(0)
    for step in range(20):
        lp.update(rng.normal(0, 1, 8), step=step)
    ctx = lp.get_context()
    assert np.isfinite(ctx).all()


def test_predict_rhythm_returns_finite():
    """predict_rhythm 返回有限值。"""
    lp = LayeredPredictor(dim_l0=8, dim_l1=4, dim_l2=2, seed=42,
                          l1_interval=2, l2_interval=2)
    rng = np.random.default_rng(0)
    for step in range(10):
        lp.update(rng.normal(0, 1, 8), step=step)
    r = lp.predict_rhythm()
    assert np.isfinite(r)
    assert 0.0 <= r <= 1.0


# --------------------------------------------------------------------------- #
# 军事级修复点
# --------------------------------------------------------------------------- #


def test_update_with_nan_observation_returns_safe_default():
    """NaN 钳制：update 后 get_context 返回值全部有限。"""
    lp = LayeredPredictor(dim_l0=8, dim_l1=4, dim_l2=2, seed=42,
                          l1_interval=2, l2_interval=2)
    bad = np.array([np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # 不应抛异常
    result = lp.update(bad, step=0)
    assert "l0_error" in result
    # 即使输入 NaN，get_context 也必须返回有限值
    ctx = lp.get_context()
    assert np.isfinite(ctx).all()


def test_get_context_clamps_nan_to_zero():
    """显式注入 NaN 到 belief，get_context 应通过 nan_to_num 钳为 0。"""
    lp = LayeredPredictor(dim_l0=8, dim_l1=4, dim_l2=2, seed=42)
    # 直接污染 belief 模拟异常状态
    lp.l0.belief[:] = np.nan
    lp.l1.belief[:] = np.inf
    lp.l2.belief[:] = -np.inf
    ctx = lp.get_context()
    assert np.isfinite(ctx).all()
    assert np.all(ctx == 0.0)


def test_spectral_radius_normalized_below_one_plus_margin():
    """谱半径归一化：transition 谱半径 <= 1.0 + margin（约 0.833 目标）。"""
    lp = LayeredPredictor(dim_l0=8, dim_l1=4, dim_l2=2, seed=42)
    for layer in (lp.l0, lp.l1, lp.l2):
        eigvals = np.linalg.eigvals(layer.transition)
        sr = float(np.max(np.abs(eigvals)))
        # 目标 ≈ 1/1.2 ≈ 0.833，留少量数值容差
        assert sr <= 1.0 + 1e-6


def test_update_clamps_non_finite_belief():
    """belief 更新后若含 NaN/inf 应被重置为零。"""
    lp = LayeredPredictor(dim_l0=8, dim_l1=4, dim_l2=2, seed=42,
                          l1_interval=2, l2_interval=2)
    # 把 transition 设为全 NaN → 模拟 update_layer 内部异常
    lp.l0.transition[:] = np.nan
    rng = np.random.default_rng(0)
    # 不应抛异常
    lp.update(rng.normal(0, 1, 8), step=0)
    # l0.belief 应被重置为 0（含 NaN → 重置）
    ctx = lp.get_context()
    assert np.isfinite(ctx).all()


def test_error_history_deque_maxlen():
    """deque maxlen 限长：error_history 上限 200。"""
    lp = LayeredPredictor(dim_l0=8, dim_l1=4, dim_l2=2, seed=42,
                          l1_interval=1, l2_interval=1)
    rng = np.random.default_rng(0)
    for step in range(500):
        lp.update(rng.normal(0, 1, 8), step=step)
    # 每层 error_history maxlen=200
    assert len(lp.l0.error_history) == 200
    assert len(lp.l1.error_history) <= 200
    assert len(lp.l2.error_history) <= 200


def test_concurrent_update_does_not_crash():
    """线程安全：并发 update 不抛异常。"""
    lp = LayeredPredictor(dim_l0=8, dim_l1=4, dim_l2=2, seed=42,
                          l1_interval=2, l2_interval=4)
    rng = np.random.default_rng(0)
    obs_list = [rng.normal(0, 1, 8) for _ in range(50)]

    def worker() -> None:
        for i, obs in enumerate(obs_list):
            lp.update(obs, step=i)

    threads = [Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # 最终 context 应有限
    assert np.isfinite(lp.get_context()).all()


def test_concurrent_get_context_does_not_crash():
    """线程安全：并发 get_context 不抛异常。"""
    lp = LayeredPredictor(dim_l0=8, dim_l1=4, dim_l2=2, seed=42,
                          l1_interval=2, l2_interval=4)
    rng = np.random.default_rng(0)
    obs_list = [rng.normal(0, 1, 8) for _ in range(20)]

    def updater() -> None:
        for i, obs in enumerate(obs_list):
            lp.update(obs, step=i)

    def reader() -> None:
        for _ in range(20):
            ctx = lp.get_context()
            assert ctx.shape == (14,)

    threads = [Thread(target=updater)]
    threads.extend(Thread(target=reader) for _ in range(3))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
