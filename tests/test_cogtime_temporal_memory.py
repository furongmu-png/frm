from __future__ import annotations

from threading import Thread

import numpy as np
import pytest

from zero_data_model.cogtime.temporal_memory import TemporalMemory


# --------------------------------------------------------------------------- #
# 构造
# --------------------------------------------------------------------------- #


def test_construct_default():
    tm = TemporalMemory(input_dim=8)
    assert tm.input_dim == 8
    assert tm.hidden_dim == 32
    assert tm.output_dim == 8
    assert tm.hidden.shape == (32,)
    assert np.all(tm.hidden == 0.0)
    assert tm.W_out.shape == (32, 8)
    assert np.all(tm.W_out == 0.0)


def test_construct_with_output_dim():
    tm = TemporalMemory(input_dim=4, hidden_dim=16, output_dim=2, seed=7)
    assert tm.output_dim == 2
    assert tm.W_out.shape == (16, 2)


def test_spectral_radius_property_returns_finite():
    """spectral_radius 属性应返回有限正值。"""
    tm = TemporalMemory(input_dim=4, hidden_dim=8, seed=42)
    sr = tm.spectral_radius
    assert np.isfinite(sr)
    # W_hh 经过归一化，目标谱半径 ≈ 1/1.2 ≈ 0.833
    assert 0.0 < sr < 1.5


# --------------------------------------------------------------------------- #
# forward / update / get_context / reset
# --------------------------------------------------------------------------- #


def test_forward_returns_output_dim():
    tm = TemporalMemory(input_dim=4, hidden_dim=8, output_dim=4, seed=42)
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 4)
    out = tm.forward(x)
    assert out.shape == (4,)


def test_forward_updates_hidden():
    tm = TemporalMemory(input_dim=4, hidden_dim=8, output_dim=4, seed=42)
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 4)
    hidden_before = tm.hidden.copy()
    tm.forward(x)
    assert not np.allclose(hidden_before, tm.hidden)


def test_forward_short_input_pads_zeros():
    """输入过短会被 _fit 补零到 input_dim。"""
    tm = TemporalMemory(input_dim=4, hidden_dim=8, output_dim=4, seed=42)
    out = tm.forward(np.array([1.0, 2.0]))  # 长度 2 → 补零到 4
    assert out.shape == (4,)
    assert np.isfinite(out).all()


def test_forward_long_input_truncates():
    """输入过长会被截断到 input_dim。"""
    tm = TemporalMemory(input_dim=4, hidden_dim=8, output_dim=4, seed=42)
    out = tm.forward(np.arange(10, dtype=float))
    assert out.shape == (4,)


def test_update_returns_mse_and_finite():
    tm = TemporalMemory(input_dim=4, hidden_dim=8, output_dim=4, seed=42)
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 4)
    target = rng.normal(0, 1, 4)
    mse = tm.update(x, target, lr=0.01)
    assert np.isfinite(mse)
    assert mse >= 0.0


def test_get_context_returns_hidden_copy():
    """get_context 返回拷贝，修改不影响内部 hidden。"""
    tm = TemporalMemory(input_dim=4, hidden_dim=8, output_dim=4, seed=42)
    rng = np.random.default_rng(0)
    tm.forward(rng.normal(0, 1, 4))
    ctx = tm.get_context()
    assert ctx.shape == (8,)
    ctx[0] = 999.0
    # 内部 hidden 不应被修改
    assert tm.hidden[0] != 999.0


def test_reset_zeros_state():
    tm = TemporalMemory(input_dim=4, hidden_dim=8, output_dim=4, seed=42)
    rng = np.random.default_rng(0)
    tm.forward(rng.normal(0, 1, 4))
    tm.reset()
    assert np.all(tm.hidden == 0.0)


# --------------------------------------------------------------------------- #
# 军事级修复点
# --------------------------------------------------------------------------- #


def test_nan_input_does_not_pollute_hidden():
    """核心修复点：forward 一个 NaN 输入，再 forward 正常输入，hidden 应保持有限。"""
    tm = TemporalMemory(input_dim=4, hidden_dim=8, output_dim=4, seed=42)
    rng = np.random.default_rng(0)
    # 先用正常输入初始化 hidden
    tm.forward(rng.normal(0, 1, 4))
    hidden_before = tm.hidden.copy()
    # 输入 NaN
    bad = np.array([np.nan, 0.0, 0.0, 0.0])
    out = tm.forward(bad)
    # NaN 输入 → 返回 last_valid_context（有限），hidden 不变
    assert np.isfinite(out).all()
    assert np.allclose(hidden_before, tm.hidden)
    # 再 forward 正常输入 → hidden 应被更新且有限
    tm.forward(rng.normal(0, 1, 4))
    assert np.isfinite(tm.hidden).all()


def test_inf_input_does_not_pollute_hidden():
    """inf 输入同样不污染 hidden。"""
    tm = TemporalMemory(input_dim=4, hidden_dim=8, output_dim=4, seed=42)
    rng = np.random.default_rng(0)
    tm.forward(rng.normal(0, 1, 4))
    hidden_before = tm.hidden.copy()
    bad = np.array([np.inf, 0.0, 0.0, 0.0])
    out = tm.forward(bad)
    assert np.isfinite(out).all()
    assert np.allclose(hidden_before, tm.hidden)


def test_update_with_nan_target_skips_w_out_update():
    """更新含 NaN target 时不更新 W_out，返回 0.0。"""
    tm = TemporalMemory(input_dim=4, hidden_dim=8, output_dim=4, seed=42)
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 4)
    tm.forward(x)
    w_out_before = tm.W_out.copy()
    bad_target = np.array([np.nan, 0.0, 0.0, 0.0])
    mse = tm.update(x, bad_target)
    # 0.0（不更新）或非有限值都触发跳过 → 应不更新 W_out
    assert mse == 0.0 or not np.isfinite(mse)
    # W_out 未被污染
    assert np.allclose(w_out_before, tm.W_out)


def test_concurrent_forward_does_not_crash():
    """线程安全：并发 forward 不抛异常，hidden 最终保持有限。"""
    tm = TemporalMemory(input_dim=4, hidden_dim=8, output_dim=4, seed=42)
    rng = np.random.default_rng(0)
    xs = [rng.normal(0, 1, 4) for _ in range(50)]

    def worker() -> None:
        for x in xs:
            tm.forward(x)

    threads = [Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert np.isfinite(tm.hidden).all()


def test_concurrent_update_does_not_crash():
    """线程安全：并发 update 不抛异常。"""
    tm = TemporalMemory(input_dim=4, hidden_dim=8, output_dim=4, seed=42)
    rng = np.random.default_rng(0)
    pairs = [(rng.normal(0, 1, 4), rng.normal(0, 1, 4)) for _ in range(50)]

    def worker() -> None:
        for x, t in pairs:
            tm.update(x, t, lr=0.01)

    threads = [Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert np.isfinite(tm.W_out).all()
    assert np.isfinite(tm.hidden).all()
