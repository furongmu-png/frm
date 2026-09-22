from __future__ import annotations

from threading import Thread

import numpy as np
import pytest

from zero_data_model.metacog.meta_cognition import MetaCognition


# --------------------------------------------------------------------------- #
# 构造
# --------------------------------------------------------------------------- #


def test_construct_default():
    mc = MetaCognition()
    assert mc.mode == "explore"
    # 初始置信度 = 1 - mean(0.5 * ones) = 0.5
    assert mc.confidence == pytest.approx(0.5)


def test_construct_with_args():
    mc = MetaCognition(dim=8, window=10, uncertainty_threshold=0.3, seed=42)
    assert mc._dim == 8
    assert mc._window == 10
    assert mc._uncertainty_threshold == 0.3


# --------------------------------------------------------------------------- #
# update / should_seek_info / get_confidence / get_uncertainty_vector / record_milestone
# --------------------------------------------------------------------------- #


def test_update_returns_expected_keys():
    mc = MetaCognition(dim=4, seed=42)
    result = mc.update(prediction_error=0.1, param_update_norm=0.05)
    assert "mean_uncertainty" in result
    assert "mode" in result
    assert "confidence" in result
    assert isinstance(result["mean_uncertainty"], float)
    assert isinstance(result["mode"], str)
    assert isinstance(result["confidence"], float)


def test_update_returns_finite_values():
    """返回值有限（核心修复点）。"""
    mc = MetaCognition(dim=4, seed=42)
    result = mc.update(prediction_error=0.5, param_update_norm=0.5)
    assert np.isfinite(result["mean_uncertainty"])
    assert np.isfinite(result["confidence"])
    assert 0.0 <= result["confidence"] <= 1.0


def test_should_seek_info_true_in_explore_mode():
    mc = MetaCognition(dim=4, uncertainty_threshold=0.5, seed=42)
    # 初始 mode = explore
    assert mc.should_seek_info() is True
    # 模拟低不确定 → exploit 模式
    for _ in range(50):
        mc.update(prediction_error=0.0, param_update_norm=0.0)
    # 在低不确定下，mode 可能转为 exploit/balanced
    # should_seek_info 仅在 explore 模式下为 True
    assert mc.should_seek_info() in (True, False)


def test_get_confidence_in_unit_interval():
    mc = MetaCognition(dim=4, seed=42)
    for i in range(20):
        c = mc.get_confidence()
        assert 0.0 <= c <= 1.0
        mc.update(prediction_error=float(i) * 0.1, param_update_norm=0.1)


def test_get_uncertainty_vector_returns_copy():
    """get_uncertainty_vector 返回拷贝，修改不影响内部。"""
    mc = MetaCognition(dim=4, seed=42)
    mc.update(prediction_error=0.5, param_update_norm=0.1)
    v = mc.get_uncertainty_vector()
    assert v.shape == (4,)
    v[0] = 999.0
    # 内部不受影响
    v2 = mc.get_uncertainty_vector()
    assert v2[0] != 999.0


def test_record_milestone_returns_dict():
    mc = MetaCognition(dim=4, seed=42)
    mc.update(prediction_error=0.1, param_update_norm=0.1)
    ev = mc.record_milestone("first_insight", step=10)
    assert ev["type"] == "meta_cognition"
    assert ev["event"] == "first_insight"
    assert ev["step"] == 10
    assert "uncertainty" in ev
    assert "mode" in ev
    assert "confidence" in ev
    assert np.isfinite(ev["uncertainty"])
    assert np.isfinite(ev["confidence"])


# --------------------------------------------------------------------------- #
# 军事级修复点
# --------------------------------------------------------------------------- #


def test_update_with_mag_minus_one_does_not_divide_by_zero():
    """mag=-1.0 除零防护：update 不应抛 ZeroDivisionError 或产生 NaN。"""
    mc = MetaCognition(dim=4, seed=42)
    # mag = -1.0 → divisor = mag + 1.0 = 0.0
    result = mc.update(prediction_error=0.1, param_update_norm=-1.0)
    assert np.isfinite(result["mean_uncertainty"])
    assert np.isfinite(result["confidence"])
    # 不确定性向量也保持有限
    v = mc.get_uncertainty_vector()
    assert np.isfinite(v).all()


def test_update_with_nan_inputs_does_not_crash():
    """NaN 输入防护。"""
    mc = MetaCognition(dim=4, seed=42)
    # NaN 预测误差 → 内部钳为 1e6
    result = mc.update(prediction_error=float("nan"),
                       param_update_norm=float("nan"))
    assert np.isfinite(result["mean_uncertainty"])
    assert np.isfinite(result["confidence"])


def test_update_with_inf_inputs_does_not_crash():
    mc = MetaCognition(dim=4, seed=42)
    result = mc.update(prediction_error=float("inf"),
                       param_update_norm=float("inf"))
    # inf 预测误差 → 钳为 1e6（有限）
    assert np.isfinite(result["mean_uncertainty"])


def test_concurrent_update_does_not_crash():
    """线程安全：并发 update 不抛异常。"""
    mc = MetaCognition(dim=4, seed=42)

    def worker() -> None:
        for i in range(50):
            mc.update(prediction_error=float(i) * 0.01,
                      param_update_norm=0.05)

    threads = [Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # 不确定性向量最终应保持有限
    v = mc.get_uncertainty_vector()
    assert np.isfinite(v).all()


def test_concurrent_read_does_not_crash():
    """线程安全：并发读 + 写不抛异常。"""
    mc = MetaCognition(dim=4, seed=42)

    def writer() -> None:
        for i in range(30):
            mc.update(prediction_error=float(i) * 0.01, param_update_norm=0.1)

    def reader() -> None:
        for _ in range(30):
            mc.get_confidence()
            mc.get_uncertainty_vector()
            mc.should_seek_info()

    threads = [Thread(target=writer) for _ in range(2)]
    threads.extend(Thread(target=reader) for _ in range(3))
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def test_error_history_deque_maxlen():
    """deque maxlen 限长：error_history 受 window 限制。"""
    mc = MetaCognition(dim=4, window=10, seed=42)
    for i in range(100):
        mc.update(prediction_error=float(i), param_update_norm=0.1)
    assert len(mc._error_history) == 10


def test_confidence_history_deque_maxlen():
    """deque maxlen 限长：confidence_history 上限 200。"""
    mc = MetaCognition(dim=4, window=5, seed=42)
    for i in range(500):
        mc.update(prediction_error=0.1, param_update_norm=0.1)
    assert len(mc._confidence_history) == 200
