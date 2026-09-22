# src/zero_data_model/cogtime/temporal_memory.py
"""时序记忆：固定随机权重的循环网络用于上下文建模。

本模块为 ZeroDataModel 提供可选的"时序上下文"钩子。采用回声状态网络
（Echo State Network）思想：输入-隐层权重 ``W_in`` 与隐层-隐层权重 ``W_hh``
固定随机（仅对 ``W_hh`` 做谱半径归一化），只在线训练输出层 ``W_out``
（Hebbian / 最小二乘更新）。这样既能为下游模块提供富含时序历史的隐状态
上下文，又避免了全网络训练的开销与不稳定。

设计要点：
- 完全独立、自包含、可插拔；不修改 model.py。
- ``W_in``、``W_hh`` 固定随机；``W_out`` 初始化为零，仅 ``W_out`` 在线学习。
- 所有 numpy 操作做防御性处理（窄异常捕获），形状不匹配/数值异常时返回安全默认值。
- 输入含 NaN/inf 时跳过隐状态更新，返回上次有效上下文，避免单次坏输入
  永久污染 ``hidden``/``W_out``。
- 通过 ``self._lock``（``RLock``）保护 ``hidden``/``W_out`` 的并发访问。
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

_logger = logging.getLogger(__name__)


class TemporalMemory:
    """固定随机权重 RNN 时序记忆（回声状态网络风格）。"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        output_dim: int | None = None,
        seed: int = 42,
    ) -> None:
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        # output_dim 默认与 input_dim 一致
        self.output_dim = (
            int(output_dim) if output_dim is not None else self.input_dim
        )
        self.seed = seed

        rng = np.random.default_rng(seed)

        # 固定随机输入权重： (input_dim, hidden_dim)
        self.W_in = rng.normal(0.0, 0.1, (self.input_dim, self.hidden_dim))
        # 固定随机隐层权重： (hidden_dim, hidden_dim)，并做谱半径归一化
        W_hh = rng.normal(0.0, 0.1, (self.hidden_dim, self.hidden_dim))
        self.W_hh = self._normalize_spectral(W_hh, margin=1.2)
        # P2.11 性能优化: ``W_hh`` 在 ``__init__`` 后不可变（只有 ``W_out`` 与
        # ``hidden`` 在线更新），但 ``spectral_radius`` 属性原本每次调用都做
        # ``np.linalg.eigvals``（O(n^3)），而 think() 每周期都在指标块里读取它。
        # 在此预计算并缓存，``spectral_radius`` 属性直接返回缓存值。若未来
        # ``W_hh`` 可变（如热启动重置），需在写入处调用 ``_recompute_spectral_radius``。
        self._cached_spectral_radius: float = self._compute_spectral_radius()
        # 输出权重：仅此层可学习，初始化为零
        self.W_out = np.zeros(
            (self.hidden_dim, self.output_dim), dtype=float
        )
        # 隐状态
        self.hidden = np.zeros(self.hidden_dim, dtype=float)
        # 上次有效前向输出（context）。输入含 NaN/inf 时返回该缓存，
        # 避免 NaN 永久污染 hidden/W_out。初始化为零向量。
        self._last_valid_context: np.ndarray = np.zeros(
            self.output_dim, dtype=float
        )
        # 可重入锁：保护 hidden / W_out / _last_valid_context 的并发访问。
        # think() 会并发调用各模块，forward/update/get_context/reset/
        # spectral_radius 均需在 self._lock 下访问可变状态。
        self._lock: threading.RLock = threading.RLock()

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播：更新隐状态并返回输出。

        ``hidden = tanh(x @ W_in + hidden @ W_hh)``，
        ``output = hidden @ W_out``。

        - 输入含 NaN/inf 时**不更新** ``hidden``，直接返回上次有效上下文
          ``self._last_valid_context``，防止单次坏输入永久污染隐状态。
        - 先将新隐状态计算到局部变量，全部成功后再赋值给 ``self.hidden``，
          避免异常导致部分更新。
        - 仅捕获 ``(ValueError, FloatingPointError, OverflowError)``，
          其它异常向上抛出（不静默吞）。
        """
        with self._lock:
            try:
                xx = self._fit(x, self.input_dim)
                # 输入校验：含 NaN/inf 则跳过 hidden 更新，返回上次有效上下文
                if not np.isfinite(xx).all():
                    return self._last_valid_context.copy()
                # 先计算新隐状态到局部变量，全部成功后再提交
                new_hidden = np.tanh(xx @ self.W_in + self.hidden @ self.W_hh)
                output = new_hidden @ self.W_out
                # 提交隐状态更新
                self.hidden = new_hidden
                # 缓存本次有效输出作为下次的"上次有效上下文"
                self._last_valid_context = output.copy()
                return output
            except (ValueError, FloatingPointError, OverflowError) as exc:
                _logger.warning("TemporalMemory.forward failed: %r", exc)
                return np.zeros(self.output_dim, dtype=float)

    def update(
        self, x: np.ndarray, target: np.ndarray, lr: float = 0.01
    ) -> float:
        """在线更新 ``W_out``（Hebbian / 最小二乘）并返回预测 MSE。

        先做一次前向传播（更新隐状态），再计算输出误差，仅更新 ``W_out``：
        ``W_out += lr * outer(hidden, target - output)``。

        - 若计算出的 ``mse`` 非有限（NaN/inf），**不更新** ``W_out``、
          **不更新** ``_last_valid_context``，返回 0.0，避免坏样本污染权重。
        - 先将新 ``W_out`` 计算到局部变量，全部成功后再赋值，避免部分更新。
        - 仅捕获 ``(ValueError, FloatingPointError, OverflowError)``。
        """
        with self._lock:
            try:
                output = self.forward(x)
                tt = self._fit(target, self.output_dim)
                error = tt - output
                # 先计算 mse：非有限则跳过 W_out 更新
                mse = float(np.mean(error * error)) if error.size else 0.0
                if not np.isfinite(mse):
                    # 坏样本：不更新 W_out，不更新 _last_valid_context，
                    # 返回 0.0（安全默认，避免 NaN 向上传播）
                    return 0.0
                # 先计算新 W_out 到局部变量，成功后再提交
                new_w_out = self.W_out + lr * np.outer(self.hidden, error)
                self.W_out = new_w_out
                return mse
            except (ValueError, FloatingPointError, OverflowError) as exc:
                _logger.warning("TemporalMemory.update failed: %r", exc)
                return 0.0

    def get_context(self) -> np.ndarray:
        """返回当前隐状态的拷贝（用于拼接为下游模块的额外输入）。"""
        with self._lock:
            try:
                return self.hidden.copy()
            except (ValueError, FloatingPointError, OverflowError) as exc:
                _logger.warning("TemporalMemory.get_context failed: %r", exc)
                return np.zeros(self.hidden_dim, dtype=float)

    def reset(self) -> None:
        """清零隐状态与上次有效上下文缓存。"""
        with self._lock:
            self.hidden = np.zeros(self.hidden_dim, dtype=float)
            self._last_valid_context = np.zeros(self.output_dim, dtype=float)

    @property
    def spectral_radius(self) -> float:
        """``W_hh`` 的谱半径（最大特征值模长）。

        P2.11 性能优化: ``W_hh`` 在 ``__init__`` 后不可变，谱半径在构造时
        预计算并缓存（``_cached_spectral_radius``）。原实现每次调用都做
        ``np.linalg.eigvals``（O(n^3)），而 think() 每周期都在指标块读取它，
        造成冗余的 O(n^3) 开销。现在直接返回缓存值，O(1)。
        """
        with self._lock:
            return self._cached_spectral_radius

    def _compute_spectral_radius(self) -> float:
        """计算 ``W_hh`` 的谱半径（供构造时缓存与未来重算使用）。"""
        try:
            eigvals = np.linalg.eigvals(self.W_hh)
            return float(np.max(np.abs(eigvals)))
        except (ValueError, FloatingPointError, OverflowError) as exc:
            _logger.warning(
                "TemporalMemory.spectral_radius failed: %r", exc
            )
            return 0.0

    def _recompute_spectral_radius(self) -> None:
        """重算并刷新缓存的谱半径。

        供未来 ``W_hh`` 写入路径调用（当前 ``W_hh`` 不可变，无需调用）。
        """
        with self._lock:
            self._cached_spectral_radius = self._compute_spectral_radius()

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_spectral(
        W: np.ndarray, margin: float = 1.2
    ) -> np.ndarray:
        """对 ``W_hh`` 做谱半径归一化：除以 (最大特征值模 * margin)。

        归一化后谱半径约为 ``1 / margin``，保证循环动力学稳定（margin 默认
        1.2，对应目标谱半径约 0.833）。最大特征值模过小或计算异常时返回原矩阵。
        """
        try:
            eigvals = np.linalg.eigvals(W)
            max_abs = float(np.max(np.abs(eigvals)))
            if max_abs < 1e-12 or not np.isfinite(max_abs):
                return W
            return W / (max_abs * margin)
        except (ValueError, FloatingPointError, OverflowError):
            return W

    @staticmethod
    def _fit(arr: Any, dim: int) -> np.ndarray:
        """将输入适配到指定维度（过长截断、过短补零）。"""
        a = np.asarray(arr, dtype=float).flatten()
        if a.size < dim:
            a = np.pad(a, (0, dim - a.size))
        elif a.size > dim:
            a = a[:dim]
        return a.astype(float)
