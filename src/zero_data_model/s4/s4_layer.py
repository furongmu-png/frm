"""S4 (Structured State Space) layer — Diagonal State Space (DSS) variant.

理论基础
========

S4 模型 (Gu et al. 2022, "Efficiently Modeling Long Sequences with
Structured State Spaces") 用线性时不变状态空间方程建模长序列：

    dx(t)/dt = A x(t) + B u(t)              (状态方程)
    y(t)     = C x(t) + D u(t)              (观测方程)

其中 x(t) ∈ R^N 是隐状态，u(t) ∈ R^U 是输入，y(t) ∈ R^V 是输出。

连续 → 离散
----------
采用零阶保持 (ZOH) 离散化，步长 Δ：
    A_bar = exp(A Δ)
    B_bar = A^{-1} (A_bar - I) B
    x_{k+1} = A_bar x_k + B_bar u_k
    y_k     = C x_k + D u_k

DSS 简化
--------
完整 S4 用 HiPPO-LegS 矩阵（稠密），DSS (Diagonal State Space, Gupta et al.
2022) 把 A 限制为对角矩阵。对角化后：
    A = diag(a_1, ..., a_N)
    exp(A Δ) = diag(exp(a_1 Δ), ..., exp(a_N Δ))  (element-wise)

HiPPO-LegS 对角初始化：a_n ≈ -n (主对角) + 小幅随机扰动。

Hebbian 更新（无反向传播）
--------------------------
传统 S4 用 BPTT 训练，但本项目无 PyTorch。采用预测误差驱动的局部
Hebbian 规则：
    err = y_pred - y_target
    ΔC = -η * err * x^T           (输出权重 Hebbian)
    ΔB = -η * x * u^T             (输入权重 anti-Hebbian，减少预测误差)
    A 不更新（保持稳定性，HiPPO 矩阵有理论保证）

这与项目现有 `topos.classifier += noise` 的局部更新风格一致。
"""
from __future__ import annotations

import threading

import numpy as np


class S4Layer:
    """Diagonal State Space (DSS) layer with HiPPO initialization.

    用于替代 ActiveInferenceEngine 的固定转移矩阵。支持：
    - forward(u): 处理整段序列（卷积模式批量计算）
    - step(u): 单步在线更新（递归模式，用于 think() 循环）
    - update(error): 预测误差驱动的局部 Hebbian 权重更新

    Parameters
    ----------
    state_dim : int
        隐状态维度 N（建议 dim 的 2 倍，因 S4 状态维度通常 > 输入维度）
    input_dim : int
        输入维度 U
    output_dim : int
        输出维度 V
    dt : float
        离散化步长 Δ（默认 0.1，决定记忆时间尺度：大 Δ 短记忆，小 Δ 长记忆）
    lr : float
        Hebbian 学习率
    seed : int | None
        随机种子（用于可复现的 B, C 初始化）
    """

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        output_dim: int,
        *,
        dt: float = 0.1,
        lr: float = 0.01,
        seed: int | None = None,
    ):
        if state_dim < 1:
            raise ValueError(f"state_dim must be >= 1, got {state_dim}")
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        if output_dim < 1:
            raise ValueError(f"output_dim must be >= 1, got {output_dim}")
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")

        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dt = dt
        self.lr = lr

        self._rng = np.random.default_rng(seed)
        self._lock = threading.RLock()

        # HiPPO-LegS 对角初始化：a_n = -n + 小扰动
        # 主对角 -1, -2, ..., -N 保证指数衰减（稳定性）
        # 加小扰动打破对称性（避免所有 mode 同步衰减）
        perturbation = self._rng.standard_normal(state_dim) * 0.01
        self._A_diag = -(np.arange(1, state_dim + 1, dtype=np.float64)) + perturbation

        # 离散化：A_bar = exp(A * dt)，对角 element-wise
        self._A_bar = np.exp(self._A_diag * dt)

        # B_bar = A^{-1} (A_bar - I) B_continuous
        # 对角情况下 A^{-1} = diag(1/a_1, ..., 1/a_N)
        # B 是 (state_dim, input_dim)，fan_in = input_dim。用 Kaiming fan-in
        # 缩放（与 C 的 Xavier 一致），而非 fan-out（state_dim）——否则
        # ZOH scale (exp(a*dt)-1)/a 会把 B_bar 压到极小，隐状态幅值不足，
        # Hebbian 学习的有效步长 lr*||x||^2 过小无法收敛。
        B_continuous = self._rng.standard_normal((state_dim, input_dim)) * np.sqrt(2.0 / input_dim)
        # A^{-1} (A_bar - I) = (exp(a*dt) - 1) / a  (element-wise on diag)
        # 处理 a=0 的奇异情况（实际不会发生，因 a_n = -n）
        scale = np.where(
            np.abs(self._A_diag) > 1e-10,
            (self._A_bar - 1.0) / self._A_diag,
            self.dt,  # 极限：a→0 时 (exp(a*dt)-1)/a → dt
        )
        self._B_bar = scale[:, np.newaxis] * B_continuous

        # C: 输出投影矩阵（state_dim → output_dim）
        # Xavier 初始化
        self._C = (
            self._rng.standard_normal((output_dim, state_dim))
            * np.sqrt(2.0 / (state_dim + output_dim))
        )

        # D: 旁路（input_dim → output_dim），通常初始化为 0 或小随机
        self._D = np.zeros((output_dim, input_dim))

        # 隐状态
        self._state = np.zeros(state_dim, dtype=np.float64)

        # NaN 防护：上一次有效的状态（用于 NaN 输入恢复）
        self._last_valid_state = self._state.copy()

    @property
    def state(self) -> np.ndarray:
        """当前隐状态（只读副本）。"""
        with self._lock:
            return self._state.copy()

    @property
    def spectral_radius(self) -> float:
        """离散化状态矩阵的谱半径（应 < 1 保证稳定性）。"""
        return float(np.max(np.abs(self._A_bar)))

    def step(self, u: np.ndarray) -> np.ndarray:
        """单步递归：x_{k+1} = A_bar x_k + B_bar u_k, y_k = C x_k + D u_k.

        用于 think() 循环中的在线预测。线程安全。

        Parameters
        ----------
        u : ndarray, shape (input_dim,)
            当前输入

        Returns
        -------
        y : ndarray, shape (output_dim,)
            当前输出预测

        Raises
        ------
        ValueError
            如果 u 的形状不匹配 input_dim
        """
        u = np.asarray(u, dtype=np.float64)
        if u.shape != (self.input_dim,):
            raise ValueError(f"u must have shape ({self.input_dim},), got {u.shape}")

        # NaN 防护
        if not np.all(np.isfinite(u)):
            # 返回基于上一次有效状态的预测（不污染 state）
            return self._C @ self._last_valid_state

        with self._lock:
            # 状态更新：x_{k+1} = A_bar * x_k + B_bar * u_k
            # 对角 A_bar：element-wise 乘法（高效）
            new_state = self._A_bar * self._state + self._B_bar @ u

            # NaN 防护（极端数值下可能产生 NaN）
            if not np.all(np.isfinite(new_state)):
                new_state = self._last_valid_state.copy()
            else:
                self._last_valid_state = new_state.copy()

            self._state = new_state

            # 输出：y = C x + D u
            y = self._C @ self._state + self._D @ u
            return y

    def forward(self, u: np.ndarray) -> np.ndarray:
        """处理整段序列（卷积/递归混合模式）。

        默认用递归模式逐 step 调用 step()（保持与在线学习一致）。

        Parameters
        ----------
        u : ndarray, shape (L, input_dim)
            长度 L 的输入序列

        Returns
        -------
        y : ndarray, shape (L, output_dim)
            输出序列
        """
        u = np.asarray(u, dtype=np.float64)
        if u.ndim != 2 or u.shape[1] != self.input_dim:
            raise ValueError(
                f"u must have shape (L, {self.input_dim}), got {u.shape}"
            )

        L = u.shape[0]
        outputs = np.zeros((L, self.output_dim))
        # 保存当前 state，forward 不污染在线状态
        with self._lock:
            saved_state = self._state.copy()
            saved_last_valid = self._last_valid_state.copy()

        try:
            for t in range(L):
                outputs[t] = self.step(u[t])
        finally:
            # 恢复 state（forward 是纯查询，不改在线状态）
            with self._lock:
                self._state = saved_state
                self._last_valid_state = saved_last_valid

        return outputs

    def update(self, error: np.ndarray, u: np.ndarray | None = None) -> None:
        """预测误差驱动的局部 Hebbian 权重更新。

        更新规则：
            ΔC = -lr * err * x^T           (输出权重，减小预测误差)
            ΔB = -lr * x * u^T * scale      (输入权重，anti-Hebbian)
            A 不更新（HiPPO 稳定性保证）

        Parameters
        ----------
        error : ndarray, shape (output_dim,)
            预测误差 y_pred - y_target
        u : ndarray, shape (input_dim,) | None
            当前输入（用于 B 更新）；None 时跳过 B 更新
        """
        error = np.asarray(error, dtype=np.float64)
        if error.shape != (self.output_dim,):
            raise ValueError(f"error must have shape ({self.output_dim},), got {error.shape}")

        # NaN 防护
        if not np.all(np.isfinite(error)):
            return

        with self._lock:
            # C 更新：ΔC = -lr * err * x^T
            # x 是当前隐状态（C 的输入）
            delta_C = -self.lr * np.outer(error, self._state)
            # 梯度裁剪（防止爆炸）
            norm_C = np.linalg.norm(delta_C)
            if norm_C > 1.0:
                delta_C = delta_C * (1.0 / norm_C)
            self._C += delta_C

            # B 更新：ΔB = -lr * scale * x * u^T
            # scale = (A_bar - 1) / A_diag (从 B_bar 推导)
            if u is not None:
                u = np.asarray(u, dtype=np.float64)
                if u.shape == (self.input_dim,) and np.all(np.isfinite(u)):
                    delta_B = -self.lr * np.outer(self._state, u)
                    norm_B = np.linalg.norm(delta_B)
                    if norm_B > 1.0:
                        delta_B = delta_B * (1.0 / norm_B)
                    # 注意：B_bar 的更新较敏感，使用更小学习率
                    self._B_bar += delta_B * 0.1

    def reset(self) -> None:
        """重置隐状态为零。"""
        with self._lock:
            self._state = np.zeros(self.state_dim, dtype=np.float64)
            self._last_valid_state = self._state.copy()

    def get_state_snapshot(self) -> dict:
        """返回状态快照（用于观测/可视化）。"""
        with self._lock:
            return {
                "state_norm": float(np.linalg.norm(self._state)),
                "state_first_5": self._state[:5].tolist(),
                "spectral_radius": self.spectral_radius,
                "a_bar_max": float(np.max(np.abs(self._A_bar))),
            }

    def __getstate__(self) -> dict:
        """支持 pickle（RLock 不可序列化）。"""
        with self._lock:
            return {
                "state_dim": self.state_dim,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "dt": self.dt,
                "lr": self.lr,
                "_A_diag": self._A_diag.copy(),
                "_A_bar": self._A_bar.copy(),
                "_B_bar": self._B_bar.copy(),
                "_C": self._C.copy(),
                "_D": self._D.copy(),
                "_state": self._state.copy(),
                "_last_valid_state": self._last_valid_state.copy(),
            }

    def __setstate__(self, state: dict) -> None:
        for k, v in state.items():
            setattr(self, k, v)
        self._lock = threading.RLock()
