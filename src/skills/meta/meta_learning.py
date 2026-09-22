"""学习如何学习：贝叶斯优化（高斯过程）自动调整模型超参数。

目标：长期自由能（累计预测误差）最小化。

设计：
- ``HyperparameterSpace`` 描述可调超参数（连续区间）。
- ``GaussianProcess`` 简化 GP（RBF 核 + Gaussian noise）拟合观测。
- ``MetaLearner`` 维护历史，每轮通过 acquisition 函数（EI/UCB）
  选下一个候选超参数组合。
- 不依赖外部库（如 scikit-optimize）；为可控复杂度采用解析解。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..base import SkillBase, SkillContext, SkillResult

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# 超参数空间
# ------------------------------------------------------------------ #


@dataclass
class Hyperparameter:
    name: str
    low: float
    high: float
    default: float
    log_scale: bool = False  # 是否在对数空间搜索

    def to_unit(self, x: float) -> float:
        """映射到 [0, 1] 单位空间。"""
        if self.log_scale:
            lo, hi = np.log(self.low), np.log(self.high)
            return float((np.log(x) - lo) / (hi - lo))
        return float((x - self.low) / (self.high - self.low))

    def from_unit(self, u: float) -> float:
        """从 [0, 1] 映射回原始空间。"""
        u = float(np.clip(u, 0.0, 1.0))
        if self.log_scale:
            lo, hi = np.log(self.low), np.log(self.high)
            return float(np.exp(lo + u * (hi - lo)))
        return float(self.low + u * (self.high - self.low))


class HyperparameterSpace:
    """超参数空间。"""

    def __init__(self, params: list[Hyperparameter]) -> None:
        self.params = params
        self.dim = len(params)

    def to_unit(self, values: dict[str, float]) -> np.ndarray:
        out = np.zeros(self.dim, dtype=np.float64)
        for i, p in enumerate(self.params):
            out[i] = p.to_unit(values[p.name])
        return out

    def from_unit(self, u: np.ndarray) -> dict[str, float]:
        u = np.clip(u, 0.0, 1.0)
        return {p.name: p.from_unit(u[i]) for i, p in enumerate(self.params)}

    def default_values(self) -> dict[str, float]:
        return {p.name: p.default for p in self.params}

    def random_sample(self, rng: np.random.Generator) -> dict[str, float]:
        return self.from_unit(rng.uniform(0.0, 1.0, self.dim))


# ------------------------------------------------------------------ #
# 简化高斯过程（RBF 核）
# ------------------------------------------------------------------ #


def _rbf_kernel(x1: np.ndarray, x2: np.ndarray, length_scale: float = 0.3) -> np.ndarray:
    """RBF 核，输入是 [n, d] 或 [d]。"""
    if x1.ndim == 1:
        x1 = x1[None, :]
    if x2.ndim == 1:
        x2 = x2[None, :]
    sq = (
        np.sum(x1 ** 2, axis=1)[:, None]
        + np.sum(x2 ** 2, axis=1)[None, :]
        - 2.0 * x1 @ x2.T
    )
    sq = np.maximum(sq, 0.0)
    return np.exp(-sq / (2.0 * length_scale ** 2))


class GaussianProcess:
    """简化 GP 回归：RBF + Gaussian noise。

    仅做点估计预测（mean + variance），不做完整后验采样，
    以避免 scipy 依赖。
    """

    def __init__(
        self,
        length_scale: float = 0.3,
        noise_var: float = 1e-3,
    ) -> None:
        self.length_scale = length_scale
        self.noise_var = noise_var
        self._X: np.ndarray = np.zeros((0, 0))
        self._y: np.ndarray = np.zeros(0)
        self._y_mean: float = 0.0  # 中心化前的均值，预测时需加回
        self._K_inv: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.ndim != 2:
            raise ValueError("X must be 2D [n, d]")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y shape mismatch")
        self._X = X
        self._y_mean = float(np.mean(y))
        self._y = y - self._y_mean  # 中心化
        K = _rbf_kernel(X, X, self.length_scale) + self.noise_var * np.eye(
            X.shape[0]
        )
        # 解析逆，加 jitter 防止奇异
        K += 1e-8 * np.eye(X.shape[0])
        self._K_inv = np.linalg.inv(K)

    def predict(self, x: np.ndarray) -> tuple[float, float]:
        if self._X.shape[0] == 0 or self._K_inv is None:
            return self._y_mean, 1.0
        if x.ndim == 1:
            x = x[None, :]
        k_star = _rbf_kernel(self._X, x, self.length_scale).ravel()  # [n]
        k_ss = 1.0 + self.noise_var
        mean = float(k_star @ self._K_inv @ self._y) + self._y_mean
        var = float(k_ss - k_star @ self._K_inv @ k_star)
        var = max(var, 1e-9)
        return mean, var


# ------------------------------------------------------------------ #
# Acquisition 函数
# ------------------------------------------------------------------ #


def expected_improvement(
    gp: GaussianProcess,
    X_candidate: np.ndarray,
    best_y: float,
    xi: float = 0.01,
) -> np.ndarray:
    """Expected Improvement。

    EI(x) = E[max(0, f* - f(x) - xi)]
    """
    means = np.zeros(X_candidate.shape[0])
    stds = np.zeros(X_candidate.shape[0])
    for i in range(X_candidate.shape[0]):
        m, v = gp.predict(X_candidate[i])
        means[i] = m
        stds[i] = float(np.sqrt(v))
    improvement = best_y - means - xi
    # 假设最小化目标：improvement = best - f - xi
    # 注意：这里 best_y 是历史最小（误差）。
    # 若 std 接近 0 → EI = max(0, improvement)
    # 否则用正态 CDF 近似
    safe_std = np.maximum(stds, 1e-9)
    z = improvement / safe_std
    # 标准正态 CDF
    cdf = 0.5 * (1.0 + np.tanh(0.7978 * z))  # tanh 近似 Φ(z)
    pdf = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    ei = improvement * cdf + safe_std * pdf
    return np.maximum(ei, 0.0)


# ------------------------------------------------------------------ #
# MetaLearner
# ------------------------------------------------------------------ #


class MetaLearner(SkillBase):
    """学习如何学习：贝叶斯优化自动调整超参数。

    工作流：
    1. ``observe(values, loss)``：记录一次超参数→损失观测。
    2. ``suggest()``：基于 GP + EI 选下一组候选超参数。
    3. 外部应用并再次 ``observe``，循环往复。
    """

    name = "meta_learning"
    dimension = "meta"

    #: GP 拟合 + EI 在候选集上评估有一定开销，节流到每 3 步刷新。
    #: ``observe()`` / ``suggest()`` 直接调用不受节流影响。
    process_interval = 3

    def __init__(
        self,
        *,
        space: HyperparameterSpace | None = None,
        n_initial_random: int = 3,
        length_scale: float = 0.3,
        noise_var: float = 1e-3,
        enabled: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(enabled=enabled, **kwargs)
        self.space = space or HyperparameterSpace(
            [
                Hyperparameter("learning_rate", 0.001, 0.5, 0.05, log_scale=True),
                Hyperparameter("beta_decay", 0.5, 0.999, 0.95, log_scale=False),
                Hyperparameter("consolidation_freq", 1, 20, 5, log_scale=False),
                Hyperparameter("exploration_beta", 0.1, 5.0, 1.0, log_scale=False),
            ]
        )
        self.n_initial_random = n_initial_random
        self.gp = GaussianProcess(length_scale=length_scale, noise_var=noise_var)
        self._X: list[np.ndarray] = []
        self._y: list[float] = []
        self._best_y: float = float("inf")
        self._best_params: dict[str, float] | None = None
        self._current_params: dict[str, float] = self.space.default_values()
        self._history: list[dict[str, Any]] = []
        self._rng = np.random.default_rng(42)
        self._step: int = 0

    # ------------------------------------------------------------------ #
    # 对外 API
    # ------------------------------------------------------------------ #

    def observe(self, params: dict[str, float], loss: float) -> None:
        """记录一次超参数→损失观测。"""
        u = self.space.to_unit(params)
        self._X.append(u)
        self._y.append(float(loss))
        self._history.append({"params": params, "loss": float(loss), "step": self._step})
        if loss < self._best_y:
            self._best_y = float(loss)
            self._best_params = dict(params)
        # 拟合 GP（数据点 > 0 时）
        if len(self._X) >= 1:
            X = np.array(self._X)
            y = np.array(self._y)
            self.gp.fit(X, y)
        self._step += 1

    def suggest(self, n_candidates: int = 64) -> dict[str, float]:
        """基于 GP + EI 选下一组候选超参数。"""
        # 前几次随机探索
        if len(self._X) < self.n_initial_random:
            return self.space.random_sample(self._rng)

        # 候选集：随机采样
        candidates = self._rng.uniform(0.0, 1.0, (n_candidates, self.space.dim))
        # 加上探索已有观测点的邻域
        if self._best_params is not None:
            best_u = self.space.to_unit(self._best_params)
            neighbors = best_u + self._rng.normal(0, 0.05, (n_candidates // 2, self.space.dim))
            neighbors = np.clip(neighbors, 0.0, 1.0)
            candidates = np.vstack([candidates, neighbors])

        ei = expected_improvement(self.gp, candidates, self._best_y)
        best_idx = int(np.argmax(ei))
        return self.space.from_unit(candidates[best_idx])

    def current_params(self) -> dict[str, float]:
        return dict(self._current_params)

    def best_params(self) -> dict[str, float] | None:
        return dict(self._best_params) if self._best_params is not None else None

    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def stats(self) -> dict[str, Any]:
        return {
            "n_observations": len(self._X),
            "best_loss": self._best_y if self._X else None,
            "best_params": self._best_params,
            "current_params": self._current_params,
        }

    # ------------------------------------------------------------------ #
    # SkillBase 接口
    # ------------------------------------------------------------------ #

    def process(self, ctx: SkillContext) -> SkillResult:
        """每步：用 ctx.prediction_error 作为当前损失，建议下一组超参数。"""
        loss = float(ctx.prediction_error)
        self.observe(self._current_params, loss)
        suggested = self.suggest()
        self._current_params = suggested
        return SkillResult(
            name=self.name,
            data={
                "step": self._step,
                "current_loss": loss,
                "best_loss": self._best_y if self._X else None,
                "best_params": self._best_params,
                "suggested_params": suggested,
                "n_observations": len(self._X),
            },
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "ready": len(self._X) > 0,
            "data": {
                "n_observations": len(self._X),
                "best_loss": self._best_y if self._X else None,
                "best_params": self._best_params,
                "current_params": self._current_params,
                "history_size": len(self._history),
            },
        }
