"""JEPA Module — orchestrates target encoder + online predictor in think().

将 JEPA 预测误差与原有预测误差加权求和，作为自由能驱动信号。
提供 get_target_embedding(obs) 供下游任务（检索、类比）使用。
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .target_encoder import TargetEncoder
from .predictor import OnlinePredictor


class JEPAModule:
    """JEPA 集成模块：在 think() 中协调目标编码器与在线预测器。

    每步执行:
      1. target = target_encoder.encode(obs)           # EMA 目标
      2. pred = predictor.predict(latent_state)        # 在线预测
      3. jepa_error = ||pred - target||
      4. predictor.update(latent_state, target)        # 误差驱动
      5. target_encoder.update_from_online(...)         # EMA 追踪

    Parameters
    ----------
    input_dim : int
        原始观测展平维度（物理帧 128×128=16384 或文本块）。
    latent_dim : int, default 64
        目标隐空间维度。
    lambda_jepa : float, default 0.5
        JEPA 误差在总自由能中的权重。初期 0.5，可配置。
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        lambda_jepa: float = 0.5,
        ema_tau: float = 0.996,
        lr: float = 1e-3,
        seed: int = 42,
    ) -> None:
        if not 0.0 <= lambda_jepa <= 1.0:
            raise ValueError(f"lambda_jepa must be in [0, 1], got {lambda_jepa}")

        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.lambda_jepa = float(lambda_jepa)
        self._enabled = True

        self.target_encoder = TargetEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            ema_tau=ema_tau,
            seed=seed,
        )
        # 预测器输入维度为 latent_dim（假设在线隐状态已对齐，
        # 实际使用时由 think() 传入与 latent_dim 兼容的状态或自动适配）
        self.predictor = OnlinePredictor(
            latent_dim=latent_dim,
            target_dim=latent_dim,
            lr=lr,
            seed=seed + 1,
        )

        self._jepa_error_history: list[float] = []
        self._max_history = 200
        self._step = 0
        self._latency_budget_ms = 0.5

    # ------------------------------------------------------------------ #
    # 主循环：每步调用
    # ------------------------------------------------------------------ #
    def process(
        self,
        observation: np.ndarray,
        latent_state: np.ndarray,
    ) -> dict[str, Any]:
        """执行一次 JEPA 前向 + 更新。

        Parameters
        ----------
        observation : np.ndarray
            原始观测（物理帧/文本块展平）。
        latent_state : np.ndarray
            在线隐状态（S4 输出或 PCN L0 state），维度需匹配 latent_dim。

        Returns
        -------
        result : dict
            包含 jepa_error, combined_free_energy, target_embedding 等。
        """
        if not self._enabled:
            return {"enabled": False}

        t0 = time.perf_counter()
        try:
            # 1. 目标编码（EMA，自身不学习）
            target = self.target_encoder.encode(observation)

            # 2. 在线预测
            pred = self.predictor.predict(latent_state)

            # 3. 预测误差
            jepa_error = float(np.linalg.norm(pred - target))

            # 4. 误差驱动更新预测器
            self.predictor.update(latent_state, target)

            # 5. EMA 追踪：用预测器在线权重更新目标编码器
            W_online, b_online = self.predictor.get_online_weights()
            # 注意：W_online 是 (target_dim, hidden_dim)，需转置/适配到
            # target_encoder 的 (latent_dim, input_dim) 形状。
            # 这里用对角追踪（取 W_online 的 latent_dim×latent_dim 子块），
            # 保持 EMA 追踪语义而不强求形状完全匹配。
            self._ema_track(W_online, b_online)

            # 记录历史
            self._jepa_error_history.append(jepa_error)
            if len(self._jepa_error_history) > self._max_history:
                self._jepa_error_history = self._jepa_error_history[-self._max_history :]

            self._step += 1
            latency_ms = (time.perf_counter() - t0) * 1000.0

            return {
                "enabled": True,
                "jepa_error": round(jepa_error, 6),
                "target_norm": round(float(np.linalg.norm(target)), 6),
                "pred_norm": round(float(np.linalg.norm(pred)), 6),
                "latency_ms": round(latency_ms, 4),
                "step": self._step,
            }
        except (ValueError, FloatingPointError, IndexError) as exc:
            return {"enabled": True, "error": f"{type(exc).__name__}: {exc}"}

    def _ema_track(self, W_online: np.ndarray, b_online: np.ndarray) -> None:
        """将预测器在线权重的 latent_dim×latent_dim 子块作为 EMA 源。"""
        # 取 W_online (target_dim, hidden) 的前 latent_dim 行与前 latent_dim 列
        n = min(self.latent_dim, W_online.shape[1])
        sub = W_online[: self.latent_dim, :n]
        # 扩展到 (latent_dim, input_dim) 通过重复或填充
        if sub.shape[1] < self.input_dim:
            reps = self.input_dim // sub.shape[1] + 1
            expanded = np.tile(sub, (1, reps))[:, : self.input_dim]
        else:
            expanded = sub[:, : self.input_dim]
        # 缩放以匹配目标编码器权重的尺度
        scale = np.sqrt(self.input_dim / self.latent_dim)
        self.target_encoder.update_from_online(expanded * scale, b_online[: self.latent_dim])

    # ------------------------------------------------------------------ #
    # 自由能合并
    # ------------------------------------------------------------------ #
    def combine_free_energy(
        self, base_prediction_error: float, jepa_error: float | None = None
    ) -> float:
        """将 JEPA 误差与原有预测误差加权求和。

        F_total = (1 - λ) * F_pixel + λ * F_jepa
        """
        if jepa_error is None:
            jepa_error = self._jepa_error_history[-1] if self._jepa_error_history else 0.0
        return (1.0 - self.lambda_jepa) * float(base_prediction_error) + self.lambda_jepa * float(
            jepa_error
        )

    # ------------------------------------------------------------------ #
    # 下游接口
    # ------------------------------------------------------------------ #
    def get_target_embedding(self, observation: np.ndarray) -> np.ndarray:
        """供下游任务（检索、类比）使用的目标隐表示。"""
        return self.target_encoder.encode(observation)

    def get_online_embedding(self, latent_state: np.ndarray) -> np.ndarray:
        """供前端可视化对比的在线预测表示。"""
        return self.predictor.predict(latent_state)

    # ------------------------------------------------------------------ #
    # 配置与快照
    # ------------------------------------------------------------------ #
    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def configure(self, **kwargs: Any) -> None:
        if "lambda_jepa" in kwargs:
            v = float(kwargs["lambda_jepa"])
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"lambda_jepa must be in [0,1], got {v}")
            self.lambda_jepa = v
        if "enabled" in kwargs:
            self._enabled = bool(kwargs["enabled"])

    def snapshot(self) -> dict:
        """返回前端可视化的快照数据。"""
        te_snap = self.target_encoder.snapshot()
        pr_snap = self.predictor.snapshot()
        # 在线 vs 目标的对齐度（余弦相似度）
        online_emb = self.predictor.last_prediction
        target_emb = self.target_encoder.last_target
        alignment = 0.0
        if online_emb is not None and target_emb is not None:
            no = np.linalg.norm(online_emb)
            nt = np.linalg.norm(target_emb)
            if no > 1e-8 and nt > 1e-8:
                alignment = float(np.dot(online_emb, target_emb) / (no * nt))

        return {
            "enabled": self._enabled,
            "lambda_jepa": self.lambda_jepa,
            "jepa_error": (
                round(self._jepa_error_history[-1], 6) if self._jepa_error_history else 0.0
            ),
            "error_trend": (
                [round(e, 6) for e in self._jepa_error_history[-20:]]
                if self._jepa_error_history
                else []
            ),
            "alignment": round(alignment, 6),
            "target_encoder": te_snap,
            "predictor": pr_snap,
            # 3D 投影供 LatentSpace3D 面板叠加显示
            "online_3d": pr_snap["last_prediction_3d"],
            "target_3d": te_snap["last_target_3d"],
        }
