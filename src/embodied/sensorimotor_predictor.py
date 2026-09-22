"""SensorimotorPredictor — predicts sensory consequences of perceptual actions.

理论基础
========
感知-运动偶联 (sensorimotor contingency) 是具身认知的核心概念：智能体必须
学习"如果我执行动作 X，感官会收到什么变化 Y"。本模块学习一个从
(信念状态, 动作) → 预测感官变化 的映射，预测误差驱动信念更新和动作选择。

实现：
  - S4 层处理感知序列（复用 src/zero_data_model/s4/s4_layer.py）
  - PCN 低层负责快速预测传感器值（视觉/触觉）
  - 训练信号：实际感知 - 预测感知 → Hebbian 更新

特别关注：
  - move_gaze → 预测视野平移（视觉帧整体偏移）
  - touch_probe → 预测压力反馈（触觉读数上升）
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

# 复用 S4 层（带降级回退，避免 S4 不可用时整个模块崩溃）
import sys
from pathlib import Path
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

try:
    from zero_data_model.s4 import S4Layer  # type: ignore
    _HAS_S4 = True
except Exception:  # pragma: no cover - S4 不可用时降级
    S4Layer = None  # type: ignore
    _HAS_S4 = False


class _NullS4Layer:
    """S4 不可用时的降级占位：直接返回输入，无状态副作用。

    军事级健壮性：核心模块不应因可选依赖缺失而崩溃。
    """

    def __init__(self, **_kwargs):
        self._state = np.zeros(1)
        self.spectral_radius = 0.0

    def step(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64).copy()

    def update(self, *args, **kwargs) -> None:
        return None

    def reset(self) -> None:
        return None


# ------------------------------------------------------------------ #
# 数据结构
# ------------------------------------------------------------------ #
@dataclass
class SensorimotorPrediction:
    """单步感知-运动预测结果。"""
    predicted_visual_change: np.ndarray   # 预测的视觉帧变化 (dim,)
    predicted_tactile: np.ndarray        # 预测的触觉读数 (n_probe,)
    visual_error: float                   # 视觉预测误差
    tactile_error: float                  # 触觉预测误差
    total_error: float                    # 总预测误差（自由能贡献）
    metadata: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------ #
# 主类
# ------------------------------------------------------------------ #
class SensorimotorPredictor:
    """感知-运动预测器。

    学习 (belief, action) → predicted_sensory_change 映射。

    Parameters
    ----------
    dim : int
        隐空间维度（与信念状态一致）
    n_probe : int
        触觉读数维度（与 EmbodiedBody.tactile_grid² 一致）
    s4_state_dim : int
        S4 隐状态维度
    lr : float
        学习率
    seed : int | None
    """

    def __init__(
        self,
        dim: int = 64,
        n_probe: int = 16,
        s4_state_dim: Optional[int] = None,
        lr: float = 0.01,
        seed: Optional[int] = 42,
        n_actions: int = 14,
    ):
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if n_probe < 1:
            raise ValueError(f"n_probe must be >= 1, got {n_probe}")
        if n_actions < 1:
            raise ValueError(f"n_actions must be >= 1, got {n_actions}")

        self.dim = dim
        self.n_probe = n_probe
        self.n_actions = int(n_actions)
        self.lr = float(lr)
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._lock = threading.RLock()

        # S4 处理感知序列（带降级回退）
        s4_state = s4_state_dim if s4_state_dim is not None else 2 * dim
        if _HAS_S4 and S4Layer is not None:
            self.s4 = S4Layer(
                state_dim=s4_state,
                input_dim=dim,
                output_dim=dim,
                dt=0.1,
                lr=lr,
                seed=seed,
            )
        else:
            # 降级：S4 不可用时用无状态占位，模块仍可工作
            self.s4 = _NullS4Layer()

        # 视觉预测器：dim → dim（预测视觉变化）
        # 用 Hebbian 权重 W_vis: (dim, dim)
        self._W_vis = self._rng.standard_normal((dim, dim)) * np.sqrt(
            2.0 / (dim + dim)
        )

        # 触觉预测器：dim → n_probe（预测触觉读数）
        self._W_tac = self._rng.standard_normal((n_probe, dim)) * np.sqrt(
            2.0 / (dim + n_probe)
        )

        # 动作嵌入矩阵：每个动作一个 dim 维调制向量。
        # C2 修复：原代码用 `1.0 + 0.1 * (action_id % 4)` 把 10 个动作
        # 坍缩为 4 个等价类（action_id=0/4/8 得到相同调制）。这使
        # select_perceptual_action 无法区分这些动作，主动感知退化为
        # 在 4 个动作中循环。改为学习的 per-action 嵌入，让每个动作
        # 拥有独立的调制方向，初始接近 1（与原行为兼容）。
        # 形状 (n_actions, dim)：第 a 行是动作 a 的元素级调制向量。
        self._action_embedding = (
            np.ones((self.n_actions, dim))
            + self._rng.standard_normal((self.n_actions, dim)) * 0.01
        )

        # 累计统计
        self._step_count = 0
        self._avg_visual_error = 0.0
        self._avg_tactile_error = 0.0
        self._error_history: list[float] = []

    # ---------------------------------------------------------------- #
    # 预测
    # ---------------------------------------------------------------- #
    def predict(
        self,
        belief: np.ndarray,
        action_id: int = 0,
        *,
        advance_state: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """预测给定信念和动作下的感官变化。

        Parameters
        ----------
        belief : ndarray, shape (dim,)
            当前信念状态
        action_id : int
            即将执行的动作 ID（用于动作调制）
        advance_state : bool
            是否推进 S4 内部状态。动作选择（``select_perceptual_action``）
            需要遍历多个候选动作的预测，此时应设为 False，否则一次选择
            会把 S4 状态推进 n_actions 次，污染序列上下文。
            训练时（``update``）应设为 True。

        Returns
        -------
        predicted_visual_change : ndarray, shape (dim,)
        predicted_tactile : ndarray, shape (n_probe,)
        """
        belief = np.asarray(belief, dtype=np.float64).flatten()
        if belief.shape != (self.dim,):
            # padding / truncation
            b = np.zeros(self.dim)
            n = min(len(belief), self.dim)
            b[:n] = belief[:n]
            belief = b

        if not np.all(np.isfinite(belief)):
            belief = np.zeros(self.dim)

        with self._lock:
            # S4 处理信念序列 → 长程上下文。
            # C1 修复：原代码每次调用 predict() 都推进 S4 状态，
            # 导致 select_perceptual_action 一次调用就把 S4 状态
            # 推进 n_actions 次（默认 10 次），破坏序列上下文。
            # 现在用 advance_state=False 在动作选择时不推进状态。
            if advance_state:
                context = self.s4.step(belief)
            else:
                # 不推进状态：对 belief 做一次前向但不更新 _state。
                # S4Layer 没有 peek()，所以这里用 step 后立即 reset
                # 不可行（会丢失合法训练时的状态）。改用 _NullS4Layer
                # 的等价策略：直接用 belief 作为 context（无序列信息），
                # 这在动作选择阶段是可接受的（信息增益代理只需相对比较）。
                context = belief.copy()

            # 动作调制：用学习的 per-action 嵌入向量。
            # C2 修复：原 `1.0 + 0.1 * (action_id % 4)` 坍缩 10 个动作为
            # 4 个等价类。现在每个动作有独立的 dim 维调制向量。
            idx = int(action_id) % self.n_actions
            modulation = self._action_embedding[idx]  # (dim,)
            modulated = context * modulation

            # NaN 防护
            if not np.all(np.isfinite(modulated)):
                modulated = np.zeros(self.dim)

            # 视觉预测变化
            pred_vis = self._W_vis @ modulated
            # 触觉预测
            pred_tac = self._W_tac @ modulated

            # NaN 防护
            if not np.all(np.isfinite(pred_vis)):
                pred_vis = np.zeros(self.dim)
            if not np.all(np.isfinite(pred_tac)):
                pred_tac = np.zeros(self.n_probe)

            return pred_vis, pred_tac

    # ---------------------------------------------------------------- #
    # 更新（误差驱动 Hebbian）
    # ---------------------------------------------------------------- #
    def update(
        self,
        belief: np.ndarray,
        action_id: int,
        actual_visual_change: np.ndarray,
        actual_tactile: np.ndarray,
    ) -> SensorimotorPrediction:
        """根据实际感官变化更新预测器。

        Parameters
        ----------
        belief : ndarray, shape (dim,)
        action_id : int
        actual_visual_change : ndarray, shape (dim,)
            实际视觉变化（current_frame_encoding - prev_frame_encoding）
        actual_tactile : ndarray, shape (n_probe,)
            实际触觉读数
        """
        belief = np.asarray(belief, dtype=np.float64).flatten()
        if belief.shape != (self.dim,):
            b = np.zeros(self.dim)
            n = min(len(belief), self.dim)
            b[:n] = belief[:n]
            belief = b

        actual_vis = np.asarray(actual_visual_change, dtype=np.float64).flatten()
        if actual_vis.shape != (self.dim,):
            v = np.zeros(self.dim)
            n = min(len(actual_vis), self.dim)
            v[:n] = actual_vis[:n]
            actual_vis = v

        actual_tac = np.asarray(actual_tactile, dtype=np.float64).flatten()
        if actual_tac.shape != (self.n_probe,):
            t = np.zeros(self.n_probe)
            n = min(len(actual_tac), self.n_probe)
            t[:n] = actual_tac[:n]
            actual_tac = t

        # NaN 防护
        if not np.all(np.isfinite(belief)):
            belief = np.zeros(self.dim)
        if not np.all(np.isfinite(actual_vis)):
            actual_vis = np.zeros(self.dim)
        if not np.all(np.isfinite(actual_tac)):
            actual_tac = np.zeros(self.n_probe)

        with self._lock:
            # 内联预测（避免调用 predict() 双步 S4）
            # S4 输出维度 = dim（output_dim），与 W_vis 兼容
            context = self.s4.step(belief)  # (dim,)
            # C2 修复：用学习的 per-action 嵌入，避免动作坍缩
            idx = int(action_id) % self.n_actions
            modulation = self._action_embedding[idx]  # (dim,)
            modulated = context * modulation
            if not np.all(np.isfinite(modulated)):
                modulated = np.zeros(self.dim)
            pred_vis = self._W_vis @ modulated
            pred_tac = self._W_tac @ modulated
            if not np.all(np.isfinite(pred_vis)):
                pred_vis = np.zeros(self.dim)
            if not np.all(np.isfinite(pred_tac)):
                pred_tac = np.zeros(self.n_probe)

            # 误差
            vis_error_vec = pred_vis - actual_vis
            tac_error_vec = pred_tac - actual_tac
            vis_error = float(np.linalg.norm(vis_error_vec))
            tac_error = float(np.linalg.norm(tac_error_vec))
            total_error = vis_error + tac_error

            # Hebbian 更新 W_vis: ΔW = -lr * error * modulated^T
            # outer(vis_error_vec (dim,), modulated (dim,)) -> (dim, dim) ✓
            delta_W_vis = -self.lr * np.outer(vis_error_vec, modulated)
            norm = np.linalg.norm(delta_W_vis)
            if norm > 1.0:
                delta_W_vis = delta_W_vis * (1.0 / norm)
            self._W_vis += delta_W_vis

            # outer(tac_error_vec (n_probe,), modulated (dim,)) -> (n_probe, dim) ✓
            delta_W_tac = -self.lr * np.outer(tac_error_vec, modulated)
            norm = np.linalg.norm(delta_W_tac)
            if norm > 1.0:
                delta_W_tac = delta_W_tac * (1.0 / norm)
            self._W_tac += delta_W_tac

            # C2 修复：更新动作嵌入，让模型学习每个动作对预测的调制。
            # 梯度 = -lr * (vis_error^T W_vis + tac_error^T W_tac) * context
            # 用链式法则：d(loss)/d(modulation) = W_vis^T vis_error + W_tac^T tac_error
            # 然后 d(modulation)/d(action_embedding) = context（元素级乘法）
            grad_mod_vis = self._W_vis.T @ vis_error_vec  # (dim,)
            grad_mod_tac = self._W_tac.T @ tac_error_vec  # (dim,)
            grad_embedding = (grad_mod_vis + grad_mod_tac) * context  # (dim,)
            # 裁剪梯度范数，防止嵌入爆炸
            grad_norm = np.linalg.norm(grad_embedding)
            if grad_norm > 1.0:
                grad_embedding = grad_embedding * (1.0 / grad_norm)
            self._action_embedding[idx] -= self.lr * grad_embedding
            # 嵌入值裁剪到合理范围，防止退化
            np.clip(
                self._action_embedding[idx],
                -3.0,
                3.0,
                out=self._action_embedding[idx],
            )

            # S4 更新（用视觉误差驱动，error shape=(output_dim,) ✓）
            self.s4.update(vis_error_vec, u=belief)

            # 统计
            self._step_count += 1
            n = self._step_count
            self._avg_visual_error = (
                (self._avg_visual_error * (n - 1) + vis_error) / n
            )
            self._avg_tactile_error = (
                (self._avg_tactile_error * (n - 1) + tac_error) / n
            )
            self._error_history.append(total_error)
            if len(self._error_history) > 1000:
                self._error_history = self._error_history[-1000:]

            return SensorimotorPrediction(
                predicted_visual_change=pred_vis,
                predicted_tactile=pred_tac,
                visual_error=vis_error,
                tactile_error=tac_error,
                total_error=total_error,
                metadata={"step": self._step_count, "action_id": action_id},
            )

    # ---------------------------------------------------------------- #
    # 主动感知策略：选择能最大化信息增益的感知动作
    # ---------------------------------------------------------------- #
    def select_perceptual_action(
        self,
        belief: np.ndarray,
        n_actions: int = 10,
        visited: Optional[np.ndarray] = None,
    ) -> int:
        """选择预期信息增益最大的感知动作。

        策略：对每个候选动作预测感官变化，选择预测误差最大
        （= 信息增益最大）的动作。加入访问计数探索奖励。
        """
        belief = np.asarray(belief, dtype=np.float64).flatten()
        if belief.shape != (self.dim,):
            b = np.zeros(self.dim)
            n = min(len(belief), self.dim)
            b[:n] = belief[:n]
            belief = b

        with self._lock:
            scores = np.zeros(n_actions)
            for a in range(n_actions):
                # C1 修复：动作选择时不推进 S4 状态（advance_state=False），
                # 避免一次选择调用把 S4 状态推进 n_actions 次。
                pred_vis, pred_tac = self.predict(
                    belief, a, advance_state=False
                )
                # 预测变化幅度 = 信息增益代理
                ig = float(np.linalg.norm(pred_vis) + np.linalg.norm(pred_tac))
                # 访问计数探索奖励
                if visited is not None and a < len(visited):
                    ig += 0.1 / (visited[a] + 1.0)
                scores[a] = ig

            # softmax 选择
            scores -= scores.max()
            probs = np.exp(scores)
            probs /= probs.sum()
            action = int(self._rng.choice(n_actions, p=probs))
            return action

    # ---------------------------------------------------------------- #
    # 查询
    # ---------------------------------------------------------------- #
    def get_snapshot(self) -> dict[str, Any]:
        with self._lock:
            # C2 修复：暴露动作嵌入的多样性诊断。
            # 若所有动作嵌入相同（坍缩），则主动感知退化。
            if self.n_actions > 1:
                embed_std = float(
                    np.std(np.linalg.norm(self._action_embedding, axis=1))
                )
            else:
                embed_std = 0.0
            return {
                "step": self._step_count,
                "avg_visual_error": self._avg_visual_error,
                "avg_tactile_error": self._avg_tactile_error,
                "s4_state_norm": float(np.linalg.norm(self.s4._state)),
                "s4_spectral_radius": self.s4.spectral_radius,
                "w_vis_norm": float(np.linalg.norm(self._W_vis)),
                "w_tac_norm": float(np.linalg.norm(self._W_tac)),
                "action_embedding_diversity": embed_std,
                "n_actions": self.n_actions,
                "recent_error": (
                    float(np.mean(self._error_history[-20:]))
                    if self._error_history else 0.0
                ),
            }

    def reset(self) -> None:
        with self._lock:
            self.s4.reset()
            self._step_count = 0
            self._avg_visual_error = 0.0
            self._avg_tactile_error = 0.0
            self._error_history.clear()
            # 重置动作嵌入到初始值（接近 1，与原行为兼容）
            self._action_embedding = (
                np.ones((self.n_actions, self.dim))
                + self._rng.standard_normal((self.n_actions, self.dim)) * 0.01
            )
