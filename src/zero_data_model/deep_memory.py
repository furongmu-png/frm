"""深度认知记忆模块（S4 + PCN + Hopfield 三合一协调器）。

理论基础
========

第一阶段认知升级把三个已实现的组件整合为一个协调循环：

  1. **S4** (Structured State Space)：长程序列记忆
     - 维护隐状态 x_t，跨多个 think() 步保留时序信息
     - 替代 ActiveInferenceEngine 的固定转移矩阵

  2. **PCN** (Predictive Coding Network)：层次化预测编码
     - 3 层 L0/L1/L2 形成层级抽象
     - 自顶向下预测 + 自底向上误差传播
     - 自由能 = ||最底层误差||² / 2

  3. **Hopfield** (Modern Hopfield Network)：联想记忆
     - 大容量存储关键 (key, value) 对
     - 检索靠 softmax attention，1 步收敛
     - 用于模式补全、上下文召回

协调循环（per step）
-------------------

输入：观测向量 obs（dim 维）

  1. S4.step(obs) → 长程上下文预测 s_pred（dim 维）
  2. PCN.forward(obs) → 层级误差 e_l0, e_l1, e_l2
     自由能 FE ≈ ||e_l0||² / 2
  3. Hopfield.retrieve(obs) → 关联记忆 h_assoc
     若 h_assoc 与 obs 相似度高 → 说明"已知"模式
     若相似度低 → "新奇"，触发存储
  4. 融合：fused = α·obs + β·s_pred + γ·h_assoc
     α + β + γ = 1
  5. S4.update(e_l0)：用 PCN 最底层误差驱动 S4 学习
  6. 若 FE 低于阈值且为"新奇"模式 → Hopfield.store(obs)

输出：fused（融合后的隐表示），FE（自由能），meta（各组件状态）

这个循环把"时序记忆 + 抽象预测 + 联想召回"三者打通，形成比单一组件
更强的认知深度。所有更新都是局部 Hebbian，无反向传播。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .s4 import S4Layer
from .hopfield import HopfieldMemory
from .pcn import PCNLayer


# ------------------------------------------------------------------ #
# 数据结构
# ------------------------------------------------------------------ #
@dataclass
class DeepMemoryState:
    """单步深度记忆循环的输出状态。"""
    fused: np.ndarray              # 融合后的隐表示
    free_energy: float             # 自由能（PCN 最底层误差²/2）
    prediction_error: float        # 预测误差（|s_pred - obs|）
    hopfield_similarity: float     # Hopfield 检索相似度
    is_novel: bool                 # 是否为新模式
    s4_norm: float                 # S4 状态范数（稳定性监控）
    pcn_layer_errors: dict         # {layer_idx: error_norm}
    hopfield_size: int             # 当前记忆库大小


@dataclass
class DeepMemoryStats:
    """累计统计。"""
    total_steps: int = 0
    novel_count: int = 0           # 新奇模式数
    stored_count: int = 0          # 实际存储数
    avg_free_energy: float = 0.0
    avg_similarity: float = 0.0
    fe_history: list = field(default_factory=list)
    similarity_history: list = field(default_factory=list)


# ------------------------------------------------------------------ #
# 主类
# ------------------------------------------------------------------ #
class DeepMemoryModule:
    """S4 + PCN + Hopfield 协调的深度认知记忆模块。

    Parameters
    ----------
    dim : int
        隐空间维度（三者共用）
    s4_state_dim : int
        S4 隐状态维度（建议 2*dim）
    hopfield_capacity : int
        Hopfield 记忆槽位上限
    novelty_threshold : float
        相似度低于此值视为"新奇"模式（默认 0.7）
    store_fe_threshold : float
        自由能低于此值时才存储（默认 1.0，避免存噪声）
    alpha, beta, gamma : float
        融合权重（obs, s4_pred, hopfield_assoc），和应为 1
    lr : float
        三组件统一学习率
    seed : int | None
        随机种子
    """

    def __init__(
        self,
        dim: int = 64,
        *,
        s4_state_dim: Optional[int] = None,
        hopfield_capacity: int = 1024,
        novelty_threshold: float = 0.7,
        store_fe_threshold: float = 1.0,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
        lr: float = 0.01,
        seed: Optional[int] = None,
    ):
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        # 归一化融合权重
        s = alpha + beta + gamma
        if s <= 0:
            raise ValueError(f"alpha+beta+gamma must be > 0, got {s}")
        self.alpha = alpha / s
        self.beta = beta / s
        self.gamma = gamma / s

        self.dim = dim
        self.novelty_threshold = float(novelty_threshold)
        self.store_fe_threshold = float(store_fe_threshold)
        self.lr = float(lr)
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._lock = threading.RLock()

        # ---- S4：长程时序记忆 ----
        s4_state = s4_state_dim if s4_state_dim is not None else 2 * dim
        self.s4 = S4Layer(
            state_dim=s4_state,
            input_dim=dim,
            output_dim=dim,
            dt=0.1,
            lr=lr,
            seed=seed,
        )

        # ---- PCN：3 层层次预测编码 ----
        # 层级：L2 (顶层, 抽象) → L1 (中间) → L0 (底层, 感知)
        #   - 预测自顶向下：L2.predict() → L1.predict() → L0
        #   - 误差自底向上：L0 → L1 → L2
        # L0: dim=dim, higher_dim=dim  (接收 L1 的预测)
        # L1: dim=dim, lower_dim=dim, higher_dim=L2_dim  (接收 L2 的预测)
        # L2: dim=L2_dim, lower_dim=dim  (顶层，无上层)
        L2_dim = max(dim // 2, 8)
        self.pcn_l2 = PCNLayer(dim=L2_dim, lower_dim=dim, lr=lr, seed=seed)
        self.pcn_l1 = PCNLayer(dim=dim, lower_dim=dim, higher_dim=L2_dim,
                               lr=lr, seed=(seed + 1) if seed else None)
        self.pcn_l0 = PCNLayer(dim=dim, higher_dim=dim, lr=lr,
                               seed=(seed + 2) if seed else None)

        # ---- Hopfield：联想记忆 ----
        hf_beta = max(2.0, dim / 2.0)
        self.hopfield = HopfieldMemory(
            memory_dim=dim,
            capacity=hopfield_capacity,
            beta=hf_beta,
            seed=seed,
        )

        # ---- 累计统计 ----
        self.stats = DeepMemoryStats()

    # ---------------------------------------------------------------- #
    # 单步循环
    # ---------------------------------------------------------------- #
    def step(self, obs: np.ndarray) -> DeepMemoryState:
        """执行一次深度记忆循环。

        Parameters
        ----------
        obs : ndarray, shape (dim,)
            观测向量

        Returns
        -------
        DeepMemoryState
            融合后的隐表示 + 自由能 + 各组件状态
        """
        obs = np.asarray(obs, dtype=np.float64).flatten()
        if obs.shape != (self.dim,):
            raise ValueError(
                f"obs must have shape ({self.dim},), got {obs.shape}"
            )

        with self._lock:
            # 1. S4：长程预测
            s_pred = self.s4.step(obs)

            # 2. PCN：层次预测编码
            #     自顶向下：L2.predict() → L1.predict() → L0
            #     自底向上：L0.update() → L1.update() → L2.update()
            # L0 接收观测
            self.pcn_l0.set_state(obs)
            # L1 对 L0 的预测
            l1_pred_for_l0 = self.pcn_l1.predict()
            # L0 误差（基于 L1 的预测）
            e_l0 = self.pcn_l0.update(prediction_from_higher=l1_pred_for_l0)
            # L2 对 L1 的预测
            l2_pred_for_l1 = self.pcn_l2.predict()
            # L1 误差（基于 L2 的预测 + L0 的误差）
            e_l1 = self.pcn_l1.update(
                error_from_lower=e_l0,
                prediction_from_higher=l2_pred_for_l1,
            )
            # L2 误差（基于 L1 的误差）
            e_l2 = self.pcn_l2.update(error_from_lower=e_l1)
            layer_errors = {
                "l0": float(np.linalg.norm(e_l0)),
                "l1": float(np.linalg.norm(e_l1)),
                "l2": float(np.linalg.norm(e_l2)),
            }
            free_energy = 0.5 * float(np.dot(e_l0, e_l0))

            # 3. Hopfield：联想检索
            hopfield_sim = 0.0
            h_assoc = np.zeros_like(obs)
            if self.hopfield.size > 0:
                try:
                    h_assoc, sims = self.hopfield.retrieve(obs, k=1)
                    hopfield_sim = float(sims[0]) if len(sims) > 0 else 0.0
                except Exception:
                    h_assoc = np.zeros_like(obs)
                    hopfield_sim = 0.0
            is_novel = hopfield_sim < self.novelty_threshold

            # 4. 融合
            fused = (self.alpha * obs
                     + self.beta * s_pred
                     + self.gamma * h_assoc)
            # NaN 防护
            if not np.all(np.isfinite(fused)):
                fused = obs.copy()

            # 5. S4 更新：用 PCN 最底层误差驱动
            s4_error = s_pred - obs
            self.s4.update(s4_error, u=obs)

            # 6. 存储：低自由能 + 新奇
            stored = False
            if (is_novel
                and free_energy < self.store_fe_threshold
                and self.hopfield.size < self.hopfield.capacity):
                try:
                    self.hopfield.store(fused)
                    stored = True
                except Exception:
                    pass

            # 7. 累计统计
            self.stats.total_steps += 1
            if is_novel:
                self.stats.novel_count += 1
            if stored:
                self.stats.stored_count += 1
            n = self.stats.total_steps
            self.stats.avg_free_energy = (
                (self.stats.avg_free_energy * (n - 1) + free_energy) / n
            )
            self.stats.avg_similarity = (
                (self.stats.avg_similarity * (n - 1) + hopfield_sim) / n
            )
            self.stats.fe_history.append(free_energy)
            self.stats.similarity_history.append(hopfield_sim)
            # 限制历史长度避免内存爆炸
            if len(self.stats.fe_history) > 10000:
                self.stats.fe_history = self.stats.fe_history[-10000:]
                self.stats.similarity_history = \
                    self.stats.similarity_history[-10000:]

            return DeepMemoryState(
                fused=fused,
                free_energy=free_energy,
                prediction_error=float(np.linalg.norm(s4_error)),
                hopfield_similarity=hopfield_sim,
                is_novel=is_novel,
                s4_norm=float(np.linalg.norm(self.s4._state)),
                pcn_layer_errors=layer_errors,
                hopfield_size=self.hopfield.size,
            )

    # ---------------------------------------------------------------- #
    # 查询接口
    # ---------------------------------------------------------------- #
    def retrieve_associated(self, query: np.ndarray,
                            k: int = 1) -> Optional[np.ndarray]:
        """用 Hopfield 检索关联记忆。"""
        if self.hopfield.size == 0:
            return None
        try:
            h, _ = self.hopfield.retrieve(query, k=k)
            return h
        except Exception:
            return None

    def get_snapshot(self) -> dict:
        """返回各组件当前状态的快照（用于可视化）。"""
        with self._lock:
            return {
                "s4": {
                    "state_dim": self.s4.state_dim,
                    "input_dim": self.s4.input_dim,
                    "output_dim": self.s4.output_dim,
                    "spectral_radius": self.s4.spectral_radius,
                    "state_norm": float(np.linalg.norm(self.s4._state)),
                },
                "pcn": {
                    "l0_dim": self.pcn_l0.dim,
                    "l1_dim": self.pcn_l1.dim,
                    "l2_dim": self.pcn_l2.dim,
                    "l0_error": float(self.pcn_l0.last_error_norm),
                    "l1_error": float(self.pcn_l1.last_error_norm),
                    "l2_error": float(self.pcn_l2.last_error_norm),
                },
                "hopfield": {
                    "size": self.hopfield.size,
                    "capacity": self.hopfield.capacity,
                    "beta": float(self.hopfield.beta),
                    "fill_ratio": self.hopfield.size / max(self.hopfield.capacity, 1),
                },
                "stats": {
                    "total_steps": self.stats.total_steps,
                    "novel_count": self.stats.novel_count,
                    "stored_count": self.stats.stored_count,
                    "avg_free_energy": self.stats.avg_free_energy,
                    "avg_similarity": self.stats.avg_similarity,
                },
            }

    def reset(self) -> None:
        """重置所有组件状态（保留学习到的权重）。"""
        with self._lock:
            self.s4._state[:] = 0
            self.pcn_l0.reset()
            self.pcn_l1.reset()
            self.pcn_l2.reset()
            self.stats = DeepMemoryStats()
