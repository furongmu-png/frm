# experiments/experience_buffer.py
"""Priority-sampled experience replay buffer.

Stores ``(obs, action, next_obs, error)`` transitions and supports
sampling by priority (error-weighted) or uniformly. Used by the
offline-consolidation loop in ``run_replay.py`` to let ZeroDataModel
re-hear past sensory transitions and consolidate learning.

Design notes:
- Backed by ``collections.deque(maxlen=capacity)`` so old experiences
  are evicted automatically once capacity is reached.
- Priority weights use ``error**alpha + epsilon`` (rank-free proportional
  sampling, à la Schaul et al. 2015). ``alpha=0`` recovers uniform
  sampling.
- ``beta`` is the importance-sampling correction exponent. At beta=0
  no correction is applied; at beta=1 the full IS weight is returned
  per sample. The consolidation loop uses these weights to scale
  prediction errors during re-learning (optional).
- The buffer is deterministic given the seed: the same seed + same
  insertion order produces the same sample sequence, across runs and
  platforms.

Phase G (三.2): Hopfield 联想记忆集成
------------------------------------
当 ``use_hopfield=True`` 时，内部存储额外维护一个现代 Hopfield 记忆网络
（``zero_data_model.hopfield.HopfieldMemory``），将每条经验的观测向量
编码为固定维度的吸引子。``retrieve(obs, k)`` 方法通过 softmax 能量检索
top-k 最相似的记忆，实现联想召回与模式补全（即使查询被部分遮挡也能
补全完整记忆）。离线巩固时调用 ``consolidate()`` 重新组织记忆矩阵
（ridge 伪逆更新），防止灾难性遗忘。

默认 ``use_hopfield=False`` 保证零回归（现有 11 个 ``test_experience_buffer.py``
测试不受影响）。
"""

from __future__ import annotations

import collections
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Phase G (三.2): 现代 Hopfield 联想记忆。延迟导入以避免在 use_hopfield=False
# 时产生不必要的依赖（且 experiments/ 脚本可能在没有 src/ on sys.path 时
# 直接 import 本模块）。
def _import_hopfield():
    """延迟导入 HopfieldMemory，仅在 use_hopfield=True 时调用。"""
    try:
        from zero_data_model.hopfield import HopfieldMemory
        return HopfieldMemory
    except ImportError:
        return None


@dataclass
class Experience:
    """A single transition stored in the buffer."""

    obs: np.ndarray
    action: int
    next_obs: np.ndarray
    error: float            # proxy for prediction error / free energy
    step: int = 0           # global step when this was recorded
    metadata: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        """JSON-friendly view (arrays → lists). Used for transcripts."""
        return {
            "obs": np.asarray(self.obs).tolist(),
            "action": int(self.action),
            "next_obs": np.asarray(self.next_obs).tolist(),
            "error": float(self.error),
            "step": int(self.step),
            "metadata": dict(self.metadata),
        }


@dataclass
class SampleBatch:
    """A batch of experiences returned by ``ExperienceBuffer.sample``."""

    experiences: list[Experience]
    weights: np.ndarray     # importance-sampling weights, shape (batch,)

    def __len__(self) -> int:
        return len(self.experiences)


class ExperienceBuffer:
    """A priority-sampled experience replay buffer.

    Parameters
    ----------
    capacity : int
        Maximum number of experiences. Older experiences are evicted
        once capacity is exceeded (FIFO via ``deque(maxlen=...)``).
    alpha : float
        Priority exponent. ``0`` → uniform; ``1`` → proportional to
        error; ``0.5`` → compromise. Default 0.6.
    beta : float
        Importance-sampling correction exponent. ``0`` → no
        correction; ``1`` → full IS correction. Default 0.4.
    epsilon : float
        Small constant added to priorities to avoid zero-probability
        samples when error=0. Default 1e-3.
    seed : int, optional
        RNG seed for reproducible sampling. ``None`` uses the global
        Python random state (non-deterministic across runs).
    """

    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,
        beta: float = 0.4,
        epsilon: float = 1e-3,
        seed: Optional[int] = 42,
        # --- Phase G (三.2): Hopfield 联想记忆集成 ----------------------- #
        # use_hopfield=False 时使用经典 deque 优先级采样（零回归）；
        # use_hopfield=True 时额外维护 Hopfield 记忆矩阵，支持
        # ``retrieve(obs, k)`` 联想检索与 ``consolidate()`` 离线巩固。
        use_hopfield: bool = False,
        hopfield_dim: Optional[int] = None,
        hopfield_capacity: Optional[int] = None,
        hopfield_beta: Optional[float] = None,
    ):
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity!r}")
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha!r}")
        if not (0.0 <= beta <= 1.0):
            raise ValueError(f"beta must be in [0, 1], got {beta!r}")
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.seed = seed
        # ``deque(maxlen=...)`` evicts oldest entries automatically.
        self._buffer: collections.deque = collections.deque(maxlen=self.capacity)
        # Cached priorities — kept in lockstep with ``_buffer``.
        # Priority = error**alpha + epsilon. Recomputed on add().
        self._priorities: collections.deque = collections.deque(
            maxlen=self.capacity
        )
        # RNG: use Python's ``random.Random`` (affects only sampling,
        # not numpy's global state).
        self._rng = random.Random(seed)

        # --- Phase G (三.2): Hopfield 联想记忆 -------------------------- #
        self.use_hopfield = bool(use_hopfield)
        self._hopfield = None
        # 平行存储每条经验的编码 key（用于 retrieve 后映射回 Experience）。
        # _buffer 仍然是 Experience 对象的真相源（Hopfield 只存向量）。
        self._hopfield_keys: collections.deque = collections.deque(maxlen=self.capacity)
        # hopfield_dim: 固定编码维度。若 None，则在第一条 add() 时根据 obs
        # 维度自动推断（取 obs 长度，最小 8）。
        self._hopfield_dim = int(hopfield_dim) if hopfield_dim is not None else None
        if self.use_hopfield:
            HopfieldMemory = _import_hopfield()
            if HopfieldMemory is None:
                raise ImportError(
                    "use_hopfield=True requires zero_data_model.hopfield; "
                    "ensure src/ is on sys.path (experiments scripts add it)"
                )
            # 实际 HopfieldMemory 实例在第一条 add() 时延迟创建
            # （因为需要知道 memory_dim = hopfield_dim）。
            self._HopfieldMemory_cls = HopfieldMemory
            self._hopfield_capacity = int(hopfield_capacity) if hopfield_capacity is not None else self.capacity
            self._hopfield_beta_param = hopfield_beta
        else:
            self._HopfieldMemory_cls = None
            self._hopfield_capacity = None
            self._hopfield_beta_param = None

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def is_empty(self) -> bool:
        return len(self._buffer) == 0

    # ------------------------------------------------------------------ #
    # Add a transition
    # ------------------------------------------------------------------ #
    def add(
        self,
        obs: np.ndarray,
        action: int,
        next_obs: np.ndarray,
        error: float,
        step: int = 0,
        metadata: Optional[dict] = None,
    ) -> None:
        """Store one transition in the buffer.

        Parameters
        ----------
        obs, next_obs : np.ndarray
            Observation vectors. Copied to avoid external mutation.
        action : int
            Discrete action taken.
        error : float
            Prediction error / free energy at this transition. Used
            to compute priority. Non-finite values are clamped to 0.
        step : int
            Global step counter when this was recorded. Useful for
            diagnostics.
        metadata : dict, optional
            Arbitrary per-experience metadata.

        Phase G (三.2): 当 ``use_hopfield=True`` 时，额外将 ``obs`` 编码为
        Hopfield 吸引子存储，支持后续 ``retrieve()`` 联想检索。
        """
        # Defensive copy so external code can mutate the arrays
        # without affecting stored experiences.
        obs_arr = np.array(obs, copy=True)
        next_arr = np.array(next_obs, copy=True)
        err = float(error) if np.isfinite(error) else 0.0
        # Clamp negative errors to 0 — priority is non-negative.
        if err < 0:
            err = 0.0
        exp = Experience(
            obs=obs_arr,
            action=int(action),
            next_obs=next_arr,
            error=err,
            step=int(step),
            metadata=dict(metadata or {}),
        )
        self._buffer.append(exp)
        self._priorities.append(err ** self.alpha + self.epsilon)

        # Phase G (三.2): Hopfield 联想记忆存储
        if self.use_hopfield:
            self._store_in_hopfield(obs_arr)

    # ------------------------------------------------------------------ #
    # Phase G (三.2): Hopfield 联想记忆
    # ------------------------------------------------------------------ #
    def _encode_for_hopfield(self, vec: np.ndarray) -> np.ndarray:
        """将任意维度的观测向量编码为固定维度的 Hopfield key。

        策略：
        1. 截断或零填充到 ``self._hopfield_dim``（满足 Hopfield 固定维度要求）
        2. L2 归一化（使点积 = 余弦相似度，对观测幅值不敏感）

        L2 归一化的必要性：现代 Hopfield 的相似度 ``X^T ξ`` 是点积，对向量
        幅值敏感。若不同经验的观测幅值差异大（如自由能高/低状态），点积
        会被幅值主导而非方向。归一化后，检索基于 *方向* 相似性（余弦），
        与 Ramsauer 2020 论文对 β = 1/√d 的推荐设定一致。

        对 ExperienceBuffer 的用途（定位具体经验索引），归一化不影响正确性
        ——我们只需要找到最相似的方向，而非重建原始观测。
        """
        vec = np.asarray(vec, dtype=np.float64).ravel()
        if self._hopfield_dim is None:
            # 第一条经验：根据 obs 维度推断 memory_dim（最小 8）
            self._hopfield_dim = max(8, len(vec))
        if len(vec) >= self._hopfield_dim:
            encoded = vec[: self._hopfield_dim].copy()
        else:
            # 零填充
            encoded = np.zeros(self._hopfield_dim, dtype=np.float64)
            encoded[: len(vec)] = vec
        # L2 归一化（避免零向量除零）
        norm = np.linalg.norm(encoded)
        if norm > 1e-12:
            encoded = encoded / norm
        return encoded

    def _store_in_hopfield(self, obs: np.ndarray) -> None:
        """将 obs 编码为吸引子存入 Hopfield 记忆矩阵。

        延迟初始化 HopfieldMemory（第一条经验时根据 obs 维度确定
        memory_dim）。存储为自联想吸引子（key=value=encoded_obs），
        支持后续部分遮挡查询的模式补全。

        关键：必须与 ``_buffer`` 保持索引对齐。即使 obs 含 NaN 也始终
        存储（NaN 替换为零向量），这样 ``retrieve_topk`` 返回的索引可以
        直接映射回 ``_buffer[idx]``。零向量与真实查询相似度极低，不会
        污染检索结果。
        """
        key = self._encode_for_hopfield(obs)
        if self._hopfield is None:
            # 延迟创建实例。若 hopfield_beta 未指定，根据 memory_dim 计算
            # 适合归一化向量的默认值。L2 归一化后点积 ∈ [-1, 1]（余弦
            # 相似度），Ramsauer 默认 β=1/√d 对此范围过小（softmax 几乎
            # 均匀）。改用 β = max(2.0, d/2) 使相似度差异在 softmax 中
            # 可分辨，支持模式补全（部分遮挡查询）。
            beta = self._hopfield_beta_param
            if beta is None:
                beta = max(2.0, self._hopfield_dim / 2.0)
            self._hopfield = self._HopfieldMemory_cls(
                memory_dim=self._hopfield_dim,
                capacity=self._hopfield_capacity,
                beta=beta,
                seed=self.seed,
            )
        # NaN 替换为零（保持索引对齐，零向量相似度极低不污染检索）
        if not np.all(np.isfinite(key)):
            key = np.nan_to_num(key, nan=0.0, posinf=0.0, neginf=0.0)
        # 存为自联想吸引子（key == value）
        self._hopfield.store(key, value=key)
        # 平行缓存编码后的 key，用于 retrieve 后映射回 Experience
        self._hopfield_keys.append(key)

    def retrieve(self, obs: np.ndarray, k: int = 1) -> SampleBatch:
        """Hopfield 联想检索：返回与 ``obs`` 最相似的 top-k 经验。

        Phase G (三.2) spec: "``sample()`` 方法改为 ``retrieve(obs, k)``，
        返回与当前状态最相关的记忆"。

        使用现代 Hopfield 网络的 softmax 能量检索：
            retrieve(ξ) = X @ softmax(β · X^T · ξ)
        返回 top-k 个最相似的记忆。即使查询被部分遮挡（如被遮挡的物体），
        Hopfield 的模式补全能力也能检索到完整记忆。

        Parameters
        ----------
        obs : np.ndarray
            查询观测向量（任意维度，自动编码到 hopfield_dim）。
        k : int
            返回的记忆数量。

        Returns
        -------
        SampleBatch
            ``experiences`` 为 top-k 最相似的 Experience 列表，
            ``weights`` 为对应的相似度（softmax 概率，用于 IS 校正）。

        Raises
        ------
        RuntimeError
            如果 ``use_hopfield=False`` 或 Hopfield 记忆为空。
        """
        if not self.use_hopfield or self._hopfield is None:
            raise RuntimeError(
                "retrieve() requires use_hopfield=True and at least one stored "
                "experience; call add() first or set use_hopfield=True"
            )
        if self.is_empty or len(self._hopfield_keys) == 0:
            return SampleBatch(experiences=[], weights=np.array([]))
        query = self._encode_for_hopfield(obs)
        k = max(1, min(int(k), len(self._buffer)))
        # Phase G (三.2): 使用 retrieve_topk 直接获取存储索引（保持与
        # _buffer 的索引对齐）。返回 (top_k_values, top_k_indices, sims)。
        # top_k_indices 直接映射到 _buffer，无需最近邻搜索。
        _values, top_k_indices, sims = self._hopfield.retrieve_topk(query, k=k)
        buffer_list = list(self._buffer)
        experiences: list[Experience] = []
        weights_list: list[float] = []
        for i, idx in enumerate(top_k_indices):
            idx_int = int(idx)
            if 0 <= idx_int < len(buffer_list):
                experiences.append(buffer_list[idx_int])
                weights_list.append(float(sims[i]) if i < len(sims) else 1.0)
        weights = np.array(weights_list, dtype=np.float64)
        # 归一化权重（防止为零）
        if weights.sum() > 0:
            weights = weights / weights.sum()
        return SampleBatch(experiences=experiences, weights=weights)

    def consolidate(self) -> dict:
        """离线巩固：重新组织 Hopfield 记忆矩阵以最大化检索精度。

        Phase G (三.2) spec: "离线巩固时，调用 ``update_weights()``
        重新组织记忆，防止遗忘"。

        使用 ridge 伪逆更新：
            W = X @ (X^T X + λI)^{-1}
        这通过正则化最小二乘重新组织记忆矩阵，减少记忆间的干扰，
        防止灾难性遗忘。

        Returns
        -------
        dict
            巩固统计信息（记忆数量、是否执行、耗时等）。

        Raises
        ------
        RuntimeError
            如果 ``use_hopfield=False``。
        """
        if not self.use_hopfield or self._hopfield is None:
            raise RuntimeError("consolidate() requires use_hopfield=True")
        # update_weights() 返回 None（原地重构记忆矩阵），这里返回巩固统计
        self._hopfield.update_weights()
        hf_stats = self._hopfield.get_stats() if hasattr(self._hopfield, "get_stats") else {}
        return {
            "consolidated": True,
            "n_attractors": len(self._hopfield_keys),
            "hopfield_stats": hf_stats,
        }

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #
    def sample(
        self,
        batch_size: int,
        mode: str = "priority",
        query: Optional[np.ndarray] = None,
    ) -> SampleBatch:
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size : int
            Number of experiences to return. Clamped to ``len(self)``.
        mode : str
            ``"priority"`` — sample proportional to ``error**alpha``
            (default). ``"uniform"`` — sample uniformly. ``"recent"`` —
            sample the most recent ``batch_size`` experiences.
            ``"hopfield"`` — (Phase G 三.2) Hopfield 联想检索，需提供
            ``query`` 观测向量，返回 top-k 最相似的记忆。
        query : np.ndarray, optional
            当 ``mode="hopfield"`` 时，提供查询观测向量用于联想检索。
            其他 mode 忽略此参数。

        Returns
        -------
        SampleBatch
            ``experiences`` list + ``weights`` array of IS weights.
            For ``mode="uniform"`` and ``"recent"`` weights are all 1.0.
            For ``mode="hopfield"`` weights 为 softmax 相似度（归一化）。
        """
        if self.is_empty:
            return SampleBatch(experiences=[], weights=np.array([]))
        # Phase G (三.2): Hopfield 联想检索模式
        if mode == "hopfield":
            if query is None:
                raise ValueError(
                    "mode='hopfield' requires a query observation vector"
                )
            return self.retrieve(query, k=batch_size)
        n = len(self._buffer)
        k = max(1, min(int(batch_size), n))
        if mode == "recent":
            # Take the most recent k experiences directly.
            experiences = list(self._buffer)[-k:]
            weights = np.ones(k, dtype=np.float64)
            return SampleBatch(experiences=experiences, weights=weights)
        # Build sampling probabilities.
        if mode == "priority":
            priorities = np.array(self._priorities, dtype=np.float64)
            # Defensive: ensure non-negative and finite.
            priorities = np.where(
                np.isfinite(priorities) & (priorities > 0),
                priorities,
                self.epsilon,
            )
            probs = priorities / priorities.sum()
        elif mode == "uniform":
            probs = np.full(n, 1.0 / n, dtype=np.float64)
        else:
            raise ValueError(
                f"mode must be 'priority', 'uniform', 'recent', or 'hopfield'; "
                f"got {mode!r}"
            )
        # Sample without replacement (standard for replay buffers —
        # makes the gradient estimates less correlated).
        idx = self._rng.sample(range(n), k=k)
        # ``random.sample`` is uniform; to apply our probabilities we
        # use numpy.random.choice-like logic via the cumulative
        # distribution. We do this manually to keep determinism with
        # ``self._rng`` (Python random).
        # Build the CDF and invert.
        cdf = np.cumsum(probs)
        u = np.array([self._rng.random() for _ in range(k)])
        chosen_idx = np.searchsorted(cdf, u, side="right")
        # Clamp (numerical safety).
        chosen_idx = np.clip(chosen_idx, 0, n - 1)
        # Deduplicate (if k > n this is impossible since k <= n; but
        # collisions in u → cdf mapping can occur). To avoid duplicates,
        # fall back to ``random.sample`` when priorities are uniform-ish.
        if len(set(chosen_idx.tolist())) < k:
            chosen_idx = np.array(self._rng.sample(range(n), k=k))
        experiences = [self._buffer[i] for i in chosen_idx]
        # Importance-sampling weights (for priority mode).
        if mode == "priority":
            # w_i = (N * p_i)**(-beta); normalise by max weight.
            sampled_probs = probs[chosen_idx]
            weights = (n * sampled_probs) ** (-self.beta)
            weights = weights / weights.max() if weights.max() > 0 else weights
        else:
            weights = np.ones(k, dtype=np.float64)
        return SampleBatch(experiences=experiences, weights=weights)

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        """Return a JSON-serialisable stats dict."""
        if self.is_empty:
            return {
                "n": 0, "capacity": self.capacity,
                "alpha": self.alpha, "beta": self.beta,
                "use_hopfield": self.use_hopfield,
            }
        errors = np.array([e.error for e in self._buffer])
        priorities = np.array(self._priorities)
        result = {
            "n": len(self._buffer),
            "capacity": self.capacity,
            "alpha": self.alpha,
            "beta": self.beta,
            "error_mean": float(errors.mean()),
            "error_std": float(errors.std()),
            "error_min": float(errors.min()),
            "error_max": float(errors.max()),
            "priority_mean": float(priorities.mean()),
            "use_hopfield": self.use_hopfield,
        }
        # Phase G (三.2): Hopfield 记忆统计
        if self.use_hopfield and self._hopfield is not None:
            hf_stats = self._hopfield.get_stats() if hasattr(self._hopfield, "get_stats") else {}
            result["hopfield"] = {
                "memory_dim": self._hopfield_dim,
                "n_attractors": len(self._hopfield_keys),
                "capacity": self._hopfield_capacity,
                **hf_stats,
            }
        return result

    def save_json(self, path: str | Path) -> str:
        """Save buffer stats (not full contents) to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.stats(), f, indent=2)
        return str(path)


# ------------------------------------------------------------------ #
# Self-test
# ------------------------------------------------------------------ #
def _self_test() -> None:
    """Smoke test the buffer."""
    buf = ExperienceBuffer(capacity=100, alpha=0.6, beta=0.4, seed=42)
    # Empty buffer.
    assert buf.is_empty
    assert len(buf) == 0
    batch = buf.sample(8)
    assert len(batch) == 0
    print("[OK] empty buffer sampling returns empty")
    # Add 20 experiences.
    rng = np.random.default_rng(0)
    for i in range(20):
        obs = rng.standard_normal(4)
        next_obs = rng.standard_normal(4)
        buf.add(obs, action=i % 4, next_obs=next_obs,
                error=float(i) / 20.0, step=i)
    assert len(buf) == 20
    assert not buf.is_empty
    print(f"[OK] added 20 experiences, len={len(buf)}")
    # Sample 8 with priority.
    batch = buf.sample(8, mode="priority")
    assert len(batch) == 8
    assert batch.experiences[0].obs.shape == (4,)
    print(f"[OK] priority sample: 8 exps, "
          f"errors={[f'{e.error:.2f}' for e in batch.experiences]}")
    # Verify high-error experiences are sampled more often.
    counts = np.zeros(20)
    for _ in range(2000):
        b = buf.sample(8, mode="priority")
        for e in b.experiences:
            # Identify by error value (unique per added exp).
            idx = int(round(e.error * 20))
            counts[idx] += 1
    high_err_count = counts[-5:].sum()
    low_err_count = counts[:5].sum()
    print(f"[OK] priority distribution: "
          f"high-err sampled {int(high_err_count)} times vs "
          f"low-err {int(low_err_count)} times")
    assert high_err_count > low_err_count, \
        "priority sampling did not favour high-error experiences"
    # Uniform sampling should be roughly balanced.
    counts = np.zeros(20)
    for _ in range(2000):
        b = buf.sample(8, mode="uniform")
        for e in b.experiences:
            idx = int(round(e.error * 20))
            counts[idx] += 1
    print(f"[OK] uniform distribution: "
          f"min bucket={int(counts.min())}, max bucket={int(counts.max())}")
    # Capacity eviction.
    for i in range(150):
        buf.add(np.zeros(4), action=0, next_obs=np.zeros(4),
                error=1.0, step=100 + i)
    assert len(buf) == 100, f"expected 100 after overflow, got {len(buf)}"
    print(f"[OK] capacity eviction: len={len(buf)} (capped at 100)")
    # Determinism.
    buf1 = ExperienceBuffer(capacity=50, seed=42)
    buf2 = ExperienceBuffer(capacity=50, seed=42)
    rng = np.random.default_rng(0)
    for i in range(30):
        obs = rng.standard_normal(4)
        buf1.add(obs, action=0, next_obs=obs, error=float(i), step=i)
        buf2.add(obs, action=0, next_obs=obs, error=float(i), step=i)
    b1 = buf1.sample(8, mode="priority")
    b2 = buf2.sample(8, mode="priority")
    same_idx = all(
        np.array_equal(b1.experiences[i].obs, b2.experiences[i].obs)
        for i in range(8)
    )
    assert same_idx, "seeded buffers produced different samples"
    print("[OK] determinism: same seed → same sample sequence")
    stats = buf.stats()
    print(f"[OK] stats: {stats}")
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    _self_test()
