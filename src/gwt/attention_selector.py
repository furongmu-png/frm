"""GWT Attention Selector — softmax 竞争选择广播候选。

升级现有 GlobalWorkspace：每个模块输出"广播候选"向量 + "注意请求"标量，
通过 softmax 竞争选择胜者，胜者被广播到所有模块作为下一时刻上下文先验。
生物合理性：胜者抑制其他竞争者，未选中模块被短暂抑制。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class BroadcastCandidate:
    """模块输出的广播候选。"""

    module_name: str
    #: 候选隐向量（信念状态或预测误差摘要）
    vector: np.ndarray
    #: 注意请求强度（如预测误差幅度），越大越可能胜出
    attention_request: float
    #: 置信度标记（来自元认知模块的不确定度，0-1，越高越可信）
    confidence: float = 1.0
    #: 附加元数据
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompetitionResult:
    """竞争结果。"""

    winner: BroadcastCandidate | None
    #: 各模块的选择概率
    probabilities: dict[str, float] = field(default_factory=dict)
    #: 广播向量（胜者的 vector）
    broadcast: np.ndarray | None = None
    #: 广播置信度
    broadcast_confidence: float = 0.0
    #: 全局时间步标记（意识"时刻"）
    timestamp: int = 0
    #: 是否采样（True）或 argmax（False）
    sampled: bool = False


class AttentionSelector:
    """全局工作空间竞争选择器。

    Parameters
    ----------
    dim : int
        广播向量维度（所有候选需对齐到此维度）。
    temperature : float, default 1.0
        softmax 温度。越低竞争越激烈（胜者更突出），越高越平均。
    sample : bool, default False
        True: 按概率采样胜者；False: argmax。
    seed : int, default 42
    inhibition_steps : int, default 3
        未选中模块被抑制的步数（降低其学习率），模拟生物侧抑制。
    """

    def __init__(
        self,
        dim: int = 64,
        temperature: float = 1.0,
        sample: bool = False,
        seed: int = 42,
        inhibition_steps: int = 3,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if inhibition_steps < 0:
            raise ValueError(f"inhibition_steps must be >= 0, got {inhibition_steps}")

        self.dim = int(dim)
        self.temperature = float(temperature)
        self.sample = bool(sample)
        self.inhibition_steps = int(inhibition_steps)
        self._rng = np.random.default_rng(seed)

        #: 各模块被抑制的剩余步数
        self._inhibition: dict[str, int] = {}
        #: 历史胜者序列（意识流）
        self._winner_history: list[str] = []
        self._max_history = 500
        self._step = 0

    # ------------------------------------------------------------------ #
    # 竞争选择
    # ------------------------------------------------------------------ #
    def compete(self, candidates: list[BroadcastCandidate]) -> CompetitionResult:
        """对一组候选执行 softmax 竞争选择。

        Returns
        -------
        CompetitionResult
            包含胜者、各模块概率、广播向量。
        """
        if not candidates:
            return CompetitionResult(winner=None)

        # 过滤被抑制的模块（仍参与概率计算但请求降低）
        names = [c.module_name for c in candidates]
        # 注意请求衰减：被抑制模块的请求 ×0.1
        requests = np.array(
            [
                c.attention_request
                * (0.1 if self._inhibition.get(c.module_name, 0) > 0 else 1.0)
                for c in candidates
            ],
            dtype=np.float64,
        )
        # NaN/inf 安全：替换为 0（模块无有效请求）
        requests = np.where(np.isfinite(requests), requests, 0.0)
        # 数值稳定 softmax
        requests = np.clip(requests, -50.0, 50.0)
        scaled = requests / max(self.temperature, 1e-8)
        scaled = scaled - np.max(scaled)
        exp = np.exp(scaled)
        probs = exp / np.sum(exp)
        # 防御浮点误差导致 np.random.choice 报错
        probs = np.clip(probs, 0.0, 1.0)
        probs = probs / probs.sum()

        prob_map = {name: float(p) for name, p in zip(names, probs)}

        # 选择胜者
        if self.sample:
            idx = int(self._rng.choice(len(candidates), p=probs))
            sampled = True
        else:
            idx = int(np.argmax(probs))
            sampled = False

        winner = candidates[idx]
        # 广播向量对齐到 dim
        broadcast = self._align_vector(winner.vector)

        # 更新抑制：先衰减已存在的抑制（避免 off-by-one），
        # 再为新败者设置完整抑制，最后清除胜者抑制。
        # 顺序很重要：先衰减 → 再设新值，确保新设的不在同一步被衰减
        # （inhibition_steps=3 应持续 3 步而非 2 步）。
        for k in list(self._inhibition.keys()):
            if self._inhibition[k] > 0:
                self._inhibition[k] -= 1
            if self._inhibition[k] <= 0:
                del self._inhibition[k]
        # 胜者清除抑制
        self._inhibition[winner.module_name] = 0
        if self.inhibition_steps > 0:
            for c in candidates:
                if c.module_name != winner.module_name:
                    self._inhibition[c.module_name] = self.inhibition_steps

        # 记录历史
        self._winner_history.append(winner.module_name)
        if len(self._winner_history) > self._max_history:
            self._winner_history = self._winner_history[-self._max_history :]
        self._step += 1

        return CompetitionResult(
            winner=winner,
            probabilities=prob_map,
            broadcast=broadcast,
            broadcast_confidence=winner.confidence,
            timestamp=self._step,
            sampled=sampled,
        )

    def _align_vector(self, v: np.ndarray) -> np.ndarray:
        """对齐向量到 dim 维度（填充/截断 + L2 归一化）。"""
        vec = np.asarray(v, dtype=np.float64).flatten()
        if vec.size != self.dim:
            if vec.size < self.dim:
                vec = np.pad(vec, (0, self.dim - vec.size))
            else:
                vec = vec[: self.dim]
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm
        return vec

    # ------------------------------------------------------------------ #
    # 访问器
    # ------------------------------------------------------------------ #
    def get_inhibition(self, module_name: str) -> int:
        """返回模块剩余抑制步数。"""
        return self._inhibition.get(module_name, 0)

    @property
    def winner_history(self) -> list[str]:
        """历史胜者序列（意识流）。"""
        return list(self._winner_history)

    @property
    def step(self) -> int:
        return self._step

    def configure(self, **kwargs: Any) -> None:
        if "temperature" in kwargs:
            v = float(kwargs["temperature"])
            if v <= 0:
                raise ValueError("temperature must be positive")
            self.temperature = v
        if "sample" in kwargs:
            self.sample = bool(kwargs["sample"])
        if "inhibition_steps" in kwargs:
            v = int(kwargs["inhibition_steps"])
            if v < 0:
                raise ValueError("inhibition_steps must be >= 0")
            self.inhibition_steps = v

    def snapshot(self) -> dict:
        return {
            "dim": self.dim,
            "temperature": self.temperature,
            "sample": self.sample,
            "step": self._step,
            "inhibited_modules": list(self._inhibition.keys()),
            "recent_winners": self._winner_history[-10:],
            "winner_distribution": self._winner_distribution(),
        }

    def _winner_distribution(self) -> dict[str, float]:
        """各模块胜出的频率（意识流统计）。"""
        if not self._winner_history:
            return {}
        counts: dict[str, int] = {}
        for w in self._winner_history:
            counts[w] = counts.get(w, 0) + 1
        total = len(self._winner_history)
        return {k: round(v / total, 4) for k, v in counts.items()}
