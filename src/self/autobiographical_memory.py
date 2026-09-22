"""AutobiographicalMemory — self-episode store with narrative retrieval.

理论基础
========
自传体记忆 (autobiographical memory) 是关于"我自己经历过的"事件的记忆，
区别于语义记忆（关于世界的事实）。本模块用 Hopfield 网络存储"自我片段"，
每个片段包含 (状态快照, 自我图式预测, 实际结果, 自由能, 时间戳)。

检索时，从记忆中找出与当前 prompt 最相关的自我片段，通过模板化生成
连贯的自传体叙述（如"我上次看到红球时，它向右滚动了，所以我这次预期…"）。

离线巩固时，自传体记忆被重新激活，强化自我连续性——让智能体"回忆"
过去经历，巩固"我"的同一性。

自由能映射：自传体记忆检索相似度 = 自我连续性的置信度。相似度高意味着
当前情境与过去经历高度相关，自我图式可以可靠地预测。
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

# A5 修复：原代码用 `except Exception: return {"error": ...}` 吞下所有
# 检索错误且不记录日志，使生产环境无法诊断 Hopfield 检索失败。改为
# 用模块级 logger 记录错误上下文（军事级：错误可见、可追溯）。
_LOGGER = logging.getLogger(__name__)

# 复用 Hopfield 网络
import sys
from pathlib import Path
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from zero_data_model.hopfield import HopfieldMemory  # type: ignore


# ------------------------------------------------------------------ #
# 数据结构
# ------------------------------------------------------------------ #
@dataclass
class SelfEpisode:
    """一个自传体记忆片段。"""
    state_snapshot: np.ndarray       # 当时的状态快照 (dim,)
    self_prediction: dict[str, Any]  # 当时自我图式的预测
    actual_outcome: dict[str, Any]   # 实际发生的结果
    free_energy: float                # 当时的自由能
    timestamp: float                  # 时间戳
    step: int                         # 步数
    modality: str                     # 当时所处的模态/阶段
    narrative: str = ""               # 生成叙述（离线巩固时填充）


@dataclass
class NarrativeResult:
    """自传体叙述检索结果。"""
    narrative: str                    # 生成的叙述文本
    retrieved_episodes: list[SelfEpisode]  # 检索到的相关片段
    similarity: float                 # 最高相似度
    metadata: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------ #
# 主类
# ------------------------------------------------------------------ #
class AutobiographicalMemory:
    """自传体记忆：存储与检索自我片段。

    Parameters
    ----------
    dim : int
        状态快照维度（与 Hopfield memory_dim 一致）
    capacity : int
        记忆容量上限
    seed : int | None
    """

    def __init__(
        self,
        dim: int = 64,
        capacity: int = 1024,
        seed: Optional[int] = 42,
    ):
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")

        self.dim = int(dim)
        self.capacity = int(capacity)
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._lock = threading.RLock()

        # Hopfield 存储状态快照
        self.hopfield = HopfieldMemory(
            memory_dim=dim,
            capacity=capacity,
            beta=max(2.0, dim / 2.0),
            seed=seed,
        )

        # 完整的 SelfEpisode 列表（key 与 Hopfield 对齐）
        self._episodes: list[SelfEpisode] = []
        self._max_episodes = capacity

    # ---------------------------------------------------------------- #
    # 存储
    # ---------------------------------------------------------------- #
    def store(
        self,
        state_snapshot: np.ndarray,
        self_prediction: dict[str, Any],
        actual_outcome: dict[str, Any],
        free_energy: float,
        step: int,
        modality: str = "unknown",
    ) -> SelfEpisode:
        """存储一个自我片段。

        Parameters
        ----------
        state_snapshot : ndarray, shape (dim,)
            当时的状态快照
        self_prediction : dict
            当时自我图式的预测（注意/动作/情感）
        actual_outcome : dict
            实际发生的结果
        free_energy : float
            当时的自由能
        step : int
            步数
        modality : str
            当时所处的模态/阶段
        """
        snap = np.asarray(state_snapshot, dtype=np.float64).flatten()
        if snap.shape != (self.dim,):
            s = np.zeros(self.dim)
            n = min(len(snap), self.dim)
            s[:n] = snap[:n]
            snap = s
        if not np.all(np.isfinite(snap)):
            snap = np.nan_to_num(snap)

        with self._lock:
            episode = SelfEpisode(
                state_snapshot=snap.copy(),
                self_prediction=dict(self_prediction) if isinstance(self_prediction, dict) else {},
                actual_outcome=dict(actual_outcome) if isinstance(actual_outcome, dict) else {},
                free_energy=float(free_energy),
                timestamp=time.time(),
                step=int(step),
                modality=str(modality),
            )
            # FIFO
            if len(self._episodes) >= self._max_episodes:
                self._episodes.pop(0)
            self._episodes.append(episode)
            # Hopfield 存储快照（用于检索）
            self.hopfield.store(snap)
            return episode

    # ---------------------------------------------------------------- #
    # 叙述检索
    # ---------------------------------------------------------------- #
    def self_narrative_query(
        self,
        prompt: np.ndarray,
        top_k: int = 3,
        text_decoder: Optional[object] = None,
    ) -> NarrativeResult:
        """从记忆中检索相关自我片段，生成连贯的自传体叙述。

        Parameters
        ----------
        prompt : ndarray, shape (dim,)
            当前情境的状态向量
        top_k : int
            检索的相关片段数
        text_decoder : object | None
            可选的文本解码器（有 decode 方法）；None 时用模板化生成
        """
        p = np.asarray(prompt, dtype=np.float64).flatten()
        if p.shape != (self.dim,):
            s = np.zeros(self.dim)
            n = min(len(p), self.dim)
            s[:n] = p[:n]
            p = s
        if not np.all(np.isfinite(p)):
            p = np.zeros(self.dim)

        with self._lock:
            if self.hopfield.size == 0:
                return NarrativeResult(
                    narrative="我没有相关的过去经历可以回忆。",
                    retrieved_episodes=[],
                    similarity=0.0,
                    metadata={"empty": True},
                )

            # Hopfield 检索 top-k
            try:
                top_k_values, top_k_indices, top_k_sims = (
                    self.hopfield.retrieve_topk(p, k=top_k)
                )
            except Exception as exc:
                # A5 修复：原代码静默吞下所有错误，无法诊断 Hopfield
                # 检索失败（如 NaN query、空记忆、维度不匹配等）。
                # 现在记录到 logger，便于生产环境调试。
                _LOGGER.warning(
                    "autobiographical_memory retrieve_topk failed: %s",
                    exc,
                    exc_info=True,
                )
                return NarrativeResult(
                    narrative="检索失败。",
                    retrieved_episodes=[],
                    similarity=0.0,
                    metadata={
                        "error": "retrieve_failed",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    },
                )

            # 收集对应的 SelfEpisode
            retrieved: list[SelfEpisode] = []
            for idx in top_k_indices:
                idx_int = int(idx)
                if 0 <= idx_int < len(self._episodes):
                    retrieved.append(self._episodes[idx_int])

            max_sim = float(top_k_sims[0]) if len(top_k_sims) > 0 else 0.0

            # 生成叙述
            if text_decoder is not None and hasattr(text_decoder, "decode"):
                try:
                    narrative = text_decoder.decode(retrieved)
                except Exception:
                    narrative = self._template_narrative(retrieved, max_sim)
            else:
                narrative = self._template_narrative(retrieved, max_sim)

            return NarrativeResult(
                narrative=narrative,
                retrieved_episodes=retrieved,
                similarity=max_sim,
                metadata={
                    "n_retrieved": len(retrieved),
                    "top_k_sims": top_k_sims.tolist(),
                },
            )

    # ---------------------------------------------------------------- #
    # 模板化叙述生成
    # ---------------------------------------------------------------- #
    def _template_narrative(
        self, episodes: list[SelfEpisode], similarity: float
    ) -> str:
        """用模板生成自传体叙述（无文本解码器时的回退）。"""
        if not episodes:
            return "我没有相关的过去经历可以回忆。"

        lines: list[str] = []
        lines.append(f"回忆起 {len(episodes)} 个相关经历（相似度 {similarity:.3f}）：")
        for i, ep in enumerate(episodes):
            # 提取自我图式预测
            pred = ep.self_prediction
            pred_attn = pred.get("predicted_attention", [])
            pred_action = pred.get("predicted_action", "?")
            # 提取实际结果
            outcome = ep.actual_outcome
            actual_action = outcome.get("actual_action", "?")
            fe = ep.free_energy
            # 简化叙述
            attn_str = (
                f"关注模块 {np.argmax(pred_attn)}" if len(pred_attn) > 0
                else "关注未知"
            )
            lines.append(
                f"  [{i+1}] 步骤 {ep.step}（{ep.modality} 阶段）："
                f"我当时{attn_str}，预测动作 {pred_action}，"
                f"实际做了 {actual_action}，自由能 {fe:.3f}。"
            )
        # 总结
        if similarity > 0.8:
            lines.append("当前情境与过去高度相似，我可以可靠地预测。")
        elif similarity > 0.5:
            lines.append("当前情境与过去部分相关，需要谨慎预测。")
        else:
            lines.append("当前情境较为新颖，缺乏可借鉴的经历。")
        return "\n".join(lines)

    # ---------------------------------------------------------------- #
    # 离线巩固
    # ---------------------------------------------------------------- #
    def consolidate(self) -> dict[str, Any]:
        """离线巩固：用伪逆重构记忆矩阵，强化自我连续性。

        Returns
        -------
        dict
            巩固诊断
        """
        with self._lock:
            n_before = self.hopfield.size
            # A4 修复诊断：记录巩固前后 keys 矩阵的 Frobenius 范数变化，
            # 用于验证 consolidate 真正修改了检索矩阵（而非 no-op）。
            # 原 bug：retrieve_topk 用 _values 而 retrieve 用 _keys，
            # 所以 update_weights 修改 _keys 后对 retrieve_topk 无效。
            # 修复后 retrieve_topk 也用 _keys，故 keys_norm 变化现在
            # 反映在 retrieve_topk 的检索结果上。
            keys_norm_before = 0.0
            if (
                self.hopfield._keys is not None
                and self.hopfield._keys.size > 0
            ):
                keys_norm_before = float(
                    np.linalg.norm(self.hopfield._keys)
                )
            self.hopfield.update_weights()
            n_after = self.hopfield.size
            keys_norm_after = 0.0
            if (
                self.hopfield._keys is not None
                and self.hopfield._keys.size > 0
            ):
                keys_norm_after = float(
                    np.linalg.norm(self.hopfield._keys)
                )
            # 为已存储片段填充叙述（如果还没有）
            filled = 0
            for ep in self._episodes:
                if not ep.narrative:
                    ep.narrative = self._template_narrative([ep], 1.0)
                    filled += 1
            return {
                "episodes_before": n_before,
                "episodes_after": n_after,
                "narratives_filled": filled,
                "consolidated": True,
                "keys_norm_before": keys_norm_before,
                "keys_norm_after": keys_norm_after,
                "keys_norm_delta": (
                    keys_norm_after - keys_norm_before
                ),
            }

    # ---------------------------------------------------------------- #
    # 查询
    # ---------------------------------------------------------------- #
    def get_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "n_episodes": len(self._episodes),
                "capacity": self.capacity,
                "fill_ratio": len(self._episodes) / max(self.capacity, 1),
                "hopfield_beta": float(self.hopfield.beta),
                "is_consolidated": self.hopfield._consolidated,
                "recent_steps": [e.step for e in self._episodes[-5:]],
                "recent_modalities": [e.modality for e in self._episodes[-5:]],
            }

    @property
    def size(self) -> int:
        return len(self._episodes)

    def reset(self) -> None:
        with self._lock:
            self._episodes.clear()
            self.hopfield.clear()
