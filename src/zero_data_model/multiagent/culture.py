"""文化传承：跨代际的文化演化与知识累积曲线。"""

from __future__ import annotations

import copy
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Generation:
    """单代际的快照：知识规模、平均自由能与学习步数。"""

    gen_id: int
    knowledge_graph_size: int
    mean_free_energy: float
    n_steps: int
    weights_snapshot: dict | None = None


class CulturePropagation:
    """跨代际文化演化跟踪器。

    记录每一代的“知识图规模 / 学习步数”比值，比较相邻代际的比值得到
    ``cultural_acceleration``：>1 表示文化让学习加速（后代学得更快）。
    支持以变异方式把上一代权重遗传给下一代。

    Parameters
    ----------
    max_generations : int
        代际历史的上限（超出后丢弃最旧者，使用 deque(maxlen=...)）。
    seed : int
        随机种子。
    """

    def __init__(
        self,
        max_generations: int = 256,
        seed: int = 42,
    ) -> None:
        self._lock = threading.RLock()
        # np.random.Generator 非线程安全，所有 RNG 访问须在 _lock 下进行。
        self._rng = np.random.default_rng(seed)
        # 使用有界 deque 防止代际历史无界增长。
        self.generations: deque[Generation] = deque(maxlen=max_generations)
        self._current_gen: int = 0

    # ------------------------------------------------------------------
    # 代际记录
    # ------------------------------------------------------------------
    def record_generation(
        self,
        knowledge_graph_size: int,
        mean_free_energy: float,
        n_steps: int,
        weights: dict | None = None,
    ) -> Generation:
        """记录一代的快照，追加到 ``self.generations`` 并推进代际计数。"""
        with self._lock:
            gen = Generation(
                gen_id=self._current_gen,
                knowledge_graph_size=knowledge_graph_size,
                mean_free_energy=mean_free_energy,
                n_steps=n_steps,
                weights_snapshot=weights,
            )
            self.generations.append(gen)
            self._current_gen += 1
            return gen

    # ------------------------------------------------------------------
    # 权重遗传
    # ------------------------------------------------------------------
    def inherit_weights(
        self, parent_weights: dict, mutation_rate: float = 0.01
    ) -> dict:
        """深拷贝父代权重，并对每个 ndarray 值叠加小幅高斯噪声。

        ndarray 值先转为 float 拷贝再叠加噪声（避免 int-dtype 数组原地
        ``+=`` 浮点噪声时被静默截断）；非 ndarray 值使用
        :func:`copy.deepcopy` 深拷贝，确保子代与父代无共享引用。
        """
        with self._lock:
            child: dict[str, Any] = {}
            for key, value in parent_weights.items():
                if isinstance(value, np.ndarray):
                    # astype(float, copy=True) 同时完成拷贝与浮点化，
                    # 防止 int-dtype 数组 += 浮点噪声时被截断。
                    arr = value.astype(float, copy=True)
                    arr += self._rng.normal(0, mutation_rate, arr.shape)
                    child[key] = arr
                else:
                    # 非 ndarray：深拷贝以匹配文档承诺（dict/list 等容器
                    # 不与父代共享内部引用）。
                    child[key] = copy.deepcopy(value)
            return child

    # ------------------------------------------------------------------
    # 文化加速度
    # ------------------------------------------------------------------
    def compute_cultural_acceleration(self) -> float:
        """计算相邻代际“知识图规模/步数”比值的均值。

        比值 > 1 表示后代在更少步数里积累了更多知识，即文化加速了学习。
        不足两代时返回 1.0（中性基线）。
        """
        with self._lock:
            gens = list(self.generations)
            if len(gens) < 2:
                return 1.0
            ratios: list[float] = []
            for i in range(len(gens) - 1):
                prev = gens[i]
                nxt = gens[i + 1]
                if prev.n_steps <= 0 or nxt.n_steps <= 0:
                    continue
                prev_rate = prev.knowledge_graph_size / prev.n_steps
                nxt_rate = nxt.knowledge_graph_size / nxt.n_steps
                if prev_rate == 0:
                    continue
                ratios.append(nxt_rate / prev_rate)
            if not ratios:
                return 1.0
            return float(np.mean(ratios))

    # ------------------------------------------------------------------
    # 知识曲线
    # ------------------------------------------------------------------
    def get_knowledge_curve(self) -> list[tuple[int, int]]:
        """返回 [(gen_id, knowledge_graph_size), ...]，便于绘图。"""
        with self._lock:
            return [
                (gen.gen_id, gen.knowledge_graph_size)
                for gen in self.generations
            ]

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------
    @property
    def stats(self) -> dict:
        """文化传承的汇总统计。"""
        with self._lock:
            latest_kg = (
                self.generations[-1].knowledge_graph_size
                if self.generations
                else 0
            )
            return {
                "n_generations": len(self.generations),
                "latest_kg_size": latest_kg,
                "acceleration_ratio": self.compute_cultural_acceleration(),
            }

    # ------------------------------------------------------------------
    # Phase 3 扩展：文化快照保存/回放、任务完成时间跟踪
    # ------------------------------------------------------------------
    def save_snapshot(self) -> dict[str, Any]:
        """保存当前文化状态为可序列化字典（供回放）。

        Returns
        -------
        dict
            含 ``generations`` 列表和 ``current_gen`` 计数。
        """
        with self._lock:
            return {
                "current_gen": self._current_gen,
                "generations": [
                    {
                        "gen_id": g.gen_id,
                        "knowledge_graph_size": g.knowledge_graph_size,
                        "mean_free_energy": g.mean_free_energy,
                        "n_steps": g.n_steps,
                    }
                    for g in self.generations
                ],
            }

    def load_snapshot(self, snapshot: dict[str, Any]) -> None:
        """从快照恢复文化状态（清空当前历史）。

        Parameters
        ----------
        snapshot:
            :meth:`save_snapshot` 返回的字典。
        """
        with self._lock:
            self.generations.clear()
            self._current_gen = int(snapshot.get("current_gen", 0))
            for g_dict in snapshot.get("generations", []):
                gen = Generation(
                    gen_id=int(g_dict["gen_id"]),
                    knowledge_graph_size=int(g_dict["knowledge_graph_size"]),
                    mean_free_energy=float(g_dict["mean_free_energy"]),
                    n_steps=int(g_dict["n_steps"]),
                    weights_snapshot=None,  # 快照不保存完整权重
                )
                self.generations.append(gen)

    def get_task_completion_curve(self) -> list[tuple[int, int]]:
        """返回 [(gen_id, n_steps), ...] 任务完成步数曲线。

        步数越少表示该代学得越快（文化加速效应）。
        """
        with self._lock:
            return [(g.gen_id, g.n_steps) for g in self.generations]

    def get_cultural_acceleration_curve(self) -> list[tuple[int, float]]:
        """返回 [(gen_id, acceleration_ratio), ...] 逐代加速度曲线。

        每个点的加速度 = 该代的 知识/步数 比值 / 前一代的比值。
        第一代为 1.0（基线）。
        """
        with self._lock:
            gens = list(self.generations)
            if not gens:
                return []
            curve: list[tuple[int, float]] = [(gens[0].gen_id, 1.0)]
            for i in range(1, len(gens)):
                prev = gens[i - 1]
                curr = gens[i]
                if prev.n_steps <= 0 or curr.n_steps <= 0:
                    curve.append((curr.gen_id, 1.0))
                    continue
                prev_rate = prev.knowledge_graph_size / prev.n_steps
                curr_rate = curr.knowledge_graph_size / curr.n_steps
                ratio = curr_rate / prev_rate if prev_rate > 0 else 1.0
                curve.append((curr.gen_id, float(ratio)))
            return curve
