# src/zero_data_model/experiment/hypothesis_tester.py
"""假设生成与验证（第六阶段，任务 6.2）。

从因果图中提取不确定边（边强度接近 0），生成"A 导致 B" vs
"A 不导致 B"的候选假设。通过设计实验固定 A、观察 B 的方差，
用贝叶斯因子更新假设的后验几率。
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


# ------------------------------------------------------------------ #
# 假设
# ------------------------------------------------------------------ #
@dataclass
class Hypothesis:
    """一个因果假设。"""
    statement: str
    cause_var: int
    effect_var: int
    direction: str  # "causes" 或 "no_effect"
    prior_odds: float = 1.0
    bayes_factor: float = 1.0
    posterior_odds: float = 1.0
    tested: bool = False
    supported: bool = False


# ------------------------------------------------------------------ #
# 假设检验器
# ------------------------------------------------------------------ #
class HypothesisTester:
    """从因果图生成假设并用实验数据检验。

    Parameters
    ----------
    edge_threshold : float
        边强度低于此值视为"接近 0"（生成 no_effect 假设）。
    seed : int
        随机种子。
    max_hypotheses : int
        历史假设上限（超出后丢弃最旧者，使用 deque(maxlen=...)）。
    """

    def __init__(
        self,
        edge_threshold: float = 0.05,
        seed: int = 42,
        max_hypotheses: int = 256,
    ) -> None:
        self._edge_threshold = float(edge_threshold)
        # np.random.Generator 非线程安全，所有 RNG 访问须在 _lock 下进行。
        self._rng = np.random.default_rng(seed)
        self._lock = threading.RLock()
        # 使用有界 deque 防止无界增长。
        self.hypotheses: deque[Hypothesis] = deque(maxlen=max_hypotheses)

    # ------------------------------------------------------------------ #
    # 假设生成
    # ------------------------------------------------------------------ #
    def generate_from_causal_graph(
        self, transition_matrix: np.ndarray, max_hyps: int = 10
    ) -> list[Hypothesis]:
        """从转移矩阵提取不确定边，生成候选假设。

        - |transition[i,j]| < edge_threshold → "no_effect" 假设
        - |transition[i,j]| > 0.2 → "causes" 假设
        """
        if transition_matrix is None:
            raise TypeError(
                "generate_from_causal_graph requires non-None transition_matrix"
            )
        # 形状校验：必须是二维方阵，避免 [i, j] 索引出错。
        tm = np.asarray(transition_matrix, dtype=float)
        if tm.ndim != 2:
            raise ValueError(
                f"transition_matrix 必须为 2-D，收到 {tm.ndim}-D"
            )
        if tm.shape[0] != tm.shape[1]:
            raise ValueError(
                f"transition_matrix 必须为方阵，收到 shape {tm.shape}"
            )
        with self._lock:
            new_hyps: list[Hypothesis] = []
            n = tm.shape[0]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    w = float(tm[i, j])
                    if abs(w) < self._edge_threshold:
                        hyp = Hypothesis(
                            statement=f"var {i} does NOT cause var {j}",
                            cause_var=i,
                            effect_var=j,
                            direction="no_effect",
                        )
                    elif abs(w) > 0.2:
                        hyp = Hypothesis(
                            statement=f"var {i} causes var {j}",
                            cause_var=i,
                            effect_var=j,
                            direction="causes",
                        )
                    else:
                        continue
                    new_hyps.append(hyp)
                    if len(new_hyps) >= max_hyps:
                        self.hypotheses.extend(new_hyps)
                        return new_hyps
            self.hypotheses.extend(new_hyps)
            return new_hyps

    # ------------------------------------------------------------------ #
    # 实验设计与执行
    # ------------------------------------------------------------------ #
    def design_experiment(self, hyp: Hypothesis) -> dict[str, Any]:
        """设计一个实验来检验假设。"""
        with self._lock:
            return {
                "fix_var": hyp.cause_var,
                "observe_var": hyp.effect_var,
                "n_trials": 10,
                "intervention_value": float(self._rng.normal()),
            }

    def update_with_data(self, hyp: Hypothesis, observations: np.ndarray) -> None:
        """用实验数据更新假设的后验几率。

        方差小（<0.01）→ 支持"causes"（BF=5）。
        方差大（>0.1）→ 支持"no_effect"（BF=5）。
        """
        if observations is None:
            raise TypeError("update_with_data requires non-None observations")
        with self._lock:
            obs = np.asarray(observations, dtype=float).flatten()
            # 观测不足（n<2）时方差无意义，按 0.0（无信息）处理，避免给出误导性的 0 方差。
            variance = 0.0 if obs.size < 2 else float(np.var(obs))
            if variance < 0.01:
                # 固定 A 后 B 几乎不变 → A 导致 B。
                hyp.bayes_factor = 5.0 if hyp.direction == "causes" else 0.2
            elif variance > 0.1:
                # 固定 A 后 B 仍波动 → A 不导致 B。
                hyp.bayes_factor = 5.0 if hyp.direction == "no_effect" else 0.2
            else:
                hyp.bayes_factor = 1.0  # 不确定
            hyp.posterior_odds = hyp.prior_odds * hyp.bayes_factor
            hyp.tested = True
            hyp.supported = hyp.posterior_odds > 1.0

    # ------------------------------------------------------------------ #
    # 配对假设生成（第二阶段 §3.2）
    # ------------------------------------------------------------------ #
    def generate_paired_hypotheses(
        self, transition_matrix: np.ndarray, max_pairs: int = 10
    ) -> list[dict[str, Any]]:
        """从因果图不确定边生成配对假设 H0 vs H1。

        对每条不确定边 ``A→B``（强度接近 0），同时生成：
        - ``H0``: A 不导致 B（``direction="no_effect"``）
        - ``H1``: A 导致 B（``direction="causes"``）

        两个假设共享同一 ``(cause_var, effect_var)``，便于后续用
        贝叶斯因子 ``BF = P(data|H1) / P(data|H0)`` 直接比较。

        Parameters
        ----------
        transition_matrix:
            转移矩阵（方阵）。
        max_pairs:
            最大配对数量。

        Returns
        -------
        list[dict]
            每个元素为 ``{"h0": Hypothesis, "h1": Hypothesis,
            "cause_var": int, "effect_var": int}``。
        """
        if transition_matrix is None:
            raise TypeError(
                "generate_paired_hypotheses requires non-None transition_matrix"
            )
        tm = np.asarray(transition_matrix, dtype=float)
        if tm.ndim != 2 or tm.shape[0] != tm.shape[1]:
            raise ValueError(
                f"transition_matrix 必须为方阵，收到 shape {tm.shape}"
            )
        with self._lock:
            pairs: list[dict[str, Any]] = []
            n = tm.shape[0]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    w = float(tm[i, j])
                    # 不确定边：强度接近 0 → 生成配对。
                    if abs(w) < self._edge_threshold:
                        h0 = Hypothesis(
                            statement=f"var {i} does NOT cause var {j}",
                            cause_var=i,
                            effect_var=j,
                            direction="no_effect",
                        )
                        h1 = Hypothesis(
                            statement=f"var {i} causes var {j}",
                            cause_var=i,
                            effect_var=j,
                            direction="causes",
                        )
                        self.hypotheses.extend([h0, h1])
                        pairs.append({
                            "h0": h0,
                            "h1": h1,
                            "cause_var": i,
                            "effect_var": j,
                        })
                        if len(pairs) >= max_pairs:
                            return pairs
            return pairs

    def update_causal_graph(
        self,
        transition_matrix: np.ndarray,
        cause_var: int,
        effect_var: int,
        bayes_factor: float,
    ) -> np.ndarray:
        """根据贝叶斯因子更新因果图（转移矩阵）。

        ``BF > 1`` → 增强 ``cause→effect`` 边；
        ``BF < 1`` → 减弱 ``cause→effect`` 边。

        Parameters
        ----------
        transition_matrix:
            当前转移矩阵（会被复制后修改）。
        cause_var:
            原因变量索引。
        effect_var:
            结果变量索引。
        bayes_factor:
            贝叶斯因子 ``BF = P(data|H1) / P(data|H0)``。

        Returns
        -------
        np.ndarray
            更新后的转移矩阵（新数组，不修改输入）。
        """
        tm = np.asarray(transition_matrix, dtype=float).copy()
        if tm.ndim != 2 or tm.shape[0] != tm.shape[1]:
            raise ValueError(
                f"transition_matrix 必须为方阵，收到 shape {tm.shape}"
            )
        n = tm.shape[0]
        if not (0 <= cause_var < n and 0 <= effect_var < n):
            raise ValueError(
                f"cause_var/effect_var 越界: {cause_var}/{effect_var}, n={n}"
            )
        with self._lock:
            # BF > 1 → 增强边（log scale）。
            # 更新量 = log(BF) * learning_rate
            lr = 0.1
            delta = float(np.log(max(bayes_factor, 1e-10))) * lr
            tm[cause_var, effect_var] = float(
                np.clip(tm[cause_var, effect_var] + delta, -1.0, 1.0)
            )
            return tm

    # ------------------------------------------------------------------ #
    # 查询
    # ------------------------------------------------------------------ #
    def get_supported(self) -> list[Hypothesis]:
        """返回已检验且被支持的假设。"""
        with self._lock:
            return [h for h in self.hypotheses if h.tested and h.supported]

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            tested = [h for h in self.hypotheses if h.tested]
            return {
                "n_hypotheses": len(self.hypotheses),
                "n_tested": len(tested),
                "n_supported": len(self.get_supported()),
            }
