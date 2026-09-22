# src/zero_data_model/experiment/experiment_planner.py
"""贝叶斯实验规划（第六阶段，任务 6.1）。

模型从被动学习进化为主动实验者：维护候选干预列表，利用世界模型
预测每个干预的预期信息增益，选择最大化信息增益的干预执行。

信息增益启发式：``gain = current_uncertainty * intervention_complexity * uniform(0.5, 1.5)``
其中 intervention_complexity 与干预涉及的变量数成正比。实际执行后，
用自由能下降量作为实际信息增益的代理，更新参数不确定性先验。
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

# 参数不确定性的下限：避免衰减到恰好 0 而冻结探索（信息增益估计恒为 0）。
MIN_UNCERTAINTY = 1e-6


# ------------------------------------------------------------------ #
# 候选实验
# ------------------------------------------------------------------ #
@dataclass
class CandidateExperiment:
    """一个候选干预实验。"""
    name: str
    intervention: dict[str, Any]
    predicted_info_gain: float = 0.0
    executed: bool = False
    result_fe_before: float = 0.0
    result_fe_after: float = 0.0
    actual_gain: float = 0.0


# ------------------------------------------------------------------ #
# 贝叶斯实验规划器
# ------------------------------------------------------------------ #
class BayesianExperimentPlanner:
    """主动实验设计：选择最大化预期信息增益的干预。

    Parameters
    ----------
    eval_interval : int
        每多少步评估一次（默认 1000，必须 > 0）。
    seed : int
        随机种子。
    max_candidates : int
        候选实验历史的上限（超出后丢弃最旧者，使用 deque(maxlen=...)）。
    max_history : int
        执行结果历史的上限（超出后丢弃最旧者）。
    """

    def __init__(
        self,
        eval_interval: int = 1000,
        seed: int = 42,
        max_candidates: int = 256,
        max_history: int = 1024,
    ) -> None:
        # fail fast：eval_interval <= 0 会在 evaluate() 中触发除零/取模错误。
        if eval_interval <= 0:
            raise ValueError(
                f"eval_interval 必须为正整数，收到 {eval_interval}"
            )
        self._eval_interval = int(eval_interval)
        # np.random.Generator 非线程安全，所有 RNG 访问须在 _lock 下进行。
        self._rng = np.random.default_rng(seed)
        self._lock = threading.RLock()
        # 使用有界 deque 防止无界增长。
        self.candidates: deque[CandidateExperiment] = deque(maxlen=max_candidates)
        self._history: deque[dict[str, Any]] = deque(maxlen=max_history)
        self._step_count: int = 0
        # 参数不确定性先验（1.0 = 完全不确定）。
        self._param_uncertainty: float = 1.0

    # ------------------------------------------------------------------ #
    # 候选管理
    # ------------------------------------------------------------------ #
    def register_candidate(self, name: str, intervention: dict[str, Any]) -> None:
        """注册一个候选干预实验。"""
        with self._lock:
            self.candidates.append(
                CandidateExperiment(name=name, intervention=intervention)
            )

    def default_candidates(self, dim: int = 64) -> None:
        """注册一组默认候选实验（物理沙盒场景）。"""
        with self._lock:
            self.register_candidate("perturb_gravity", {"type": "scalar", "factor": 1.5})
            self.register_candidate("add_object", {"type": "vector", "dim": dim})
            self.register_candidate("change_friction", {"type": "scalar", "factor": 0.5})
            self.register_candidate("rotate_scene", {"type": "matrix"})
            self.register_candidate("remove_object", {"type": "index"})

    # ------------------------------------------------------------------ #
    # 信息增益估计
    # ------------------------------------------------------------------ #
    def estimate_info_gain(
        self, candidate: CandidateExperiment, current_uncertainty: float
    ) -> float:
        """启发式估计候选实验的预期信息增益。

        gain = current_uncertainty * (1 - 1/(1+|intervention|)) * uniform(0.5, 1.5)
        """
        with self._lock:
            complexity = max(1, len(candidate.intervention))
            gain = (
                current_uncertainty
                * (1.0 - 1.0 / (1.0 + complexity))
                * float(self._rng.uniform(0.5, 1.5))
            )
            candidate.predicted_info_gain = float(gain)
            return float(gain)

    def select_best(
        self, current_uncertainty: float
    ) -> CandidateExperiment | None:
        """选择预期信息增益最大的未执行候选。"""
        with self._lock:
            unexecuted = [c for c in self.candidates if not c.executed]
            if not unexecuted:
                return None
            for c in unexecuted:
                self.estimate_info_gain(c, current_uncertainty)
            return max(unexecuted, key=lambda c: c.predicted_info_gain)

    # ------------------------------------------------------------------ #
    # 执行与记录
    # ------------------------------------------------------------------ #
    def record_result(
        self,
        candidate: CandidateExperiment,
        fe_before: float,
        fe_after: float,
    ) -> None:
        """记录实验执行结果，更新参数不确定性先验。"""
        with self._lock:
            candidate.executed = True
            candidate.result_fe_before = float(fe_before)
            candidate.result_fe_after = float(fe_after)
            candidate.actual_gain = float(fe_before - fe_after)  # 正=FE下降=好
            self._history.append(
                {
                    "name": candidate.name,
                    "predicted_gain": candidate.predicted_info_gain,
                    "actual_gain": candidate.actual_gain,
                    "fe_before": candidate.result_fe_before,
                    "fe_after": candidate.result_fe_after,
                }
            )
            # 贝叶斯更新参数不确定性：实际增益越大，不确定性下降越多。
            gain = max(0.0, candidate.actual_gain)
            decay = 1.0 - gain / (gain + 1.0)
            # 钳制到下限，防止衰减到恰好 0 而冻结探索。
            self._param_uncertainty = max(
                self._param_uncertainty * decay, MIN_UNCERTAINTY
            )

    def evaluate(
        self, current_uncertainty: float, step: int
    ) -> dict[str, Any] | None:
        """每 eval_interval 步评估一次，返回最优候选。"""
        with self._lock:
            self._step_count = step
            # 防御性：__init__ 已校验 eval_interval > 0，此处再守一次。
            if self._eval_interval <= 0:
                return None
            if step % self._eval_interval != 0:
                return None
            best = self.select_best(current_uncertainty)
            if best is None:
                return None
            return {
                "experiment": best.name,
                "intervention": best.intervention,
                "predicted_gain": best.predicted_info_gain,
            }

    # ------------------------------------------------------------------ #
    # 属性
    # ------------------------------------------------------------------ #
    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            executed = [c for c in self.candidates if c.executed]
            gains = [c.actual_gain for c in executed]
            return {
                "n_candidates": len(self.candidates),
                "n_executed": len(executed),
                "mean_actual_gain": float(np.mean(gains)) if gains else 0.0,
                "param_uncertainty": float(self._param_uncertainty),
            }
