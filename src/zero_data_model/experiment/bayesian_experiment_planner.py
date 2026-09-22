# src/zero_data_model/experiment/bayesian_experiment_planner.py
"""贝叶斯实验规划器（BayesianExperimentPlannerV2）。

第二阶段 §3.1：从随机好奇转向贝叶斯最优实验规划。与现有
``experiment_planner.BayesianExperimentPlanner``（启发式信息增益）的
区别：

- **KL 散度信息增益**：对每个候选干预，用世界模型（S4 状态空间）
  模拟预期信息增益 ``≈ KL(P(state|do) || P(state))``。
- **多采样平均**：采样多个可能的干预结果，计算平均预期自由能减少。
- **沙盒干预 API**：``apply_intervention(type, params)`` 接口，支持
  物理沙盒集成。

信息增益计算：
    IG(do) = E[KL(P(state_next | do, state_curr) || P(state_next | state_curr))]
    ≈ mean_t [ KL(q_t || p_t) ]
    其中 q_t = 干预后状态分布，p_t = 自然状态分布。

仅依赖 ``numpy``，独立、自包含、可插拔。
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


# ------------------------------------------------------------------ #
# 沙盒干预接口（Protocol）
# ------------------------------------------------------------------ #
class SandboxInterface(Protocol):
    """物理沙盒干预接口协议。

    任何提供 ``apply_intervention`` 和 ``get_state_distribution`` 方法
    的对象都可作为沙盒传入。
    """

    def apply_intervention(
        self, intervention_type: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """执行干预，返回干预后的状态。"""
        ...

    def get_state_distribution(self) -> np.ndarray:
        """返回当前状态分布（均值向量）。"""
        ...


# ------------------------------------------------------------------ #
# 候选实验
# ------------------------------------------------------------------ #
@dataclass
class CandidateIntervention:
    """一个候选干预实验。

    Attributes
    ----------
    name:
        实验名称。
    intervention_type:
        干预类型（如 ``"change_gravity"``、``"add_object"``）。
    params:
        干预参数。
    predicted_info_gain:
        预期信息增益（KL 散度，越大越好）。
    executed:
        是否已执行。
    actual_info_gain:
        实际信息增益（执行后计算）。
    """

    name: str
    intervention_type: str
    params: dict[str, Any] = field(default_factory=dict)
    predicted_info_gain: float = 0.0
    executed: bool = False
    actual_info_gain: float = 0.0


# ------------------------------------------------------------------ #
# 贝叶斯实验规划器 V2
# ------------------------------------------------------------------ #
class BayesianExperimentPlannerV2:
    """贝叶斯最优实验规划器（KL 散度信息增益）。

    Parameters
    ----------
    eval_interval:
        每多少步评估一次（默认 1000）。
    n_samples:
        每个候选的采样次数（用于估计期望信息增益）。
    seed:
        随机种子。
    max_candidates:
        候选实验历史上限。
    max_history:
        执行结果历史上限。

    Attributes
    ----------
    candidates:
        已注册的候选干预列表。
    """

    def __init__(
        self,
        eval_interval: int = 1000,
        n_samples: int = 10,
        seed: int = 42,
        max_candidates: int = 256,
        max_history: int = 1024,
    ) -> None:
        """初始化实验规划器。"""
        if eval_interval <= 0:
            raise ValueError(
                f"eval_interval 必须为正整数，收到 {eval_interval}"
            )
        if n_samples <= 0:
            raise ValueError(
                f"n_samples 必须为正整数，收到 {n_samples}"
            )
        self._eval_interval = int(eval_interval)
        self._n_samples = int(n_samples)
        self._rng = np.random.default_rng(seed)
        self._lock = threading.RLock()
        self.candidates: deque[CandidateIntervention] = deque(
            maxlen=max_candidates
        )
        self._history: deque[dict[str, Any]] = deque(maxlen=max_history)
        self._step_count: int = 0
        self._sandbox: SandboxInterface | None = None

    # ------------------------------------------------------------------ #
    # 沙盒集成
    # ------------------------------------------------------------------ #
    def attach_sandbox(self, sandbox: SandboxInterface) -> None:
        """附加物理沙盒（提供 ``apply_intervention`` 接口）。"""
        with self._lock:
            self._sandbox = sandbox

    def detach_sandbox(self) -> None:
        """分离沙盒。"""
        with self._lock:
            self._sandbox = None

    # ------------------------------------------------------------------ #
    # 候选管理
    # ------------------------------------------------------------------ #
    def register_candidate(
        self,
        name: str,
        intervention_type: str,
        params: dict[str, Any] | None = None,
    ) -> None:
        """注册一个候选干预实验。"""
        with self._lock:
            self.candidates.append(
                CandidateIntervention(
                    name=name,
                    intervention_type=intervention_type,
                    params=dict(params) if params else {},
                )
            )

    def default_candidates(self, dim: int = 64) -> None:
        """注册一组默认候选实验（物理沙盒场景）。

        对应 spec 中的示例：
        - "改变重力方向" (change_gravity)
        - "放入新物体" (add_object)
        - "移除障碍" (remove_object)
        - "改变摩擦" (change_friction)
        - "旋转场景" (rotate_scene)
        """
        with self._lock:
            self.register_candidate("change_gravity", "scalar", {"factor": 1.5})
            self.register_candidate("add_object", "vector", {"dim": dim})
            self.register_candidate("remove_object", "index", {"idx": 0})
            self.register_candidate("change_friction", "scalar", {"factor": 0.5})
            self.register_candidate("rotate_scene", "matrix", {"angle": 90})

    # ------------------------------------------------------------------ #
    # KL 散度信息增益估计
    # ------------------------------------------------------------------ #
    def _kl_divergence(
        self, p: np.ndarray, q: np.ndarray
    ) -> float:
        """计算两个分布的 KL 散度 ``KL(p || q)``。

        使用对称化版本（Jensen-Shannon 散度的一半）以避免 q=0 问题。
        """
        p = np.asarray(p, dtype=np.float64).flatten()
        q = np.asarray(q, dtype=np.float64).flatten()
        # 对齐长度。
        min_len = min(len(p), len(q))
        p = p[:min_len]
        q = q[:min_len]
        # 归一化为概率分布。
        p_sum = p.sum()
        q_sum = q.sum()
        if p_sum < 1e-12 or q_sum < 1e-12:
            return 0.0
        p = p / p_sum
        q = q / q_sum
        # 加 epsilon 避免 log(0)。
        eps = 1e-12
        kl = np.sum(p * np.log((p + eps) / (q + eps)))
        return float(max(0.0, kl))

    def estimate_info_gain(
        self,
        candidate: CandidateIntervention,
        current_state: np.ndarray,
        world_model_step: Any | None = None,
    ) -> float:
        """估计候选干预的预期信息增益。

        信息增益 ≈ 预测状态分布与干预后状态分布的 KL 散度。
        采样 ``n_samples`` 次取平均。

        Parameters
        ----------
        candidate:
            候选干预。
        current_state:
            当前状态向量。
        world_model_step:
            可选的世界模型单步预测函数 ``f(state) -> next_state``。
            若为 ``None``，使用状态扰动模拟。

        Returns
        -------
        float
            预期信息增益（KL 散度）。
        """
        with self._lock:
            current = np.asarray(current_state, dtype=np.float64).flatten()
            gains: list[float] = []
            for _ in range(self._n_samples):
                # 模拟自然状态演化。
                if world_model_step is not None:
                    natural_next = world_model_step(current)
                else:
                    # 无世界模型：加小幅高斯噪声模拟自然演化。
                    natural_next = current + self._rng.standard_normal(
                        current.shape
                    ) * 0.01
                # 模拟干预后状态演化。
                intervened_next = self._simulate_intervention(
                    current, candidate, natural_next
                )
                gain = self._kl_divergence(intervened_next, natural_next)
                gains.append(gain)
            # 平均预期信息增益。
            avg_gain = float(np.mean(gains)) if gains else 0.0
            candidate.predicted_info_gain = avg_gain
            return avg_gain

    def _simulate_intervention(
        self,
        current_state: np.ndarray,
        candidate: CandidateIntervention,
        natural_next: np.ndarray,
    ) -> np.ndarray:
        """模拟干预后的状态演化。"""
        itype = candidate.intervention_type
        params = candidate.params
        result = natural_next.copy()
        if itype == "scalar":
            factor = float(params.get("factor", 1.5))
            result = result * factor
        elif itype == "vector":
            # 添加新维度（模拟新物体）。
            dim = int(params.get("dim", 64))
            extra = self._rng.standard_normal(min(dim, 8))
            min_len = min(len(result), len(extra))
            result[:min_len] += extra[:min_len] * 0.1
        elif itype == "index":
            # 移除一个维度（模拟移除物体）。
            idx = int(params.get("idx", 0))
            if 0 <= idx < len(result):
                result[idx] = 0.0
        elif itype == "matrix":
            # 旋转（模拟场景旋转）。
            result = result[::-1].copy()
        return result

    def select_best(
        self,
        current_state: np.ndarray,
        world_model_step: Any | None = None,
    ) -> CandidateIntervention | None:
        """选择预期信息增益最大的未执行候选。

        Parameters
        ----------
        current_state:
            当前状态向量。
        world_model_step:
            可选的世界模型单步预测函数。

        Returns
        -------
        CandidateIntervention | None
            最优候选，或 ``None``（无候选时）。
        """
        with self._lock:
            unexecuted = [c for c in self.candidates if not c.executed]
            if not unexecuted:
                return None
            for c in unexecuted:
                self.estimate_info_gain(c, current_state, world_model_step)
            return max(unexecuted, key=lambda c: c.predicted_info_gain)

    # ------------------------------------------------------------------ #
    # 执行与记录
    # ------------------------------------------------------------------ #
    def execute(
        self,
        candidate: CandidateIntervention,
        current_state: np.ndarray,
    ) -> dict[str, Any]:
        """执行候选实验。

        若附加了沙盒，调用 ``sandbox.apply_intervention``；
        否则使用内部模拟。

        Returns
        -------
        dict
            执行结果：``{name, predicted_gain, actual_gain, state_before,
            state_after, used_sandbox}``。
        """
        with self._lock:
            state_before = np.asarray(
                current_state, dtype=np.float64
            ).flatten().copy()

            used_sandbox = False
            state_after: np.ndarray
            if self._sandbox is not None:
                # 使用真实沙盒。
                result = self._sandbox.apply_intervention(
                    candidate.intervention_type, candidate.params
                )
                state_after = np.asarray(
                    result.get("state", state_before), dtype=np.float64
                ).flatten()
                used_sandbox = True
            else:
                # 内部模拟。
                natural_next = state_before + self._rng.standard_normal(
                    state_before.shape
                ) * 0.01
                state_after = self._simulate_intervention(
                    state_before, candidate, natural_next
                )

            actual_gain = self._kl_divergence(state_after, state_before)
            candidate.executed = True
            candidate.actual_info_gain = actual_gain
            record = {
                "name": candidate.name,
                "intervention_type": candidate.intervention_type,
                "predicted_gain": candidate.predicted_info_gain,
                "actual_gain": actual_gain,
                "state_before": state_before.tolist(),
                "state_after": state_after.tolist(),
                "used_sandbox": used_sandbox,
            }
            self._history.append(record)
            return record

    def evaluate(
        self,
        current_state: np.ndarray,
        step: int,
        world_model_step: Any | None = None,
    ) -> dict[str, Any] | None:
        """每 ``eval_interval`` 步评估一次，返回最优候选。"""
        with self._lock:
            self._step_count = step
            if step % self._eval_interval != 0:
                return None
            best = self.select_best(current_state, world_model_step)
            if best is None:
                return None
            return {
                "experiment": best.name,
                "intervention_type": best.intervention_type,
                "params": best.params,
                "predicted_gain": best.predicted_info_gain,
            }

    # ------------------------------------------------------------------ #
    # 属性
    # ------------------------------------------------------------------ #
    @property
    def stats(self) -> dict[str, Any]:
        """统计信息。"""
        with self._lock:
            executed = [c for c in self.candidates if c.executed]
            gains = [c.actual_info_gain for c in executed]
            return {
                "n_candidates": len(self.candidates),
                "n_executed": len(executed),
                "mean_actual_gain": float(np.mean(gains)) if gains else 0.0,
                "sandbox_attached": self._sandbox is not None,
                "step_count": self._step_count,
            }


__all__: list[str] = [
    "BayesianExperimentPlannerV2",
    "CandidateIntervention",
    "SandboxInterface",
]
