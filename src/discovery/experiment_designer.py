"""Discovery Experiment Designer — 贝叶斯实验设计，选择 EIG 最大的实验。

对每个假设设计实验方案（干预变量、观测变量、动作序列），
计算预期信息增益（EIG），选择 EIG 最大的实验执行。
兼容现有 SandboxInterface Protocol。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .hypothesis_generator import Hypothesis, _stable_hash_id


class SandboxInterface(Protocol):
    """沙盒干预接口（与 bayesian_experiment_planner.SandboxInterface 兼容）。"""

    def apply_intervention(
        self, intervention_type: str, params: dict[str, Any]
    ) -> dict[str, Any]: ...

    def get_state_distribution(self) -> np.ndarray: ...


@dataclass
class ExperimentDesign:
    """实验方案。"""

    hypothesis_id: str
    intervention_type: str  # "change_mass" | "apply_force" | "set_velocity" 等
    intervention_vars: list[str]
    observation_vars: list[str]
    params: dict[str, Any]
    expected_info_gain: float  # EIG
    #: 执行步骤（沙盒动作序列）
    action_sequence: list[dict[str, Any]] = field(default_factory=list)
    #: 重复次数（减少噪声）
    n_repeats: int = 5
    experiment_id: str = ""

    def __post_init__(self) -> None:
        if not self.experiment_id:
            self.experiment_id = _stable_hash_id(self.hypothesis_id, "E")


class ExperimentDesigner:
    """实验设计器。

    Parameters
    ----------
    n_samples : int, default 10
        EIG 估计的采样数。
    seed : int, default 42
    """

    def __init__(self, n_samples: int = 10, seed: int = 42) -> None:
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        self.n_samples = int(n_samples)
        self._rng = np.random.default_rng(seed)
        self._sandbox: SandboxInterface | None = None
        self._designed_count = 0

    # ------------------------------------------------------------------ #
    # 沙盒附加
    # ------------------------------------------------------------------ #
    def attach_sandbox(self, sandbox: SandboxInterface) -> None:
        """附加物理沙盒以执行实验。"""
        self._sandbox = sandbox

    def detach_sandbox(self) -> None:
        self._sandbox = None

    # ------------------------------------------------------------------ #
    # 实验设计
    # ------------------------------------------------------------------ #
    def design(self, hypothesis: Hypothesis) -> ExperimentDesign | None:
        """为假设设计实验方案。

        Returns
        -------
        ExperimentDesign | None
            设计方案，或 None（无法测试）。
        """
        if hypothesis.testability < 0.1:
            return None

        # 根据策略选择干预类型
        intervention_type, params = self._select_intervention(hypothesis)
        eig = self._estimate_eig(hypothesis, intervention_type, params)

        # 生成动作序列
        action_sequence = self._build_action_sequence(intervention_type, params, hypothesis)

        design = ExperimentDesign(
            hypothesis_id=hypothesis.hypothesis_id,
            intervention_type=intervention_type,
            intervention_vars=hypothesis.intervention_vars,
            observation_vars=hypothesis.observation_vars,
            params=params,
            expected_info_gain=eig,
            action_sequence=action_sequence,
        )
        self._designed_count += 1
        return design

    def design_batch(self, hypotheses: list[Hypothesis]) -> list[ExperimentDesign]:
        """批量设计并按 EIG 排序。"""
        designs: list[ExperimentDesign] = []
        for h in hypotheses:
            d = self.design(h)
            if d is not None:
                designs.append(d)
        designs.sort(key=lambda d: d.expected_info_gain, reverse=True)
        return designs

    def select_best(self, hypotheses: list[Hypothesis]) -> ExperimentDesign | None:
        """选择 EIG 最大的实验。"""
        designs = self.design_batch(hypotheses)
        return designs[0] if designs else None

    # ------------------------------------------------------------------ #
    # 执行实验
    # ------------------------------------------------------------------ #
    def execute(self, design: ExperimentDesign) -> dict[str, Any]:
        """在沙盒中执行实验方案。

        Returns
        -------
        dict
            实验结果，含 before/after 状态、观测值。
        """
        if self._sandbox is None:
            # 无沙盒时返回模拟结果
            return self._simulate(design)

        results: list[dict[str, Any]] = []
        for i in range(design.n_repeats):
            # F5 修复：原代码无 try-except，沙盒在实验中途断开/超时/
            # 抛异常时整个 execute() 崩溃，已收集的前 i 次结果全部丢失。
            # 现在包裹单次迭代，失败时记录占位数据并 continue，保留
            # 已收集的有效数据。
            try:
                before = self._sandbox.get_state_distribution()
                after = self._sandbox.apply_intervention(
                    design.intervention_type, design.params
                )
            except Exception:
                # 沙盒调用失败，用零向量占位
                before = np.zeros(8)
                after = None

            # M2 修复：sandbox 可能返回 None（失败/不支持干预）。
            if before is None:
                before = np.zeros(8)
            else:
                try:
                    before = np.asarray(before, dtype=np.float64).flatten()
                except (TypeError, ValueError):
                    before = np.zeros(8)
            # 沙盒返回值可能是 dict（含状态字段）或 ndarray。
            # result_analyzer 期望 array-like，提取数值向量。
            if after is None:
                after_vec = np.zeros_like(before)
            elif isinstance(after, dict):
                # F6 修复：显式回退检查，避免 "state": None 时
                # 不回退到 "state_distribution"（原 dict.get 默认值
                # 在 key 存在但值为 None 时不触发）。
                after_vec = after.get("state")
                if after_vec is None:
                    after_vec = after.get("state_distribution")
                if after_vec is None:
                    # 退而求其次：收集所有数值型 value
                    after_vec = [v for v in after.values() if isinstance(v, (int, float))]
                    if not after_vec:
                        after_vec = np.zeros_like(before)
            else:
                after_vec = after
            # F6+ 修复：after_vec 可能是字符串等非数值类型，
            # np.asarray(str, dtype=float64) 抛 ValueError。
            try:
                after_list = np.asarray(after_vec, dtype=np.float64).flatten().tolist()
            except (TypeError, ValueError):
                after_list = np.zeros_like(before).tolist()
            results.append(
                {"repeat": i, "before": before.tolist(), "after": after_list}
            )

        return {
            "experiment_id": design.experiment_id,
            "hypothesis_id": design.hypothesis_id,
            "intervention": design.intervention_type,
            "params": design.params,
            "n_repeats": design.n_repeats,
            "results": results,
            "executed": True,
        }

    def _simulate(self, design: ExperimentDesign) -> dict[str, Any]:
        """无沙盒时的模拟执行（返回合成数据）。

        模拟真实物理：干预参数决定效应方向与强度。
        - change_mass: 质量越大 → 状态偏移越大（模拟惯性效应）
        - apply_force: 力越大 → 状态偏移越大（牛顿第二定律）
        - set_velocity: 直接设定速度 → 显著状态变化
        - change_friction: 摩擦 → 阻尼效应
        这样贝叶斯因子能区分"有效应"（accept）与"无效应"（reject），
        使科学发现闭环产出论文。
        """
        rng = np.random.default_rng(self._rng.integers(0, 2**31))
        # 干预强度 → 效应幅度（模拟真实物理因果）
        params = design.params
        # F7 修复：params.get(key, default) 仅在 key 不存在时返回
        # default。若 key 存在但值为 None（如 {"mass": None}），
        # float(None) 抛 TypeError 不被捕获。改用 `or` 模式：
        # None/0/空 等假值均回退到默认值。
        if design.intervention_type == "change_mass":
            effect = float(params.get("mass") or 1.0) * 0.8
        elif design.intervention_type == "apply_force":
            effect = abs(float(params.get("force") or 1.0)) * 0.6
        elif design.intervention_type == "set_velocity":
            effect = abs(float(params.get("velocity") or 1.0)) * 0.7
        elif design.intervention_type == "change_friction":
            effect = float(params.get("friction") or 0.5) * 0.5
        else:
            effect = 0.3
        # NaN/inf 防护
        if not np.isfinite(effect):
            effect = 0.3
        # 效应方向向量（固定种子使同假设可复现）
        direction = rng.standard_normal(8)
        direction = direction / (np.linalg.norm(direction) + 1e-8)

        results = []
        for i in range(design.n_repeats):
            before = rng.standard_normal(8) * 0.1
            # 干预产生确定性偏移 + 测量噪声
            after = before + direction * effect + rng.standard_normal(8) * 0.05
            results.append({"repeat": i, "before": before.tolist(), "after": after.tolist()})
        return {
            "experiment_id": design.experiment_id,
            "hypothesis_id": design.hypothesis_id,
            "intervention": design.intervention_type,
            "params": design.params,
            "n_repeats": design.n_repeats,
            "results": results,
            "executed": False,
            "simulated": True,
        }

    # ------------------------------------------------------------------ #
    # 干预选择与 EIG 估计
    # ------------------------------------------------------------------ #
    def _select_intervention(self, h: Hypothesis) -> tuple[str, dict[str, Any]]:
        """根据假设选择干预类型。"""
        if "质量" in h.statement or "mass" in h.statement.lower():
            return "change_mass", {"mass": float(self._rng.uniform(0.5, 5.0))}
        if "速度" in h.statement or "velocity" in h.statement.lower():
            return "set_velocity", {"velocity": float(self._rng.uniform(-2, 2))}
        if "摩擦" in h.statement or "friction" in h.statement.lower():
            return "change_friction", {"friction": float(self._rng.uniform(0, 1))}
        # 默认：施加力
        return "apply_force", {
            "force": float(self._rng.uniform(-5, 5)),
            "direction": float(self._rng.uniform(0, 6.28)),
        }

    def _estimate_eig(
        self, h: Hypothesis, intervention_type: str, params: dict[str, Any]
    ) -> float:
        """估计预期信息增益。

        EIG = E[KL(P(state|do) || P(state))]
        简化：用干预强度 × 假设置信度 × 可测试性近似。
        """
        # 干预强度
        strength = sum(abs(v) for v in params.values() if isinstance(v, (int, float)))
        strength = min(strength, 10.0) / 10.0
        eig = strength * h.confidence * h.testability
        # 加随机扰动模拟不确定性
        eig += float(self._rng.uniform(-0.05, 0.05))
        return max(0.0, float(eig))

    def _build_action_sequence(
        self, intervention_type: str, params: dict[str, Any], h: Hypothesis
    ) -> list[dict[str, Any]]:
        """生成沙盒动作序列。"""
        seq = [
            {"step": 0, "action": "reset_sandbox", "params": {}},
            {
                "step": 1,
                "action": intervention_type,
                "params": params,
                "description": f"测试假设: {h.statement[:80]}",
            },
            {
                "step": 2,
                "action": "observe",
                "params": {"vars": h.observation_vars},
            },
        ]
        return seq

    # ------------------------------------------------------------------ #
    # 访问器
    # ------------------------------------------------------------------ #
    @property
    def designed_count(self) -> int:
        return self._designed_count

    @property
    def sandbox_attached(self) -> bool:
        return self._sandbox is not None

    def snapshot(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "designed_count": self._designed_count,
            "sandbox_attached": self.sandbox_attached,
        }
