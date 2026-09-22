"""CounterfactualSelf — "what if I had acted differently" explanations.

理论基础
========
反事实思考 (counterfactual thinking) 是人类自我反思的核心能力：回顾过去
决策，想象"如果当时做了不同选择，结果会怎样"。这在认知科学中与后悔、
学习、因果推理紧密相关。

本模块利用世界模型和自我图式，回答"如果当时我做了不同的选择，结果会怎样"。
生成反事实叙述：通过文本解码器输出类似"如果我当时选择了向左移动，我可能
不会撞到墙壁，因为…"。

这些解释连同自传体记忆一同存储，强化自我反思能力。

自由能映射：反事实推理 = 对未实现的可能性空间的探索，是一种"认知主动推理"。
模型在反事实空间中寻找"最小化预期自由能"的路径，从而学习更好的决策。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


# ------------------------------------------------------------------ #
# 数据结构
# ------------------------------------------------------------------ #
@dataclass
class CounterfactualResult:
    """反事实推理结果。"""
    actual_action: int
    counterfactual_action: int
    actual_outcome: dict[str, Any]
    predicted_counterfactual_outcome: dict[str, Any]
    narrative: str
    regret: float                    # 后悔度（反事实优于实际的量）
    metadata: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------ #
# 主类
# ------------------------------------------------------------------ #
class CounterfactualSelf:
    """反事实自我解释器。

    利用世界模型（简化为状态转移预测）和自我图式，生成反事实叙述。

    Parameters
    ----------
    dim : int
        隐空间维度
    n_actions : int
        动作空间大小
    lr : float
        状态转移模型学习率
    seed : int | None
    """

    def __init__(
        self,
        dim: int = 64,
        n_actions: int = 14,
        lr: float = 0.01,
        seed: Optional[int] = 42,
    ):
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if n_actions < 2:
            raise ValueError(
                f"n_actions must be >= 2 (need alternatives), got {n_actions}"
            )

        self.dim = int(dim)
        self.n_actions = int(n_actions)
        self.lr = float(lr)
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._lock = threading.RLock()

        # 状态转移模型：每个动作一个转移矩阵 (dim, dim)
        # T_a: s_t → s_{t+1} 的预测
        # 用低秩近似：T_a ≈ W_a (dim, dim)，初始化为接近单位
        self._transition_models = [
            np.eye(dim) * 0.9 + self._rng.standard_normal((dim, dim)) * 0.01
            for _ in range(n_actions)
        ]

        # 奖励模型：每个动作的预期自由能（越低越好）。
        # C3 修复：原代码初始化为全 0，导致未执行过的动作永远 EFE=0，
        # 在 argmin 选择中被永远优先选中（spurious regret）。
        # 改为初始化为 NaN 标记"未知"，并在选择时用均值回退。
        self._expected_free_energy = np.full(n_actions, np.nan)
        # 记录每个动作是否被实际执行过（用于 EFE 选择策略）
        self._action_executed = np.zeros(n_actions, dtype=bool)

        # 上一时刻状态 + 动作（用于反事实推理）
        self._last_state = np.zeros(dim, dtype=np.float64)
        self._last_action = 0
        self._last_outcome: dict[str, Any] = {}

        # 统计
        self._step_count = 0
        self._regret_history: list[float] = []
        self._narratives_generated = 0

    # ---------------------------------------------------------------- #
    # 记录实际发生的转移
    # ---------------------------------------------------------------- #
    def record_transition(
        self,
        state_before: np.ndarray,
        action: int,
        state_after: np.ndarray,
        outcome: dict[str, Any],
        free_energy: float = 0.0,
    ) -> None:
        """记录一次实际的状态转移，更新转移模型。

        Parameters
        ----------
        state_before : ndarray, shape (dim,)
        action : int
        state_after : ndarray, shape (dim,)
        outcome : dict
            实际结果（如碰撞、奖励等）
        free_energy : float
            该步的自由能（用于后悔度计算）
        """
        s_before = self._normalize(state_before)
        s_after = self._normalize(state_after)
        if not (0 <= action < self.n_actions):
            action = 0

        # C3 修复：free_energy 必须是有限实数，否则 NaN 会污染 EFE。
        try:
            fe = float(free_energy)
        except (TypeError, ValueError):
            fe = 0.0
        if not np.isfinite(fe):
            fe = 0.0

        with self._lock:
            # Hebbian 更新转移模型：T_a 应满足 T_a @ s_before ≈ s_after
            pred_after = self._transition_models[action] @ s_before
            error = pred_after - s_after
            # ΔT = -lr * outer(error, s_before)
            delta = -self.lr * np.outer(error, s_before)
            norm = np.linalg.norm(delta)
            if norm > 0.5:
                delta = delta * (0.5 / norm)
            self._transition_models[action] += delta

            # 更新预期自由能（EMA）。
            # C3 修复：原代码对未初始化的 EFE 用 0 做基线，导致
            # 未执行过的动作永远停留在 0。现在首次执行时直接赋值，
            # 后续用 EMA 平滑。
            old_efe = self._expected_free_energy[action]
            if np.isnan(old_efe):
                self._expected_free_energy[action] = fe
            else:
                self._expected_free_energy[action] = (
                    0.9 * old_efe + 0.1 * fe
                )
            self._action_executed[action] = True

            # 记录上一时刻
            self._last_state = s_before.copy()
            self._last_action = action
            self._last_outcome = dict(outcome) if isinstance(outcome, dict) else {}
            self._step_count += 1

    # ---------------------------------------------------------------- #
    # 反事实推理
    # ---------------------------------------------------------------- #
    def generate_counterfactual(
        self,
        state_before: Optional[np.ndarray] = None,
        actual_action: Optional[int] = None,
        actual_outcome: Optional[dict[str, Any]] = None,
        alternative_action: Optional[int] = None,
        text_decoder: Optional[object] = None,
    ) -> CounterfactualResult:
        """生成反事实叙述。

        Parameters
        ----------
        state_before : ndarray | None
            决策前的状态；None 时用上一记录的状态
        actual_action : int | None
            实际采取的动作；None 时用上一记录的动作
        actual_outcome : dict | None
            实际结果；None 时用上一记录的结果
        alternative_action : int | None
            反事实假设的动作；None 时自动选择预期自由能最低的动作
        text_decoder : object | None
            可选文本解码器；None 时用模板生成
        """
        with self._lock:
            s_before = (
                self._normalize(state_before)
                if state_before is not None
                else self._last_state.copy()
            )
            act = (
                int(actual_action)
                if actual_action is not None
                else self._last_action
            )
            if not (0 <= act < self.n_actions):
                act = 0

            outcome = (
                dict(actual_outcome) if actual_outcome is not None
                else dict(self._last_outcome)
            )

            # 选择反事实动作
            if alternative_action is not None and 0 <= alternative_action < self.n_actions:
                cf_action = int(alternative_action)
            else:
                # 自动选择：预期自由能最低的动作（排除实际动作）。
                # C3 修复：原代码对未执行过的动作用 EFE=0，导致
                # argmin 永远选未执行过的动作（虚假后悔）。现在
                # 用已执行动作的 EFE 均值回退未执行动作的 EFE，
                # 没有已执行动作时用 0（与原行为兼容但显式标记）。
                efe = self._expected_free_energy.copy()
                executed_mask = self._action_executed.copy()
                # 排除实际动作
                efe[act] = np.inf
                executed_mask[act] = False
                if executed_mask.any():
                    executed_mean = float(
                        np.nanmean(self._expected_free_energy[executed_mask])
                    )
                    if not np.isfinite(executed_mean):
                        executed_mean = 0.0
                else:
                    executed_mean = 0.0
                # 对未执行过的动作，用已执行动作的均值回退
                for i in range(self.n_actions):
                    if i == act:
                        continue
                    if np.isnan(efe[i]):
                        efe[i] = executed_mean
                cf_action = int(np.argmin(efe))

            # 预测反事实结果
            pred_cf_state = self._transition_models[cf_action] @ s_before
            # C3 修复：删除未使用的 pred_actual_state（死代码）。

            # 预期自由能差异 = 后悔度。
            # C3 修复：actual_efe 可能仍为 NaN（act 从未执行过），
            # 此时用 cf_efe 回退，避免 NaN 传播到 regret。
            actual_efe = self._expected_free_energy[act]
            cf_efe = self._expected_free_energy[cf_action]
            if np.isnan(actual_efe):
                actual_efe = cf_efe if not np.isnan(cf_efe) else 0.0
            if np.isnan(cf_efe):
                cf_efe = actual_efe if not np.isnan(actual_efe) else 0.0
            regret = float(actual_efe - cf_efe)  # 正 = 后悔（反事实更好）
            # NaN 防护
            if not np.isfinite(regret):
                regret = 0.0

            # 反事实预测结果
            cf_outcome = {
                "predicted_state_norm": float(np.linalg.norm(pred_cf_state)),
                "predicted_state_change": float(
                    np.linalg.norm(pred_cf_state - s_before)
                ),
                "expected_free_energy": float(cf_efe),
                "alternative_action": cf_action,
            }

            # 生成叙述
            if text_decoder is not None and hasattr(text_decoder, "decode"):
                try:
                    narrative = text_decoder.decode_counterfactual(
                        actual_action=act,
                        counterfactual_action=cf_action,
                        actual_outcome=outcome,
                        cf_outcome=cf_outcome,
                        regret=regret,
                    )
                except Exception:
                    narrative = self._template_narrative(
                        act, cf_action, outcome, cf_outcome, regret
                    )
            else:
                narrative = self._template_narrative(
                    act, cf_action, outcome, cf_outcome, regret
                )

            self._narratives_generated += 1
            self._regret_history.append(regret)
            if len(self._regret_history) > 1000:
                self._regret_history = self._regret_history[-1000:]

            return CounterfactualResult(
                actual_action=act,
                counterfactual_action=cf_action,
                actual_outcome=outcome,
                predicted_counterfactual_outcome=cf_outcome,
                narrative=narrative,
                regret=regret,
                metadata={
                    "step": self._step_count,
                    "actual_efe": float(actual_efe),
                    "cf_efe": float(cf_efe),
                    "narratives_generated": self._narratives_generated,
                },
            )

    # ---------------------------------------------------------------- #
    # 模板化叙述
    # ---------------------------------------------------------------- #
    def _template_narrative(
        self,
        actual_action: int,
        cf_action: int,
        actual_outcome: dict[str, Any],
        cf_outcome: dict[str, Any],
        regret: float,
    ) -> str:
        """用模板生成反事实叙述。"""
        # 动作描述映射（简化）
        action_names = {
            0: "向左移动", 1: "向右移动", 2: "向上移动", 3: "保持静止",
        }
        # 扩展动作空间描述
        if self.n_actions > 4:
            for i in range(4, self.n_actions):
                action_names[i] = f"执行感知动作 {i}"

        actual_desc = action_names.get(actual_action, f"动作 {actual_action}")
        cf_desc = action_names.get(cf_action, f"动作 {cf_action}")

        # 实际结果描述
        actual_fe = actual_outcome.get("free_energy", "?")
        actual_collision = actual_outcome.get("collision", False)

        # 叙述
        lines: list[str] = []
        lines.append(f"反思：我当时选择了「{actual_desc}」。")
        if regret > 0.01:
            lines.append(
                f"如果当时我选择了「{cf_desc}」，预期自由能会是 "
                f"{cf_outcome['expected_free_energy']:.3f}（实际 {actual_outcome.get('free_energy', 0):.3f}），"
                f"可能会更好。"
            )
            if actual_collision:
                lines.append(f"因为实际结果发生了碰撞，而「{cf_desc}」可能避免这一情况。")
            else:
                lines.append(f"因为「{cf_desc}」的预期状态变化更小，可能更稳定。")
            lines.append(f"后悔度：{regret:.3f}。我会记住这次教训。")
        elif regret < -0.01:
            lines.append(
                f"回想起来，「{actual_desc}」是较好的选择（预期自由能 "
                f"{actual_outcome.get('free_energy', 0):.3f} 低于「{cf_desc}」的 "
                f"{cf_outcome['expected_free_energy']:.3f}）。"
            )
            lines.append("这次决策是合理的。")
        else:
            lines.append(
                f"「{actual_desc}」和「{cf_desc}」的预期结果相近，"
                f"决策影响不大。"
            )
        return "\n".join(lines)

    # ---------------------------------------------------------------- #
    # 工具
    # ---------------------------------------------------------------- #
    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        v = np.asarray(vec, dtype=np.float64).flatten()
        if v.shape != (self.dim,):
            r = np.zeros(self.dim)
            n = min(len(v), self.dim)
            r[:n] = v[:n]
            v = r
        if not np.all(np.isfinite(v)):
            v = np.zeros(self.dim)
        return v

    def get_snapshot(self) -> dict[str, Any]:
        with self._lock:
            # C3 修复：EFE 数组含 NaN（未执行动作），序列化时转为 null。
            efe_list = []
            for v in self._expected_free_energy:
                if np.isnan(v):
                    efe_list.append(None)
                else:
                    efe_list.append(float(v))
            return {
                "step": self._step_count,
                "narratives_generated": self._narratives_generated,
                "avg_regret": (
                    float(np.mean(self._regret_history[-20:]))
                    if self._regret_history else 0.0
                ),
                "expected_free_energy": efe_list,
                "n_executed_actions": int(self._action_executed.sum()),
                "last_action": self._last_action,
                "last_outcome_keys": (
                    list(self._last_outcome.keys())
                    if self._last_outcome else []
                ),
            }

    def reset(self) -> None:
        with self._lock:
            self._transition_models = [
                np.eye(self.dim) * 0.9
                + self._rng.standard_normal((self.dim, self.dim)) * 0.01
                for _ in range(self.n_actions)
            ]
            # C3 修复：reset 时也用 NaN 初始化（与 __init__ 一致）
            self._expected_free_energy = np.full(self.n_actions, np.nan)
            self._action_executed = np.zeros(self.n_actions, dtype=bool)
            self._last_state = np.zeros(self.dim, dtype=np.float64)
            self._last_action = 0
            self._last_outcome = {}
            self._step_count = 0
            self._regret_history.clear()
            self._narratives_generated = 0
