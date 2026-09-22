# src/zero_data_model/experiment/experiment_logger.py
"""实验日志与可视化（ExperimentLogger）。

第二阶段 §3.3：记录每次实验的假设、干预动作、观察结果、贝叶斯因子、
结论，并通过回调推送实验日志到前端（类似实验室笔记本）。

在故事模式中自动生成"实验里程碑"（如"模型首次自主设计实验验证了
碰撞守恒定律"）。

仅依赖 ``numpy``，独立、自包含、可插拔。WebSocket 推送通过
可注入的回调函数实现，避免对特定框架的硬依赖。
"""
from __future__ import annotations

import contextlib
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ------------------------------------------------------------------ #
# 实验记录
# ------------------------------------------------------------------ #
@dataclass
class ExperimentRecord:
    """一次实验的完整记录。

    Attributes
    ----------
    step:
        实验发生的步数。
    hypothesis:
        被检验的假设描述。
    intervention:
        执行的干预动作描述。
    observation:
        观察结果描述。
    bayes_factor:
        贝叶斯因子 ``BF = P(data|H1) / P(data|H0)``。
    conclusion:
        实验结论（``"supported"`` / ``"rejected"`` / ``"inconclusive"``）。
    info_gain:
        信息增益（KL 散度）。
    timestamp:
        记录时间戳。
    """

    step: int
    hypothesis: str = ""
    intervention: str = ""
    observation: str = ""
    bayes_factor: float = 1.0
    conclusion: str = "inconclusive"
    info_gain: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """转为字典（便于序列化与 WebSocket 推送）。"""
        return {
            "step": self.step,
            "hypothesis": self.hypothesis,
            "intervention": self.intervention,
            "observation": self.observation,
            "bayes_factor": round(float(self.bayes_factor), 6),
            "conclusion": self.conclusion,
            "info_gain": round(float(self.info_gain), 6),
            "timestamp": self.timestamp,
        }


# ------------------------------------------------------------------ #
# 实验日志器
# ------------------------------------------------------------------ #
class ExperimentLogger:
    """实验日志记录器与前端推送。

    Parameters
    ----------
    max_records:
        历史记录上限（超出后丢弃最旧者）。
    push_callback:
        可选的推送回调函数 ``f(record_dict) -> None``，用于 WebSocket
        推送到前端。若为 ``None``，日志仅在内存中累积。

    Attributes
    ----------
    records:
        已记录的实验记录列表。
    milestones:
        实验里程碑事件列表（供故事模式）。
    """

    def __init__(
        self,
        max_records: int = 1000,
        push_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """初始化实验日志器。"""
        self._lock = threading.RLock()
        self.records: deque[ExperimentRecord] = deque(maxlen=max_records)
        self.milestones: deque[dict[str, Any]] = deque(maxlen=200)
        self._push_callback = push_callback
        self._milestone_counters: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    # 记录实验
    # ------------------------------------------------------------------ #
    def log(
        self,
        step: int,
        hypothesis: str = "",
        intervention: str = "",
        observation: str = "",
        bayes_factor: float = 1.0,
        conclusion: str = "inconclusive",
        info_gain: float = 0.0,
    ) -> ExperimentRecord:
        """记录一次实验并推送到前端。

        Parameters
        ----------
        step:
            实验发生的步数。
        hypothesis:
            被检验的假设描述。
        intervention:
            执行的干预动作描述。
        observation:
            观察结果描述。
        bayes_factor:
            贝叶斯因子 ``BF = P(data|H1) / P(data|H0)``。
        conclusion:
            实验结论。
        info_gain:
            信息增益。

        Returns
        -------
        ExperimentRecord
            创建的实验记录。
        """
        record = ExperimentRecord(
            step=int(step),
            hypothesis=str(hypothesis),
            intervention=str(intervention),
            observation=str(observation),
            bayes_factor=float(bayes_factor),
            conclusion=str(conclusion),
            info_gain=float(info_gain),
        )
        with self._lock:
            self.records.append(record)
            # 检查里程碑。
            self._check_milestone(record)
        # 推送到前端（在锁外执行，避免回调中再次获取锁导致死锁）。
        if self._push_callback is not None:
            # 回调失败不应影响主流程。
            with contextlib.suppress(Exception):
                self._push_callback(record.to_dict())
        return record

    def log_from_dict(self, data: dict[str, Any]) -> ExperimentRecord:
        """从字典记录实验（便于与实验规划器输出对接）。"""
        return self.log(
            step=data.get("step", 0),
            hypothesis=data.get("hypothesis", ""),
            intervention=data.get("intervention", data.get("name", "")),
            observation=data.get("observation", ""),
            bayes_factor=data.get("bayes_factor", 1.0),
            conclusion=data.get("conclusion", "inconclusive"),
            info_gain=data.get("info_gain", data.get("actual_gain", 0.0)),
        )

    # ------------------------------------------------------------------ #
    # 里程碑检测
    # ------------------------------------------------------------------ #
    def _check_milestone(self, record: ExperimentRecord) -> None:
        """检测实验里程碑并记录（供故事模式）。

        里程碑类型：
        - ``first_experiment``: 首次自主实验
        - ``first_supported``: 首次假设被支持
        - ``first_rejected``: 首次假设被拒绝
        - ``high_info_gain``: 信息增益超过阈值
        - ``strong_evidence``: 贝叶斯因子 > 10（强证据）
        """
        new_milestones: list[dict[str, Any]] = []

        if "first_experiment" not in self._milestone_counters:
            self._milestone_counters["first_experiment"] = 1
            new_milestones.append({
                "type": "experiment_milestone",
                "event": "first_experiment",
                "step": record.step,
                "description": (
                    f"在第{record.step}步，模型首次自主设计实验"
                    f"验证了{record.hypothesis}"
                ),
            })

        if (
            record.conclusion == "supported"
            and "first_supported" not in self._milestone_counters
        ):
            self._milestone_counters["first_supported"] = 1
            new_milestones.append({
                "type": "experiment_milestone",
                "event": "first_supported_hypothesis",
                "step": record.step,
                "description": (
                    f"在第{record.step}步，模型通过实验支持了假设："
                    f"{record.hypothesis}（BF={record.bayes_factor:.2f}）"
                ),
            })

        if (
            record.conclusion == "rejected"
            and "first_rejected" not in self._milestone_counters
        ):
            self._milestone_counters["first_rejected"] = 1
            new_milestones.append({
                "type": "experiment_milestone",
                "event": "first_rejected_hypothesis",
                "step": record.step,
                "description": (
                    f"在第{record.step}步，模型通过实验拒绝了假设："
                    f"{record.hypothesis}"
                ),
            })

        if record.bayes_factor > 10.0 and "strong_evidence" not in (
            self._milestone_counters
        ):
            self._milestone_counters["strong_evidence"] = 1
            new_milestones.append({
                "type": "experiment_milestone",
                "event": "strong_evidence",
                "step": record.step,
                "description": (
                    f"在第{record.step}步，实验产生了强证据"
                    f"（BF={record.bayes_factor:.2f} > 10）"
                ),
            })

        for ms in new_milestones:
            self.milestones.append(ms)

    # ------------------------------------------------------------------ #
    # 查询接口
    # ------------------------------------------------------------------ #
    def get_records(self, limit: int = 50) -> list[dict[str, Any]]:
        """获取最近的实验记录（字典列表）。"""
        with self._lock:
            return [r.to_dict() for r in list(self.records)[-limit:]]

    def get_milestones(self) -> list[dict[str, Any]]:
        """获取所有实验里程碑（供故事模式）。"""
        with self._lock:
            return list(self.milestones)

    def get_pending_milestones(self) -> list[dict[str, Any]]:
        """获取并清空待处理的里程碑。"""
        with self._lock:
            ms = list(self.milestones)
            self.milestones.clear()
            return ms

    # ------------------------------------------------------------------ #
    # 属性
    # ------------------------------------------------------------------ #
    @property
    def stats(self) -> dict[str, Any]:
        """统计信息。"""
        with self._lock:
            records_list = list(self.records)
            supported = sum(1 for r in records_list if r.conclusion == "supported")
            rejected = sum(1 for r in records_list if r.conclusion == "rejected")
            mean_bf = (
                float(np.mean([r.bayes_factor for r in records_list]))
                if records_list
                else 0.0
            )
            return {
                "n_records": len(records_list),
                "n_supported": supported,
                "n_rejected": rejected,
                "n_milestones": len(self.milestones),
                "mean_bayes_factor": mean_bf,
                "has_push_callback": self._push_callback is not None,
            }


__all__: list[str] = ["ExperimentLogger", "ExperimentRecord"]
