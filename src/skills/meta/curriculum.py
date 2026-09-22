"""好奇心驱动课程学习。

维护任务池，为每个任务计算"学习进展"（近期误差下降率），
优先调度进展最快的任务。

设计：
- ``CurriculumTask``：单个任务（难度、初始误差、当前误差）。
- ``CurriculumScheduler``：调度器，按学习进展（LP）排序选下一任务。
- LP = (recent_error_history[0] - recent_error_history[-1]) / horizon
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..base import SkillBase, SkillContext, SkillResult

logger = logging.getLogger(__name__)


@dataclass
class CurriculumTask:
    """单个课程任务。"""
    task_id: str
    difficulty: float  # 0-1
    description: str = ""
    error_history: deque[float] = field(default_factory=lambda: deque(maxlen=20))
    n_attempts: int = 0
    last_error: float = 0.0

    def record(self, error: float) -> None:
        self.error_history.append(float(error))
        self.last_error = float(error)
        self.n_attempts += 1

    def learning_progress(self, window: int = 5) -> float:
        """学习进展 = 早期误差 - 最近误差。

        正值表示在进步；负值表示在退步；0 表示无变化或样本不足。
        """
        if len(self.error_history) < 2:
            return 0.0
        history = list(self.error_history)
        if len(history) <= window:
            return float(history[0] - history[-1])
        early = float(np.mean(history[:window]))
        recent = float(np.mean(history[-window:]))
        return early - recent

    def mastery(self) -> float:
        """掌握度 = 1 - 误差（裁剪到 [0, 1]）。"""
        if not self.error_history:
            return 0.0
        recent = float(np.mean(list(self.error_history)[-5:]))
        return float(np.clip(1.0 - recent, 0.0, 1.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "difficulty": self.difficulty,
            "description": self.description,
            "n_attempts": self.n_attempts,
            "last_error": self.last_error,
            "learning_progress": self.learning_progress(),
            "mastery": self.mastery(),
        }


class CurriculumScheduler(SkillBase):
    """课程调度器。"""

    name = "curriculum"
    dimension = "meta"

    def __init__(
        self,
        *,
        lp_window: int = 5,
        exploration_rate: float = 0.1,
        enabled: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(enabled=enabled, **kwargs)
        self.lp_window = lp_window
        self.exploration_rate = exploration_rate
        self._tasks: dict[str, CurriculumTask] = {}
        self._current_task_id: str | None = None
        self._selection_history: list[str] = []
        self._rng = np.random.default_rng(7)

    # ------------------------------------------------------------------ #
    # 对外 API
    # ------------------------------------------------------------------ #

    def add_task(
        self,
        task_id: str,
        difficulty: float,
        description: str = "",
    ) -> CurriculumTask:
        """添加任务到任务池。"""
        if task_id in self._tasks:
            raise ValueError(f"task {task_id!r} already exists")
        task = CurriculumTask(
            task_id=task_id,
            difficulty=float(np.clip(difficulty, 0.0, 1.0)),
            description=description,
        )
        self._tasks[task_id] = task
        return task

    def record_attempt(self, task_id: str, error: float) -> None:
        if task_id not in self._tasks:
            raise KeyError(f"unknown task: {task_id!r}")
        self._tasks[task_id].record(error)

    def select_next(self) -> str:
        """选下一个任务。

        策略：
        - 以 exploration_rate 概率随机选（探索）
        - 否则按学习进展排序（exploitation）
        """
        if not self._tasks:
            raise RuntimeError("no tasks in pool")
        task_ids = list(self._tasks.keys())

        # 探索
        if self._rng.random() < self.exploration_rate:
            chosen = task_ids[int(self._rng.integers(0, len(task_ids)))]
        else:
            # 学习进展排序，ties 用更简单（难度低）的任务优先
            scored = sorted(
                task_ids,
                key=lambda tid: (
                    -self._tasks[tid].learning_progress(self.lp_window),
                    self._tasks[tid].difficulty,
                ),
            )
            chosen = scored[0]
        self._current_task_id = chosen
        self._selection_history.append(chosen)
        return chosen

    def current_task(self) -> CurriculumTask | None:
        if self._current_task_id is None:
            return None
        return self._tasks.get(self._current_task_id)

    def task_list(self) -> list[dict[str, Any]]:
        return [t.to_dict() for t in self._tasks.values()]

    def stats(self) -> dict[str, Any]:
        return {
            "n_tasks": len(self._tasks),
            "current_task": self._current_task_id,
            "selection_history": list(self._selection_history),
            "avg_mastery": (
                float(np.mean([t.mastery() for t in self._tasks.values()]))
                if self._tasks
                else 0.0
            ),
        }

    # ------------------------------------------------------------------ #
    # SkillBase 接口
    # ------------------------------------------------------------------ #

    def process(self, ctx: SkillContext) -> SkillResult:
        """默认演示：从 ctx.prediction_error 推送到当前任务，
        然后选下一任务。"""
        if self._tasks:
            if self._current_task_id is not None:
                self.record_attempt(self._current_task_id, ctx.prediction_error)
            next_id = self.select_next()
        else:
            next_id = ""
        return SkillResult(
            name=self.name,
            data={
                "current_task": next_id,
                "n_tasks": len(self._tasks),
                "avg_mastery": (
                    float(np.mean([t.mastery() for t in self._tasks.values()]))
                    if self._tasks
                    else 0.0
                ),
                "tasks": self.task_list()[:10],  # 限制前端载荷
            },
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "ready": len(self._tasks) > 0,
            "data": {
                "n_tasks": len(self._tasks),
                "current_task": self._current_task_id,
                "avg_mastery": (
                    float(np.mean([t.mastery() for t in self._tasks.values()]))
                    if self._tasks
                    else 0.0
                ),
                "selection_history_size": len(self._selection_history),
            },
        }
