"""主动教学（机器教人）技能。

维护一个"学生模型"（简化版 ZeroDataModel 或线性模型），
选择教学范例以最大程度降低学生的预测误差。
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np

from ..base import SkillBase, SkillContext, SkillResult

logger = logging.getLogger(__name__)


class StudentModel:
    """简化学生模型：线性预测器。

    教师通过选择范例来降低学生在特定输入上的预测误差。
    """

    def __init__(self, dim: int = 16, lr: float = 0.1) -> None:
        self.dim = dim
        self.lr = lr
        self._weights = np.zeros(dim, dtype=np.float64)
        self._bias = 0.0
        self._seen_examples: int = 0
        self._ema_error: float = 0.0  # 指数移动平均误差

    def predict(self, x: np.ndarray) -> float:
        x = self._fit(x)
        return float(x @ self._weights + self._bias)

    def teach(self, x: np.ndarray, target: float) -> float:
        """用单个范例教学，返回教学后的预测误差。"""
        x = self._fit(x)
        pred = x @ self._weights + self._bias
        error = target - pred
        # 梯度下降
        self._weights += self.lr * error * x
        self._bias += self.lr * error
        self._seen_examples += 1
        abs_err = abs(error)
        # 指数移动平均（EMA），alpha=0.3
        alpha = 0.3
        self._ema_error = (1 - alpha) * self._ema_error + alpha * abs_err
        return abs_err

    def evaluate(self, examples: list[tuple[np.ndarray, float]]) -> float:
        """评估学生在示例集上的平均误差。"""
        if not examples:
            return 0.0
        errors = []
        for x, target in examples:
            pred = self.predict(x)
            errors.append(abs(target - pred))
        return float(np.mean(errors))

    def knowledge_state(self) -> dict[str, Any]:
        """返回学生知识状态（用于前端雷达图）。"""
        return {
            "weight_norm": float(np.linalg.norm(self._weights)),
            "bias": float(self._bias),
            "n_examples": self._seen_examples,
            "avg_error": self._ema_error,
            "mastery": float(1.0 / (1.0 + self._ema_error)),
        }

    def _fit(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).flatten()[: self.dim]
        if x.size < self.dim:
            x = np.pad(x, (0, self.dim - x.size))
        return x


class TeachingModule(SkillBase):
    """主动教学模块。

    - 维护学生模型
    - 选择最能降低学生误差的范例
    - 输出教学内容
    """

    name = "machine_teaching"
    dimension = "interaction"

    def __init__(
        self,
        *,
        student_dim: int = 16,
        n_skills: int = 4,
        enabled: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(enabled=enabled, **kwargs)
        self.student = StudentModel(dim=student_dim)
        self.n_skills = n_skills

        # 教学范例池：每个 skill 一组 (input, target) 对
        self._example_pool: dict[int, list[tuple[np.ndarray, float]]] = {
            i: [] for i in range(n_skills)
        }

        # 教学历史
        self._teaching_history: deque[dict[str, Any]] = deque(maxlen=100)

        self._current_skill: int = 0
        self._rng = np.random.default_rng(333)

        # 生成默认范例（必须在 _rng 之后）
        self._init_examples()

    def _init_examples(self) -> None:
        """初始化教学范例池。"""
        for skill_id in range(self.n_skills):
            for _ in range(10):
                x = self._rng.standard_normal(16) * 0.5
                # 目标 = 线性函数（不同 skill 有不同的权重模式）
                target_w = np.zeros(16)
                target_w[skill_id * 4 : (skill_id + 1) * 4] = 1.0
                target = float(x @ target_w)
                self._example_pool[skill_id].append((x, target))

    def add_example(
        self, skill_id: int, x: np.ndarray, target: float
    ) -> None:
        """添加自定义教学范例。"""
        if 0 <= skill_id < self.n_skills:
            self._example_pool[skill_id].append(
                (np.asarray(x, dtype=np.float64).flatten(), target)
            )

    def select_best_example(
        self, skill_id: int
    ) -> tuple[np.ndarray, float] | None:
        """选择最能降低学生误差的范例。

        策略：评估每个范例对学生的信息增益（误差降低幅度），
        选择信息增益最大的。
        """
        examples = self._example_pool.get(skill_id, [])
        if not examples:
            return None

        best_gain = -1.0
        best_example = None

        # 临时保存学生状态
        saved_w = self.student._weights.copy()
        saved_b = self.student._bias
        saved_n = self.student._seen_examples
        saved_e = self.student._ema_error

        for x, target in examples:
            # 计算教学前的误差
            pre_error = abs(target - self.student.predict(x))

            # 模拟教学
            self.student.teach(x, target)

            # 计算教学后的误差（在另一范例上）
            post_error = abs(target - self.student.predict(x))

            # 信息增益 = 误差降低
            gain = pre_error - post_error

            if gain > best_gain:
                best_gain = gain
                best_example = (x, target)

            # 恢复学生状态
            self.student._weights = saved_w.copy()
            self.student._bias = saved_b
            self.student._seen_examples = saved_n
            self.student._ema_error = saved_e

        return best_example

    def teach_step(self, skill_id: int | None = None) -> dict[str, Any]:
        """执行一步教学。"""
        if skill_id is None:
            skill_id = self._current_skill

        example = self.select_best_example(skill_id)
        if example is None:
            return {"taught": False, "skill_id": skill_id}

        x, target = example
        error = self.student.teach(x, target)

        result = {
            "taught": True,
            "skill_id": skill_id,
            "error": error,
            "student_state": self.student.knowledge_state(),
        }
        self._teaching_history.append(result)
        return result

    def process(self, ctx: SkillContext) -> SkillResult:
        # 执行一步教学
        teaching_result = self.teach_step(self._current_skill)

        # 评估所有 skills
        skill_evaluations = {}
        for sid in range(self.n_skills):
            examples = self._example_pool.get(sid, [])
            avg_error = self.student.evaluate(examples)
            skill_evaluations[f"skill_{sid}"] = {
                "avg_error": avg_error,
                "n_examples": len(examples),
            }

        # 切换到下一个需要最多教学的 skill
        worst_skill = max(
            skill_evaluations.items(),
            key=lambda x: x[1]["avg_error"],
        )
        self._current_skill = int(worst_skill[0].split("_")[1])

        return SkillResult(
            name=self.name,
            data={
                "teaching_result": teaching_result,
                "student_state": self.student.knowledge_state(),
                "skill_evaluations": skill_evaluations,
                "current_skill": self._current_skill,
                "n_taught": len(self._teaching_history),
            },
        )
