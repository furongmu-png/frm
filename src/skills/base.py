"""技能基类与共享数据结构。

每个技能继承 :class:`SkillBase`，实现 :meth:`process` 方法。技能由
:class:`~skills.registry.SkillRegistry` 管理，在
``HierarchicalZeroDataModel.think()`` 的技能阶段被统一调用。
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SkillContext:
    """每次 ``think()`` 调用时传递给技能的上下文。

    Attributes
    ----------
    belief : np.ndarray
        当前信念向量（L0 状态副本）。
    observation : np.ndarray | None
        本步观测（若来自物理沙盒则为 ``encode_image`` 后的向量）。
    prediction_error : float
        L0 层误差范数。
    prediction_errors : dict
        各层误差 ``{"L0": float, "L1": float, "L2": float}``。
    confidence : float
        元认知置信度（0-1，来自 Phase 2）。
    step : int
        全局步数。
    model : Any
        底层 ``ZeroDataModel`` 引用（只读访问，技能不应修改）。
    raw_observation : Any
        原始观测（如 128×128 物理帧），由技能按需消费。
    """

    belief: np.ndarray
    observation: np.ndarray | None = None
    prediction_error: float = 0.0
    prediction_errors: dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0
    step: int = 0
    model: Any = None
    raw_observation: Any = None


@dataclass
class SkillResult:
    """技能执行结果，写入 ``signal.metadata["skills"][name]``。"""

    name: str
    enabled: bool = True
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "data": self.data,
            **({"error": self.error} if self.error else {}),
        }


class SkillBase(ABC):
    """所有技能的抽象基类。

    子类需实现 :meth:`process`。基类提供：
    - ``name`` / ``enabled`` 属性
    - ``snapshot()`` 返回前端可消费的状态摘要
    - ``safe_process()`` 包装 try/except，确保单个技能失败不影响全局
    - ``process_interval`` 节流：计算密集型技能（MCTS、自我对弈、
      沙盒执行等）可声明大于 1 的间隔，非刷新步直接返回上次结果，
      避免在每个 ``think()`` 中重跑昂贵计算。首次调用（``step % interval
      == 0``）仍会真正执行，保证技能语义完整。
    """

    #: 技能名称（唯一标识，用于注册和 metadata key）
    name: str = "base"

    #: 所属维度：perception / cognition / interaction / expertise / meta
    dimension: str = "meta"

    #: 每隔多少步真正执行一次 ``process``。``1`` 表示每步都执行
    #: （默认，轻量技能保持）。计算密集型技能应覆盖为更大的值。
    process_interval: int = 1

    def __init__(self, *, enabled: bool = True, **kwargs: Any) -> None:
        self.enabled = enabled
        self._params = dict(kwargs)
        self._last_result: SkillResult | None = None

    @abstractmethod
    def process(self, ctx: SkillContext) -> SkillResult:
        """执行技能逻辑，返回 :class:`SkillResult`。"""

    def _is_refresh_step(self, step: int) -> bool:
        """当前步是否需要真正执行 ``process``。"""
        if self.process_interval <= 1:
            return True
        return step % self.process_interval == 0

    def safe_process(self, ctx: SkillContext) -> SkillResult:
        """带异常隔离的执行包装。

        若 ``process_interval > 1`` 且当前步非刷新步，且已有缓存结果，
        直接返回上次结果（零计算），从而把昂贵技能的均摊成本压到
        ``1/interval``。
        """
        if not self.enabled:
            return SkillResult(name=self.name, enabled=False)
        # 节流：非刷新步返回缓存（首次调用必刷新，保证有数据）
        if not self._is_refresh_step(ctx.step) and self._last_result is not None:
            return self._last_result
        try:
            result = self.process(ctx)
            self._last_result = result
            return result
        except (
            ValueError,
            KeyError,
            IndexError,
            FloatingPointError,
            TypeError,
            RuntimeError,
        ) as exc:
            logger.warning(
                "skill %s failed: %s: %s", self.name, type(exc).__name__, exc
            )
            result = SkillResult(
                name=self.name, enabled=True, error=f"{type(exc).__name__}: {exc}"
            )
            self._last_result = result
            return result

    def snapshot(self) -> dict[str, Any]:
        """返回前端面板可消费的状态快照。"""
        if self._last_result is None:
            return {"name": self.name, "enabled": self.enabled, "ready": False}
        return {
            "name": self.name,
            "enabled": self.enabled,
            "ready": True,
            "data": self._last_result.data,
            **({"error": self._last_result.error} if self._last_result.error else {}),
        }

    def configure(self, **kwargs: Any) -> None:
        """运行时更新技能参数。"""
        self._params.update(kwargs)

    def get_param(self, key: str, default: Any = None) -> Any:
        return self._params.get(key, default)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} enabled={self.enabled}>"
