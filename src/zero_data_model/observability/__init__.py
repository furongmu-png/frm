"""Phase 3 §3 全栈可观测性包。

提供思维链可视化、反事实解释、人类教学接口和审计日志，使模型
认知过程完全透明且可干预。
"""
from __future__ import annotations

from .audit import (
    AuditEntry,
    AuditEventType,
    AuditLogger,
    ValueDimension,
    ValueVector,
)
from .counterfactual_explainer import CounterfactualExplainer, CounterfactualResult
from .human_teaching import (
    FeedbackRecord,
    FeedbackType,
    HumanFeedback,
    TeachingMode,
)
from .thought_chain import ThoughtChain, ThoughtRecord

__all__: list[str] = [
    "AuditEntry",
    "AuditEventType",
    "AuditLogger",
    "CounterfactualExplainer",
    "CounterfactualResult",
    "FeedbackRecord",
    "FeedbackType",
    "HumanFeedback",
    "TeachingMode",
    "ThoughtChain",
    "ThoughtRecord",
    "ValueDimension",
    "ValueVector",
]
