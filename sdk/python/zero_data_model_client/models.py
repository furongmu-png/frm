"""Pydantic models for ZeroDataModel SDK requests and responses.

The client returns plain ``dict`` payloads by default for maximum flexibility,
but these models are provided for callers that want typed validation. They
mirror the shapes documented for the core and phase-7 endpoints.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Core responses
# --------------------------------------------------------------------------- #


class HealthResponse(BaseModel):
    status: str
    model_ready: bool | None = None


class ThinkRequest(BaseModel):
    cycles: int = Field(default=1, ge=1)
    input: list[float] | None = Field(default=None, max_length=1024)


class ThinkResponse(BaseModel):
    cycle: int
    output: list[float] | None = None
    confidence: float | None = None
    free_energy: float | None = None
    metadata: dict[str, Any] | None = None


class KnowledgeGraphResponse(BaseModel):
    nodes: list[Any] = Field(default_factory=list)
    edges: list[Any] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Phase-7 request models (subset with well-defined shapes)
# --------------------------------------------------------------------------- #


class ArchitectActionRequest(BaseModel):
    action_type: str
    module_name: str


class LayeredPredictorUpdateRequest(BaseModel):
    observation: list[float]


class EpisodicGraphPlanRequest(BaseModel):
    start_id: int
    goal_id: int


class SemanticIndexSearchRequest(BaseModel):
    vector: list[float]
    k: int = Field(default=5, ge=1)


class LogicRuleIdRequest(BaseModel):
    rule_id: str


class HypothesisTestRequest(BaseModel):
    hypothesis_id: str


class ExperimentResultRequest(BaseModel):
    candidate_id: str
    outcome: float
