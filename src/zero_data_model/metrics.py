# src/zero_data_model/metrics.py
"""Centralized Prometheus metric definitions for ZeroDataModel.

All ``zdm_*`` custom metrics live here so both :mod:`zero_data_model.api`
(the FastAPI observability layer) and :mod:`zero_data_model.model` (the
``think()`` loop and cognitive-upgrade hooks) import from a single source
of truth.

When ``prometheus_client`` is not installed, every metric degrades to a
no-op (:class:`_NoopMetric`) so the application keeps functioning without
observability dependencies -- metric calls become cheap no-ops and
``ZDM_METRICS`` always exposes the same attribute set either way.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

# --------------------------------------------------------------------------- #
# Optional prometheus_client import -- no-op fallback when missing.
# --------------------------------------------------------------------------- #

try:  # pragma: no cover - environment-dependent import
    from prometheus_client import Counter, Gauge, Histogram

    _HAS_PROMETHEUS = True
except Exception:  # pragma: no cover - optional dep missing
    _HAS_PROMETHEUS = False

    class _NoopMetric:
        """No-op stand-in usable as a Counter / Gauge / Histogram.

        Constructing one accepts and ignores the (name, help, labelnames,
        buckets, ...) positional/keyword arguments the real
        ``prometheus_client`` constructors take, so
        ``Counter("name", "help", ["l"])`` yields a usable no-op.
        ``labels(...)`` returns ``self`` so chained calls
        (``m.labels(m="x").inc()``) keep working without a real backend.
        All mutating methods are silent no-ops.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            return None

        def labels(self, *args: Any, **kwargs: Any) -> _NoopMetric:
            return self

        def inc(self, amount: float = 1.0) -> None:
            return None

        def dec(self, amount: float = 1.0) -> None:
            return None

        def set(self, value: float) -> None:
            return None

        def observe(self, value: float) -> None:
            return None

    Counter = _NoopMetric  # type: ignore[assignment, misc]
    Gauge = _NoopMetric  # type: ignore[assignment, misc]
    Histogram = _NoopMetric  # type: ignore[assignment, misc]


# --------------------------------------------------------------------------- #
# Core think() metrics (re-homed from api.py for central ownership).
# --------------------------------------------------------------------------- #

_think_duration = Histogram(
    "zdm_think_duration_seconds",
    "Wall-clock duration of /think cycles in seconds.",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)
_cycle_count = Gauge(
    "zdm_cycle_count",
    "Current model cycle count.",
)
_free_energy = Gauge(
    "zdm_free_energy_last",
    "Last think() cycle free energy",
)

# --------------------------------------------------------------------------- #
# Generic cognitive-upgrade module metrics (cover all 13 new modules).
# --------------------------------------------------------------------------- #

_cognitive_module_calls_total = Counter(
    "zdm_cognitive_module_calls_total",
    "Total calls to cognitive-upgrade modules",
    ["module", "method"],
)
_cognitive_module_duration_seconds = Histogram(
    "zdm_cognitive_module_duration_seconds",
    "Duration of cognitive-upgrade module calls",
    ["module", "method"],
)
_cognitive_module_errors_total = Counter(
    "zdm_cognitive_module_errors_total",
    "Total errors from cognitive-upgrade modules",
    ["module", "method"],
)

# --------------------------------------------------------------------------- #
# Module-specialized metrics.
# --------------------------------------------------------------------------- #

# architect
_architect_dormant_count = Gauge(
    "zdm_architect_dormant_count", "Dormant modules count"
)
_architect_actions_total = Counter(
    "zdm_architect_actions_total", "Architect actions", ["action_type"]
)

# temporal_memory
_temporal_spectral_radius = Gauge(
    "zdm_temporal_spectral_radius", "Temporal memory spectral radius"
)
_temporal_mse = Gauge("zdm_temporal_mse", "Temporal memory MSE")

# layered_predictor
_layered_belief_norm = Gauge(
    "zdm_layered_belief_norm", "Layered predictor L2 belief norm"
)

# episodic_graph
_episodic_node_count = Gauge(
    "zdm_episodic_node_count", "Episodic graph node count"
)
_episodic_edge_count = Gauge(
    "zdm_episodic_edge_count", "Episodic graph edge count"
)

# semantic_index
_semantic_index_size = Gauge("zdm_semantic_index_size", "Semantic index size")

# logic_layer
_logic_penalty = Gauge("zdm_logic_penalty", "Logic layer penalty signal")
_logic_violation_count = Gauge(
    "zdm_logic_violation_count", "Logic rule violations"
)

# meta_cognition
_metacog_confidence = Gauge(
    "zdm_metacog_confidence", "Meta-cognition confidence"
)
_metacog_uncertainty = Gauge(
    "zdm_metacog_uncertainty", "Meta-cognition mean uncertainty"
)

# experiment_planner
_experiment_candidates = Gauge(
    "zdm_experiment_candidates", "Experiment candidate count"
)
_experiment_info_gain = Gauge(
    "zdm_experiment_info_gain", "Last selected experiment info gain"
)

# hypothesis_tester
_hypothesis_supported = Gauge(
    "zdm_hypothesis_supported", "Supported hypothesis count"
)

# multiagent (world / communication / culture)
_multiagent_collaboration_events = Gauge(
    "zdm_multiagent_collaboration_events", "Collaboration events count"
)
_communication_usage = Gauge(
    "zdm_communication_usage", "Communication channel usage total"
)
_culture_generations = Gauge(
    "zdm_culture_generations", "Culture generation count"
)

# --------------------------------------------------------------------------- #
# Public namespace -- import as ``from .metrics import ZDM_METRICS``.
#
# ``prometheus_available`` is the single non-metric attribute (it reports
# whether a real prometheus_client backend is wired), which is why
# validation scripts count ``len(ZDM_METRICS.__dict__) - 1`` metrics.
# --------------------------------------------------------------------------- #

ZDM_METRICS = SimpleNamespace(
    prometheus_available=_HAS_PROMETHEUS,
    # core think() metrics
    think_duration=_think_duration,
    cycle_count=_cycle_count,
    free_energy=_free_energy,
    # generic cognitive-module metrics
    cognitive_module_calls_total=_cognitive_module_calls_total,
    cognitive_module_duration_seconds=_cognitive_module_duration_seconds,
    cognitive_module_errors_total=_cognitive_module_errors_total,
    # architect
    architect_dormant_count=_architect_dormant_count,
    architect_actions_total=_architect_actions_total,
    # temporal_memory
    temporal_spectral_radius=_temporal_spectral_radius,
    temporal_mse=_temporal_mse,
    # layered_predictor
    layered_belief_norm=_layered_belief_norm,
    # episodic_graph
    episodic_node_count=_episodic_node_count,
    episodic_edge_count=_episodic_edge_count,
    # semantic_index
    semantic_index_size=_semantic_index_size,
    # logic_layer
    logic_penalty=_logic_penalty,
    logic_violation_count=_logic_violation_count,
    # meta_cognition
    metacog_confidence=_metacog_confidence,
    metacog_uncertainty=_metacog_uncertainty,
    # experiment_planner
    experiment_candidates=_experiment_candidates,
    experiment_info_gain=_experiment_info_gain,
    # hypothesis_tester
    hypothesis_supported=_hypothesis_supported,
    # multiagent
    multiagent_collaboration_events=_multiagent_collaboration_events,
    communication_usage=_communication_usage,
    culture_generations=_culture_generations,
)
