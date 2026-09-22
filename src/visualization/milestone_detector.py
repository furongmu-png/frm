# src/visualization/milestone_detector.py
"""Milestone detector — analyzes snapshot history for learning milestones.

Detects events like:
  - free_energy_drop: FE drops by >50% relative to a recent window.
  - high_surprise: prediction_error spikes above 2x moving average.
  - first_causal_edge: first causal graph edge appears.
  - kg_growth_burst: knowledge graph gains >5 nodes in one step.
  - concept_cluster: a new modality category appears.
  - convergence: FE stabilizes (std < 0.001 over 50 steps).

Each Milestone has: step, type, title, description, snapshot_step (for replay).
"""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass


# ------------------------------------------------------------------ #
# Milestone dataclass
# ------------------------------------------------------------------ #
@dataclass
class Milestone:
    """A single detected learning milestone (JSON-serialisable via
    :meth:`MilestoneDetector.get_milestones`)."""

    step: int
    type: str
    title: str
    description: str
    snapshot_step: int


# ------------------------------------------------------------------ #
# Milestone detector
# ------------------------------------------------------------------ #
class MilestoneDetector:
    """Analyzes snapshots and detects learning milestones.

    Call :meth:`analyze` after each cognitive cycle. The detector tracks
    free-energy, prediction-error, KG and modality state across calls and
    emits :class:`Milestone` objects when detection rules fire.

    A minimum 10-step gap is enforced between milestones to prevent
    spamming (``first_causal_edge`` is always emitted regardless of the
    gap).
    """

    # Minimum steps between emitted milestones (anti-spam).
    MIN_GAP = 10
    # Free-energy window size for drop / convergence detection.
    FE_WINDOW_SIZE = 50
    # Prediction-error window size for high-surprise detection.
    PE_WINDOW_SIZE = 20

    def __init__(self):
        self.milestones: list[Milestone] = []
        self._last_fe: float | None = None
        self._fe_window: deque = deque(maxlen=self.FE_WINDOW_SIZE)
        self._last_kg_node_count: int = 0
        self._last_causal_edges: int = 0
        self._seen_modalities: set = set()
        self._last_milestone_step: int = -100
        # Prediction-error window for high-surprise detection.
        self._pe_window: deque = deque(maxlen=self.PE_WINDOW_SIZE)
        # Phase 2 tracking: has a logic discovery / experiment milestone
        # been emitted yet (each fires only once).
        self._logic_discovery_emitted: bool = False
        self._experiment_milestone_emitted: bool = False

    # ------------------------------------------------------------------ #
    # Internal: emit a milestone (with gap enforcement)
    # ------------------------------------------------------------------ #
    def _emit(
        self,
        step: int,
        mtype: str,
        title: str,
        description: str,
        always_emit: bool = False,
    ) -> Milestone | None:
        """Create and record a milestone, enforcing the min-gap rule.

        Returns the :class:`Milestone` if emitted, or ``None`` if
        suppressed by the gap rule. ``always_emit=True`` bypasses the
        gap check (used for ``first_causal_edge``).
        """
        if not always_emit and (step - self._last_milestone_step) < self.MIN_GAP:
            return None
        m = Milestone(
            step=step,
            type=mtype,
            title=title,
            description=description,
            snapshot_step=step,
        )
        self.milestones.append(m)
        self._last_milestone_step = step
        return m

    # ------------------------------------------------------------------ #
    # Main analysis method
    # ------------------------------------------------------------------ #
    def analyze(self, snapshot: dict) -> Milestone | None:
        """Check all detection rules against ``snapshot``.

        Returns the first :class:`Milestone` found, or ``None``. When a
        milestone is found it is appended to :attr:`milestones` and
        returned immediately; the fe/pe tracking windows are **not**
        updated in that case so the milestone-bearing snapshot does not
        contaminate the baseline windows.
        """
        step = int(snapshot.get("step", 0))
        fe = float(snapshot.get("free_energy", 0.0))
        pe = float(snapshot.get("prediction_error", 0.0))
        modality = snapshot.get("modality", "unknown")

        causal_graph = snapshot.get("causal_graph") or {}
        causal_edges = (
            causal_graph.get("edges")
            if isinstance(causal_graph, dict)
            else None
        )
        n_edges = len(causal_edges) if isinstance(causal_edges, list) else 0

        kg_update = snapshot.get("kg_update") or {}
        new_nodes = (
            kg_update.get("new_nodes")
            if isinstance(kg_update, dict)
            else None
        )
        n_new_nodes = len(new_nodes) if isinstance(new_nodes, list) else 0

        milestone: Milestone | None = None

        # a. first_causal_edge — always emitted (bypasses gap).
        if n_edges > 0 and self._last_causal_edges == 0:
            milestone = self._emit(
                step,
                "first_causal_edge",
                "First Causal Edge",
                f"First causal graph edge appeared at step {step} "
                f"({n_edges} edge(s)).",
                always_emit=True,
            )
        self._last_causal_edges = n_edges

        # b. kg_growth_burst — >=5 new KG nodes in one step.
        if milestone is None and n_new_nodes >= 5:
            milestone = self._emit(
                step,
                "kg_growth_burst",
                "Knowledge Graph Growth Burst",
                f"Knowledge graph gained {n_new_nodes} new nodes in one step.",
            )
        self._last_kg_node_count += n_new_nodes

        # c. concept_cluster — new modality category.
        #    The modality is added to the seen-set *before* emitting so
        #    that it is not re-detected on subsequent calls (even if the
        #    gap rule suppresses the emit).
        if milestone is None and modality not in self._seen_modalities:
            self._seen_modalities.add(modality)
            milestone = self._emit(
                step,
                "concept_cluster",
                "New Concept Cluster",
                f"New modality category appeared: '{modality}'.",
            )

        # d. free_energy_drop — current FE < 50% of recent-window mean.
        if milestone is None and len(self._fe_window) >= self.FE_WINDOW_SIZE:
            mean_fe = sum(self._fe_window) / len(self._fe_window)
            if fe < 0.5 * mean_fe:
                milestone = self._emit(
                    step,
                    "free_energy_drop",
                    "Free Energy Drop",
                    f"Free energy dropped from {mean_fe:.4f} to {fe:.4f} "
                    f"(>50% reduction over {self.FE_WINDOW_SIZE}-step window).",
                )

        # e. high_surprise — prediction error > 2x moving average.
        if milestone is None and len(self._pe_window) > 0:
            mean_pe = sum(self._pe_window) / len(self._pe_window)
            if pe > 2.0 * mean_pe:
                milestone = self._emit(
                    step,
                    "high_surprise",
                    "High Surprise",
                    f"Prediction error spiked to {pe:.4f} "
                    f"(2x moving average {mean_pe:.4f}).",
                )

        # f. convergence — FE std < 0.001 over the full window.
        if milestone is None and len(self._fe_window) >= self.FE_WINDOW_SIZE:
            std_fe = statistics.pstdev(self._fe_window)
            if std_fe < 0.001:
                milestone = self._emit(
                    step,
                    "convergence",
                    "Convergence",
                    f"Free energy converged (std={std_fe:.6f} over "
                    f"{self.FE_WINDOW_SIZE} steps).",
                )

        # g. logic_discovery — Phase 2: first time logic contradiction
        #    exceeds 0.3 (model detected an inconsistency between its
        #    beliefs and declared logical rules).
        if milestone is None and not self._logic_discovery_emitted:
            _meta = snapshot.get("metadata") or {}
            if isinstance(_meta, dict):
                _upgrades = _meta.get("cognitive_upgrades") or {}
                if isinstance(_upgrades, dict):
                    _lv = _upgrades.get("logic_violations") or {}
                    if isinstance(_lv, dict):
                        _contr = _lv.get("contradiction")
                        if (
                            isinstance(_contr, (int, float))
                            and float(_contr) > 0.3
                        ):
                            m = self._emit(
                                step,
                                "logic_discovery",
                                "Logic Inconsistency Detected",
                                f"Model detected a logical contradiction "
                                f"(contradiction={float(_contr):.3f}) "
                                f"between its beliefs and declared rules.",
                            )
                            if m is not None:
                                self._logic_discovery_emitted = True
                                milestone = m

        # h. experiment_milestone — Phase 2: first time the experiment
        #    logger records a milestone (first experiment, first
        #    supported/rejected hypothesis, or strong evidence).
        if milestone is None and not self._experiment_milestone_emitted:
            _meta = snapshot.get("metadata") or {}
            if isinstance(_meta, dict):
                _upgrades = _meta.get("cognitive_upgrades") or {}
                if isinstance(_upgrades, dict):
                    _exp = _upgrades.get("experiment") or {}
                    if isinstance(_exp, dict):
                        _ms = _exp.get("milestones")
                        if isinstance(_ms, list) and len(_ms) > 0:
                            m = self._emit(
                                step,
                                "experiment_milestone",
                                "First Autonomous Experiment",
                                f"Model designed and executed its first "
                                f"autonomous experiment "
                                f"({len(_ms)} milestone(s) recorded).",
                            )
                            if m is not None:
                                self._experiment_milestone_emitted = True
                                milestone = m

        # Update tracking windows only when no milestone fired so the
        # milestone-bearing snapshot does not contaminate the baseline.
        if milestone is None:
            self._fe_window.append(fe)
            self._pe_window.append(pe)

        self._last_fe = fe
        return milestone

    # ------------------------------------------------------------------ #
    # Public accessors
    # ------------------------------------------------------------------ #
    def get_milestones(self) -> list[dict]:
        """Return all milestones as a list of JSON-serialisable dicts."""
        return [
            {
                "step": m.step,
                "type": m.type,
                "title": m.title,
                "description": m.description,
                "snapshot_step": m.snapshot_step,
            }
            for m in self.milestones
        ]


# ------------------------------------------------------------------ #
# Module-level helper
# ------------------------------------------------------------------ #
def detect_from_history(history: list[dict]) -> list[Milestone]:
    """Replay ``history`` through a fresh detector and return all milestones."""
    detector = MilestoneDetector()
    for snap in history:
        detector.analyze(snap)
    return detector.milestones
