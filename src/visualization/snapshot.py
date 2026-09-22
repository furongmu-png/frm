# src/visualization/snapshot.py
"""Snapshot collector — lightweight model introspection after each think().

The ``SnapshotCollector`` reads the internal state of a ``ZeroDataModel``
instance and produces a JSON-serialisable dictionary conforming to the
"Window of Consciousness" protocol.  Collection is designed to be cheap
(< 1 ms for most fields) so it can be called after every cognitive cycle
without measurably impacting think() performance.

Protocol fields (see spec § V):
    step, timestamp, modality, frame_b64, text_block, action,
    free_energy, prediction_error, beta, belief_state,
    causal_graph, categories, kg_update, self_authoring, event,
    text_pos, latent_3d

Phase-5 additions:
    text_pos   : int   — character offset in the current text stream
                         (-1 when not reading text).
    latent_3d  : list  — first 3 components of belief_state, used by the
                         3D latent-space projection panel. [0,0,0] if no
                         belief is available.

Phase-G additions (cognitive upgrades: S4 / PCN / Hopfield):
    s4_state        : list[float] — first ≤8 dims of the S4 hidden state
                                    (HiPPO/DSS layer). Empty when ``use_s4``
                                    is off or before the first think().
    layer_errors    : dict        — per-layer PCN prediction-error norms
                                    ({"L0": float, "L1": float, "L2": float}).
                                    Empty when ``use_pcn`` is off.
    memory_retrieved: dict        — Hopfield retrieval summary
                                    ({"n_memories": int, "top1_similarity":
                                    float}). Empty when ``use_hopfield`` is off
                                    or no memories have been stored yet.

Phase-2 additions (neuro-symbolic fusion / metacognition / experiments):
    metadata        : dict        — structured cognitive-upgrade metadata
                                    forwarded from ``think()``'s signal.
                                    Contains ``cognitive_upgrades`` sub-dict
                                    with keys: ``meta_cognition``,
                                    ``logic_violations``, ``experiment``,
                                    ``reasoning_chain``. Empty when Phase 2
                                    modules are disabled. This is the primary
                                    data source for the Confidence, Logic,
                                    Experiment and Reasoning-Chain frontend
                                    panels.

These three fields are populated from ``think()``'s ``signal.metadata``
(when passed via the optional ``metadata`` argument to ``collect()``) with
a fallback to direct model attribute reads. They are always present in the
serialized dict (as empty containers when the upgrades are disabled) so
frontend panels can render without null checks.
"""

from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass, field

import numpy as np


# ------------------------------------------------------------------ #
# Snapshot dataclass
# ------------------------------------------------------------------ #
@dataclass
class Snapshot:
    """A single cognitive-cycle snapshot, JSON-serialisable."""

    step: int = 0
    timestamp: float = 0.0
    modality: str = "unknown"
    frame_b64: str | None = None        # base64-encoded PNG
    text_block: str = ""
    action: int = 0
    free_energy: float = 0.0
    prediction_error: float = 0.0
    beta: float = 0.0
    belief_state: list[float] = field(default_factory=list)
    causal_graph: dict = field(default_factory=dict)
    categories: dict = field(default_factory=dict)
    kg_update: dict = field(default_factory=dict)
    self_authoring: str | None = None
    event: str = ""
    # Phase-5 additions.
    text_pos: int = -1                 # char offset in text stream, -1 if N/A.
    latent_3d: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    # Phase-G additions (S4 / PCN / Hopfield cognitive upgrades).
    # See module docstring for the per-field contract. All three default to
    # empty containers so snapshots produced when the upgrades are disabled
    # remain JSON-serialisable and frontend-friendly.
    s4_state: list[float] = field(default_factory=list)
    layer_errors: dict = field(default_factory=dict)
    memory_retrieved: dict = field(default_factory=dict)
    # Phase-2: structured metadata forwarded from think()'s signal.
    # Contains ``cognitive_upgrades`` with Phase 2 module outputs.
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict."""
        return {
            "step": self.step,
            "timestamp": self.timestamp,
            "modality": self.modality,
            "frame_b64": self.frame_b64,
            "text_block": self.text_block,
            "action": self.action,
            "free_energy": round(self.free_energy, 6),
            "prediction_error": round(self.prediction_error, 6),
            "beta": round(self.beta, 6),
            "belief_state": [round(float(x), 6) for x in self.belief_state],
            "causal_graph": self.causal_graph,
            "categories": self.categories,
            "kg_update": self.kg_update,
            "self_authoring": self.self_authoring,
            "event": self.event,
            "text_pos": int(self.text_pos),
            "latent_3d": [round(float(x), 6) for x in self.latent_3d],
            # Phase-G: round floats to 6 dp to keep payload small + JSON-safe.
            # ``memory_retrieved`` mixes ints (n_memories) and floats
            # (top1_similarity), so round only float leaves; preserve ints
            # and any other types verbatim.
            "s4_state": [round(float(x), 6) for x in self.s4_state],
            "layer_errors": {
                k: round(float(v), 6) for k, v in self.layer_errors.items()
            },
            "memory_retrieved": {
                k: (round(float(v), 6) if isinstance(v, float) else v)
                for k, v in self.memory_retrieved.items()
            },
            # Phase-2: forward structured cognitive-upgrade metadata so the
            # Confidence / Logic / Experiment / Reasoning-Chain frontend
            # panels can render. Passes through verbatim (already JSON-safe
            # dicts written by HierarchicalZeroDataModel._run_phase2_cycle).
            "metadata": self.metadata,
        }


# ------------------------------------------------------------------ #
# Snapshot collector
# ------------------------------------------------------------------ #
class SnapshotCollector:
    """Collects lightweight snapshots from a running ZeroDataModel.

    Usage (zero-intrusion pattern):

        collector = SnapshotCollector()
        # ... after each model.think(obs):
        snap = collector.collect(model, step=step, modality="physics", ...)
        streamer.broadcast(snap.to_dict())
    """

    def __init__(self, max_history: int = 5000):
        self._history: list[dict] = []
        self._max_history = max_history
        self._milestones: list[dict] = []
        # Track KG state for incremental updates.
        self._prev_kg_nodes: set[str] = set()
        self._prev_kg_edges: set[tuple] = set()
        # Free-energy tracking for event detection.
        self._fe_history: list[float] = []
        self._last_fe: float = 0.0

    # ------------------------------------------------------------------ #
    # Main collection method
    # ------------------------------------------------------------------ #
    def collect(
        self,
        model,
        step: int = 0,
        modality: str = "unknown",
        frame: np.ndarray | None = None,
        text_block: str = "",
        action: int = 0,
        free_energy: float | None = None,
        prediction_error: float | None = None,
        beta: float | None = None,
        kg_nodes: list[dict] | None = None,
        kg_edges: list[dict] | None = None,
        self_authoring: str | None = None,
        text_pos: int = -1,
        metadata: dict | None = None,
    ) -> Snapshot:
        """Collect a snapshot from the model.

        Parameters
        ----------
        model : ZeroDataModel
            The model instance (read-only access to internal state).
        step : int
            Current cognitive-cycle step number.
        modality : str
            Active modality label: "physics", "text", "image", "audio", "code".
        frame : np.ndarray | None
            Optional raw frame (e.g. 128×128 uint8) to encode as base64 PNG.
        text_block : str
            The text currently being processed (if modality == "text").
        action : int
            Action taken this step.
        free_energy, prediction_error, beta : float | None
            Override values; if None, they are read from the model.
        kg_nodes, kg_edges : list[dict] | None
            Knowledge graph state for incremental diff.
        self_authoring : str | None
            Self-authored text (if generated this step).
        text_pos : int
            Character offset in the current text stream (call
            ``stream.tell()`` if a TextStream is being read). Pass -1
            when not reading text. Forwarded verbatim into the snapshot.
        metadata : dict | None
            Optional ``signal.metadata`` produced by ``model.think()``.
            Phase G cognitive upgrades (S4 / PCN / Hopfield) write their
            per-cycle state into ``metadata["cognitive_upgrades"]``. When
            provided, this dict is the primary source for the new
            ``s4_state`` / ``layer_errors`` / ``memory_retrieved`` snapshot
            fields; when omitted, the collector falls back to direct model
            attribute reads (``model.s4_state_cache`` etc.). Pass ``None``
            for legacy callers — the new fields will simply be empty.
        """
        ts = time.time()

        # --- Belief state --- #
        try:
            gm = model.active_inference.generative_model
            belief = gm.belief_state
            belief_list = np.asarray(belief, dtype=float).flatten().tolist()
        except Exception:
            belief_list = []

        # --- Free energy (pragmatic) --- #
        if free_energy is None:
            try:
                free_energy = float(model.active_inference.free_energy_history[-1]) \
                    if len(model.active_inference.free_energy_history) > 0 else 0.0
            except Exception:
                free_energy = 0.0

        # --- Prediction error --- #
        if prediction_error is None:
            try:
                prediction_error = float(
                    np.linalg.norm(belief - model.active_inference.generative_model.predict_observation(belief))
                )
            except Exception:
                prediction_error = 0.0

        # --- Beta (exploration strength) --- #
        if beta is None:
            try:
                beta = float(model.active_inference._exploration_step)
            except Exception:
                beta = 0.0

        # --- Frame encoding --- #
        frame_b64 = None
        if frame is not None:
            frame_b64 = self._encode_frame_png(frame)

        # --- Causal graph (simplified) --- #
        causal_graph = self._extract_causal_graph(model)

        # --- Categories --- #
        categories = self._extract_categories(model)

        # --- KG incremental update --- #
        kg_update = self._compute_kg_diff(kg_nodes, kg_edges)

        # --- Event detection --- #
        event = self._detect_event(free_energy, modality, kg_update)

        # --- Latent 3D projection (first 3 belief components) --- #
        if belief_list and len(belief_list) >= 3:
            latent_3d = [float(belief_list[0]), float(belief_list[1]), float(belief_list[2])]
        else:
            latent_3d = [0.0, 0.0, 0.0]

        # --- Phase G: S4 / PCN / Hopfield cognitive-upgrade state --- #
        s4_state, layer_errors, memory_retrieved = self._extract_phase_g_state(
            model, metadata
        )

        # --- Phase 2: forward structured metadata for frontend panels --- #
        # The ``metadata`` dict (from think()'s signal) may contain
        # ``cognitive_upgrades`` with Phase 2 module outputs (meta_cognition,
        # logic_violations, experiment, reasoning_chain). Forward it
        # verbatim so the Confidence / Logic / Experiment / Reasoning-Chain
        # panels can render. Guard against non-serialisable values.
        snap_metadata = self._extract_metadata(metadata)

        snap = Snapshot(
            step=step,
            timestamp=ts,
            modality=modality,
            frame_b64=frame_b64,
            text_block=text_block[:500] if text_block else "",
            action=action,
            free_energy=float(free_energy),
            prediction_error=float(prediction_error),
            beta=float(beta),
            belief_state=belief_list,
            causal_graph=causal_graph,
            categories=categories,
            kg_update=kg_update,
            self_authoring=self_authoring,
            event=event,
            text_pos=int(text_pos),
            latent_3d=latent_3d,
            s4_state=s4_state,
            layer_errors=layer_errors,
            memory_retrieved=memory_retrieved,
            metadata=snap_metadata,
        )

        # Store in history.
        snap_dict = snap.to_dict()
        self._history.append(snap_dict)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        # Track milestones.
        self._check_milestone(snap_dict)

        return snap

    # ------------------------------------------------------------------ #
    # Frame encoding
    # ------------------------------------------------------------------ #
    @staticmethod
    def _encode_frame_png(frame: np.ndarray) -> str:
        """Encode a numpy array as base64 PNG."""
        try:
            from PIL import Image
            arr = np.asarray(frame)
            if arr.ndim == 2:
                img = Image.fromarray(arr.astype(np.uint8), mode="L")
            elif arr.ndim == 3 and arr.shape[2] == 3:
                img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
            elif arr.ndim == 3 and arr.shape[2] == 4:
                img = Image.fromarray(arr.astype(np.uint8), mode="RGBA")
            else:
                img = Image.fromarray(arr.astype(np.uint8))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except ImportError:
            # Fallback: raw bytes as base64 (no PNG compression).
            arr = np.asarray(frame, dtype=np.uint8).flatten()
            return base64.b64encode(arr.tobytes()).decode("ascii")

    # ------------------------------------------------------------------ #
    # Causal graph extraction
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_causal_graph(model) -> dict:
        """Extract a simplified causal graph from the model's emergence engine."""
        try:
            emergence = model.emergence
            memory = emergence.memory
            patterns = memory._patterns
            labels = memory._labels
            nodes = []
            edges = []
            for i, (pat, label) in enumerate(zip(patterns, labels)):
                nodes.append({"id": i, "label": str(label) if label else f"node_{i}"})
                if i > 0:
                    # Simple chain: each node connected to previous.
                    strength = float(np.dot(pat, patterns[i - 1])) / (
                        np.linalg.norm(pat) * np.linalg.norm(patterns[i - 1]) + 1e-12
                    )
                    edges.append({
                        "source": i - 1, "target": i,
                        "strength": round(strength, 4),
                    })
            return {"nodes": nodes[:20], "edges": edges[:20]}
        except Exception:
            return {"nodes": [], "edges": []}

    # ------------------------------------------------------------------ #
    # Categories extraction
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_categories(model) -> dict:
        """Extract category engine state."""
        try:
            cat = model.category_engine
            categories = {}
            for name, cat_obj in list(cat.categories.items())[:10]:
                categories[name] = {
                    "n_objects": len(cat_obj.objects),
                }
            return {
                "n_categories": len(cat.categories),
                "categories": categories,
                "n_functors": len(cat.functors),
            }
        except Exception:
            return {}

    # ------------------------------------------------------------------ #
    # Phase G state extraction (S4 / PCN / Hopfield)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_phase_g_state(model, metadata: dict | None) -> tuple:
        """Extract Phase G cognitive-upgrade state for the snapshot.

        Returns a 3-tuple ``(s4_state, layer_errors, memory_retrieved)``:

        - ``s4_state``: ``list[float]`` — first ≤8 dims of the S4 hidden
          state. Empty list when ``use_s4`` is off or unavailable.
        - ``layer_errors``: ``dict[str, float]`` — per-layer PCN
          prediction-error norms ``{"L0":..., "L1":..., "L2":...}``.
          Empty dict when ``use_pcn`` is off.
        - ``memory_retrieved``: ``dict`` — Hopfield retrieval summary
          ``{"n_memories": int, "top1_similarity": float}``.
          Empty dict when ``use_hopfield`` is off or no memories stored.

        Resolution order per field:
            1. ``metadata["cognitive_upgrades"]`` sub-dict (when ``metadata``
               is passed from ``think()``'s signal — the canonical source
               populated by the hooks in ``model.py``).
            2. Direct model attribute read (fallback for legacy callers
               that don't forward ``signal.metadata``). For S4 this reads
               ``model.s4_state_cache``; for PCN/Hopfield there is no
               cached attribute on the model, so the fields stay empty in
               the fallback path.

        Every branch is wrapped in try/except so a misbehaving upgrade
        never crashes snapshot collection — it just yields an empty field.
        """
        s4_state: list[float] = []
        layer_errors: dict = {}
        memory_retrieved: dict = {}

        # --- (1) Primary source: signal.metadata["cognitive_upgrades"] --- #
        if isinstance(metadata, dict):
            upgrades = metadata.get("cognitive_upgrades")
            if isinstance(upgrades, dict):
                # S4: cached state list (already capped at 8 dims in model.py)
                _s4_meta = upgrades.get("s4_state")
                if isinstance(_s4_meta, list):
                    try:
                        s4_state = [float(x) for x in _s4_meta if np.isfinite(float(x))]
                    except (TypeError, ValueError):
                        s4_state = []
                # PCN: run_pcn_cycle() returns {"layer_errors": {...}, ...}
                _pcn_meta = upgrades.get("pcn")
                if isinstance(_pcn_meta, dict):
                    _le = _pcn_meta.get("layer_errors")
                    if isinstance(_le, dict):
                        try:
                            layer_errors = {
                                str(k): float(v)
                                for k, v in _le.items()
                                if np.isfinite(float(v))
                            }
                        except (TypeError, ValueError):
                            layer_errors = {}
                # Hopfield: {"n_memories": int, "top1_similarity": float}
                _mem_meta = upgrades.get("memory_retrieved")
                if isinstance(_mem_meta, dict):
                    memory_retrieved = dict(_mem_meta)

        # --- (2) Fallback: direct model attribute reads ------------- #
        # S4 state is cached on model.s4_state_cache by think() even when
        # metadata isn't forwarded; PCN/Hopfield have no cached attribute,
        # so they remain empty in the fallback path.
        if not s4_state:
            _cache = getattr(model, "s4_state_cache", None)
            if isinstance(_cache, list) and _cache:
                try:
                    s4_state = [float(x) for x in _cache if np.isfinite(float(x))]
                except (TypeError, ValueError):
                    s4_state = []

        return s4_state, layer_errors, memory_retrieved

    # ------------------------------------------------------------------ #
    # Phase 2 metadata extraction (cognitive_upgrades forwarding)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_metadata(metadata: dict | None) -> dict:
        """Extract JSON-safe metadata from think()'s signal for the frontend.

        Returns a dict with ``cognitive_upgrades`` (if present) containing
        Phase 2 module outputs. All values are coerced to JSON-safe types
        (dict/list/str/int/float/bool/None) so the snapshot remains
        serialisable. Non-serialisable values are stringified as a
        fallback.
        """
        if not isinstance(metadata, dict):
            return {}

        def _safe(v):
            """Recursively coerce to JSON-safe types."""
            if isinstance(v, (str, int, float, bool)) or v is None:
                return v
            if isinstance(v, dict):
                return {str(k): _safe(val) for k, val in v.items()}
            if isinstance(v, (list, tuple)):
                return [_safe(item) for item in v]
            # numpy arrays, ndarrays, custom objects → string fallback.
            try:
                return float(v)
            except (TypeError, ValueError):
                return str(v)

        result: dict = {}
        for key, val in metadata.items():
            result[str(key)] = _safe(val)
        return result

    # ------------------------------------------------------------------ #
    # KG incremental diff
    # ------------------------------------------------------------------ #
    def _compute_kg_diff(
        self,
        nodes: list[dict] | None,
        edges: list[dict] | None,
    ) -> dict:
        """Compute incremental KG changes since last snapshot."""
        if nodes is None and edges is None:
            return {"new_nodes": [], "new_edges": []}

        new_nodes = []
        new_edges = []
        if nodes is not None:
            current_node_ids = set()
            for n in nodes:
                nid = str(n.get("id", n.get("title", "")))
                current_node_ids.add(nid)
                if nid not in self._prev_kg_nodes:
                    new_nodes.append(nid)
            self._prev_kg_nodes = current_node_ids

        if edges is not None:
            current_edge_ids = set()
            for e in edges:
                src = str(e.get("source", ""))
                tgt = str(e.get("target", ""))
                eid = (src, tgt)
                current_edge_ids.add(eid)
                if eid not in self._prev_kg_edges:
                    new_edges.append({
                        "source": src, "target": tgt,
                        "weight": e.get("weight", 1.0),
                    })
            self._prev_kg_edges = current_edge_ids

        return {"new_nodes": new_nodes, "new_edges": new_edges}

    # ------------------------------------------------------------------ #
    # Event detection
    # ------------------------------------------------------------------ #
    def _detect_event(self, fe: float, modality: str, kg_update: dict) -> str:
        """Detect notable events for milestone tracking."""
        self._fe_history.append(fe)
        if len(self._fe_history) > 10:
            self._fe_history = self._fe_history[-10:]

        # High surprise: FE spike.
        if len(self._fe_history) >= 3:
            mean_recent = np.mean(self._fe_history[-3:])
            if fe > 2 * mean_recent and fe > 0.01:
                return "high_surprise"

        # New KG node discovered.
        if kg_update.get("new_nodes"):
            return "new_concept"

        # Collision (physics-specific, detected externally).
        if modality == "physics" and fe > self._last_fe * 3 and self._last_fe > 0:
            return "collision"

        self._last_fe = fe
        return ""

    # ------------------------------------------------------------------ #
    # Milestone tracking
    # ------------------------------------------------------------------ #
    def _check_milestone(self, snap: dict) -> None:
        """Record milestones for the story-mode timeline."""
        if snap["event"]:
            self._milestones.append({
                "step": snap["step"],
                "event": snap["event"],
                "description": f"Step {snap['step']}: {snap['event']} "
                               f"(FE={snap['free_energy']:.4f}, mod={snap['modality']})",
            })

        # FE reduction milestone (every 200 steps of progress).
        if snap["step"] > 0 and snap["step"] % 200 == 0:
            self._milestones.append({
                "step": snap["step"],
                "event": "checkpoint",
                "description": f"Reached step {snap['step']}, "
                               f"FE={snap['free_energy']:.4f}",
            })

    # ------------------------------------------------------------------ #
    # Public accessors
    # ------------------------------------------------------------------ #
    def get_history(self, start: int = 0, end: int | None = None) -> list[dict]:
        """Return historical snapshots in [start, end) range."""
        if end is None:
            end = len(self._history)
        return self._history[start:end]

    def get_milestones(self) -> list[dict]:
        """Return all recorded milestones."""
        return self._milestones

    def get_current_kg(self) -> dict:
        """Return the current KG state (nodes + edges from diff tracking)."""
        return {
            "nodes": list(self._prev_kg_nodes),
            "edges": [
                {"source": s, "target": t}
                for s, t in self._prev_kg_edges
            ],
        }

    @property
    def history_size(self) -> int:
        return len(self._history)
