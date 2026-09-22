# src/visualization/api_server.py
"""REST API server for history, knowledge graph, and interventions.

Extends the ZeroDataModel API with endpoints for the "Window of
Consciousness" frontend:

  GET  /history?start=<int>&end=<int>   — paginated snapshot history
  GET  /knowledge-graph                  — current KG (nodes + edges)
  POST /intervention                     — inject physics/agent changes
  GET  /story-milestones                  — key learning events timeline
  WS   /ws                               — real-time snapshot stream

Uses FastAPI if available; otherwise falls back to a simple
``http.server``-based stub.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict
from typing import Any

from .milestone_detector import MilestoneDetector, detect_from_history
from .snapshot import SnapshotCollector
from .streamer import WebSocketStreamer


# ------------------------------------------------------------------ #
# API server
# ------------------------------------------------------------------ #
class VisualizationAPI:
    """REST + WebSocket API for the visualization frontend.

    Wraps a ``SnapshotCollector`` (for history/milestones/KG) and a
    ``WebSocketStreamer`` (for real-time push). Can be embedded in a
    FastAPI app or run standalone.
    """

    def __init__(
        self,
        collector: SnapshotCollector,
        streamer: WebSocketStreamer,
        milestone_detector: MilestoneDetector | None = None,
        host: str = "localhost",
        rest_port: int = 8000,
    ):
        self.collector = collector
        self.streamer = streamer
        self.milestone_detector = milestone_detector
        self.host = host
        self.rest_port = rest_port
        self._app = None
        self._server_thread: threading.Thread | None = None
        self._interventions: list[dict] = []

    # ------------------------------------------------------------------ #
    # FastAPI app factory
    # ------------------------------------------------------------------ #
    def create_app(self):
        """Create and return a FastAPI app with all endpoints."""
        try:
            from fastapi import FastAPI, HTTPException, Query
            from fastapi.middleware.cors import CORSMiddleware
            from pydantic import BaseModel
        except ImportError:
            print("[api] FastAPI not installed; REST endpoints unavailable")
            return None

        app = FastAPI(title="ZeroDataModel Visualization API")
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # --- GET /health --- #
        @app.get("/health")
        def health():
            return {
                "status": "ok",
                "history_size": self.collector.history_size,
                "n_milestones": len(self.collector.get_milestones()),
                "n_ws_clients": self.streamer.n_clients,
                "paused": self.streamer.paused,
                "speed": self.streamer.speed,
            }

        # --- GET /history --- #
        @app.get("/history")
        def get_history(
            start: int = Query(0, ge=0),
            end: int | None = Query(None, ge=0),
        ):
            return {
                "snapshots": self.collector.get_history(start, end),
                "total": self.collector.history_size,
            }

        # --- GET /knowledge-graph --- #
        @app.get("/knowledge-graph")
        def get_kg():
            return self.collector.get_current_kg()

        # --- GET /story-milestones --- #
        @app.get("/story-milestones")
        def get_milestones():
            if self.milestone_detector is not None:
                return {"milestones": self.milestone_detector.get_milestones()}
            # Fallback: detect on the fly from collected history.
            detected = detect_from_history(self.collector.get_history())
            return {"milestones": [asdict(m) for m in detected]}

        # --- POST /intervention --- #
        class InterventionRequest(BaseModel):
            type: str = "add_object"
            params: dict = {}

        @app.post("/intervention")
        def post_intervention(req: InterventionRequest):
            self._interventions.append(req.dict())
            # Forward to streamer command queue.
            self.streamer._handle_client_message({
                "type": "intervention",
                "subtype": req.type,
                "params": req.params,
            })
            return {"status": "queued", "queue_size": len(self._interventions)}

        # --- GET /interventions --- #
        @app.get("/interventions")
        def get_interventions():
            return {"pending": self._interventions}

        # --- POST /command --- #
        class CommandRequest(BaseModel):
            type: str = "pause"

        @app.post("/command")
        def post_command(req: CommandRequest):
            self.streamer._handle_client_message(req.dict())
            return {"status": "ok", "paused": self.streamer.paused}

        self._app = app
        return app

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Start the REST server in a background thread (uvicorn)."""
        app = self.create_app()
        if app is None:
            return
        try:
            import uvicorn
            config = uvicorn.Config(app, host=self.host, port=self.rest_port, log_level="warning")
            server = uvicorn.Server(config)
            self._server_thread = threading.Thread(target=server.run, daemon=True)
            self._server_thread.start()
            print(f"[api] REST server on http://{self.host}:{self.rest_port}")
        except ImportError:
            print("[api] uvicorn not installed; REST server unavailable")

    def stop(self) -> None:
        self.streamer.stop()
