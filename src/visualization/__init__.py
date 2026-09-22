# src/visualization/__init__.py
"""Visualization package for ZeroDataModel — the "Window of Consciousness".

Provides:
  - SnapshotCollector: lightweight introspection of model state after each think()
  - WebSocketStreamer: broadcasts snapshots to connected frontends
  - APIServer: REST + WebSocket endpoints for history, knowledge graph, interventions
"""

from .snapshot import SnapshotCollector, Snapshot
from .streamer import WebSocketStreamer

__all__ = ["SnapshotCollector", "Snapshot", "WebSocketStreamer"]
