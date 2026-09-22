# src/visualization/streamer.py
"""WebSocket streamer — broadcasts snapshots to connected frontends.

Runs an ``asyncio`` WebSocket server in a background thread so it can
coexist with the model's synchronous think() loop. Supports:

  - **Push**: ``broadcast(snapshot_dict)`` (alias ``push``) sends to all
    connected clients. New clients automatically receive the last 500
    history items on connect.
  - **Control**: clients send commands (pause, resume, step, set_speed)
    which are queued for the model loop to consume.
    - ``pause`` / ``resume``: toggle the ``paused`` flag.
    - ``step``: set the ``single_step`` event so the broadcast loop
      releases exactly one snapshot, then re-pauses.
    - ``set_speed``: adjusts the broadcast loop sleep interval
      (base 0.05s; effective sleep = 0.05 / speed).
  - **History**: a deque(maxlen=2000) of recent snapshots; new clients
    receive the last 500 on connect.
  - **Intervention**: clients send intervention parameters (add object,
    apply force, inject question) which are forwarded to the model.

Usage:

    streamer = WebSocketStreamer(host="localhost", port=8765)
    streamer.start()  # starts background thread
    # ... in model loop:
    streamer.broadcast(snap.to_dict())
    cmd = streamer.get_pending_command()  # non-blocking
    if cmd:
        handle(cmd)
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import deque
from typing import Any

try:
    import websockets
    from websockets.server import serve
    _HAS_WEBSOCKETS = True
except ImportError:
    _HAS_WEBSOCKETS = False


# ------------------------------------------------------------------ #
# WebSocket streamer
# ------------------------------------------------------------------ #
class WebSocketStreamer:
    """Async WebSocket server running in a background thread.

    The model's synchronous loop calls ``broadcast()`` (thread-safe)
    to push snapshots. Control commands from clients are queued and
    retrieved via ``get_pending_command()``.
    """

    # Base broadcast loop sleep (seconds). Effective sleep is
    # ``BASE_SLEEP / speed`` so higher speed => shorter interval.
    BASE_SLEEP = 0.05
    # Maximum number of snapshots retained in history.
    MAX_HISTORY = 2000
    # Number of history items sent to a newly connected client.
    HISTORY_SEND_COUNT = 500

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        max_command_queue: int = 100,
    ):
        self.host = host
        self.port = port
        self._clients: set = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server_task: asyncio.Task | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._paused = False
        self._speed = 1.0

        # single_step event: when set, the broadcast loop releases
        # exactly one queued snapshot then re-pauses.
        self._single_step = threading.Event()

        # Thread-safe command queue (filled by WebSocket clients).
        self._command_queue: deque = deque(maxlen=max_command_queue)
        self._lock = threading.Lock()

        # Broadcast queue (filled by the model thread, consumed by asyncio).
        self._broadcast_queue: asyncio.Queue = None  # set in _run

        # Snapshot history (thread-safe via _history_lock).
        self._history: deque = deque(maxlen=self.MAX_HISTORY)
        self._history_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Start the WebSocket server in a background thread."""
        if self._running:
            return
        if not _HAS_WEBSOCKETS:
            print("[streamer] websockets not installed; running in stub mode")
            self._running = True
            return

        self._thread = threading.Thread(target=self._run_thread, daemon=True)
        self._thread.start()
        self._running = True
        # Wait for the event loop to be ready.
        time.sleep(0.3)

    def stop(self) -> None:
        """Stop the server."""
        self._running = False
        # Wake any blocked single_step wait.
        self._single_step.set()
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._shutdown(), self._loop
            )
        if self._thread:
            self._thread.join(timeout=3.0)

    async def _shutdown(self) -> None:
        if self._server_task:
            self._server_task.cancel()
        # Close all clients.
        for ws in list(self._clients):
            await ws.close()
        self._clients.clear()

    def _run_thread(self) -> None:
        """Run the asyncio event loop in this thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._broadcast_queue = asyncio.Queue(maxsize=200)
        self._loop.run_until_complete(self._main())

    async def _main(self) -> None:
        """Main asyncio coroutine: start server + process broadcasts.

        This is the broadcast loop. It honours the ``paused`` flag and
        ``single_step`` event:

          - If paused AND single_step is not set → sleep BASE_SLEEP and
            continue (drop any queued snapshot, do not broadcast).
          - If single_step is set → clear it and process exactly one
            queued snapshot, then continue (still paused).
          - Otherwise (not paused) → process one queued snapshot.
          - After processing, sleep ``BASE_SLEEP / speed`` to throttle
            the broadcast rate.
        """
        async with serve(self._handle_client, self.host, self.port):
            print(f"[streamer] WebSocket server listening on ws://{self.host}:{self.port}")
            while self._running:
                # Paused gate: if paused and no single_step signal,
                # sleep and continue without broadcasting.
                if self._paused and not self._single_step.is_set():
                    await asyncio.sleep(self.BASE_SLEEP)
                    continue
                # If single_step is set, consume it (release one snapshot).
                if self._single_step.is_set():
                    self._single_step.clear()

                try:
                    msg = await asyncio.wait_for(
                        self._broadcast_queue.get(), timeout=0.1
                    )
                    if msg is not None:
                        self._append_history(msg)
                        await self._do_broadcast(msg)
                except asyncio.TimeoutError:
                    pass
                except Exception as e:
                    print(f"[streamer] broadcast error: {e}")

                # Speed-controlled throttle: higher speed => shorter sleep.
                await asyncio.sleep(self.BASE_SLEEP / max(self._speed, 0.01))

    # ------------------------------------------------------------------ #
    # Client handler
    # ------------------------------------------------------------------ #
    async def _handle_client(self, websocket) -> None:
        """Handle a single WebSocket client connection.

        On connect, automatically sends the last ``HISTORY_SEND_COUNT``
        history items as a ``history`` message so the frontend can
        populate its time-series immediately.
        """
        self._clients.add(websocket)
        peer = getattr(websocket, "remote_address", "?")
        print(f"[streamer] client connected: {peer} (total: {len(self._clients)})")
        # Send recent history to the new client.
        await self._send_history(websocket)
        try:
            async for raw in websocket:
                try:
                    msg = json.loads(raw)
                    self._handle_client_message(msg)
                except json.JSONDecodeError:
                    pass
        except Exception:
            pass  # Client disconnected.
        finally:
            self._clients.discard(websocket)
            print(f"[streamer] client disconnected (total: {len(self._clients)})")

    async def _send_history(self, websocket) -> None:
        """Send the last ``HISTORY_SEND_COUNT`` snapshots to a client."""
        with self._history_lock:
            items = list(self._history)[-self.HISTORY_SEND_COUNT:]
        if not items:
            return
        try:
            data = json.dumps(
                {"type": "history", "snapshots": items}, default=str
            )
            await websocket.send(data)
        except Exception as e:
            print(f"[streamer] send_history error: {e}")

    def _handle_client_message(self, msg: dict) -> None:
        """Process a command from a client.

        Commands:
          - ``pause``  → set paused = True.
          - ``resume`` → set paused = False, clear any single_step.
          - ``step``   → set paused = True, then set single_step so the
                         broadcast loop releases exactly one snapshot.
          - ``set_speed`` → adjust the broadcast loop sleep interval.
        """
        cmd_type = msg.get("type", "")
        if cmd_type == "pause":
            self._paused = True
        elif cmd_type == "resume":
            self._paused = False
            self._single_step.clear()
        elif cmd_type == "step":
            self._paused = True  # ensure paused
            self._single_step.set()  # release one snapshot
        elif cmd_type == "set_speed":
            self._speed = float(msg.get("speed", 1.0))
        # Queue all commands for the model loop.
        with self._lock:
            self._command_queue.append(msg)

    # ------------------------------------------------------------------ #
    # History (called from the asyncio loop)
    # ------------------------------------------------------------------ #
    def _append_history(self, msg: dict) -> None:
        """Append a snapshot to the history deque (thread-safe)."""
        with self._history_lock:
            self._history.append(msg)

    def get_history(self, start: int = 0, end: int | None = None) -> list[dict]:
        """Return history items in ``[start, end)`` (thread-safe copy)."""
        with self._history_lock:
            items = list(self._history)
        if end is None:
            return items[start:]
        return items[start:end]

    # ------------------------------------------------------------------ #
    # Broadcast (called from the model thread)
    # ------------------------------------------------------------------ #
    def broadcast(self, snapshot: dict) -> None:
        """Push a snapshot to all connected clients (thread-safe).

        The snapshot is queued; the asyncio broadcast loop will forward
        it to clients (subject to the paused / single_step gate) and
        append it to history.
        """
        if not self._running or not self._loop:
            return
        try:
            self._loop.call_soon_threadsafe(
                self._broadcast_queue.put_nowait, snapshot
            )
        except Exception:
            pass  # Queue full or loop closed.

    # Alias: ``push`` is the user-facing name in the spec.
    push = broadcast

    async def _do_broadcast(self, msg: dict) -> None:
        """Send a message to all connected clients."""
        if not self._clients:
            return
        data = json.dumps(msg, default=str)
        # Send to all, ignoring failures.
        dead = set()
        for ws in self._clients:
            try:
                await ws.send(data)
            except Exception:
                dead.add(ws)
        self._clients -= dead

    # ------------------------------------------------------------------ #
    # Command accessors (called from the model thread)
    # ------------------------------------------------------------------ #
    def get_pending_command(self) -> dict | None:
        """Non-blocking: return the next pending command, or None."""
        with self._lock:
            if self._command_queue:
                return self._command_queue.popleft()
            return None

    def clear_single_step(self) -> None:
        """Clear the single_step signal (called by the model after a step)."""
        self._single_step.clear()

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def single_step_pending(self) -> bool:
        """True if a single-step is requested (model may consult this)."""
        return self._single_step.is_set()

    @property
    def history_size(self) -> int:
        with self._history_lock:
            return len(self._history)

    @property
    def n_clients(self) -> int:
        return len(self._clients)

    @property
    def is_running(self) -> bool:
        return self._running
