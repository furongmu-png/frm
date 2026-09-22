# experiments/physics_sandbox.py
"""Minimal 2D physics sandbox for ZeroDataModel sensory input.

Single-file, numpy-only implementation. Renders 128x128 grayscale frames
suitable as the sole sensory stream for ``ZeroDataModel``. All randomness
flows through ``numpy.random.Generator`` so a fixed ``seed`` reproduces
the exact initial state and frame sequence across platforms.

This file is **intentionally self-contained** — it does NOT import from
``examples/physics_sandbox.py`` and may diverge from it. The focus here
is on clarity and minimal dependencies for tutorial / experimentation
purposes.

Run with:  python experiments/physics_sandbox.py
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

# Optional: only used by ``_demo`` to actually display frames.
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_MPL = False


# ------------------------------------------------------------------ #
# Module-level constants — keep the magic numbers in one place.
# ------------------------------------------------------------------ #
WORLD_SIZE = 128          # canvas side in pixels
AGENT_INDEX = 0           # bodies[0] is the force-controlled agent
DELTA_V = 1.0             # pixels / frame^2 applied by step(action)
AGENT_GRAY = 255          # brightest pixel so the agent stands out
BODY_GRAY_MIN = 200      # inclusive lower bound for non-agent bodies
BODY_GRAY_MAX = 254      # inclusive upper bound for non-agent bodies
MIN_OBJECTS = 1
MAX_OBJECTS = 3


class Shape(Enum):
    """Supported body shapes."""

    RECT = "rect"
    CIRCLE = "circle"


@dataclass
class Body:
    """A single rigid body in the world.

    All collision tests use the *bounding circle* of the body
    (``radius``) — for circles this is exact, for rectangles we use
    the half-diagonal so the rect is fully enclosed. This is the
    user-approved simplification that keeps the collision code O(n^2)
    and free of corner-case geometry.
    """

    cx: float
    cy: float
    vx: float
    vy: float
    mass: float
    shape: Shape
    radius: float            # circle radius OR half-diagonal of rect
    width: float = 0.0       # only meaningful for RECT
    height: float = 0.0     # only meaningful for RECT
    gray: int = BODY_GRAY_MIN


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def _bounding_radius(shape: Shape, width: float, height: float,
                     radius: float) -> float:
    """Return the bounding-circle radius of a body."""
    if shape is Shape.CIRCLE:
        return float(radius)
    # For a rectangle, use the half-diagonal so the whole rect is
    # enclosed by the bounding circle.
    return float(np.sqrt(width * width + height * height) / 2.0)


# ------------------------------------------------------------------ #
# Main class
# ------------------------------------------------------------------ #
class PhysicsSandbox:
    """A minimal 2D physics sandbox.

    The world is a ``WORLD_SIZE x WORLD_SIZE`` pixel canvas. Bodies
    obey frictionless straight-line motion and undergo fully elastic
    collisions (impulse-based, conserving momentum and kinetic energy
    along the contact normal). The first body is the *agent* and is
    controlled via ``step(action)``.

    Parameters
    ----------
    num_objects : int
        Number of bodies in the scene, including the agent. Must be
        in ``[1, 3]``.
    seed : int, optional
        RNG seed for reproducible initial state.

    Attributes
    ----------
    bodies : list[Body]
        The current rigid bodies. ``bodies[0]`` is the agent.
    frame : np.ndarray
        The last rendered frame, shape ``(WORLD_SIZE, WORLD_SIZE)``,
        dtype ``uint8``.
    """

    # ------------------------------------------------------------------ #
    # Construction / reset
    # ------------------------------------------------------------------ #
    def __init__(self, num_objects: int = 2, seed: Optional[int] = 42):
        if not isinstance(num_objects, int) or num_objects < MIN_OBJECTS \
                or num_objects > MAX_OBJECTS:
            raise ValueError(
                f"num_objects must be in [{MIN_OBJECTS}, {MAX_OBJECTS}], "
                f"got {num_objects!r}"
            )
        self.num_objects = num_objects
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.bodies: list[Body] = []
        self.frame: np.ndarray = np.zeros(
            (WORLD_SIZE, WORLD_SIZE), dtype=np.uint8
        )
        # Generate the initial scene and render the first frame.
        self.reset(seed=seed)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Re-randomise the scene and return the initial frame.

        If ``seed`` is ``None``, the existing RNG is continued (so
        successive ``reset()`` calls produce different scenes). If
        ``seed`` is an int, a fresh RNG is constructed for full
        reproducibility.
        """
        if seed is not None:
            self.seed = seed
            self._rng = np.random.default_rng(seed)
        self.bodies = self._sample_non_overlapping()
        self.frame = self.render()
        return self.frame

    # ------------------------------------------------------------------ #
    # Random scene generation
    # ------------------------------------------------------------------ #
    def _sample_non_overlapping(self) -> list[Body]:
        """Sample bodies that do not overlap and lie fully inside the world.

        Each body's centre is drawn uniformly inside the safe margin
        ``[r+1, WORLD_SIZE-r-1]`` so the body cannot start touching a
        wall. Rejection sampling guarantees non-overlap between bodies.
        """
        bodies: list[Body] = []
        max_attempts = 200
        for i in range(self.num_objects):
            for _ in range(max_attempts):
                body = self._sample_one_body(index=i, existing=bodies)
                if body is None:
                    continue
                # Check non-overlap with all previously placed bodies.
                ok = True
                for other in bodies:
                    dx = body.cx - other.cx
                    dy = body.cy - other.cy
                    dist = float(np.hypot(dx, dy))
                    if dist < (body.radius + other.radius) + 1.0:
                        ok = False
                        break
                if ok:
                    bodies.append(body)
                    break
            else:
                # Exhausted attempts — fall back to placing the body
                # at the centre with zero velocity so the simulation
                # can still proceed. This is extremely unlikely for
                # ``num_objects <= 3`` in a 128x128 world.
                body = self._sample_one_body(index=i, existing=bodies)
                if body is not None:
                    bodies.append(body)
        return bodies

    def _sample_one_body(self, index: int,
                         existing: list[Body]) -> Optional[Body]:
        """Sample a single body. The agent (index 0) is a circle."""
        if index == AGENT_INDEX:
            shape = Shape.CIRCLE
            radius = float(self._rng.uniform(4.0, 6.0))
            width = 0.0
            height = 0.0
            mass = 1.0
            gray = AGENT_GRAY
        else:
            # Non-agent bodies are randomly rect or circle.
            shape = Shape.CIRCLE if self._rng.random() < 0.5 else Shape.RECT
            if shape is Shape.CIRCLE:
                radius = float(self._rng.uniform(5.0, 8.0))
                width = 0.0
                height = 0.0
            else:
                width = float(self._rng.uniform(8.0, 14.0))
                height = float(self._rng.uniform(8.0, 14.0))
                radius = _bounding_radius(shape, width, height, 0.0)
            mass = float(self._rng.uniform(0.8, 1.5))
            # int() truncation gives an int in [MIN, MAX].
            gray = int(self._rng.integers(BODY_GRAY_MIN, BODY_GRAY_MAX + 1))
        # Safe centre: keep the body fully inside the canvas.
        margin = radius + 1.0
        cx = float(self._rng.uniform(margin, WORLD_SIZE - margin))
        cy = float(self._rng.uniform(margin, WORLD_SIZE - margin))
        # Velocity: small, biased so the world isn't static.
        vx = float(self._rng.uniform(-1.5, 1.5))
        vy = float(self._rng.uniform(-1.5, 1.5))
        return Body(
            cx=cx, cy=cy, vx=vx, vy=vy,
            mass=mass, shape=shape, radius=radius,
            width=width, height=height, gray=gray,
        )

    # ------------------------------------------------------------------ #
    # Physics step
    # ------------------------------------------------------------------ #
    def step(self, action: int) -> np.ndarray:
        """Advance the simulation by one frame.

        Parameters
        ----------
        action : int
            One of ``0`` (left), ``1`` (right), ``2`` (up), ``3`` (no-op).
            Applies a fixed impulse ``DELTA_V`` to the agent's velocity
            *before* the physics integration step.

        Returns
        -------
        np.ndarray
            The new frame, shape ``(WORLD_SIZE, WORLD_SIZE)``, dtype ``uint8``.
        """
        if not isinstance(action, (int, np.integer)) or action < 0 or action > 3:
            raise ValueError(f"action must be an int in [0, 3], got {action!r}")
        # 1) Apply the agent's action — modify velocity only.
        agent = self.bodies[AGENT_INDEX]
        if action == 0:    # left
            agent.vx -= DELTA_V
        elif action == 1:  # right
            agent.vx += DELTA_V
        elif action == 2:  # up
            agent.vy -= DELTA_V   # image y grows downward, so "up" = -y
        # action == 3: no-op
        # 2) Integrate positions (frictionless, dt = 1).
        for b in self.bodies:
            b.cx += b.vx
            b.cy += b.vy
        # 3) Resolve wall collisions: reflect velocity, push body back in.
        self._resolve_walls()
        # 4) Resolve pairwise body collisions (elastic, impulse-based).
        self._resolve_body_collisions()
        # 5) Render and return.
        self.frame = self.render()
        return self.frame

    # ------------------------------------------------------------------ #
    # Collision resolution
    # ------------------------------------------------------------------ #
    def _resolve_walls(self) -> None:
        """Reflect any body that crosses a world boundary."""
        for b in self.bodies:
            # Left wall.
            if b.cx - b.radius < 0:
                b.cx = b.radius
                b.vx = -b.vx
            # Right wall.
            if b.cx + b.radius > WORLD_SIZE:
                b.cx = WORLD_SIZE - b.radius
                b.vx = -b.vx
            # Top wall.
            if b.cy - b.radius < 0:
                b.cy = b.radius
                b.vy = -b.vy
            # Bottom wall.
            if b.cy + b.radius > WORLD_SIZE:
                b.cy = WORLD_SIZE - b.radius
                b.vy = -b.vy

    def _resolve_body_collisions(self) -> None:
        """Resolve all pairwise collisions using elastic impulse.

        For each colliding pair, the velocity components along the
        contact normal are exchanged using the standard 1D elastic
        collision formula (mass-weighted). Tangential components are
        preserved, which conserves both linear momentum and kinetic
        energy for the frictionless, non-rotating case.
        """
        n = len(self.bodies)
        for i in range(n):
            for j in range(i + 1, n):
                a = self.bodies[i]
                b = self.bodies[j]
                dx = b.cx - a.cx
                dy = b.cy - a.cy
                dist = float(np.hypot(dx, dy))
                rsum = a.radius + b.radius
                if dist >= rsum:
                    continue  # no collision
                # Avoid divide-by-zero on perfectly overlapping bodies.
                if dist < 1e-9:
                    nx, ny = 1.0, 0.0
                    dist = 0.0
                else:
                    nx = dx / dist
                    ny = dy / dist
                # Positional correction: push bodies apart along the
                # normal so they no longer overlap.
                overlap = rsum - dist
                ma = max(a.mass, 1e-9)
                mb = max(b.mass, 1e-9)
                total = ma + mb
                a.cx -= nx * overlap * (mb / total)
                a.cy -= ny * overlap * (mb / total)
                b.cx += nx * overlap * (ma / total)
                b.cy += ny * overlap * (ma / total)
                # Velocity update along the normal (1D elastic collision).
                # v1n' = (v1n*(m1-m2) + 2*m2*v2n) / (m1+m2)
                # v2n' = (v2n*(m2-m1) + 2*m1*v1n) / (m1+m2)
                v1n = a.vx * nx + a.vy * ny
                v2n = b.vx * nx + b.vy * ny
                # Skip if they are separating.
                if v2n - v1n > 0:
                    continue
                v1n_new = (v1n * (ma - mb) + 2.0 * mb * v2n) / total
                v2n_new = (v2n * (mb - ma) + 2.0 * ma * v1n) / total
                # Apply delta along the normal.
                a.vx += (v1n_new - v1n) * nx
                a.vy += (v1n_new - v1n) * ny
                b.vx += (v2n_new - v2n) * nx
                b.vy += (v2n_new - v2n) * ny

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def render(self) -> np.ndarray:
        """Render the current scene to a grayscale frame.

        Returns
        -------
        np.ndarray
            ``(WORLD_SIZE, WORLD_SIZE)`` uint8 array. Background is 0
            (black); bodies are filled with their ``gray`` value.
        """
        canvas = np.zeros((WORLD_SIZE, WORLD_SIZE), dtype=np.uint8)
        for body in self.bodies:
            self._draw_body(canvas, body)
        return canvas

    def _draw_body(self, canvas: np.ndarray, body: Body) -> None:
        """Rasterise a single body onto ``canvas`` (in place)."""
        # Compute the bounding box of pixel indices we need to touch.
        r = body.radius
        x0 = int(np.floor(body.cx - r))
        x1 = int(np.ceil(body.cx + r))
        y0 = int(np.floor(body.cy - r))
        y1 = int(np.ceil(body.cy + r))
        # Clip to canvas bounds.
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(WORLD_SIZE - 1, x1)
        y1 = min(WORLD_SIZE - 1, y1)
        if x0 > x1 or y0 > y1:
            return
        # Build a coordinate grid for the bounding box.
        xs = np.arange(x0, x1 + 1)
        ys = np.arange(y0, y1 + 1)
        gx, gy = np.meshgrid(xs, ys)
        if body.shape is Shape.CIRCLE:
            # Pixels whose centre is inside the circle get filled.
            dx = gx - body.cx
            dy = gy - body.cy
            mask = (dx * dx + dy * dy) <= (r * r)
        else:
            # Rectangle: pixels inside the half-width / half-height box.
            half_w = body.width / 2.0
            half_h = body.height / 2.0
            mask = (np.abs(gx - body.cx) <= half_w) & \
                   (np.abs(gy - body.cy) <= half_h)
        # Write the gray value into masked pixels.
        # NOTE: ``canvas[np.ix_(ys, xs)][mask] = ...`` does NOT work —
        # fancy indexing returns a *copy*, so the assignment is lost.
        # Instead, build a full-world mask and assign directly. This is
        # slightly more allocation but guaranteed to write back.
        full_mask = np.zeros_like(canvas, dtype=bool)
        # Place the local bbox mask back into the full-canvas mask.
        # ``mask`` has shape (len(ys), len(xs)); we scatter it via
        # broadcasting index arrays.
        y_idx = gy[mask]
        x_idx = gx[mask]
        full_mask[y_idx, x_idx] = True
        canvas[full_mask] = np.uint8(body.gray)

    # ------------------------------------------------------------------ #
    # Introspection / utility
    # ------------------------------------------------------------------ #
    def state_dict(self) -> dict:
        """Return a JSON-serialisable snapshot of the current state."""
        return {
            "num_objects": self.num_objects,
            "seed": self.seed,
            "bodies": [
                {
                    "cx": b.cx, "cy": b.cy,
                    "vx": b.vx, "vy": b.vy,
                    "mass": b.mass,
                    "shape": b.shape.value,
                    "radius": b.radius,
                    "width": b.width,
                    "height": b.height,
                    "gray": b.gray,
                }
                for b in self.bodies
            ],
        }

    def make_object_disappear(self, object_index: int) -> None:
        """Remove a non-agent body from the scene.

        Used by intervention experiments: the agent remains, but a
        chosen obstacle is deleted. The agent (index 0) cannot be
        removed.
        """
        if object_index == AGENT_INDEX:
            raise ValueError("cannot remove the agent (index 0)")
        if not 0 <= object_index < len(self.bodies):
            raise ValueError(f"invalid object_index {object_index!r}")
        del self.bodies[object_index]


# ------------------------------------------------------------------ #
# Demo
# ------------------------------------------------------------------ #
def _demo(num_steps: int = 8, seed: int = 42) -> None:
    """Run a short random-action demo and optionally display frames."""
    sandbox = PhysicsSandbox(num_objects=2, seed=seed)
    rng = np.random.default_rng(seed + 1)
    print(f"Initial state: {sandbox.state_dict()['bodies']}")
    frames = [sandbox.frame.copy()]
    for step in range(num_steps):
        action = int(rng.integers(0, 4))
        frame = sandbox.step(action)
        frames.append(frame.copy())
        print(f"step {step + 1:2d}  action={action}  "
              f"agent=(cx={sandbox.bodies[0].cx:.2f}, "
              f"cy={sandbox.bodies[0].cy:.2f}, "
              f"vx={sandbox.bodies[0].vx:.2f}, "
              f"vy={sandbox.bodies[0].vy:.2f})  "
              f"frame_mean={float(frame.mean()):.2f}")
    if _HAS_MPL:
        # Plot the first, middle, and last frame in a row.
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        for ax, fr, title in zip(
            axes,
            [frames[0], frames[len(frames) // 2], frames[-1]],
            ["initial", "mid", "final"],
        ):
            ax.imshow(fr, cmap="gray", vmin=0, vmax=255)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"PhysicsSandbox demo (seed={seed})")
        fig.tight_layout()
        plt.show()
    else:
        print("(matplotlib not available; skipping display)")


if __name__ == "__main__":
    _demo()
