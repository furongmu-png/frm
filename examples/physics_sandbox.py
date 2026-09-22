# examples/physics_sandbox.py
"""Minimal 2D physics sandbox for ZeroDataModel sensory input.

Single-file, numpy-only implementation. Renders 128x128 grayscale frames
suitable as the sole sensory stream for ``ZeroDataModel``. All randomness
flows through ``numpy.random.Generator`` so a fixed ``seed`` reproduces
the exact initial state and frame sequence across platforms.

Run with:  python examples/physics_sandbox.py
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

# Optional: only used by ``_demo`` to actually display frames.
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_MPL = False


# Module-level constants keep the magic numbers in one place.
WORLD_SIZE = 128          # canvas side in pixels
AGENT_INDEX = 0           # bodies[0] is the force-controlled agent
DELTA_V = 1.0             # pixels / frame^2 applied by step(action)
AGENT_GRAY = 255          # brightest pixel so the agent is visible
BODY_GRAY_MIN = 200
BODY_GRAY_MAX = 254


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
    in the body count and branch-free.
    """

    cx: float             # center x in pixels
    cy: float             # center y in pixels
    vx: float             # velocity x in pixels / frame
    vy: float             # velocity y in pixels / frame
    mass: float
    shape: Shape
    radius: float         # bounding-circle radius (circle: exact, rect: half-diagonal)
    width: float = 0.0    # rect: full width (unused for circles)
    height: float = 0.0   # rect: full height (unused for circles)
    gray: int = AGENT_GRAY


class PhysicsSandbox:
    """Deterministic 2D physics sandbox.

    Renders a 128x128 uint8 grayscale frame. Bodies move frictionlessly,
    bounce elastically off walls and each other. ``step(action)`` applies
    a fixed impulse to the agent (``bodies[0]``) along one of 4 directions
    and returns the next rendered frame.

    Determinism contract: with a fixed ``seed`` the full sequence of
    frames produced by ``reset()`` followed by any sequence of
    ``step(action)`` calls is identical across runs and platforms. All
    randomness is drawn from a single ``numpy.random.Generator`` and all
    physics math uses numpy so there are no Python-float-ordering
    ambiguities.
    """

    def __init__(self, num_objects: int = 2, seed: int | None = 42):
        if not 1 <= num_objects <= 3:
            raise ValueError(f"num_objects must be in [1, 3], got {num_objects}")
        self.num_objects = num_objects
        self.seed = seed
        self.rng: np.random.Generator | None = None
        self.bodies: list[Body] = []
        # ``reset`` initialises ``rng`` and ``bodies``.
        self.reset(seed=seed)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def reset(self, seed: int | None = None) -> np.ndarray:
        """Re-randomise the scene and return the initial frame.

        ``seed=None`` reuses the seed passed to ``__init__`` so repeated
        ``reset()`` calls without arguments reproduce the same scene.
        """
        if seed is None:
            seed = self.seed
        self.rng = np.random.default_rng(seed)
        self.bodies = self._spawn_bodies()
        return self.render()

    def step(self, action: int) -> np.ndarray:
        """Apply ``action`` to the agent, advance physics, return frame.

        action: 0=left, 1=right, 2=up, 3=no-op. The impulse magnitude is
        ``DELTA_V`` pixels/frame^2, applied to the agent's velocity
        BEFORE the physics step so it takes effect this frame.
        """
        if action not in (0, 1, 2, 3):
            raise ValueError(f"action must be 0-3, got {action!r}")
        agent = self.bodies[AGENT_INDEX]
        # Image y grows downward: "up" pushes toward row 0 => negative vy.
        if action == 0:      # left
            agent.vx -= DELTA_V
        elif action == 1:    # right
            agent.vx += DELTA_V
        elif action == 2:    # up
            agent.vy -= DELTA_V
        # action == 3: no-op
        self._physics_step()
        return self.render()

    # ------------------------------------------------------------------ #
    # Phase G: intervention API (cognitive-emergence evaluation)
    # ------------------------------------------------------------------ #
    def make_object_disappear(self, object_index: int) -> None:
        """Remove a non-agent body from the scene (object-permanence test).

        Removes ``self.bodies[object_index]`` in place. The agent
        (index 0) cannot be removed (raises ``ValueError``) — only
        secondary objects can disappear, since the intervention tests
        whether the model notices a NON-SELF object vanishing.

        After removal, ``self.bodies`` is shorter by one and subsequent
        ``render()`` calls no longer draw the object. The model's
        prediction error is expected to spike (surprise) if it has
        built an internal representation of the disappeared object.

        Used by ``experiments/run_sandbox_replay.py`` for the
        disappearance intervention experiment.
        """
        if object_index == AGENT_INDEX:
            raise ValueError(
                f"cannot remove the agent (index {AGENT_INDEX}); "
                f"only secondary objects can disappear"
            )
        if not 0 <= object_index < len(self.bodies):
            raise ValueError(
                f"object_index {object_index} out of range "
                f"[0, {len(self.bodies)})"
            )
        # Pop in place. Subsequent indices shift down by 1, but since
        # the caller is expected to use this once per run, that's fine.
        self.bodies.pop(object_index)

    def render(self) -> np.ndarray:
        """Return the current 128x128 uint8 grayscale frame.

        Background is 0 (black); bodies are drawn at their ``gray`` value
        (agent=255, others in [200, 254]).
        """
        frame = np.zeros((WORLD_SIZE, WORLD_SIZE), dtype=np.uint8)
        for body in self.bodies:
            if body.shape == Shape.CIRCLE:
                self._draw_circle(frame, body)
            else:
                self._draw_rect(frame, body)
        return frame

    # ------------------------------------------------------------------ #
    # Initialisation
    # ------------------------------------------------------------------ #
    def _spawn_bodies(self) -> list[Body]:
        """Sample ``num_objects`` non-overlapping, fully-contained bodies."""
        bodies: list[Body] = []
        for i in range(self.num_objects):
            body = self._sample_non_overlapping(bodies, is_agent=(i == AGENT_INDEX))
            bodies.append(body)
        return bodies

    def _sample_non_overlapping(
        self, existing: list[Body], is_agent: bool
    ) -> Body:
        """Reject-sample a body that does not overlap any existing one."""
        for _ in range(1000):
            body = self._sample_body(is_agent=is_agent)
            ok = True
            for b in existing:
                d = float(np.hypot(body.cx - b.cx, body.cy - b.cy))
                # +2 px gap so the very first frame has visible separation.
                if d < body.radius + b.radius + 2.0:
                    ok = False
                    break
            if ok:
                return body
        raise RuntimeError("failed to place a non-overlapping body")

    def _sample_body(self, is_agent: bool = False) -> Body:
        """Sample one body. The agent is fixed at mass=1 and gray=255."""
        shape = Shape.CIRCLE if self.rng.random() < 0.5 else Shape.RECT
        if shape == Shape.CIRCLE:
            r = float(self.rng.integers(5, 10))
            w = h = 2.0 * r
        else:
            w = float(self.rng.integers(10, 20))
            h = float(self.rng.integers(10, 20))
            # Half-diagonal so the rect is fully enclosed by its bounding
            # circle — this is what the collision code uses.
            r = 0.5 * float(np.hypot(w, h))
        margin = r + 2.0
        cx = float(self.rng.uniform(margin, WORLD_SIZE - margin))
        cy = float(self.rng.uniform(margin, WORLD_SIZE - margin))
        vx = float(self.rng.uniform(-2.0, 2.0))
        vy = float(self.rng.uniform(-2.0, 2.0))
        # Agent mass is fixed at 1 so the action impulse (DELTA_V) has a
        # predictable effect on the agent's velocity regardless of the
        # other bodies' masses. Non-agent bodies draw from {1, 1.5, 2}.
        mass = 1.0 if is_agent else float(self.rng.choice([1.0, 1.5, 2.0]))
        gray = AGENT_GRAY if is_agent else int(
            self.rng.integers(BODY_GRAY_MIN, BODY_GRAY_MAX + 1)
        )
        return Body(
            cx=cx, cy=cy, vx=vx, vy=vy,
            mass=mass, shape=shape, radius=r,
            width=w, height=h, gray=gray,
        )

    # ------------------------------------------------------------------ #
    # Physics
    # ------------------------------------------------------------------ #
    def _physics_step(self) -> None:
        """Advance one frame: move, wall-bounce, resolve pair collisions."""
        # 1) advance positions (frictionless uniform motion).
        for b in self.bodies:
            b.cx += b.vx
            b.cy += b.vy
        # 2) wall reflection — clamp position then flip the offending
        #    velocity component so the body is back inside and moving away.
        for b in self.bodies:
            r = b.radius
            if b.cx - r < 0.0:
                b.cx = r
                b.vx = -b.vx
            elif b.cx + r > WORLD_SIZE:
                b.cx = WORLD_SIZE - r
                b.vx = -b.vx
            if b.cy - r < 0.0:
                b.cy = r
                b.vy = -b.vy
            elif b.cy + r > WORLD_SIZE:
                b.cy = WORLD_SIZE - r
                b.vy = -b.vy
        # 3) pairwise elastic collisions (bounding-circle approximation).
        n = len(self.bodies)
        for i in range(n):
            for j in range(i + 1, n):
                self._resolve_collision(self.bodies[i], self.bodies[j])

    def _resolve_collision(self, a: Body, b: Body) -> None:
        """Resolve one elastic collision between ``a`` and ``b``.

        Uses the standard 1D-elastic-collision impulse along the contact
        normal (a -> b). This conserves both total momentum and total
        kinetic energy along the normal direction; tangential components
        are untouched. If the bodies are already separating we skip
        (avoids re-colliding sticky pairs).
        """
        dx = b.cx - a.cx
        dy = b.cy - a.cy
        dist = float(np.hypot(dx, dy))
        rsum = a.radius + b.radius
        if dist == 0.0 or dist >= rsum:
            return
        # unit normal (a -> b).
        nx = dx / dist
        ny = dy / dist
        # relative velocity along normal (b - a).
        rvx = b.vx - a.vx
        rvy = b.vy - a.vy
        vn = rvx * nx + rvy * ny
        if vn > 0.0:
            return  # already separating, no impulse needed
        # 1D elastic-collision impulse along normal:
        #   j = 2 * <v_rel, n> / (m1 + m2)
        #   v_a += j * m_b * n
        #   v_b -= j * m_a * n
        # Closed form: conserves momentum (sum of m_i * v_i) and kinetic
        # energy (sum of 0.5 * m_i * |v_i|^2) along the normal direction.
        m1, m2 = a.mass, b.mass
        j = (2.0 * vn) / (m1 + m2)
        a.vx += j * m2 * nx
        a.vy += j * m2 * ny
        b.vx -= j * m1 * nx
        b.vy -= j * m1 * ny
        # Positional correction: push the bodies apart by half the
        # overlap each so the next frame does not start already
        # overlapping (which would re-trigger the impulse with vn ~ 0
        # and produce a jitter lock). The 1e-6 bias guarantees forward
        # progress out of the contact.
        overlap = rsum - dist
        push = 0.5 * overlap + 1e-6
        a.cx -= push * nx
        a.cy -= push * ny
        b.cx += push * nx
        b.cy += push * ny

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _draw_circle(self, frame: np.ndarray, body: Body) -> None:
        """Vectorised filled-circle rasterisation.

        Uses ``np.ogrid`` to build a coordinate grid over the circle's
        bounding box and a single mask to set all interior pixels. The
        center is taken as the float ``cx/cy`` (not rounded) so the
        rendered shape moves sub-pixel-smoothly with the physics.
        """
        r = body.radius
        x0 = max(0, int(np.floor(body.cx - r)))
        x1 = min(WORLD_SIZE, int(np.ceil(body.cx + r)) + 1)
        y0 = max(0, int(np.floor(body.cy - r)))
        y1 = min(WORLD_SIZE, int(np.ceil(body.cy + r)) + 1)
        if x1 <= x0 or y1 <= y0:
            return
        ys, xs = np.ogrid[y0:y1, x0:x1]
        mask = (xs - body.cx) ** 2 + (ys - body.cy) ** 2 <= r * r
        frame[y0:y1, x0:x1][mask] = body.gray

    def _draw_rect(self, frame: np.ndarray, body: Body) -> None:
        """Axis-aligned rectangle rasterisation via slice assignment."""
        half_w = 0.5 * body.width
        half_h = 0.5 * body.height
        x0 = max(0, int(round(body.cx - half_w)))
        x1 = min(WORLD_SIZE, int(round(body.cx + half_w)))
        y0 = max(0, int(round(body.cy - half_h)))
        y1 = min(WORLD_SIZE, int(round(body.cy + half_h)))
        if x1 <= x0 or y1 <= y0:
            return
        frame[y0:y1, x0:x1] = body.gray


def _demo() -> None:
    """Random-walk the agent for 50 steps; print state; plot if mpl is present."""
    sb = PhysicsSandbox(num_objects=2, seed=42)
    rng = np.random.default_rng(0)
    frames: list[np.ndarray] = [sb.render()]
    for _ in range(50):
        a = int(rng.integers(0, 4))
        frames.append(sb.step(a))
    agent = sb.bodies[AGENT_INDEX]
    print(
        f"agent: cx={agent.cx:.2f} cy={agent.cy:.2f} "
        f"vx={agent.vx:.2f} vy={agent.vy:.2f}"
    )
    # Determinism check: a fresh sandbox with the same seed must produce
    # the identical initial frame byte-for-byte.
    sb2 = PhysicsSandbox(num_objects=2, seed=42)
    print(f"reset determinism: {np.array_equal(sb2.render(), frames[0])}")
    if _HAS_MPL:
        fig, axes = plt.subplots(1, 4, figsize=(10, 3))
        for ax, f in zip(axes, [frames[0], frames[10], frames[25], frames[50]], strict=False):
            ax.imshow(f, cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    _demo()
