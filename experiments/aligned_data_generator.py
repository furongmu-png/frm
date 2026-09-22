# experiments/aligned_data_generator.py
"""Generate time-aligned (physics_frame, text_description) pairs (Phase J, Task 1).

The generator drives a ``PhysicsSandbox`` with a random action schedule and,
at every step, produces BOTH a rendered frame AND a short natural-language
description of what just happened. The description is built by a simple
event detector (collision / wall-hit / direction-of-motion / stationary)
plus a small bank of sentence templates — NO external NLP, NO pretrained
embeddings, fully deterministic given the seed.

The resulting dataset is the training signal for ``multimodal_bridge.py``:
each sample is a tuple (frame, text) describing the SAME physical event,
so a cross-modal predictor that has learned the alignment should be able
to map one to the other.

Output (.npz):
    frames:    (N, 128, 128) uint8   — raw sandbox frames
    texts:     (N,) object (str)     — text descriptions
    actions:   (N,) int32            — action taken at each step
    events:   (N,) object (str)     — event label per step
    meta:      dict (pickled)        — config + generation stats

Run with:
    python experiments/aligned_data_generator.py --n_samples 2000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from physics_sandbox import PhysicsSandbox  # noqa: E402


# ------------------------------------------------------------------ #
# Defaults
# ------------------------------------------------------------------ #
DEFAULT_N_SAMPLES = 2000
DEFAULT_SEED = 42
DEFAULT_NUM_OBJECTS = 2
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "aligned_data"
WORLD_SIZE = 128
# Action space (mirrors run_closed_loop.SANDBOX_ACTIONS).
SANDBOX_ACTIONS = (0, 1, 2, 3)  # left, right, up, no-op
ACTION_NAMES = {0: "left", 1: "right", 2: "up", 3: "no-op"}

# Collision / wall thresholds (in pixel units). The sandbox world is
# 128×128; objects are circles of radius ``radius`` (default 6 px).
COLLISION_DIST_FACTOR = 2.0  # collision if centre-distance < factor * radius
WALL_MARGIN = 8              # px from edge to count as "near wall"


# ------------------------------------------------------------------ #
# Event types
# ------------------------------------------------------------------ #
@dataclass
class PhysicsEvent:
    """A detected physical event in a single sandbox step."""

    label: str          # short canonical label, e.g. "collision"
    description: str    # full sentence, e.g. "The agent collided with object 1."
    action: int = 3     # action that produced this event
    details: dict = field(default_factory=dict)


# ------------------------------------------------------------------ #
# Event detector
# ------------------------------------------------------------------ #
class EventDetector:
    """Detect physical events from sandbox state across a step.

    The sandbox does not expose velocities directly, so we reconstruct
    them by finite-differencing positions between consecutive frames.
    """

    def __init__(self, num_objects: int, seed: int = DEFAULT_SEED):
        self.num_objects = num_objects
        self._prev_positions: np.ndarray | None = None
        self._rng = np.random.default_rng(seed)
        # Track cumulative collisions to vary phrasing.
        self._collision_count = 0
        self._wall_count = 0

    def reset(self) -> None:
        self._prev_positions = None
        self._collision_count = 0
        self._wall_count = 0

    def detect(
        self,
        sandbox: PhysicsSandbox,
        action: int,
    ) -> PhysicsEvent:
        """Inspect ``sandbox`` after a step and emit one event."""
        # Positions: shape (num_objects, 2). Each body is a ``Body``
        # dataclass with ``cx, cy, vx, vy, radius``.
        positions = np.array(
            [[b.cx, b.cy] for b in sandbox.bodies],
            dtype=float,
        )
        radii = [float(b.radius) for b in sandbox.bodies]

        # Velocity by finite difference.
        if self._prev_positions is not None:
            velocities = positions - self._prev_positions
        else:
            velocities = np.zeros_like(positions)
        self._prev_positions = positions.copy()

        # --- Priority 1: body-body collision ------------------------ #
        if self.num_objects >= 2:
            for i in range(self.num_objects):
                for j in range(i + 1, self.num_objects):
                    dist = float(np.linalg.norm(positions[i] - positions[j]))
                    threshold = COLLISION_DIST_FACTOR * max(
                        radii[i], radii[j]
                    )
                    if dist < threshold:
                        self._collision_count += 1
                        label = "collision"
                        if i == 0 or j == 0:
                            # Agent involved.
                            other = j if i == 0 else i
                            desc = (
                                f"The agent collided with object {other}."
                            )
                        else:
                            desc = (
                                f"Object {i} collided with object {j}."
                            )
                        return PhysicsEvent(
                            label=label, description=desc, action=action,
                            details={"distance": dist, "pair": (i, j)},
                        )

        # --- Priority 2: wall hit (agent near boundary) ------------- #
        agent_pos = positions[0]
        near_wall = (
            agent_pos[0] < WALL_MARGIN
            or agent_pos[0] > WORLD_SIZE - WALL_MARGIN
            or agent_pos[1] < WALL_MARGIN
            or agent_pos[1] > WORLD_SIZE - WALL_MARGIN
        )
        if near_wall and action != 3:
            self._wall_count += 1
            label = "wall_hit"
            # Identify which wall.
            if agent_pos[0] < WALL_MARGIN:
                wall = "left"
            elif agent_pos[0] > WORLD_SIZE - WALL_MARGIN:
                wall = "right"
            elif agent_pos[1] < WALL_MARGIN:
                wall = "bottom"
            else:
                wall = "top"
            desc = f"The agent hit the {wall} wall."
            return PhysicsEvent(
                label=label, description=desc, action=action,
                details={"wall": wall, "position": agent_pos.tolist()},
            )

        # --- Priority 3: movement direction ------------------------- #
        agent_vel = velocities[0]
        speed = float(np.linalg.norm(agent_vel))
        if speed > 0.5 and action != 3:
            # Dominant axis.
            if abs(agent_vel[0]) > abs(agent_vel[1]):
                direction = "right" if agent_vel[0] > 0 else "left"
            else:
                direction = "up" if agent_vel[1] > 0 else "down"
            label = "movement"
            desc = f"The agent moved {direction}."
            return PhysicsEvent(
                label=label, description=desc, action=action,
                details={"direction": direction, "speed": speed},
            )

        # --- Priority 4: stationary --------------------------------- #
        if speed < 0.1:
            label = "stationary"
            desc = "The agent stayed still."
            return PhysicsEvent(
                label=label, description=desc, action=action,
                details={"speed": speed},
            )

        # --- Fallback: generic action ------------------------------- #
        label = "action"
        desc = f"The agent performed a {ACTION_NAMES.get(action, 'unknown')} action."
        return PhysicsEvent(
            label=label, description=desc, action=action,
            details={"speed": speed},
        )


# ------------------------------------------------------------------ #
# Dataset generator
# ------------------------------------------------------------------ #
@dataclass
class AlignedDataset:
    """Container for an aligned (frame, text) dataset."""

    frames: np.ndarray           # (N, 128, 128) uint8
    texts: list[str]             # length N
    actions: np.ndarray          # (N,) int32
    events: list[str]            # length N, event labels
    meta: dict                   # config + stats

    def __len__(self) -> int:
        return len(self.texts)

    def save(self, path: Path) -> str:
        """Save as .npz (frames + actions) + .pkl (texts + events + meta).

        Returns the path to the .npz file. The .pkl sidecar is written
        next to it with the same stem.
        """
        import pickle
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            frames=self.frames,
            actions=self.actions,
        )
        sidecar = path.with_suffix(".pkl")
        with sidecar.open("wb") as f:
            pickle.dump(
                {"texts": self.texts, "events": self.events, "meta": self.meta},
                f, protocol=pickle.HIGHEST_PROTOCOL,
            )
        return str(path)

    @classmethod
    def load(cls, path: str | Path) -> "AlignedDataset":
        import pickle
        path = Path(path)
        data = np.load(path, allow_pickle=False)
        sidecar = path.with_suffix(".pkl")
        with sidecar.open("rb") as f:
            payload = pickle.load(f)
        return cls(
            frames=data["frames"],
            texts=payload["texts"],
            actions=data["actions"],
            events=payload["events"],
            meta=payload["meta"],
        )


def generate_dataset(
    n_samples: int = DEFAULT_N_SAMPLES,
    num_objects: int = DEFAULT_NUM_OBJECTS,
    seed: int = DEFAULT_SEED,
    output_path: Path | None = None,
) -> AlignedDataset:
    """Run the sandbox with random actions and collect aligned pairs."""
    sandbox = PhysicsSandbox(num_objects=num_objects, seed=seed)
    detector = EventDetector(num_objects=num_objects, seed=seed + 1)
    rng = np.random.default_rng(seed)

    frames = np.zeros((n_samples, WORLD_SIZE, WORLD_SIZE), dtype=np.uint8)
    actions_arr = np.zeros(n_samples, dtype=np.int32)
    texts: list[str] = []
    events: list[str] = []

    t0 = time.perf_counter()
    print("=" * 64)
    print("Aligned Data Generation")
    print("=" * 64)
    print(f"  n_samples   = {n_samples}")
    print(f"  num_objects = {num_objects}")
    print(f"  seed        = {seed}")
    print()

    # Event label counts for stats.
    label_counts: dict[str, int] = {}

    for step in range(n_samples):
        # Random action (uniform over SANDBOX_ACTIONS).
        action = int(rng.choice(SANDBOX_ACTIONS))
        frame = sandbox.step(action)
        event = detector.detect(sandbox, action)

        frames[step] = frame
        actions_arr[step] = action
        texts.append(event.description)
        events.append(event.label)
        label_counts[event.label] = label_counts.get(event.label, 0) + 1

        if (step + 1) % 200 == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"  step {step + 1:5d}/{n_samples}  "
                f"events={label_counts}  "
                f"rate={((step + 1) / max(elapsed, 1e-9)):.1f} sample/s"
            )

    elapsed = time.perf_counter() - t0
    print()
    print(f"Generated {n_samples} aligned pairs in {elapsed:.1f}s")
    print(f"Event distribution: {label_counts}")

    meta = {
        "n_samples": n_samples,
        "num_objects": num_objects,
        "seed": seed,
        "world_size": WORLD_SIZE,
        "event_counts": label_counts,
        "elapsed_s": elapsed,
        "action_names": ACTION_NAMES,
    }
    ds = AlignedDataset(
        frames=frames, texts=texts, actions=actions_arr,
        events=events, meta=meta,
    )
    if output_path is not None:
        path = ds.save(Path(output_path))
        print(f"Saved dataset to {path}")
    return ds


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate aligned (frame, text) pairs from the physics sandbox.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n_samples", type=int, default=DEFAULT_N_SAMPLES)
    p.add_argument("--num_objects", type=int, default=DEFAULT_NUM_OBJECTS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--output_path", type=str,
        default=str(DEFAULT_OUTPUT_DIR / "aligned_dataset.npz"),
        help="Path for the output .npz file (a .pkl sidecar is also written).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    generate_dataset(
        n_samples=args.n_samples,
        num_objects=args.num_objects,
        seed=args.seed,
        output_path=Path(args.output_path),
    )


if __name__ == "__main__":
    main()
