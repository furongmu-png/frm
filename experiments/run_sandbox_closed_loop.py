# experiments/run_sandbox_closed_loop.py
"""Closed-loop integration: ZeroDataModel <-> PhysicsSandbox.

Bridges the model's continuous-action active-inference loop with the
sandbox's discrete 4-action interface (0=left, 1=right, 2=up, 3=no-op),
forming a perception -> cognition -> action -> environment cycle.

Run with:  python experiments/run_sandbox_closed_loop.py
"""

from __future__ import annotations

import gc
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from zero_data_model.model import ZeroDataModel

# Import the sandbox from the examples directory (sibling package).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
from physics_sandbox import PhysicsSandbox  # noqa: E402

# --- Configuration constants ---------------------------------------- #
DEFAULT_DIM = 64               # model observation dimension
DEFAULT_NUM_OBJECTS = 2        # bodies in the sandbox
DEFAULT_SEED = 42
DEFAULT_MAX_STEPS = 1000       # default loop length
STRESS_STEPS = 2000            # stress-test length (>= "thousands")
MEMORY_CHECK_INTERVAL = 200    # run gc + sample RSS every N steps
PROGRESS_INTERVAL = 500        # print progress every N steps
# Threshold below which the action vector is considered "no signal" → no-op.
# ``select_action`` samples around ``belief @ active_weights`` with 0.5 std
# noise, so a threshold of 0.1 filters out pure-noise directions while still
# responding to genuine belief-driven signals.
ACTION_THRESHOLD = 0.1


@dataclass
class CycleRecord:
    """Per-step metrics captured during the closed loop."""

    step: int
    action: int               # discretised action (0-3)
    action_vec_norm: float    # ||action_vec|| — signal magnitude
    free_energy: float        # model's free energy this cycle
    confidence: float         # model's integrated signal confidence
    obs_norm: float            # ||observation|| — sanity check (not NaN)
    frame_mean: float          # mean pixel value — sanity check (in range)
    rss_kb: float             # process RSS after this step (0 if unavailable)


@dataclass
class RunSummary:
    """Aggregated results from one ``run()`` call."""

    n_steps: int = 0
    n_crashes: int = 0
    crash_messages: list[str] = field(default_factory=list)
    n_nan_obs: int = 0
    n_nan_signal: int = 0
    actions: list[int] = field(default_factory=list)  # discretised action distribution
    elapsed_s: float = 0.0
    cycles_per_sec: float = 0.0
    rss_start_kb: float = 0.0
    rss_end_kb: float = 0.0
    rss_peak_kb: float = 0.0
    rss_growth_kb: float = 0.0      # end - start (negative = shrinkage)
    tracemalloc_peak_kb: float = 0.0
    records: list[CycleRecord] = field(default_factory=list)


class SandboxRunner:
    """Drives the ZeroDataModel <-> PhysicsSandbox closed loop.

    The loop is: ``frame -> encode_image -> think -> select_action ->
    discretise -> sandbox.step -> next frame``. Each cycle records metrics
    for later analysis (free energy, confidence, action distribution,
    RSS). The runner detects crashes (exceptions per step, not fatal),
    NaN observations/signals, and memory growth (leak proxy).
    """

    def __init__(
        self,
        model: ZeroDataModel,
        sandbox: PhysicsSandbox,
        max_steps: int = DEFAULT_MAX_STEPS,
    ):
        self.model = model
        self.sandbox = sandbox
        self.max_steps = max_steps
        self.summary = RunSummary()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self) -> RunSummary:
        """Execute the closed loop for ``self.max_steps`` steps.

        Returns a ``RunSummary`` with per-step records and aggregate
        metrics. The loop is resilient: a single step's exception is
        caught and recorded as a crash, and the loop continues (so a
        transient numerical issue does not abort a 2000-step stress
        test).
        """
        # Start memory tracking BEFORE reset so the baseline is clean.
        tracemalloc.start()
        rss_start = self._rss_kb()
        t0 = time.perf_counter()

        frame = self.sandbox.reset()
        self.summary.rss_start_kb = rss_start
        self.summary.rss_peak_kb = rss_start

        for step in range(self.max_steps):
            try:
                frame = self._cycle(frame, step)
            except Exception as exc:  # noqa: BLE001 — intentional broad catch
                self.summary.n_crashes += 1
                msg = f"step {step}: {type(exc).__name__}: {exc}"
                self.summary.crash_messages.append(msg)
                # Re-init the sandbox so the loop can continue (a crash
                # may have left the sandbox in a bad state). We keep the
                # same seed so the recovery is deterministic.
                self.sandbox.reset()
                frame = self.sandbox.render()

            # Periodic memory sampling (after gc so we measure retained,
            # not transient, allocations).
            if (step + 1) % MEMORY_CHECK_INTERVAL == 0:
                gc.collect()
                rss = self._rss_kb()
                if rss > self.summary.rss_peak_kb:
                    self.summary.rss_peak_kb = rss
                if self.summary.records:
                    self.summary.records[-1].rss_kb = rss

            if (step + 1) % PROGRESS_INTERVAL == 0:
                elapsed = time.perf_counter() - t0
                rate = (step + 1) / elapsed
                print(
                    f"  step {step + 1}/{self.max_steps}  "
                    f"crashes={self.summary.n_crashes}  "
                    f"rate={rate:.1f} cyc/s  "
                    f"rss={self._rss_kb():.0f}KB"
                )

        elapsed = time.perf_counter() - t0
        rss_end = self._rss_kb()
        tm_curr, tm_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.summary.n_steps = self.max_steps
        self.summary.elapsed_s = elapsed
        self.summary.cycles_per_sec = self.max_steps / elapsed if elapsed > 0 else 0.0
        self.summary.rss_end_kb = rss_end
        self.summary.rss_growth_kb = rss_end - rss_start
        self.summary.tracemalloc_peak_kb = tm_peak / 1024.0
        self.summary.actions = [r.action for r in self.summary.records]
        return self.summary

    # ------------------------------------------------------------------ #
    # One perception-cognition-action cycle
    # ------------------------------------------------------------------ #
    def _cycle(self, frame: np.ndarray, step: int) -> np.ndarray:
        """Run one full cycle: encode -> think -> select -> act."""
        # 1) PERCEPTION: encode the 128x128 grayscale frame into a
        #    ``dim``-length observation vector.
        obs = self.model.encode_image(frame)
        if not np.all(np.isfinite(obs)):
            self.summary.n_nan_obs += 1
        obs_norm = float(np.linalg.norm(obs)) if np.all(np.isfinite(obs)) else float("nan")

        # 2) COGNITION: think() runs the 6-module parallel process ->
        #    predict -> update cycle, updates the belief state, and
        #    returns the integrated Signal. This is where learning
        #    happens (emission/transition weights drift toward the
        #    observation).
        signal = self.model.think(obs)
        if not np.all(np.isfinite(signal.data)):
            self.summary.n_nan_signal += 1

        # 3) ACTION SELECTION: select_action evaluates 8 candidate
        #    actions around ``belief @ active_weights`` and returns the
        #    one minimising expected free energy. Returns a CONTINUOUS
        #    (active_dim,) vector — we must discretise it to 0-3 for
        #    the sandbox.
        belief = self.model.active_inference.generative_model.belief_state
        action_vec = self.model.active_inference.select_action(
            belief, current_observation=obs
        )
        action = self._discretize_action(action_vec)

        # 4) ENVIRONMENT: apply the discretised action and get the
        #    next frame.
        next_frame = self.sandbox.step(action)

        # 5) RECORD: capture metrics for offline analysis.
        free_energy = float(self.model.active_inference.compute_free_energy(obs))
        self.summary.records.append(CycleRecord(
            step=step,
            action=action,
            action_vec_norm=float(np.linalg.norm(action_vec)),
            free_energy=free_energy,
            confidence=float(signal.confidence),
            obs_norm=obs_norm,
            frame_mean=float(np.mean(next_frame)),
            rss_kb=0.0,  # filled by periodic memory check
        ))
        return next_frame

    # ------------------------------------------------------------------ #
    # Continuous -> discrete action mapping
    # ------------------------------------------------------------------ #
    def _discretize_action(self, action_vec: np.ndarray) -> int:
        """Map a continuous ``(active_dim,)`` action vector to 0-3.

        Strategy: interpret the first two components as a (vx, vy)
        velocity signal. The sandbox's action semantics are:
            0 = left  (vx < -threshold)
            1 = right (vx > +threshold)
            2 = up    (vy < -threshold; image y grows downward)
            3 = no-op (signal too weak)

        This is the simplest faithful mapping: it respects the sandbox's
        directional semantics, uses the model's actual output (not a
        random projection), and has a no-op fallback so the agent does
        not jitter when the model is uncertain.
        """
        vx = float(action_vec[0]) if action_vec.shape[0] > 0 else 0.0
        vy = float(action_vec[1]) if action_vec.shape[0] > 1 else 0.0
        # Check x-axis first (left/right), then y-axis (up). This
        # priority order means a strong lateral signal always wins over
        # a weaker vertical one — matching the sandbox's 2D layout where
        # horizontal movement is the primary degree of freedom.
        if vx > ACTION_THRESHOLD:
            return 1  # right
        if vx < -ACTION_THRESHOLD:
            return 0  # left
        if vy < -ACTION_THRESHOLD:
            return 2  # up (negative vy = toward row 0)
        return 3  # no-op

    # ------------------------------------------------------------------ #
    # Memory tracking
    # ------------------------------------------------------------------ #
    @staticmethod
    def _rss_kb() -> float:
        """Return current process RSS in KB, or 0.0 if unavailable."""
        # ``resource`` is Unix-only; on Windows we fall back to 0.0
        # (tracemalloc still tracks Python allocations).
        try:
            import resource
            # ru_maxrss is in KB on Linux, bytes on macOS — normalise.
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == "darwin":
                rss /= 1024.0
            return float(rss)
        except (ImportError, OSError):
            return 0.0


# ------------------------------------------------------------------ #
# Determinism verification
# ------------------------------------------------------------------ #
def verify_determinism(
    n_steps: int = 100,
    seed: int = DEFAULT_SEED,
    dim: int = DEFAULT_DIM,
) -> bool:
    """Run two identical loops and check frame-by-frame equality.

    With a fixed seed, both the sandbox (numpy Generator) and the model
    (per-module child Generators spawned from the seed) are
    deterministic, so the entire frame sequence must match byte-for-byte.
    """
    print(f"\n--- Determinism check ({n_steps} steps, seed={seed}) ---")

    def run_once() -> list[np.ndarray]:
        model = ZeroDataModel(dim=dim, seed=seed)
        sandbox = PhysicsSandbox(num_objects=DEFAULT_NUM_OBJECTS, seed=seed)
        runner = SandboxRunner(model, sandbox, max_steps=n_steps)
        # We need the frames, not just the summary — capture them inline.
        frames = [sandbox.reset().copy()]
        for _ in range(n_steps):
            obs = model.encode_image(frames[-1])
            model.think(obs)
            belief = model.active_inference.generative_model.belief_state
            action_vec = model.active_inference.select_action(
                belief, current_observation=obs
            )
            action = runner._discretize_action(action_vec)
            frames.append(sandbox.step(action).copy())
        return frames

    frames_a = run_once()
    frames_b = run_once()
    if len(frames_a) != len(frames_b):
        print(f"  FAIL: length mismatch {len(frames_a)} vs {len(frames_b)}")
        return False
    mismatches = sum(
        1 for fa, fb in zip(frames_a, frames_b, strict=False)
        if not np.array_equal(fa, fb)
    )
    ok = mismatches == 0
    print(f"  {'PASS' if ok else 'FAIL'}: {len(frames_a)} frames, "
          f"{mismatches} mismatches")
    return ok


# ------------------------------------------------------------------ #
# Main entry point
# ------------------------------------------------------------------ #
def main() -> None:
    """Run a smoke test, a stress test, and a determinism check."""
    print("=" * 64)
    print("ZeroDataModel <-> PhysicsSandbox Closed-Loop Integration")
    print("=" * 64)

    # --- Smoke test: 100 steps, verify no crashes/NaNs ---------------- #
    print(f"\n--- Smoke test (100 steps, dim={DEFAULT_DIM}) ---")
    model = ZeroDataModel(dim=DEFAULT_DIM, seed=DEFAULT_SEED)
    sandbox = PhysicsSandbox(num_objects=DEFAULT_NUM_OBJECTS, seed=DEFAULT_SEED)
    runner = SandboxRunner(model, sandbox, max_steps=100)
    summary = runner.run()
    print(f"  steps={summary.n_steps}  crashes={summary.n_crashes}  "
          f"nan_obs={summary.n_nan_obs}  nan_signal={summary.n_nan_signal}")
    print(f"  elapsed={summary.elapsed_s:.1f}s  rate={summary.cycles_per_sec:.1f} cyc/s")
    action_counts = [summary.actions.count(a) for a in range(4)]
    print(f"  action distribution [L,R,Up,NOOP]: {action_counts}")
    print(f"  rss: start={summary.rss_start_kb:.0f}KB end={summary.rss_end_kb:.0f}KB "
          f"peak={summary.rss_peak_kb:.0f}KB growth={summary.rss_growth_kb:+.0f}KB")
    print(f"  tracemalloc peak: {summary.tracemalloc_peak_kb:.0f}KB")
    if summary.crash_messages:
        print(f"  crashes (first 3): {summary.crash_messages[:3]}")

    # --- Stress test: 2000 steps, verify stability -------------------- #
    print(f"\n--- Stress test ({STRESS_STEPS} steps) ---")
    model2 = ZeroDataModel(dim=DEFAULT_DIM, seed=DEFAULT_SEED)
    sandbox2 = PhysicsSandbox(num_objects=DEFAULT_NUM_OBJECTS, seed=DEFAULT_SEED)
    runner2 = SandboxRunner(model2, sandbox2, max_steps=STRESS_STEPS)
    summary2 = runner2.run()
    print(f"  steps={summary2.n_steps}  crashes={summary2.n_crashes}  "
          f"nan_obs={summary2.n_nan_obs}  nan_signal={summary2.n_nan_signal}")
    print(f"  elapsed={summary2.elapsed_s:.1f}s  rate={summary2.cycles_per_sec:.1f} cyc/s")
    action_counts2 = [summary2.actions.count(a) for a in range(4)]
    print(f"  action distribution [L,R,Up,NOOP]: {action_counts2}")
    print(f"  rss: start={summary2.rss_start_kb:.0f}KB end={summary2.rss_end_kb:.0f}KB "
          f"peak={summary2.rss_peak_kb:.0f}KB growth={summary2.rss_growth_kb:+.0f}KB")
    print(f"  tracemalloc peak: {summary2.tracemalloc_peak_kb:.0f}KB")
    if summary2.crash_messages:
        print(f"  crashes (first 3): {summary2.crash_messages[:3]}")

    # Leak heuristic: if RSS grew by more than 5% over 2000 steps (after
    # gc), flag it. A genuine leak grows linearly with steps; normal
    # cache warm-up plateaus within the first few hundred steps.
    if summary2.rss_start_kb > 0:
        growth_pct = (summary2.rss_growth_kb / summary2.rss_start_kb) * 100
        verdict = "LEAK SUSPECTED" if growth_pct > 5.0 else "no leak"
        print(f"  RSS growth: {growth_pct:+.1f}% → {verdict}")

    # --- Determinism check -------------------------------------------- #
    verify_determinism(n_steps=50, seed=DEFAULT_SEED, dim=DEFAULT_DIM)

    print("\n" + "=" * 64)
    print("Done.")
    print("=" * 64)


if __name__ == "__main__":
    main()
