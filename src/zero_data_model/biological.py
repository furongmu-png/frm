# src/zero_data_model/biological.py
"""Biological Computation Substrate — DNA storage, morphogenetic fields, cellular automata."""

from __future__ import annotations

import threading
from collections import OrderedDict

import numpy as np

from .base import CognitiveModule, KnowledgeStore, Prediction, Signal

# JIT kernels -- graceful fallback to pure numpy if numba missing.
try:
    from .hardware.kernels import _cellular_automata_step, _morphogenetic_laplacian
    _HAS_JIT = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_JIT = False


class DNAStorage(KnowledgeStore):
    """DNA-inspired knowledge storage using quaternary encoding (A=0, T=1, C=2, G=3).

    C-batch fix: the original implementation normalised to [0, 1] and quantised
    to 4 levels, which is *lossy* (only 4 distinct values survive). The new
    implementation is byte-exact lossless: each float64 is viewed as 8 bytes,
    each byte is split into 4 quaternary digits (4^4 = 256 = 2^8), so the
    quaternary stream round-trips to the *exact* original float bytes.
    """

    def __init__(self, capacity: int = 1024, rng: np.random.Generator | None = None):
        self.capacity = capacity
        # Round-3 audit CRIT-1: per-module Generator
        self._rng = rng if rng is not None else np.random.default_rng()
        # ``OrderedDict`` so we can evict the oldest entry when ``capacity``
        # is exceeded (Fix 20) -- the original ``dict`` declared ``capacity``
        # but never enforced it.
        self._store: OrderedDict[str, dict] = OrderedDict()

    def _encode(self, data: np.ndarray) -> np.ndarray:
        """Lossless byte->quaternary encoding.

        View the float64 buffer as raw bytes, then split every byte into 4
        base-4 digits. Returns a ``uint8`` array of length ``4 * nbytes`` with
        values in {0, 1, 2, 3}.
        """
        arr = np.ascontiguousarray(data, dtype=np.float64)
        # View as raw bytes (uint8) without copying.
        raw = arr.view(np.uint8).ravel()
        # Split each byte into 4 base-4 digits: digit_k = (byte >> (2*k)) & 3
        shifts = np.array([0, 2, 4, 6], dtype=np.uint8)
        # Shape (n_bytes, 4): digit[i, k] = (raw[i] >> shifts[k]) & 3
        digits = (raw[:, None] >> shifts[None, :]) & np.uint8(3)
        return digits.ravel()

    def _decode(self, encoded: np.ndarray, nbytes: int, shape: tuple[int, ...]) -> np.ndarray:
        """Reverse ``_encode``: pack 4 quaternary digits back into one byte."""
        e = np.asarray(encoded, dtype=np.uint8).reshape(nbytes, 4)
        # Recombine: byte = sum(digit_k << (2*k))
        shifts = np.array([0, 2, 4, 6], dtype=np.uint8)
        raw = (e.astype(np.uint16) << shifts[None, :]).sum(axis=1).astype(np.uint8)
        # View the packed bytes as float64 and reshape.
        return raw.view(np.float64).reshape(shape)

    def store(self, key: str, value: np.ndarray) -> None:
        arr = np.ascontiguousarray(value, dtype=np.float64)
        self._store[key] = {
            "encoded": self._encode(arr),
            "nbytes": arr.nbytes,
            "shape": arr.shape,
        }
        self._store.move_to_end(key)
        # Enforce capacity by evicting the oldest entries (Fix 20).
        while len(self._store) > self.capacity:
            self._store.popitem(last=False)

    def retrieve(self, key: str) -> np.ndarray | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        return self._decode(entry["encoded"], entry["nbytes"], entry["shape"])

    def generate(self, query: Signal) -> Signal:
        """Self-generate knowledge by recombining stored sequences."""
        if len(self._store) < 2:
            # Round-3 audit CRIT-1: per-module Generator
            return Signal(data=self._rng.standard_normal(64), metadata={"source": "dna_random"})
        keys = list(self._store.keys())
        k1, k2 = self._rng.choice(keys, size=2, replace=False)
        v1 = self.retrieve(k1)
        v2 = self.retrieve(k2)
        if v1 is None or v2 is None:
            return Signal(data=self._rng.standard_normal(64), metadata={"source": "dna_random"})
        min_len = min(len(v1.flatten()), len(v2.flatten()))
        # ``np.random.randint(1, min_len)`` crashes when ``min_len == 1``;
        # clamp the lower bound to 2 so crossover is always valid (Fix 19).
        crossover = int(self._rng.integers(1, max(2, min_len)))
        flat1 = v1.flatten()[:min_len]
        flat2 = v2.flatten()[:min_len]
        child = np.concatenate([flat1[:crossover], flat2[crossover:]])
        return Signal(data=child, metadata={"source": "dna_crossover", "parents": [k1, k2]})


class MorphogeneticField:
    """Self-organizing structure development inspired by morphogenesis.

    C-batch fix: the original ``MorphogeneticField`` evolved ``grid`` and each
    morphogen under a *pure linear Laplacian diffusion* (no reaction term),
    which can only smooth features out -- it cannot create the self-organising
    patterns morphogenesis is famous for. The new implementation runs a true
    Gray-Scott reaction-diffusion system on the first two morphogens
    ``(u, v)``:

        du/dt = Du * laplacian(u) - u*v^2 + f*(1-u)
        dv/dt = Dv * laplacian(v) + u*v^2 - (f+k)*v

    Gray-Scott is the canonical activator-inhibitor model that produces
    spots, stripes, mazes and self-replicating patterns from a small seed --
    genuine self-organisation rather than decay-to-uniform.

    The legacy ``grid`` attribute is kept as a passive pattern carrier that
    still evolves under plain diffusion (so ``BiologicalSubstrate.predict``'s
    save/restore/perturb path keeps working). ``morphogens`` is a list whose
    first two entries are the Gray-Scott ``(u, v)`` fields (extra morphogens
    fall back to linear diffusion for backward compatibility with callers
    that constructed with ``n_signals > 2``).
    """

    # Gray-Scott default parameters (Pearson 1993 spots regime).
    _DEFAULT_DU = 0.16
    _DEFAULT_DV = 0.08
    _DEFAULT_FEED = 0.035
    _DEFAULT_KILL = 0.065

    def __init__(
        self, grid_size: int = 16, n_signals: int = 3, rng: np.random.Generator | None = None
    ):
        self.grid_size = grid_size
        # Round-3 audit CRIT-1: per-module Generator
        self._rng = rng if rng is not None else np.random.default_rng()
        # Passive pattern grid -- still evolves under plain diffusion so the
        # ``predict`` save/restore/perturb path keeps working unchanged.
        self.grid = self._rng.standard_normal((grid_size, grid_size)) * 0.1
        # Gray-Scott state. u starts at 1.0 everywhere; v at 0.0; then a
        # central square is seeded with u=0.5, v=0.25 -- the standard
        # perturbation that kicks off pattern formation.
        u = np.ones((grid_size, grid_size), dtype=float)
        v = np.zeros((grid_size, grid_size), dtype=float)
        c = grid_size // 2
        r = max(1, grid_size // 8)
        u[c - r : c + r, c - r : c + r] = 0.5
        v[c - r : c + r, c - r : c + r] = 0.25
        self.morphogens: list[np.ndarray] = [u, v]
        # Extra morphogens (n_signals > 2): keep linear-diffusion behaviour so
        # callers that constructed with more signals still get a list of the
        # expected length. These are not part of the Gray-Scott system.
        for _ in range(max(0, n_signals - 2)):
            self.morphogens.append(self._rng.standard_normal((grid_size, grid_size)) * 0.1)
        # Gray-Scott coefficients.
        self.du_rate = self._DEFAULT_DU
        self.dv_rate = self._DEFAULT_DV
        self.feed_rate = self._DEFAULT_FEED
        self.kill_rate = self._DEFAULT_KILL
        # ``diffusion_rate`` is retained as the explicit-Euler timestep for
        # the linear-diffusion grid AND as the Gray-Scott timestep (kept small
        # for stability of the explicit scheme). ``update()`` mutates it.
        self.diffusion_rate = 0.05

    def _laplacian(self, field: np.ndarray) -> np.ndarray:
        if _HAS_JIT:
            return _morphogenetic_laplacian(np.ascontiguousarray(field, dtype=float))
        return (
            np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0)
            + np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1)
            - 4 * field
        )

    def step(self) -> None:
        # 1) Gray-Scott reaction-diffusion update on (u, v).
        u, v = self.morphogens[0], self.morphogens[1]
        u_lap = self._laplacian(u)
        v_lap = self._laplacian(v)
        uvv = u * v * v
        dt = self.diffusion_rate
        u_new = u + dt * (
            self.du_rate * u_lap - uvv + self.feed_rate * (1.0 - u)
        )
        v_new = v + dt * (
            self.dv_rate * v_lap + uvv - (self.feed_rate + self.kill_rate) * v
        )
        # Clip to the Gray-Scott domain [0, 1] for explicit-Euler stability.
        np.clip(u_new, 0.0, 1.0, out=u_new)
        np.clip(v_new, 0.0, 1.0, out=v_new)
        self.morphogens[0] = u_new
        self.morphogens[1] = v_new
        # 2) Legacy linear diffusion on grid + any extra morphogens (n_signals>2).
        self.grid += dt * self._laplacian(self.grid)
        # Round-9 audit R9-008: clip ``grid`` to [0, 1] after the diffusion
        # update, matching the Gray-Scott (u, v) pair above and the extra
        # morphogens below. ``grid`` is initialised from N(0, 0.1²) (can be
        # negative) and evolved under the same plain Laplacian diffusion as
        # the extra morphogens, but was the ONLY diffusion-evolved field
        # lacking the [0, 1] safety net. ``develop`` returns
        # ``pattern = grid * (1 + 0.1 * morphogen_sum)``, so a signed grid
        # mixed a [-0.3, 0.3]-ish quantity with [0, 1]-clipped morphogens --
        # the exact semantic inconsistency Round-6 THEORY6-18 sought to
        # eliminate. The default ``diffusion_rate=0.05`` (clamped to
        # [0.001, 0.2] by update()) keeps the update stable (CFL bound 0.25),
        # but ``grid`` was the sole field that would diverge unchecked if
        # the clamp were ever raised above 0.25.
        np.clip(self.grid, 0.0, 1.0, out=self.grid)
        # Round-6 audit THEORY6-18 / NEW5-6: clip extra morphogens (n_signals>2)
        # to [0, 1] after the diffusion update. The Gray-Scott (u, v) pair is
        # clipped above, but the extra morphogens were left unbounded — they
        # are initialised from N(0, 0.1²) (can be negative) and evolved under
        # plain Laplacian diffusion with no boundary enforcement, contradicting
        # the docstring's claim that morphogens are concentrations (which are
        # physically non-negative) and contaminating ``develop``'s
        # ``pattern = grid * (1 + 0.1 * morphogen_sum)`` via unbounded sums.
        for m in self.morphogens[2:]:
            m += dt * self._laplacian(m)
            np.clip(m, 0.0, 1.0, out=m)

    def develop(self, n_steps: int = 50) -> np.ndarray:
        for _ in range(n_steps):
            self.step()
        if self.morphogens:
            morphogen_sum = np.add.reduce(self.morphogens)
        else:
            morphogen_sum = np.zeros_like(self.grid)
        pattern = self.grid * (1.0 + 0.1 * morphogen_sum)
        return pattern


class CellularAutomata:
    """Cellular automaton for distributed computation."""

    def __init__(self, size: int = 64, rule: int = 30, rng: np.random.Generator | None = None):
        self.size = size
        self.rule = rule
        # Round-3 audit CRIT-1: per-module Generator
        self._rng = rng if rng is not None else np.random.default_rng()
        self.state = self._rng.integers(0, 2, size=size)

    def _apply_rule(self, left: int, center: int, right: int) -> int:
        index = (left << 2) | (center << 1) | right
        return (self.rule >> index) & 1

    def step(self) -> None:
        if _HAS_JIT:
            # JIT path: all cells updated in one pass -- the biggest win.
            self.state = _cellular_automata_step(
                np.ascontiguousarray(self.state),
                int(self.rule),
                int(self.size),
            )
            return
        new_state = np.zeros(self.size, dtype=int)
        for i in range(self.size):
            left = self.state[(i - 1) % self.size]
            center = self.state[i]
            right = self.state[(i + 1) % self.size]
            new_state[i] = self._apply_rule(left, center, right)
        self.state = new_state

    def evolve(self, n_steps: int = 50) -> np.ndarray:
        history = [self.state.copy()]
        for _ in range(n_steps):
            self.step()
            history.append(self.state.copy())
        return np.array(history)

    def step_n(self, n: int) -> np.ndarray:
        """Advance ``n`` steps WITHOUT recording history (Fix 11).

        Returns the final state. Used by hot loops (e.g.
        ``PatternMiner._best_automaton_rule``) that only need the final state
        rather than the full evolution tape -- avoids building 256 histories.
        """
        for _ in range(n):
            self.step()
        return self.state


class BiologicalSubstrate(CognitiveModule):
    """
    Biological computation substrate.
    - DNA storage with crossover-based self-generation
    - Morphogenetic field for self-organization
    - Cellular automata for distributed computation
    """

    def __init__(self, dim: int = 64, rng: np.random.Generator | None = None):
        self.dim = dim
        # Round-3 audit CRIT-1: per-module Generator
        self._rng = rng if rng is not None else np.random.default_rng()
        # Per-module re-entrant lock (Phase D / think() lockless update):
        # protects ``process`` / ``predict`` / ``update`` from concurrent
        # think() calls racing on ``self._rng`` and shared mutable state
        # (dna_storage, morphogenetic, automata, _cycle_count). RLock
        # allows re-entry. ``predict``'s save/restore pattern for the
        # morphogenetic grid benefits most from locking — without it, two
        # concurrent ``predict`` calls could each save a different grid,
        # then both restore, clobbering each other.
        self._lock = threading.RLock()
        self.dna_storage = DNAStorage(capacity=16, rng=self._rng)
        self.morphogenetic = MorphogeneticField(grid_size=16, rng=self._rng)
        self.automata = CellularAutomata(size=dim, rng=self._rng)
        # Sequence counter so each ``process`` call stores under a unique key
        # (Fix 4): the original code always stored under "input", overwriting
        # the previous entry, so ``generate`` (which needs >=2 stored
        # sequences) never actually performed crossover.
        self._cycle_count = 0
        # Round-8 audit PERF8-3: cache the last ``process`` output so ``predict``
        # does not re-run ``morphogenetic.develop(n_steps=5)`` every think()
        # cycle. ``BiologicalSubstrate`` was the only cognitive module without
        # this cache — every other module (ConsciousnessCore, MathUniverse,
        # ...) already reused its process output in predict (Fix 12). The
        # morphogenetic develop step is the heaviest per-cycle op in this
        # module (a 16x16 grid x 5 diffusion iterations), so the redundant
        # pass was a real per-cycle cost.
        self._last_process_output: np.ndarray | None = None

    def __getstate__(self) -> dict:
        # Phase D: per-module RLock is not picklable. Strip it here and
        # rebuild in __setstate__. Hold the lock so concurrent think()
        # blocks during the snapshot (consistent array copy).
        with self._lock:
            return {k: v for k, v in self.__dict__.items() if k != "_lock"}

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._lock = threading.RLock()

    def process(self, signal: Signal) -> Signal:
        with self._lock:
            # Use a unique per-cycle key so the store accumulates a population of
            # signals and crossover/recombination actually runs (Fix 4). Capacity
            # is bounded by DNAStorage (oldest evicted automatically).
            self.dna_storage.store(f"signal_{self._cycle_count}", signal.data)
            self._cycle_count += 1
            generated = self.dna_storage.generate(signal)
            pattern = self.morphogenetic.develop(n_steps=10)
            pattern_flat = pattern.flatten()[: self.dim]
            if len(pattern_flat) < self.dim:
                pattern_flat = np.pad(pattern_flat, (0, self.dim - len(pattern_flat)))
            self.automata.step_n(5)
            ca_signal = self.automata.state.astype(float)
            combined = 0.4 * generated.data[: self.dim] + 0.3 * pattern_flat + 0.3 * ca_signal
            if len(combined) < self.dim:
                combined = np.pad(combined, (0, self.dim - len(combined)))
            # Round-8 audit PERF8-3: cache for ``predict`` to reuse.
            self._last_process_output = combined[: self.dim]
            return Signal(data=combined[: self.dim], metadata={"source": "biological"})

    def predict(self, signal: Signal) -> Prediction:
        with self._lock:
            # Round-8 audit PERF8-3: reuse the cached process output when available
            # so we do not re-run ``morphogenetic.develop(n_steps=5)`` (the
            # heaviest op in this module) twice per think() cycle. Falls back to
            # the full perturbed-develop pass when ``predict`` is called
            # standalone (no prior ``process`` in this cycle), matching the
            # ConsciousnessCore/MathUniverse pattern (Fix 12).
            if self._last_process_output is not None:
                predicted = self._last_process_output
                var = float(np.var(predicted))
                uncertainty = var if np.isfinite(var) else 1.0
                return Prediction(value=predicted[: self.dim], uncertainty=uncertainty)
            # Seed the morphogenetic field development from the incoming signal
            # rather than ignoring it (Fix 17): project the signal onto the grid
            # via an outer product so the prediction actually reflects the input.
            seed = signal.data[: self.dim]
            if len(seed) < self.dim:
                seed = np.pad(seed, (0, self.dim - len(seed)))
            gs = self.morphogenetic.grid.shape[0]
            # Build a (gs, gs) perturbation from the first ``gs`` signal samples.
            seed_vec = seed[:gs]
            if seed_vec.shape[0] < gs:
                seed_vec = np.pad(seed_vec, (0, gs - seed_vec.shape[0]))
            perturb = 0.05 * np.outer(seed_vec, seed_vec)
            # Save/restore the morphogenetic grid so predict() stays read-only
            # w.r.t. substrate state across the parallel predict step.
            saved_grid = self.morphogenetic.grid.copy()
            saved_morphogens = [m.copy() for m in self.morphogenetic.morphogens]
            saved_rate = self.morphogenetic.diffusion_rate
            try:
                self.morphogenetic.grid = self.morphogenetic.grid + perturb
                pattern = self.morphogenetic.develop(n_steps=5)
            finally:
                self.morphogenetic.grid = saved_grid
                self.morphogenetic.morphogens = saved_morphogens
                self.morphogenetic.diffusion_rate = saved_rate
            predicted = pattern.flatten()[: self.dim]
            if len(predicted) < self.dim:
                predicted = np.pad(predicted, (0, self.dim - len(predicted)))
            # Round-3 audit: np.var of empty/NaN returns NaN; guard.
            var = float(np.var(predicted))
            uncertainty = var if np.isfinite(var) else 1.0
            return Prediction(value=predicted[: self.dim], uncertainty=uncertainty)

    def update(self, prediction_error: float) -> None:
        with self._lock:
            if not np.isfinite(prediction_error):
                return
            # Round-8 audit THEORY8-11: clip to NON-NEGATIVE (MSE is non-negative
            # by construction; a negative value would invert the rate update,
            # matching the THEORY8-7 fix in active_inference.update).
            prediction_error = float(np.clip(prediction_error, 0.0, 1e6))
            if prediction_error == 0.0:
                return
            # Round-8 audit THEORY8-11: use ``tanh`` to keep the rate update
            # bounded and proportional. The previous ``rate += err * 0.01`` saturated
            # to the [0.001, 0.2] clip boundaries for any |err| > 20, turning the
            # update into a binary "small error -> floor, large error -> ceiling"
            # control with no gradient in between. ``tanh`` preserves the
            # small-error/small-update gradient while capping large errors.
            # ``0.01 * tanh(err * 0.001)`` ranges in [-0.01, 0.01] (err=1e3 ->
            # ~0.01, err=1 -> 1e-5), so a single update never moves the rate by
            # more than 0.01 — well inside the [0.001, 0.2] stability window.
            delta = 0.01 * float(np.tanh(prediction_error * 0.001))
            new_rate = self.morphogenetic.diffusion_rate + delta
            self.morphogenetic.diffusion_rate = min(0.2, max(0.001, new_rate))
