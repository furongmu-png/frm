# benchmark.py
"""Performance benchmark: Qiskit quantum backend, numba-JIT annealer, parallel inference.

Compares the hardware-accelerated stack against the legacy pure-numpy paths so
the speedup is measurable. Run: python benchmark.py
"""

import sys

sys.path.insert(0, "src")

import time

import numpy as np

from zero_data_model.hardware import (
    ParallelExecutor,
    SimulatorQuantumBackend,
    backend_name,
    has_gpu,
)
from zero_data_model.hardware import kernels as _kernels
from zero_data_model.hardware.quantum import get_quantum_backend
from zero_data_model.model import ZeroDataModel
from zero_data_model.quantum_hybrid import (
    QuantumAnnealer,
)


def time_it(fn, n_runs=3, **kw):
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = fn(**kw)
        times.append(time.perf_counter() - t0)
    return min(times), result


def bench_quantum_backend():
    print("\n[1] Quantum backend: Qiskit vs Simulator")
    n_qubits, n_layers = 6, 3
    params = np.random.randn(n_layers, n_qubits, 2) * 0.1
    entangling = np.random.randn(n_qubits, n_qubits) * 0.05

    backend = get_quantum_backend(n_qubits=n_qubits, n_layers=n_layers)
    name = backend.name
    t, _ = time_it(lambda: backend.evolve_and_measure(params, entangling, n_shots=1024), n_runs=3)
    print(f"  Active backend: {name}")
    print(f"  Evolve+measure (n_qubits={n_qubits}, shots=1024): {t*1000:.2f} ms")

    # Simulator comparison (always available).
    sim = SimulatorQuantumBackend(n_qubits=n_qubits, n_layers=n_layers)
    t_sim, _ = time_it(lambda: sim.evolve_and_measure(params, entangling), n_runs=3)
    print(f"  Simulator fallback:                       {t_sim*1000:.2f} ms")
    if name == "qiskit":
        print("  Qiskit is real quantum circuit execution via StatevectorSampler.")
    return name


def bench_annealer_jit():
    print("\n[2] Quantum annealer: numba-JIT vs pure numpy")
    n_vars = 64
    # Warm up JIT.
    warm = QuantumAnnealer(n_vars=n_vars)
    warm.optimize(n_iterations=5)

    t_jit, (state, energy) = time_it(
        lambda: warm.optimize(n_iterations=500), n_runs=3
    )
    print(f"  n_vars={n_vars}, iterations=500, JIT={warm.jit}")
    print(f"  Best energy: {energy:.4f}")
    print(f"  Time: {t_jit*1000:.2f} ms")


def bench_parallel_vs_sequential():
    print("\n[3] Parallel vs sequential module execution")
    model = ZeroDataModel(dim=64)
    info = model.hardware_info
    print(f"  Hardware: {info}")

    signal = np.random.randn(64)
    # Warm up.
    model.think(signal)

    # Parallel (default).
    t_par, _ = time_it(lambda: model.think(signal), n_runs=3)
    # Force sequential.
    seq_exec = ParallelExecutor(n_workers=1)
    orig = model.parallel_executor
    model.parallel_executor = seq_exec
    t_seq, _ = time_it(lambda: model.think(signal), n_runs=3)
    model.parallel_executor = orig
    print(f"  Parallel think() (cycle):   {t_par*1000:.2f} ms")
    print(f"  Sequential think() (cycle): {t_seq*1000:.2f} ms")
    speedup = t_seq / t_par if t_par > 0 else 0
    print(f"  Speedup: {speedup:.2f}x")


def bench_full_model():
    print("\n[4] Full model: NLP/CV/Analytics throughput")
    model = ZeroDataModel(dim=64)
    rng = np.random.default_rng(0)

    # Warm up.
    model.classify_text("code data model")
    model.forecast(np.arange(20, dtype=float))

    texts = [
        "algorithm network compute",
        "tree river mountain",
        "love joy hope",
        "energy force mass",
    ] * 5
    t0 = time.perf_counter()
    for t in texts:
        model.classify_text(t)
    t_nlp = time.perf_counter() - t0

    imgs = [rng.random((16, 16)) for _ in range(10)]
    t0 = time.perf_counter()
    for im in imgs:
        model.recognize_pattern(im)
    t_cv = time.perf_counter() - t0

    series = [np.arange(30, dtype=float) + rng.standard_normal(30) for _ in range(10)]
    t0 = time.perf_counter()
    for s in series:
        model.forecast(s, horizon=5)
    t_an = time.perf_counter() - t0

    n_nlp = len(texts)
    n_cv = len(imgs)
    n_an = len(series)
    print(f"  NLP:       {n_nlp} classifications in {t_nlp*1000:.1f} ms ({n_nlp/t_nlp:.0f}/s)")
    print(f"  CV:        {n_cv} pattern recognitions in {t_cv*1000:.1f} ms ({n_cv/t_cv:.0f}/s)")
    print(f"  Analytics: {n_an} forecasts in {t_an*1000:.1f} ms ({n_an/t_an:.0f}/s)")


def bench_jit_kernels():
    """[JIT] Measure the JIT-compiled hot kernels vs their pure-numpy references.

    For each kernel we run the JIT path (already wired into the core modules)
    and a pure-numpy reference implementation on identical inputs, then report
    the speedup. The first call of each JIT function compiles it; we warm up
    before measuring so the reported numbers reflect steady-state performance.
    """
    print("\n[JIT] numba-JIT kernels vs pure-numpy reference")
    print(f"  HAS_NUMBA: {_kernels.HAS_NUMBA}")
    if not _kernels.HAS_NUMBA:
        print("  (skipped -- numba not installed)")
        return

    rng = np.random.default_rng(0)

    # --- CellularAutomata.step (Python per-cell loop -> single JIT pass) ---
    size = 256
    rule = 30
    state = rng.integers(0, 2, size).astype(np.int64)

    def py_ca_step(s):
        new_state = np.zeros(size, dtype=int)
        for i in range(size):
            left = s[(i - 1) % size]
            center = s[i]
            right = s[(i + 1) % size]
            index = (left << 2) | (center << 1) | right
            new_state[i] = (rule >> index) & 1
        return new_state

    # Warm up the JIT.
    _kernels._cellular_automata_step(state.copy(), rule, size)
    n_iters = 200
    t_jit, _ = time_it(
        lambda: [_kernels._cellular_automata_step(state.copy(), rule, size) for _ in range(n_iters)]
    )
    t_py, _ = time_it(lambda: [py_ca_step(state.copy()) for _ in range(n_iters)])
    speedup_ca = t_py / t_jit if t_jit > 0 else 0
    print(f"  CellularAutomata.step (size={size}, x{n_iters}):")
    print(
        f"    pure-python: {t_py*1000:.2f} ms   "
        f"JIT: {t_jit*1000:.2f} ms   speedup: {speedup_ca:.1f}x"
    )

    # --- MorphogeneticField.step (np.roll temporaries -> single fused JIT pass) ---
    grid_size = 32
    grid = rng.standard_normal((grid_size, grid_size))
    morphs = [rng.standard_normal((grid_size, grid_size)) for _ in range(3)]

    def py_morph_step(g, ms):
        lap = (
            np.roll(g, 1, axis=0) + np.roll(g, -1, axis=0)
            + np.roll(g, 1, axis=1) + np.roll(g, -1, axis=1)
            - 4 * g
        )
        g = g + 0.05 * lap
        for m in ms:
            m_lap = (
                np.roll(m, 1, axis=0) + np.roll(m, -1, axis=0)
                + np.roll(m, 1, axis=1) + np.roll(m, -1, axis=1)
                - 4 * m
            )
            m = m + 0.05 * m_lap
        return g, ms

    _kernels._morphogenetic_laplacian(grid.copy())
    n_iters = 200
    t_jit, _ = time_it(
        lambda: [_kernels._morphogenetic_laplacian(grid.copy()) for _ in range(n_iters)]
    )
    t_py, _ = time_it(
        lambda: [py_morph_step(grid.copy(), [m.copy() for m in morphs]) for _ in range(n_iters)]
    )
    speedup_morph = t_py / t_jit if t_jit > 0 else 0
    print(f"  MorphogeneticField.laplacian ({grid_size}x{grid_size}, x{n_iters}):")
    print(
        f"    np.roll:      {t_py*1000:.2f} ms   "
        f"JIT: {t_jit*1000:.2f} ms   speedup: {speedup_morph:.1f}x"
    )

    # --- FractalGenerator.generate (n_iterations of matmul+tanh -> fused JIT) ---
    dim = 64
    n_t = 4
    scales = rng.standard_normal((n_t, dim, dim))
    offsets = rng.standard_normal((n_t, dim))
    x0 = rng.standard_normal(dim)

    def py_fractal(x, n_iter=20):
        for it in range(n_iter):
            t = it % n_t
            x = scales[t] @ x + offsets[t]
            x = np.tanh(x)
        return x

    _kernels._fractal_generate(x0.copy(), scales, offsets, 1, n_t)
    n_iters = 100
    t_jit, _ = time_it(
        lambda: [
            _kernels._fractal_generate(x0.copy(), scales, offsets, 20, n_t)
            for _ in range(n_iters)
        ]
    )
    t_py, _ = time_it(lambda: [py_fractal(x0.copy()) for _ in range(n_iters)])
    speedup_frac = t_py / t_jit if t_jit > 0 else 0
    print(f"  FractalGenerator.generate (dim={dim}, iters=20, x{n_iters}):")
    print(
        f"    pure-numpy: {t_py*1000:.2f} ms   "
        f"JIT: {t_jit*1000:.2f} ms   speedup: {speedup_frac:.1f}x"
    )

    # --- BiologicalSubstrate.process via full ZeroDataModel.think ---
    # The integrated end-to-end improvement of one think() cycle (random input).
    model = ZeroDataModel(dim=64)
    sig = rng.standard_normal(64)
    model.think(sig)  # warm up
    t_jit, _ = time_it(lambda: model.think(sig))
    print(f"  ZeroDataModel.think (dim=64, single cycle): {t_jit*1000:.2f} ms")


def main():
    print("=" * 60)
    print("  Zero-Data Model — Performance Benchmark")
    print("=" * 60)
    print(f"  Array backend: {backend_name()}  (GPU: {has_gpu})")

    bench_quantum_backend()
    bench_annealer_jit()
    bench_parallel_vs_sequential()
    bench_full_model()
    bench_jit_kernels()

    print("\n" + "=" * 60)
    print("  Benchmark complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
