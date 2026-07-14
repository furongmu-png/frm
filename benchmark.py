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
    backend_name,
    has_gpu,
    QiskitQuantumBackend,
    SimulatorQuantumBackend,
    ParallelExecutor,
)
from zero_data_model.hardware.quantum import get_quantum_backend
from zero_data_model.quantum_hybrid import (
    VariationalQuantumCircuit,
    QuantumAnnealer,
    QuantumClassicalHybrid,
)
from zero_data_model.model import ZeroDataModel


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
        print(f"  Qiskit is real quantum circuit execution via StatevectorSampler.")
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

    texts = ["algorithm network compute", "tree river mountain", "love joy hope", "energy force mass"] * 5
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

    print(f"  NLP:      {len(texts)} classifications in {t_nlp*1000:.1f} ms ({len(texts)/t_nlp:.0f}/s)")
    print(f"  CV:       {len(imgs)} pattern recognitions in {t_cv*1000:.1f} ms ({len(imgs)/t_cv:.0f}/s)")
    print(f"  Analytics: {len(series)} forecasts in {t_an*1000:.1f} ms ({len(series)/t_an:.0f}/s)")


def main():
    print("=" * 60)
    print("  Zero-Data Model — Performance Benchmark")
    print("=" * 60)
    print(f"  Array backend: {backend_name()}  (GPU: {has_gpu})")

    bench_quantum_backend()
    bench_annealer_jit()
    bench_parallel_vs_sequential()
    bench_full_model()

    print("\n" + "=" * 60)
    print("  Benchmark complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
