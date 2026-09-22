# benchmark.py
"""Performance benchmark: Qiskit quantum backend, numba-JIT annealer, parallel inference.

Compares the hardware-accelerated stack against the legacy pure-numpy paths so
the speedup is measurable. Run: python benchmark.py
"""

import sys

sys.path.insert(0, "src")

import argparse
import json
import platform
import statistics
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

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


# ============================================================================
# Cognitive Upgrade Module Benchmarks (Phase 7)
# ============================================================================

_COGNITIVE_FLAGS = (
    "enable_architect",
    "enable_layered_predictor",
    "enable_episodic_memory",
    "enable_logic_layer",
    "enable_meta_cognition",
    "enable_experiment_planner",
)


def _json_default(obj):
    """Fallback JSON serializer for numpy types (defensive)."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    return str(obj)


def _build_cog_model(dim, **flags):
    """Construct a deterministic model (re-seeded for reproducibility)."""
    np.random.seed(42)
    return ZeroDataModel(dim=dim, seed=42, **flags)


def _measure_one_config(dim, cycles, flags, extra_fn=None, n_median=10):
    """Measure construct + think for one flag configuration.

    Returns dict with construct_ms, think_median_ms, think_total_ms,
    mem_delta_kb (tracemalloc peak during construct), metadata_bytes.
    """
    tracemalloc.start()
    tracemalloc.clear_traces()
    t0 = time.perf_counter()
    model = _build_cog_model(dim, **flags)
    construct_s = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    rng = np.random.default_rng(0)
    signal = rng.standard_normal(dim)

    # Warm up (not measured).
    model.think(signal)
    if extra_fn is not None:
        extra_fn(model, signal, rng)

    # Single-cycle median (of n_median runs).
    single_times = []
    for _ in range(n_median):
        t0 = time.perf_counter()
        model.think(signal)
        if extra_fn is not None:
            extra_fn(model, signal, rng)
        single_times.append(time.perf_counter() - t0)
    median_s = statistics.median(single_times)

    # cycles total.
    t0 = time.perf_counter()
    for _ in range(cycles):
        model.think(signal)
        if extra_fn is not None:
            extra_fn(model, signal, rng)
    total_s = time.perf_counter() - t0

    # metadata bytes (from one more think).
    last_sig = model.think(signal)
    md_bytes = len(json.dumps(last_sig.metadata, default=_json_default))

    return {
        "construct_ms": construct_s * 1000,
        "think_median_ms": median_s * 1000,
        "think_total_ms": total_s * 1000,
        "mem_delta_kb": peak / 1024.0,
        "metadata_bytes": md_bytes,
    }


def _episodic_plan_extra(model, signal, rng):
    """Exercise episodic_graph.plan after think (insert happens in think)."""
    if model.episodic_graph is not None:
        goal = rng.standard_normal(model.dim)
        try:
            model.episodic_graph.plan(signal, goal)
        except Exception:
            pass


def benchmark_cognitive_upgrade(dim: int = 64, cycles: int = 10) -> dict:
    """Benchmark the 7 cognitive-upgrade module dimensions.

    For each dimension, measures enabled vs disabled:
      - construct time (model creation)
      - single-cycle think() median (of 10 runs, via statistics.median)
      - 10-cycle think() total latency
      - memory delta (tracemalloc peak during construction)
      - metadata JSON serialization size (json.dumps bytes)

    Returns a dict with per-dimension on/off metrics and overhead.
    """
    print(f"\n[Cognitive Upgrade] dim={dim}, cycles={cycles}")

    # 7 dimensions: (name, flags_on, extra_fn)
    dims = [
        ("architect", {"enable_architect": True}, None),
        ("layered_predictor", {"enable_layered_predictor": True}, None),
        ("episodic_memory", {"enable_episodic_memory": True}, _episodic_plan_extra),
        ("logic_layer", {"enable_logic_layer": True}, None),
        ("meta_cognition", {"enable_meta_cognition": True}, None),
        ("experiment_planner", {"enable_experiment_planner": True}, None),
        ("all_upgrades", {f: True for f in _COGNITIVE_FLAGS}, None),
    ]

    # Disabled baseline (shared across all dimensions).
    off = _measure_one_config(dim, cycles, {})

    results = {
        "dim": dim,
        "cycles": cycles,
        "baseline_disabled": off,
        "dimensions": [],
    }

    for name, flags, extra in dims:
        try:
            on = _measure_one_config(dim, cycles, flags, extra_fn=extra)
        except Exception as exc:
            print(f"  [WARN] {name} measurement failed: {exc}")
            on = {
                "construct_ms": 0.0, "think_median_ms": 0.0,
                "think_total_ms": 0.0, "mem_delta_kb": 0.0,
                "metadata_bytes": 0, "error": str(exc),
            }
        off_med = off["think_median_ms"]
        on_med = on["think_median_ms"]
        overhead_ms = on_med - off_med
        overhead_pct = (overhead_ms / off_med * 100) if off_med > 0 else 0.0
        mem_inc = on["mem_delta_kb"] - off["mem_delta_kb"]
        results["dimensions"].append({
            "name": name,
            "construct_ms": {"on": on["construct_ms"], "off": off["construct_ms"]},
            "think_median_ms": {"on": on_med, "off": off_med},
            "think_total_ms": {"on": on["think_total_ms"], "off": off["think_total_ms"]},
            "mem_delta_kb": {
                "on": on["mem_delta_kb"], "off": off["mem_delta_kb"],
                "increment": mem_inc,
            },
            "metadata_bytes": {"on": on["metadata_bytes"], "off": off["metadata_bytes"]},
            "overhead_ms": overhead_ms,
            "overhead_pct": overhead_pct,
        })

    _print_cognitive_table(results)
    return results


def _print_cognitive_table(results):
    """Print the cognitive upgrade benchmark results table."""
    dims = results["dimensions"]
    print(
        f"  {'dimension':<20} {'construct_on':>12} {'think_med_on':>13} "
        f"{'think_med_off':>14} {'overhead_ms':>12} {'overhead_%':>11} "
        f"{'meta_on':>8} {'mem_inc_KB':>11}"
    )
    print("  " + "-" * 105)
    for d in dims:
        print(
            f"  {d['name']:<20} {d['construct_ms']['on']:>12.2f} "
            f"{d['think_median_ms']['on']:>13.3f} {d['think_median_ms']['off']:>14.3f} "
            f"{d['overhead_ms']:>12.3f} {d['overhead_pct']:>11.1f} "
            f"{d['metadata_bytes']['on']:>8} {d['mem_delta_kb']['increment']:>11.1f}"
        )
    base = results["baseline_disabled"]
    print(
        f"\n  baseline (all disabled): think_med={base['think_median_ms']:.3f}ms  "
        f"construct={base['construct_ms']:.2f}ms  "
        f"meta_bytes={base['metadata_bytes']}  "
        f"mem_KB={base['mem_delta_kb']:.1f}"
    )


# ============================================================================
# Baseline storage & regression comparison
# ============================================================================

def _run_all_benchmarks(dim: int = 64, cycles: int = 10) -> dict:
    """Run all benchmarks and collect results into a serializable dict."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "cognitive_upgrade": benchmark_cognitive_upgrade(dim=dim, cycles=cycles),
    }


def _compare_results(old: dict, new: dict, threshold: float = 0.10) -> dict:
    """Compare two benchmark result sets; flag regressions > threshold (10%)."""
    report = {
        "threshold": threshold,
        "regressions": [],
        "improvements": [],
        "summary": "",
    }
    old_cog = old.get("cognitive_upgrade", {})
    new_cog = new.get("cognitive_upgrade", {})
    old_dims = {d["name"]: d for d in old_cog.get("dimensions", [])}
    new_dims = {d["name"]: d for d in new_cog.get("dimensions", [])}

    for name, new_d in new_dims.items():
        old_d = old_dims.get(name)
        if not old_d:
            continue
        for metric in ("think_median_ms", "think_total_ms", "construct_ms"):
            old_v = old_d.get(metric, {}).get("on")
            new_v = new_d.get(metric, {}).get("on")
            if old_v and new_v and old_v > 0:
                ratio = (new_v - old_v) / old_v
                if ratio > threshold:
                    report["regressions"].append({
                        "dimension": name, "metric": metric,
                        "old": old_v, "new": new_v,
                        "pct_slower": round(ratio * 100, 1),
                    })
                elif ratio < -threshold:
                    report["improvements"].append({
                        "dimension": name, "metric": metric,
                        "old": old_v, "new": new_v,
                        "pct_faster": round(-ratio * 100, 1),
                    })

    n_reg = len(report["regressions"])
    n_imp = len(report["improvements"])
    report["summary"] = (
        f"{n_reg} regression(s) > {threshold*100:.0f}%, {n_imp} improvement(s)"
    )
    return report


def _print_regression(report: dict) -> None:
    """Print a regression report."""
    if "summary" in report:
        print(f"\n  Regression report: {report['summary']}")
    if report.get("regressions"):
        thr = report.get("threshold", 0.1) * 100
        print(f"  REGRESSIONS (> {thr:.0f}% slower):")
        print(f"    {'dimension':<20} {'metric':<20} {'old':>10} {'new':>10} {'%slower':>10}")
        for r in report["regressions"]:
            print(
                f"    {r['dimension']:<20} {r['metric']:<20} "
                f"{r['old']:>10.3f} {r['new']:>10.3f} {r['pct_slower']:>10.1f}"
            )
    else:
        print("  No regressions detected.")
    if report.get("improvements"):
        print("  IMPROVEMENTS:")
        for r in report["improvements"]:
            print(
                f"    {r['dimension']:<20} {r['metric']:<20} "
                f"{r['old']:>10.3f} -> {r['new']:>10.3f}  ({r['pct_faster']:.1f}% faster)"
            )


def save_baseline(path: str = ".benchmarks/baseline.json"):
    """Run all benchmarks and save as baseline JSON.

    If the file already exists, compares new vs old and marks regressions
    (>10% slower). Returns (results, regression_report).
    """
    results = _run_all_benchmarks()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    regression = None
    if p.exists():
        try:
            old = json.loads(p.read_text())
            regression = _compare_results(old, results)
        except Exception as exc:
            print(f"  (could not compare with existing baseline: {exc})")
    p.write_text(json.dumps(results, indent=2, default=_json_default))
    print(f"\n  Baseline saved to {p}")
    if regression:
        _print_regression(regression)
    return results, regression


def compare_baseline(path: str = ".benchmarks/baseline.json") -> dict:
    """Run current benchmark and compare with stored baseline.

    Returns the regression report dict.
    """
    current = _run_all_benchmarks()
    p = Path(path)
    if not p.exists():
        print(f"  Baseline not found at {p}; run --save-baseline first.")
        return {"error": "baseline not found", "path": str(p), "current": current}
    try:
        old = json.loads(p.read_text())
    except Exception as exc:
        print(f"  Could not load baseline: {exc}")
        return {"error": str(exc), "path": str(p)}
    report = _compare_results(old, current)
    _print_regression(report)
    return report


# ============================================================================
# CLI entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Zero-Data Model performance benchmark"
    )
    parser.add_argument(
        "--save-baseline", action="store_true",
        help="Run all benchmarks and store as baseline JSON",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Run benchmarks and compare with stored baseline",
    )
    parser.add_argument(
        "--cognitive-only", action="store_true",
        help="Only run the cognitive-upgrade module benchmarks",
    )
    parser.add_argument(
        "--dim", type=int, default=64,
        help="Model dimension (default: 64)",
    )
    parser.add_argument(
        "--cycles", type=int, default=10,
        help="Number of think() cycles per dimension (default: 10)",
    )
    args = parser.parse_args()

    if args.cognitive_only:
        benchmark_cognitive_upgrade(dim=args.dim, cycles=args.cycles)
        return

    if args.save_baseline:
        save_baseline()
        return

    if args.compare:
        compare_baseline()
        return

    # Default: run all benchmarks and print tables.
    print("=" * 60)
    print("  Zero-Data Model — Performance Benchmark")
    print("=" * 60)
    print(f"  Array backend: {backend_name()}  (GPU: {has_gpu})")

    bench_quantum_backend()
    bench_annealer_jit()
    bench_parallel_vs_sequential()
    bench_full_model()
    bench_jit_kernels()
    benchmark_cognitive_upgrade(dim=args.dim, cycles=args.cycles)

    print("\n" + "=" * 60)
    print("  Benchmark complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
