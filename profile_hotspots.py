"""Profile the hot numpy kernels of the zero-data cognitive model.

Uses ``line_profiler`` to obtain line-by-line timings for one ``model.think()``
cycle and the key inner methods, then prints the top hotspots so the numba
JIT work can be focused where it matters.

The script degrades gracefully if ``line_profiler`` is not installed: it falls
back to a coarse ``time.perf_counter`` measurement of each method so the
script always runs. The ``@profile`` decorator is also defined as a no-op
when line_profiler is missing, so any module that does ``from profile import
profile`` style imports keeps working.
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, "src")

import numpy as np

# --- Guard the @profile decorator (line_profiler magic) ----------------------
# line_profiler injects a builtin ``profile`` decorator when running under
# ``kernprof``. When line_profiler is installed but the script is run as plain
# python, we expose the same name as a no-op so module imports never break.
try:
    import line_profiler  # noqa: F401

    def profile(func):
        """No-op decorator; line_profiler uses LineProfiler.add_function instead."""
        return func

    _HAS_LINE_PROFILER = True
except ImportError:  # pragma: no cover - optional dependency
    def profile(func):
        return func

    _HAS_LINE_PROFILER = False


def _collect_target_functions():
    """Collect the hot functions/methods to be profiled."""
    from zero_data_model.active_inference import (
        ActiveInferenceEngine,
        GenerativeModel,
    )
    from zero_data_model.biological import (
        BiologicalSubstrate,
        CellularAutomata,
        MorphogeneticField,
    )
    from zero_data_model.category_engine import (
        CategoryTheoryEngine,
        ToposEngine,
    )
    from zero_data_model.consciousness_core import (
        ConsciousnessCore,
        GlobalWorkspace,
        PredictiveLayer,
    )
    from zero_data_model.math_universe import (
        FractalGenerator,
        InformationGeometry,
        MathematicalUniverse,
        TopologicalAnalyzer,
    )
    from zero_data_model.quantum_hybrid import QuantumClassicalHybrid

    # Bound method references resolved against pre-built instances for clean
    # line-by-line reporting.
    rng = np.random.default_rng(0)
    dim = 64

    cc = ConsciousnessCore(dim=dim, n_layers=3)
    ai = ActiveInferenceEngine(state_dim=dim, obs_dim=dim, action_dim=dim // 2)
    ce = CategoryTheoryEngine(dim=dim)
    qh = QuantumClassicalHybrid(dim=dim)
    bio = BiologicalSubstrate(dim=dim)
    mu = MathematicalUniverse(dim=dim)

    # Small inputs that exercise each kernel.
    sig_data = rng.standard_normal(dim)

    targets = []

    # Consciousness core hot loops.
    targets.append(("PredictiveLayer.predict", PredictiveLayer.predict))
    targets.append(("GlobalWorkspace.broadcast", GlobalWorkspace.broadcast))
    targets.append(("ConsciousnessCore.process", ConsciousnessCore.process))

    # Active inference hot loops.
    targets.append(("GenerativeModel.predict_observation", GenerativeModel.predict_observation))
    targets.append(("GenerativeModel.predict_next_state", GenerativeModel.predict_next_state))
    targets.append(("GenerativeModel.infer_state", GenerativeModel.infer_state))
    targets.append(("ActiveInferenceEngine.select_action", ActiveInferenceEngine.select_action))
    targets.append(("ActiveInferenceEngine.process", ActiveInferenceEngine.process))

    # Category engine hot loops.
    targets.append(("ToposEngine.classify", ToposEngine.classify))
    targets.append(("CategoryTheoryEngine.find_isomorphism", CategoryTheoryEngine.find_isomorphism))
    targets.append(("CategoryTheoryEngine.process", CategoryTheoryEngine.process))

    # Quantum hybrid hot loop (classical path).
    targets.append(("QuantumClassicalHybrid.process", QuantumClassicalHybrid.process))

    # Biological hot loops -- the CA per-cell loop is the biggest hotspot.
    targets.append(("MorphogeneticField.step", MorphogeneticField.step))
    targets.append(("MorphogeneticField.develop", MorphogeneticField.develop))
    targets.append(("CellularAutomata.step", CellularAutomata.step))
    targets.append(("CellularAutomata.evolve", CellularAutomata.evolve))
    targets.append(("BiologicalSubstrate.process", BiologicalSubstrate.process))

    # Math universe hot loops.
    targets.append(("InformationGeometry.kl_divergence", InformationGeometry.kl_divergence))
    targets.append((
        "TopologicalAnalyzer.compute_betti_numbers",
        TopologicalAnalyzer.compute_betti_numbers,
    ))
    targets.append((
        "TopologicalAnalyzer.topological_features",
        TopologicalAnalyzer.topological_features,
    ))
    targets.append(("FractalGenerator.generate", FractalGenerator.generate))
    targets.append(("MathematicalUniverse.process", MathematicalUniverse.process))

    return targets, (cc, ai, ce, qh, bio, mu, sig_data, dim, rng)


def run_line_profiler():
    """Run line_profiler against all target functions for one think() cycle."""
    from line_profiler import LineProfiler

    from zero_data_model.model import ZeroDataModel

    targets, _ = _collect_target_functions()

    # Also profile the bound .think() method on a fresh model so we capture
    # the integration overhead as well.
    model = ZeroDataModel(dim=64)
    # Warm up the JIT annealer so its compile time is not in the profile.
    model.quantum_hybrid.annealer.optimize(n_iterations=5)

    rng = np.random.default_rng(0)
    sig_data = rng.standard_normal(64)

    lp = LineProfiler()
    for _name, fn in targets:
        lp.add_function(fn)
    lp.add_function(model.think)

    lp.enable_by_count()
    # Run several think cycles to get a meaningful sample.
    for _ in range(5):
        model.think(sig_data)
    lp.disable_by_count()

    print("\n" + "=" * 78)
    print("  Line-by-line profile: top hotspots in one model.think() cycle")
    print("=" * 78)
    lp.print_stats(stream=sys.stdout, output_unit=1e-6)


def run_coarse_timing():
    """Fallback: time each module's process() and the full think() cycle."""
    from zero_data_model.base import Signal
    from zero_data_model.model import ZeroDataModel

    model = ZeroDataModel(dim=64)
    rng = np.random.default_rng(0)
    sig = Signal(data=rng.standard_normal(64))

    # Warm up.
    model.think(sig.data)

    n_runs = 50
    t0 = time.perf_counter()
    for _ in range(n_runs):
        model.think(sig.data)
    total = (time.perf_counter() - t0) / n_runs
    print(f"\nCoarse timing (line_profiler not available): {total*1000:.2f} ms / think()")
    print("Install line_profiler for line-by-line breakdown: pip install line_profiler")


def main():
    if _HAS_LINE_PROFILER:
        run_line_profiler()
    else:
        run_coarse_timing()


if __name__ == "__main__":
    main()
