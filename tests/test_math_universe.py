# tests/test_math_universe.py
import numpy as np
from zero_data_model.math_universe import MathematicalUniverse, InformationGeometry, TopologicalAnalyzer, FractalGenerator
from zero_data_model.base import Signal


def test_fisher_metric():
    ig = InformationGeometry(dim=8)
    dist = np.abs(np.random.randn(8)) + 0.1
    metric = ig.fisher_metric(dist)
    assert metric.shape == (8,)
    assert np.all(metric > 0)


def test_geodesic():
    ig = InformationGeometry(dim=8)
    p = np.abs(np.random.randn(8)) + 0.1
    q = np.abs(np.random.randn(8)) + 0.1
    mid = ig.geodesic(p, q, t=0.5)
    assert abs(np.sum(mid) - 1.0) < 1e-6


def test_topological_features():
    ta = TopologicalAnalyzer(dim=16)
    data = np.random.randn(16)
    features = ta.topological_features(data)
    assert features.shape == (16,)


def test_fractal_generate():
    fg = FractalGenerator(dim=16)
    initial = np.random.randn(16)
    result = fg.generate(initial, n_iterations=5)
    assert result.shape == (16,)


def test_math_universe_process():
    mu = MathematicalUniverse(dim=16)
    signal = Signal(data=np.random.randn(16))
    result = mu.process(signal)
    assert result.data.shape == (16,)
