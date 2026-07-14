# tests/test_benchmark_comparison.py
"""Smoke tests for the zero-data vs sklearn comparison benchmark.

These tests do NOT assert accuracy/RMSE values (those depend on the seeded
model and data); they only verify that each comparison function runs and
returns the expected structure/types, so the benchmark stays runnable.
"""

from __future__ import annotations

import numpy as np
import pytest

# scikit-learn is required for the sklearn side of every comparison; skip the
# whole module gracefully if it is not installed.
pytest.importorskip("sklearn")

import benchmark_comparison as bench  # noqa: E402


@pytest.fixture(autouse=True)
def _seed_global_rng() -> None:
    """Pin the global RNG so model construction inside the benchmark is reproducible."""
    np.random.seed(bench.SEED)


def test_text_classification_comparison_runs() -> None:
    """Returns a list of (n_samples, zero_data_acc, sklearn_acc) tuples.

    sklearn_acc is None at N=0 (cannot train without data) and a float otherwise.
    """
    rows = bench.text_classification_comparison()

    assert isinstance(rows, list)
    assert len(rows) == len(bench.TEXT_N_VALUES)

    for idx, row in enumerate(rows):
        assert isinstance(row, tuple)
        assert len(row) == 3
        n_samples, zd_acc, sk_acc = row
        assert n_samples == bench.TEXT_N_VALUES[idx]
        assert isinstance(zd_acc, float)
        assert 0.0 <= zd_acc <= 1.0
        if n_samples == 0:
            # No training data -> sklearn cannot run.
            assert sk_acc is None
        else:
            assert isinstance(sk_acc, float)
            assert 0.0 <= sk_acc <= 1.0


def test_anomaly_comparison_runs() -> None:
    """Returns structured results: dicts with n_fit, zd metrics, and sk metrics.

    The sklearn entry is None at N_fit=0 (IsolationForest needs normal data to fit).
    """
    rows = bench.anomaly_comparison()

    assert isinstance(rows, list)
    assert len(rows) == len(bench.ANOMALY_N_VALUES)

    for idx, row in enumerate(rows):
        assert isinstance(row, dict)
        assert row["n_fit"] == bench.ANOMALY_N_VALUES[idx]
        # Zero-data metrics are always present.
        zd = row["zd"]
        assert set(zd.keys()) == {"precision", "recall", "f1"}
        for key in ("precision", "recall", "f1"):
            assert isinstance(zd[key], float)
            assert 0.0 <= zd[key] <= 1.0
        # sklearn metrics are None at N_fit=0, else a metrics dict.
        if row["n_fit"] == 0:
            assert row["sk"] is None
        else:
            sk = row["sk"]
            assert isinstance(sk, dict)
            assert set(sk.keys()) == {"precision", "recall", "f1"}
            for key in ("precision", "recall", "f1"):
                assert isinstance(sk[key], float)
                assert 0.0 <= sk[key] <= 1.0


def test_forecast_comparison_runs() -> None:
    """Returns a list of (n_samples, zero_data_rmse, sklearn_rmse) tuples.

    sklearn_rmse is None at N=0 (and when too few points to fit), else a float.
    """
    rows = bench.forecast_comparison()

    assert isinstance(rows, list)
    assert len(rows) == len(bench.FORECAST_N_VALUES)

    for idx, row in enumerate(rows):
        assert isinstance(row, tuple)
        assert len(row) == 3
        n_samples, zd_rmse, sk_rmse = row
        assert n_samples == bench.FORECAST_N_VALUES[idx]
        assert isinstance(zd_rmse, float)
        assert zd_rmse >= 0.0
        assert np.isfinite(zd_rmse)
        if sk_rmse is not None:
            assert isinstance(sk_rmse, float)
            assert sk_rmse >= 0.0
            assert np.isfinite(sk_rmse)


def test_similarity_comparison_runs() -> None:
    """Returns a finite Pearson correlation in [-1, 1]."""
    corr = bench.similarity_comparison()

    assert isinstance(corr, float)
    assert np.isfinite(corr)
    assert -1.0 <= corr <= 1.0
