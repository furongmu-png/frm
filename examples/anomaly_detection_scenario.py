# examples/anomaly_detection_scenario.py
"""Real-world-style anomaly detection scenario for the zero-data model.

Simulates server CPU metrics (normal operation + a spike + a level shift) and
runs the analytics stack: anomaly detection, trend analysis and forecasting.
No external datasets are used.

Run with:  python examples/anomaly_detection_scenario.py
"""

from __future__ import annotations

import numpy as np

from zero_data_model.model import ZeroDataModel


def main() -> None:
    model = ZeroDataModel(dim=64)
    rng = np.random.default_rng(42)

    print("=" * 64)
    print("Anomaly Detection Scenario: Simulated Server CPU Metrics")
    print("=" * 64)

    # 60 minutes of CPU load: ~40% baseline, a spike, then a level shift to ~70%.
    # The shifted region is kept a minority of the window so it shows up as a
    # statistical outlier (mirrors how a real ops team would notice a sustained
    # change after a stable baseline).
    normal = 40.0 + rng.normal(0.0, 2.0, 50)   # 50 min baseline ~40%
    spike = np.array([95.0])                    # sudden spike at minute 50
    shifted = 70.0 + rng.normal(0.0, 2.0, 9)    # 9 min level shift to ~70%
    cpu = np.concatenate([normal, spike, shifted])

    print(f"\nSeries length : {len(cpu)} minutes")
    print(f"Baseline      : ~40% (first {len(normal)} min)")
    print(f"Spike         : {spike[0]:.0f}% at t={len(normal)}")
    print(f"Level shift   : ~70% (last {len(shifted)} min)")

    # Anomaly detection.
    mask = model.detect_anomalies(cpu)
    anomaly_idx = np.where(mask)[0]
    print(f"\n[Anomaly Detection] flagged {int(mask.sum())} point(s): {anomaly_idx.tolist()}")

    # Trend analysis.
    trend = model.analyze_trend(cpu)
    print("\n[Trend Analysis]")
    print(f"  regime            = {trend['regime']}")
    print(f"  trend_slope       = {trend['trend_slope']:.4f}")
    print(f"  curvature         = {trend['curvature']:.4f}")
    print(f"  geodesic_deviation= {trend['geodesic_deviation']:.4f}")
    print(f"  isomorphism_score = {trend['isomorphism_score']:.4f}")

    # Forecast the next 5 minutes from the current (shifted) state.
    forecast = model.forecast(cpu, horizon=5)
    print("\n[Forecast] next 5 minutes of CPU load:")
    print(f"  {np.round(forecast, 2).tolist()}")

    print("\nHardware backends:", model.hardware_info)


if __name__ == "__main__":
    main()
