# demo.py
"""Demo: Zero-Data Model in action — no external data required."""

import sys
sys.path.insert(0, "src")

import numpy as np
from zero_data_model.model import ZeroDataModel


def main():
    print("=" * 60)
    print("  Zero-Data Model Demo")
    print("  A self-sufficient cognitive system")
    print("=" * 60)

    model = ZeroDataModel(dim=32)

    print("\n[1] Self-Generated Thought (no input data)")
    for i in range(3):
        result = model.think()
        print(f"  Cycle {result.metadata['cycle']}: "
              f"output norm={np.linalg.norm(result.data):.4f}, "
              f"confidence={result.metadata['self_reflection']['self_confidence']:.4f}")

    print("\n[2] Processing External Signal")
    external_input = np.sin(np.linspace(0, 2 * np.pi, 32))
    result = model.think(external_input)
    print(f"  Input norm: {np.linalg.norm(external_input):.4f}")
    print(f"  Output norm: {np.linalg.norm(result.data):.4f}")

    print("\n[3] Self-Generated Knowledge")
    knowledge = model.generate_knowledge()
    print(f"  Generated knowledge vector norm: {np.linalg.norm(knowledge.data):.4f}")

    print("\n[4] Cross-Domain Analogy Detection")
    problem_a = np.random.randn(32)
    problem_b = problem_a + np.random.randn(32) * 0.1
    problem_c = np.random.randn(32)
    sim_ab = model.find_analogies(problem_a, problem_b)
    sim_ac = model.find_analogies(problem_a, problem_c)
    print(f"  Similarity(A, B) = {sim_ab:.4f}  (should be high)")
    print(f"  Similarity(A, C) = {sim_ac:.4f}  (should be lower)")

    print("\n[5] Optimization via Quantum Annealing")
    solution = model.solve(np.random.randn(32))
    print(f"  Solution energy: {solution.metadata['energy']:.4f}")

    # --- Domain Capabilities (NLP / CV / Analytics) ---

    print("\n" + "-" * 60)
    print("  Domain Capabilities")
    print("-" * 60)

    print("\n[6] NLP: Text Encoding & Semantic Similarity")
    t1, t2, t3 = "code data model algorithm", "code data model algorithm", "tree river mountain ocean"
    sim_same = model.text_similarity(t1, t2)
    sim_diff = model.text_similarity(t1, t3)
    print(f"  similarity('{t1}', '{t2}') = {sim_same:.4f}  (identical)")
    print(f"  similarity('{t1}', '{t3}') = {sim_diff:.4f}  (different topic)")

    print("\n[7] NLP: Zero-Shot Text Classification")
    samples = [
        "the algorithm computes the network",
        "trees rivers mountains forests",
        "love joy happiness hope",
        "energy force mass quantum field",
    ]
    for s in samples:
        topic, conf = model.classify_text(s)
        print(f"  '{s}' -> topic='{topic}', confidence={conf:.4f}")

    print("\n[8] NLP: Self-Generated Text")
    generated = model.generate_text("seed", length=48)
    print(f"  Generated (48 chars): {generated!r}")

    print("\n[9] CV: Image Encoding")
    rng = np.random.default_rng(42)
    circle_img = np.zeros((16, 16))
    yy, xx = np.indices((16, 16))
    circle_img[(yy - 8) ** 2 + (xx - 8) ** 2 <= 25] = 1.0
    enc = model.encode_image(circle_img)
    print(f"  Circle image encoded: shape={enc.shape}, norm={np.linalg.norm(enc):.4f}")

    print("\n[10] CV: Feature Extraction")
    feats = model.extract_image_features(circle_img)
    print(f"  Edges:      shape={feats['edges'].shape}, magnitude_sum={np.sum(feats['edges']):.4f}")
    print(f"  Texture:    shape={feats['texture'].shape}")
    print(f"  Morphology:  shape={feats['morphology'].shape}")
    print(f"  Stats:      {feats['stats']}")

    print("\n[11] CV: Pattern Recognition (zero-shot)")
    shape, conf = model.recognize_pattern(circle_img)
    print(f"  Recognized shape: '{shape}', confidence={conf:.4f}")
    square_img = np.zeros((16, 16))
    square_img[4:12, 4:12] = 1.0
    shape2, conf2 = model.recognize_pattern(square_img)
    print(f"  Square recognized as: '{shape2}', confidence={conf2:.4f}")

    print("\n[12] CV: Shape Analysis")
    analysis = model.analyze_shape(circle_img)
    print(f"  Aspect ratio: {analysis['aspect_ratio']:.4f}")
    print(f"  Symmetry:     {analysis['symmetry']:.4f}")
    print(f"  Complexity:    {analysis['complexity']:.4f}")
    print(f"  Betti-0:       {analysis['beti0']:.0f}")

    print("\n[13] Analytics: Time-Series Forecasting")
    series = np.arange(20, dtype=float) + rng.standard_normal(20) * 0.5
    forecast = model.forecast(series, horizon=5)
    print(f"  Series (last 5): {np.round(series[-5:], 2)}")
    print(f"  Forecast (5):    {np.round(forecast, 2)}")
    print(f"  Series mean: {np.mean(series):.4f}, forecast mean: {np.mean(forecast):.4f}")

    print("\n[14] Analytics: Anomaly Detection")
    anomaly_series = np.array([1, 1, 1, 1, 1, 100, 1, 1, 1, 1], dtype=float)
    anomalies = model.detect_anomalies(anomaly_series)
    print(f"  Series:    {anomaly_series}")
    print(f"  Anomalies: {anomalies}")
    print(f"  Detected {int(anomalies.sum())} anomaly point(s) at index {np.where(anomalies)[0].tolist()}")

    print("\n[15] Analytics: Pattern Mining")
    mined = model.mine_patterns(series)
    print(f"  Self-similarity:   {mined['self_similarity']:.4f}")
    print(f"  Best CA rule:      {mined['automaton_rule']}")
    print(f"  Periodicity lag:   {mined['periodicity']}")
    print(f"  Topology features: {np.round(mined['topology'][:5], 4)}")

    print("\n[16] Analytics: Trend Analysis")
    up_series = np.arange(50, dtype=float)
    trend = model.analyze_trend(up_series)
    print(f"  Up-trend series -> regime='{trend['regime']}'")
    print(f"  Slope={trend['trend_slope']:.4f}, Curvature={trend['curvature']:.4f}")
    print(f"  Geodesic deviation={trend['geodesic_deviation']:.4f}, Isomorphism={trend['isomorphism_score']:.4f}")

    print("\n" + "=" * 60)
    print("  Demo complete. No external data was fed to the model.")
    print("  All NLP/CV/Analytics capabilities used zero-data priors")
    print("  composed with the 6 self-generating core modules.")
    print("=" * 60)


if __name__ == "__main__":
    main()
