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
    print(f"  Source: {knowledge.metadata.get('source', 'internal')}")

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
    print(f"  Solution norm: {np.linalg.norm(solution.data):.4f}")

    print("\n" + "=" * 60)
    print("  Demo complete. No external data was fed to the model.")
    print("=" * 60)


if __name__ == "__main__":
    main()
