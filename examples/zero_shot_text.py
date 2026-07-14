# examples/zero_shot_text.py
"""Zero-shot NLP demonstration for the zero-data cognitive model.

Runs three demonstrations with NO training data and NO external datasets:
  1. Zero-shot classification of 8 sentences across 4 topics.
  2. Semantic similarity ranking of a query against 5 candidates.
  3. Zero-data text generation from a seed.

Run with:  python examples/zero_shot_text.py
"""

from __future__ import annotations

from zero_data_model.model import ZeroDataModel


def demo_classification(model: ZeroDataModel) -> None:
    print("=" * 64)
    print("Demo 1: Zero-Shot Text Classification (no training data)")
    print("=" * 64)
    sentences = [
        ("the algorithm computes the network", "tech"),
        ("code data model system", "tech"),
        ("trees rivers mountains forests", "nature"),
        ("sky ocean flower tree", "nature"),
        ("love joy happiness hope", "emotion"),
        ("sad fear anger happy", "emotion"),
        ("energy force mass quantum field", "science"),
        ("light force mass atom", "science"),
    ]
    correct = 0
    for text, expected in sentences:
        topic, conf = model.classify_text(text)
        flag = "OK" if topic == expected else "??"
        if topic == expected:
            correct += 1
        print(f"  [{flag}] '{text}'")
        print(f"        -> topic={topic} (expected={expected})  conf={conf:.3f}")
    print(f"\n  Accuracy: {correct}/{len(sentences)}")


def demo_similarity(model: ZeroDataModel) -> None:
    print("\n" + "=" * 64)
    print("Demo 2: Semantic Similarity Ranking  (query: 'code data model')")
    print("=" * 64)
    query = "code data model"
    candidates = [
        "compute data code",
        "algorithm network system",
        "the model computes data",
        "love joy hope",
        "energy quantum field",
    ]
    ranked = sorted(
        ((model.text_similarity(query, c), c) for c in candidates),
        reverse=True,
    )
    for rank, (sim, text) in enumerate(ranked, 1):
        print(f"  {rank}. sim={sim:.4f}  '{text}'")


def demo_generation(model: ZeroDataModel) -> None:
    print("\n" + "=" * 64)
    print("Demo 3: Zero-Data Text Generation (from a seed)")
    print("=" * 64)
    for seed in ["cognition", "quantum", "forest"]:
        generated = model.generate_text(seed, length=40)
        print(f"  seed='{seed}'")
        print(f"    -> '{generated}'")


def main() -> None:
    model = ZeroDataModel(dim=64)
    demo_classification(model)
    demo_similarity(model)
    demo_generation(model)
    print("\nHardware backends:", model.hardware_info)


if __name__ == "__main__":
    main()
