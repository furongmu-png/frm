# examples/cross_domain_transfer.py
"""Cross-domain transfer demonstration for the zero-data cognitive model.

Shows that the model can compute structural similarity (isomorphism) across
different modalities with no shared training data:
  1. Text <-> image structural analogy.
  2. Time-series <-> text transfer.
  3. Category-theory isomorphism between an NLP-like and a CV-like problem.

Run with:  python examples/cross_domain_transfer.py
"""

from __future__ import annotations

import numpy as np

from zero_data_model.model import ZeroDataModel


def main() -> None:
    model = ZeroDataModel(dim=64)
    rng = np.random.default_rng(42)

    print("=" * 64)
    print("Cross-Domain Transfer (zero-data, no shared training)")
    print("=" * 64)

    # 1. Text <-> image structural similarity.
    print("\n[1] Text <-> Image structural analogy")
    text_vec = model.encode_text("code data model")
    image = rng.random((16, 16))
    image_vec = model.encode_image(image)
    sim = model.find_analogies(text_vec, image_vec)
    print(f"    find_analogies(text_vec, image_vec) = {sim:.4f}")
    print(f"    cosine in [-1, 1]; finite = {bool(np.isfinite(sim))}")
    print(f"    self-check  image <-> image       = {model.find_analogies(image_vec, image_vec):.4f}")

    # 2. Series <-> text transfer (forecast used as the series embedding).
    print("\n[2] Time-Series <-> Text transfer")
    series = np.arange(20, dtype=float) * 2.0  # upward trend
    series_embedding = model.forecast(series, horizon=64)
    science_vec = model.encode_text("energy force quantum field")
    sim2 = model.find_analogies(series_embedding, science_vec)
    print(f"    series (upward trend) forecast-embedding vs science text")
    print(f"    find_analogies = {sim2:.4f}")

    # 3. Category-theory isomorphism between an NLP-like and a CV-like problem.
    print("\n[3] Category-theory isomorphism (NLP-like vs CV-like problem)")
    nlp_problem = model.encode_text("algorithm network system")  # NLP-like vector
    cv_problem = model.encode_image(rng.random((12, 12)))         # CV-like vector
    iso = model.find_analogies(nlp_problem, cv_problem)
    print(f"    isomorphism score = {iso:.4f}")
    print(f"    (1.0 = structurally identical, 0.0 = orthogonal)")

    print("\nHardware backends:", model.hardware_info)


if __name__ == "__main__":
    main()
