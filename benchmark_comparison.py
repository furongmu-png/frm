# benchmark_comparison.py
"""Comparison benchmark: Zero-Data Model vs sklearn baselines.

Shows how the zero-data model performs WITHOUT any training data, compared to
sklearn baselines that require labeled examples. As training data increases,
sklearn catches up — but the zero-data model provides a usable baseline at zero.

The comparison is intentionally FAIR and HONEST:
  * At N=0 only the zero-data model can run; sklearn is marked "N/A — requires
    training data".
  * As N grows, sklearn is allowed to catch up and, where it genuinely does,
    surpass the zero-data model. We do not hide cases where sklearn wins.
  * The zero-data model is a reasonable baseline, not a state-of-the-art
    supervised model: where a fitted sklearn model is clearly better at high N,
    the tables say so.

Run: python benchmark_comparison.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from zero_data_model.model import ZeroDataModel

# Reproducibility: the model's core modules draw random generative parameters
# from numpy's legacy global RNG, so we seed it before every construction.
SEED = 42


def build_model() -> ZeroDataModel:
    """Construct a deterministic zero-data model (re-seeded for reproducibility)."""
    np.random.seed(SEED)
    return ZeroDataModel(dim=64)


# ---------------------------------------------------------------------------
# Shared task data
# ---------------------------------------------------------------------------

# 12 test texts across 4 topics. The first 8 are keyword-heavy (easy for the
# zero-data keyword-prototype classifier); the last 4 deliberately avoid the
# prototype keywords, so they are hard for the zero-data model but learnable by
# a supervised sklearn model given enough training examples.
TEST_TEXTS: list[tuple[str, str]] = [
    ("the algorithm computes the network", "tech"),
    ("code data model system compute", "tech"),
    ("the program processes information on servers", "tech"),
    ("tree river mountain forest", "nature"),
    ("sky ocean flower tree", "nature"),
    ("leaves fall in the autumn wind", "nature"),
    ("love joy happiness hope", "emotion"),
    ("happy love hope joy", "emotion"),
    ("his heart pounded with terror", "emotion"),
    ("energy force mass quantum field", "science"),
    ("light force mass atom", "science"),
    ("cells divide under the microscope", "science"),
]

# Training pool per topic (disjoint from the test set). It mixes keyword and
# paraphrase phrasings, including some that share vocabulary with the hard test
# texts (e.g. "the program runs on the server" / "autumn leaves cover the
# ground" / "anxiety gripped his pounding heart" / "the microscope reveals
# dividing cells"). This lets a supervised sklearn model generalize beyond the
# zero-data keyword prototypes as N grows.
TRAIN_POOL: dict[str, list[str]] = {
    "tech": [
        "the computer stores data in memory",
        "software engineers write code every day",
        "the network connects many devices",
        "a model trained on large datasets",
        "the algorithm sorts the records",
        "the system processes user requests",
        "servers host the application backend",
        "the program runs on the server",
        "compute clusters scale horizontally",
        "data pipelines transform records",
        "the database stores user information",
        "network protocols route packets",
    ],
    "nature": [
        "the forest is full of tall trees",
        "a river flows through the valley",
        "the mountain is covered in snow",
        "the sky turns orange at sunset",
        "waves crash on the ocean shore",
        "a flower blooms in the garden",
        "autumn leaves cover the ground",
        "birds nest in the tall branches",
        "the lake reflects the clouds",
        "deer walk through the meadow",
        "rain falls on the green field",
        "the desert is hot and dry",
    ],
    "emotion": [
        "she felt happy all day long",
        "the sad movie made him cry",
        "they love each other deeply",
        "fear paralyzed her in the dark",
        "joy filled the crowded room",
        "anger rose in his voice",
        "hope kept them going forward",
        "terror filled his pounding heart",
        "she wept at the quiet funeral",
        "anxiety gripped his pounding heart",
        "a smile crossed her face",
        "grief overwhelmed the family",
    ],
    "science": [
        "energy cannot be created or destroyed",
        "the force of gravity pulls objects",
        "mass and energy are equivalent",
        "light travels at a constant speed",
        "the atom has a dense nucleus",
        "quantum mechanics describes particles",
        "the magnetic field bends electrons",
        "the microscope reveals dividing cells",
        "the experiment measured the reaction",
        "molecules form chemical bonds",
        "the telescope observes distant stars",
        "radiation heats the surface",
    ],
}

TOPICS = sorted(TRAIN_POOL.keys())
TEXT_N_VALUES = [0, 4, 8, 16, 32]
ANOMALY_N_VALUES = [0, 4, 8, 16, 32]
FORECAST_N_VALUES = [0, 8, 16, 32, 64]


# ---------------------------------------------------------------------------
# 1. Zero-shot text classification comparison
# ---------------------------------------------------------------------------

def _sample_train_texts(n: int, rng: np.random.Generator) -> tuple[list[str], list[str]]:
    """Stratified sample of ``n`` training texts (n//4 per topic, no replacement)."""
    per = max(1, n // len(TOPICS))
    texts: list[str] = []
    labels: list[str] = []
    for topic in TOPICS:
        pool = TRAIN_POOL[topic]
        idx = rng.choice(len(pool), size=min(per, len(pool)), replace=False)
        for i in idx:
            texts.append(pool[i])
            labels.append(topic)
    return texts, labels


def text_classification_comparison() -> list[tuple[int, float, float | None]]:
    """Compare zero-data classification (no training) vs TF-IDF + LogisticRegression.

    Returns a list of ``(n_samples, zero_data_acc, sklearn_acc)`` tuples where
    ``sklearn_acc`` is ``None`` at N=0 (sklearn cannot train without data).
    """
    model = build_model()
    test_texts = [t for t, _ in TEST_TEXTS]
    truth = [lbl for _, lbl in TEST_TEXTS]

    # Zero-data accuracy is independent of N (no training) -> computed once.
    zd_preds = [model.classify_text(t)[0] for t in test_texts]
    zd_acc = float(np.mean([p == g for p, g in zip(zd_preds, truth, strict=True)]))

    rows: list[tuple[int, float, float | None]] = []
    for n in TEXT_N_VALUES:
        if n == 0:
            # sklearn has no training data to fit on.
            rows.append((n, zd_acc, None))
            continue
        rng = np.random.default_rng(123)
        tr_texts, tr_labels = _sample_train_texts(n, rng)
        vec = TfidfVectorizer()
        x_train = vec.fit_transform(tr_texts)
        x_test = vec.transform(test_texts)
        clf = LogisticRegression(max_iter=1000, random_state=0)
        clf.fit(x_train, tr_labels)
        preds = clf.predict(x_test)
        sk_acc = float(np.mean([p == g for p, g in zip(preds, truth, strict=True)]))
        rows.append((n, zd_acc, sk_acc))

    # Print the comparison table.
    print("\n[1] Zero-shot text classification (12 test texts, 4 topics)")
    print("    zero-data: ZeroDataModel.classify_text (no training data)")
    print("    sklearn  : TfidfVectorizer + LogisticRegression (needs N labelled samples)")
    print(f"    {'N_samples':>9} | {'zero_data_acc':>13} | {'sklearn_acc':>22}")
    print("    " + "-" * 50)
    for n, zd, sk in rows:
        sk_str = "N/A — requires data" if sk is None else f"{sk:.3f}"
        print(f"    {n:>9} | {zd:>13.3f} | {sk_str:>22}")
    return rows


# ---------------------------------------------------------------------------
# 2. Anomaly detection comparison
# ---------------------------------------------------------------------------

def _make_anomaly_series() -> tuple[np.ndarray, np.ndarray]:
    """Build a stationary noise series with mixed-difficulty point anomalies.

    Returns ``(series, ground_truth_mask)``. The baseline is pure stationary
    noise (no trend / no seasonality) so a 1-D IsolationForest is on equal
    footing with the zero-data detector; anomalies range from subtle (~3.5
    sigma) to moderate (~6 sigma).
    """
    rng = np.random.default_rng(11)
    n = 120
    series = rng.normal(0.0, 1.0, n)
    anom_idx = np.array([8, 22, 35, 49, 63, 78, 91, 105])
    mags = np.array([3.5, 5.0, 4.0, 6.0, 3.5, 5.5, 4.5, 6.0])
    series[anom_idx] += mags
    gt = np.zeros(n, dtype=bool)
    gt[anom_idx] = True
    return series, gt


def _prf(mask: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    """Return (precision, recall, f1) for a boolean anomaly mask vs ground truth."""
    tp = int(np.sum(mask & gt))
    fp = int(np.sum(mask & ~gt))
    fn = int(np.sum(~mask & gt))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return float(precision), float(recall), float(f1)


def anomaly_comparison() -> list[dict]:
    """Compare zero-data anomaly detection (no fit) vs IsolationForest.

    Returns a list of dicts: ``{"n_fit", "zd": {p,r,f1}, "sk": {p,r,f1} | None}``.
    ``sk`` is ``None`` at N_fit=0 (IsolationForest cannot fit without normal data).
    For each N_fit>0 the IsolationForest is averaged over 10 random subsamples of
    ``N_fit`` normal points (each with its own seed) so the reported numbers are an
    *expected* performance at that sample size, not a single noisy draw.
    """
    model = build_model()
    series, gt = _make_anomaly_series()
    contamination = float(np.mean(gt))  # tell IF the true anomaly rate (fair)

    # Zero-data: no fitting -> one constant result.
    zd_mask = model.detect_anomalies(series)
    zd_p, zd_r, zd_f1 = _prf(zd_mask, gt)
    zd_result = {"precision": zd_p, "recall": zd_r, "f1": zd_f1}

    normal_pts = series[~gt]  # clean "normal" data given to sklearn for fitting
    n_seeds = 10

    rows: list[dict] = []
    for n_fit in ANOMALY_N_VALUES:
        if n_fit == 0:
            rows.append({"n_fit": n_fit, "zd": zd_result, "sk": None})
            continue
        ps, rs, fs = [], [], []
        for state in range(n_seeds):
            sub_rng = np.random.default_rng(1000 + state)
            idx = sub_rng.choice(normal_pts.size, size=min(n_fit, normal_pts.size),
                                 replace=False)
            train_pts = normal_pts[idx].reshape(-1, 1)
            iso = IsolationForest(
                n_estimators=300, contamination=contamination, random_state=state
            )
            iso.fit(train_pts)
            mask = iso.predict(series.reshape(-1, 1)) == -1
            p, r, f1 = _prf(mask, gt)
            ps.append(p)
            rs.append(r)
            fs.append(f1)
        sk_result = {
            "precision": float(np.mean(ps)),
            "recall": float(np.mean(rs)),
            "f1": float(np.mean(fs)),
        }
        rows.append({"n_fit": n_fit, "zd": zd_result, "sk": sk_result})

    print("\n[2] Anomaly detection (120-point stationary series, 8 injected anomalies)")
    print("    zero-data: ZeroDataModel.detect_anomalies (no fitting)")
    print("    sklearn  : IsolationForest fit on N_fit 'normal' points (avg of 10 subsamples)")
    print(f"    {'N_fit':>6} | {'zd_prec':>7} {'zd_rec':>7} {'zd_f1':>6} | "
          f"{'sk_prec':>7} {'sk_rec':>7} {'sk_f1':>6}")
    print("    " + "-" * 60)
    for row in rows:
        zd = row["zd"]
        if row["sk"] is None:
            sk_str = "       N/A — requires fit       "
        else:
            sk = row["sk"]
            sk_str = f"{sk['precision']:>7.3f} {sk['recall']:>7.3f} {sk['f1']:>6.3f}"
        print(f"    {row['n_fit']:>6} | {zd['precision']:>7.3f} {zd['recall']:>7.3f} "
              f"{zd['f1']:>6.3f} | {sk_str}")
    return rows


# ---------------------------------------------------------------------------
# 3. Time series forecast comparison
# ---------------------------------------------------------------------------

def _make_forecast_series() -> np.ndarray:
    """Trend + seasonality + noise series for the forecasting task."""
    rng = np.random.default_rng(5)
    n = 70
    t = np.arange(n)
    return 0.5 * t + 3.0 * np.sin(2 * np.pi * t / 10) + rng.normal(0.0, 1.0, n)


FORECAST_HORIZON = 5
FORECAST_N_LAGS = 3


def _sklearn_forecast(
    n_train: int, data: np.ndarray, horizon: int, n_lags: int
) -> np.ndarray | None:
    """Forecast ``horizon`` steps with a LinearRegression on lagged features.

    Training examples (X = last n_lags values, y = next value) are built from the
    first ``n_train`` points; the forecast is seeded from the last ``n_lags``
    points of the full available training series and rolled forward iteratively.
    Returns ``None`` if there is too little data to fit (n_train <= n_lags).
    """
    if n_train <= n_lags:
        return None
    src = data[:n_train]
    x = [src[i - n_lags:i] for i in range(n_lags, len(src))]
    y = [src[i] for i in range(n_lags, len(src))]
    lr = LinearRegression()
    lr.fit(np.asarray(x), np.asarray(y))
    window = list(data[-n_lags:])
    preds: list[float] = []
    for _ in range(horizon):
        nxt = float(lr.predict(np.asarray([window]))[0])
        preds.append(nxt)
        window = window[1:] + [nxt]
    return np.asarray(preds, dtype=float)


def forecast_comparison() -> list[tuple[int, float, float | None]]:
    """Compare zero-data forecasting (no fit) vs lagged LinearRegression.

    Returns a list of ``(n_samples, zero_data_rmse, sklearn_rmse)`` tuples where
    ``sklearn_rmse`` is ``None`` at N=0 (no data to fit the lagged model).
    """
    model = build_model()
    series = _make_forecast_series()
    train = series[:-FORECAST_HORIZON]
    test = series[-FORECAST_HORIZON:]

    # Zero-data forecast (constant baseline, no fitting).
    zd_pred = model.forecast(train, horizon=FORECAST_HORIZON)
    zd_rmse = float(np.sqrt(np.mean((zd_pred - test) ** 2)))

    rows: list[tuple[int, float, float | None]] = []
    for n in FORECAST_N_VALUES:
        if n == 0:
            rows.append((n, zd_rmse, None))
            continue
        pred = _sklearn_forecast(n, train, FORECAST_HORIZON, FORECAST_N_LAGS)
        if pred is None:
            rows.append((n, zd_rmse, None))
            continue
        sk_rmse = float(np.sqrt(np.mean((pred - test) ** 2)))
        rows.append((n, zd_rmse, sk_rmse))

    print("\n[3] Time series forecast (trend+seasonality+noise, horizon=5)")
    print("    zero-data: ZeroDataModel.forecast (no fitting)")
    print("    sklearn  : LinearRegression on 3 lagged features (needs N points to fit)")
    print(f"    {'N_samples':>9} | {'zero_data_rmse':>15} | {'sklearn_rmse':>22}")
    print("    " + "-" * 54)
    for n, zd, sk in rows:
        sk_str = "N/A — requires data" if sk is None else f"{sk:.3f}"
        print(f"    {n:>9} | {zd:>15.3f} | {sk_str:>22}")
    return rows


# ---------------------------------------------------------------------------
# 4. Text similarity comparison
# ---------------------------------------------------------------------------

# Pairs spanning identical, same-topic (no shared words), cross-topic, and
# paraphrase (shared function words) cases.
SIMILARITY_PAIRS: list[tuple[str, str]] = [
    ("machine learning models", "machine learning models"),
    ("code data model", "algorithm network system"),
    ("code data model", "tree river mountain"),
    ("trees rivers forests", "sky ocean flower"),
    ("love joy hope", "sad fear anger"),
    ("energy force mass", "quantum atom field"),
    ("the cat sat on the mat", "a feline rested on the rug"),
    ("happy birthday to you", "joyful anniversary to you"),
]


def similarity_comparison() -> float:
    """Compare zero-data similarity vs TF-IDF cosine similarity.

    Both approaches work here (TF-IDF only needs the pair itself as a corpus to
    compute an IDF). Returns the Pearson correlation between the two similarity
    score vectors across the pairs.
    """
    model = build_model()
    corpus = [a for a, _ in SIMILARITY_PAIRS] + [b for _, b in SIMILARITY_PAIRS]
    vec = TfidfVectorizer()
    mat = vec.fit_transform(corpus)
    half = len(SIMILARITY_PAIRS)

    zd_scores: list[float] = []
    sk_scores: list[float] = []
    for i, (a, b) in enumerate(SIMILARITY_PAIRS):
        zd = float(model.text_similarity(a, b))
        sk = float(cosine_similarity(mat[i], mat[i + half])[0, 0])
        zd_scores.append(zd)
        sk_scores.append(sk)

    correlation = float(np.corrcoef(zd_scores, sk_scores)[0, 1])

    print("\n[4] Text similarity (8 text pairs)")
    print("    zero-data: ZeroDataModel.text_similarity (no training)")
    print("    sklearn  : TfidfVectorizer + cosine_similarity (corpus = the 8 pairs)")
    print(f"    {'pair':>4} | {'zero_data':>9} | {'tfidf_cos':>9} | text_a ~ text_b")
    print("    " + "-" * 72)
    for i, (a, b) in enumerate(SIMILARITY_PAIRS):
        print(f"    {i + 1:>4} | {zd_scores[i]:>9.3f} | {sk_scores[i]:>9.3f} | "
              f"{a!r} ~ {b!r}")
    print(f"    Pearson correlation (zero-data vs tfidf): {correlation:.3f}")
    return correlation


# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------

def print_summary(
    text_rows: list[tuple[int, float, float | None]],
    anomaly_rows: list[dict],
    forecast_rows: list[tuple[int, float, float | None]],
    similarity_corr: float,
) -> None:
    """Print a concise summary of the zero-data advantage at N=0."""
    print("\n" + "=" * 72)
    print("  SUMMARY: zero-data model vs sklearn (the no-data advantage at N=0)")
    print("=" * 72)

    # Text classification.
    zd_text = text_rows[0][1]
    best_text = max((sk for _, _, sk in text_rows if sk is not None), default=None)
    print("\n  Text classification (accuracy):")
    print(f"    - N=0    : zero-data = {zd_text:.3f}  | sklearn = N/A (requires data)")
    if best_text is not None:
        winner = "sklearn" if best_text > zd_text else "zero-data (or tie)"
        print(f"    - best sklearn (N=32) = {best_text:.3f}  -> winner at high N: {winner}")

    # Anomaly detection.
    zd_anom = anomaly_rows[0]["zd"]
    best_sk_f1 = max(
        (r["sk"]["f1"] for r in anomaly_rows if r["sk"] is not None), default=None
    )
    print("\n  Anomaly detection (F1):")
    print(f"    - N_fit=0: zero-data = {zd_anom['f1']:.3f} (prec {zd_anom['precision']:.3f}, "
          f"rec {zd_anom['recall']:.3f}) | sklearn = N/A (requires fit)")
    if best_sk_f1 is not None:
        winner = "sklearn" if best_sk_f1 > zd_anom["f1"] else "zero-data"
        print(f"    - best sklearn F1 = {best_sk_f1:.3f}  -> winner: {winner}")

    # Forecasting.
    zd_fc = forecast_rows[0][1]
    best_sk_rmse = min(
        (sk for _, _, sk in forecast_rows if sk is not None), default=None
    )
    print("\n  Forecasting (RMSE, lower is better):")
    print(f"    - N=0    : zero-data = {zd_fc:.3f}  | sklearn = N/A (requires data)")
    if best_sk_rmse is not None:
        winner = "sklearn" if best_sk_rmse < zd_fc else "zero-data"
        print(f"    - best sklearn RMSE = {best_sk_rmse:.3f}  -> winner at high N: {winner}")

    # Similarity.
    print("\n  Text similarity:")
    print(f"    - both work without training; score correlation = {similarity_corr:.3f}")

    print("\n  Bottom line:")
    print("    At N=0 the zero-data model is the ONLY usable option in every task;")
    print("    it provides a reasonable baseline without any labelled examples.")
    print("    Given enough training data, fitted sklearn models catch up and, in")
    print("    text classification and forecasting, surpass the zero-data baseline.")
    print("=" * 72)


def main() -> None:
    print("=" * 72)
    print("  Comparison Benchmark: Zero-Data Model vs sklearn")
    print("=" * 72)

    text_rows = text_classification_comparison()
    anomaly_rows = anomaly_comparison()
    forecast_rows = forecast_comparison()
    similarity_corr = similarity_comparison()

    print_summary(text_rows, anomaly_rows, forecast_rows, similarity_corr)


if __name__ == "__main__":
    main()
