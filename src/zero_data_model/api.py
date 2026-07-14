# src/zero_data_model/api.py
"""FastAPI REST service exposing the :class:`ZeroDataModel` as HTTP endpoints.

The app holds a single module-level :class:`ZeroDataModel` instance that is
lazily initialized on first use (``dim=32`` for fast startup). Numpy arrays
returned by the model are converted to plain Python lists so they serialize
to JSON cleanly (FastAPI / pydantic cannot serialize numpy directly).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel

from .model import ZeroDataModel
from .persistence import ModelSerializer

# Module-level singleton. Lazily initialized so importing the module is cheap
# and the first request pays the construction cost.
_model: Optional[ZeroDataModel] = None


def get_model() -> ZeroDataModel:
    """Return the lazily-initialized module-level model instance."""
    global _model
    if _model is None:
        _model = ZeroDataModel(dim=32)
    return _model


def set_model(model: ZeroDataModel) -> None:
    """Replace the module-level model (used by /load and tests)."""
    global _model
    _model = model


# --------------------------------------------------------------------------- #
# Request schemas
# --------------------------------------------------------------------------- #


class ThinkRequest(BaseModel):
    input: Optional[list[float]] = None


class ClassifyRequest(BaseModel):
    text: str


class SimilarityRequest(BaseModel):
    a: str
    b: str


class GenerateRequest(BaseModel):
    seed: str
    length: int = 32


class ForecastRequest(BaseModel):
    series: list[float]
    horizon: int = 5


class AnomaliesRequest(BaseModel):
    series: list[float]


class TrendRequest(BaseModel):
    series: list[float]


class RecognizeRequest(BaseModel):
    image: list[list[float]]


class PathRequest(BaseModel):
    path: str


# --------------------------------------------------------------------------- #
# Response schemas
# --------------------------------------------------------------------------- #


class HealthResponse(BaseModel):
    status: str
    hardware_info: dict


class ThinkResponse(BaseModel):
    cycle: int
    output: list[float]
    confidence: float


class ClassifyResponse(BaseModel):
    topic: str
    confidence: float


class SimilarityResponse(BaseModel):
    similarity: float


class GenerateResponse(BaseModel):
    text: str


class ForecastResponse(BaseModel):
    forecast: list[float]


class AnomaliesResponse(BaseModel):
    anomalies: list[bool]


class TrendResponse(BaseModel):
    trend_slope: float
    regime: str
    curvature: float
    geodesic_deviation: float
    isomorphism_score: float


class RecognizeResponse(BaseModel):
    shape: str
    confidence: float


class SaveResponse(BaseModel):
    saved: bool


class LoadResponse(BaseModel):
    loaded: bool


# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #


def create_app() -> FastAPI:
    """Build and return a configured FastAPI application."""
    app = FastAPI(title="ZeroDataModel API", version="0.1.0")

    @app.get("/", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Health check + active hardware backends."""
        model = get_model()
        return HealthResponse(status="ok", hardware_info=model.hardware_info)

    @app.post("/think", response_model=ThinkResponse)
    def think(req: Optional[ThinkRequest] = Body(default=None)) -> ThinkResponse:
        """Run one thought cycle. With no ``input`` the model self-generates."""
        model = get_model()
        body = req or ThinkRequest()
        input_data = (
            np.asarray(body.input, dtype=float) if body.input else None
        )
        signal = model.think(input_data)
        return ThinkResponse(
            cycle=int(signal.metadata.get("cycle", model.cycle_count)),
            output=[float(x) for x in np.asarray(signal.data).flatten().tolist()],
            confidence=float(signal.confidence),
        )

    @app.post("/classify", response_model=ClassifyResponse)
    def classify(req: ClassifyRequest) -> ClassifyResponse:
        """Zero-shot classify ``text`` into a topic."""
        model = get_model()
        topic, conf = model.classify_text(req.text)
        return ClassifyResponse(topic=str(topic), confidence=float(conf))

    @app.post("/similarity", response_model=SimilarityResponse)
    def similarity(req: SimilarityRequest) -> SimilarityResponse:
        """Semantic similarity in [0, 1] between two texts."""
        model = get_model()
        score = model.text_similarity(req.a, req.b)
        return SimilarityResponse(similarity=float(score))

    @app.post("/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest) -> GenerateResponse:
        """Generate ``length`` printable characters from a seed."""
        model = get_model()
        text = model.generate_text(req.seed, length=req.length)
        return GenerateResponse(text=str(text))

    @app.post("/forecast", response_model=ForecastResponse)
    def forecast(req: ForecastRequest) -> ForecastResponse:
        """Forecast ``horizon`` future values of the input series."""
        model = get_model()
        series = np.asarray(req.series, dtype=float).flatten()
        if series.size == 0:
            raise HTTPException(status_code=400, detail="series must be non-empty")
        preds = model.forecast(series, horizon=req.horizon)
        return ForecastResponse(
            forecast=[float(x) for x in np.asarray(preds).flatten().tolist()]
        )

    @app.post("/anomalies", response_model=AnomaliesResponse)
    def anomalies(req: AnomaliesRequest) -> AnomaliesResponse:
        """Flag anomalies in a 1D series (returns a boolean mask)."""
        model = get_model()
        series = np.asarray(req.series, dtype=float).flatten()
        if series.size == 0:
            raise HTTPException(status_code=400, detail="series must be non-empty")
        mask = model.detect_anomalies(series)
        return AnomaliesResponse(anomalies=[bool(x) for x in np.asarray(mask).tolist()])

    @app.post("/trend", response_model=TrendResponse)
    def trend(req: TrendRequest) -> TrendResponse:
        """Analyze trend / regime / curvature / geodesic / isomorphism."""
        model = get_model()
        series = np.asarray(req.series, dtype=float).flatten()
        if series.size == 0:
            raise HTTPException(status_code=400, detail="series must be non-empty")
        out = model.analyze_trend(series)
        return TrendResponse(
            trend_slope=float(out["trend_slope"]),
            regime=str(out["regime"]),
            curvature=float(out["curvature"]),
            geodesic_deviation=float(out["geodesic_deviation"]),
            isomorphism_score=float(out["isomorphism_score"]),
        )

    @app.post("/recognize", response_model=RecognizeResponse)
    def recognize(req: RecognizeRequest) -> RecognizeResponse:
        """Recognize a shape/pattern in a 2D grayscale image (list of rows)."""
        model = get_model()
        if not req.image or not req.image[0]:
            raise HTTPException(status_code=400, detail="image must be non-empty")
        image = np.asarray(req.image, dtype=float)
        if image.ndim != 2:
            raise HTTPException(status_code=400, detail="image must be 2D")
        shape, conf = model.recognize_pattern(image)
        return RecognizeResponse(shape=str(shape), confidence=float(conf))

    @app.post("/save", response_model=SaveResponse)
    def save(req: PathRequest) -> SaveResponse:
        """Persist the current model state to ``path`` on disk."""
        model = get_model()
        try:
            ModelSerializer.save(model, req.path)
        except OSError as exc:
            raise HTTPException(status_code=400, detail=f"save failed: {exc}")
        return SaveResponse(saved=True)

    @app.post("/load", response_model=LoadResponse)
    def load(req: PathRequest) -> LoadResponse:
        """Replace the live model with one loaded from ``path``."""
        try:
            new_model = ModelSerializer.load(req.path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"load failed: {exc}")
        set_model(new_model)
        return LoadResponse(loaded=True)

    return app


# A module-level app instance so ``uvicorn zero_data_model.api:app`` works.
app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(create_app(), host="0.0.0.0", port=8000)
