"""FastAPI app serving the champion readmission model.

Run locally (after ``make export``):

    uv run uvicorn readmission_mlops.service.app:app --port 8000

Then open http://127.0.0.1:8000/docs for the interactive API documentation.

Configuration (environment variables):

- ``MODEL_DIR``: folder with ``model.skops`` and ``metadata.json``
  (default ``artifacts/model``).
- ``PREDICTION_LOG``: JSON-lines file for the prediction log (default
  ``logs/predictions.jsonl``); set it to an empty string to disable logging.
"""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from readmission_mlops.config import PROJECT_ROOT
from readmission_mlops.service.model_store import LoadedModel
from readmission_mlops.service.prediction_log import PredictionLog
from readmission_mlops.service.schemas import (
    BatchRequest,
    BatchResponse,
    Encounter,
    Health,
    ModelInfo,
    PredictionResponse,
)

logger = logging.getLogger(__name__)

DESCRIPTION = """
Scores the risk that a patient with diabetes is readmitted within 30 days of
discharge, and explains each score.

**For prioritising follow-up only, never for excluding anyone from care.**
Trained on public US data from 1999-2008 (UCI, CC BY 4.0); performance in other
settings is unknown. Model card:
https://github.com/MulengaSnr/readmission-risk-model/blob/v1.0/docs/MODEL_CARD.md
"""


def create_app(model_dir: Path | None = None, log_path: Path | None = None) -> FastAPI:
    """Build the app. Arguments override the environment (used in tests)."""
    model_dir = model_dir or Path(
        os.environ.get("MODEL_DIR", PROJECT_ROOT / "artifacts" / "model")
    )
    if log_path is None:
        env_log = os.environ.get("PREDICTION_LOG", "logs/predictions.jsonl")
        log_path = Path(env_log) if env_log else None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            app.state.model = LoadedModel.from_dir(model_dir)
            logger.info("Loaded model version %s from %s",
                        app.state.model.version, model_dir)  # fmt: skip
        except (FileNotFoundError, OSError) as exc:
            app.state.model = None
            logger.error("No model loaded from %s: %s", model_dir, exc)
        app.state.log = PredictionLog(log_path)
        yield

    app = FastAPI(
        title="Readmission risk API",
        version="2.0.0",
        description=DESCRIPTION,
        lifespan=lifespan,
    )

    def loaded(request: Request) -> LoadedModel:
        model = request.app.state.model
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        return model

    def score(request: Request, encounters: list[Encounter]) -> tuple[str, list]:
        model = loaded(request)
        records = [e.model_dump(by_alias=True) for e in encounters]
        predictions = model.predict(records)
        request_id = str(uuid.uuid4())
        request.app.state.log.write(request_id, model.version, records, predictions)
        return request_id, predictions

    @app.get("/health", response_model=Health)
    def health(request: Request) -> Health:
        """Liveness and readiness: is a model loaded?"""
        model = request.app.state.model
        if model is None:
            return Health(status="model not loaded", model_version=None)
        return Health(status="ok", model_version=model.version)

    @app.get("/model-info", response_model=ModelInfo)
    def model_info(request: Request) -> ModelInfo:
        """Version, provenance and offline metrics of the serving model."""
        meta = loaded(request).metadata
        return ModelInfo(**{**meta, "model_version": str(meta["model_version"])})

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: Request, encounter: Encounter) -> PredictionResponse:
        """Risk, risk band and top-3 reasons for one discharge."""
        request_id, predictions = score(request, [encounter])
        return PredictionResponse(
            **predictions[0],
            request_id=request_id,
            model_version=request.app.state.model.version,
        )

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(request: Request, batch: BatchRequest) -> BatchResponse:
        """Score up to 1,000 discharges in one call (used for replay)."""
        request_id, predictions = score(request, batch.encounters)
        return BatchResponse(
            request_id=request_id,
            model_version=request.app.state.model.version,
            predictions=predictions,
        )

    return app


app = create_app()
