"""API HTTP para clasificar imágenes de hojas con ResNet50."""

from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.config import ALLOWED_ORIGINS, MAX_IMAGE_SIZE_BYTES, MAX_IMAGE_SIZE_MB, MODEL_NAME
from app.predictor import ModelNotReadyError, predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}


class TopPrediction(BaseModel):
    class_name: str
    crop: str
    condition: str
    confidence: float = Field(ge=0.0, le=1.0)


class PredictionResponse(TopPrediction):
    model: str
    top_3: list[TopPrediction]


@asynccontextmanager
async def lifespan(_: FastAPI):
    # La API puede iniciar sin pesos para permitir el desarrollo del resto del backend.
    try:
        predictor.load()
    except ModelNotReadyError as exc:
        logger.warning("Modelo aún no disponible: %s", exc)
    yield


app = FastAPI(
    title="ProyectoIA - API de clasificación foliar",
    description=(
        "API para clasificar imágenes de hojas de manzana, maíz, pimiento, "
        "papa y tomate mediante ResNet50."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
def root() -> dict[str, str]:
    return {
        "project": "ProyectoIA",
        "service": "API de clasificación de patologías foliares",
        "model": MODEL_NAME,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["General"])
def health() -> dict:
    model_status = predictor.status()
    return {
        "status": "ok" if model_status["model_loaded"] else "degraded",
        **model_status,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Predicción"])
async def predict_image(
    file: Annotated[UploadFile, File(description="Imagen de una hoja: JPEG, PNG o WEBP")],
) -> PredictionResponse:
    content = await file.read(MAX_IMAGE_SIZE_BYTES + 1)
    await file.close()

    if not content:
        raise HTTPException(status_code=400, detail="El archivo está vacío.")
    if len(content) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"La imagen supera el límite de {MAX_IMAGE_SIZE_MB} MB.",
        )

    try:
        with Image.open(io.BytesIO(content)) as image:
            image_format = image.format
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="El archivo no contiene una imagen válida.",
        ) from exc

    if image_format not in ALLOWED_IMAGE_FORMATS:
        raise HTTPException(
            status_code=415,
            detail="Formato no compatible. Utilice JPEG, PNG o WEBP.",
        )

    try:
        result = await run_in_threadpool(predictor.predict, content)
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Error inesperado en /predict.")
        raise HTTPException(
            status_code=500,
            detail="Ocurrió un error interno al clasificar la imagen.",
        ) from exc

    return PredictionResponse(**result)
