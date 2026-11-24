
"""API HTTP para clasificar imágenes de hojas con ResNet50."""

from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.config import (
    ALLOWED_ORIGINS,
    MAX_BATCH_FILES,
    MAX_IMAGE_PIXELS,
    MAX_IMAGE_SIZE_BYTES,
    MAX_IMAGE_SIZE_MB,
    MODEL_NAME,
)
from app.labels import describe_class
from app.predictor import ModelNotReadyError, predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}


class ClassLabel(BaseModel):
    class_name: str
    crop: str
    condition: str
    crop_name: str
    condition_name: str


class TopPrediction(ClassLabel):
    confidence: float = Field(ge=0.0, le=1.0)


class PredictionResponse(TopPrediction):
    model: str
    top_3: list[TopPrediction]


class ModelResponse(BaseModel):
    model: str
    input_size: str
    max_files: int
    max_image_size_mb: int
    max_image_pixels: int
    image_formats: list[str]
    classes: list[ClassLabel]


class BatchPrediction(BaseModel):
    index: int
    filename: str | None
    status: Literal["ok"] = "ok"
    prediction: PredictionResponse


class BatchError(BaseModel):
    index: int
    filename: str | None
    status: Literal["error"] = "error"
    status_code: int
    detail: str


class BatchResponse(BaseModel):
    model: str
    results: list[Annotated[BatchPrediction | BatchError, Field(discriminator="status")]]


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        predictor.load()
    except ModelNotReadyError as exc:
        logger.warning("Modelo aún no disponible: %s", exc)
    yield


app = FastAPI(
    title="ProyectoIA - API de clasificación foliar",
    description=(
        "API para clasificar imágenes de hojas de manzana, maíz, "
        "pimiento, papa y tomate mediante ResNet50."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS: utiliza los orígenes configurados en config.py y .env
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
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


@app.get("/model", response_model=ModelResponse, tags=["General"])
def model_info() -> ModelResponse:
    try:
        class_names = predictor.class_names()
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ModelResponse(
        model=MODEL_NAME,
        input_size=predictor.status()["input_size"],
        max_files=MAX_BATCH_FILES,
        max_image_size_mb=MAX_IMAGE_SIZE_MB,
        max_image_pixels=MAX_IMAGE_PIXELS,
        image_formats=sorted(ALLOWED_IMAGE_FORMATS),
        classes=[ClassLabel(**describe_class(name)) for name in class_names],
    )


async def classify_upload(file: UploadFile) -> PredictionResponse:
    try:
        content = await file.read(MAX_IMAGE_SIZE_BYTES + 1)
    finally:
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
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="La imagen supera los 25 megapíxeles.")
            image.verify()
    except Image.DecompressionBombError as exc:
        raise HTTPException(status_code=413, detail="La imagen supera los 25 megapíxeles.") from exc
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
        result = await run_in_threadpool(predictor.predict, content, image_format)

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


@app.post("/predict", response_model=PredictionResponse, tags=["Predicción"])
async def predict_image(
    file: Annotated[UploadFile, File(description="Imagen de una hoja: JPEG, PNG o WEBP")],
) -> PredictionResponse:
    return await classify_upload(file)


@app.post("/predict/batch", response_model=BatchResponse, tags=["Predicción"])
async def predict_batch(
    files: Annotated[list[UploadFile], File(description="Repite el campo files por imagen")],
) -> BatchResponse:
    if not 1 <= len(files) <= MAX_BATCH_FILES:
        raise HTTPException(status_code=422, detail=f"Envía entre 1 y {MAX_BATCH_FILES} imágenes.")

    results: list[BatchPrediction | BatchError] = []
    for index, file in enumerate(files):
        try:
            prediction = await classify_upload(file)
        except HTTPException as exc:
            if exc.status_code == 503:
                raise
            results.append(BatchError(
                index=index, filename=file.filename, status_code=exc.status_code, detail=exc.detail
            ))
        else:
            results.append(BatchPrediction(index=index, filename=file.filename, prediction=prediction))
    return BatchResponse(model=MODEL_NAME, results=results)
