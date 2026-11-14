"""Carga ResNet50 y realiza predicciones sobre imágenes."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

from app.config import CLASS_NAMES_PATH, IMAGE_SIZE, MODEL_NAME, MODEL_PATH

logger = logging.getLogger(__name__)


class ModelNotReadyError(RuntimeError):
    """Se produce cuando todavía no existen los artefactos entrenados."""


class PlantDiseasePredictor:
    def __init__(self, model_path: Path, class_names_path: Path) -> None:
        self.model_path = model_path
        self.class_names_path = class_names_path
        self._model: Any | None = None
        self._tf: Any | None = None
        self._class_names: list[str] = []
        self._lock = threading.RLock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tf is not None

    def status(self) -> dict[str, Any]:
        return {
            "model_name": MODEL_NAME,
            "model_loaded": self.is_loaded,
            "model_file_exists": self.model_path.is_file(),
            "class_names_file_exists": self.class_names_path.is_file(),
            "num_classes": len(self._class_names) if self.is_loaded else None,
            "input_size": f"{IMAGE_SIZE}x{IMAGE_SIZE} RGB",
        }

    def load(self) -> None:
        """Carga una sola vez el modelo y su correspondencia de clases."""
        with self._lock:
            if self.is_loaded:
                return

            if not self.model_path.is_file() or not self.class_names_path.is_file():
                raise ModelNotReadyError(
                    "No se encontraron los artefactos de ResNet50. "
                    "Ejecute scripts/train_resnet.py para generarlos."
                )

            try:
                import tensorflow as tf
            except ImportError as exc:
                raise ModelNotReadyError(
                    "TensorFlow no está instalado. Instale backend/requirements.txt."
                ) from exc

            try:
                with self.class_names_path.open("r", encoding="utf-8") as file:
                    class_names = json.load(file)

                if (
                        not isinstance(class_names, list)
                        or not class_names
                        or not all(isinstance(name, str) for name in class_names)
                ):
                    raise ValueError("class_names.json debe contener una lista de nombres.")

                model = tf.keras.models.load_model(self.model_path, compile=False)
                output_classes = int(model.output_shape[-1])
                if output_classes != len(class_names):
                    raise ValueError(
                        "La cantidad de salidas del modelo no coincide con "
                        "la cantidad de nombres de clases."
                    )
            except Exception as exc:
                logger.exception("No se pudo cargar el modelo de clasificación.")
                raise ModelNotReadyError(
                    "No se pudo cargar el modelo o sus clases. Revise los artefactos."
                ) from exc

            self._tf = tf
            self._model = model
            self._class_names = class_names
            logger.info("%s cargado correctamente (%s clases).", MODEL_NAME, len(class_names))

    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        """Procesa bytes de imagen y devuelve las tres predicciones principales."""
        if not self.is_loaded:
            self.load()

        assert self._tf is not None
        assert self._model is not None

        tf = self._tf
        try:
            # Se normaliza al rango [0, 1] para evitar la saturación de confianza
            image = tf.io.decode_image(
                image_bytes,
                channels=3,
                expand_animations=False,
            )
            image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
            image = tf.cast(image, tf.float32) / 255.0
            image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
            batch = tf.expand_dims(image, axis=0)

            with self._lock:
                scores = self._model(batch, training=False).numpy()[0]
        except Exception as exc:
            logger.exception("Error durante la predicción.")
            raise ValueError("No se pudo procesar la imagen con ResNet50.") from exc

        if scores.ndim != 1 or scores.shape[0] != len(self._class_names):
            raise ValueError("El modelo devolvió una salida con dimensiones inesperadas.")
        if not np.all(np.isfinite(scores)):
            raise ValueError("El modelo devolvió puntuaciones no válidas.")

        top_indices = np.argsort(scores)[::-1][: min(3, len(self._class_names))]
        top_predictions = []
        for index in top_indices:
            class_name = self._class_names[int(index)]
            crop, separator, condition = class_name.partition("___")
            top_predictions.append(
                {
                    "class_name": class_name,
                    "crop": crop,
                    "condition": condition if separator else "",
                    "confidence": float(scores[index]),
                }
            )

        best = top_predictions[0]
        return {
            "model": MODEL_NAME,
            **best,
            "top_3": top_predictions,
        }


predictor = PlantDiseasePredictor(MODEL_PATH, CLASS_NAMES_PATH)