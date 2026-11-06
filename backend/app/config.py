"""Configuración del backend."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_DIR / ".env")
ARTIFACTS_DIR = Path(
    os.getenv("ARTIFACTS_DIR", str(BACKEND_DIR / "artifacts"))
).expanduser().resolve()
MODEL_PATH = Path(
    os.getenv("MODEL_PATH", str(ARTIFACTS_DIR / "resnet50.keras"))
).expanduser().resolve()
CLASS_NAMES_PATH = Path(
    os.getenv("CLASS_NAMES_PATH", str(ARTIFACTS_DIR / "class_names.json"))
).expanduser().resolve()

try:
    MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
except ValueError as exc:
    raise ValueError("MAX_IMAGE_SIZE_MB debe ser un número entero.") from exc

if MAX_IMAGE_SIZE_MB < 1:
    raise ValueError("MAX_IMAGE_SIZE_MB debe ser como mínimo 1.")

MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]

IMAGE_SIZE = 224
MODEL_NAME = "ResNet50"
