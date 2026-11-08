"""Entrena ResNet50 de forma independiente al notebook original.

Ejemplo:
    python scripts/train_resnet.py --dataset-dir "D:/datos/plantvillage dataset"

La carpeta indicada debe contener la subcarpeta `color` del dataset PlantVillage.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

SEED = 123
IMG_SIZE = 224
BATCH_SIZE = 32
MAX_PER_CLASS = 100
EPOCHS = 10
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
CLASS_PREFIXES = (
    "Apple___",
    "Corn_(maize)___",
    "Pepper,_bell___",
    "Potato___",
    "Tomato___",
)
AUTOTUNE = tf.data.AUTOTUNE


def get_class_names(color_dir: Path) -> list[str]:
    if not color_dir.is_dir():
        raise FileNotFoundError(f"No se encontró la carpeta de imágenes: {color_dir}")

    return sorted(
        name
        for name in os.listdir(color_dir)
        if (color_dir / name).is_dir()
        and name.startswith(CLASS_PREFIXES)
    )


def collect_images(color_dir: Path, class_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    file_paths: list[str] = []
    labels: list[int] = []
    rng = np.random.default_rng(SEED)

    for label_id, class_name in enumerate(class_names):
        class_dir = color_dir / class_name
        images = [
            str(class_dir / filename)
            for filename in os.listdir(class_dir)
            if filename.lower().endswith(IMAGE_EXTENSIONS)
        ]
        rng.shuffle(images)
        images = images[:MAX_PER_CLASS]

        if len(images) < 3:
            raise ValueError(
                f"La clase {class_name} tiene menos de 3 imágenes válidas; "
                "no se puede realizar una división estratificada."
            )

        file_paths.extend(images)
        labels.extend([label_id] * len(images))

    if not class_names:
        raise ValueError("No se encontraron clases de los cinco cultivos seleccionados.")

    return np.asarray(file_paths), np.asarray(labels, dtype=np.int32)


def split_dataset(
    file_paths: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        file_paths,
        labels,
        test_size=0.30,
        random_state=SEED,
        stratify=labels,
    )

    # Se conserva el reparto final del notebook: 230 test y 460 validation
    # cuando hay 2,300 imágenes (100 por cada una de las 23 clases).
    test_paths, val_paths, test_labels, val_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=2 / 3,
        random_state=SEED,
        stratify=temp_labels,
    )

    return (
        train_paths,
        train_labels,
        val_paths,
        val_labels,
        test_paths,
        test_labels,
    )


def load_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    image.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return image, label


def make_dataset(
    paths: np.ndarray,
    labels: np.ndarray,
    training: bool = False,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=SEED,
        )
    dataset = dataset.map(load_image, num_parallel_calls=AUTOTUNE)
    return dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)


def build_resnet50(num_classes: int) -> keras.Model:
    inputs = keras.Input(
        shape=(IMG_SIZE, IMG_SIZE, 3),
        name="imagen",
    )

    # Mismo flujo del Colab: datos [0,1] -> Rescaling(255) -> preprocess_input.
    x = layers.Rescaling(255.0)(inputs)
    base = keras.applications.ResNet50(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    x = keras.applications.resnet50.preprocess_input(x)
    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="clasificacion",
    )(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="ResNet")


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenar ResNet50 para ProyectoIA.")
    parser.add_argument(
        "--dataset-dir",
        required=True,
        type=Path,
        help="Carpeta que contiene la subcarpeta 'color' de PlantVillage.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts",
        help="Carpeta de salida de modelo, clases y métricas.",
    )
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(SEED)
    color_dir = args.dataset_dir.expanduser().resolve() / "color"
    artifacts_dir = args.artifacts_dir.expanduser().resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    class_names = get_class_names(color_dir)
    if len(class_names) != 23:
        raise ValueError(
            f"Se esperaban 23 clases y se encontraron {len(class_names)}. "
            "Verifique que el dataset corresponda al utilizado por el proyecto."
        )

    file_paths, labels = collect_images(color_dir, class_names)
    (
        train_paths,
        train_labels,
        val_paths,
        val_labels,
        test_paths,
        test_labels,
    ) = split_dataset(file_paths, labels)

    print(f"Clases: {len(class_names)} | Imágenes: {len(file_paths)}")
    print(
        "Training: %d (%.1f%%) | Validation: %d (%.1f%%) | Test: %d (%.1f%%)"
        % (
            len(train_paths), 100 * len(train_paths) / len(file_paths),
            len(val_paths), 100 * len(val_paths) / len(file_paths),
            len(test_paths), 100 * len(test_paths) / len(file_paths),
        )
    )

    train_ds = make_dataset(train_paths, train_labels, training=True)
    val_ds = make_dataset(val_paths, val_labels)
    test_ds = make_dataset(test_paths, test_labels)

    model = build_resnet50(num_classes=len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
    )

    probabilities = model.predict(test_ds, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    metrics: dict[str, Any] = {
        "model": "ResNet50",
        "num_classes": len(class_names),
        "test_images": int(len(test_labels)),
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "precision_macro": float(
            precision_score(test_labels, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(test_labels, predictions, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(test_labels, predictions, average="macro", zero_division=0)
        ),
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "image_size": IMG_SIZE,
    }

    model_path = artifacts_dir / "resnet50.keras"
    class_names_path = artifacts_dir / "class_names.json"
    metrics_path = artifacts_dir / "metrics.json"

    model.save(model_path)
    class_names_path.write_text(
        json.dumps(class_names, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nEvaluación en el conjunto de prueba:")
    for name in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
        print(f"{name}: {metrics[name] * 100:.2f}%")
    print(f"\nModelo guardado en: {model_path}")
    print(f"Clases guardadas en: {class_names_path}")
    print(f"Métricas guardadas en: {metrics_path}")


if __name__ == "__main__":
    main()
