"""Entrena ResNet50 de forma independiente al notebook original.

Ejemplo:
    python scripts/train_resnet.py --dataset-dir "D:/datos/plantvillage dataset"

La carpeta indicada debe contener la subcarpeta `color` del dataset PlantVillage.

Mejoras respecto a la versión anterior:
  * más imágenes por clase (--max-per-class),
  * aumento de datos dentro del modelo (solo actúa al entrenar),
  * EarlyStopping: se queda con los mejores pesos, no con los de la última época,
  * fine-tuning de las últimas capas de ResNet50 con learning rate bajo,
  * métricas de train / validation / test e historial guardados en artifacts/.

Prueba rápida (poco tiempo, sin fine-tuning):
    python scripts/train_resnet.py --dataset-dir "..." --max-per-class 100 --epochs 5 --fine-tune-epochs 0
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
DEFAULT_MAX_PER_CLASS = 300      # antes: 100
DEFAULT_EPOCHS = 15              # fase 1: solo la cabeza clasificadora
DEFAULT_FINE_TUNE_EPOCHS = 10    # fase 2: últimas capas de ResNet50
FINE_TUNE_LAYERS = 30
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


def collect_images(
    color_dir: Path,
    class_names: list[str],
    max_per_class: int = DEFAULT_MAX_PER_CLASS,
) -> tuple[np.ndarray, np.ndarray]:
    """Junta rutas y etiquetas. max_per_class <= 0 usa todas las imágenes."""
    file_paths: list[str] = []
    labels: list[int] = []
    rng = np.random.default_rng(SEED)

    for label_id, class_name in enumerate(class_names):
        class_dir = color_dir / class_name
        # sorted() para que el orden sea el mismo en cualquier computadora
        images = sorted(
            str(class_dir / filename)
            for filename in os.listdir(class_dir)
            if filename.lower().endswith(IMAGE_EXTENSIONS)
        )
        rng.shuffle(images)
        if max_per_class > 0:
            images = images[:max_per_class]

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
    """70% train, 20% validation, 10% test (estratificado)."""
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        file_paths,
        labels,
        test_size=0.30,
        random_state=SEED,
        stratify=labels,
    )

    # Mismo reparto del notebook: del 30% restante, 1/3 es test y 2/3 validation.
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


def build_resnet50(num_classes: int) -> tuple[keras.Model, keras.Model]:
    inputs = keras.Input(
        shape=(IMG_SIZE, IMG_SIZE, 3),
        name="imagen",
    )

    # Aumento de datos: solo se aplica al entrenar, al predecir no hace nada.
    x = layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)

    # Mismo flujo del Colab: datos [0,1] -> Rescaling(255) -> preprocess_input.
    x = layers.Rescaling(255.0)(x)
    base = keras.applications.ResNet50(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    x = keras.applications.resnet50.preprocess_input(x)
    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="clasificacion",
    )(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="ResNet"), base


def compile_model(model: keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def unfreeze_top_layers(base: keras.Model, n_layers: int) -> None:
    """Descongela las últimas n capas de ResNet50 (sin tocar BatchNorm)."""
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def make_callbacks() -> list[keras.callbacks.Callback]:
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ]


def merge_histories(histories: list[keras.callbacks.History]) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {}
    for history in histories:
        for key, values in history.history.items():
            merged.setdefault(key, []).extend(float(v) for v in values)
    return merged


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
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=DEFAULT_MAX_PER_CLASS,
        help="Imágenes máximas por clase (0 = todas).",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=DEFAULT_FINE_TUNE_EPOCHS,
        help="Épocas de fine-tuning (0 = sin fine-tuning).",
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

    file_paths, labels = collect_images(color_dir, class_names, args.max_per_class)
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

    model, base = build_resnet50(num_classes=len(class_names))

    # ---- Fase 1: entrenar solo la cabeza clasificadora ----
    print("\n=== Fase 1: entrenando la cabeza clasificadora ===")
    compile_model(model, learning_rate=1e-3)
    histories = [
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=make_callbacks(),
        )
    ]

    # ---- Fase 2: fine-tuning de las últimas capas de ResNet50 ----
    if args.fine_tune_epochs > 0:
        best_weights = model.get_weights()
        best_val_acc = model.evaluate(val_ds, verbose=0)[1]

        print(f"\n=== Fase 2: fine-tuning de las últimas {FINE_TUNE_LAYERS} capas ===")
        unfreeze_top_layers(base, FINE_TUNE_LAYERS)
        compile_model(model, learning_rate=1e-5)
        histories.append(
            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=args.fine_tune_epochs,
                callbacks=make_callbacks(),
            )
        )

        fine_tuned_val_acc = model.evaluate(val_ds, verbose=0)[1]
        if fine_tuned_val_acc < best_val_acc:
            print(
                f"El fine-tuning no mejoró la validación "
                f"({fine_tuned_val_acc:.4f} < {best_val_acc:.4f}); "
                "se restauran los pesos de la fase 1."
            )
            model.set_weights(best_weights)

    # ---- Evaluación final ----
    train_accuracy = float(model.evaluate(train_ds, verbose=0)[1])
    val_accuracy = float(model.evaluate(val_ds, verbose=0)[1])

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
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "seed": SEED,
        "epochs": args.epochs,
        "fine_tune_epochs": args.fine_tune_epochs,
        "max_per_class": args.max_per_class,
        "total_images": int(len(file_paths)),
        "batch_size": BATCH_SIZE,
        "image_size": IMG_SIZE,
    }

    model_path = artifacts_dir / "resnet50.keras"
    class_names_path = artifacts_dir / "class_names.json"
    metrics_path = artifacts_dir / "metrics.json"
    history_path = artifacts_dir / "history.json"

    model.save(model_path)
    class_names_path.write_text(
        json.dumps(class_names, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    history_path.write_text(
        json.dumps(merge_histories(histories), indent=2),
        encoding="utf-8",
    )

    print("\nPrecisión por conjunto (si train >> val/test hay sobreajuste):")
    print(f"train: {train_accuracy * 100:.2f}% | val: {val_accuracy * 100:.2f}%")
    print("\nEvaluación en el conjunto de prueba:")
    for name in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
        print(f"{name}: {metrics[name] * 100:.2f}%")
    print(f"\nModelo guardado en: {model_path}")
    print(f"Clases guardadas en: {class_names_path}")
    print(f"Métricas guardadas en: {metrics_path}")
    print(f"Historial guardado en: {history_path}")


if __name__ == "__main__":
    main()
