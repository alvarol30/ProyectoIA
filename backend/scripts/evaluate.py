"""Valida el modelo entrenado y lo compara con imágenes distintas (punto 2).

Ejemplos (desde la carpeta backend/):

    # 1) Evaluar en el conjunto de test de PlantVillage (mismo split del entrenamiento)
    python scripts/evaluate.py --dataset-dir "D:/datos/plantvillage dataset"

    # 2) Probar con imágenes externas (fotos reales, de internet, del celular...)
    python scripts/evaluate.py --images-dir "D:/imagenes_prueba"

    # 3) Ambas cosas a la vez
    python scripts/evaluate.py --dataset-dir "..." --images-dir "..."

Para las imágenes externas hay dos formas de organizarlas:
  * Con subcarpetas con el nombre exacto de la clase (por ejemplo
    imagenes_prueba/Tomato___Late_blight/foto1.jpg). Así se calcula la precisión.
  * Sueltas en una sola carpeta: solo se muestran las predicciones.

Resultados en backend/artifacts/evaluation/.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow import keras

from train_resnet import (
    DEFAULT_MAX_PER_CLASS,
    IMAGE_EXTENSIONS,
    collect_images,
    load_image,
    make_dataset,
    split_dataset,
)

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def save_confusion_png(cm: np.ndarray, class_names: list[str], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib no está instalado: se omite la imagen (pip install matplotlib).")
        return

    labels = [name.replace("___", " - ")[:30] for name in class_names]
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Clase real")
    ax.set_title("Matriz de confusión (conjunto de test)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6, color=color)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate_test_split(
    model: keras.Model,
    class_names: list[str],
    dataset_dir: Path,
    max_per_class: int,
    out_dir: Path,
) -> None:
    color_dir = dataset_dir.expanduser().resolve() / "color"
    file_paths, labels = collect_images(color_dir, class_names, max_per_class)
    _, _, _, _, test_paths, test_labels = split_dataset(file_paths, labels)

    probabilities = model.predict(make_dataset(test_paths, test_labels), verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    n = len(class_names)

    accuracy = accuracy_score(test_labels, predictions)
    print(f"\n=== Conjunto de test: {len(test_labels)} imágenes ===")
    print(f"Accuracy: {accuracy * 100:.2f}%\n")

    report_text = classification_report(
        test_labels,
        predictions,
        labels=list(range(n)),
        target_names=class_names,
        zero_division=0,
    )
    print(report_text)
    (out_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")

    cm = confusion_matrix(test_labels, predictions, labels=list(range(n)))
    np.savetxt(out_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    save_confusion_png(cm, class_names, out_dir / "confusion_matrix.png")

    off_diagonal = cm.copy()
    np.fill_diagonal(off_diagonal, 0)
    rows, cols = np.nonzero(off_diagonal)
    confused = sorted(
        ((int(off_diagonal[i, j]), int(i), int(j)) for i, j in zip(rows, cols)),
        reverse=True,
    )[:5]
    if confused:
        print("Confusiones más frecuentes (real -> predicha):")
        for count, i, j in confused:
            print(f"  {count}x  {class_names[i]}  ->  {class_names[j]}")
    else:
        print("No hubo confusiones en el conjunto de test.")

    print(
        "\nNota: PlantVillage tiene varias fotos de la misma hoja, así que el test "
        "puede dar resultados optimistas. Compáralo con imágenes externas (--images-dir)."
    )


def evaluate_external(
    model: keras.Model,
    class_names: list[str],
    images_dir: Path,
    out_dir: Path,
) -> None:
    images_dir = images_dir.expanduser().resolve()
    paths = sorted(
        p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"No se encontraron imágenes (jpg/jpeg/png) en {images_dir}")

    dataset = (
        tf.data.Dataset.from_tensor_slices([str(p) for p in paths])
        .map(lambda p: load_image(p, tf.constant(0))[0])
        .batch(32)
    )
    probabilities = model.predict(dataset, verbose=0)
    top3 = np.argsort(-probabilities, axis=1)[:, :3]

    rows = []
    correct = 0
    labeled = 0
    print(f"\n=== Imágenes externas: {len(paths)} ===")
    for path, probs, top in zip(paths, probabilities, top3):
        predicted = class_names[int(top[0])]
        confidence = float(probs[int(top[0])])
        real = path.parent.name if path.parent.name in class_names else ""
        is_correct = ""
        if real:
            labeled += 1
            is_correct = real == predicted
            correct += int(is_correct)
        rows.append(
            {
                "imagen": str(path.relative_to(images_dir)),
                "clase_real": real,
                "prediccion": predicted,
                "confianza": f"{confidence * 100:.1f}%",
                "top3": " | ".join(
                    f"{class_names[int(k)]} ({probs[int(k)] * 100:.1f}%)" for k in top
                ),
                "correcta": is_correct,
            }
        )
        mark = "" if is_correct == "" else ("  OK" if is_correct else "  ERROR")
        print(f"{path.name[:35]:35s} -> {predicted} ({confidence * 100:.1f}%){mark}")

    csv_path = out_dir / "external_predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    if labeled:
        print(f"\nPrecisión en imágenes externas: {correct}/{labeled} = {100 * correct / labeled:.1f}%")
    else:
        print(
            "\nLas imágenes no están en subcarpetas con el nombre de la clase, "
            "así que no se calculó la precisión."
        )
    print(f"Tabla guardada en: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluar el modelo de ProyectoIA.")
    parser.add_argument("--dataset-dir", type=Path, help="Carpeta que contiene 'color' de PlantVillage.")
    parser.add_argument("--images-dir", type=Path, help="Carpeta con imágenes externas para probar.")
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Debe ser el mismo valor usado al entrenar (por defecto se lee de metrics.json).",
    )
    args = parser.parse_args()

    if not args.dataset_dir and not args.images_dir:
        parser.error("Indica --dataset-dir y/o --images-dir.")

    artifacts_dir = args.artifacts_dir.expanduser().resolve()
    model_path = args.model_path or artifacts_dir / "resnet50.keras"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"No existe el modelo {model_path}. Primero ejecuta scripts/train_resnet.py."
        )

    class_names = json.loads((artifacts_dir / "class_names.json").read_text(encoding="utf-8"))

    max_per_class = args.max_per_class
    if max_per_class is None:
        metrics_file = artifacts_dir / "metrics.json"
        if metrics_file.is_file():
            metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
            max_per_class = metrics.get("max_per_class", 100)
        else:
            max_per_class = DEFAULT_MAX_PER_CLASS

    out_dir = artifacts_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = keras.models.load_model(model_path)

    if args.dataset_dir:
        evaluate_test_split(model, class_names, args.dataset_dir, max_per_class, out_dir)
    if args.images_dir:
        evaluate_external(model, class_names, args.images_dir, out_dir)

    print(f"\nResultados guardados en: {out_dir}")


if __name__ == "__main__":
    main()
