# ProyectoIA

Demostración local para clasificar hojas de manzano, maíz, pimiento, papa y tomate en 23 clases de PlantVillage. ResNet50 devuelve una posible condición y las tres clases con mayor puntuación. La [evaluación](docs/evaluacion.md) muestra resultados dentro del dataset y con fotos externas.

## Ejecutar la demo en Windows

Desde la raíz del repositorio, abre PowerShell con Python 3.11:

```powershell
py -3.11 -m venv backend\.venv
.\backend\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

Consigue del equipo el modelo exportado y el `class_names.json` de la misma ejecución. Colócalos en `backend/artifacts/` como `resnet50.keras` y `class_names.json`. Los pesos se distribuyen por separado. Si no los tienes, usa los pasos de entrenamiento de abajo. Para cargar artefactos desde una ruta independiente, define `ARTIFACTS_DIR` en PowerShell antes de iniciar la API.

En una terminal, desde la raíz:

```powershell
cd backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

En una segunda terminal, desde la raíz:

```powershell
.\backend\.venv\Scripts\python.exe -m http.server 5500 --bind 127.0.0.1 --directory frontend
```

Abre [la demo](http://127.0.0.1:5500/) para subir una imagen o [Swagger](http://127.0.0.1:8000/docs) para probar todos los endpoints, incluida la carga múltiple. [El endpoint de salud](http://127.0.0.1:8000/health) debe mostrar `model_loaded: true`. Si muestra `false`, revisa la ruta de los artefactos y el registro de arranque.

## Contrato de la API

| Endpoint | Entrada y respuesta |
| --- | --- |
| `GET /health` | Estado de carga del modelo. |
| `GET /model` | Nombre del modelo, clases, formatos y límites. |
| `POST /predict` | Una imagen en el campo multipart `file`. |
| `POST /predict/batch` | De 1 a 10 imágenes mediante campos multipart repetidos `files`. |

La API acepta JPEG, PNG y WebP, hasta 10 MB por archivo de forma predeterminada y 25 megapíxeles. Cada predicción contiene `model`, `class_name`, `crop_name`, `condition_name`, `confidence` (de 0 a 1) y `top_3`. También expone `crop` y `condition` con los identificadores originales.

La respuesta por lote devuelve `results` en orden. Cada resultado identifica el archivo por `index` (desde cero) y `filename`. Los elementos de estado `ok` incluyen `prediction`. Los de estado `error` incluyen `status_code` y `detail`, y el lote sigue procesando los otros archivos. Si se necesita un modelo y no está disponible, la API devuelve HTTP 503.

```powershell
curl.exe -F "files=@hoja1.jpg" -F "files=@hoja2.png" http://127.0.0.1:8000/predict/batch
```

La interfaz incluida en `frontend/` usa la predicción individual. La carga múltiple se puede probar en Swagger o integrar con el mismo contrato. La API permite los orígenes locales de los puertos 3000 y 5500. Usa `backend/.env.example` como referencia para configurar otros orígenes.

## Entrenamiento y evaluación

Descarga [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset) y descomprime las imágenes en color. Desde `backend`, pasa al script la carpeta que contiene `color`:

```powershell
.\.venv\Scripts\python.exe scripts\train_resnet.py --dataset-dir "D:\datos\plantvillage dataset" --max-per-class 300 --epochs 15 --fine-tune-epochs 10
```

El script genera `resnet50.keras`, `class_names.json`, `metrics.json` e `history.json` en `backend/artifacts/`. Usa `--artifacts-dir` para guardar una ejecución aparte.

Para evaluar fotos externas, ponlas en subcarpetas con las clases de `GET /model`, por ejemplo `fotos/Tomato___Late_blight`. Desde `backend`:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --images-dir "D:\fotos" --output-dir "D:\resultados"
```

El CSV incluye la clase esperada (si existe), la predicción y las tres primeras clases. Los [experimentos, resultados y pasos de Colab](docs/evaluacion.md) explican qué se probó y por qué se conservó el modelo de la demo.

## Alcance de la demo

Usa una foto de una hoja de uno de los cinco cultivos. La puntuación expresa la salida del clasificador entre sus 23 clases. Fotos de otros cultivos u objetos también pueden recibir una clase con puntuación alta. Los resultados de fotos de campo sirven para evaluar el prototipo y no constituyen un diagnóstico agronómico.
