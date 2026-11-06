# ProyectoIA — backend de inteligencia artificial

Primera implementación independiente del notebook original. El backend recibe imágenes mediante una API, carga un modelo ResNet50 guardado y devuelve la clase predicha y sus probabilidades.

> Este directorio no modifica el Google Colab del equipo. El entrenamiento se reproduce desde `backend/scripts/train_resnet.py` cuando se disponga del dataset PlantVillage.

## Estructura

```text
ProyectoIA/
├── README.md
└── backend/
    ├── app/
    │   ├── __init__.py
    │   ├── config.py
    │   ├── main.py
    │   └── predictor.py
    ├── artifacts/
    │   └── README.md
    ├── scripts/
    │   └── train_resnet.py
    ├── .env.example
    └── requirements.txt
```

## Requisitos

- Python 3.11 recomendado.
- El dataset PlantVillage utilizado por el equipo, descomprimido.
- En Windows, TensorFlow 2.21 tiene paquetes CPU para Python 3.10–3.13. Esta primera versión funciona en CPU; no requiere GPU.

## 1. Instalar el backend (Windows PowerShell)

Abre PowerShell en la carpeta `backend`:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-api.txt
```

Si PowerShell bloquea la activación del entorno, ejecuta solamente para esta sesión:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 2. Iniciar la API

Desde la carpeta `backend`, con el entorno virtual activo:

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Abre estas direcciones en el navegador:

- `http://127.0.0.1:8000/` — información básica.
- `http://127.0.0.1:8000/health` — estado del backend y del modelo.
- `http://127.0.0.1:8000/docs` — documentación interactiva de la API.

**Resultado esperado antes de entrenar:** la API inicia, pero `/health` indica `model_loaded: false` y `/predict` devuelve HTTP 503. Eso es normal: todavía no existe el archivo del modelo.

Cuando se vaya a entrenar o ejecutar predicciones con los pesos, instala también las dependencias de IA:

```powershell
pip install -r requirements.txt
```

Este segundo archivo incluye TensorFlow 2.21 y scikit-learn; TensorFlow es una descarga grande.

## 3. Entrenar y guardar ResNet50 (cuando esté disponible el dataset)

El argumento `--dataset-dir` debe apuntar a la carpeta que contiene la subcarpeta `color`. Por ejemplo, si la ruta es `D:\datos\plantvillage dataset\color`, pasa `D:\datos\plantvillage dataset`:

```powershell
python scripts/train_resnet.py --dataset-dir "D:\datos\plantvillage dataset"
```

El script selecciona las mismas cinco familias de cultivos y hasta 100 imágenes por clase, usa entrada 224×224 RGB, división 70/20/10, ResNet50 preentrenado con ImageNet, base congelada, GlobalAveragePooling2D, Dropout(0.2), capa Dense de 23 salidas, Adam (0.001) y 10 épocas.

El entrenamiento genera:

- `artifacts/resnet50.keras` — arquitectura y pesos entrenados.
- `artifacts/class_names.json` — nombres de clases en el orden de las salidas.
- `artifacts/metrics.json` — métricas obtenidas en la prueba.

El reentrenamiento independiente sigue la metodología del Colab, pero no se garantiza que reproduzca exactamente el 91,30 % del informe: la inicialización y el entorno de ejecución pueden cambiar los resultados.

## 4. Probar una predicción

Después de generar los artefactos, reinicia Uvicorn y entra en `http://127.0.0.1:8000/docs`.

En `POST /predict`, selecciona una imagen JPEG, PNG o WEBP. El resultado incluye la clase, cultivo, condición, confianza y las tres clases con mayor puntuación.

## Notas de integración

- El frontend puede enviar la imagen como `multipart/form-data`, con el campo `file`.
- CORS permite por defecto `http://localhost:3000` y `http://127.0.0.1:3000`.
- Ajusta `ALLOWED_ORIGINS` con una lista separada por comas cuando se conozca la dirección del frontend.
- Los archivos `.keras` se excluyen de Git porque pueden ser grandes. No subas pesos ni datasets al repositorio sin acordar con el equipo dónde almacenarlos.
- Este backend es una base de desarrollo. Antes del despliegue deberán medirse tiempo de respuesta, memoria, tamaño de artefactos y límites del proveedor de hosting.
