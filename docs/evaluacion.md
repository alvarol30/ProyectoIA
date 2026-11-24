# Evaluación de ProyectoIA

El estudio académico comparó SVM, Random Forest, MobileNetV2 y ResNet50. En el experimento original de 230 imágenes de PlantVillage, las exactitudes fueron 74,35 %, 79,57 %, 85,22 % y 91,30 %, respectivamente. ResNet50 fue la arquitectura elegida para la demo.

## Experimentos de ResNet50

| Variante | PlantVillage, exactitud / F1 macro | PlantDoc, primera predicción / entre las tres primeras | PLD papa, primera predicción |
| --- | --- | --- | --- |
| Modelo de la demo | 93,61 % / 93,65 % (673 fotos) | 27/122 / 60/122 | 44/150 |
| Candidato con partición por hoja | 95,70 % / 95,89 % (5.901 fotos) | 30/122 / 62/122 | 45/150 |
| Candidato con 2.232 fotos de campo | 95,09 % / 95,31 % (5.901 fotos) | 28/122 / 65/122 | 46/150 |
| Candidato con papa balanceada | 94,27 % / 94,00 % (5.901 fotos) | Pendiente de evaluación | 63/150 |

La prueba del modelo de la demo en PlantVillage usa otra partición. La comparación directa en PlantVillage corresponde a los tres candidatos sobre las mismas 5.901 imágenes. Los candidatos conservaron las 23 etiquetas y la entrada RGB de 224 × 224 con normalización a [0, 1].

- **PlantDoc:** 25 fotos de la primera prueba y 97 adicionales, 14 clases en conjunto. Las tres variantes evaluadas vieron las mismas 122 fotos, separadas del entrenamiento. [Predicciones por imagen](resultados/plantdoc.csv).
- **PLD:** prueba reservada de 50 fotos por condición, de la carpeta Testing de los autores. El modelo balanceado acertó 16/50 de tizón temprano, 43/50 de tizón tardío y 4/50 de papa sana. Confundió 37/50 fotos de papa sana con manzano sano. [Predicciones por imagen](resultados/pld-papa.csv).

En PLD, el paso de 45 a 63 aciertos implicó **+12 puntos porcentuales** de exactitud balanceada frente al candidato previo, con IC pareado del 95 % de **+6 a +18 puntos** por bootstrap estratificado por clase. El intervalo asume independencia entre imágenes. La colección carece de identificadores de planta o sesión verificados para estimar la dependencia entre fotos.

El [criterio previo](protocolo-papa.txt) exigía una mejora de al menos cinco puntos en PLD, un intervalo por encima de cero y una pérdida de hasta un punto de F1 macro en PlantVillage. La pérdida fue de **1,89 puntos**, así que mantuvimos el modelo de la demo. Los resultados documentan el efecto en las colecciones elegidas. La evaluación externa no cubre las 23 clases ni demuestra rendimiento en cultivos de Perú.

## Reproducir los experimentos en Colab

Los [notebooks](experimentos/) se ejecutan con GPU y guardan pesos, etiquetas y métricas en Google Drive. Todos esperan el ZIP de PlantVillage y el CSV de particiones en `Mi unidad/ProyectoIA-datos`. El CSV se genera con `python docs/experimentos/preparar_particiones_plantvillage.py <directorio-de-insumos> <directorio-de-salida>`. El directorio de insumos debe contener `plantvillage-dataset.zip` y el [archivo de clases del proyecto](../backend/artifacts/class_names.json). La preparación combina las particiones publicadas y el mapa de hojas en una revisión fija. Excluye del entrenamiento 284 archivos cuyo grupo reaparece en prueba.

Para las dos variantes de campo, el notebook también requiere la muestra de imágenes empaquetada del experimento. Los paquetes se distribuyen por separado del código. Cada ZIP incluye un manifiesto de procedencia con nombres originales y SHA-256.

| Notebook | Entradas de campo | Carpeta de salida |
| --- | --- | --- |
| [Candidato](experimentos/candidato.ipynb) | PlantVillage | `resnet50_candidato` |
| [Campo](experimentos/campo.ipynb) | `campo_entrenamiento.zip`, 1.782 fotos originales de tomate de PlantCity y 450 de papa con tizón tardío de Tanzania | `resnet50_campo` |
| [Papa balanceada](experimentos/papa-balanceada.ipynb) | `campo_entrenamiento_balanceado.zip`, las mismas 1.782 fotos de tomate y 300 fotos por cada condición de papa de Tanzania | `resnet50_papa_balanceada` |

## Fuentes de imágenes

- [PlantVillage y mapa de agrupación por hoja](https://huggingface.co/datasets/mohanty/PlantVillage/tree/main/leaf_grouping).
- [PlantDoc, revisión de la prueba](https://github.com/pratikkayal/PlantDoc-Dataset/tree/5467f6012d78d1c446145d5f582da6096f852ae8/test).
- [PlantCity, colección de campo](https://data.mendeley.com/datasets/w8kh2xkspx/2).
- [Irish Potato Imagery, Tanzania](https://zenodo.org/records/8286529).
- [Potato Leaf Dataset, Pakistán](https://doi.org/10.3390/electronics10172064).
