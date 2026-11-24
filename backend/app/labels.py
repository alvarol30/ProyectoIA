"""Etiquetas de presentación de PlantVillage."""

NAMES_ES = {
    "Apple": "Manzano",
    "Corn_(maize)": "Maíz",
    "Pepper,_bell": "Pimiento morrón",
    "Potato": "Papa",
    "Tomato": "Tomate",
    "Apple_scab": "Roña del manzano",
    "Black_rot": "Podredumbre negra",
    "Cedar_apple_rust": "Roya del cedro y del manzano",
    "Cercospora_leaf_spot Gray_leaf_spot": "Mancha gris por Cercospora",
    "Common_rust_": "Roya común",
    "Northern_Leaf_Blight": "Tizón norteño de la hoja",
    "Bacterial_spot": "Mancha bacteriana",
    "Early_blight": "Tizón temprano",
    "Late_blight": "Tizón tardío",
    "Leaf_Mold": "Moho de la hoja",
    "Septoria_leaf_spot": "Mancha foliar por Septoria",
    "Spider_mites Two-spotted_spider_mite": "Ácaro de dos manchas",
    "Target_Spot": "Mancha en diana",
    "Tomato_Yellow_Leaf_Curl_Virus": "Virus del rizado amarillo de la hoja del tomate",
    "Tomato_mosaic_virus": "Virus del mosaico del tomate",
    "healthy": "Hoja sana según el modelo",
}


def describe_class(class_name: str) -> dict[str, str]:
    crop, _, condition = class_name.partition("___")
    return {
        "class_name": class_name,
        "crop": crop,
        "condition": condition,
        "crop_name": NAMES_ES.get(crop, crop),
        "condition_name": NAMES_ES.get(condition, condition),
    }
