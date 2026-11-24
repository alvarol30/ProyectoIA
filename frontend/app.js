const API_URL = 'http://127.0.0.1:8000/predict';

const dropZone = document.getElementById('dropZone');
const imageInput = document.getElementById('imageInput');
const previewContainer = document.getElementById('previewContainer');
const imagePreview = document.getElementById('imagePreview');
const removeImgBtn = document.getElementById('removeImgBtn');
const analyzeBtn = document.getElementById('analyzeBtn');

const initialState = document.getElementById('initialState');
const loadingState = document.getElementById('loadingState');
const resultContent = document.getElementById('resultContent');

const cropName = document.getElementById('cropName');
const conditionName = document.getElementById('conditionName');
const confidenceBar = document.getElementById('confidenceBar');
const confidenceText = document.getElementById('confidenceText');
const topPredictionsList = document.getElementById('topPredictionsList');

let selectedFile = null;

// Manejo de eventos de selección de archivo
imageInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) {
        handleFileSelect(e.dataTransfer.files[0]);
    }
});

function handleFileSelect(file) {
    if (!file.type.startsWith('image/')) {
        alert('Por favor selecciona una imagen válida.');
        return;
    }
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        dropZone.classList.add('hidden');
        previewContainer.classList.remove('hidden');
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

removeImgBtn.addEventListener('click', () => {
    selectedFile = null;
    imageInput.value = '';
    previewContainer.classList.add('hidden');
    dropZone.classList.remove('hidden');
    analyzeBtn.disabled = true;
    resetResults();
});

// Diccionario completo de traducción para las 23 clases en español
const translations = {
    // Cultivos
    "Apple": "Manzano",
    "Corn (maize)": "Maíz",
    "Corn_(maize)": "Maíz",
    "Pepper, bell": "Pimiento morrón",
    "Pepper,_bell": "Pimiento morrón",
    "Potato": "Papa",
    "Tomato": "Tomate",

    // Patologías y estados
    "Apple scab": "Roña del manzano",
    "Black rot": "Podredumbre negra",
    "Cedar apple rust": "Roya del cedro y del manzano",
    "Cercospora leaf spot Gray leaf spot": "Mancha foliar por Cercospora y mancha gris",
    "Common rust": "Roya común",
    "Northern Leaf Blight": "Tizón norteño de la hoja",
    "Bacterial spot": "Mancha bacteriana",
    "Early blight": "Tizón temprano",
    "Late blight": "Tizón tardío",
    "Leaf Mold": "Moho de la hoja",
    "Septoria leaf spot": "Mancha foliar por Septoria",
    "Spider mites Two-spotted spider mite": "Ácaros / Arañuela roja",
    "Target Spot": "Mancha en diana",
    "Tomato Yellow Leaf Curl Virus": "Virus del rizado amarillo de la hoja",
    "Tomato mosaic virus": "Virus del mosaico del tomate",
    "healthy": "Sano (Sin patología detectada)"
};

// Función auxiliar para traducir y limpiar textos robustamente
function translateText(text) {
    if (!text) return "";
    let clean = text.replace(/___/g, " ").replace(/_/g, " ").replace(/\(|\)/g, "").trim();
    return translations[clean] || translations[text] || text;
}

function resetResults() {
    initialState.classList.remove('hidden');
    loadingState.classList.add('hidden');
    resultContent.classList.add('hidden');
}

// Envío a la API FastAPI
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    initialState.classList.add('hidden');
    loadingState.classList.remove('hidden');
    resultContent.classList.add('hidden');

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Error en el servidor backend');

        const data = await response.json();
        renderResults(data);
    } catch (error) {
        alert('No se pudo conectar con el servidor de IA (http://127.0.0.1:8000). Asegúrate de que el backend esté corriendo.');
        resetResults();
    }
});

// Función para renderizar los resultados traducidos en español sin duplicación
function renderResults(data) {
    loadingState.classList.add('hidden');
    resultContent.classList.remove('hidden');

    // Traducir cultivo y condición al español usando el diccionario
    const cultivoEspanol = translateText(data.crop);
    const condicionEspanol = translateText(data.condition || data.class_name);

    // Separar correctamente el cultivo y la condición para evitar duplicados
    if (cropName) {
        cropName.textContent = cultivoEspanol;
    }

    if (conditionName) {
        conditionName.textContent = condicionEspanol;
    }

    const percentage = Math.round((data.confidence || 0) * 100);
    confidenceBar.style.width = `${percentage}%`;
    confidenceText.textContent = `${percentage}%`;

    // Información del modelo de IA
    const modelInfo = document.getElementById('modelInfo');
    if (modelInfo) {
        modelInfo.textContent = `Modelo: ${data.model} (23 clases)`;
    }

    // Renderizar Top 3 con nombres traducidos al español y cálculo corregido
    topPredictionsList.innerHTML = '';
    if (data.top_3 && Array.isArray(data.top_3)) {
        data.top_3.forEach((item) => {
            const li = document.createElement('li');
            const perc = Math.round((item.confidence || 0) * 100);
            const nombreCultivoTop = translateText(item.crop);
            const nombreClaseTop = translateText(item.condition || item.class_name);
            li.innerHTML = `<span>${nombreCultivoTop} — ${nombreClaseTop}</span><strong>${perc}%</strong>`;
            topPredictionsList.appendChild(li);
        });
    }
}