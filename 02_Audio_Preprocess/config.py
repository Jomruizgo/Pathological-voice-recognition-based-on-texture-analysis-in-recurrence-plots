import os

# --- Configuración de Rutas ---
# La ruta base donde se encuentra la carpeta 'data'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR,'..', 'data')
ORIGINAL_NORMAL_AUDIO_DIR = os.path.join(DATA_DIR, 'Normal')
ORIGINAL_PATHOL_AUDIO_DIR = os.path.join(DATA_DIR, 'Pathol')


# Directorios de AUDIO PREPROCESADO
OUTPUT_PREPROCESSED_AUDIO_DIR = os.path.join(BASE_DIR, "output")
PREPROCESSED_NORMAL_AUDIO_DIR = os.path.join(OUTPUT_PREPROCESSED_AUDIO_DIR, "Normal")
PREPROCESSED_PATHOL_AUDIO_DIR = os.path.join(OUTPUT_PREPROCESSED_AUDIO_DIR, "Pathol")


# --- Hiperparámetros de Audio ---
# Estos ya están definidos por ti: 25 kHz, mono, 16 bits
TARGET_SAMPLE_RATE = 25000  # Hz
MAX_AUDIO_DURATION_SECONDS = 1.0 # Duración máxima del audio a procesar (en segundos)

# --- Configuración para Detección de Actividad de Sonido (SAD) y Recorte Inteligente ---
SAD_TOP_DB = 30 # Umbral en dB para librosa.effects.split (ej. 20-40, 30 es un buen punto de partida)
SAD_ENERGY_THRESHOLD = 0.01 # No directamente usado por librosa.effects.split, pero se mantiene por compatibilidad.

# --- Configuración de Guardado de Plots ---
# Indica si se deben guardar los plots de preprocesamiento
SAVE_PREPROCESS_PLOTS=True

# Directorio base para los plots
BASE_PREPROCESS_PLOTS_DIR = os.path.join(OUTPUT_PREPROCESSED_AUDIO_DIR, "preprocess_plots")

# Subdirectorios para los plots según la categoría
PREPROCESS_PLOTS_NORMAL_DIR = os.path.join(BASE_PREPROCESS_PLOTS_DIR, "Normal")
PREPROCESS_PLOTS_PATHOL_DIR = os.path.join(BASE_PREPROCESS_PLOTS_DIR, "Pathol")