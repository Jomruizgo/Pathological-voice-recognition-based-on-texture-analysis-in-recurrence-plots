# project_root/03_Full_Optimal_Parameters_Analysis/config.py

import os

# --- Configuración de Rutas ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directorio de entrada: Audios preprocesados del módulo 02_Audio_Preprocess
PREPROCESSED_AUDIO_BASE_DIR = os.path.join(BASE_DIR, '..', '02_Audio_Preprocess', 'output')
INPUT_NORMAL_AUDIO_DIR = os.path.join(PREPROCESSED_AUDIO_BASE_DIR, 'Normal')
INPUT_PATHOL_AUDIO_DIR = os.path.join(PREPROCESSED_AUDIO_BASE_DIR, 'Pathol')

# Directorio de salida para los gráficos del análisis de parámetros
OUTPUT_ANALYSIS_DIR = os.path.join(BASE_DIR, 'output')

# Archivo de checkpoint para guardar el progreso
CHECKPOINT_FILE = os.path.join(OUTPUT_ANALYSIS_DIR, 'checkpoint.json')

# --- Configuración para la Carga de Audio ---
# Frecuencia de muestreo (debe coincidir con la de preprocesamiento)
TARGET_SAMPLE_RATE = 25000 # Hz

# Duración del audio a procesar (None = audio completo)
# Si el audio es más corto, se procesa completamente
# Si es más largo, se pueden procesar los primeros N segundos
AUDIO_DURATION = None  # None para procesar el audio completo

# --- Procesamiento de Archivos ---
# Procesar TODOS los archivos (no muestreo)
PROCESS_ALL_FILES = True  # True = procesar todos, False = usar muestreo

# Si PROCESS_ALL_FILES = False, estos valores se usarán para muestreo
NUM_NORMAL_SAMPLES_FOR_ANALYSIS = 24
NUM_PATHOL_SAMPLES_FOR_ANALYSIS = 20

# --- Configuración para la Determinación de TAU ---
# Rango de valores de tiempo de retraso (tau) a probar
TAU_MAX = 100 # Número máximo de retraso a probar
TAU_STEP = 1  # Paso para probar los valores de tau

# --- Configuración para la Determinación de DIMENSIÓN (D) ---
# Dimensión máxima a probar para Falsos Vecinos
DIM_MAX = 10

# Porcentaje de vecinos falsos (umbral para FNN)
FNN_THRESHOLD = 0.05 # Generalmente se busca donde el porcentaje de FNN cae por debajo de un umbral pequeño (ej. 1-5%)

# --- Configuración para Guardar Resultados ---
SAVE_TAU_PLOT = True
SAVE_DIM_PLOT = True
SAVE_AGGREGATE_PLOTS = True

# --- Configuración de Checkpoints ---
# Guardar checkpoint cada N archivos procesados
CHECKPOINT_FREQUENCY = 5  # Guardar cada 5 archivos (para paralelización)

# Habilitar logging detallado
VERBOSE = True

# --- Configuración de Paralelización ---
# Número de cores a usar para procesamiento paralelo
NUM_CORES = 7  # Usar 7 de los 8 cores disponibles

# Tamaño de batch para procesamiento paralelo
# Procesa N archivos en paralelo, guarda checkpoint, y continúa con el siguiente batch
BATCH_SIZE = 10  # Guardar checkpoint cada 10 archivos procesados
