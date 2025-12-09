# project_root/04_Recurrence_Plot_Generator/config.py
import os

# --- Configuración de Rutas ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directorios de entrada: Audios preprocesados del módulo 02_Audio_Preprocess
PREPROCESSED_AUDIO_BASE_DIR = os.path.join(BASE_DIR, '..', '02_Audio_Preprocess', 'output')
INPUT_NORMAL_AUDIO_DIR = os.path.join(PREPROCESSED_AUDIO_BASE_DIR, 'Normal')
INPUT_PATHOL_AUDIO_DIR = os.path.join(PREPROCESSED_AUDIO_BASE_DIR, 'Pathol')

# Directorios de salida: Donde se guardarán los Recurrence Plots y Phase Space Plots
OUTPUT_RP_BASE_DIR = os.path.join(BASE_DIR, 'output')
OUTPUT_RP_NORMAL_DIR = os.path.join(OUTPUT_RP_BASE_DIR, 'Recurrence_Plots', 'Normal') # Nueva subcarpeta
OUTPUT_RP_PATHOL_DIR = os.path.join(OUTPUT_RP_BASE_DIR, 'Recurrence_Plots', 'Pathol') # Nueva subcarpeta

# Directorios para Phase Space Plots
OUTPUT_PS_NORMAL_DIR = os.path.join(OUTPUT_RP_BASE_DIR, 'Phase_Space_Plots', 'Normal') # Nueva subcarpeta
OUTPUT_PS_PATHOL_DIR = os.path.join(OUTPUT_RP_BASE_DIR, 'Phase_Space_Plots', 'Pathol') # Nueva subcarpeta

# --- Configuración de Frecuencia de Muestreo ---
TARGET_SAMPLE_RATE = 25000 # Hz (25 kHz) 


# --- Parámetros para la Generación de Recurrence Plots ---
# IMPORTANTE: Estos valores se basan en los resultados del módulo 03_Full_Optimal_Parameters_Analysis
# m=7: Cubre 97.5% de señales patológicas (propiedad acumulativa de dimensión)
# τ=9: Mediana global de MI (valor PROVISIONAL, pendiente validación con literatura)
EMBEDDING_DIM = 7
TIME_DELAY = 9

EPSILON_MODE = 'percentage'
EPSILON_VALUE = 0.1 # 10% del rango de distancias

# Precisión numérica para matrices (float32 usa menos memoria, float64 más preciso)
# float32: ~2.3 GB por matriz vs float64: ~4.6 GB para 1 seg de audio a 25kHz
DTYPE = 'float64'

# Número de núcleos para procesamiento paralelo
# N_JOBS = 1: secuencial, N_JOBS = -1: usar todos los núcleos disponibles
# Cada proceso usa ~4.6 GB de RAM (float64), ajustar según memoria disponible
N_JOBS = 2
NORMALIZATION_METHOD = 'minmax'
DISTANCE_METRIC = 'euclidean'
BINARY_RP = False

# --- Configuración para la Salida de Imágenes de Recurrence Plots Puros ---
# Tamaño de la figura en pulgadas (ej. 8x8 pulgadas)
RP_FIGSIZE_INCHES = 8 
# Dots Per Inch (píxeles por pulgada). Determina la resolución final de la imagen.
# Para una imagen de 8x8 pulgadas con 300 DPI, la resolución será 2400x2400 píxeles.
RP_DPI = 300 

# --- Configuración para Habilitar/Deshabilitar la Generación ---
SAVE_RECURRENCE_PLOTS = True
SAVE_PHASE_SPACE_PLOTS = False # Nueva variable para controlar los plots de espacio de fase

# --- Archivos de registro ---
LOG_FILE_PATH = os.path.join(BASE_DIR, 'rp_generation.log')

# --- Archivo de historial de procesamiento ---
# Registra qué archivos fueron procesados y con qué parámetros
HISTORY_FILE_PATH = os.path.join(OUTPUT_RP_BASE_DIR, 'processing_history.json')