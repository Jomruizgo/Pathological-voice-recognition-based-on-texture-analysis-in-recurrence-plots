import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
EDA_REPORT_FILE = os.path.join(OUTPUT_DIR, "eda_summary_report.txt")

DATA_DIR = os.path.join(BASE_DIR,'..', 'data')
#DATA_DIR = os.path.join(BASE_DIR,'..','..','BDAtos HUPA Segmentada')
ORIGINAL_NORMAL_AUDIO_DIR = os.path.join(DATA_DIR, 'Normal')
ORIGINAL_PATHOL_AUDIO_DIR = os.path.join(DATA_DIR, 'Pathol')

# Directorio para plots de forma de onda con detección de silencio
EDA_WAVEFORM_PLOTS_DIR = os.path.join(OUTPUT_DIR, "waveform_plots")
EDA_WAVEFORM_PLOTS_NORMAL_DIR = os.path.join(EDA_WAVEFORM_PLOTS_DIR, "Normal")
EDA_WAVEFORM_PLOTS_PATHOL_DIR = os.path.join(EDA_WAVEFORM_PLOTS_DIR, "Pathol")


# --- Hiperparámetros Comunes (deben ser consistentes con preprocesamiento si se usan) ---
# Frecuencia de muestreo objetivo para carga de audio en EDA (para plots/SAD)
# ¡IMPORTANTE! Debe ser la misma que TARGET_SAMPLE_RATE en 02_Audio_Preprocess/config.py
# Su propósito es asegurar que el proceso de análisis de la forma de onda y la detección de actividad de sonido para la visualización 
# en el EDA se realice de manera consistente y comparable en todos los archivos, y que esta visualización sea una representación fiel de 
# cómo operará el SAD en la etapa de preprocesamiento real
TARGET_SAMPLE_RATE = 25000  # Hz (Asegúrate de que coincida con tu config de preprocesamiento)

# Configuración SAD para detección de silencio en EDA (debe ser consistente con preprocesamiento)
SAD_TOP_DB = 30 # Umbral en dB para librosa.effects.split (ej. 20-40, 30 es un buen punto de partida)

# --- Opciones de Visualización de EDA ---
EDA_SAVE_WAVEFORM_PLOTS = True # Habilitar/deshabilitar la generación de plots de forma de onda (individuales)