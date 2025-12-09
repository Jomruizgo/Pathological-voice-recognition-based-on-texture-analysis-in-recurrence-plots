# project_root/05_Texture_Descriptors/config.py
"""
Configuración para el módulo de extracción de descriptores de textura.

Este módulo analiza los Recurrence Plots generados en el módulo 04 y extrae
características de textura que serán utilizadas para la clasificación en el módulo 06.

IMPORTANTE: Este módulo está diseñado para ser interrumpible y reanudable.
Si el proceso se detiene, al volver a ejecutarlo continuará desde donde quedó.
"""

import os
import numpy as np

# --- Configuración de Rutas ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directorios de entrada: Recurrence Plots del módulo 04
RP_INPUT_BASE_DIR = os.path.join(BASE_DIR, '..', '04_RP_Generator_m7_tau9', 'output', 'Recurrence_Plots')
RP_INPUT_NORMAL_DIR = os.path.join(RP_INPUT_BASE_DIR, 'Normal')
RP_INPUT_PATHOL_DIR = os.path.join(RP_INPUT_BASE_DIR, 'Pathol')

# Directorios de salida
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, 'output')
OUTPUT_FEATURES_DIR = os.path.join(OUTPUT_BASE_DIR, 'features')
OUTPUT_CHECKPOINTS_DIR = os.path.join(OUTPUT_BASE_DIR, 'checkpoints')

# Archivo principal de características extraídas
FEATURES_OUTPUT_FILE = os.path.join(OUTPUT_FEATURES_DIR, 'texture_features.csv')
# Archivo de metadata con información sobre las características
FEATURES_METADATA_FILE = os.path.join(OUTPUT_FEATURES_DIR, 'features_metadata.json')

# --- Sistema de Checkpoints ---
# El sistema de checkpoints permite reanudar el procesamiento si se interrumpe
CHECKPOINT_FILE = os.path.join(OUTPUT_CHECKPOINTS_DIR, 'processing_checkpoint.json')
ENABLE_CHECKPOINTS = True  # Activar/desactivar sistema de checkpoints
CHECKPOINT_FREQUENCY = 10  # Guardar checkpoint cada N imágenes procesadas

# --- Configuración de Procesamiento ---
# Formato de imagen esperado
IMAGE_FORMAT = '.png'  # Extensión de los archivos de RP

# Normalización de imágenes antes de extraer características
NORMALIZE_IMAGES = True  # Normalizar imágenes a rango [0, 1]

# Procesamiento en paralelo
ENABLE_PARALLEL = True  # Activar procesamiento paralelo
N_JOBS = -1  # Número de trabajos paralelos (-1 = usar todos los cores)

# Configuración de procesamiento de imágenes
IMAGE_MIN_SIZE = (2000, 2000)      # Tamaño mínimo de imagen (ancho, alto)
IMAGE_MAX_SIZE = (25000, 25000)  # Tamaño máximo (None para sin límite)
                               # Nota: RPs típicos son 2400x2400, límite anterior era 2048x2048
IMAGE_TARGET_DTYPE = 'uint8'   # Tipo de datos objetivo
IMAGE_NORMALIZE_RANGE = (0, 255)  # Rango de normalización

# Configuración de checkpoints granulares
CHECKPOINT_BATCH_SIZE = 5     # Número de imágenes procesadas antes de guardar checkpoint parcial
ENABLE_PARTIAL_CHECKPOINTS = True  # Habilitar checkpoints durante procesamiento de descriptor

# --- Configuración por defecto de Descriptores ---
# ═══════════════════════════════════════════════════════════════
# SISTEMA DE CONFIGURACIÓN HÍBRIDO:
#
# 🔄 ESTAS configuraciones tienen PRIORIDAD sobre los defaults de los constructores
#    cuando se usa el pipeline principal (main.py, GUI, ModularPipeline)
#
# 📋 Los defaults en los constructores sirven como:
#    • Documentación de valores recomendados
#    • Fallback para uso directo del descriptor
#
# 💡 Para cambiar la configuración del pipeline, MODIFICAR AQUÍ, no en constructores
# ═══════════════════════════════════════════════════════════════

DEFAULT_DESCRIPTORS = {
    'glcm': {  
        'enabled': True,
        # Multi-escala: micro (1), local (2), medio (5)
        'distances': [1],
        # Cobertura direccional completa para RPs
        'angles': [0, np.pi/4, np.pi/2, 3*np.pi/4], 
        # 256: Max detalle | 64: Recomendado para RPs | 32: Max robustez
        'levels': 256,
        # Estabilidad estadística y comparabilidad
        'symmetric': True,
        'normed': True,
        # Propiedades Haralick más discriminativas
        'properties': ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    },
    
    'lbp': {
        'enabled': True,
        # Multi-escala: micro-local (1), local (2), medio (3)
        'radius': [1],
        # Diferentes resoluciones angulares: básica (8), media (16), alta (24)
        'n_points': [4],
        # 'uniform': Solo patrones con ≤2 transiciones (recomendado para robustez)
        'method': 'uniform'
    },
    
    'statistical': {
        'enabled': True,
        'compute_moments': True,
        'moments': ['mean', 'std', 'skewness', 'kurtosis'],
        'compute_percentiles': True,
        'percentiles': [10, 25, 50, 75, 90],
        'compute_histogram': True,
        'n_bins': 64,
        'compute_entropy': True
    },
    
    'gabor': {
        'enabled': True,
        'frequencies': [0.1, 0.2, 0.4],
        'orientations': [0, 45, 90, 135],
        'sigma': 1.0,
        'compute_magnitude': True,
        'compute_phase': False,
        'compute_energy': True
    },
    
    'wavelet': {
        'enabled': True,
        'wavelet': 'db4',
        'levels': 3,
        'feature_types': ['energy', 'entropy', 'mean', 'std'],
        'compute_ratios': True
    },
    
    'rqa': {
        'enabled': True,
        'epsilon': None,  # None = usa percentil 10 adaptativo
        'min_line_length': 2  # Longitud mínima de línea para DET y LAM
    }
}

# --- Configuración de Análisis ---
# NOTA: Los análisis de características se han movido al módulo 06_Feature_Analysis

# --- Configuración de Logging ---
LOG_LEVEL = 'INFO'  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
LOG_FILE = os.path.join(BASE_DIR, 'texture_extraction.log')

# --- Configuración de Performance ---
# Límites de memoria para procesamiento por lotes
MAX_MEMORY_MB = 20480  # Máximo uso de memoria en MB
BATCH_SIZE_AUTO = True  # Ajustar batch_size automáticamente según memoria

# Configuración de cache
ENABLE_FEATURE_CACHE = True  # Cachear características ya calculadas
CACHE_DIR = os.path.join(OUTPUT_BASE_DIR, 'cache')

print(f"Configuración cargada para extracción de descriptores de textura")
print(f"Directorios de entrada: {RP_INPUT_NORMAL_DIR}, {RP_INPUT_PATHOL_DIR}")
print(f"Directorio de salida: {OUTPUT_FEATURES_DIR}")