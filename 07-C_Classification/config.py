"""
Configuración para el módulo 07-C: Clasificación con Features Auto-Seleccionadas.

Este módulo carga automáticamente las características seleccionadas por el
módulo 06-C, eliminando la necesidad de configuración manual.

NOTA: Este módulo trabaja con características extraídas de RPs generados
      con m=7, τ=9 (05_Texture_Descriptors_m7_tau9 → 06-C → 07-C).
"""

import os
import json
from pathlib import Path

# ============================================================================
# RUTAS DE DIRECTORIOS
# ============================================================================

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / 'output'
MODELS_DIR = OUTPUT_DIR / 'models'
METRICS_DIR = OUTPUT_DIR / 'metrics'
PLOTS_DIR = OUTPUT_DIR / 'plots'

# ============================================================================
# CONEXIÓN AUTOMÁTICA CON MÓDULO 06-C
# ============================================================================

# Path al JSON de selección de características del módulo 06-C
FEATURE_RANKING_JSON = BASE_DIR.parent / '06-C_Feature_Selection' / 'output' / 'feature_ranking.json'

# Path alternativo si se usa un CSV diferente
CUSTOM_FEATURE_RANKING = None  # Dejar en None para usar automático


def load_selected_features():
    """
    Carga automáticamente las características seleccionadas por el módulo 06-C.

    Returns:
        tuple: (feature_names, feature_metadata, selection_config)

    Raises:
        FileNotFoundError: Si el JSON no existe
        ValueError: Si el JSON tiene formato inválido
    """
    json_path = CUSTOM_FEATURE_RANKING or FEATURE_RANKING_JSON

    if not json_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de selección de características: {json_path}\n"
            f"Ejecuta primero el módulo 06-C: python 06-C_Feature_Selection/main.py"
        )

    with open(json_path, 'r', encoding='utf-8') as f:
        ranking_data = json.load(f)

    # Extraer nombres de características
    feature_names = [feat['name'] for feat in ranking_data['selected_features']]

    # Metadata completa
    feature_metadata = ranking_data['selected_features']

    # Configuración usada en selección
    selection_config = ranking_data['configuration']

    return feature_names, feature_metadata, selection_config


# ============================================================================
# DATOS DE ENTRADA
# ============================================================================

# Usar el mismo CSV que el módulo 06-C (m7_tau9)
INPUT_DATA_DIR = BASE_DIR.parent / '05_Texture_Descriptors_m7_tau9' / 'output' / 'features' / 'combined'
INPUT_FEATURES_FILE = 'combined_glcm_lbp_wavelet_rqa_statistical_20251204_173601.csv'

# ============================================================================
# CARACTERÍSTICAS SELECCIONADAS (CARGA AUTOMÁTICA)
# ============================================================================

try:
    # Cargar automáticamente desde 06-C
    SELECTED_FEATURES, FEATURE_METADATA, SELECTION_CONFIG = load_selected_features()

    print(f"✓ Características cargadas automáticamente desde 06-C (m7_tau9):")
    print(f"  - Total: {len(SELECTED_FEATURES)} características")
    print(f"  - Configuración: α={SELECTION_CONFIG['alpha']}, "
          f"Cohen's d≥{SELECTION_CONFIG['min_cohens_d']}, "
          f"r<{SELECTION_CONFIG['max_correlation']}")

    # Crear subsets automáticamente basados en ranking
    FEATURE_SUBSETS = {
        'top_5': SELECTED_FEATURES[:5],
        'top_10': SELECTED_FEATURES[:10],
        'all_selected': SELECTED_FEATURES,  # Todas las seleccionadas por 06-C
    }

    # Subset por descriptor (opcional)
    from collections import defaultdict
    by_descriptor = defaultdict(list)
    for feat_meta in FEATURE_METADATA:
        by_descriptor[feat_meta['descriptor']].append(feat_meta['name'])

    # Añadir subsets por descriptor si tienen suficientes características
    for descriptor, features in by_descriptor.items():
        if len(features) >= 3:
            FEATURE_SUBSETS[f'{descriptor}_only'] = features

except FileNotFoundError as e:
    print(f"⚠️  ADVERTENCIA: {e}")
    print(f"   Usando configuración por defecto (vacía)")
    print(f"   Ejecuta: python 06-C_Feature_Selection/main.py")

    SELECTED_FEATURES = []
    FEATURE_METADATA = []
    SELECTION_CONFIG = {}
    FEATURE_SUBSETS = {}

# ============================================================================
# CONFIGURACIONES DE MODELOS
# ============================================================================

MODEL_CONFIGS = {
    'logistic_regression': {
        'random_state': 42,
        'max_iter': 1000,
        'solver': 'liblinear',
        'class_weight': 'balanced'
    },

    'naive_bayes': {
        'var_smoothing': 1e-9
    },

    'knn': {
        'n_neighbors': 5,
        'weights': 'distance',
        'metric': 'euclidean'
    },

    'decision_tree': {
        'random_state': 42,
        'max_depth': 10,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'class_weight': 'balanced'
    },

    'svm': {
        'random_state': 42,
        'kernel': 'rbf',
        'C': 10.0,
        'gamma': 'scale',
        'class_weight': 'balanced',
        'probability': True
    },

    'random_forest': {
        'random_state': 42,
        'n_estimators': 500,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'class_weight': 'balanced',
        'oob_score': True,
        'n_jobs': -1
    },

    'neural_network': {
        'random_state': 42,
        'hidden_layer_sizes': (100, 50),
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.005,
        'learning_rate': 'adaptive',
        'learning_rate_init': 0.001,
        'max_iter': 800,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'n_iter_no_change': 15,
        'batch_size': 'auto',
        'shuffle': True
    },

    'xgboost': {
        'random_state': 42,
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'gamma': 0,
        'reg_alpha': 0.1,
        'reg_lambda': 1,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'n_jobs': -1
    }
}

# ============================================================================
# PARÁMETROS DE VALIDACIÓN
# ============================================================================

VALIDATION_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'stratify': True,
    'cv_folds': 10,
    'scoring_metrics': [
        'accuracy',
        'precision',
        'recall',
        'f1',
        'roc_auc'
    ]
}

# ============================================================================
# CONFIGURACIÓN DE VISUALIZACIÓN
# ============================================================================

PLOT_CONFIG = {
    'figsize': (12, 8),
    'dpi': 300,
    'style': 'seaborn-v0_8',
    'color_palette': ['#3498db', '#e74c3c'],  # Azul para Normal, Rojo para Pathol
    'font_size': 12
}

# ============================================================================
# CONFIGURACIÓN DE REPORTES
# ============================================================================

REPORT_CONFIG = {
    'decimal_places': 4,
    'include_std': True,
    'save_detailed_results': True,
    'generate_html_report': True,
    'include_feature_justifications': True  # Incluir justificaciones de 06-C
}

# Semillas para reproducibilidad
RANDOM_SEEDS = [42, 123, 456, 789, 999]

# Configuración de logging
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'classification.log'
}
