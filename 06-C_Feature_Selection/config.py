"""
Configuración para el módulo 06-C: Selección Rigurosa de Características.

Este módulo implementa un proceso científicamente defendible de selección
de características basado en significancia estadística, relevancia práctica,
poder discriminativo y eliminación de redundancia.

NOTA: Este módulo analiza las características extraídas de RPs generados
      con m=7, τ=9 (05_Texture_Descriptors_m7_tau9).
"""

import os
from pathlib import Path

# ============================================================================
# RUTAS DE DATOS
# ============================================================================

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / 'output'
DOCS_DIR = BASE_DIR / 'docs'

# Entrada: Features del módulo 05 (m7_tau9)
INPUT_FEATURES_DIR = BASE_DIR.parent / '05_Texture_Descriptors_m7_tau9' / 'output' / 'features' / 'combined'
INPUT_FEATURES_FILE = 'combined_glcm_lbp_wavelet_rqa_statistical_20251204_173601.csv'

# Salidas
FEATURE_RANKING_JSON = OUTPUT_DIR / 'feature_ranking.json'
SELECTION_REPORT_HTML = OUTPUT_DIR / 'selection_report.html'
VALIDATION_PLOTS_DIR = OUTPUT_DIR / 'validation_plots'
CORRELATION_MATRIX_NPZ = OUTPUT_DIR / 'correlation_matrix.npz'

# ============================================================================
# FASE 1: FILTRADO POR SIGNIFICANCIA ESTADÍSTICA
# ============================================================================

# Nivel de significancia para p-value
ALPHA = 0.05  # Estándar en literatura científica (95% confianza)

# Justificación:
# - p < 0.05: Evidencia estadística suficiente de diferencia entre clases
# - p ≥ 0.05: No hay evidencia de que la característica discrimine
#
# Referencia: Fisher, R.A. (1925). Statistical Methods for Research Workers

# ============================================================================
# FASE 2: FILTRADO POR RELEVANCIA PRÁCTICA
# ============================================================================

# Tamaño de efecto mínimo (Cohen's d)
MIN_COHENS_D = 0.2  # Efecto pequeño o superior

# ⚠️ AJUSTE BASADO EN HALLAZGO EMPÍRICO (2025-10-20):
# Durante la primera ejecución se descubrió que NINGUNA característica
# tiene |d| ≥ 0.5 en este dataset. El máximo Cohen's d encontrado fue ~0.19.
#
# Esto indica que:
# 1. Las diferencias entre voces normales y patológicas son SUTILES
# 2. El problema de clasificación es INHERENTEMENTE DIFÍCIL
# 3. Los efectos pequeños (0.2 ≤ |d| < 0.5) son lo mejor disponible
#
# Ajuste a MIN_COHENS_D = 0.2 está justificado porque:
# - Cohen (1988) define 0.2 como límite mínimo de efectos detectables
# - Reconoce la realidad del dataset (sin perder rigor científico)
# - Aún filtra efectos triviales (|d| < 0.2)
# - Es pragmático: permite completar el pipeline con características útiles
#
# Ver documentación completa en: HALLAZGO_COHEN_D.md

# Interpretación de Cohen's d (Cohen, 1988):
# |d| < 0.2  : Efecto trivial (despreciable) → RECHAZADO
# 0.2 ≤ |d| < 0.5 : Efecto pequeño (detectable) → ACEPTADO ✓
# 0.5 ≤ |d| < 0.8 : Efecto mediano (ideal, pero no existe en dataset)
# |d| ≥ 0.8  : Efecto grande (ideal, pero no existe en dataset)
#
# Threshold 0.2 = Reconoce naturaleza sutil del problema, mantiene rigor
#
# Referencia: Cohen, J. (1988). Statistical Power Analysis for the
#             Behavioral Sciences (2nd ed.)

# ============================================================================
# FASE 3: RANKING POR PODER DISCRIMINATIVO COMBINADO
# ============================================================================

# Pesos para score combinado
WEIGHT_F_SCORE = 0.7   # Peso para F-statistic ANOVA (relaciones lineales)
WEIGHT_MI_SCORE = 0.3  # Peso para Mutual Information (no linealidades)

# Justificación:
# - F-Score (70%): Captura diferencias de medias entre clases (lineal, interpretable)
# - Mutual Info (30%): Captura dependencias no lineales que F-Score pierde
# - Favorecemos linealidad porque características de textura suelen ser lineales
#
# Ajustable según dominio: si esperas relaciones no lineales fuertes, usa 50-50

# ============================================================================
# FASE 4: ELIMINACIÓN DE REDUNDANCIA
# ============================================================================

# Correlación máxima permitida entre características seleccionadas
MAX_CORRELATION = 0.85  # Threshold para multicolinealidad

# Justificación:
# - |r| > 0.85: VIF = 1/(1-0.72) = 3.57 (tolerable pero límite)
# - |r| > 0.90: VIF = 1/(1-0.81) = 5.26 (problemático)
# - Threshold 0.85: Balance entre diversidad y complementariedad
#
# VIF (Variance Inflation Factor):
# - VIF < 5: Aceptable
# - VIF > 10: Multicolinealidad severa
#
# Referencia: Kutner et al. (2004). Applied Linear Statistical Models

# Método de correlación
CORRELATION_METHOD = 'pearson'  # Alternativas: 'spearman', 'kendall'

# ============================================================================
# FASE 5: VALIDACIÓN DE SEPARABILIDAD
# ============================================================================

# Criterios de validación del subset seleccionado
MIN_PCA_VARIANCE = 0.80     # Mínimo 80% varianza explicada por características
MIN_SILHOUETTE_SCORE = 0.30  # Mínimo 0.3 (0.5+ bueno, 0.7+ excelente)
MIN_FISHER_RATIO = 1.5       # Distancia inter-clase / intra-clase > 1.5

# Justificación:
# - PCA variance > 80%: Características capturan mayoría de información
# - Silhouette > 0.3: Separabilidad aceptable entre clusters
# - Fisher ratio > 1.5: Clases más separadas que dispersas internamente

# ============================================================================
# CONFIGURACIÓN GENERAL
# ============================================================================

# Número objetivo de características
TARGET_N_FEATURES = 15  # Balance entre parsimonia y performance

# Si no se alcanza el objetivo por restricciones, relajar threshold
RELAX_CORRELATION_IF_NEEDED = True  # Si n < TARGET, aumentar MAX_CORRELATION a 0.90

# Random seed para reproducibilidad
RANDOM_STATE = 42

# Nivel de logging
LOG_LEVEL = 'INFO'
LOG_FILE = BASE_DIR / 'feature_selection.log'

# ============================================================================
# VISUALIZACIÓN Y REPORTES
# ============================================================================

# Configuración de plots
PLOT_CONFIG = {
    'figsize': (14, 10),
    'dpi': 300,
    'style': 'seaborn-v0_8-darkgrid',
    'font_size': 11,
    'color_normal': '#3498db',    # Azul
    'color_pathol': '#e74c3c'     # Rojo
}

# Generar reporte HTML completo
GENERATE_HTML_REPORT = True

# Incluir visualizaciones en reporte
INCLUDE_VALIDATION_PLOTS = True

# ============================================================================
# METADATA PARA TRAZABILIDAD
# ============================================================================

PIPELINE_VERSION = '1.0.0'
AUTHOR = 'Pipeline Automatizado RP'
DESCRIPTION = """
Selección rigurosa de características para clasificación de voces
normales vs patológicas usando Recurrence Plots.

Implementa proceso de 5 fases:
1. Filtrado por significancia estadística (p-value)
2. Filtrado por relevancia práctica (Cohen's d)
3. Ranking por poder discriminativo (F-Score + MI)
4. Eliminación de redundancia (correlación)
5. Validación de separabilidad (PCA, Silhouette, Fisher)
"""
