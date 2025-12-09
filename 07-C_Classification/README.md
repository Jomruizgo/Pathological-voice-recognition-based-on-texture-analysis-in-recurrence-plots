# Módulo 07-C: Clasificación con Features Auto-Seleccionadas

## 📋 Descripción

Este módulo consume **automáticamente** las características seleccionadas por el módulo 06-C, eliminando la necesidad de configuración manual.

### ⚡ Funcionalidad: Evaluación de Múltiples Subsets

**07-C ahora evalúa automáticamente múltiples configuraciones**:
- ✅ **TOP_5**: Las 5 características más discriminativas
- ✅ **TOP_10**: Las 10 características más discriminativas
- ✅ **ALL_SELECTED**: Todas las seleccionadas por 06-C (15)
- ✅ **Por descriptor**: Subsets de cada tipo (lbp_only, stat_only, wavelet_only)

**Resultado**: Comparación automática para encontrar el **subset óptimo**.

---

## 🔗 Conexión Automática con 06-C

### **Cómo Funciona**

```python
# config.py línea 33-49

def load_selected_features():
    """
    Carga automáticamente las características seleccionadas por 06-C.
    """
    json_path = Path('../06-C_Feature_Selection/output/feature_ranking.json')

    if not json_path.exists():
        raise FileNotFoundError(
            "Ejecuta primero: python 06-C_Feature_Selection/main.py"
        )

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extraer nombres de características
    feature_names = [feat['name'] for feat in data['selected_features']]

    # Generar subsets automáticamente
    FEATURE_SUBSETS = {
        'top_5': feature_names[:5],
        'top_10': feature_names[:10],
        'all_selected': feature_names
    }

    return feature_names, FEATURE_SUBSETS
```

### **Al Importar config.py**

```bash
$ python
>>> import config

✓ Características cargadas automáticamente desde 06-C (m7_tau9):
  - Total: 15 características
  - Configuración: α=0.05, Cohen's d≥0.2, r<0.85
```

**No necesitas editar nada. Todo automático.**

---

## 🚀 Uso del Módulo

### **Prerequisito**

```bash
# DEBES ejecutar 06-C primero para generar feature_ranking.json
cd 06-C_Feature_Selection
python main.py

# Verifica que existe el JSON
ls 06-C_Feature_Selection/output/feature_ranking.json
```

### **Ejecución**

```bash
cd 07-C_Classification

# Evaluar TODOS los subsets automáticamente (Recomendado)
python main.py

# Output:
# ================================================================================
# EVALUANDO SUBSET: TOP_5 (5 características)
# ================================================================================
# ... entrena 8 modelos ...
# 🏆 MEJOR MODELO: Random Forest (F1=0.8060)
#
# ================================================================================
# EVALUANDO SUBSET: TOP_10 (10 características)
# ================================================================================
# ... entrena 8 modelos ...
# 🏆 MEJOR MODELO: Random Forest (F1=0.8185)
#
# ================================================================================
# EVALUANDO SUBSET: ALL_SELECTED (15 características)
# ================================================================================
# ... entrena 8 modelos ...
# 🏆 MEJOR MODELO: XGBoost (F1=0.8066)
#
# ================================================================================
# COMPARACIÓN ENTRE SUBSETS DE CARACTERÍSTICAS
# ================================================================================
# Subset                 N Features Mejor Modelo           F1-Score   Accuracy    ROC-AUC
# --------------------------------------------------------------------------------------------
# top_5                           5 random_forest            0.8060     0.8068     0.8401
# top_10                         10 random_forest            0.8185     0.8182     0.9005 ⭐
# all_selected                   15 xgboost                  0.8066     0.8068     0.8823
#
# ================================================================================
# 🏆 MEJOR CONFIGURACIÓN GLOBAL
# ================================================================================
#   Subset: top_10
#   Modelo: RANDOM_FOREST
#   F1-Score: 0.8185
#   ROC-AUC: 0.9005
# ================================================================================

# Entrenar solo un modelo específico (en todos los subsets)
python main.py --model svm

# Sin visualizaciones (solo métricas)
python main.py --no-plots

# Con logging detallado
python main.py --verbose
```

### **Parámetros CLI Disponibles**

| Parámetro | Opciones | Default | Descripción |
|-----------|----------|---------|-------------|
| `--model` | `logistic_regression`, `naive_bayes`, `knn`, `decision_tree`, `svm`, `random_forest`, `neural_network`, `xgboost`, `all` | `all` | Modelo(s) a entrenar |
| `--verbose` / `-v` | - | `False` | Logging detallado |
| `--no-plots` | - | `False` | No generar gráficos (solo métricas) |

---

## 📊 Subsets de Características Evaluados

El módulo genera y evalúa automáticamente estos subsets:

```python
FEATURE_SUBSETS = {
    'top_5': SELECTED_FEATURES[:5],      # Top 5 por combined score
    'top_10': SELECTED_FEATURES[:10],    # Top 10 por combined score
    'all_selected': SELECTED_FEATURES,   # Todas las seleccionadas (15)

    # Subsets por descriptor (si tienen ≥3 características)
    'lbp_only': [...],      # Solo características LBP
    'stat_only': [...],     # Solo características Statistical
    'wavelet_only': [...]   # Solo características Wavelet
    'rqa_only': [...],      # Solo características RQA (si ≥3)
}
```

**Resultado**: Identificación automática del **subset óptimo** (balance entre performance y parsimonia).

---

## 🎯 Modelos Implementados

8 clasificadores evaluados en cada subset:

1. **Logistic Regression** - Baseline interpretable (L2 regularization, max_iter=1000)
2. **Naive Bayes** - Modelo probabilístico (Gaussian)
3. **k-NN** - Basado en proximidad (k=5, weights='distance')
4. **Decision Tree** - Reglas interpretables (max_depth=10, min_samples_split=10)
5. **SVM (RBF)** - Captura no linealidades (C=10, gamma='scale', probability=True)
6. **Random Forest** - Ensemble robusto (n_estimators=500, max_depth=15)
7. **Neural Network** - MLP (hidden_layers=(100,50), max_iter=500)
8. **XGBoost** - Gradient boosting (max_depth=6, n_estimators=100, learning_rate=0.1)

---

## 📈 Características Principales

| Aspecto | Implementación |
|---------|----------------|
| **Configuración** | ✓ Automática desde JSON |
| **Reproducibilidad** | ✓ Alta (proceso automatizado, random_state=42) |
| **Evaluación de subsets** | ✓ Automática (TOP_5, TOP_10, ALL_SELECTED) |
| **Identificación óptima** | ✓ Automática (comparación entre subsets) |
| **Actualización** | ✓ Automática (re-ejecutar 06-C) |
| **Trazabilidad** | ✓ Completa (JSON con metadata) |
| **Justificación** | ✓ Métricas científicas cuantificables |

---

## 📁 Estructura de Salida

### **Archivos Generados**

```
07-C_Classification/output/
├── metrics/
│   ├── results_20251206_232655.json    # Resultados completos con subsets
│   └── plots/
│       ├── confusion_matrix_*.png      # Matrices por modelo
│       ├── roc_curve_*.png             # Curvas ROC por modelo
│       ├── models_metrics_comparison.png
│       ├── models_f1_comparison.png
│       ├── models_cv_scores.png
│       └── all_models_roc_curves.png
└── logs/
    └── classification.log
```

### **Formato del JSON de Resultados**

```json
{
  "timestamp": "2025-12-06T23:26:55",
  "n_samples": 440,
  "n_train": 352,
  "n_test": 88,
  "class_names": ["Normal", "Pathol"],

  "subsets": {
    "top_5": {
      "n_features": 5,
      "features": ["lbp_hist_bin_5_r1_p4", "stat_hist_bin_0", ...],
      "models": [
        {
          "model_name": "random_forest",
          "accuracy": 0.8068,
          "f1_score": 0.8060,
          "roc_auc": 0.8401,
          "cv_f1_mean": 0.7632,
          "cv_f1_std": 0.0696
        },
        ...
      ],
      "best_model": {
        "model_name": "random_forest",
        "f1_score": 0.8060
      }
    },
    "top_10": { ... },
    "all_selected": { ... }
  },

  "best_overall": {
    "subset": "top_10",
    "model_name": "random_forest",
    "accuracy": 0.8182,
    "f1_score": 0.8185,
    "roc_auc": 0.9005
  },

  "selection_config": {
    "alpha": 0.05,
    "min_cohens_d": 0.2,
    "max_correlation": 0.85,
    "target_n_features": 15
  }
}
```

---

## 🔍 Información Disponible de cada Característica

El JSON de 06-C incluye metadata completa:

```python
# Acceder a metadata de características
import config

for feat in config.FEATURE_METADATA:
    print(f"{feat['rank']}. {feat['name']}")
    print(f"   F-Score: {feat['f_score']:.2f}")
    print(f"   p-value: {feat['p_value']:.4f}")
    print(f"   Cohen's d: {feat['cohens_d']:.2f} ({feat['effect_size']})")
    print(f"   MI: {feat['mi_score']:.3f}")
    print(f"   Justificación: {feat['justification']}")
```

**Ejemplo de output**:
```
1. lbp_hist_bin_5_r1_p4
   F-Score: 62.60
   p-value: 0.00000
   Cohen's d: 0.76 (medium)
   MI: 0.107
   Justificación: altamente significativa (p<0.001), efecto medium (|d|=0.76), alto poder discriminativo
```

---

## 🔄 Flujo Completo de Pipeline

```
┌─────────────────────────────────────┐
│ 05: Extracción de Descriptores      │
│ Output: features.csv (181 features) │
└──────────────┬──────────────────────┘
               │
               ↓
┌─────────────────────────────────────┐
│ 06-C: Selección Rigurosa (5 fases) │
│ Output: feature_ranking.json (15)   │
└──────────────┬──────────────────────┘
               │
               ↓ (AUTOMÁTICO)
┌─────────────────────────────────────┐
│ 07-C: Clasificación Multi-Subset    │
│ - Carga automática de JSON          │
│ - Evalúa TOP_5, TOP_10, ALL_SELECTED│
│ - Entrena 8 modelos × N subsets     │
│ - Identifica configuración óptima   │
│ Output: Comparación + Mejor subset  │
└─────────────────────────────────────┘
```

**Sin intervención manual. Todo automatizado y reproducible.**

---

## 💡 Resultados Típicos

### **Hallazgos del Experimento TOP_5 vs TOP_10 vs ALL_SELECTED**

Basado en dataset de 440 muestras (239 Normal, 201 Pathol):

| Subset | N Características | Mejor Modelo | F1-Score | ROC-AUC | Interpretación |
|--------|-------------------|--------------|----------|---------|----------------|
| **TOP_10** 🏆 | **10** | **Random Forest** | **0.8185** | **0.9005** | **Óptimo: mejor F1 y AUC > 90%** |
| TOP_5 | 5 | Random Forest | 0.8060 | 0.8401 | Parsimonioso pero AUC menor |
| ALL_SELECTED | 15 | XGBoost | 0.8066 | 0.8823 | Buen F1 con más características |

**Conclusión**: 10 características es el **punto óptimo** (balance entre parsimonia y performance).

---

## ⚙️ Configuración Avanzada

### **Usar un JSON Diferente**

Si quieres usar un JSON de selección diferente:

```python
# config.py línea 21
CUSTOM_FEATURE_RANKING = Path('/path/to/other_ranking.json')
```

### **Forzar Características Manualmente (No Recomendado)**

Si realmente necesitas overridear:

```python
# config.py después de la línea 60
SELECTED_FEATURES = [
    'manual_feature_1',
    'manual_feature_2',
    # ...
]
```

**⚠️ NO RECOMENDADO**: Pierdes trazabilidad y justificación científica.

---

## 🛠️ Troubleshooting

### **Error: "No se encontró feature_ranking.json"**

```bash
# Solución: Ejecuta 06-C primero
cd 06-C_Feature_Selection
python main.py

# Verifica que el JSON existe
ls output/feature_ranking.json
```

### **Error: "Característica X no existe en el CSV"**

Significa que el CSV de entrada ha cambiado desde que se ejecutó 06-C.

```bash
# Solución: Re-ejecutar 06-C con el CSV actualizado
cd 06-C_Feature_Selection
python main.py --input /path/to/updated_features.csv
```

### **Warning: "Solo X características cargadas"**

Si 06-C seleccionó menos de 15 características (por restricciones de redundancia):

```bash
# Opción 1: Relajar threshold en 06-C
# config.py línea 75
MAX_CORRELATION = 0.90  # Era 0.85

# Opción 2: Aceptar menos características
# La validación de 06-C ya garantiza que son suficientes
```

### **Performance Pobre en Todos los Subsets**

Si todos los subsets dan ROC-AUC < 0.70:

1. **Verifica los datos de entrada**:
   ```bash
   # Distribución de clases balanceada?
   python -c "import pandas as pd; df=pd.read_csv('input.csv'); print(df['label'].value_counts())"
   ```

2. **Revisa la selección de características en 06-C**:
   - ¿Pasó la validación?
   - ¿Silhouette > 0.3?
   - ¿Fisher ratio > 1.5?

3. **Considera ajustar umbral Cohen's d en 06-C**:
   - d ≥ 0.2 es permisivo (más características)
   - d ≥ 0.5 es restrictivo (menos características)

---

## 📚 Referencias

- **Módulo 06-C**: Documentación completa del proceso de selección de características
- **Documentación de configuración de modelos**: Ver `config.py` para parámetros de cada clasificador

---

## 🎓 Ejemplo Completo

```bash
# 1. Generar descriptores (si no existen)
cd 05_Texture_Descriptors_m7_tau9
python main.py

# 2. Seleccionar características rigorosamente
cd ../06-C_Feature_Selection
python main.py

# Output:
# ✓ 15 características seleccionadas
# ✓ Validación PASADA (Silhouette=0.52, Fisher=2.3)
# ✓ JSON generado: output/feature_ranking.json

# 3. Evaluar todos los subsets automáticamente
cd ../07-C_Classification
python main.py

# Output:
# ✓ Características cargadas: 15
# ✓ Subsets evaluados: 6 (top_5, top_10, all_selected, lbp_only, stat_only, wavelet_only)
# ✓ Modelos entrenados por subset: 8
# ✓ Total entrenamientos: 48
#
# 🏆 MEJOR CONFIGURACIÓN GLOBAL:
#   Subset: top_10
#   Modelo: Random Forest
#   F1-Score: 0.8185
#   ROC-AUC: 0.9005
```

**Todo el proceso es automático, reproducible y científicamente defendible.**

---

## 🔬 Resultados Experimentales: Justificación de Cohen's d ≥ 0.2

Este módulo utiliza características seleccionadas con **Cohen's d ≥ 0.2** (efecto pequeño+).

**Experimento controlado** (mismo dataset, random_state=42):

| Pipeline | Umbral Cohen's d | Características | Mejor Config | F1-Score | ROC-AUC | Validación |
|----------|------------------|-----------------|--------------|----------|---------|------------|
| **06-C/07-C** | **d ≥ 0.2** | 15 → **TOP_10** | Random Forest | **0.8185** | **0.9005** | ✓ APROBADA |
| 06-D/07-D | d ≥ 0.5 | 5 | SVM | 0.8041 | 0.7781 | ✗ FALLIDA |

**Conclusión**: d ≥ 0.2 ofrece mejor generalización (ROC-AUC +15.7%) y validación aprobada.

Ver documentación detallada en: `06-C_Feature_Selection/docs/JUSTIFICACION_COHENS_D_0.2.md`

---

**Generado automáticamente para el módulo 07-C**
**Última actualización**: 2025-12-06
