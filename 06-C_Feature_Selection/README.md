# Módulo 06-C: Selección Rigurosa de Características

## 📋 Descripción

Este módulo implementa un **proceso científicamente defendible de 5 fases** para seleccionar características óptimas para clasificación de voces normales vs patológicas.

**06-C genera automáticamente un JSON** con las características seleccionadas que el módulo 07-C consume sin intervención manual, permitiendo un pipeline completamente automatizado y reproducible.

## 🎯 Objetivo

Eliminar la **desconexión manual** entre análisis (módulo 06) y clasificación (módulo 07), implementando un pipeline automático y reproducible basado en:

1. **Significancia estadística** (p-value)
2. **Relevancia práctica** (Cohen's d)
3. **Poder discriminativo** (F-Score + Mutual Information)
4. **Eliminación de redundancia** (correlación)
5. **Validación de separabilidad** (PCA, Silhouette, Fisher)

---

## 🔬 Las 5 Fases del Pipeline

### **Fase 1: Filtrado por Significancia Estadística**

**Objetivo**: Eliminar características cuyas diferencias entre clases podrían ser azar.

**Método**: F-statistic ANOVA (sklearn.feature_selection.f_classif)

**Criterio**:
```python
p-value < α = 0.05
```

**Interpretación**:
- **p < 0.05**: Hay evidencia estadística suficiente de que la característica discrimina
- **p ≥ 0.05**: No hay evidencia (diferencia podría ser azar) → **RECHAZADA**

**Ejemplo**:
```
Característica A: F=8.5, p=0.003  → ✓ Pasa (evidencia estadística)
Característica B: F=2.1, p=0.082  → ✗ Rechazada (sin evidencia)
```

---

### **Fase 2: Filtrado por Relevancia Práctica**

**Objetivo**: Eliminar características estadísticamente significativas pero con efecto trivial.

**Método**: Cohen's d (tamaño del efecto)

**Fórmula**:
```
Cohen's d = (μ₁ - μ₂) / σ_pooled

donde σ_pooled = √[((n₁-1)σ₁² + (n₂-1)σ₂²) / (n₁+n₂-2)]
```

**Criterio**:
```python
|Cohen's d| ≥ 0.2  (efecto pequeño o superior)
```

**Interpretación** (Cohen, 1988):
- **|d| < 0.2**: Efecto **trivial** (despreciable) → **RECHAZADA**
- **0.2 ≤ |d| < 0.5**: Efecto **pequeño** (aceptable) → ✓ **ACEPTADA**
- **0.5 ≤ |d| < 0.8**: Efecto **mediano** (ideal) → ✓ **ACEPTADA**
- **|d| ≥ 0.8**: Efecto **grande** (ideal) → ✓ **ACEPTADA**

**¿Por Qué Es Importante?**

```
Ejemplo problemático:
- Característica: lbp_feature_X
- Normal:     μ=0.023, σ=0.008
- Patológica: μ=0.025, σ=0.008
- Diferencia: 0.002 (2 milésimas)
- Con n=500: p=0.003 (significativo ✓)
- Cohen's d: 0.25 (efecto pequeño ✗)

Conclusión: Diferencia es REAL (no azar), pero TAN PEQUEÑA que:
  1. El clasificador tendrá dificultad usándola
  2. Es sensible a ruido de medición
  3. No aporta poder discriminativo práctico
```

---

### **Fase 3: Ranking por Poder Discriminativo Combinado**

**Objetivo**: Rankear características considerando relaciones lineales Y no lineales.

**Métodos**:
1. **F-Score ANOVA**: Captura diferencias de medias (lineal)
2. **Mutual Information**: Captura dependencias no lineales

**Score Combinado**:
```python
# Normalizar ambos a [0, 1]
F_norm = (F - F_min) / (F_max - F_min)
MI_norm = (MI - MI_min) / (MI_max - MI_min)

# Score combinado (70% F-Score, 30% MI)
Combined = 0.7 × F_norm + 0.3 × MI_norm
```

**Justificación de pesos**:
- **70% F-Score**: Características de textura suelen tener relaciones lineales con la clase
- **30% MI**: Captura relaciones no lineales que F-Score pierde
- Ajustable según dominio (usa 50-50 si esperas fuertes no linealidades)

---

### **Fase 4: Eliminación de Redundancia** ⚠️ **CRÍTICO**

**Objetivo**: Evitar multicolinealidad seleccionando características complementarias.

**Algoritmo Greedy**:
```python
selected = []
for feature in ranking (ordenado por Combined Score):
    # Calcular correlación con características ya seleccionadas
    max_corr = max(|r(feature, s)| for s in selected)

    if max_corr < 0.85:  # Threshold
        selected.append(feature)
    else:
        RECHAZAR (redundante)
```

**Criterio**:
```python
|r| < 0.85  (correlación de Pearson)
```

**Justificación (VIF - Variance Inflation Factor)**:
```
VIF = 1 / (1 - r²)

Con r=0.85: VIF = 1/(1-0.72) = 3.57  (tolerable, límite es ~5)
Con r=0.90: VIF = 1/(1-0.81) = 5.26  (problemático)

Threshold 0.85 = Balance entre diversidad y complementariedad
```

**¿Por Qué Es Crítico?**

```
Ejemplo de problema de redundancia:
- lbp_hist_bin_5_r1_p4   F=62.6, Combined=0.90
- lbp_hist_bin_0_r1_p4   F=56.1, Combined=0.86
- Correlación entre ellas: r=0.97 (muy alta!)

Sin Fase 4:
  → Seleccionas AMBAS
  → Aportan esencialmente la misma información
  → Multicolinealidad en el modelo
  → Desperdicio de 1 slot de tus 15 características

Con Fase 4:
  → Seleccionas lbp_hist_bin_5_r1_p4 (rank 1)
  → Rechazas lbp_hist_bin_0_r1_p4 (redundante con anterior)
  → Usas ese slot para una característica complementaria
```

---

### **Fase 5: Validación de Separabilidad**

**Objetivo**: Verificar que el subset seleccionado realmente separa bien las clases.

**Métricas**:

1. **Varianza Explicada por PCA**
   ```python
   PCA con k componentes → Σ explained_variance_ratio
   Criterio: > 80%
   ```
   Interpretación: Las características capturan suficiente información

2. **Silhouette Score**
   ```python
   Silhouette = (b - a) / max(a, b)
   donde:
     a = distancia promedio intra-cluster
     b = distancia promedio inter-cluster

   Criterio: > 0.3 (aceptable), > 0.5 (bueno), > 0.7 (excelente)
   ```
   Interpretación: Qué tan bien separadas están las clases

3. **Fisher Ratio**
   ```python
   Fisher = distancia_inter_clase / distancia_intra_clase
   Criterio: > 1.5
   ```
   Interpretación: Clases más separadas que dispersas internamente

---

## 🚀 Uso del Módulo

### **Instalación de Dependencias**

```bash
pip install pandas numpy scikit-learn scipy
```

### **Ejecución**

```bash
# Desde la raíz del proyecto
cd 06-C_Feature_Selection

# Pipeline completo
python main.py

# Con logging detallado
python main.py --verbose

# Especificar archivo de entrada
python main.py --input /path/to/features.csv
```

### **Salidas Generadas**

```
06-C_Feature_Selection/output/
├── feature_ranking.json          # ← USADO POR 07-C AUTOMÁTICAMENTE
├── selection_report.md           # Reporte detallado en Markdown
└── feature_selection.log         # Log completo del proceso
```

---

## 📊 Formato del JSON de Salida

```json
{
  "metadata": {
    "timestamp": "2025-08-12T17:30:00",
    "pipeline_version": "1.0.0"
  },
  "configuration": {
    "alpha": 0.05,
    "min_cohens_d": 0.5,
    "weight_f_score": 0.7,
    "weight_mi": 0.3,
    "max_correlation": 0.85,
    "target_n_features": 15
  },
  "selected_features": [
    {
      "rank": 1,
      "name": "lbp_hist_bin_5_r1_p4",
      "descriptor": "lbp",
      "f_score": 62.600,
      "p_value": 2.09e-14,
      "cohens_d": -0.757,
      "effect_size": "medium",
      "mi_score": 0.107,
      "combined_score": 0.900,
      "justification": "altamente significativa (p<0.001), efecto medium (|d|=0.76), alto poder discriminativo"
    },
    ...
  ],
  "validation": {
    "pca_variance_explained": 0.87,
    "silhouette_score": 0.52,
    "fisher_ratio": 2.3,
    "validation_passed": true
  }
}
```

---

## 🔗 Conexión con Módulo 07-C

El módulo **07-C** consume automáticamente el JSON:

```python
# 07-C_Classification/config.py

def load_selected_features():
    """Carga automáticamente desde 06-C."""
    with open('06-C_Feature_Selection/output/feature_ranking.json') as f:
        data = json.load(f)

    return [feat['name'] for feat in data['selected_features']]

# Carga automática al importar config
SELECTED_FEATURES = load_selected_features()
```

**No más configuración manual. Todo automático y reproducible.**

---

## 📈 Parámetros Configurables

En `config.py`:

| Parámetro | Valor Default | Descripción |
|-----------|---------------|-------------|
| `ALPHA` | 0.05 | Nivel de significancia (95% confianza) |
| `MIN_COHENS_D` | 0.2 | Efecto mínimo (pequeño+) |
| `WEIGHT_F_SCORE` | 0.7 | Peso F-Score en ranking |
| `WEIGHT_MI_SCORE` | 0.3 | Peso MI en ranking |
| `MAX_CORRELATION` | 0.85 | Threshold redundancia |
| `TARGET_N_FEATURES` | 15 | Objetivo de características |
| `MIN_PCA_VARIANCE` | 0.80 | Mínima varianza explicada |
| `MIN_SILHOUETTE_SCORE` | 0.30 | Mínima separabilidad |
| `MIN_FISHER_RATIO` | 1.5 | Mínimo ratio inter/intra |

---

## 📚 Referencias Científicas

1. **Fisher, R.A. (1925)**. *Statistical Methods for Research Workers*.
   - Fundamento del test F-ANOVA

2. **Cohen, J. (1988)**. *Statistical Power Analysis for the Behavioral Sciences (2nd ed.)*.
   - Definición de tamaños de efecto (Cohen's d)

3. **Kutner et al. (2004)**. *Applied Linear Statistical Models*.
   - Variance Inflation Factor (VIF) y multicolinealidad

4. **Rousseeuw, P.J. (1987)**. *Silhouettes: A graphical aid to the interpretation and validation of cluster analysis*.
   - Silhouette score para validación

---

## ⚙️ Flujo Completo del Pipeline

```
ENTRADA: features.csv (181 características)
    ↓
┌─────────────────────────────────────────────┐
│ FASE 1: Significancia Estadística          │
│ Filtro: p < 0.05                            │
│ Output: 79 características                  │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ FASE 2: Relevancia Práctica                │
│ Filtro: |Cohen's d| ≥ 0.2                   │
│ Output: 71 características                  │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ FASE 3: Ranking Discriminativo              │
│ Score: 0.7×F_norm + 0.3×MI_norm             │
│ Output: Características rankeadas           │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ FASE 4: Eliminación de Redundancia          │
│ Filtro: |r| < 0.85 (greedy)                 │
│ Output: 15 características                  │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ FASE 5: Validación de Separabilidad         │
│ Verifica: PCA > 80%, Silhouette > 0.3       │
│ Output: feature_ranking.json                │
└─────────────────────────────────────────────┘
    ↓
SALIDA: 15 características optimizadas
        (listas para clasificación en 07-C)
```

---

## 🎓 Características Principales del Pipeline

| Aspecto | Implementación en 06-C |
|---------|------------------------|
| **Significancia estadística** | ✓ Filtro por p-value < 0.05 (F-statistic ANOVA) |
| **Relevancia práctica** | ✓ Filtro por Cohen's d ≥ 0.2 (efecto pequeño+) |
| **Poder discriminativo** | ✓ Ranking combinado: 70% F-Score + 30% Mutual Information |
| **Eliminación de redundancia** | ✓ Filtro activo por correlación (r < 0.85, VIF < 3.6) |
| **Validación de separabilidad** | ✓ Métricas cuantitativas (PCA variance, Silhouette, Fisher ratio) |
| **Salida** | ✓ JSON estructurado con metadata completa + Reporte Markdown |
| **Conexión con clasificación** | ✓ Integración automática con módulo 07-C (sin configuración manual) |
| **Reproducibilidad** | ✓ Proceso completamente automatizado (random_state, thresholds fijos) |
| **Justificación científica** | ✓ Métricas cuantificables para cada característica seleccionada |

---

## 🔍 Ejemplo de Ejecución

```bash
$ python main.py

================================================================================
MÓDULO 06-C: SELECCIÓN RIGUROSA DE CARACTERÍSTICAS
================================================================================

✓ Características cargadas automáticamente desde 06-C:
  - Total: 181 características iniciales

================================================================================
FASE 1: FILTRADO POR SIGNIFICANCIA ESTADÍSTICA
================================================================================
Total características: 181
✓ Significativas (p < 0.05): 79
✗ Rechazadas (p ≥ 0.05): 102

================================================================================
FASE 2: FILTRADO POR RELEVANCIA PRÁCTICA (COHEN'S D)
================================================================================
Entrada: 79 características
✓ Relevantes (|d| ≥ 0.2): 71
✗ Rechazadas (|d| < 0.2): 8

Distribución de tamaños de efecto:
  - small: 58
  - medium: 13

================================================================================
FASE 3: RANKING POR PODER DISCRIMINATIVO
================================================================================
Top 10 características:
  1. lbp_hist_bin_5_r1_p4                 | Combined=0.900 | F= 62.60 | MI=0.107
  2. lbp_hist_bin_0_r1_p4                 | Combined=0.858 | F= 56.09 | MI=0.103
  ...

================================================================================
FASE 4: ELIMINACIÓN DE REDUNDANCIA
================================================================================
✓ lbp_hist_bin_5_r1_p4                   | Primera característica
✗ lbp_hist_bin_0_r1_p4                   | REDUNDANTE: r=0.97 con lbp_hist_bin_5_r1_p4
✓ stat_hist_bin_0                        | max_r=0.38 con lbp_hist_bin_5_r1_p4
...

Características seleccionadas: 15/15

================================================================================
FASE 5: VALIDACIÓN DE SEPARABILIDAD
================================================================================
  PCA varianza: 87% (mín: 80%)
  Silhouette: 0.52 (Buena separabilidad)
  Fisher ratio: 2.3 (mín: 1.5)

✓ VALIDACIÓN EXITOSA: Subset de características es apropiado

================================================================================
✓ PIPELINE COMPLETADO EXITOSAMENTE
================================================================================

Archivos generados:
  - JSON: output/feature_ranking.json
  - Markdown: output/selection_report.md
  - Log: feature_selection.log
```

---

## 💡 Preguntas Frecuentes

### ¿Por qué necesito ambos Cohen's d y p-value?

- **p-value**: "¿La diferencia es real o azar?" (significancia **estadística**)
- **Cohen's d**: "¿La diferencia importa en la práctica?" (significancia **práctica**)

Con muestras grandes, diferencias microscópicas dan p < 0.05 pero d < 0.2 (efecto trivial). Necesitas **ambos**.

### ¿Por qué 70% F-Score y 30% MI?

F-Score captura relaciones lineales (diferencias de medias). MI captura relaciones no lineales. Las características de textura suelen ser lineales, por eso favorecemos F-Score. Ajusta los pesos si tu dominio es diferente.

### ¿Qué pasa si no alcanzo 15 características?

Si después de eliminar redundancia tienes menos de 15, el algoritmo se detiene. Puedes relajar `MAX_CORRELATION` a 0.90 en config.py.

### ¿Puedo usar un CSV diferente?

Sí. Especifica con `--input`:
```bash
python main.py --input /path/to/other_features.csv
```

---

## 🛠️ Troubleshooting

**Error: "Archivo de entrada no encontrado"**
```bash
# Solución: Ejecuta primero el módulo 05
cd 05_Texture_Descriptors
python main.py
```

**Warning: "Validación FALLIDA"**
```
Significa que el subset no cumple todos los criterios (PCA < 80%, Silhouette < 0.3, o Fisher < 1.5).

Soluciones:
1. Relajar MIN_COHENS_D a 0.4 para incluir más características
2. Relajar MAX_CORRELATION a 0.90 para permitir más redundancia
3. Revisar si los datos tienen suficiente poder discriminativo
```

---

**Generado automáticamente por el pipeline 06-C**
