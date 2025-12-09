# Módulo 05: Extracción de Descriptores de Textura (m=7, τ=9)

## Descripción

Este módulo analiza los Recurrence Plots generados con parámetros **m=7** y **τ=9** (módulo 04_RP_Generator_m7_tau9) y extrae características de textura multiescala para la clasificación binaria de voces normales y patológicas.

## Dataset Procesado

- **Total de RPs**: 440 (239 Normal, 201 Patológica)
- **Parámetros RP**: m=7, τ=9, ε=10%
- **Tamaño de imágenes**: ~25,000 × 25,000 píxeles
- **Formato**: PNG, escala de grises

## Descriptores Implementados

El módulo extrae **181 características** de 5 familias de descriptores:

### 📊 Descriptores Activos (Usados en Análisis Final)

1. **GLCM (Gray-Level Co-occurrence Matrix)** - 6 características
   - Propiedades de co-ocurrencia espacial
   - Distancia: [1], Ángulos: [0°, 45°, 90°, 135°]
   - Propiedades: contraste, disimilaridad, homogeneidad, energía, correlación, ASM
   - Niveles de gris: 256

2. **LBP (Local Binary Patterns)** - 14 características
   - Patrones binarios locales uniformes
   - Radio: [1], Puntos: [4]
   - Método: 'uniform' (patrones con ≤2 transiciones)
   - Características: histogramas, ratios, entropía, uniformidad

3. **Wavelet** - 65 características
   - Descomposición multi-escala (db4, 3 niveles)
   - Características: energía, entropía, media, desviación estándar
   - Subbandas: Aproximación (A) + Detalles (H, V, D) por nivel
   - Ratios direccionales: horizontal, vertical, diagonal

4. **RQA (Recurrence Quantification Analysis)** - 10 características
   - Cuantificación de recurrencia dinámica
   - Métricas: RR, DET, LAM, L_max, L_mean, V_max, V_mean, ENTR, DIV, TT
   - Epsilon: adaptativo (percentil 10)
   - Longitud mínima de línea: 2

5. **Statistical** - 86 características
   - Momentos estadísticos: media, std, skewness, kurtosis
   - Percentiles: [10, 25, 50, 75, 90]
   - Histogramas: 64 bins
   - Entropía de Shannon

### 🔒 Descriptores Disponibles pero NO Usados

- **Gabor**: Calculado pero excluido del análisis final
- **Tamura**: Calculado pero excluido del análisis final

## Características del Sistema

### 🔧 Sistema Modular e Incremental
- Cada descriptor se calcula y almacena **independientemente**
- Añade nuevos descriptores **sin recalcular** los existentes
- Reutilización automática si la configuración no cambió
- Sistema de checkpoints parciales para procesos interrumpibles

### ⚙️ Configuración Centralizada
- Configuración en `config.py` tiene prioridad sobre defaults
- Optimizado específicamente para Recurrence Plots
- Sistema de detección de cambios mediante hash de configuración

## Uso

### Línea de Comandos

```bash
# Calcular todos los descriptores habilitados en config.py
python main.py

# Solo descriptores específicos
python main.py --descriptors glcm lbp wavelet rqa statistical

# Calcular solo descriptores faltantes (incremental)
python main.py --descriptors glcm lbp wavelet

# Ver estado del sistema
python main.py --checkpoint-info

# Interfaz gráfica (si disponible)
python main.py --gui
```

### Combinación de Descriptores

Para generar el archivo combinado usado en el análisis:

```bash
python main.py --descriptors glcm lbp wavelet rqa statistical
```

Esto genera: `combined_glcm_lbp_wavelet_rqa_statistical_YYYYMMDD_HHMMSS.csv`

## Estructura de Salida

```
output/
├── features/
│   ├── by_descriptor/                   # Características por descriptor
│   │   ├── glcm/
│   │   │   ├── features.csv            # 440 muestras × 6 características
│   │   │   ├── metadata.json           # Configuración y estadísticas
│   │   │   └── partial_checkpoint.json # Checkpoint (si existe)
│   │   ├── lbp/
│   │   │   └── features.csv            # 440 muestras × 14 características
│   │   ├── wavelet/
│   │   │   └── features.csv            # 440 muestras × 65 características
│   │   ├── rqa/
│   │   │   └── features.csv            # 440 muestras × 10 características
│   │   ├── statistical/
│   │   │   └── features.csv            # 440 muestras × 86 características
│   │   ├── gabor/                      # NO usado en análisis final, por cuestión de recurso computacional
│   │   └── tamura/                     # NO usado en análisis final, por cuestión de recurso computacional
│   ├── combined/                        # Características combinadas
│   │   └── combined_glcm_lbp_wavelet_rqa_statistical_20251204_173601.csv
│   └── manifest.json                    # Estado global del sistema
└── checkpoints/                         # Checkpoints de procesamiento
```

## Dataset Final Generado

**Archivo**: `combined_glcm_lbp_wavelet_rqa_statistical_20251204_173601.csv`

- **Dimensiones**: 440 muestras × 181 características
- **Distribución de clases**:
  - Normal: 239 (54.3%)
  - Patológica: 201 (45.7%)
- **Características por descriptor**:
  - GLCM: 6 (3.3%)
  - LBP: 14 (7.7%)
  - Wavelet: 65 (35.9%)
  - RQA: 10 (5.5%)
  - Statistical: 86 (47.5%)

## Sistema de Checkpoints

El módulo implementa un sistema robusto de checkpoints:

### Niveles de Checkpoint

1. **manifest.json**: Rastrea descriptores calculados y estado global
2. **Checkpoints parciales**: Guardan progreso cada N imágenes (configurable)
3. **Detección de cambios**: Recalcula automáticamente si cambió la configuración

### Configuración de Checkpoints

```python
CHECKPOINT_BATCH_SIZE = 5              # Guardar cada 5 imágenes
ENABLE_PARTIAL_CHECKPOINTS = True      # Habilitar checkpoints granulares
CHECKPOINT_FREQUENCY = 10              # Frecuencia global
```

## Flujo de Trabajo

### Primera Ejecución
1. Lee RPs de `../04_RP_Generator_m7_tau9/output/Recurrence_Plots/`
2. Calcula descriptores habilitados en `config.py`
3. Guarda CSVs individuales en `by_descriptor/`
4. Genera manifest.json con metadata

### Ejecuciones Posteriores
1. Verifica manifest.json
2. Compara hash de configuración
3. **Si NO cambió**: Reutiliza descriptores existentes
4. **Si cambió**: Recalcula solo el descriptor modificado
5. **Si hay nuevas imágenes**: Procesa solo las nuevas (incremental)

### Generación de Combinado
1. Lee CSVs individuales de `by_descriptor/`
2. Verifica mismo orden de muestras
3. Concatena horizontalmente
4. Guarda en `combined/` con timestamp

## Configuración de Procesamiento

### Imágenes
```python
IMAGE_MIN_SIZE = (2000, 2000)      # Tamaño mínimo aceptado
IMAGE_MAX_SIZE = (25000, 25000)    # Tamaño máximo (RPs ~25,000×25,000)
IMAGE_TARGET_DTYPE = 'uint8'       # Tipo de datos
IMAGE_NORMALIZE_RANGE = (0, 255)   # Rango de normalización
```

### Paralelización
```python
ENABLE_PARALLEL = True             # Activar procesamiento paralelo
N_JOBS = -1                        # Usar todos los cores disponibles
```

## Integración con Pipeline

### Entrada
- **Fuente**: `../04_RP_Generator_m7_tau9/output/Recurrence_Plots/`
- **Archivos**: `Normal/*.png`, `Pathol/*.png`
- **Total**: 440 RPs (239 + 201)

### Salida
- **Destino**: `../06-C_Feature_Selection/`
- **Archivo**: `combined_glcm_lbp_wavelet_rqa_statistical_20251204_173601.csv`
- **Formato**: CSV con columnas [filename, label, feature_1, ..., feature_181]

## Resultados del Análisis

El dataset de 181 características generado por este módulo fue procesado en los módulos posteriores:

- **06-C_Feature_Selection**: Reducción a 15 características óptimas
- **07-C_Classification**: Clasificación con Random Forest (79.55% accuracy, 90.47% AUC)

### Top 5 Características Más Importantes (Random Forest)

1. `lbp_hist_bin_5_r1_p4` (18.5% importancia)
2. `stat_hist_bin_0` (14.2%)
3. `wavelet_energy_detail_H_L1` (11.9%)
4. `rqa_LAM` (9.8%)
5. `lbp_nonuniform_ratio_r1_p4` (8.9%)

## Notas Importantes

- ✅ El sistema es **interrumpible** y **reanudable** en cualquier momento
- ✅ Los descriptores se calculan de forma **independiente** y **paralela**
- ✅ La configuración en `config.py` tiene **prioridad absoluta**
- ✅ Compatible con **procesamiento incremental** (solo nuevas imágenes)
- ⚠️  Gabor y Tamura están disponibles pero **NO se usan** en el pipeline final
- ✅  Los archivos deben estar en el **mismo orden** en todos los CSVs para combinar


## Referencias

- **Módulo anterior**: `04_RP_m7_tau9` (Generación de Recurrence Plots)
- **Módulo siguiente**: `06-C_Feature_Selection` (Selección de Características)
- **Resultados finales**: Ver `RESULTADOS_PIPELINE_m7_tau9.md` en la raíz del proyecto
