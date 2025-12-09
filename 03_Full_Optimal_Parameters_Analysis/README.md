# Módulo 03_Full: Análisis de Parámetros Óptimos para Recurrence Plots

## 📋 Descripción

Este módulo determina los **parámetros óptimos** para la generación de Recurrence Plots (RPs) mediante análisis de series temporales no lineales: **tiempo de retardo (τ, tau)** y **dimensión de embedding (m)**.

**Característica clave**: Procesamiento paralelo de **TODOS los archivos** del dataset con sistema de checkpoints para reanudar análisis interrumpidos.

---

## 🎯 Objetivo

Calcular los parámetros óptimos para la reconstrucción del espacio de fase:

1. **Tiempo de retardo (τ)**: Usando **Información Mutua** y **Autocorrelación**
2. **Dimensión de embedding (m)**: Usando **False Nearest Neighbors (FNN)**
3. **Análisis estadístico agregado**: Medianas, percentiles, distribuciones por categoría

**Output**: Valores óptimos validados para usar en el módulo 04 (generación de RPs)

---

## 🔬 Fundamento Teórico

### 1. Tiempo de Retardo (τ)

El tiempo de retardo óptimo determina cuántos pasos temporales separar las muestras al reconstruir el espacio de fase.

#### Método 1: Información Mutua (MI)

**Fórmula**:
```
I(τ) = ∑ P(x(t), x(t+τ)) log₂ [P(x(t), x(t+τ)) / (P(x(t)) · P(x(t+τ)))]
```

**Criterio**: **Primer mínimo local** de I(τ)
- Minimiza redundancia entre x(t) y x(t+τ)
- Captura dependencias no lineales

#### Método 2: Autocorrelación

**Fórmula**:
```
R(τ) = E[(x(t) - μ)(x(t+τ) - μ)] / σ²
```

**Criterio**: Primer cruce por cero o caída a 1/e ≈ 0.368

**IMPORTANTE**: Este módulo corrige un bug crítico donde versiones anteriores seleccionaban el **valor mínimo global** en lugar del **primer mínimo local**. El primer mínimo local es teóricamente más apropiado.

### 2. Dimensión de Embedding (m)

La dimensión óptima para "desplegar" el atractor en el espacio de fase.

#### Método: False Nearest Neighbors (FNN)

**Criterio**:
```
FNN(m) < 5%  (umbral típico: 1-5%)
```

- Vecinos son "falsos" si la distancia aumenta mucho al incrementar la dimensión
- La dimensión óptima es la mínima donde FNN cae por debajo del umbral

---

## 🚀 Uso del Módulo

### **Instalación de Dependencias**

```bash
pip install librosa numpy scipy matplotlib seaborn pandas
```

### **Ejecución**

```bash
# Desde la raíz del proyecto
cd 03_Full_Optimal_Parameters_Analysis

# Procesamiento paralelo completo (usa 7 cores por defecto)
python main.py

# Si el proceso se interrumpe, al ejecutar nuevamente:
# → Detecta checkpoint y continúa desde donde quedó
```

### **Entrada Esperada**

```
02_Audio_Preprocess/output/
├── Normal/
│   ├── N001.wav  # Archivos preprocesados
│   └── ...
└── Pathol/
    ├── P001.wav
    └── ...
```

### **Salidas Generadas**

```
03_Full_Optimal_Parameters_Analysis/output/
├── checkpoint.json                  # Progreso del análisis
├── aggregate_statistics.json        # Estadísticas globales
├── tau_analysis_aggregate.png       # Distribución de τ óptimos
├── dim_analysis_aggregate.png       # Distribución de m óptimos
├── tau_analysis/                    # Gráficos individuales τ
│   ├── Normal/
│   │   ├── N001_tau_analysis.png
│   │   └── ...
│   └── Pathol/
│       ├── P001_tau_analysis.png
│       └── ...
└── dim_analysis/                    # Gráficos individuales m
    ├── Normal/
    │   ├── N001_dim_analysis.png
    │   └── ...
    └── Pathol/
        ├── P001_dim_analysis.png
        └── ...
```

---

## 📈 Parámetros Configurables

En `config.py`:

| Parámetro | Valor Default | Descripción |
|-----------|---------------|-------------|
| `TARGET_SAMPLE_RATE` | 25000 | Frecuencia de muestreo (Hz) - **DEBE COINCIDIR CON MÓDULO 02** |
| `PROCESS_ALL_FILES` | True | Procesar todos los archivos (no muestreo) |
| `TAU_MAX` | 100 | Rango máximo de τ a evaluar |
| `TAU_STEP` | 1 | Paso para evaluar τ |
| `DIM_MAX` | 10 | Dimensión máxima a evaluar |
| `FNN_THRESHOLD` | 0.05 | Umbral de FNN (5%) |
| `NUM_CORES` | 7 | Cores para procesamiento paralelo |
| `BATCH_SIZE` | 10 | Archivos por batch (para checkpoints) |
| `CHECKPOINT_FREQUENCY` | 5 | Guardar checkpoint cada N archivos |
| `SAVE_TAU_PLOT` | True | Guardar gráficos individuales de τ |
| `SAVE_DIM_PLOT` | True | Guardar gráficos individuales de m |
| `SAVE_AGGREGATE_PLOTS` | True | Guardar gráficos agregados |

---

## 🔄 Flujo del Proceso

```
ENTRADA: Audios preprocesados (02_Audio_Preprocess/output)
    ↓
┌─────────────────────────────────────────────┐
│ 1. Verificación de Checkpoint               │
│    - Si existe: continúa desde progreso     │
│    - Si no: inicia análisis completo        │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ 2. Procesamiento Paralelo (7 cores)        │
│    - Divide archivos en batches            │
│    - Procesa cada batch en paralelo        │
│    - Guarda checkpoint cada batch          │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ 3. Cálculo de τ óptimo (por archivo)       │
│    a) Información Mutua → τ_MI             │
│       Criterio: PRIMER mínimo local        │
│    b) Autocorrelación → τ_AC               │
│       Criterio: Primer cruce por cero      │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ 4. Cálculo de m óptimo (por archivo)       │
│    - False Nearest Neighbors (FNN)         │
│    - m óptimo: min{m | FNN(m) < 5%}        │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ 5. Agregación Estadística                  │
│    - Medianas, percentiles (25, 50, 75)    │
│    - Distribuciones por categoría          │
│    - Tests estadísticos (Normal vs Pathol) │
└─────────────────────────────────────────────┘
    ↓
SALIDA: Parámetros óptimos + Estadísticas + Gráficos
```

---

## 📊 Formato del JSON de Salida

### `aggregate_statistics.json`

```json
{
  "timestamp": "2025-12-07T15:30:00",
  "total_files_analyzed": 440,
  "categories": {
    "Normal": 239,
    "Pathol": 201
  },
  "tau_statistics": {
    "mutual_information": {
      "global": {
        "median": 9,
        "mean": 11.2,
        "std": 5.8,
        "p25": 7,
        "p50": 9,
        "p75": 13
      },
      "by_category": {
        "Normal": {
          "median": 9,
          "mean": 10.8,
          "std": 5.5
        },
        "Pathol": {
          "median": 9,
          "mean": 11.7,
          "std": 6.2
        }
      }
    },
    "autocorrelation": {
      "global": {
        "median": 12,
        "mean": 14.5,
        "std": 7.3
      }
    }
  },
  "dim_statistics": {
    "false_nearest_neighbors": {
      "global": {
        "median": 4,
        "mean": 4.3,
        "std": 1.2,
        "p25": 3,
        "p50": 4,
        "p75": 5
      },
      "by_category": {
        "Normal": {
          "median": 4,
          "mean": 4.1
        },
        "Pathol": {
          "median": 4,
          "mean": 4.5
        }
      }
    }
  },
  "recommendations": {
    "tau_optimal": 9,
    "dim_optimal": 4,
    "rationale": "Mediana de primer mínimo local de MI (tau) y mediana de FNN (dim)"
  }
}
```

---

## 💡 Resultados Típicos (Dataset de 440 Audios)

### Tiempo de Retardo (τ)

| Método | Mediana | Rango Típico | Recomendación |
|--------|---------|--------------|---------------|
| **Información Mutua** | **9** | 7-13 | **Usar este valor** |
| Autocorrelación | 12 | 8-18 | Validación |

### Dimensión de Embedding (m)

| Método | Mediana | Rango Típico | Recomendación |
|--------|---------|--------------|---------------|
| **FNN (umbral 5%)** | **4** | 3-5 | **Usar este valor** |

### Configuración Recomendada para Módulo 04:

```python
# 04_Recurrence_Plot_Generator/config.py
EMBEDDING_DIM = 4   # Mediana de FNN
TIME_DELAY = 9      # Mediana de primer mínimo local de MI
```

---

## 🔍 Sistema de Checkpoints

### Funcionalidad

- **Guarda progreso** cada `BATCH_SIZE` archivos procesados
- **Detecta cambios de configuración** mediante hash MD5
- **Reanuda automáticamente** desde el último checkpoint válido

### Estructura del Checkpoint

```json
{
  "version": "1.0",
  "config_hash": "a3f2b1c4...",
  "last_update": "2025-12-07T10:15:30",
  "total_files": 440,
  "processed_files": 250,
  "results": {
    "Normal_N001.wav": {
      "tau_mi": 9,
      "tau_ac": 12,
      "dim_fnn": 4
    },
    ...
  },
  "processed_file_list": [
    "Normal_N001.wav",
    "Normal_N002.wav",
    ...
  ]
}
```

### Casos de Uso

**Interrupción del proceso**:
```bash
# Primera ejecución (se interrumpe en archivo 250/440)
python main.py
# ... procesamiento ...
# ^C (interrupción manual)

# Segunda ejecución (continúa automáticamente desde archivo 251)
python main.py
✓ Checkpoint encontrado: 250/440 procesados
Continuando desde archivo 251...
```

**Cambio de configuración**:
```bash
# Si modificas TAU_MAX, DIM_MAX, etc. en config.py:
python main.py
⚠️ Configuración cambiada. Iniciando nuevo análisis.
```

---

## 🛠️ Troubleshooting

**Error: "Configuración inconsistente con checkpoint"**
```
Causa: Modificaste config.py después de un análisis parcial
Solución: El módulo reiniciará automáticamente desde cero
```

**Warning: "No se encontró mínimo local de MI"**
```
Causa: Curva de MI es monótona (sin mínimos locales claros)
Solución: Se usa τ=1 por defecto. Revisar gráfico individual del archivo.
```

**Error: "Memoria insuficiente en procesamiento paralelo"**
```
Causa: NUM_CORES muy alto para RAM disponible
Solución: Reducir NUM_CORES en config.py (ej. de 7 a 4)
```

**Proceso muy lento**:
```
Solución 1: Deshabilitar plots individuales (SAVE_TAU_PLOT=False, SAVE_DIM_PLOT=False)
Solución 2: Aumentar BATCH_SIZE para menos escrituras de checkpoint
Solución 3: Verificar que PROCESS_ALL_FILES=True (más eficiente que muestreo)
```

---

## 📚 Referencias Científicas

1. **Fraser, A.M., & Swinney, H.L. (1986)**. *Independent coordinates for strange attractors from mutual information*. Physical Review A, 33(2), 1134.
   - Método de información mutua para τ

2. **Kennel, M.B., Brown, R., & Abarbanel, H.D. (1992)**. *Determining embedding dimension for phase-space reconstruction using a geometrical construction*. Physical Review A, 45(6), 3403.
   - Método de False Nearest Neighbors para m

3. **Takens, F. (1981)**. *Detecting strange attractors in turbulence*. Lecture Notes in Mathematics, 898, 366-381.
   - Teorema de embedding para reconstrucción del espacio de fase

4. **Kantz, H., & Schreiber, T. (2004)**. *Nonlinear Time Series Analysis* (2nd ed.). Cambridge University Press.
   - Métodos completos de análisis no lineal

---

## ⚙️ Tiempo de Ejecución Estimado

Para **440 archivos** (~1 segundo cada uno):

| Configuración | Tiempo Estimado |
|---------------|-----------------|
| 7 cores, plots habilitados | ~45-60 minutos |
| 7 cores, plots deshabilitados | ~25-35 minutos |
| 4 cores, plots habilitados | ~75-90 minutos |
| 1 core, plots deshabilitados | ~3-4 horas |

---

**Generado para el pipeline de análisis de voz mediante Recurrence Plots**
