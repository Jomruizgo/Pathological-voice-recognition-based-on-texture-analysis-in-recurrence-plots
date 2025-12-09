# Pipeline de Clasificación de Voces Patológicas mediante Recurrence Plots

Pipeline modular para la detección y clasificación binaria de voces patológicas vs normales utilizando análisis de texturas en Recurrence Plots (RPs).

## Descripción

Este proyecto implementa un pipeline completo de procesamiento de señales de audio para clasificar voces patológicas. El enfoque combina técnicas de análisis no lineal de series temporales (Recurrence Plots) con extracción de descriptores de textura de imágenes y clasificación mediante machine learning.

### Metodología

1. **Preprocesamiento de audio**: Normalización, remuestreo y detección de actividad vocal
2. **Análisis de parámetros óptimos**: Cálculo de dimensión de embedding (m) y retardo temporal (τ) mediante Mutual Information y False Nearest Neighbors
3. **Generación de Recurrence Plots**: Construcción del espacio de fases y matriz de recurrencia
4. **Extracción de características**: Descriptores de textura (GLCM, LBP, Wavelet, RQA, estadísticos)
5. **Selección de características**: Pipeline de 5 fases con validación estadística
6. **Clasificación**: Evaluación de 8 modelos con validación cruzada

## Estructura del Proyecto

```
Pipeline-Code/
├── 01_Audio_EDA/                    # Análisis exploratorio de datos de audio
├── 02_Audio_Preprocess/             # Preprocesamiento de señales de audio
├── 03_Full_Optimal_Parameters_Analysis/  # Cálculo de parámetros óptimos (τ, m)
├── 04_RP_Generator_m7_tau9/         # Generación de Recurrence Plots
├── 05_Texture_Descriptors_m7_tau9/  # Extracción de descriptores de textura
├── 06-C_Feature_Selection/          # Selección rigurosa de características
└── 07-C_Classification/             # Clasificación y evaluación de modelos
```

## Requisitos

### Dependencias Principales

```
numpy
pandas
scipy
librosa
soundfile
scikit-learn
scikit-image
opencv-python
matplotlib
seaborn
xgboost
streamlit
pydub
pillow
```

### Instalación

```bash
pip install numpy pandas scipy librosa soundfile scikit-learn scikit-image opencv-python matplotlib seaborn xgboost streamlit pydub pillow
```

## Uso

Cada módulo es independiente y se ejecuta secuencialmente. Navegar al directorio del módulo y ejecutar:

```bash
python main.py
```

### Ejecución del Pipeline Completo

```bash
# 1. Análisis exploratorio
cd 01_Audio_EDA && python main.py

# 2. Preprocesamiento
cd ../02_Audio_Preprocess && python main.py

# 3. Análisis de parámetros óptimos
cd ../03_Full_Optimal_Parameters_Analysis && python main.py

# 4. Generación de Recurrence Plots
cd ../04_RP_Generator_m7_tau9 && python main.py

# 5. Extracción de descriptores
cd ../05_Texture_Descriptors_m7_tau9 && python main.py

# 6. Selección de características
cd ../06-C_Feature_Selection && python main.py

# 7. Clasificación
cd ../07-C_Classification && python main.py
```

### Opciones Disponibles

| Módulo | Opciones |
|--------|----------|
| 05 | `--descriptors glcm lbp` - Seleccionar descriptores específicos |
| 05 | `--gui` - Interfaz gráfica Streamlit |
| 05 | `--list-descriptors` - Ver descriptores disponibles |
| 06-C | `--verbose` - Modo detallado |
| 06-C | `--input /path/to/file.csv` - CSV de entrada personalizado |
| 07-C | `--model all` - Entrenar todos los modelos |
| 07-C | `--no-plots` - Solo métricas, sin gráficos |

## Configuración

Cada módulo tiene su archivo `config.py` con parámetros específicos:

### Parámetros Principales

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `TARGET_SAMPLE_RATE` | 25000 Hz | Frecuencia de muestreo |
| `MAX_AUDIO_DURATION_SECONDS` | 1.0 s | Duración máxima del audio |
| `EMBEDDING_DIM` | 7 | Dimensión de embedding (m) |
| `TIME_DELAY` | 9 | Retardo temporal (τ) |
| `EPSILON_VALUE` | 0.1 | Umbral de recurrencia (10% del rango) |

### Descriptores de Textura Activos

- **GLCM**: Gray Level Co-occurrence Matrix (6 características)
- **LBP**: Local Binary Patterns (14 características)
- **Wavelet**: Descomposición wavelet (65 características)
- **RQA**: Recurrence Quantification Analysis (10 características)
- **Statistical**: Estadísticas de imagen (86 características)

## Resultados

### Dataset

- **Total de muestras**: 440 audios (352 entrenamiento / 88 test)
- **Clase Normal**: 239 muestras
- **Clase Patológica**: 201 muestras

### Selección de Características

- **Características iniciales**: 181
- **Fase 1 (Significancia estadística p<0.05)**: 79 características
- **Fase 2 (Relevancia práctica |d|≥0.2)**: 71 características
- **Fase 4 (Eliminación de redundancia r<0.85)**: 15 características seleccionadas

### Rendimiento de Clasificación (subset top_10 características)

| Modelo | Accuracy | F1-Score | AUC-ROC |
|--------|----------|----------|---------|
| **Random Forest** | **81.82%** | **81.85%** | **90.05%** |
| Decision Tree | 81.82% | 81.82% | 84.58% |
| XGBoost | 77.27% | 77.27% | 88.91% |
| SVM | 73.86% | 73.89% | 82.92% |
| Neural Network | 73.86% | 73.50% | 80.47% |
| Naive Bayes | 72.73% | 71.27% | 83.07% |
| Logistic Regression | 71.59% | 71.62% | 81.72% |
| KNN | 68.18% | 68.25% | 80.49% |

**Mejor modelo**: Random Forest con 10 características seleccionadas (Accuracy: 81.82%, AUC-ROC: 90.05%)

## Módulos en Detalle

### 01_Audio_EDA

Análisis exploratorio de los archivos de audio:
- Estadísticas de duración, amplitud y frecuencia
- Detección de silencios
- Visualización de formas de onda

### 02_Audio_Preprocess

Preprocesamiento de señales:
- Remuestreo a frecuencia objetivo
- Recorte temporal uniforme
- Speech Activity Detection (SAD)
- Normalización de amplitud

### 03_Full_Optimal_Parameters_Analysis

Determinación de parámetros óptimos para Recurrence Plots:
- Cálculo de τ mediante Mutual Information y Autocorrelación
- Cálculo de m mediante False Nearest Neighbors (FNN)
- Análisis estadístico de resultados por clase

### 04_RP_Generator_m7_tau9

Generación de Recurrence Plots:
- Construcción del espacio de fases mediante time-delay embedding
- Cálculo de matriz de distancias
- Aplicación de umbral para matriz de recurrencia
- Exportación como imágenes de alta resolución (~2400x2400 px)

### 05_Texture_Descriptors_m7_tau9

Extracción de descriptores de textura:
- Sistema modular con 5 familias de descriptores
- Procesamiento con checkpoints para reanudación
- Interfaz gráfica opcional (Streamlit)
- Combinación automática de características

### 06-C_Feature_Selection

Selección rigurosa de características en 5 fases:
1. **Significancia estadística**: Test ANOVA (p < 0.05)
2. **Relevancia práctica**: Cohen's d (|d| ≥ 0.2)
3. **Ranking**: F-Score + Mutual Information
4. **Redundancia**: Correlación de Pearson (r < 0.85)
5. **Validación**: PCA, Silhouette Score, Fisher's Ratio

### 07-C_Classification

Clasificación y evaluación:
- 8 modelos: Logistic Regression, Naive Bayes, KNN, Decision Tree, SVM, Random Forest, Neural Network, XGBoost
- Validación cruzada estratificada (10 folds)
- Métricas: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- Matrices de confusión y curvas ROC

## Estructura de Datos de Entrada

Los archivos de audio deben organizarse en:

```
data/
├── Normal/
│   ├── audio_001.wav
│   └── ...
└── Pathol/
    ├── audio_001.wav
    └── ...
```

## Referencias Bibliográficas

- Eckmann, J.P., Kamphorst, S.O., Ruelle, D. (1987). Recurrence Plots of Dynamical Systems. *Europhysics Letters*.
- Marwan, N., et al. (2007). Recurrence plots for the analysis of complex systems. *Physics Reports*.
- Haralick, R.M., Shanmugam, K., Dinstein, I. (1973). Textural Features for Image Classification. *IEEE Transactions on Systems, Man, and Cybernetics*.

## Licencia

Este proyecto fue desarrollado como parte del curso "Joven Investigador" (2025-I) del Instituto Tecnológico Metropolitano (ITM).

## Autor

Desarrollado para investigación en clasificación de voces patológicas mediante análisis no lineal de señales de audio.
