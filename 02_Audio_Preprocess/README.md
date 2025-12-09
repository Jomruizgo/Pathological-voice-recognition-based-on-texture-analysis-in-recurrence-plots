# Módulo 02: Preprocesamiento de Audio

## 📋 Descripción

Este módulo implementa el preprocesamiento estandarizado de archivos de audio WAV para garantizar uniformidad en el pipeline de análisis de Recurrence Plots (RP).

**Función principal**: Normalizar las señales de audio a una frecuencia de muestreo común, seleccionar un segmento de 1 segundo que contenga actividad vocal usando SAD (Speech Activity Detection), y generar visualizaciones opcionales del proceso.

---

## 🎯 Objetivo

Preparar archivos de audio en formato WAV para el análisis posterior, asegurando:

1. **Frecuencia de muestreo uniforme**: 25 kHz para todos los audios
2. **Duración estandarizada**: Exactamente 1 segundo por archivo
3. **Formato mono**: Conversión de estéreo a mono si es necesario
4. **Selección inteligente**: Prioriza el segmento activo más largo usando SAD
5. **Trazabilidad**: Visualizaciones opcionales del proceso de selección

---

## 🔧 Características Principales

### Normalización de Audio

- **Remuestreo a 25 kHz**: Frecuencia consistente para todo el pipeline
- **Conversión a mono**: Si el audio es estéreo, se convierte a monocanal
- **Normalización de amplitud**: Escala las muestras al rango [-1, 1]

### Detección de Actividad de Sonido (SAD) y Selección Inteligente

Utiliza `librosa.effects.split()` para **identificar** (no eliminar) regiones con actividad vocal:

```python
# Parámetros configurables en config.py
SAD_TOP_DB = 30  # Umbral en dB: segmentos con energía por debajo de este valor se consideran silencio
MAX_AUDIO_DURATION_SECONDS = 1.0  # Duración objetivo del audio preprocesado
```

**Proceso de selección del segmento de 1 segundo**:

1. **Detecta todos los segmentos activos**: Intervalos donde la energía está por encima de `TOP_DB`
2. **Identifica el segmento activo MÁS LARGO**: No necesariamente el de mayor energía
3. **Selecciona 1 segundo según el caso**:

   - **Si segmento activo ≥ 1 segundo**:
     - Compara energía (RMS) de los extremos (primeros y últimos 0.1 segundos)
     - Recorta del extremo con MENOR energía para obtener exactamente 1 segundo

   - **Si segmento activo < 1 segundo**:
     - Extiende el segmento usando audio ALREDEDOR (incluyendo silencios) del audio original
     - Intenta centrar el segmento activo en el segundo resultante

   - **Si NO se detecta actividad**:
     - Toma los primeros 1 segundo del audio original completo

**IMPORTANTE**: El proceso **NO elimina silencios**. Utiliza SAD para identificar el segmento activo más largo (no necesariamente el de mayor energía) y selecciona 1 segundo basándose en este segmento. Cuando el segmento activo es mayor a 1s, recorta comparando la energía de pequeñas ventanas (0.1s) en los extremos. Cuando es menor, extiende con audio circundante preservando contexto temporal.

### Visualización (Opcional)

Genera gráficos comparativos mostrando:
- **Plot superior**: Forma de onda original con el segmento seleccionado resaltado en rojo
- **Plot inferior**: Segmento final de 1 segundo preprocesado
- Anotaciones indicando el criterio de selección usado (recorte/extensión/primeros 1s)

---

## 🚀 Uso del Módulo

### **Instalación de Dependencias**

```bash
pip install librosa soundfile numpy matplotlib
```

### **Ejecución**

```bash
# Desde la raíz del proyecto
cd 02_Audio_Preprocess

# Ejecutar preprocesamiento
python main.py
```

### **Entrada Esperada**

```
data/
├── Normal/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
└── Pathol/
    ├── audio1.wav
    ├── audio2.wav
    └── ...
```

### **Salidas Generadas**

```
02_Audio_Preprocess/output/
├── Normal/
│   ├── audio1.wav  # Audio preprocesado
│   ├── audio2.wav
│   └── ...
├── Pathol/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
└── preprocess_plots/  # (si SAVE_PREPROCESS_PLOTS=True)
    ├── Normal/
    │   ├── audio1_preprocess.png
    │   └── ...
    └── Pathol/
        ├── audio1_preprocess.png
        └── ...
```

---

## 📈 Parámetros Configurables

En `config.py`:

| Parámetro | Valor Default | Descripción |
|-----------|---------------|-------------|
| `TARGET_SAMPLE_RATE` | 25000 | Frecuencia de muestreo objetivo (Hz) |
| `MAX_AUDIO_DURATION_SECONDS` | 1.0 | Duración máxima a procesar (segundos) |
| `SAD_TOP_DB` | 30 | Umbral en dB para detección de silencios |
| `SAVE_PREPROCESS_PLOTS` | True | Guardar gráficos de preprocesamiento |

**IMPORTANTE**: `TARGET_SAMPLE_RATE` debe ser **consistente en todos los módulos** (02, 03, 04).

---

## 🔄 Flujo del Proceso

```
ENTRADA: Archivos WAV originales (data/Normal, data/Pathol)
    ↓
┌─────────────────────────────────────────────┐
│ 1. Carga de Audio                           │
│    - Remuestreo a 25 kHz                    │
│    - Conversión a mono                      │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ 2. Detección de Actividad de Sonido (SAD)  │
│    - librosa.effects.split()                │
│    - Umbral: TOP_DB = 30 dB                 │
│    - Identifica segmento activo MÁS LARGO   │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ 3. Selección Inteligente de 1 Segundo      │
│    - Si segmento ≥ 1s: recorta por energía  │
│    - Si segmento < 1s: extiende con audio   │
│    - Si sin actividad: primeros 1s          │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ 4. Normalización y Guardado                │
│    - Normaliza amplitud a [-1, 1]           │
│    - Formato: WAV mono, 25 kHz, 16-bit      │
│    - Plot opcional de comparación           │
└─────────────────────────────────────────────┘
    ↓
SALIDA: Archivos de 1 segundo en output/Normal y output/Pathol
```

---

## 📊 Ejemplo de Ejecución

```bash
$ python main.py

Iniciando la etapa de Preprocesamiento de Audio (con SAD y visualización segregada)...

Procesando audios de la categoría 'Normal' desde: ../data/Normal
[1/239] Procesando: N001.wav
  ✓ Audio cargado: 25000 Hz, 44100 muestras (1.76s)
  ✓ SAD aplicado: Segmento activo más largo detectado: 1.45s
  ✓ Segmento de 1.0s seleccionado mediante recorte inteligente
  ✓ Audio guardado: output/Normal/N001.wav (25000 Hz, 25000 muestras, 1.00s)
  ✓ Plot guardado: output/preprocess_plots/Normal/N001_preprocess_plot.png
[2/239] Procesando: N002.wav
...

Procesando audios de la categoría 'Patológico' desde: ../data/Pathol
[1/201] Procesando: P001.wav
...

--- Resumen de Preprocesamiento ---
Audios procesados exitosamente: 440
Audios omitidos (errores/problemas): 0
Archivos preprocesados guardados en: output/
Plots de preprocesamiento (si SAVE_PREPROCESS_PLOTS=True) en: output/preprocess_plots/[Normal|Pathol]
Preprocesamiento de Audio completado.
```

---

## 💡 Consideraciones Importantes

### 1. Consistencia de Parámetros

El valor de `TARGET_SAMPLE_RATE` debe ser idéntico en:
- `02_Audio_Preprocess/config.py`
- `03_Full_Optimal_Parameters_Analysis/config.py`
- `04_Recurrence_Plot_Generator/config.py`

**Recomendado**: 25000 Hz (25 kHz) para análisis de voz

### 2. Umbral SAD (TOP_DB)

**Valor usado en este pipeline: TOP_DB = 30 dB**

**Justificación para este dataset específico**:
- El pipeline selecciona **1 segundo** de cada audio (`MAX_AUDIO_DURATION_SECONDS = 1.0`) basándose en el segmento activo más largo detectado
- El valor de 30 dB fue seleccionado tras **inspección visual** de los plots de preprocesamiento
- Para este dataset específico, 30 dB identifica apropiadamente las regiones con actividad vocal (señales con energía por encima de 30 dB por debajo del pico máximo se consideran activas)

**IMPORTANTE**: El umbral óptimo **depende de las características del dataset**:
- Relación señal-ruido (SNR)
- Calidad de grabación
- Nivel de ruido de fondo
- Duración de audio a procesar

**Recomendaciones**:
- Si se procesa toda la señal de audio (no solo 1 segundo), se debe realizar un **análisis más profundo** para seleccionar el valor apropiado
- Valores más bajos (ej. 20 dB) detectan más segmentos como "activos" (incluye audio de menor energía)
- Valores más altos (ej. 40 dB) son más restrictivos en qué se considera "activo" (solo alta energía)
- **Siempre validar** inspeccionando visualmente los plots de preprocesamiento para verificar que el segmento seleccionado es representativo

### 3. Calidad de Entrada

- Se recomienda audios con SNR (relación señal-ruido) > 20 dB
- Evitar audios con ruido de fondo constante (ej. ventilador, tráfico)
- Verificar que los plots de preprocesamiento muestren segmentación apropiada

### 4. Eficiencia

- Para 440 archivos (~1 segundo cada uno): ~2-3 minutos de procesamiento
- La generación de plots aumenta el tiempo ~30%
- Deshabilitar plots (`SAVE_PREPROCESS_PLOTS=False`) para procesamiento rápido

---

## 🔍 Validación de Resultados

### Verificaciones Recomendadas:

1. **Duración consistente**: Todos los audios preprocesados deben tener duraciones razonables (no vacíos)
   ```bash
   # Verificar duraciones
   soxi -D output/Normal/*.wav | sort -n
   ```

2. **Frecuencia de muestreo**: Confirmar que todos son 25000 Hz
   ```bash
   # Verificar sample rate
   soxi -r output/Normal/*.wav | uniq
   ```

3. **Visualización**: Revisar algunos plots para confirmar que la selección es apropiada
   - Plot superior: Área sombreada en rojo = segmento de 1 segundo seleccionado
   - Plot inferior: Segmento final preprocesado de 1 segundo

---

## 🛠️ Troubleshooting

**Error: "No se encontraron archivos de audio"**
```
Solución: Verificar que data/Normal y data/Pathol existen y contienen archivos .wav
```

**Warning: "Archivo muy corto después de SAD"**
```
Posible causa: Archivo es casi todo silencio
Solución: Revisar calidad del audio original o ajustar SAD_TOP_DB a un valor menor
```

**Error: "Frecuencia de muestreo inconsistente"**
```
Solución: Verificar que TARGET_SAMPLE_RATE sea igual en módulos 02, 03 y 04
```

---

## 📚 Referencias Técnicas

1. **librosa.effects.split()**: McFee, B. et al. (2015). *librosa: Audio and Music Signal Analysis in Python*.
   - Implementación de detección de actividad basada en energía

2. **Voice Activity Detection (VAD)**: Estándares ITU-T G.729 Annex B
   - Fundamentos de detección de actividad de voz

---

**Generado para el pipeline de análisis de voz mediante Recurrence Plots**
