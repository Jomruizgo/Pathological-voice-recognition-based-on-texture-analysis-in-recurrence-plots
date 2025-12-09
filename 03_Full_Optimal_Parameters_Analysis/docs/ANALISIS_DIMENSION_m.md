# Análisis para Selección de Dimensión de Embedding (m)

**Fecha de creación**: 2025-11-30
**Estado**: En progreso

---

## 1. Fundamento Teórico

### 1.1 ¿Qué es la dimensión de embedding?

La dimensión de embedding (m) determina el número de coordenadas en el espacio de fase reconstruido. Según el teorema de Takens, para reconstruir adecuadamente la dinámica de un sistema, se requiere:

$$m \geq 2d + 1$$

donde *d* es la dimensión del atractor del sistema original.

### 1.2 Método de Falsos Vecinos Cercanos (FNN)

El método FNN identifica la dimensión mínima donde los puntos que parecen cercanos en dimensiones bajas (falsos vecinos) se separan al aumentar la dimensión. Cuando el porcentaje de FNN cae por debajo de un umbral (típicamente 5%), se considera que la dimensión es suficiente.

### 1.3 Propiedad Acumulativa de m

**Importante**: La dimensión tiene una propiedad acumulativa:
- Un embedding de dimensión m=5 **contiene** toda la información de m=4
- Si una señal requiere m=5 para desenrollar su atractor, usar m=4 sería insuficiente
- Usar m=6 cuando m=5 es suficiente añade redundancia pero no pierde información

Por esta propiedad, el criterio de selección debe ser **conservador**: elegir un m que cubra la mayoría de los casos.

---

## 2. Resultados del Análisis FNN

### 2.1 Configuración

| Parámetro | Valor |
|-----------|-------|
| DIM_MAX | 10 |
| FNN_THRESHOLD | 0.05 (5%) |
| Sample rate | 25000 Hz |
| Archivos analizados | 440 |

### 2.2 Distribución de Dimensiones por Categoría

| Dimensión | Global | Normal (239) | Pathol (201) |
|-----------|--------|--------------|--------------|
| 3 | 2 (0.5%) | 0 (0.0%) | 2 (1.0%) |
| 4 | 348 (79.1%) | 208 (87.0%) | 140 (69.7%) |
| 5 | 76 (17.3%) | 31 (13.0%) | 45 (22.4%) |
| 6 | 2 (0.5%) | 0 (0.0%) | 2 (1.0%) |
| 7 | 7 (1.6%) | 0 (0.0%) | 7 (3.5%) |
| 8 | 2 (0.5%) | 0 (0.0%) | 2 (1.0%) |
| 10 | 3 (0.7%) | 0 (0.0%) | 3 (1.5%) |

### 2.3 Estadísticas Descriptivas

| Métrica | Global | Normal | Pathol |
|---------|--------|--------|--------|
| Promedio | 4.28 | 4.13 | 4.47 |
| Mediana | 4 | 4 | 4 |
| Moda | 4 | 4 | 4 |
| Mínimo | 3 | 4 | 3 |
| Máximo | 10 | 5 | 10 |

### 2.4 Análisis de Cobertura Acumulada

| Dimensión | Normal | Pathol | Interpretación |
|-----------|--------|--------|----------------|
| m=4 | 87.0% | 69.7% | Cubre mayoría de Normal, pero 30% de Pathol requiere más |
| m=5 | 100% | 92.0% | Cubre todo Normal, 8% de Pathol requiere más |
| m=7 | 100% | 97.5% | Solo 2.5% de Pathol requiere m>7 |
| m=10 | 100% | 100% | Cobertura total |

---

## 3. Observaciones Clave

### 3.1 Diferencias entre categorías

1. **Normal es más homogéneo**: Todas las voces normales tienen m entre 4-5
2. **Pathol es más variable**: Rango de 3-10, con casos que requieren dimensiones altas
3. **Voces patológicas más complejas**: La mayor variabilidad sugiere dinámicas más complejas o irregulares

### 3.2 Consistencia con literatura

- Estudios previos de dinámica no lineal en voz reportan m=2-5 típicamente
- Valores m>7 son inusuales y podrían indicar:
  - Señales con alta complejidad
  - Ruido en la señal
  - Dinámicas caóticas de alta dimensión

---

## 4. Opciones de Selección

### Opción A: m = 4 (Moda global)
- **Ventajas**: Cubre 79.1% global, mínima redundancia
- **Desventajas**: No cubre 30.3% de Pathol
- **Riesgo**: Información perdida en casos patológicos complejos

### Opción B: m = 5 (Cobertura 93% Pathol)
- **Ventajas**: Cubre 100% Normal, 92% Pathol
- **Desventajas**: 8% de Pathol podría tener embedding insuficiente
- **Balance**: Buena cobertura con dimensión moderada

### Opción C: m = 7 (Cobertura 97.5% Pathol) ✓ SELECCIONADA
- **Ventajas**: Cubre casi todos los casos (97.5% Pathol)
- **Desventajas**: Posible redundancia para señales simples
- **Conservador**: Minimiza riesgo de información perdida

### Opción D: m = 10 (Cobertura total)
- **Ventajas**: Cubre 100% de todos los casos
- **Desventajas**: Alto costo computacional, máxima redundancia
- **Uso**: Solo si se quiere garantía absoluta

---

## 5. Discusión

### 5.1 Criterio de selección

Dado que m tiene propiedad acumulativa (m alto contiene a m bajo), el criterio debe priorizar **cobertura sobre eficiencia**. Un embedding insuficiente pierde información de la dinámica, mientras que uno redundante solo añade costo computacional.

### 5.2 Pregunta abierta

¿Cuál es el umbral de cobertura aceptable?
- 90%? → m = 5
- 95%? → m = 5 o m = 6
- 97.5%? → m = 7
- 100%? → m = 10

### 5.3 Consideración práctica

Para clasificación de voz patológica, es más importante no perder información de los casos patológicos (que son el objetivo de detección) que optimizar eficiencia computacional.

### 5.4 Análisis de Costo Computacional: m=5 vs m=7

Para evaluar el impacto real de elegir m=7 sobre m=5, se comparó el tamaño de los Recurrence Plots generados con el mismo archivo de audio (n_adga):

| Parámetro | m=5, τ=9 | m=7, τ=9 | Diferencia |
|-----------|----------|----------|------------|
| Tamaño RP | 24,964 × 24,964 | 24,946 × 24,946 | -18 píxeles/lado |
| Píxeles totales | 623,201,296 | 622,302,916 | -898,380 (-0.14%) |

**Fórmula del tamaño del RP:**

El número de vectores de embedding (y por tanto el tamaño N×N del RP) se calcula como:

$$N = n_{samples} - (m-1) \times \tau$$

Para el mismo audio con n_samples muestras:
- m=5: $N = n_{samples} - 4 \times 9 = n_{samples} - 36$
- m=7: $N = n_{samples} - 6 \times 9 = n_{samples} - 54$

**Diferencia**: $(m_7-1)\tau - (m_5-1)\tau = 6 \times 9 - 4 \times 9 = 18$ vectores

**Conclusión del análisis de costo:**

La elección de m=7 sobre m=5 representa una reducción de apenas **18 puntos** en matrices de ~25,000 × 25,000 (equivalente al **0.07%**). Este costo computacional adicional es **despreciable** comparado con el beneficio de incrementar la cobertura de casos patológicos del 92% al 97.5%.

---

## 6. Decisión

**Estado**: DECIDIDO

| Opción considerada | m | Cobertura Pathol | Justificación |
|--------------------|---|------------------|---------------|
| Opción C | 7 | 97.5% | Trade-off favorable: máxima cobertura con costo marginal |

**Decisión final**: m = 7

**Fecha de decisión**: 2025-12-04

**Justificación**:

1. **Cobertura**: m=7 cubre el 97.5% de las señales patológicas, comparado con 92% de m=5
2. **Propiedad acumulativa**: Al usar m=7, se preserva toda la información que se obtendría con m=4, 5 o 6
3. **Costo computacional marginal**: La diferencia en tamaño del RP es de solo 18 píxeles por lado (0.07%), lo cual es despreciable en matrices de ~25,000 × 25,000
4. **Prioridad en detección**: Para clasificación de voz patológica, es preferible no perder información de los casos complejos (objetivo de detección) que optimizar eficiencia computacional
5. **Consistencia**: Todas las voces normales quedan cubiertas (100%), y solo el 2.5% de patológicas quedarían con embedding potencialmente insuficiente

---

## 7. Historial de Actualizaciones

| Fecha | Cambio |
|-------|--------|
| 2025-11-30 | Creación del documento |
| 2025-12-04 | Agregado análisis de costo computacional m=5 vs m=7 |
| 2025-12-04 | Decisión final: m=7 |

