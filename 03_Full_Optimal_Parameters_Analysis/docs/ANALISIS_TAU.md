# Análisis para Selección de Tiempo de Retardo (τ)

**Fecha de creación**: 2025-11-30
**Estado**: En progreso

---

## 1. Fundamento Teórico

### 1.1 ¿Qué es el tiempo de retardo?

El tiempo de retardo (τ) determina la separación temporal entre las coordenadas del embedding. Para una serie temporal x(t), el vector de embedding es:

$$\vec{X}(t) = [x(t), x(t+\tau), x(t+2\tau), ..., x(t+(m-1)\tau)]$$

### 1.2 Criterio de selección

El τ óptimo debe lograr un balance:
- **τ muy pequeño**: Las coordenadas están muy correlacionadas, el atractor colapsa hacia la diagonal
- **τ muy grande**: Las coordenadas pierden relación, el atractor se dispersa sin estructura

### 1.3 Métodos de cálculo

#### Información Mutua (MI)
- Mide dependencia lineal Y no lineal entre x(t) y x(t+τ)
- **Criterio**: Primer mínimo local de la función MI(τ)
- Más completo teóricamente para sistemas no lineales

#### Autocorrelación (ACF)
- Mide solo dependencia lineal entre x(t) y x(t+τ)
- **Criterio**: Primer cruce por cero de la función ACF(τ)
- Más simple pero limitado a correlaciones lineales

### 1.4 ¿Por qué MI y ACF dan valores diferentes?

- MI captura dependencias no lineales que ACF ignora
- En señales de voz (sistemas no lineales), MI típicamente da valores de τ más pequeños
- La diferencia indica presencia de correlaciones no lineales significativas

### 1.5 Propiedad NO Acumulativa de τ

**Importante**: A diferencia de la dimensión m, el τ NO tiene propiedad acumulativa:
- τ=9 NO "contiene" a τ=8
- Cada valor de τ produce un embedding **diferente**, no uno que contenga al otro
- No existe un criterio de "cobertura" como en m

---

## 2. Revisión de Literatura

### 2.1 Enfoques en estudios publicados

La literatura muestra **dos enfoques principales** para seleccionar τ en estudios con múltiples señales:

#### Enfoque A: Parámetros calculados por señal individual

Algunos estudios calculan τ (y m) para cada señal individualmente usando MI y FNN:

> "The traditional mutual information method is used to obtain the time delay, τ... and the false nearest neighbors (FNN) algorithm is used to find the best embedding dimension."
> — Costa et al., 2022 [1]

**Ventaja**: Óptimo para cada señal
**Desventaja**: Los RPs no son directamente comparables entre señales

#### Enfoque B: Parámetros fijos mediante calibración

Otros estudios usan **búsqueda por rejilla** con señales de calibración para encontrar parámetros fijos:

> "Systematic search of values of these parameters m = 2, 3...10, τ = 2,3... 50, and for r = 0.02, 0.04,...0.5, on a perfectly periodic test signal."
> — Little et al., 2007 [2]

**Ventaja**: Comparabilidad, reproducibilidad
**Desventaja**: No óptimo para cada señal individual

#### Enfoque C: Parámetros fijos basados en dataset representativo

> "Embedding dimension = 2 and time delay = 6, determined based on the false nearest neighbor method and average mutual information."
> — Pham et al., 2024 [3]

En este caso, parece que calcularon los parámetros una vez y los aplicaron uniformemente.

### 2.2 Parámetros reportados en literatura de voz

| Estudio | Frecuencia | τ | m | Dataset | Método selección |
|---------|------------|---|---|---------|------------------|
| Little et al., 2007 [2] | 25 kHz | 35 | 4 | 707 (53 normal, 654 pathol) | Búsqueda por rejilla |
| Costa et al., 2022 [1] | 25 kHz | Individual | Individual | 226 (53 normal, 173 pathol) | MI y FNN por señal |
| Pham et al., 2024 [3] | 16 kHz | 6 | 2 | Múltiple | MI y FNN (aparentemente fijo) |

### 2.3 Observaciones de la literatura

1. **No hay consenso** sobre usar τ fijo vs adaptativo
2. **Los valores varían ampliamente**: desde τ=6 hasta τ=35
3. **La frecuencia de muestreo influye**: mayor fs → mayor τ (en muestras)
4. **Ningún estudio encontrado usa estadísticos** (mediana, promedio) para agregar valores individuales de τ

### 2.4 Referencias

[1] Costa, V. et al. (2022). "Multi-Scale Recurrence Quantification Measurements for Voice Disorder Detection." Applied Sciences, 12(18), 9196. https://www.mdpi.com/2076-3417/12/18/9196

[2] Little, M.A. et al. (2007). "Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection." BioMedical Engineering OnLine, 6:23. https://pmc.ncbi.nlm.nih.gov/articles/PMC1913514/

[3] Pham, T. et al. (2024). "Recurrence plot embeddings as short segment nonlinear features for multimodal speaker identification." Scientific Reports. https://pmc.ncbi.nlm.nih.gov/articles/PMC11143305/

---

## 3. Resultados de Nuestro Análisis

### 3.1 Configuración

| Parámetro | Valor |
|-----------|-------|
| TAU_MAX | 100 |
| Bins para MI | 10 |
| Sample rate | 25000 Hz |
| Archivos analizados | 440 (239 Normal, 201 Pathol) |

### 3.2 Estadísticas de τ por Método

#### Información Mutua (MI) - Primer Mínimo Local

| Métrica | Global | Normal (239) | Pathol (201) |
|---------|--------|--------------|--------------|
| Promedio | 10.55 | 9.34 | 11.98 |
| Mediana | 9 | 8 | 9 |
| Desv. Estándar | 6.01 | 4.73 | 7.18 |
| Mínimo | 2 | 5 | 2 |
| Máximo | 46 | 38 | 46 |

#### Autocorrelación (ACF) - Primer Cruce por Cero

| Métrica | Global | Normal (239) | Pathol (201) |
|---------|--------|--------------|--------------|
| Promedio | 21.82 | 21.72 | 21.93 |
| Mediana | 16 | 14 | 22 |
| Desv. Estándar | 13.75 | 14.56 | 12.73 |
| Mínimo | 5 | 5 | 5 |
| Máximo | 59 | 57 | 59 |

### 3.3 Comparación MI vs ACF

| Categoría | Mediana MI | Mediana ACF | Ratio ACF/MI |
|-----------|------------|-------------|--------------|
| Global | 9 | 16 | 1.78 |
| Normal | 8 | 14 | 1.75 |
| Pathol | 9 | 22 | 2.44 |

**Observación**: ACF da valores ~2x mayores que MI, especialmente en Pathol.

---

## 4. Problema: ¿Cómo seleccionar τ único para el dataset?

### 4.1 El dilema

La literatura no ofrece un método estándar para agregar valores individuales de τ cuando se desea usar un valor fijo para todo el dataset.

### 4.2 Opciones identificadas

| Enfoque | Descripción | Usado en literatura |
|---------|-------------|---------------------|
| **Por señal** | Calcular τ individual | Costa et al., 2022 |
| **Búsqueda por rejilla** | Optimizar en señal de calibración | Little et al., 2007 |
| **Estadístico (mediana/promedio)** | Agregar valores individuales | No encontrado |
| **Valor fijo de literatura** | Adoptar valor publicado | Pham et al., 2024 |

### 4.3 Análisis crítico

**Usar estadísticos (mediana, promedio)**:
- No tiene justificación teórica sólida
- La mediana asume que el "mejor" es el central
- El promedio es sensible a outliers
- **Ningún estudio encontrado usa este enfoque**

**Usar búsqueda por rejilla (como Little et al.)**:
- Tiene justificación: optimiza para reproducir resultados teóricos
- Requiere señal de calibración
- El valor resultante (τ=35) es mucho mayor que nuestra mediana (τ=9)

**Usar τ por señal individual**:
- Es el más riguroso teóricamente
- Complicaría la comparabilidad de RPs
- Algunos estudios lo usan exitosamente

---

## 5. Observaciones Clave

### 5.1 Selección de método: MI vs ACF

**Recomendación: Usar MI** porque:
1. Captura dependencias no lineales (importante para voz)
2. Da valores más conservadores (τ más pequeño)
3. Es el método más citado en literatura de Recurrence Plots

### 5.2 Comparación con Little et al. (2007)

| Aspecto | Little et al. | Nuestro estudio |
|---------|---------------|-----------------|
| Frecuencia | 25 kHz | 25 kHz |
| τ | 35 | 9 (mediana MI) |
| m | 4 | 4-5 |
| Método | Búsqueda por rejilla | MI primer mínimo local |

La diferencia en τ (35 vs 9) es notable y requiere investigación adicional.

### 5.3 Diferencias entre categorías

- τ_MI es similar entre Normal (8) y Pathol (9)
- τ_ACF difiere más: Normal (14) vs Pathol (22)
- Pathol tiene mayor variabilidad en ambos métodos

---

## 6. Opciones de Selección

### Opción A: τ por señal individual
- Calcular τ_MI para cada señal y usarlo individualmente
- **Pros**: Teóricamente óptimo, usado en literatura
- **Cons**: RPs no directamente comparables

### Opción B: τ = 9 (Mediana MI global)
- Usar valor central de la distribución empírica
- **Pros**: Simple, representativo del dataset
- **Cons**: Sin precedente claro en literatura

### Opción C: Búsqueda por rejilla (como Little et al.)
- Calibrar con señal periódica de prueba
- **Pros**: Metodología publicada y validada
- **Cons**: Podría dar τ muy diferente (35 vs 9)

### Opción D: Adoptar valor de literatura
- Usar τ=35 de Little et al. (mismo fs=25kHz)
- **Pros**: Reproducibilidad, comparabilidad con estudio previo
- **Cons**: Muy diferente a nuestros valores de MI

---

## 7. Discusión Pendiente

1. ¿Por qué Little et al. obtuvieron τ=35 mientras nuestra mediana MI es 9?
2. ¿Es válido usar la mediana aunque no haya precedente en literatura?
3. ¿Deberíamos implementar búsqueda por rejilla para validar?
4. ¿Es mejor usar τ individual por señal?

---

## 8. Decisión

**Estado**: PENDIENTE

| Aspecto | Decisión |
|---------|----------|
| Método (MI vs ACF) | MI (justificado en sección 5.1) |
| Valor de τ | PENDIENTE |
| Criterio usado | PENDIENTE |

**Decisión final**: τ = ___

**Fecha de decisión**: ___

**Justificación**: ___

---

## 9. Historial de Actualizaciones

| Fecha | Cambio |
|-------|--------|
| 2025-11-30 | Creación del documento |
| 2025-11-30 | Agregada revisión de literatura con referencias concretas |

