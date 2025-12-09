"""
Descriptores de textura de Tamura para análisis de Recurrence Plots.

Los descriptores de Tamura capturan propiedades perceptuales de textura
que son psicológicamente significativas para el sistema visual humano.

================================================================================
PROCESAMIENTO POR BLOQUES PARA IMÁGENES GRANDES
================================================================================

Este descriptor implementa un sistema de procesamiento por bloques (tiling)
diseñado específicamente para manejar Recurrence Plots de gran tamaño
(típicamente 25000x25000 píxeles en este proyecto).

¿POR QUÉ ES NECESARIO EL PROCESAMIENTO POR BLOQUES?
----------------------------------------------------
El algoritmo original de Tamura para Coarseness requiere crear arrays
intermedios de tamaño (kmax × rows × cols). Para una imagen de 25000×25000
con kmax=6:

    Memoria requerida = 6 × 25000 × 25000 × 8 bytes ≈ 28 GB de RAM

Esto excede la memoria disponible en la mayoría de sistemas, causando que
el proceso sea terminado por el sistema operativo (OOM Killer).

SOLUCIÓN IMPLEMENTADA:
----------------------
1. La imagen se divide en bloques de tamaño configurable (default: 8192×8192)
2. Se calculan las características Tamura para cada bloque independientemente
3. Los resultados se agregan usando estadísticas robustas

CARACTERÍSTICAS EXTRAÍDAS (por bloque, luego agregadas):
--------------------------------------------------------
- tamura_coarseness_mean: Promedio de coarseness entre bloques
- tamura_coarseness_std: Variabilidad de coarseness (heterogeneidad espacial)
- tamura_contrast_mean: Promedio de contraste
- tamura_contrast_std: Variabilidad de contraste
- tamura_directionality_mean: Direccionalidad promedio
- tamura_directionality_std: Variabilidad direccional
- tamura_linelikeness_mean: Line-likeness promedio
- tamura_linelikeness_std: Variabilidad de estructuras lineales
- tamura_roughness_mean: Rugosidad promedio (coarseness + contrast)
- tamura_roughness_std: Variabilidad de rugosidad

IMPACTO EN LOS RESULTADOS:
--------------------------
- Los valores agregados (mean) correlacionan bien con el cálculo global
  cuando la textura es relativamente homogénea (como en RPs).
- Las desviaciones estándar (std) proveen información adicional sobre
  la variabilidad espacial de la textura, útil para detectar regiones
  con diferentes características.
- Patrones mayores al tamaño del bloque pueden no capturarse completamente,
  pero esto es un trade-off aceptable dado que:
  a) Los RPs tienen estructura diagonal que se preserva en cada bloque
  b) La mayoría de patrones relevantes ocurren a escalas menores
  c) El overlap entre bloques mitiga parcialmente este efecto

Referencias:
- Tamura et al. (1978): "Textural Features Corresponding to Visual Perception"
- Bianconi et al. (2015): "Discrimination between tumour epithelium and stroma"
================================================================================
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import logging
from scipy import ndimage
from scipy.signal import convolve2d
from scipy.stats import kurtosis
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

from .base import BaseDescriptor
from . import register_descriptor


@register_descriptor("tamura", enabled_by_default=True)
class TamuraDescriptor(BaseDescriptor):
    """
    Implementa los descriptores de textura de Tamura con soporte para imágenes grandes.

    Esta implementación utiliza procesamiento por bloques (tiling) para manejar
    Recurrence Plots de gran tamaño (>10000×10000 píxeles) sin exceder la memoria
    disponible del sistema.

    Los descriptores de Tamura están diseñados para corresponder con la
    percepción visual humana de textura, incluyendo:
    - Coarseness: grosor o granularidad
    - Contrast: rango dinámico de intensidades
    - Directionality: orientación predominante
    - Line-likeness: presencia de estructuras lineales
    - Roughness: combinación de coarseness y contrast

    Attributes:
        tile_size (int): Tamaño de los bloques para procesamiento (default: 8192)
        tile_overlap (int): Solapamiento entre bloques para reducir efectos de borde
        memory_threshold (int): Umbral en píxeles para activar procesamiento por bloques
        kmax (int): Máximo k para ventanas 2^k × 2^k en coarseness
        hist_bins (int): Número de bins para histograma de direccionalidad
    """

    # Umbral de tamaño para activar procesamiento por bloques
    # Imágenes con más de este número de píxeles usarán tiling
    # 10000×10000 = 100M píxeles, requeriría ~4.5GB solo para averages array
    LARGE_IMAGE_THRESHOLD = 100_000_000  # 100 millones de píxeles

    # Tamaño por defecto de bloques (8192×8192 = 67M píxeles)
    # Usa ~3GB para el array averages, balance entre velocidad y memoria
    # Con imágenes de ~25000×25000, esto genera 9-16 bloques
    DEFAULT_TILE_SIZE = 8192

    # Solapamiento entre bloques
    # Ayuda a reducir artefactos en los bordes entre bloques
    DEFAULT_TILE_OVERLAP = 512

    # Número de workers para procesamiento paralelo de bloques
    # None = usar número de CPUs disponibles (limitado a 4 para evitar OOM)
    DEFAULT_N_WORKERS = None

    def __init__(self, kmax: int = 6, hist_bins: int = 16,
                 directionality_threshold: float = 0.1,
                 tile_size: int = None, tile_overlap: int = None,
                 n_workers: int = None):
        """
        Inicializa el descriptor de Tamura.

        Args:
            kmax (int): Máximo k para ventanas 2^k x 2^k en coarseness (default: 6)
            hist_bins (int): Número de bins para histograma de direccionalidad (default: 16)
            directionality_threshold (float): Umbral para magnitud significativa (default: 0.1)
            tile_size (int): Tamaño de bloques para imágenes grandes (default: 8192)
            tile_overlap (int): Solapamiento entre bloques (default: 512)
            n_workers (int): Número de workers para procesamiento paralelo (default: auto)
        """
        super().__init__(kmax=kmax, hist_bins=hist_bins,
                        directionality_threshold=directionality_threshold)
        self._name = "tamura"
        self.logger = logging.getLogger(__name__)

        # Parámetros para cálculos
        self.kmax = kmax
        self.hist_bins = hist_bins
        self.directionality_threshold = directionality_threshold

        # Parámetros para procesamiento por bloques
        self.tile_size = tile_size or self.DEFAULT_TILE_SIZE
        self.tile_overlap = tile_overlap or self.DEFAULT_TILE_OVERLAP

        # Configurar número de workers para procesamiento paralelo
        # IMPORTANTE: Cada bloque de 8192×8192 requiere ~3GB para arrays intermedios
        # Con 20GB de RAM y la imagen ya cargada (~5GB), solo podemos procesar
        # ~4-5 bloques simultáneamente. Usamos 2 workers para ser conservadores.
        if n_workers is None:
            self.n_workers = 2  # Conservador para evitar OOM con imágenes grandes
        else:
            self.n_workers = n_workers

    @property
    def name(self) -> str:
        """Retorna el nombre del descriptor."""
        return self._name

    def _needs_tiling(self, image: np.ndarray) -> bool:
        """
        Determina si la imagen requiere procesamiento por bloques.

        Args:
            image: Imagen a evaluar

        Returns:
            bool: True si la imagen excede el umbral de tamaño
        """
        return image.size > self.LARGE_IMAGE_THRESHOLD

    def _generate_tiles(self, image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Genera bloques (tiles) de la imagen con solapamiento.

        Args:
            image: Imagen completa normalizada

        Returns:
            Lista de tuplas (tile, (y_start, y_end, x_start, x_end))
        """
        rows, cols = image.shape
        tiles = []

        # Calcular paso efectivo (tile_size - overlap)
        step = self.tile_size - self.tile_overlap

        y = 0
        while y < rows:
            x = 0
            y_end = min(y + self.tile_size, rows)

            while x < cols:
                x_end = min(x + self.tile_size, cols)

                # Extraer bloque
                tile = image[y:y_end, x:x_end]

                # Solo procesar si el bloque tiene tamaño mínimo razonable
                if tile.shape[0] >= 64 and tile.shape[1] >= 64:
                    tiles.append((tile, (y, y_end, x, x_end)))

                x += step
                if x_end == cols:
                    break

            y += step
            if y_end == rows:
                break

        return tiles

    def extract(self, image: np.ndarray, **kwargs) -> Dict[str, float]:
        """
        Extrae los descriptores de Tamura del Recurrence Plot.

        Para imágenes grandes (>100M píxeles), utiliza procesamiento por bloques
        para evitar problemas de memoria. Los resultados se agregan usando
        estadísticas (media y desviación estándar).

        Args:
            image (np.ndarray): RP en escala de grises
            **kwargs: Parámetros adicionales

        Returns:
            Dict[str, float]: Características de Tamura (10 características para
                              imágenes grandes, 5 para imágenes pequeñas)
        """
        self.logger.debug(f"Extrayendo características Tamura de imagen {image.shape}")

        # Validar imagen
        if len(image.shape) != 2:
            self.logger.warning("Tamura requiere imagen en escala de grises")
            return self._get_default_features()

        if image.size == 0:
            self.logger.warning("Imagen vacía")
            return self._get_default_features()

        # Normalizar imagen a [0, 1] si es necesario
        if image.max() > 1.0:
            image_norm = image.astype(np.float64) / 255.0
        else:
            image_norm = image.astype(np.float64)

        # Decidir estrategia de procesamiento
        if self._needs_tiling(image_norm):
            self.logger.info(f"Imagen grande ({image.shape}), usando procesamiento por bloques "
                           f"(tile_size={self.tile_size}, overlap={self.tile_overlap})")
            return self._extract_tiled(image_norm)
        else:
            self.logger.debug("Imagen pequeña, usando procesamiento directo")
            return self._extract_direct(image_norm)

    def _extract_direct(self, image_norm: np.ndarray) -> Dict[str, float]:
        """
        Extrae características directamente (sin tiling) para imágenes pequeñas.

        Args:
            image_norm: Imagen normalizada [0, 1]

        Returns:
            Dict con 5 características Tamura básicas
        """
        try:
            coarseness = self._compute_coarseness(image_norm)
            contrast = self._compute_contrast(image_norm)
            directionality = self._compute_directionality(image_norm)
            linelikeness = self._compute_linelikeness(image_norm)
            roughness = self._compute_roughness(coarseness, contrast)

            # Para mantener consistencia, también generamos las versiones con _mean
            features = {
                f"{self.name}_coarseness_mean": float(coarseness),
                f"{self.name}_coarseness_std": 0.0,
                f"{self.name}_contrast_mean": float(contrast),
                f"{self.name}_contrast_std": 0.0,
                f"{self.name}_directionality_mean": float(directionality),
                f"{self.name}_directionality_std": 0.0,
                f"{self.name}_linelikeness_mean": float(linelikeness),
                f"{self.name}_linelikeness_std": 0.0,
                f"{self.name}_roughness_mean": float(roughness),
                f"{self.name}_roughness_std": 0.0
            }

            self.logger.debug(f"Características Tamura extraídas (directo): {len(features)}")
            return features

        except Exception as e:
            self.logger.error(f"Error extrayendo características Tamura (directo): {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return self._get_default_features()

    def _process_single_tile(self, tile_data: Tuple[np.ndarray, Tuple[int, int, int, int]]) -> Optional[Tuple[float, float, float, float, float]]:
        """
        Procesa un único bloque y retorna sus características Tamura.

        Este método está diseñado para ser ejecutado en paralelo por ThreadPoolExecutor.

        Args:
            tile_data: Tupla (tile_array, coords) donde coords son (y_start, y_end, x_start, x_end)

        Returns:
            Tupla (coarseness, contrast, directionality, linelikeness, roughness) o None si falla
        """
        try:
            tile, coords = tile_data

            # Calcular características del bloque
            coarseness = self._compute_coarseness(tile)
            contrast = self._compute_contrast(tile)
            directionality = self._compute_directionality(tile)
            linelikeness = self._compute_linelikeness(tile)
            roughness = self._compute_roughness(coarseness, contrast)

            # Validar que los valores son finitos
            if all(np.isfinite([coarseness, contrast, directionality, linelikeness, roughness])):
                return (coarseness, contrast, directionality, linelikeness, roughness)
            else:
                return None

        except Exception:
            return None

    def _extract_tiled(self, image_norm: np.ndarray) -> Dict[str, float]:
        """
        Extrae características usando procesamiento PARALELO por bloques para imágenes grandes.

        Divide la imagen en bloques, calcula características para cada uno EN PARALELO,
        y luego agrega los resultados usando media y desviación estándar.

        Args:
            image_norm: Imagen normalizada [0, 1]

        Returns:
            Dict con 10 características Tamura (5 medias + 5 desviaciones estándar)
        """
        try:
            # Generar bloques
            tiles = self._generate_tiles(image_norm)
            n_tiles = len(tiles)

            self.logger.info(f"Procesando {n_tiles} bloques en paralelo ({self.n_workers} workers)...")

            if n_tiles == 0:
                self.logger.error("No se generaron bloques válidos")
                return self._get_default_features()

            # Almacenar características de cada bloque
            coarseness_values = []
            contrast_values = []
            directionality_values = []
            linelikeness_values = []
            roughness_values = []

            # Procesar bloques en paralelo usando ThreadPoolExecutor
            # ThreadPoolExecutor funciona bien porque numpy libera el GIL durante operaciones
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                # Enviar todos los bloques al pool
                future_to_idx = {
                    executor.submit(self._process_single_tile, tile_data): idx
                    for idx, tile_data in enumerate(tiles)
                }

                # Recolectar resultados a medida que completan
                completed = 0
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        if result is not None:
                            coarseness, contrast, directionality, linelikeness, roughness = result
                            coarseness_values.append(coarseness)
                            contrast_values.append(contrast)
                            directionality_values.append(directionality)
                            linelikeness_values.append(linelikeness)
                            roughness_values.append(roughness)
                        else:
                            self.logger.warning(f"Bloque {idx+1}/{n_tiles} produjo valores no finitos, ignorando")

                    except Exception as e:
                        self.logger.warning(f"Error en bloque {idx+1}/{n_tiles}: {e}")

                    completed += 1
                    # Log de progreso
                    if completed % 4 == 0 or completed == n_tiles:
                        self.logger.debug(f"Procesados {completed}/{n_tiles} bloques")

            # Verificar que tenemos suficientes bloques procesados
            n_valid = len(coarseness_values)
            if n_valid == 0:
                self.logger.error("Ningún bloque se procesó exitosamente")
                return self._get_default_features()

            self.logger.info(f"Bloques procesados exitosamente: {n_valid}/{n_tiles}")

            # Agregar resultados
            features = {
                # Medias
                f"{self.name}_coarseness_mean": float(np.mean(coarseness_values)),
                f"{self.name}_contrast_mean": float(np.mean(contrast_values)),
                f"{self.name}_directionality_mean": float(np.mean(directionality_values)),
                f"{self.name}_linelikeness_mean": float(np.mean(linelikeness_values)),
                f"{self.name}_roughness_mean": float(np.mean(roughness_values)),
                # Desviaciones estándar (variabilidad espacial)
                f"{self.name}_coarseness_std": float(np.std(coarseness_values)),
                f"{self.name}_contrast_std": float(np.std(contrast_values)),
                f"{self.name}_directionality_std": float(np.std(directionality_values)),
                f"{self.name}_linelikeness_std": float(np.std(linelikeness_values)),
                f"{self.name}_roughness_std": float(np.std(roughness_values))
            }

            self.logger.debug(f"Características Tamura extraídas (tiled): {len(features)}")
            return features

        except Exception as e:
            self.logger.error(f"Error extrayendo características Tamura (tiled): {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return self._get_default_features()

    def _compute_coarseness(self, image: np.ndarray) -> float:
        """
        Calcula el descriptor Coarseness (grosor/granularidad).

        Mide el tamaño promedio de las regiones con intensidad uniforme.
        Valores altos indican textura gruesa, valores bajos textura fina.

        Args:
            image: Imagen normalizada [0, 1]

        Returns:
            float: Valor de coarseness
        """
        rows, cols = image.shape

        # Ajustar kmax si la imagen es pequeña
        effective_kmax = min(self.kmax, int(np.log2(min(rows, cols) // 4)))
        effective_kmax = max(1, effective_kmax)

        # Calcular promedios en ventanas de diferentes tamaños
        # A_k(i,j) = promedio en ventana 2^k x 2^k centrada en (i,j)
        averages = np.zeros((effective_kmax, rows, cols), dtype=np.float32)

        for k in range(effective_kmax):
            window_size = 2 ** (k + 1)
            if window_size > min(rows, cols) // 2:
                break

            kernel = np.ones((window_size, window_size), dtype=np.float32) / (window_size ** 2)

            # Usar cv2 para convolución más rápida
            if window_size > 32:
                averages[k] = cv2.filter2D(image.astype(np.float32), -1, kernel,
                                          borderType=cv2.BORDER_REFLECT)
            else:
                averages[k] = convolve2d(image, kernel, mode='same', boundary='symm')

        # Calcular diferencias usando operaciones vectorizadas
        E_kh = np.zeros((effective_kmax, rows, cols), dtype=np.float32)
        E_kv = np.zeros((effective_kmax, rows, cols), dtype=np.float32)

        for k in range(effective_kmax):
            step = 2 ** k
            if step >= min(rows, cols) // 2:
                break

            # Diferencias horizontales
            if 2*step < rows:
                E_kh[k, step:rows-step, :] = np.abs(
                    averages[k, 2*step:, :] - averages[k, :rows-2*step, :]
                )

            # Diferencias verticales
            if 2*step < cols:
                E_kv[k, :, step:cols-step] = np.abs(
                    averages[k, :, 2*step:] - averages[k, :, :cols-2*step]
                )

        # Encontrar k óptimo para cada píxel
        E_max = np.maximum(E_kh, E_kv)
        k_best = np.argmax(E_max, axis=0)
        S_best = 2.0 ** k_best

        # Coarseness es el promedio de S_best
        coarseness = np.mean(S_best)

        return coarseness

    def _compute_contrast(self, image: np.ndarray) -> float:
        """
        Calcula el descriptor Contrast.

        Mide la distribución de intensidades y el rango dinámico.
        Combina la desviación estándar con la kurtosis.

        Args:
            image: Imagen normalizada [0, 1]

        Returns:
            float: Valor de contrast
        """
        # Calcular estadísticas básicas
        sigma = np.std(image)

        # Calcular kurtosis (cuarto momento)
        pixels = image.flatten()
        alpha4 = kurtosis(pixels, fisher=False)

        # Evitar división por cero
        if alpha4 <= 0:
            alpha4 = 0.0001

        # Fórmula de Tamura para contrast
        contrast = sigma / (alpha4 ** 0.25)

        return contrast

    def _compute_directionality(self, image: np.ndarray) -> float:
        """
        Calcula el descriptor Directionality.

        Mide qué tan fuerte es la orientación predominante de la textura.
        Valores bajos indican textura isotrópica, valores altos indican
        fuerte direccionalidad.

        Args:
            image: Imagen normalizada [0, 1]

        Returns:
            float: Valor de directionality (1 - entropía normalizada)
        """
        # Calcular gradientes usando operadores de Sobel
        grad_x = cv2.Sobel(image.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)

        # Calcular magnitud y dirección
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)

        # Convertir direcciones a grados [0, 180]
        direction_deg = np.degrees(direction) % 180

        # Filtrar píxeles con magnitud significativa
        threshold = self.directionality_threshold * np.max(magnitude)
        significant_pixels = magnitude > threshold

        if not np.any(significant_pixels):
            return 0.0

        # Crear histograma de direcciones ponderado por magnitud
        hist, bins = np.histogram(
            direction_deg[significant_pixels],
            bins=self.hist_bins,
            range=(0, 180),
            weights=magnitude[significant_pixels]
        )

        # Normalizar histograma
        if np.sum(hist) > 0:
            hist = hist / np.sum(hist)

        # Calcular entropía del histograma
        hist_nonzero = hist[hist > 0]
        if len(hist_nonzero) > 0:
            entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))
            max_entropy = np.log2(self.hist_bins)
            normalized_entropy = entropy / max_entropy
            directionality = 1.0 - normalized_entropy
        else:
            directionality = 0.0

        return directionality

    def _compute_linelikeness(self, image: np.ndarray) -> float:
        """
        Calcula el descriptor Line-likeness.

        Mide la presencia de estructuras lineales en la textura.
        Se basa en la correlación de píxeles en diferentes direcciones.

        Args:
            image: Imagen normalizada [0, 1]

        Returns:
            float: Valor de line-likeness
        """
        # Definir direcciones para análisis (0°, 45°, 90°, 135°)
        directions = [(0, 1), (1, 1), (1, 0), (1, -1)]
        correlations = []

        for dy, dx in directions:
            corr = self._directional_correlation(image, dy, dx, max_distance=5)
            correlations.append(corr)

        # Line-likeness es la diferencia entre máxima y mínima correlación
        linelikeness = np.max(correlations) - np.min(correlations)

        return linelikeness

    def _directional_correlation(self, image: np.ndarray, dy: int, dx: int,
                                max_distance: int = 5) -> float:
        """
        Calcula correlación direccional promedio.

        Args:
            image: Imagen normalizada
            dy, dx: Dirección del desplazamiento
            max_distance: Distancia máxima a considerar

        Returns:
            float: Correlación promedio en la dirección dada
        """
        rows, cols = image.shape
        correlations = []

        for d in range(1, max_distance + 1):
            offset_y = d * dy
            offset_x = d * dx

            if offset_y >= 0:
                y1, y2 = 0, rows - offset_y
            else:
                y1, y2 = -offset_y, rows

            if offset_x >= 0:
                x1, x2 = 0, cols - offset_x
            else:
                x1, x2 = -offset_x, cols

            if y2 > y1 and x2 > x1:
                region1 = image[y1:y2, x1:x2]
                region2 = image[y1+offset_y:y2+offset_y, x1+offset_x:x2+offset_x]

                if region1.size > 0:
                    r1_flat = region1.flatten()
                    r2_flat = region2.flatten()

                    if np.std(r1_flat) > 0 and np.std(r2_flat) > 0:
                        corr = np.corrcoef(r1_flat, r2_flat)[0, 1]
                        if np.isfinite(corr):
                            correlations.append(corr)

        if correlations:
            return np.mean(correlations)
        else:
            return 0.0

    def _compute_roughness(self, coarseness: float, contrast: float) -> float:
        """
        Calcula el descriptor Roughness.

        Combina coarseness y contrast para dar una medida general de rugosidad.

        Args:
            coarseness: Valor de coarseness calculado
            contrast: Valor de contrast calculado

        Returns:
            float: Valor de roughness
        """
        roughness = coarseness + contrast
        return roughness

    def get_feature_names(self) -> list:
        """
        Retorna los nombres de las características extraídas.

        Para imágenes grandes (procesamiento por bloques), se retornan 10 características
        (5 medias + 5 desviaciones estándar). Para imágenes pequeñas, se retornan las
        mismas 10 características pero con std=0.

        Returns:
            list: Lista de nombres de características
        """
        return [
            f"{self.name}_coarseness_mean",
            f"{self.name}_coarseness_std",
            f"{self.name}_contrast_mean",
            f"{self.name}_contrast_std",
            f"{self.name}_directionality_mean",
            f"{self.name}_directionality_std",
            f"{self.name}_linelikeness_mean",
            f"{self.name}_linelikeness_std",
            f"{self.name}_roughness_mean",
            f"{self.name}_roughness_std"
        ]

    def _get_default_features(self) -> Dict[str, float]:
        """Retorna características por defecto en caso de error."""
        return {name: 0.0 for name in self.get_feature_names()}
