"""
Descriptor RQA (Recurrence Quantification Analysis) para Recurrence Plots.

Este módulo implementa métricas cuantitativas de análisis de recurrencia
que caracterizan las estructuras dinámicas en los Recurrence Plots.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
from scipy import ndimage
from dataclasses import dataclass

from .base import BaseDescriptor
from . import register_descriptor
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


@dataclass
class RQAMetrics:
    """Contenedor para métricas RQA."""
    RR: float      # Recurrence Rate
    DET: float     # Determinism
    LAM: float     # Laminarity  
    ENTR: float    # Entropy
    L_max: float   # Maximum diagonal line length
    L_mean: float  # Mean diagonal line length
    V_max: float   # Maximum vertical line length
    V_mean: float  # Mean vertical line length
    TT: float      # Trapping time
    DIV: float     # Divergence (1/L_max)


@register_descriptor("rqa", enabled_by_default=True)
class RQADescriptor(BaseDescriptor):
    """
    Implementa Recurrence Quantification Analysis (RQA).
    
    RQA extrae medidas cuantitativas de las estructuras en Recurrence Plots,
    proporcionando información sobre la dinámica del sistema subyacente.
    """
    
    def __init__(self, epsilon: Optional[float] = None, min_line_length: int = 2):
        """
        Inicializa el descriptor RQA.
        
        Args:
            epsilon (float, optional): Umbral para binarización. Si None, usa percentil 10
            min_line_length (int): Longitud mínima de línea a considerar
        """
        super().__init__()
        self.epsilon = epsilon
        self.min_line_length = min_line_length
        self.logger = logging.getLogger(__name__)
        
    @property
    def name(self) -> str:
        """Nombre del descriptor."""
        return "rqa"
    
    @property
    def required_image_type(self) -> str:
        """Tipo de imagen requerida."""
        return "grayscale"
    
    def extract(self, image: np.ndarray, **kwargs) -> Dict[str, float]:
        """
        Extrae características RQA del Recurrence Plot.
        
        Args:
            image (np.ndarray): Recurrence Plot en escala de grises
            **kwargs: Parámetros adicionales
            
        Returns:
            Dict[str, float]: Métricas RQA calculadas
        """
        self.logger.debug(f"Extrayendo características RQA de imagen {image.shape}")
        
        # Validar imagen
        if len(image.shape) != 2:
            self.logger.warning("RQA requiere imagen en escala de grises")
            return self._get_default_features()
        
        if image.size == 0 or image.shape[0] < 10 or image.shape[1] < 10:
            self.logger.warning("Imagen demasiado pequeña para RQA")
            return self._get_default_features()
        
        # Binarizar RP si es necesario
        binary_rp = self._binarize_rp(image)
        
        # Calcular métricas RQA
        metrics = self._calculate_rqa_metrics(binary_rp)
        
        # Convertir a diccionario de características
        features = {
            f"{self.name}_RR": metrics.RR,
            f"{self.name}_DET": metrics.DET,
            f"{self.name}_LAM": metrics.LAM,
            f"{self.name}_ENTR": metrics.ENTR,
            f"{self.name}_L_max": metrics.L_max,
            f"{self.name}_L_mean": metrics.L_mean,
            f"{self.name}_V_max": metrics.V_max,
            f"{self.name}_V_mean": metrics.V_mean,
            f"{self.name}_TT": metrics.TT,
            f"{self.name}_DIV": metrics.DIV
        }
        
        self.logger.debug(f"Características RQA extraídas: RR={metrics.RR:.3f}, DET={metrics.DET:.3f}")
        
        return features
    
    def _binarize_rp(self, rp: np.ndarray) -> np.ndarray:
        """
        Binariza el Recurrence Plot usando umbral.
        
        En un RP típico:
        - Píxeles oscuros (valores bajos) = recurrencia (deben ser 1)
        - Píxeles claros (valores altos) = no recurrencia (deben ser 0)
        
        Args:
            rp (np.ndarray): RP en escala de grises (0-255)
            
        Returns:
            np.ndarray: RP binarizado (0 o 1)
        """
        # Normalizar a [0, 1] si está en [0, 255]
        if rp.max() > 1.0:
            rp_norm = rp.astype(np.float32) / 255.0
        else:
            rp_norm = rp.astype(np.float32)
            
        if self.epsilon is not None:
            # Usar epsilon definido (debe estar en [0, 1])
            threshold = self.epsilon
        else:
            # Usar percentil como umbral adaptativo
            # Valores MENORES al percentil serán considerados recurrentes
            threshold = np.percentile(rp_norm, 10)  # 10% más oscuro es recurrente
        
        # Los valores MENORES al threshold son recurrencia (1)
        # Los valores MAYORES al threshold son no-recurrencia (0)
        binary_rp = (rp_norm <= threshold).astype(np.uint8)
        
        # Asegurar simetría
        binary_rp = np.maximum(binary_rp, binary_rp.T)
        
        return binary_rp
    
    def _calculate_rqa_metrics(self, binary_rp: np.ndarray) -> RQAMetrics:
        """
        Calcula todas las métricas RQA.
        
        Args:
            binary_rp (np.ndarray): RP binarizado
            
        Returns:
            RQAMetrics: Métricas calculadas
        """
        N = binary_rp.shape[0]
        
        # Recurrence Rate (RR)
        RR = np.sum(binary_rp) / (N * N)
        
        # Encontrar líneas diagonales
        diag_lines = self._find_diagonal_lines(binary_rp)
        diag_hist = self._compute_line_histogram(diag_lines)
        
        # Determinism (DET)
        if len(diag_hist) > 0:
            total_diag_points = sum(length * count for length, count in diag_hist.items() 
                                  if length >= self.min_line_length)
            DET = total_diag_points / max(np.sum(binary_rp), 1)
        else:
            DET = 0.0
        
        # Líneas diagonales estadísticas
        L_max, L_mean, ENTR = self._compute_line_statistics(diag_hist)
        
        # Encontrar líneas verticales
        vert_lines = self._find_vertical_lines(binary_rp)
        vert_hist = self._compute_line_histogram(vert_lines)
        
        # Laminarity (LAM)
        if len(vert_hist) > 0:
            total_vert_points = sum(length * count for length, count in vert_hist.items() 
                                  if length >= self.min_line_length)
            LAM = total_vert_points / max(np.sum(binary_rp), 1)
        else:
            LAM = 0.0
        
        # Líneas verticales estadísticas
        V_max, V_mean, _ = self._compute_line_statistics(vert_hist)
        
        # Trapping Time (TT)
        TT = V_mean
        
        # Divergence (DIV)
        DIV = 1.0 / L_max if L_max > 0 else float('inf')
        
        return RQAMetrics(
            RR=RR,
            DET=DET,
            LAM=LAM,
            ENTR=ENTR,
            L_max=L_max,
            L_mean=L_mean,
            V_max=V_max,
            V_mean=V_mean,
            TT=TT,
            DIV=DIV
        )
    
    def _find_diagonal_lines(self, binary_rp: np.ndarray) -> List[int]:
        """
        Encuentra todas las líneas diagonales en el RP.
        
        Args:
            binary_rp (np.ndarray): RP binarizado
            
        Returns:
            List[int]: Lista de longitudes de líneas diagonales
        """
        lines = []
        N = binary_rp.shape[0]
        
        # Buscar en todas las diagonales (excluyendo la principal)
        for k in range(1, N):
            # Diagonal superior
            diag = np.diagonal(binary_rp, k)
            lines.extend(self._extract_line_lengths(diag))
            
            # Diagonal inferior
            diag = np.diagonal(binary_rp, -k)
            lines.extend(self._extract_line_lengths(diag))
        
        return lines
    
    def _find_vertical_lines(self, binary_rp: np.ndarray) -> List[int]:
        """
        Encuentra todas las líneas verticales en el RP.
        
        Args:
            binary_rp (np.ndarray): RP binarizado
            
        Returns:
            List[int]: Lista de longitudes de líneas verticales
        """
        lines = []
        N = binary_rp.shape[0]
        
        # Buscar en cada columna
        for i in range(N):
            col = binary_rp[:, i]
            lines.extend(self._extract_line_lengths(col))
        
        return lines
    
    def _extract_line_lengths(self, sequence: np.ndarray) -> List[int]:
        """
        Extrae longitudes de líneas consecutivas de 1s en una secuencia.
        
        Args:
            sequence (np.ndarray): Secuencia binaria
            
        Returns:
            List[int]: Lista de longitudes de líneas
        """
        if len(sequence) == 0:
            return []
        
        # Agregar 0s al inicio y final para detectar bordes
        padded = np.concatenate(([0], sequence, [0]))
        
        # Encontrar cambios de 0 a 1 y de 1 a 0
        changes = np.diff(padded)
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        
        # Calcular longitudes
        lengths = ends - starts
        
        return lengths.tolist()
    
    def _compute_line_histogram(self, lines: List[int]) -> Dict[int, int]:
        """
        Calcula histograma de longitudes de línea.
        
        Args:
            lines (List[int]): Lista de longitudes
            
        Returns:
            Dict[int, int]: Histograma {longitud: frecuencia}
        """
        histogram = {}
        for length in lines:
            if length >= self.min_line_length:
                histogram[length] = histogram.get(length, 0) + 1
        return histogram
    
    def _compute_line_statistics(self, histogram: Dict[int, int]) -> Tuple[float, float, float]:
        """
        Calcula estadísticas de líneas desde histograma.
        
        Args:
            histogram (Dict[int, int]): Histograma de longitudes
            
        Returns:
            Tuple[float, float, float]: (max_length, mean_length, entropy)
        """
        if not histogram:
            return 0.0, 0.0, 0.0
        
        lengths = list(histogram.keys())
        counts = list(histogram.values())
        
        # Longitud máxima
        max_length = float(max(lengths))
        
        # Longitud media ponderada
        total_points = sum(l * c for l, c in zip(lengths, counts))
        total_lines = sum(counts)
        mean_length = total_points / total_lines if total_lines > 0 else 0.0
        
        # Entropía de Shannon de la distribución de longitudes
        probabilities = np.array(counts) / total_lines
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-15))
        
        return max_length, mean_length, entropy
    
    def _get_default_features(self) -> Dict[str, float]:
        """
        Retorna características por defecto cuando falla la extracción.
        
        Returns:
            Dict[str, float]: Características con valores por defecto
        """
        return {
            f"{self.name}_RR": 0.0,
            f"{self.name}_DET": 0.0,
            f"{self.name}_LAM": 0.0,
            f"{self.name}_ENTR": 0.0,
            f"{self.name}_L_max": 0.0,
            f"{self.name}_L_mean": 0.0,
            f"{self.name}_V_max": 0.0,
            f"{self.name}_V_mean": 0.0,
            f"{self.name}_TT": 0.0,
            f"{self.name}_DIV": float('inf')
        }
    
    def get_feature_names(self) -> List[str]:
        """
        Retorna lista de nombres de características.
        
        Returns:
            List[str]: Lista de nombres de características RQA
        """
        return [
            f"{self.name}_RR",
            f"{self.name}_DET",
            f"{self.name}_LAM",
            f"{self.name}_ENTR",
            f"{self.name}_L_max",
            f"{self.name}_L_mean",
            f"{self.name}_V_max",
            f"{self.name}_V_mean",
            f"{self.name}_TT",
            f"{self.name}_DIV"
        ]
    
    def get_feature_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Retorna información sobre las características extraídas.
        
        Returns:
            Dict[str, Dict[str, Any]]: Información de cada característica
        """
        return {
            f"{self.name}_RR": {
                "description": "Recurrence Rate - Densidad de puntos recurrentes",
                "range": [0, 1],
                "units": "ratio"
            },
            f"{self.name}_DET": {
                "description": "Determinism - Predictibilidad del sistema",
                "range": [0, 1],
                "units": "ratio"
            },
            f"{self.name}_LAM": {
                "description": "Laminarity - Estados laminares/intermitentes",
                "range": [0, 1],
                "units": "ratio"
            },
            f"{self.name}_ENTR": {
                "description": "Entropy - Complejidad de líneas diagonales",
                "range": [0, float('inf')],
                "units": "bits"
            },
            f"{self.name}_L_max": {
                "description": "Longitud máxima de línea diagonal",
                "range": [0, float('inf')],
                "units": "points"
            },
            f"{self.name}_L_mean": {
                "description": "Longitud media de líneas diagonales",
                "range": [0, float('inf')],
                "units": "points"
            },
            f"{self.name}_V_max": {
                "description": "Longitud máxima de línea vertical",
                "range": [0, float('inf')],
                "units": "points"
            },
            f"{self.name}_V_mean": {
                "description": "Longitud media de líneas verticales",
                "range": [0, float('inf')],
                "units": "points"
            },
            f"{self.name}_TT": {
                "description": "Trapping Time - Tiempo en estados laminares",
                "range": [0, float('inf')],
                "units": "points"
            },
            f"{self.name}_DIV": {
                "description": "Divergence - Inverso de L_max",
                "range": [0, float('inf')],
                "units": "1/points"
            }
        }