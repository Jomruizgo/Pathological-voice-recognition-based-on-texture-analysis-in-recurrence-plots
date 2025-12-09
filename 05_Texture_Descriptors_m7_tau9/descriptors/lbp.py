"""
Descriptor LBP (Local Binary Patterns) para análisis de textura.

LBP es un descriptor muy eficiente que caracteriza la textura local
comparando cada píxel con sus vecinos en un patrón circular.

Para Recurrence Plots, LBP es especialmente útil porque:
- Detecta micro-patrones y estructuras locales
- Es robusto a cambios de iluminación
- Captura la organización espacial de los puntos recurrentes
- Puede detectar patrones direccionales y repetitivos

El algoritmo:
1. Para cada píxel, compara su valor con N vecinos en un círculo de radio R
2. Si el vecino >= píxel central, asigna 1, sino 0
3. El patrón binario resultante se convierte a decimal
4. Se cuenta la frecuencia de cada patrón en la imagen
"""

import numpy as np
from typing import Dict, List
from skimage.feature import local_binary_pattern
import warnings

from . import register_descriptor
from .base import BaseDescriptor


@register_descriptor("lbp", enabled_by_default=True)
class LBPDescriptor(BaseDescriptor):
    """
    Extractor de características LBP (Local Binary Patterns).
    
    LBP analiza la estructura local de la textura comparando cada píxel
    con sus vecinos circulares. Es muy efectivo para:
    
    - Detectar micro-texturas en Recurrence Plots
    - Identificar patrones repetitivos locales
    - Análisis robusto independiente de la iluminación
    - Caracterización de la rugosidad y uniformidad local
    
    Example:
        >>> descriptor = LBPDescriptor(
        ...     radius=[1, 2, 3], 
        ...     n_points=[8, 16, 24],
        ...     method='uniform'
        ... )
        >>> features = descriptor.extract(recurrence_plot)
    """
    
    def __init__(self, 
                 radius: List[int] = [1, 2, 3], 
                 n_points: List[int] = [8, 16, 24],
                 method: str = 'uniform'):
        """
        Inicializa el descriptor LBP.
        
        SISTEMA DE CONFIGURACIÓN HÍBRIDO:
        ═══════════════════════════════════════════════════════════════
        📋 Los defaults en esta función sirven como:
           • Documentación de valores recomendados
           • Fallback cuando se usa el descriptor directamente
           
        🔄 Cuando se usa a través del pipeline principal:
           • config.py::DEFAULT_DESCRIPTORS['lbp'] sobrescribe estos valores
           • Los defaults aquí son ignorados
           
        📖 Para ver la configuración REAL usada en el pipeline:
           • Ver: config.py línea ~64: DEFAULT_DESCRIPTORS['lbp']
        ═══════════════════════════════════════════════════════════════
        
        Args:
            radius (List[int]): Radios del vecindario circular a analizar.
                               Default: [1, 2, 3] - Multi-escala desde local hasta medio alcance
                               • 1: Patrones micro-locales (vecinos inmediatos)
                               • 2: Patrones locales (un píxel de separación)  
                               • 3: Patrones de medio alcance (mayor contexto)
                               
            n_points (List[int]): Número de puntos de muestreo en cada círculo.
                                 Default: [8, 16, 24] - Diferentes resoluciones angulares
                                 • 8: Resolución básica (menos sensible a ruido)
                                 • 16: Resolución media (balance precisión/robustez)
                                 • 24: Alta resolución (máximo detalle)
                                 
            method (str): Método de LBP a usar.
                         Default: 'uniform' - Recomendado para robustez
                         • 'uniform': Solo patrones con ≤2 transiciones 0→1
                         • 'ror': Rotation invariant uniform
                         • 'nri_uniform': Non-rotation invariant uniform  
                         • 'var': Variance measure
                         
        Note:
            ⚠️  IMPORTANTE: Cuando uses el pipeline (main.py, GUI), estos defaults
               son IGNORADOS y se usa la configuración de config.py en su lugar.
               
            ✅ Para uso directo: LBPDescriptor() usa estos defaults
            🔄 Para pipeline: ModularPipeline() usa config.py
        """
        super().__init__(
            radius=radius,
            n_points=n_points,
            method=method
        )
        
        # Validar parámetros
        self._validate_parameters()
        
        # Cache para nombres de características
        self._feature_names_cache = None
        
        self.logger.info(f"LBP inicializado: {len(radius)} radios, "
                        f"{len(n_points)} configuraciones de puntos, método '{method}'")
    
    def _validate_parameters(self):
        """Valida que los parámetros sean correctos."""
        
        # Validar radios
        if not self.config['radius'] or not all(isinstance(r, int) and r > 0 
                                               for r in self.config['radius']):
            raise ValueError("radius debe ser una lista de enteros positivos")
        
        # Validar n_points
        if not self.config['n_points'] or not all(isinstance(n, int) and n >= 3 
                                                 for n in self.config['n_points']):
            raise ValueError("n_points debe ser una lista de enteros >= 3")
        
        # Validar método
        valid_methods = ['uniform', 'ror', 'nri_uniform', 'var']
        if self.config['method'] not in valid_methods:
            raise ValueError(f"method debe ser uno de: {valid_methods}")
    
    @property
    def name(self) -> str:
        """Nombre del descriptor."""
        return "lbp"
    
    def extract(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extrae características LBP de una imagen.
        
        El proceso es:
        1. Para cada combinación (radio, n_points), calcular LBP
        2. Obtener histograma de patrones LBP
        3. Calcular estadísticas del histograma como características
        4. Retornar características con nombres descriptivos
        
        Args:
            image (np.ndarray): Imagen en escala de grises [0-255]
            
        Returns:
            Dict[str, float]: Características extraídas con nombres como:
                             'lbp_hist_bin_0_r1_p8', 'lbp_uniformity_r2_p16', etc.
        """
        features = {}
        
        # Convertir a float para evitar problemas numéricos
        image_float = image.astype(np.float64)
        
        # Procesar cada combinación de parámetros
        for radius in self.config['radius']:
            for n_points in self.config['n_points']:
                
                # Calcular LBP
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    lbp = local_binary_pattern(
                        image_float, 
                        n_points, 
                        radius, 
                        method=self.config['method']
                    )
                
                # Crear sufijo para nombres de características
                suffix = f"_r{radius}_p{n_points}"
                
                # Calcular histograma de patrones LBP
                if self.config['method'] == 'uniform':
                    # Para uniform, hay n_points + 2 bins (n_points uniformes + 1 para no-uniformes + 1 extra)
                    n_bins = n_points + 2
                else:
                    # Para otros métodos, usar el rango completo
                    n_bins = int(lbp.max()) + 1
                
                hist, _ = np.histogram(lbp.ravel(), bins=n_bins, 
                                     range=(0, n_bins), density=True)
                
                # Características del histograma
                # 1. Valores del histograma normalizado
                for i, val in enumerate(hist):
                    features[f"lbp_hist_bin_{i}{suffix}"] = float(val)
                
                # 2. Estadísticas globales del histograma
                features[f"lbp_mean{suffix}"] = float(np.mean(lbp))
                features[f"lbp_std{suffix}"] = float(np.std(lbp))
                features[f"lbp_max{suffix}"] = float(np.max(lbp))
                features[f"lbp_min{suffix}"] = float(np.min(lbp))
                
                # 3. Medidas de uniformidad y variabilidad
                # Uniformity: qué tan concentrado está el histograma
                features[f"lbp_uniformity{suffix}"] = float(np.sum(hist ** 2))
                
                # Entropy: medida de desorden en el histograma
                hist_nonzero = hist[hist > 0]  # Evitar log(0)
                if len(hist_nonzero) > 0:
                    entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))
                    features[f"lbp_entropy{suffix}"] = float(entropy)
                else:
                    features[f"lbp_entropy{suffix}"] = 0.0
                
                # 4. Si es método uniform, características especiales
                if self.config['method'] == 'uniform':
                    # Proporción de patrones uniformes vs no-uniformes
                    uniform_sum = np.sum(hist[:-2])  # Todos excepto los últimos 2 bins
                    non_uniform = hist[-2]  # Penúltimo bin son los no-uniformes
                    
                    features[f"lbp_uniform_ratio{suffix}"] = float(uniform_sum)
                    features[f"lbp_nonuniform_ratio{suffix}"] = float(non_uniform)
        
        self.logger.debug(f"LBP extraído: {len(features)} características")
        return features
    
    def get_feature_names(self) -> List[str]:
        """
        Retorna nombres de todas las características que extrae.
        
        Los nombres incluyen:
        - Histograma: 'lbp_hist_bin_{i}_r{radio}_p{puntos}'
        - Estadísticas: 'lbp_mean_r{radio}_p{puntos}', etc.
        - Uniformidad: 'lbp_uniformity_r{radio}_p{puntos}'
        
        Returns:
            List[str]: Lista ordenada de nombres de características
        """
        if self._feature_names_cache is None:
            names = []
            
            for radius in self.config['radius']:
                for n_points in self.config['n_points']:
                    suffix = f"_r{radius}_p{n_points}"
                    
                    # Nombres para histograma
                    if self.config['method'] == 'uniform':
                        n_bins = n_points + 2
                    else:
                        # Para estimación, usar un valor típico
                        n_bins = 2 ** n_points if n_points <= 8 else 256
                    
                    for i in range(n_bins):
                        names.append(f"lbp_hist_bin_{i}{suffix}")
                    
                    # Nombres para estadísticas
                    names.extend([
                        f"lbp_mean{suffix}",
                        f"lbp_std{suffix}",
                        f"lbp_max{suffix}",
                        f"lbp_min{suffix}",
                        f"lbp_uniformity{suffix}",
                        f"lbp_entropy{suffix}"
                    ])
                    
                    # Nombres adicionales para método uniform
                    if self.config['method'] == 'uniform':
                        names.extend([
                            f"lbp_uniform_ratio{suffix}",
                            f"lbp_nonuniform_ratio{suffix}"
                        ])
            
            self._feature_names_cache = sorted(names)
        
        return self._feature_names_cache
    
    def get_description(self) -> str:
        """
        Retorna descripción detallada del descriptor.
        
        Returns:
            str: Descripción de qué mide este descriptor
        """
        return (
            f"LBP (Local Binary Patterns) con {len(self.config['radius'])} radios "
            f"({self.config['radius']}), {len(self.config['n_points'])} configuraciones "
            f"de puntos ({self.config['n_points']}) y método '{self.config['method']}'. "
            f"Analiza patrones de textura local comparando píxeles con sus vecinos "
            f"circulares. Total estimado de características: ~{len(self.get_feature_names())}"
        )