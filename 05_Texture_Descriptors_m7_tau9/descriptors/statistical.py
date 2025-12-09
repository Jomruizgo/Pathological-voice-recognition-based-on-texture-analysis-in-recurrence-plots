"""
Descriptor Statistical para análisis básico de estadísticas de imagen.

Este descriptor extrae características estadísticas fundamentales que son
la base para cualquier análisis de textura. Incluye:

- Momentos estadísticos (media, desviación estándar, asimetría, curtosis)
- Histograma y percentiles
- Medidas de dispersión y forma de la distribución
- Características de entropía e información

Para Recurrence Plots, estas estadísticas son importantes porque:
- La media indica la densidad promedio de recurrencia
- La desviación estándar mide la variabilidad
- Asimetría y curtosis caracterizan la forma de la distribución
- Percentiles capturan la distribución completa de intensidades
"""

import numpy as np
from typing import Dict, List
from scipy import stats
import warnings

from . import register_descriptor
from .base import BaseDescriptor


@register_descriptor("statistical", enabled_by_default=True)
class StatisticalDescriptor(BaseDescriptor):
    """
    Extractor de características estadísticas básicas de imagen.
    
    Este descriptor calcula estadísticas fundamentales que proporcionan
    una caracterización global de la imagen. Es especialmente útil como
    línea base y complemento a descriptores más especializados.
    
    Características extraídas:
    - Momentos: media, std, asimetría (skewness), curtosis
    - Distribución: percentiles, rango, mediana
    - Información: entropía del histograma
    - Forma: coeficiente de variación, ratio de rangos
    
    Example:
        >>> descriptor = StatisticalDescriptor(
        ...     compute_moments=True,
        ...     compute_histogram=True,
        ...     n_bins=64
        ... )
        >>> features = descriptor.extract(recurrence_plot)
    """
    
    def __init__(self, 
                 compute_moments: bool = True,
                 moments: List[str] = ['mean', 'std', 'skewness', 'kurtosis'],
                 compute_percentiles: bool = True,
                 percentiles: List[int] = [10, 25, 50, 75, 90],
                 compute_histogram: bool = True,
                 n_bins: int = 64,
                 compute_entropy: bool = True):
        """
        Inicializa el descriptor estadístico.
        
        SISTEMA DE CONFIGURACIÓN HÍBRIDO:
        ═══════════════════════════════════════════════════════════════
        📋 Los defaults aquí sirven como documentación y fallback
        🔄 config.py::DEFAULT_DESCRIPTORS['statistical'] tiene prioridad en pipeline
        📖 Ver config.py línea ~90 para configuración real del pipeline
        ═══════════════════════════════════════════════════════════════
        
        Args:
            compute_moments (bool): Si calcular momentos estadísticos.
            
            moments (List[str]): Qué momentos calcular.
                                Opciones: 'mean', 'std', 'var', 'skewness', 'kurtosis'
                                - mean: valor promedio de píxeles
                                - std: dispersión de valores
                                - var: varianza (std²)
                                - skewness: asimetría de la distribución
                                - kurtosis: forma de las colas de la distribución
            
            compute_percentiles (bool): Si calcular percentiles.
            
            percentiles (List[int]): Qué percentiles calcular (0-100).
                                    Ejemplo: [25, 50, 75] calcula cuartiles.
                                    Útil para entender la distribución completa.
            
            compute_histogram (bool): Si calcular características del histograma.
            
            n_bins (int): Número de bins para el histograma.
                         Más bins = mayor resolución pero más características.
                         
            compute_entropy (bool): Si calcular entropía del histograma.
                                   Mide el desorden/información en la imagen.
        """
        super().__init__(
            compute_moments=compute_moments,
            moments=moments,
            compute_percentiles=compute_percentiles,
            percentiles=percentiles,
            compute_histogram=compute_histogram,
            n_bins=n_bins,
            compute_entropy=compute_entropy
        )
        
        # Validar parámetros
        self._validate_parameters()
        
        # Cache para nombres de características
        self._feature_names_cache = None
        
        self.logger.info(f"Statistical inicializado: momentos={compute_moments}, "
                        f"percentiles={compute_percentiles}, histograma={compute_histogram}")
    
    def _validate_parameters(self):
        """Valida que los parámetros sean correctos."""
        
        # Validar momentos
        valid_moments = ['mean', 'std', 'var', 'skewness', 'kurtosis']
        invalid_moments = set(self.config['moments']) - set(valid_moments)
        if invalid_moments:
            raise ValueError(f"Momentos inválidos: {invalid_moments}. "
                           f"Válidos: {valid_moments}")
        
        # Validar percentiles
        if self.config['compute_percentiles']:
            percentiles = self.config['percentiles']
            if not all(0 <= p <= 100 for p in percentiles):
                raise ValueError("Percentiles deben estar entre 0 y 100")
        
        # Validar n_bins
        if self.config['n_bins'] < 2:
            raise ValueError("n_bins debe ser >= 2")
    
    @property
    def name(self) -> str:
        """Nombre del descriptor."""
        return "statistical"
    
    def extract(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extrae características estadísticas de una imagen.
        
        Args:
            image (np.ndarray): Imagen en escala de grises [0-255]
            
        Returns:
            Dict[str, float]: Características extraídas con nombres como:
                             'stat_mean', 'stat_std', 'stat_p50', 'stat_entropy', etc.
        """
        features = {}
        
        # Convertir imagen a array plano para cálculos estadísticos
        image_flat = image.flatten().astype(np.float64)
        
        # 1. MOMENTOS ESTADÍSTICOS
        if self.config['compute_moments']:
            
            if 'mean' in self.config['moments']:
                features['stat_mean'] = float(np.mean(image_flat))
            
            if 'std' in self.config['moments']:
                features['stat_std'] = float(np.std(image_flat))
            
            if 'var' in self.config['moments']:
                features['stat_var'] = float(np.var(image_flat))
            
            if 'skewness' in self.config['moments']:
                # Asimetría: mide si la distribución está sesgada
                # > 0: cola derecha más larga, < 0: cola izquierda más larga
                features['stat_skewness'] = float(stats.skew(image_flat))
            
            if 'kurtosis' in self.config['moments']:
                # Curtosis: mide la forma de las colas de la distribución
                # > 0: colas más pesadas que normal, < 0: colas más ligeras
                features['stat_kurtosis'] = float(stats.kurtosis(image_flat))
        
        # 2. PERCENTILES
        if self.config['compute_percentiles']:
            percentile_values = np.percentile(image_flat, self.config['percentiles'])
            
            for p, value in zip(self.config['percentiles'], percentile_values):
                features[f'stat_p{p}'] = float(value)
        
        # 3. CARACTERÍSTICAS ADICIONALES DE DISTRIBUCIÓN
        
        # Rango de valores
        features['stat_range'] = float(np.max(image_flat) - np.min(image_flat))
        features['stat_min'] = float(np.min(image_flat))
        features['stat_max'] = float(np.max(image_flat))
        
        # Coeficiente de variación (std/mean) - normaliza la variabilidad
        mean_val = np.mean(image_flat)
        if mean_val != 0:
            features['stat_cv'] = float(np.std(image_flat) / mean_val)
        else:
            features['stat_cv'] = 0.0
        
        # Rango intercuartílico (IQR) - medida robusta de dispersión
        q75, q25 = np.percentile(image_flat, [75, 25])
        features['stat_iqr'] = float(q75 - q25)
        
        # 4. HISTOGRAMA Y ENTROPÍA
        if self.config['compute_histogram'] or self.config['compute_entropy']:
            
            # Calcular histograma
            hist, bin_edges = np.histogram(image_flat, bins=self.config['n_bins'], 
                                         range=(0, 255), density=True)
            
            if self.config['compute_histogram']:
                # Características del histograma normalizado
                for i, val in enumerate(hist):
                    features[f'stat_hist_bin_{i}'] = float(val)
                
                # Estadísticas del histograma
                features['stat_hist_mean'] = float(np.mean(hist))
                features['stat_hist_std'] = float(np.std(hist))
                features['stat_hist_max'] = float(np.max(hist))
                
                # Número efectivo de bins (bins con contenido significativo)
                significant_bins = np.sum(hist > np.max(hist) * 0.01)  # >1% del máximo
                features['stat_hist_effective_bins'] = float(significant_bins)
            
            if self.config['compute_entropy']:
                # Entropía de Shannon del histograma
                # Mide la cantidad de información/desorden en la imagen
                hist_nonzero = hist[hist > 0]  # Evitar log(0)
                if len(hist_nonzero) > 0:
                    entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))
                    features['stat_entropy'] = float(entropy)
                else:
                    features['stat_entropy'] = 0.0
                
                # Entropía normalizada (dividida por log2(n_bins))
                max_entropy = np.log2(self.config['n_bins'])
                if max_entropy > 0:
                    features['stat_entropy_norm'] = float(features['stat_entropy'] / max_entropy)
                else:
                    features['stat_entropy_norm'] = 0.0
        
        # 5. CARACTERÍSTICAS GEOMÉTRICAS BÁSICAS
        
        # Momentos espaciales básicos (centroide de masa)
        height, width = image.shape
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        
        # Centroide ponderado por intensidad
        total_intensity = np.sum(image_flat)
        if total_intensity > 0:
            centroid_x = float(np.sum(x_coords * image) / total_intensity)
            centroid_y = float(np.sum(y_coords * image) / total_intensity)
            
            features['stat_centroid_x'] = centroid_x / width  # Normalizado
            features['stat_centroid_y'] = centroid_y / height  # Normalizado
        else:
            features['stat_centroid_x'] = 0.5  # Centro por defecto
            features['stat_centroid_y'] = 0.5
        
        self.logger.debug(f"Statistical extraído: {len(features)} características")
        return features
    
    def get_feature_names(self) -> List[str]:
        """
        Retorna nombres de todas las características que extrae.
        
        Returns:
            List[str]: Lista ordenada de nombres de características
        """
        if self._feature_names_cache is None:
            names = []
            
            # Momentos
            if self.config['compute_moments']:
                for moment in self.config['moments']:
                    names.append(f'stat_{moment}')
            
            # Percentiles
            if self.config['compute_percentiles']:
                for p in self.config['percentiles']:
                    names.append(f'stat_p{p}')
            
            # Características básicas de distribución
            names.extend([
                'stat_range', 'stat_min', 'stat_max', 'stat_cv', 'stat_iqr'
            ])
            
            # Histograma
            if self.config['compute_histogram']:
                for i in range(self.config['n_bins']):
                    names.append(f'stat_hist_bin_{i}')
                names.extend([
                    'stat_hist_mean', 'stat_hist_std', 'stat_hist_max', 
                    'stat_hist_effective_bins'
                ])
            
            # Entropía
            if self.config['compute_entropy']:
                names.extend(['stat_entropy', 'stat_entropy_norm'])
            
            # Características geométricas
            names.extend(['stat_centroid_x', 'stat_centroid_y'])
            
            self._feature_names_cache = sorted(names)
        
        return self._feature_names_cache
    
    def get_description(self) -> str:
        """
        Retorna descripción detallada del descriptor.
        
        Returns:
            str: Descripción de qué mide este descriptor
        """
        components = []
        
        if self.config['compute_moments']:
            components.append(f"momentos ({', '.join(self.config['moments'])})")
        
        if self.config['compute_percentiles']:
            components.append(f"percentiles ({len(self.config['percentiles'])} valores)")
        
        if self.config['compute_histogram']:
            components.append(f"histograma ({self.config['n_bins']} bins)")
        
        if self.config['compute_entropy']:
            components.append("entropía")
        
        return (
            f"Descriptor estadístico que extrae: {', '.join(components)}. "
            f"Proporciona caracterización global de la distribución de intensidades. "
            f"Total de características: {len(self.get_feature_names())}"
        )