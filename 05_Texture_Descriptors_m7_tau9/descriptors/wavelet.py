"""
Descriptor Wavelet para análisis de textura multi-escala.

Las transformadas Wavelet descomponen una imagen en diferentes escalas y
orientaciones simultáneamente, proporcionando un análisis tiempo-frecuencia
localizado. Son especialmente útiles para texturas con características
multi-escala.

Para Recurrence Plots, las wavelets son valiosas porque:
- Analizan patrones recurrentes a múltiples escalas temporales
- Detectan estructuras jerárquicas en la recurrencia
- Capturan tanto detalles finos como patrones globales
- Proporcionan descomposición direccional (horizontal, vertical, diagonal)

La transformada wavelet descompone la imagen en:
- Aproximación: contenido de baja frecuencia (estructura global)
- Detalles: contenido de alta frecuencia en diferentes orientaciones
"""

import numpy as np
from typing import Dict, List
import pywt
from scipy import stats
import warnings

from . import register_descriptor
from .base import BaseDescriptor


@register_descriptor("wavelet", enabled_by_default=True)
class WaveletDescriptor(BaseDescriptor):
    """
    Extractor de características usando transformada Wavelet.
    
    La transformada wavelet proporciona análisis multi-resolución,
    descomponiendo la imagen en componentes de aproximación y detalle
    a diferentes escalas y orientaciones.
    
    Especialmente útil para:
    - Análisis multi-escala de Recurrence Plots
    - Detección de estructuras jerárquicas
    - Caracterización de rugosidad a diferentes escalas
    - Análisis direccional de patrones
    
    Example:
        >>> descriptor = WaveletDescriptor(
        ...     wavelet='db4', 
        ...     levels=3,
        ...     feature_types=['energy', 'entropy']
        ... )
        >>> features = descriptor.extract(recurrence_plot)
    """
    
    def __init__(self, 
                 wavelet: str = 'db4',
                 levels: int = 3,
                 feature_types: List[str] = ['energy', 'entropy', 'mean', 'std'],
                 compute_ratios: bool = True):
        """
        Inicializa el descriptor Wavelet.
        
        Args:
            wavelet (str): Tipo de wavelet a usar.
                          Opciones populares:
                          - 'db4', 'db8': Daubechies (buenas para texturas generales)
                          - 'haar': Simple y rápida (buena para bordes)
                          - 'bior2.2': Biorthogonal (buena para compresión)
                          - 'coif2': Coiflets (balanceada)
                          
            levels (int): Número de niveles de descomposición.
                         Más niveles = análisis más grueso.
                         Típicamente 2-4 niveles.
                         
            feature_types (List[str]): Tipos de características a extraer.
                                      Opciones:
                                      - 'energy': Energía de cada subband
                                      - 'entropy': Entropía de cada subband  
                                      - 'mean': Valor promedio
                                      - 'std': Desviación estándar
                                      - 'var': Varianza
                                      
            compute_ratios (bool): Si calcular ratios entre subbands.
                                  Útil para caracterizar la distribución
                                  relativa de energía entre escalas.
        """
        super().__init__(
            wavelet=wavelet,
            levels=levels,
            feature_types=feature_types,
            compute_ratios=compute_ratios
        )
        
        # Validar parámetros
        self._validate_parameters()
        
        # Cache para nombres de características
        self._feature_names_cache = None
        
        self.logger.info(f"Wavelet inicializado: {wavelet}, {levels} niveles, "
                        f"características: {feature_types}")
    
    def _validate_parameters(self):
        """Valida que los parámetros sean correctos."""
        
        # Validar wavelet
        available_wavelets = pywt.wavelist(kind='discrete')
        if self.config['wavelet'] not in available_wavelets:
            raise ValueError(f"Wavelet '{self.config['wavelet']}' no válida. "
                           f"Disponibles: {available_wavelets}")
        
        # Validar levels
        if not isinstance(self.config['levels'], int) or self.config['levels'] < 1:
            raise ValueError("levels debe ser un entero >= 1")
        
        if self.config['levels'] > 6:
            self.logger.warning("Más de 6 niveles puede ser excesivo para la mayoría de imágenes")
        
        # Validar feature_types
        valid_features = ['energy', 'entropy', 'mean', 'std', 'var']
        invalid_features = set(self.config['feature_types']) - set(valid_features)
        if invalid_features:
            raise ValueError(f"Características inválidas: {invalid_features}. "
                           f"Válidas: {valid_features}")
    
    @property
    def name(self) -> str:
        """Nombre del descriptor."""
        return "wavelet"
    
    def extract(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extrae características Wavelet de una imagen.
        
        El proceso es:
        1. Aplicar transformada wavelet multi-nivel
        2. Para cada subband (aproximación y detalles), extraer características
        3. Calcular ratios entre subbands si está habilitado
        4. Retornar características con nombres descriptivos
        
        Args:
            image (np.ndarray): Imagen en escala de grises [0-255]
            
        Returns:
            Dict[str, float]: Características extraídas con nombres como:
                             'wavelet_energy_approx_L3', 'wavelet_std_detail_H_L2', etc.
        """
        features = {}
        
        # Convertir a float para evitar problemas numéricos
        image_float = image.astype(np.float64)
        
        # Aplicar transformada wavelet multi-nivel
        coeffs = pywt.wavedec2(image_float, self.config['wavelet'], 
                              level=self.config['levels'], mode='symmetric')
        
        # coeffs[0] es la aproximación del último nivel
        # coeffs[1:] son las tuplas (cH, cV, cD) de detalles de cada nivel
        
        # Procesar aproximación (coeficientes de baja frecuencia)
        approx = coeffs[0]
        self._extract_subband_features(approx, f"approx_L{self.config['levels']}", features)
        
        # Procesar detalles de cada nivel
        for level in range(1, self.config['levels'] + 1):
            # Los detalles están en orden inverso: coeffs[1] es el nivel más alto
            detail_idx = level
            cH, cV, cD = coeffs[detail_idx]  # Horizontal, Vertical, Diagonal
            
            current_level = self.config['levels'] - level + 1
            
            # Extractar características para cada dirección
            self._extract_subband_features(cH, f"detail_H_L{current_level}", features)
            self._extract_subband_features(cV, f"detail_V_L{current_level}", features)
            self._extract_subband_features(cD, f"detail_D_L{current_level}", features)
        
        # Calcular ratios entre subbands si está habilitado
        if self.config['compute_ratios']:
            self._compute_energy_ratios(coeffs, features)
        
        self.logger.debug(f"Wavelet extraído: {len(features)} características")
        return features
    
    def _extract_subband_features(self, subband: np.ndarray, name: str, features: Dict[str, float]):
        """
        Extrae características de una subband específica.
        
        Args:
            subband (np.ndarray): Coeficientes de la subband
            name (str): Nombre identificador de la subband
            features (Dict[str, float]): Diccionario donde agregar características
        """
        # Convertir a array plano para cálculos estadísticos
        coeffs_flat = subband.flatten()
        
        # Extraer según tipos de características configurados
        for feature_type in self.config['feature_types']:
            
            if feature_type == 'energy':
                # Energía = suma de cuadrados de coeficientes
                energy = float(np.sum(coeffs_flat ** 2))
                features[f'wavelet_energy_{name}'] = energy
            
            elif feature_type == 'entropy':
                # Entropía de Shannon de la distribución de coeficientes
                # Primero creamos histograma normalizado
                try:
                    hist, _ = np.histogram(coeffs_flat, bins=64, density=True)
                    hist_nonzero = hist[hist > 0]
                    if len(hist_nonzero) > 0:
                        entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))
                        features[f'wavelet_entropy_{name}'] = float(entropy)
                    else:
                        features[f'wavelet_entropy_{name}'] = 0.0
                except:
                    features[f'wavelet_entropy_{name}'] = 0.0
            
            elif feature_type == 'mean':
                features[f'wavelet_mean_{name}'] = float(np.mean(coeffs_flat))
            
            elif feature_type == 'std':
                features[f'wavelet_std_{name}'] = float(np.std(coeffs_flat))
            
            elif feature_type == 'var':
                features[f'wavelet_var_{name}'] = float(np.var(coeffs_flat))
        
        # Características adicionales útiles
        features[f'wavelet_max_{name}'] = float(np.max(np.abs(coeffs_flat)))
        features[f'wavelet_n_coeffs_{name}'] = float(len(coeffs_flat))
    
    def _compute_energy_ratios(self, coeffs, features: Dict[str, float]):
        """
        Calcula ratios de energía entre diferentes subbands.
        
        Args:
            coeffs: Coeficientes wavelet de pywt.wavedec2
            features (Dict[str, float]): Diccionario donde agregar ratios
        """
        # Calcular energía total de cada subband
        energies = {}
        
        # Energía de aproximación
        approx_energy = np.sum(coeffs[0] ** 2)
        energies['approx'] = approx_energy
        
        # Energía de detalles por nivel y dirección
        total_detail_energy = 0
        for level in range(1, self.config['levels'] + 1):
            detail_idx = level
            cH, cV, cD = coeffs[detail_idx]
            
            current_level = self.config['levels'] - level + 1
            
            energies[f'detail_H_L{current_level}'] = np.sum(cH ** 2)
            energies[f'detail_V_L{current_level}'] = np.sum(cV ** 2)
            energies[f'detail_D_L{current_level}'] = np.sum(cD ** 2)
            
            total_detail_energy += energies[f'detail_H_L{current_level}']
            total_detail_energy += energies[f'detail_V_L{current_level}']
            total_detail_energy += energies[f'detail_D_L{current_level}']
        
        # Energía total
        total_energy = approx_energy + total_detail_energy
        
        # Calcular ratios normalizados
        if total_energy > 0:
            # Ratio aproximación vs detalles
            features['wavelet_ratio_approx_total'] = float(approx_energy / total_energy)
            features['wavelet_ratio_detail_total'] = float(total_detail_energy / total_energy)
            
            # Ratios por dirección (promediando todos los niveles)
            h_energy = sum(energies[k] for k in energies.keys() if 'detail_H' in k)
            v_energy = sum(energies[k] for k in energies.keys() if 'detail_V' in k)
            d_energy = sum(energies[k] for k in energies.keys() if 'detail_D' in k)
            
            if total_detail_energy > 0:
                features['wavelet_ratio_horizontal'] = float(h_energy / total_detail_energy)
                features['wavelet_ratio_vertical'] = float(v_energy / total_detail_energy)
                features['wavelet_ratio_diagonal'] = float(d_energy / total_detail_energy)
            else:
                features['wavelet_ratio_horizontal'] = 0.0
                features['wavelet_ratio_vertical'] = 0.0
                features['wavelet_ratio_diagonal'] = 0.0
        else:
            # Si no hay energía, todos los ratios son 0
            features['wavelet_ratio_approx_total'] = 0.0
            features['wavelet_ratio_detail_total'] = 0.0
            features['wavelet_ratio_horizontal'] = 0.0
            features['wavelet_ratio_vertical'] = 0.0
            features['wavelet_ratio_diagonal'] = 0.0
    
    def get_feature_names(self) -> List[str]:
        """
        Retorna nombres de todas las características que extrae.
        
        Returns:
            List[str]: Lista ordenada de nombres de características
        """
        if self._feature_names_cache is None:
            names = []
            
            # Nombres para aproximación
            for feature_type in self.config['feature_types']:
                names.append(f'wavelet_{feature_type}_approx_L{self.config["levels"]}')
            names.extend([
                f'wavelet_max_approx_L{self.config["levels"]}',
                f'wavelet_n_coeffs_approx_L{self.config["levels"]}'
            ])
            
            # Nombres para detalles de cada nivel
            for level in range(1, self.config['levels'] + 1):
                for direction in ['H', 'V', 'D']:
                    suffix = f'{direction}_L{level}'
                    
                    for feature_type in self.config['feature_types']:
                        names.append(f'wavelet_{feature_type}_detail_{suffix}')
                    
                    names.extend([
                        f'wavelet_max_detail_{suffix}',
                        f'wavelet_n_coeffs_detail_{suffix}'
                    ])
            
            # Nombres para ratios
            if self.config['compute_ratios']:
                names.extend([
                    'wavelet_ratio_approx_total',
                    'wavelet_ratio_detail_total',
                    'wavelet_ratio_horizontal',
                    'wavelet_ratio_vertical',
                    'wavelet_ratio_diagonal'
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
            f"Transformada Wavelet usando '{self.config['wavelet']}' con "
            f"{self.config['levels']} niveles de descomposición. "
            f"Extrae {', '.join(self.config['feature_types'])} de cada subband. "
            f"{'Incluye ratios de energía. ' if self.config['compute_ratios'] else ''}"
            f"Analiza texturas a múltiples escalas y orientaciones. "
            f"Total de características: {len(self.get_feature_names())}"
        )