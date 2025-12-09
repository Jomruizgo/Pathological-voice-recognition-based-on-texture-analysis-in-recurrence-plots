"""
Descriptor Gabor para análisis de textura basado en frecuencia y orientación.

Los filtros de Gabor son especialmente útiles para analizar texturas porque
combinan información espacial y frecuencial. Son similares a cómo funciona
el sistema visual humano.

Para Recurrence Plots, los filtros de Gabor son valiosos porque:
- Detectan patrones direccionales y periódicos
- Analizan diferentes escalas de recurrencia
- Capturan estructuras orientadas en el RP
- Identifican frecuencias dominantes en la textura

Un filtro de Gabor es esencialmente una función gaussiana modulada por
una onda sinusoidal, lo que permite detectar patrones oscilatorios
en direcciones específicas.
"""

import numpy as np
from typing import Dict, List
from skimage.filters import gabor
import warnings

from . import register_descriptor
from .base import BaseDescriptor


@register_descriptor("gabor", enabled_by_default=True)
class GaborDescriptor(BaseDescriptor):
    """
    Extractor de características usando filtros de Gabor.
    
    Los filtros de Gabor analizan la imagen en el dominio espacial-frecuencial,
    detectando patrones oscilatorios en diferentes orientaciones y escalas.
    
    Son especialmente efectivos para:
    - Detectar texturas direccionales en Recurrence Plots
    - Identificar periodicidades y estructuras repetitivas
    - Analizar diferentes escalas de recurrencia
    - Capturar información de frecuencia espacial
    
    Example:
        >>> descriptor = GaborDescriptor(
        ...     frequencies=[0.1, 0.2, 0.4], 
        ...     orientations=[0, 45, 90, 135],
        ...     sigma=1.0
        ... )
        >>> features = descriptor.extract(recurrence_plot)
    """
    
    def __init__(self, 
                 frequencies: List[float] = [0.1, 0.2, 0.4],
                 orientations: List[float] = [0, 45, 90, 135],
                 sigma: float = 1.0,
                 compute_magnitude: bool = True,
                 compute_phase: bool = False,
                 compute_energy: bool = True):
        """
        Inicializa el descriptor Gabor.
        
        Args:
            frequencies (List[float]): Frecuencias de los filtros de Gabor.
                                      Valores típicos: 0.1-0.5
                                      - Frecuencias bajas: patrones grandes
                                      - Frecuencias altas: detalles finos
                                      
            orientations (List[float]): Orientaciones en grados.
                                       Ejemplo: [0, 45, 90, 135] cubre las
                                       principales direcciones.
                                       
            sigma (float): Desviación estándar del kernel gaussiano.
                          Controla el tamaño del filtro:
                          - Sigma pequeño: localización espacial precisa
                          - Sigma grande: mejor resolución frecuencial
                          
            compute_magnitude (bool): Si calcular la magnitud de la respuesta.
                                     Recomendado: True (es la característica principal)
                                     
            compute_phase (bool): Si calcular la fase de la respuesta.
                                 Útil para análisis avanzados pero costoso.
                                 
            compute_energy (bool): Si calcular la energía total de cada filtro.
                                  Proporciona medida global de activación.
        """
        super().__init__(
            frequencies=frequencies,
            orientations=orientations,
            sigma=sigma,
            compute_magnitude=compute_magnitude,
            compute_phase=compute_phase,
            compute_energy=compute_energy
        )
        
        # Validar parámetros
        self._validate_parameters()
        
        # Cache para nombres de características
        self._feature_names_cache = None
        
        self.logger.info(f"Gabor inicializado: {len(frequencies)} frecuencias, "
                        f"{len(orientations)} orientaciones, sigma={sigma}")
    
    def _validate_parameters(self):
        """Valida que los parámetros sean correctos."""
        
        # Validar frecuencias
        frequencies = self.config['frequencies']
        if not frequencies or not all(isinstance(f, (int, float)) and f > 0 
                                     for f in frequencies):
            raise ValueError("frequencies debe ser una lista de números positivos")
        
        if any(f > 1.0 for f in frequencies):
            self.logger.warning("Frecuencias > 1.0 pueden causar artefactos")
        
        # Validar orientaciones
        orientations = self.config['orientations']
        if not orientations or not all(isinstance(o, (int, float)) 
                                      for o in orientations):
            raise ValueError("orientations debe ser una lista de números")
        
        # Validar sigma
        if not isinstance(self.config['sigma'], (int, float)) or self.config['sigma'] <= 0:
            raise ValueError("sigma debe ser un número positivo")
        
        # Validar que al menos una opción de cómputo esté activa
        if not any([self.config['compute_magnitude'], 
                   self.config['compute_phase'], 
                   self.config['compute_energy']]):
            raise ValueError("Al menos una opción de cómputo debe estar activa")
    
    @property
    def name(self) -> str:
        """Nombre del descriptor."""
        return "gabor"
    
    def extract(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extrae características Gabor de una imagen.
        
        El proceso es:
        1. Para cada combinación (frecuencia, orientación), aplicar filtro Gabor
        2. Calcular respuesta compleja (parte real e imaginaria)
        3. Extraer magnitud, fase y/o energía según configuración
        4. Calcular estadísticas de las respuestas filtradas
        
        Args:
            image (np.ndarray): Imagen en escala de grises [0-255]
            
        Returns:
            Dict[str, float]: Características extraídas con nombres como:
                             'gabor_mag_mean_f0.1_o0', 'gabor_energy_f0.2_o90', etc.
        """
        features = {}
        
        # Convertir imagen a float para evitar problemas numéricos
        image_float = image.astype(np.float64) / 255.0  # Normalizar a [0,1]
        
        # Procesar cada combinación frecuencia-orientación
        for frequency in self.config['frequencies']:
            for orientation_deg in self.config['orientations']:
                
                # Convertir orientación a radianes
                orientation_rad = np.deg2rad(orientation_deg)
                
                # Aplicar filtro de Gabor
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    # gabor() retorna (respuesta_real, respuesta_imaginaria)
                    real_response, imag_response = gabor(
                        image_float,
                        frequency=frequency,
                        theta=orientation_rad,
                        sigma_x=self.config['sigma'],
                        sigma_y=self.config['sigma']
                    )
                
                # Crear sufijo para nombres de características
                suffix = f"_f{frequency}_o{int(orientation_deg)}"
                
                # 1. MAGNITUD (amplitude)
                if self.config['compute_magnitude']:
                    magnitude = np.sqrt(real_response**2 + imag_response**2)
                    
                    # Estadísticas de la magnitud
                    features[f'gabor_mag_mean{suffix}'] = float(np.mean(magnitude))
                    features[f'gabor_mag_std{suffix}'] = float(np.std(magnitude))
                    features[f'gabor_mag_max{suffix}'] = float(np.max(magnitude))
                    features[f'gabor_mag_min{suffix}'] = float(np.min(magnitude))
                    
                    # Percentiles de magnitud
                    percentiles = np.percentile(magnitude, [25, 50, 75])
                    features[f'gabor_mag_q25{suffix}'] = float(percentiles[0])
                    features[f'gabor_mag_median{suffix}'] = float(percentiles[1])
                    features[f'gabor_mag_q75{suffix}'] = float(percentiles[2])
                
                # 2. FASE
                if self.config['compute_phase']:
                    phase = np.arctan2(imag_response, real_response)
                    
                    # Estadísticas de la fase
                    features[f'gabor_phase_mean{suffix}'] = float(np.mean(phase))
                    features[f'gabor_phase_std{suffix}'] = float(np.std(phase))
                    
                    # Coherencia de fase (qué tan consistente es la fase)
                    # Calculamos la magnitud del promedio de exponenciales complejas
                    complex_responses = real_response + 1j * imag_response
                    phase_coherence = np.abs(np.mean(np.exp(1j * phase)))
                    features[f'gabor_phase_coherence{suffix}'] = float(phase_coherence)
                
                # 3. ENERGÍA
                if self.config['compute_energy']:
                    # Energía total de la respuesta
                    energy_real = np.sum(real_response**2)
                    energy_imag = np.sum(imag_response**2)
                    energy_total = energy_real + energy_imag
                    
                    features[f'gabor_energy_real{suffix}'] = float(energy_real)
                    features[f'gabor_energy_imag{suffix}'] = float(energy_imag)
                    features[f'gabor_energy_total{suffix}'] = float(energy_total)
                    
                    # Ratio de energías (indica dominancia de parte real vs imaginaria)
                    if energy_total > 0:
                        energy_ratio = energy_real / energy_total
                        features[f'gabor_energy_ratio{suffix}'] = float(energy_ratio)
                    else:
                        features[f'gabor_energy_ratio{suffix}'] = 0.5
                
                # 4. CARACTERÍSTICAS ADICIONALES
                
                # Respuesta promedio (parte real e imaginaria por separado)
                features[f'gabor_real_mean{suffix}'] = float(np.mean(real_response))
                features[f'gabor_imag_mean{suffix}'] = float(np.mean(imag_response))
                
                # Variabilidad de la respuesta
                features[f'gabor_real_std{suffix}'] = float(np.std(real_response))
                features[f'gabor_imag_std{suffix}'] = float(np.std(imag_response))
        
        self.logger.debug(f"Gabor extraído: {len(features)} características")
        return features
    
    def get_feature_names(self) -> List[str]:
        """
        Retorna nombres de todas las características que extrae.
        
        Los nombres siguen el patrón: gabor_{tipo}_{estadística}_f{freq}_o{orientación}
        
        Returns:
            List[str]: Lista ordenada de nombres de características
        """
        if self._feature_names_cache is None:
            names = []
            
            for frequency in self.config['frequencies']:
                for orientation_deg in self.config['orientations']:
                    suffix = f"_f{frequency}_o{int(orientation_deg)}"
                    
                    # Nombres para magnitud
                    if self.config['compute_magnitude']:
                        names.extend([
                            f'gabor_mag_mean{suffix}',
                            f'gabor_mag_std{suffix}',
                            f'gabor_mag_max{suffix}',
                            f'gabor_mag_min{suffix}',
                            f'gabor_mag_q25{suffix}',
                            f'gabor_mag_median{suffix}',
                            f'gabor_mag_q75{suffix}'
                        ])
                    
                    # Nombres para fase
                    if self.config['compute_phase']:
                        names.extend([
                            f'gabor_phase_mean{suffix}',
                            f'gabor_phase_std{suffix}',
                            f'gabor_phase_coherence{suffix}'
                        ])
                    
                    # Nombres para energía
                    if self.config['compute_energy']:
                        names.extend([
                            f'gabor_energy_real{suffix}',
                            f'gabor_energy_imag{suffix}',
                            f'gabor_energy_total{suffix}',
                            f'gabor_energy_ratio{suffix}'
                        ])
                    
                    # Nombres para respuestas básicas
                    names.extend([
                        f'gabor_real_mean{suffix}',
                        f'gabor_imag_mean{suffix}',
                        f'gabor_real_std{suffix}',
                        f'gabor_imag_std{suffix}'
                    ])
            
            self._feature_names_cache = sorted(names)
        
        return self._feature_names_cache
    
    def get_description(self) -> str:
        """
        Retorna descripción detallada del descriptor.
        
        Returns:
            str: Descripción de qué mide este descriptor
        """
        components = []
        if self.config['compute_magnitude']:
            components.append("magnitud")
        if self.config['compute_phase']:
            components.append("fase")
        if self.config['compute_energy']:
            components.append("energía")
        
        return (
            f"Filtros de Gabor con {len(self.config['frequencies'])} frecuencias "
            f"({self.config['frequencies']}) y {len(self.config['orientations'])} "
            f"orientaciones ({self.config['orientations']}°). "
            f"Sigma={self.config['sigma']}. Extrae: {', '.join(components)}. "
            f"Detecta patrones direccionales y frecuencias espaciales. "
            f"Total de características: {len(self.get_feature_names())}"
        )