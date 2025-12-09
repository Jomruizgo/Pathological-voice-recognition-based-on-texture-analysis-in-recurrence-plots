"""
Descriptor GLCM (Gray Level Co-occurrence Matrix) para análisis de textura.

GLCM analiza las relaciones espaciales entre píxeles de diferentes niveles de gris.
Es especialmente útil para detectar patrones regulares y texturas direccionales
en los Recurrence Plots.

Este descriptor extrae varias propiedades estadísticas de la matriz de coocurrencia:
- Contrast: Mide la diferencia local en intensidad
- Dissimilarity: Mide la variación local  
- Homogeneity: Mide la uniformidad local
- Energy: Mide la uniformidad de textura
- Correlation: Mide la correlación lineal entre píxeles
- ASM (Angular Second Moment): Mide la uniformidad de energía
"""

import numpy as np
from typing import Dict, List
from skimage.feature import graycoprops, graycomatrix
import warnings

# Importar el sistema de registro
from . import register_descriptor
from .base import BaseDescriptor


@register_descriptor("glcm", enabled_by_default=True)
class GLCMDescriptor(BaseDescriptor):
    """
    Extractor de características GLCM (Gray Level Co-occurrence Matrix).
    
    La matriz de coocurrencia analiza cómo se distribuyen los pares de píxeles
    con diferentes niveles de gris a diferentes distancias y orientaciones.
    Esto es muy útil para Recurrence Plots porque captura:
    
    - Patrones repetitivos (homogeneity)
    - Variabilidad local (contrast, dissimilarity)  
    - Estructura direccional (diferentes ángulos)
    - Uniformidad de textura (energy, ASM)
    
    Example:
        >>> descriptor = GLCMDescriptor(
        ...     distances=[1, 2], 
        ...     angles=[0, np.pi/2],
        ...     levels=64
        ... )
        >>> features = descriptor.extract(recurrence_plot)
    """
    
    def __init__(self, 
                 distances: List[int] = [1], 
                 angles: List[float] = [0, np.pi/4, np.pi/2, 3*np.pi/4],
                 levels: int = 256,
                 symmetric: bool = True,
                 normed: bool = True,
                 properties: List[str] = ['contrast', 'dissimilarity', 'homogeneity', 
                                        'energy', 'correlation', 'ASM']):
        """
        Inicializa el descriptor GLCM.
        
        SISTEMA DE CONFIGURACIÓN HÍBRIDO:
        ═══════════════════════════════════════════════════════════════
        📋 Los defaults en esta función sirven como:
           • Documentación de valores recomendados
           • Fallback cuando se usa el descriptor directamente
           
        🔄 Cuando se usa a través del pipeline principal:
           • config.py::DEFAULT_DESCRIPTORS['glcm'] sobrescribe estos valores
           • Los defaults aquí son ignorados
           
        📖 Para ver la configuración REAL usada en el pipeline:
           • Ver: config.py línea ~54: DEFAULT_DESCRIPTORS['glcm']
        ═══════════════════════════════════════════════════════════════
        
        Args:
            distances (List[int]): Distancias entre píxeles a analizar.
                                  Default: [1, 2, 5] - Multi-escala espacial
                                  • 1: Relaciones píxeles adyacentes (micro-textura)
                                  • 2: Relaciones locales (salto de 1 píxel)
                                  • 5: Relaciones medio alcance (patrones amplios)
                                  
            angles (List[float]): Ángulos en radianes a analizar.
                                 Default: [0, π/4, π/2, 3π/4] - Cobertura direccional completa
                                 • 0: Horizontal → (relaciones temporales directas)
                                 • π/4: Diagonal ↗ (periodicidades ascendentes)
                                 • π/2: Vertical ↑ (correlaciones cruzadas)
                                 • 3π/4: Diagonal ↖ (periodicidades descendentes)
                                 
            levels (int): Número de niveles de gris para discretizar la imagen.
                         Default: 256 - Máxima resolución tonal
                         • 256: Máximo detalle pero sensible a ruido
                         • 64: Balance robustez/detalle (recomendado para RPs)
                         • 32: Mayor robustez, menor detalle
                         
            symmetric (bool): Si hacer la matriz simétrica. 
                             Default: True - Recomendado para estabilidad estadística
            
            normed (bool): Si normalizar la matriz.
                          Default: True - Recomendado para comparabilidad
            
            properties (List[str]): Propiedades estadísticas a extraer.
                                   Default: ['contrast', 'dissimilarity', 'homogeneity', 
                                            'energy', 'correlation', 'ASM']
                                   • contrast: Diferencias locales de intensidad
                                   • dissimilarity: Variación entre píxeles relacionados  
                                   • homogeneity: Uniformidad de la textura
                                   • energy: Concentración de pares de valores
                                   • correlation: Dependencia lineal entre píxeles
                                   • ASM: Angular Second Moment (uniformidad²)
        
        Note:
            ⚠️  IMPORTANTE: Cuando uses el pipeline (main.py, GUI), estos defaults
               son IGNORADOS y se usa la configuración de config.py en su lugar.
               
            ✅ Para uso directo: GLCMDescriptor() usa estos defaults
            🔄 Para pipeline: ModularPipeline() usa config.py
        """
        super().__init__(
            distances=distances,
            angles=angles, 
            levels=levels,
            symmetric=symmetric,
            normed=normed,
            properties=properties
        )
        
        # Validar parámetros
        self._validate_parameters()
        
        # Cache para nombres de características
        self._feature_names_cache = None
        
        self.logger.info(f"GLCM inicializado: {len(distances)} distancias, "
                        f"{len(angles)} ángulos, {levels} niveles")
    
    def _validate_parameters(self):
        """Valida que los parámetros sean correctos."""
        
        # Validar distancias
        if not self.config['distances'] or not all(isinstance(d, int) and d > 0 
                                                  for d in self.config['distances']):
            raise ValueError("distances debe ser una lista de enteros positivos")
        
        # Validar ángulos  
        if not self.config['angles'] or not all(isinstance(a, (int, float)) 
                                               for a in self.config['angles']):
            raise ValueError("angles debe ser una lista de números")
        
        # Validar niveles
        if not isinstance(self.config['levels'], int) or self.config['levels'] < 2:
            raise ValueError("levels debe ser un entero >= 2")
        
        # Validar propiedades
        valid_props = ['contrast', 'dissimilarity', 'homogeneity', 
                      'energy', 'correlation', 'ASM']
        invalid_props = set(self.config['properties']) - set(valid_props)
        if invalid_props:
            raise ValueError(f"Propiedades inválidas: {invalid_props}. "
                           f"Válidas: {valid_props}")
    
    @property
    def name(self) -> str:
        """Nombre del descriptor."""
        return "glcm"
    
    def extract(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extrae características GLCM de una imagen usando promediado.
        
        El proceso es:
        1. Reducir niveles de gris si es necesario (para eficiencia)
        2. Calcular matriz de coocurrencia para cada distancia/ángulo
        3. Extraer propiedades estadísticas de cada matriz
        4. PROMEDIAR todas las combinaciones distancia-ángulo por propiedad
        5. Retornar una característica robusta por propiedad
        
        Args:
            image (np.ndarray): Imagen en escala de grises [0-255]
            
        Returns:
            Dict[str, float]: Características promediadas con nombres como:
                             'glcm_contrast', 'glcm_energy', 'glcm_homogeneity', etc.
        """
        # Reducir niveles de gris si es necesario
        # Esto mejora la eficiencia sin perder información significativa
        if self.config['levels'] < 256:
            # Reescalar imagen al número de niveles deseado
            image_scaled = (image * (self.config['levels'] - 1) / 255).astype(np.uint8)
        else:
            image_scaled = image.astype(np.uint8)
        
        # Calcular matriz de coocurrencia
        # Esta es la operación más costosa computacionalmente
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suprimir warnings de skimage
            
            glcm = graycomatrix(
                image_scaled,
                distances=self.config['distances'],
                angles=self.config['angles'],
                levels=self.config['levels'],
                symmetric=self.config['symmetric'],
                normed=self.config['normed']
            )
        
        # Extraer características de cada propiedad
        features = {}
        
        for prop in self.config['properties']:
            # Calcular propiedad para todas las combinaciones distancia/ángulo
            prop_values = graycoprops(glcm, prop)
            
            # PROMEDIADO: Obtener valor promedio de todas las combinaciones
            # Esto reduce de N_distancias × N_ángulos características a 1 por propiedad
            # Ventajas: robustez, menos features, mejor para ML
            avg_value = np.mean(prop_values)
            
            feature_name = f"glcm_{prop}"
            features[feature_name] = float(avg_value)
        
        self.logger.debug(f"GLCM extraído: {len(features)} características")
        return features
    
    def get_feature_names(self) -> List[str]:
        """
        Retorna nombres de todas las características que extrae.
        
        Con implementación promediada, cada propiedad genera una sola característica.
        Los nombres siguen el patrón: glcm_{propiedad}
        Ejemplo: 'glcm_contrast', 'glcm_energy', 'glcm_homogeneity'
        
        Returns:
            List[str]: Lista ordenada de nombres de características
        """
        if self._feature_names_cache is None:
            names = []
            
            for prop in self.config['properties']:
                name = f"glcm_{prop}"
                names.append(name)
            
            # Ordenar para consistencia
            self._feature_names_cache = sorted(names)
        
        return self._feature_names_cache
    
    def get_description(self) -> str:
        """
        Retorna descripción detallada del descriptor.
        
        Returns:
            str: Descripción de qué mide este descriptor
        """
        return (
            f"GLCM (Gray Level Co-occurrence Matrix) con {len(self.config['distances'])} "
            f"distancias, {len(self.config['angles'])} ángulos y {self.config['levels']} "
            f"niveles de gris. Usa PROMEDIADO de todas las combinaciones distancia-ángulo "
            f"para mayor robustez y eficiencia en ML. Extrae {len(self.config['properties'])} "
            f"propiedades: {', '.join(self.config['properties'])}. "
            f"Total de características: {len(self.get_feature_names())}"
        )