"""
Clase base para todos los descriptores de textura.

Este módulo define la interfaz que todos los descriptores deben implementar.
También incluye funcionalidad común como validación, logging y serialización.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
import logging
from datetime import datetime


class BaseDescriptor(ABC):
    """
    Clase base abstracta para todos los descriptores de textura.
    
    Esta clase define la interfaz que todos los descriptores deben implementar
    y proporciona funcionalidad común como logging y validación.
    
    Attributes:
        config (dict): Configuración del descriptor
        logger (logging.Logger): Logger para este descriptor
        _feature_names (List[str]): Cache de nombres de características
    """
    
    def __init__(self, **kwargs):
        """
        Inicializa el descriptor base.
        
        Args:
            **kwargs: Parámetros de configuración específicos del descriptor
        """
        self.config = kwargs
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._feature_names = None  # Cache para nombres de características
        self._last_extraction_time = None
        
    @abstractmethod
    def extract(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extrae características de textura de una imagen.
        
        Este es el método principal que cada descriptor debe implementar.
        Debe procesar la imagen y retornar un diccionario con las características.
        
        Args:
            image (np.ndarray): Imagen de entrada (típicamente un Recurrence Plot)
                               Formato esperado: 2D array en escala de grises [0-255]
        
        Returns:
            Dict[str, float]: Diccionario con pares nombre_característica: valor
                             Ejemplo: {'glcm_contrast_d1_a0': 0.123, ...}
        
        Raises:
            ValueError: Si la imagen tiene formato incorrecto
        """
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Retorna la lista de nombres de todas las características que extrae.
        
        Este método es importante para:
        - Crear los headers del CSV de salida
        - Validar que todas las imágenes generan las mismas características
        - Documentación automática
        
        Returns:
            List[str]: Lista ordenada de nombres de características
                      Ejemplo: ['glcm_contrast_d1_a0', 'glcm_energy_d1_a0', ...]
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Nombre único del descriptor.
        
        Este nombre se usa para:
        - Identificar el descriptor en logs
        - Prefijo en nombres de características
        - Referencia en configuración
        
        Returns:
            str: Nombre del descriptor (ej: "glcm", "lbp", etc.)
        """
        pass
    
    @property
    def description(self) -> str:
        """
        Descripción detallada del descriptor.
        
        Returns:
            str: Descripción legible del descriptor y qué mide
        """
        return self.__class__.__doc__ or "Sin descripción disponible"
    
    def validate_image(self, image: np.ndarray) -> None:
        """
        Valida que la imagen tenga el formato correcto.
        
        Args:
            image (np.ndarray): Imagen a validar
            
        Raises:
            ValueError: Si la imagen no cumple los requisitos
        """
        if image is None:
            raise ValueError("La imagen no puede ser None")
            
        if not isinstance(image, np.ndarray):
            raise ValueError(f"La imagen debe ser numpy.ndarray, recibido: {type(image)}")
            
        if image.ndim != 2:
            raise ValueError(f"La imagen debe ser 2D (escala de grises), recibido: {image.ndim}D")
            
        if image.size == 0:
            raise ValueError("La imagen está vacía")
            
        if not np.isfinite(image).all():
            raise ValueError("La imagen contiene valores NaN o infinitos")
            
        self.logger.debug(f"Imagen validada: shape={image.shape}, dtype={image.dtype}, "
                         f"rango=[{image.min():.2f}, {image.max():.2f}]")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocesa la imagen antes de extraer características.
        
        Este método puede ser sobrescrito por descriptores específicos
        si necesitan preprocesamiento especial.
        
        Args:
            image (np.ndarray): Imagen original
            
        Returns:
            np.ndarray: Imagen preprocesada
        """
        # Asegurar que la imagen esté en el rango correcto
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                # Imagen normalizada [0, 1] -> [0, 255]
                image = (image * 255).astype(np.uint8)
            else:
                # Imagen en otro rango -> normalizar a [0, 255]
                image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                
        return image
    
    def extract_with_validation(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extrae características con validación y manejo de errores.
        
        Este método envuelve extract() con validación, preprocesamiento
        y manejo de errores robusto.
        
        Args:
            image (np.ndarray): Imagen de entrada
            
        Returns:
            Dict[str, float]: Características extraídas
            
        Raises:
            Exception: Propaga excepciones de extract() con contexto adicional
        """
        try:
            # Validar imagen
            self.validate_image(image)
            
            # Preprocesar
            processed_image = self.preprocess_image(image)
            
            # Medir tiempo de extracción
            start_time = datetime.now()
            
            # Extraer características
            self.logger.info(f"Extrayendo características con {self.name}...")
            features = self.extract(processed_image)
            
            # Guardar tiempo de extracción
            self._last_extraction_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Extracción completada en {self._last_extraction_time:.2f}s")
            
            # Validar resultado
            self._validate_features(features)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error en {self.name}: {str(e)}")
            raise
    
    def _validate_features(self, features: Dict[str, float]) -> None:
        """
        Valida que las características extraídas sean correctas.
        
        Args:
            features (Dict[str, float]): Características a validar
            
        Raises:
            ValueError: Si las características no son válidas
        """
        if not features:
            raise ValueError("No se extrajeron características")
            
        expected_names = set(self.get_feature_names())
        actual_names = set(features.keys())
        
        if expected_names != actual_names:
            missing = expected_names - actual_names
            extra = actual_names - expected_names
            msg = f"Características no coinciden. "
            if missing:
                msg += f"Faltan: {missing}. "
            if extra:
                msg += f"Extras: {extra}."
            raise ValueError(msg)
            
        # Verificar que todos los valores sean numéricos y finitos
        for name, value in features.items():
            if not isinstance(value, (int, float, np.number)):
                raise ValueError(f"Característica '{name}' no es numérica: {type(value)}")
            if not np.isfinite(value):
                raise ValueError(f"Característica '{name}' no es finita: {value}")
    
    def get_config(self) -> Dict[str, Any]:
        """
        Retorna la configuración actual del descriptor.
        
        Returns:
            Dict[str, Any]: Configuración serializable
        """
        return {
            'name': self.name,
            'class': self.__class__.__name__,
            'config': self.config,
            'description': self.description
        }
    
    def to_json(self) -> str:
        """
        Serializa el descriptor a JSON.
        
        Returns:
            str: Representación JSON del descriptor
        """
        return json.dumps(self.get_config(), indent=2)
    
    def __repr__(self) -> str:
        """Representación legible del descriptor."""
        return f"{self.__class__.__name__}(name='{self.name}', config={self.config})"