"""
Cargador de imágenes con validación y preprocesamiento.

Este módulo se encarga de cargar los Recurrence Plots desde archivos,
aplicar validaciones necesarias y preprocesamiento básico para
asegurar que las imágenes estén en el formato correcto para
la extracción de características.
"""

import os
import numpy as np
from typing import Optional, Tuple
import logging
from PIL import Image
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    cv2 = None
from pathlib import Path


class ImageLoader:
    """
    Cargador robusto de imágenes con validación y preprocesamiento.
    
    Esta clase maneja la carga de Recurrence Plots desde archivos,
    aplicando las validaciones y transformaciones necesarias para
    asegurar compatibilidad con los descriptores de textura.
    
    Funcionalidades:
    - Carga múltiples formatos (PNG, JPG, etc.)
    - Conversión automática a escala de grises
    - Validación de dimensiones y contenido
    - Normalización de valores de píxel
    - Manejo robusto de errores
    """
    
    def __init__(self, 
                 target_dtype: str = 'uint8',
                 normalize_range: Tuple[int, int] = (0, 255),
                 min_size: Tuple[int, int] = (32, 32),
                 max_size: Optional[Tuple[int, int]] = (2048, 2048)):
        """
        Inicializa el cargador de imágenes.
        
        Args:
            target_dtype (str): Tipo de datos objetivo ('uint8', 'float32', etc.)
                               'uint8' es recomendado para la mayoría de descriptores.
                               
            normalize_range (Tuple[int, int]): Rango de normalización de píxeles.
                                              (0, 255) para uint8, (0, 1) para float.
                                              
            min_size (Tuple[int, int]): Tamaño mínimo aceptable (ancho, alto).
                                       Imágenes menores se rechazarán.
                                       
            max_size (Optional[Tuple[int, int]]): Tamaño máximo aceptable (ancho, alto).
                                             Imágenes mayores se redimensionarán.
                                             Si es None, no hay límite de tamaño.
        """
        self.target_dtype = target_dtype
        self.normalize_range = normalize_range
        self.min_size = min_size
        self.max_size = max_size
        
        self.logger = logging.getLogger(__name__)
        
        # Estadísticas de carga
        self.stats = {
            'total_loaded': 0,
            'successful_loads': 0,
            'failed_loads': 0,
            'resized_images': 0,
            'converted_images': 0
        }
        
        max_size_str = "sin límite" if max_size is None else str(max_size)
        self.logger.info(f"ImageLoader inicializado: dtype={target_dtype}, "
                        f"rango={normalize_range}, tamaño={min_size}-{max_size_str}")
    
    def load_image(self, file_path: str) -> Optional[np.ndarray]:
        """
        Carga una imagen desde archivo con validación completa.
        
        Args:
            file_path (str): Ruta al archivo de imagen
            
        Returns:
            Optional[np.ndarray]: Imagen cargada como array 2D en escala de grises,
                                 o None si no se pudo cargar
        """
        self.stats['total_loaded'] += 1
        
        try:
            # Validar que el archivo existe
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
            
            # Validar extensión de archivo
            valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in valid_extensions:
                raise ValueError(f"Formato no soportado: {file_ext}")
            
            # Intentar cargar con PIL primero (más robusto)
            image = self._load_with_pil(file_path)
            
            # Si PIL falla y OpenCV está disponible, intentar con OpenCV
            if image is None and HAS_OPENCV:
                image = self._load_with_opencv(file_path)
            
            # Si ambos fallan, reportar error
            if image is None:
                if HAS_OPENCV:
                    raise RuntimeError("No se pudo cargar con PIL ni OpenCV")
                else:
                    raise RuntimeError("No se pudo cargar con PIL (OpenCV no disponible)")
            
            # Validar y procesar la imagen cargada
            processed_image = self._process_loaded_image(image, file_path)
            
            if processed_image is not None:
                self.stats['successful_loads'] += 1
                self.logger.debug(f"Imagen cargada exitosamente: {file_path} "
                                f"-> shape={processed_image.shape}")
            
            return processed_image
            
        except Exception as e:
            self.stats['failed_loads'] += 1
            self.logger.error(f"Error cargando {file_path}: {str(e)}")
            return None
    
    def _load_with_pil(self, file_path: str) -> Optional[np.ndarray]:
        """
        Intenta cargar imagen usando PIL/Pillow.
        
        Args:
            file_path (str): Ruta al archivo
            
        Returns:
            Optional[np.ndarray]: Array de imagen o None si falla
        """
        try:
            with Image.open(file_path) as pil_image:
                # Convertir a escala de grises si es necesario
                if pil_image.mode != 'L':
                    pil_image = pil_image.convert('L')
                    self.stats['converted_images'] += 1
                
                # Convertir a numpy array
                image_array = np.array(pil_image)
                
                return image_array
                
        except Exception as e:
            self.logger.debug(f"PIL falló para {file_path}: {str(e)}")
            return None
    
    def _load_with_opencv(self, file_path: str) -> Optional[np.ndarray]:
        """
        Intenta cargar imagen usando OpenCV.
        
        Args:
            file_path (str): Ruta al archivo
            
        Returns:
            Optional[np.ndarray]: Array de imagen o None si falla
        """
        if not HAS_OPENCV:
            return None
            
        try:
            # Cargar en escala de grises directamente
            image_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            
            if image_array is None:
                raise RuntimeError("OpenCV retornó None")
            
            return image_array
            
        except Exception as e:
            self.logger.debug(f"OpenCV falló para {file_path}: {str(e)}")
            return None
    
    def _process_loaded_image(self, image: np.ndarray, file_path: str) -> Optional[np.ndarray]:
        """
        Procesa y valida una imagen recién cargada.
        
        Args:
            image (np.ndarray): Imagen cruda cargada
            file_path (str): Ruta original para logging
            
        Returns:
            Optional[np.ndarray]: Imagen procesada o None si no es válida
        """
        try:
            # Validar que sea un array numpy
            if not isinstance(image, np.ndarray):
                raise TypeError(f"La imagen no es un numpy array: {type(image)}")
            
            # Asegurar que sea 2D (escala de grises)
            if image.ndim == 3:
                if image.shape[2] == 1:
                    # Imagen con 1 canal, remover dimensión extra
                    image = image.squeeze(axis=2)
                elif image.shape[2] == 3:
                    # Imagen RGB, convertir a escala de grises
                    image = np.mean(image, axis=2).astype(image.dtype)
                    self.stats['converted_images'] += 1
                else:
                    raise ValueError(f"Imagen con {image.shape[2]} canales no soportada")
            
            elif image.ndim != 2:
                raise ValueError(f"Imagen debe ser 2D, recibida: {image.ndim}D")
            
            # Validar tamaño
            height, width = image.shape
            
            # Verificar tamaño mínimo
            if height < self.min_size[1] or width < self.min_size[0]:
                raise ValueError(f"Imagen muy pequeña: {width}x{height}, "
                               f"mínimo: {self.min_size[0]}x{self.min_size[1]}")
            
            # Redimensionar si es muy grande (solo si hay límite configurado)
            if (self.max_size is not None and 
                (height > self.max_size[1] or width > self.max_size[0])):
                self.logger.warning(f"Redimensionando imagen grande: {width}x{height} "
                                  f"-> máximo: {self.max_size[0]}x{self.max_size[1]}")
                image = self._resize_image(image)
                self.stats['resized_images'] += 1
            
            # Validar contenido
            if image.size == 0:
                raise ValueError("Imagen vacía")
            
            # Verificar que no todos los píxeles sean iguales
            if np.all(image == image.flat[0]):
                self.logger.warning(f"Imagen uniforme detectada: {file_path}")
            
            # Normalizar valores si es necesario
            image = self._normalize_values(image)
            
            # Convertir al tipo de dato objetivo
            image = self._convert_dtype(image)
            
            return image
            
        except Exception as e:
            self.logger.error(f"Error procesando imagen {file_path}: {str(e)}")
            return None
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Redimensiona una imagen manteniendo la relación de aspecto.
        
        Args:
            image (np.ndarray): Imagen original
            
        Returns:
            np.ndarray: Imagen redimensionada
            
        Note:
            Este método solo se llama cuando max_size no es None.
        """
        height, width = image.shape
        
        if self.max_size is None:
            return image  # No redimensionar si no hay límite
            
        max_width, max_height = self.max_size
        
        # Calcular escala manteniendo aspecto
        scale_w = max_width / width
        scale_h = max_height / height
        scale = min(scale_w, scale_h)
        
        # Nuevas dimensiones
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Redimensionar usando OpenCV si está disponible, sino PIL
        if HAS_OPENCV:
            resized = cv2.resize(image, (new_width, new_height), 
                               interpolation=cv2.INTER_AREA)
        else:
            # Usar PIL como alternativa
            from PIL import Image as PILImage
            pil_image = PILImage.fromarray(image)
            pil_resized = pil_image.resize((new_width, new_height), PILImage.LANCZOS)
            resized = np.array(pil_resized)
        
        return resized
    
    def _normalize_values(self, image: np.ndarray) -> np.ndarray:
        """
        Normaliza los valores de píxel al rango objetivo.
        
        Args:
            image (np.ndarray): Imagen original
            
        Returns:
            np.ndarray: Imagen normalizada
        """
        # Obtener rango actual
        min_val = float(np.min(image))
        max_val = float(np.max(image))
        
        # Si ya está en el rango correcto, no hacer nada
        target_min, target_max = self.normalize_range
        if min_val >= target_min and max_val <= target_max:
            return image
        
        # Normalizar al rango [0, 1] primero
        if max_val > min_val:
            normalized = (image.astype(np.float64) - min_val) / (max_val - min_val)
        else:
            # Imagen uniforme
            normalized = np.zeros_like(image, dtype=np.float64)
        
        # Escalar al rango objetivo
        scaled = normalized * (target_max - target_min) + target_min
        
        return scaled
    
    def _convert_dtype(self, image: np.ndarray) -> np.ndarray:
        """
        Convierte la imagen al tipo de dato objetivo.
        
        Args:
            image (np.ndarray): Imagen a convertir
            
        Returns:
            np.ndarray: Imagen con el tipo de dato correcto
        """
        if self.target_dtype == 'uint8':
            return np.clip(image, 0, 255).astype(np.uint8)
        elif self.target_dtype == 'float32':
            return image.astype(np.float32)
        elif self.target_dtype == 'float64':
            return image.astype(np.float64)
        else:
            # Para otros tipos, intentar conversión directa
            return image.astype(self.target_dtype)
    
    def get_stats(self) -> dict:
        """
        Obtiene estadísticas de carga de imágenes.
        
        Returns:
            dict: Estadísticas de rendimiento
        """
        stats = self.stats.copy()
        
        # Calcular porcentajes
        if stats['total_loaded'] > 0:
            stats['success_rate'] = stats['successful_loads'] / stats['total_loaded']
            stats['failure_rate'] = stats['failed_loads'] / stats['total_loaded']
            stats['conversion_rate'] = stats['converted_images'] / stats['total_loaded']
            stats['resize_rate'] = stats['resized_images'] / stats['total_loaded']
        else:
            stats['success_rate'] = 0.0
            stats['failure_rate'] = 0.0
            stats['conversion_rate'] = 0.0
            stats['resize_rate'] = 0.0
        
        return stats
    
    def reset_stats(self):
        """Reinicia las estadísticas de carga."""
        self.stats = {
            'total_loaded': 0,
            'successful_loads': 0,
            'failed_loads': 0,
            'resized_images': 0,
            'converted_images': 0
        }
    
    def batch_load(self, file_paths: list, max_failures: int = 10) -> list:
        """
        Carga múltiples imágenes en lote.
        
        Args:
            file_paths (list): Lista de rutas de archivos
            max_failures (int): Máximo número de fallos antes de abortar
            
        Returns:
            list: Lista de arrays de imágenes (None para fallos)
        """
        images = []
        failure_count = 0
        
        for file_path in file_paths:
            image = self.load_image(file_path)
            images.append(image)
            
            if image is None:
                failure_count += 1
                if failure_count >= max_failures:
                    self.logger.error(f"Abortando carga en lote: {failure_count} fallos")
                    break
        
        return images