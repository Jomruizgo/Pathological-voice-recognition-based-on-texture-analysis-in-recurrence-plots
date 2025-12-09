"""
Tracker de progreso para monitoreo en tiempo real.

Este módulo proporciona seguimiento detallado del progreso del procesamiento,
incluyendo métricas de rendimiento, estimaciones de tiempo y historial.
Es especialmente útil para interfaces gráficas y logging.
"""

import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import threading
from collections import deque


class ProgressTracker:
    """
    Rastrea el progreso del procesamiento con métricas detalladas.
    
    Proporciona:
    - Progreso actual (imágenes procesadas / total)
    - Velocidad de procesamiento (imágenes/segundo)
    - Tiempo estimado restante (ETA)
    - Historial de progreso para gráficos
    - Uso de memoria (si está disponible)
    """
    
    def __init__(self, history_size: int = 100):
        """
        Inicializa el tracker de progreso.
        
        Args:
            history_size (int): Número máximo de puntos en el historial.
                               Más puntos = gráficos más detallados pero más memoria.
        """
        self.history_size = history_size
        
        # Estado del progreso
        self.total_images = 0
        self.total_descriptors = 0
        self.processed_images = 0
        self.start_time = None
        self.last_update_time = None
        
        # Historial para cálculos de velocidad y gráficos
        self.history = deque(maxlen=history_size)
        
        # Threading lock para acceso concurrente
        self._lock = threading.Lock()
        
        # Métricas calculadas
        self._current_speed = 0.0
        self._eta_seconds = 0
        self._memory_usage = 0.0
    
    def initialize(self, total_images: int, total_descriptors: int):
        """
        Inicializa el tracker con los totales a procesar.
        
        Args:
            total_images (int): Total de imágenes a procesar
            total_descriptors (int): Total de descriptores por imagen
        """
        with self._lock:
            self.total_images = total_images
            self.total_descriptors = total_descriptors
            self.processed_images = 0
            self.start_time = datetime.now()
            self.last_update_time = self.start_time
            
            # Limpiar historial
            self.history.clear()
            
            # Agregar punto inicial
            self.history.append({
                'timestamp': self.start_time,
                'processed': 0,
                'speed': 0.0,
                'memory_mb': self._get_memory_usage()
            })
    
    def update(self, processed_images: int, total_images: Optional[int] = None) -> Dict:
        """
        Actualiza el progreso y calcula métricas.
        
        Args:
            processed_images (int): Número de imágenes procesadas hasta ahora
            total_images (Optional[int]): Total actualizado si cambió
            
        Returns:
            Dict: Información completa del progreso
        """
        current_time = datetime.now()
        
        with self._lock:
            # Actualizar totales si se proporcionaron
            if total_images is not None:
                self.total_images = total_images
            
            # Actualizar progreso
            self.processed_images = processed_images
            
            # Calcular métricas solo si ha pasado tiempo
            if self.last_update_time and self.start_time:
                elapsed_total = (current_time - self.start_time).total_seconds()
                elapsed_since_last = (current_time - self.last_update_time).total_seconds()
                
                # Calcular velocidad promedio
                if elapsed_total > 0:
                    avg_speed = self.processed_images / elapsed_total
                else:
                    avg_speed = 0.0
                
                # Calcular velocidad instantánea (usando ventana móvil)
                instant_speed = self._calculate_instant_speed()
                
                # Usar velocidad más estable para ETA
                self._current_speed = instant_speed if instant_speed > 0 else avg_speed
                
                # Calcular ETA
                if self._current_speed > 0 and self.processed_images < self.total_images:
                    remaining_images = self.total_images - self.processed_images
                    self._eta_seconds = remaining_images / self._current_speed
                else:
                    self._eta_seconds = 0
                
                # Obtener uso de memoria
                self._memory_usage = self._get_memory_usage()
                
                # Agregar punto al historial
                self.history.append({
                    'timestamp': current_time,
                    'processed': self.processed_images,
                    'speed': self._current_speed,
                    'memory_mb': self._memory_usage
                })
            
            self.last_update_time = current_time
            
            # Crear información de progreso
            progress_info = self._create_progress_info()
            
        return progress_info
    
    def _calculate_instant_speed(self) -> float:
        """
        Calcula velocidad instantánea usando ventana móvil.
        
        Returns:
            float: Velocidad en imágenes por segundo
        """
        if len(self.history) < 2:
            return 0.0
        
        # Usar últimos puntos para velocidad instantánea
        window_size = min(10, len(self.history))  # Ventana de 10 puntos máximo
        recent_points = list(self.history)[-window_size:]
        
        if len(recent_points) < 2:
            return 0.0
        
        # Calcular diferencias
        time_diff = (recent_points[-1]['timestamp'] - recent_points[0]['timestamp']).total_seconds()
        processed_diff = recent_points[-1]['processed'] - recent_points[0]['processed']
        
        if time_diff > 0:
            return processed_diff / time_diff
        else:
            return 0.0
    
    def _get_memory_usage(self) -> float:
        """
        Obtiene el uso actual de memoria.
        
        Returns:
            float: Uso de memoria en MB
        """
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convertir a MB
        except ImportError:
            # psutil no disponible
            return 0.0
        except Exception:
            # Error obteniendo memoria
            return 0.0
    
    def _create_progress_info(self) -> Dict:
        """
        Crea el diccionario completo de información de progreso.
        
        Returns:
            Dict: Información completa del progreso
        """
        # Calcular porcentaje
        if self.total_images > 0:
            progress_percent = (self.processed_images / self.total_images) * 100
        else:
            progress_percent = 0.0
        
        # Tiempo transcurrido
        if self.start_time:
            elapsed_seconds = (datetime.now() - self.start_time).total_seconds()
        else:
            elapsed_seconds = 0
        
        # Crear historial para gráficos (solo timestamps recientes)
        history_for_graphs = []
        for point in self.history:
            history_for_graphs.append({
                'time': point['timestamp'].strftime('%H:%M:%S'),
                'processed': point['processed'],
                'speed': point['speed'],
                'memory_mb': point['memory_mb']
            })
        
        return {
            # Progreso básico
            'processed': self.processed_images,
            'total': self.total_images,
            'progress_percent': progress_percent,
            
            # Métricas de tiempo
            'elapsed_seconds': elapsed_seconds,
            'eta_seconds': self._eta_seconds,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            
            # Métricas de rendimiento
            'speed': self._current_speed,  # imágenes/segundo
            'memory_mb': self._memory_usage,
            
            # Información adicional
            'total_descriptors': self.total_descriptors,
            'estimated_total_features': self.processed_images * self.total_descriptors,
            
            # Historial para gráficos
            'history': history_for_graphs,
            
            # Estado
            'is_complete': self.processed_images >= self.total_images,
            'last_update': datetime.now().isoformat()
        }
    
    def get_summary(self) -> Dict:
        """
        Obtiene un resumen del progreso actual.
        
        Returns:
            Dict: Resumen del progreso
        """
        with self._lock:
            return self._create_progress_info()
    
    def get_eta_formatted(self) -> str:
        """
        Obtiene el tiempo estimado restante formateado.
        
        Returns:
            str: ETA en formato legible (ej: "2m 30s", "1h 15m")
        """
        if self._eta_seconds <= 0:
            return "Calculando..."
        
        eta_delta = timedelta(seconds=int(self._eta_seconds))
        
        # Formatear según la duración
        total_seconds = int(eta_delta.total_seconds())
        
        if total_seconds < 60:
            return f"{total_seconds}s"
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            return f"{minutes}m {seconds}s"
        else:
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            return f"{hours}h {minutes}m"
    
    def get_speed_formatted(self) -> str:
        """
        Obtiene la velocidad formateada.
        
        Returns:
            str: Velocidad en formato legible
        """
        if self._current_speed < 0.01:
            return "0.0 img/s"
        elif self._current_speed < 1:
            return f"{self._current_speed:.2f} img/s"
        else:
            return f"{self._current_speed:.1f} img/s"
    
    def reset(self):
        """Reinicia el tracker."""
        with self._lock:
            self.total_images = 0
            self.total_descriptors = 0
            self.processed_images = 0
            self.start_time = None
            self.last_update_time = None
            self.history.clear()
            self._current_speed = 0.0
            self._eta_seconds = 0
            self._memory_usage = 0.0