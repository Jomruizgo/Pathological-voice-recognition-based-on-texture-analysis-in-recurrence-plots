"""
Sistema de persistencia de progreso basado en archivos.

Este módulo permite compartir el estado del progreso entre diferentes
sesiones de Streamlit usando archivos persistentes dentro del módulo.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Optional
import threading

class ProgressFileManager:
    """
    Gestiona el progreso de la extracción usando archivos persistentes.
    
    Esto permite que el progreso persista entre recargas de página,
    reinicios del sistema y sea accesible desde cualquier sesión de Streamlit.
    """
    
    def __init__(self, progress_dir: Optional[str] = None):
        """
        Inicializa el gestor de progreso.
        
        Args:
            progress_dir: Directorio para archivos de progreso.
                         Si es None, usa directorio dentro del módulo.
        """
        if progress_dir is None:
            # Usar directorio dentro del módulo para persistencia entre reinicios
            module_dir = Path(__file__).parent.parent  # Subir al directorio del módulo
            progress_dir = module_dir / "output" / "gui_config"
        
        self.progress_dir = Path(progress_dir)
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        
        self.progress_file = self.progress_dir / "current_progress.json"
        self.status_file = self.progress_dir / "extraction_status.json"
        self.config_file = self.progress_dir / "extraction_config.json"
        self.lock = threading.Lock()
    
    def update_progress(self, progress_info: Dict):
        """
        Actualiza el archivo de progreso.
        
        Args:
            progress_info: Información de progreso a guardar
        """
        with self.lock:
            try:
                current_time = time.time()
                
                # Leer progreso anterior para calcular velocidad
                previous_data = None
                try:
                    if self.progress_file.exists():
                        with open(self.progress_file, 'r') as f:
                            previous_data = json.load(f)
                except:
                    previous_data = None
                
                progress_data = {
                    **progress_info,
                    'timestamp': current_time,
                    'last_update': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Normalizar campos de progreso para compatibilidad
                if 'processed' in progress_data and 'progress' not in progress_data:
                    progress_data['progress'] = progress_data['processed']
                
                # Calcular velocidad
                current_progress = progress_data.get('progress', 0)
                if previous_data and current_progress > 0:
                    time_diff = current_time - previous_data.get('timestamp', current_time)
                    progress_diff = current_progress - previous_data.get('progress', 0)
                    
                    if time_diff > 0 and progress_diff > 0:
                        # Velocidad instantánea
                        instant_speed = progress_diff / time_diff
                        
                        # Usar promedio móvil para suavizar la velocidad
                        previous_speed = previous_data.get('speed', 0)
                        if previous_speed > 0:
                            # Promedio ponderado: 30% velocidad anterior, 70% velocidad actual
                            smoothed_speed = 0.3 * previous_speed + 0.7 * instant_speed
                        else:
                            smoothed_speed = instant_speed
                            
                        progress_data['speed'] = round(smoothed_speed, 2)
                    elif progress_diff == 0:
                        # Sin cambio de progreso, mantener velocidad anterior
                        progress_data['speed'] = previous_data.get('speed', 0)
                    else:
                        progress_data['speed'] = 0
                else:
                    progress_data['speed'] = 0
                
                with open(self.progress_file, 'w') as f:
                    json.dump(progress_data, f, indent=2)
                print(f"DEBUG FILE: Progreso guardado - {progress_data.get('progress', 'N/A')}/{progress_data.get('total', 'N/A')} - {progress_data.get('speed', 0):.1f} img/s")
                    
            except Exception as e:
                print(f"Error guardando progreso: {str(e)}")
    
    def read_progress(self) -> Optional[Dict]:
        """
        Lee el progreso actual del archivo.
        
        Returns:
            Dict con información de progreso o None si no hay progreso
        """
        with self.lock:
            try:
                if self.progress_file.exists():
                    with open(self.progress_file, 'r') as f:
                        data = json.load(f)
                    
                    # Verificar que no sea muy antiguo (más de 5 minutos)
                    if time.time() - data.get('timestamp', 0) < 300:
                        return data
                    
                return None
                
            except Exception as e:
                print(f"Error leyendo progreso: {str(e)}")
                return None
    
    def update_status(self, status: str, **kwargs):
        """
        Actualiza el estado de la extracción.
        
        Args:
            status: Estado actual ('running', 'completed', 'error', 'stopped')
            **kwargs: Información adicional del estado
        """
        with self.lock:
            try:
                status_data = {
                    'status': status,
                    'timestamp': time.time(),
                    'last_update': time.strftime('%Y-%m-%d %H:%M:%S'),
                    **kwargs
                }
                
                with open(self.status_file, 'w') as f:
                    json.dump(status_data, f, indent=2)
                    
            except Exception as e:
                print(f"Error guardando estado: {str(e)}")
    
    def read_status(self) -> Optional[Dict]:
        """
        Lee el estado actual de la extracción.
        
        Returns:
            Dict con información de estado o None
        """
        with self.lock:
            try:
                if self.status_file.exists():
                    with open(self.status_file, 'r') as f:
                        return json.load(f)
                return None
                
            except Exception as e:
                print(f"Error leyendo estado: {str(e)}")
                return None
    
    def clear_progress(self):
        """Limpia todos los archivos de progreso."""
        with self.lock:
            try:
                if self.progress_file.exists():
                    self.progress_file.unlink()
                if self.status_file.exists():
                    self.status_file.unlink()
            except Exception as e:
                print(f"Error limpiando progreso: {str(e)}")
    
    def is_extraction_running(self) -> bool:
        """
        Verifica si hay una extracción en progreso.
        
        Returns:
            bool: True si hay extracción activa
        """
        status = self.read_status()
        if status:
            # Verificar que no sea muy antiguo (2 minutos, más estricto)
            time_elapsed = time.time() - status.get('timestamp', 0)
            if time_elapsed < 120:  # 2 minutos en lugar de 5
                # Solo considerar activo si está específicamente en 'running'
                # 'ready' significa que está listo para continuar pero no corriendo
                return status.get('status') == 'running'
        return False
    
    def save_config(self, config: Dict):
        """
        Guarda la configuración de la extracción.
        
        Args:
            config: Configuración a guardar
        """
        with self.lock:
            try:
                config_data = {
                    **config,
                    'timestamp': time.time(),
                    'last_update': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
                with open(self.config_file, 'w') as f:
                    json.dump(config_data, f, indent=2)
                    
            except Exception as e:
                print(f"Error guardando configuración: {str(e)}")
    
    def load_config(self) -> Optional[Dict]:
        """
        Carga la configuración guardada.
        
        Returns:
            Dict con configuración o None
        """
        with self.lock:
            try:
                if self.config_file.exists():
                    with open(self.config_file, 'r') as f:
                        return json.load(f)
                return None
                
            except Exception as e:
                print(f"Error cargando configuración: {str(e)}")
                return None