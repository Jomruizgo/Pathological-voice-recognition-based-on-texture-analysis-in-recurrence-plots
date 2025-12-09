"""
Gestor de checkpoints para procesamiento interrumpible y reanudable.

Este módulo permite:
- Guardar el estado del procesamiento periódicamente
- Reanudar desde el último checkpoint si se interrumpe
- Mantener un historial de procesamiento
- Recuperación ante fallos

El sistema guarda:
- Lista de archivos procesados
- Características extraídas hasta el momento
- Estado de configuración
- Métricas de progreso
"""

import json
import os
import pickle
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
import pandas as pd
import numpy as np
import logging


class CheckpointManager:
    """
    Gestiona el guardado y recuperación de checkpoints durante el procesamiento.
    
    Esta clase implementa un sistema robusto de checkpoints que permite:
    - Interrumpir el procesamiento en cualquier momento
    - Reanudar exactamente donde se quedó
    - Mantener múltiples checkpoints con rotación
    - Recuperación automática ante fallos
    """
    
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 3):
        """
        Inicializa el gestor de checkpoints.
        
        Args:
            checkpoint_dir (str): Directorio donde se guardarán los checkpoints
            max_checkpoints (int): Número máximo de checkpoints a mantener
                                  (los más antiguos se eliminan automáticamente)
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.logger = logging.getLogger(__name__)
        
        # Crear directorio si no existe
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Estado actual
        self.processed_files: Set[str] = set()
        self.features_df: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, Any] = {
            'start_time': None,
            'last_checkpoint': None,
            'total_processed': 0,
            'configuration': {},
            'descriptor_info': {}
        }
        
    def save_checkpoint(self, 
                       processed_files: List[str],
                       features_df: pd.DataFrame,
                       metadata: Dict[str, Any]) -> str:
        """
        Guarda un checkpoint con el estado actual del procesamiento.
        
        Args:
            processed_files (List[str]): Lista de archivos ya procesados
            features_df (pd.DataFrame): DataFrame con características extraídas
            metadata (Dict[str, Any]): Metadata adicional del procesamiento
            
        Returns:
            str: Ruta del checkpoint guardado
        """
        # Crear nombre único para el checkpoint
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_{timestamp}"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        # Crear directorio para este checkpoint
        os.makedirs(checkpoint_path, exist_ok=True)
        
        try:
            # 1. Guardar lista de archivos procesados
            files_path = os.path.join(checkpoint_path, 'processed_files.json')
            with open(files_path, 'w') as f:
                json.dump(sorted(processed_files), f, indent=2)
            
            # 2. Guardar DataFrame de características
            # Usamos múltiples formatos para robustez
            features_path_csv = os.path.join(checkpoint_path, 'features.csv')
            features_path_pkl = os.path.join(checkpoint_path, 'features.pkl')
            
            features_df.to_csv(features_path_csv, index=False)
            features_df.to_pickle(features_path_pkl)
            
            # 3. Guardar metadata
            metadata_enriched = {
                **metadata,
                'checkpoint_time': datetime.now().isoformat(),
                'n_processed': len(processed_files),
                'n_features': len(features_df.columns) if features_df is not None else 0,
                'n_samples': len(features_df) if features_df is not None else 0
            }
            
            metadata_path = os.path.join(checkpoint_path, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata_enriched, f, indent=2)
            
            # 4. Crear archivo de validación
            validation_path = os.path.join(checkpoint_path, 'validation.txt')
            with open(validation_path, 'w') as f:
                f.write(f"Checkpoint válido creado en: {datetime.now()}\n")
                f.write(f"Archivos procesados: {len(processed_files)}\n")
                f.write(f"Características extraídas: {len(features_df) if features_df is not None else 0}\n")
            
            self.logger.info(f"✅ Checkpoint guardado: {checkpoint_path}")
            self.logger.info(f"   - Archivos procesados: {len(processed_files)}")
            self.logger.info(f"   - Características guardadas: {len(features_df) if features_df is not None else 0}")
            
            # Limpiar checkpoints antiguos
            self._cleanup_old_checkpoints()
            
            return checkpoint_path
            
        except Exception as e:
            self.logger.error(f"❌ Error al guardar checkpoint: {str(e)}")
            # Intentar eliminar checkpoint corrupto
            if os.path.exists(checkpoint_path):
                shutil.rmtree(checkpoint_path)
            raise
    
    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Carga el checkpoint más reciente.
        
        Returns:
            Optional[Dict[str, Any]]: Diccionario con el estado recuperado o None
                                     Contiene: 'processed_files', 'features_df', 'metadata'
        """
        # Buscar todos los checkpoints
        checkpoints = self._find_all_checkpoints()
        
        if not checkpoints:
            self.logger.info("No se encontraron checkpoints previos")
            return None
        
        # Intentar cargar desde el más reciente
        for checkpoint_dir in checkpoints:
            try:
                state = self._load_checkpoint(checkpoint_dir)
                if state:
                    self.logger.info(f"✅ Checkpoint cargado: {checkpoint_dir}")
                    return state
            except Exception as e:
                self.logger.warning(f"Error al cargar checkpoint {checkpoint_dir}: {str(e)}")
                continue
        
        self.logger.error("No se pudo cargar ningún checkpoint válido")
        return None
    
    def _load_checkpoint(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """
        Carga un checkpoint específico.
        
        Args:
            checkpoint_path (str): Ruta del checkpoint a cargar
            
        Returns:
            Optional[Dict[str, Any]]: Estado recuperado o None si falla
        """
        # Verificar que existe el directorio
        if not os.path.exists(checkpoint_path):
            return None
        
        # Verificar archivo de validación
        validation_path = os.path.join(checkpoint_path, 'validation.txt')
        if not os.path.exists(validation_path):
            self.logger.warning(f"Checkpoint sin archivo de validación: {checkpoint_path}")
            return None
        
        try:
            # 1. Cargar archivos procesados
            files_path = os.path.join(checkpoint_path, 'processed_files.json')
            with open(files_path, 'r') as f:
                processed_files = set(json.load(f))
            
            # 2. Cargar DataFrame de características
            # Intentar primero pickle (más rápido), luego CSV
            features_path_pkl = os.path.join(checkpoint_path, 'features.pkl')
            features_path_csv = os.path.join(checkpoint_path, 'features.csv')
            
            features_df = None
            if os.path.exists(features_path_pkl):
                try:
                    features_df = pd.read_pickle(features_path_pkl)
                except:
                    self.logger.warning("Error al cargar pickle, intentando CSV...")
            
            if features_df is None and os.path.exists(features_path_csv):
                features_df = pd.read_csv(features_path_csv)
            
            if features_df is None:
                self.logger.error("No se pudo cargar el DataFrame de características")
                return None
            
            # 3. Cargar metadata
            metadata_path = os.path.join(checkpoint_path, 'metadata.json')
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Actualizar estado interno
            self.processed_files = processed_files
            self.features_df = features_df
            self.metadata = metadata
            
            return {
                'processed_files': processed_files,
                'features_df': features_df,
                'metadata': metadata,
                'checkpoint_path': checkpoint_path
            }
            
        except Exception as e:
            self.logger.error(f"Error al cargar checkpoint {checkpoint_path}: {str(e)}")
            return None
    
    def _find_all_checkpoints(self) -> List[str]:
        """
        Encuentra todos los checkpoints disponibles, ordenados por fecha.
        
        Returns:
            List[str]: Lista de rutas de checkpoints (más reciente primero)
        """
        checkpoints = []
        
        # Buscar directorios que empiecen con 'checkpoint_'
        for item in os.listdir(self.checkpoint_dir):
            item_path = os.path.join(self.checkpoint_dir, item)
            if os.path.isdir(item_path) and item.startswith('checkpoint_'):
                # Verificar que tenga archivo de validación
                if os.path.exists(os.path.join(item_path, 'validation.txt')):
                    checkpoints.append(item_path)
        
        # Ordenar por fecha de modificación (más reciente primero)
        checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        return checkpoints
    
    def _cleanup_old_checkpoints(self):
        """
        Elimina checkpoints antiguos manteniendo solo los más recientes.
        """
        checkpoints = self._find_all_checkpoints()
        
        # Si hay más checkpoints que el máximo permitido
        if len(checkpoints) > self.max_checkpoints:
            # Eliminar los más antiguos
            for checkpoint_path in checkpoints[self.max_checkpoints:]:
                try:
                    shutil.rmtree(checkpoint_path)
                    self.logger.info(f"🗑️ Checkpoint antiguo eliminado: {checkpoint_path}")
                except Exception as e:
                    self.logger.warning(f"Error al eliminar checkpoint: {str(e)}")
    
    def get_resume_info(self) -> Dict[str, Any]:
        """
        Obtiene información sobre el estado de reanudación.
        
        Returns:
            Dict[str, Any]: Información sobre qué se puede reanudar
        """
        latest = self.load_latest_checkpoint()
        
        if not latest:
            return {
                'can_resume': False,
                'reason': 'No hay checkpoints disponibles'
            }
        
        return {
            'can_resume': True,
            'checkpoint_time': latest['metadata'].get('checkpoint_time'),
            'processed_files': len(latest['processed_files']),
            'total_features': len(latest['features_df']) if latest['features_df'] is not None else 0,
            'checkpoint_path': latest['checkpoint_path']
        }
    
    def should_create_checkpoint(self, current_count: int, batch_size: int) -> bool:
        """
        Determina si se debe crear un checkpoint basado en el progreso.
        
        Args:
            current_count (int): Número actual de elementos procesados
            batch_size (int): Tamaño del lote para checkpoints
            
        Returns:
            bool: True si se debe crear checkpoint
        """
        return current_count > 0 and current_count % batch_size == 0
    
    @staticmethod
    def checkpoint_exists(checkpoint_dir: str = None) -> bool:
        """
        Verifica si existen checkpoints.
        
        Args:
            checkpoint_dir (str): Directorio de checkpoints (usa config si es None)
            
        Returns:
            bool: True si hay al menos un checkpoint válido
        """
        if checkpoint_dir is None:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            import config
            checkpoint_dir = config.OUTPUT_CHECKPOINTS_DIR
        
        manager = CheckpointManager(checkpoint_dir)
        return len(manager._find_all_checkpoints()) > 0