"""
Pipeline principal para extracción de descriptores de textura.

Este módulo implementa el flujo completo de procesamiento:
1. Descubrimiento y carga de imágenes
2. Inicialización de descriptores seleccionados
3. Procesamiento con checkpoints y recuperación  
4. Extracción de características con manejo de errores
5. Consolidación y guardado de resultados

El pipeline está diseñado para ser interrumpible y reanudable,
con manejo robusto de errores y logging detallado.
"""

import os
import glob
import pandas as pd
from typing import Dict, List, Optional, Callable, Any
import logging
from datetime import datetime
import traceback
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

# Imports del proyecto
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from descriptors import (
    list_available_descriptors, 
    create_descriptor
)
from core.checkpoint_manager import CheckpointManager
from core.progress_tracker import ProgressTracker
from utils.image_loader import ImageLoader
import config


class TextureExtractionPipeline:
    """
    Pipeline principal para extracción masiva de descriptores de textura.
    
    Esta clase coordina todo el proceso de extracción:
    - Gestión de checkpoints para procesamiento interrumpible
    - Procesamiento paralelo configurable
    - Manejo robusto de errores por imagen
    - Tracking de progreso en tiempo real
    - Consolidación automática de resultados
    
    El pipeline puede ser pausado y reanudado en cualquier momento,
    manteniendo un estado consistente.
    """
    
    def __init__(self, 
                 descriptors: List[str],
                 descriptor_configs: Dict[str, Dict] = None,
                 batch_size: int = 10,
                 n_jobs: int = 1,
                 enable_checkpoints: bool = True,
                 progress_callback: Optional[Callable] = None):
        """
        Inicializa el pipeline de extracción.
        
        Args:
            descriptors (List[str]): Lista de nombres de descriptores a usar.
                                    Ejemplo: ['glcm', 'lbp', 'statistical']
                                    
            descriptor_configs (Dict[str, Dict]): Configuración específica 
                                                 para cada descriptor.
                                                 Si None, usa configuraciones por defecto.
                                                 
            batch_size (int): Número de imágenes a procesar antes de crear checkpoint.
                             Menor = checkpoints más frecuentes pero menor eficiencia.
                             
            n_jobs (int): Número de trabajos paralelos.
                         1 = secuencial, >1 = paralelo, -1 = todos los cores.
                         
            enable_checkpoints (bool): Si habilitar sistema de checkpoints.
                                      Recomendado True para procesamiento largo.
                                      
            progress_callback (Callable): Función callback para reportar progreso.
                                         Útil para GUIs. Recibe dict con info de progreso.
        """
        # Configuración básica
        self.descriptors = descriptors
        self.descriptor_configs = descriptor_configs or {}
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.enable_checkpoints = enable_checkpoints
        self.progress_callback = progress_callback
        
        # Componentes internos
        self.logger = logging.getLogger(__name__)
        self.checkpoint_manager = CheckpointManager(config.OUTPUT_CHECKPOINTS_DIR) if enable_checkpoints else None
        self.progress_tracker = ProgressTracker()
        self.image_loader = ImageLoader(
            target_dtype=config.IMAGE_TARGET_DTYPE,
            normalize_range=config.IMAGE_NORMALIZE_RANGE,
            min_size=config.IMAGE_MIN_SIZE,
            max_size=config.IMAGE_MAX_SIZE
        )
        
        # Estados de control
        self._is_running = False
        self._should_pause = False
        self._should_stop = False
        self._lock = threading.Lock()
        
        # Validar configuración
        self._validate_configuration()
        
        self.logger.info(f"Pipeline inicializado: {len(descriptors)} descriptores, "
                        f"batch_size={batch_size}, n_jobs={n_jobs}")
    
    def _validate_configuration(self):
        """Valida que la configuración del pipeline sea correcta."""
        
        # Validar descriptores
        available = list_available_descriptors()
        invalid_descriptors = set(self.descriptors) - set(available)
        if invalid_descriptors:
            raise ValueError(f"Descriptores no válidos: {invalid_descriptors}. "
                           f"Disponibles: {available}")
        
        # Validar directorios de entrada
        for label, directory in [('Normal', config.RP_INPUT_NORMAL_DIR), 
                                ('Pathol', config.RP_INPUT_PATHOL_DIR)]:
            if not os.path.exists(directory):
                raise FileNotFoundError(f"Directorio {label} no encontrado: {directory}")
        
        # Crear directorios de salida
        os.makedirs(config.OUTPUT_FEATURES_DIR, exist_ok=True)
        
        self.logger.info("Configuración validada correctamente")
    
    def run(self) -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo de extracción.
        
        Returns:
            Dict[str, Any]: Resultados del procesamiento incluyendo:
                           - total_images: número de imágenes procesadas
                           - total_features: número de características extraídas
                           - processing_time: tiempo total de procesamiento
                           - output_files: archivos generados
                           - errors: lista de errores encontrados
        """
        self.logger.info("=== INICIANDO PIPELINE DE EXTRACCIÓN DE TEXTURA ===")
        start_time = datetime.now()
        
        try:
            with self._lock:
                self._is_running = True
                self._should_pause = False
                self._should_stop = False
            
            # 1. Intentar recuperar desde checkpoint
            resume_state = self._attempt_resume()
            
            # 2. Descubrir imágenes a procesar
            image_paths = self._discover_images(resume_state)
            
            if not image_paths:
                self.logger.warning("No hay imágenes para procesar")
                return self._create_empty_result()
            
            # 3. Inicializar descriptores
            descriptors_instances = self._initialize_descriptors()
            
            # 4. Configurar tracking de progreso
            self.progress_tracker.initialize(
                total_images=len(image_paths),
                total_descriptors=len(descriptors_instances)
            )
            
            # 5. Procesar imágenes
            results_df, errors = self._process_images(
                image_paths, descriptors_instances, resume_state
            )
            
            # 6. Guardar resultados finales
            output_files = self._save_final_results(results_df)
            
            # 7. Generar reporte final
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            final_results = {
                'success': True,
                'total_images': len(image_paths),
                'images_processed': len(results_df) if results_df is not None else 0,
                'total_features': len(results_df.columns) - 2 if results_df is not None else 0,  # -2 por filename y label
                'processing_time': processing_time,
                'output_files': output_files,
                'errors': errors,
                'descriptors_used': self.descriptors,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat()
            }
            
            self.logger.info(f"=== PIPELINE COMPLETADO EXITOSAMENTE ===")
            self.logger.info(f"Imágenes procesadas: {final_results['images_processed']}")
            self.logger.info(f"Características extraídas: {final_results['total_features']}")
            self.logger.info(f"Tiempo total: {processing_time:.2f}s")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Error fatal en pipeline: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            return {
                'success': False,
                'error': str(e),
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
        finally:
            with self._lock:
                self._is_running = False
    
    def _attempt_resume(self) -> Optional[Dict[str, Any]]:
        """
        Intenta reanudar desde un checkpoint existente.
        
        Returns:
            Optional[Dict[str, Any]]: Estado recuperado o None si no hay checkpoint
        """
        if not self.enable_checkpoints:
            return None
        
        resume_state = self.checkpoint_manager.load_latest_checkpoint()
        
        if resume_state:
            self.logger.info(f"Reanudando desde checkpoint: "
                           f"{len(resume_state['processed_files'])} archivos ya procesados")
            return resume_state
        
        return None
    
    def _discover_images(self, resume_state: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Descubre todas las imágenes a procesar.
        
        Args:
            resume_state: Estado de reanudación si existe
            
        Returns:
            List[Dict[str, str]]: Lista de imágenes con 'path' y 'label'
        """
        self.logger.info("Descubriendo imágenes a procesar...")
        
        # Obtener archivos ya procesados
        processed_files = set()
        if resume_state:
            processed_files = resume_state['processed_files']
        
        image_paths = []
        
        # Buscar imágenes normales
        normal_pattern = os.path.join(config.RP_INPUT_NORMAL_DIR, f"*{config.IMAGE_FORMAT}")
        for path in glob.glob(normal_pattern):
            filename = os.path.basename(path)
            if filename not in processed_files:
                image_paths.append({
                    'path': path,
                    'filename': filename,
                    'label': 'Normal'
                })
        
        # Buscar imágenes patológicas
        pathol_pattern = os.path.join(config.RP_INPUT_PATHOL_DIR, f"*{config.IMAGE_FORMAT}")
        for path in glob.glob(pathol_pattern):
            filename = os.path.basename(path)
            if filename not in processed_files:
                image_paths.append({
                    'path': path,
                    'filename': filename,
                    'label': 'Pathol'
                })
        
        self.logger.info(f"Encontradas {len(image_paths)} imágenes para procesar")
        self.logger.info(f"  - {len(processed_files)} ya procesadas (saltadas)")
        
        return image_paths
    
    def _initialize_descriptors(self) -> Dict[str, Any]:
        """
        Inicializa las instancias de descriptores configurados.
        
        Returns:
            Dict[str, Any]: Diccionario de descriptores inicializados
        """
        self.logger.info("Inicializando descriptores...")
        
        descriptors_instances = {}
        
        for desc_name in self.descriptors:
            try:
                # Obtener configuración específica o usar por defecto
                desc_config = self.descriptor_configs.get(desc_name, {})
                
                # Crear instancia
                descriptor = create_descriptor(desc_name, **desc_config)
                descriptors_instances[desc_name] = descriptor
                
                self.logger.info(f"  ✓ {desc_name}: {descriptor.get_description()}")
                
            except Exception as e:
                self.logger.error(f"  ✗ Error inicializando {desc_name}: {str(e)}")
                raise
        
        return descriptors_instances
    
    def _process_images(self, 
                       image_paths: List[Dict[str, str]], 
                       descriptors: Dict[str, Any],
                       resume_state: Optional[Dict[str, Any]]) -> tuple:
        """
        Procesa todas las imágenes extrayendo características.
        
        Args:
            image_paths: Lista de imágenes a procesar
            descriptors: Descriptores inicializados
            resume_state: Estado de reanudación si existe
            
        Returns:
            tuple: (DataFrame con resultados, Lista de errores)
        """
        self.logger.info(f"Procesando {len(image_paths)} imágenes...")
        
        # Inicializar estructuras de datos
        all_features = []
        errors = []
        processed_count = 0
        
        # Cargar datos existentes si hay reanudación
        if resume_state and 'features_df' in resume_state:
            existing_df = resume_state['features_df']
            all_features = existing_df.to_dict('records')
            processed_count = len(all_features)
            print(f"DEBUG: Checkpoint cargado - {processed_count} imágenes ya procesadas")
            
            # Filtrar imágenes ya procesadas
            processed_filenames = set()
            for feature in all_features:
                if 'filename' in feature:
                    processed_filenames.add(feature['filename'])
            
            # Crear lista de imágenes pendientes
            pending_images = []
            for img_info in image_paths:
                if img_info['filename'] not in processed_filenames:
                    pending_images.append(img_info)
            
            print(f"DEBUG: Filtrando imágenes - {len(pending_images)} pendientes de {len(image_paths)} totales")
            image_paths = pending_images
            
        # Actualizar el total correcto para el progreso
        original_total = processed_count + len(image_paths)
        print(f"DEBUG: Total real de imágenes: {original_total} (ya procesadas: {processed_count}, pendientes: {len(image_paths)})")
        
        # Procesar por lotes para checkpoints
        for batch_start in range(0, len(image_paths), self.batch_size):
            
            # Verificar si debemos parar
            if self._should_stop:
                self.logger.info("Procesamiento detenido por usuario")
                break
            
            # Esperar si estamos pausados
            while self._should_pause and not self._should_stop:
                time.sleep(0.1)
            
            # Definir lote actual
            batch_end = min(batch_start + self.batch_size, len(image_paths))
            current_batch = image_paths[batch_start:batch_end]
            
            self.logger.info(f"Procesando lote {batch_start + 1}-{batch_end}...")
            print(f"DEBUG: Iniciando procesamiento del lote {batch_start + 1}-{batch_end} ({len(current_batch)} imágenes)")
            
            # Procesar lote
            batch_features, batch_errors = self._process_batch(current_batch, descriptors)
            print(f"DEBUG: Lote procesado. Características extraídas: {len(batch_features)}, Errores: {len(batch_errors)}")
            
            # Agregar resultados
            all_features.extend(batch_features)
            errors.extend(batch_errors)
            processed_count += len(batch_features)
            
            # Actualizar progreso
            print(f"DEBUG: Actualizando progreso - procesadas: {processed_count}, total: {original_total}")
            self._update_progress(processed_count, original_total)
            
            # Crear checkpoint
            if self.enable_checkpoints and processed_count > 0:
                self._create_checkpoint(all_features, processed_count)
        
        # Convertir a DataFrame
        if all_features:
            results_df = pd.DataFrame(all_features)
            self.logger.info(f"Características consolidadas: {results_df.shape}")
        else:
            results_df = None
            self.logger.warning("No se procesaron características")
        
        return results_df, errors
    
    def _process_batch(self, 
                      batch: List[Dict[str, str]], 
                      descriptors: Dict[str, Any]) -> tuple:
        """
        Procesa un lote de imágenes.
        
        Args:
            batch: Lote de imágenes a procesar
            descriptors: Descriptores a aplicar
            
        Returns:
            tuple: (Lista de características, Lista de errores)
        """
        batch_features = []
        batch_errors = []
        
        if self.n_jobs == 1:
            # Procesamiento secuencial
            for image_info in batch:
                # Verificar si debemos parar antes de procesar cada imagen
                if self._should_stop:
                    self.logger.info("Procesamiento detenido durante lote secuencial")
                    break
                    
                try:
                    features = self._process_single_image(image_info, descriptors)
                    if features:
                        batch_features.append(features)
                except Exception as e:
                    error_info = {
                        'filename': image_info['filename'],
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                    batch_errors.append(error_info)
                    self.logger.error(f"Error procesando {image_info['filename']}: {str(e)}")
        
        else:
            # Procesamiento paralelo
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                # Enviar trabajos
                future_to_image = {
                    executor.submit(self._process_single_image, image_info, descriptors): image_info
                    for image_info in batch
                }
                
                # Recoger resultados
                for future in as_completed(future_to_image):
                    # Verificar si debemos parar antes de procesar cada resultado
                    if self._should_stop:
                        self.logger.info("Procesamiento detenido durante lote paralelo")
                        # Cancelar trabajos pendientes
                        for remaining_future in future_to_image:
                            remaining_future.cancel()
                        break
                        
                    image_info = future_to_image[future]
                    try:
                        features = future.result()
                        if features:
                            batch_features.append(features)
                    except Exception as e:
                        error_info = {
                            'filename': image_info['filename'],
                            'error': str(e),
                            'timestamp': datetime.now().isoformat()
                        }
                        batch_errors.append(error_info)
                        self.logger.error(f"Error procesando {image_info['filename']}: {str(e)}")
        
        return batch_features, batch_errors
    
    def _process_single_image(self, 
                             image_info: Dict[str, str], 
                             descriptors: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Procesa una sola imagen con todos los descriptores.
        
        Args:
            image_info: Información de la imagen
            descriptors: Descriptores a aplicar
            
        Returns:
            Optional[Dict[str, Any]]: Características extraídas o None si falla
        """
        filename = image_info['filename']
        image_path = image_info['path']
        label = image_info['label']
        
        print(f"DEBUG: Procesando imagen: {filename}")
        
        try:
            # Verificar si debemos parar antes de procesar
            if self._should_stop:
                return None
                
            # Cargar imagen
            image = self.image_loader.load_image(image_path)
            if image is None:
                raise ValueError(f"No se pudo cargar la imagen: {image_path}")
            
            # Extraer características con cada descriptor
            features = {
                'filename': filename,
                'label': label
            }
            
            for desc_name, descriptor in descriptors.items():
                # Verificar si debemos parar antes de cada descriptor
                if self._should_stop:
                    return None
                    
                try:
                    # Extraer características del descriptor
                    desc_features = descriptor.extract_with_validation(image)
                    
                    # Agregar al diccionario principal
                    features.update(desc_features)
                    
                except Exception as e:
                    self.logger.warning(f"Error en descriptor {desc_name} para {filename}: {str(e)}")
                    # Continuar con otros descriptores
                    continue
            
            # Verificar que se extrajeron características
            if len(features) <= 2:  # Solo filename y label
                raise ValueError("No se pudieron extraer características")
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error procesando imagen {filename}: {str(e)}")
            raise
    
    def _create_checkpoint(self, all_features: List[Dict], processed_count: int):
        """Crea un checkpoint con el estado actual."""
        if not self.checkpoint_manager:
            return
        
        try:
            # Crear DataFrame temporal
            features_df = pd.DataFrame(all_features)
            
            # Crear lista de archivos procesados
            processed_files = [f['filename'] for f in all_features]
            
            # Metadata del checkpoint
            metadata = {
                'descriptors': self.descriptors,
                'descriptor_configs': self.descriptor_configs,
                'processed_count': processed_count,
                'timestamp': datetime.now().isoformat()
            }
            
            # Guardar checkpoint
            self.checkpoint_manager.save_checkpoint(
                processed_files, features_df, metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error creando checkpoint: {str(e)}")
    
    def _update_progress(self, processed: int, total: int):
        """Actualiza el progreso y notifica callback si existe."""
        progress_info = self.progress_tracker.update(processed, total)
        
        print(f"DEBUG: Actualizando progreso: {processed}/{total}")
        
        if self.progress_callback:
            try:
                self.progress_callback(progress_info)
                print(f"DEBUG: Callback de progreso llamado exitosamente")
            except Exception as e:
                self.logger.warning(f"Error en callback de progreso: {str(e)}")
                print(f"DEBUG: Error en callback: {str(e)}")
    
    def _save_final_results(self, results_df: Optional[pd.DataFrame]) -> List[str]:
        """
        Guarda los resultados finales.
        
        Args:
            results_df: DataFrame con todas las características
            
        Returns:
            List[str]: Lista de archivos guardados
        """
        if results_df is None:
            return []
        
        output_files = []
        
        try:
            # Guardar CSV principal
            csv_path = config.FEATURES_OUTPUT_FILE
            results_df.to_csv(csv_path, index=False)
            output_files.append(csv_path)
            self.logger.info(f"Características guardadas en: {csv_path}")
            
            # Guardar metadata
            metadata = {
                'extraction_date': datetime.now().isoformat(),
                'descriptors_used': self.descriptors,
                'descriptor_configs': self.descriptor_configs,
                'total_images': len(results_df),
                'total_features': len(results_df.columns) - 2,  # -2 por filename y label
                'feature_names': list(results_df.columns),
                'class_distribution': results_df['label'].value_counts().to_dict()
            }
            
            metadata_path = config.FEATURES_METADATA_FILE
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            output_files.append(metadata_path)
            self.logger.info(f"Metadata guardada en: {metadata_path}")
            
        except Exception as e:
            self.logger.error(f"Error guardando resultados: {str(e)}")
        
        return output_files
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """Crea un resultado vacío."""
        return {
            'success': True,
            'total_images': 0,
            'images_processed': 0,
            'total_features': 0,
            'processing_time': 0,
            'output_files': [],
            'errors': [],
            'descriptors_used': self.descriptors
        }
    
    # Métodos de control público
    
    def pause(self):
        """Pausa el procesamiento."""
        with self._lock:
            self._should_pause = True
        self.logger.info("Pipeline pausado")
    
    def resume(self):
        """Reanuda el procesamiento."""
        with self._lock:
            self._should_pause = False
        self.logger.info("Pipeline reanudado")
    
    def stop(self):
        """Detiene completamente el procesamiento."""
        with self._lock:
            self._should_stop = True
            self._should_pause = False
        self.logger.info("Pipeline detenido")
    
    @property
    def is_running(self) -> bool:
        """Retorna si el pipeline está corriendo."""
        with self._lock:
            return self._is_running
    
    @property
    def is_paused(self) -> bool:
        """Retorna si el pipeline está pausado."""
        with self._lock:
            return self._should_pause