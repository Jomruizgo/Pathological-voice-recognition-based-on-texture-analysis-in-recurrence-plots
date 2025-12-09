"""
Pipeline modular para extracción de descriptores de textura.

Este módulo implementa un sistema modular que permite:
- Calcular descriptores de forma independiente
- Añadir nuevos descriptores sin recalcular los existentes
- Combinar características de forma flexible
- Mantener un índice de descriptores calculados

Diseñado para ser evolutivo y eficiente en el uso de recursos.
"""

import os
import json
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import hashlib

# Configurar imports relativos para ejecución directa
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pipeline import TextureExtractionPipeline
from descriptors import create_descriptor
import config


class ModularPipeline(TextureExtractionPipeline):
    """
    Pipeline modular que extiende TextureExtractionPipeline para soportar
    cálculo y almacenamiento independiente de descriptores.
    
    Cada descriptor se calcula y almacena por separado, permitiendo:
    - Añadir descriptores sin recalcular existentes
    - Combinar subsets de características bajo demanda
    - Versionado independiente por descriptor
    
    NOTA: Este pipeline NO usa checkpoints tradicionales, solo el sistema modular
    con manifest.json y checkpoints parciales por descriptor.
    """
    
    def __init__(self, *args, **kwargs):
        """Inicializa el pipeline modular."""
        # Forzar enable_checkpoints=False para evitar CheckpointManager
        kwargs['enable_checkpoints'] = False
        super().__init__(*args, **kwargs)
        
        # Directorios específicos para el sistema modular
        self.features_by_descriptor_dir = os.path.join(config.OUTPUT_FEATURES_DIR, 'by_descriptor')
        self.combined_features_dir = os.path.join(config.OUTPUT_FEATURES_DIR, 'combined')
        self.manifest_path = os.path.join(config.OUTPUT_FEATURES_DIR, 'manifest.json')
        
        # Crear directorios si no existen
        os.makedirs(self.features_by_descriptor_dir, exist_ok=True)
        os.makedirs(self.combined_features_dir, exist_ok=True)
        
        # Cargar manifest existente
        self.manifest = self._load_manifest()
        
        # Inicializar control de parada (necesario para la GUI)
        self._should_stop = False
        
    def _load_manifest(self) -> Dict[str, Any]:
        """Carga el manifest de descriptores calculados."""
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Error cargando manifest: {e}")
        
        # Manifest vacío por defecto
        return {
            'descriptors': {},
            'images_processed': [],
            'last_update': None
        }
    
    def _save_manifest(self):
        """Guarda el manifest actualizado."""
        self.manifest['last_update'] = datetime.now().isoformat()
        
        try:
            with open(self.manifest_path, 'w') as f:
                json.dump(self.manifest, f, indent=2)
            self.logger.info("Manifest actualizado")
        except Exception as e:
            self.logger.error(f"Error guardando manifest: {e}")
    
    def _get_config_hash(self, descriptor_name: str, config: Dict) -> str:
        """
        Genera un hash único para la configuración de un descriptor.
        
        Esto permite detectar si la configuración cambió.
        """
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def _get_descriptor_dir(self, descriptor_name: str) -> str:
        """Obtiene el directorio para un descriptor específico."""
        return os.path.join(self.features_by_descriptor_dir, descriptor_name)
    
    def _check_descriptor_status(self, descriptor_name: str, config: Dict) -> Dict[str, Any]:
        """
        Verifica el estado de un descriptor.
        
        Returns:
            Dict con información sobre si el descriptor está calculado,
            si la configuración cambió, etc.
        """
        status = {
            'exists': False,
            'config_changed': False,
            'images_missing': [],
            'can_reuse': False
        }
        
        try:
            # Verificar si el descriptor está en el manifest
            if descriptor_name not in self.manifest['descriptors']:
                return status
            
            descriptor_info = self.manifest['descriptors'][descriptor_name]
            status['exists'] = True
            
            # Verificar si la configuración cambió
            current_hash = self._get_config_hash(descriptor_name, config)
            if descriptor_info.get('config_hash') != current_hash:
                status['config_changed'] = True
                return status
            
            # Verificar si hay nuevas imágenes
            descriptor_dir = self._get_descriptor_dir(descriptor_name)
            features_path = os.path.join(descriptor_dir, 'features.csv')
            
            if os.path.exists(features_path):
                try:
                    existing_df = pd.read_csv(features_path)
                    existing_files = set(existing_df['filename'].values)
                    
                    # Obtener lista actual de imágenes
                    self.logger.debug("Obteniendo lista de imágenes actuales...")
                    current_images = self._discover_all_images()
                    current_files = {img['filename'] for img in current_images}
                    self.logger.debug(f"Encontradas {len(current_files)} imágenes actuales")
                    
                    # Encontrar imágenes faltantes
                    missing = current_files - existing_files
                    status['images_missing'] = list(missing)
                    
                    # Se puede reusar si no hay imágenes faltantes
                    status['can_reuse'] = len(missing) == 0
                    
                except Exception as e:
                    self.logger.error(f"Error verificando archivos de {descriptor_name}: {e}")
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error crítico verificando estado de {descriptor_name}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return status
    
    def _discover_all_images(self) -> List[Dict[str, str]]:
        """Descubre todas las imágenes disponibles (sin filtrar por procesadas)."""
        try:
            self.logger.debug("Iniciando descubrimiento de imágenes...")
            images = []
            
            # Descubrir imágenes directamente sin usar super()
            for directory, label in [(config.RP_INPUT_NORMAL_DIR, 'Normal'),
                                   (config.RP_INPUT_PATHOL_DIR, 'Pathol')]:
                if not os.path.exists(directory):
                    self.logger.warning(f"Directorio no encontrado: {directory}")
                    continue
                
                for filename in os.listdir(directory):
                    if filename.endswith(config.IMAGE_FORMAT):
                        images.append({
                            'filename': filename,
                            'path': os.path.join(directory, filename),
                            'label': label
                        })
                        
            self.logger.debug(f"Descubrimiento completado: {len(images)} imágenes encontradas")
            return images
        except Exception as e:
            self.logger.error(f"Error en _discover_all_images: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return []
    
    def compute_descriptors(self, descriptors: Optional[List[str]] = None,
                          force_recompute: bool = False) -> Dict[str, Any]:
        """
        Calcula descriptores de forma modular.
        
        Args:
            descriptors: Lista de descriptores a calcular. 
                        Si None, usa los configurados en el pipeline.
            force_recompute: Si True, recalcula incluso si ya existen.
            
        Returns:
            Dict con resultados del cálculo por descriptor.
        """
        if descriptors is None:
            descriptors = self.descriptors
        
        results = {
            'computed': {},
            'reused': {},
            'failed': {},
            'summary': {
                'total_computed': 0,
                'total_reused': 0,
                'total_failed': 0
            }
        }
        
        self.logger.info(f"=== INICIANDO CÁLCULO MODULAR DE DESCRIPTORES ===")
        self.logger.info(f"Descriptores solicitados: {descriptors}")
        self.logger.info(f"Orden de procesamiento: {list(enumerate(descriptors))}")
        
        # Callback de progreso global para múltiples descriptores
        if self.progress_callback:
            global_progress = {
                'total_descriptors': len(descriptors),
                'descriptors_completed': 0,
                'descriptors_reused': 0,
                'current_descriptor': None,
                'descriptor_progress': 0,
                'descriptor_total': 0,
                'phase': 'analyzing',
                # Campos de compatibilidad para update_progress
                'progress': 0,
                'total': 100  # Valor genérico inicial
            }
            self.progress_callback(global_progress)
        
        # Analizar estado de cada descriptor
        for desc_idx, desc_name in enumerate(descriptors):
            # Almacenar contexto global para uso en callbacks
            self._current_descriptor_index = desc_idx + 1
            self._total_descriptors = len(descriptors)
            self._descriptors_completed = results['summary']['total_computed']
            self._descriptors_reused = results['summary']['total_reused']
            self.logger.info(f"\n--- Procesando descriptor: {desc_name} ---")
            
            try:
                # Obtener configuración
                desc_config = self.descriptor_configs.get(desc_name, {})
                self.logger.info(f"Configuración obtenida para {desc_name}: {desc_config}")
                
                # Verificar estado
                self.logger.info(f"Verificando estado de {desc_name}...")
                status = self._check_descriptor_status(desc_name, desc_config)
                self.logger.info(f"Estado verificado: {status}")
                
                if status['can_reuse'] and not force_recompute:
                    self.logger.info(f"✓ {desc_name} ya calculado y actualizado, reusando")
                    results['reused'][desc_name] = status
                    results['summary']['total_reused'] += 1
                    
                    # Actualizar progreso: descriptor reutilizado
                    if self.progress_callback:
                        # Usar valores actualizados después del incremento
                        global_progress = {
                            'total_descriptors': len(descriptors),
                            'descriptors_completed': results['summary']['total_computed'],
                            'descriptors_reused': results['summary']['total_reused'],  # Valor ya incrementado
                            'current_descriptor': desc_name,
                            'descriptor_progress': 100,
                            'descriptor_total': 100,
                            'phase': 'reused',
                            'descriptor_index': desc_idx + 1,
                            # Campos de compatibilidad
                            'progress': 100,
                            'total': 100
                        }
                        self.progress_callback(global_progress)
                    
                    self.logger.info(f"✓ {desc_name} reutilizado, pasando al siguiente descriptor")
                    
                    # Pequeño delay para asegurar que la GUI se actualice
                    import time
                    time.sleep(0.1)
                    
                    continue
                
                # Necesita cálculo
                if status['exists'] and status['config_changed']:
                    self.logger.warning(f"⚠ Configuración de {desc_name} cambió, recalculando todo")
                elif status['exists'] and status['images_missing']:
                    self.logger.info(f"📝 Calculando {len(status['images_missing'])} imágenes nuevas para {desc_name}")
                else:
                    self.logger.info(f"🆕 Calculando {desc_name} por primera vez")
                
                # Actualizar progreso: iniciando cálculo
                if self.progress_callback:
                    global_progress = {
                        'total_descriptors': len(descriptors),
                        'descriptors_completed': results['summary']['total_computed'],
                        'descriptors_reused': results['summary']['total_reused'],
                        'current_descriptor': desc_name,
                        'descriptor_progress': 0,
                        'descriptor_total': 0,  # Se actualizará en _compute_single_descriptor
                        'phase': 'computing',
                        'descriptor_index': desc_idx + 1,
                        # Campos de compatibilidad
                        'progress': 0,
                        'total': 100
                    }
                    self.progress_callback(global_progress)
                
                # Calcular descriptor
                compute_result = self._compute_single_descriptor(
                    desc_name, desc_config, 
                    only_missing=status['images_missing'] if status['exists'] else None
                )
                
                if compute_result['success']:
                    results['computed'][desc_name] = compute_result
                    results['summary']['total_computed'] += 1
                    
                    # Actualizar progreso: descriptor completado
                    if self.progress_callback:
                        # Usar valores actualizados después del incremento
                        global_progress = {
                            'total_descriptors': len(descriptors),
                            'descriptors_completed': results['summary']['total_computed'],  # Valor ya incrementado
                            'descriptors_reused': results['summary']['total_reused'],
                            'current_descriptor': desc_name,
                            'descriptor_progress': 100,
                            'descriptor_total': 100,
                            'phase': 'completed',
                            'descriptor_index': desc_idx + 1,
                            # Campos de compatibilidad
                            'progress': 100,
                            'total': 100
                        }
                        self.progress_callback(global_progress)
                else:
                    results['failed'][desc_name] = compute_result
                    results['summary']['total_failed'] += 1
                    
                    # Actualizar progreso: descriptor falló
                    if self.progress_callback:
                        global_progress = {
                            'total_descriptors': len(descriptors),
                            'descriptors_completed': results['summary']['total_computed'],
                            'descriptors_reused': results['summary']['total_reused'],
                            'current_descriptor': desc_name,
                            'descriptor_progress': 0,
                            'descriptor_total': 100,
                            'phase': 'failed',
                            'descriptor_index': desc_idx + 1,
                            # Campos de compatibilidad
                            'progress': 0,
                            'total': 100
                        }
                        self.progress_callback(global_progress)
                    
            except Exception as e:
                self.logger.error(f"Error procesando {desc_name}: {e}")
                results['failed'][desc_name] = {'error': str(e)}
                results['summary']['total_failed'] += 1
        
        self.logger.info(f"=== LOOP COMPLETADO ===")
        self.logger.info(f"Descriptores procesados: {list(results['computed'].keys())}")
        self.logger.info(f"Descriptores reutilizados: {list(results['reused'].keys())}")
        self.logger.info(f"Descriptores fallidos: {list(results['failed'].keys())}")
        
        # Actualizar manifest
        self._save_manifest()
        
        # Callback final: proceso completamente terminado
        if self.progress_callback:
            final_progress = {
                'total_descriptors': len(descriptors),
                'descriptors_completed': results['summary']['total_computed'],
                'descriptors_reused': results['summary']['total_reused'],
                'current_descriptor': None,
                'descriptor_progress': 100,
                'descriptor_total': 100,
                'phase': 'all_finished',  # Nueva fase para indicar que TODO terminó
                'descriptor_index': len(descriptors),
                # Campos de compatibilidad
                'progress': 100,
                'total': 100
            }
            self.progress_callback(final_progress)
        
        self.logger.info("\n=== CÁLCULO MODULAR COMPLETADO ===")
        self.logger.info(f"Computados: {results['summary']['total_computed']}")
        self.logger.info(f"Reusados: {results['summary']['total_reused']}")
        self.logger.info(f"Fallidos: {results['summary']['total_failed']}")
        
        return results
    
    def _compute_single_descriptor(self, descriptor_name: str, desc_config: Dict,
                                 only_missing: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calcula un descriptor individual.
        
        Args:
            descriptor_name: Nombre del descriptor
            desc_config: Configuración del descriptor
            only_missing: Si se especifica, solo calcula estas imágenes
            
        Returns:
            Dict con resultados del cálculo
        """
        start_time = datetime.now()
        result = {
            'success': False,
            'images_processed': 0,
            'features_extracted': 0,
            'errors': [],
            'time_elapsed': 0
        }
        
        try:
            # Crear directorio del descriptor
            descriptor_dir = self._get_descriptor_dir(descriptor_name)
            os.makedirs(descriptor_dir, exist_ok=True)
            
            # Crear instancia del descriptor
            descriptor = create_descriptor(descriptor_name, **desc_config)
            
            # Obtener imágenes a procesar
            if only_missing:
                # Filtrar solo las imágenes faltantes
                all_images = self._discover_all_images()
                images_to_process = [img for img in all_images if img['filename'] in only_missing]
            else:
                # Procesar todas las imágenes
                images_to_process = self._discover_all_images()
            
            self.logger.info(f"Procesando {len(images_to_process)} imágenes para {descriptor_name}")
            
            # Intentar cargar checkpoint parcial
            features_list = []
            processed_filenames = []
            
            # Calcular total de imágenes ANTES de filtrar
            total_images_to_process = len(images_to_process)
            
            partial_checkpoint = self._load_partial_checkpoint(descriptor_name, desc_config)
            if partial_checkpoint and not only_missing:  # Solo usar checkpoint si procesamos todo
                self.logger.info(f"Reanudando desde checkpoint: {partial_checkpoint['total_processed']} imágenes ya procesadas")
                features_list = partial_checkpoint['features_data']
                processed_filenames = partial_checkpoint['processed_filenames']
                
                # Filtrar imágenes ya procesadas
                processed_set = set(processed_filenames)
                images_to_process = [img for img in images_to_process if img['filename'] not in processed_set]
                result['images_processed'] = len(processed_filenames)
                
                self.logger.info(f"Quedan {len(images_to_process)} imágenes por procesar")
            
            # TEMPORALMENTE DESACTIVADO: El wrapped_progress_callback causa recursión infinita
            # Usar callback original hasta resolver el problema
            
            # # Wrapper para callback que combina información global y del descriptor  
            # def wrapped_progress_callback(progress_info):
            #     """Callback que añade información global del progreso multi-descriptor."""
            #     if self.progress_callback:
            #         # ... código del wrapper ...
            #         self.progress_callback(enhanced_progress)
            
            # Mantener el callback original por ahora
            original_callback = self.progress_callback
            # self.progress_callback = wrapped_progress_callback  # DESACTIVADO
            
            try:
                # Extraer características
                for i, img_info in enumerate(images_to_process):
                    try:
                        # Verificar si debemos detenernos
                        if self._should_stop:
                            self.logger.info("Procesamiento detenido por usuario")
                            break
                        
                        # Cargar imagen
                        image = self.image_loader.load_image(img_info['path'])
                        if image is None:
                            raise ValueError(f"No se pudo cargar: {img_info['path']}")
                        
                        # Extraer características
                        features = descriptor.extract_with_validation(image)
                        
                        # Añadir metadata
                        features['filename'] = img_info['filename']
                        features['label'] = img_info['label']
                        
                        features_list.append(features)
                        processed_filenames.append(img_info['filename'])
                        result['images_processed'] += 1
                        
                        # Actualizar progreso si hay callback (después de cada imagen)
                        if self.progress_callback:
                            progress_info = {
                                'descriptor': descriptor_name,
                                'processed': len(processed_filenames),  # Total procesadas hasta ahora  
                                'total': total_images_to_process,  # Total original de imágenes
                                'progress': len(processed_filenames),  # Para compatibilidad
                                'current_file': img_info['filename']
                            }
                            self.progress_callback(progress_info)
                        
                        # Guardar checkpoint parcial si es necesario
                        if (config.ENABLE_PARTIAL_CHECKPOINTS and 
                            result['images_processed'] % config.CHECKPOINT_BATCH_SIZE == 0):
                            self._save_partial_checkpoint(
                                descriptor_name, 
                                processed_filenames.copy(),
                                features_list.copy(),
                                desc_config  # Este es el diccionario del descriptor
                            )
                        
                    except Exception as e:
                        error_info = {
                            'filename': img_info['filename'],
                            'error': str(e)
                        }
                        result['errors'].append(error_info)
                        self.logger.error(f"Error en {img_info['filename']}: {e}")
            
            finally:
                # Restaurar callback original
                self.progress_callback = original_callback
            
            # Guardar resultados
            if features_list:
                # Crear DataFrame
                features_df = pd.DataFrame(features_list)
                
                # Si estamos añadiendo a un descriptor existente, combinar
                if only_missing:
                    existing_path = os.path.join(descriptor_dir, 'features.csv')
                    if os.path.exists(existing_path):
                        existing_df = pd.read_csv(existing_path)
                        features_df = pd.concat([existing_df, features_df], ignore_index=True)
                
                # Guardar CSV
                csv_path = os.path.join(descriptor_dir, 'features.csv')
                features_df.to_csv(csv_path, index=False)
                
                # Guardar metadata del descriptor
                metadata = {
                    'descriptor_name': descriptor_name,
                    'config': desc_config,
                    'config_hash': self._get_config_hash(descriptor_name, desc_config),
                    'total_images': len(features_df),
                    'feature_names': [col for col in features_df.columns if col not in ['filename', 'label']],
                    'feature_count': len(features_df.columns) - 2,  # -2 por filename y label
                    'computation_date': datetime.now().isoformat(),
                    'errors': result['errors']
                }
                
                metadata_path = os.path.join(descriptor_dir, 'metadata.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Actualizar manifest
                self.manifest['descriptors'][descriptor_name] = {
                    'computed_date': datetime.now().isoformat(),
                    'total_images': len(features_df),
                    'feature_count': metadata['feature_count'],
                    'config_hash': metadata['config_hash']
                }
                
                # Actualizar lista de imágenes procesadas en manifest
                if not only_missing:  # Solo si procesamos todas
                    current_images = set(self.manifest.get('images_processed', []))
                    new_images = set(features_df['filename'].values)
                    self.manifest['images_processed'] = sorted(current_images.union(new_images))
                
                result['success'] = True
                result['features_extracted'] = metadata['feature_count']
                
                # Limpiar checkpoint parcial al completar exitosamente
                self._clean_partial_checkpoint(descriptor_name)
                
                self.logger.info(f"✓ {descriptor_name} guardado: {result['images_processed']} imágenes, "
                               f"{result['features_extracted']} características")
            
        except Exception as e:
            self.logger.error(f"Error en cálculo de {descriptor_name}: {e}")
            result['error'] = str(e)
        
        result['time_elapsed'] = (datetime.now() - start_time).total_seconds()
        return result
    
    def combine_features(self, descriptors: Optional[List[str]] = None,
                        output_name: str = 'combined_features') -> Optional[pd.DataFrame]:
        """
        Combina características de múltiples descriptores.
        
        Args:
            descriptors: Lista de descriptores a combinar. 
                        Si None, combina todos los disponibles.
            output_name: Nombre base para el archivo de salida
            
        Returns:
            DataFrame combinado o None si falla
        """
        self.logger.info(f"=== COMBINANDO CARACTERÍSTICAS ===")
        
        # Determinar qué descriptores combinar
        if descriptors is None:
            descriptors = list(self.manifest['descriptors'].keys())
        
        if not descriptors:
            self.logger.warning("No hay descriptores para combinar")
            return None
        
        self.logger.info(f"Combinando descriptores: {descriptors}")
        
        try:
            # Iniciar con el primer descriptor
            first_desc = descriptors[0]
            first_path = os.path.join(self._get_descriptor_dir(first_desc), 'features.csv')
            
            if not os.path.exists(first_path):
                self.logger.error(f"No se encuentra {first_desc}")
                return None
            
            # Leer primer DataFrame
            combined_df = pd.read_csv(first_path)
            self.logger.info(f"- {first_desc}: {combined_df.shape[1]-2} características")
            
            # Combinar con los demás descriptores
            for desc_name in descriptors[1:]:
                desc_path = os.path.join(self._get_descriptor_dir(desc_name), 'features.csv')
                
                if not os.path.exists(desc_path):
                    self.logger.warning(f"Saltando {desc_name} - no encontrado")
                    continue
                
                # Leer descriptor
                desc_df = pd.read_csv(desc_path)
                self.logger.info(f"- {desc_name}: {desc_df.shape[1]-2} características")
                
                # Combinar por filename
                # Mantener solo filename y label del primer DataFrame
                desc_df_features = desc_df.drop(['label'], axis=1)
                combined_df = combined_df.merge(desc_df_features, on='filename', how='inner')
            
            self.logger.info(f"\nResultado combinado:")
            self.logger.info(f"- Imágenes: {len(combined_df)}")
            self.logger.info(f"- Características totales: {combined_df.shape[1]-2}")
            
            # Guardar resultado combinado
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"{output_name}_{timestamp}.csv"
            output_path = os.path.join(self.combined_features_dir, output_filename)
            
            combined_df.to_csv(output_path, index=False)
            self.logger.info(f"✓ Guardado en: {output_path}")
            
            # También guardar metadata de la combinación
            combination_metadata = {
                'descriptors_combined': descriptors,
                'combination_date': datetime.now().isoformat(),
                'total_images': len(combined_df),
                'total_features': combined_df.shape[1] - 2,
                'output_file': output_filename
            }
            
            metadata_path = os.path.join(self.combined_features_dir, f"{output_name}_{timestamp}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(combination_metadata, f, indent=2)
            
            return combined_df
            
        except Exception as e:
            self.logger.error(f"Error combinando características: {e}")
            return None
    
    def get_available_descriptors(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtiene información sobre descriptores disponibles (calculados).
        
        Returns:
            Dict con información de cada descriptor calculado
        """
        return self.manifest.get('descriptors', {})
    
    def get_descriptor_features(self, descriptor_name: str) -> Optional[pd.DataFrame]:
        """
        Carga las características de un descriptor específico.
        
        Args:
            descriptor_name: Nombre del descriptor
            
        Returns:
            DataFrame con las características o None
        """
        features_path = os.path.join(self._get_descriptor_dir(descriptor_name), 'features.csv')
        
        if os.path.exists(features_path):
            return pd.read_csv(features_path)
        
        return None
    
    def clean_descriptor(self, descriptor_name: str) -> bool:
        """
        Elimina los datos calculados de un descriptor.
        
        Args:
            descriptor_name: Nombre del descriptor a limpiar
            
        Returns:
            bool: True si se limpió exitosamente
        """
        try:
            import shutil
            
            # Eliminar directorio del descriptor
            descriptor_dir = self._get_descriptor_dir(descriptor_name)
            if os.path.exists(descriptor_dir):
                shutil.rmtree(descriptor_dir)
                self.logger.info(f"Directorio de {descriptor_name} eliminado")
            
            # Eliminar del manifest
            if descriptor_name in self.manifest['descriptors']:
                del self.manifest['descriptors'][descriptor_name]
                self._save_manifest()
                self.logger.info(f"{descriptor_name} eliminado del manifest")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error limpiando {descriptor_name}: {e}")
            return False
    
    # ═══════════════════════════════════════════════════════════════
    # Métodos de Checkpoints Granulares
    # ═══════════════════════════════════════════════════════════════
    
    def _get_partial_checkpoint_path(self, descriptor_name: str) -> str:
        """Obtiene la ruta del checkpoint parcial de un descriptor."""
        descriptor_dir = self._get_descriptor_dir(descriptor_name)
        return os.path.join(descriptor_dir, 'partial_checkpoint.json')
    
    def _save_partial_checkpoint(self, descriptor_name: str, 
                                processed_filenames: List[str],
                                features_data: List[Dict],
                                descriptor_config: Dict) -> bool:
        """
        Guarda checkpoint parcial durante el procesamiento de un descriptor.
        
        Args:
            descriptor_name: Nombre del descriptor
            processed_filenames: Lista de archivos ya procesados
            features_data: Lista de características extraídas hasta ahora
            descriptor_config: Configuración del descriptor
            
        Returns:
            bool: True si se guardó exitosamente
        """
        if not config.ENABLE_PARTIAL_CHECKPOINTS:
            return True
            
        try:
            checkpoint_path = self._get_partial_checkpoint_path(descriptor_name)
            
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            
            checkpoint_data = {
                'descriptor_name': descriptor_name,
                'timestamp': datetime.now().isoformat(),
                'processed_filenames': processed_filenames,
                'total_processed': len(processed_filenames),
                'config_hash': self._get_config_hash(descriptor_name, descriptor_config),
                'features_count': len(features_data)
            }
            
            # Guardar checkpoint
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            # Guardar características parciales
            if features_data:
                partial_features_path = checkpoint_path.replace('.json', '_features.csv')
                features_df = pd.DataFrame(features_data)
                features_df.to_csv(partial_features_path, index=False)
            
            self.logger.debug(f"Checkpoint parcial guardado: {len(processed_filenames)} imágenes")
            return True
            
        except Exception as e:
            self.logger.warning(f"Error guardando checkpoint parcial: {e}")
            return False
    
    def _load_partial_checkpoint(self, descriptor_name: str, 
                                descriptor_config: Dict) -> Optional[Dict[str, Any]]:
        """
        Carga checkpoint parcial de un descriptor.
        
        Args:
            descriptor_name: Nombre del descriptor
            descriptor_config: Configuración actual del descriptor
            
        Returns:
            Optional[Dict]: Información del checkpoint o None si no existe/válido
        """
        if not config.ENABLE_PARTIAL_CHECKPOINTS:
            return None
            
        try:
            checkpoint_path = self._get_partial_checkpoint_path(descriptor_name)
            
            if not os.path.exists(checkpoint_path):
                return None
            
            # Cargar checkpoint
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            # Verificar que la configuración no haya cambiado
            current_hash = self._get_config_hash(descriptor_name, descriptor_config)
            if checkpoint_data.get('config_hash') != current_hash:
                self.logger.info(f"Configuración de {descriptor_name} cambió, "
                               f"descartando checkpoint parcial")
                self._clean_partial_checkpoint(descriptor_name)
                return None
            
            # Cargar características parciales si existen
            partial_features_path = checkpoint_path.replace('.json', '_features.csv')
            features_data = []
            if os.path.exists(partial_features_path):
                features_df = pd.read_csv(partial_features_path)
                features_data = features_df.to_dict('records')
            
            checkpoint_data['features_data'] = features_data
            
            self.logger.info(f"Checkpoint parcial cargado: {len(checkpoint_data['processed_filenames'])} imágenes")
            return checkpoint_data
            
        except Exception as e:
            self.logger.warning(f"Error cargando checkpoint parcial: {e}")
            self._clean_partial_checkpoint(descriptor_name)
            return None
    
    def _clean_partial_checkpoint(self, descriptor_name: str) -> bool:
        """
        Limpia checkpoint parcial de un descriptor.
        
        Args:
            descriptor_name: Nombre del descriptor
            
        Returns:
            bool: True si se limpió exitosamente
        """
        try:
            checkpoint_path = self._get_partial_checkpoint_path(descriptor_name)
            partial_features_path = checkpoint_path.replace('.json', '_features.csv')
            
            for path in [checkpoint_path, partial_features_path]:
                if os.path.exists(path):
                    os.remove(path)
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Error limpiando checkpoint parcial: {e}")
            return False
    
    def stop(self):
        """Detiene la ejecución del pipeline."""
        self.logger.info("Deteniendo pipeline modular...")
        self._should_stop = True