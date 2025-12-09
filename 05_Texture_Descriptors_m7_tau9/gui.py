"""
Interfaz gráfica para el módulo de extracción de descriptores de textura.

Esta GUI proporciona:
- Monitoreo en tiempo real del progreso
- Configuración visual de descriptores
- Visualización de resultados
- Control de ejecución (iniciar/pausar/reanudar)

Ejecutar con: streamlit run gui.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import os
import time
import json
from typing import Dict, List
import threading

# Importar módulos del proyecto
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from descriptors import list_available_descriptors, get_descriptor_info
from core.modular_pipeline import ModularPipeline
from utils.progress_file import ProgressFileManager
import config

# Variables globales para comunicación entre threads
extraction_results = None
extraction_error = None
extraction_running = False

# Gestor de progreso basado en archivos
progress_manager = ProgressFileManager()

# Configuración de la página
st.set_page_config(
    page_title="Extractor de Descriptores de Textura",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estado global de la aplicación
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'error' not in st.session_state:
    st.session_state.error = None


def main():
    """Función principal de la interfaz gráfica."""
    global progress_manager
    
    # Título y descripción
    st.title("🔬 Extractor de Descriptores de Textura")
    st.markdown("""
    Esta herramienta analiza los Recurrence Plots generados y extrae características 
    de textura para la clasificación de voces normales y patológicas.
    """)
    
    # Cargar configuración guardada si existe
    saved_config = progress_manager.load_config()
    
    # Sidebar para configuración
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Selector de descriptores
        st.subheader("Descriptores Disponibles")
        available_descriptors = list_available_descriptors()
        
        selected_descriptors = []
        
        # Obtener descriptores guardados si existen
        saved_descriptors = saved_config.get('descriptors', []) if saved_config else []
        
        for desc_name in available_descriptors:
            desc_info = get_descriptor_info(desc_name)
            
            # Checkbox para activar/desactivar
            col1, col2 = st.columns([3, 1])
            with col1:
                # Usar valor guardado si existe, sino usar valor por defecto
                default_enabled = desc_name in saved_descriptors if saved_descriptors else desc_info['enabled_by_default']
                is_enabled = st.checkbox(
                    desc_name.upper(),
                    value=default_enabled,
                    help=desc_info['description']
                )
            
            with col2:
                show_info = st.button("ℹ️", key=f"info_{desc_name}", help="Ver información del descriptor")
            
            if is_enabled:
                selected_descriptors.append(desc_name)
            
            # Mostrar información si se solicitó (fuera del if is_enabled)
            if show_info:
                with st.expander(f"Información de {desc_name}", expanded=True):
                    show_descriptor_info(desc_name, desc_info)
        
        st.divider()
        
        # Configuración de procesamiento
        st.subheader("Procesamiento")
        
        batch_size = st.number_input(
            "Tamaño de lote",
            min_value=1,
            max_value=100,
            value=saved_config.get('batch_size', 10) if saved_config else 10,
            help="Número de imágenes a procesar antes de guardar checkpoint"
        )
        
        n_jobs = st.number_input(
            "Trabajos paralelos",
            min_value=1,
            max_value=os.cpu_count(),
            value=saved_config.get('n_jobs', os.cpu_count() - 1) if saved_config else os.cpu_count() - 1,
            help="Número de cores a utilizar"
        )
        
        enable_checkpoints = st.checkbox(
            "Habilitar checkpoints",
            value=True,
            help="Permite reanudar el procesamiento si se interrumpe"
        )
        
        # Botón para limpiar checkpoints
        if st.button("🗑️ Limpiar Checkpoints", help="Elimina checkpoints parciales de TODOS los descriptores (no solo los seleccionados)"):
            try:
                # 1. Limpiar archivos de progreso de GUI
                progress_manager.clear_progress()
                
                # 2. Limpiar checkpoints parciales modulares DE TODOS los descriptores
                features_by_desc_dir = os.path.join(config.OUTPUT_FEATURES_DIR, 'by_descriptor')
                checkpoints_removed = 0
                descriptors_cleaned = []
                
                if os.path.exists(features_by_desc_dir):
                    for desc_dir in os.listdir(features_by_desc_dir):
                        checkpoint_path = os.path.join(features_by_desc_dir, desc_dir, 'partial_checkpoint.json')
                        features_path = os.path.join(features_by_desc_dir, desc_dir, 'partial_checkpoint_features.csv')
                        
                        desc_had_checkpoints = False
                        for path in [checkpoint_path, features_path]:
                            if os.path.exists(path):
                                os.remove(path)
                                checkpoints_removed += 1
                                desc_had_checkpoints = True
                        
                        if desc_had_checkpoints:
                            descriptors_cleaned.append(desc_dir)
                
                if checkpoints_removed > 0:
                    st.success(f"✅ {checkpoints_removed} archivos de checkpoint eliminados.")
                    st.info(f"📊 Descriptores limpiados: {', '.join(descriptors_cleaned)}")
                else:
                    st.info("No había checkpoints parciales para eliminar.")
                    
                st.info("💡 Los descriptores calculados se mantienen. Para eliminarlos usa '🗑️ Limpiar Todo'.")
            except Exception as e:
                st.error(f"Error limpiando checkpoints: {str(e)}")
        
        # Estado para el botón de limpiar todo
        if 'confirm_delete_all' not in st.session_state:
            st.session_state.confirm_delete_all = False
        
        # Botón para limpiar TODO (más agresivo)
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🗑️ Limpiar Todo", help="⚠️ PRECAUCIÓN: Elimina TODO - checkpoints Y descriptores calculados"):
                st.session_state.confirm_delete_all = True
        
        # Mostrar confirmación si se presionó el botón
        if st.session_state.confirm_delete_all:
            st.warning("⚠️ **CONFIRMACIÓN REQUERIDA**")
            st.markdown("Esto eliminará **TODOS** los descriptores calculados y checkpoints.")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                if st.button("✅ Sí, eliminar TODO", type="primary"):
                    import shutil
                    try:
                        # 1. Limpiar todo el directorio de características
                        if os.path.exists(config.OUTPUT_FEATURES_DIR):
                            shutil.rmtree(config.OUTPUT_FEATURES_DIR)
                            os.makedirs(config.OUTPUT_FEATURES_DIR, exist_ok=True)
                        
                        # 2. Limpiar archivos de progreso
                        progress_manager.clear_progress()
                        
                        st.success("✅ TODO eliminado. El módulo está como recién instalado.")
                        st.warning("⚠️ Deberás recalcular todos los descriptores desde cero.")
                        st.session_state.confirm_delete_all = False
                    except Exception as e:
                        st.error(f"Error limpiando todo: {str(e)}")
            
            with col2:
                if st.button("❌ Cancelar"):
                    st.session_state.confirm_delete_all = False
                    st.rerun()
    
    # Área principal
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.metric("Descriptores Seleccionados", len(selected_descriptors))
    
    with col2:
        # Contar imágenes disponibles
        normal_count = count_images(config.RP_INPUT_NORMAL_DIR)
        pathol_count = count_images(config.RP_INPUT_PATHOL_DIR)
        st.metric("Imágenes a Procesar", normal_count + pathol_count)
    
    with col3:
        # Verificar si hay descriptores calculados para mostrar estado
        manifest_path = os.path.join(config.OUTPUT_FEATURES_DIR, 'manifest.json')
        has_descriptors = os.path.exists(manifest_path)
        desc_count = 0
        
        if has_descriptors:
            try:
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                    desc_count = len(manifest.get('descriptors', {}))
                    if desc_count > 0:
                        st.info(f"📊 {desc_count} descriptores disponibles")
            except:
                pass
    
    st.divider()
    
    # Verificar si hay una extracción en progreso (desde archivo)
    extraction_running = progress_manager.is_extraction_running()
    
    # También verificar el progreso actual para detectar si terminó
    current_progress = progress_manager.read_progress()
    if current_progress and current_progress.get('phase') == 'all_finished':
        # El proceso terminó, marcar como no activo y limpiar session_state si existe
        extraction_running = False
        if st.session_state.is_running:
            st.session_state.is_running = False
            st.session_state.results = {
                'computed': current_progress.get('computed', {}),
                'reused': current_progress.get('reused', {}),
                'failed': current_progress.get('failed', {}),
                'summary': current_progress.get('summary', {})
            }
        # Limpiar progreso para evitar estados inconsistentes
        progress_manager.clear_progress()
    
    # Verificación robusta: detectar y limpiar cualquier estado inconsistente
    if current_progress and not extraction_running and not st.session_state.is_running:
        phase = current_progress.get('phase', '')
        progress_timestamp = current_progress.get('timestamp', 0)
        time_elapsed = time.time() - progress_timestamp
        
        # Limpiar estados que no deberían persistir cuando no hay ejecución activa
        should_clear = False
        warning_msg = ""
        
        if phase == 'analyzing' and time_elapsed > 10:  # Reducido a 10 segundos
            should_clear = True
            warning_msg = "⚠️ Estado 'analyzing' persistente detectado, limpiando..."
        elif phase == 'computing' and time_elapsed > 60:  # 1 minuto para computing
            should_clear = True
            warning_msg = "⚠️ Estado 'computing' abandonado detectado, limpiando..."
        elif phase in ['completed', 'failed', 'reused'] and time_elapsed > 300:  # 5 minutos para estados finales
            should_clear = True
            warning_msg = "⚠️ Progreso antiguo detectado, limpiando..."
        
        if should_clear:
            st.warning(warning_msg)
            progress_manager.clear_progress()
            extraction_running = False
            current_progress = None
    
    # Verificación final: asegurar que todos los estados sean consistentes
    if current_progress and not st.session_state.is_running and not extraction_running:
        # Si hay progreso pero ningún indicador de ejecución activa, algo está mal
        phase = current_progress.get('phase', '')
        if phase in ['analyzing', 'computing']:
            st.warning("🔄 Limpiando estado de progreso inconsistente...")
            progress_manager.clear_progress()
            current_progress = None
    
    # Determinar si hay extracción activa (desde session_state o archivo)
    is_extraction_active = st.session_state.is_running or extraction_running
    
    # Explicación de controles
    with st.expander("ℹ️ Ayuda sobre los controles", expanded=False):
        st.markdown("""
        **▶️ Iniciar/Continuar**: 
        - Si hay checkpoints, continúa desde donde se interrumpió
        - Si no hay checkpoints, inicia una nueva extracción
        
        **🛑 Detener**: Interrumpe la extracción actual (se puede continuar después)
        
        **🗑️ Limpiar Checkpoints**: Elimina checkpoints para empezar completamente desde cero
        """)
    
    # Mostrar resumen de configuración que se aplicará
    if selected_descriptors and not is_extraction_active:
        with st.expander("📋 Configuración que se aplicará al ejecutar", expanded=False):
            st.markdown("### Descriptores seleccionados y sus parámetros:")
            
            for desc_name in selected_descriptors:
                st.markdown(f"**{desc_name.upper()}**")
                
                # Obtener configuración de config.py
                if desc_name in config.DEFAULT_DESCRIPTORS:
                    config_params = config.DEFAULT_DESCRIPTORS[desc_name].copy()
                    config_params.pop('enabled', None)
                    
                    # Formatear parámetros de forma compacta
                    param_strs = []
                    for key, value in config_params.items():
                        if isinstance(value, list) and len(value) > 3:
                            param_strs.append(f"`{key}`: {value[:3]}... ({len(value)} elementos)")
                        elif isinstance(value, float):
                            param_strs.append(f"`{key}`: {value:.2f}")
                        else:
                            param_strs.append(f"`{key}`: {value}")
                    
                    st.markdown("  " + ", ".join(param_strs))
                else:
                    st.markdown("  *Usando valores por defecto del descriptor*")
                
            st.divider()
            st.markdown(f"**Configuración de procesamiento:**")
            st.markdown(f"- Tamaño de lote: {batch_size}")
            st.markdown(f"- Trabajos paralelos: {n_jobs}")
            st.markdown(f"- Checkpoints: {'Habilitados' if enable_checkpoints else 'Deshabilitados'}")
    
    # Controles de ejecución
    col1, col2 = st.columns(2)
    
    # Determinar el texto del botón según si hay descriptores disponibles
    if has_descriptors and desc_count > 0:
        button_text = "▶️ Continuar/Actualizar"
        button_help = "Continuar con descriptores nuevos o actualizar existentes"
    else:
        button_text = "▶️ Iniciar"
        button_help = "Comenzar nueva extracción"
    
    with col1:
        if st.button(button_text, type="primary", disabled=is_extraction_active, 
                    help=button_help):
            if selected_descriptors:
                start_extraction(selected_descriptors, {}, batch_size, n_jobs)
            else:
                st.error("Selecciona al menos un descriptor")
    
    with col2:
        if st.button("🛑 Detener", disabled=not is_extraction_active,
                    help="Detener extracción actual"):
            stop_extraction()
    
    # Mostrar errores si existen
    if st.session_state.error:
        st.error(st.session_state.error)
        if st.button("Limpiar error"):
            st.session_state.error = None
            st.rerun()
    
    # La verificación de resultados ahora se hace dentro del bloque de procesamiento
    # para evitar perderlos en el auto-rerun
    
    # Si hay extracción en archivo pero no in session_state, sincronizar
    if extraction_running and not st.session_state.is_running:
        st.session_state.is_running = True
        st.info("🔄 Extracción en progreso detectada - reconectando al proceso activo")
    
    # Mostrar progreso si hay extracción activa (usando OR para evitar duplicación)
    if st.session_state.is_running or extraction_running:
        # Mostrar progreso básico mientras se ejecuta
        st.header("📊 Procesando...")
        
        # Leer progreso desde archivo
        try:
            # Intentar obtener progreso desde archivo
            file_progress = progress_manager.read_progress()
            
            if file_progress:
                # Verificar si hay información multi-descriptor
                is_multi_descriptor = ('total_descriptors' in file_progress and 
                                     file_progress.get('total_descriptors', 1) > 1)
                
                if is_multi_descriptor:
                    # === SECCIÓN MULTI-DESCRIPTOR ===
                    st.subheader("🔄 Progreso General de Descriptores")
                    
                    # Información de descriptores
                    total_desc = file_progress.get('total_descriptors', 1)
                    completed_desc = file_progress.get('descriptors_completed', 0)
                    reused_desc = file_progress.get('descriptors_reused', 0)
                    current_desc = file_progress.get('current_descriptor', 'N/A')
                    desc_index = file_progress.get('descriptor_index', 1)
                    phase = file_progress.get('phase', 'computing')
                    
                    # Barra de progreso de descriptores
                    desc_progress = (completed_desc + reused_desc) / total_desc if total_desc > 0 else 0
                    st.progress(desc_progress)
                    
                    # Métricas de descriptores
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Descriptor Actual", f"{desc_index}/{total_desc}")
                    with col2:
                        st.metric("Completados", completed_desc)
                    with col3:
                        st.metric("Reusados", reused_desc)
                    with col4:
                        phase_emoji = {
                            'analyzing': '🔍',
                            'computing': '⚙️',
                            'reused': '♻️',
                            'completed': '✅',
                            'failed': '❌',
                            'all_finished': '🎉'
                        }.get(phase, '🔄')
                        st.metric("Fase", f"{phase_emoji} {phase}")
                    
                    # Mostrar mensaje según la fase
                    if phase == 'all_finished':
                        if completed_desc > 0 and reused_desc > 0:
                            st.success(f"🎉 **¡Proceso Completado!** {completed_desc} descriptores calculados y {reused_desc} reutilizados.")
                        elif completed_desc > 0:
                            st.success(f"🎉 **¡Proceso Completado!** {completed_desc} descriptores calculados exitosamente.")
                        elif reused_desc > 0:
                            st.success(f"♻️ **¡Todos los descriptores ya estaban calculados!** {reused_desc} descriptores reutilizados.")
                        else:
                            st.success("🎉 **¡Proceso Completado!**")
                    elif current_desc and current_desc != 'None':
                        st.markdown(f"**Procesando:** `{current_desc.upper()}`")
                    else:
                        st.markdown("**Estado:** Inicializando...")
                    
                    # === PROGRESO DEL DESCRIPTOR ACTUAL ===
                    if phase == 'computing':
                        st.subheader(f"📊 Progreso de {current_desc.upper()}")
                        
                        desc_progress = file_progress.get('descriptor_progress', 0)
                        desc_total = file_progress.get('descriptor_total', 100)
                        
                        # Barra de progreso del descriptor actual
                        if desc_total > 0:
                            desc_progress_pct = desc_progress / desc_total
                            st.progress(desc_progress_pct)
                        else:
                            st.progress(0)
                        
                        # Métricas del descriptor actual
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Imágenes", f"{desc_progress}/{desc_total}")
                        with col2:
                            percentage = (desc_progress/desc_total*100) if desc_total > 0 else 0
                            st.metric("Porcentaje", f"{percentage:.1f}%")
                        with col3:
                            st.metric("Velocidad", f"{file_progress.get('speed', 0):.1f} img/s")
                        
                        # Archivo actual siendo procesado
                        current_file = file_progress.get('current_file', '')
                        if current_file:
                            st.caption(f"📄 Procesando: {current_file}")
                else:
                    # === MODO COMPATIBILIDAD SIMPLE ===
                    progress = file_progress.get('progress', 0)
                    total = file_progress.get('total', 100)
                    
                    # Barra de progreso simple
                    if total > 0:
                        progress_pct = progress / total
                        st.progress(progress_pct)
                    else:
                        st.progress(0)
                    
                    # Métricas simples
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Imágenes Procesadas", f"{progress}/{total}")
                    with col2:
                        st.metric("Porcentaje", f"{(progress/total*100):.1f}%" if total > 0 else "0.0%")
                    with col3:
                        st.metric("Velocidad", f"{file_progress.get('speed', 0):.1f} img/s")
                
                # Información común
                if 'last_update' in file_progress:
                    st.caption(f"⏰ Última actualización: {file_progress['last_update']}")
                
                # Debug info (solo en desarrollo)
                with st.expander("🐛 Información de debug", expanded=False):
                    st.json(file_progress)
            else:
                st.info("Iniciando procesamiento... (esperando primera actualización)")
        except Exception as e:
            st.error(f"Error mostrando progreso: {str(e)}")
            import traceback
            st.text(traceback.format_exc())
        
        # *** VERIFICAR RESULTADOS DE MÚLTIPLES FORMAS ***
        
        # 1. Verificar variables globales
        global extraction_results, extraction_error
        st.write(f"DEBUG GLOBAL: extraction_results={extraction_results is not None}, extraction_error={extraction_error is not None}")
        
        # 2. Verificar estado del progreso manager
        import time as time_module
        current_time = time_module.time()
        status_data = progress_manager.read_status()
        if status_data:
            status_timestamp = status_data.get('timestamp', 0)
            age = current_time - status_timestamp
            st.write(f"DEBUG STATUS: {status_data.get('status')} - timestamp={status_timestamp:.1f} - age={age:.1f}s")
            
            if status_data.get('status') == 'completed':
                if status_data.get('results'):
                    st.success("🎉 Resultados encontrados en status! Transfiriendo...")
                    st.session_state.results = status_data['results']
                    st.session_state.is_running = False
                    progress_manager.clear_progress()
                    st.rerun()
                else:
                    st.warning("Status es 'completed' pero no hay 'results' en el status")
        else:
            st.write("DEBUG STATUS: No status data found")
        
        if extraction_results is not None:
            st.success("🎉 Resultados encontrados en variables globales! Transfiriendo...")
            st.session_state.results = extraction_results
            extraction_results = None
            st.session_state.is_running = False
            st.rerun()
        
        if extraction_error is not None:
            st.session_state.error = extraction_error
            extraction_error = None
            st.session_state.is_running = False
            st.rerun()
        
        # Auto-actualizar cada 2 segundos
        time.sleep(2)
        st.rerun()
    
    elif st.session_state.results:
        # Mostrar resumen de resultados primero
        show_results_summary_brief()
        
        show_results_dashboard()
        
        # Mostrar estado modular y opciones de combinación
        show_modular_status()


def show_modular_status():
    """Muestra el estado modular de los descriptores y opciones de combinación."""
    st.header("🧩 Estado Modular de Descriptores")
    
    # Crear una instancia temporal del pipeline para acceder al manifest
    temp_pipeline = ModularPipeline(descriptors=[], n_jobs=1, enable_checkpoints=False)
    available_descriptors = temp_pipeline.get_available_descriptors()
    
    if not available_descriptors:
        st.info("No hay descriptores calculados aún")
        return
    
    # Mostrar tabla de descriptores calculados
    st.subheader("📊 Descriptores Disponibles")
    
    desc_data = []
    for desc_name, desc_info in available_descriptors.items():
        desc_data.append({
            "Descriptor": desc_name,
            "Imágenes": desc_info['total_images'],
            "Características": desc_info['feature_count'],
            "Fecha Cálculo": desc_info['computed_date'][:10],
            "Config Hash": desc_info['config_hash']
        })
    
    df_descriptors = pd.DataFrame(desc_data)
    st.dataframe(df_descriptors, use_container_width=True)
    
    # Verificar inconsistencias en el número de imágenes
    if len(desc_data) > 1:
        image_counts = [d['Imágenes'] for d in desc_data]
        max_images = max(image_counts)
        min_images = min(image_counts)
        
        if max_images != min_images:
            st.warning(f"⚠️ **Inconsistencia detectada**: Los descriptores tienen diferente número de imágenes ({min_images}-{max_images})")
            
            # Mostrar qué descriptores necesitan actualización
            outdated_descriptors = [d['Descriptor'] for d in desc_data if d['Imágenes'] < max_images]
            if outdated_descriptors:
                st.write(f"**Descriptores desactualizados:** {', '.join(outdated_descriptors)}")
                st.write(f"**Solución:** Usar 'Limpiar Descriptores' para los desactualizados y recalcularlos, o usar 'Continuar' para actualizar automáticamente.")
    
    # Información adicional
    if desc_data:
        st.info(f"💡 **Total de imágenes actuales disponibles:** 440 (239 Normal + 201 Pathol)")
    
    # Sección de combinación
    st.subheader("🔗 Combinar Características")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_to_combine = st.multiselect(
            "Selecciona descriptores para combinar:",
            options=list(available_descriptors.keys()),
            default=list(available_descriptors.keys()),
            help="Puedes crear diferentes combinaciones de características"
        )
    
    with col2:
        if st.button("🔀 Combinar Seleccionados", type="primary", disabled=len(selected_to_combine) < 2):
            if selected_to_combine:
                with st.spinner("Combinando características..."):
                    # Crear nombre descriptivo
                    output_name = f"combined_{'_'.join(selected_to_combine)}"
                    
                    # Combinar
                    combined_df = temp_pipeline.combine_features(selected_to_combine, output_name)
                    
                    if combined_df is not None:
                        st.success(f"✅ Combinación exitosa: {len(combined_df)} imágenes, {combined_df.shape[1]-2} características")
                        
                        # Mostrar preview
                        with st.expander("Vista previa del resultado"):
                            st.dataframe(combined_df.head())
                    else:
                        st.error("Error al combinar características")
    
    # Opciones avanzadas
    with st.expander("⚙️ Opciones Avanzadas"):
        st.markdown("### Limpiar Descriptores")
        st.warning("⚠️ Esta acción eliminará los datos calculados del descriptor seleccionado")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            desc_to_clean = st.selectbox(
                "Descriptor a limpiar:",
                options=list(available_descriptors.keys())
            )
        
        with col2:
            if st.button("🗑️ Limpiar", type="secondary"):
                if temp_pipeline.clean_descriptor(desc_to_clean):
                    st.success(f"✅ {desc_to_clean} eliminado")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"Error al limpiar {desc_to_clean}")


def show_descriptor_info(desc_name: str, desc_info: dict):
    """
    Muestra información detallada sobre un descriptor específico.
    
    Args:
        desc_name: Nombre del descriptor
        desc_info: Información del descriptor
    """
    # Información básica
    st.markdown(f"**Clase:** `{desc_info['class']}`")
    st.markdown(f"**Descripción:** {desc_info['description']}")
    
    # Configuración que se usará (sistema híbrido)
    st.subheader("Configuración que se aplicará")
    
    # Obtener configuración real de config.py si existe
    config_values = {}
    if desc_name in config.DEFAULT_DESCRIPTORS:
        config_values = config.DEFAULT_DESCRIPTORS[desc_name].copy()
        config_values.pop('enabled', None)  # Quitar el campo 'enabled'
    
    # Obtener parámetros del constructor como referencia
    params = desc_info.get('parameters', {})
    
    if config_values or params:
        # Mostrar configuración real que se usará
        param_data = []
        
        # Primero, mostrar valores de config.py
        for param_name, value in config_values.items():
            param_type = 'Configurado'
            
            # Formatear el valor de manera legible
            if isinstance(value, list):
                if len(value) > 5:
                    formatted_value = f"Lista de {len(value)} elementos: {value[:3]}..."
                else:
                    formatted_value = str(value)
            elif isinstance(value, (dict, tuple)):
                formatted_value = str(value)
            elif isinstance(value, float):
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            
            param_data.append({
                "Parámetro": param_name,
                "Valor que se usará": formatted_value,
                "Origen": "config.py"
            })
        
        # Luego, mostrar parámetros del constructor que no estén en config
        for param_name, param_info in params.items():
            if param_name not in config_values:
                default_value = param_info.get('default')
                
                # Formatear el valor por defecto
                if isinstance(default_value, list):
                    if len(default_value) > 5:
                        formatted_value = f"Lista de {len(default_value)} elementos"
                    else:
                        formatted_value = str(default_value)
                elif default_value is None:
                    formatted_value = "None"
                else:
                    formatted_value = str(default_value)
                
                param_data.append({
                    "Parámetro": param_name,
                    "Valor que se usará": formatted_value,
                    "Origen": "default del constructor"
                })
        
        # Mostrar como tabla
        df = pd.DataFrame(param_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Este descriptor no tiene parámetros configurables.")
    
    # Mostrar explicación del sistema híbrido
    with st.expander("ℹ️ Sistema de Configuración Híbrido"):
        st.markdown("""
        **¿Cómo funciona la configuración?**
        
        1. **Prioridad**: Los valores en `config.py` tienen prioridad sobre los defaults del constructor
        2. **Origen config.py**: Valores optimizados específicamente para Recurrence Plots
        3. **Origen constructor**: Valores por defecto cuando no hay configuración específica
        
        **¿Por qué sistema híbrido?**
        - Permite configuración centralizada en `config.py`
        - Mantiene defaults sensatos en los constructores
        - Facilita el uso independiente de los descriptores
        """)
    
    # Información adicional específica del descriptor
    if desc_name == 'glcm':
        st.info("📊 GLCM extrae propiedades de textura basadas en co-ocurrencia de niveles de gris")
    elif desc_name == 'lbp':
        st.info("🔵 LBP detecta patrones locales binarios para caracterizar texturas")
    elif desc_name == 'statistical':
        st.info("📈 Statistical calcula momentos estadísticos y distribuciones")
    elif desc_name == 'gabor':
        st.info("🌊 Gabor analiza frecuencias y orientaciones específicas")
    elif desc_name == 'wavelet':
        st.info("🌀 Wavelet descompone la imagen en múltiples escalas y frecuencias")


def count_images(directory: str) -> int:
    """Cuenta las imágenes en un directorio."""
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory) if f.endswith(config.IMAGE_FORMAT)])


def start_extraction(descriptors: List[str], configs: Dict, batch_size: int, n_jobs: int):
    """Inicia el proceso de extracción."""
    # Limpiar estado global antes de iniciar
    global extraction_results, extraction_error, extraction_running
    extraction_results = None
    extraction_error = None
    extraction_running = False
    
    # Limpiar progreso anterior y resultados para evitar estados inconsistentes
    st.session_state.results = None
    st.session_state.error = None
    progress_manager.clear_progress()
    
    st.session_state.is_running = True
    
    # Guardar configuración para persistencia
    progress_manager.save_config({
        'descriptors': descriptors,
        'batch_size': batch_size,
        'n_jobs': n_jobs
    })
    
    # Usar configuraciones por defecto en lugar de las de la GUI
    # Las configuraciones ya están optimizadas en config.py
    descriptor_configs = {}
    for desc_name in descriptors:
        if desc_name in config.DEFAULT_DESCRIPTORS:
            # Usar configuración por defecto, excluyendo 'enabled'
            desc_config = config.DEFAULT_DESCRIPTORS[desc_name].copy()
            desc_config.pop('enabled', None)
            descriptor_configs[desc_name] = desc_config
    
    # Crear pipeline
    pipeline_config = {
        'descriptors': descriptors,
        'descriptor_configs': descriptor_configs,
        'batch_size': batch_size,
        'n_jobs': n_jobs,
        'progress_callback': update_progress
    }
    
    st.session_state.pipeline = ModularPipeline(**pipeline_config)
    
    # Iniciar en thread separado, pasando el pipeline como argumento
    thread = threading.Thread(target=run_extraction, args=(st.session_state.pipeline,))
    thread.start()
    
    st.success("✅ Extracción iniciada con configuraciones optimizadas")
    st.rerun()


def run_extraction(pipeline):
    """
    Ejecuta la extracción en un thread separado.
    
    Args:
        pipeline: Instancia del pipeline a ejecutar
    """
    try:
        progress_manager.update_status('running')
        
        # Usar el método modular compute_descriptors
        results = pipeline.compute_descriptors()
        
        # Guardar resultados
        global extraction_results, extraction_error
        extraction_results = results
        extraction_error = None
        print(f"DEBUG THREAD: Resultados guardados - {len(results.get('computed', {}))} computed, {len(results.get('reused', {}))} reused")
        
        # NOTA: No podemos modificar session_state desde thread sin contexto
        # Los resultados se guardan en variables globales y luego la GUI principal los lee
        
        # Verificar si el pipeline fue detenido o completado
        print(f"DEBUG THREAD: pipeline._should_stop = {pipeline._should_stop}")
        if pipeline._should_stop:
            # El pipeline fue detenido intencionalmente
            print("DEBUG THREAD: Pipeline fue detenido, status = ready")
            progress_manager.update_status('ready')  # Listo para continuar
        else:
            # El pipeline terminó normalmente
            print("DEBUG THREAD: Pipeline terminó normal, status = completed")
            progress_manager.update_status('completed', results=results)
        
    except Exception as e:
        error_msg = f"Error durante la extracción: {str(e)}"
        import traceback
        traceback.print_exc()
        
        # Guardar el error
        extraction_results = None
        extraction_error = error_msg
        
        # NOTA: No podemos modificar session_state desde thread sin contexto
        # Los errores se guardan en variables globales
        
        # Actualizar estado a error
        progress_manager.update_status('error', error=error_msg)
        
    finally:
        # Marcar como no en ejecución y limpiar estado global
        global extraction_running
        extraction_running = False
        # NOTA: No podemos modificar session_state desde thread
        
        # Limpiar progreso solo después de que sea seguro hacerlo
        # IMPORTANTE: Darle tiempo a la GUI para leer el status 'completed'
        print("DEBUG THREAD: Esperando 5 segundos antes de limpiar progreso...")
        time.sleep(5)
        print("DEBUG THREAD: Limpiando progreso ahora")
        progress_manager.clear_progress()


def update_progress(progress_info: dict):
    """Callback para actualizar el progreso."""
    # Actualizar progreso en el archivo para persistencia
    progress_manager.update_progress(progress_info)
    
    # NOTA: No marcamos como completado aquí automáticamente.
    # Dejamos que run_extraction() maneje el estado final para evitar condiciones de carrera.


def stop_extraction():
    """Detiene completamente la extracción."""
    global progress_manager
    if st.session_state.pipeline:
        st.session_state.pipeline.stop()
        st.session_state.is_running = False
        
        # Actualizar estado en archivo para que sea persistente
        # Usar 'ready' en lugar de 'stopped' para indicar que está listo para continuar
        progress_manager.update_status('ready')
        
        st.warning("🛑 Extracción detenida - Puedes continuar después")
        st.rerun()


def show_monitoring_dashboard():
    """Muestra el dashboard de monitoreo en tiempo real."""
    st.header("📊 Monitoreo en Tiempo Real")
    
    # Contenedores para actualización dinámica
    progress_container = st.container()
    metrics_container = st.container()
    chart_container = st.container()
    
    # Placeholder para actualización en tiempo real
    with progress_container:
        progress_placeholder = st.empty()
    
    with metrics_container:
        col1, col2, col3, col4 = st.columns(4)
        metric1 = col1.empty()
        metric2 = col2.empty()
        metric3 = col3.empty()
        metric4 = col4.empty()
    
    with chart_container:
        chart_placeholder = st.empty()
    
    # Actualizar mientras esté corriendo
    while st.session_state.is_running:
        try:
            # Obtener última actualización de progreso
            # progress_info = progress_queue.get_nowait()  # TODO: Actualizar para pipeline modular
            progress_info = progress_manager.read_progress() or {}
            
            # Actualizar barra de progreso
            with progress_placeholder.container():
                progress = progress_info.get('progress', 0)
                total = progress_info.get('total', 100)
                st.progress(progress / total)
                st.text(f"Procesando: {progress}/{total} imágenes")
            
            # Actualizar métricas
            metric1.metric("Imágenes Procesadas", progress)
            metric2.metric("Velocidad", f"{progress_info.get('speed', 0):.1f} img/s")
            metric3.metric("Tiempo Restante", format_time(progress_info.get('eta', 0)))
            metric4.metric("Memoria Usada", f"{progress_info.get('memory', 0):.1f} MB")
            
            # Actualizar gráfico de progreso
            if 'history' in progress_info:
                df = pd.DataFrame(progress_info['history'])
                fig = px.line(df, x='time', y='processed', 
                             title='Progreso de Procesamiento')
                chart_placeholder.plotly_chart(fig, use_container_width=True)
        
        except queue.Empty:
            pass
        
        time.sleep(0.5)  # Actualizar cada 500ms


def show_results_summary_brief():
    """Muestra un resumen breve del proceso completado."""
    results = st.session_state.results
    if not results:
        return
    
    # Obtener información del ModularPipeline si disponible
    computed = results.get('computed', {})
    reused = results.get('reused', {})
    failed = results.get('failed', {})
    summary = results.get('summary', {})
    
    total_computed = summary.get('total_computed', 0)
    total_reused = summary.get('total_reused', 0) 
    total_failed = summary.get('total_failed', 0)
    
    # Determinar mensaje apropiado
    if total_computed == 0 and total_reused > 0 and total_failed == 0:
        # Todos fueron reutilizados
        st.success(
            f"🎉 **¡Proceso Completado!** Todos los descriptores ({total_reused}) "
            f"ya estaban calculados y fueron reutilizados exitosamente."
        )
    elif total_computed > 0 and total_reused == 0 and total_failed == 0:
        # Todos fueron calculados
        st.success(
            f"🎉 **¡Proceso Completado!** Se calcularon {total_computed} "
            f"descriptor{'es' if total_computed > 1 else ''} exitosamente."
        )
    elif total_computed > 0 and total_reused > 0 and total_failed == 0:
        # Combinación de calculados y reutilizados
        st.success(
            f"🎉 **¡Proceso Completado!** {total_computed} descriptor{'es' if total_computed > 1 else ''} "
            f"calculado{'s' if total_computed > 1 else ''} y {total_reused} reutilizado{'s' if total_reused > 1 else ''}."
        )
    elif total_failed > 0:
        # Hubo errores
        if total_computed + total_reused > 0:
            st.warning(
                f"⚠️ **Proceso Completado con Errores.** {total_computed + total_reused} descriptor{'es' if total_computed + total_reused > 1 else ''} "
                f"procesado{'s' if total_computed + total_reused > 1 else ''} exitosamente, {total_failed} falló."
            )
        else:
            st.error(f"❌ **Proceso Falló.** {total_failed} descriptor{'es' if total_failed > 1 else ''} no pudo ser procesado.")
    
    # Mostrar detalles adicionales si es útil
    if total_computed > 0 or total_reused > 0:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Calculados", total_computed)
            
        with col2:
            st.metric("Reutilizados", total_reused)
            
        with col3:
            st.metric("Errores", total_failed)


def show_results_dashboard():
    """Muestra el dashboard de resultados."""
    st.header("📈 Resultados del Análisis")
    
    if not st.session_state.results:
        st.warning("No hay resultados disponibles")
        return
    
    # Tabs simplificados - análisis detallado se hace en módulo 06
    tab1, tab2 = st.tabs([
        "Resumen", "Exportar"
    ])
    
    with tab1:
        show_results_summary()
    
    with tab2:
        show_export_options()


def show_results_summary():
    """Muestra resumen de resultados."""
    results = st.session_state.results
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Estadísticas Generales")
        
        # Adaptado para ModularPipeline results format
        if 'computed' in results or 'reused' in results:
            # ModularPipeline format
            computed = results.get('computed', {})
            reused = results.get('reused', {})
            summary = results.get('summary', {})
            
            # Obtener info de un descriptor para contar imágenes
            total_images = 0
            total_features = 0
            
            # Usar ModularPipeline para obtener datos actuales
            try:
                temp_pipeline = ModularPipeline(descriptors=[], n_jobs=1, enable_checkpoints=False)
                available_descriptors = temp_pipeline.get_available_descriptors()
                
                if available_descriptors:
                    # Sumar imágenes (usar el máximo entre descriptores)
                    total_images = max(desc_info['total_images'] for desc_info in available_descriptors.values())
                    # Sumar características de todos los descriptores
                    total_features = sum(desc_info['feature_count'] for desc_info in available_descriptors.values())
            except:
                pass
                
            st.metric("Descriptores procesados", 
                     summary.get('total_computed', 0) + summary.get('total_reused', 0))
            st.metric("Total de imágenes procesadas", total_images)
            st.metric("Total de características extraídas", total_features)
            
        else:
            # Formato original
            st.metric("Total de imágenes procesadas", results.get('total_images', 0))
            st.metric("Total de características extraídas", results.get('total_features', 0))
            st.metric("Tiempo total de procesamiento", format_time(results.get('total_time', 0)))
    
    with col2:
        st.subheader("Distribución por Descriptores")
        
        # Mostrar distribución de descriptores en lugar de clases
        if 'computed' in results or 'reused' in results:
            computed = len(results.get('computed', {}))
            reused = len(results.get('reused', {}))
            failed = len(results.get('failed', {}))
            
            if computed + reused + failed > 0:
                fig = px.pie(
                    values=[computed, reused, failed] if failed > 0 else [computed, reused],
                    names=['Calculados', 'Reutilizados', 'Fallidos'] if failed > 0 else ['Calculados', 'Reutilizados'],
                    title="Estado de Descriptores",
                    color_discrete_sequence=['#2E8B57', '#4682B4', '#DC143C']
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            # Formato original
            class_dist = results.get('class_distribution', {})
            if class_dist:
                fig = px.pie(
                    values=list(class_dist.values()),
                    names=list(class_dist.keys()),
                    title="Distribución de Clases"
                )
                st.plotly_chart(fig, use_container_width=True)


# Funciones eliminadas - análisis detallado movido al módulo 06
# def show_feature_distributions() - ELIMINADA
# def show_feature_correlations() - ELIMINADA


def show_export_options():
    """Muestra opciones de exportación."""
    st.subheader("Opciones de Exportación")
    
    st.info("💡 **Análisis Detallado:** Las distribuciones, correlaciones y análisis estadísticos avanzados se realizan en el **Módulo 06 - Feature Analysis**.")
    
    st.subheader("Archivos Disponibles")
    
    # Mostrar dónde están los archivos generados
    st.write("Las características extraídas están disponibles en:")
    st.code("05_Texture_Descriptors/output/features/", language="bash")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📁 Por Descriptor Individual:**")
        st.code("""
by_descriptor/
├── glcm/features.csv
└── lbp/features.csv
        """)
    
    with col2:
        st.write("**🔗 Archivos Combinados:**")
        st.code("""
combined/
└── combined_features_*.csv
        """)


def format_time(seconds: float) -> str:
    """Formatea segundos a formato legible."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.0f}m {seconds%60:.0f}s"
    else:
        return f"{seconds/3600:.0f}h {(seconds%3600)/60:.0f}m"


if __name__ == "__main__":
    main()