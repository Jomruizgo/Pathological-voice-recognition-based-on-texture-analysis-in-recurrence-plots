#02_Audio_Preprocess/processor.py
# Dentro de load_and_preprocess_audio_with_sad en 02_Audio_Preprocess/processor.py
import os
import numpy as np
import librosa
import soundfile as sf
import warnings
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Importar el config del mismo módulo
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import config
sys.path.pop(0)

def load_and_preprocess_audio_with_sad(file_path, plot_output_dir=None):
    """
    Carga un archivo de audio, lo convierte a la frecuencia de muestreo deseada,
    lo convierte a mono. Siempre busca un segmento de MAX_AUDIO_DURATION_SECONDS.
    Prioriza el audio original (incluyendo silencios) antes de padear con ceros.
    Genera plots visuales si config.SAVE_PREPROCESS_PLOTS es True.
    """
    original_audio_data_for_plot = None # Para el primer plot (original)
    
    selected_segment_start_sample_plot = 0 # Inicio del área resaltada en el plot original
    selected_segment_end_sample_plot = 0   # Fin del área resaltada en el plot original

    try:
        # 1. Cargar audio original
        audio_data_raw, original_sr = librosa.load(file_path, sr=None, mono=False)
        original_audio_data_for_plot = audio_data_raw.copy() # Copia para el plot original
        
        # 2. Convertir a mono y remuestrear para procesamiento
        if audio_data_raw.ndim > 1:
            audio_data_mono = librosa.to_mono(audio_data_raw)
        else:
            audio_data_mono = audio_data_raw

        if original_sr != config.TARGET_SAMPLE_RATE:
            audio_data_processed = librosa.resample(y=audio_data_mono, orig_sr=original_sr, target_sr=config.TARGET_SAMPLE_RATE)
        else:
            audio_data_processed = audio_data_mono

        current_sr = config.TARGET_SAMPLE_RATE
        
        # Duración objetivo en muestras
        target_num_samples = int(config.MAX_AUDIO_DURATION_SECONDS * current_sr)
        
        # Duración total del audio original procesado en muestras
        original_audio_duration_samples = len(audio_data_processed)

        final_audio_segment = None # Este será el segmento final a devolver

        # --- Paso 1: Determinar el segmento activo más largo ---
        active_segments = librosa.effects.split(y=audio_data_processed, top_db=config.SAD_TOP_DB)
        
        longest_segment_interval = None
        longest_segment_length = 0

        if active_segments.shape[0] > 0:
            for start_s, end_s in active_segments:
                current_len = end_s - start_s
                if current_len > longest_segment_length:
                    longest_segment_length = current_len
                    longest_segment_interval = (start_s, end_s)
        
        # --- Paso 2: Seleccionar el audio final según las reglas de duración y actividad ---
        
        # Caso A: El audio original es más corto que la duración objetivo.
        if original_audio_duration_samples < target_num_samples:
            padding_needed = target_num_samples - original_audio_duration_samples
            final_audio_segment = np.pad(audio_data_processed, (0, padding_needed), 'constant')
            
            selected_segment_start_sample_plot = 0
            selected_segment_end_sample_plot = original_audio_duration_samples
            warnings.warn(f"Audio '{os.path.basename(file_path)}': Original ({original_audio_duration_samples/current_sr:.2f}s) es más corto que objetivo ({config.MAX_AUDIO_DURATION_SECONDS}s). Padeado con silencio.")
            
        # Caso B: El audio original es igual o más largo que la duración objetivo.
        else:
            if longest_segment_interval:
                start_sad, end_sad = longest_segment_interval
                
                # Si el segmento activo más largo es >= a la duración objetivo, aplicamos recorte inteligente
                if longest_segment_length >= target_num_samples:
                    # --- RECORTAR INTELIGENTEMENTE: Mantener la parte con más información ---
                    # El segmento activo es más largo de lo necesario.
                    # Necesitamos reducirlo a target_num_samples.
                    # Calculamos cuántas muestras debemos eliminar
                    samples_to_remove = longest_segment_length - target_num_samples

                    # Evaluar la "información" (energía) en los extremos
                    # Tomamos pequeñas ventanas al principio y al final para comparar
                    eval_window_samples = int(current_sr * 0.1) # Ventana de 0.1 segundos para evaluar

                    # Asegurarse de que las ventanas no excedan el segmento
                    if eval_window_samples * 2 > longest_segment_length:
                        # Si el segmento es muy corto para dos ventanas de evaluación,
                        # simplemente tomamos los primeros target_num_samples
                        # o podemos intentar centrarlo si es casi igual.
                        # Para simplificar, si es casi el tamaño, recortamos del final.
                        # Si no hay espacio para dos ventanas de 0.1s, significa que el exceso es pequeño.
                        # En este caso, simplemente recortamos el exceso del final.
                        final_audio_segment = audio_data_processed[start_sad : start_sad + target_num_samples]
                        selected_segment_start_sample_plot = start_sad
                        selected_segment_end_sample_plot = start_sad + target_num_samples
                        warnings.warn(f"Audio '{os.path.basename(file_path)}': Segmento activo ({longest_segment_length/current_sr:.2f}s) es ligeramente más largo. Recortado de forma simple al inicio para {config.MAX_AUDIO_DURATION_SECONDS}s.")
                    else:
                        # Calcular energía (RMS) de una pequeña porción al inicio y al final
                        energy_start = np.sum(audio_data_processed[start_sad : start_sad + eval_window_samples]**2)
                        energy_end = np.sum(audio_data_processed[end_sad - eval_window_samples : end_sad]**2)

                        new_start_offset = 0
                        new_end_offset = 0

                        # Decidir si quitar del inicio o del final
                        if energy_start < energy_end:
                            # Quitar del inicio (mover el inicio hacia adelante)
                            new_start_offset = min(samples_to_remove, longest_segment_length - eval_window_samples) # No quitar más allá del segmento
                            new_start_offset = min(new_start_offset, start_sad + longest_segment_length - target_num_samples) # No exceder lo necesario
                            
                            new_start = start_sad + new_start_offset
                            new_end = new_start + target_num_samples # Mantener la longitud deseada
                            # Ajuste de seguridad para no exceder el original
                            if new_end > end_sad:
                                new_end = end_sad
                                new_start = new_end - target_num_samples

                        else:
                            # Quitar del final (mover el final hacia atrás)
                            new_end_offset = min(samples_to_remove, longest_segment_length - eval_window_samples) # No quitar más allá del segmento
                            new_end_offset = min(new_end_offset, end_sad - start_sad - target_num_samples) # No exceder lo necesario

                            new_end = end_sad - new_end_offset
                            new_start = new_end - target_num_samples # Mantener la longitud deseada
                            # Ajuste de seguridad para no quedar por debajo de 0
                            if new_start < start_sad:
                                new_start = start_sad
                                new_end = new_start + target_num_samples
                        
                        # Si después de la decisión, la ventana no es exactamente target_num_samples
                        # (debido a los límites o redondeos), ajustamos para que sea exacta.
                        if (new_end - new_start) != target_num_samples:
                            final_audio_segment = audio_data_processed[start_sad : start_sad + target_num_samples]
                            selected_segment_start_sample_plot = start_sad
                            selected_segment_end_sample_plot = start_sad + target_num_samples
                            warnings.warn(f"Audio '{os.path.basename(file_path)}': Recorte inteligente complejo, se optó por recorte simple al inicio.")
                        else:
                            final_audio_segment = audio_data_processed[new_start : new_end]
                            selected_segment_start_sample_plot = new_start
                            selected_segment_end_sample_plot = new_end
                            warnings.warn(f"Audio '{os.path.basename(file_path)}': Segmento activo ({longest_segment_length/current_sr:.2f}s) recortado inteligentemente a {config.MAX_AUDIO_DURATION_SECONDS}s.")

                # Si el segmento activo más largo es más corto que la duración objetivo,
                # lo extendemos usando el audio original (incluyendo silencio)
                else: # longest_segment_length < target_num_samples
                    # ... (Esta parte del código no cambia de la última versión) ...
                    # Intentar centrar el segmento activo si es posible
                    needed_additional_samples = target_num_samples - longest_segment_length
                    
                    available_before = start_sad
                    available_after = original_audio_duration_samples - end_sad

                    take_before = min(available_before, needed_additional_samples // 2)
                    take_after = min(available_after, needed_additional_samples - take_before)
                    
                    # Ajustar si no se pudo tomar suficiente de un lado
                    if (take_before + take_after) < needed_additional_samples:
                        remaining_needed = needed_additional_samples - (take_before + take_after)
                        if available_before - take_before > 0:
                            take_before += min(available_before - take_before, remaining_needed)
                        # ¡CORRECCIÓN AQUÍ! Cambiado de needed_needed_additional_samples a needed_additional_samples
                        if (take_before + take_after) < needed_additional_samples and available_after - take_after > 0:
                            take_after += min(available_after - take_after, remaining_needed - (take_before + take_after))

                    new_start = max(0, start_sad - take_before)
                    new_end = min(original_audio_duration_samples, end_sad + take_after)
                    
                    # Asegurarse de que la longitud final sea target_num_samples si es posible
                    if (new_end - new_start) < target_num_samples and new_end < original_audio_duration_samples:
                        extend_more = target_num_samples - (new_end - new_start)
                        new_end = min(original_audio_duration_samples, new_end + extend_more)
                    
                    if (new_end - new_start) < target_num_samples and new_start > 0:
                        extend_more = target_num_samples - (new_end - new_start)
                        new_start = max(0, new_start - extend_more)

                    # Última comprobación para asegurar target_num_samples si hay material original
                    if (new_end - new_start) < target_num_samples and original_audio_duration_samples >= target_num_samples:
                        # Si por alguna razón sigue siendo corto pero el original es largo, tomamos desde el inicio
                        final_audio_segment = audio_data_processed[:target_num_samples]
                        selected_segment_start_sample_plot = 0
                        selected_segment_end_sample_plot = target_num_samples
                        warnings.warn(f"Audio '{os.path.basename(file_path)}': Segmento activo corto. Recortando primeros {config.MAX_AUDIO_DURATION_SECONDS}s del original.")
                    else:
                        final_audio_segment = audio_data_processed[new_start : new_end]
                        selected_segment_start_sample_plot = new_start
                        selected_segment_end_sample_plot = new_end
                        warnings.warn(f"Audio '{os.path.basename(file_path)}': Segmento activo ({longest_segment_length/current_sr:.2f}s) extendido a {len(final_audio_segment)/current_sr:.2f}s usando silencio del original.")

            else:
                # No se detectó actividad de sonido, pero el original es lo suficientemente largo.
                # Tomar los primeros MAX_AUDIO_DURATION_SECONDS del audio original.
                final_audio_segment = audio_data_processed[:target_num_samples]
                selected_segment_start_sample_plot = 0
                selected_segment_end_sample_plot = target_num_samples
                warnings.warn(f"Audio '{os.path.basename(file_path)}': No se detectó actividad. Tomando los primeros {config.MAX_AUDIO_DURATION_SECONDS}s del audio original.")

        # Asegurarse de que el audio final esté normalizado
        if final_audio_segment is not None:
            # Asegurar que el segmento final tenga exactamente la duración deseada (si el original lo permite)
            # Esto es una doble verificación para el caso donde el original es largo,
            # para asegurar que siempre sea MAX_AUDIO_DURATION_SECONDS
            if original_audio_duration_samples >= target_num_samples:
                if len(final_audio_segment) != target_num_samples:
                    # Esto debería ocurrir solo si las lógicas de recorte/extensión no fueron perfectas.
                    # Recortamos/ajustamos el final si es necesario.
                    final_audio_segment = final_audio_segment[:target_num_samples]
                    selected_segment_end_sample_plot = selected_segment_start_sample_plot + target_num_samples


            final_audio_segment = librosa.util.normalize(final_audio_segment)
        else:
            warnings.warn(f"Audio '{os.path.basename(file_path)}': No se pudo generar un segmento de audio final. Devolviendo None.")
            return None, None

        # --- Generación de Plots para Visualización ---
        if config.SAVE_PREPROCESS_PLOTS and plot_output_dir:
            try:
                fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=False)
                file_basename = os.path.basename(file_path)
                
                # Plot del audio original
                librosa.display.waveshow(y=original_audio_data_for_plot, sr=original_sr, ax=axes[0], color='blue', alpha=0.7)
                axes[0].set(title=f'Audio Original: {file_basename}',
                            xlabel='Tiempo (s)', ylabel='Amplitud')
                
                # Resaltar el segmento tomado del original para el preprocesamiento
                x_start_sec_plot = selected_segment_start_sample_plot / current_sr
                x_end_sec_plot = selected_segment_end_sample_plot / current_sr # Usa el fin real del segmento tomado del original
                
                axes[0].axvspan(x_start_sec_plot, x_end_sec_plot, color='red', alpha=0.3, label='Segmento Final (base del preprocesado)')
                
                # Mensaje en el plot original
                if target_num_samples > original_audio_duration_samples:
                    axes[0].text(original_audio_duration_samples / current_sr / 2, axes[0].get_ylim()[1] * 0.9, 
                                 f"Original ({original_audio_duration_samples/current_sr:.2f}s) padeado con silencio.", 
                                 color='red', fontsize=8, ha='center')
                elif longest_segment_interval and longest_segment_length < target_num_samples:
                    axes[0].text(x_start_sec_plot, axes[0].get_ylim()[1] * 0.9, 
                                 f"Activo ({longest_segment_length/current_sr:.2f}s) extendido con silencio original.", 
                                 color='red', fontsize=8)
                elif longest_segment_interval and longest_segment_length >= target_num_samples:
                     axes[0].text(x_start_sec_plot, axes[0].get_ylim()[1] * 0.9, 
                                 f"Activo ({longest_segment_length/current_sr:.2f}s) recortado del inicio.", 
                                 color='red', fontsize=8)
                else: # No actividad detectada, original suficientemente largo, se tomó el inicio
                    axes[0].text(x_start_sec_plot, axes[0].get_ylim()[1] * 0.9, 
                                 f"No actividad. Primeros {config.MAX_AUDIO_DURATION_SECONDS}s tomados.", 
                                 color='red', fontsize=8)


                axes[0].legend()

                # Plot del audio preprocesado final
                final_segment_duration_seconds = len(final_audio_segment) / current_sr
                librosa.display.waveshow(y=final_audio_segment, sr=current_sr, ax=axes[1], color='green', alpha=0.7)
                axes[1].set(title=f'Audio Preprocesado Final (Duración: {final_segment_duration_seconds:.2f}s)',
                            xlabel='Tiempo (s)', ylabel='Amplitud')
                axes[1].set_xlim(0, config.MAX_AUDIO_DURATION_SECONDS) # Aseguramos que el eje X vaya hasta la duración máxima
                                                                      # para ver si hay relleno al final

                plt.tight_layout()
                
                plot_filename = os.path.join(plot_output_dir, f"{os.path.splitext(file_basename)[0]}_preprocess_plot.png")
                plt.savefig(plot_filename)
                plt.close(fig)
            except Exception as plot_e:
                warnings.warn(f"Error al generar plot de preprocesamiento para {file_path}: {plot_e}")

        return final_audio_segment, current_sr

    except Exception as e:
        warnings.warn(f"Error al cargar o preprocesar audio {file_path}: {e}")
        return None, None

def save_preprocessed_audio(audio_data, sr, output_filepath):
    """
    Guarda los datos de audio preprocesados en un archivo WAV.
    """
    try:
        sf.write(output_filepath, audio_data, sr)
        return True
    except Exception as e:
        warnings.warn(f"Error al guardar el audio preprocesado {output_filepath}: {e}")
        return False