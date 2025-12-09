# 01_EDA/main.py
import os
import pandas as pd
from pydub import AudioSegment
import matplotlib.pyplot as plt
import seaborn as sns
import librosa # Necesario para carga de audio y SAD
import librosa.display # Necesario para plots de forma de onda
import soundfile as sf # Para cargar WAV en librosa
import numpy as np # Para manejo de arrays

# Importar el config del mismo módulo
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import config
sys.path.pop(0)


# Asegurarse de que los directorios de salida existan
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

def analyze_audio_file(filepath, label):
    """
    Carga un archivo de audio y extrae sus propiedades básicas.
    """
    try:
        audio = AudioSegment.from_wav(filepath)
        duration_seconds = len(audio) / 1000.0
        sample_rate = audio.frame_rate
        channels = audio.channels
        bit_depth = audio.sample_width * 8
        file_size_bytes = os.path.getsize(filepath)

        return {
            'filename': os.path.basename(filepath),
            'label': label,
            'duration_s': duration_seconds,
            'sample_rate_hz': sample_rate,
            'channels': channels,
            'bit_depth_bits': bit_depth,
            'file_size_bytes': file_size_bytes
        }
    except Exception as e:
        print(f"Error al analizar {filepath}: {e}")
        return None

def plot_audio_waveform_with_silence(filepath, output_dir, label, target_sr, sad_top_db):
    """
    Carga un archivo de audio, detecta secciones de silencio y genera un gráfico
    de la forma de onda resaltando esas secciones.
    """
    try:
        # Cargar audio con librosa, remuestreando a la frecuencia objetivo
        audio_data, sr = librosa.load(filepath, sr=target_sr, mono=True)

        # Detectar segmentos activos usando librosa.effects.split
        # Esto devuelve intervalos [inicio_muestra, fin_muestra] de secciones con sonido
        active_segments = librosa.effects.split(y=audio_data, top_db=sad_top_db)

        # Identificar los segmentos de silencio (lo contrario de los activos)
        silent_segments = []
        current_pos = 0
        for start_sample, end_sample in active_segments:
            if start_sample > current_pos: # Si hay un hueco antes del segmento activo actual
                silent_segments.append((current_pos, start_sample))
            current_pos = end_sample # Avanzar a después del segmento activo
        
        # Si queda audio después del último segmento activo, es silencio
        if current_pos < len(audio_data):
            silent_segments.append((current_pos, len(audio_data)))


        fig, ax = plt.subplots(figsize=(15, 6))
        
        # Plotear la forma de onda completa
        librosa.display.waveshow(y=audio_data, sr=sr, ax=ax, color='blue', alpha=0.7)
        
        # Resaltar las secciones de silencio
        for start_silence, end_silence in silent_segments:
            start_sec = start_silence / sr
            end_sec = end_silence / sr
            ax.axvspan(start_sec, end_sec, color='red', alpha=0.3, label='Sección de Silencio' if silent_segments.index((start_silence, end_silence)) == 0 else "") # Solo una etiqueta
        
        # Añadir las etiquetas para el rango de tiempo
        ax.set(title=f'Forma de Onda de Audio - {label}: {os.path.basename(filepath)}\n(Secciones de Silencio resaltadas)',
               xlabel='Tiempo (segundos)', ylabel='Amplitud')
        ax.legend()
        ax.grid(True)
        
        # Asegurarse de que el directorio de salida para plots exista
        os.makedirs(output_dir, exist_ok=True)
        
        plot_filename = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(filepath))[0]}_waveform_silence.png")
        plt.savefig(plot_filename)
        plt.close(fig) # Cierra la figura para liberar memoria
        # print(f"Gráfico de forma de onda guardado en: {plot_filename}") # Para depuración
        
    except Exception as e:
        print(f"Error al generar gráfico de forma de onda para {filepath}: {e}")


def run_eda():
    """
    Realiza un análisis exploratorio de datos en los archivos de audio.
    """
    print("Iniciando el Análisis Exploratorio de Datos (EDA) de audios...")

    all_audio_data = []

    # Crear directorios para los plots de forma de onda si están habilitados
    if config.EDA_SAVE_WAVEFORM_PLOTS:
        os.makedirs(config.EDA_WAVEFORM_PLOTS_DIR, exist_ok=True)
        os.makedirs(config.EDA_WAVEFORM_PLOTS_NORMAL_DIR, exist_ok=True)
        os.makedirs(config.EDA_WAVEFORM_PLOTS_PATHOL_DIR, exist_ok=True)

    # Analizar y plotear audios normales
    normal_files = [os.path.join(config.ORIGINAL_NORMAL_AUDIO_DIR, f)
                    for f in os.listdir(config.ORIGINAL_NORMAL_AUDIO_DIR) if f.endswith('.wav')]
    print(f"Analizando {len(normal_files)} archivos en {config.ORIGINAL_NORMAL_AUDIO_DIR}...")
    for f_path in normal_files:
        data = analyze_audio_file(f_path, 'Normal')
        if data:
            all_audio_data.append(data)
            if config.EDA_SAVE_WAVEFORM_PLOTS:
                plot_audio_waveform_with_silence(
                    f_path, 
                    config.EDA_WAVEFORM_PLOTS_NORMAL_DIR, # Directorio de salida específico
                    'Normal', 
                    config.TARGET_SAMPLE_RATE, 
                    config.SAD_TOP_DB
                )

    # Analizar y plotear audios patológicos
    pathol_files = [os.path.join(config.ORIGINAL_PATHOL_AUDIO_DIR, f)
                    for f in os.listdir(config.ORIGINAL_PATHOL_AUDIO_DIR) if f.endswith('.wav')]
    print(f"Analizando {len(pathol_files)} archivos en {config.ORIGINAL_PATHOL_AUDIO_DIR}...")
    for f_path in pathol_files:
        data = analyze_audio_file(f_path, 'Pathol')
        if data:
            all_audio_data.append(data)
            if config.EDA_SAVE_WAVEFORM_PLOTS:
                plot_audio_waveform_with_silence(
                    f_path, 
                    config.EDA_WAVEFORM_PLOTS_PATHOL_DIR, # Directorio de salida específico
                    'Pathol', 
                    config.TARGET_SAMPLE_RATE, 
                    config.SAD_TOP_DB
                )

    if not all_audio_data:
        print("No se encontraron archivos de audio válidos para el análisis.")
        return

    df_eda = pd.DataFrame(all_audio_data)

    print("\n--- Resumen General de los Datos de Audio ---")
    print(df_eda.head())
    print(f"\nTotal de archivos analizados: {len(df_eda)}")

    # Guardar resumen en un archivo de texto
    with open(config.EDA_REPORT_FILE, 'w') as f:
        f.write("--- Resumen Estadístico de Propiedades de Audio ---\n")
        f.write(df_eda.describe().to_string())
        f.write("\n\n--- Conteo de Canales por Tipo de Audio ---\n")
        f.write(df_eda.groupby('label')['channels'].value_counts().unstack(fill_value=0).to_string())
        f.write("\n\n--- Conteo de Profundidad de Bits por Tipo de Audio ---\n")
        f.write(df_eda.groupby('label')['bit_depth_bits'].value_counts().unstack(fill_value=0).to_string())
        f.write("\n\n--- Conteo de Frecuencia de Muestreo por Tipo de Audio ---\n")
        f.write(df_eda.groupby('label')['sample_rate_hz'].value_counts().unstack(fill_value=0).to_string())


    print(f"\nResumen estadístico guardado en: {config.EDA_REPORT_FILE}")

    # --- Visualizaciones Agregadas ---
    print("\nGenerando visualizaciones agregadas (distribuciones)...")

    # Distribución de Duración
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df_eda, x='duration_s', hue='label', kde=True, bins=20)
    plt.title('Distribución de la Duración de los Audios')
    plt.xlabel('Duración (segundos)')
    plt.ylabel('Conteo de Audios')
    plt.grid(True)
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'duration_distribution.png'))
    plt.close()
    print(f"Gráfico de duración guardado en: {os.path.join(config.OUTPUT_DIR, 'duration_distribution.png')}")

    # Distribución de Frecuencia de Muestreo (si hay variaciones)
    if df_eda['sample_rate_hz'].nunique() > 1:
        plt.figure(figsize=(10, 5))
        sns.countplot(data=df_eda, x='sample_rate_hz', hue='label')
        plt.title('Frecuencia de Conteo por Frecuencia de Muestreo')
        plt.xlabel('Frecuencia de Muestreo (Hz)')
        plt.ylabel('Conteo')
        plt.grid(True) # Añadir grid para mejor legibilidad
        plt.savefig(os.path.join(config.OUTPUT_DIR, 'sample_rate_counts.png'))
        plt.close()
        print(f"Gráfico de frecuencia de muestreo guardado en: {os.path.join(config.OUTPUT_DIR, 'sample_rate_counts.png')}")
    else:
        print("Todos los audios tienen la misma frecuencia de muestreo. No se generó gráfico de frecuencia de muestreo.")

    print(f"EDA completado. Revisa el archivo de reporte en '{config.EDA_REPORT_FILE}' y los gráficos en '{config.OUTPUT_DIR}' y '{config.EDA_WAVEFORM_PLOTS_DIR}'.")

# Esta parte es para cuando se ejecuta este script directamente
if __name__ == "__main__":
    run_eda()