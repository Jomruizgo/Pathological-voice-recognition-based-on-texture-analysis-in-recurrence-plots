#02_Audio_Preprocess/main.py

import os
import sys
import glob
import warnings

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import config
from processor import load_and_preprocess_audio_with_sad, save_preprocessed_audio

def main():
    print("Iniciando la etapa de Preprocesamiento de Audio (con SAD y visualización segregada)...")

    # Crear directorios de salida de audio preprocesado
    os.makedirs(config.OUTPUT_PREPROCESSED_AUDIO_DIR, exist_ok=True)
    os.makedirs(config.PREPROCESSED_NORMAL_AUDIO_DIR, exist_ok=True)
    os.makedirs(config.PREPROCESSED_PATHOL_AUDIO_DIR, exist_ok=True)

    # Crear directorios para los plots de preprocesamiento si está habilitado
    if config.SAVE_PREPROCESS_PLOTS:
        os.makedirs(config.BASE_PREPROCESS_PLOTS_DIR, exist_ok=True) # Directorio base de plots
        os.makedirs(config.PREPROCESS_PLOTS_NORMAL_DIR, exist_ok=True) # Subdirectorio 'Normal' para plots
        os.makedirs(config.PREPROCESS_PLOTS_PATHOL_DIR, exist_ok=True) # Subdirectorio 'Pathol' para plots

    processed_count = 0
    skipped_count = 0

    # --- Procesar audios de la categoría "Normal" ---
    print(f"\nProcesando audios de la categoría 'Normal' desde: {config.ORIGINAL_NORMAL_AUDIO_DIR}")
    
    for filepath in glob.glob(os.path.join(config.ORIGINAL_NORMAL_AUDIO_DIR, "*.wav")):
        filename = os.path.basename(filepath)
        output_audio_filepath = os.path.join(config.PREPROCESSED_NORMAL_AUDIO_DIR, filename)

        # Pasamos el directorio específico para los plots de Normal
        audio_data, sr = load_and_preprocess_audio_with_sad(filepath, config.PREPROCESS_PLOTS_NORMAL_DIR)

        if audio_data is not None:
            if save_preprocessed_audio(audio_data, sr, output_audio_filepath):
                processed_count += 1
            else:
                skipped_count += 1
        else:
            skipped_count += 1

    # --- Procesar audios de la categoría "Patológico" ---
    print(f"\nProcesando audios de la categoría 'Patológico' desde: {config.ORIGINAL_PATHOL_AUDIO_DIR}")
    
    for filepath in glob.glob(os.path.join(config.ORIGINAL_PATHOL_AUDIO_DIR, "*.wav")):
        filename = os.path.basename(filepath)
        output_audio_filepath = os.path.join(config.PREPROCESSED_PATHOL_AUDIO_DIR, filename)

        # Pasamos el directorio específico para los plots de Pathol
        audio_data, sr = load_and_preprocess_audio_with_sad(filepath, config.PREPROCESS_PLOTS_PATHOL_DIR)

        if audio_data is not None:
            if save_preprocessed_audio(audio_data, sr, output_audio_filepath):
                processed_count += 1
            else:
                skipped_count += 1
        else:
            skipped_count += 1

    print(f"\n--- Resumen de Preprocesamiento ---")
    print(f"Audios procesados exitosamente: {processed_count}")
    print(f"Audios omitidos (errores/problemas): {skipped_count}")
    print(f"Archivos preprocesados guardados en: {config.OUTPUT_PREPROCESSED_AUDIO_DIR}")
    print(f"Plots de preprocesamiento (si SAVE_PREPROCESS_PLOTS=True) en: {config.BASE_PREPROCESS_PLOTS_DIR}/[Normal|Pathol]")
    print("Preprocesamiento de Audio completado.")

if __name__ == "__main__":
    main()