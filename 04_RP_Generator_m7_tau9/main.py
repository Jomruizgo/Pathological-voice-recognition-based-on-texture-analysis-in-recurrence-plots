# project_root/04_Recurrence_Plot_Generator/main.py

import os
import sys
import glob
import warnings
from multiprocessing import Pool, current_process

# Importar las funciones y la configuración desde rp_generator.py
from rp_generator import generate_and_save_pure_rp, generate_and_save_phase_space_plot
import config # Importa el config específico de este módulo (04_Recurrence_Plot_Generator/config.py)


def process_single_audio(args):
    """
    Procesa un solo archivo de audio para generar RP y/o PS.
    Esta función se ejecuta en paralelo.
    Recibe una tupla con todos los argumentos necesarios.
    """
    audio_filepath, output_rp_dir, output_ps_dir, category, save_rp, save_ps = args

    # Importar dentro del proceso hijo
    from rp_generator import generate_and_save_pure_rp, generate_and_save_phase_space_plot
    from multiprocessing import current_process

    base_filename = os.path.basename(audio_filepath)
    base_name = os.path.splitext(base_filename)[0]
    prefix = 'n_' if category == 'Normal' else 'p_'

    # Mostrar inicio del procesamiento con ID del proceso
    pid = current_process().name
    print(f"    [{pid}] Iniciando: {base_filename}", flush=True)

    result = {"file": base_filename, "rp": None, "ps": None}

    try:
        if save_rp:
            print(f"    [{pid}] Calculando RP: {base_filename}", flush=True)
            generate_and_save_pure_rp(audio_filepath, output_rp_dir, category)
            print(f"    [{pid}] Guardado: {base_filename}", flush=True)
            result["rp"] = "generated"
        else:
            result["rp"] = "disabled"

        if save_ps:
            generate_and_save_phase_space_plot(audio_filepath, output_ps_dir, category)
            result["ps"] = "generated"
        else:
            result["ps"] = "disabled"

    except Exception as e:
        import traceback
        result["error"] = f"{str(e)}\n{traceback.format_exc()}"

    return result


def _check_if_plot_exists(audio_filepath, output_dir, plot_type):
    """
    Verifica si un plot ya ha sido generado y existe en el directorio de salida.
    plot_type puede ser 'rp_pure' o 'ps'.
    """
    # Construir el nombre del archivo de salida esperado
    base_filename = os.path.splitext(os.path.basename(audio_filepath))[0]
    expected_filename = f"{base_filename}_{plot_type}.png"
    
    # Para RP, el output_dir es directamente la carpeta de la categoría
    # Para PS, generate_and_save_phase_space_plot asume output_dir ya apunta a la subcarpeta específica
    # Revisamos la función generate_and_save_pure_rp y generate_and_save_phase_space_plot en rp_generator.py
    # rp_pure: rp_output_dir = os.path.join(output_dir, 'Recurrence_Plots')
    # ps: ps_output_dir = os.path.join(output_dir, 'Phase_Space_Plots')
    # Ah, espera, las funciones en rp_generator.py ya crean el subdirectorio 'Recurrence_Plots' o 'Phase_Space_Plots'
    # dentro del output_dir que se les pasa.
    # Entonces, para _check_if_plot_exists, necesitamos saber la ruta COMPLETA donde se guardará.

    # En config.py tienes:
    # RP_OUTPUT_BASE_DIR = os.path.join(BASE_DIR, 'output', 'Recurrence_Plots')
    # PS_OUTPUT_BASE_DIR = os.path.join(BASE_DIR, 'output', 'Phase_Space_Plots')
    # Y luego en main.py se dividen por categoría:
    # output_rp_dirs = { "Normal": os.path.join(RP_OUTPUT_BASE_DIR, 'Normal'), ... }

    # Entonces, el path final es: current_output_rp_dir/filename_rp_pure.png
    # o current_output_ps_dir/filename_ps.png

    full_output_path = os.path.join(output_dir, expected_filename)
    return os.path.exists(full_output_path)


def main():
    print("Iniciando la generación de Recurrence Plots y Phase Space Plots...")

    # Rutas de entrada base (desde 02_Audio_Preprocess/output)
    input_dirs = {
        "Normal": config.INPUT_NORMAL_AUDIO_DIR, # <-- ¡CORREGIDO! Usar INPUT_NORMAL_AUDIO_DIR
        "Pathol": config.INPUT_PATHOL_AUDIO_DIR  # <-- ¡CORREGIDO! Usar INPUT_PATHOL_AUDIO_DIR
    }

    # Rutas de salida para los plots (también definidas en config.py)
    # Estas son las BASES para las categorías. Las funciones generadoras se encargarán
    # de los subdirectorios 'Recurrence_Plots' y 'Phase_Space_Plots' dentro de ellas.
    # NOTA: Tu config.py ya define directamente las rutas con las subcarpetas Normal/Pathol
    # por ejemplo: OUTPUT_RP_NORMAL_DIR = os.path.join(OUTPUT_RP_BASE_DIR, 'Recurrence_Plots', 'Normal')
    # Así que podemos usarlas directamente.

    output_rp_category_base_dirs = {
        "Normal": config.OUTPUT_RP_NORMAL_DIR,
        "Pathol": config.OUTPUT_RP_PATHOL_DIR
    }
    output_ps_category_base_dirs = {
        "Normal": config.OUTPUT_PS_NORMAL_DIR,
        "Pathol": config.OUTPUT_PS_PATHOL_DIR
    }

    # Crear todos los directorios de salida según la configuración
    for category in input_dirs.keys():
        # Asegurarse de crear los directorios específicos de categoría
        # No usamos os.path.join aquí porque las variables de config.py ya son rutas completas
        os.makedirs(output_rp_category_base_dirs[category], exist_ok=True)
        os.makedirs(output_ps_category_base_dirs[category], exist_ok=True)
        print(f"Directorios de salida creados/verificados para {category}:")
        print(f"  RPs: {output_rp_category_base_dirs[category]}")
        print(f"  PSs: {output_ps_category_base_dirs[category]}")


    # Definir las categorías de audio
    audio_categories = ["Normal", "Pathol"] # Ajusta si tienes otras categorías

    # Obtener número de workers
    n_jobs = getattr(config, 'N_JOBS', 1)
    if n_jobs == -1:
        n_jobs = os.cpu_count()

    print(f"\nUsando {n_jobs} núcleo(s) para procesamiento paralelo")

    for category in audio_categories:
        category_input_dir = input_dirs[category]

        # Obtener todos los archivos .wav en la categoría
        audio_files = glob.glob(os.path.join(category_input_dir, "*.wav"))

        if not audio_files:
            warnings.warn(f"No se encontraron archivos de audio en {category_input_dir}. Saltando la categoría '{category}'.")
            continue

        print(f"\nProcesando {len(audio_files)} audios para la categoría '{category}' desde {category_input_dir}...")

        # Obtener los directorios de salida específicos para la categoría actual
        current_output_rp_dir = output_rp_category_base_dirs[category]
        current_output_ps_dir = output_ps_category_base_dirs[category]

        # Filtrar archivos que necesitan procesamiento
        files_to_process = []
        for audio_filepath in audio_files:
            base_filename = os.path.basename(audio_filepath)
            base_name = os.path.splitext(base_filename)[0]
            prefix = 'n_' if category == 'Normal' else 'p_'

            rp_output_path = os.path.join(current_output_rp_dir, f"{prefix}{base_name}_rp_pure.png")
            rp_exists = os.path.exists(rp_output_path)

            ps_output_path = os.path.join(current_output_ps_dir, f"{prefix}{base_name}_ps.png")
            ps_exists = os.path.exists(ps_output_path)

            should_process_rp = config.SAVE_RECURRENCE_PLOTS and not rp_exists
            should_process_ps = config.SAVE_PHASE_SPACE_PLOTS and not ps_exists

            if should_process_rp or should_process_ps:
                files_to_process.append(audio_filepath)
            else:
                print(f"  Saltando {base_filename}: Plots ya existen.")

        if not files_to_process:
            print(f"  Todos los plots para '{category}' ya existen.")
            continue

        print(f"  Generando {len(files_to_process)} nuevos plots...")

        # Preparar argumentos para cada archivo
        save_rp = config.SAVE_RECURRENCE_PLOTS
        save_ps = config.SAVE_PHASE_SPACE_PLOTS

        task_args = [
            (audio_filepath, current_output_rp_dir, current_output_ps_dir, category, save_rp, save_ps)
            for audio_filepath in files_to_process
        ]

        # Procesamiento paralelo
        if n_jobs > 1:
            total = len(task_args)
            # maxtasksperchild=1 fuerza reinicio del worker después de cada tarea
            # para liberar memoria correctamente
            with Pool(processes=n_jobs, maxtasksperchild=1) as pool:
                for i, result in enumerate(pool.imap(process_single_audio, task_args), 1):
                    if "error" in result:
                        print(f"  Error en {result['file']}:\n{result['error']}")
                    else:
                        print(f"  [{i}/{total}] {result['file']} - RP: {result['rp']}")
        else:
            # Procesamiento secuencial
            for i, args in enumerate(task_args, 1):
                result = process_single_audio(args)
                if "error" in result:
                    print(f"  Error en {result['file']}:\n{result['error']}")
                else:
                    print(f"  [{i}/{len(files_to_process)}] {result['file']} - RP: {result['rp']}")

    print("\nGeneración de Recurrence Plots y Phase Space Plots completada.")
    print(f"Plots guardados según la estructura definida en config.py")


if __name__ == "__main__":
    main()