# project_root/03_Full_Optimal_Parameters_Analysis/main.py

import os
import librosa
import numpy as np
import warnings
import glob
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import hashlib
from datetime import datetime
from multiprocessing import Pool, Lock, Manager
from functools import partial

# Importar funciones de los módulos separados
import config
from calculate_optimal_tau import calculate_mutual_information, calculate_autocorrelation, plot_tau_analysis, find_first_local_minimum
from calculate_optimal_dim import calculate_false_nearest_neighbors, plot_dim_analysis


def get_config_hash():
    """Genera un hash de la configuración para detectar cambios."""
    config_params = {
        'TAU_MAX': config.TAU_MAX,
        'DIM_MAX': config.DIM_MAX,
        'FNN_THRESHOLD': config.FNN_THRESHOLD,
        'TARGET_SAMPLE_RATE': config.TARGET_SAMPLE_RATE
    }
    return hashlib.md5(json.dumps(config_params, sort_keys=True).encode()).hexdigest()


def load_checkpoint():
    """Carga el checkpoint si existe."""
    if not os.path.exists(config.CHECKPOINT_FILE):
        return None, False

    try:
        with open(config.CHECKPOINT_FILE, 'r') as f:
            checkpoint = json.load(f)

        if checkpoint.get('config_hash') != get_config_hash():
            print("⚠️  Configuración cambiada. Iniciando nuevo análisis.")
            return None, False

        print(f"✓ Checkpoint encontrado: {checkpoint['processed_files']}/{checkpoint['total_files']} procesados")
        return checkpoint, True
    except Exception as e:
        print(f"⚠️  Error al cargar checkpoint: {e}")
        return None, False


def save_checkpoint(results, processed_files_unique_ids, total_files):
    """Guarda el progreso actual usando IDs únicos (categoria_filename)."""
    checkpoint = {
        'version': '1.0',
        'config_hash': get_config_hash(),
        'last_update': datetime.now().isoformat(),
        'total_files': total_files,
        'processed_files': len(processed_files_unique_ids),
        'results': results,
        'processed_file_list': sorted(list(processed_files_unique_ids))
    }

    temp_file = config.CHECKPOINT_FILE + '.tmp'
    with open(temp_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    os.replace(temp_file, config.CHECKPOINT_FILE)

    if config.VERBOSE:
        print(f"  → Checkpoint guardado ({len(processed_files_unique_ids)}/{total_files})")


def load_and_process_audio_fragment(filepath, duration_s, sr):
    """Carga un audio completo o un fragmento."""
    try:
        y, _ = librosa.load(filepath, sr=sr, duration=duration_s, mono=True)
        if y.ndim > 1:
            y = y.flatten()
        return y
    except Exception as e:
        warnings.warn(f"No se pudo cargar {filepath}: {e}")
        return None


def process_single_file(args):
    """
    Procesa un único archivo de audio. Esta función es llamada por cada proceso.
    Args: tupla (filepath, category, index, total_files)
    """
    filepath, category, index, total_files = args
    filename = os.path.basename(filepath)

    try:
        print(f"[{index}/{total_files}] Procesando: {filename} ({category})")

        y = load_and_process_audio_fragment(filepath, config.AUDIO_DURATION, config.TARGET_SAMPLE_RATE)
        if y is None:
            return None

        min_required = (config.DIM_MAX - 1) * config.TAU_MAX + 1
        if len(y) < min_required:
            warnings.warn(f"Audio muy corto ({len(y)} < {min_required} requeridas)")
            return None

        # Calcular Tau
        mi_values = calculate_mutual_information(y, config.TAU_MAX, num_bins=10)
        acf_values = calculate_autocorrelation(y, config.TAU_MAX)

        tau_suggestions = {}
        taus_range = np.arange(1, config.TAU_MAX + 1)

        if mi_values.size > 0 and not np.isnan(mi_values).all():
            # Usar PRIMER MÍNIMO LOCAL (no mínimo global)
            first_local_min_idx = find_first_local_minimum(mi_values)
            tau_suggestions['MI_min'] = int(taus_range[first_local_min_idx])

        first_zero = np.where(np.diff(np.sign(acf_values)))[0]
        if first_zero.size > 0:
            tau_suggestions['ACF_zero_cross'] = int(taus_range[first_zero[0]])

        # Crear prefix único con categoría para evitar sobrescrituras
        file_prefix = f"{category}_{os.path.splitext(filename)[0]}"

        if config.SAVE_TAU_PLOT:
            plot_tau_analysis(mi_values, acf_values, config.TAU_MAX,
                            file_prefix, config.OUTPUT_ANALYSIS_DIR)

        # Calcular Dimensión
        chosen_tau = tau_suggestions.get('MI_min', tau_suggestions.get('ACF_zero_cross', 1))
        fnn_values = calculate_false_nearest_neighbors(y, chosen_tau, config.DIM_MAX,
                                                       r_tolerance=15.0, a_tolerance=2.0)

        optimal_dim = None
        dims_range = np.arange(1, config.DIM_MAX + 1)
        if fnn_values.size > 0 and not np.isnan(fnn_values).all():
            for d_idx, fnn_percent in enumerate(fnn_values):
                if fnn_percent <= config.FNN_THRESHOLD * 100:
                    optimal_dim = int(dims_range[d_idx])
                    break
            if optimal_dim is None:
                optimal_dim = int(dims_range[-1])

        if config.SAVE_DIM_PLOT:
            plot_dim_analysis(fnn_values, config.DIM_MAX, config.FNN_THRESHOLD,
                            file_prefix, config.OUTPUT_ANALYSIS_DIR)

        print(f"  ✓ {filename}: Tau={tau_suggestions}, Dim={optimal_dim}")

        # Crear ID único combinando categoría y filename
        unique_id = f"{category}_{filename}"

        return {
            'unique_id': unique_id,
            'filename': filename,
            'category': category,
            'tau_mi': tau_suggestions.get('MI_min'),
            'tau_acf': tau_suggestions.get('ACF_zero_cross'),
            'dim_fnn': optimal_dim
        }

    except Exception as e:
        print(f"  ✗ Error procesando {filename}: {e}")
        return None


def main():
    print("="*70)
    print("ANÁLISIS COMPLETO DE PARÁMETROS ÓPTIMOS (PARALELO)")
    print("="*70)
    print(f"Usando {config.NUM_CORES} cores para procesamiento paralelo")
    print("="*70)

    os.makedirs(config.OUTPUT_ANALYSIS_DIR, exist_ok=True)

    # Cargar checkpoint
    checkpoint, is_valid = load_checkpoint()

    # Obtener archivos
    all_normal_files = sorted(glob.glob(os.path.join(config.INPUT_NORMAL_AUDIO_DIR, "*.wav")))
    all_pathol_files = sorted(glob.glob(os.path.join(config.INPUT_PATHOL_AUDIO_DIR, "*.wav")))

    if config.PROCESS_ALL_FILES:
        sample_files_normal = all_normal_files
        sample_files_pathol = all_pathol_files
    else:
        num_normal = min(config.NUM_NORMAL_SAMPLES_FOR_ANALYSIS, len(all_normal_files))
        num_pathol = min(config.NUM_PATHOL_SAMPLES_FOR_ANALYSIS, len(all_pathol_files))
        sample_files_normal = random.sample(all_normal_files, num_normal)
        sample_files_pathol = random.sample(all_pathol_files, num_pathol)

    all_sample_files = [(f, 'Normal') for f in sample_files_normal] + \
                       [(f, 'Pathol') for f in sample_files_pathol]

    if not all_sample_files:
        print("ERROR: No se encontraron archivos de audio.")
        return

    total_files = len(all_sample_files)
    print(f"Total archivos: {total_files} (Normal: {len(sample_files_normal)}, Pathol: {len(sample_files_pathol)})")

    # Restaurar progreso usando IDs únicos (categoria_filename)
    results = checkpoint.get('results', []) if is_valid and checkpoint else []
    processed_unique_ids = set(checkpoint.get('processed_file_list', []) if is_valid and checkpoint else [])

    # Crear mapa de resultados para verificar parámetros completos
    results_map = {r['unique_id']: r for r in results}
    results_ids = set(results_map.keys())

    # Identificar archivos problemáticos:
    # 1. En processed_file_list pero SIN registro en results
    missing_results_ids = processed_unique_ids - results_ids

    # 2. Con registro pero parámetros incompletos (null)
    incomplete_ids = set()
    for r in results:
        if r.get('tau_mi') is None or r.get('tau_acf') is None or r.get('dim_fnn') is None:
            incomplete_ids.add(r['unique_id'])

    # Combinar todos los que necesitan reprocesarse
    needs_reprocess = missing_results_ids | incomplete_ids

    if processed_unique_ids:
        print(f"Archivos en checkpoint: {len(processed_unique_ids)}")
        print(f"Registros en results: {len(results)}")
        print(f"Sin registro en results: {len(missing_results_ids)}")
        print(f"Con parámetros incompletos: {len(incomplete_ids)}")
        print(f"Total a reprocesar: {len(needs_reprocess)}\n")

    # Filtrar archivos pendientes: no procesados O necesitan reprocesarse
    files_to_process = []
    for i, (filepath, category) in enumerate(all_sample_files, 1):
        filename = os.path.basename(filepath)
        unique_id = f"{category}_{filename}"

        # Procesar si: no está en checkpoint O necesita reprocesarse
        if unique_id not in processed_unique_ids or unique_id in needs_reprocess:
            files_to_process.append((filepath, category, i, total_files))
            # Remover del set para que se reprocese
            if unique_id in needs_reprocess:
                processed_unique_ids.discard(unique_id)
                # Eliminar resultado anterior si existe
                results = [r for r in results if r['unique_id'] != unique_id]
        elif config.VERBOSE:
            print(f"[{i}/{total_files}] ⊘ Saltando: {filename} ({category})")

    if not files_to_process:
        print("\n✓ Todos los archivos ya fueron procesados!")
    else:
        print(f"\nProcesando {len(files_to_process)} archivos pendientes...\n")

        # Procesamiento paralelo en batches
        total_to_process = len(files_to_process)

        for batch_start in range(0, total_to_process, config.BATCH_SIZE):
            batch_end = min(batch_start + config.BATCH_SIZE, total_to_process)
            batch = files_to_process[batch_start:batch_end]

            print(f"\n--- Procesando batch {batch_start//config.BATCH_SIZE + 1} ({batch_start+1}-{batch_end}/{total_to_process}) ---")

            with Pool(processes=config.NUM_CORES) as pool:
                # Procesar batch en paralelo
                batch_results = pool.map(process_single_file, batch)

            # Recopilar resultados del batch
            for result in batch_results:
                if result is not None:
                    results.append(result)
                    processed_unique_ids.add(result['unique_id'])

            # Guardar checkpoint después de cada batch
            save_checkpoint(results, processed_unique_ids, total_files)
            print(f"  → Checkpoint guardado: {len(processed_unique_ids)}/{total_files} archivos completados")

    # Checkpoint final
    save_checkpoint(results, processed_unique_ids, total_files)

    # Análisis agregado
    if results:
        results_df = pd.DataFrame(results)
        csv_path = os.path.join(config.OUTPUT_ANALYSIS_DIR, 'parametros_optimos_completo.csv')
        results_df.to_csv(csv_path, index=False)

        print("\n" + "="*70)
        print("ANÁLISIS AGREGADO")
        print("="*70)
        print(results_df.to_string(index=False))

        # Estadísticas
        for col, name in [('tau_mi', 'Tau (MI)'), ('tau_acf', 'Tau (ACF)'), ('dim_fnn', 'Dimensión (FNN)')]:
            values = results_df[col].dropna()
            if not values.empty:
                values = values.astype(int)
                print(f"\n{name}: Promedio={values.mean():.2f}, Mediana={values.median()}, Moda={values.mode().tolist()}")

        # Gráficos
        if config.SAVE_AGGREGATE_PLOTS:
            for col, title, fname in [
                ('tau_mi', 'Tau (MI)', 'tau_mi'),
                ('dim_fnn', 'Dimensión (FNN)', 'dim_fnn')
            ]:
                values = results_df[col].dropna().astype(int)
                if not values.empty:
                    # Histograma
                    plt.figure(figsize=(10, 5))
                    bins = np.arange(values.min() - 0.5, values.max() + 1.5, 1)
                    sns.histplot(values, kde=False, bins=bins, stat='count')
                    plt.title(f'Distribución de {title}')
                    plt.xlabel(title)
                    plt.ylabel('Frecuencia')
                    plt.grid(True)
                    plt.savefig(os.path.join(config.OUTPUT_ANALYSIS_DIR, f'{fname}_distribution.png'))
                    plt.close()

                    # Boxplot
                    plt.figure(figsize=(10, 3))
                    sns.boxplot(x=values)
                    plt.title(f'Boxplot de {title}')
                    plt.grid(True)
                    plt.savefig(os.path.join(config.OUTPUT_ANALYSIS_DIR, f'{fname}_boxplot.png'))
                    plt.close()

    print("\n" + "="*70)
    print("✓ ANÁLISIS COMPLETADO")
    print(f"Procesados: {len(processed_unique_ids)}/{total_files}")
    print("="*70)


if __name__ == "__main__":
    main()
