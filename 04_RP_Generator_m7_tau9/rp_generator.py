# project_root/03_Recurrence_Plot_Generator/rp_generator.py
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from PIL import Image
import warnings

# Importar el config del mismo módulo
import config


def _calculate_rp_matrix(y):
    """
    Función auxiliar para calcular la matriz de recurrencia de forma manual.
    """
    # 1. Embedding de Takens (lógica ya probada)
    if y.ndim > 1:
        y = y.flatten()

    N = len(y)
    D = config.EMBEDDING_DIM
    tau = config.TIME_DELAY
    
    min_length_for_embedding = (D - 1) * tau + 1
    if N < min_length_for_embedding:
        raise ValueError(f"Audio es demasiado corto ({N} muestras) para generar RP con los parámetros (D={D}, tau={tau}). Mínimo requerido: {min_length_for_embedding} muestras.")

    dtype = getattr(np, config.DTYPE) if hasattr(config, 'DTYPE') else np.float64
    embedded_series = np.zeros((N - (D - 1) * tau, D), dtype=dtype)
    for i in range(D):
        embedded_series[:, i] = y[i * tau : N - (D - 1 - i) * tau]
    
    if embedded_series.shape[0] == 0:
        raise ValueError(f"La serie embebida resultante para RP está vacía. Ajuste los parámetros de embedding (EMBEDDING_DIM, TIME_DELAY) o la duración del audio.")

    # 2. Calcular la Matriz de Distancias (Euclidiana)
    # Forma eficiente de calcular distancias euclidianas entre filas de una matriz
    # Esto es equivalente a scipy.spatial.distance.pdist con metric='euclidean'
    # y luego scipy.spatial.distance.squareform
    
    # Calcular la matriz de distancias euclidianas
    # reshape(1, -1) si es necesario para cdist (no debería serlo si embedded_series es (M, D))
    # Para ser claros: M es el número de puntos embebidos, D es la dimensión
    
    # Método 1: Usando broadcasting de NumPy (más eficiente)
    # (A - B)^2 = A^2 + B^2 - 2AB
    # sum( (embedded_series[i,:] - embedded_series[j,:])^2 )
    
    # Calcular matriz de distancias directamente en dtype especificado
    # Usamos scipy para cálculo más eficiente en memoria
    from scipy.spatial.distance import pdist, squareform

    distance_matrix_unthresholded = squareform(pdist(embedded_series, metric='euclidean')).astype(dtype)

    # 3. Determinar el umbral (epsilon)
    epsilon_abs = 0.0 # Inicializar
    
    if config.EPSILON_MODE == 'percentage':
        # Excluir la diagonal principal antes de calcular el percentil para ser más preciso
        # (aunque en distancias euclidianas ya son 0, es buena práctica)
        distances_flat = distance_matrix_unthresholded[np.triu_indices(distance_matrix_unthresholded.shape[0], k=1)]
        
        if len(distances_flat) == 0:
            warnings.warn("No hay suficientes distancias para calcular el percentil. Estableciendo epsilon_abs a 0.")
            epsilon_abs = 0.0
        else:
            epsilon_abs = np.percentile(distances_flat, config.EPSILON_VALUE * 100)
            
    elif config.EPSILON_MODE == 'fixed':
        epsilon_abs = config.EPSILON_VALUE
    else:
        raise ValueError(f"EPSILON_MODE '{config.EPSILON_MODE}' no es válido. Debe ser 'percentage' o 'fixed'.")

    # 4. Crear la Matriz de Recurrencia (Binaria o No Binaria)
    if config.BINARY_RP:
        recurrence_matrix = (distance_matrix_unthresholded <= epsilon_abs).astype(int)
    else:
        # --- SOLUCIÓN PARA LA DIAGONAL BLANCA EN ESCALA DE GRISES ---
        max_dist_overall = np.max(distance_matrix_unthresholded)
        if max_dist_overall == 0:
            # Si todas las distancias son 0 (e.g., serie constante), todo es recurrente (negro)
            recurrence_matrix = np.zeros_like(distance_matrix_unthresholded)
        else:
            # Normaliza las distancias para que 0 (recurrente) sea negro y 1 (no recurrente) sea blanco.
            # dist_norm = distancia_actual / distancia_maxima
            recurrence_matrix = distance_matrix_unthresholded / max_dist_overall
            
            # Opcional: Aplicar un "clip" visual si se desea que distancias muy grandes se vean todas blancas
            # Esto NO es una binarización, solo una mejora visual.
            # recurrence_matrix[recurrence_matrix > 1] = 1 # Ya está en [0,1] si está normalizada por max_dist

    return recurrence_matrix

def generate_and_save_pure_rp(audio_filepath, output_dir, label):
    """
    Carga un archivo de audio, genera su Recurrence Plot PURO (sin ejes, títulos, etc.)
    y lo guarda como imagen usando PIL (más eficiente en memoria que matplotlib).
    Puede ser binario o en escala de grises.
    """
    try:
        y, sr = librosa.load(audio_filepath, sr=config.TARGET_SAMPLE_RATE, mono=True)

        # Calcular la matriz de recurrencia
        recurrence_matrix = _calculate_rp_matrix(y)

        # Convertir matriz a imagen usando PIL (mucho más eficiente que matplotlib)
        # La matriz está normalizada [0, 1], donde 0=negro (recurrente) y 1=blanco (no recurrente)
        # Para escala de grises: multiplicar por 255
        # Para binario: 0 -> 0 (negro), 1 -> 255 (blanco)

        if config.BINARY_RP:
            # Matriz binaria: 1=recurrente (negro), 0=no recurrente (blanco)
            # Invertir para que recurrente sea negro
            img_array = ((1 - recurrence_matrix) * 255).astype(np.uint8)
        else:
            # Escala de grises: 0=recurrente (negro), 1=no recurrente (blanco)
            img_array = (recurrence_matrix * 255).astype(np.uint8)

        # Voltear verticalmente para que origin='lower' (como en matplotlib)
        img_array = np.flipud(img_array)

        # Crear imagen PIL en escala de grises
        img = Image.fromarray(img_array, mode='L')

        # Generar nombre único con prefijo según la clase
        base_name = os.path.splitext(os.path.basename(audio_filepath))[0]
        prefix = 'n_' if label == 'Normal' else 'p_'
        rp_filename = os.path.join(output_dir, f"{prefix}{base_name}_rp_pure.png")

        # Guardar imagen
        img.save(rp_filename)

    except ValueError as ve:
        warnings.warn(f"Error (Valor Inválido) al generar Recurrence Plot PURO para {audio_filepath}: {ve}")
    except Exception as e:
        warnings.warn(f"Error al generar Recurrence Plot PURO para {audio_filepath}: {e}")



def generate_and_save_phase_space_plot(audio_filepath, output_dir, label):
    """
    Carga un archivo de audio, genera su Phase Space Plot y lo guarda como imagen.
    Solo para EMBEDDING_DIM = 2 o 3.
    """
    try:
        y, sr = librosa.load(audio_filepath, sr=config.TARGET_SAMPLE_RATE, mono=True)

        if config.EMBEDDING_DIM < 2 or config.EMBEDDING_DIM > 3:
            warnings.warn(f"El Phase Space Plot solo es visualizable para EMBEDDING_DIM = 2 o 3. Actualmente es {config.EMBEDDING_DIM}. Saltando.")
            return

        # MANUAL TAKENS' EMBEDDING: Confirmed working by separate test
        # Ensure y is 1D
        if y.ndim > 1:
            y = y.flatten()

        # Calculate the length of the embedded series
        N = len(y)
        D = config.EMBEDDING_DIM
        tau = config.TIME_DELAY
        
        # Minimum required length check for embedding
        min_length_for_embedding = (D - 1) * tau + 1
        if N < min_length_for_embedding:
            raise ValueError(f"Audio es demasiado corto ({N} muestras) para generar Phase Space Plot con los parámetros actuales (D={D}, tau={tau}). Mínimo requerido: {min_length_for_embedding} muestras.")

        # Create the embedded series
        embedded_series = np.zeros((N - (D - 1) * tau, D))
        for i in range(D):
            embedded_series[:, i] = y[i * tau : N - (D - 1 - i) * tau]

        # Ensure the embedded series is not empty
        if embedded_series.shape[0] == 0:
            raise ValueError(f"La serie embebida resultante está vacía. Ajuste los parámetros de embedding (EMBEDDING_DIM, TIME_DELAY) o la duración del audio.")
            
        fig = plt.figure(figsize=(10, 8))
        
        if D == 2:
            ax = fig.add_subplot(111)
            ax.plot(embedded_series[:, 0], embedded_series[:, 1], 'k-', linewidth=0.5, alpha=0.7)
            ax.set_xlabel(f'x(t)')
            ax.set_ylabel(f'x(t + {tau}*tau)')
            ax.set_title(f'Phase Space Plot (2D): {os.path.basename(audio_filepath)} ({label})')
            ax.grid(True)
        elif D == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(embedded_series[:, 0], embedded_series[:, 1], embedded_series[:, 2], 'k-', linewidth=0.5, alpha=0.7)
            ax.set_xlabel(f'x(t)')
            ax.set_ylabel(f'x(t + {tau}*tau)')
            ax.set_zlabel(f'x(t + {2*tau}*tau)')
            ax.set_title(f'Phase Space Plot (3D): {os.path.basename(audio_filepath)} ({label})')
        
        plt.tight_layout()

        # output_dir ya apunta a la subcarpeta específica (Normal o Pathol)
        # por ejemplo: .../output/Phase_Space_Plots/Normal
        #os.makedirs(output_dir, exist_ok=True) 
        
        # Generar nombre único con prefijo según la clase
        base_name = os.path.splitext(os.path.basename(audio_filepath))[0]
        prefix = 'n_' if label == 'Normal' else 'p_'
        ps_filename = os.path.join(output_dir, f"{prefix}{base_name}_ps.png")
        plt.savefig(ps_filename, dpi=200)
        plt.close(fig)

    except ValueError as ve:
        warnings.warn(f"Error (Valor Inválido) al generar Phase Space Plot para {audio_filepath}: {ve}")
    except Exception as e:
        warnings.warn(f"Error al generar Phase Space Plot para {audio_filepath}: {e}")
