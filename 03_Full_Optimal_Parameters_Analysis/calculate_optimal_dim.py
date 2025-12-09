# project_root/04_Optimal_Parameters_Analysis/calculate_optimal_dim.py

import numpy as np
import matplotlib.pyplot as plt
import os

# Adaptado de la lógica de FNN, simplificado para claridad.
# Una implementación completa de FNN es bastante compleja.
# Esta es una versión simplificada del concepto.
def _takens_embedding_for_fnn(series, dim, tau):
    """Realiza el embedding de Takens para FNN."""
    N = len(series)
    if N < (dim - 1) * tau + 1:
        return np.array([]) # No se puede embedir
    
    embedded_series = np.zeros((N - (dim - 1) * tau, dim))
    for i in range(dim):
        embedded_series[:, i] = series[i * tau : N - (dim - 1 - i) * tau]
    return embedded_series


def calculate_false_nearest_neighbors(series, tau, dim_max, r_tolerance=15.0, a_tolerance=2.0):
    """
    Calcula el porcentaje de Falsos Vecinos Más Cercanos (FNN) para diferentes dimensiones.
    Esta es una implementación simplificada para fines educativos.
    Para una implementación robusta de FNN, se recomienda una librería como nolds.
    
    Parametros:
        series (np.array): La serie temporal de entrada.
        tau (int): El retraso de tiempo ya determinado (ej. del análisis de MI/ACF).
        dim_max (int): La dimensión máxima de embedding a probar.
        r_tolerance (float): Criterio R de Kennel et al. (distancia relativa).
        a_tolerance (float): Criterio A de Kennel et al. (distancia absoluta).
    """
    fnn_percentages = []
    
    # Iterar sobre las dimensiones de embedding
    for current_dim in range(1, dim_max + 1):
        if len(series) < (current_dim - 1) * tau + 1:
            fnn_percentages.append(100.0) # Si no se puede embedir, consideramos 100% FNN
            continue

        # 1. Embeber la serie en la dimensión actual (d)
        embedded_d = _takens_embedding_for_fnn(series, current_dim, tau)
        if embedded_d.shape[0] == 0:
            fnn_percentages.append(100.0)
            continue
            
        # 2. Encontrar el vecino más cercano para cada punto en la dimensión d
        #    Usaremos un enfoque simple de distancia euclidiana bruta.
        #    Esto es computacionalmente costoso para series muy largas.
        num_points_d = embedded_d.shape[0]
        distances_d = np.zeros(num_points_d)
        nearest_neighbors_idx = np.zeros(num_points_d, dtype=int)

        for i in range(num_points_d):
            # Calcular distancias desde el punto i a todos los demás puntos
            diffs = embedded_d - embedded_d[i]
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            
            # Excluir la distancia a sí mismo (que es 0)
            dists[i] = np.inf 
            
            # Encontrar el índice del vecino más cercano
            nearest_neighbors_idx[i] = np.argmin(dists)
            distances_d[i] = dists[nearest_neighbors_idx[i]]
        
        # 3. Embeber la serie en la dimensión siguiente (d+1)
        next_dim = current_dim + 1
        embedded_d_plus_1 = _takens_embedding_for_fnn(series, next_dim, tau)

        # Si la serie embebida en d+1 es más corta, solo consideramos los puntos que existen en ambas.
        # Esto es crucial para comparar el mismo par de puntos en ambas dimensiones.
        num_points_d_plus_1 = embedded_d_plus_1.shape[0]
        num_common_points = min(num_points_d, num_points_d_plus_1)

        if num_common_points == 0:
            fnn_percentages.append(100.0)
            continue

        # 4. Calcular el porcentaje de FNN
        false_neighbors_count = 0
        total_neighbors_checked = 0
        
        for i in range(num_common_points):
            # El vecino más cercano de P_i en d es P_j (donde j = nearest_neighbors_idx[i])
            j = nearest_neighbors_idx[i]

            # Asegurarse de que j también esté dentro de los límites de num_common_points
            if j >= num_common_points:
                continue # Este par no se puede comparar en d+1

            # Distancia entre P_i y P_j en la dimensión d
            dist_d_ij = distances_d[i]

            # Distancia entre P_i y P_j en la dimensión d+1
            diff_d_plus_1 = embedded_d_plus_1[i] - embedded_d_plus_1[j]
            dist_d_plus_1_ij = np.sqrt(np.sum(diff_d_plus_1**2))
            
            # Criterio de FNN (Kennel et al.)
            # Si el vecino más cercano en d está muy lejos en d+1 (criterio R)
            if dist_d_ij > 0 and (dist_d_plus_1_ij / dist_d_ij) > r_tolerance:
                false_neighbors_count += 1
            # O si la distancia en d+1 es simplemente muy grande (criterio A)
            elif dist_d_plus_1_ij > a_tolerance * np.std(series): # Normalizar por la desviación estándar de la serie original
                 false_neighbors_count += 1
            
            total_neighbors_checked += 1
        
        if total_neighbors_checked > 0:
            fnn_percentage = (false_neighbors_count / total_neighbors_checked) * 100
        else:
            fnn_percentage = 100.0 # No se pudieron comparar vecinos

        fnn_percentages.append(fnn_percentage)

    return np.array(fnn_percentages)


def plot_dim_analysis(fnn_values, dim_max, fnn_threshold, filename_prefix, output_dir):
    """
    Genera y guarda el gráfico de Falsos Vecinos Más Cercanos (FNN).
    """
    dims = np.arange(1, dim_max + 1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dims, fnn_values, marker='o', linestyle='-', color='purple')
    ax.set_title('Análisis de Dimensión Óptima (Falsos Vecinos Más Cercanos)')
    ax.set_xlabel('Dimensión de Embedding (D)')
    ax.set_ylabel('Porcentaje de Falsos Vecinos Más Cercanos (%)')
    ax.grid(True)
    
    # Marcar el umbral
    ax.axhline(y=fnn_threshold * 100, color='r', linestyle='--', label=f'Umbral: {fnn_threshold*100:.0f}%')
    
    # Intentar identificar la dimensión óptima
    optimal_dim = None
    for i, fnn_percent in enumerate(fnn_values):
        if fnn_percent <= fnn_threshold * 100:
            optimal_dim = dims[i]
            break
    
    if optimal_dim is not None:
        ax.axvline(x=optimal_dim, color='blue', linestyle='-.', label=f'Dim. Óptima Sugerida: D={optimal_dim}')
    else:
        print("Advertencia: No se encontró una dimensión donde FNN caiga por debajo del umbral.")

    ax.legend()
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{filename_prefix}_dim_analysis.png"))
    plt.close(fig)

    return optimal_dim # Devuelve la dimensión sugerida
