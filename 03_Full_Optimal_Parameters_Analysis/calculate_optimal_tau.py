# project_root/04_Optimal_Parameters_Analysis/calculate_optimal_tau.py

import numpy as np
from scipy.stats import entropy as shannon_entropy
import matplotlib.pyplot as plt
import os

# Función auxiliar para el binning (necesaria para Mutual Information de forma discreta)
# Adaptado de ejemplos comunes de cálculo de MI para series temporales
def _bin_series(series, num_bins=10):
    """Discretiza una serie temporal en un número de bins."""
    min_val = np.min(series)
    max_val = np.max(series)
    bins = np.linspace(min_val, max_val, num_bins + 1)
    return np.digitize(series, bins) - 1 # -1 para que los bins empiecen en 0

def calculate_mutual_information(series, tau_max, num_bins=10):
    """
    Calcula la Información Mutua para diferentes retrasos (tau).
    Basado en la estimación de MI para variables discretas.
    """
    mi_values = []
    
    # Discretizar la serie una vez
    binned_series = _bin_series(series, num_bins)
    
    # Calcular las probabilidades marginales para la serie original binned_series
    p_x = np.histogram(binned_series, bins=num_bins, density=True)[0]
    
    # Shannon Entropy H(X)
    h_x = shannon_entropy(p_x[p_x > 0]) # Solo considera bins con probabilidad > 0

    for tau in range(1, tau_max + 1):
        if len(binned_series) <= tau:
            mi_values.append(np.nan) # No se puede calcular MI si tau es demasiado grande
            continue
            
        # Series retrasadas
        x_t = binned_series[:-tau]
        x_t_plus_tau = binned_series[tau:]
        
        # Calcular las probabilidades conjuntas p(x, x_tau)
        hist_2d, _, _ = np.histogram2d(x_t, x_t_plus_tau, bins=num_bins, density=True)
        p_xy = hist_2d.flatten() # Aplanar para calcular entropía conjunta
        
        # Shannon Entropy H(X, Y)
        h_xy = shannon_entropy(p_xy[p_xy > 0])
        
        # Calcular H(Y) de forma similar a H(X), pero con la serie retrasada
        p_y = np.histogram(x_t_plus_tau, bins=num_bins, density=True)[0]
        h_y = shannon_entropy(p_y[p_y > 0])

        # MI(X,Y) = H(X) + H(Y) - H(X,Y)
        mi = h_x + h_y - h_xy
        mi_values.append(mi)
        
    return np.array(mi_values)


def calculate_autocorrelation(series, tau_max):
    """
    Calcula la función de autocorrelación (ACF) para diferentes retrasos (tau).
    """
    # La implementación de numpy.correlate ya maneja el padding y la normalización para autocorrelación
    # 'full' devuelve la correlación para todos los solapamientos posibles
    # Luego tomamos solo la mitad derecha para los retrasos positivos
    n = len(series)
    if n == 0:
        return np.array([])
    
    # Desviación estándar de la serie (para normalizar)
    std_dev = np.std(series)
    if std_dev == 0: # Si la serie es constante, la autocorrelación es 0 o 1 (en el retraso 0)
        return np.zeros(tau_max + 1) # O manejar esto como 1 en el retraso 0 y 0 en otros

    # Centrar la serie (restar la media)
    series_centered = series - np.mean(series)

    # Calcular autocorrelación
    # np.correlate(a, v, mode) -> (a * v')
    # Para autocorrelación, 'a' y 'v' son la misma serie
    autocorr = np.correlate(series_centered, series_centered, mode='full')
    
    # La correlación en modo 'full' tiene n-1+n-1 = 2n-1 puntos.
    # El centro (retraso 0) está en el índice n-1.
    # Queremos desde el retraso 0 hasta tau_max.
    autocorr = autocorr[n-1 : n-1 + tau_max + 1] # Incluye retraso 0
    
    # Normalizar por la varianza (o n * std_dev^2, o n * variance_of_uncentered_series)
    # Autocorrelación normalizada: R_k = E[(X_t - mu)(X_{t+k} - mu)] / sigma^2
    # La autocorrelación de np.correlate en modo 'full' es la autocovarianza no normalizada.
    # Normalizamos por la varianza en el retraso 0, que es autocorr[0]
    if autocorr[0] == 0: # Si la varianza es cero (serie constante)
        return np.zeros(tau_max + 1)
        
    autocorr = autocorr / autocorr[0] # Normaliza por la autocovarianza en retraso 0

    return autocorr[1:] # Excluir retraso 0, ya que Tau_max es para retrasos > 0


def find_first_local_minimum(values):
    """
    Encuentra el índice del primer mínimo local en una serie.
    Un mínimo local es donde el valor es menor que sus vecinos.
    """
    for i in range(1, len(values) - 1):
        if values[i] < values[i-1] and values[i] < values[i+1]:
            return i
    # Si no hay mínimo local, retornar el mínimo global
    return np.argmin(values)


def plot_tau_analysis(mi_values, acf_values, tau_max, filename_prefix, output_dir):
    """
    Genera y guarda gráficos de Información Mutua y Autocorrelación.
    """
    taus = np.arange(1, tau_max + 1) # Taus para el eje X

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle('Análisis para Determinación de Tau Óptimo')

    # Gráfico de Información Mutua
    ax1.plot(taus, mi_values, marker='o', linestyle='-', color='blue')
    ax1.set_title('Información Mutua')
    ax1.set_xlabel('Retraso (Tau)')
    ax1.set_ylabel('Información Mutua')
    ax1.grid(True)
    # Buscar el PRIMER MÍNIMO LOCAL (no el mínimo global)
    first_local_min_idx = find_first_local_minimum(mi_values)
    optimal_tau_mi = taus[first_local_min_idx]
    ax1.axvline(x=optimal_tau_mi, color='r', linestyle='--', label=f'Primer Mínimo Local: Tau={optimal_tau_mi}')
    ax1.legend()


    # Gráfico de Autocorrelación
    ax2.plot(taus, acf_values, marker='o', linestyle='-', color='green')
    ax2.set_title('Función de Autocorrelación')
    ax2.set_xlabel('Retraso (Tau)')
    ax2.set_ylabel('Autocorrelación')
    ax2.axhline(0, color='gray', linestyle='--') # Línea en 0 para identificar el primer cruce
    ax2.grid(True)
    # Opcional: Marcar el primer cruce por cero
    first_zero_cross_idx = np.where(np.diff(np.sign(acf_values)))[0]
    if first_zero_cross_idx.size > 0:
        optimal_tau_acf = taus[first_zero_cross_idx[0]]
        ax2.axvline(x=optimal_tau_acf, color='r', linestyle='--', label=f'Primer Cruce por Cero: Tau={optimal_tau_acf}')
        ax2.legend()
    else:
        print("Advertencia: No se encontró un cruce por cero en la función de autocorrelación dentro del rango probado.")


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajustar para el suptitle
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{filename_prefix}_tau_analysis.png"))
    plt.close(fig)

    # Devolver los Tau óptimos sugeridos (del primer mínimo LOCAL de MI y primer cruce por cero de ACF)
    # Esto es solo una sugerencia, la interpretación visual es clave.
    tau_suggestions = {}
    if mi_values.size > 0:
        # Usar primer mínimo LOCAL, no el mínimo global
        first_local_min_idx = find_first_local_minimum(mi_values)
        tau_suggestions['MI_min'] = taus[first_local_min_idx]
    if first_zero_cross_idx.size > 0:
        tau_suggestions['ACF_zero_cross'] = taus[first_zero_cross_idx[0]]

    return tau_suggestions
