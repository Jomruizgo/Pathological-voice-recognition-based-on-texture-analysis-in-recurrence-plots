#!/usr/bin/env python3
"""
Script para regenerar plots desde JSON de resultados existente.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix
import sys

# Agregar directorio padre al path para imports
sys.path.append(str(Path(__file__).parent))
import config

def plot_models_comparison(results: dict, output_dir: Path):
    """Genera gráfico de comparación de modelos para un subset."""
    models_data = results['models']

    # Extraer datos
    model_names = [m['model_name'].replace('_', ' ').title() for m in models_data]
    accuracies = [m['accuracy'] for m in models_data]
    precisions = [m['precision'] for m in models_data]
    recalls = [m['recall'] for m in models_data]
    f1_scores = [m['f1_score'] for m in models_data]

    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(model_names))
    width = 0.2

    ax.bar(x - 1.5*width, accuracies, width, label='Accuracy', alpha=0.8)
    ax.bar(x - 0.5*width, precisions, width, label='Precision', alpha=0.8)
    ax.bar(x + 0.5*width, recalls, width, label='Recall', alpha=0.8)
    ax.bar(x + 1.5*width, f1_scores, width, label='F1-Score', alpha=0.8)

    ax.set_xlabel('Modelo', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Comparación de Métricas por Modelo', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    output_file = output_dir / 'models_metrics_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Generado: {output_file.name}")

def plot_f1_comparison(results: dict, output_dir: Path):
    """Genera gráfico de F1-Scores."""
    models_data = sorted(results['models'], key=lambda x: x['f1_score'], reverse=True)

    model_names = [m['model_name'].replace('_', ' ').title() for m in models_data]
    f1_scores = [m['f1_score'] for m in models_data]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(model_names, f1_scores, alpha=0.7)

    # Colorear la barra del mejor modelo
    bars[0].set_color('green')
    bars[0].set_alpha(0.8)

    ax.set_xlabel('F1-Score', fontsize=12)
    ax.set_title('F1-Score por Modelo (ordenado)', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.grid(axis='x', alpha=0.3)

    # Añadir valores en las barras
    for i, (name, score) in enumerate(zip(model_names, f1_scores)):
        ax.text(score + 0.01, i, f'{score:.4f}', va='center', fontsize=10)

    plt.tight_layout()
    output_file = output_dir / 'models_f1_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Generado: {output_file.name}")

def plot_cv_scores(results: dict, output_dir: Path):
    """Genera gráfico de CV scores con barras de error."""
    models_data = results['models']

    model_names = [m['model_name'].replace('_', ' ').title() for m in models_data]
    cv_means = [m['cv_f1_mean'] for m in models_data]
    cv_stds = [m['cv_f1_std'] for m in models_data]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(model_names))

    ax.bar(x, cv_means, alpha=0.7, yerr=cv_stds, capsize=5)
    ax.set_xlabel('Modelo', fontsize=12)
    ax.set_ylabel('CV F1-Score (mean ± std)', fontsize=12)
    ax.set_title('Validación Cruzada: F1-Score por Modelo', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    output_file = output_dir / 'models_cv_scores.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Generado: {output_file.name}")

def plot_all_roc_curves(results: dict, output_dir: Path):
    """Genera gráfico con todas las curvas ROC."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Línea diagonal (clasificador aleatorio)
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC=0.50)', linewidth=1)

    # Plotear cada modelo
    for model in results['models']:
        fpr = model.get('roc_fpr', [])
        tpr = model.get('roc_tpr', [])
        roc_auc = model['roc_auc']

        if fpr and tpr:
            label = f"{model['model_name'].replace('_', ' ').title()} (AUC={roc_auc:.4f})"
            ax.plot(fpr, tpr, label=label, linewidth=2, alpha=0.8)

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Curvas ROC - Todos los Modelos', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / 'all_models_roc_curves.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Generado: {output_file.name}")

def main():
    # Cargar JSON más reciente
    results_dir = Path('output/metrics')
    json_files = list(results_dir.glob('results_*.json'))

    if not json_files:
        print("❌ No se encontraron archivos de resultados")
        return

    latest_json = max(json_files, key=lambda p: p.stat().st_mtime)
    print(f"\n📊 Regenerando plots desde: {latest_json.name}")

    with open(latest_json, 'r') as f:
        data = json.load(f)

    # Directorio de salida
    plots_dir = results_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # Usar el subset ALL_SELECTED que tiene todas las características
    subset = 'all_selected'
    if subset not in data['subsets']:
        print(f"❌ Subset '{subset}' no encontrado en JSON")
        return

    results = data['subsets'][subset]

    print(f"\n🎨 Generando plots comparativos para subset: {subset.upper()}")
    print(f"   Modelos: {len(results['models'])}")

    # Generar plots
    plot_models_comparison(results, plots_dir)
    plot_f1_comparison(results, plots_dir)
    plot_cv_scores(results, plots_dir)
    plot_all_roc_curves(results, plots_dir)

    print(f"\n✅ Plots regenerados exitosamente en: {plots_dir}")
    print(f"   Total: 4 plots actualizados con {len(results['models'])} modelos")

if __name__ == '__main__':
    main()
