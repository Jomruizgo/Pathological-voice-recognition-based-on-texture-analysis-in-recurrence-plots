#!/usr/bin/env python3
"""
Módulo de visualización para 07-C Classification.

Genera gráficos de:
- Matrices de confusión
- Curvas ROC
- Comparación de métricas entre modelos
- Gráficos de F1-Scores
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional

import config


class Visualizer:
    """Generador de visualizaciones para clasificación."""

    def __init__(self, output_dir: Path):
        """
        Inicializa el visualizador.

        Args:
            output_dir: Directorio donde guardar gráficos
        """
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / 'plots'
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)
        plt.style.use('seaborn-v0_8-darkgrid')

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                              model_name: str, class_names: List[str] = None) -> Path:
        """
        Genera matriz de confusión.

        Args:
            y_true: Etiquetas verdaderas
            y_pred: Predicciones
            model_name: Nombre del modelo
            class_names: Nombres de clases

        Returns:
            Path al archivo guardado
        """
        if class_names is None:
            class_names = ['Normal', 'Pathol']

        # Calcular matriz
        cm = confusion_matrix(y_true, y_pred)

        # Normalizar
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Crear figura
        fig, ax = plt.subplots(figsize=(10, 8))

        # Heatmap
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax, cbar_kws={'label': 'Proporción'})

        # Añadir conteos absolutos
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j + 0.5, i + 0.7, f'({cm[i, j]})',
                       ha='center', va='center', fontsize=10, color='gray')

        ax.set_title(f'Matriz de Confusión - {model_name}', fontsize=14, pad=20)
        ax.set_xlabel('Predicción', fontsize=12)
        ax.set_ylabel('Verdadero', fontsize=12)

        plt.tight_layout()

        # Guardar
        filename = f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
        save_path = self.plots_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Matriz de confusión guardada: {save_path}")
        return save_path

    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                      model_name: str) -> Path:
        """
        Genera curva ROC.

        Args:
            y_true: Etiquetas verdaderas
            y_proba: Probabilidades predichas
            model_name: Nombre del modelo

        Returns:
            Path al archivo guardado
        """
        # Calcular curva ROC
        fpr, tpr, thresholds = roc_curve(y_true, y_proba[:, 1] if y_proba.ndim == 2 else y_proba)
        roc_auc = auc(fpr, tpr)

        # Crear figura
        fig, ax = plt.subplots(figsize=(10, 8))

        # Curva ROC
        ax.plot(fpr, tpr, color='darkorange', lw=2,
               label=f'ROC curve (AUC = {roc_auc:.4f})')

        # Línea diagonal (clasificador aleatorio)
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
               label='Random classifier')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curve - {model_name}', fontsize=14, pad=20)
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Guardar
        filename = f'roc_curve_{model_name.lower().replace(" ", "_")}.png'
        save_path = self.plots_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Curva ROC guardada: {save_path}")
        return save_path

    def plot_metrics_comparison(self, results: Dict[str, Any]) -> Path:
        """
        Genera gráfico de comparación de métricas entre modelos.

        Args:
            results: Diccionario con resultados de todos los modelos

        Returns:
            Path al archivo guardado
        """
        # Extraer datos
        models = []
        metrics_data = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': []}

        for model_name, result in results.items():
            models.append(model_name)
            metrics_data['Accuracy'].append(result['accuracy'])
            metrics_data['Precision'].append(result['precision'])
            metrics_data['Recall'].append(result['recall'])
            metrics_data['F1-Score'].append(result['f1_score'])

        # Crear figura
        fig, ax = plt.subplots(figsize=(14, 8))

        x = np.arange(len(models))
        width = 0.2
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        # Barras para cada métrica
        for i, (metric_name, values) in enumerate(metrics_data.items()):
            ax.bar(x + i * width, values, width, label=metric_name, color=colors[i])

        ax.set_xlabel('Modelos', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Comparación de Métricas por Modelo', fontsize=14, pad=20)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.1)

        plt.tight_layout()

        # Guardar
        save_path = self.plots_dir / 'models_metrics_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Comparación de métricas guardada: {save_path}")
        return save_path

    def plot_f1_comparison(self, results: Dict[str, Any]) -> Path:
        """
        Genera gráfico de comparación de F1-Scores.

        Args:
            results: Diccionario con resultados de todos los modelos

        Returns:
            Path al archivo guardado
        """
        # Extraer datos
        models = list(results.keys())
        f1_scores = [result['f1_score'] for result in results.values()]

        # Ordenar por F1-Score descendente
        sorted_indices = np.argsort(f1_scores)[::-1]
        models = [models[i] for i in sorted_indices]
        f1_scores = [f1_scores[i] for i in sorted_indices]

        # Crear figura
        fig, ax = plt.subplots(figsize=(12, 7))

        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
        bars = ax.bar(models, f1_scores, color=colors)

        # Añadir valores sobre las barras
        for bar, score in zip(bars, f1_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                   f'{score:.4f}', ha='center', va='bottom',
                   fontweight='bold', fontsize=10)

        ax.set_xlabel('Modelos', fontsize=12)
        ax.set_ylabel('F1-Score', fontsize=12)
        ax.set_title('Comparación de F1-Scores (Ordenado)', fontsize=14, pad=20)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.1)

        plt.tight_layout()

        # Guardar
        save_path = self.plots_dir / 'models_f1_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Comparación de F1-Scores guardada: {save_path}")
        return save_path

    def plot_cv_scores(self, results: Dict[str, Any]) -> Path:
        """
        Genera gráfico de scores de validación cruzada.

        Args:
            results: Diccionario con resultados de todos los modelos

        Returns:
            Path al archivo guardado
        """
        # Extraer datos
        models = list(results.keys())
        cv_means = [result['cv_f1_mean'] for result in results.values()]
        cv_stds = [result['cv_f1_std'] for result in results.values()]

        # Crear figura
        fig, ax = plt.subplots(figsize=(12, 7))

        x = np.arange(len(models))
        ax.bar(x, cv_means, yerr=cv_stds, alpha=0.7, capsize=5,
              color='steelblue', error_kw={'elinewidth': 2, 'alpha': 0.8})

        # Añadir valores
        for i, (mean, std) in enumerate(zip(cv_means, cv_stds)):
            ax.text(i, mean + std + 0.02, f'{mean:.3f}±{std:.3f}',
                   ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Modelos', fontsize=12)
        ax.set_ylabel('CV F1-Score (mean ± std)', fontsize=12)
        ax.set_title('Validación Cruzada - F1-Scores', fontsize=14, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.1)

        plt.tight_layout()

        # Guardar
        save_path = self.plots_dir / 'models_cv_scores.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Gráfico de CV guardado: {save_path}")
        return save_path

    def plot_all_roc_curves(self, all_results: Dict[str, Any],
                           y_test: np.ndarray) -> Path:
        """
        Genera curvas ROC de todos los modelos en un solo gráfico.

        Args:
            all_results: Diccionario con resultados y probabilidades de todos los modelos
            y_test: Etiquetas verdaderas de test

        Returns:
            Path al archivo guardado
        """
        fig, ax = plt.subplots(figsize=(12, 9))

        colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

        for (model_name, result), color in zip(all_results.items(), colors):
            if 'y_proba' in result and result['y_proba'] is not None:
                y_proba = result['y_proba']

                # Calcular curva ROC
                fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1] if y_proba.ndim == 2 else y_proba)
                roc_auc = auc(fpr, tpr)

                ax.plot(fpr, tpr, color=color, lw=2,
                       label=f'{model_name} (AUC = {roc_auc:.3f})')

        # Línea diagonal
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random classifier')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - Comparación de Todos los Modelos', fontsize=14, pad=20)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Guardar
        save_path = self.plots_dir / 'all_models_roc_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Curvas ROC combinadas guardadas: {save_path}")
        return save_path
