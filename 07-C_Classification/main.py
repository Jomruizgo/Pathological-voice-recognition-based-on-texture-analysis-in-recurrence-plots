#!/usr/bin/env python3
"""
Módulo 07-C: Clasificación con Features Auto-Seleccionadas

Script principal que entrena modelos de clasificación usando las características
seleccionadas automáticamente por el módulo 06-C.

Uso:
    python main.py                    # Entrenar todos los modelos
    python main.py --model svm        # Entrenar solo SVM
    python main.py --verbose          # Con logging detallado
"""

import argparse
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json

# Modelos de clasificación
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# Métricas y validación
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

import config
from visualization import Visualizer


def setup_logging(verbose: bool = False):
    """Configura el sistema de logging."""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger(__name__)


def load_data(feature_list=None):
    """
    Carga datos y extrae características especificadas.

    Args:
        feature_list: Lista de características a usar. Si None, usa SELECTED_FEATURES

    Returns:
        X, y, feature_names
    """
    logger = logging.getLogger(__name__)

    # Cargar CSV
    csv_path = config.INPUT_DATA_DIR / config.INPUT_FEATURES_FILE

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV no encontrado: {csv_path}")

    logger.info(f"Cargando datos desde: {csv_path.name}")
    df = pd.read_csv(csv_path)

    # Usar feature_list o las seleccionadas por defecto
    features_to_use = feature_list if feature_list is not None else config.SELECTED_FEATURES

    # Verificar que todas las características existen
    missing_features = [f for f in features_to_use if f not in df.columns]
    if missing_features:
        raise ValueError(f"Características no encontradas en CSV: {missing_features}")

    # Extraer features y labels
    X = df[features_to_use].values
    y = df['label'].values

    logger.info(f"Datos cargados: {len(df)} muestras, {len(features_to_use)} características")
    logger.info(f"Distribución de clases: {dict(pd.Series(y).value_counts())}")

    return X, y, features_to_use


def train_model(model_name, X_train, X_test, y_train, y_test):
    """
    Entrena y evalúa un modelo.

    Returns:
        dict con métricas, predicciones y probabilidades
    """
    logger = logging.getLogger(__name__)

    # Crear modelo según configuración
    model_configs = {
        'logistic_regression': LogisticRegression(**config.MODEL_CONFIGS['logistic_regression']),
        'naive_bayes': GaussianNB(**config.MODEL_CONFIGS['naive_bayes']),
        'knn': KNeighborsClassifier(**config.MODEL_CONFIGS['knn']),
        'decision_tree': DecisionTreeClassifier(**config.MODEL_CONFIGS['decision_tree']),
        'svm': SVC(**config.MODEL_CONFIGS['svm']),
        'random_forest': RandomForestClassifier(**config.MODEL_CONFIGS['random_forest']),
        'neural_network': MLPClassifier(**config.MODEL_CONFIGS['neural_network']),
        'xgboost': XGBClassifier(**config.MODEL_CONFIGS['xgboost'])
    }

    if model_name not in model_configs:
        raise ValueError(f"Modelo no reconocido: {model_name}")

    model = model_configs[model_name]

    logger.info(f"\n{'='*60}")
    logger.info(f"Entrenando: {model_name.upper()}")
    logger.info(f"{'='*60}")

    # Entrenar
    start_time = datetime.now()
    model.fit(X_train, y_train)
    train_time = (datetime.now() - start_time).total_seconds()

    # Predecir
    start_time = datetime.now()
    y_pred = model.predict(X_test)
    predict_time = (datetime.now() - start_time).total_seconds()

    # Probabilidades (si están disponibles)
    y_proba = None
    try:
        y_proba = model.predict_proba(X_test)
    except:
        logger.warning(f"  {model_name} no soporta predict_proba")

    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # ROC-AUC (solo si el modelo tiene predict_proba)
    roc_auc = None
    if y_proba is not None:
        try:
            # Encode labels
            le = LabelEncoder()
            y_test_encoded = le.fit_transform(y_test)
            roc_auc = roc_auc_score(y_test_encoded, y_proba[:, 1])
        except:
            logger.warning(f"  No se pudo calcular ROC-AUC para {model_name}")

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)

    # Validación cruzada
    logger.info(f"Realizando validación cruzada ({config.VALIDATION_CONFIG['cv_folds']} folds)...")
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=config.VALIDATION_CONFIG['cv_folds'],
        scoring='f1_weighted'
    )

    # Log resultados
    logger.info(f"\nRESULTADOS ({model_name}):")
    logger.info(f"  Accuracy:  {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  F1-Score:  {f1:.4f}")
    if roc_auc:
        logger.info(f"  ROC-AUC:   {roc_auc:.4f}")
    logger.info(f"  CV F1 (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    logger.info(f"  Tiempo entrenamiento: {train_time:.3f}s")
    logger.info(f"  Tiempo predicción: {predict_time:.3f}s")

    logger.info(f"\nMatriz de Confusión:")
    logger.info(f"  {cm}")

    # Retornar métricas
    return {
        'model_name': model_name,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc) if roc_auc else None,
        'cv_f1_mean': float(cv_scores.mean()),
        'cv_f1_std': float(cv_scores.std()),
        'train_time': float(train_time),
        'predict_time': float(predict_time),
        'confusion_matrix': cm.tolist(),
        'n_train': len(X_train),
        'n_test': len(X_test),
        # Datos para visualización
        'y_pred': y_pred,
        'y_proba': y_proba
    }


def save_results(results, output_dir):
    """Guarda resultados en JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Función auxiliar para limpiar resultados de modelos
    def clean_model_result(model_result):
        return {k: v for k, v in model_result.items()
                if k not in ['y_pred', 'y_proba']}

    # Guardar JSON (sin y_pred y y_proba que son arrays)
    results_to_save = {}
    for key, value in results.items():
        if key == 'models':
            # Estructura antigua: lista directa de modelos
            results_to_save[key] = [clean_model_result(m) for m in value]
        elif key == 'subsets':
            # Estructura nueva: subsets con modelos dentro
            results_to_save[key] = {}
            for subset_name, subset_data in value.items():
                results_to_save[key][subset_name] = {
                    'n_features': subset_data['n_features'],
                    'features': subset_data['features'],
                    'models': [clean_model_result(m) for m in subset_data['models']],
                    'best_model': clean_model_result(subset_data['best_model']) if subset_data['best_model'] else None
                }
        elif key == 'best_overall':
            # Limpiar best_overall también
            if value:
                results_to_save[key] = clean_model_result(value)
            else:
                results_to_save[key] = value
        else:
            results_to_save[key] = value

    output_file = output_dir / f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

    with open(output_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)

    print(f"\n✓ Resultados guardados en: {output_file}")

    return output_file


def generate_summary(all_results):
    """Genera resumen de todos los modelos."""
    logger = logging.getLogger(__name__)

    logger.info(f"\n{'='*80}")
    logger.info("RESUMEN COMPARATIVO DE MODELOS")
    logger.info(f"{'='*80}\n")

    # Tabla comparativa
    header = f"{'Modelo':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'CV F1':>12}"
    logger.info(header)
    logger.info("-" * len(header))

    for result in sorted(all_results, key=lambda x: x['f1_score'], reverse=True):
        logger.info(
            f"{result['model_name']:<20} "
            f"{result['accuracy']:>10.4f} "
            f"{result['precision']:>10.4f} "
            f"{result['recall']:>10.4f} "
            f"{result['f1_score']:>10.4f} "
            f"{result['cv_f1_mean']:>6.4f}±{result['cv_f1_std']:.4f}"
        )

    # Mejor modelo
    best_model = max(all_results, key=lambda x: x['f1_score'])
    logger.info(f"\n🏆 MEJOR MODELO: {best_model['model_name'].upper()}")
    logger.info(f"   F1-Score: {best_model['f1_score']:.4f}")
    logger.info(f"   Accuracy: {best_model['accuracy']:.4f}")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Clasificación con características auto-seleccionadas por 06-C"
    )

    parser.add_argument(
        '--model',
        choices=['logistic_regression', 'naive_bayes', 'knn', 'decision_tree',
                 'svm', 'random_forest', 'neural_network', 'all'],
        default='all',
        help='Modelo a entrenar (por defecto: all)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Logging detallado'
    )

    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='No generar gráficos (solo métricas)'
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.verbose)

    print("\n" + "="*80)
    print("MÓDULO 07-C: CLASIFICACIÓN CON FEATURES AUTO-SELECCIONADAS")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    try:
        # Obtener subsets de características a evaluar
        feature_subsets = config.FEATURE_SUBSETS if hasattr(config, 'FEATURE_SUBSETS') else {
            'all_selected': config.SELECTED_FEATURES
        }

        logger.info(f"\n✓ Evaluando {len(feature_subsets)} subsets de características:")
        for name, features in feature_subsets.items():
            logger.info(f"  - {name}: {len(features)} características")

        # Almacenar resultados por subset
        results_by_subset = {}

        # Iterar sobre cada subset
        for subset_name, subset_features in feature_subsets.items():
            print(f"\n{'='*80}")
            print(f"EVALUANDO SUBSET: {subset_name.upper()} ({len(subset_features)} características)")
            print(f"{'='*80}\n")

            # Cargar datos con el subset específico
            X, y, feature_names = load_data(feature_list=subset_features)

            # Estandarizar características
            logger.info("Estandarizando características...")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Encode labels
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            class_names = le.classes_.tolist()

            # Split train/test (IMPORTANTE: usar mismo random_state para comparación justa)
            logger.info(f"Dividiendo datos: {int((1-config.VALIDATION_CONFIG['test_size'])*100)}% train, {int(config.VALIDATION_CONFIG['test_size']*100)}% test")
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded,
                test_size=config.VALIDATION_CONFIG['test_size'],
                random_state=config.VALIDATION_CONFIG['random_state'],
                stratify=y_encoded if config.VALIDATION_CONFIG['stratify'] else None
            )

            logger.info(f"Train: {len(X_train)} muestras | Test: {len(X_test)} muestras")

            # Entrenar modelos
            models_to_train = (
                ['logistic_regression', 'naive_bayes', 'knn', 'decision_tree',
                 'svm', 'random_forest', 'neural_network', 'xgboost']
                if args.model == 'all'
                else [args.model]
            )

            all_results = []

            for model_name in models_to_train:
                try:
                    result = train_model(model_name, X_train, X_test, y_train, y_test)
                    all_results.append(result)
                except Exception as e:
                    logger.error(f"Error entrenando {model_name}: {e}")
                    continue

            # Generar resumen para este subset
            if len(all_results) > 1:
                generate_summary(all_results)

            # Guardar resultados de este subset
            results_by_subset[subset_name] = {
                'n_features': len(feature_names),
                'features': feature_names,
                'models': all_results,
                'best_model': max(all_results, key=lambda x: x['f1_score']) if all_results else None
            }

        # Generar resumen comparativo entre subsets
        print(f"\n{'='*80}")
        print("COMPARACIÓN ENTRE SUBSETS DE CARACTERÍSTICAS")
        print(f"{'='*80}\n")

        logger.info(f"{'Subset':<20} {'N Features':>12} {'Mejor Modelo':<20} {'F1-Score':>10} {'Accuracy':>10} {'ROC-AUC':>10}")
        logger.info("-" * 92)

        best_overall = None
        best_overall_f1 = 0

        for subset_name, subset_data in results_by_subset.items():
            best = subset_data['best_model']
            if best:
                logger.info(
                    f"{subset_name:<20} {subset_data['n_features']:>12} "
                    f"{best['model_name']:<20} "
                    f"{best['f1_score']:>10.4f} "
                    f"{best['accuracy']:>10.4f} "
                    f"{best['roc_auc'] if best['roc_auc'] else 0:>10.4f}"
                )

                if best['f1_score'] > best_overall_f1:
                    best_overall_f1 = best['f1_score']
                    best_overall = {'subset': subset_name, **best}

        if best_overall:
            print(f"\n{'='*80}")
            print(f"🏆 MEJOR CONFIGURACIÓN GLOBAL")
            print(f"{'='*80}")
            print(f"  Subset: {best_overall['subset']}")
            print(f"  Modelo: {best_overall['model_name'].upper()}")
            print(f"  F1-Score: {best_overall['f1_score']:.4f}")
            print(f"  Accuracy: {best_overall['accuracy']:.4f}")
            if best_overall['roc_auc']:
                print(f"  ROC-AUC: {best_overall['roc_auc']:.4f}")
            print(f"{'='*80}")

        # Generar visualizaciones para el mejor subset
        if not args.no_plots:
            # Usar el MEJOR subset (con mayor F1-Score) para visualizaciones
            if best_overall:
                best_subset_results = results_by_subset[best_overall['subset']]['models']
                logger.info("\n" + "="*80)
                logger.info(f"GENERANDO VISUALIZACIONES (BASADO EN MEJOR SUBSET: {best_overall['subset'].upper()})")
                logger.info("="*80)

                visualizer = Visualizer(config.METRICS_DIR)

                # Matrices de confusión individuales
                for result in best_subset_results:
                    visualizer.plot_confusion_matrix(
                        y_test, result['y_pred'],
                        result['model_name'], class_names
                    )

                # Curvas ROC individuales
                for result in best_subset_results:
                    if result['y_proba'] is not None:
                        visualizer.plot_roc_curve(
                            y_test, result['y_proba'],
                            result['model_name']
                        )

                # Gráficos comparativos
                if len(best_subset_results) > 1:
                    results_dict = {r['model_name']: r for r in best_subset_results}

                    visualizer.plot_metrics_comparison(results_dict)
                    visualizer.plot_f1_comparison(results_dict)
                    visualizer.plot_cv_scores(results_dict)
                    visualizer.plot_all_roc_curves(results_dict, y_test)

                logger.info(f"\n✓ Gráficos guardados en: {visualizer.plots_dir}")

        # Guardar resultados completos
        output = {
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(X),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'class_names': class_names,
            'subsets': results_by_subset,
            'best_overall': best_overall,
            'selection_config': config.SELECTION_CONFIG if hasattr(config, 'SELECTION_CONFIG') else None
        }

        save_results(output, config.METRICS_DIR)

        print("\n" + "="*80)
        print("✓ CLASIFICACIÓN COMPLETADA EXITOSAMENTE")
        print("="*80)

        return 0

    except Exception as e:
        logger.exception(f"Error durante ejecución: {e}")
        print(f"\n❌ ERROR: {str(e)}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
