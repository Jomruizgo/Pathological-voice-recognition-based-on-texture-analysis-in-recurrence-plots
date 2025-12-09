#!/usr/bin/env python3
"""
Módulo 06-C: Selección Rigurosa de Características

Script principal que ejecuta el pipeline de 5 fases para selección
científicamente defendible de características.

Uso:
    python main.py                    # Pipeline completo
    python main.py --verbose         # Con logging detallado
    python main.py --validate-only   # Solo validar features existentes
"""

import argparse
import sys
import logging
import json
from pathlib import Path
from datetime import datetime

import config
from feature_selector import FeatureSelector


def setup_logging(verbose: bool = False):
    """Configura el sistema de logging."""
    level = logging.DEBUG if verbose else logging.INFO

    # Crear directorio de logs
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Configurar logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging configurado: nivel={logging.getLevelName(level)}")

    return logger


def save_results_to_json(results: dict, output_path: Path):
    """Guarda resultados en formato JSON."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✓ Resultados guardados en: {output_path}")


def generate_markdown_report(results: dict, output_path: Path):
    """Genera reporte en formato Markdown."""

    report = f"""# Reporte de Selección de Características

**Fecha**: {results['metadata']['timestamp']}
**Pipeline**: Versión {results['metadata']['pipeline_version']}
**Autor**: {results['metadata']['author']}

---

## Resumen Ejecutivo

- **Características iniciales**: {results['data_summary']['n_features_initial']}
- **Características seleccionadas**: {len(results['selected_features'])}
- **Muestras totales**: {results['data_summary']['n_samples']}
- **Distribución de clases**: {results['data_summary']['class_distribution']}

---

## Configuración del Pipeline

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| α (p-value) | {results['configuration']['alpha']} | Significancia estadística |
| Cohen's d mínimo | {results['configuration']['min_cohens_d']} | Relevancia práctica |
| Peso F-Score | {results['configuration']['weight_f_score']} | Peso en ranking combinado |
| Peso MI | {results['configuration']['weight_mi']} | Peso en ranking combinado |
| Correlación máxima | {results['configuration']['max_correlation']} | Threshold redundancia |

---

## Resultados por Fase

### Fase 1: Significancia Estadística
- **Entrada**: {results['phase_results']['phase_1_statistical']['n_input']} características
- **Salida**: {results['phase_results']['phase_1_statistical']['n_output']} características
- **Rechazadas**: {results['phase_results']['phase_1_statistical']['n_rejected']} (p ≥ {results['configuration']['alpha']})

### Fase 2: Relevancia Práctica
- **Entrada**: {results['phase_results']['phase_2_practical']['n_input']} características
- **Salida**: {results['phase_results']['phase_2_practical']['n_output']} características
- **Rechazadas**: {results['phase_results']['phase_2_practical']['n_rejected']} (|Cohen's d| < {results['configuration']['min_cohens_d']})
- **Distribución de efectos**: {results['phase_results']['phase_2_practical']['effect_size_distribution']}

### Fase 3: Ranking Discriminativo
- **Top 10 características**:
"""

    for i, feat in enumerate(results['phase_results']['phase_3_ranking']['top_10_features'], 1):
        report += f"  {i}. `{feat}`\n"

    report += f"""
### Fase 4: Eliminación de Redundancia
- **Seleccionadas**: {results['phase_results']['phase_4_redundancy']['n_selected']} características
- **Rechazadas**: {results['phase_results']['phase_4_redundancy']['n_rejected']} (correlación > {results['configuration']['max_correlation']})

### Fase 5: Validación de Separabilidad
- **PCA Varianza Explicada**: {results['validation']['pca_variance_explained']:.2%} (mín: {results['validation']['criteria']['pca_variance']['threshold']:.0%})
- **Silhouette Score**: {results['validation']['silhouette_score']:.3f} (mín: {results['validation']['criteria']['silhouette']['threshold']})
- **Fisher Ratio**: {results['validation']['fisher_ratio']:.3f} (mín: {results['validation']['criteria']['fisher_ratio']['threshold']})
- **Validación**: {'✓ PASADA' if results['validation']['validation_passed'] else '✗ FALLIDA'}

---

## Características Seleccionadas (Top {len(results['selected_features'])})

| Rank | Característica | Descriptor | F-Score | p-value | Cohen's d | MI | Combined Score |
|------|---------------|------------|---------|---------|-----------|----|--------------------|
"""

    for feat in results['selected_features']:
        report += (f"| {feat['rank']} | `{feat['name']}` | {feat['descriptor']} | "
                  f"{feat['f_score']:.2f} | {feat['p_value']:.4f} | {feat['cohens_d']:.2f} | "
                  f"{feat['mi_score']:.3f} | {feat['combined_score']:.3f} |\n")

    report += f"""
---

## Justificaciones

"""

    for feat in results['selected_features']:
        report += f"### {feat['rank']}. {feat['name']}\n"
        report += f"**Descriptor**: {feat['descriptor']}  \n"
        report += f"**Justificación**: {feat['justification']}  \n\n"

    report += f"""
---

## Interpretación de Resultados

### Validación de Separabilidad

**PCA ({results['validation']['pca_variance_explained']:.2%} varianza)**:
- {'✓' if results['validation']['criteria']['pca_variance']['passed'] else '✗'} Las características capturan {'suficiente' if results['validation']['criteria']['pca_variance']['passed'] else 'insuficiente'} información para discriminar entre clases.

**Silhouette Score ({results['validation']['silhouette_score']:.3f})**:
- Interpretación: {results['validation']['silhouette_interpretation']}
- {'✓' if results['validation']['criteria']['silhouette']['passed'] else '✗'} Separabilidad {'aceptable' if results['validation']['criteria']['silhouette']['passed'] else 'insuficiente'} entre clases.

**Fisher Ratio ({results['validation']['fisher_ratio']:.3f})**:
- Distancia inter-clase: {results['validation']['inter_class_distance']:.3f}
- Distancia intra-clase: {results['validation']['intra_class_distance']:.3f}
- {'✓' if results['validation']['criteria']['fisher_ratio']['passed'] else '✗'} Clases {'bien separadas' if results['validation']['criteria']['fisher_ratio']['passed'] else 'pobremente separadas'} en el espacio de características.

---

## Conclusiones

Este subset de {len(results['selected_features'])} características fue seleccionado mediante un proceso riguroso de 5 fases que garantiza:

1. **Significancia estadística**: Todas las características tienen p < {results['configuration']['alpha']}
2. **Relevancia práctica**: Todas tienen |Cohen's d| ≥ {results['configuration']['min_cohens_d']} (efecto mediano+)
3. **Poder discriminativo**: Rankeadas por F-Score (70%) + MI (30%)
4. **No redundancia**: Correlación máxima entre seleccionadas < {results['configuration']['max_correlation']}
5. **Separabilidad validada**: {'Cumple' if results['validation']['validation_passed'] else 'No cumple'} todos los criterios de validación

---

**Generado automáticamente por el pipeline 06-C**
*Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"✓ Reporte Markdown generado: {output_path}")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Selección rigurosa de características (Pipeline 5 fases)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Logging detallado (DEBUG level)'
    )

    parser.add_argument(
        '--input',
        type=str,
        help='Path al CSV de entrada (por defecto: configurado en config.py)'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Path para el JSON de salida (por defecto: configurado en config.py)'
    )

    args = parser.parse_args()

    # Configurar logging
    logger = setup_logging(args.verbose)

    print("\n" + "="*80)
    print("MÓDULO 06-C: SELECCIÓN RIGUROSA DE CARACTERÍSTICAS")
    print("="*80)
    print(f"Pipeline de 5 fases para selección científicamente defendible")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    try:
        # Determinar archivo de entrada
        if args.input:
            input_path = Path(args.input)
        else:
            input_path = config.INPUT_FEATURES_DIR / config.INPUT_FEATURES_FILE

        if not input_path.exists():
            logger.error(f"Archivo de entrada no encontrado: {input_path}")
            print(f"\n❌ ERROR: Archivo no encontrado: {input_path}")
            print(f"   Ejecuta primero el módulo 05 para generar características")
            return 1

        # Crear directorio de salida
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Ejecutar pipeline
        selector = FeatureSelector()
        results = selector.run_full_pipeline(input_path)

        # Guardar resultados
        output_json = Path(args.output) if args.output else config.FEATURE_RANKING_JSON
        save_results_to_json(results, output_json)

        # Generar reporte Markdown
        report_md = config.OUTPUT_DIR / 'selection_report.md'
        generate_markdown_report(results, report_md)

        # Resumen final
        print("\n" + "="*80)
        print("RESUMEN FINAL")
        print("="*80)
        print(f"✓ Características iniciales: {results['data_summary']['n_features_initial']}")
        print(f"✓ Características seleccionadas: {len(results['selected_features'])}")
        print(f"✓ Validación: {'PASADA' if results['validation']['validation_passed'] else 'FALLIDA'}")
        print(f"\nArchivos generados:")
        print(f"  - JSON: {output_json}")
        print(f"  - Markdown: {report_md}")
        print(f"  - Log: {config.LOG_FILE}")
        print("\n" + "="*80)
        print("✓ PIPELINE COMPLETADO EXITOSAMENTE")
        print("="*80)

        return 0

    except Exception as e:
        logger.exception("Error durante ejecución del pipeline")
        print(f"\n❌ ERROR: {str(e)}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
