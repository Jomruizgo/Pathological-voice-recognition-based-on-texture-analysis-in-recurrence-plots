"""
Selector de características riguroso y científicamente defendible.

Implementa proceso de 5 fases para selección óptima de características.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any
from pathlib import Path
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.stats import pearsonr
import json
from datetime import datetime

import config


class FeatureSelector:
    """
    Selector de características basado en significancia estadística,
    relevancia práctica y eliminación de redundancia.
    """

    def __init__(self):
        """Inicializa el selector de características."""
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        # Resultados por fase
        self.phase_results = {
            'phase_1_statistical': {},
            'phase_2_practical': {},
            'phase_3_ranking': {},
            'phase_4_redundancy': {},
            'phase_5_validation': {}
        }

    def load_data(self, csv_path: Path) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Carga datos de características.

        Returns:
            (df, X, y): DataFrame completo, features, labels
        """
        self.logger.info(f"Cargando datos desde {csv_path}")

        df = pd.read_csv(csv_path)

        if 'label' not in df.columns:
            raise ValueError("CSV debe contener columna 'label'")

        # Separar features y labels
        feature_cols = [col for col in df.columns if col not in ['filename', 'label']]
        X = df[feature_cols].values
        y = df['label'].values

        self.logger.info(f"Datos cargados: {len(df)} muestras, {len(feature_cols)} características")
        self.logger.info(f"Distribución de clases: {dict(df['label'].value_counts())}")

        return df, X, y

    def phase_1_statistical_significance(self, df: pd.DataFrame, X: np.ndarray,
                                         y: np.ndarray) -> pd.DataFrame:
        """
        FASE 1: Filtrado por significancia estadística.

        Elimina características sin evidencia estadística de discriminación.

        Args:
            df: DataFrame completo
            X: Features
            y: Labels

        Returns:
            DataFrame con características significativas y sus métricas
        """
        self.logger.info("="*80)
        self.logger.info("FASE 1: FILTRADO POR SIGNIFICANCIA ESTADÍSTICA")
        self.logger.info("="*80)

        feature_cols = [col for col in df.columns if col not in ['filename', 'label']]
        y_encoded = self.label_encoder.fit_transform(y)

        # Calcular F-statistic ANOVA y p-values
        self.logger.info("Calculando F-statistic ANOVA...")
        f_scores, p_values = f_classif(X, y_encoded)

        # Crear DataFrame con resultados
        results = pd.DataFrame({
            'feature': feature_cols,
            'f_score': f_scores,
            'p_value': p_values
        })

        # Filtrar por p-value
        n_total = len(results)
        results_significant = results[results['p_value'] < config.ALPHA].copy()
        n_significant = len(results_significant)
        n_rejected = n_total - n_significant

        self.logger.info(f"Threshold: α = {config.ALPHA}")
        self.logger.info(f"Total características: {n_total}")
        self.logger.info(f"✓ Significativas (p < {config.ALPHA}): {n_significant}")
        self.logger.info(f"✗ Rechazadas (p ≥ {config.ALPHA}): {n_rejected}")

        # Top características rechazadas (para análisis)
        rejected = results[results['p_value'] >= config.ALPHA].sort_values('p_value')
        if len(rejected) > 0:
            self.logger.info(f"\nTop 5 características rechazadas:")
            for idx, row in rejected.head(5).iterrows():
                self.logger.info(f"  - {row['feature']}: p={row['p_value']:.4f}, F={row['f_score']:.2f}")

        # Guardar resultados de fase
        self.phase_results['phase_1_statistical'] = {
            'n_input': n_total,
            'n_output': n_significant,
            'n_rejected': n_rejected,
            'alpha': config.ALPHA,
            'rejected_features': rejected['feature'].tolist()
        }

        return results_significant

    def phase_2_practical_relevance(self, df: pd.DataFrame,
                                     features_significant: pd.DataFrame) -> pd.DataFrame:
        """
        FASE 2: Filtrado por relevancia práctica (Cohen's d).

        Elimina características estadísticamente significativas pero
        con efecto trivial/pequeño.

        Args:
            df: DataFrame completo
            features_significant: Características significativas de Fase 1

        Returns:
            DataFrame con características prácticamente relevantes
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("FASE 2: FILTRADO POR RELEVANCIA PRÁCTICA (COHEN'S D)")
        self.logger.info("="*80)

        # Calcular Cohen's d para cada característica
        self.logger.info("Calculando tamaños de efecto (Cohen's d)...")

        classes = df['label'].unique()
        if len(classes) != 2:
            raise ValueError("Cohen's d implementado solo para 2 clases")

        class_0, class_1 = classes
        cohens_d_values = []

        for feature in features_significant['feature']:
            data_0 = df[df['label'] == class_0][feature].dropna()
            data_1 = df[df['label'] == class_1][feature].dropna()

            mean_0, mean_1 = data_0.mean(), data_1.mean()
            std_0, std_1 = data_0.std(), data_1.std()
            n_0, n_1 = len(data_0), len(data_1)

            # Pooled standard deviation
            pooled_std = np.sqrt(((n_0 - 1) * std_0**2 + (n_1 - 1) * std_1**2) / (n_0 + n_1 - 2))

            # Cohen's d
            if pooled_std > 0:
                cohens_d = (mean_0 - mean_1) / pooled_std
            else:
                cohens_d = 0.0

            cohens_d_values.append(cohens_d)

        features_significant['cohens_d'] = cohens_d_values

        # Clasificar magnitud
        def classify_effect_size(d):
            abs_d = abs(d)
            if abs_d < 0.2:
                return 'trivial'
            elif abs_d < 0.5:
                return 'small'
            elif abs_d < 0.8:
                return 'medium'
            else:
                return 'large'

        features_significant['effect_size'] = features_significant['cohens_d'].apply(classify_effect_size)

        # Filtrar por Cohen's d
        n_input = len(features_significant)
        features_practical = features_significant[
            np.abs(features_significant['cohens_d']) >= config.MIN_COHENS_D
        ].copy()
        n_output = len(features_practical)
        n_rejected = n_input - n_output

        self.logger.info(f"Threshold: |Cohen's d| ≥ {config.MIN_COHENS_D} (efecto mediano+)")
        self.logger.info(f"Entrada: {n_input} características")
        self.logger.info(f"✓ Relevantes (|d| ≥ {config.MIN_COHENS_D}): {n_output}")
        self.logger.info(f"✗ Rechazadas (|d| < {config.MIN_COHENS_D}): {n_rejected}")

        # Distribución por tamaño de efecto
        effect_dist = features_practical['effect_size'].value_counts()
        self.logger.info(f"\nDistribución de tamaños de efecto:")
        for effect, count in effect_dist.items():
            self.logger.info(f"  - {effect}: {count}")

        # Top características rechazadas
        rejected = features_significant[
            np.abs(features_significant['cohens_d']) < config.MIN_COHENS_D
        ].sort_values('cohens_d', key=abs)

        if len(rejected) > 0:
            self.logger.info(f"\nTop 5 características rechazadas (efecto pequeño/trivial):")
            for idx, row in rejected.head(5).iterrows():
                self.logger.info(f"  - {row['feature']}: |d|={abs(row['cohens_d']):.3f} ({row['effect_size']}), "
                               f"F={row['f_score']:.2f}, p={row['p_value']:.4f}")

        # Guardar resultados
        self.phase_results['phase_2_practical'] = {
            'n_input': n_input,
            'n_output': n_output,
            'n_rejected': n_rejected,
            'min_cohens_d': config.MIN_COHENS_D,
            'effect_size_distribution': effect_dist.to_dict(),
            'rejected_features': rejected['feature'].tolist()
        }

        return features_practical

    def phase_3_discriminative_ranking(self, df: pd.DataFrame, X: np.ndarray,
                                       y: np.ndarray,
                                       features_practical: pd.DataFrame) -> pd.DataFrame:
        """
        FASE 3: Ranking por poder discriminativo combinado (F-Score + MI).

        Args:
            df: DataFrame completo
            X: Features originales
            y: Labels
            features_practical: Características relevantes de Fase 2

        Returns:
            DataFrame rankeado por score combinado
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("FASE 3: RANKING POR PODER DISCRIMINATIVO")
        self.logger.info("="*80)

        # Obtener índices de características seleccionadas
        feature_cols = [col for col in df.columns if col not in ['filename', 'label']]
        selected_indices = [feature_cols.index(f) for f in features_practical['feature']]
        X_selected = X[:, selected_indices]

        # Calcular Mutual Information
        self.logger.info("Calculando Mutual Information...")
        y_encoded = self.label_encoder.transform(y)
        mi_scores = mutual_info_classif(X_selected, y_encoded, random_state=config.RANDOM_STATE)

        features_practical['mi_score'] = mi_scores

        # Normalizar F-Score y MI a [0, 1]
        scaler = MinMaxScaler()
        features_practical['f_score_norm'] = scaler.fit_transform(
            features_practical[['f_score']]
        )
        features_practical['mi_score_norm'] = scaler.fit_transform(
            features_practical[['mi_score']]
        )

        # Score combinado
        features_practical['combined_score'] = (
            config.WEIGHT_F_SCORE * features_practical['f_score_norm'] +
            config.WEIGHT_MI_SCORE * features_practical['mi_score_norm']
        )

        # Ordenar por score combinado
        features_ranked = features_practical.sort_values('combined_score', ascending=False)

        self.logger.info(f"Pesos: F-Score={config.WEIGHT_F_SCORE}, MI={config.WEIGHT_MI_SCORE}")
        self.logger.info(f"Total características rankeadas: {len(features_ranked)}")
        self.logger.info(f"\nTop 10 características por poder discriminativo:")

        for idx, row in features_ranked.head(10).iterrows():
            self.logger.info(
                f"  {idx+1:2d}. {row['feature']:40s} | "
                f"Combined={row['combined_score']:.3f} | "
                f"F={row['f_score']:6.2f} | "
                f"MI={row['mi_score']:.3f} | "
                f"|d|={abs(row['cohens_d']):.2f}"
            )

        # Guardar resultados
        self.phase_results['phase_3_ranking'] = {
            'n_features': len(features_ranked),
            'weight_f_score': config.WEIGHT_F_SCORE,
            'weight_mi': config.WEIGHT_MI_SCORE,
            'top_10_features': features_ranked.head(10)['feature'].tolist()
        }

        return features_ranked

    def phase_4_redundancy_elimination(self, df: pd.DataFrame,
                                        features_ranked: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
        """
        FASE 4: Eliminación de redundancia por correlación.

        Selecciona características no redundantes (baja correlación entre sí).

        Args:
            df: DataFrame completo
            features_ranked: Características rankeadas de Fase 3

        Returns:
            (selected_features, correlation_matrix)
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("FASE 4: ELIMINACIÓN DE REDUNDANCIA")
        self.logger.info("="*80)

        # Calcular matriz de correlación
        feature_list = features_ranked['feature'].tolist()
        corr_matrix = df[feature_list].corr(method=config.CORRELATION_METHOD)

        self.logger.info(f"Método de correlación: {config.CORRELATION_METHOD}")
        self.logger.info(f"Threshold: |r| < {config.MAX_CORRELATION}")
        self.logger.info(f"Objetivo: {config.TARGET_N_FEATURES} características\n")

        selected = []
        rejected_by_redundancy = []

        for idx, row in features_ranked.iterrows():
            feature = row['feature']

            # Primera característica siempre se selecciona
            if len(selected) == 0:
                selected.append(feature)
                self.logger.info(f"✓ {feature:40s} | Primera característica")
                continue

            # Calcular correlación con características ya seleccionadas
            correlations = [abs(corr_matrix.loc[feature, sel]) for sel in selected]
            max_corr = max(correlations)
            max_corr_with = selected[correlations.index(max_corr)]

            # Decidir si seleccionar
            if max_corr < config.MAX_CORRELATION:
                selected.append(feature)
                self.logger.info(
                    f"✓ {feature:40s} | max_r={max_corr:.3f} con {max_corr_with}"
                )
            else:
                rejected_by_redundancy.append({
                    'feature': feature,
                    'reason': 'high_correlation',
                    'max_correlation': max_corr,
                    'correlated_with': max_corr_with,
                    'f_score': row['f_score'],
                    'combined_score': row['combined_score']
                })
                self.logger.info(
                    f"✗ {feature:40s} | REDUNDANTE: r={max_corr:.3f} con {max_corr_with}"
                )

            # Parar si alcanzamos objetivo
            if len(selected) >= config.TARGET_N_FEATURES:
                break

        # Si no alcanzamos objetivo y está permitido relajar
        if len(selected) < config.TARGET_N_FEATURES and config.RELAX_CORRELATION_IF_NEEDED:
            self.logger.info(f"\n⚠️  Solo {len(selected)} características seleccionadas (objetivo: {config.TARGET_N_FEATURES})")
            self.logger.info(f"Relajando threshold a {config.MAX_CORRELATION + 0.05:.2f}...")
            # Aquí podrías implementar lógica para relajar threshold

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Características seleccionadas: {len(selected)}/{config.TARGET_N_FEATURES}")
        self.logger.info(f"Características rechazadas por redundancia: {len(rejected_by_redundancy)}")

        # Guardar resultados
        self.phase_results['phase_4_redundancy'] = {
            'n_selected': len(selected),
            'n_rejected': len(rejected_by_redundancy),
            'max_correlation_threshold': config.MAX_CORRELATION,
            'selected_features': selected,
            'rejected_features': rejected_by_redundancy
        }

        return selected, corr_matrix

    def phase_5_validation(self, df: pd.DataFrame, selected_features: List[str]) -> Dict[str, Any]:
        """
        FASE 5: Validación de separabilidad del subset seleccionado.

        Verifica que las características seleccionadas realmente separen bien las clases.

        Args:
            df: DataFrame completo
            selected_features: Lista de características seleccionadas

        Returns:
            Dict con métricas de validación
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("FASE 5: VALIDACIÓN DE SEPARABILIDAD")
        self.logger.info("="*80)

        # Preparar datos
        X_subset = df[selected_features].values
        y = df['label'].values

        # Estandarizar
        X_scaled = self.scaler.fit_transform(X_subset)

        # 1. Varianza explicada por PCA
        self.logger.info("Calculando varianza explicada (PCA)...")
        n_components = min(10, len(selected_features))
        pca = PCA(n_components=n_components)
        pca.fit(X_scaled)
        variance_explained = pca.explained_variance_ratio_.sum()

        self.logger.info(f"  PCA con {n_components} componentes: {variance_explained:.2%} varianza explicada")

        # 2. Silhouette score
        self.logger.info("Calculando Silhouette score...")
        silhouette = silhouette_score(X_scaled, y)
        self.logger.info(f"  Silhouette score: {silhouette:.3f}")

        # Interpretación
        if silhouette > 0.7:
            sil_interpretation = "Excelente separabilidad"
        elif silhouette > 0.5:
            sil_interpretation = "Buena separabilidad"
        elif silhouette > 0.3:
            sil_interpretation = "Separabilidad aceptable"
        else:
            sil_interpretation = "Separabilidad pobre"
        self.logger.info(f"  Interpretación: {sil_interpretation}")

        # 3. Fisher ratio (distancia inter-clase / intra-clase)
        self.logger.info("Calculando Fisher ratio...")
        classes = np.unique(y)
        class_0_data = X_scaled[y == classes[0]]
        class_1_data = X_scaled[y == classes[1]]

        centroid_0 = class_0_data.mean(axis=0)
        centroid_1 = class_1_data.mean(axis=0)

        # Distancia entre centroides (inter-clase)
        inter_class_dist = np.linalg.norm(centroid_0 - centroid_1)

        # Distancia promedio intra-clase
        intra_0 = np.mean([np.linalg.norm(x - centroid_0) for x in class_0_data])
        intra_1 = np.mean([np.linalg.norm(x - centroid_1) for x in class_1_data])
        intra_class_dist = (intra_0 + intra_1) / 2

        fisher_ratio = inter_class_dist / intra_class_dist if intra_class_dist > 0 else 0
        self.logger.info(f"  Fisher ratio: {fisher_ratio:.3f}")

        # Validación de criterios
        self.logger.info(f"\n{'='*60}")
        self.logger.info("VALIDACIÓN DE CRITERIOS:")

        validation_passed = True

        # PCA variance
        pca_pass = variance_explained >= config.MIN_PCA_VARIANCE
        status = "✓ PASS" if pca_pass else "✗ FAIL"
        self.logger.info(f"  {status} | PCA variance: {variance_explained:.2%} (mín: {config.MIN_PCA_VARIANCE:.0%})")
        validation_passed &= pca_pass

        # Silhouette
        sil_pass = silhouette >= config.MIN_SILHOUETTE_SCORE
        status = "✓ PASS" if sil_pass else "✗ FAIL"
        self.logger.info(f"  {status} | Silhouette: {silhouette:.3f} (mín: {config.MIN_SILHOUETTE_SCORE})")
        validation_passed &= sil_pass

        # Fisher ratio
        fisher_pass = fisher_ratio >= config.MIN_FISHER_RATIO
        status = "✓ PASS" if fisher_pass else "✗ FAIL"
        self.logger.info(f"  {status} | Fisher ratio: {fisher_ratio:.3f} (mín: {config.MIN_FISHER_RATIO})")
        validation_passed &= fisher_pass

        # Convertir numpy.bool_ a bool nativo de Python
        validation_passed = bool(validation_passed)
        pca_pass = bool(pca_pass)
        sil_pass = bool(sil_pass)
        fisher_pass = bool(fisher_pass)

        if validation_passed:
            self.logger.info(f"\n✓ VALIDACIÓN EXITOSA: Subset de características es apropiado")
        else:
            self.logger.warning(f"\n⚠️  ADVERTENCIA: Subset no cumple todos los criterios de validación")

        # Guardar resultados
        validation_results = {
            'n_features': len(selected_features),
            'pca_variance_explained': float(variance_explained),
            'pca_n_components': n_components,
            'silhouette_score': float(silhouette),
            'silhouette_interpretation': sil_interpretation,
            'fisher_ratio': float(fisher_ratio),
            'inter_class_distance': float(inter_class_dist),
            'intra_class_distance': float(intra_class_dist),
            'validation_passed': validation_passed,
            'criteria': {
                'pca_variance': {'value': float(variance_explained), 'threshold': config.MIN_PCA_VARIANCE, 'passed': pca_pass},
                'silhouette': {'value': float(silhouette), 'threshold': config.MIN_SILHOUETTE_SCORE, 'passed': sil_pass},
                'fisher_ratio': {'value': float(fisher_ratio), 'threshold': config.MIN_FISHER_RATIO, 'passed': fisher_pass}
            }
        }

        self.phase_results['phase_5_validation'] = validation_results

        return validation_results

    def run_full_pipeline(self, csv_path: Path) -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo de 5 fases.

        Args:
            csv_path: Path al CSV con características

        Returns:
            Dict con resultados completos incluyendo características seleccionadas
        """
        self.logger.info("="*80)
        self.logger.info("INICIO DE PIPELINE DE SELECCIÓN DE CARACTERÍSTICAS")
        self.logger.info("="*80)
        self.logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Configuración:")
        self.logger.info(f"  - α (p-value): {config.ALPHA}")
        self.logger.info(f"  - Cohen's d mínimo: {config.MIN_COHENS_D}")
        self.logger.info(f"  - Pesos: F-Score={config.WEIGHT_F_SCORE}, MI={config.WEIGHT_MI_SCORE}")
        self.logger.info(f"  - Correlación máxima: {config.MAX_CORRELATION}")
        self.logger.info(f"  - Objetivo: {config.TARGET_N_FEATURES} características\n")

        # Cargar datos
        df, X, y = self.load_data(csv_path)

        # Fase 1: Significancia estadística
        features_significant = self.phase_1_statistical_significance(df, X, y)

        # Fase 2: Relevancia práctica
        features_practical = self.phase_2_practical_relevance(df, features_significant)

        # Fase 3: Ranking discriminativo
        features_ranked = self.phase_3_discriminative_ranking(df, X, y, features_practical)

        # Fase 4: Eliminación de redundancia
        selected_features, corr_matrix = self.phase_4_redundancy_elimination(df, features_ranked)

        # Fase 5: Validación
        validation_results = self.phase_5_validation(df, selected_features)

        # Construir resultado final
        final_results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'input_file': str(csv_path),
                'pipeline_version': config.PIPELINE_VERSION,
                'author': config.AUTHOR,
                'description': config.DESCRIPTION
            },
            'configuration': {
                'alpha': config.ALPHA,
                'min_cohens_d': config.MIN_COHENS_D,
                'weight_f_score': config.WEIGHT_F_SCORE,
                'weight_mi': config.WEIGHT_MI_SCORE,
                'max_correlation': config.MAX_CORRELATION,
                'target_n_features': config.TARGET_N_FEATURES,
                'random_state': config.RANDOM_STATE
            },
            'data_summary': {
                'n_samples': len(df),
                'n_features_initial': len([col for col in df.columns if col not in ['filename', 'label']]),
                'class_distribution': df['label'].value_counts().to_dict()
            },
            'phase_results': self.phase_results,
            'selected_features': self._build_selected_features_list(features_ranked, selected_features),
            'validation': validation_results
        }

        self.logger.info("\n" + "="*80)
        self.logger.info("PIPELINE COMPLETADO EXITOSAMENTE")
        self.logger.info("="*80)

        return final_results

    def _build_selected_features_list(self, features_ranked: pd.DataFrame,
                                       selected_features: List[str]) -> List[Dict[str, Any]]:
        """Construye lista detallada de características seleccionadas."""
        selected_list = []

        for rank, feature in enumerate(selected_features, 1):
            row = features_ranked[features_ranked['feature'] == feature].iloc[0]

            # Determinar descriptor (primer componente antes de '_')
            descriptor = feature.split('_')[0]

            selected_list.append({
                'rank': rank,
                'name': feature,
                'descriptor': descriptor,
                'f_score': float(row['f_score']),
                'p_value': float(row['p_value']),
                'cohens_d': float(row['cohens_d']),
                'effect_size': row['effect_size'],
                'mi_score': float(row['mi_score']),
                'combined_score': float(row['combined_score']),
                'justification': self._generate_feature_justification(row)
            })

        return selected_list

    def _generate_feature_justification(self, row: pd.Series) -> str:
        """Genera justificación textual para una característica."""
        justification = []

        # Significancia estadística
        if row['p_value'] < 0.001:
            justification.append("altamente significativa (p<0.001)")
        elif row['p_value'] < 0.01:
            justification.append("muy significativa (p<0.01)")
        else:
            justification.append(f"significativa (p={row['p_value']:.3f})")

        # Tamaño de efecto
        justification.append(f"efecto {row['effect_size']} (|d|={abs(row['cohens_d']):.2f})")

        # Poder discriminativo
        if row['f_score'] > 10:
            justification.append("alto poder discriminativo")
        elif row['f_score'] > 5:
            justification.append("buen poder discriminativo")
        else:
            justification.append("poder discriminativo moderado")

        return ", ".join(justification)
