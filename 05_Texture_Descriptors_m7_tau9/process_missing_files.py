#!/usr/bin/env python3
"""
Script para procesar archivos específicos faltantes sin recalcular todo.

Este script procesa solo los archivos especificados y los agrega a los
CSVs existentes sin modificar el manifest ni triggerar recálculo completo.
"""

import os
import sys
import pandas as pd
import numpy as np
from PIL import Image
import logging

# Agregar directorio padre al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from descriptors import create_descriptor
from utils.image_loader import ImageLoader

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Archivos faltantes
MISSING_FILES = [
    'n_adga_rp_pure.png',
    'n_ega_rp_pure.png'
]

# Descriptores a procesar
DESCRIPTORS_TO_PROCESS = ['glcm', 'lbp', 'wavelet']

def find_image_path(filename):
    """Encuentra la ruta completa de una imagen."""
    for directory in [config.RP_INPUT_NORMAL_DIR, config.RP_INPUT_PATHOL_DIR]:
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            label = 'Normal' if 'Normal' in directory else 'Pathol'
            return path, label
    return None, None

def process_file_with_descriptor(descriptor_name, image_path, filename, label):
    """Procesa un archivo con un descriptor específico."""
    try:
        # Configuración del descriptor
        desc_config = config.DEFAULT_DESCRIPTORS.get(descriptor_name, {}).copy()
        desc_config.pop('enabled', None)  # Remover enabled si existe

        # Crear descriptor
        descriptor = create_descriptor(descriptor_name, **desc_config)

        # Cargar imagen
        image_loader = ImageLoader(
            target_dtype=config.IMAGE_TARGET_DTYPE,
            normalize_range=config.IMAGE_NORMALIZE_RANGE,
            min_size=config.IMAGE_MIN_SIZE,
            max_size=config.IMAGE_MAX_SIZE
        )

        image = image_loader.load_image(image_path)
        if image is None:
            logger.error(f"No se pudo cargar imagen: {image_path}")
            return None

        # Extraer características
        logger.info(f"  Extrayendo {descriptor_name} de {filename}...")
        features = descriptor.extract_with_validation(image)

        # Agregar metadata
        features['filename'] = filename
        features['label'] = label

        return features

    except Exception as e:
        logger.error(f"Error procesando {filename} con {descriptor_name}: {e}")
        return None

def append_to_csv(descriptor_name, features_dict):
    """Agrega características al CSV existente del descriptor."""
    descriptor_dir = os.path.join(config.OUTPUT_FEATURES_DIR, 'by_descriptor', descriptor_name)
    csv_path = os.path.join(descriptor_dir, 'features.csv')

    if not os.path.exists(csv_path):
        logger.error(f"CSV no existe para {descriptor_name}: {csv_path}")
        return False

    try:
        # Cargar CSV existente
        existing_df = pd.read_csv(csv_path)
        logger.info(f"  CSV existente: {len(existing_df)} filas")

        # Crear DataFrame con nuevas características
        new_df = pd.DataFrame([features_dict])

        # Combinar
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)

        # Guardar
        combined_df.to_csv(csv_path, index=False)
        logger.info(f"  ✓ CSV actualizado: {len(combined_df)} filas totales")

        return True

    except Exception as e:
        logger.error(f"Error agregando a CSV de {descriptor_name}: {e}")
        return False

def main():
    """Procesa los archivos faltantes."""
    logger.info("=== PROCESANDO ARCHIVOS FALTANTES ===")
    logger.info(f"Archivos a procesar: {MISSING_FILES}")
    logger.info(f"Descriptores: {DESCRIPTORS_TO_PROCESS}")
    logger.info("")

    results = {
        'processed': 0,
        'failed': 0,
        'details': []
    }

    for filename in MISSING_FILES:
        logger.info(f"\n--- Procesando: {filename} ---")

        # Encontrar imagen
        image_path, label = find_image_path(filename)

        if image_path is None:
            logger.error(f"✗ Archivo no encontrado: {filename}")
            results['failed'] += 1
            results['details'].append({'file': filename, 'status': 'not_found'})
            continue

        logger.info(f"Encontrado en: {image_path} (Label: {label})")

        file_success = True

        # Procesar con cada descriptor
        for descriptor_name in DESCRIPTORS_TO_PROCESS:
            logger.info(f"\nDescriptor: {descriptor_name}")

            # Extraer características
            features = process_file_with_descriptor(
                descriptor_name, image_path, filename, label
            )

            if features is None:
                file_success = False
                continue

            # Agregar al CSV
            if not append_to_csv(descriptor_name, features):
                file_success = False

        if file_success:
            logger.info(f"✓ {filename} procesado exitosamente")
            results['processed'] += 1
            results['details'].append({'file': filename, 'status': 'success'})
        else:
            logger.error(f"✗ {filename} falló en uno o más descriptores")
            results['failed'] += 1
            results['details'].append({'file': filename, 'status': 'partial_failure'})

    # Resumen final
    logger.info("\n" + "=" * 50)
    logger.info("RESUMEN FINAL")
    logger.info("=" * 50)
    logger.info(f"Procesados exitosamente: {results['processed']}/{len(MISSING_FILES)}")
    logger.info(f"Fallidos: {results['failed']}/{len(MISSING_FILES)}")
    logger.info("")

    for detail in results['details']:
        status_symbol = "✓" if detail['status'] == 'success' else "✗"
        logger.info(f"{status_symbol} {detail['file']}: {detail['status']}")

    logger.info("\n¡Proceso completado!")

    return results['failed'] == 0

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
