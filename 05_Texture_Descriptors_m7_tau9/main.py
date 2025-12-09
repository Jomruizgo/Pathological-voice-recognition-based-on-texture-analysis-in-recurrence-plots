#!/usr/bin/env python3
"""
Módulo principal para extracción modular de descriptores de textura.

Este script implementa un sistema evolutivo que permite calcular descriptores
de forma independiente y combinar características bajo demanda.

SISTEMA MODULAR:
- Cada descriptor se calcula y almacena por separado
- Añade nuevos descriptores sin recalcular los existentes
- Combina cualquier subset de características automáticamente
- Detecta cambios de configuración y reutiliza cuando es posible

Uso:
    python main.py                          # Calcula descriptores faltantes
    python main.py --descriptors glcm lbp  # Solo descriptores específicos
    python main.py --gui                    # Interfaz gráfica modular
    python main.py --list-descriptors      # Ver descriptores disponibles

FLUJO EVOLUTIVO:
1. Primera vez: python main.py --descriptors glcm lbp
2. Más tarde: python main.py --descriptors glcm lbp gabor (solo calcula gabor)
3. El sistema automáticamente combina todas las características disponibles
"""

import argparse
import sys
import os
import json
import logging
from typing import List, Optional
import signal
from datetime import datetime

# Imports del proyecto
import config
from descriptors import list_available_descriptors, get_descriptor_info
from core.modular_pipeline import ModularPipeline


def setup_logging(level: str = 'INFO') -> logging.Logger:
    """
    Configura el sistema de logging.
    
    Args:
        level (str): Nivel de logging ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        
    Returns:
        logging.Logger: Logger configurado
    """
    # Crear directorio de logs si no existe
    log_dir = os.path.dirname(config.LOG_FILE)
    os.makedirs(log_dir, exist_ok=True)
    
    # Configurar formato
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configurar logging tanto para archivo como consola
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler(config.LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Sistema de logging iniciado")
    return logger


def show_available_descriptors():
    """Muestra información sobre descriptores disponibles."""
    available = list_available_descriptors()
    
    print("\n=== DESCRIPTORES DE TEXTURA DISPONIBLES ===")
    print(f"Total: {len(available)} descriptores\n")
    
    for desc_name in sorted(available):
        try:
            info = get_descriptor_info(desc_name)
            print(f"📊 {desc_name.upper()}")
            print(f"   Clase: {info['class']}")
            print(f"   Habilitado por defecto: {'Sí' if info['enabled_by_default'] else 'No'}")
            print(f"   Descripción: {info['description'][:100]}...")
            if info['parameters']:
                print(f"   Parámetros: {len(info['parameters'])} configurables")
            print()
        except Exception as e:
            print(f"❌ Error obteniendo info de {desc_name}: {str(e)}")


def show_checkpoint_info():
    """Muestra información sobre el estado del sistema modular."""
    print("\n=== INFORMACIÓN DEL SISTEMA MODULAR ===")
    
    manifest_path = os.path.join(config.OUTPUT_FEATURES_DIR, 'manifest.json')
    
    if not os.path.exists(manifest_path):
        print("❌ No hay descriptores calculados")
        print("   Ejecuta una extracción para calcular descriptores")
        return
    
    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        descriptors = manifest.get('descriptors', {})
        if descriptors:
            print(f"✅ {len(descriptors)} descriptores disponibles:")
            for desc_name, desc_info in descriptors.items():
                print(f"\n   📊 {desc_name}")
                print(f"      - Imágenes: {desc_info['total_images']}")
                print(f"      - Características: {desc_info['feature_count']}")
                print(f"      - Fecha: {desc_info['computed_date'][:10]}")
                
                # Verificar checkpoints parciales
                checkpoint_path = os.path.join(config.OUTPUT_FEATURES_DIR, 'by_descriptor', 
                                             desc_name, 'partial_checkpoint.json')
                if os.path.exists(checkpoint_path):
                    print(f"      - ⚠️ Checkpoint parcial disponible")
        else:
            print("❌ No hay descriptores calculados")
            
    except Exception as e:
        print(f"❌ Error verificando estado: {str(e)}")


def validate_descriptors(descriptor_names: List[str]) -> bool:
    """
    Valida que los descriptores especificados existan.
    
    Args:
        descriptor_names (List[str]): Lista de nombres de descriptores
        
    Returns:
        bool: True si todos son válidos
    """
    available = list_available_descriptors()
    invalid = set(descriptor_names) - set(available)
    
    if invalid:
        print(f"❌ Descriptores no válidos: {', '.join(invalid)}")
        print(f"   Disponibles: {', '.join(sorted(available))}")
        return False
    
    return True


def setup_signal_handlers(pipeline: Optional[ModularPipeline]):
    """
    Configura manejadores de señales para interrupción grácil.
    
    Args:
        pipeline: Pipeline de extracción (puede ser None)
    """
    def signal_handler(signum, frame):
        signal_name = signal.Signals(signum).name
        print(f"\n⚠️  Señal {signal_name} recibida")
        
        if pipeline and pipeline.is_running:
            print("🛑 Deteniendo pipeline...")
            pipeline.stop()
            print("✅ Pipeline detenido. Los checkpoints se han guardado.")
        
        print("👋 Saliendo...")
        sys.exit(0)
    
    # Registrar manejadores para interrupciones comunes
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination


def run_extraction(args) -> bool:
    """
    Ejecuta la extracción de características.
    
    Args:
        args: Argumentos de línea de comandos
        
    Returns:
        bool: True si fue exitoso
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Determinar qué descriptores usar
        if args.descriptors:
            descriptors_to_use = args.descriptors
        else:
            # Usar descriptores habilitados por defecto
            descriptors_to_use = [
                name for name, config_dict in config.DEFAULT_DESCRIPTORS.items()
                if config_dict.get('enabled', True)
            ]
        
        if not descriptors_to_use:
            print("❌ No hay descriptores seleccionados para extracción")
            return False
        
        print(f"🚀 Iniciando extracción con descriptores: {', '.join(descriptors_to_use)}")
        
        # Crear configuraciones de descriptores
        descriptor_configs = {}
        for desc_name in descriptors_to_use:
            if desc_name in config.DEFAULT_DESCRIPTORS:
                # Usar configuración por defecto, excluyendo 'enabled'
                desc_config = config.DEFAULT_DESCRIPTORS[desc_name].copy()
                desc_config.pop('enabled', None)
                descriptor_configs[desc_name] = desc_config
        
        # Crear pipeline modular
        pipeline = ModularPipeline(
            descriptors=descriptors_to_use,
            descriptor_configs=descriptor_configs,
            batch_size=args.batch_size,
            n_jobs=args.n_jobs,
            enable_checkpoints=not args.no_checkpoints
        )
        
        # Configurar manejadores de señales
        setup_signal_handlers(pipeline)
        
        # Ejecutar cálculo de descriptores modular
        results = pipeline.compute_descriptors()
        
        # Mostrar resultados modulares
        total_computed = results['summary']['total_computed']
        total_reused = results['summary']['total_reused'] 
        total_failed = results['summary']['total_failed']
        
        print("\n🎉 CÁLCULO MODULAR COMPLETADO")
        print(f"   🆕 Descriptores calculados: {total_computed}")
        print(f"   ♻️  Descriptores reutilizados: {total_reused}")
        print(f"   ❌ Descriptores fallidos: {total_failed}")
        
        # Mostrar detalle por descriptor
        if results['computed']:
            print("\n📊 Descriptores calculados:")
            for desc_name, desc_result in results['computed'].items():
                print(f"   • {desc_name}: {desc_result['images_processed']} imágenes, "
                      f"{desc_result['features_extracted']} características")
        
        if results['reused']:
            print("\n♻️  Descriptores reutilizados:")
            for desc_name in results['reused']:
                print(f"   • {desc_name}")
        
        if results['failed']:
            print("\n❌ Descriptores fallidos:")
            for desc_name, error_info in results['failed'].items():
                print(f"   • {desc_name}: {error_info.get('error', 'Error desconocido')}")
        
        # Ofrecer combinar características si hay múltiples descriptores calculados
        available_descriptors = pipeline.get_available_descriptors()
        if len(available_descriptors) > 1:
            print(f"\n🔗 Combinando todas las características disponibles...")
            combined_df = pipeline.combine_features()
            if combined_df is not None:
                print(f"   ✅ CSV combinado generado: {len(combined_df)} imágenes, {combined_df.shape[1]-2} características")
            
        return total_failed == 0  # Éxito si no hay fallidos
            
    except KeyboardInterrupt:
        print("\n⚠️  Extracción interrumpida por el usuario")
        return False
    except Exception as e:
        logger.error(f"Error durante extracción: {str(e)}")
        print(f"\n❌ Error durante extracción: {str(e)}")
        return False


def run_gui():
    """Ejecuta la interfaz gráfica."""
    try:
        import subprocess
        import sys
        from setup_streamlit import setup_streamlit_config
        
        # Configurar Streamlit
        setup_streamlit_config()
        
        # Ejecutar Streamlit
        gui_script = os.path.join(os.path.dirname(__file__), 'gui.py')
        cmd = [sys.executable, '-m', 'streamlit', 'run', gui_script]
        
        print("🖥️  Iniciando interfaz gráfica...")
        print("   La GUI se abrirá en tu navegador web")
        print("   Presiona Ctrl+C para detener")
        
        subprocess.run(cmd)
        
    except ImportError as e:
        if 'streamlit' in str(e).lower():
            print("❌ Streamlit no está instalado")
            print("   Instala con: pip install streamlit")
            print("   O ejecuta desde la raíz del proyecto: pip install -r requirements.txt")
        else:
            print(f"❌ Error de importación: {str(e)}")
    except Exception as e:
        print(f"❌ Error iniciando GUI: {str(e)}")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Extractor de descriptores de textura para Recurrence Plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python main.py                                # Extracción por defecto
  python main.py --descriptors glcm lbp         # Solo GLCM y LBP
  python main.py --batch-size 5 --n-jobs 4     # Configuración personalizada
  python main.py --gui                          # Interfaz gráfica
  python main.py --resume                       # Reanudar desde checkpoint
  python main.py --list-descriptors             # Ver descriptores disponibles
        """
    )
    
    # Argumentos principales
    parser.add_argument(
        '--descriptors', 
        nargs='*', 
        help='Descriptores a usar (por defecto: todos los habilitados)'
    )
    
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=5,
        help='Tamaño de lote para checkpoints (por defecto: 5)'
    )
    
    parser.add_argument(
        '--n-jobs', 
        type=int, 
        default=1,
        help='Número de trabajos paralelos (por defecto: 1, -1 = todos los cores)'
    )
    
    # Opciones de control
    parser.add_argument(
        '--no-checkpoints', 
        action='store_true',
        help='Deshabilitar sistema de checkpoints'
    )
    
    parser.add_argument(
        '--resume', 
        action='store_true',
        help='Reanudar desde último checkpoint'
    )
    
    parser.add_argument(
        '--gui', 
        action='store_true',
        help='Ejecutar interfaz gráfica'
    )
    
    # Opciones de información
    parser.add_argument(
        '--list-descriptors', 
        action='store_true',
        help='Mostrar descriptores disponibles'
    )
    
    parser.add_argument(
        '--checkpoint-info', 
        action='store_true',
        help='Mostrar información de checkpoints'
    )
    
    parser.add_argument(
        '--log-level', 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Nivel de logging (por defecto: INFO)'
    )
    
    args = parser.parse_args()
    
    # Configurar logging
    logger = setup_logging(args.log_level)
    
    # Mostrar información si se solicita
    if args.list_descriptors:
        show_available_descriptors()
        return
    
    if args.checkpoint_info:
        show_checkpoint_info()
        return
    
    # Ejecutar GUI si se solicita
    if args.gui:
        run_gui()
        return
    
    # Validar descriptores si se especificaron
    if args.descriptors and not validate_descriptors(args.descriptors):
        sys.exit(1)
    
    # Mostrar configuración
    print("\n=== CONFIGURACIÓN DE EXTRACCIÓN ===")
    if args.descriptors:
        print(f"Descriptores: {', '.join(args.descriptors)}")
    else:
        enabled_descriptors = [
            name for name, config_dict in config.DEFAULT_DESCRIPTORS.items()
            if config_dict.get('enabled', True)
        ]
        print(f"Descriptores (por defecto): {', '.join(enabled_descriptors)}")
    
    print(f"Tamaño de lote: {args.batch_size}")
    print(f"Trabajos paralelos: {args.n_jobs}")
    print(f"Checkpoints: {'Deshabilitados' if args.no_checkpoints else 'Habilitados'}")
    print(f"Nivel de logging: {args.log_level}")
    print()
    
    # Si se solicita reanudar, mostrar info de checkpoint
    if args.resume:
        show_checkpoint_info()
        print()
    
    # Ejecutar extracción
    success = run_extraction(args)
    
    # Salir con código apropiado
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()