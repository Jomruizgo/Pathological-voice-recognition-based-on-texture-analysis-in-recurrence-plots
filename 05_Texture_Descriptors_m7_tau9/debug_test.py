#!/usr/bin/env python3
"""
Script de debug para identificar el problema en ModularPipeline
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.modular_pipeline import ModularPipeline
import config

def debug_test():
    """Test simple para debug"""
    print("=== DEBUG TEST ===")
    
    try:
        print("1. Creando ModularPipeline...")
        pipeline = ModularPipeline(
            descriptors=['glcm', 'lbp'], 
            n_jobs=1, 
            enable_checkpoints=False,
            progress_callback=lambda x: print(f"PROGRESS: {x}")
        )
        print("✓ Pipeline creado")
        
        print("2. Iniciando compute_descriptors...")
        results = pipeline.compute_descriptors()
        print("✓ compute_descriptors completado")
        print(f"Resultados: {results}")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_test()