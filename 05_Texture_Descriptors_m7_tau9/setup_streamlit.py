#!/usr/bin/env python3
"""
Script para configurar Streamlit sin preguntas de email.

Ejecutar una vez para configurar Streamlit globalmente.
"""

from pathlib import Path
import os

def setup_streamlit_config():
    """Configura Streamlit para no preguntar por email ni estadísticas."""
    
    print("🔧 Configurando Streamlit...")
    
    # Directorio de configuración de Streamlit
    streamlit_dir = Path.home() / '.streamlit'
    config_file = streamlit_dir / 'config.toml'
    
    # Crear directorio si no existe
    streamlit_dir.mkdir(exist_ok=True)
    print(f"   Directorio creado: {streamlit_dir}")
    
    # Contenido de configuración
    config_content = '''# Configuración de Streamlit para el proyecto de Descriptores de Textura
# Generado automáticamente - no preguntar por email ni estadísticas

[general]
email = ""

[browser]
gatherUsageStats = false
showErrorDetails = true

[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
'''
    
    # Escribir configuración
    config_file.write_text(config_content)
    print(f"   Archivo de configuración creado: {config_file}")
    
    # Verificar que se creó correctamente
    if config_file.exists():
        print("✅ Streamlit configurado exitosamente")
        print("   - Sin pregunta de email")
        print("   - Sin recolección de estadísticas")
        print("   - Configuración de tema personalizada")
        print("\n💡 Ahora puedes ejecutar: python main.py --gui")
    else:
        print("❌ Error: No se pudo crear el archivo de configuración")
        return False
    
    return True

if __name__ == "__main__":
    setup_streamlit_config()