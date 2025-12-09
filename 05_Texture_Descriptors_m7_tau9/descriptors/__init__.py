"""
Sistema de registro automático de descriptores usando el patrón Decorator.

Este módulo implementa un sistema de autodescubrimiento que permite agregar
nuevos descriptores sin modificar código existente. Simplemente crea una nueva
clase que herede de BaseDescriptor y decórala con @register_descriptor.

Ejemplo de uso:
    @register_descriptor("mi_descriptor")
    class MiDescriptor(BaseDescriptor):
        # ... implementación ...
"""

import os
import importlib
import inspect
from typing import Dict, Type, List
from .base import BaseDescriptor

# Registro global de descriptores
# Este diccionario se llena automáticamente cuando se importan los módulos
_DESCRIPTOR_REGISTRY: Dict[str, Type[BaseDescriptor]] = {}


def register_descriptor(name: str, enabled_by_default: bool = True):
    """
    Decorador para registrar automáticamente descriptores de textura.
    
    Este decorador permite que las clases de descriptores se registren
    automáticamente sin necesidad de modificar código central.
    
    Args:
        name (str): Nombre único del descriptor (ej: "glcm", "lbp")
        enabled_by_default (bool): Si el descriptor está activo por defecto
        
    Returns:
        function: Decorador que registra la clase
        
    Ejemplo:
        @register_descriptor("glcm")
        class GLCMDescriptor(BaseDescriptor):
            pass
    """
    def decorator(cls):
        # Verificar que la clase herede de BaseDescriptor
        if not issubclass(cls, BaseDescriptor):
            raise TypeError(f"{cls.__name__} debe heredar de BaseDescriptor")
        
        # Registrar la clase
        _DESCRIPTOR_REGISTRY[name] = cls
        
        # Agregar metadata a la clase
        cls._registry_name = name
        cls._enabled_by_default = enabled_by_default
        
        # Log del registro
        print(f"✓ Descriptor '{name}' registrado: {cls.__name__}")
        
        return cls
    
    return decorator


def get_descriptor(name: str) -> Type[BaseDescriptor]:
    """
    Obtiene una clase de descriptor por su nombre.
    
    Args:
        name (str): Nombre del descriptor registrado
        
    Returns:
        Type[BaseDescriptor]: Clase del descriptor
        
    Raises:
        KeyError: Si el descriptor no está registrado
    """
    if name not in _DESCRIPTOR_REGISTRY:
        available = ", ".join(_DESCRIPTOR_REGISTRY.keys())
        raise KeyError(f"Descriptor '{name}' no encontrado. Disponibles: {available}")
    
    return _DESCRIPTOR_REGISTRY[name]


def list_available_descriptors() -> List[str]:
    """
    Lista todos los descriptores disponibles.
    
    Returns:
        List[str]: Lista de nombres de descriptores registrados
    """
    return list(_DESCRIPTOR_REGISTRY.keys())


def get_descriptor_info(name: str) -> dict:
    """
    Obtiene información detallada sobre un descriptor.
    
    Args:
        name (str): Nombre del descriptor
        
    Returns:
        dict: Información del descriptor incluyendo nombre, clase, docstring, etc.
    """
    cls = get_descriptor(name)
    return {
        'name': name,
        'class': cls.__name__,
        'module': cls.__module__,
        'enabled_by_default': getattr(cls, '_enabled_by_default', True),
        'description': inspect.getdoc(cls) or "Sin descripción",
        'parameters': _extract_init_parameters(cls)
    }


def _extract_init_parameters(cls):
    """
    Extrae los parámetros del constructor de una clase.
    
    Args:
        cls: Clase a inspeccionar
        
    Returns:
        dict: Parámetros con sus valores por defecto
    """
    try:
        sig = inspect.signature(cls.__init__)
        params = {}
        for name, param in sig.parameters.items():
            if name not in ['self', 'args', 'kwargs']:
                params[name] = {
                    'default': param.default if param.default != param.empty else None,
                    'annotation': str(param.annotation) if param.annotation != param.empty else 'Any'
                }
        return params
    except:
        return {}


def create_descriptor(name: str, **kwargs) -> BaseDescriptor:
    """
    Crea una instancia de un descriptor con los parámetros dados.
    
    Args:
        name (str): Nombre del descriptor
        **kwargs: Parámetros para el constructor del descriptor
        
    Returns:
        BaseDescriptor: Instancia del descriptor
    """
    descriptor_class = get_descriptor(name)
    return descriptor_class(**kwargs)


# Auto-importar todos los módulos de descriptores en este directorio
# Esto asegura que todos los descriptores se registren automáticamente
def _auto_import_descriptors():
    """
    Importa automáticamente todos los módulos de descriptores.
    
    Esta función busca todos los archivos .py en el directorio actual
    (excepto __init__.py y base.py) y los importa, lo que causa que
    sus decoradores @register_descriptor se ejecuten.
    """
    current_dir = os.path.dirname(__file__)
    
    # Buscar todos los archivos .py en el directorio
    for filename in os.listdir(current_dir):
        if filename.endswith('.py') and filename not in ['__init__.py', 'base.py']:
            module_name = filename[:-3]  # Quitar .py
            
            try:
                # Importar el módulo
                importlib.import_module(f'.{module_name}', package=__name__)
                print(f"  → Módulo '{module_name}' importado")
            except Exception as e:
                print(f"  ⚠ Error al importar '{module_name}': {e}")


# Ejecutar auto-importación cuando se importe este módulo
print("🔍 Buscando descriptores de textura...")
_auto_import_descriptors()
print(f"📊 Total de descriptores registrados: {len(_DESCRIPTOR_REGISTRY)}")