"""
conftest.py
===========
Configuración de pytest para BIOETHICARE 360.

Inserta la raíz del repositorio en `sys.path` para que las pruebas puedan importar
`core.*` y `services.*` sin necesidad de instalar el proyecto como paquete.
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent

if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
