"""
services/ai/base.py
===================
Patrón Strategy (M6) para proveedores de IA.

Todos los proveedores (Gemini, OpenAI, Kiro) implementan la misma interfaz
`AIProvider`, de modo que añadir un nuevo modelo NO requiere más ramas if/else
dispersas por la aplicación. La UI y la lógica de negocio solo conocen esta interfaz.

`RespuestaIA` transporta además los metadatos que exige el audit log inmutable (M8):
modelo exacto usado, versión del prompt y si la respuesta fue estructurada.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RespuestaIA:
    """Resultado de una llamada a un proveedor de IA, con metadatos de trazabilidad."""

    texto: str = ""
    estructurado: Optional[Dict[str, Any]] = None
    proveedor: str = ""
    modelo: str = ""
    prompt_version: str = ""
    bloqueado: bool = False
    error: Optional[str] = None
    intentos: int = 1
    metadatos: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and not self.bloqueado and bool(
            self.texto or self.estructurado
        )

    def __str__(self) -> str:  # permite usar la respuesta donde se esperaba un str
        return self.texto


class AIProvider(ABC):
    """Interfaz común para todos los proveedores de IA."""

    #: Nombre legible del proveedor (para UI y audit log)
    nombre: str = "AIProvider"

    #: Soporta salida estructurada validable (Opción K1)
    soporta_schema: bool = False

    def __init__(self) -> None:
        #: Modelo efectivamente usado en la última llamada (para audit log / sidebar)
        self.modelo_activo: Optional[str] = None

    @abstractmethod
    def analizar(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        prompt_version: str = "",
    ) -> RespuestaIA:
        """
        Ejecuta un análisis.

        Parameters
        ----------
        prompt:
            Texto de entrada YA ANONIMIZADO. La anonimización (M2) es responsabilidad
            del llamador; los proveedores nunca deben recibir PII.
        system:
            Instrucción de sistema opcional (rol del modelo).
        response_schema:
            Si se provee y `soporta_schema` es True, el proveedor devuelve un objeto
            estructurado conforme al esquema (Opción K1).
        prompt_version:
            Identificador de versión del prompt, propagado al audit log (M8).

        Returns
        -------
        RespuestaIA
        """
        raise NotImplementedError
