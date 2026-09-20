"""
services/ai/__init__.py
=======================
Fábrica central de proveedores de IA (patrón Strategy, M6).

Reemplaza los `if provider == "..."` dispersos por una única función `get_provider`.
Añadir un proveedor nuevo = una implementación de `AIProvider` + una rama aquí.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Union

from .base import AIProvider, RespuestaIA
from .gemini_provider import GeminiProvider
from .openai_provider import OpenAIProvider
from .kiro_provider import KiroProvider

__all__ = [
    "AIProvider",
    "RespuestaIA",
    "GeminiProvider",
    "OpenAIProvider",
    "KiroProvider",
    "get_provider",
    "PROVEEDORES_DISPONIBLES",
]

#: Nombres mostrados en la UI. Kiro se suma sin alterar los existentes.
PROVEEDORES_DISPONIBLES = ("Google Gemini", "OpenAI", "Kiro")

SecretsLike = Union[Dict[str, str], Callable[[str], Optional[str]]]


def _resolver(secrets: SecretsLike, key: str) -> Optional[str]:
    """
    Permite usar tanto un dict como `st.secrets.get` indistintamente.

    Ambas ramas van protegidas a propósito: `st.secrets.get` NO devuelve None cuando
    no existe ningún `secrets.toml`, sino que lanza `StreamlitSecretNotFoundError`.
    Sin este guardia, una instalación sin secretos configurados no degradaba con un
    aviso en la UI: rompía el arranque de la aplicación.
    """
    try:
        if callable(secrets):
            return secrets(key)
        return secrets.get(key)
    except Exception:
        return None


def get_provider(nombre: str, secrets: SecretsLike) -> Optional[AIProvider]:
    """
    Devuelve una instancia de `AIProvider` según el nombre seleccionado,
    o None si faltan las credenciales de ese proveedor.

    Parameters
    ----------
    nombre:
        Uno de `PROVEEDORES_DISPONIBLES`.
    secrets:
        Un dict o un callable tipo `st.secrets.get` para resolver claves.
    """
    if nombre == "Kiro":
        api_key = _resolver(secrets, "KIRO_API_KEY")
        base_url = _resolver(secrets, "KIRO_BASE_URL")
        if not api_key:
            return None
        return KiroProvider(api_key=api_key, base_url=base_url)

    if nombre == "OpenAI":
        api_key = _resolver(secrets, "OPENAI_API_KEY")
        return OpenAIProvider(api_key=api_key) if api_key else None

    # Por defecto: Google Gemini
    api_key = _resolver(secrets, "GEMINI_API_KEY")
    return GeminiProvider(api_key=api_key) if api_key else None
