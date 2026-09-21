"""
services/ai/openai_provider.py
==============================
Proveedor OpenAI adaptado a la interfaz `AIProvider`.

Soporta salida estructurada (K1) vía `response_format={"type": "json_object"}`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .base import AIProvider, RespuestaIA
from .prompts import SYSTEM_BIOETICA

logger = logging.getLogger(__name__)

MODELO_POR_DEFECTO = "gpt-4o"


class OpenAIProvider(AIProvider):
    nombre = "OpenAI"
    soporta_schema = True

    def __init__(self, api_key: str, modelo: str = MODELO_POR_DEFECTO) -> None:
        super().__init__()
        self.api_key = api_key
        self.modelo = modelo

    def analizar(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        prompt_version: str = "",
    ) -> RespuestaIA:
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            return RespuestaIA(
                proveedor=self.nombre,
                error=f"openai no está instalado: {e}",
                prompt_version=prompt_version,
            )

        system_msg = system or SYSTEM_BIOETICA
        if response_schema is not None:
            system_msg += (
                "\n\nDebes responder EXCLUSIVAMENTE con un objeto JSON válido que cumpla "
                f"este esquema: {response_schema}"
            )

        try:
            client = OpenAI(api_key=self.api_key)
            kwargs: Dict[str, Any] = {
                "model": self.modelo,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 4096,
            }
            if response_schema is not None:
                kwargs["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**kwargs)
            texto = response.choices[0].message.content or ""
            self.modelo_activo = self.modelo

            respuesta = RespuestaIA(
                texto=texto,
                proveedor=self.nombre,
                modelo=self.modelo,
                prompt_version=prompt_version,
            )
            if response_schema is not None:
                from .schemas import parsear_json_tolerante

                respuesta.estructurado = parsear_json_tolerante(texto)
            return respuesta

        except Exception as e:
            logger.error("Error en llamada a OpenAI: %s", e)
            return RespuestaIA(
                proveedor=self.nombre,
                modelo=self.modelo,
                prompt_version=prompt_version,
                error=f"Error al contactar a OpenAI: {e}",
            )
