"""
services/ai/kiro_provider.py
============================
Proveedor Kiro (Opción K1: salida estructurada validable).

Se integra como una implementación más de `AIProvider`, por lo que convive con
Gemini y OpenAI sin modificar la lógica de negocio ni la UI.

Contrato HTTP asumido
---------------------
POST {base_url}/v1/analyze
Headers: Authorization: Bearer <KIRO_API_KEY>
Body:
    {
      "prompt": "...",
      "system": "...",                       # opcional
      "temperature": 0.3,
      "max_tokens": 4096,
      "response_format": {                   # solo si se pide salida estructurada (K1)
          "type": "json_schema",
          "schema": { ... }
      }
    }
Respuesta esperada (se aceptan varias formas para tolerar cambios de contrato):
    { "text": "...", "structured": {...}, "model": "kiro-..." }
    { "output": "..." } | { "content": "..." } | { "choices": [{"message": {"content": "..."}}] }

Si tu endpoint real difiere, ajusta `_extraer_payload` y `RUTA_ANALYZE`: el resto
de la aplicación no necesita cambios.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .base import AIProvider, RespuestaIA
from .prompts import SYSTEM_BIOETICA

logger = logging.getLogger(__name__)

BASE_URL_POR_DEFECTO = "https://api.kiro.ai"
RUTA_ANALYZE = "/v1/analyze"
TIMEOUT_SEGUNDOS = 60
MODELO_POR_DEFECTO = "kiro-1"


class KiroProvider(AIProvider):
    nombre = "Kiro"
    soporta_schema = True  # Opción K1

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        modelo: str = MODELO_POR_DEFECTO,
        timeout: int = TIMEOUT_SEGUNDOS,
    ) -> None:
        super().__init__()
        self.api_key = api_key
        self.base_url = (base_url or BASE_URL_POR_DEFECTO).rstrip("/")
        self.modelo = modelo
        self.timeout = timeout

    def analizar(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        prompt_version: str = "",
    ) -> RespuestaIA:
        import requests

        payload: Dict[str, Any] = {
            "model": self.modelo,
            "prompt": prompt,
            "system": system or SYSTEM_BIOETICA,
            "temperature": 0.3,
            "max_tokens": 4096,
        }
        if response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "schema": response_schema,
            }

        try:
            resp = requests.post(
                f"{self.base_url}{RUTA_ANALYZE}",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error("Error en llamada a Kiro: %s", e)
            return RespuestaIA(
                proveedor=self.nombre,
                modelo=self.modelo,
                prompt_version=prompt_version,
                error=f"Error al contactar a Kiro: {e}",
            )

        texto, estructurado, modelo = self._extraer_payload(data)
        self.modelo_activo = modelo or self.modelo

        if response_schema is not None and estructurado is None and texto:
            from .schemas import parsear_json_tolerante

            estructurado = parsear_json_tolerante(texto)

        if not texto and not estructurado:
            return RespuestaIA(
                proveedor=self.nombre,
                modelo=self.modelo_activo or self.modelo,
                prompt_version=prompt_version,
                error="Kiro devolvió una respuesta vacía o con formato no reconocido.",
            )

        return RespuestaIA(
            texto=texto,
            estructurado=estructurado,
            proveedor=self.nombre,
            modelo=self.modelo_activo or self.modelo,
            prompt_version=prompt_version,
        )

    # -- Internos ------------------------------------------------------------------

    @staticmethod
    def _extraer_payload(data: Any):
        """Tolera varias formas de respuesta para no acoplarse a un contrato exacto."""
        if isinstance(data, str):
            return data, None, None
        if not isinstance(data, dict):
            return "", None, None

        modelo = data.get("model") or data.get("modelo")
        estructurado = data.get("structured") or data.get("data") or data.get("json")
        if estructurado is not None and not isinstance(estructurado, dict):
            estructurado = None

        texto = (
            data.get("text")
            or data.get("output")
            or data.get("content")
            or data.get("respuesta")
            or ""
        )
        if not texto:
            try:
                texto = data["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError, TypeError):
                texto = ""

        return texto, estructurado, modelo
