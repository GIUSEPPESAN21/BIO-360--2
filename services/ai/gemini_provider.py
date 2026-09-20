"""
services/ai/gemini_provider.py
=============================
Proveedor Google Gemini.

CAMBIO DE SEGURIDAD (Fase 1)
----------------------------
El código original usaba `BLOCK_ONLY_HIGH` en las cuatro categorías, debilitando
la seguridad global de forma permanente para evitar bloqueos de terminología médica.

Ahora:
1. El umbral por defecto es `BLOCK_MEDIUM_AND_ABOVE` (estándar, no debilitado).
2. Si el modelo bloquea la respuesta, NO se baja la seguridad global: se reintenta
   con un prompt reformulado de forma más restrictiva y clínicamente encuadrado
   (`reencuadrar_prompt_clinico`), que declara el contexto académico-deliberativo.
3. Solo si el reintento clínico también es bloqueado se escala el umbral a
   `BLOCK_ONLY_HIGH` para ESA llamada puntual, registrándolo en la respuesta para
   que quede trazado en el audit log. Nunca es el comportamiento por defecto.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .base import AIProvider, RespuestaIA
from .prompts import reencuadrar_prompt_clinico

logger = logging.getLogger(__name__)

# Umbral estándar: no debilitado.
UMBRAL_ESTANDAR = "BLOCK_MEDIUM_AND_ABOVE"
# Umbral de escalamiento, usado SOLO como último recurso y trazado en el audit log.
UMBRAL_ESCALADO = "BLOCK_ONLY_HIGH"

CATEGORIAS = (
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
)

MODELOS_CANDIDATOS = (
    "gemini-2.0-flash-exp",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro-latest",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
)


def _safety(umbral: str) -> List[Dict[str, str]]:
    return [{"category": c, "threshold": umbral} for c in CATEGORIAS]


class GeminiProvider(AIProvider):
    nombre = "Google Gemini"
    soporta_schema = True  # Gemini soporta response_mime_type=application/json

    def __init__(self, api_key: str, modelos: Optional[tuple] = None) -> None:
        super().__init__()
        self.api_key = api_key
        self.modelos = modelos or MODELOS_CANDIDATOS

    # -- API pública ---------------------------------------------------------------

    def analizar(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        prompt_version: str = "",
    ) -> RespuestaIA:
        try:
            import google.generativeai as genai
        except ImportError as e:  # pragma: no cover - dependencia de runtime
            return RespuestaIA(
                proveedor=self.nombre,
                error=f"google-generativeai no está instalado: {e}",
                prompt_version=prompt_version,
            )

        try:
            genai.configure(api_key=self.api_key)
        except Exception as e:
            return RespuestaIA(
                proveedor=self.nombre,
                error=f"Error configurando Gemini: {e}",
                prompt_version=prompt_version,
            )

        # Estrategia de 3 pasos: (prompt original, umbral estándar) ->
        # (prompt reencuadrado, umbral estándar) -> (prompt reencuadrado, umbral escalado)
        estrategias = [
            (prompt, UMBRAL_ESTANDAR, "original"),
            (reencuadrar_prompt_clinico(prompt), UMBRAL_ESTANDAR, "reencuadrado"),
            (reencuadrar_prompt_clinico(prompt), UMBRAL_ESCALADO, "reencuadrado+umbral_escalado"),
        ]

        ultimo_error: Optional[str] = None
        intentos = 0

        for texto_prompt, umbral, etiqueta in estrategias:
            for modelo in self.modelos:
                intentos += 1
                try:
                    generation_config: Dict[str, Any] = {
                        "temperature": 0.3,
                        "top_p": 0.95,
                        "top_k": 40,
                        "max_output_tokens": 4096,
                    }
                    if response_schema is not None:
                        generation_config["response_mime_type"] = "application/json"

                    kwargs: Dict[str, Any] = {
                        "model_name": modelo,
                        "generation_config": generation_config,
                        "safety_settings": _safety(umbral),
                    }
                    if system:
                        kwargs["system_instruction"] = system

                    try:
                        model = genai.GenerativeModel(**kwargs)
                    except TypeError:
                        # Versiones antiguas del SDK no aceptan system_instruction
                        kwargs.pop("system_instruction", None)
                        model = genai.GenerativeModel(**kwargs)
                        if system:
                            texto_prompt = f"{system}\n\n{texto_prompt}"

                    response = model.generate_content(texto_prompt)

                    texto = self._extraer_texto(response)
                    if texto:
                        self.modelo_activo = modelo
                        respuesta = RespuestaIA(
                            texto=texto,
                            proveedor=self.nombre,
                            modelo=modelo,
                            prompt_version=prompt_version,
                            intentos=intentos,
                            metadatos={
                                "estrategia": etiqueta,
                                "safety_threshold": umbral,
                            },
                        )
                        if response_schema is not None:
                            respuesta.estructurado = self._parsear_json(texto)
                        if umbral == UMBRAL_ESCALADO:
                            logger.warning(
                                "Gemini requirió escalar el umbral de seguridad para responder "
                                "(modelo=%s). Registrado en audit log.",
                                modelo,
                            )
                        return respuesta

                    motivo = self._motivo_bloqueo(response)
                    if motivo:
                        logger.warning(
                            "Gemini bloqueó la respuesta (modelo=%s, estrategia=%s): %s",
                            modelo, etiqueta, motivo,
                        )
                        ultimo_error = f"Bloqueado: {motivo}"
                        # Pasamos al siguiente modelo/estrategia sin debilitar seguridad global
                        continue

                except Exception as e:
                    ultimo_error = str(e)
                    logger.warning("Error con modelo %s (%s): %s", modelo, etiqueta, e)
                    continue

        return RespuestaIA(
            proveedor=self.nombre,
            prompt_version=prompt_version,
            bloqueado=True,
            intentos=intentos,
            error=(
                "No se obtuvo respuesta de ningún modelo de Gemini tras reintentos "
                f"con reencuadre clínico. Último detalle: {ultimo_error}"
            ),
        )

    # -- Internos ------------------------------------------------------------------

    @staticmethod
    def _extraer_texto(response: Any) -> str:
        try:
            partes = getattr(response, "parts", None)
            if partes:
                texto = "".join(getattr(p, "text", "") or "" for p in partes)
                if texto.strip():
                    return texto
            texto = getattr(response, "text", "") or ""
            return texto if texto.strip() else ""
        except Exception:
            return ""

    @staticmethod
    def _motivo_bloqueo(response: Any) -> Optional[str]:
        feedback = getattr(response, "prompt_feedback", None)
        if feedback:
            razon = getattr(feedback, "block_reason", None)
            return str(razon) if razon else str(feedback)
        return None

    @staticmethod
    def _parsear_json(texto: str) -> Optional[Dict[str, Any]]:
        from .schemas import parsear_json_tolerante

        return parsear_json_tolerante(texto)
