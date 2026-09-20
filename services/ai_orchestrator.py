"""
services/ai_orchestrator.py
===========================
Orquestador de las llamadas a IA.

Es el ÚNICO punto por el que la aplicación habla con un modelo. Garantiza que en
cada llamada se cumplan, en este orden, las tres reglas del sistema:

1. **Anonimización (M2).** Ningún texto con PII sale hacia una API externa.
2. **Strategy (M6).** El proveedor concreto (Gemini/OpenAI/Kiro) es intercambiable.
3. **Auditoría (M8).** Toda llamada queda registrada con modelo, versión de prompt,
   hash del prompt y timestamp.

La rehidratación de los tokens ([PACIENTE_1] -> nombre real) se hace al final, sobre
la respuesta, para que el usuario autorizado vea el texto legible en el frontend.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from core.anonimizacion import Anonimizador
from services.ai import AIProvider
from services.ai.prompts import (
    PROMPT_VERSION_ANALISIS_CLINICO,
    PROMPT_VERSION_CHAT,
    PROMPT_VERSION_DELIBERACION,
    PROMPT_VERSION_EXPLICACION_SEMAFORO,
    SYSTEM_BIOETICA,
    prompt_analisis_clinico,
    prompt_chat,
    prompt_deliberacion,
    prompt_explicacion_semaforo,
)
from services.ai.schemas import (
    SCHEMA_ANALISIS_CLINICO,
    formatear_analisis_clinico,
    validar_analisis_clinico,
)
from services.audit import (
    ACCION_ANALISIS_CLINICO,
    ACCION_CHAT,
    ACCION_DELIBERACION,
    ACCION_EXPLICACION_SEMAFORO,
)
from services.reportes import anonimizar_reporte_para_ia

logger = logging.getLogger(__name__)

#: Longitud máxima de historia clínica aceptada (evita desbordar el contexto y el costo)
MAX_CARACTERES_HISTORIA = 30_000


class AIOrchestrator:
    """Coordina anonimización, proveedor y auditoría para cada operación de IA."""

    def __init__(
        self,
        provider: Optional[AIProvider],
        dilemas_data: Dict[str, Any],
        audit_log: Any = None,
    ) -> None:
        self.provider = provider
        self.dilemas_data = dilemas_data or {}
        self.audit_log = audit_log

    # -- Helpers -------------------------------------------------------------------

    @property
    def disponible(self) -> bool:
        return self.provider is not None

    def _sin_proveedor(self) -> Tuple[str, Any]:
        from services.ai.base import RespuestaIA

        return (
            "No hay un proveedor de IA configurado. Verifique las claves de API en "
            "'Perfil y Configuración'.",
            RespuestaIA(error="proveedor_no_configurado"),
        )

    def _auditar(self, accion: str, respuesta: Any, caso_id: str, prompt: str) -> None:
        if self.audit_log is None:
            return
        try:
            self.audit_log.registrar_respuesta_ia(
                accion, respuesta, caso_id=caso_id, prompt_texto=prompt, anonimizado=True
            )
        except Exception as e:  # el fallo de auditoría no debe romper el flujo clínico
            logger.error("Fallo registrando en audit log: %s", e)

    # -- Operación 1: análisis previo de historia clínica (K1 + K2) -----------------

    def analizar_historia_clinica(
        self,
        historia: str,
        nombres_pii: Optional[List[str]] = None,
        caso_id: str = "",
    ) -> Dict[str, Any]:
        """
        Analiza una historia clínica y devuelve un resultado estructurado.

        Returns
        -------
        dict con claves: texto_markdown, dilema_sugerido, estructurado, respuesta, error
        """
        resultado: Dict[str, Any] = {
            "texto_markdown": "",
            "dilema_sugerido": "",
            "estructurado": None,
            "respuesta": None,
            "error": None,
        }

        if not self.disponible:
            mensaje, respuesta = self._sin_proveedor()
            resultado.update(error=mensaje, respuesta=respuesta)
            return resultado

        historia = (historia or "").strip()
        if not historia:
            resultado["error"] = "La historia clínica está vacía."
            return resultado

        if len(historia) > MAX_CARACTERES_HISTORIA:
            historia = historia[:MAX_CARACTERES_HISTORIA]
            resultado["truncado"] = True

        # 1) Anonimizar (M2)
        anonimizador = Anonimizador(nombres_conocidos=nombres_pii or [])
        historia_anon = anonimizador.anonimizar(historia)

        # 2) Prompt con grounding normativo (K2) + esquema estructurado (K1)
        prompt = prompt_analisis_clinico(historia_anon, self.dilemas_data)
        schema = SCHEMA_ANALISIS_CLINICO if self.provider.soporta_schema else None

        respuesta = self.provider.analizar(
            prompt,
            system=SYSTEM_BIOETICA,
            response_schema=schema,
            prompt_version=PROMPT_VERSION_ANALISIS_CLINICO,
        )
        resultado["respuesta"] = respuesta

        # 3) Auditar (M8)
        self._auditar(ACCION_ANALISIS_CLINICO, respuesta, caso_id, prompt)

        if not respuesta.ok:
            resultado["error"] = respuesta.error or "La IA no devolvió una respuesta utilizable."
            return resultado

        # 4) Validar salida estructurada contra la base de conocimiento (K1)
        dilemas_validos = list(self.dilemas_data.keys())
        if respuesta.estructurado:
            validado = validar_analisis_clinico(respuesta.estructurado, dilemas_validos)
            if validado["valido"]:
                resultado["estructurado"] = validado
                resultado["dilema_sugerido"] = validado["dilema_sugerido"]
                texto = formatear_analisis_clinico(validado)
                resultado["texto_markdown"] = anonimizador.rehidratar(texto)
                return resultado

        # Fallback: texto libre rehidratado
        resultado["texto_markdown"] = anonimizador.rehidratar(respuesta.texto)
        return resultado

    # -- Operación 2: análisis deliberativo del comité (K2) -------------------------

    def generar_deliberacion(
        self,
        reporte: Dict[str, Any],
        nombres_pii: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        resultado: Dict[str, Any] = {"texto": "", "respuesta": None, "error": None}

        if not self.disponible:
            mensaje, respuesta = self._sin_proveedor()
            resultado.update(error=mensaje, respuesta=respuesta)
            return resultado

        anonimizador = Anonimizador(nombres_conocidos=nombres_pii or [])
        reporte_anon = anonimizar_reporte_para_ia(reporte, anonimizador)
        dilema = reporte.get("Dilema Ético Principal (Seleccionado)")

        prompt = prompt_deliberacion(reporte_anon, self.dilemas_data, dilema)
        respuesta = self.provider.analizar(
            prompt, system=SYSTEM_BIOETICA, prompt_version=PROMPT_VERSION_DELIBERACION
        )
        resultado["respuesta"] = respuesta

        self._auditar(
            ACCION_DELIBERACION, respuesta, reporte.get("ID del Caso", ""), prompt
        )

        if not respuesta.ok:
            resultado["error"] = respuesta.error or "La IA no devolvió una respuesta utilizable."
            return resultado

        resultado["texto"] = anonimizador.rehidratar(respuesta.texto)
        return resultado

    # -- Operación 3: chatbot contextual (K2) ---------------------------------------

    def responder_chat(
        self,
        reporte: Dict[str, Any],
        pregunta: str,
        nombres_pii: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        resultado: Dict[str, Any] = {"texto": "", "respuesta": None, "error": None}

        if not self.disponible:
            mensaje, respuesta = self._sin_proveedor()
            resultado.update(error=mensaje, respuesta=respuesta)
            return resultado

        anonimizador = Anonimizador(nombres_conocidos=nombres_pii or [])
        reporte_anon = anonimizar_reporte_para_ia(reporte or {}, anonimizador)
        pregunta_anon = anonimizador.anonimizar(pregunta or "")
        dilema = (reporte or {}).get("Dilema Ético Principal (Seleccionado)")

        prompt = prompt_chat(reporte_anon, pregunta_anon, self.dilemas_data, dilema)
        respuesta = self.provider.analizar(
            prompt, system=SYSTEM_BIOETICA, prompt_version=PROMPT_VERSION_CHAT
        )
        resultado["respuesta"] = respuesta

        self._auditar(ACCION_CHAT, respuesta, (reporte or {}).get("ID del Caso", ""), prompt)

        if not respuesta.ok:
            resultado["error"] = respuesta.error or "La IA no devolvió una respuesta utilizable."
            return resultado

        resultado["texto"] = anonimizador.rehidratar(respuesta.texto)
        return resultado

    # -- Operación 4: explicación del semáforo ético determinista -------------------

    def explicar_semaforo(
        self,
        analisis_etico: Dict[str, Any],
        dilema: Optional[str] = None,
        caso_id: str = "",
    ) -> Dict[str, Any]:
        """
        La IA EXPLICA el resultado del motor de reglas; no lo recalcula ni lo decide.
        El análisis ético no contiene PII (solo nombres de perspectivas y principios).
        """
        resultado: Dict[str, Any] = {"texto": "", "respuesta": None, "error": None}

        if not self.disponible:
            mensaje, respuesta = self._sin_proveedor()
            resultado.update(error=mensaje, respuesta=respuesta)
            return resultado

        prompt = prompt_explicacion_semaforo(analisis_etico, dilema, self.dilemas_data)
        respuesta = self.provider.analizar(
            prompt,
            system=SYSTEM_BIOETICA,
            prompt_version=PROMPT_VERSION_EXPLICACION_SEMAFORO,
        )
        resultado["respuesta"] = respuesta
        self._auditar(ACCION_EXPLICACION_SEMAFORO, respuesta, caso_id, prompt)

        if not respuesta.ok:
            resultado["error"] = respuesta.error or "La IA no devolvió una respuesta utilizable."
            return resultado

        resultado["texto"] = respuesta.texto
        return resultado

    # -- Trazabilidad para el reporte ----------------------------------------------

    @staticmethod
    def trazabilidad(respuesta: Any) -> Dict[str, Any]:
        """Metadatos de trazabilidad para incrustar en el reporte y el PDF (M8)."""
        from datetime import datetime, timezone

        return {
            "proveedor": getattr(respuesta, "proveedor", ""),
            "modelo": getattr(respuesta, "modelo", ""),
            "prompt_version": getattr(respuesta, "prompt_version", ""),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "datos_anonimizados": True,
        }
