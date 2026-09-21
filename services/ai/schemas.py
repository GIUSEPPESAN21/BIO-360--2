"""
services/ai/schemas.py
======================
Esquemas JSON validables para la salida estructurada de Kiro (Opción K1).

Ventaja: elimina el parsing frágil de texto libre y permite poblar automáticamente
variables como `dilema_sugerido`, validándolas contra la base de conocimiento
(`dilemas.json`) antes de usarlas.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

# --- Esquema: análisis previo de historia clínica ---------------------------------

SCHEMA_ANALISIS_CLINICO: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "dilema_sugerido": {
            "type": "string",
            "description": "Nombre EXACTO de uno de los dilemas de la base de conocimiento provista.",
        },
        "justificacion_dilema": {"type": "string"},
        "principios_en_conflicto": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["autonomia", "beneficencia", "no_maleficencia", "justicia"],
            },
        },
        "elementos_bioeticos_clave": {"type": "array", "items": {"type": "string"}},
        "normativas_aplicables": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Citas textuales tomadas ÚNICAMENTE de la base normativa provista.",
        },
        "cursos_de_accion": {"type": "array", "items": {"type": "string"}},
        "nivel_riesgo": {"type": "string", "enum": ["Bajo", "Moderado", "Crítico"]},
        "resumen": {"type": "string"},
    },
    "required": ["dilema_sugerido", "elementos_bioeticos_clave", "resumen"],
    "additionalProperties": False,
}


# --- Esquema: análisis deliberativo del comité ------------------------------------

SCHEMA_DELIBERACION: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "analisis_deliberativo": {"type": "string"},
        "principios_en_tension": {"type": "array", "items": {"type": "string"}},
        "normativas_citadas": {"type": "array", "items": {"type": "string"}},
        "cursos_de_accion": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "accion": {"type": "string"},
                    "fundamento_etico": {"type": "string"},
                    "riesgos": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["accion", "fundamento_etico"],
            },
        },
        "recomendacion_final": {"type": "string"},
        "advertencias": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["analisis_deliberativo", "recomendacion_final"],
    "additionalProperties": False,
}


# --- Utilidades de parsing y validación -------------------------------------------

def parsear_json_tolerante(texto: str) -> Optional[Dict[str, Any]]:
    """
    Intenta extraer un objeto JSON de la respuesta del modelo.

    Tolera envolturas comunes: bloques ```json ... ```, texto antes/después del objeto.
    Devuelve None si no se encuentra JSON válido.
    """
    if not texto:
        return None

    candidato = texto.strip()

    # 1) Intento directo
    try:
        obj = json.loads(candidato)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, TypeError):
        pass

    # 2) Bloque de código markdown
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidato, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass

    # 3) Primer objeto balanceado del texto
    inicio = candidato.find("{")
    if inicio != -1:
        profundidad = 0
        for i in range(inicio, len(candidato)):
            if candidato[i] == "{":
                profundidad += 1
            elif candidato[i] == "}":
                profundidad -= 1
                if profundidad == 0:
                    try:
                        obj = json.loads(candidato[inicio : i + 1])
                        return obj if isinstance(obj, dict) else None
                    except json.JSONDecodeError:
                        break
    return None


def validar_analisis_clinico(
    data: Optional[Dict[str, Any]],
    dilemas_validos: List[str],
) -> Dict[str, Any]:
    """
    Valida y normaliza la salida estructurada del análisis clínico (K1).

    Reglas:
    - `dilema_sugerido` debe coincidir con un dilema de la base de conocimiento.
      Si no coincide exactamente, se intenta un match laxo (case-insensitive / subcadena);
      si tampoco, se descarta para no inventar categorías.
    - Los campos de lista se normalizan a listas de str.
    """
    resultado: Dict[str, Any] = {
        "dilema_sugerido": "",
        "justificacion_dilema": "",
        "principios_en_conflicto": [],
        "elementos_bioeticos_clave": [],
        "normativas_aplicables": [],
        "cursos_de_accion": [],
        "nivel_riesgo": "",
        "resumen": "",
        "valido": False,
    }
    if not isinstance(data, dict):
        return resultado

    sugerido = str(data.get("dilema_sugerido") or "").strip()
    if sugerido:
        if sugerido in dilemas_validos:
            resultado["dilema_sugerido"] = sugerido
        else:
            bajo = sugerido.lower()
            match = next(
                (d for d in dilemas_validos if d.lower() == bajo),
                None,
            ) or next(
                (d for d in dilemas_validos if bajo in d.lower() or d.lower() in bajo),
                None,
            )
            resultado["dilema_sugerido"] = match or ""

    for campo in (
        "principios_en_conflicto",
        "elementos_bioeticos_clave",
        "normativas_aplicables",
        "cursos_de_accion",
    ):
        valor = data.get(campo)
        if isinstance(valor, list):
            # `v is not None` es imprescindible: `str(None)` da "None", que supera el
            # filtro de cadena no vacía. Sin este guardia, un `null` en la respuesta del
            # modelo terminaba impreso como la viñeta literal "- None" en la UI y el PDF.
            resultado[campo] = [
                str(v).strip() for v in valor if v is not None and str(v).strip()
            ]
        elif isinstance(valor, str) and valor.strip():
            resultado[campo] = [valor.strip()]

    for campo in ("justificacion_dilema", "nivel_riesgo", "resumen"):
        valor = data.get(campo)
        if isinstance(valor, str):
            resultado[campo] = valor.strip()

    if resultado["nivel_riesgo"] not in ("Bajo", "Moderado", "Crítico", ""):
        resultado["nivel_riesgo"] = ""

    resultado["valido"] = bool(resultado["resumen"] or resultado["elementos_bioeticos_clave"])
    return resultado


def formatear_analisis_clinico(validado: Dict[str, Any]) -> str:
    """Convierte la salida estructurada (K1) en markdown legible para la UI y el PDF."""
    if not validado.get("valido"):
        return ""

    lineas: List[str] = []
    if validado.get("resumen"):
        lineas.append(validado["resumen"])
        lineas.append("")
    if validado.get("dilema_sugerido"):
        lineas.append(f"**Dilema sugerido:** {validado['dilema_sugerido']}")
        if validado.get("justificacion_dilema"):
            lineas.append(f"_{validado['justificacion_dilema']}_")
        lineas.append("")
    if validado.get("nivel_riesgo"):
        lineas.append(f"**Nivel de riesgo estimado:** {validado['nivel_riesgo']}")
        lineas.append("")

    secciones = [
        ("Elementos bioéticos clave", "elementos_bioeticos_clave"),
        ("Principios en conflicto", "principios_en_conflicto"),
        ("Normativas aplicables (base verificada)", "normativas_aplicables"),
        ("Cursos de acción", "cursos_de_accion"),
    ]
    for titulo, clave in secciones:
        items = validado.get(clave) or []
        if items:
            lineas.append(f"**{titulo}:**")
            lineas.extend(f"- {i}" for i in items)
            lineas.append("")

    return "\n".join(lineas).strip()
