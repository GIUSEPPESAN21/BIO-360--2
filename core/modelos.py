"""
core/modelos.py
===============
Modelos de dominio y utilidades de conversión segura para BIOETHICARE 360.

No dependen de Streamlit ni de servicios externos, por lo que son testeables de
forma aislada con pytest.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict


# --- Constantes de dominio compartidas -------------------------------------------

PRINCIPIOS = ("autonomia", "beneficencia", "no_maleficencia", "justicia")
PRINCIPIOS_LABELS = ("Autonomía", "Beneficencia", "No Maleficencia", "Justicia")

PERSPECTIVAS_NOMBRES = {
    "medico": "Equipo Médico",
    "familia": "Familia/Paciente",
    "comite": "Comité de Bioética",
}

# Colores centralizados (antes duplicados en dos funciones de gráficos)
PERSPECTIVAS_COLORES_SOLIDOS = {
    "medico": "#EF4444",
    "familia": "#3B82F6",
    "comite": "#22C55E",
}
PERSPECTIVAS_COLORES_TRANSLUCIDOS = {
    "medico": "rgba(239, 68, 68, 0.7)",
    "familia": "rgba(59, 130, 246, 0.7)",
    "comite": "rgba(34, 197, 94, 0.7)",
}
COLOR_POR_DEFECTO = "#6c757d"


# --- Utilidades de conversión segura ---------------------------------------------

def safe_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


# --- Modelo principal ------------------------------------------------------------

class CasoBioetico:
    """Representa un caso bioético con sus ponderaciones multiperspectiva."""

    def __init__(self, dilemas_opciones: list[str] | None = None, **kwargs: Any) -> None:
        dilemas_opciones = dilemas_opciones or []
        self.nombre_paciente = safe_str(kwargs.get("nombre_paciente"), "N/A")
        self.historia_clinica = safe_str(
            kwargs.get("historia_clinica"), f"caso_{int(datetime.now().timestamp())}"
        )
        self.edad = safe_int(kwargs.get("edad"))
        self.genero = safe_str(kwargs.get("genero"), "N/A")
        self.nombre_analista = safe_str(kwargs.get("nombre_analista"), "N/A")
        self.dilema_etico = safe_str(
            kwargs.get("dilema_etico", dilemas_opciones[0] if dilemas_opciones else "")
        )
        self.descripcion_caso = safe_str(kwargs.get("descripcion_caso"))
        self.antecedentes_culturales = safe_str(kwargs.get("antecedentes_culturales"))
        self.condicion = safe_str(kwargs.get("condicion", "Estable"))
        self.semanas_gestacion = safe_int(kwargs.get("semanas_gestacion"))
        self.puntos_clave_ia = safe_str(kwargs.get("puntos_clave_ia"))
        self.ai_clinical_analysis_summary = safe_str(kwargs.get("ai_clinical_analysis_summary"))
        self.perspectivas: Dict[str, Dict[str, int]] = {
            "medico": self._extract_perspective("medico", kwargs),
            "familia": self._extract_perspective("familia", kwargs),
            "comite": self._extract_perspective("comite", kwargs),
        }

    @staticmethod
    def _extract_perspective(prefix: str, kwargs: Dict[str, Any]) -> Dict[str, int]:
        return {
            "autonomia": safe_int(kwargs.get(f"nivel_autonomia_{prefix}")),
            "beneficencia": safe_int(kwargs.get(f"nivel_beneficencia_{prefix}")),
            "no_maleficencia": safe_int(kwargs.get(f"nivel_no_maleficencia_{prefix}")),
            "justicia": safe_int(kwargs.get(f"nivel_justicia_{prefix}")),
        }

    def nombres_pii(self) -> list[str]:
        """Devuelve los nombres propios del caso que deben anonimizarse antes de enviarse a la IA."""
        return [n for n in (self.nombre_paciente, self.nombre_analista) if n and n != "N/A"]
