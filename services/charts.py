"""
services/charts.py
==================
Generación de visualizaciones Plotly.

Optimización M5
---------------
Los gráficos se regeneran BAJO DEMANDA a partir de las ponderaciones en bruto.
Ya no se persisten los JSON de Plotly en Firestore (eran documentos de decenas de KB,
acercándose al límite de 1 MiB por documento y encareciendo cada lectura).

Las funciones aceptan tanto un `CasoBioetico` como el dict de perspectivas
almacenado en el reporte, de modo que los casos guardados con el esquema antiguo
siguen visualizándose.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import plotly.graph_objects as go

from core.modelos import (
    COLOR_POR_DEFECTO,
    PERSPECTIVAS_COLORES_SOLIDOS,
    PERSPECTIVAS_COLORES_TRANSLUCIDOS,
    PERSPECTIVAS_NOMBRES,
    PRINCIPIOS,
    PRINCIPIOS_LABELS,
    safe_int,
)

logger = logging.getLogger(__name__)

# Mapa inverso: "Equipo Médico" -> "medico" (para leer reportes ya guardados)
_NOMBRE_LARGO_A_CORTO = {v: k for k, v in PERSPECTIVAS_NOMBRES.items()}


def normalizar_perspectivas(fuente: Any) -> Dict[str, Dict[str, int]]:
    """
    Normaliza las ponderaciones a la forma {clave_corta: {principio: int}}.

    Acepta:
    - un `CasoBioetico` (atributo `.perspectivas`)
    - {"medico": {...}, "familia": {...}, "comite": {...}}
    - {"Equipo Médico": {...}, "Familia/Paciente": {...}, "Comité de Bioética": {...}}
    """
    if fuente is None:
        return {}

    datos = getattr(fuente, "perspectivas", fuente)
    if not isinstance(datos, dict):
        return {}

    resultado: Dict[str, Dict[str, int]] = {}
    for clave, valores in datos.items():
        if not isinstance(valores, dict):
            continue
        corta = clave if clave in PERSPECTIVAS_NOMBRES else _NOMBRE_LARGO_A_CORTO.get(clave)
        if corta is None:
            corta = str(clave).strip().lower()
        resultado[corta] = {p: safe_int(valores.get(p)) for p in PRINCIPIOS}
    return resultado


def _nombre(corta: str) -> str:
    return PERSPECTIVAS_NOMBRES.get(corta, str(corta).capitalize())


def figura_radar(fuente: Any) -> Optional[go.Figure]:
    """Radar de ponderación por perspectiva."""
    perspectivas = normalizar_perspectivas(fuente)
    if not perspectivas:
        return None
    try:
        fig = go.Figure()
        for corta, valores in perspectivas.items():
            fig.add_trace(
                go.Scatterpolar(
                    r=[valores[p] for p in PRINCIPIOS],
                    theta=list(PRINCIPIOS_LABELS),
                    fill="toself",
                    name=_nombre(corta),
                    # .get() evita el KeyError del código original
                    line_color=PERSPECTIVAS_COLORES_TRANSLUCIDOS.get(corta, COLOR_POR_DEFECTO),
                )
            )
        fig.update_layout(
            title_text="<b>Ponderación por Perspectiva</b>",
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            showlegend=True,
            font_size=14,
        )
        return fig
    except Exception as e:
        logger.error("Error generando gráfico radar: %s", e)
        return None


def figura_consenso(fuente: Any) -> Optional[go.Figure]:
    """Barras de media por principio con desviación estándar (consenso/disenso)."""
    perspectivas = normalizar_perspectivas(fuente)
    if not perspectivas:
        return None
    try:
        scores = np.array([[v[p] for p in PRINCIPIOS] for v in perspectivas.values()])
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=list(PRINCIPIOS_LABELS),
                y=np.mean(scores, axis=0),
                error_y=dict(type="data", array=np.std(scores, axis=0), visible=True),
                marker_color="#636EFA",
            )
        )
        fig.update_layout(
            title_text="<b>Análisis de Consenso y Disenso</b>",
            yaxis=dict(range=[0, 6]),
            font_size=14,
        )
        return fig
    except Exception as e:
        logger.error("Error generando gráfico de consenso: %s", e)
        return None


def figura_equilibrio(fuente: Any) -> Optional[go.Figure]:
    """Barras agrupadas comparando principios entre perspectivas."""
    perspectivas = normalizar_perspectivas(fuente)
    if not perspectivas:
        return None
    try:
        fig = go.Figure()
        for corta, valores in perspectivas.items():
            fig.add_trace(
                go.Bar(
                    x=list(PRINCIPIOS_LABELS),
                    y=[valores[p] for p in PRINCIPIOS],
                    name=_nombre(corta),
                    marker_color=PERSPECTIVAS_COLORES_SOLIDOS.get(corta, COLOR_POR_DEFECTO),
                )
            )
        fig.update_layout(
            title_text="<b>Análisis Comparativo de Principios</b>",
            barmode="group",
            yaxis=dict(title="Puntaje Asignado", range=[0, 5.5]),
            legend_title_text="Perspectivas",
            font_size=12,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#2E3A47",
        )
        return fig
    except Exception as e:
        logger.error("Error generando gráfico de equilibrio: %s", e)
        return None


def generar_todas_las_figuras(fuente: Any) -> Dict[str, Optional[go.Figure]]:
    """Devuelve las tres figuras del dashboard, regeneradas desde las ponderaciones."""
    return {
        "radar": figura_radar(fuente),
        "consenso": figura_consenso(fuente),
        "equilibrio": figura_equilibrio(fuente),
    }
