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


# --- DeliberIA: indicadores de deliberación y explicabilidad (XAI) -------------------

COLOR_SEVERIDAD = {"Bajo": "#16A34A", "Moderado": "#D97706", "Crítico": "#DC2626"}
COLOR_PRIMARIO = "#2563EB"
COLOR_NEUTRO = "#64748B"


def figura_divergencia(fuente: Any) -> Optional[go.Figure]:
    """
    Mapa de calor de divergencia: cuánto se aparta cada perspectiva de la media del
    grupo en cada principio. Rojo = pondera por encima del grupo; azul = por debajo.
    Hace visible DÓNDE está el disenso, no solo cuánto hay.
    """
    perspectivas = normalizar_perspectivas(fuente)
    activas = {k: v for k, v in perspectivas.items() if sum(v.values()) > 0}
    if len(activas) < 2:
        return None
    try:
        medias = {p: np.mean([v[p] for v in activas.values()]) for p in PRINCIPIOS}
        z = [[round(v[p] - medias[p], 2) for p in PRINCIPIOS] for v in activas.values()]
        fig = go.Figure(
            go.Heatmap(
                z=z,
                x=list(PRINCIPIOS_LABELS),
                y=[_nombre(k) for k in activas],
                zmin=-3, zmax=3, zmid=0,
                colorscale="RdBu_r",
                text=[[f"{val:+.1f}" for val in fila] for fila in z],
                texttemplate="%{text}",
                colorbar=dict(title="Δ vs. media"),
                hovertemplate="%{y} · %{x}: %{z:+.2f}<extra></extra>",
            )
        )
        fig.update_layout(title_text="<b>Mapa de Divergencia entre Perspectivas</b>", font_size=13)
        return fig
    except Exception as e:
        logger.error("Error generando mapa de divergencia: %s", e)
        return None


def figura_contribuciones(xai: Optional[Dict[str, Any]]) -> Optional[go.Figure]:
    """
    Atribución de la severidad (XAI): cascada con los puntos que aporta cada hallazgo
    del semáforo, y líneas en los umbrales de Moderado y Crítico.
    """
    hallazgos = (xai or {}).get("hallazgos") or []
    if not hallazgos:
        return None
    try:
        from core.etica import (
            ETIQUETAS_TIPO,
            UMBRAL_SEVERIDAD_CRITICO,
            UMBRAL_SEVERIDAD_MODERADO,
        )

        etiquetas, valores = [], []
        for i, h in enumerate(hallazgos, 1):
            partes = [ETIQUETAS_TIPO.get(h.get("tipo"), h.get("tipo", ""))]
            if h.get("perspectiva"):
                partes.append(_nombre(h["perspectiva"]))
            if h.get("principio"):
                partes.append(h["principio"].replace("_", " "))
            etiquetas.append(f"{i}. " + " · ".join(partes))
            valores.append(safe_int(h.get("puntos")))
        fig = go.Figure(
            go.Waterfall(
                x=etiquetas + ["Total"],
                y=valores + [0],
                measure=["relative"] * len(valores) + ["total"],
                increasing=dict(marker_color="#F59E0B"),
                totals=dict(marker_color=COLOR_SEVERIDAD.get(xai.get("severidad"), COLOR_NEUTRO)),
                connector=dict(line=dict(color="#CBD5E1")),
            )
        )
        for umbral, texto in (
            (UMBRAL_SEVERIDAD_MODERADO, "Moderado"),
            (UMBRAL_SEVERIDAD_CRITICO, "Crítico"),
        ):
            fig.add_hline(y=umbral, line_dash="dash", line_color=COLOR_SEVERIDAD[texto],
                          annotation_text=f"Umbral {texto} ({umbral})", annotation_position="top left")
        fig.update_layout(
            title_text="<b>Atribución de la Severidad por Hallazgo (XAI)</b>",
            yaxis_title="Puntos de severidad", showlegend=False, font_size=12,
            xaxis=dict(tickangle=-30),
        )
        return fig
    except Exception as e:
        logger.error("Error generando gráfico de contribuciones: %s", e)
        return None


def figura_consenso_por_principio(indicadores: Optional[Dict[str, Any]]) -> Optional[go.Figure]:
    """Índice de consenso (0-1) por principio, con los cortes de clasificación."""
    por_principio = (indicadores or {}).get("por_principio") or {}
    datos = [(v.get("etiqueta", k), v.get("indice_consenso")) for k, v in por_principio.items()]
    datos = [(e, v) for e, v in datos if v is not None]
    if not datos:
        return None
    try:
        from core.indicadores import CORTE_CONSENSO_ALTO, CORTE_CONSENSO_MODERADO

        colores = [
            "#16A34A" if v >= CORTE_CONSENSO_ALTO
            else "#D97706" if v >= CORTE_CONSENSO_MODERADO
            else "#DC2626"
            for _, v in datos
        ]
        fig = go.Figure(
            go.Bar(x=[e for e, _ in datos], y=[v for _, v in datos], marker_color=colores,
                   text=[f"{v:.2f}" for _, v in datos], textposition="outside")
        )
        fig.add_hline(y=CORTE_CONSENSO_ALTO, line_dash="dot", line_color="#16A34A")
        fig.add_hline(y=CORTE_CONSENSO_MODERADO, line_dash="dot", line_color="#D97706")
        fig.update_layout(title_text="<b>Índice de Consenso por Principio</b>",
                          yaxis=dict(range=[0, 1.15], title="1 = acuerdo total"), font_size=13)
        return fig
    except Exception as e:
        logger.error("Error generando consenso por principio: %s", e)
        return None


# --- DeliberIA: analítica de la base y auditoría de sesgos ---------------------------

def figura_distribucion_severidad(resumen_por_grupo: Dict[str, Dict[str, Any]]) -> Optional[go.Figure]:
    """Barras apiladas de severidad por grupo (p. ej., por dominio clínico)."""
    if not resumen_por_grupo:
        return None
    try:
        grupos = list(resumen_por_grupo)
        fig = go.Figure()
        for nivel in ("Bajo", "Moderado", "Crítico"):
            fig.add_trace(go.Bar(
                name=nivel, x=grupos,
                y=[resumen_por_grupo[g]["distribucion_severidad"].get(nivel, 0) for g in grupos],
                marker_color=COLOR_SEVERIDAD[nivel],
            ))
        fig.update_layout(barmode="stack", title_text="<b>Semáforo Ético por Grupo</b>",
                          yaxis_title="Casos", font_size=12)
        return fig
    except Exception as e:
        logger.error("Error generando distribución de severidad: %s", e)
        return None


def figura_paridad(paridad: Dict[str, Any]) -> Optional[go.Figure]:
    """Tasa de severidad elevada por subgrupo, con la banda de la regla de 4/5."""
    subgrupos = (paridad or {}).get("subgrupos") or []
    if not subgrupos:
        return None
    try:
        x = [f"{s['subgrupo']} (n={s['n']})" for s in subgrupos]
        y = [s["tasa_severidad_elevada"] or 0 for s in subgrupos]
        colores = [COLOR_PRIMARIO if s["suficiente"] else "#CBD5E1" for s in subgrupos]
        fig = go.Figure(go.Bar(x=x, y=y, marker_color=colores,
                               text=[f"{v:.0%}" for v in y], textposition="outside"))
        validos = [s["tasa_severidad_elevada"] for s in subgrupos if s["suficiente"]]
        if validos:
            fig.add_hline(y=max(validos) * 0.8, line_dash="dash", line_color="#DC2626",
                          annotation_text="80% de la tasa máxima (regla 4/5)",
                          annotation_position="bottom right")
        fig.update_layout(
            title_text=f"<b>Paridad de Severidad Elevada — {paridad.get('etiqueta', '')}</b>",
            yaxis=dict(range=[0, 1.1], tickformat=".0%", title="Moderado o Crítico"),
            font_size=12,
        )
        return fig
    except Exception as e:
        logger.error("Error generando gráfico de paridad: %s", e)
        return None


def figura_matriz_calibracion(calibracion: Dict[str, Any]) -> Optional[go.Figure]:
    """Matriz de confusión: riesgo estimado por la IA vs. semáforo determinista."""
    matriz = (calibracion or {}).get("matriz_confusion") or {}
    if not matriz or not calibracion.get("n_pares_riesgo"):
        return None
    try:
        niveles = ["Bajo", "Moderado", "Crítico"]
        z = [[matriz.get(ia, {}).get(m, 0) for m in niveles] for ia in niveles]
        fig = go.Figure(go.Heatmap(
            z=z, x=[f"Motor: {n}" for n in niveles], y=[f"IA: {n}" for n in niveles],
            colorscale="Blues", text=z, texttemplate="%{text}", showscale=False,
        ))
        fig.update_layout(title_text="<b>Calibración IA vs. Motor Determinista</b>", font_size=12)
        return fig
    except Exception as e:
        logger.error("Error generando matriz de calibración: %s", e)
        return None


def figura_histograma(valores: list, titulo: str, eje_x: str, rango: Optional[list] = None) -> Optional[go.Figure]:
    valores = [v for v in valores if v not in ("", None)]
    if not valores:
        return None
    try:
        fig = go.Figure(go.Histogram(x=valores, nbinsx=10, marker_color=COLOR_PRIMARIO))
        fig.update_layout(title_text=f"<b>{titulo}</b>", xaxis_title=eje_x, yaxis_title="Casos",
                          font_size=12, bargap=0.05)
        if rango:
            fig.update_xaxes(range=rango)
        return fig
    except Exception as e:
        logger.error("Error generando histograma: %s", e)
        return None
