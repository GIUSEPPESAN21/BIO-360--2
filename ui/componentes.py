"""
ui/componentes.py
=================
Componentes de UI compartidos: dashboard de detalle de un caso.

Cambio clave (M5): los gráficos se REGENERAN desde las ponderaciones del reporte con
`services.charts`, en lugar de deserializar JSON de Plotly guardados en Firestore.
Esto funciona igual para casos nuevos (schema v2) y antiguos (schema v1).
"""

from __future__ import annotations

import logging

import streamlit as st

from core.modelos import PRINCIPIOS, PRINCIPIOS_LABELS, safe_int, safe_str
from services.charts import figura_consenso, figura_equilibrio, figura_radar

logger = logging.getLogger(__name__)

COLOR_SEVERIDAD = {"Bajo": "#28a745", "Moderado": "#ffc107", "Crítico": "#dc3545"}


def display_case_details(report_data: dict, key_prefix: str) -> None:
    """Renderiza el dashboard completo de un caso."""
    try:
        case_id = safe_str(report_data.get("ID del Caso", "caso_desconocido"))
        sanitized_id = "".join(filter(str.isalnum, case_id)) or "caso"

        st.subheader(f"Dashboard del Caso: `{case_id}`", anchor=False)
        st.markdown("---")

        _seccion_semaforo(report_data)
        st.markdown("---")
        _seccion_graficos(report_data, key_prefix, sanitized_id)
        st.markdown("---")

        if report_data.get("Análisis Deliberativo (IA)"):
            st.markdown("##### Análisis Deliberativo por IA")
            st.info(report_data["Análisis Deliberativo (IA)"])
            _seccion_trazabilidad(report_data)
            st.markdown("---")

        _seccion_resumen(report_data, key_prefix, sanitized_id)

    except Exception as e:
        logger.error("Error fatal en display_case_details: %s", e)
        st.error("Ocurrió un error al mostrar los detalles del caso. Revise los logs.")


def _seccion_semaforo(report_data: dict) -> None:
    analisis = report_data.get("AnalisisEtico") or {}
    if not analisis:
        return

    severidad = analisis.get("severidad", "Bajo")
    advertencias = analisis.get("advertencias", []) or []
    color = COLOR_SEVERIDAD.get(severidad, "#6c757d")

    st.markdown(
        f"<h5 style='color:{color};'>Análisis de Coherencia Ética: {severidad}</h5>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Resultado calculado por un motor de reglas determinista, no por un modelo de IA."
    )

    if advertencias:
        with st.expander(
            "Ver detalles y recomendaciones del análisis ético",
            expanded=(severidad != "Bajo"),
        ):
            for adv in advertencias:
                st.warning(adv)
            recomendaciones = analisis.get("recomendaciones", []) or []
            if recomendaciones:
                st.info(f"**Recomendaciones:** {' '.join(recomendaciones)}")
            if analisis.get("explicacion_ia"):
                st.markdown("**Explicación generada por IA:**")
                st.info(analisis["explicacion_ia"])
    else:
        st.success("El análisis no encontró desequilibrios éticos significativos.")


def _seccion_graficos(report_data: dict, key_prefix: str, sanitized_id: str) -> None:
    st.markdown("##### Visualizaciones del Caso")
    perspectivas = report_data.get("AnalisisMultiperspectiva") or {}

    if not perspectivas:
        st.info("No hay ponderaciones registradas para graficar.")
        return

    tab1, tab2 = st.tabs(["Análisis de Perspectivas", "Análisis Comparativo de Principios"])

    with tab1:
        fig_radar = figura_radar(perspectivas)
        fig_consenso = figura_consenso(perspectivas)
        if fig_radar is None and fig_consenso is None:
            st.warning("No se pudieron generar los gráficos de perspectivas.")
        else:
            c1, c2 = st.columns(2)
            if fig_radar is not None:
                c1.plotly_chart(
                    fig_radar, use_container_width=True,
                    key=f"{key_prefix}_radar_{sanitized_id}",
                )
            if fig_consenso is not None:
                c2.plotly_chart(
                    fig_consenso, use_container_width=True,
                    key=f"{key_prefix}_stats_{sanitized_id}",
                )

    with tab2:
        fig_equilibrio = figura_equilibrio(perspectivas)
        if fig_equilibrio is not None:
            st.plotly_chart(
                fig_equilibrio, use_container_width=True,
                key=f"{key_prefix}_equilibrio_{sanitized_id}",
            )
        else:
            st.info("Gráfico de equilibrio no disponible.")


def _seccion_trazabilidad(report_data: dict) -> None:
    traza = report_data.get("Trazabilidad IA") or {}
    if not traza:
        return
    partes = []
    if traza.get("proveedor"):
        partes.append(f"Proveedor: {traza['proveedor']}")
    if traza.get("modelo"):
        partes.append(f"Modelo: `{traza['modelo']}`")
    if traza.get("prompt_version"):
        partes.append(f"Prompt: `{traza['prompt_version']}`")
    if traza.get("timestamp"):
        partes.append(f"UTC: {traza['timestamp']}")
    if partes:
        st.caption("Trazabilidad — " + " · ".join(partes))


def _seccion_resumen(report_data: dict, key_prefix: str, sanitized_id: str) -> None:
    st.markdown("##### Resumen y Contexto del Caso")
    col_a, col_b = st.columns(2)
    col_a.markdown(f"**Paciente:** {safe_str(report_data.get('Resumen del Paciente'))}")
    col_a.markdown(f"**Analista:** {safe_str(report_data.get('Analista'))}")
    col_b.markdown(
        f"**Dilema Seleccionado:** {safe_str(report_data.get('Dilema Ético Principal (Seleccionado)'))}"
    )
    if report_data.get("Dilema Sugerido por IA"):
        col_b.markdown(f"**Dilema Sugerido por IA:** {safe_str(report_data.get('Dilema Sugerido por IA'))}")

    with st.expander("Ver Detalles Completos, Ponderación y Chat"):
        st.text_area(
            "Descripción:",
            value=safe_str(report_data.get("Descripción Detallada del Caso")),
            height=150, disabled=True, key=f"{key_prefix}_desc_{sanitized_id}",
        )
        st.text_area(
            "Contexto Sociocultural:",
            value=safe_str(report_data.get("Contexto Sociocultural y Familiar")),
            height=100, disabled=True, key=f"{key_prefix}_context_{sanitized_id}",
        )

        if report_data.get("Análisis IA de Historia Clínica"):
            st.markdown("**Análisis IA de Historia Clínica (Elementos Clave)**")
            st.info(report_data["Análisis IA de Historia Clínica"])

        st.markdown("**Ponderación por Perspectiva (escala 0-5)**")
        multiperspectiva = report_data.get("AnalisisMultiperspectiva") or {}
        if isinstance(multiperspectiva, dict):
            for nombre, valores in multiperspectiva.items():
                if not isinstance(valores, dict):
                    continue
                st.markdown(f"**{nombre}**")
                cols = st.columns(4)
                for i, (label, clave) in enumerate(zip(PRINCIPIOS_LABELS, PRINCIPIOS)):
                    cols[i].metric(label, safe_int(valores.get(clave, 0)))

        st.markdown("**Historial del Chat**")
        chat = report_data.get("Historial del Chat de Deliberación") or []
        if chat:
            for msg in chat:
                if not isinstance(msg, dict):
                    continue
                with st.chat_message(safe_str(msg.get("role"), "assistant")):
                    st.markdown(safe_str(msg.get("content")))
        else:
            st.info("No hay historial de chat disponible.")
