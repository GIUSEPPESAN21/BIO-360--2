"""
ui/tab_consultar.py
===================
Pestaña de consulta de casos guardados, con paginación real (M5).

Antes se hacía `.stream()` sobre toda la colección y se materializaba cada reporte
completo en memoria. Ahora se piden páginas de tamaño fijo con `limit()` + cursor, y
el listado solo descarga campos de resumen; el reporte completo se lee únicamente
cuando el usuario selecciona un caso.
"""

from __future__ import annotations

import logging

import streamlit as st

from services.firebase_service import PAGINA_POR_DEFECTO
from ui.componentes import display_case_details
from ui.estado import reset_paginacion

logger = logging.getLogger(__name__)


def render(repo) -> None:
    st.header("🔍 Consultar Mis Casos Guardados", anchor=False)

    if repo is None:
        st.error("La conexión con Firebase no está disponible.")
        return

    c1, c2 = st.columns([1, 3])
    tamano = c1.selectbox("Casos por página", [10, 20, 50], index=1, key="casos_por_pagina")
    if c2.button("🔄 Recargar listado", use_container_width=False):
        reset_paginacion()

    try:
        cursor = None
        pagina = st.session_state.get("casos_pagina", 0)
        cursores = st.session_state.get("casos_cursores", [])
        if pagina > 0 and len(cursores) >= pagina:
            cursor = cursores[pagina - 1]

        resumenes, siguiente = repo.listar_pagina(limite=tamano, cursor=cursor)
        st.session_state.casos_resumenes = resumenes

        if not resumenes and pagina == 0:
            st.info("No tienes casos guardados.")
            return

        st.caption(f"Página {pagina + 1} · {len(resumenes)} caso(s) en esta página")

        opciones = {
            f"{r['ID del Caso']} — {r.get('Fecha Análisis', 's/f')}": r["id"]
            for r in resumenes
        }
        etiqueta_sel = st.selectbox(
            "Selecciona un caso para ver sus detalles",
            options=list(opciones.keys()),
            key="case_selector_consultar",
        )

        nav1, nav2, _ = st.columns([1, 1, 3])
        if nav1.button("⬅️ Anterior", disabled=(pagina == 0), use_container_width=True):
            st.session_state.casos_pagina = max(0, pagina - 1)
            st.rerun()
        if nav2.button("Siguiente ➡️", disabled=(siguiente is None), use_container_width=True):
            cursores = list(cursores[:pagina])
            cursores.append(siguiente)
            st.session_state.casos_cursores = cursores
            st.session_state.casos_pagina = pagina + 1
            st.rerun()

        if etiqueta_sel:
            caso_id = opciones[etiqueta_sel]
            # Lectura completa solo del caso seleccionado (M5)
            with st.spinner("Cargando caso..."):
                reporte = repo.obtener(caso_id)
            if reporte is None:
                st.warning("El caso seleccionado ya no está disponible.")
            else:
                display_case_details(reporte, key_prefix="consult")

    except Exception as e:
        logger.error("Error consultando casos desde Firestore: %s", e)
        st.error(f"Ocurrió un error al consultar tus casos: {e}")
