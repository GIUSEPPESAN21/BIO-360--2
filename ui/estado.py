"""
ui/estado.py
============
Gestión centralizada del `st.session_state`.

Corrige además el estado engañoso de `selected_model`: antes se inicializaba con
"gemini-2.0-flash-exp" y el sidebar mostraba ese modelo como "activo" aunque nunca
se hubiera hecho una llamada. Ahora arranca en None y solo se puebla con el modelo
realmente usado por el proveedor.
"""

from __future__ import annotations

import tempfile
from typing import Any, Dict

import streamlit as st

SESSION_DEFAULTS: Dict[str, Any] = {
    "reporte": None,
    "temp_dir": None,
    "case_id": None,
    "chat_history": [],
    "last_question": "",
    "dilema_sugerido": None,
    "ai_clinical_analysis_output": "",
    "ai_clinical_analysis_estructurado": None,
    "clinical_history_input": "",
    "user": None,
    "consentimiento_texto": None,
    "ai_provider": "Google Gemini",
    # None = aún no se ha invocado ningún modelo (no se miente al usuario)
    "selected_model": None,
    "ultima_trazabilidad": None,
    # Paginación de casos (M5)
    "casos_pagina": 0,
    "casos_cursores": [],
    "casos_resumenes": [],
}


def inicializar_estado() -> None:
    """Crea las claves del estado de sesión que falten."""
    for key, default in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = (
                list(default) if isinstance(default, list) else default
            )
    if not st.session_state.temp_dir:
        st.session_state.temp_dir = tempfile.mkdtemp()


def reset_caso_activo() -> None:
    """Limpia el caso activo (al iniciar un análisis nuevo)."""
    st.session_state.reporte = None
    st.session_state.case_id = None
    st.session_state.chat_history = []
    st.session_state.consentimiento_texto = None
    st.session_state.ultima_trazabilidad = None


def reset_paginacion() -> None:
    """Reinicia el estado de paginación de la consulta de casos."""
    st.session_state.casos_pagina = 0
    st.session_state.casos_cursores = []
    st.session_state.casos_resumenes = []


def uid_usuario() -> str:
    """UID del usuario autenticado, o cadena vacía."""
    user = st.session_state.get("user")
    if isinstance(user, dict):
        return user.get("localId") or ""
    return ""


def email_usuario() -> str:
    """Email del usuario autenticado, o un marcador."""
    user = st.session_state.get("user")
    if isinstance(user, dict):
        return user.get("email") or "Analista Desconocido"
    return "Analista Desconocido"
