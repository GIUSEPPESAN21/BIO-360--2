"""
BIOETHICARE 360º — Punto de entrada de la aplicación Streamlit
==============================================================
Autores: Anderson Díaz Pérez (titular de los derechos de BioEthicCare360®) y
         Joseph Javier Sánchez Acuña (App Web).

Este archivo es deliberadamente DELGADO. Su única responsabilidad es componer la
aplicación: inicializar el estado, abrir las conexiones externas, construir el
orquestador de IA y delegar cada pestaña en su módulo de `ui/`.

La lógica vive en tres capas, y no en este archivo:

    core/      dominio puro — modelos, reglas del semáforo ético, anonimización de PII
    services/  integraciones — Firebase, proveedores de IA, PDF, auditoría
    ui/        interfaz Streamlit — una pestaña por módulo

El monolito anterior (~700 líneas) está preservado en `app_legacy.py` y en el
historial de git. NO debe ejecutarse: enviaba las historias clínicas sin anonimizar a
la API de IA, fijaba los filtros de seguridad en `BLOCK_ONLY_HIGH` de forma
permanente y no registraba nada en el audit log.

Garantías que esta composición hace cumplir en cada llamada a un modelo:
  M2  Anonimización de PII antes de que el texto salga hacia una API externa.
  M6  Proveedor intercambiable (Gemini / OpenAI / Kiro) vía patrón Strategy.
  M8  Registro de auditoría inmutable con modelo, versión de prompt y hash.
"""

from __future__ import annotations

import logging

import streamlit as st

# --- Configuración de página: debe ser la primera llamada a Streamlit -------------
st.set_page_config(layout="wide", page_title="BIOETHICARE 360", page_icon="🏥")

from core.conocimiento import cargar_dilemas, listar_dilemas
from services.ai import PROVEEDORES_DISPONIBLES, get_provider
from services.ai_orchestrator import AIOrchestrator
from services.audit import AuditLog
from services.firebase_service import (
    CasosRepository,
    initialize_firebase_admin,
    initialize_firebase_auth,
)
from ui import tab_acerca_de, tab_analisis, tab_chatbot, tab_configuracion, tab_consultar
from ui.estado import email_usuario, inicializar_estado, uid_usuario
from ui.login import display_login_form

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# --- Recursos cacheados -----------------------------------------------------------
# Los recursos se leen de st.secrets DENTRO de la función: st.secrets no es hashable
# y no puede pasarse como argumento a una función cacheada.

@st.cache_resource(show_spinner=False)
def _firestore_client():
    """Cliente de Firestore (Admin SDK). None si faltan credenciales."""
    return initialize_firebase_admin(st.secrets)


@st.cache_resource(show_spinner=False)
def _auth_app():
    """App de Pyrebase para autenticación de cliente. None si falta configuración."""
    return initialize_firebase_auth(st.secrets)


@st.cache_data(show_spinner=False)
def _dilemas() -> dict:
    """Base de conocimiento de dilemas, validada y saneada."""
    return cargar_dilemas()


# --- Composición de la aplicación principal ---------------------------------------

def _construir_contexto():
    """
    Crea los colaboradores que necesitan las pestañas.

    Returns
    -------
    (orquestador, repo, audit_log, proveedor_ok)
        `repo` y `audit_log` son None cuando no hay base de datos o no hay usuario,
        y las pestañas ya degradan con un mensaje en ese caso.
    """
    db = _firestore_client()
    uid = uid_usuario()

    # Proveedor de IA según la selección del usuario (patrón Strategy, M6).
    # Devuelve None si falta la clave de API de ese proveedor.
    proveedor_nombre = st.session_state.get("ai_provider") or PROVEEDORES_DISPONIBLES[0]
    provider = get_provider(proveedor_nombre, st.secrets.get)
    proveedor_ok = provider is not None

    audit_log = AuditLog(db, uid, email_usuario()) if (db is not None and uid) else None

    repo = None
    if db is not None and uid:
        try:
            repo = CasosRepository(db, uid)
        except ValueError as e:  # uid vacío: no debería ocurrir aquí, pero no rompemos
            logger.error("No se pudo crear el repositorio de casos: %s", e)

    # El orquestador es el ÚNICO camino por el que la app habla con un modelo:
    # garantiza anonimización (M2) y auditoría (M8) en cada llamada.
    orquestador = AIOrchestrator(provider, _dilemas(), audit_log)

    return orquestador, repo, audit_log, proveedor_ok


def display_main_app() -> None:
    """Renderiza la aplicación para un usuario ya autenticado."""
    st.title("BIOETHICARE 360º 🏥")

    dilemas_data = _dilemas()
    dilemas_opciones = listar_dilemas(dilemas_data)

    orquestador, repo, audit_log, proveedor_ok = _construir_contexto()

    if not dilemas_data:
        st.error(
            "No se pudo cargar la base de conocimiento de dilemas (`dilemas.json`). "
            "El registro de casos no estará disponible."
        )

    if not proveedor_ok:
        st.warning(
            f"⚠️ Clave de API para {st.session_state.ai_provider} no encontrada. "
            "Las funciones de IA están deshabilitadas. Revise 'Perfil y Configuración'.",
            icon="⚠️",
        )

    if repo is None:
        st.info(
            "La conexión con la base de datos no está disponible: puede analizar casos "
            "y descargar PDFs, pero no se guardarán ni podrán consultarse después."
        )

    t_analisis, t_chatbot, t_consultar, t_config, t_acerca = st.tabs(
        [
            "**Análisis de Caso**",
            "**Asistente de Bioética (Chatbot)**",
            "**Consultar Casos Anteriores**",
            "**Perfil y Configuración**",
            "**Acerca de**",
        ]
    )

    with t_analisis:
        tab_analisis.render(orquestador, repo, audit_log, dilemas_data, dilemas_opciones)

    with t_chatbot:
        tab_chatbot.render(orquestador, repo)

    with t_consultar:
        tab_consultar.render(repo)

    with t_config:
        tab_configuracion.render(proveedor_ok, PROVEEDORES_DISPONIBLES)

    with t_acerca:
        tab_acerca_de.render()


# --- Flujo principal ---------------------------------------------------------------

def main() -> None:
    inicializar_estado()

    if not st.session_state.get("user"):
        display_login_form(_auth_app())
    else:
        display_main_app()


if __name__ == "__main__":
    main()
