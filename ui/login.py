"""
ui/login.py
===========
Formulario de autenticación contra Firebase.

La autenticación va por la API REST de Identity Toolkit
(`services.auth_service`), no por Pyrebase: esa librería arrastraba `gcloud`, que
importa el `pkg_resources` retirado de las versiones recientes de `setuptools` y hacía
fallar el arranque en Python 3.13.

El objeto de autenticación conserva la forma `app.auth()`, así que este módulo funciona
igual con ambas implementaciones.
"""

from __future__ import annotations

import logging

import streamlit as st

logger = logging.getLogger(__name__)

LONGITUD_MINIMA_PASSWORD = 8


def display_login_form(firebase_auth_app) -> None:
    st.header("BIOETHICARE 360 - Acceso de Usuario")

    if not firebase_auth_app:
        st.error(
            "La configuración de autenticación de Firebase no está disponible. "
            "Revise los secrets de la aplicación (`firebase_client_config`), tomando "
            "como guía el archivo `secrets_template.toml`."
        )
        return

    auth_client = firebase_auth_app.auth()

    with st.container(border=True):
        choice = st.selectbox("Elige una opción", ["Iniciar Sesión", "Registrarse"], key="auth_choice")
        email = st.text_input("Correo electrónico", key="auth_email")
        password = st.text_input("Contraseña", type="password", key="auth_password")

        if choice == "Iniciar Sesión":
            if st.button("Iniciar Sesión", use_container_width=True, type="primary"):
                _iniciar_sesion(auth_client, email, password)
        else:
            if st.button("Registrarse", use_container_width=True):
                _registrar(auth_client, email, password)

    st.markdown("---")
    st.caption(
        "Sistema de deliberación bioética. El acceso está restringido a personal "
        "autorizado. Los datos clínicos se anonimizan antes de cualquier procesamiento "
        "por inteligencia artificial."
    )


def _iniciar_sesion(auth_client, email: str, password: str) -> None:
    if not (email and password):
        st.warning("Por favor, introduce tu email y contraseña.")
        return

    try:
        usuario = auth_client.sign_in_with_email_and_password(email, password)
    except Exception as e:
        # El servicio de autenticación ya devuelve un mensaje apto para el usuario y
        # deliberadamente genérico ante credenciales inválidas: distinguir "email no
        # existe" de "contraseña incorrecta" permitiría enumerar las cuentas.
        st.error(_mensaje_de(e, "No se pudo iniciar sesión. Verifique sus credenciales."))
        logger.warning("Fallo en inicio de sesión: %s", getattr(e, "codigo", "") or e)
        return

    if not usuario or not usuario.get("localId"):
        st.error("La respuesta de autenticación no incluyó un identificador de usuario.")
        return

    st.session_state.user = usuario
    st.rerun()


def _registrar(auth_client, email: str, password: str) -> None:
    if not (email and password):
        st.warning("Introduce un email y contraseña válidos para registrarte.")
        return

    if len(password) < LONGITUD_MINIMA_PASSWORD:
        st.warning(
            f"La contraseña debe tener al menos {LONGITUD_MINIMA_PASSWORD} caracteres. "
            "Esta aplicación maneja datos clínicos sensibles."
        )
        return

    try:
        auth_client.create_user_with_email_and_password(email, password)
    except Exception as e:
        st.error(_mensaje_de(e, "No se pudo completar el registro."))
        logger.warning("Fallo en registro de usuario: %s", getattr(e, "codigo", "") or e)
        return

    st.success("¡Cuenta creada exitosamente! Proceda a iniciar sesión.")


def _mensaje_de(error: Exception, respaldo: str) -> str:
    """Usa el mensaje del servicio de autenticación si lo trae; si no, uno genérico."""
    texto = str(error).strip()
    return texto or respaldo
