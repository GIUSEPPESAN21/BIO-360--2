"""
ui/login.py
===========
Formulario de autenticación (Firebase / Pyrebase).
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
            "Revise los secrets de la aplicación (`firebase_client_config`)."
        )
        return

    auth_client = firebase_auth_app.auth()

    with st.container(border=True):
        choice = st.selectbox("Elige una opción", ["Iniciar Sesión", "Registrarse"], key="auth_choice")
        email = st.text_input("Correo electrónico", key="auth_email")
        password = st.text_input("Contraseña", type="password", key="auth_password")

        if choice == "Iniciar Sesión":
            if st.button("Iniciar Sesión", use_container_width=True, type="primary"):
                if not (email and password):
                    st.warning("Por favor, introduce tu email y contraseña.")
                    return
                try:
                    user = auth_client.sign_in_with_email_and_password(email, password)
                    st.session_state.user = user
                    st.rerun()
                except Exception as e:
                    # Mensaje genérico a propósito: no revelar si el email existe
                    st.error("Error: Email o contraseña incorrectos. Verifique sus credenciales.")
                    logger.warning("Fallo en inicio de sesión: %s", e)
        else:
            if st.button("Registrarse", use_container_width=True):
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
                    st.success("¡Cuenta creada exitosamente! Proceda a iniciar sesión.")
                except Exception as e:
                    st.error(
                        "Error al registrar: es posible que el correo ya esté en uso "
                        "o que la contraseña sea muy débil."
                    )
                    logger.warning("Fallo en registro de usuario: %s", e)

    st.markdown("---")
    st.caption(
        "Sistema de deliberación bioética. El acceso está restringido a personal "
        "autorizado. Los datos clínicos se anonimizan antes de cualquier procesamiento "
        "por inteligencia artificial."
    )
