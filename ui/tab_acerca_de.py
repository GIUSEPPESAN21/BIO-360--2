"""
ui/tab_acerca_de.py
===================
Pestaña "Acerca de": propósito de la herramienta, autoría, contacto y aviso de uso
responsable.

El contenido informativo (descripción, autores, enlaces de contacto) se preserva tal
como estaba en el monolito original: en `display_main_app` la autoría vivía en un
expander "Autores" al inicio de la página y el contacto en la pestaña "Acerca de".
Aquí ambos quedan reunidos en un único lugar.
"""

from __future__ import annotations

import streamlit as st

# Datos de contacto, tomados literalmente del app.py original.
URL_LINKEDIN = "https://www.linkedin.com/in/joseph-javier-sánchez-acuña-150410275"
URL_GITHUB = "https://github.com/GIUSEPPESAN21"
EMAIL_CONTACTO = "joseph.sanchez@uniminuto.edu.co"


def render() -> None:
    """Renderiza la pestaña informativa. No recibe argumentos ni toca el estado."""
    _seccion_descripcion()
    st.divider()
    _seccion_autores()
    st.divider()
    _seccion_contacto()
    st.divider()
    _seccion_uso_responsable()


def _seccion_descripcion() -> None:
    st.markdown("### Acerca de esta Herramienta")
    st.markdown(
        "Esta es una suite de software diseñada para asistir a profesionales de la "
        "salud en el análisis y deliberación de casos bioéticos complejos. Utiliza "
        "inteligencia artificial para generar análisis, identificar puntos clave y "
        "facilitar un proceso de toma de decisiones estructurado y ético."
    )


def _seccion_autores() -> None:
    st.markdown("### Autores")
    st.markdown(
        "**Anderson Díaz Pérez** — Creador y titular de los derechos de autor de "
        "BioEthicCare360®.\n\n"
        "_Doctor en Bioética, Doctor en Salud Pública, Magíster en Ciencias Básicas "
        "Biomédicas (énfasis en Inmunología), Especialista en Inteligencia Artificial._"
    )
    st.markdown(
        "**Joseph Javier Sánchez Acuña** — Creador de la App Web.\n\n"
        "_Ingeniero Industrial, Experto en Inteligencia Artificial y Desarrollo de "
        "Software._"
    )


def _seccion_contacto() -> None:
    st.markdown("### Contacto")
    st.markdown(f"🔗 [Perfil de LinkedIn]({URL_LINKEDIN})")
    st.markdown(f"📂 [Repositorio en GitHub]({URL_GITHUB})")
    st.markdown(f"📧 {EMAIL_CONTACTO}")


def _seccion_uso_responsable() -> None:
    st.markdown("### Uso Responsable")
    st.info(
        "Esta herramienta **apoya** la deliberación bioética; **no sustituye** la "
        "decisión del comité de bioética ni del equipo tratante, y no emite "
        "diagnósticos médicos ni indicaciones terapéuticas.\n\n"
        "El semáforo ético (nivel de severidad, advertencias y recomendaciones sobre "
        "las ponderaciones) lo calcula un **motor de reglas determinista**, no un "
        "modelo de IA. La inteligencia artificial se usa para explicar y contextualizar "
        "ese resultado, nunca para decidirlo."
    )
