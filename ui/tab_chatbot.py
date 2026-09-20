"""
ui/tab_chatbot.py
=================
Pestaña del asistente de bioética (chatbot contextual).

Corrección del bug de precedencia
---------------------------------
El código original era:

    if prompt := st.chat_input("...") or st.session_state.get('last_question'):

Por precedencia de operadores, Python evalúa `st.chat_input(...) or st.session_state.get(...)`
PRIMERO y asigna ese resultado a `prompt`. Funcionaba por casualidad con las preguntas
guiadas, pero la semántica era incorrecta y frágil.

Ahora se parentiza el walrus explícitamente y se resuelve la fuente de la pregunta
de forma determinista:

    if (entrada := st.chat_input("...")) or st.session_state.get('last_question'):
        pregunta = entrada or st.session_state.last_question
"""

from __future__ import annotations

import logging

import streamlit as st

from core.modelos import safe_str
from services.ai_orchestrator import AIOrchestrator

logger = logging.getLogger(__name__)

PREGUNTAS_GUIADAS = [
    "¿Cuál es el conflicto principal entre los principios bioéticos en este caso?",
    "Desde un punto de vista legal, ¿qué normativas o sentencias son relevantes aquí?",
    "¿Qué estrategias de mediación se podrían usar entre el equipo médico y la familia?",
    "¿Qué cursos de acción alternativos no se han considerado todavía?",
    "¿Cómo influyen los factores culturales o religiosos en la toma de decisiones?",
    "Si priorizamos el principio de beneficencia, ¿cuál sería el curso de acción recomendado?",
    "Analiza el caso a partir de las metodologías de Diego Gracia y Anderson Díaz Pérez (MIEC).",
    "¿Qué metodología sería la más adecuada para analizar el caso? Brinda el propósito y el desarrollo del mismo.",
]


def _set_pregunta(q: str) -> None:
    st.session_state.last_question = q


def render(orquestador: AIOrchestrator, repo) -> None:
    st.header(f"🤖 Asistente de Bioética con {st.session_state.ai_provider}", anchor=False)

    if not st.session_state.get("case_id"):
        st.info("Primero analiza un caso para poder usar el chatbot contextual.")
        return

    st.info(f"Chatbot activo para el caso: **{st.session_state.case_id}**.")
    st.caption(
        "🔒 El caso se envía anonimizado al modelo. Las respuestas se rehidratan "
        "localmente para su lectura."
    )

    st.subheader("Preguntas Guiadas para Deliberación", anchor=False)
    cols = st.columns(2)
    for i, q in enumerate(PREGUNTAS_GUIADAS):
        cols[i % 2].button(
            q, on_click=_set_pregunta, args=(q,), use_container_width=True, key=f"q_{i}"
        )

    # --- Entrada del chat: walrus correctamente parentizado ---
    entrada = st.chat_input("Escribe tu pregunta...")
    pendiente = st.session_state.get("last_question") or ""

    if (pregunta := (entrada or pendiente).strip()):
        st.session_state.last_question = ""

        if not orquestador.disponible:
            st.warning(
                "No hay un proveedor de IA configurado. Verifique las claves en "
                "'Perfil y Configuración'."
            )
        else:
            st.session_state.chat_history.append({"role": "user", "content": pregunta})

            with st.spinner("Pensando..."):
                resultado = orquestador.responder_chat(
                    st.session_state.reporte or {},
                    pregunta,
                    nombres_pii=st.session_state.get("nombres_pii"),
                )

            if resultado.get("error"):
                st.error(resultado["error"])
                # No se persiste una respuesta fallida, pero sí se conserva la pregunta
            else:
                respuesta_texto = resultado["texto"]
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": respuesta_texto}
                )
                respuesta = resultado.get("respuesta")
                if getattr(respuesta, "modelo", ""):
                    st.session_state.selected_model = respuesta.modelo

                if repo is not None and st.session_state.case_id:
                    try:
                        repo.actualizar(
                            st.session_state.case_id,
                            {"Historial del Chat de Deliberación": st.session_state.chat_history},
                        )
                    except Exception as e:
                        logger.error("Error actualizando historial de chat: %s", e)
                        st.warning("No se pudo guardar el historial de chat en la base de datos.")

                if isinstance(st.session_state.reporte, dict):
                    st.session_state.reporte["Historial del Chat de Deliberación"] = (
                        st.session_state.chat_history
                    )
            st.rerun()

    st.subheader("Historial del Chat", anchor=False)
    for msg in st.session_state.chat_history:
        if not isinstance(msg, dict):
            continue
        with st.chat_message(safe_str(msg.get("role"), "assistant")):
            st.markdown(safe_str(msg.get("content")))
