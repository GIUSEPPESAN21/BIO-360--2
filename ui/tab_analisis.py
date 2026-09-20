"""
ui/tab_analisis.py
==================
Pestaña de análisis de caso: análisis previo con IA, registro del caso,
ponderación multiperspectiva, dashboard y descargas.
"""

from __future__ import annotations

import logging
import os

import streamlit as st

from core.etica import verificar_sesgo_etico
from core.modelos import CasoBioetico
from services.ai_orchestrator import MAX_CARACTERES_HISTORIA, AIOrchestrator
from services.audit import ACCION_CASO_CREADO, ACCION_CONSENTIMIENTO
from services.pdf_service import crear_consentimiento_pdf, crear_reporte_pdf_completo
from services.reportes import generar_reporte_completo, generar_texto_consentimiento
from ui.componentes import display_case_details
from ui.estado import email_usuario, reset_caso_activo

logger = logging.getLogger(__name__)


def render(
    orquestador: AIOrchestrator,
    repo,
    audit_log,
    dilemas_data: dict,
    dilemas_opciones: list,
) -> None:
    _seccion_analisis_previo(orquestador)
    _seccion_formulario(orquestador, repo, audit_log, dilemas_data, dilemas_opciones)
    _seccion_resultados(orquestador, repo, audit_log)


# --- 1. Análisis previo de historia clínica ---------------------------------------

def _seccion_analisis_previo(orquestador: AIOrchestrator) -> None:
    st.header("1. Asistente de Análisis Previo (Opcional)", anchor=False)
    st.caption(
        "🔒 La historia clínica se anonimiza automáticamente (nombres, fechas, lugares, "
        "documentos y contactos se reemplazan por marcadores) antes de enviarse al "
        "modelo de IA. Los datos reales no salen de este servidor."
    )

    st.text_area(
        "Pega aquí la historia clínica del paciente...",
        key="clinical_history_input",
        height=250,
        max_chars=MAX_CARACTERES_HISTORIA,
    )

    etiqueta = f"🤖 Analizar Historia Clínica con {st.session_state.ai_provider}"
    if st.button(etiqueta, use_container_width=True, disabled=not orquestador.disponible):
        historia = (st.session_state.clinical_history_input or "").strip()
        if not historia:
            st.warning("Pega la historia clínica antes de analizar.")
            return

        with st.spinner(f"Anonimizando y analizando con {st.session_state.ai_provider}..."):
            resultado = orquestador.analizar_historia_clinica(historia)

        if resultado.get("error"):
            st.error(resultado["error"])
            return

        st.session_state.ai_clinical_analysis_output = resultado["texto_markdown"]
        st.session_state.ai_clinical_analysis_estructurado = resultado.get("estructurado")
        if resultado.get("dilema_sugerido"):
            st.session_state.dilema_sugerido = resultado["dilema_sugerido"]

        respuesta = resultado.get("respuesta")
        if respuesta is not None and getattr(respuesta, "modelo", ""):
            st.session_state.selected_model = respuesta.modelo

    if st.session_state.ai_clinical_analysis_output:
        st.info(st.session_state.ai_clinical_analysis_output)
        if st.session_state.dilema_sugerido:
            st.success(
                f"**Dilema sugerido por IA:** {st.session_state.dilema_sugerido} "
                "(se preselecciona abajo; puede cambiarlo)"
            )


# --- 2. Formulario del caso -------------------------------------------------------

def _seccion_formulario(
    orquestador: AIOrchestrator,
    repo,
    audit_log,
    dilemas_data: dict,
    dilemas_opciones: list,
) -> None:
    st.header("2. Registro y Contexto del Caso", anchor=False)

    if not dilemas_opciones:
        st.error("No se pudo cargar la base de conocimiento de dilemas (dilemas.json).")
        return

    # Preselección del dilema sugerido por la IA (K1)
    indice_dilema = 0
    sugerido = st.session_state.get("dilema_sugerido")
    if sugerido in dilemas_opciones:
        indice_dilema = dilemas_opciones.index(sugerido)

    analista_email = email_usuario()

    with st.form("caso_form"):
        col1, col2 = st.columns(2)
        with col1:
            nombre_paciente = st.text_input("Nombre del Paciente")
            edad = st.number_input("Edad (años)", 0, 120, value=0)
            genero = st.selectbox("Género", ["Masculino", "Femenino", "Otro"])
            semanas_gestacion = st.number_input("Semanas Gestación (si aplica)", 0, 42, value=0)
        with col2:
            historia_clinica = st.text_input("Nº Historia Clínica / ID del Caso")
            st.text_input("Nombre del Analista", value=analista_email, disabled=True)
            condicion = st.selectbox("Condición", ["Estable", "Crítico", "Terminal", "Neonato"])
            dilema_etico = st.selectbox(
                "Dilema Ético Principal", options=dilemas_opciones, index=indice_dilema
            )

        descripcion_caso = st.text_area("Descripción Detallada del Caso", height=150)
        antecedentes_culturales = st.text_area("Contexto Sociocultural y Familiar", height=100)
        puntos_clave_ia = st.text_area("Puntos Clave para Deliberación IA (Opcional)", height=100)

        st.header("3. Ponderación Multiperspectiva (0-5)", anchor=False)
        ponderaciones = {}
        etiquetas = {
            "medico": "Perspectiva del Equipo Médico",
            "familia": "Perspectiva de la Familia / Paciente",
            "comite": "Perspectiva del Comité de Bioética",
        }
        for prefijo, titulo in etiquetas.items():
            with st.expander(titulo):
                c = st.columns(4)
                ponderaciones[f"nivel_autonomia_{prefijo}"] = c[0].slider(
                    "Autonomía", 0, 5, 3, key=f"a_{prefijo}"
                )
                ponderaciones[f"nivel_beneficencia_{prefijo}"] = c[1].slider(
                    "Beneficencia", 0, 5, 3, key=f"b_{prefijo}"
                )
                ponderaciones[f"nivel_no_maleficencia_{prefijo}"] = c[2].slider(
                    "No Maleficencia", 0, 5, 3, key=f"nm_{prefijo}"
                )
                ponderaciones[f"nivel_justicia_{prefijo}"] = c[3].slider(
                    "Justicia", 0, 5, 3, key=f"j_{prefijo}"
                )

        generar_consentimiento = st.checkbox("📄 Generar Consentimiento Informado", value=False)
        explicar_con_ia = st.checkbox(
            "🤖 Explicar el semáforo ético con IA", value=False,
            help="La IA solo explica el resultado del motor de reglas; no lo modifica.",
        )
        submitted = st.form_submit_button(
            "Analizar Caso y Generar Dashboard", use_container_width=True
        )

    if not submitted:
        return

    if not historia_clinica.strip():
        st.error("El campo 'Nº Historia Clínica / ID del Caso' es obligatorio.")
        return

    with st.spinner("Procesando y generando reporte..."):
        reset_caso_activo()

        form_data = {
            "nombre_paciente": nombre_paciente,
            "historia_clinica": historia_clinica,
            "edad": edad,
            "genero": genero,
            "nombre_analista": analista_email,
            "dilema_etico": dilema_etico,
            "descripcion_caso": descripcion_caso,
            "antecedentes_culturales": antecedentes_culturales,
            "condicion": condicion,
            "semanas_gestacion": semanas_gestacion,
            "puntos_clave_ia": puntos_clave_ia,
            "ai_clinical_analysis_summary": st.session_state.ai_clinical_analysis_output,
            **ponderaciones,
        }

        caso = CasoBioetico(dilemas_opciones=dilemas_opciones, **form_data)

        # Semáforo ético determinista
        adv, rec, sev = verificar_sesgo_etico(caso)
        analisis_etico = {"advertencias": adv, "recomendaciones": rec, "severidad": sev}

        # Explicación opcional por IA (no decide, solo explica)
        if explicar_con_ia and orquestador.disponible:
            explicacion = orquestador.explicar_semaforo(
                analisis_etico, dilema_etico, caso.historia_clinica
            )
            if explicacion.get("texto"):
                analisis_etico["explicacion_ia"] = explicacion["texto"]

        reporte = generar_reporte_completo(
            caso,
            dilema_sugerido=st.session_state.get("dilema_sugerido"),
            chat_history=[],
            ethical_analysis=analisis_etico,
            analisis_estructurado=st.session_state.get("ai_clinical_analysis_estructurado"),
        )

        st.session_state.reporte = reporte
        st.session_state.case_id = caso.historia_clinica
        st.session_state.nombres_pii = caso.nombres_pii()

        st.session_state.consentimiento_texto = (
            generar_texto_consentimiento(caso, dilemas_data) if generar_consentimiento else None
        )

        # Persistencia
        if repo is not None:
            try:
                repo.guardar(caso.historia_clinica, reporte)
                st.success(f"Caso '{caso.historia_clinica}' guardado correctamente.")
                if audit_log is not None:
                    audit_log.registrar(
                        ACCION_CASO_CREADO,
                        caso_id=caso.historia_clinica,
                        metadatos={"severidad": sev, "dilema": dilema_etico},
                    )
                    if generar_consentimiento:
                        audit_log.registrar(
                            ACCION_CONSENTIMIENTO, caso_id=caso.historia_clinica
                        )
            except Exception as e:
                logger.error("Error guardando caso %s: %s", caso.historia_clinica, e)
                st.error(f"No se pudo guardar el caso en la base de datos: {e}")

    st.rerun()


# --- 3. Resultados, análisis deliberativo y descargas -----------------------------

def _seccion_resultados(orquestador: AIOrchestrator, repo, audit_log) -> None:
    reporte = st.session_state.get("reporte")
    if not reporte:
        return

    st.markdown("---")
    display_case_details(reporte, key_prefix="active")

    a1, a2, a3 = st.columns([2, 1, 1])

    etiqueta = f"🤖 Generar Análisis Deliberativo con {st.session_state.ai_provider}"
    if a1.button(
        etiqueta, use_container_width=True, key="gen_analysis_button",
        disabled=not orquestador.disponible,
    ):
        with st.spinner(f"Contactando a {st.session_state.ai_provider}..."):
            resultado = orquestador.generar_deliberacion(
                reporte, nombres_pii=st.session_state.get("nombres_pii")
            )

        if resultado.get("error"):
            st.error(resultado["error"])
        else:
            analisis = resultado["texto"]
            respuesta = resultado.get("respuesta")
            traza = AIOrchestrator.trazabilidad(respuesta)

            st.session_state.reporte["Análisis Deliberativo (IA)"] = analisis
            st.session_state.reporte["Trazabilidad IA"] = traza
            st.session_state.ultima_trazabilidad = traza
            if getattr(respuesta, "modelo", ""):
                st.session_state.selected_model = respuesta.modelo

            if repo is not None and st.session_state.case_id:
                try:
                    repo.actualizar(
                        st.session_state.case_id,
                        {
                            "Análisis Deliberativo (IA)": analisis,
                            "Trazabilidad IA": traza,
                        },
                    )
                except Exception as e:
                    logger.error("Error actualizando análisis deliberativo: %s", e)
                    st.warning("El análisis se generó pero no se pudo guardar en la base de datos.")
            st.rerun()

    _boton_pdf_reporte(a2)
    _boton_pdf_consentimiento(a3)


def _boton_pdf_reporte(columna) -> None:
    try:
        nombre = f"Reporte_{st.session_state.get('case_id') or 'caso'}.pdf"
        ruta = os.path.join(st.session_state.temp_dir, nombre)
        crear_reporte_pdf_completo(st.session_state.reporte, ruta)
        with open(ruta, "rb") as f:
            columna.download_button(
                "📄 Descargar Reporte PDF", f, os.path.basename(ruta),
                "application/pdf", use_container_width=True, key="download_pdf_button",
            )
    except Exception as e:
        logger.error("Error generando PDF del reporte: %s", e)
        columna.error("Error al generar PDF.")


def _boton_pdf_consentimiento(columna) -> None:
    if not st.session_state.get("consentimiento_texto"):
        return
    try:
        nombre = f"Consentimiento_{st.session_state.get('case_id') or 'caso'}.pdf"
        ruta = os.path.join(st.session_state.temp_dir, nombre)
        crear_consentimiento_pdf(st.session_state.consentimiento_texto, ruta)
        with open(ruta, "rb") as f:
            columna.download_button(
                "✍️ Descargar Consentimiento", f, os.path.basename(ruta),
                "application/pdf", use_container_width=True, key="download_consent_button",
            )
    except Exception as e:
        logger.error("Error generando PDF de consentimiento: %s", e)
        columna.error("Error al generar PDF de consentimiento.")
