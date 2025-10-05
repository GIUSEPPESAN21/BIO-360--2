# BIOETHICARE 360º
# Autores: Anderson Díaz Pérez & Joseph Javier Sánchez Acuña

# --- 1. Importaciones ---
import os
import json
import requests
import streamlit as st
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import tempfile
import shutil
import plotly.io as pio
import time
import logging

# --- MEJORA: Importación para OpenAI ---
from openai import OpenAI

# Importaciones para PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, HRFlowable
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.units import inch
from reportlab.lib import colors

# Se añade pyrebase para la autenticación del cliente
import firebase_admin
from firebase_admin import credentials, firestore
import pyrebase

# --- 2. Configuración de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 3. Configuración Inicial y Estado de la Sesión ---
st.set_page_config(layout="wide", page_title="BIOETHICARE 360")

session_defaults = {
    'reporte': None,
    'temp_dir': None,
    'case_id': None,
    'chat_history': [],
    'last_question': "",
    'dilema_sugerido': None,
    'ai_clinical_analysis_output': "",
    'clinical_history_input': "",
    'key_counter': 0,
    'user': None, 
    'consentimiento_texto': None,
    'ai_provider': 'Google Gemini' # Proveedor de IA por defecto
}
for key, default_value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# --- 4. Funciones Utilitarias ---
def safe_int(value, default=0):
    if value is None or value == '': return default
    try: return int(value)
    except (ValueError, TypeError): return default

def safe_str(value, default=""):
    if value is None: return default
    return str(value).strip()

def log_error(error_msg, exception=None):
    logger.error(f"BIOETHICARE ERROR: {error_msg}")
    if exception: logger.error(f"Exception details: {str(exception)}")

# --- 5. MÓDULO DE ANÁLISIS ÉTICO ---
def verificar_sesgo_etico(caso):
    advertencias = []
    recomendaciones = []
    puntos_severidad = 0
    for nombre, valores in caso.perspectivas.items():
        if sum(valores.values()) == 0:
            advertencias.append(f"**Perspectiva Omitida:** La perspectiva de '{nombre}' no asignó puntuación a ningún principio.")
            recomendaciones.append(f"Se recomienda verificar si la ponderación de '{nombre}' fue omitida accidentalmente para asegurar una deliberación completa.")
            puntos_severidad += 3
        for principio, valor in valores.items():
            if valor == 0:
                advertencias.append(f"**Principio Omitido en '{nombre.title()}':** El principio de '{principio.replace('_', ' ').capitalize()}' tiene un valor de 0.")
                recomendaciones.append(f"Evaluar si la omisión del principio de '{principio.replace('_', ' ').capitalize()}' en la perspectiva de '{nombre.title()}' es intencional y justificada.")
                puntos_severidad += 1
        if sum(valores.values()) > 0:
            max_diff = max(valores.values()) - min(valores.values())
            if max_diff >= 4:
                advertencias.append(f"**Alto Desequilibrio Interno:** En la perspectiva de '{nombre.title()}', existe un alto desequilibrio entre los principios (diferencia de {max_diff} puntos).")
                recomendaciones.append("Se sugiere revisar si la alta disparidad en la ponderación de esta perspectiva está suficientemente justificada o si requiere una deliberación más balanceada.")
                puntos_severidad += 2
    puntajes_totales = {nombre: sum(valores.values()) for nombre, valores in caso.perspectivas.items()}
    if len(puntajes_totales) > 1:
        max_perspectiva = max(puntajes_totales, key=puntajes_totales.get)
        min_perspectiva = min(puntajes_totales, key=puntajes_totales.get)
        if puntajes_totales[max_perspectiva] - puntajes_totales[min_perspectiva] >= 8:
            advertencias.append(f"**Alto Desequilibrio Externo:** La perspectiva de '{max_perspectiva.title()}' tiene un peso total significativamente mayor que la de '{min_perspectiva.title()}'.")
            recomendaciones.append("Analizar si esta dominancia de una perspectiva sobre otra es adecuada para el caso o si es necesario re-equilibrar las ponderaciones para una decisión más equitativa.")
            puntos_severidad += 2
    if puntos_severidad >= 5:
        severidad = "Crítico"
    elif puntos_severidad >= 2:
        severidad = "Moderado"
    else:
        severidad = "Bajo"
    return advertencias, recomendaciones, severidad

def generar_grafico_equilibrio_etico(caso):
    try:
        fig = go.Figure()
        colores = {"medico": "#EF4444", "familia": "#3B82F6", "comite": "#22C55E"}
        nombres_perspectivas = {'medico': 'Equipo Médico', 'familia': 'Familia/Paciente', 'comite': 'Comité de Bioética'}
        for nombre_corto, valores in caso.perspectivas.items():
            nombre_largo = nombres_perspectivas.get(nombre_corto, nombre_corto.capitalize())
            fig.add_trace(go.Bar(
                x=["Autonomía", "Beneficencia", "No Maleficencia", "Justicia"],
                y=list(valores.values()),
                name=nombre_largo,
                marker_color=colores[nombre_corto]
            ))
        fig.update_layout(
            title_text="<b>Análisis Comparativo de Principios</b>",
            barmode="group",
            yaxis=dict(title="Puntaje Asignado", range=[0, 5.5]),
            legend_title_text="Perspectivas",
            font_size=12,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#2E3A47'
        )
        return fig.to_json()
    except Exception as e:
        log_error("Error generando gráfico de equilibrio ético", e)
        return None

# --- 6. Conexión con Firebase ---
@st.cache_resource
def initialize_firebase_admin():
    try:
        if "firebase_credentials" in st.secrets:
            creds_dict = dict(st.secrets["firebase_credentials"])
            creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
            cred = credentials.Certificate(creds_dict)
            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred)
            logger.info("Conexión con Firebase Admin SDK establecida.")
            return firestore.client()
        else:
            log_error("Credenciales de Firebase Admin no encontradas en st.secrets.")
            return None
    except Exception as e:
        log_error("Error crítico al conectar con Firebase Admin SDK", e)
        return None

@st.cache_resource
def initialize_firebase_auth():
    try:
        if "firebase_client_config" in st.secrets:
            firebase_client_config = dict(st.secrets["firebase_client_config"])
            return pyrebase.initialize_app(firebase_client_config)
        else:
            log_error("Configuración de cliente de Firebase (firebase_client_config) no encontrada en st.secrets.")
            return None
    except Exception as e:
        log_error("Error crítico al inicializar Pyrebase para autenticación", e)
        return None

db = initialize_firebase_admin()
firebase_auth_app = initialize_firebase_auth()

# --- 7. Base de Conocimiento ---
@st.cache_data
def cargar_dilemas():
    try:
        with open('dilemas.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        log_error("El archivo dilemas.json no fue encontrado.")
        st.error("Error: No se pudo cargar la base de conocimiento de dilemas.")
        return {}
    except json.JSONDecodeError:
        log_error("Error al decodificar el archivo dilemas.json.")
        st.error("Error: El formato del archivo de dilemas es inválido.")
        return {}


dilemas_data = cargar_dilemas()
dilemas_opciones = list(dilemas_data.keys())


# --- 8. Clases de Modelo ---
class CasoBioetico:
    def __init__(self, **kwargs):
        self.nombre_paciente = safe_str(kwargs.get('nombre_paciente'), 'N/A')
        self.historia_clinica = safe_str(kwargs.get('historia_clinica'), f"caso_{int(datetime.now().timestamp())}")
        self.edad = safe_int(kwargs.get('edad'))
        self.genero = safe_str(kwargs.get('genero'), 'N/A')
        self.nombre_analista = safe_str(kwargs.get('nombre_analista'), 'N/A')
        self.dilema_etico = safe_str(kwargs.get('dilema_etico', dilemas_opciones[0] if dilemas_opciones else ""))
        self.descripcion_caso = safe_str(kwargs.get('descripcion_caso'))
        self.antecedentes_culturales = safe_str(kwargs.get('antecedentes_culturales'))
        self.condicion = safe_str(kwargs.get('condicion', 'Estable'))
        self.semanas_gestacion = safe_int(kwargs.get('semanas_gestacion'))
        self.puntos_clave_ia = safe_str(kwargs.get('puntos_clave_ia'))
        self.ai_clinical_analysis_summary = safe_str(kwargs.get('ai_clinical_analysis_summary'))
        self.perspectivas = {
            "medico": self._extract_perspective("medico", kwargs),
            "familia": self._extract_perspective("familia", kwargs),
            "comite": self._extract_perspective("comite", kwargs),
        }
    def _extract_perspective(self, prefix, kwargs):
        return {
            "autonomia": safe_int(kwargs.get(f'nivel_autonomia_{prefix}')),
            "beneficencia": safe_int(kwargs.get(f'nivel_beneficencia_{prefix}')),
            "no_maleficencia": safe_int(kwargs.get(f'nivel_no_maleficencia_{prefix}')),
            "justicia": safe_int(kwargs.get(f'nivel_justicia_{prefix}')),
        }

# --- 9. Funciones de Generación de Reportes ---
def generar_reporte_completo(caso, dilema_sugerido, chat_history, chart_jsons, ethical_analysis):
    resumen_paciente = f"Paciente {caso.nombre_paciente}, {caso.edad} años, género {caso.genero}, condición {caso.condicion}."
    if caso.semanas_gestacion > 0:
        resumen_paciente += f" Neonato de {caso.semanas_gestacion} sem."
    return {
        "ID del Caso": caso.historia_clinica, "Fecha Análisis": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Analista": caso.nombre_analista, "Resumen del Paciente": resumen_paciente,
        "Dilema Ético Principal (Seleccionado)": caso.dilema_etico, "Dilema Sugerido por IA": dilema_sugerido or "",
        "Descripción Detallada del Caso": caso.descripcion_caso, "Contexto Sociocultural y Familiar": caso.antecedentes_culturales,
        "Puntos Clave para Deliberación IA": caso.puntos_clave_ia, "Análisis IA de Historia Clínica": caso.ai_clinical_analysis_summary,
        "AnalisisMultiperspectiva": {"Equipo Médico": caso.perspectivas["medico"], "Familia/Paciente": caso.perspectivas["familia"], "Comité de Bioética": caso.perspectivas["comite"]},
        "AnalisisEtico": ethical_analysis, "Análisis Deliberativo (IA)": "", "Historial del Chat de Deliberación": chat_history,
        "radar_chart_json": chart_jsons.get('radar_comparativo_json'), "stats_chart_json": chart_jsons.get('stats_chart_json'),
        "equilibrio_chart_json": chart_jsons.get('equilibrio_chart_json'),
    }

def generar_visualizaciones_avanzadas(caso):
    try:
        perspectivas_data = caso.perspectivas
        labels = ["Autonomía", "Beneficencia", "No Maleficencia", "Justicia"]
        fig_radar = go.Figure()
        colors_map = {'medico': 'rgba(239, 68, 68, 0.7)', 'familia': 'rgba(59, 130, 246, 0.7)', 'comite': 'rgba(34, 197, 94, 0.7)'}
        nombres = {'medico': 'Equipo Médico', 'familia': 'Familia/Paciente', 'comite': 'Comité de Bioética'}
        for key, data in perspectivas_data.items():
            fig_radar.add_trace(go.Scatterpolar(r=list(data.values()), theta=labels, fill='toself', name=nombres[key], line_color=colors_map[key]))
        fig_radar.update_layout(title_text="<b>Ponderación por Perspectiva</b>", polar=dict(radialaxis=dict(visible=True, range=[0, 5])), showlegend=True, font_size=14)
        scores = np.array([list(d.values()) for d in perspectivas_data.values()])
        fig_stats = go.Figure()
        fig_stats.add_trace(go.Bar(x=labels, y=np.mean(scores, axis=0), error_y=dict(type='data', array=np.std(scores, axis=0), visible=True), marker_color='#636EFA'))
        fig_stats.update_layout(title_text="<b>Análisis de Consenso y Disenso</b>", yaxis=dict(range=[0, 6]), font_size=14)
        return {'radar_comparativo_json': fig_radar.to_json(), 'stats_chart_json': fig_stats.to_json()}
    except Exception as e:
        log_error("Error generando visualizaciones", e)
        return {'radar_comparativo_json': None, 'stats_chart_json': None}

def crear_reporte_pdf_completo(data, filename):
    try:
        doc = SimpleDocTemplate(filename, pagesize=letter, topMargin=inch/2, bottomMargin=inch/2)
        styles = getSampleStyleSheet()
        story = []
        h1 = ParagraphStyle(name='H1', fontSize=18, fontName='Helvetica-Bold', alignment=TA_CENTER, spaceAfter=20)
        h2 = ParagraphStyle(name='H2', fontSize=14, fontName='Helvetica-Bold', spaceBefore=12, spaceAfter=6, textColor=colors.darkblue)
        body = ParagraphStyle(name='Body', fontSize=10, fontName='Helvetica', leading=14, alignment=TA_JUSTIFY, spaceAfter=10)
        chat_style = ParagraphStyle(name='Chat', fontSize=9, fontName='Helvetica-Oblique', backColor=colors.whitesmoke, borderWidth=1, padding=5)
        story.append(Paragraph("Reporte Deliberativo - BIOETHICARE 360", h1))
        order = ["ID del Caso", "Fecha Análisis", "Analista", "Resumen del Paciente", "Dilema Ético Principal (Seleccionado)", "Dilema Sugerido por IA", "Descripción Detallada del Caso", "Contexto Sociocultural y Familiar", "Puntos Clave para Deliberación IA", "Análisis IA de Historia Clínica"]
        for key in order:
            if data.get(key):
                story.append(Paragraph(key, h2))
                story.append(Paragraph(safe_str(data[key]).replace('\n', '<br/>'), body))
        if "AnalisisEtico" in data:
            story.append(Paragraph("Análisis de Coherencia Ética", h2))
            analisis = data["AnalisisEtico"]
            story.append(Paragraph(f"<b>Nivel de Severidad:</b> {analisis.get('severidad', 'N/A')}", body))
            for adv in analisis.get("advertencias", []):
                story.append(Paragraph(f"<li>{adv}</li>", body))
            story.append(Paragraph(f"<b>Recomendaciones:</b> {' '.join(analisis.get('recomendaciones', []))}", body))
        if "AnalisisMultiperspectiva" in data:
            story.append(Paragraph("Análisis Multiperspectiva", h2))
            for nombre, valores in data["AnalisisMultiperspectiva"].items():
                texto = f"<b>{nombre}:</b> Autonomía: {valores.get('autonomia', 0)}, Beneficencia: {valores.get('beneficencia', 0)}, No Maleficencia: {valores.get('no_maleficencia', 0)}, Justicia: {valores.get('justicia', 0)}"
                story.append(Paragraph(texto, body))
        if data.get("Análisis Deliberativo (IA)"):
            story.append(Paragraph("Análisis Deliberativo (IA)", h2))
            story.append(Paragraph(safe_str(data["Análisis Deliberativo (IA)"]).replace('\n', '<br/>'), body))
        story.append(PageBreak())
        story.append(Paragraph("Visualizaciones de Datos", h1))
        story.append(Paragraph("Los gráficos de radar y consenso/disenso se muestran de forma interactiva en la aplicación web.", body))
        if data.get("Historial del Chat de Deliberación"):
            story.append(PageBreak())
            story.append(Paragraph("Historial del Chat de Deliberación", h1))
            for msg in data["Historial del Chat de Deliberación"]:
                role_text = f"<b>{safe_str(msg.get('role', 'unknown')).capitalize()}:</b> {safe_str(msg.get('content'))}"
                story.append(Paragraph(role_text, chat_style))
        doc.build(story)
        logger.info(f"PDF generado exitosamente: {filename}")
    except Exception as e:
        log_error(f"Error generando PDF {filename}", e)
        raise e

def generar_texto_consentimiento(caso):
    dilema_info = dilemas_data.get(caso.dilema_etico, {})
    riesgos = "\n".join([f"- {r}" for r in dilema_info.get("riesgos", ["No especificados"])])
    beneficios = "\n".join([f"- {b}" for b in dilema_info.get("beneficios", ["No especificados"])])
    alternativas = "\n".join([f"- {a}" for a in dilema_info.get("alternativas", ["No especificadas"])])
    normativas = "\n".join([f"- {n}" for n in dilema_info.get("normativas", ["No especificadas"])])
    texto = f"""
CONSENTIMIENTO/ASENTIMIENTO INFORMADO (BIOETHICARE 360)
Fecha: {datetime.now().strftime("%Y-%m-%d")}
ID del Caso: {caso.historia_clinica}
------------------------------------------------------------------
DATOS DEL PACIENTE
------------------------------------------------------------------
Nombre: {caso.nombre_paciente}
Edad: {caso.edad} años
Género: {caso.genero}
Dilema Ético Principal: {caso.dilema_etico}
------------------------------------------------------------------
INFORMACIÓN SOBRE LA DECISIÓN
------------------------------------------------------------------
En el contexto de su situación clínica, se ha identificado un dilema ético principal relacionado con "{caso.dilema_etico}". A continuación, se presenta la información relevante para que usted (o su representante) pueda tomar una decisión informada.
1. RIESGOS POTENCIALES:
{riesgos}
2. BENEFICIOS ESPERADOS:
{beneficios}
3. ALTERNATIVAS DISPONIBLES:
{alternativas}
4. MARCO NORMATIVO Y ÉTICO:
Esta deliberación se enmarca en las siguientes normativas y principios:
{normativas}
------------------------------------------------------------------
DECLARACIÓN Y FIRMA
------------------------------------------------------------------
Declaro que he leído (o me han leído) y comprendido la información anterior. He tenido la oportunidad de hacer preguntas y todas han sido respondidas a mi satisfacción.
Entiendo que mi decisión es voluntaria y que puedo retirarla en cualquier momento sin que ello afecte la calidad de mi atención médica.
Firma del Paciente/Tutor Legal: _________________________
Nombre: _________________________
Fecha: _________________________
Firma del Profesional de la Salud: _________________________
Nombre: {caso.nombre_analista}
Fecha: _________________________
"""
    return texto

def crear_consentimiento_pdf(texto, filename):
    try:
        doc = SimpleDocTemplate(filename, pagesize=letter, topMargin=inch/2, bottomMargin=inch/2)
        styles = getSampleStyleSheet()
        story = []
        h1 = ParagraphStyle(name='H1', fontSize=14, fontName='Helvetica-Bold', alignment=TA_CENTER, spaceAfter=18)
        h2 = ParagraphStyle(name='H2', fontSize=11, fontName='Helvetica-Bold', spaceBefore=10, spaceAfter=4, textColor=colors.darkblue)
        body = ParagraphStyle(name='Body', fontSize=10, fontName='Helvetica', leading=14, alignment=TA_LEFT, spaceAfter=8)
        lines = texto.split('\n')
        for line in lines:
            if line.isupper() and not line.startswith("-"):
                if "CONSENTIMIENTO" in line:
                    story.append(Paragraph(line, h1))
                else:
                    story.append(Spacer(1, 0.1*inch))
                    story.append(Paragraph(line, h2))
                    story.append(HRFlowable(width="100%", thickness=1, color=colors.black))
            else:
                story.append(Paragraph(line.replace('\n', '<br/>'), body))
        doc.build(story)
        logger.info(f"PDF de consentimiento generado: {filename}")
    except Exception as e:
        log_error(f"Error generando PDF de consentimiento {filename}", e)
        raise e

# --- 10. APIs de IA ---
# --- MEJORA: Manejo de errores más detallado ---
def llamar_gemini(prompt, api_key):
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        result = response.json()
        if 'candidates' in result and result['candidates'] and 'content' in result['candidates'][0] and 'parts' in result['candidates'][0]['content']:
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            log_error(f"Respuesta inesperada de Gemini: {result}")
            st.error("Respuesta inválida de la API de Gemini.")
            return "Respuesta inválida de la API de Gemini."
    except requests.exceptions.HTTPError as http_err:
        error_details = "No se pudieron obtener detalles del error."
        try:
            error_details_json = http_err.response.json()
            error_details = error_details_json.get("error", {}).get("message", json.dumps(error_details_json))
        except json.JSONDecodeError:
            error_details = http_err.text
        log_error(f"Error HTTP con Gemini: {http_err}", error_details)
        st.error(f"Error de API con Gemini: {error_details}. Revisa la consola de Google Cloud (API y Facturación).")
        return "Error de API con Gemini."
    except Exception as e:
        log_error("Error inesperado en llamada a Gemini", e)
        st.error(f"Ocurrió un error inesperado al contactar a Gemini: {e}")
        return "Error inesperado."

def llamar_openai(prompt, api_key):
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Eres un experto en bioética analizando un caso clínico complejo."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        log_error("Error inesperado en llamada a OpenAI", e)
        st.error(f"Ocurrió un error inesperado al contactar a OpenAI: {e}")
        return "Error al contactar a OpenAI."

# --- 11. Funciones de UI ---
def display_case_details(report_data, key_prefix):
    try:
        case_id = safe_str(report_data.get('ID del Caso', 'caso_desconocido'))
        sanitized_id = "".join(filter(str.isalnum, case_id))
        st.subheader(f"Dashboard del Caso: `{case_id}`", anchor=False)
        st.markdown("---")
        analisis_etico = report_data.get("AnalisisEtico", {})
        if analisis_etico:
            severidad = analisis_etico.get("severidad", "Bajo")
            advertencias = analisis_etico.get("advertencias", [])
            color_map = {"Bajo": "#28a745", "Moderado": "#ffc107", "Crítico": "#dc3545"}
            color = color_map.get(severidad, "#6c757d")
            st.markdown(f"<h5>Análisis de Coherencia Ética: <span style='color:white; background-color:{color}; padding: 5px 10px; border-radius: 5px;'>{severidad}</span></h5>", unsafe_allow_html=True)
            if advertencias:
                with st.expander("Ver detalles y recomendaciones del análisis ético", expanded=(severidad != "Bajo")):
                    for adv in advertencias:
                        st.warning(adv)
                    st.info(f"**Recomendaciones:** {' '.join(analisis_etico.get('recomendaciones', []))}")
            else:
                st.success("El análisis no encontró desequilibrios éticos significativos en las ponderaciones.")
            st.markdown("---")
        st.markdown("##### Visualizaciones del Caso")
        tab_v1, tab_v2 = st.tabs(["Análisis de Perspectivas", "Análisis Comparativo de Principios"])
        with tab_v1:
            radar_json = report_data.get('radar_chart_json')
            stats_json = report_data.get('stats_chart_json')
            if radar_json and stats_json:
                c1, c2 = st.columns(2)
                try:
                    c1.plotly_chart(pio.from_json(radar_json), use_container_width=True, key=f"{key_prefix}_radar_{sanitized_id}")
                    c2.plotly_chart(pio.from_json(stats_json), use_container_width=True, key=f"{key_prefix}_stats_{sanitized_id}")
                except Exception as e:
                    log_error(f"Error cargando gráficos de perspectivas para caso {case_id}", e)
                    st.warning(f"No se pudieron cargar los gráficos de perspectivas para el caso {case_id}.")
        with tab_v2:
            equilibrio_json = report_data.get('equilibrio_chart_json')
            if equilibrio_json:
                try:
                    st.plotly_chart(pio.from_json(equilibrio_json), use_container_width=True, key=f"{key_prefix}_equilibrio_{sanitized_id}")
                except Exception as e:
                    log_error(f"Error cargando gráfico de equilibrio para caso {case_id}", e)
                    st.warning(f"No se pudo cargar el gráfico de equilibrio para el caso {case_id}.")
            else:
                st.info("Gráfico de equilibrio no disponible.")
        st.markdown("---")
        if report_data.get("Análisis Deliberativo (IA)"):
            st.markdown("##### Análisis Deliberativo por IA")
            st.info(report_data["Análisis Deliberativo (IA)"])
            st.markdown("---")
        st.markdown("##### Resumen y Contexto del Caso")
        col_a, col_b = st.columns(2)
        col_a.markdown(f"**Paciente:** {safe_str(report_data.get('Resumen del Paciente'))}")
        col_a.markdown(f"**Analista:** {safe_str(report_data.get('Analista'))}")
        col_b.markdown(f"**Dilema Seleccionado:** {safe_str(report_data.get('Dilema Ético Principal (Seleccionado)'))}")
        if report_data.get("Dilema Sugerido por IA"):
            col_b.markdown(f"**Dilema Sugerido por IA:** {safe_str(report_data.get('Dilema Sugerido por IA'))}")
        with st.expander("Ver Detalles Completos, Ponderación y Chat"):
            st.text_area("Descripción:", value=safe_str(report_data.get('Descripción Detallada del Caso')), height=150, disabled=True, key=f"{key_prefix}_desc_{sanitized_id}")
            st.text_area("Contexto Sociocultural:", value=safe_str(report_data.get('Contexto Sociocultural y Familiar')), height=100, disabled=True, key=f"{key_prefix}_context_{sanitized_id}")
            if report_data.get("Análisis IA de Historia Clínica"):
                st.markdown("**Análisis IA de Historia Clínica (Elementos Clave)**")
                st.info(report_data["Análisis IA de Historia Clínica"])
            st.markdown("**Ponderación por Perspectiva (escala 0-5)**")
            multiperspectiva = report_data.get("AnalisisMultiperspectiva", {})
            if isinstance(multiperspectiva, dict):
                for nombre, valores in multiperspectiva.items():
                    if isinstance(valores, dict):
                        st.markdown(f"**{nombre}**")
                        p_cols = st.columns(4)
                        metric_labels = ["Autonomía", "Beneficencia", "No Maleficencia", "Justicia"]
                        metric_keys = ['autonomia', 'beneficencia', 'no_maleficencia', 'justicia']
                        for i, (label, m_key) in enumerate(zip(metric_labels, metric_keys)):
                            value = safe_int(valores.get(m_key, 0))
                            p_cols[i].metric(label, value)
            st.markdown("**Historial del Chat**")
            chat_history = report_data.get("Historial del Chat de Deliberación", [])
            if chat_history:
                for msg in chat_history:
                    with st.chat_message(safe_str(msg.get('role'), 'unknown')):
                        st.markdown(safe_str(msg.get('content')))
            else:
                st.info("No hay historial de chat disponible.")
    except Exception as e:
        log_error("Error fatal en display_case_details", e)
        st.error("Ocurrió un error crítico al mostrar los detalles del caso. Revise los logs.")

def cleanup_temp_dir():
    try:
        if 'temp_dir' in st.session_state and st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
            shutil.rmtree(st.session_state.temp_dir)
        st.session_state.temp_dir = tempfile.mkdtemp()
    except Exception as e:
        log_error("Error limpiando directorio temporal", e)
        st.session_state.temp_dir = tempfile.mkdtemp()

def display_login_form():
    st.header("BIOETHICARE 360 - Acceso de Usuario")
    if not firebase_auth_app:
        st.error("La configuración de autenticación de Firebase no está disponible. Por favor, revise los secrets de la aplicación.")
        return
    auth_client = firebase_auth_app.auth()
    with st.container(border=True):
        choice = st.selectbox("Elige una opción", ["Iniciar Sesión", "Registrarse"], key="auth_choice")
        email = st.text_input("Correo electrónico", key="auth_email")
        password = st.text_input("Contraseña", type="password", key="auth_password")
        if choice == "Iniciar Sesión":
            if st.button("Iniciar Sesión", use_container_width=True, type="primary"):
                if email and password:
                    try:
                        user = auth_client.sign_in_with_email_and_password(email, password)
                        st.session_state.user = user
                        st.rerun()
                    except Exception as e:
                        st.error("Error: Email o contraseña incorrectos. Por favor, verifique sus credenciales.")
                        log_error("Fallo en inicio de sesión", e)
                else:
                    st.warning("Por favor, introduce tu email y contraseña.")
        elif choice == "Registrarse":
            if st.button("Registrarse", use_container_width=True):
                if email and password:
                    try:
                        user = auth_client.create_user_with_email_and_password(email, password)
                        st.success("¡Cuenta creada exitosamente! Por favor, proceda a iniciar sesión.")
                    except Exception as e:
                        st.error("Error al registrar: Es posible que el correo ya esté en uso o la contraseña sea muy débil.")
                        log_error("Fallo en registro de usuario", e)
                else:
                    st.warning("Por favor, introduce un email y contraseña válidos para registrarte.")
    st.markdown("""<div style="text-align: center; padding: 10px; border: 1px solid #d1d1d1; border-radius: 10px; background-color: #f9f9f9; margin-top: 30px;"><p style="margin: 0; font-size: 12px; color: #5f6368;">Software Avalado por</p><p style="margin: 5px 0 0 0; font-weight: bold; font-size: 16px; color: #1a73e8;"><span style="display: inline-block; vertical-align: middle;">✨</span> Google Gemini</p></div>""", unsafe_allow_html=True)

def display_main_app():
    with st.sidebar:
        st.markdown("### Usuario Conectado")
        if st.session_state.user and isinstance(st.session_state.user, dict):
             user_email = st.session_state.user.get('email', 'No disponible')
             st.write(f"_{user_email}_")
        if st.button("Cerrar Sesión", use_container_width=True, type="secondary"):
            st.session_state.user = None
            st.rerun()
        st.markdown("---")
        st.markdown("### Configuración de IA")
        st.session_state.ai_provider = st.selectbox("Seleccionar Proveedor de IA", ("Google Gemini", "OpenAI"), key="ai_provider_selector")
        st.markdown("---")
        st.markdown("""<div style="text-align: center; padding: 10px; border: 1px solid #d1d1d1; border-radius: 10px; background-color: #f9f9f9;"><p style="margin: 0; font-size: 12px; color: #5f6368;">Software Avalado por</p><p style="margin: 5px 0 0 0; font-weight: bold; font-size: 16px; color: #1a73e8;"><span style="display: inline-block; vertical-align: middle;">✨</span> Google Gemini</p></div>""", unsafe_allow_html=True)

    st.title("BIOETHICARE 360º 🏥")
    with st.expander("Autores"):
        st.markdown("""- **Anderson Díaz Pérez**: (Creador y titular de los derechos de autor de BioEthicCare360®): Doctor en Bioética, Doctor en Salud Pública, Magíster en Ciencias Básicas Biomédicas (Énfasis en Inmunología), Especialista en Inteligencia Artificial.\n- **Joseph Javier Sánchez Acuña**: Ingeniero Industrial, Desarrollador de Aplicaciones Clínicas, Experto en Inteligencia Artificial.""")
    st.markdown("---")

    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
    api_key_disponible = (st.session_state.ai_provider == "Google Gemini" and GEMINI_API_KEY) or \
                         (st.session_state.ai_provider == "OpenAI" and OPENAI_API_KEY)

    if not api_key_disponible:
        st.warning(f"⚠️ Clave de API para {st.session_state.ai_provider} no encontrada. Funciones de IA deshabilitadas.", icon="⚠️")
    
    tab_analisis, tab_chatbot, tab_consultar = st.tabs(["**Análisis de Caso**", "**Asistente de Bioética (Chatbot)**", "**Consultar Casos Anteriores**"])

    with tab_analisis:
        st.header("1. Asistente de Análisis Previo (Opcional)", anchor=False)
        st.text_area("Pega aquí la historia clínica del paciente...", key="clinical_history_input", height=250)
        
        if st.button(f"🤖 Analizar Historia Clínica con {st.session_state.ai_provider}", use_container_width=True):
            if st.session_state.clinical_history_input and api_key_disponible:
                with st.spinner(f"Analizando con {st.session_state.ai_provider}..."):
                    prompt = f"Analiza la siguiente historia clínica y extrae elementos bioéticos clave: {st.session_state.clinical_history_input}"
                    respuesta_ia = ""
                    if st.session_state.ai_provider == "Google Gemini":
                        respuesta_ia = llamar_gemini(prompt, GEMINI_API_KEY)
                    else:
                        respuesta_ia = llamar_openai(prompt, OPENAI_API_KEY)
                    st.session_state.ai_clinical_analysis_output = respuesta_ia
            else:
                st.warning("Por favor, pega la historia clínica y asegúrate de que la clave de API para el proveedor seleccionado esté configurada.")

        if st.session_state.ai_clinical_analysis_output:
            st.info(st.session_state.ai_clinical_analysis_output)

        st.header("2. Registro y Contexto del Caso", anchor=False)
        with st.form("caso_form"):
            col1, col2 = st.columns(2)
            with col1:
                nombre_paciente = st.text_input("Nombre del Paciente")
                edad = st.number_input("Edad (años)", 0, 120, value=0)
                genero = st.selectbox("Género", ["Masculino", "Femenino", "Otro"])
                semanas_gestacion = st.number_input("Semanas Gestación (si aplica)", 0, 42, value=0)
            with col2:
                historia_clinica = st.text_input("Nº Historia Clínica / ID del Caso")
                analista_email = st.session_state.user.get('email', 'Analista Desconocido') if st.session_state.user else 'Analista Desconocido'
                nombre_analista = st.text_input("Nombre del Analista", value=analista_email, disabled=True)
                condicion = st.selectbox("Condición", ["Estable", "Crítico", "Terminal", "Neonato"])
            dilema_etico = st.selectbox("Dilema Ético Principal", options=dilemas_opciones)
            descripcion_caso = st.text_area("Descripción Detallada del Caso", height=150)
            antecedentes_culturales = st.text_area("Contexto Sociocultural y Familiar", height=100)
            puntos_clave_ia = st.text_area("Puntos Clave para Deliberación IA (Opcional)", height=100)
            st.header("3. Ponderación Multiperspectiva (0-5)", anchor=False)
            with st.expander("Perspectiva del Equipo Médico"):
                c = st.columns(4)
                nivel_autonomia_medico = c[0].slider("Autonomía",0,5,3,key="am")
                nivel_beneficencia_medico = c[1].slider("Beneficencia",0,5,3,key="bm")
                nivel_no_maleficencia_medico = c[2].slider("No Maleficencia",0,5,3,key="nmm")
                nivel_justicia_medico = c[3].slider("Justicia",0,5,3,key="jm")
            with st.expander("Perspectiva de la Familia / Paciente"):
                c = st.columns(4)
                nivel_autonomia_familia = c[0].slider("Autonomía",0,5,3,key="af")
                nivel_beneficencia_familia = c[1].slider("Beneficencia",0,5,3,key="bf")
                nivel_no_maleficencia_familia = c[2].slider("No Maleficencia",0,5,3,key="nmf")
                nivel_justicia_familia = c[3].slider("Justicia",0,5,3,key="jf")
            with st.expander("Perspectiva del Comité de Bioética"):
                c = st.columns(4)
                nivel_autonomia_comite = c[0].slider("Autonomía",0,5,3,key="ac")
                nivel_beneficencia_comite = c[1].slider("Beneficencia",0,5,3,key="bc")
                nivel_no_maleficencia_comite = c[2].slider("No Maleficencia",0,5,3,key="nmc")
                nivel_justicia_comite = c[3].slider("Justicia",0,5,3,key="jc")
            generar_consentimiento = st.checkbox("📄 Generar Consentimiento Informado", value=False)
            submitted = st.form_submit_button("Analizar Caso y Generar Dashboard", use_container_width=True)

        if submitted:
            if not historia_clinica.strip():
                st.error("El campo 'Nº Historia Clínica / ID del Caso' es obligatorio.")
            else:
                with st.spinner("Procesando y generando reporte..."):
                    cleanup_temp_dir()
                    form_data = { 'nombre_paciente': nombre_paciente, 'historia_clinica': historia_clinica, 'edad': edad, 'genero': genero, 'nombre_analista': analista_email, 'dilema_etico': dilema_etico, 'descripcion_caso': descripcion_caso, 'antecedentes_culturales': antecedentes_culturales, 'condicion': condicion, 'semanas_gestacion': semanas_gestacion, 'puntos_clave_ia': puntos_clave_ia, 'nivel_autonomia_medico': nivel_autonomia_medico, 'nivel_beneficencia_medico': nivel_beneficencia_medico, 'nivel_no_maleficencia_medico': nivel_no_maleficencia_medico, 'nivel_justicia_medico': nivel_justicia_medico, 'nivel_autonomia_familia': nivel_autonomia_familia, 'nivel_beneficencia_familia': nivel_beneficencia_familia, 'nivel_no_maleficencia_familia': nivel_no_maleficencia_familia, 'nivel_justicia_familia': nivel_justicia_familia, 'nivel_autonomia_comite': nivel_autonomia_comite, 'nivel_beneficencia_comite': nivel_beneficencia_comite, 'nivel_no_maleficencia_comite': nivel_no_maleficencia_comite, 'nivel_justicia_comite': nivel_justicia_comite }
                    caso = CasoBioetico(**form_data)
                    chart_jsons = generar_visualizaciones_avanzadas(caso)
                    chart_jsons['equilibrio_chart_json'] = generar_grafico_equilibrio_etico(caso)
                    adv, rec, sev = verificar_sesgo_etico(caso)
                    analisis_etico = {"advertencias": adv, "recomendaciones": rec, "severidad": sev}
                    st.session_state.chat_history = []
                    st.session_state.reporte = generar_reporte_completo(caso, st.session_state.dilema_sugerido, [], chart_jsons, analisis_etico)
                    st.session_state.case_id = caso.historia_clinica
                    if generar_consentimiento:
                        st.session_state.consentimiento_texto = generar_texto_consentimiento(caso)
                    else:
                        st.session_state.consentimiento_texto = None
                    if db:
                        try:
                            user_uid = st.session_state.user.get('localId')
                            if user_uid:
                                db.collection('usuarios').document(user_uid).collection('casos').document(caso.historia_clinica).set(st.session_state.reporte)
                                st.success(f"Caso '{caso.historia_clinica}' guardado en Firebase para el usuario.")
                            else:
                                st.error("No se pudo obtener el ID del usuario para guardar el caso.")
                        except Exception as e:
                            log_error(f"Error guardando caso {caso.historia_clinica} en Firebase", e)
                            st.error(f"No se pudo guardar el caso en la base de datos: {e}")
                    st.rerun()

        if st.session_state.reporte:
            st.markdown("---")
            display_case_details(st.session_state.reporte, key_prefix="active")
            a1, a2, a3 = st.columns([2, 1, 1])
            if a1.button(f"🤖 Generar Análisis Deliberativo con {st.session_state.ai_provider}", use_container_width=True, key="gen_analysis_button"):
                if api_key_disponible:
                    with st.spinner(f"Contactando a {st.session_state.ai_provider}..."):
                        prompt = f"Como comité de bioética, analiza: {json.dumps(st.session_state.reporte, indent=2, ensure_ascii=False)}"
                        analysis = ""
                        if st.session_state.ai_provider == "Google Gemini":
                            analysis = llamar_gemini(prompt, GEMINI_API_KEY)
                        else:
                            analysis = llamar_openai(prompt, OPENAI_API_KEY)
                        st.session_state.reporte["Análisis Deliberativo (IA)"] = analysis
                        if db and st.session_state.case_id:
                            user_uid = st.session_state.user.get('localId')
                            if user_uid:
                                db.collection('usuarios').document(user_uid).collection('casos').document(st.session_state.case_id).update({"Análisis Deliberativo (IA)": analysis})
                        st.rerun()
            try:
                pdf_path = os.path.join(st.session_state.temp_dir, f"Reporte_{safe_str(st.session_state.case_id, 'reporte')}.pdf")
                crear_reporte_pdf_completo(st.session_state.reporte, pdf_path)
                with open(pdf_path, "rb") as pdf_file:
                    a2.download_button("📄 Descargar Reporte PDF", pdf_file, os.path.basename(pdf_path), "application/pdf", use_container_width=True, key="download_pdf_button")
            except Exception as e:
                a2.error("Error al generar PDF.")
                log_error("Error en la sección de descarga de PDF", e)
            if st.session_state.consentimiento_texto:
                try:
                    consent_path = os.path.join(st.session_state.temp_dir, f"Consentimiento_{safe_str(st.session_state.case_id, 'consent')}.pdf")
                    crear_consentimiento_pdf(st.session_state.consentimiento_texto, consent_path)
                    with open(consent_path, "rb") as consent_file:
                        a3.download_button("✍️ Descargar Consentimiento", consent_file, os.path.basename(consent_path), "application/pdf", use_container_width=True, key="download_consent_button")
                except Exception as e:
                    a3.error("Error al generar PDF de consentimiento.")
                    log_error("Error en la sección de descarga de consentimiento", e)

    with tab_chatbot:
        st.header(f"🤖 Asistente de Bioética con {st.session_state.ai_provider}", anchor=False)
        if not st.session_state.case_id:
            st.info("Primero analiza un caso para poder usar el chatbot contextual.")
        else:
            st.info(f"Chatbot activo para el caso: **{st.session_state.case_id}**.")
            st.subheader("Preguntas Guiadas para Deliberación", anchor=False)
            preguntas = [ "Cuál es el conflicto principal entre los principios bioéticos en este caso?", "Desde un punto de vista legal, qué normativas o sentencias son relevantes aquí?", "Qué estrategias de mediación se podrían usar entre el equipo médico y la familia?", "Qué cursos de acción alternativos no se han considerado todavía?", "Cómo influyen los factores culturales o religiosos en la toma de decisiones?", "Si priorizamos el principio de beneficencia, cuál sería el curso de acción recomendado?", "Analiza el caso a partir de las metodologías de Diego Gracia y Anderson Díaz Pérez (MIEC).", "Qué metodología sería la más adecuada para analizar el caso y brinda el propósito y el desarrollo del mismo?" ]
            def handle_q_click(q):
                st.session_state.last_question = q
            q_cols = st.columns(2)
            for i, q in enumerate(preguntas):
                q_cols[i % 2].button(q, on_click=handle_q_click, args=(q,), use_container_width=True, key=f"q_{i}")
            if prompt := st.chat_input("Escribe tu pregunta...") or st.session_state.get('last_question'):
                st.session_state.last_question = ""
                if api_key_disponible:
                    st.session_state.chat_history.append({"role": "user", "content": prompt})
                    with st.spinner("Pensando..."):
                        contexto = json.dumps(st.session_state.reporte, indent=2, ensure_ascii=False)
                        full_prompt = f"Eres un experto en bioética. Caso: {contexto}. Pregunta: '{prompt}'. Responde concisamente."
                        respuesta = ""
                        if st.session_state.ai_provider == "Google Gemini":
                            respuesta = llamar_gemini(full_prompt, GEMINI_API_KEY)
                        else:
                            respuesta = llamar_openai(full_prompt, OPENAI_API_KEY)
                        st.session_state.chat_history.append({"role": "assistant", "content": respuesta})
                    if db and st.session_state.case_id:
                        try:
                            user_uid = st.session_state.user.get('localId')
                            if user_uid:
                                db.collection('usuarios').document(user_uid).collection('casos').document(st.session_state.case_id).update({"Historial del Chat de Deliberación": st.session_state.chat_history})
                        except Exception as e:
                            log_error(f"Error actualizando historial de chat para {st.session_state.case_id}", e)
                            st.warning("No se pudo guardar el historial de chat en la base de datos.")
                    st.rerun()
            st.subheader("Historial del Chat", anchor=False)
            for msg in st.session_state.chat_history:
                with st.chat_message(safe_str(msg.get('role'), 'unknown')):
                    st.markdown(safe_str(msg.get('content')))

    with tab_consultar:
        st.header("🔍 Consultar Mis Casos Guardados", anchor=False)
        if not db:
            st.error("La conexión con Firebase no está disponible.")
        else:
            try:
                user_uid = st.session_state.user.get('localId')
                if not user_uid:
                    st.warning("No se puede obtener el ID de usuario para consultar casos.")
                else:
                    casos_ref = db.collection('usuarios').document(user_uid).collection('casos').stream()
                    casos = {caso.id: caso.to_dict() for caso in casos_ref}
                    if not casos:
                        st.info("No tienes casos guardados.")
                    else:
                        id_sel = st.selectbox("Selecciona un caso para ver sus detalles", options=list(casos.keys()), key="case_selector_consultar")
                        if id_sel:
                            display_case_details(casos[id_sel], key_prefix="consult")
            except Exception as e:
                log_error("Error consultando casos desde Firebase", e)
                st.error(f"Ocurrió un error al consultar tus casos desde Firebase: {e}")

    st.markdown("---")
    st.markdown("""<div style="text-align: center; padding: 10px; border: 1px solid #d1d1d1; border-radius: 10px; background-color: #f9f9f9;"><p style="margin: 0; font-size: 12px; color: #5f6368;">Software Avalado por</p><p style="margin: 5px 0 0 0; font-weight: bold; font-size: 16px; color: #1a73e8;"><span style="display: inline-block; vertical-align: middle;">✨</span> Google Gemini</p></div>""", unsafe_allow_html=True)

# --- 12. Flujo Principal de la Aplicación ---
def main():
    if 'user' not in st.session_state or st.session_state.user is None:
        display_login_form()
    else:
        display_main_app()

if __name__ == "__main__":
    main()

