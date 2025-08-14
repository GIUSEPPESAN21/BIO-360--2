# ==============================================================================
# BIOETHICARE 360 2.0 - Versión Mejorada
# Autores: Anderson Díaz Pérez & Joseph Javier Sánchez Acuña
# Descripción: Aplicación web para el análisis deliberativo de dilemas
#              bioéticos, con autenticación de usuarios, generación de reportes,
#              consentimiento informado y asistencia por IA.
# ==============================================================================

# --- 1. Importaciones de Librerías ---
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

# Importaciones para generación de PDF con ReportLab
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, HRFlowable
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.units import inch
from reportlab.lib import colors

# Importaciones para Firebase (Autenticación y Base de Datos)
import firebase_admin
from firebase_admin import credentials, firestore
import pyrebase

# --- 2. Configuración Global y Constantes ---

# Configuración del logging para monitoreo y depuración
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# URL del logo de la aplicación
LOGO_URL = "https://i.imgur.com/iZ6w2fJ.png"

# Configuración de la página de Streamlit (debe ser el primer comando de st)
st.set_page_config(layout="wide", page_title="BIOETHICARE 360", page_icon=LOGO_URL)

# --- 3. Gestión del Estado de la Sesión ---

# Define los valores por defecto para el estado de la sesión
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
    'consentimiento_texto': None
}

# Inicializa el estado de la sesión si no existe
for key, default_value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# --- 4. Funciones Utilitarias ---

def safe_int(value, default=0):
    """Convierte un valor a entero de forma segura."""
    if value is None or value == '': return default
    try: return int(value)
    except (ValueError, TypeError): return default

def safe_str(value, default=""):
    """Convierte un valor a string de forma segura."""
    if value is None: return default
    return str(value).strip()

def log_error(error_msg, exception=None):
    """Registra un mensaje de error en el log."""
    logger.error(f"BIOETHICARE ERROR: {error_msg}")
    if exception: logger.error(f"Exception details: {str(exception)}")

# --- 5. Conexión a Servicios Externos (Firebase y Gemini) ---

@st.cache_resource
def initialize_firebase_admin():
    """Inicializa el SDK de Admin de Firebase para interactuar con Firestore."""
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
    """Inicializa Pyrebase para la autenticación de usuarios."""
    try:
        if "firebase_client_config" in st.secrets:
            firebase_client_config = dict(st.secrets["firebase_client_config"])
            return pyrebase.initialize_app(firebase_client_config)
        else:
            log_error("Configuración de cliente de Firebase no encontrada en st.secrets.")
            return None
    except Exception as e:
        log_error("Error crítico al inicializar Pyrebase para autenticación", e)
        return None

def llamar_gemini(prompt, api_key):
    """Realiza una llamada a la API de Google Gemini."""
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        result = response.json()
        if 'candidates' in result and result['candidates']:
            return result['candidates'][0]['content']['parts'][0]['text']
        log_error(f"Respuesta inesperada de la API de Gemini: {result}")
        st.warning("Respuesta inesperada de la API.")
        return "No se pudo obtener una respuesta válida."
    except requests.exceptions.RequestException as e:
        log_error("Error de conexión con Gemini", e)
        st.error(f"Error de conexión con la API de Gemini: {e}")
        return "Error de conexión."
    except Exception as e:
        log_error("Error inesperado en llamada a Gemini", e)
        st.error(f"Ocurrió un error inesperado al contactar a la IA: {e}")
        return "Error inesperado."

# Inicialización de servicios
db = initialize_firebase_admin()
firebase_auth_app = initialize_firebase_auth()

# --- 6. Base de Conocimiento y Modelo de Datos ---

@st.cache_data
def cargar_dilemas():
    """Carga la información estructurada de los dilemas desde un archivo JSON."""
    try:
        with open('dilemas.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        log_error("El archivo dilemas.json no fue encontrado.")
        st.error("Error: No se pudo cargar la base de conocimiento de dilemas.")
        return {}

dilemas_data = cargar_dilemas()
dilemas_opciones = list(dilemas_data.keys())

class CasoBioetico:
    """Clase que representa la estructura de datos de un caso bioético."""
    def __init__(self, **kwargs):
        self.nombre_paciente = safe_str(kwargs.get('nombre_paciente'), 'N/A')
        self.historia_clinica = safe_str(kwargs.get('historia_clinica'), f"caso_{int(datetime.now().timestamp())}")
        self.edad = safe_int(kwargs.get('edad'))
        self.genero = safe_str(kwargs.get('genero'), 'N/A')
        self.nombre_analista = safe_str(kwargs.get('nombre_analista'), 'N/A')
        self.dilema_etico = safe_str(kwargs.get('dilema_etico', dilemas_opciones[0]))
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

# --- 7. Lógica de Análisis y Generación de Gráficos ---

def verificar_sesgo_etico(caso):
    """Analiza las ponderaciones para detectar posibles sesgos o desequilibrios."""
    # (El código de esta función permanece sin cambios, es robusto)
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

def generar_visualizaciones_avanzadas(caso):
    """Genera los gráficos de radar y de barras para el dashboard."""
    try:
        perspectivas_data = caso.perspectivas
        labels = ["Autonomía", "Beneficencia", "No Maleficencia", "Justicia"]
        
        # Gráfico de Radar
        fig_radar = go.Figure()
        colors_map = {'medico': 'rgba(239, 68, 68, 0.7)', 'familia': 'rgba(59, 130, 246, 0.7)', 'comite': 'rgba(34, 197, 94, 0.7)'}
        nombres = {'medico': 'Equipo Médico', 'familia': 'Familia/Paciente', 'comite': 'Comité de Bioética'}
        for key, data in perspectivas_data.items():
            fig_radar.add_trace(go.Scatterpolar(r=list(data.values()), theta=labels, fill='toself', name=nombres[key], line_color=colors_map[key]))
        fig_radar.update_layout(title_text="<b>Ponderación por Perspectiva</b>", polar=dict(radialaxis=dict(visible=True, range=[0, 5])), showlegend=True, font_size=14)
        
        # Gráfico de Barras (Consenso y Disenso)
        scores = np.array([list(d.values()) for d in perspectivas_data.values()])
        fig_stats = go.Figure()
        fig_stats.add_trace(go.Bar(x=labels, y=np.mean(scores, axis=0), error_y=dict(type='data', array=np.std(scores, axis=0), visible=True), marker_color='#636EFA'))
        fig_stats.update_layout(title_text="<b>Análisis de Consenso y Disenso</b>", yaxis=dict(range=[0, 6]), font_size=14)
        
        return {'radar_comparativo_json': fig_radar.to_json(), 'stats_chart_json': fig_stats.to_json()}
    except Exception as e:
        log_error("Error generando visualizaciones", e)
        return {'radar_comparativo_json': None, 'stats_chart_json': None}

def generar_grafico_equilibrio_etico(caso):
    """Genera el gráfico de barras comparativo de principios."""
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

# --- 8. Generación de Documentos (Reporte y Consentimiento) ---

def generar_reporte_completo(caso, dilema_sugerido, chat_history, chart_jsons, ethical_analysis):
    """Compila todos los datos del caso en un diccionario para el reporte."""
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

def crear_reporte_pdf_completo(data, filename):
    """Crea el PDF del reporte deliberativo."""
    # (El código de esta función permanece sin cambios, es robusto)
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
    """Genera el texto del consentimiento informado basado en el caso."""
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
    """Crea un archivo PDF a partir del texto del consentimiento."""
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

# --- 9. Componentes de la Interfaz de Usuario (UI) ---

def display_login_form():
    """Muestra el formulario de inicio de sesión y registro."""
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.image(LOGO_URL)
        st.header("Acceso de Usuario")
    
        if not firebase_auth_app:
            st.error("La configuración de autenticación no está disponible. Contacte al administrador.")
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
                            st.error("Error: Email o contraseña incorrectos.")
                            log_error("Fallo en inicio de sesión", e)
                    else:
                        st.warning("Por favor, introduce tu email y contraseña.")

            elif choice == "Registrarse":
                if st.button("Registrarse", use_container_width=True):
                    if email and password:
                        try:
                            user = auth_client.create_user_with_email_and_password(email, password)
                            st.success("¡Cuenta creada exitosamente! Por favor, inicie sesión.")
                        except Exception as e:
                            st.error("Error al registrar: El correo ya puede estar en uso o la contraseña es muy débil.")
                            log_error("Fallo en registro de usuario", e)
                    else:
                        st.warning("Por favor, introduce un email y contraseña válidos.")

def display_main_app():
    """Muestra la aplicación principal una vez que el usuario está autenticado."""
    
    # --- Barra Lateral ---
    with st.sidebar:
        st.image(LOGO_URL, width=150)
        st.markdown("### Usuario Conectado")
        if st.session_state.user and isinstance(st.session_state.user, dict):
             user_email = st.session_state.user.get('email', 'No disponible')
             st.write(f"_{user_email}_")
        
        if st.button("Cerrar Sesión", use_container_width=True, type="secondary"):
            st.session_state.user = None
            st.rerun()
        st.markdown("---")

    # --- Encabezado Principal ---
    c1, c2 = st.columns([1, 5])
    with c1:
        st.image(LOGO_URL, width=100)
    with c2:
        st.title("BioEthicCare 360º")

    with st.expander("Autores y Propósito de la Herramienta"):
        st.markdown("""
        - **Anderson Díaz Pérez**: (Creador y titular de los derechos de autor de BioEthicCare360®): Doctor en Bioética, Doctor en Salud Pública, Magíster en Ciencias Básicas Biomédicas (Énfasis en Inmunología), Especialista en Inteligencia Artificial.
        - **Joseph Javier Sánchez Acuña**: Ingeniero Industrial, Desarrollador de Aplicaciones Clínicas, Experto en Inteligencia Artificial.
        
        **Propósito:** Facilitar la toma de decisiones informadas y humanizadas en dilemas bioéticos, previniendo sanciones y garantizando el respeto a la dignidad del paciente.
        """)
    st.markdown("---")

    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        st.warning("⚠️ Clave de API de Gemini no encontrada. Funciones de IA deshabilitadas.", icon="⚠️")
    
    # --- Pestañas de Navegación ---
    tab_analisis, tab_chatbot, tab_consultar = st.tabs(["**Análisis de Caso**", "**Asistente de Bioética (Chatbot)**", "**Consultar Casos Anteriores**"])

    # --- Pestaña: Análisis de Caso ---
    with tab_analisis:
        display_tab_analisis(GEMINI_API_KEY)

    # --- Pestaña: Asistente de Bioética ---
    with tab_chatbot:
        display_tab_chatbot(GEMINI_API_KEY)

    # --- Pestaña: Consultar Casos ---
    with tab_consultar:
        display_tab_consultar()

def display_tab_analisis(api_key):
    """Renderiza el contenido de la pestaña de análisis de caso."""
    st.header("1. Asistente de Análisis Previo (Opcional)", anchor=False)
    st.text_area("Pega aquí la historia clínica del paciente para un análisis preliminar por IA...", key="clinical_history_input", height=250)
    
    if st.button("🤖 Analizar Historia Clínica con IA", use_container_width=True):
        if st.session_state.clinical_history_input and api_key:
            with st.spinner("Analizando historia clínica con Gemini..."):
                prompt = f"Analiza la siguiente historia clínica y extrae elementos bioéticos clave: {st.session_state.clinical_history_input}"
                st.session_state.ai_clinical_analysis_output = llamar_gemini(prompt, api_key)
        else:
            st.warning("Por favor, pega la historia clínica y asegúrate de que la clave de API de Gemini está configurada.")

    if st.session_state.ai_clinical_analysis_output:
        st.info(st.session_state.ai_clinical_analysis_output)

    st.header("2. Registro y Contexto del Caso", anchor=False)
    with st.form("caso_form"):
        # --- Columnas para datos del paciente ---
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

        # --- Campos principales del caso ---
        dilema_etico = st.selectbox("Dilema Ético Principal", options=dilemas_opciones)
        descripcion_caso = st.text_area("Descripción Detallada del Caso", height=150)
        antecedentes_culturales = st.text_area("Contexto Sociocultural y Familiar", height=100)
        puntos_clave_ia = st.text_area("Puntos Clave para Deliberación IA (Opcional)", height=100)
        
        # --- Ponderación Multiperspectiva ---
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
        submitted = st.form_submit_button("Analizar Caso y Generar Dashboard", use_container_width=True, type="primary")

    if submitted:
        handle_form_submission(locals())

    if st.session_state.reporte:
        display_dashboard_and_actions(api_key)

def handle_form_submission(form_locals):
    """Procesa los datos del formulario cuando es enviado."""
    if not form_locals['historia_clinica'].strip():
        st.error("El campo 'Nº Historia Clínica / ID del Caso' es obligatorio.")
        return

    with st.spinner("Procesando y generando reporte..."):
        cleanup_temp_dir()
        
        # Recolecta los datos del formulario en un diccionario
        form_data = {key: value for key, value in form_locals.items() if key not in ['submitted', 'api_key']}
        caso = CasoBioetico(**form_data)
        
        # Genera análisis y gráficos
        chart_jsons = generar_visualizaciones_avanzadas(caso)
        chart_jsons['equilibrio_chart_json'] = generar_grafico_equilibrio_etico(caso)
        adv, rec, sev = verificar_sesgo_etico(caso)
        analisis_etico = {"advertencias": adv, "recomendaciones": rec, "severidad": sev}

        # Actualiza el estado de la sesión
        st.session_state.chat_history = []
        st.session_state.reporte = generar_reporte_completo(caso, st.session_state.dilema_sugerido, [], chart_jsons, analisis_etico)
        st.session_state.case_id = caso.historia_clinica
        
        if form_locals['generar_consentimiento']:
            st.session_state.consentimiento_texto = generar_texto_consentimiento(caso)
        else:
            st.session_state.consentimiento_texto = None

        # Guarda en Firebase si está disponible
        if db:
            try:
                user_uid = st.session_state.user.get('localId')
                if user_uid:
                    db.collection('usuarios').document(user_uid).collection('casos').document(caso.historia_clinica).set(st.session_state.reporte)
                    st.success(f"Caso '{caso.historia_clinica}' guardado en Firebase.")
                else:
                    st.error("No se pudo obtener el ID del usuario para guardar el caso.")
            except Exception as e:
                log_error(f"Error guardando caso {caso.historia_clinica} en Firebase", e)
                st.error(f"No se pudo guardar el caso en la base de datos: {e}")
        
        st.rerun()

def display_dashboard_and_actions(api_key):
    """Muestra el dashboard del caso y los botones de acción."""
    st.markdown("---")
    display_case_details(st.session_state.reporte, key_prefix="active")
    
    a1, a2, a3 = st.columns([2, 1, 1])
    
    # Botón para análisis con Gemini
    if a1.button("🤖 Generar/Regenerar Análisis Deliberativo con Gemini", use_container_width=True, key="gen_analysis_button"):
        if api_key:
            with st.spinner("Contactando a Gemini..."):
                prompt = f"Como comité de bioética, analiza: {json.dumps(st.session_state.reporte, indent=2, ensure_ascii=False)}"
                analysis = llamar_gemini(prompt, api_key)
                st.session_state.reporte["Análisis Deliberativo (IA)"] = analysis
                if db and st.session_state.case_id:
                    user_uid = st.session_state.user.get('localId')
                    if user_uid:
                        db.collection('usuarios').document(user_uid).collection('casos').document(st.session_state.case_id).update({"Análisis Deliberativo (IA)": analysis})
                st.rerun()

    # Botón para descargar reporte
    try:
        pdf_path = os.path.join(st.session_state.temp_dir, f"Reporte_{safe_str(st.session_state.case_id, 'reporte')}.pdf")
        crear_reporte_pdf_completo(st.session_state.reporte, pdf_path)
        with open(pdf_path, "rb") as pdf_file:
            a2.download_button("📄 Descargar Reporte PDF", pdf_file, os.path.basename(pdf_path), "application/pdf", use_container_width=True, key="download_pdf_button")
    except Exception as e:
        a2.error("Error al generar PDF.")
        log_error("Error en la sección de descarga de PDF", e)

    # Botón para descargar consentimiento
    if st.session_state.consentimiento_texto:
        try:
            consent_path = os.path.join(st.session_state.temp_dir, f"Consentimiento_{safe_str(st.session_state.case_id, 'consent')}.pdf")
            crear_consentimiento_pdf(st.session_state.consentimiento_texto, consent_path)
            with open(consent_path, "rb") as consent_file:
                a3.download_button("✍️ Descargar Consentimiento", consent_file, os.path.basename(consent_path), "application/pdf", use_container_width=True, key="download_consent_button")
        except Exception as e:
            a3.error("Error al generar PDF de consentimiento.")
            log_error("Error en la sección de descarga de consentimiento", e)

def display_tab_chatbot(api_key):
    """Renderiza el contenido de la pestaña del chatbot."""
    st.header("🤖 Asistente de Bioética con Gemini", anchor=False)
    if not st.session_state.case_id:
        st.info("Primero analiza un caso en la pestaña 'Análisis de Caso' para poder usar el chatbot contextual.")
        return

    st.info(f"Chatbot activo para el caso: **{st.session_state.case_id}**.")
    st.subheader("Preguntas Guiadas para Deliberación", anchor=False)
    preguntas = [
        "¿Cuál es el conflicto principal entre los principios bioéticos en este caso?",
        "Desde un punto de vista legal, ¿qué normativas o sentencias son relevantes aquí?",
        "¿Qué estrategias de mediación se podrían usar entre el equipo médico y la familia?",
        "¿Qué cursos de acción alternativos no se han considerado todavía?",
        "¿Cómo influyen los factores culturales o religiosos en la toma de decisiones?",
        "Si priorizamos el principio de beneficencia, ¿cuál sería el curso de acción recomendado?",
        "Analiza el caso a partir de las metodologías de Diego Gracia y Anderson Díaz Pérez (MIEC).",
        "¿Qué metodología sería la más adecuada para analizar el caso y brinda el propósito y el desarrollo del mismo?"
    ]
    
    def handle_q_click(q):
        st.session_state.last_question = q
    
    q_cols = st.columns(2)
    for i, q in enumerate(preguntas):
        q_cols[i % 2].button(q, on_click=handle_q_click, args=(q,), use_container_width=True, key=f"q_{i}")
    
    if prompt := st.chat_input("Escribe tu pregunta...") or st.session_state.get('last_question'):
        st.session_state.last_question = ""
        if api_key:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.spinner("Pensando..."):
                contexto = json.dumps(st.session_state.reporte, indent=2, ensure_ascii=False)
                full_prompt = f"Eres un experto en bioética. Caso: {contexto}. Pregunta: '{prompt}'. Responde concisamente."
                respuesta = llamar_gemini(full_prompt, api_key)
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

def display_tab_consultar():
    """Renderiza el contenido de la pestaña de consulta de casos."""
    st.header("🔍 Consultar Mis Casos Guardados", anchor=False)
    if not db:
        st.error("La conexión con la base de datos no está disponible.")
        return

    try:
        user_uid = st.session_state.user.get('localId')
        if not user_uid:
            st.warning("No se puede obtener el ID de usuario para consultar casos.")
            return
            
        casos_ref = db.collection('usuarios').document(user_uid).collection('casos').stream()
        casos = {caso.id: caso.to_dict() for caso in casos_ref}
        
        if not casos:
            st.info("Aún no tienes casos guardados.")
        else:
            id_sel = st.selectbox("Selecciona un caso para ver sus detalles", options=list(casos.keys()), key="case_selector_consultar")
            if id_sel:
                display_case_details(casos[id_sel], key_prefix="consult")
    except Exception as e:
        log_error("Error consultando casos desde Firebase", e)
        st.error(f"Ocurrió un error al consultar tus casos desde Firebase: {e}")

# --- 10. Flujo Principal de la Aplicación ---

def main():
    """Función principal que dirige al login o a la aplicación principal."""
    if 'user' not in st.session_state or st.session_state.user is None:
        display_login_form()
    else:
        display_main_app()

if __name__ == "__main__":
    main()
