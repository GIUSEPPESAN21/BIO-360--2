"""
ui/tab_configuracion.py
=======================
Pestaña "Perfil y Configuración": usuario conectado, cierre de sesión, selección del
proveedor de IA y estado de cumplimiento.

Diferencias respecto al monolito original
-----------------------------------------
1. **El selector se alimenta de la fábrica de proveedores**, no de una tupla escrita a
   mano. Así, `Kiro` (y cualquier proveedor futuro) aparece automáticamente en cuanto
   se registra en `services.ai.PROVEEDORES_DISPONIBLES`, sin tocar la UI.
2. **El estado del modelo no miente.** El original inicializaba `selected_model` con
   `"gemini-2.0-flash-exp"` y lo mostraba como "modelo activo" aunque nunca se hubiera
   invocado. Aquí, si `selected_model` es `None`, se dice explícitamente que todavía no
   se ha llamado a ningún modelo.
3. **Cerrar sesión también limpia el caso activo**, para que los datos clínicos del
   paciente anterior no queden en la memoria de sesión tras el logout.
4. Se retiró la imagen remota del logo de Gemini (`storage.googleapis.com`): era una
   dependencia de red externa innecesaria y resultaba incorrecta cuando el proveedor
   seleccionado no era Gemini.

Nunca se muestra ni se registra el VALOR de una clave de API: solo si está presente.
"""

from __future__ import annotations

import logging

import streamlit as st

from ui.estado import email_usuario, reset_caso_activo

logger = logging.getLogger(__name__)

#: Secret que hay que definir para cada proveedor, para poder nombrarlo en el aviso.
SECRET_POR_PROVEEDOR: dict[str, str] = {
    "Google Gemini": "GEMINI_API_KEY",
    "OpenAI": "OPENAI_API_KEY",
    "Kiro": "KIRO_API_KEY",
}


def render(proveedor_ok: bool, proveedores: tuple) -> None:
    """
    Renderiza la pestaña de perfil y configuración.

    Parameters
    ----------
    proveedor_ok:
        True si hay una clave de API disponible para el proveedor actualmente
        seleccionado. Se calcula en el entrypoint, que es quien lee los secrets.
    proveedores:
        Nombres de proveedores disponibles (`services.ai.PROVEEDORES_DISPONIBLES`).
    """
    st.header("👤 Perfil y Configuración del Sistema", anchor=False)

    _seccion_usuario()
    st.divider()
    _seccion_proveedor(proveedor_ok, proveedores)
    st.divider()
    _seccion_motor_ia()
    st.divider()
    _seccion_cumplimiento()


# --- Usuario ----------------------------------------------------------------------

def _seccion_usuario() -> None:
    st.markdown("### Usuario Conectado")
    st.success(f"Sesión iniciada como: **{email_usuario()}**")

    if st.button("Cerrar Sesión", use_container_width=True, type="secondary"):
        # Privacidad: al cerrar sesión se descarta también el caso activo. De lo
        # contrario el reporte clínico del paciente anterior (nombre, historia,
        # chat de deliberación) seguiría en st.session_state y sería visible para
        # quien iniciara sesión a continuación en el mismo navegador.
        reset_caso_activo()
        st.session_state.user = None
        st.rerun()


# --- Proveedor de IA --------------------------------------------------------------

def _seccion_proveedor(proveedor_ok: bool, proveedores: tuple) -> None:
    st.markdown("### ⚙️ Configuración de IA")

    opciones = list(proveedores) or ["Google Gemini"]
    actual = st.session_state.get("ai_provider") or opciones[0]
    indice = opciones.index(actual) if actual in opciones else 0

    # OJO: la key del widget NO puede ser "ai_provider", porque colisionaría con la
    # clave de session_state que escribimos justo después.
    seleccion = st.selectbox(
        "Seleccionar Proveedor de IA",
        options=opciones,
        index=indice,
        key="ai_provider_selector",
    )
    st.session_state.ai_provider = seleccion

    if proveedor_ok:
        st.info(f"Proveedor '{seleccion}' seleccionado y con credenciales disponibles.")
    else:
        secret = SECRET_POR_PROVEEDOR.get(seleccion, "la clave de API correspondiente")
        st.warning(
            f"⚠️ No se encontró `{secret}` para el proveedor '{seleccion}'. "
            "Las funciones de IA están deshabilitadas.\n\n"
            "Defínela en `.streamlit/secrets.toml`, tomando como guía el archivo "
            "`secrets_template.toml` de la raíz del proyecto.",
            icon="⚠️",
        )

    if seleccion == "Kiro":
        st.caption(
            "ℹ️ Aviso de integración: el contrato HTTP que asume "
            "`services/ai/kiro_provider.py` (`POST {base_url}/v1/analyze`) **no ha sido "
            "verificado** contra la API real de Kiro. Si tu endpoint difiere, ajusta "
            "`RUTA_ANALYZE` y `_extraer_payload` en ese archivo; el resto de la "
            "aplicación no necesita cambios. `KIRO_BASE_URL` es opcional."
        )


# --- Estado del motor -------------------------------------------------------------

def _seccion_motor_ia() -> None:
    st.markdown("### 🤖 Estado del Motor de IA")

    modelo = st.session_state.get("selected_model")
    if modelo:
        st.info(f"**Modelo usado en la última llamada:**\n\n`{modelo}`")
    else:
        # No se presenta un modelo por defecto como si estuviera activo.
        st.info(
            "Todavía no se ha invocado ningún modelo en esta sesión. "
            "El modelo efectivo se mostrará aquí tras el primer análisis."
        )

    traza = st.session_state.get("ultima_trazabilidad") or {}
    if traza:
        partes = [
            f"Proveedor: {traza.get('proveedor', '—')}",
            f"Versión de prompt: `{traza.get('prompt_version', '—')}`",
            f"UTC: {traza.get('timestamp', '—')}",
        ]
        st.caption("Última trazabilidad registrada — " + " · ".join(partes))


# --- Cumplimiento -----------------------------------------------------------------

def _seccion_cumplimiento() -> None:
    st.markdown("### 🔒 Estado de Cumplimiento")
    st.markdown(
        "- **Anonimización de datos clínicos:** las historias clínicas y los reportes "
        "se anonimizan (nombres, fechas, lugares, documentos y contactos se sustituyen "
        "por marcadores) **antes** de enviarse a cualquier API de IA externa. Los "
        "valores reales se rehidratan solo localmente, al mostrarte el resultado.\n"
        "- **Registro de auditoría inmutable:** cada llamada a un modelo queda "
        "registrada con usuario, modelo exacto, versión de prompt y hash del prompt, "
        "encadenada con SHA-256. Las reglas de Firestore permiten crear entradas pero "
        "prohíben modificarlas o borrarlas.\n"
        "- **Reglas de base de datos:** el aislamiento por usuario se define en "
        "`firestore.rules` y **debe estar desplegado** para tener efecto. La "
        "autenticación del frontend por sí sola no es suficiente."
    )
    st.code("firebase deploy --only firestore:rules", language="bash")
    st.caption(
        "Ejecuta ese comando desde la raíz del proyecto para aplicar las reglas del "
        "lado del servidor. Sin ese despliegue, un usuario autenticado podría leer "
        "rutas que no le pertenecen."
    )
