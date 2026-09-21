"""
tests/test_app.py
=================
Pruebas de integración del punto de entrada (`app.py`) con `streamlit.testing.AppTest`.

Por qué existen estas pruebas
-----------------------------
`app.py` es puro cableado: compone core/, services/ y ui/. Los errores que introduce no
son de lógica, son de integración — una firma que no coincide, un módulo que no importa,
un secreto que se lee mal. Ese tipo de fallo no lo detecta ninguna prueba unitaria, solo
ejecutar el script.

`AppTest` ejecuta el script de Streamlit en memoria, sin navegador ni servidor, y expone
las excepciones que se hayan producido durante el renderizado.

Nota sobre credenciales
-----------------------
Estas pruebas corren DELIBERADAMENTE sin `secrets.toml`. Eso verifica el camino de
degradación elegante: sin credenciales la aplicación debe avisar por la interfaz, nunca
romperse. Fue justamente aquí donde se detectó que `st.secrets.get` lanza
`StreamlitSecretNotFoundError` en lugar de devolver None.

Lo que estas pruebas NO cubren: login real contra Firebase, persistencia en Firestore y
llamadas efectivas a los modelos de IA. Todo eso exige credenciales reales.
"""

from __future__ import annotations

import pytest

from streamlit.testing.v1 import AppTest

TIEMPO_LIMITE = 120
USUARIO_DE_PRUEBA = {"localId": "uid_de_prueba", "email": "prueba@ejemplo.com"}


@pytest.fixture
def app_sin_autenticar() -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=TIEMPO_LIMITE)
    at.run()
    return at


@pytest.fixture
def app_autenticada() -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=TIEMPO_LIMITE)
    at.session_state["user"] = dict(USUARIO_DE_PRUEBA)
    at.run()
    return at


# --- Ruta sin autenticar -----------------------------------------------------------

def test_la_app_arranca_sin_autenticar_y_sin_excepciones(app_sin_autenticar):
    assert not app_sin_autenticar.exception


def test_sin_autenticar_se_muestra_la_pantalla_de_acceso(app_sin_autenticar):
    encabezados = [h.value for h in app_sin_autenticar.header]
    assert any("Acceso de Usuario" in h for h in encabezados)


def test_sin_autenticar_no_se_filtra_la_aplicacion_principal(app_sin_autenticar):
    """Sin sesión no debe renderizarse el dashboard ni sus pestañas."""
    titulos = [t.value for t in app_sin_autenticar.title]
    assert not any("BIOETHICARE 360º" in t for t in titulos)


# --- Ruta autenticada: aquí se cablean las cinco pestañas --------------------------

def test_la_app_autenticada_renderiza_sin_excepciones(app_autenticada):
    assert not app_autenticada.exception


def test_la_app_autenticada_muestra_el_titulo(app_autenticada):
    titulos = [t.value for t in app_autenticada.title]
    assert any("BIOETHICARE 360º" in t for t in titulos)


def test_estan_los_doce_sliders_de_ponderacion(app_autenticada):
    """3 perspectivas x 4 principios. Si falta alguno, el semáforo recibe datos parciales."""
    assert len(app_autenticada.slider) == 12


def test_los_sliders_cubren_el_rango_cero_a_cinco(app_autenticada):
    for slider in app_autenticada.slider:
        assert slider.min == 0
        assert slider.max == 5


def test_esta_el_formulario_de_registro_del_caso(app_autenticada):
    """Descripción, contexto sociocultural, puntos clave e historia clínica."""
    assert len(app_autenticada.text_area) >= 4


def test_esta_el_selector_de_dilema_etico(app_autenticada):
    """El desplegable se puebla desde dilemas.json a través de core.conocimiento."""
    opciones = [tuple(s.options) for s in app_autenticada.selectbox if s.options]
    assert any(len(o) >= 9 for o in opciones), "No se encontró el selector de dilemas poblado"


# --- Degradación elegante sin credenciales -----------------------------------------

def test_sin_credenciales_se_avisa_por_la_interfaz_en_vez_de_romper(app_autenticada):
    """
    REGRESIÓN del defecto de `st.secrets`: sin `secrets.toml` la aplicación debe avisar,
    no lanzar. Si esta prueba falla con una excepción, el guardia de `_resolver` se perdió.
    """
    assert not app_autenticada.exception

    avisos = " ".join(w.value for w in app_autenticada.warning)
    assert "API" in avisos, "No se avisó de la ausencia de clave de API"


def test_sin_base_de_datos_se_informa_que_no_se_guardaran_los_casos(app_autenticada):
    mensajes = " ".join(i.value for i in app_autenticada.info)
    assert "base de datos" in mensajes.lower()


# --- El estado de sesión queda bien inicializado -----------------------------------

def test_el_estado_de_sesion_se_inicializa(app_autenticada):
    estado = app_autenticada.session_state

    assert estado["ai_provider"]
    assert estado["temp_dir"]
    assert estado["chat_history"] == []


def test_no_se_declara_un_modelo_activo_antes_de_la_primera_llamada(app_autenticada):
    """
    El monolito inicializaba `selected_model` con "gemini-2.0-flash-exp" y lo mostraba
    como modelo activo aunque nunca se hubiera invocado. No debe volver a ocurrir.
    """
    assert app_autenticada.session_state["selected_model"] is None
