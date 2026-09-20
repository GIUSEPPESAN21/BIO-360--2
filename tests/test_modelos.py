"""
tests/test_modelos.py
=====================
Pruebas de los modelos de dominio y las conversiones seguras (`core.modelos`).

`safe_int` y `safe_str` son la frontera entre los datos del formulario de Streamlit
(donde un campo vacío llega como cadena) y la aritmética del semáforo ético. Si
`safe_int` devolviera None en algún borde, `sum(valores.values())` en `core.etica`
lanzaría y el análisis del caso caería a mitad de camino.
"""

from __future__ import annotations

import pytest

from core.modelos import (
    PERSPECTIVAS_NOMBRES,
    PRINCIPIOS,
    PRINCIPIOS_LABELS,
    CasoBioetico,
    safe_int,
    safe_str,
)


# --- safe_int ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "entrada, esperado",
    [
        (None, 0),
        ("", 0),
        ("abc", 0),
        ("5", 5),
        (5, 5),
        (5.9, 5),      # truncamiento, no redondeo
        (-3, -3),
        (True, 1),
        ([], 0),       # tipo no convertible
        ({}, 0),
    ],
)
def test_safe_int_bordes(entrada, esperado):
    assert safe_int(entrada) == esperado


def test_safe_int_respeta_el_default_propio():
    assert safe_int(None, default=7) == 7
    assert safe_int("", default=7) == 7
    assert safe_int("no es numero", default=7) == 7
    # Un valor válido ignora el default
    assert safe_int("3", default=7) == 3


def test_safe_int_siempre_devuelve_int():
    """Garantía que `core.etica` da por supuesta al sumar ponderaciones."""
    for entrada in (None, "", "abc", "5", 5.9, [], {}):
        assert isinstance(safe_int(entrada), int)


# --- safe_str ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "entrada, esperado",
    [
        (None, ""),
        ("", ""),
        ("  hola  ", "hola"),
        ("hola", "hola"),
        (5, "5"),
        (5.5, "5.5"),
    ],
)
def test_safe_str_bordes(entrada, esperado):
    assert safe_str(entrada) == esperado


def test_safe_str_respeta_el_default_solo_para_none():
    assert safe_str(None, default="N/A") == "N/A"
    # Una cadena vacía NO es None: se normaliza a vacío, no al default
    assert safe_str("", default="N/A") == ""


# --- Constantes de dominio ---------------------------------------------------------

def test_principios_y_etiquetas_estan_alineados():
    """Los gráficos emparejan PRINCIPIOS con PRINCIPIOS_LABELS por posición."""
    assert len(PRINCIPIOS) == len(PRINCIPIOS_LABELS) == 4


def test_las_tres_perspectivas_estan_definidas():
    assert set(PERSPECTIVAS_NOMBRES) == {"medico", "familia", "comite"}


# --- CasoBioetico ------------------------------------------------------------------

def test_caso_sin_kwargs_usa_defaults_sin_lanzar():
    caso = CasoBioetico()

    assert caso.nombre_paciente == "N/A"
    assert caso.nombre_analista == "N/A"
    assert caso.edad == 0
    assert caso.genero == "N/A"
    assert caso.condicion == "Estable"
    assert caso.historia_clinica.startswith("caso_")


def test_caso_extrae_las_tres_perspectivas_con_sus_cuatro_principios():
    caso = CasoBioetico()

    assert set(caso.perspectivas) == {"medico", "familia", "comite"}
    for valores in caso.perspectivas.values():
        assert set(valores) == set(PRINCIPIOS)
        assert all(isinstance(v, int) for v in valores.values())


def test_caso_lee_las_ponderaciones_del_formulario():
    caso = CasoBioetico(
        nivel_autonomia_medico=5,
        nivel_beneficencia_medico="4",   # llega como cadena desde el formulario
        nivel_no_maleficencia_medico=None,
        nivel_justicia_medico=2,
    )

    assert caso.perspectivas["medico"]["autonomia"] == 5
    assert caso.perspectivas["medico"]["beneficencia"] == 4
    assert caso.perspectivas["medico"]["no_maleficencia"] == 0
    assert caso.perspectivas["medico"]["justicia"] == 2


def test_dilema_por_defecto_es_el_primero_de_las_opciones():
    caso = CasoBioetico(dilemas_opciones=["Dilema A", "Dilema B"])
    assert caso.dilema_etico == "Dilema A"


def test_dilema_explicito_gana_sobre_el_default():
    caso = CasoBioetico(dilemas_opciones=["Dilema A"], dilema_etico="Dilema B")
    assert caso.dilema_etico == "Dilema B"


def test_sin_opciones_de_dilema_no_lanza():
    caso = CasoBioetico(dilemas_opciones=[])
    assert caso.dilema_etico == ""


# --- nombres_pii: lo que se debe anonimizar antes de salir a la IA -----------------

def test_nombres_pii_excluye_placeholders_y_vacios():
    caso = CasoBioetico()  # ambos nombres quedan en 'N/A'
    assert caso.nombres_pii() == []


def test_nombres_pii_devuelve_paciente_y_analista():
    caso = CasoBioetico(nombre_paciente="Ana Ruiz", nombre_analista="dr@hospital.co")
    nombres = caso.nombres_pii()

    assert "Ana Ruiz" in nombres
    assert "dr@hospital.co" in nombres
    assert len(nombres) == 2


def test_nombres_pii_omite_solo_el_ausente():
    caso = CasoBioetico(nombre_paciente="Ana Ruiz")
    assert caso.nombres_pii() == ["Ana Ruiz"]
