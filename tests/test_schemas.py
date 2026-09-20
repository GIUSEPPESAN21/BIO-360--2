"""
tests/test_schemas.py
=====================
Pruebas del parsing y la validación de la salida estructurada (`services.ai.schemas`).

Por qué existen estas pruebas
-----------------------------
La Opción K1 sustituye el parsing frágil de texto libre por un esquema JSON validable.
Pero un modelo de lenguaje sigue siendo una fuente no confiable: puede envolver el JSON
en un bloque markdown, añadir texto de cortesía alrededor, o inventarse un nombre de
dilema que no existe en la base de conocimiento.

La regla más importante que se prueba aquí: si el modelo propone un dilema que NO está
en `dilemas.json`, el sistema lo DESCARTA en lugar de aceptarlo. Aceptarlo significaría
crear categorías clínicas inexistentes a partir de una alucinación.
"""

from __future__ import annotations

from services.ai.schemas import (
    SCHEMA_ANALISIS_CLINICO,
    formatear_analisis_clinico,
    parsear_json_tolerante,
    validar_analisis_clinico,
)

DILEMAS_VALIDOS = [
    "Dilemas Éticos en Neonatología",
    "Limitación del Esfuerzo Terapéutico",
]


# --- parsear_json_tolerante --------------------------------------------------------

def test_json_crudo():
    assert parsear_json_tolerante('{"a": 1}') == {"a": 1}


def test_json_en_bloque_markdown():
    texto = '```json\n{"a": 1, "b": "dos"}\n```'
    assert parsear_json_tolerante(texto) == {"a": 1, "b": "dos"}


def test_json_en_bloque_markdown_sin_etiqueta():
    texto = '```\n{"a": 1}\n```'
    assert parsear_json_tolerante(texto) == {"a": 1}


def test_json_rodeado_de_texto_de_cortesia():
    texto = 'Claro, aquí tienes el análisis:\n{"a": 1}\nEspero que sea útil.'
    assert parsear_json_tolerante(texto) == {"a": 1}


def test_json_anidado_se_extrae_completo():
    texto = 'Resultado: {"a": {"b": [1, 2]}, "c": 3} listo.'
    assert parsear_json_tolerante(texto) == {"a": {"b": [1, 2]}, "c": 3}


def test_texto_sin_json_devuelve_none():
    assert parsear_json_tolerante("No pude generar el análisis.") is None


def test_cadena_vacia_devuelve_none():
    assert parsear_json_tolerante("") is None
    assert parsear_json_tolerante(None) is None


def test_array_json_en_la_raiz_devuelve_none():
    """Solo se acepta un objeto: una lista no encaja en el esquema esperado."""
    assert parsear_json_tolerante('[1, 2, 3]') is None


def test_json_malformado_devuelve_none():
    assert parsear_json_tolerante('{"a": 1,,}') is None


# --- validar_analisis_clinico ------------------------------------------------------

def test_dilema_exacto_se_acepta():
    data = {
        "dilema_sugerido": "Dilemas Éticos en Neonatología",
        "resumen": "Resumen del caso.",
    }
    validado = validar_analisis_clinico(data, DILEMAS_VALIDOS)

    assert validado["valido"] is True
    assert validado["dilema_sugerido"] == "Dilemas Éticos en Neonatología"


def test_coincidencia_laxa_por_mayusculas():
    data = {
        "dilema_sugerido": "dilemas éticos en neonatología",
        "resumen": "Resumen.",
    }
    validado = validar_analisis_clinico(data, DILEMAS_VALIDOS)

    assert validado["dilema_sugerido"] == "Dilemas Éticos en Neonatología"


def test_dilema_inventado_se_descarta():
    """
    Regla crítica: un dilema que no está en la base de conocimiento NO se acepta.
    El sistema prefiere no sugerir nada antes que inventar una categoría clínica.
    """
    data = {
        "dilema_sugerido": "Conflicto Interplanetario de Recursos",
        "resumen": "Resumen.",
    }
    validado = validar_analisis_clinico(data, DILEMAS_VALIDOS)

    assert validado["dilema_sugerido"] == ""


def test_campos_de_lista_se_normalizan():
    data = {
        "resumen": "R.",
        "elementos_bioeticos_clave": ["  uno  ", "", "dos", None],
        "normativas_aplicables": "una sola norma como cadena",
    }
    validado = validar_analisis_clinico(data, DILEMAS_VALIDOS)

    assert validado["elementos_bioeticos_clave"] == ["uno", "dos"]
    assert validado["normativas_aplicables"] == ["una sola norma como cadena"]


def test_nivel_de_riesgo_invalido_se_descarta():
    data = {"resumen": "R.", "nivel_riesgo": "Apocalíptico"}
    validado = validar_analisis_clinico(data, DILEMAS_VALIDOS)

    assert validado["nivel_riesgo"] == ""


def test_nivel_de_riesgo_valido_se_conserva():
    for nivel in ("Bajo", "Moderado", "Crítico"):
        validado = validar_analisis_clinico(
            {"resumen": "R.", "nivel_riesgo": nivel}, DILEMAS_VALIDOS
        )
        assert validado["nivel_riesgo"] == nivel


def test_entrada_no_dict_devuelve_estructura_vacia_sin_lanzar():
    for entrada in (None, "texto", [1, 2], 42):
        validado = validar_analisis_clinico(entrada, DILEMAS_VALIDOS)
        assert validado["valido"] is False
        assert validado["dilema_sugerido"] == ""


def test_sin_resumen_ni_elementos_no_es_valido():
    validado = validar_analisis_clinico({"nivel_riesgo": "Bajo"}, DILEMAS_VALIDOS)
    assert validado["valido"] is False


def test_solo_con_elementos_clave_ya_es_valido():
    validado = validar_analisis_clinico(
        {"elementos_bioeticos_clave": ["autonomía en conflicto"]}, DILEMAS_VALIDOS
    )
    assert validado["valido"] is True


def test_todas_las_claves_estan_siempre_presentes():
    """La UI y el PDF leen estas claves sin comprobar su existencia."""
    validado = validar_analisis_clinico({}, DILEMAS_VALIDOS)
    esperadas = {
        "dilema_sugerido", "justificacion_dilema", "principios_en_conflicto",
        "elementos_bioeticos_clave", "normativas_aplicables", "cursos_de_accion",
        "nivel_riesgo", "resumen", "valido",
    }
    assert esperadas.issubset(set(validado))


# --- formatear_analisis_clinico ----------------------------------------------------

def test_formateo_de_analisis_invalido_da_cadena_vacia():
    assert formatear_analisis_clinico({"valido": False}) == ""


def test_formateo_incluye_las_secciones_pobladas():
    validado = validar_analisis_clinico(
        {
            "dilema_sugerido": "Dilemas Éticos en Neonatología",
            "resumen": "Caso de neonato prematuro.",
            "nivel_riesgo": "Crítico",
            "elementos_bioeticos_clave": ["autonomía de los padres"],
            "normativas_aplicables": ["Ley 1733 de 2014"],
        },
        DILEMAS_VALIDOS,
    )
    texto = formatear_analisis_clinico(validado)

    assert "Caso de neonato prematuro." in texto
    assert "Dilemas Éticos en Neonatología" in texto
    assert "Crítico" in texto
    assert "autonomía de los padres" in texto
    assert "Ley 1733 de 2014" in texto


# --- El esquema en sí --------------------------------------------------------------

def test_el_esquema_exige_los_campos_minimos():
    assert set(SCHEMA_ANALISIS_CLINICO["required"]) == {
        "dilema_sugerido", "elementos_bioeticos_clave", "resumen",
    }


def test_el_esquema_restringe_los_principios_a_los_cuatro_canonicos():
    enum = SCHEMA_ANALISIS_CLINICO["properties"]["principios_en_conflicto"]["items"]["enum"]
    assert set(enum) == {"autonomia", "beneficencia", "no_maleficencia", "justicia"}
