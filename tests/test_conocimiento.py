"""
tests/test_conocimiento.py
==========================
Pruebas de la carga y validación de la base de conocimiento (`core.conocimiento`).

Por qué existen estas pruebas
-----------------------------
El contenido de `dilemas.json` no es solo configuración: se inyecta como contexto de
grounding en los prompts, y al modelo se le instruye a citar normativa ÚNICAMENTE desde
ahí. Una entrada malformada no produce un error visible, produce un prompt degradado.

Se comprueban las dos mitades del contrato de diseño:
  - La base REAL del proyecto está sana (si deja de estarlo, alguien la rompió).
  - Una base corrupta degrada con elegancia en lugar de tumbar la aplicación.
"""

from __future__ import annotations

import json

import pytest

from core.conocimiento import (
    CLAVES_ESPERADAS,
    cargar_dilemas,
    info_dilema,
    listar_dilemas,
    normalizar_dilema,
    validar_base_dilemas,
)


# --- La base real del proyecto -----------------------------------------------------

def test_la_base_real_carga_y_esta_sana():
    base = cargar_dilemas()

    assert base, "No se pudo cargar dilemas.json"
    assert validar_base_dilemas(base) == [], "La base de dilemas del proyecto tiene problemas de esquema"


def test_la_carga_no_pierde_dilemas():
    """El saneamiento no debe descartar entradas legítimas."""
    base = cargar_dilemas()
    with open("dilemas.json", "r", encoding="utf-8") as f:
        crudo = json.load(f)

    assert len(base) == len(crudo)
    assert set(base) == set(crudo)


def test_modo_estricto_no_lanza_sobre_la_base_real():
    """Si esto falla, la base del proyecto dejó de cumplir su propio esquema."""
    base = cargar_dilemas(estricto=True)
    assert base


def test_cada_dilema_real_expone_las_cuatro_claves():
    base = cargar_dilemas()

    for nombre, info in base.items():
        assert set(info) == set(CLAVES_ESPERADAS), f"Dilema '{nombre}' con claves inesperadas"
        for clave in CLAVES_ESPERADAS:
            assert isinstance(info[clave], list)
            assert info[clave], f"Dilema '{nombre}': '{clave}' vacío"
            assert all(isinstance(x, str) for x in info[clave])


# --- validar_base_dilemas ----------------------------------------------------------

def test_base_sana_no_reporta_problemas():
    sana = {
        "Dilema X": {
            "riesgos": ["r1"], "beneficios": ["b1"],
            "alternativas": ["a1"], "normativas": ["n1"],
        }
    }
    assert validar_base_dilemas(sana) == []


def test_detecta_entrada_irrecuperable():
    problemas = validar_base_dilemas({"Dilema X": "esto no es un objeto"})
    assert any("irrecuperable" in p for p in problemas)


def test_detecta_clave_ausente():
    problemas = validar_base_dilemas({"Dilema X": {"riesgos": ["r"]}})
    assert any("beneficios" in p for p in problemas)


def test_detecta_clave_que_no_es_lista():
    problemas = validar_base_dilemas(
        {"Dilema X": {"riesgos": "no soy lista", "beneficios": ["b"],
                      "alternativas": ["a"], "normativas": ["n"]}}
    )
    assert any("no es una lista" in p for p in problemas)


def test_detecta_lista_vacia():
    problemas = validar_base_dilemas(
        {"Dilema X": {"riesgos": [], "beneficios": ["b"],
                      "alternativas": ["a"], "normativas": ["n"]}}
    )
    assert any("está vacía" in p for p in problemas)


def test_detecta_elemento_que_no_es_cadena():
    problemas = validar_base_dilemas(
        {"Dilema X": {"riesgos": ["ok", 42], "beneficios": ["b"],
                      "alternativas": ["a"], "normativas": ["n"]}}
    )
    assert any("no es una cadena" in p for p in problemas)


def test_base_vacia_o_no_dict_se_reporta_sin_lanzar():
    assert validar_base_dilemas({}) != []
    assert validar_base_dilemas(None) != []
    assert validar_base_dilemas("texto") != []


# --- normalizar_dilema -------------------------------------------------------------

def test_normalizar_garantiza_las_cuatro_claves():
    for entrada in (None, "texto", {}, 42, []):
        normalizado = normalizar_dilema(entrada)
        assert set(normalizado) == set(CLAVES_ESPERADAS)
        assert all(isinstance(v, list) and v for v in normalizado.values())


def test_normalizar_descarta_elementos_invalidos():
    normalizado = normalizar_dilema(
        {"riesgos": ["  válido  ", "", None, 42, "otro"]}
    )
    assert normalizado["riesgos"] == ["válido", "otro"]


def test_normalizar_tolera_una_cadena_suelta_como_lista():
    normalizado = normalizar_dilema({"normativas": "Ley 1733 de 2014"})
    assert normalizado["normativas"] == ["Ley 1733 de 2014"]


def test_normalizar_aplica_valores_por_defecto_cuando_queda_vacio():
    normalizado = normalizar_dilema({"riesgos": [None, 42]})
    assert normalizado["riesgos"] == ["No especificados"]


# --- Degradación elegante ----------------------------------------------------------

def test_archivo_inexistente_devuelve_dict_vacio_sin_lanzar():
    assert cargar_dilemas("no_existe_este_archivo.json") == {}


def test_archivo_inexistente_en_modo_estricto_lanza():
    with pytest.raises(ValueError):
        cargar_dilemas("no_existe_este_archivo.json", estricto=True)


def test_json_invalido_devuelve_dict_vacio_sin_lanzar(tmp_path):
    ruta = tmp_path / "roto.json"
    ruta.write_text("{esto no es json valido", encoding="utf-8")

    assert cargar_dilemas(str(ruta)) == {}


def test_json_que_no_es_objeto_en_la_raiz_devuelve_vacio(tmp_path):
    ruta = tmp_path / "lista.json"
    ruta.write_text('["a", "b"]', encoding="utf-8")

    assert cargar_dilemas(str(ruta)) == {}


def test_base_parcialmente_corrupta_conserva_la_parte_sana(tmp_path):
    """
    Decisión de diseño: la aplicación sigue operando con los dilemas válidos en lugar
    de caer por uno mal escrito. El irrecuperable se descarta.
    """
    ruta = tmp_path / "mixto.json"
    ruta.write_text(
        json.dumps({
            "Bueno": {"riesgos": ["r"], "beneficios": ["b"],
                      "alternativas": ["a"], "normativas": ["n"]},
            "Irrecuperable": "no soy un objeto",
            "Incompleto": {"riesgos": ["r"]},
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    base = cargar_dilemas(str(ruta))

    assert "Bueno" in base
    assert "Irrecuperable" not in base, "La entrada irrecuperable debió descartarse"
    # El incompleto se conserva pero normalizado, con defaults explícitos
    assert "Incompleto" in base
    assert base["Incompleto"]["beneficios"] == ["No especificados"]


def test_base_corrupta_en_modo_estricto_lanza(tmp_path):
    ruta = tmp_path / "corrupto.json"
    ruta.write_text('{"X": "no soy objeto"}', encoding="utf-8")

    with pytest.raises(ValueError):
        cargar_dilemas(str(ruta), estricto=True)


# --- API pública: compatibilidad hacia atrás ---------------------------------------

def test_listar_dilemas_preserva_el_orden_del_archivo():
    base = cargar_dilemas()
    assert listar_dilemas(base) == list(base.keys())


def test_listar_dilemas_tolera_entrada_invalida():
    assert listar_dilemas(None) == []
    assert listar_dilemas("texto") == []


def test_info_dilema_de_uno_existente():
    base = cargar_dilemas()
    primero = listar_dilemas(base)[0]

    ficha = info_dilema(base, primero)
    assert set(ficha) == set(CLAVES_ESPERADAS)


def test_info_dilema_inexistente_devuelve_ficha_por_defecto_completa():
    """`services.reportes.generar_texto_consentimiento` lee las 4 claves sin comprobar."""
    ficha = info_dilema(cargar_dilemas(), "DILEMA_QUE_NO_EXISTE")

    assert set(ficha) == set(CLAVES_ESPERADAS)
    assert ficha["riesgos"] == ["No especificados"]
    assert ficha["normativas"] == ["No especificadas"]


def test_info_dilema_tolera_base_invalida():
    ficha = info_dilema(None, "cualquiera")
    assert set(ficha) == set(CLAVES_ESPERADAS)
