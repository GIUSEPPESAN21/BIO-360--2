"""
tests/test_pdf.py
=================
Pruebas de la generación de PDFs (`services.pdf_service`) y de los gráficos que
incrusta (`services.charts`).

Por qué existen estas pruebas
-----------------------------
El PDF es el entregable formal que llega al comité de bioética. Dos fallos históricos
justifican estas pruebas:

1. El reporte omitía los gráficos y se limitaba a decir que estaban "en la aplicación
   web", dejando sin evidencia visual al documento que se archiva (mejora M4).
2. Un carácter `<` suelto en una historia clínica rompía el build de ReportLab, porque
   el texto del usuario se interpolaba en su XML interno sin escapar.

La incrustación de gráficos depende de `kaleido`, que puede no estar disponible en
cualquier entorno. El contrato es de DEGRADACIÓN ELEGANTE: sin kaleido el PDF se genera
igual, con una nota explícita. Estas pruebas verifican el PDF siempre, y reportan por
separado si los gráficos pudieron incrustarse.
"""

from __future__ import annotations

import pytest

from services.charts import (
    figura_consenso,
    figura_equilibrio,
    figura_radar,
    generar_todas_las_figuras,
    normalizar_perspectivas,
)
from services.pdf_service import (
    crear_consentimiento_pdf,
    crear_reporte_pdf_completo,
    figura_a_imagen,
)

PERSPECTIVAS = {
    "Equipo Médico": {"autonomia": 4, "beneficencia": 5, "no_maleficencia": 3, "justicia": 2},
    "Familia/Paciente": {"autonomia": 5, "beneficencia": 3, "no_maleficencia": 2, "justicia": 3},
    "Comité de Bioética": {"autonomia": 3, "beneficencia": 4, "no_maleficencia": 4, "justicia": 4},
}


def reporte_de_prueba(**overrides) -> dict:
    base = {
        "ID del Caso": "HC-001",
        "Fecha Análisis": "2026-09-20 10:00:00",
        "Analista": "analista@hospital.co",
        "Resumen del Paciente": "Paciente de 72 años, condición Terminal.",
        "Dilema Ético Principal (Seleccionado)": "Limitación del Esfuerzo Terapéutico",
        "Descripción Detallada del Caso": "Descripción con **negrita** y salto\nde línea.",
        "AnalisisMultiperspectiva": PERSPECTIVAS,
        "AnalisisEtico": {
            "severidad": "Moderado",
            "advertencias": ["**Alto Desequilibrio Interno:** en el equipo médico."],
            "recomendaciones": ["Revisar la ponderación."],
        },
        "Análisis Deliberativo (IA)": "Análisis generado por el modelo.",
        "Trazabilidad IA": {
            "proveedor": "Kiro",
            "modelo": "kiro-1",
            "prompt_version": "deliberacion_comite_v2",
            "timestamp": "2026-09-20T15:00:00+00:00",
        },
        "Historial del Chat de Deliberación": [
            {"role": "user", "content": "¿Cuál es el conflicto principal?"},
            {"role": "assistant", "content": "Autonomía frente a no maleficencia."},
        ],
    }
    base.update(overrides)
    return base


def es_pdf(ruta) -> bool:
    with open(ruta, "rb") as f:
        return f.read(5) == b"%PDF-"


# --- Normalización de ponderaciones ------------------------------------------------

def test_normalizar_acepta_nombres_largos_del_reporte():
    normalizado = normalizar_perspectivas(PERSPECTIVAS)
    assert set(normalizado) == {"medico", "familia", "comite"}


def test_normalizar_acepta_claves_cortas():
    cortas = {"medico": PERSPECTIVAS["Equipo Médico"]}
    assert "medico" in normalizar_perspectivas(cortas)


def test_normalizar_acepta_un_objeto_con_atributo_perspectivas():
    class CasoDoble:
        perspectivas = {"medico": PERSPECTIVAS["Equipo Médico"]}

    assert "medico" in normalizar_perspectivas(CasoDoble())


def test_normalizar_tolera_entradas_invalidas():
    assert normalizar_perspectivas(None) == {}
    assert normalizar_perspectivas("texto") == {}


# --- Figuras -----------------------------------------------------------------------

def test_las_tres_figuras_se_generan():
    figuras = generar_todas_las_figuras(PERSPECTIVAS)

    assert set(figuras) == {"radar", "consenso", "equilibrio"}
    assert all(f is not None for f in figuras.values())


def test_sin_ponderaciones_no_hay_figuras():
    for fabrica in (figura_radar, figura_consenso, figura_equilibrio):
        assert fabrica({}) is None


def test_una_perspectiva_desconocida_no_lanza_por_color():
    """El monolito usaba colores[clave] y reventaba con un KeyError."""
    raras = {"perspectiva_inventada": {"autonomia": 1, "beneficencia": 1,
                                       "no_maleficencia": 1, "justicia": 1}}
    assert figura_radar(raras) is not None
    assert figura_equilibrio(raras) is not None


# --- Reporte deliberativo en PDF ---------------------------------------------------

def test_el_reporte_pdf_se_genera(tmp_path):
    destino = tmp_path / "reporte.pdf"
    crear_reporte_pdf_completo(reporte_de_prueba(), str(destino))

    assert destino.exists()
    assert es_pdf(destino)
    assert destino.stat().st_size > 1500


def test_el_reporte_pdf_sobrevive_a_caracteres_que_rompen_el_xml(tmp_path):
    """
    REGRESIÓN. Una historia clínica con `<`, `&` o `>` rompía el build de ReportLab.
    """
    destino = tmp_path / "reporte_xml.pdf"
    peligroso = "Saturación <90% & frecuencia >100 lpm; ver nota <urgente>."
    crear_reporte_pdf_completo(
        reporte_de_prueba(**{"Descripción Detallada del Caso": peligroso}), str(destino)
    )

    assert destino.exists()
    assert es_pdf(destino)


def test_el_reporte_pdf_funciona_sin_ponderaciones(tmp_path):
    """Un caso antiguo o incompleto no debe impedir la descarga del PDF."""
    destino = tmp_path / "reporte_sin_graficos.pdf"
    crear_reporte_pdf_completo(
        reporte_de_prueba(**{"AnalisisMultiperspectiva": {}}), str(destino)
    )

    assert destino.exists()
    assert es_pdf(destino)


def test_el_reporte_pdf_funciona_con_un_reporte_minimo(tmp_path):
    destino = tmp_path / "minimo.pdf"
    crear_reporte_pdf_completo({"ID del Caso": "HC-002"}, str(destino))

    assert destino.exists()
    assert es_pdf(destino)


def test_incrustar_los_graficos_agranda_el_pdf(tmp_path):
    """
    Si kaleido está disponible, el PDF con gráficos debe pesar claramente más que el
    mismo reporte sin ponderaciones. Si no lo está, la prueba se omite en lugar de
    fallar: la degradación elegante es el comportamiento esperado, no un error.
    """
    if figura_a_imagen(figura_radar(PERSPECTIVAS)) is None:
        pytest.skip("kaleido no disponible en este entorno: el PDF degrada sin gráficos")

    con = tmp_path / "con.pdf"
    sin = tmp_path / "sin.pdf"
    crear_reporte_pdf_completo(reporte_de_prueba(), str(con))
    crear_reporte_pdf_completo(
        reporte_de_prueba(**{"AnalisisMultiperspectiva": {}}), str(sin)
    )

    assert con.stat().st_size > sin.stat().st_size


# --- Consentimiento informado en PDF ------------------------------------------------

def test_el_consentimiento_pdf_se_genera(tmp_path):
    destino = tmp_path / "consentimiento.pdf"
    texto = (
        "CONSENTIMIENTO/ASENTIMIENTO INFORMADO (BIOETHICARE 360)\n"
        "Fecha: 2026-09-20\n"
        "DATOS DEL PACIENTE\n"
        "Nombre: Ana Ruiz\n"
        "1. RIESGOS POTENCIALES:\n- Riesgo uno\n"
        "DECLARACIÓN Y FIRMA\n"
        "Firma del Paciente/Tutor Legal: _______\n"
    )
    crear_consentimiento_pdf(texto, str(destino))

    assert destino.exists()
    assert es_pdf(destino)


def test_el_consentimiento_pdf_tolera_texto_vacio(tmp_path):
    destino = tmp_path / "consentimiento_vacio.pdf"
    crear_consentimiento_pdf("", str(destino))

    assert destino.exists()
    assert es_pdf(destino)
