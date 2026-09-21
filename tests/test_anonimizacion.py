"""
tests/test_anonimizacion.py
===========================
Pruebas de la capa de anonimización de PII (`core.anonimizacion`).

Por qué existen estas pruebas
-----------------------------
Esta capa es el control técnico que sostiene el cumplimiento de la Ley 1581, HIPAA y
GDPR: es lo único que impide que el nombre, la cédula o la fecha de ingreso de un
paciente real viajen a la API de un tercero. Un fallo aquí no produce un error visible
en la aplicación; produce una fuga silenciosa de datos clínicos.

Dos propiedades son críticas y se prueban explícitamente:
  1. REVERSIBILIDAD  — el usuario autorizado debe poder leer el texto real.
  2. ESTABILIDAD     — un mismo valor siempre da el mismo token, para que el modelo
                       mantenga la coherencia referencial ([PACIENTE_1] es siempre la
                       misma persona a lo largo del análisis).
"""

from __future__ import annotations

from core.anonimizacion import Anonimizador, anonimizar_texto


# --- Reversibilidad ----------------------------------------------------------------

def test_round_trip_devuelve_el_texto_original():
    original = "Juan Pérez ingresó el 12/03/2024 en Bogotá."
    anon = Anonimizador(nombres_conocidos=["Juan Pérez"])

    limpio = anon.anonimizar(original)
    assert limpio != original, "El texto no fue anonimizado en absoluto"

    assert anon.rehidratar(limpio) == original


def test_round_trip_con_varios_tipos_de_pii():
    original = (
        "Paciente María Gómez, correo maria.gomez@hospital.co, "
        "teléfono 3001234567, documento CC 1.234.567."
    )
    anon = Anonimizador(nombres_conocidos=["María Gómez"])

    limpio = anon.anonimizar(original)
    assert anon.rehidratar(limpio) == original


# --- Estabilidad de tokens ---------------------------------------------------------

def test_mismo_valor_produce_siempre_el_mismo_token():
    anon = Anonimizador(nombres_conocidos=["Ana Ruiz"])
    limpio = anon.anonimizar("Ana Ruiz habló con la familia. Luego Ana Ruiz firmó.")

    # El nombre aparece dos veces: debe haber un solo token, repetido.
    tokens_paciente = [t for t in anon.mapa if t.startswith("[PACIENTE_")]
    assert len(tokens_paciente) == 1
    assert limpio.count(tokens_paciente[0]) == 2


def test_valores_distintos_reciben_tokens_distintos():
    anon = Anonimizador(nombres_conocidos=["Ana Ruiz", "Luis Mora"])
    anon.anonimizar("Ana Ruiz y Luis Mora asistieron.")

    tokens = [t for t in anon.mapa if t.startswith("[PACIENTE_")]
    assert len(tokens) == 2
    assert len(set(tokens)) == 2


# --- La PII desaparece del texto que sale hacia la IA ------------------------------

def test_el_email_no_sobrevive_a_la_anonimizacion():
    anon = Anonimizador()
    limpio = anon.anonimizar("Contacto: paciente@dominio.com para seguimiento.")

    assert "paciente@dominio.com" not in limpio
    assert "[EMAIL_" in limpio


def test_el_telefono_no_sobrevive_a_la_anonimizacion():
    anon = Anonimizador()
    limpio = anon.anonimizar("Celular del acudiente: 3009876543.")

    assert "3009876543" not in limpio


def test_la_fecha_numerica_no_sobrevive_a_la_anonimizacion():
    anon = Anonimizador()
    limpio = anon.anonimizar("Ingresó el 12/03/2024 por urgencias.")

    assert "12/03/2024" not in limpio
    assert "[FECHA_" in limpio


def test_la_fecha_textual_en_espanol_no_sobrevive():
    anon = Anonimizador()
    limpio = anon.anonimizar("La junta se reunió el 12 de marzo de 2024 a las 9.")

    assert "12 de marzo de 2024" not in limpio


def test_el_documento_de_identidad_no_sobrevive():
    anon = Anonimizador()
    limpio = anon.anonimizar("Identificado con CC 1.234.567 del registro.")

    assert "1.234.567" not in limpio


def test_el_nombre_se_sustituye_sin_importar_la_capitalizacion():
    anon = Anonimizador(nombres_conocidos=["Juan Pérez"])
    limpio = anon.anonimizar("JUAN PÉREZ y juan pérez son la misma persona.")

    assert "JUAN PÉREZ" not in limpio
    assert "juan pérez" not in limpio
    assert "[PACIENTE_1]" in limpio


def test_nombres_demasiado_cortos_o_placeholder_se_ignoran():
    """'N/A' y cadenas de menos de 3 caracteres no deben generar tokens basura."""
    anon = Anonimizador(nombres_conocidos=["N/A", "ab", ""])
    limpio = anon.anonimizar("El paciente ab no tiene nombre registrado: N/A.")

    assert not [t for t in anon.mapa if t.startswith("[PACIENTE_")]
    assert "N/A" in limpio


# --- El mapa de rehidratación ------------------------------------------------------

def test_el_mapa_contiene_los_valores_originales():
    anon = Anonimizador(nombres_conocidos=["Ana Ruiz"])
    anon.anonimizar("Ana Ruiz, correo ana@x.com.")

    assert "Ana Ruiz" in anon.mapa.values()
    assert "ana@x.com" in anon.mapa.values()


def test_el_mapa_es_una_copia_defensiva():
    """Mutar el mapa devuelto no debe corromper el estado interno del anonimizador."""
    anon = Anonimizador(nombres_conocidos=["Ana Ruiz"])
    anon.anonimizar("Ana Ruiz firmó.")

    copia = anon.mapa
    copia.clear()

    assert anon.mapa, "El mapa interno quedó vacío: se expuso la estructura real"


# --- Robustez ----------------------------------------------------------------------

def test_texto_vacio_y_none_no_lanzan():
    anon = Anonimizador()

    assert anon.anonimizar("") == ""
    assert anon.rehidratar("") == ""
    assert anon.anonimizar(None) == ""
    assert anon.rehidratar(None) == ""


def test_texto_sin_pii_queda_intacto():
    anon = Anonimizador()
    texto = "Se discutió el principio de autonomía frente a la beneficencia."

    assert anon.anonimizar(texto) == texto


def test_funcion_de_conveniencia_devuelve_texto_y_mapa():
    limpio, mapa = anonimizar_texto("Ana Ruiz llamó.", nombres_conocidos=["Ana Ruiz"])

    assert "Ana Ruiz" not in limpio
    assert isinstance(mapa, dict)
    assert "Ana Ruiz" in mapa.values()


def test_funcion_de_conveniencia_sin_nombres_conocidos():
    limpio, mapa = anonimizar_texto("Correo: x@y.com")

    assert "x@y.com" not in limpio
    assert "x@y.com" in mapa.values()
