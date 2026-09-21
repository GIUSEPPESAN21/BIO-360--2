"""
tests/test_etica.py
===================
Pruebas del semáforo ético determinista (`core.etica`).

Por qué existen estas pruebas
-----------------------------
Esta es la única parte del sistema que DECIDE algo con consecuencia clínica: el nivel
de severidad que el comité ve en el dashboard y en el PDF. No la calcula un modelo de
IA, sino un motor de reglas, precisamente para que sea auditable y reproducible.

Un cambio silencioso en cualquiera de los umbrales alteraría diagnósticos pasados y
futuros sin que nadie lo note. Por eso hay una prueba que falla explícitamente si los
umbrales cambian: no es redundante, es un candado.
"""

from __future__ import annotations

import pytest

from core.etica import (
    PESO_DESEQUILIBRIO_EXTERNO,
    PESO_DESEQUILIBRIO_INTERNO,
    PESO_PERSPECTIVA_OMITIDA,
    PESO_PRINCIPIO_OMITIDO,
    UMBRAL_DESEQUILIBRIO_EXTERNO,
    UMBRAL_DESEQUILIBRIO_INTERNO,
    UMBRAL_SEVERIDAD_CRITICO,
    UMBRAL_SEVERIDAD_MODERADO,
    verificar_sesgo_etico,
)

PRINCIPIOS = ("autonomia", "beneficencia", "no_maleficencia", "justicia")


class CasoDoble:
    """Doble de prueba mínimo: `verificar_sesgo_etico` solo necesita `.perspectivas`."""

    def __init__(self, perspectivas: dict) -> None:
        self.perspectivas = perspectivas


def perspectiva(valor: int = 3, **overrides: int) -> dict:
    """Crea una perspectiva con los 4 principios en `valor`, salvo los sobrescritos."""
    base = {p: valor for p in PRINCIPIOS}
    base.update(overrides)
    return base


def caso(medico=None, familia=None, comite=None) -> CasoDoble:
    return CasoDoble(
        {
            "medico": medico if medico is not None else perspectiva(),
            "familia": familia if familia is not None else perspectiva(),
            "comite": comite if comite is not None else perspectiva(),
        }
    )


# --- Caso equilibrado -------------------------------------------------------------

def test_caso_equilibrado_es_bajo_y_sin_advertencias():
    advertencias, recomendaciones, severidad = verificar_sesgo_etico(caso())

    assert severidad == "Bajo"
    assert advertencias == []
    assert recomendaciones == []


def test_advertencias_y_recomendaciones_vienen_emparejadas():
    """Cada advertencia debe traer su recomendación: la UI las presenta en conjunto."""
    _caso = caso(medico=perspectiva(0))
    advertencias, recomendaciones, _ = verificar_sesgo_etico(_caso)

    assert len(advertencias) == len(recomendaciones)
    assert len(advertencias) > 0


# --- Perspectiva y principios omitidos --------------------------------------------

def test_perspectiva_completamente_omitida_se_detecta():
    advertencias, _, severidad = verificar_sesgo_etico(caso(medico=perspectiva(0)))

    texto = " ".join(advertencias)
    assert "Perspectiva Omitida" in texto
    assert "medico" in texto
    # 3 (perspectiva omitida) + 4x1 (principios en 0) + 2 (desequilibrio externo) = 9
    assert severidad == "Crítico"


def test_principio_individual_en_cero_se_detecta():
    advertencias, _, _ = verificar_sesgo_etico(
        caso(medico=perspectiva(3, autonomia=0))
    )

    texto = " ".join(advertencias)
    assert "Principio Omitido" in texto
    assert "Autonomia" in texto or "Autonomía" in texto


def test_peso_de_principio_omitido_es_menor_que_perspectiva_omitida():
    """Omitir una perspectiva entera es más grave que omitir un principio."""
    assert PESO_PERSPECTIVA_OMITIDA > PESO_PRINCIPIO_OMITIDO


# --- Desequilibrios ----------------------------------------------------------------

def test_desequilibrio_interno_en_el_umbral_se_detecta():
    """Diferencia máx-mín dentro de una perspectiva igual al umbral: debe avisar."""
    desequilibrada = perspectiva(3, autonomia=5, beneficencia=1)
    assert max(desequilibrada.values()) - min(desequilibrada.values()) == UMBRAL_DESEQUILIBRIO_INTERNO

    advertencias, _, severidad = verificar_sesgo_etico(caso(medico=desequilibrada))

    assert "Alto Desequilibrio Interno" in " ".join(advertencias)
    # Solo aporta PESO_DESEQUILIBRIO_INTERNO = 2 -> Moderado, no Crítico
    assert severidad == "Moderado"


def test_desequilibrio_interno_por_debajo_del_umbral_no_avisa():
    apenas = perspectiva(3, autonomia=5, beneficencia=2)
    assert max(apenas.values()) - min(apenas.values()) == UMBRAL_DESEQUILIBRIO_INTERNO - 1

    advertencias, _, severidad = verificar_sesgo_etico(caso(medico=apenas))

    assert "Alto Desequilibrio Interno" not in " ".join(advertencias)
    assert severidad == "Bajo"


def test_desequilibrio_externo_se_detecta():
    """Totales muy dispares entre perspectivas, sin ningún principio en cero."""
    _caso = caso(
        medico=perspectiva(5),   # total 20
        familia=perspectiva(3),  # total 12
        comite=perspectiva(1),   # total 4
    )
    assert (20 - 4) >= UMBRAL_DESEQUILIBRIO_EXTERNO

    advertencias, _, severidad = verificar_sesgo_etico(_caso)

    assert "Alto Desequilibrio Externo" in " ".join(advertencias)
    assert severidad == "Moderado"


def test_pesos_de_desequilibrio_son_los_esperados():
    assert PESO_DESEQUILIBRIO_INTERNO == 2
    assert PESO_DESEQUILIBRIO_EXTERNO == 2


# --- El candado de los umbrales ----------------------------------------------------

def test_umbrales_de_severidad_no_deben_cambiar_silenciosamente():
    """
    CANDADO DELIBERADO.

    Si esta prueba falla, alguien movió un umbral del semáforo ético. Eso reclasifica
    casos clínicos ya analizados (un 'Moderado' puede volverse 'Bajo') y debe ser una
    decisión explícita y documentada, nunca un efecto colateral de otro cambio.
    """
    assert UMBRAL_SEVERIDAD_CRITICO == 5, "Umbral de 'Crítico' modificado"
    assert UMBRAL_SEVERIDAD_MODERADO == 2, "Umbral de 'Moderado' modificado"
    assert UMBRAL_DESEQUILIBRIO_INTERNO == 4, "Umbral de desequilibrio interno modificado"
    assert UMBRAL_DESEQUILIBRIO_EXTERNO == 8, "Umbral de desequilibrio externo modificado"


@pytest.mark.parametrize(
    "puntos_objetivo, severidad_esperada",
    [
        (0, "Bajo"),
        (UMBRAL_SEVERIDAD_MODERADO, "Moderado"),
        (UMBRAL_SEVERIDAD_CRITICO, "Crítico"),
    ],
)
def test_fronteras_de_severidad(puntos_objetivo, severidad_esperada):
    """
    Comprueba el comportamiento EN la frontera de cada umbral, construyendo casos con
    una puntuación de severidad conocida.

    - 0 puntos  -> caso equilibrado.
    - 2 puntos  -> un solo desequilibrio interno (PESO_DESEQUILIBRIO_INTERNO).
    - 5 puntos  -> un principio en 0 (1) + desequilibrio interno (2) + externo (2).
    """
    if puntos_objetivo == 0:
        _caso = caso()
    elif puntos_objetivo == UMBRAL_SEVERIDAD_MODERADO:
        _caso = caso(medico=perspectiva(3, autonomia=5, beneficencia=1))
    else:
        # medico: un principio en 0 (+1) y diferencia 5-0=5 >= 4 (+2)
        # totales: medico 5+0+3+3=11, comite 1 cada uno = 4 -> diff 7 < 8, no externo.
        # Añadimos comite bajo para forzar el externo (+2). Total = 1+2+2 = 5.
        _caso = caso(
            medico=perspectiva(3, autonomia=5, beneficencia=0),
            familia=perspectiva(3),
            comite=perspectiva(1),
        )

    _, _, severidad = verificar_sesgo_etico(_caso)
    assert severidad == severidad_esperada
