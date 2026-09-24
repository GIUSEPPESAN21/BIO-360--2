"""
core/usabilidad.py
==================
Validación con actores clínicos: System Usability Scale (SUS) y guía de entrevista.

La Fase 3 del proyecto DeliberIA (meses 5-9) valida el motor "con expertos en bioética
y actores clínicos mediante entrevistas semiestructuradas y pruebas de usabilidad".
Este módulo aporta el instrumento cuantitativo estándar para la usabilidad —la SUS de
Brooke (1996), 10 ítems Likert 1-5— con su puntuación e interpretación, y la guía de
entrevista semiestructurada para la parte cualitativa.

Puntuación SUS
--------------
Ítems impares (redactados en positivo): aporte = respuesta - 1.
Ítems pares (redactados en negativo):  aporte = 5 - respuesta.
Puntuación = suma de aportes × 2.5, en el rango 0-100.

Interpretación: escala de adjetivos de Bangor, Kortum y Miller (2009), aceptabilidad
de Bangor et al. (2008) y calificación por letras de Sauro y Lewis (2016).
El valor de referencia (media de cientos de estudios) es 68.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from core.modelos import safe_int

VERSION_INSTRUMENTO = "sus_es_v1"
REFERENCIA_SUS = 68.0

#: Ítems de la SUS en español (adaptación usual de Brooke, 1996).
ITEMS_SUS: tuple[str, ...] = (
    "Creo que me gustaría usar este sistema con frecuencia.",
    "Encontré el sistema innecesariamente complejo.",
    "Pensé que el sistema era fácil de usar.",
    "Creo que necesitaría el apoyo de un técnico para poder usar este sistema.",
    "Encontré que las diversas funciones del sistema estaban bien integradas.",
    "Pensé que había demasiada inconsistencia en el sistema.",
    "Imagino que la mayoría de las personas aprenderían a usar este sistema muy rápidamente.",
    "Encontré el sistema muy engorroso de usar.",
    "Me sentí muy seguro/a usando el sistema.",
    "Necesité aprender muchas cosas antes de poder empezar a usar este sistema.",
)

ESCALA_LIKERT = {
    1: "Totalmente en desacuerdo",
    2: "En desacuerdo",
    3: "Neutral",
    4: "De acuerdo",
    5: "Totalmente de acuerdo",
}

ROLES_EVALUADOR = (
    "Miembro del comité de ética",
    "Médico/a tratante",
    "Enfermería",
    "Experto/a en bioética",
    "Otro profesional de la salud",
    "Investigador/a",
)

#: Guía de entrevista semiestructurada (Fase 3). Agrupada por dimensión.
GUIA_ENTREVISTA: Dict[str, List[str]] = {
    "Utilidad para la deliberación": [
        "¿En qué momento del trabajo del comité le resultaría útil DeliberIA?",
        "¿El semáforo ético y los indicadores de consenso le ayudaron a identificar "
        "tensiones que no había advertido? Dé un ejemplo.",
    ],
    "Explicabilidad y confianza": [
        "¿Entendió por qué el caso obtuvo esa severidad? ¿Qué parte de la explicación "
        "fue más clara y cuál menos?",
        "¿Confiaría en el análisis deliberativo generado por IA? ¿Qué necesitaría para "
        "confiar más (o menos)?",
    ],
    "Equidad y sesgos": [
        "¿Percibe que la herramienta podría favorecer alguna perspectiva (equipo médico, "
        "familia, comité) sobre otra?",
        "¿Hay grupos de pacientes para los que el análisis le parezca menos adecuado?",
    ],
    "Integración en la práctica": [
        "¿Qué información de la historia clínica o de las actas del comité debería "
        "incorporar la herramienta y hoy falta?",
        "¿Qué barreras institucionales ve para adoptarla (tiempo, formación, "
        "infraestructura, normativa)?",
    ],
    "Cierre": [
        "Si pudiera cambiar una sola cosa de DeliberIA, ¿cuál sería?",
    ],
}


def puntuar_sus(respuestas: Sequence[Any]) -> float:
    """
    Puntuación SUS (0-100) de un cuestionario completo.

    Raises
    ------
    ValueError
        Si no hay exactamente 10 respuestas o alguna está fuera de 1-5.
    """
    if len(respuestas) != len(ITEMS_SUS):
        raise ValueError(f"La SUS requiere {len(ITEMS_SUS)} respuestas; se recibieron {len(respuestas)}.")
    valores = [safe_int(r, 0) for r in respuestas]
    if any(v < 1 or v > 5 for v in valores):
        raise ValueError("Todas las respuestas de la SUS deben estar entre 1 y 5.")
    total = 0
    for i, v in enumerate(valores):
        total += (v - 1) if i % 2 == 0 else (5 - v)
    return total * 2.5


def adjetivo_sus(puntuacion: float) -> str:
    """Escala de adjetivos de Bangor, Kortum y Miller (2009)."""
    if puntuacion >= 84.1:
        return "Excelente"
    if puntuacion >= 72.6:
        return "Buena"
    if puntuacion >= 51.7:
        return "Aceptable (OK)"
    if puntuacion >= 25.1:
        return "Pobre"
    return "La peor imaginable"


def aceptabilidad_sus(puntuacion: float) -> str:
    """Rangos de aceptabilidad de Bangor, Kortum y Miller (2008)."""
    if puntuacion >= 70:
        return "Aceptable"
    if puntuacion >= 50:
        return "Marginal"
    return "No aceptable"


def calificacion_sus(puntuacion: float) -> str:
    """Calificación por letras (curva de Sauro y Lewis, simplificada)."""
    if puntuacion >= 80.3:
        return "A"
    if puntuacion >= 74.0:
        return "B"
    if puntuacion >= 68.0:
        return "C"
    if puntuacion >= 51.0:
        return "D"
    return "F"


def interpretar_sus(puntuacion: float) -> Dict[str, Any]:
    return {
        "puntuacion": round(puntuacion, 2),
        "adjetivo": adjetivo_sus(puntuacion),
        "aceptabilidad": aceptabilidad_sus(puntuacion),
        "calificacion": calificacion_sus(puntuacion),
        "sobre_referencia": puntuacion >= REFERENCIA_SUS,
    }


def resumen_sus(puntuaciones: Sequence[float]) -> Dict[str, Any]:
    """Estadísticos de un conjunto de cuestionarios SUS (media, DE, IC95% aproximado)."""
    valores = [float(p) for p in puntuaciones]
    n = len(valores)
    if n == 0:
        return {"n": 0, "media": None, "desviacion": None, "ic95": None, "interpretacion": None}
    media = sum(valores) / n
    if n > 1:
        de = (sum((v - media) ** 2 for v in valores) / (n - 1)) ** 0.5
        margen = 1.96 * de / n ** 0.5
        ic = (round(max(0.0, media - margen), 2), round(min(100.0, media + margen), 2))
    else:
        de, ic = 0.0, None
    return {
        "n": n,
        "media": round(media, 2),
        "desviacion": round(de, 2),
        "ic95": ic,
        "interpretacion": interpretar_sus(media),
    }
