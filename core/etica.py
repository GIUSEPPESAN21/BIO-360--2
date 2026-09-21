"""
core/etica.py
=============
Reglas deterministas del "semáforo ético" de BIOETHICARE 360.

Estas reglas NO usan IA: son determinísticas y auditables. La IA (Kiro/Gemini/OpenAI)
solo se usa para *explicar* el resultado, nunca para decidirlo (patrón usado también
en SaludIA). Por eso esta lógica está cubierta por pytest (M3): un cambio silencioso
en los umbrales alteraría diagnósticos clínicos.

Umbrales de severidad (deben permanecer estables salvo decisión explícita):
    puntos_severidad >= 5  -> "Crítico"
    puntos_severidad >= 2  -> "Moderado"
    en otro caso           -> "Bajo"
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# Puntos que aporta cada tipo de hallazgo a la severidad total
PESO_PERSPECTIVA_OMITIDA = 3
PESO_PRINCIPIO_OMITIDO = 1
PESO_DESEQUILIBRIO_INTERNO = 2
PESO_DESEQUILIBRIO_EXTERNO = 2

# Umbrales
UMBRAL_DESEQUILIBRIO_INTERNO = 4   # diferencia máx-mín dentro de una perspectiva
UMBRAL_DESEQUILIBRIO_EXTERNO = 8   # diferencia de totales entre perspectivas
UMBRAL_SEVERIDAD_CRITICO = 5
UMBRAL_SEVERIDAD_MODERADO = 2


def verificar_sesgo_etico(caso) -> Tuple[List[str], List[str], str]:
    """
    Evalúa las ponderaciones de un `CasoBioetico` y devuelve
    (advertencias, recomendaciones, severidad).

    `caso` debe exponer `.perspectivas: dict[str, dict[str, int]]`.
    """
    advertencias: List[str] = []
    recomendaciones: List[str] = []
    puntos_severidad = 0

    for nombre, valores in caso.perspectivas.items():
        if sum(valores.values()) == 0:
            advertencias.append(
                f"**Perspectiva Omitida:** La perspectiva de '{nombre}' no asignó puntuación a ningún principio."
            )
            recomendaciones.append(
                f"Se recomienda verificar si la ponderación de '{nombre}' fue omitida accidentalmente "
                f"para asegurar una deliberación completa."
            )
            puntos_severidad += PESO_PERSPECTIVA_OMITIDA

        for principio, valor in valores.items():
            if valor == 0:
                principio_legible = principio.replace("_", " ").capitalize()
                advertencias.append(
                    f"**Principio Omitido en '{nombre.title()}':** El principio de "
                    f"'{principio_legible}' tiene un valor de 0."
                )
                recomendaciones.append(
                    f"Evaluar si la omisión del principio de '{principio_legible}' en la "
                    f"perspectiva de '{nombre.title()}' es intencional y justificada."
                )
                puntos_severidad += PESO_PRINCIPIO_OMITIDO

        if sum(valores.values()) > 0:
            max_diff = max(valores.values()) - min(valores.values())
            if max_diff >= UMBRAL_DESEQUILIBRIO_INTERNO:
                advertencias.append(
                    f"**Alto Desequilibrio Interno:** En la perspectiva de '{nombre.title()}', "
                    f"existe un alto desequilibrio entre los principios (diferencia de {max_diff} puntos)."
                )
                recomendaciones.append(
                    "Se sugiere revisar si la alta disparidad en la ponderación de esta perspectiva "
                    "está suficientemente justificada o si requiere una deliberación más balanceada."
                )
                puntos_severidad += PESO_DESEQUILIBRIO_INTERNO

    puntajes_totales = {n: sum(v.values()) for n, v in caso.perspectivas.items()}
    if len(puntajes_totales) > 1:
        max_persp = max(puntajes_totales, key=puntajes_totales.get)
        min_persp = min(puntajes_totales, key=puntajes_totales.get)
        if puntajes_totales[max_persp] - puntajes_totales[min_persp] >= UMBRAL_DESEQUILIBRIO_EXTERNO:
            advertencias.append(
                f"**Alto Desequilibrio Externo:** La perspectiva de '{max_persp.title()}' tiene un peso "
                f"total significativamente mayor que la de '{min_persp.title()}'."
            )
            recomendaciones.append(
                "Analizar si esta dominancia de una perspectiva sobre otra es adecuada para el caso o si "
                "es necesario re-equilibrar las ponderaciones para una decisión más equitativa."
            )
            puntos_severidad += PESO_DESEQUILIBRIO_EXTERNO

    if puntos_severidad >= UMBRAL_SEVERIDAD_CRITICO:
        severidad = "Crítico"
    elif puntos_severidad >= UMBRAL_SEVERIDAD_MODERADO:
        severidad = "Moderado"
    else:
        severidad = "Bajo"

    return advertencias, recomendaciones, severidad
