"""
core/indicadores.py
===================
Indicadores cuantitativos de deliberación de DeliberIA (consenso, disenso,
concordancia y priorización de principios).

Contexto de investigación
-------------------------
El proyecto "DeliberIA: De Datos Clínicos Fragmentados a Decisiones Éticas Confiables"
(Convocatoria Jóvenes Investigadores UNIMINUTO 2026) compromete la cuantificación
rigurosa de los indicadores de consenso, disenso y desempeño del motor de deliberación
(OE1, actividad 3). Este módulo los calcula de forma DETERMINISTA y reproducible a
partir de la ponderación multiperspectiva (0-5) de los cuatro principios.

Indicadores
-----------
Todos se calculan sobre las perspectivas que ponderaron al menos un principio: una
perspectiva omitida (todo en cero) ya la señala el semáforo ético y, si se incluyera
aquí, fabricaría un "disenso" que en realidad es ausencia de dato.

- **Índice de consenso por principio** = 1 - σ / σ_max, donde σ es la desviación
  estándar poblacional de las ponderaciones de ese principio entre perspectivas y
  σ_max la máxima posible en la escala 0-5 para ese número de perspectivas. Vale 1
  con acuerdo total y 0 con la polarización máxima posible.
- **Índice de consenso global** = media de los índices por principio.
  **Índice de disenso** = 1 - consenso global.
- **W de Kendall** (con corrección por empates): concordancia entre perspectivas en
  el ORDEN de prioridad de los principios, independiente de la escala que use cada
  perspectiva. 0 = sin concordancia, 1 = concordancia total.
- **Acuerdo por pares** = 1 - (diferencia absoluta media / 5) entre dos perspectivas.
- **Priorización colectiva**: principios ordenados por su media entre perspectivas.
- **Puntos de disenso**: principios cuyo rango (máx - mín) entre perspectivas es
  mayor o igual a `UMBRAL_RANGO_DISENSO`.

Este módulo no depende de Streamlit, red ni numpy, y está cubierto por pytest.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Dict, List, Optional

from core.modelos import PRINCIPIOS, PRINCIPIOS_LABELS, safe_int

VERSION_INDICADORES = "indicadores_v1"

ESCALA_MIN = 0
ESCALA_MAX = 5

#: Rango entre perspectivas a partir del cual un principio se marca como punto de disenso.
UMBRAL_RANGO_DISENSO = 3

#: Cortes de clasificación del índice de consenso global.
CORTE_CONSENSO_ALTO = 0.80
CORTE_CONSENSO_MODERADO = 0.60

_ETIQUETA_PRINCIPIO = dict(zip(PRINCIPIOS, PRINCIPIOS_LABELS))


# --- Utilidades estadísticas puras -----------------------------------------------

def _media(valores: List[float]) -> float:
    return sum(valores) / len(valores) if valores else 0.0


def _desviacion_poblacional(valores: List[float]) -> float:
    if not valores:
        return 0.0
    m = _media(valores)
    return math.sqrt(sum((v - m) ** 2 for v in valores) / len(valores))


def desviacion_maxima(n: int, minimo: float = ESCALA_MIN, maximo: float = ESCALA_MAX) -> float:
    """
    Desviación estándar poblacional MÁXIMA alcanzable por `n` valores en [minimo, maximo].

    Se alcanza con los valores en los extremos: k en el máximo y n-k en el mínimo, con
    σ = (maximo - minimo) · sqrt(p(1-p)), p = k/n. Se busca el k que la maximiza.
    """
    if n < 2:
        return 0.0
    amplitud = maximo - minimo
    return max(amplitud * math.sqrt((k / n) * (1 - k / n)) for k in range(1, n))


def rangos_con_empates(valores: List[float]) -> List[float]:
    """
    Rangos (1 = menor) con promedio para los empates, como en la W de Kendall.

    >>> rangos_con_empates([3, 5, 3, 1])
    [2.5, 4.0, 2.5, 1.0]
    """
    orden = sorted(range(len(valores)), key=lambda i: valores[i])
    rangos = [0.0] * len(valores)
    i = 0
    while i < len(orden):
        j = i
        while j + 1 < len(orden) and valores[orden[j + 1]] == valores[orden[i]]:
            j += 1
        promedio = (i + j) / 2 + 1
        for k in range(i, j + 1):
            rangos[orden[k]] = promedio
        i = j + 1
    return rangos


def w_de_kendall(matriz: List[List[float]]) -> Optional[float]:
    """
    Coeficiente de concordancia W de Kendall con corrección por empates.

    Parameters
    ----------
    matriz:
        Una fila por evaluador (perspectiva) y una columna por ítem (principio).

    Returns
    -------
    float en [0, 1], o None si no está definido (menos de 2 evaluadores, menos de 2
    ítems, o todos los evaluadores empataron todos los ítems).
    """
    m = len(matriz)
    if m < 2:
        return None
    n = len(matriz[0])
    if n < 2 or any(len(fila) != n for fila in matriz):
        return None

    rangos = [rangos_con_empates(fila) for fila in matriz]
    sumas = [sum(r[i] for r in rangos) for i in range(n)]
    media_r = m * (n + 1) / 2
    s = sum((r - media_r) ** 2 for r in sumas)

    correccion = 0.0
    for fila in matriz:
        conteo: Dict[float, int] = {}
        for v in fila:
            conteo[v] = conteo.get(v, 0) + 1
        correccion += sum(t ** 3 - t for t in conteo.values())

    denominador = m ** 2 * (n ** 3 - n) - m * correccion
    if denominador <= 0:
        return None
    return max(0.0, min(1.0, 12 * s / denominador))


# --- Normalización de la entrada ---------------------------------------------------

def _perspectivas(fuente: Any) -> Dict[str, Dict[str, int]]:
    """Acepta un CasoBioetico, un dict de perspectivas o el reporte persistido."""
    if fuente is None:
        return {}
    if isinstance(fuente, dict) and "AnalisisMultiperspectiva" in fuente:
        fuente = fuente.get("AnalisisMultiperspectiva")
    datos = getattr(fuente, "perspectivas", fuente)
    if not isinstance(datos, dict):
        return {}
    salida: Dict[str, Dict[str, int]] = {}
    for nombre, valores in datos.items():
        if isinstance(valores, dict):
            salida[str(nombre)] = {p: safe_int(valores.get(p)) for p in PRINCIPIOS}
    return salida


# --- Clasificaciones --------------------------------------------------------------

def clasificar_consenso(indice: Optional[float]) -> str:
    if indice is None:
        return "No calculable"
    if indice >= CORTE_CONSENSO_ALTO:
        return "Consenso alto"
    if indice >= CORTE_CONSENSO_MODERADO:
        return "Consenso moderado"
    return "Disenso"


def interpretar_w(w: Optional[float]) -> str:
    """Interpretación convencional de la W de Kendall."""
    if w is None:
        return "No calculable"
    if w >= 0.7:
        return "Concordancia fuerte"
    if w >= 0.5:
        return "Concordancia moderada"
    if w >= 0.3:
        return "Concordancia débil"
    return "Sin concordancia apreciable"


# --- API pública ------------------------------------------------------------------

def calcular_indicadores(fuente: Any) -> Dict[str, Any]:
    """
    Calcula todos los indicadores de deliberación de un caso.

    Returns
    -------
    dict serializable (apto para Firestore, el PDF y la base de datos FAIR).
    """
    todas = _perspectivas(fuente)
    activas = {n: v for n, v in todas.items() if sum(v.values()) > 0}
    omitidas = [n for n in todas if n not in activas]

    resultado: Dict[str, Any] = {
        "version": VERSION_INDICADORES,
        "perspectivas_consideradas": list(activas),
        "perspectivas_omitidas": omitidas,
        "por_principio": {},
        "indice_consenso": None,
        "indice_disenso": None,
        "clasificacion_consenso": "No calculable",
        "w_kendall": None,
        "interpretacion_w": "No calculable",
        "acuerdo_por_pares": {},
        "priorizacion_colectiva": [],
        "puntos_de_disenso": [],
    }

    if not activas:
        return resultado

    n = len(activas)
    sigma_max = desviacion_maxima(n)
    indices: List[float] = []

    for p in PRINCIPIOS:
        valores = [activas[nombre][p] for nombre in activas]
        sigma = _desviacion_poblacional(valores)
        rango = max(valores) - min(valores)
        indice = (1 - sigma / sigma_max) if sigma_max > 0 else None
        if indice is not None:
            indices.append(indice)
        resultado["por_principio"][p] = {
            "etiqueta": _ETIQUETA_PRINCIPIO[p],
            "media": round(_media(valores), 4),
            "desviacion": round(sigma, 4),
            "rango": rango,
            "indice_consenso": round(indice, 4) if indice is not None else None,
        }
        if n >= 2 and rango >= UMBRAL_RANGO_DISENSO:
            resultado["puntos_de_disenso"].append(p)

    if indices:
        global_ = _media(indices)
        resultado["indice_consenso"] = round(global_, 4)
        resultado["indice_disenso"] = round(1 - global_, 4)
        resultado["clasificacion_consenso"] = clasificar_consenso(global_)

    w = w_de_kendall([[activas[nombre][p] for p in PRINCIPIOS] for nombre in activas])
    resultado["w_kendall"] = round(w, 4) if w is not None else None
    resultado["interpretacion_w"] = interpretar_w(w)

    for a, b in combinations(activas, 2):
        dif = _media([abs(activas[a][p] - activas[b][p]) for p in PRINCIPIOS])
        resultado["acuerdo_por_pares"][f"{a} ↔ {b}"] = round(1 - dif / ESCALA_MAX, 4)

    resultado["priorizacion_colectiva"] = sorted(
        PRINCIPIOS, key=lambda p: (-resultado["por_principio"][p]["media"], PRINCIPIOS.index(p))
    )
    return resultado


def resumen_indicadores_texto(indicadores: Dict[str, Any]) -> List[str]:
    """Frases legibles (para la UI y el PDF) a partir de `calcular_indicadores`."""
    if not indicadores or indicadores.get("indice_consenso") is None:
        return ["No hay suficientes perspectivas ponderadas para calcular el consenso."]

    lineas = [
        f"Índice de consenso global: {indicadores['indice_consenso']:.2f} "
        f"({indicadores['clasificacion_consenso']}); índice de disenso: "
        f"{indicadores['indice_disenso']:.2f}.",
    ]
    if indicadores.get("w_kendall") is not None:
        lineas.append(
            f"W de Kendall (concordancia en la priorización de principios): "
            f"{indicadores['w_kendall']:.2f} — {indicadores['interpretacion_w']}."
        )
    prior = indicadores.get("priorizacion_colectiva") or []
    if prior:
        lineas.append(
            "Priorización colectiva: "
            + " > ".join(_ETIQUETA_PRINCIPIO.get(p, p) for p in prior)
            + "."
        )
    disenso = indicadores.get("puntos_de_disenso") or []
    if disenso:
        lineas.append(
            "Puntos de disenso (rango ≥ "
            f"{UMBRAL_RANGO_DISENSO} entre perspectivas): "
            + ", ".join(_ETIQUETA_PRINCIPIO.get(p, p) for p in disenso)
            + "."
        )
    if indicadores.get("perspectivas_omitidas"):
        lineas.append(
            "Perspectivas excluidas del cálculo por no haber ponderado: "
            + ", ".join(indicadores["perspectivas_omitidas"])
            + "."
        )
    return lineas
