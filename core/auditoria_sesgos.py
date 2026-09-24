"""
core/auditoria_sesgos.py
========================
Protocolo formal de auditoría de sesgos algorítmicos de DeliberIA.

Qué se audita
-------------
La ficha técnica del proyecto compromete "un protocolo formal de auditoría de sesgos
algorítmicos (paridad entre subgrupos, calibración y explicabilidad)" sobre las
recomendaciones del motor (OE1, actividad 3). Este módulo implementa las pruebas
sobre la base de registros analíticos de `core.dataset`:

A. **Paridad entre subgrupos** (género, grupo de edad, dominio clínico, condición)
   del resultado "severidad elevada" (Moderado o Crítico):
   - tasa por subgrupo, diferencia de paridad demográfica (máx - mín),
   - razón de impacto dispar (mín / máx) con la regla de los cuatro quintos (0.8),
   - prueba χ² de independencia con su valor p.
B. **Paridad del consenso**: índice de consenso medio por subgrupo; una brecha amplia
   indica que la deliberación es sistemáticamente más conflictiva para un grupo.
C. **Calibración de la IA frente al motor determinista**: el riesgo que estima la IA
   en el análisis previo se compara con el semáforo (referencia auditable):
   exactitud, kappa de Cohen ponderado lineal, matriz de confusión y dirección del
   error (sobre/subestimación). También la concordancia del dilema sugerido por la
   IA con el elegido por el comité.
D. **Paridad de la calibración**: exactitud de la IA por subgrupo (igualdad de
   desempeño entre grupos).

Explicabilidad: el tercer pilar del protocolo lo aporta `core.etica.evaluar_semaforo`
(atribución de cada punto de severidad y contrafactuales), no este módulo.

Interpretación
--------------
El semáforo es determinista y NO recibe género, edad ni dominio: una disparidad en A
no nace del algoritmo sino de las PONDERACIONES que registran las perspectivas. Ese es
precisamente el hallazgo que interesa al comité (posible sesgo en la deliberación
humana que el sistema hace visible). En C y D, en cambio, sí se audita al modelo de IA.

Significancia práctica y estadística: una disparidad solo es "Alerta" cuando es a la
vez PRÁCTICA (regla de los cuatro quintos y diferencia ≥ 0.20) y ESTADÍSTICA (χ² con
p < 0.05). Si solo se cumple una de las dos, el veredicto es "Vigilar": con muestras
pequeñas, las tasas fluctúan mucho por azar y una alerta sería un falso positivo. Las
brechas sin prueba de significancia (consenso y exactitud por subgrupo) también se
reportan como "Vigilar".

Salvaguarda estadística: los subgrupos con menos de `N_MINIMO_SUBGRUPO` casos se
reportan pero se EXCLUYEN de las razones y brechas, y el veredicto pasa a "Datos
insuficientes" en lugar de emitir una alerta espuria.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.modelos import safe_str

VERSION_PROTOCOLO = "auditoria_sesgos_v1"

ATRIBUTOS_PROTEGIDOS: Tuple[str, ...] = ("genero", "grupo_edad", "dominio_clinico", "condicion")
ETIQUETAS_ATRIBUTO = {
    "genero": "Género",
    "grupo_edad": "Grupo de edad",
    "dominio_clinico": "Dominio clínico",
    "condicion": "Condición clínica",
}

NIVELES = ("Bajo", "Moderado", "Crítico")
RESULTADO_POSITIVO = ("Moderado", "Crítico")

N_MINIMO_SUBGRUPO = 5
UMBRAL_REGLA_CUATRO_QUINTOS = 0.8
UMBRAL_DIFERENCIA_PARIDAD = 0.20
UMBRAL_BRECHA_CONSENSO = 0.15
UMBRAL_KAPPA_ACEPTABLE = 0.40
ALFA = 0.05

VEREDICTO_OK = "Sin evidencia de disparidad"
VEREDICTO_ALERTA = "Alerta de disparidad"
VEREDICTO_VIGILAR = "Vigilar"
VEREDICTO_INSUFICIENTE = "Datos insuficientes"


# --- Estadística pura --------------------------------------------------------------

def _gamma_inferior_regularizada(a: float, x: float) -> float:
    """P(a, x) por serie (x < a+1) o fracción continua (Numerical Recipes, 6.2)."""
    if x <= 0:
        return 0.0
    ln_pre = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1:
        termino = suma = 1.0 / a
        ap = a
        for _ in range(500):
            ap += 1
            termino *= x / ap
            suma += termino
            if abs(termino) < abs(suma) * 1e-14:
                break
        return suma * math.exp(ln_pre)
    # Fracción continua de Lentz para Q(a, x)
    tiny = 1e-300
    b = x + 1 - a
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-14:
            break
    return 1.0 - math.exp(ln_pre) * h


def chi2_sf(x: float, gl: int) -> float:
    """Función de supervivencia de χ² (valor p) con `gl` grados de libertad."""
    if gl <= 0:
        return 1.0
    if x <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - _gamma_inferior_regularizada(gl / 2, x / 2)))


def chi2_independencia(tabla: List[List[int]]) -> Optional[Dict[str, float]]:
    """
    Prueba χ² de independencia sobre una tabla de contingencia (filas = subgrupos).
    Devuelve None si la tabla es degenerada (una sola fila o columna efectiva).
    """
    filas = [f for f in tabla if sum(f) > 0]
    if len(filas) < 2:
        return None
    n_col = len(filas[0])
    tot_col = [sum(f[j] for f in filas) for j in range(n_col)]
    columnas = [j for j in range(n_col) if tot_col[j] > 0]
    if len(columnas) < 2:
        return None
    total = sum(tot_col)
    estadistico = 0.0
    esperado_min = math.inf
    for f in filas:
        tot_fila = sum(f)
        for j in columnas:
            esperado = tot_fila * tot_col[j] / total
            esperado_min = min(esperado_min, esperado)
            estadistico += (f[j] - esperado) ** 2 / esperado
    gl = (len(filas) - 1) * (len(columnas) - 1)
    return {
        "chi2": round(estadistico, 4),
        "gl": gl,
        "p_valor": round(chi2_sf(estadistico, gl), 4),
        "esperado_minimo": round(esperado_min, 2),
        # Regla de Cochran: con esperados < 5 la aproximación χ² es poco fiable.
        "aproximacion_fiable": esperado_min >= 5,
    }


def kappa_cohen(pares: Sequence[Tuple[str, str]], categorias: Sequence[str] = NIVELES,
                ponderado: bool = True) -> Optional[float]:
    """
    Kappa de Cohen entre dos clasificaciones ordinales. Con `ponderado=True` usa pesos
    lineales (un error Bajo↔Crítico pesa más que Bajo↔Moderado).
    """
    pares = [(a, b) for a, b in pares if a in categorias and b in categorias]
    n = len(pares)
    if n == 0:
        return None
    k = len(categorias)
    idx = {c: i for i, c in enumerate(categorias)}
    obs = [[0.0] * k for _ in range(k)]
    for a, b in pares:
        obs[idx[a]][idx[b]] += 1
    marg_a = [sum(obs[i]) / n for i in range(k)]
    marg_b = [sum(obs[i][j] for i in range(k)) / n for j in range(k)]

    def peso(i: int, j: int) -> float:
        if ponderado:
            return 1 - abs(i - j) / (k - 1)
        return 1.0 if i == j else 0.0

    p_o = sum(peso(i, j) * obs[i][j] / n for i in range(k) for j in range(k))
    p_e = sum(peso(i, j) * marg_a[i] * marg_b[j] for i in range(k) for j in range(k))
    if p_e >= 1:
        return 1.0 if p_o >= 1 else 0.0
    return (p_o - p_e) / (1 - p_e)


def _media(valores: List[float]) -> Optional[float]:
    return sum(valores) / len(valores) if valores else None


def _num(valor: Any) -> Optional[float]:
    if valor in ("", None):
        return None
    try:
        return float(valor)
    except (TypeError, ValueError):
        return None


def _es_verdadero(valor: Any) -> Optional[bool]:
    if valor in ("", None):
        return None
    if isinstance(valor, bool):
        return valor
    return str(valor).strip().lower() in ("true", "1", "sí", "si")


# --- A. Paridad del resultado ------------------------------------------------------

def paridad_por_subgrupo(registros: List[Dict[str, Any]], atributo: str) -> Dict[str, Any]:
    grupos: Dict[str, Dict[str, int]] = {}
    for r in registros:
        g = safe_str(r.get(atributo)) or "Sin dato"
        sev = safe_str(r.get("severidad"))
        if sev not in NIVELES:
            continue
        cuenta = grupos.setdefault(g, {n: 0 for n in NIVELES})
        cuenta[sev] += 1

    detalle = []
    for g, cuenta in sorted(grupos.items()):
        n = sum(cuenta.values())
        positivos = sum(cuenta[n_] for n_ in RESULTADO_POSITIVO)
        detalle.append(
            {
                "subgrupo": g,
                "n": n,
                "positivos": positivos,
                "tasa_severidad_elevada": round(positivos / n, 4) if n else None,
                "distribucion": dict(cuenta),
                "suficiente": n >= N_MINIMO_SUBGRUPO,
            }
        )

    validos = [d for d in detalle if d["suficiente"]]
    salida: Dict[str, Any] = {
        "atributo": atributo,
        "etiqueta": ETIQUETAS_ATRIBUTO.get(atributo, atributo),
        "subgrupos": detalle,
        "diferencia_paridad": None,
        "razon_impacto_dispar": None,
        "chi2": None,
        "veredicto": VEREDICTO_INSUFICIENTE,
        "motivos": [],
    }
    if len(validos) < 2:
        salida["motivos"].append(
            f"Menos de dos subgrupos con n ≥ {N_MINIMO_SUBGRUPO}: no se comparan tasas."
        )
        return salida

    tasas = [d["tasa_severidad_elevada"] for d in validos]
    dif = max(tasas) - min(tasas)
    razon = (min(tasas) / max(tasas)) if max(tasas) > 0 else 1.0
    salida["diferencia_paridad"] = round(dif, 4)
    salida["razon_impacto_dispar"] = round(razon, 4)
    salida["chi2"] = chi2_independencia(
        [[d["positivos"], d["n"] - d["positivos"]] for d in validos]
    )

    practica = razon < UMBRAL_REGLA_CUATRO_QUINTOS and dif >= UMBRAL_DIFERENCIA_PARIDAD
    if practica:
        salida["motivos"].append(
            f"Razón de impacto {razon:.2f} < {UMBRAL_REGLA_CUATRO_QUINTOS} y diferencia de "
            f"paridad {dif:.2f} ≥ {UMBRAL_DIFERENCIA_PARIDAD}."
        )
    chi = salida["chi2"]
    estadistica = bool(chi and chi["p_valor"] < ALFA)
    if chi:
        nota = "" if chi["aproximacion_fiable"] else " (aproximación poco fiable: esperados < 5)"
        comparacion = "<" if estadistica else "≥"
        salida["motivos"].append(
            f"χ² = {chi['chi2']:.2f}, gl = {chi['gl']}, p = {chi['p_valor']:.3f} "
            f"{comparacion} {ALFA}{nota}."
        )
    if practica and estadistica:
        salida["veredicto"] = VEREDICTO_ALERTA
    elif practica or estadistica:
        salida["veredicto"] = VEREDICTO_VIGILAR
    else:
        salida["veredicto"] = VEREDICTO_OK
    return salida


# --- B. Paridad del consenso -------------------------------------------------------

def consenso_por_subgrupo(registros: List[Dict[str, Any]], atributo: str) -> Dict[str, Any]:
    grupos: Dict[str, List[float]] = {}
    for r in registros:
        v = _num(r.get("indice_consenso"))
        if v is None:
            continue
        grupos.setdefault(safe_str(r.get(atributo)) or "Sin dato", []).append(v)

    detalle = [
        {"subgrupo": g, "n": len(v), "consenso_medio": round(_media(v), 4),
         "suficiente": len(v) >= N_MINIMO_SUBGRUPO}
        for g, v in sorted(grupos.items())
    ]
    validos = [d for d in detalle if d["suficiente"]]
    salida = {
        "atributo": atributo,
        "etiqueta": ETIQUETAS_ATRIBUTO.get(atributo, atributo),
        "subgrupos": detalle,
        "brecha": None,
        "veredicto": VEREDICTO_INSUFICIENTE,
    }
    if len(validos) >= 2:
        medias = [d["consenso_medio"] for d in validos]
        brecha = max(medias) - min(medias)
        salida["brecha"] = round(brecha, 4)
        salida["veredicto"] = VEREDICTO_VIGILAR if brecha >= UMBRAL_BRECHA_CONSENSO else VEREDICTO_OK
    return salida


# --- C. Calibración de la IA -------------------------------------------------------

def calibracion_ia(registros: List[Dict[str, Any]]) -> Dict[str, Any]:
    pares = [
        (safe_str(r.get("nivel_riesgo_ia")), safe_str(r.get("severidad")))
        for r in registros
        if safe_str(r.get("nivel_riesgo_ia")) in NIVELES and safe_str(r.get("severidad")) in NIVELES
    ]
    n = len(pares)
    matriz = {ia: {m: 0 for m in NIVELES} for ia in NIVELES}
    sobre = sub = 0
    for ia, motor in pares:
        matriz[ia][motor] += 1
        diferencia = NIVELES.index(ia) - NIVELES.index(motor)
        sobre += diferencia > 0
        sub += diferencia < 0

    concordancias_dilema = [
        c for c in (_es_verdadero(r.get("concordancia_dilema_ia")) for r in registros) if c is not None
    ]

    salida: Dict[str, Any] = {
        "n_pares_riesgo": n,
        "matriz_confusion": matriz,  # filas: IA, columnas: motor determinista
        "exactitud": round(sum(matriz[c][c] for c in NIVELES) / n, 4) if n else None,
        "kappa_ponderado": None,
        "tasa_sobreestimacion": round(sobre / n, 4) if n else None,
        "tasa_subestimacion": round(sub / n, 4) if n else None,
        "n_dilemas_comparados": len(concordancias_dilema),
        "concordancia_dilema": (
            round(sum(concordancias_dilema) / len(concordancias_dilema), 4)
            if concordancias_dilema else None
        ),
        "veredicto": VEREDICTO_INSUFICIENTE,
        "motivos": [],
    }
    k = kappa_cohen(pares)
    salida["kappa_ponderado"] = round(k, 4) if k is not None else None

    if n < N_MINIMO_SUBGRUPO * 2:
        salida["motivos"].append(
            f"Solo {n} caso(s) con riesgo estimado por la IA: se requieren al menos "
            f"{N_MINIMO_SUBGRUPO * 2}."
        )
        return salida
    if k is not None and k < UMBRAL_KAPPA_ACEPTABLE:
        salida["motivos"].append(
            f"Kappa ponderado {k:.2f} < {UMBRAL_KAPPA_ACEPTABLE}: la IA no está calibrada "
            "con el motor determinista; su estimación de riesgo no debe mostrarse como guía."
        )
    if salida["tasa_sobreestimacion"] is not None and abs(
        salida["tasa_sobreestimacion"] - salida["tasa_subestimacion"]
    ) >= UMBRAL_DIFERENCIA_PARIDAD:
        direccion = (
            "sobreestima" if salida["tasa_sobreestimacion"] > salida["tasa_subestimacion"]
            else "subestima"
        )
        salida["motivos"].append(f"Error direccional: la IA {direccion} sistemáticamente el riesgo.")
    salida["veredicto"] = VEREDICTO_ALERTA if salida["motivos"] else VEREDICTO_OK
    return salida


# --- D. Paridad de la calibración --------------------------------------------------

def calibracion_por_subgrupo(registros: List[Dict[str, Any]], atributo: str) -> Dict[str, Any]:
    grupos: Dict[str, List[bool]] = {}
    for r in registros:
        c = _es_verdadero(r.get("concordancia_riesgo_ia"))
        if c is None:
            continue
        grupos.setdefault(safe_str(r.get(atributo)) or "Sin dato", []).append(c)
    detalle = [
        {"subgrupo": g, "n": len(v), "exactitud": round(sum(v) / len(v), 4),
         "suficiente": len(v) >= N_MINIMO_SUBGRUPO}
        for g, v in sorted(grupos.items())
    ]
    validos = [d for d in detalle if d["suficiente"]]
    salida = {
        "atributo": atributo,
        "etiqueta": ETIQUETAS_ATRIBUTO.get(atributo, atributo),
        "subgrupos": detalle,
        "brecha_exactitud": None,
        "veredicto": VEREDICTO_INSUFICIENTE,
    }
    if len(validos) >= 2:
        valores = [d["exactitud"] for d in validos]
        brecha = max(valores) - min(valores)
        salida["brecha_exactitud"] = round(brecha, 4)
        salida["veredicto"] = VEREDICTO_VIGILAR if brecha >= UMBRAL_DIFERENCIA_PARIDAD else VEREDICTO_OK
    return salida


# --- Auditoría completa ------------------------------------------------------------

def auditar(
    registros: List[Dict[str, Any]],
    atributos: Sequence[str] = ATRIBUTOS_PROTEGIDOS,
) -> Dict[str, Any]:
    """Ejecuta el protocolo completo y devuelve un informe serializable."""
    paridad = [paridad_por_subgrupo(registros, a) for a in atributos]
    consenso = [consenso_por_subgrupo(registros, a) for a in atributos]
    calibracion = calibracion_ia(registros)
    calibracion_sub = [calibracion_por_subgrupo(registros, a) for a in atributos]

    veredictos = (
        [p["veredicto"] for p in paridad]
        + [c["veredicto"] for c in consenso]
        + [calibracion["veredicto"]]
        + [c["veredicto"] for c in calibracion_sub]
    )
    alertas: List[str] = []
    vigilancias: List[str] = []

    def _anotar(veredicto: str, texto: str) -> None:
        if veredicto == VEREDICTO_ALERTA:
            alertas.append(texto)
        elif veredicto == VEREDICTO_VIGILAR:
            vigilancias.append(texto)

    for p in paridad:
        _anotar(
            p["veredicto"],
            f"Paridad de severidad por {p['etiqueta'].lower()}: " + " ".join(p["motivos"]),
        )
    for c in consenso:
        if c["brecha"] is not None:
            _anotar(
                c["veredicto"],
                f"Brecha de consenso por {c['etiqueta'].lower()}: {c['brecha']:.2f} "
                f"(umbral {UMBRAL_BRECHA_CONSENSO}).",
            )
    _anotar(calibracion["veredicto"], "Calibración IA vs. motor: " + " ".join(calibracion["motivos"]))
    for c in calibracion_sub:
        if c["brecha_exactitud"] is not None:
            _anotar(
                c["veredicto"],
                f"Desempeño desigual de la IA por {c['etiqueta'].lower()}: brecha de "
                f"exactitud {c['brecha_exactitud']:.2f}.",
            )

    if alertas:
        global_ = VEREDICTO_ALERTA
    elif vigilancias:
        global_ = VEREDICTO_VIGILAR
    elif all(v == VEREDICTO_INSUFICIENTE for v in veredictos):
        global_ = VEREDICTO_INSUFICIENTE
    else:
        global_ = VEREDICTO_OK

    return {
        "version_protocolo": VERSION_PROTOCOLO,
        "n_registros": len(registros),
        "parametros": {
            "n_minimo_subgrupo": N_MINIMO_SUBGRUPO,
            "regla_cuatro_quintos": UMBRAL_REGLA_CUATRO_QUINTOS,
            "umbral_diferencia_paridad": UMBRAL_DIFERENCIA_PARIDAD,
            "umbral_brecha_consenso": UMBRAL_BRECHA_CONSENSO,
            "umbral_kappa": UMBRAL_KAPPA_ACEPTABLE,
            "alfa": ALFA,
            "resultado_positivo": list(RESULTADO_POSITIVO),
        },
        "paridad_severidad": paridad,
        "paridad_consenso": consenso,
        "calibracion_ia": calibracion,
        "paridad_calibracion": calibracion_sub,
        "veredicto_global": global_,
        "alertas": alertas,
        "vigilancias": vigilancias,
    }
