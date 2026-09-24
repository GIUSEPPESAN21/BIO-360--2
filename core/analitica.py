"""
core/analitica.py
=================
Estadística descriptiva de la base de casos de DeliberIA (indicadores de desempeño).

Complementa a `core.auditoria_sesgos`: aquí se resume QUÉ muestra la base (severidad,
consenso, concordancia, tiempo de deliberación, validación experta) en total y por
dominio clínico; allí se evalúa si esos resultados son EQUITATIVOS entre subgrupos.

Opera sobre los registros analíticos de `core.dataset` (sin PII). Módulo puro.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.modelos import PRINCIPIOS, PRINCIPIOS_LABELS, safe_str

NIVELES = ("Bajo", "Moderado", "Crítico")
_ETIQUETA_PRINCIPIO = dict(zip(PRINCIPIOS, PRINCIPIOS_LABELS))
_CAMPOS_VALIDACION = ("val_utilidad", "val_pertinencia", "val_fundamentacion", "val_claridad")


def _num(valor: Any) -> Optional[float]:
    if valor in ("", None):
        return None
    try:
        return float(valor)
    except (TypeError, ValueError):
        return None


def _bool(valor: Any) -> Optional[bool]:
    if valor in ("", None):
        return None
    if isinstance(valor, bool):
        return valor
    return str(valor).strip().lower() in ("true", "1", "sí", "si")


def _estadisticos(valores: List[float]) -> Dict[str, Optional[float]]:
    if not valores:
        return {"n": 0, "media": None, "mediana": None, "desviacion": None, "min": None, "max": None}
    ordenados = sorted(valores)
    n = len(ordenados)
    media = sum(ordenados) / n
    mitad = n // 2
    mediana = ordenados[mitad] if n % 2 else (ordenados[mitad - 1] + ordenados[mitad]) / 2
    de = (sum((v - media) ** 2 for v in ordenados) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return {
        "n": n,
        "media": round(media, 4),
        "mediana": round(mediana, 4),
        "desviacion": round(de, 4),
        "min": round(ordenados[0], 4),
        "max": round(ordenados[-1], 4),
    }


def _proporcion(valores: List[Optional[bool]]) -> Optional[float]:
    validos = [v for v in valores if v is not None]
    return round(sum(validos) / len(validos), 4) if validos else None


def resumir(registros: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Resumen descriptivo de un conjunto de registros."""
    n = len(registros)
    severidad = {nivel: 0 for nivel in NIVELES}
    for r in registros:
        s = safe_str(r.get("severidad"))
        if s in severidad:
            severidad[s] += 1

    prioritarios: Dict[str, int] = {}
    for r in registros:
        p = safe_str(r.get("principio_prioritario"))
        if p:
            prioritarios[p] = prioritarios.get(p, 0) + 1

    medias_ponderacion = {}
    for p in PRINCIPIOS:
        valores = [
            _num(r.get(f"p_{persp}_{p}"))
            for r in registros
            for persp in ("medico", "familia", "comite")
        ]
        valores = [v for v in valores if v is not None]
        medias_ponderacion[p] = round(sum(valores) / len(valores), 4) if valores else None

    tiempos = [t for t in (_num(r.get("tiempo_deliberacion_s")) for r in registros) if t]

    validacion = {}
    for campo in _CAMPOS_VALIDACION:
        validacion[campo] = _estadisticos(
            [v for v in (_num(r.get(campo)) for r in registros) if v is not None]
        )

    return {
        "n": n,
        "distribucion_severidad": severidad,
        "proporcion_severidad_elevada": (
            round((severidad["Moderado"] + severidad["Crítico"]) / n, 4) if n else None
        ),
        "consenso": _estadisticos(
            [v for v in (_num(r.get("indice_consenso")) for r in registros) if v is not None]
        ),
        "w_kendall": _estadisticos(
            [v for v in (_num(r.get("w_kendall")) for r in registros) if v is not None]
        ),
        "tiempo_deliberacion_min": _estadisticos([t / 60 for t in tiempos]),
        "concordancia_dilema_ia": _proporcion([_bool(r.get("concordancia_dilema_ia")) for r in registros]),
        "concordancia_riesgo_ia": _proporcion([_bool(r.get("concordancia_riesgo_ia")) for r in registros]),
        "proporcion_deliberacion_ia": _proporcion(
            [_bool(r.get("deliberacion_ia_generada")) for r in registros]
        ),
        "principio_prioritario_frecuencia": {
            _ETIQUETA_PRINCIPIO.get(k, k): v
            for k, v in sorted(prioritarios.items(), key=lambda kv: -kv[1])
        },
        "ponderacion_media_por_principio": {
            _ETIQUETA_PRINCIPIO[p]: v for p, v in medias_ponderacion.items()
        },
        "validacion_experta": validacion,
        "proporcion_recomendacion_aceptable": _proporcion([_bool(r.get("val_aceptable")) for r in registros]),
    }


def resumir_por(registros: List[Dict[str, Any]], atributo: str = "dominio_clinico") -> Dict[str, Dict[str, Any]]:
    """Resumen descriptivo por cada valor de `atributo` (por defecto, dominio clínico)."""
    grupos: Dict[str, List[Dict[str, Any]]] = {}
    for r in registros:
        grupos.setdefault(safe_str(r.get(atributo)) or "Sin dato", []).append(r)
    return {g: resumir(rs) for g, rs in sorted(grupos.items())}
