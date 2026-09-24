"""
core/dataset.py
===============
Base de datos clínico-ética estructurada, seudonimizada y documentada (DeliberIA).

Producto comprometido
---------------------
La ficha técnica del proyecto DeliberIA compromete como Producto 3 una "base de datos
clínico-ética estructurada, depurada y documentada (casos, principios, indicadores)",
documentada con metadatos bajo principios FAIR y con potencial de apertura como
recurso de ciencia abierta. Este módulo convierte cada reporte deliberativo en un
REGISTRO ANALÍTICO plano y sin PII, y construye el paquete reproducible:

    casos.csv / casos.json      registros (una fila por caso)
    diccionario_datos.csv       definición, tipo y dominio de cada variable
    metadatos.json              metadatos FAIR (JSON-LD, vocabulario schema.org)
    calidad.json                informe de control de calidad y k-anonimato
    LEEME.txt                   descripción del paquete y condiciones de uso

Garantías de privacidad
-----------------------
- Ningún campo de texto libre (descripción, contexto, chat, análisis de IA) entra en
  el registro: el texto clínico es la principal fuente de reidentificación.
- El ID del caso (a menudo el Nº de historia clínica) se sustituye por un seudónimo
  HMAC-SHA256 con clave secreta. Sin la clave no es reversible ni enlazable.
- La edad se generaliza a grupos y la fecha a periodo (AAAA-MM).
- `evaluar_calidad` mide el k-anonimato sobre los cuasi-identificadores y marca los
  grupos con k < `K_ANONIMATO_MINIMO` antes de cualquier apertura de datos.

Módulo puro (sin Streamlit, red ni pandas): testeable en aislamiento.
"""

from __future__ import annotations

import csv
import hashlib
import hmac
import io
import json
import random
import re
import zipfile
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from core.etica import ETIQUETAS_TIPO, VERSION_MOTOR, evaluar_semaforo
from core.indicadores import VERSION_INDICADORES, calcular_indicadores
from core.modelos import (
    CONDICIONES,
    DOMINIOS_CLINICOS,
    GENEROS,
    PERSPECTIVAS_NOMBRES,
    PRINCIPIOS,
    safe_int,
    safe_str,
)

VERSION_DATASET = "1.0.0"

#: Clave por defecto para el seudónimo. SOLO para demostración: en producción debe
#: definirse `DATASET_PSEUDONYM_KEY` en los secrets; los metadatos lo advierten.
CLAVE_SEUDONIMO_POR_DEFECTO = "deliberia-clave-de-demostracion-no-usar-en-produccion"

#: Tamaño mínimo aceptable de cada clase de equivalencia de cuasi-identificadores.
K_ANONIMATO_MINIMO = 5

CUASI_IDENTIFICADORES: Tuple[str, ...] = ("grupo_edad", "genero", "dominio_clinico", "condicion")

ORIGEN_REGISTRADO = "registrado"
ORIGEN_SINTETICO = "sintetico"

GRUPOS_EDAD = ("Neonato", "0-17", "18-39", "40-64", "65+", "Sin dato")
NIVELES_SEVERIDAD = ("Bajo", "Moderado", "Crítico")

_NOMBRE_LARGO_A_CORTO = {v: k for k, v in PERSPECTIVAS_NOMBRES.items()}


# --- Diccionario de datos ----------------------------------------------------------
# (variable, tipo, descripción, dominio de valores). Es la fuente única: de aquí salen
# el orden de columnas del CSV, el diccionario publicado y los metadatos FAIR.

def _columnas_ponderacion() -> List[Tuple[str, str, str, str]]:
    columnas = []
    for corta, larga in PERSPECTIVAS_NOMBRES.items():
        for p in PRINCIPIOS:
            columnas.append(
                (
                    f"p_{corta}_{p}",
                    "entero",
                    f"Ponderación del principio '{p}' asignada por la perspectiva '{larga}'.",
                    "0-5",
                )
            )
    return columnas


DICCIONARIO_DATOS: List[Tuple[str, str, str, str]] = [
    ("id_seudonimo", "texto", "Seudónimo HMAC-SHA256 (16 hex) del ID del caso. No reversible sin la clave.", "hex[16]"),
    ("origen", "categórica", "Procedencia del registro.", f"{ORIGEN_REGISTRADO} | {ORIGEN_SINTETICO}"),
    ("schema_version", "entero", "Versión del esquema del reporte de origen.", "1, 2, 3"),
    ("periodo", "texto", "Año y mes del análisis (fecha generalizada).", "AAAA-MM"),
    ("dominio_clinico", "categórica", "Dominio clínico del caso.", " | ".join(DOMINIOS_CLINICOS)),
    ("dilema_seleccionado", "categórica", "Dilema ético principal elegido por el comité.", "Catálogo de dilemas.json"),
    ("dilema_sugerido_ia", "categórica", "Dilema sugerido por la IA en el análisis previo (vacío si no se usó).", "Catálogo de dilemas.json"),
    ("concordancia_dilema_ia", "booleana", "La IA sugirió el mismo dilema que eligió el comité.", "true | false | vacío"),
    ("condicion", "categórica", "Condición clínica del paciente.", " | ".join(CONDICIONES)),
    ("genero", "categórica", "Género registrado.", " | ".join(GENEROS) + " | N/A"),
    ("grupo_edad", "categórica", "Edad generalizada en grupos.", " | ".join(GRUPOS_EDAD)),
    *_columnas_ponderacion(),
    ("severidad", "ordinal", "Semáforo ético determinista.", " < ".join(NIVELES_SEVERIDAD)),
    ("puntos_severidad", "entero", "Suma de las contribuciones de los hallazgos del semáforo.", "≥ 0"),
    ("n_hallazgos", "entero", "Número de hallazgos del semáforo.", "≥ 0"),
    *[
        (f"n_{tipo}", "entero", f"Hallazgos de tipo '{etiqueta}'.", "≥ 0")
        for tipo, etiqueta in ETIQUETAS_TIPO.items()
    ],
    ("indice_consenso", "real", "Índice de consenso global entre perspectivas (1 - σ/σmax).", "0-1"),
    ("indice_disenso", "real", "1 - índice de consenso.", "0-1"),
    ("w_kendall", "real", "W de Kendall: concordancia en la priorización de principios.", "0-1"),
    ("n_puntos_disenso", "entero", "Principios con rango entre perspectivas ≥ 3.", "0-4"),
    ("principio_prioritario", "categórica", "Principio con mayor ponderación media.", " | ".join(PRINCIPIOS)),
    ("nivel_riesgo_ia", "ordinal", "Nivel de riesgo estimado por la IA en el análisis previo.", " | ".join(NIVELES_SEVERIDAD) + " | vacío"),
    ("concordancia_riesgo_ia", "booleana", "El riesgo estimado por la IA coincide con el semáforo determinista.", "true | false | vacío"),
    ("tiempo_deliberacion_s", "entero", "Segundos entre el inicio del registro del caso y su envío (0 = no medido).", "≥ 0"),
    ("n_turnos_chat", "entero", "Preguntas formuladas al asistente de deliberación.", "≥ 0"),
    ("deliberacion_ia_generada", "booleana", "Se generó el análisis deliberativo por IA.", "true | false"),
    ("proveedor_ia", "categórica", "Proveedor del modelo usado en la deliberación.", "Google Gemini | OpenAI | Kiro | vacío"),
    ("modelo_ia", "texto", "Identificador exacto del modelo.", "texto"),
    ("prompt_version", "texto", "Versión del prompt de deliberación.", "texto"),
    ("val_utilidad", "entero", "Validación experta: utilidad del análisis para la deliberación.", "1-5 | vacío"),
    ("val_pertinencia", "entero", "Validación experta: pertinencia clínica y ética.", "1-5 | vacío"),
    ("val_fundamentacion", "entero", "Validación experta: fundamentación normativa.", "1-5 | vacío"),
    ("val_claridad", "entero", "Validación experta: claridad de la explicación.", "1-5 | vacío"),
    ("val_aceptable", "booleana", "El experto considera aceptable la recomendación.", "true | false | vacío"),
    ("version_motor", "texto", "Versión del motor de reglas del semáforo.", "texto"),
    ("version_indicadores", "texto", "Versión del cálculo de indicadores.", "texto"),
]

COLUMNAS: List[str] = [c[0] for c in DICCIONARIO_DATOS]


# --- Seudonimización y generalización ----------------------------------------------

def seudonimizar(caso_id: str, clave: Optional[str] = None) -> str:
    """Seudónimo HMAC-SHA256 (16 hex) del ID del caso. Estable para la misma clave."""
    clave_bytes = (clave or CLAVE_SEUDONIMO_POR_DEFECTO).encode("utf-8")
    return hmac.new(clave_bytes, safe_str(caso_id).encode("utf-8"), hashlib.sha256).hexdigest()[:16]


def grupo_edad(edad: Any, semanas_gestacion: Any = 0, condicion: str = "") -> str:
    if safe_int(semanas_gestacion) > 0 or condicion == "Neonato":
        return "Neonato"
    e = safe_int(edad, -1)
    if e <= 0:
        return "Sin dato"
    if e < 18:
        return "0-17"
    if e < 40:
        return "18-39"
    if e < 65:
        return "40-64"
    return "65+"


def _periodo(fecha: Any) -> str:
    texto = safe_str(fecha)
    m = re.match(r"(\d{4})-(\d{2})", texto)
    return f"{m.group(1)}-{m.group(2)}" if m else ""


_RE_RESUMEN = re.compile(
    r"(?P<edad>\d+)\s+años,\s+género\s+(?P<genero>[^,]+),\s+condición\s+(?P<condicion>[^.]+)\.",
    flags=re.IGNORECASE,
)


def datos_estructurados_de_reporte(reporte: Dict[str, Any]) -> Dict[str, Any]:
    """
    Variables estructuradas del paciente. Los reportes de esquema ≥ 3 las guardan en
    `DatosEstructurados`; para los antiguos se recuperan del "Resumen del Paciente"
    (que tiene un formato fijo generado por el sistema), sin leer el nombre.
    """
    datos = reporte.get("DatosEstructurados")
    if isinstance(datos, dict) and datos:
        return dict(datos)

    salida: Dict[str, Any] = {
        "edad": 0,
        "genero": "N/A",
        "condicion": "",
        "semanas_gestacion": 0,
        "dominio_clinico": "Otro",
    }
    m = _RE_RESUMEN.search(safe_str(reporte.get("Resumen del Paciente")))
    if m:
        salida["edad"] = safe_int(m.group("edad"))
        salida["genero"] = m.group("genero").strip()
        salida["condicion"] = m.group("condicion").strip()
    sem = re.search(r"Neonato de (\d+) sem", safe_str(reporte.get("Resumen del Paciente")))
    if sem:
        salida["semanas_gestacion"] = safe_int(sem.group(1))
    return salida


def _perspectivas_cortas(reporte: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    crudo = reporte.get("AnalisisMultiperspectiva") or {}
    salida: Dict[str, Dict[str, int]] = {}
    if isinstance(crudo, dict):
        for nombre, valores in crudo.items():
            corta = nombre if nombre in PERSPECTIVAS_NOMBRES else _NOMBRE_LARGO_A_CORTO.get(nombre)
            if corta and isinstance(valores, dict):
                salida[corta] = {p: safe_int(valores.get(p)) for p in PRINCIPIOS}
    return salida


class _CasoPonderaciones:
    """Adaptador mínimo para reutilizar el motor sobre ponderaciones persistidas."""

    def __init__(self, perspectivas: Dict[str, Dict[str, int]]) -> None:
        self.perspectivas = perspectivas


def _bool_o_vacio(valor: Optional[bool]) -> Any:
    return "" if valor is None else bool(valor)


# --- Registro analítico ------------------------------------------------------------

def registro_desde_reporte(
    reporte: Dict[str, Any],
    clave_seudonimo: Optional[str] = None,
    origen: str = ORIGEN_REGISTRADO,
) -> Dict[str, Any]:
    """
    Convierte un reporte deliberativo (tal como se persiste en Firestore) en un
    registro analítico plano y SIN PII, con exactamente las columnas de `COLUMNAS`.

    El semáforo y los indicadores se RECALCULAN desde las ponderaciones con el motor
    vigente: así todos los registros son comparables aunque procedan de versiones
    distintas de la aplicación.
    """
    reporte = reporte or {}
    datos = datos_estructurados_de_reporte(reporte)
    perspectivas = _perspectivas_cortas(reporte)
    for corta in PERSPECTIVAS_NOMBRES:
        perspectivas.setdefault(corta, {p: 0 for p in PRINCIPIOS})

    semaforo = evaluar_semaforo(_CasoPonderaciones(perspectivas))
    indicadores = calcular_indicadores(perspectivas)
    por_tipo: Dict[str, int] = {}
    for h in semaforo.hallazgos:
        por_tipo[h.tipo] = por_tipo.get(h.tipo, 0) + 1

    seleccionado = safe_str(reporte.get("Dilema Ético Principal (Seleccionado)"))
    sugerido = safe_str(reporte.get("Dilema Sugerido por IA"))
    estructurado = reporte.get("Análisis Estructurado (IA)") or {}
    riesgo_ia = safe_str(estructurado.get("nivel_riesgo")) if isinstance(estructurado, dict) else ""
    if riesgo_ia not in NIVELES_SEVERIDAD:
        riesgo_ia = ""
    traza = reporte.get("Trazabilidad IA") or {}
    validacion = reporte.get("ValidacionExperta") or {}
    chat = reporte.get("Historial del Chat de Deliberación") or []

    registro: Dict[str, Any] = {
        "id_seudonimo": seudonimizar(reporte.get("ID del Caso", ""), clave_seudonimo),
        "origen": origen,
        "schema_version": safe_int(reporte.get("schema_version"), 1),
        "periodo": _periodo(reporte.get("Fecha Análisis (UTC)") or reporte.get("Fecha Análisis")),
        "dominio_clinico": safe_str(
            reporte.get("Dominio Clínico") or datos.get("dominio_clinico"), "Otro"
        ),
        "dilema_seleccionado": seleccionado,
        "dilema_sugerido_ia": sugerido,
        "concordancia_dilema_ia": _bool_o_vacio(
            (sugerido == seleccionado) if (sugerido and seleccionado) else None
        ),
        "condicion": safe_str(datos.get("condicion")),
        "genero": safe_str(datos.get("genero"), "N/A"),
        "grupo_edad": grupo_edad(
            datos.get("edad"), datos.get("semanas_gestacion"), safe_str(datos.get("condicion"))
        ),
        "severidad": semaforo.severidad,
        "puntos_severidad": semaforo.puntos_severidad,
        "n_hallazgos": len(semaforo.hallazgos),
        "indice_consenso": _vacio_si_none(indicadores["indice_consenso"]),
        "indice_disenso": _vacio_si_none(indicadores["indice_disenso"]),
        "w_kendall": _vacio_si_none(indicadores["w_kendall"]),
        "n_puntos_disenso": len(indicadores["puntos_de_disenso"]),
        "principio_prioritario": (indicadores["priorizacion_colectiva"] or [""])[0],
        "nivel_riesgo_ia": riesgo_ia,
        "concordancia_riesgo_ia": _bool_o_vacio(
            (riesgo_ia == semaforo.severidad) if riesgo_ia else None
        ),
        "tiempo_deliberacion_s": safe_int(reporte.get("Tiempo de Deliberación (s)")),
        "n_turnos_chat": sum(
            1 for m in chat if isinstance(m, dict) and m.get("role") == "user"
        ),
        "deliberacion_ia_generada": bool(safe_str(reporte.get("Análisis Deliberativo (IA)"))),
        "proveedor_ia": safe_str(traza.get("proveedor")) if isinstance(traza, dict) else "",
        "modelo_ia": safe_str(traza.get("modelo")) if isinstance(traza, dict) else "",
        "prompt_version": safe_str(traza.get("prompt_version")) if isinstance(traza, dict) else "",
        "val_utilidad": _likert(validacion, "utilidad"),
        "val_pertinencia": _likert(validacion, "pertinencia"),
        "val_fundamentacion": _likert(validacion, "fundamentacion"),
        "val_claridad": _likert(validacion, "claridad"),
        "val_aceptable": _bool_o_vacio(
            validacion.get("aceptable") if isinstance(validacion, dict) and "aceptable" in validacion else None
        ),
        "version_motor": VERSION_MOTOR,
        "version_indicadores": VERSION_INDICADORES,
    }
    for corta in PERSPECTIVAS_NOMBRES:
        for p in PRINCIPIOS:
            registro[f"p_{corta}_{p}"] = perspectivas[corta][p]
    for tipo in ETIQUETAS_TIPO:
        registro[f"n_{tipo}"] = por_tipo.get(tipo, 0)

    return {c: registro.get(c, "") for c in COLUMNAS}


def _vacio_si_none(valor: Any) -> Any:
    return "" if valor is None else valor


def _likert(validacion: Any, clave: str) -> Any:
    if not isinstance(validacion, dict):
        return ""
    v = safe_int(validacion.get(clave), 0)
    return v if 1 <= v <= 5 else ""


def construir_registros(
    reportes: Iterable[Dict[str, Any]],
    clave_seudonimo: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Convierte varios reportes en registros, descartando los que no son dict."""
    return [
        registro_desde_reporte(r, clave_seudonimo)
        for r in reportes
        if isinstance(r, dict)
    ]


# --- Control de calidad y k-anonimato ----------------------------------------------

def evaluar_calidad(registros: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Informe de calidad de la base: completitud por variable, duplicados, casos con
    perspectivas omitidas y k-anonimato sobre los cuasi-identificadores.
    """
    n = len(registros)
    informe: Dict[str, Any] = {
        "n_registros": n,
        "completitud": {},
        "duplicados": 0,
        "registros_con_perspectiva_omitida": 0,
        "k_anonimato": None,
        "clases_en_riesgo": [],
        "k_minimo_requerido": K_ANONIMATO_MINIMO,
        "apto_para_apertura": False,
        "observaciones": [],
    }
    if n == 0:
        informe["observaciones"].append("La base está vacía.")
        return informe

    for col in COLUMNAS:
        llenos = sum(1 for r in registros if r.get(col) not in ("", None))
        informe["completitud"][col] = round(llenos / n, 4)

    ids = [r.get("id_seudonimo") for r in registros]
    informe["duplicados"] = len(ids) - len(set(ids))
    informe["registros_con_perspectiva_omitida"] = sum(
        1 for r in registros if safe_int(r.get("n_perspectiva_omitida")) > 0
    )

    clases: Dict[Tuple[str, ...], int] = {}
    for r in registros:
        clave = tuple(safe_str(r.get(q)) for q in CUASI_IDENTIFICADORES)
        clases[clave] = clases.get(clave, 0) + 1
    informe["k_anonimato"] = min(clases.values())
    informe["clases_en_riesgo"] = [
        {**dict(zip(CUASI_IDENTIFICADORES, clave)), "n": cuenta}
        for clave, cuenta in sorted(clases.items(), key=lambda kv: kv[1])
        if cuenta < K_ANONIMATO_MINIMO
    ]

    if informe["duplicados"]:
        informe["observaciones"].append(
            f"Hay {informe['duplicados']} seudónimo(s) duplicado(s): revisar casos guardados dos veces."
        )
    if informe["registros_con_perspectiva_omitida"]:
        informe["observaciones"].append(
            f"{informe['registros_con_perspectiva_omitida']} registro(s) con alguna perspectiva "
            "sin ponderar: su consenso se calcula solo con las perspectivas presentes."
        )
    if informe["clases_en_riesgo"]:
        informe["observaciones"].append(
            f"k-anonimato = {informe['k_anonimato']} (< {K_ANONIMATO_MINIMO}): "
            f"{len(informe['clases_en_riesgo'])} combinación(es) de cuasi-identificadores "
            "con muy pocos casos. Generalizar o suprimir antes de publicar."
        )
    informe["apto_para_apertura"] = (
        not informe["clases_en_riesgo"] and informe["duplicados"] == 0
    )
    return informe


# --- Metadatos FAIR ----------------------------------------------------------------

def metadatos_fair(
    registros: List[Dict[str, Any]],
    calidad: Optional[Dict[str, Any]] = None,
    clave_por_defecto: bool = False,
) -> Dict[str, Any]:
    """
    Metadatos del conjunto de datos en JSON-LD (vocabulario schema.org/Dataset),
    siguiendo los principios FAIR (Localizable, Accesible, Interoperable, Reutilizable).
    """
    dominios = sorted({safe_str(r.get("dominio_clinico")) for r in registros if r.get("dominio_clinico")})
    origenes = sorted({safe_str(r.get("origen")) for r in registros if r.get("origen")})
    periodos = sorted(p for p in {safe_str(r.get("periodo")) for r in registros} if p)
    advertencias = []
    if clave_por_defecto:
        advertencias.append(
            "Seudónimos generados con la clave de DEMOSTRACIÓN: no publicar. Defina "
            "DATASET_PSEUDONYM_KEY en los secrets antes de exportar datos reales."
        )
    if ORIGEN_SINTETICO in origenes:
        advertencias.append(
            "El paquete contiene registros SINTÉTICOS generados para pruebas y demostración; "
            "no representan pacientes reales ni deben usarse como evidencia clínica."
        )

    return {
        "@context": "https://schema.org/",
        "@type": "Dataset",
        "name": "DeliberIA — Base de casos clínico-éticos estructurada y seudonimizada",
        "alternateName": "BIOETHICARE 360º / DeliberIA clinical-ethics case base",
        "description": (
            "Registros analíticos de deliberaciones bioéticas asistidas por el motor "
            "DeliberIA: ponderación multiperspectiva de los cuatro principios "
            "(Beauchamp y Childress), semáforo ético determinista, indicadores de "
            "consenso/disenso, concordancia IA-comité y validación experta. Sin texto "
            "clínico libre ni identificadores directos."
        ),
        "version": VERSION_DATASET,
        "dateCreated": datetime.now(timezone.utc).date().isoformat(),
        "temporalCoverage": f"{periodos[0]}/{periodos[-1]}" if periodos else "",
        "inLanguage": "es",
        "keywords": [
            "bioética clínica", "inteligencia artificial explicable", "XAI",
            "deliberación ética", "consenso", "auditoría de sesgos", "cuidados paliativos",
            "comités de ética", "FAIR", "ciencia abierta",
        ],
        "creator": [
            {"@type": "Person", "name": "Joseph Javier Sánchez Acuña",
             "identifier": "https://orcid.org/0009-0007-2008-823X",
             "affiliation": "UNIMINUTO — Centro Universitario Buga"},
            {"@type": "Person", "name": "Anderson Díaz Pérez",
             "description": "Titular de los derechos de autor de BioEthicCare360®"},
        ],
        "sourceOrganization": {
            "@type": "Organization",
            "name": "Semillero GLIOSP · Grupo de investigación GICIDET · UNIMINUTO Rectoría Centro Occidente",
        },
        "funding": "Convocatoria Jóvenes Investigadores UNIMINUTO 2026",
        "isBasedOn": {
            "@type": "SoftwareApplication",
            "name": "BIOETHICARE 360º / DeliberIA",
            "identifier": "Registro DNDA No. 13-101-338",
        },
        "citation": "https://doi.org/10.21203/rs.3.rs-7078293/v1",
        "license": "Por definir por los titulares (se sugiere CC BY 4.0 para la versión abierta).",
        "conditionsOfAccess": (
            "Acceso restringido hasta verificar k-anonimato ≥ "
            f"{K_ANONIMATO_MINIMO} y contar con el aval del comité de ética."
        ),
        "measurementTechnique": [
            "Semáforo ético por reglas deterministas (" + VERSION_MOTOR + ")",
            "Índice de consenso 1 - σ/σmax y W de Kendall (" + VERSION_INDICADORES + ")",
            "Seudonimización HMAC-SHA256; generalización de edad y fecha",
        ],
        "variableMeasured": [
            {"@type": "PropertyValue", "name": v, "description": d,
             "unitText": t, "valueReference": dom}
            for v, t, d, dom in DICCIONARIO_DATOS
        ],
        "size": {"registros": len(registros), "dominios_clinicos": dominios, "origen": origenes},
        "calidad": calidad or {},
        "advertencias": advertencias,
        "distribution": [
            {"@type": "DataDownload", "encodingFormat": "text/csv", "contentUrl": "casos.csv"},
            {"@type": "DataDownload", "encodingFormat": "application/json", "contentUrl": "casos.json"},
        ],
    }


# --- Exportación -------------------------------------------------------------------

def registros_a_csv(registros: List[Dict[str, Any]]) -> str:
    buffer = io.StringIO()
    escritor = csv.DictWriter(buffer, fieldnames=COLUMNAS, extrasaction="ignore")
    escritor.writeheader()
    for r in registros:
        escritor.writerow(r)
    return buffer.getvalue()


def diccionario_a_csv() -> str:
    buffer = io.StringIO()
    escritor = csv.writer(buffer)
    escritor.writerow(["variable", "tipo", "descripcion", "dominio_de_valores"])
    for variable, tipo, descripcion, dominio in DICCIONARIO_DATOS:
        escritor.writerow([variable, tipo, descripcion, dominio])
    return buffer.getvalue()


_LEEME = """DeliberIA — Base de casos clínico-éticos (versión {version})
=====================================================================

Contenido
---------
casos.csv / casos.json   Un registro por caso deliberado ({n} registros).
diccionario_datos.csv    Definición, tipo y dominio de cada variable.
metadatos.json           Metadatos FAIR (JSON-LD, schema.org/Dataset).
calidad.json             Completitud, duplicados y k-anonimato.

Privacidad
----------
No contiene texto clínico libre ni identificadores directos. El ID del caso es un
seudónimo HMAC-SHA256; la edad está generalizada en grupos y la fecha en AAAA-MM.
k-anonimato sobre (grupo_edad, genero, dominio_clinico, condicion): {k}.
Apto para apertura según el control automático: {apto}.

Reproducibilidad
----------------
El semáforo y los indicadores se recalculan desde las ponderaciones con el motor
{motor} e indicadores {indicadores}, disponibles en el código fuente.

{advertencias}
Uso: apoyo a la investigación en bioética computacional. No sustituye la decisión
del comité de ética ni del equipo tratante.
"""


def exportar_paquete_zip(
    registros: List[Dict[str, Any]],
    clave_por_defecto: bool = False,
) -> bytes:
    """Construye el paquete reproducible (zip en memoria) con datos, diccionario y metadatos."""
    calidad = evaluar_calidad(registros)
    metadatos = metadatos_fair(registros, calidad, clave_por_defecto=clave_por_defecto)
    advertencias = "\n".join(f"ADVERTENCIA: {a}" for a in metadatos["advertencias"])

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("casos.csv", registros_a_csv(registros))
        z.writestr("casos.json", json.dumps(registros, ensure_ascii=False, indent=2))
        z.writestr("diccionario_datos.csv", diccionario_a_csv())
        z.writestr("metadatos.json", json.dumps(metadatos, ensure_ascii=False, indent=2))
        z.writestr("calidad.json", json.dumps(calidad, ensure_ascii=False, indent=2))
        z.writestr(
            "LEEME.txt",
            _LEEME.format(
                version=VERSION_DATASET,
                n=len(registros),
                k=calidad.get("k_anonimato"),
                apto="sí" if calidad.get("apto_para_apertura") else "no",
                motor=VERSION_MOTOR,
                indicadores=VERSION_INDICADORES,
                advertencias=(advertencias + "\n") if advertencias else "",
            ),
        )
    return buffer.getvalue()


# --- Datos sintéticos de demostración ----------------------------------------------

def generar_reportes_sinteticos(n: int = 60, semilla: int = 2026) -> List[Dict[str, Any]]:
    """
    Reportes SINTÉTICOS y reproducibles para demostrar y probar la analítica sin
    datos reales. Se etiquetan explícitamente y nunca se mezclan con los guardados.

    La distribución de ponderaciones es aleatoria con semilla fija: sirve para
    ejercitar el pipeline, NO para extraer conclusiones clínicas.
    """
    rng = random.Random(semilla)
    dilemas = (
        "Cuidados Paliativos y Futilidad",
        "Limitación del Esfuerzo Terapéutico (Adultos/Pediatría)",
        "Consentimiento Informado",
        "Eutanasia y Muerte Digna",
        "Asignación de Recursos Escasos",
    )
    dominios = ("Cuidados Paliativos", "Unidad de Cuidados Intensivos (UCI)", "Oncología")
    niveles = list(NIVELES_SEVERIDAD)
    reportes: List[Dict[str, Any]] = []
    for i in range(n):
        base = rng.randint(2, 4)
        perspectivas = {}
        for larga in PERSPECTIVAS_NOMBRES.values():
            perspectivas[larga] = {
                p: max(0, min(5, base + rng.choice((-2, -1, 0, 0, 1, 2)))) for p in PRINCIPIOS
            }
        seleccionado = rng.choice(dilemas)
        sugerido = seleccionado if rng.random() < 0.75 else rng.choice(dilemas)
        edad = rng.choice((8, 25, 34, 47, 58, 66, 72, 81))
        reportes.append(
            {
                "schema_version": 3,
                "ID del Caso": f"SINT-{semilla}-{i:04d}",
                "Fecha Análisis (UTC)": f"2026-{rng.randint(1, 9):02d}-15T10:00:00+00:00",
                "Dominio Clínico": rng.choice(dominios),
                "DatosEstructurados": {
                    "edad": edad,
                    "genero": rng.choice(("Masculino", "Femenino")),
                    "condicion": rng.choice(("Estable", "Crítico", "Terminal")),
                    "semanas_gestacion": 0,
                },
                "Dilema Ético Principal (Seleccionado)": seleccionado,
                "Dilema Sugerido por IA": sugerido,
                "AnalisisMultiperspectiva": perspectivas,
                "Análisis Estructurado (IA)": {"nivel_riesgo": rng.choice(niveles)},
                "Tiempo de Deliberación (s)": rng.randint(240, 2400),
                "ValidacionExperta": {
                    "utilidad": rng.randint(3, 5),
                    "pertinencia": rng.randint(3, 5),
                    "fundamentacion": rng.randint(2, 5),
                    "claridad": rng.randint(3, 5),
                    "aceptable": rng.random() < 0.8,
                },
            }
        )
    return reportes


def generar_registros_sinteticos(n: int = 60, semilla: int = 2026) -> List[Dict[str, Any]]:
    return [
        registro_desde_reporte(r, origen=ORIGEN_SINTETICO)
        for r in generar_reportes_sinteticos(n, semilla)
    ]
