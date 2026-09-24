"""
services/reportes.py
====================
Armado del reporte deliberativo y del texto de consentimiento informado.

Optimización M5
---------------
`generar_reporte_completo` ya NO incluye los JSON de Plotly. Solo persiste las
ponderaciones en bruto; los gráficos se regeneran bajo demanda con `services.charts`.
Esto reduce drásticamente el tamaño del documento en Firestore y el costo de lectura.

El campo `schema_version` permite distinguir casos nuevos de los guardados con el
esquema antiguo (que sí traían `radar_chart_json` y compañía).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.conocimiento import info_dilema
from core.indicadores import calcular_indicadores
from core.modelos import safe_str

# v1 = incluía JSON de Plotly; v2 = solo ponderaciones (M5);
# v3 = DeliberIA: dominio clínico, datos estructurados, indicadores de deliberación,
#      explicación XAI del semáforo, tiempo de deliberación y validación experta.
SCHEMA_VERSION = 3

# Claves pesadas del esquema v1 que ya no se escriben (se mantienen documentadas
# para poder limpiarlas de documentos antiguos).
CLAVES_LEGADAS_PESADAS = (
    "radar_chart_json",
    "stats_chart_json",
    "equilibrio_chart_json",
)


def generar_reporte_completo(
    caso,
    dilema_sugerido: Optional[str] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    ethical_analysis: Optional[Dict[str, Any]] = None,
    analisis_estructurado: Optional[Dict[str, Any]] = None,
    explicacion_semaforo: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Construye el dict del reporte que se muestra en la UI y se persiste en Firestore.

    `explicacion_semaforo` es `ResultadoSemaforo.como_dict()` (atribución de cada punto
    de severidad y contrafactuales): se guarda dentro de `AnalisisEtico["xai"]`.
    """
    resumen_paciente = (
        f"Paciente {caso.nombre_paciente}, {caso.edad} años, "
        f"género {caso.genero}, condición {caso.condicion}."
    )
    if caso.semanas_gestacion > 0:
        resumen_paciente += f" Neonato de {caso.semanas_gestacion} sem."

    reporte: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "ID del Caso": caso.historia_clinica,
        "Fecha Análisis": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Fecha Análisis (UTC)": datetime.now(timezone.utc).isoformat(),
        "Analista": caso.nombre_analista,
        "Resumen del Paciente": resumen_paciente,
        "Dominio Clínico": getattr(caso, "dominio_clinico", "Otro"),
        "Dilema Ético Principal (Seleccionado)": caso.dilema_etico,
        "Dilema Sugerido por IA": dilema_sugerido or "",
        "Descripción Detallada del Caso": caso.descripcion_caso,
        "Contexto Sociocultural y Familiar": caso.antecedentes_culturales,
        "Puntos Clave para Deliberación IA": caso.puntos_clave_ia,
        "Análisis IA de Historia Clínica": caso.ai_clinical_analysis_summary,
        # Fuente de verdad para regenerar los gráficos (M5)
        "AnalisisMultiperspectiva": {
            "Equipo Médico": caso.perspectivas["medico"],
            "Familia/Paciente": caso.perspectivas["familia"],
            "Comité de Bioética": caso.perspectivas["comite"],
        },
        # Variables estructuradas del paciente (sin nombre) para la base analítica
        "DatosEstructurados": (
            caso.datos_estructurados() if hasattr(caso, "datos_estructurados") else {}
        ),
        "AnalisisEtico": dict(ethical_analysis or {}),
        # Indicadores cuantitativos de deliberación (consenso, disenso, W de Kendall)
        "IndicadoresDeliberacion": calcular_indicadores(caso.perspectivas),
        "Tiempo de Deliberación (s)": getattr(caso, "tiempo_deliberacion_s", 0),
        "Análisis Deliberativo (IA)": "",
        "Historial del Chat de Deliberación": chat_history or [],
        "Trazabilidad IA": {},
    }

    if explicacion_semaforo:
        reporte["AnalisisEtico"]["xai"] = explicacion_semaforo

    if analisis_estructurado:
        # Salida estructurada de Kiro (K1), útil para auditoría y para el PDF
        reporte["Análisis Estructurado (IA)"] = analisis_estructurado

    return reporte


def limpiar_claves_legadas(reporte: Dict[str, Any]) -> Dict[str, Any]:
    """
    Elimina los JSON de Plotly de documentos guardados con el esquema v1.

    Se aplica al leer casos antiguos: los gráficos se regeneran desde las
    ponderaciones, por lo que esas claves ya no aportan nada y solo ocupan memoria.
    """
    if not isinstance(reporte, dict):
        return reporte
    return {k: v for k, v in reporte.items() if k not in CLAVES_LEGADAS_PESADAS}


def generar_texto_consentimiento(caso, dilemas_data: Dict[str, Any]) -> str:
    """Genera el texto del consentimiento/asentimiento informado a partir del caso."""
    info = info_dilema(dilemas_data, caso.dilema_etico)
    riesgos = "\n".join(f"- {r}" for r in info["riesgos"])
    beneficios = "\n".join(f"- {b}" for b in info["beneficios"])
    alternativas = "\n".join(f"- {a}" for a in info["alternativas"])
    normativas = "\n".join(f"- {n}" for n in info["normativas"])

    return f"""
CONSENTIMIENTO/ASENTIMIENTO INFORMADO (BIOETHICARE 360)
Fecha: {datetime.now().strftime("%Y-%m-%d")}
ID del Caso: {caso.historia_clinica}
------------------------------------------------------------------
DATOS DEL PACIENTE
------------------------------------------------------------------
Nombre: {caso.nombre_paciente}
Edad: {caso.edad} años
Género: {caso.genero}
Dilema Ético Principal: {caso.dilema_etico}
------------------------------------------------------------------
INFORMACIÓN SOBRE LA DECISIÓN
------------------------------------------------------------------
En el contexto de su situación clínica, se ha identificado un dilema ético principal relacionado con "{caso.dilema_etico}". A continuación, se presenta la información relevante para que usted (o su representante) pueda tomar una decisión informada.
1. RIESGOS POTENCIALES:
{riesgos}
2. BENEFICIOS ESPERADOS:
{beneficios}
3. ALTERNATIVAS DISPONIBLES:
{alternativas}
4. MARCO NORMATIVO Y ÉTICO:
Esta deliberación se enmarca en las siguientes normativas y principios:
{normativas}
------------------------------------------------------------------
DECLARACIÓN Y FIRMA
------------------------------------------------------------------
Declaro que he leído (o me han leído) y comprendido la información anterior. He tenido la oportunidad de hacer preguntas y todas han sido respondidas a mi satisfacción.
Entiendo que mi decisión es voluntaria y que puedo retirarla en cualquier momento sin que ello afecte la calidad de mi atención médica.
Firma del Paciente/Tutor Legal: _________________________
Nombre: _________________________
Fecha: _________________________
Firma del Profesional de la Salud: _________________________
Nombre: {caso.nombre_analista}
Fecha: _________________________
"""


def anonimizar_reporte_para_ia(reporte: Dict[str, Any], anonimizador) -> Dict[str, Any]:
    """
    Devuelve una copia del reporte apta para enviarse a una API de IA (M2).

    - Elimina identificadores directos (nombre del paciente/analista van tokenizados).
    - Anonimiza todos los campos de texto libre.
    - Descarta claves irrelevantes/pesadas para el modelo (gráficos, trazabilidad).
    """
    CAMPOS_TEXTO = (
        "Resumen del Paciente",
        "Descripción Detallada del Caso",
        "Contexto Sociocultural y Familiar",
        "Puntos Clave para Deliberación IA",
        "Análisis IA de Historia Clínica",
        "Análisis Deliberativo (IA)",
    )
    EXCLUIR = set(CLAVES_LEGADAS_PESADAS) | {
        "Trazabilidad IA",
        "Analista",
        "schema_version",
        "Fecha Análisis (UTC)",
        # La validación experta y el tiempo son metadatos de investigación: no
        # aportan a la deliberación y no deben sesgar al modelo.
        "ValidacionExperta",
        "Tiempo de Deliberación (s)",
    }

    limpio: Dict[str, Any] = {}
    for clave, valor in (reporte or {}).items():
        if clave in EXCLUIR:
            continue
        if clave in CAMPOS_TEXTO and isinstance(valor, str):
            limpio[clave] = anonimizador.anonimizar(valor)
        elif clave == "Historial del Chat de Deliberación" and isinstance(valor, list):
            limpio[clave] = [
                {
                    "role": safe_str(m.get("role")),
                    "content": anonimizador.anonimizar(safe_str(m.get("content"))),
                }
                for m in valor
                if isinstance(m, dict)
            ]
        elif clave == "AnalisisEtico" and isinstance(valor, dict):
            limpio[clave] = {
                "severidad": valor.get("severidad"),
                "advertencias": [
                    anonimizador.anonimizar(safe_str(a)) for a in valor.get("advertencias", [])
                ],
                "recomendaciones": [
                    anonimizador.anonimizar(safe_str(r)) for r in valor.get("recomendaciones", [])
                ],
            }
        else:
            limpio[clave] = valor

    return limpio
