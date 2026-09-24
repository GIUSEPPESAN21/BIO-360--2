"""
core/etica.py
=============
Reglas deterministas del "semáforo ético" de BIOETHICARE 360.

Estas reglas NO usan IA: son determinísticas y auditables. La IA (Kiro/Gemini/OpenAI)
solo se usa para *explicar* el resultado, nunca para decidirlo (patrón usado también
en SaludIA). Por eso esta lógica está cubierta por pytest (M3): un cambio silencioso
en los umbrales alteraría diagnósticos clínicos.

Explicabilidad (XAI)
--------------------
`evaluar_semaforo` devuelve cada hallazgo con los puntos que aporta a la severidad,
de modo que el resultado es atribuible regla por regla (y perspectiva por perspectiva)
y admite explicaciones contrafactuales ("si se resolviera X, la severidad sería Y").
Es la base del componente de IA explicable de DeliberIA: la explicación NO la inventa
un modelo, la entrega el propio motor.

Umbrales de severidad (deben permanecer estables salvo decisión explícita):
    puntos_severidad >= 5  -> "Crítico"
    puntos_severidad >= 2  -> "Moderado"
    en otro caso           -> "Bajo"
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple

#: Versión del motor de reglas. Se persiste con cada caso para que un análisis pasado
#: pueda reproducirse con exactamente la lógica que lo produjo.
VERSION_MOTOR = "semaforo_v2_xai"

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

# Tipos de hallazgo (claves estables para la base de datos y el análisis agregado)
TIPO_PERSPECTIVA_OMITIDA = "perspectiva_omitida"
TIPO_PRINCIPIO_OMITIDO = "principio_omitido"
TIPO_DESEQUILIBRIO_INTERNO = "desequilibrio_interno"
TIPO_DESEQUILIBRIO_EXTERNO = "desequilibrio_externo"

ETIQUETAS_TIPO = {
    TIPO_PERSPECTIVA_OMITIDA: "Perspectiva omitida",
    TIPO_PRINCIPIO_OMITIDO: "Principio omitido",
    TIPO_DESEQUILIBRIO_INTERNO: "Desequilibrio interno",
    TIPO_DESEQUILIBRIO_EXTERNO: "Desequilibrio externo",
}


@dataclass(frozen=True)
class Hallazgo:
    """
    Un hallazgo del semáforo ético con su contribución EXPLÍCITA a la severidad.

    Es la unidad de explicabilidad (XAI) del motor: cada punto de severidad se puede
    atribuir a una regla concreta, a una perspectiva y, si aplica, a un principio.
    """

    tipo: str               # una de las constantes TIPO_*
    perspectiva: str        # clave corta de la perspectiva ("medico", ...) o "" si es global
    principio: str          # clave del principio o "" si no aplica
    puntos: int             # contribución a `puntos_severidad`
    advertencia: str
    recomendacion: str
    detalle: Dict[str, Any] = field(default_factory=dict)

    def como_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResultadoSemaforo:
    """Resultado completo y explicable del semáforo ético."""

    hallazgos: List[Hallazgo]
    puntos_severidad: int
    severidad: str

    @property
    def advertencias(self) -> List[str]:
        return [h.advertencia for h in self.hallazgos]

    @property
    def recomendaciones(self) -> List[str]:
        return [h.recomendacion for h in self.hallazgos]

    def contribuciones_por_tipo(self) -> Dict[str, int]:
        """Puntos de severidad agregados por tipo de regla (atribución global)."""
        totales: Dict[str, int] = {}
        for h in self.hallazgos:
            totales[h.tipo] = totales.get(h.tipo, 0) + h.puntos
        return totales

    def contribuciones_por_perspectiva(self) -> Dict[str, int]:
        """Puntos de severidad atribuibles a cada perspectiva ("" = global)."""
        totales: Dict[str, int] = {}
        for h in self.hallazgos:
            totales[h.perspectiva] = totales.get(h.perspectiva, 0) + h.puntos
        return totales

    def contrafactuales(self) -> List[Dict[str, Any]]:
        """
        Explicación contrafactual: para cada tipo de hallazgo, qué severidad tendría el
        caso si ese tipo de problema se resolviera (y todo lo demás quedara igual).

        Ayuda al comité a ver QUÉ revisar primero para bajar la severidad, sin que el
        motor proponga ponderaciones concretas (eso es decisión humana).
        """
        salida: List[Dict[str, Any]] = []
        for tipo, puntos in sorted(
            self.contribuciones_por_tipo().items(), key=lambda kv: -kv[1]
        ):
            restantes = self.puntos_severidad - puntos
            salida.append(
                {
                    "tipo": tipo,
                    "etiqueta": ETIQUETAS_TIPO.get(tipo, tipo),
                    "puntos_que_aporta": puntos,
                    "puntos_si_se_resuelve": restantes,
                    "severidad_si_se_resuelve": clasificar_severidad(restantes),
                }
            )
        return salida

    def como_dict(self) -> Dict[str, Any]:
        """Forma serializable para el reporte y Firestore (sin objetos Python)."""
        return {
            "puntos_severidad": self.puntos_severidad,
            "severidad": self.severidad,
            "hallazgos": [h.como_dict() for h in self.hallazgos],
            "contribuciones_por_tipo": self.contribuciones_por_tipo(),
            "contrafactuales": self.contrafactuales(),
            "version_motor": VERSION_MOTOR,
        }


def clasificar_severidad(puntos_severidad: int) -> str:
    """Traduce puntos de severidad a la etiqueta del semáforo (umbrales bloqueados)."""
    if puntos_severidad >= UMBRAL_SEVERIDAD_CRITICO:
        return "Crítico"
    if puntos_severidad >= UMBRAL_SEVERIDAD_MODERADO:
        return "Moderado"
    return "Bajo"


def evaluar_semaforo(caso) -> ResultadoSemaforo:
    """
    Evalúa las ponderaciones de un caso y devuelve un resultado EXPLICABLE.

    `caso` debe exponer `.perspectivas: dict[str, dict[str, int]]`.
    Las reglas, pesos y umbrales son exactamente los de `verificar_sesgo_etico`.
    """
    hallazgos: List[Hallazgo] = []

    for nombre, valores in caso.perspectivas.items():
        if sum(valores.values()) == 0:
            hallazgos.append(
                Hallazgo(
                    tipo=TIPO_PERSPECTIVA_OMITIDA,
                    perspectiva=nombre,
                    principio="",
                    puntos=PESO_PERSPECTIVA_OMITIDA,
                    advertencia=(
                        f"**Perspectiva Omitida:** La perspectiva de '{nombre}' no asignó "
                        "puntuación a ningún principio."
                    ),
                    recomendacion=(
                        f"Se recomienda verificar si la ponderación de '{nombre}' fue omitida "
                        "accidentalmente para asegurar una deliberación completa."
                    ),
                )
            )

        for principio, valor in valores.items():
            if valor == 0:
                principio_legible = principio.replace("_", " ").capitalize()
                hallazgos.append(
                    Hallazgo(
                        tipo=TIPO_PRINCIPIO_OMITIDO,
                        perspectiva=nombre,
                        principio=principio,
                        puntos=PESO_PRINCIPIO_OMITIDO,
                        advertencia=(
                            f"**Principio Omitido en '{nombre.title()}':** El principio de "
                            f"'{principio_legible}' tiene un valor de 0."
                        ),
                        recomendacion=(
                            f"Evaluar si la omisión del principio de '{principio_legible}' en la "
                            f"perspectiva de '{nombre.title()}' es intencional y justificada."
                        ),
                    )
                )

        if sum(valores.values()) > 0:
            max_diff = max(valores.values()) - min(valores.values())
            if max_diff >= UMBRAL_DESEQUILIBRIO_INTERNO:
                hallazgos.append(
                    Hallazgo(
                        tipo=TIPO_DESEQUILIBRIO_INTERNO,
                        perspectiva=nombre,
                        principio="",
                        puntos=PESO_DESEQUILIBRIO_INTERNO,
                        advertencia=(
                            f"**Alto Desequilibrio Interno:** En la perspectiva de "
                            f"'{nombre.title()}', existe un alto desequilibrio entre los "
                            f"principios (diferencia de {max_diff} puntos)."
                        ),
                        recomendacion=(
                            "Se sugiere revisar si la alta disparidad en la ponderación de esta "
                            "perspectiva está suficientemente justificada o si requiere una "
                            "deliberación más balanceada."
                        ),
                        detalle={"diferencia": max_diff},
                    )
                )

    puntajes_totales = {n: sum(v.values()) for n, v in caso.perspectivas.items()}
    if len(puntajes_totales) > 1:
        max_persp = max(puntajes_totales, key=puntajes_totales.get)
        min_persp = min(puntajes_totales, key=puntajes_totales.get)
        diferencia = puntajes_totales[max_persp] - puntajes_totales[min_persp]
        if diferencia >= UMBRAL_DESEQUILIBRIO_EXTERNO:
            hallazgos.append(
                Hallazgo(
                    tipo=TIPO_DESEQUILIBRIO_EXTERNO,
                    perspectiva="",
                    principio="",
                    puntos=PESO_DESEQUILIBRIO_EXTERNO,
                    advertencia=(
                        f"**Alto Desequilibrio Externo:** La perspectiva de "
                        f"'{max_persp.title()}' tiene un peso total significativamente mayor "
                        f"que la de '{min_persp.title()}'."
                    ),
                    recomendacion=(
                        "Analizar si esta dominancia de una perspectiva sobre otra es adecuada "
                        "para el caso o si es necesario re-equilibrar las ponderaciones para una "
                        "decisión más equitativa."
                    ),
                    detalle={
                        "dominante": max_persp,
                        "minoritaria": min_persp,
                        "diferencia": diferencia,
                    },
                )
            )

    puntos = sum(h.puntos for h in hallazgos)
    return ResultadoSemaforo(
        hallazgos=hallazgos, puntos_severidad=puntos, severidad=clasificar_severidad(puntos)
    )


def verificar_sesgo_etico(caso) -> Tuple[List[str], List[str], str]:
    """
    Evalúa las ponderaciones de un `CasoBioetico` y devuelve
    (advertencias, recomendaciones, severidad).

    Interfaz histórica, conservada por compatibilidad: delega en `evaluar_semaforo`,
    que además expone la atribución de cada punto de severidad (XAI).
    """
    resultado = evaluar_semaforo(caso)
    return resultado.advertencias, resultado.recomendaciones, resultado.severidad
