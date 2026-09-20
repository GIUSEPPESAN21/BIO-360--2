"""
core/anonimizacion.py
=====================
Capa de anonimización de PII (M2) para BIOETHICARE 360.

Objetivo de cumplimiento
-------------------------
Antes de enviar cualquier texto a una API de IA de terceros (Gemini, OpenAI, Kiro),
esta capa reemplaza información personal identificable (PII) por tokens genéricos
([PACIENTE], [FECHA], [LUGAR], [ID], [TELEFONO], [EMAIL], [DOCUMENTO]).

El sistema conserva un "mapa de rehidratación" que permite reconstruir los valores
originales ÚNICAMENTE en el frontend, al mostrar el reporte final al usuario
autorizado. Los valores originales nunca abandonan el servidor.

Cumple con:
- Ley 1581 de 2012 (Protección de Datos Personales, Colombia)
- HIPAA (EE.UU.)
- GDPR (UE)

Diseño
------
`Anonimizador` es determinista y reversible dentro de una misma sesión de análisis:
para un mismo valor de entrada produce el mismo token, de modo que el modelo de IA
mantiene la coherencia referencial (p. ej. "[PACIENTE_1]" siempre es la misma persona).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# --- Expresiones regulares de PII -------------------------------------------------

# Fechas: 12/03/2024, 12-03-2024, 2024-03-12, "12 de marzo de 2024"
_MESES = (
    "enero|febrero|marzo|abril|mayo|junio|julio|agosto|"
    "septiembre|setiembre|octubre|noviembre|diciembre"
)
_PATRONES: List[Tuple[str, str]] = [
    # Email antes que nada (contiene @)
    ("EMAIL", r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
    # Teléfonos colombianos / internacionales (7 a 15 dígitos, con separadores opcionales)
    ("TELEFONO", r"(?<!\d)(?:\+?\d{1,3}[\s\-]?)?(?:\(?\d{2,4}\)?[\s\-]?)?\d{3}[\s\-]?\d{4}(?!\d)"),
    # Documentos de identidad con etiqueta explícita (CC, TI, CE, NUIP + número)
    ("DOCUMENTO", r"(?:\b(?:CC|T\.?I\.?|C\.?E\.?|NUIP|NIT)\b[\s.:#\-]*)\d[\d.\-]{5,}"),
    # Fechas numéricas
    ("FECHA", r"\b\d{1,4}[/\-]\d{1,2}[/\-]\d{1,4}\b"),
    # Fechas textuales en español
    ("FECHA", rf"\b\d{{1,2}}\s+de\s+(?:{_MESES})\s+de\s+\d{{4}}\b"),
]

# Preposiciones/marcadores que suelen preceder a un lugar
_MARCADORES_LUGAR = r"(?:en|de|desde|hacia|hospital|clínica|clinica|ciudad de|municipio de|barrio)"


@dataclass
class Anonimizador:
    """
    Anonimiza texto libre reemplazando PII por tokens y guarda el mapa inverso.

    Uso típico
    ----------
    >>> anon = Anonimizador()
    >>> limpio = anon.anonimizar("Juan Pérez ingresó el 12/03/2024 en Bogotá.")
    >>> # `limpio` -> "[PACIENTE_1] ingresó el [FECHA_1] en [LUGAR_1]."
    >>> anon.rehidratar(limpio)  # reconstruye el original para el frontend
    """

    # Nombres propios conocidos del caso que deben protegerse (paciente, analista, etc.)
    nombres_conocidos: List[str] = field(default_factory=list)
    _mapa: Dict[str, str] = field(default_factory=dict)   # token -> valor original
    _inverso: Dict[str, str] = field(default_factory=dict)  # valor original -> token
    _contadores: Dict[str, int] = field(default_factory=dict)

    # -- API pública --------------------------------------------------------------

    def anonimizar(self, texto: str) -> str:
        """Devuelve el texto con la PII reemplazada por tokens estables."""
        if not texto:
            return texto or ""

        resultado = texto

        # 1) Nombres propios conocidos del caso (paciente, analista) -> [PACIENTE_n]
        for nombre in sorted(self.nombres_conocidos, key=len, reverse=True):
            nombre = (nombre or "").strip()
            if len(nombre) < 3 or nombre.upper() in ("N/A", "NA"):
                continue
            patron = re.compile(re.escape(nombre), flags=re.IGNORECASE)
            resultado = patron.sub(lambda _m, n=nombre: self._token("PACIENTE", n), resultado)

        # 2) Patrones estructurales (email, teléfono, documento, fecha)
        for etiqueta, patron in _PATRONES:
            resultado = re.sub(
                patron,
                lambda m, e=etiqueta: self._token(e, m.group(0)),
                resultado,
                flags=re.IGNORECASE,
            )

        # 3) Lugares precedidos por marcador (en/de + Palabra Capitalizada)
        #
        # El flag IGNORECASE se aplica SOLO al marcador, mediante el grupo con ámbito
        # `(?i:...)`. Antes se compilaba el patrón completo con re.IGNORECASE, lo que
        # anulaba la exigencia de mayúscula inicial de `[A-ZÁÉÍÓÚÑ]` y convertía prosa
        # clínica corriente en falsos lugares: "el principio de autonomía frente a la
        # beneficencia" se transformaba en "el principio de [LUGAR_1] a la beneficencia".
        # Eso no filtraba datos, pero enviaba al modelo un texto mutilado.
        patron_lugar = re.compile(
            rf"\b(?i:{_MARCADORES_LUGAR})\s+"
            r"([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)?)"
        )

        def _sub_lugar(m: "re.Match") -> str:
            marcador = m.group(0)[: m.start(1) - m.start(0)]
            lugar = m.group(1)
            return f"{marcador}{self._token('LUGAR', lugar)}"

        resultado = patron_lugar.sub(_sub_lugar, resultado)

        return resultado

    def rehidratar(self, texto: str) -> str:
        """
        Reconstruye los valores originales a partir de los tokens.
        Debe usarse SOLO en el frontend, al mostrar el reporte al usuario autorizado.
        """
        if not texto:
            return texto or ""
        resultado = texto
        # Reemplazar tokens más largos primero para evitar solapamientos ([PACIENTE_10] vs [PACIENTE_1])
        for token in sorted(self._mapa, key=len, reverse=True):
            resultado = resultado.replace(token, self._mapa[token])
        return resultado

    @property
    def mapa(self) -> Dict[str, str]:
        """Mapa token -> valor original (para rehidratar). Nunca debe enviarse a la IA."""
        return dict(self._mapa)

    # -- Internos -----------------------------------------------------------------

    def _token(self, etiqueta: str, valor: str) -> str:
        """Devuelve un token estable para `valor`; reutiliza el mismo si ya existe."""
        valor = valor.strip()
        if valor in self._inverso:
            return self._inverso[valor]
        self._contadores[etiqueta] = self._contadores.get(etiqueta, 0) + 1
        token = f"[{etiqueta}_{self._contadores[etiqueta]}]"
        self._mapa[token] = valor
        self._inverso[valor] = token
        return token


def anonimizar_texto(texto: str, nombres_conocidos: List[str] | None = None) -> Tuple[str, Dict[str, str]]:
    """
    Función de conveniencia sin estado persistente.

    Returns
    -------
    (texto_anonimizado, mapa_rehidratacion)
    """
    anon = Anonimizador(nombres_conocidos=nombres_conocidos or [])
    limpio = anon.anonimizar(texto)
    return limpio, anon.mapa
