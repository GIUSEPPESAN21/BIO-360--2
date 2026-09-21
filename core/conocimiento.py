"""
core/conocimiento.py
====================
Carga y validación de la base de conocimiento de dilemas (`dilemas.json`).

Se separa de Streamlit para que sea testeable. El caché de Streamlit se aplica en
la capa de UI/servicios, no aquí.

Validación de esquema
---------------------
`dilemas.json` no es un archivo de configuración cualquiera: su contenido se inyecta
como **contexto de grounding** en los prompts del modelo de IA
(ver `services.ai.prompts.construir_contexto_normativo`). El modelo recibe la
instrucción de citar normativa ÚNICAMENTE desde esa base, así que una entrada
malformada no produce un error visible: produce un prompt degradado y, en el peor
caso, una respuesta legal mal fundamentada.

Por eso `cargar_dilemas` valida la base tras leerla. Decisión de diseño deliberada:

- Las entradas **irrecuperables** (cuyo valor no es un objeto JSON) se **descartan**,
  porque no hay forma de normalizarlas sin inventar contenido normativo.
- El resto se **normaliza** (las cuatro claves quedan garantizadas como listas de
  cadenas, descartando elementos inválidos).
- Los problemas se registran con `logger.warning`, no se lanzan.

Es decir, la aplicación **sigue arrancando con la parte sana de la base** en lugar de
caer por un dilema mal escrito. Para CI y pruebas, donde sí se quiere fallar ruidosa-
mente, existe el parámetro `estricto=True`.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

RUTA_POR_DEFECTO = "dilemas.json"

#: Claves que debe exponer la ficha de cada dilema.
CLAVES_ESPERADAS: tuple[str, ...] = ("riesgos", "beneficios", "alternativas", "normativas")

#: Valor por defecto explícito por clave cuando falta o queda vacía tras normalizar.
_POR_DEFECTO: Dict[str, str] = {
    "riesgos": "No especificados",
    "beneficios": "No especificados",
    "alternativas": "No especificadas",
    "normativas": "No especificadas",
}


# --- Validación -------------------------------------------------------------------

def validar_base_dilemas(dilemas_data: Dict[str, Any]) -> List[str]:
    """
    Valida la forma de la base de conocimiento sin lanzar excepciones.

    Returns
    -------
    list[str]
        Descripciones de los problemas encontrados. Una lista VACÍA significa que la
        base está sana.
    """
    problemas: List[str] = []

    if not isinstance(dilemas_data, dict):
        return ["La base de dilemas no es un objeto JSON en la raíz."]

    if not dilemas_data:
        return ["La base de dilemas está vacía."]

    for nombre, info in dilemas_data.items():
        if not isinstance(nombre, str) or not nombre.strip():
            problemas.append(f"Nombre de dilema inválido o vacío: {nombre!r}.")

        if not isinstance(info, dict):
            problemas.append(
                f"Dilema '{nombre}': el valor no es un objeto JSON "
                f"(se encontró {type(info).__name__}); la entrada es irrecuperable."
            )
            continue

        for clave in CLAVES_ESPERADAS:
            if clave not in info:
                problemas.append(f"Dilema '{nombre}': falta la clave '{clave}'.")
                continue

            valor = info[clave]
            if not isinstance(valor, list):
                problemas.append(
                    f"Dilema '{nombre}': la clave '{clave}' no es una lista "
                    f"(se encontró {type(valor).__name__})."
                )
                continue

            if not valor:
                problemas.append(f"Dilema '{nombre}': la lista '{clave}' está vacía.")

            for i, elemento in enumerate(valor):
                if not isinstance(elemento, str) or not elemento.strip():
                    problemas.append(
                        f"Dilema '{nombre}': el elemento {i} de '{clave}' no es una "
                        f"cadena de texto no vacía ({elemento!r})."
                    )

    return problemas


def normalizar_dilema(info: Any) -> Dict[str, List[str]]:
    """
    Devuelve la ficha de un dilema con las cuatro claves garantizadas como listas de
    cadenas, descartando elementos inválidos y aplicando valores por defecto explícitos.

    Acepta cualquier entrada: si `info` no es un dict, devuelve la ficha por defecto.
    """
    fuente = info if isinstance(info, dict) else {}
    normalizado: Dict[str, List[str]] = {}

    for clave in CLAVES_ESPERADAS:
        valor = fuente.get(clave)

        if isinstance(valor, list):
            limpios = [str(v).strip() for v in valor if isinstance(v, str) and str(v).strip()]
        elif isinstance(valor, str) and valor.strip():
            # Tolerancia: una cadena suelta se trata como lista de un elemento.
            limpios = [valor.strip()]
        else:
            limpios = []

        normalizado[clave] = limpios or [_POR_DEFECTO[clave]]

    return normalizado


# --- Carga ------------------------------------------------------------------------

def cargar_dilemas(ruta: str = RUTA_POR_DEFECTO, estricto: bool = False) -> Dict[str, Any]:
    """
    Carga la base de conocimiento de dilemas bioéticos.

    Devuelve un dict vacío (sin lanzar excepción) si el archivo no existe o es inválido,
    para que la aplicación pueda degradar con un mensaje en lugar de caer.

    Tras la lectura se valida la forma de la base:
    - Las entradas cuyo valor no es un objeto JSON se DESCARTAN (irrecuperables).
    - El resto se normaliza con `normalizar_dilema`.
    - Cada problema se registra con `logger.warning`.

    Parameters
    ----------
    ruta:
        Ruta al JSON. Si es relativa y no existe en el directorio de trabajo, se
        resuelve también relativa a la raíz del proyecto.
    estricto:
        `False` por defecto, para NO alterar el comportamiento de la aplicación.
        Si es `True`, lanza `ValueError` cuando la validación encuentra cualquier
        problema. Pensado para CI y pruebas automatizadas.

    Raises
    ------
    ValueError
        Solo cuando `estricto=True` y la validación encuentra problemas.
    """
    try:
        # Resolver también relativo a la raíz del proyecto
        if not os.path.isabs(ruta) and not os.path.exists(ruta):
            raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            candidato = os.path.join(raiz, ruta)
            if os.path.exists(candidato):
                ruta = candidato

        with open(ruta, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.error("dilemas.json no contiene un objeto JSON en la raíz.")
            if estricto:
                raise ValueError("dilemas.json no contiene un objeto JSON en la raíz.")
            return {}
    except FileNotFoundError:
        logger.error("El archivo %s no fue encontrado.", ruta)
        if estricto:
            raise ValueError(f"El archivo {ruta} no fue encontrado.")
        return {}
    except json.JSONDecodeError as e:
        logger.error("Error al decodificar %s: %s", ruta, e)
        if estricto:
            raise ValueError(f"Error al decodificar {ruta}: {e}")
        return {}

    problemas = validar_base_dilemas(data)
    if problemas:
        if estricto:
            raise ValueError(
                f"La base de dilemas tiene {len(problemas)} problema(s) de esquema: "
                + "; ".join(problemas)
            )
        for problema in problemas:
            logger.warning("Validación de dilemas.json: %s", problema)

    # Descartar irrecuperables y normalizar el resto, para seguir operando con la
    # parte sana de la base en lugar de caer.
    saneada: Dict[str, Any] = {}
    descartados: List[str] = []
    for nombre, info in data.items():
        if not isinstance(info, dict):
            descartados.append(str(nombre))
            continue
        saneada[str(nombre)] = normalizar_dilema(info)

    if descartados:
        logger.warning(
            "Se descartaron %d dilema(s) irrecuperables de la base: %s",
            len(descartados),
            ", ".join(descartados),
        )

    return saneada


def listar_dilemas(dilemas_data: Dict[str, Any]) -> List[str]:
    """Nombres de los dilemas disponibles, en el orden del archivo."""
    if not isinstance(dilemas_data, dict):
        return []
    return list(dilemas_data.keys())


def info_dilema(dilemas_data: Dict[str, Any], nombre: str) -> Dict[str, List[str]]:
    """
    Devuelve la ficha de un dilema con todas las claves garantizadas,
    usando valores por defecto explícitos si faltan.

    Delega en `normalizar_dilema` para no duplicar la lógica de saneamiento.
    """
    fuente = dilemas_data if isinstance(dilemas_data, dict) else {}
    return normalizar_dilema(fuente.get(nombre))
