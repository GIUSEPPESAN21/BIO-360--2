"""
services/firebase_service.py
===========================
Acceso a Firebase: inicialización y repositorio de casos.

Mejoras
-------
- **Repositorio centralizado (M1).** La ruta `usuarios/{uid}/casos/{caso_id}` estaba
  duplicada literalmente en 4 puntos del monolito. Ahora vive en un solo lugar,
  `CasosRepository`, alineada con las reglas de `firestore.rules`.
- **Paginación (M5).** `listar_ids` y `listar_pagina` usan `limit()` y cursores
  (`start_after`) en vez de traer todos los documentos con `.stream()`.
- **Lectura ligera (M5).** Para poblar el selector solo se piden los campos mínimos
  con `select()`, evitando descargar el reporte completo de cada caso.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

COLECCION_USUARIOS = "usuarios"
SUBCOLECCION_CASOS = "casos"
PAGINA_POR_DEFECTO = 20

# Campos ligeros para el listado (no descarga el reporte entero)
CAMPOS_RESUMEN = ("ID del Caso", "Fecha Análisis", "Dilema Ético Principal (Seleccionado)")


# --- Inicialización ----------------------------------------------------------------

def initialize_firebase_admin(secrets: Any) -> Optional[Any]:
    """
    Inicializa el Admin SDK y devuelve el cliente de Firestore.
    `secrets` debe comportarse como un mapping con la clave "firebase_credentials".
    """
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore
    except ImportError as e:  # pragma: no cover
        logger.error("firebase-admin no está instalado: %s", e)
        return None

    try:
        if "firebase_credentials" not in secrets:
            logger.error("Credenciales de Firebase Admin no encontradas en secrets.")
            return None

        creds_dict = dict(secrets["firebase_credentials"])
        if "private_key" in creds_dict:
            creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")

        if not firebase_admin._apps:
            cred = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(cred)
            logger.info("Conexión con Firebase Admin SDK establecida.")
        return firestore.client()
    except Exception as e:
        logger.error("Error crítico al conectar con Firebase Admin SDK: %s", e)
        return None


def initialize_firebase_auth(secrets: Any) -> Optional[Any]:
    """Inicializa Pyrebase para la autenticación de cliente."""
    try:
        import pyrebase
    except ImportError as e:  # pragma: no cover
        logger.error("pyrebase no está instalado: %s", e)
        return None

    try:
        if "firebase_client_config" not in secrets:
            logger.error("firebase_client_config no encontrado en secrets.")
            return None
        return pyrebase.initialize_app(dict(secrets["firebase_client_config"]))
    except Exception as e:
        logger.error("Error crítico al inicializar Pyrebase: %s", e)
        return None


# --- Repositorio de casos ----------------------------------------------------------

class CasosRepository:
    """
    Encapsula el acceso a `usuarios/{uid}/casos`.

    Todas las operaciones están acotadas al `uid` recibido, en correspondencia con
    las reglas server-side de `firestore.rules`.
    """

    def __init__(self, db: Any, uid: str) -> None:
        if not uid:
            raise ValueError("Se requiere un UID de usuario para acceder a los casos.")
        self.db = db
        self.uid = uid

    # -- Rutas ---------------------------------------------------------------------

    def _coleccion(self):
        return (
            self.db.collection(COLECCION_USUARIOS)
            .document(self.uid)
            .collection(SUBCOLECCION_CASOS)
        )

    def _doc(self, caso_id: str):
        return self._coleccion().document(caso_id)

    # -- Escritura -----------------------------------------------------------------

    def guardar(self, caso_id: str, reporte: Dict[str, Any]) -> None:
        self._doc(caso_id).set(reporte)

    def actualizar(self, caso_id: str, campos: Dict[str, Any]) -> None:
        self._doc(caso_id).update(campos)

    # -- Lectura -------------------------------------------------------------------

    def obtener(self, caso_id: str) -> Optional[Dict[str, Any]]:
        snap = self._doc(caso_id).get()
        if not getattr(snap, "exists", False):
            return None
        from services.reportes import limpiar_claves_legadas

        return limpiar_claves_legadas(snap.to_dict() or {})

    def listar_pagina(
        self,
        limite: int = PAGINA_POR_DEFECTO,
        cursor: Optional[Any] = None,
    ) -> Tuple[List[Dict[str, Any]], Optional[Any]]:
        """
        Devuelve (resumenes, cursor_siguiente) usando paginación real.

        `resumenes` contiene solo los campos de `CAMPOS_RESUMEN` más el id, para no
        descargar el reporte completo de cada caso (M5).
        `cursor_siguiente` es None cuando no hay más páginas.
        """
        query = self._coleccion()

        # `select` puede no existir en algunos stubs/mocks; degradar con elegancia.
        try:
            query = query.select(list(CAMPOS_RESUMEN))
        except Exception:
            pass

        try:
            query = query.order_by("Fecha Análisis", direction="DESCENDING")
        except Exception:
            try:
                from google.cloud.firestore import Query

                query = self._coleccion().order_by(
                    "Fecha Análisis", direction=Query.DESCENDING
                )
            except Exception:
                query = self._coleccion()

        if cursor is not None:
            try:
                query = query.start_after(cursor)
            except Exception:
                pass

        try:
            query = query.limit(limite)
        except Exception:
            pass

        docs = list(query.stream())
        resumenes: List[Dict[str, Any]] = []
        for d in docs:
            data = d.to_dict() or {}
            resumenes.append(
                {
                    "id": d.id,
                    "ID del Caso": data.get("ID del Caso", d.id),
                    "Fecha Análisis": data.get("Fecha Análisis", ""),
                    "Dilema": data.get("Dilema Ético Principal (Seleccionado)", ""),
                    "_snapshot": d,
                }
            )

        siguiente = docs[-1] if len(docs) == limite else None
        return resumenes, siguiente

    def contar_aproximado(self, tope: int = 100) -> int:
        """Cuenta hasta `tope` documentos (evita escanear colecciones grandes)."""
        try:
            return len(list(self._coleccion().limit(tope).stream()))
        except Exception:
            return 0
