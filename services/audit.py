"""
services/audit.py
=================
Audit log inmutable (M8).

Requisito ético-legal
---------------------
En decisiones de fin de vida (LET, eutanasia, cuidados paliativos) debe poder
reconstruirse a posteriori QUIÉN generó cada análisis, con QUÉ modelo exacto,
con QUÉ versión de prompt y CUÁNDO. Sin esa trazabilidad el sistema no es
defendible ante una auditoría externa ni ante una revisión judicial.

Diseño de inmutabilidad
-----------------------
- Cada entrada se escribe con `document()` (ID autogenerado) y `create()` cuando está
  disponible, de modo que una escritura nunca sobrescribe una entrada previa.
- Las reglas de `firestore.rules` permiten `create` pero prohíben `update` y `delete`
  en la colección de auditoría: la inmutabilidad se garantiza server-side, no por
  convención del cliente.
- Cada entrada incluye un `hash_encadenado` (SHA-256 del contenido + hash anterior),
  lo que permite detectar manipulación de la secuencia incluso con acceso privilegiado.
- Nunca se registra PII ni el texto clínico: solo metadatos y un hash del prompt.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

COLECCION_AUDITORIA = "audit_log"
HASH_GENESIS = "0" * 64

# Acciones registrables
ACCION_ANALISIS_CLINICO = "analisis_historia_clinica"
ACCION_DELIBERACION = "analisis_deliberativo"
ACCION_CHAT = "consulta_chat"
ACCION_EXPLICACION_SEMAFORO = "explicacion_semaforo_etico"
ACCION_CASO_CREADO = "caso_creado"
ACCION_CONSENTIMIENTO = "consentimiento_generado"
# DeliberIA: acciones del flujo de investigación
ACCION_VALIDACION_EXPERTA = "validacion_experta_registrada"
ACCION_SUS_REGISTRADO = "cuestionario_sus_registrado"
ACCION_EXPORTACION_DATASET = "base_de_datos_exportada"
ACCION_AUDITORIA_SESGOS = "auditoria_sesgos_generada"


def _hash_texto(texto: str) -> str:
    return hashlib.sha256((texto or "").encode("utf-8")).hexdigest()


def _hash_entrada(entrada: Dict[str, Any], hash_previo: str) -> str:
    """SHA-256 canónico de la entrada encadenado con el hash anterior."""
    base = json.dumps(entrada, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(f"{hash_previo}|{base}".encode("utf-8")).hexdigest()


class AuditLog:
    """
    Escritor del registro de auditoría.

    Ruta: `usuarios/{uid}/audit_log/{entrada_id}`
    (mismo árbol que los casos, para que las reglas de seguridad por UID apliquen).
    """

    def __init__(self, db: Any, uid: str, user_email: str = "") -> None:
        self.db = db
        self.uid = uid
        self.user_email = user_email
        self._ultimo_hash: Optional[str] = None

    # -- Rutas ---------------------------------------------------------------------

    def _coleccion(self):
        return (
            self.db.collection("usuarios")
            .document(self.uid)
            .collection(COLECCION_AUDITORIA)
        )

    # -- Escritura -----------------------------------------------------------------

    def registrar(
        self,
        accion: str,
        *,
        caso_id: str = "",
        proveedor: str = "",
        modelo: str = "",
        prompt_version: str = "",
        prompt_hash: str = "",
        prompt_texto: Optional[str] = None,
        anonimizado: Optional[bool] = None,
        resultado_ok: Optional[bool] = None,
        intentos: Optional[int] = None,
        metadatos: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Escribe una entrada inmutable en el audit log.

        `prompt_texto` NO se almacena: solo se usa para calcular su hash, de modo que
        el log pueda probar qué prompt se envió sin retener datos clínicos.
        """
        if self.db is None or not self.uid:
            logger.warning("Audit log no disponible (sin db o sin uid); acción=%s", accion)
            return None

        if prompt_texto is not None and not prompt_hash:
            prompt_hash = _hash_texto(prompt_texto)

        entrada: Dict[str, Any] = {
            "accion": accion,
            "usuario_uid": self.uid,
            "usuario_email": self.user_email,
            "caso_id": caso_id,
            "proveedor_ia": proveedor,
            "modelo_ia": modelo,
            "prompt_version": prompt_version,
            "prompt_sha256": prompt_hash,
            "datos_anonimizados": anonimizado,
            "resultado_ok": resultado_ok,
            "intentos": intentos,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "metadatos": metadatos or {},
        }

        hash_previo = self._obtener_ultimo_hash()
        entrada["hash_previo"] = hash_previo
        entrada["hash_encadenado"] = _hash_entrada(entrada, hash_previo)

        try:
            doc_ref = self._coleccion().document()
            # `create` falla si el documento ya existe -> refuerza la inmutabilidad
            if hasattr(doc_ref, "create"):
                doc_ref.create(entrada)
            else:  # pragma: no cover - compatibilidad con stubs
                doc_ref.set(entrada)
            self._ultimo_hash = entrada["hash_encadenado"]
            logger.info("Audit log: %s (modelo=%s, caso=%s)", accion, modelo, caso_id)
            return entrada
        except Exception as e:
            logger.error("No se pudo escribir en el audit log (%s): %s", accion, e)
            return None

    def registrar_respuesta_ia(
        self,
        accion: str,
        respuesta: Any,
        *,
        caso_id: str = "",
        prompt_texto: Optional[str] = None,
        anonimizado: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Atajo para registrar una `RespuestaIA` con todos sus metadatos."""
        return self.registrar(
            accion,
            caso_id=caso_id,
            proveedor=getattr(respuesta, "proveedor", ""),
            modelo=getattr(respuesta, "modelo", ""),
            prompt_version=getattr(respuesta, "prompt_version", ""),
            prompt_texto=prompt_texto,
            anonimizado=anonimizado,
            resultado_ok=getattr(respuesta, "ok", None),
            intentos=getattr(respuesta, "intentos", None),
            metadatos=dict(getattr(respuesta, "metadatos", {}) or {}),
        )

    # -- Lectura / verificación -----------------------------------------------------

    def _obtener_ultimo_hash(self) -> str:
        if self._ultimo_hash:
            return self._ultimo_hash
        try:
            query = self._coleccion()
            try:
                query = query.order_by("timestamp_utc", direction="DESCENDING").limit(1)
            except Exception:
                query = query.limit(1)
            docs = list(query.stream())
            if docs:
                data = docs[0].to_dict() or {}
                return data.get("hash_encadenado") or HASH_GENESIS
        except Exception as e:
            logger.warning("No se pudo leer el último hash del audit log: %s", e)
        return HASH_GENESIS

    def listar(self, limite: int = 50) -> List[Dict[str, Any]]:
        """Devuelve las entradas más recientes del audit log."""
        if self.db is None or not self.uid:
            return []
        try:
            query = self._coleccion()
            try:
                query = query.order_by("timestamp_utc", direction="DESCENDING")
            except Exception:
                pass
            try:
                query = query.limit(limite)
            except Exception:
                pass
            return [d.to_dict() or {} for d in query.stream()]
        except Exception as e:
            logger.error("Error leyendo el audit log: %s", e)
            return []


def verificar_cadena(entradas: List[Dict[str, Any]]) -> bool:
    """
    Verifica la integridad de la cadena de hashes.

    `entradas` debe venir en orden cronológico ASCENDENTE. Devuelve True si cada
    entrada encadena correctamente con la anterior.
    """
    hash_previo = HASH_GENESIS
    for entrada in entradas:
        # Reconstruye el payload exacto que se hasheó en `registrar`:
        # todos los campos excepto `hash_encadenado`, con `hash_previo` incluido.
        payload = {k: v for k, v in entrada.items() if k != "hash_encadenado"}
        if payload.get("hash_previo") != hash_previo:
            return False
        if entrada.get("hash_encadenado") != _hash_entrada(payload, hash_previo):
            return False
        hash_previo = entrada["hash_encadenado"]
    return True
