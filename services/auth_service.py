"""
services/auth_service.py
========================
Autenticación de cliente contra Firebase, mediante la API REST de Identity Toolkit.

Por qué este módulo existe (y por qué se retiró Pyrebase)
--------------------------------------------------------
La aplicación usaba `Pyrebase4` para una sola cosa: iniciar sesión y registrar usuarios
con email y contraseña. Pero `Pyrebase4` arrastra una cadena de dependencias abandonada:

    Pyrebase4 -> gcloud (ultima version de 2015) -> pkg_resources
              -> oauth2client (obsoleto desde 2018)
              -> python-jwt, requests-toolbelt, pycryptodome

`pkg_resources` formaba parte de `setuptools` y ha sido retirado de las versiones
recientes, y un entorno de Python 3.12+ ya no instala `setuptools` por defecto. El
resultado en Python 3.13 es que `import pyrebase` falla al arrancar:

    ModuleNotFoundError: No module named 'pkg_resources'

Ese fallo es irreparable desde nuestro lado: está en una dependencia transitiva sin
mantenimiento desde hace una década. Fijar `setuptools` a una versión antigua solo
aplaza el problema y arrastra una dependencia obsoleta a producción.

`Pyrebase` no es más que un envoltorio sobre la API REST de Firebase, así que la
solución correcta es llamar a esa API directamente con `requests`, que el proyecto ya
usa. Se elimina toda la cadena rota y no se pierde ninguna funcionalidad: la parte de
Pyrebase que necesitaba `gcloud` era el módulo de Storage, que esta aplicación nunca usó.

Compatibilidad
--------------
`FirebaseAuthApp` imita deliberadamente la forma de Pyrebase (`app.auth()` devuelve un
cliente con `sign_in_with_email_and_password` y `create_user_with_email_and_password`),
de modo que `ui/login.py` no necesita cambios. Las respuestas conservan las claves que
la aplicación ya leía: `localId`, `email` e `idToken`.

Referencia: https://firebase.google.com/docs/reference/rest/auth
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

BASE_IDENTITY_TOOLKIT = "https://identitytoolkit.googleapis.com/v1"
TIMEOUT_SEGUNDOS = 30


class ErrorAutenticacion(Exception):
    """
    Fallo de autenticación con un mensaje apto para mostrar al usuario.

    `codigo` conserva el código crudo de Firebase para diagnóstico en los logs; el texto
    de la excepción es el mensaje ya traducido, sin revelar si un email existe o no.
    """

    def __init__(self, mensaje: str, codigo: str = "") -> None:
        super().__init__(mensaje)
        self.codigo = codigo


#: Traducción de los códigos de error de Identity Toolkit a mensajes en español.
#: Firebase cambia el código según la configuración del proyecto: con la protección
#: contra enumeración de emails activada devuelve INVALID_LOGIN_CREDENTIALS en lugar de
#: distinguir EMAIL_NOT_FOUND de INVALID_PASSWORD. Se cubren ambas familias, y todas las
#: variantes de credencial inválida comparten un mensaje genérico a propósito: decir
#: "ese correo no existe" permitiría enumerar las cuentas del sistema.
_MENSAJES: Dict[str, str] = {
    "INVALID_LOGIN_CREDENTIALS": "Email o contraseña incorrectos. Verifique sus credenciales.",
    "EMAIL_NOT_FOUND": "Email o contraseña incorrectos. Verifique sus credenciales.",
    "INVALID_PASSWORD": "Email o contraseña incorrectos. Verifique sus credenciales.",
    "INVALID_EMAIL": "El formato del correo electrónico no es válido.",
    "MISSING_PASSWORD": "Debe introducir una contraseña.",
    "MISSING_EMAIL": "Debe introducir un correo electrónico.",
    "USER_DISABLED": "Esta cuenta ha sido deshabilitada por un administrador.",
    "EMAIL_EXISTS": "Ese correo electrónico ya está registrado.",
    "WEAK_PASSWORD": "La contraseña es demasiado débil: debe tener al menos 6 caracteres.",
    "OPERATION_NOT_ALLOWED": (
        "El inicio de sesión con email y contraseña no está habilitado en el proyecto "
        "de Firebase. Actívelo en Authentication > Sign-in method."
    ),
    "TOO_MANY_ATTEMPTS_TRY_LATER": (
        "Demasiados intentos fallidos. Espere unos minutos antes de volver a intentarlo."
    ),
}

MENSAJE_GENERICO = "No se pudo completar la operación de autenticación. Intente de nuevo."


def _mensaje_para(codigo: str) -> str:
    """Traduce un código de Firebase, tolerando sufijos como 'WEAK_PASSWORD : ...'."""
    crudo = (codigo or "").strip()
    if crudo in _MENSAJES:
        return _MENSAJES[crudo]
    # Firebase adjunta detalles tras ':' en algunos códigos
    prefijo = crudo.split(":")[0].strip()
    return _MENSAJES.get(prefijo, MENSAJE_GENERICO)


class FirebaseAuthClient:
    """
    Cliente de autenticación por email y contraseña.

    Los nombres de los métodos replican los de Pyrebase para que la capa de UI no
    necesite cambios.
    """

    def __init__(self, api_key: str, timeout: int = TIMEOUT_SEGUNDOS) -> None:
        if not api_key:
            raise ValueError("Se requiere la apiKey web del proyecto de Firebase.")
        self.api_key = api_key
        self.timeout = timeout

    # -- Internos ------------------------------------------------------------------

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Llama a un endpoint de Identity Toolkit y devuelve el JSON de respuesta.

        Raises
        ------
        ErrorAutenticacion
            Tanto si Firebase devuelve un error como si la llamada no se pudo realizar.
            La clave de API nunca se incluye en el mensaje ni en los logs.
        """
        import requests

        url = f"{BASE_IDENTITY_TOOLKIT}/{endpoint}"
        try:
            respuesta = requests.post(
                url,
                params={"key": self.api_key},
                json=payload,
                timeout=self.timeout,
            )
        except Exception as e:
            logger.error("Error de red contactando Firebase Auth (%s): %s", endpoint, e)
            raise ErrorAutenticacion(
                "No se pudo contactar con el servicio de autenticación. "
                "Revise su conexión de red."
            ) from e

        try:
            datos = respuesta.json()
        except ValueError:
            datos = {}

        if respuesta.status_code >= 400 or "error" in datos:
            codigo = ""
            if isinstance(datos.get("error"), dict):
                codigo = str(datos["error"].get("message") or "")
            # Se registra el código, nunca las credenciales ni la apiKey
            logger.warning(
                "Firebase Auth rechazó la operación %s (HTTP %s, código=%s)",
                endpoint, respuesta.status_code, codigo or "desconocido",
            )
            raise ErrorAutenticacion(_mensaje_para(codigo), codigo)

        return datos if isinstance(datos, dict) else {}

    # -- API pública (compatible con Pyrebase) -------------------------------------

    def sign_in_with_email_and_password(self, email: str, password: str) -> Dict[str, Any]:
        """
        Inicia sesión. Devuelve el dict de Firebase, que incluye `localId`, `email`,
        `idToken`, `refreshToken` y `expiresIn`.
        """
        return self._post(
            "accounts:signInWithPassword",
            {"email": email or "", "password": password or "", "returnSecureToken": True},
        )

    def create_user_with_email_and_password(self, email: str, password: str) -> Dict[str, Any]:
        """Registra un usuario nuevo. Misma forma de respuesta que el inicio de sesión."""
        return self._post(
            "accounts:signUp",
            {"email": email or "", "password": password or "", "returnSecureToken": True},
        )

    def send_email_verification(self, id_token: str) -> Dict[str, Any]:
        """
        Envía el correo de verificación de dirección.

        Útil si se decide activar la condición `emailVerificado()` de
        `firestore.rules` para escribir datos clínicos.
        """
        return self._post(
            "accounts:sendOobCode",
            {"requestType": "VERIFY_EMAIL", "idToken": id_token or ""},
        )

    def get_account_info(self, id_token: str) -> Dict[str, Any]:
        """Datos de la cuenta, incluido `emailVerified`."""
        return self._post("accounts:lookup", {"idToken": id_token or ""})


class FirebaseAuthApp:
    """
    Equivalente mínimo de la app de Pyrebase.

    Existe solo para conservar la forma `app.auth()` que ya usaba `ui/login.py`, de modo
    que retirar Pyrebase no obligue a tocar la capa de interfaz.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = dict(config or {})
        api_key = self._config.get("apiKey")
        if not api_key:
            raise ValueError(
                "La configuración de cliente de Firebase no incluye 'apiKey'. "
                "Revise la sección [firebase_client_config] de sus secrets."
            )
        self._cliente = FirebaseAuthClient(api_key)

    def auth(self) -> FirebaseAuthClient:
        return self._cliente

    @property
    def project_id(self) -> Optional[str]:
        return self._config.get("projectId")


def initialize_auth_app(config: Dict[str, Any]) -> Optional[FirebaseAuthApp]:
    """
    Crea la app de autenticación a partir de `[firebase_client_config]`.

    Devuelve None si la configuración falta o es inválida, para que la aplicación pueda
    avisar por la interfaz en lugar de romperse.
    """
    try:
        return FirebaseAuthApp(config)
    except Exception as e:
        logger.error("No se pudo inicializar la autenticación de Firebase: %s", e)
        return None
