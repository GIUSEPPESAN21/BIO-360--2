"""
tests/test_auth_service.py
==========================
Pruebas del servicio de autenticación por API REST (`services.auth_service`).

Incluye la regresión del fallo de arranque en Python 3.13
--------------------------------------------------------
`Pyrebase4` importaba `gcloud`, que importa `pkg_resources`. Ese módulo se retiró de las
versiones recientes de `setuptools`, y Python 3.12+ ya no instala `setuptools` por
defecto, así que `import pyrebase` reventaba el arranque con
`ModuleNotFoundError: No module named 'pkg_resources'`.

`test_ningun_modulo_de_produccion_importa_pyrebase` falla si alguien vuelve a introducir
esa dependencia.

Ninguna prueba hace llamadas de red: se inyecta un doble de `requests.post`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from services.auth_service import (
    ErrorAutenticacion,
    FirebaseAuthApp,
    FirebaseAuthClient,
    initialize_auth_app,
)

CONFIG = {
    "apiKey": "clave-web-de-prueba",
    "authDomain": "proyecto.firebaseapp.com",
    "projectId": "proyecto",
}

RESPUESTA_LOGIN = {
    "kind": "identitytoolkit#VerifyPasswordResponse",
    "localId": "uid-123",
    "email": "medico@hospital.co",
    "idToken": "token-de-id",
    "refreshToken": "token-de-refresco",
    "expiresIn": "3600",
    "registered": True,
}


class RespuestaFalsa:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        if self._payload is _SIN_JSON:
            raise ValueError("sin json")
        return self._payload


_SIN_JSON = object()


@pytest.fixture
def capturador(monkeypatch):
    """Sustituye requests.post y registra la llamada."""
    registro = {}

    def falso_post(url, params=None, json=None, timeout=None):
        registro["url"] = url
        registro["params"] = params or {}
        registro["json"] = json or {}
        registro["timeout"] = timeout
        return RespuestaFalsa(registro.get("_respuesta", RESPUESTA_LOGIN),
                              registro.get("_status", 200))

    import requests

    monkeypatch.setattr(requests, "post", falso_post)
    return registro


# --- Construcción -------------------------------------------------------------------

def test_se_requiere_api_key():
    with pytest.raises(ValueError):
        FirebaseAuthClient(api_key="")


def test_la_app_expone_auth_como_pyrebase():
    """Forma `app.auth()`: es lo que permite que ui/login.py no cambiara."""
    app = FirebaseAuthApp(CONFIG)
    cliente = app.auth()

    assert isinstance(cliente, FirebaseAuthClient)
    assert hasattr(cliente, "sign_in_with_email_and_password")
    assert hasattr(cliente, "create_user_with_email_and_password")
    assert app.project_id == "proyecto"


def test_config_sin_api_key_devuelve_none_en_vez_de_lanzar():
    assert initialize_auth_app({"projectId": "x"}) is None
    assert initialize_auth_app({}) is None
    assert initialize_auth_app(None) is None


def test_config_valida_construye_la_app():
    assert isinstance(initialize_auth_app(CONFIG), FirebaseAuthApp)


# --- Inicio de sesión ---------------------------------------------------------------

def test_inicio_de_sesion_exitoso_devuelve_local_id(capturador):
    cliente = FirebaseAuthClient(CONFIG["apiKey"])
    usuario = cliente.sign_in_with_email_and_password("medico@hospital.co", "secreta")

    assert usuario["localId"] == "uid-123"
    assert usuario["email"] == "medico@hospital.co"


def test_inicio_de_sesion_usa_el_endpoint_correcto(capturador):
    FirebaseAuthClient(CONFIG["apiKey"]).sign_in_with_email_and_password("a@b.co", "x")

    assert capturador["url"].endswith("/accounts:signInWithPassword")
    assert capturador["json"]["returnSecureToken"] is True
    assert capturador["json"]["email"] == "a@b.co"


def test_la_api_key_viaja_como_parametro_no_en_el_cuerpo(capturador):
    """La clave va en la query string, como exige Identity Toolkit."""
    FirebaseAuthClient(CONFIG["apiKey"]).sign_in_with_email_and_password("a@b.co", "x")

    assert capturador["params"]["key"] == CONFIG["apiKey"]
    assert "key" not in capturador["json"]


def test_registro_usa_el_endpoint_de_alta(capturador):
    FirebaseAuthClient(CONFIG["apiKey"]).create_user_with_email_and_password("a@b.co", "x")
    assert capturador["url"].endswith("/accounts:signUp")


def test_verificacion_de_email_usa_send_oob_code(capturador):
    FirebaseAuthClient(CONFIG["apiKey"]).send_email_verification("token-de-id")

    assert capturador["url"].endswith("/accounts:sendOobCode")
    assert capturador["json"]["requestType"] == "VERIFY_EMAIL"


# --- Traducción de errores ----------------------------------------------------------

@pytest.mark.parametrize(
    "codigo",
    ["INVALID_LOGIN_CREDENTIALS", "EMAIL_NOT_FOUND", "INVALID_PASSWORD"],
)
def test_credenciales_invalidas_comparten_mensaje_generico(capturador, codigo):
    """
    No debe poder distinguirse "el email no existe" de "la contraseña es incorrecta":
    esa diferencia permitiría enumerar las cuentas del sistema.
    """
    capturador["_respuesta"] = {"error": {"code": 400, "message": codigo}}
    capturador["_status"] = 400

    with pytest.raises(ErrorAutenticacion) as exc:
        FirebaseAuthClient(CONFIG["apiKey"]).sign_in_with_email_and_password("a@b.co", "x")

    assert "incorrectos" in str(exc.value)
    assert exc.value.codigo == codigo


def test_email_ya_existente_tiene_mensaje_propio(capturador):
    capturador["_respuesta"] = {"error": {"code": 400, "message": "EMAIL_EXISTS"}}
    capturador["_status"] = 400

    with pytest.raises(ErrorAutenticacion) as exc:
        FirebaseAuthClient(CONFIG["apiKey"]).create_user_with_email_and_password("a@b.co", "x")

    assert "ya está registrado" in str(exc.value)


def test_codigo_con_detalle_adjunto_se_traduce(capturador):
    """Firebase adjunta detalles tras ':' en algunos códigos."""
    capturador["_respuesta"] = {
        "error": {"code": 400, "message": "WEAK_PASSWORD : Password should be at least 6"}
    }
    capturador["_status"] = 400

    with pytest.raises(ErrorAutenticacion) as exc:
        FirebaseAuthClient(CONFIG["apiKey"]).create_user_with_email_and_password("a@b.co", "x")

    assert "débil" in str(exc.value)


def test_codigo_desconocido_cae_en_mensaje_generico(capturador):
    capturador["_respuesta"] = {"error": {"code": 400, "message": "ALGO_NUEVO_DE_FIREBASE"}}
    capturador["_status"] = 400

    with pytest.raises(ErrorAutenticacion):
        FirebaseAuthClient(CONFIG["apiKey"]).sign_in_with_email_and_password("a@b.co", "x")


def test_respuesta_sin_json_valido_se_reporta_como_error(capturador):
    capturador["_respuesta"] = _SIN_JSON
    capturador["_status"] = 500

    with pytest.raises(ErrorAutenticacion):
        FirebaseAuthClient(CONFIG["apiKey"]).sign_in_with_email_and_password("a@b.co", "x")


def test_fallo_de_red_se_traduce_a_error_de_autenticacion(monkeypatch):
    import requests

    def post_que_falla(*a, **k):
        raise requests.exceptions.ConnectionError("sin red")

    monkeypatch.setattr(requests, "post", post_que_falla)

    with pytest.raises(ErrorAutenticacion) as exc:
        FirebaseAuthClient(CONFIG["apiKey"]).sign_in_with_email_and_password("a@b.co", "x")

    assert "conexión" in str(exc.value) or "contactar" in str(exc.value)


def test_el_mensaje_de_error_nunca_incluye_la_api_key(capturador):
    capturador["_respuesta"] = {"error": {"code": 400, "message": "INVALID_LOGIN_CREDENTIALS"}}
    capturador["_status"] = 400

    with pytest.raises(ErrorAutenticacion) as exc:
        FirebaseAuthClient(CONFIG["apiKey"]).sign_in_with_email_and_password("a@b.co", "x")

    assert CONFIG["apiKey"] not in str(exc.value)


# --- Regresión: la cadena de dependencias rota no debe volver -----------------------

def _es_sentencia_de_import_de_pyrebase(linea: str) -> bool:
    """
    True solo para una SENTENCIA de importación real.

    Las docstrings de `auth_service.py` y `firebase_service.py` mencionan
    `import pyrebase` para documentar por qué se retiró; una búsqueda por subcadena las
    marcaría como infracción. Se exige que la línea empiece por la sentencia.
    """
    desnuda = linea.strip()
    if desnuda.startswith("#"):
        return False
    return (
        desnuda.startswith("import pyrebase")
        or desnuda.startswith("from pyrebase")
        or desnuda.startswith("import Pyrebase")
        or desnuda.startswith("from Pyrebase")
    )


def test_ningun_modulo_de_produccion_importa_pyrebase():
    """
    REGRESIÓN del fallo de arranque en Python 3.13.

    Pyrebase -> gcloud -> pkg_resources (retirado de setuptools). Reintroducir Pyrebase
    volvería a romper el despliegue. `app_legacy.py` se excluye: es el monolito
    conservado como referencia y no se ejecuta.
    """
    raiz = Path(__file__).resolve().parent.parent
    carpetas = ("core", "services", "ui")

    infractores = []
    for carpeta in carpetas:
        for archivo in (raiz / carpeta).rglob("*.py"):
            for linea in archivo.read_text(encoding="utf-8").splitlines():
                if _es_sentencia_de_import_de_pyrebase(linea):
                    infractores.append(f"{carpeta}/{archivo.name}: {linea.strip()}")

    for linea in (raiz / "app.py").read_text(encoding="utf-8").splitlines():
        if _es_sentencia_de_import_de_pyrebase(linea):
            infractores.append(f"app.py: {linea.strip()}")

    assert not infractores, f"Se reintrodujo Pyrebase: {infractores}"


def test_la_prueba_de_regresion_detecta_un_import_real():
    """Verifica que el detector no sea permisivo: debe cazar la sentencia real."""
    assert _es_sentencia_de_import_de_pyrebase("import pyrebase")
    assert _es_sentencia_de_import_de_pyrebase("  from pyrebase import initialize_app")
    # Y que no marque prosa ni comentarios
    assert not _es_sentencia_de_import_de_pyrebase("# import pyrebase fallaba")
    assert not _es_sentencia_de_import_de_pyrebase("El `import pyrebase` fallaba al arrancar")


def test_requirements_no_declara_la_cadena_abandonada():
    raiz = Path(__file__).resolve().parent.parent
    texto = (raiz / "requirements.txt").read_text(encoding="utf-8")

    activos = [
        l.strip() for l in texto.splitlines()
        if l.strip() and not l.strip().startswith("#")
    ]
    declarados = " ".join(activos).lower()

    for paquete in ("pyrebase", "gcloud", "oauth2client", "python-jwt"):
        assert paquete not in declarados, f"'{paquete}' volvió a requirements.txt"
