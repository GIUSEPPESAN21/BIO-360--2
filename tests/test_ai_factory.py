"""
tests/test_ai_factory.py
========================
Pruebas de la fábrica de proveedores de IA (`services.ai.get_provider`, patrón Strategy).

Incluye la PRUEBA DE REGRESIÓN del defecto que rompía el arranque
-----------------------------------------------------------------
`st.secrets.get(clave)` no devuelve None cuando no existe ningún `secrets.toml`:
lanza `StreamlitSecretNotFoundError`. La rama de callable de `_resolver` no estaba
protegida, así que una instalación sin secretos configurados no degradaba con un aviso
en la interfaz — reventaba al construir el contexto de la aplicación.

`test_resolver_tolera_un_callable_que_lanza` fija ese comportamiento. Si alguien quita
el `try/except` de `_resolver`, esa prueba falla.
"""

from __future__ import annotations

from services.ai import (
    PROVEEDORES_DISPONIBLES,
    GeminiProvider,
    KiroProvider,
    OpenAIProvider,
    get_provider,
)
from services.ai.base import AIProvider, RespuestaIA


class SecretosQueLanzan:
    """Imita a `st.secrets` cuando no hay ningún archivo de secretos configurado."""

    def get(self, clave):
        raise RuntimeError(
            "No secrets found. Valid paths for a secrets.toml file are: ..."
        )


# --- Regresión: secretos ausentes no deben romper el arranque ----------------------

def test_resolver_tolera_un_callable_que_lanza():
    """
    REGRESIÓN. Un resolvedor que lanza debe traducirse en 'no hay proveedor',
    no propagarse hasta la capa de UI.
    """
    secretos = SecretosQueLanzan()

    for nombre in PROVEEDORES_DISPONIBLES:
        assert get_provider(nombre, secretos.get) is None


def test_resolver_tolera_un_mapping_que_lanza():
    class MappingQueLanza:
        def get(self, clave):
            raise KeyError(clave)

    assert get_provider("Google Gemini", MappingQueLanza()) is None


def test_sin_claves_no_hay_proveedor():
    for nombre in PROVEEDORES_DISPONIBLES:
        assert get_provider(nombre, {}) is None


# --- Selección de proveedor (Strategy, M6) -----------------------------------------

def test_gemini_se_construye_con_su_clave():
    provider = get_provider("Google Gemini", {"GEMINI_API_KEY": "k"})

    assert isinstance(provider, GeminiProvider)
    assert isinstance(provider, AIProvider)
    assert provider.nombre == "Google Gemini"


def test_openai_se_construye_con_su_clave():
    provider = get_provider("OpenAI", {"OPENAI_API_KEY": "k"})

    assert isinstance(provider, OpenAIProvider)
    assert provider.nombre == "OpenAI"


def test_kiro_se_construye_con_su_clave():
    provider = get_provider("Kiro", {"KIRO_API_KEY": "k"})

    assert isinstance(provider, KiroProvider)
    assert provider.nombre == "Kiro"


def test_kiro_respeta_la_base_url_configurada():
    provider = get_provider(
        "Kiro", {"KIRO_API_KEY": "k", "KIRO_BASE_URL": "https://interno.example/api/"}
    )
    # La barra final se normaliza para no generar '//' al concatenar la ruta
    assert provider.base_url == "https://interno.example/api"


def test_kiro_sin_base_url_usa_el_valor_por_defecto():
    provider = get_provider("Kiro", {"KIRO_API_KEY": "k"})
    assert provider.base_url


def test_nombre_desconocido_cae_en_gemini():
    """Comportamiento documentado de la fábrica: Gemini es el proveedor por defecto."""
    provider = get_provider("Proveedor Inexistente", {"GEMINI_API_KEY": "k"})
    assert isinstance(provider, GeminiProvider)


def test_kiro_esta_expuesto_en_los_proveedores_disponibles():
    """La pestaña de configuración se alimenta de esta tupla (Opción K-A)."""
    assert "Kiro" in PROVEEDORES_DISPONIBLES
    assert "Google Gemini" in PROVEEDORES_DISPONIBLES
    assert "OpenAI" in PROVEEDORES_DISPONIBLES


# --- Contrato de la interfaz Strategy ----------------------------------------------

def test_todos_los_proveedores_implementan_la_interfaz():
    claves = {
        "Google Gemini": {"GEMINI_API_KEY": "k"},
        "OpenAI": {"OPENAI_API_KEY": "k"},
        "Kiro": {"KIRO_API_KEY": "k"},
    }
    for nombre, secretos in claves.items():
        provider = get_provider(nombre, secretos)
        assert isinstance(provider, AIProvider)
        assert hasattr(provider, "analizar")
        assert isinstance(provider.soporta_schema, bool)
        # Ningún proveedor debe declarar un modelo activo antes de la primera llamada
        assert provider.modelo_activo is None


# --- RespuestaIA -------------------------------------------------------------------

def test_respuesta_ok_requiere_contenido_y_ausencia_de_error():
    assert RespuestaIA(texto="algo").ok is True
    assert RespuestaIA(estructurado={"a": 1}).ok is True
    assert RespuestaIA().ok is False
    assert RespuestaIA(texto="algo", error="fallo").ok is False
    assert RespuestaIA(texto="algo", bloqueado=True).ok is False


def test_respuesta_se_puede_usar_como_cadena():
    """Compatibilidad: el monolito trataba la respuesta de la IA como un str."""
    assert str(RespuestaIA(texto="análisis")) == "análisis"
