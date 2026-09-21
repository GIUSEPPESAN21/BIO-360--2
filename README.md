# 🏥 BIOETHICARE 360º

Software de análisis y deliberación bioética con inteligencia artificial, autenticación
y base de datos.

Asiste a comités de bioética hospitalarios en el análisis de casos complejos: extrae
elementos bioéticos de una historia clínica, registra la ponderación multiperspectiva de
los cuatro principios, calcula un semáforo ético determinista, genera un análisis
deliberativo asistido por IA y produce el reporte y el consentimiento informado en PDF.

---

## Arquitectura

La aplicación está organizada en tres capas. `app.py` es un punto de entrada delgado que
solo las compone; no contiene lógica de negocio.

```
app.py                  Composición: estado, conexiones, orquestador, pestañas
core/                   Dominio puro, sin Streamlit ni red — testeable en aislamiento
  modelos.py              CasoBioetico, conversiones seguras, constantes de dominio
  etica.py                Semáforo ético DETERMINISTA (motor de reglas, no IA)
  anonimizacion.py        Capa de anonimización de PII y rehidratación
  conocimiento.py         Carga y validación de dilemas.json
services/               Integraciones externas
  ai_orchestrator.py      Único camino hacia un modelo: anonimiza, llama, audita
  ai/                     Proveedores intercambiables (patrón Strategy)
    base.py                 Interfaz AIProvider + RespuestaIA
    gemini_provider.py      Google Gemini
    openai_provider.py      OpenAI
    kiro_provider.py        Kiro
    prompts.py              Prompts versionados + grounding normativo (RAG)
    schemas.py              Esquemas JSON validables y parsing tolerante
  firebase_service.py     Inicialización y repositorio de casos (con paginación)
  audit.py                Registro de auditoría inmutable, encadenado con SHA-256
  charts.py               Figuras Plotly, regeneradas bajo demanda
  pdf_service.py          Reporte y consentimiento en PDF, con gráficos incrustados
  reportes.py             Armado del reporte y del texto de consentimiento
ui/                     Interfaz Streamlit, una pestaña por módulo
tests/                  Suite pytest
firestore.rules         Reglas de seguridad del lado del servidor
app_legacy.py           Monolito anterior — NO EJECUTAR (ver más abajo)
```

### Garantías que la arquitectura hace cumplir

Todas las llamadas a un modelo pasan por `services/ai_orchestrator.py`, que aplica en
este orden:

1. **Anonimización de PII.** Nombres, fechas, lugares, documentos, teléfonos y correos se
   sustituyen por marcadores (`[PACIENTE_1]`, `[FECHA_1]`) **antes** de que el texto salga
   hacia una API de terceros. Los valores reales se rehidratan solo al mostrar el
   resultado al usuario autorizado. Requisito de la Ley 1581, HIPAA y GDPR.
2. **Proveedor intercambiable.** Añadir un modelo es implementar `AIProvider` y
   registrarlo en la fábrica; no se añaden condicionales por toda la aplicación.
3. **Auditoría.** Cada llamada queda registrada con usuario, modelo exacto, versión de
   prompt, hash del prompt y timestamp. Nunca se almacena el texto clínico, solo su hash.

**La IA no decide.** El semáforo ético (severidad, advertencias, recomendaciones) lo
calcula el motor de reglas determinista de `core/etica.py`. La IA puede *explicar* ese
resultado, nunca recalcularlo.

> ⚠️ **`app_legacy.py`** es el monolito anterior, conservado como referencia histórica.
> **No debe ejecutarse:** enviaba las historias clínicas sin anonimizar a la API de IA,
> fijaba los filtros de seguridad del modelo en `BLOCK_ONLY_HIGH` de forma permanente y
> no registraba nada en el log de auditoría.

---

## Instalación

### 1. Dependencias

```bash
pip install -r requirements.txt
```

### 2. Credenciales

Copia la plantilla y rellena los valores reales:

```bash
mkdir .streamlit
copy secrets_template.toml .streamlit\secrets.toml    # Windows
# cp secrets_template.toml .streamlit/secrets.toml    # Linux / macOS
```

`secrets_template.toml` documenta cada campo: la clave del proveedor de IA que vayas a
usar (`GEMINI_API_KEY`, `OPENAI_API_KEY` o `KIRO_API_KEY`), las credenciales de servidor
de Firebase (`[firebase_credentials]`) y la configuración de cliente
(`[firebase_client_config]`).

> 🔒 `.streamlit/secrets.toml` está excluido por `.gitignore` y **nunca** debe
> versionarse: contiene la clave privada de la cuenta de servicio, que da acceso
> administrativo completo a los datos clínicos.

La aplicación arranca sin credenciales: avisa por la interfaz de lo que falta y
deshabilita las funciones correspondientes, en lugar de fallar.

### 3. Desplegar las reglas de Firestore

**Paso obligatorio.** La autenticación del frontend no es suficiente: sin estas reglas,
la base de datos no impone el aislamiento entre usuarios.

```bash
firebase deploy --only firestore:rules
```

Las reglas restringen a cada usuario a su propia ruta `usuarios/{uid}`, y hacen el
registro de auditoría inmutable (permiten crear entradas, prohíben modificarlas y
borrarlas).

### 4. Ejecutar

```bash
streamlit run app.py
```

---

## Pruebas

```bash
python -m pytest tests -q
```

La suite cubre la lógica que no puede depender de comprobaciones manuales en la interfaz:

| Archivo | Qué protege |
|---|---|
| `test_etica.py` | Umbrales del semáforo ético. Incluye un candado que falla si alguien los modifica, porque eso reclasificaría casos ya analizados |
| `test_anonimizacion.py` | Reversibilidad y estabilidad de los tokens; que ningún tipo de PII sobreviva a la anonimización |
| `test_modelos.py` | Conversiones seguras en la frontera entre el formulario y la aritmética del semáforo |
| `test_schemas.py` | Parsing tolerante de la salida del modelo; que un dilema inventado se descarte en lugar de crear categorías inexistentes |
| `test_conocimiento.py` | Que `dilemas.json` cumpla su esquema y que una base corrupta degrade sin tumbar la aplicación |
| `test_ai_factory.py` | Selección de proveedor y regresión del fallo de secretos ausentes |
| `test_pdf.py` | Generación de PDF, incrustación de gráficos y robustez ante caracteres que rompen el XML de ReportLab |
| `test_app.py` | Integración del punto de entrada con `AppTest`: que ambas rutas rendericen y que sin credenciales se avise en vez de romper |

Las pruebas corren deliberadamente **sin** `secrets.toml`, para verificar el camino de
degradación. No cubren el login real contra Firebase, la persistencia en Firestore ni las
llamadas efectivas a los modelos: eso requiere credenciales reales.

---

## Uso

1. **Inicio de sesión.** Registro y autenticación contra Firebase.
2. **Análisis previo (opcional).** Pega la historia clínica; la IA extrae los elementos
   bioéticos y sugiere un dilema del catálogo. El texto se anonimiza antes de enviarse.
3. **Registro del caso.** Datos del paciente, dilema, descripción, contexto sociocultural
   y la ponderación multiperspectiva de los cuatro principios (0–5).
4. **Dashboard.** Semáforo ético, visualizaciones comparativas y análisis de consenso.
5. **Análisis deliberativo.** La IA elabora el análisis del comité, citando únicamente la
   base normativa verificada del sistema.
6. **Descargas.** Reporte deliberativo y consentimiento informado en PDF, con los
   gráficos incrustados y la trazabilidad del modelo utilizado.
7. **Asistente de bioética.** Chatbot contextual sobre el caso activo, con preguntas
   guiadas para la deliberación.
8. **Consultar casos.** Casos guardados del usuario, con paginación.

---

## Autores

- **Anderson Díaz Pérez** — Creador y titular de los derechos de autor de
  BioEthicCare360®. Doctor en Bioética, Doctor en Salud Pública, Magíster en Ciencias
  Básicas Biomédicas (énfasis en Inmunología), Especialista en Inteligencia Artificial.
- **Joseph Javier Sánchez Acuña** — Creador de la App Web. Ingeniero Industrial, Experto
  en Inteligencia Artificial y Desarrollo de Software.

📂 [Repositorio en GitHub](https://github.com/GIUSEPPESAN21) · 📧 joseph.sanchez@uniminuto.edu.co

---

## Aviso de uso

Esta herramienta **apoya** la deliberación bioética; **no sustituye** la decisión del
comité de bioética ni del equipo tratante, y no emite diagnósticos médicos ni
indicaciones terapéuticas.
