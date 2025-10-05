🏥 BIOETHICARE 360º - Versión 2.1 Fusionada y Optimizada
Software de Análisis Bioético Avanzado con Inteligencia Artificial, Autenticación y Base de Datos

Este proyecto combina la funcionalidad completa de la aplicación original de GitHub con el motor de IA optimizado de la versión 2.0, solucionando el error 404 de Gemini y añadiendo mayor robustez.

✅ Mejoras Clave en esta Versión
Motor de IA Optimizado: Se implementó la función llamar_gemini mejorada que:

Utiliza modelos actualizados como Gemini 2.0 Flash Experimental.

Implementa un fallback automático entre varios modelos de Gemini para garantizar la disponibilidad.

Tiene una configuración de parámetros optimizada para el análisis bioético.

Funcionalidad Completa Conservada: Mantiene todas las características del proyecto original:

Autenticación de usuarios con Firebase.

Almacenamiento y consulta de casos en la base de datos Firestore.

Análisis multiperspectiva detallado.

Generación de reportes completos y consentimientos informados en formato PDF.

Visualizaciones de datos avanzadas.

Interfaz Mejorada: Se ha añadido una nueva pestaña de "Información del Sistema" y se ha mejorado el sidebar para mostrar el estado del modelo de IA en tiempo real.

🚀 Instalación y Ejecución
1. Instalar Dependencias
Asegúrate de tener todas las librerías necesarias.

pip install -r requirements.txt

2. Configurar Credenciales (secrets.toml)
Crea un archivo llamado secrets.toml dentro de una carpeta .streamlit en la raíz de tu proyecto. Este archivo contendrá tus claves de API y las credenciales de Firebase.

Utiliza el archivo secrets_template.toml como guía.

3. Ejecutar la Aplicación
Una vez configurado, ejecuta la aplicación con Streamlit.

streamlit run app.py

🔧 Solución del Error 404 de Gemini
❌ Problema Original: El código anterior usaba modelos de Gemini que quedaron obsoletos (gemini-1.5-pro-latest), lo que provocaba un error 404 Not Found.

✅ Solución Implementada: La nueva función llamar_gemini ahora utiliza una lista de modelos actuales y los prueba en orden de preferencia. Si el modelo más avanzado falla o es bloqueado, la aplicación pasa automáticamente al siguiente, garantizando una alta disponibilidad.

🎯 Modelos en Orden de Preferencia
gemini-2.0-flash-exp ⭐ (Más avanzado)

gemini-1.5-pro-001 (Más estable)

gemini-1.5-flash-001 (Más rápido)

gemini-1.5-flash (Básico)

📋 Uso de la Aplicación
Inicio de Sesión: Regístrate o inicia sesión con tus credenciales. La aplicación se conecta a Firebase para gestionar los usuarios.

Análisis de Caso:

Usa el análisis previo de IA para extraer puntos clave de una historia clínica.

Completa el formulario detallado del caso y las ponderaciones multiperspectiva.

Genera un dashboard interactivo con visualizaciones y análisis de coherencia ética.

Descarga el reporte completo y el consentimiento informado en PDF.

Asistente de Bioética: Utiliza el chatbot contextual para deliberar sobre el caso activo, con preguntas guiadas para facilitar el análisis.

Consultar Casos: Accede a todos los casos que has analizado y guardado previamente en tu cuenta.
