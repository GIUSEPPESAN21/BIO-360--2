"""
services/pdf_service.py
=======================
Generación de PDFs (reporte deliberativo y consentimiento informado).

Mejora M4
---------
El reporte ahora INCRUSTA los gráficos. Antes el PDF contenía solo la frase
"los gráficos se muestran de forma interactiva en la aplicación web", lo que dejaba
sin evidencia visual al entregable formal del comité.

Se usa `kaleido` (ya presente en requirements.txt) para convertir cada figura Plotly
a PNG **en memoria** (`fig.to_image`), sin escribir archivos temporales, y se incrusta
con `reportlab.platypus.Image`. Si kaleido no está disponible en el entorno, el PDF se
genera igual con una nota explícita (degradación elegante, no error).
"""

from __future__ import annotations

import html
import io
import logging
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

from core.modelos import safe_str
from services.charts import generar_todas_las_figuras

logger = logging.getLogger(__name__)

ANCHO_GRAFICO = 6.5 * inch
ESCALA_EXPORT = 2  # densidad de píxeles para nitidez en impresión


# --- Estilos ----------------------------------------------------------------------

def _estilos() -> Dict[str, ParagraphStyle]:
    return {
        "h1": ParagraphStyle(
            name="H1", fontSize=18, fontName="Helvetica-Bold",
            alignment=TA_CENTER, spaceAfter=20,
        ),
        "h2": ParagraphStyle(
            name="H2", fontSize=14, fontName="Helvetica-Bold",
            spaceBefore=12, spaceAfter=6, textColor=colors.darkblue,
        ),
        "h3": ParagraphStyle(
            name="H3", fontSize=11, fontName="Helvetica-Bold",
            spaceBefore=8, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            name="Body", fontSize=10, fontName="Helvetica", leading=14,
            alignment=TA_JUSTIFY, spaceAfter=10,
        ),
        "body_left": ParagraphStyle(
            name="BodyLeft", fontSize=10, fontName="Helvetica", leading=14,
            alignment=TA_LEFT, spaceAfter=8,
        ),
        "chat": ParagraphStyle(
            name="Chat", fontSize=9, fontName="Helvetica-Oblique",
            backColor=colors.whitesmoke, borderWidth=1, padding=5, spaceAfter=6,
        ),
        "nota": ParagraphStyle(
            name="Nota", fontSize=8, fontName="Helvetica-Oblique",
            textColor=colors.grey, spaceAfter=6,
        ),
    }


def _md_a_pdf(texto: str) -> str:
    """
    Convierte el markdown ligero usado por la app a las marcas que entiende ReportLab.

    Escapa primero el texto para evitar que contenido del usuario rompa el XML
    interno de ReportLab (un `<` suelto en una historia clínica hacía fallar el build).
    """
    seguro = html.escape(safe_str(texto), quote=False)
    # **negrita** -> <b>negrita</b>
    import re

    seguro = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", seguro, flags=re.DOTALL)
    seguro = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", seguro, flags=re.DOTALL)
    seguro = seguro.replace("`", "")
    return seguro.replace("\n", "<br/>")


# --- Exportación de figuras a PNG en memoria (M4) ---------------------------------

def figura_a_imagen(fig: Any, ancho: float = ANCHO_GRAFICO) -> Optional[Image]:
    """
    Convierte una figura Plotly a un flowable `Image` de ReportLab usando kaleido,
    íntegramente en memoria. Devuelve None si la conversión no es posible.
    """
    if fig is None:
        return None
    try:
        png_bytes = fig.to_image(format="png", scale=ESCALA_EXPORT, engine="kaleido")
    except Exception as e:
        logger.warning("No se pudo exportar la figura a PNG con kaleido: %s", e)
        return None

    try:
        buffer = io.BytesIO(png_bytes)
        # Preservar proporción original de la imagen exportada
        from reportlab.lib.utils import ImageReader

        lector = ImageReader(io.BytesIO(png_bytes))
        px_w, px_h = lector.getSize()
        alto = ancho * (px_h / px_w) if px_w else ancho * 0.6
        return Image(buffer, width=ancho, height=alto)
    except Exception as e:
        logger.warning("No se pudo construir el flowable de imagen: %s", e)
        return None


def _seccion_visualizaciones(data: Dict[str, Any], st_: Dict[str, ParagraphStyle]) -> List[Any]:
    """
    Construye la sección de visualizaciones del PDF incrustando los gráficos.

    Las figuras se REGENERAN desde las ponderaciones del reporte (M5), por lo que
    funciona tanto para casos nuevos como para casos guardados con el esquema antiguo.
    """
    story: List[Any] = [PageBreak(), Paragraph("Visualizaciones de Datos", st_["h1"])]

    perspectivas = data.get("AnalisisMultiperspectiva") or {}
    figuras = generar_todas_las_figuras(perspectivas)

    titulos = {
        "radar": "Ponderación por Perspectiva",
        "consenso": "Análisis de Consenso y Disenso",
        "equilibrio": "Análisis Comparativo de Principios",
    }

    incrustadas = 0
    for clave in ("radar", "consenso", "equilibrio"):
        img = figura_a_imagen(figuras.get(clave))
        if img is not None:
            story.append(Paragraph(titulos[clave], st_["h3"]))
            story.append(img)
            story.append(Spacer(1, 0.2 * inch))
            incrustadas += 1

    if incrustadas == 0:
        story.append(
            Paragraph(
                "No fue posible renderizar los gráficos en este entorno "
                "(motor de exportación kaleido no disponible). Las visualizaciones "
                "permanecen consultables de forma interactiva en la aplicación web.",
                st_["nota"],
            )
        )
    return story


# --- Reporte deliberativo completo -------------------------------------------------

ORDEN_CAMPOS = (
    "ID del Caso",
    "Fecha Análisis",
    "Analista",
    "Resumen del Paciente",
    "Dilema Ético Principal (Seleccionado)",
    "Dilema Sugerido por IA",
    "Descripción Detallada del Caso",
    "Contexto Sociocultural y Familiar",
    "Puntos Clave para Deliberación IA",
    "Análisis IA de Historia Clínica",
)


def crear_reporte_pdf_completo(data: Dict[str, Any], filename: str) -> str:
    """
    Genera el PDF del reporte deliberativo, incluyendo los gráficos incrustados (M4).

    `data` debe venir ya REHIDRATADO (con los datos reales del paciente): el PDF es un
    entregable para el usuario autorizado, no una carga para una API externa.
    """
    try:
        doc = SimpleDocTemplate(
            filename, pagesize=letter, topMargin=inch / 2, bottomMargin=inch / 2
        )
        s = _estilos()
        story: List[Any] = [Paragraph("Reporte Deliberativo - BIOETHICARE 360", s["h1"])]

        for key in ORDEN_CAMPOS:
            if data.get(key):
                story.append(Paragraph(html.escape(key), s["h2"]))
                story.append(Paragraph(_md_a_pdf(data[key]), s["body"]))

        analisis = data.get("AnalisisEtico") or {}
        if analisis:
            story.append(Paragraph("Análisis de Coherencia Ética", s["h2"]))
            story.append(
                Paragraph(
                    f"<b>Nivel de Severidad:</b> {html.escape(safe_str(analisis.get('severidad', 'N/A')))}",
                    s["body"],
                )
            )
            for adv in analisis.get("advertencias", []) or []:
                story.append(Paragraph(f"• {_md_a_pdf(adv)}", s["body_left"]))
            recomendaciones = analisis.get("recomendaciones", []) or []
            if recomendaciones:
                story.append(
                    Paragraph(
                        f"<b>Recomendaciones:</b> {_md_a_pdf(' '.join(recomendaciones))}",
                        s["body"],
                    )
                )

        multiperspectiva = data.get("AnalisisMultiperspectiva") or {}
        if multiperspectiva:
            story.append(Paragraph("Análisis Multiperspectiva", s["h2"]))
            for nombre, valores in multiperspectiva.items():
                if not isinstance(valores, dict):
                    continue
                texto = (
                    f"<b>{html.escape(safe_str(nombre))}:</b> "
                    f"Autonomía: {valores.get('autonomia', 0)}, "
                    f"Beneficencia: {valores.get('beneficencia', 0)}, "
                    f"No Maleficencia: {valores.get('no_maleficencia', 0)}, "
                    f"Justicia: {valores.get('justicia', 0)}"
                )
                story.append(Paragraph(texto, s["body"]))

        if data.get("Análisis Deliberativo (IA)"):
            story.append(Paragraph("Análisis Deliberativo (IA)", s["h2"]))
            story.append(Paragraph(_md_a_pdf(data["Análisis Deliberativo (IA)"]), s["body"]))

        # Trazabilidad del modelo usado (M8) si está disponible
        traza = data.get("Trazabilidad IA") or {}
        if traza:
            story.append(Paragraph("Trazabilidad del Análisis por IA", s["h2"]))
            for etiqueta, clave in (
                ("Proveedor", "proveedor"),
                ("Modelo", "modelo"),
                ("Versión de prompt", "prompt_version"),
                ("Fecha/hora (UTC)", "timestamp"),
            ):
                if traza.get(clave):
                    story.append(
                        Paragraph(
                            f"<b>{etiqueta}:</b> {html.escape(safe_str(traza[clave]))}",
                            s["body_left"],
                        )
                    )

        # --- Gráficos incrustados (M4) ---
        story.extend(_seccion_visualizaciones(data, s))

        chat = data.get("Historial del Chat de Deliberación") or []
        if chat:
            story.append(PageBreak())
            story.append(Paragraph("Historial del Chat de Deliberación", s["h1"]))
            for msg in chat:
                if not isinstance(msg, dict):
                    continue
                rol = safe_str(msg.get("role", "unknown")).capitalize()
                story.append(
                    Paragraph(
                        f"<b>{html.escape(rol)}:</b> {_md_a_pdf(msg.get('content'))}",
                        s["chat"],
                    )
                )

        doc.build(story)
        logger.info("PDF generado exitosamente: %s", filename)
        return filename
    except Exception as e:
        logger.error("Error generando PDF %s: %s", filename, e)
        raise


# --- Consentimiento informado ------------------------------------------------------

def crear_consentimiento_pdf(texto: str, filename: str) -> str:
    """Genera el PDF del consentimiento informado a partir del texto plano."""
    try:
        doc = SimpleDocTemplate(
            filename, pagesize=letter, topMargin=inch / 2, bottomMargin=inch / 2
        )
        s = _estilos()
        h1 = ParagraphStyle(
            name="C1", fontSize=14, fontName="Helvetica-Bold",
            alignment=TA_CENTER, spaceAfter=18,
        )
        h2 = ParagraphStyle(
            name="C2", fontSize=11, fontName="Helvetica-Bold",
            spaceBefore=10, spaceAfter=4, textColor=colors.darkblue,
        )
        story: List[Any] = []

        for line in (texto or "").split("\n"):
            desnuda = line.strip()
            if desnuda and desnuda.isupper() and not desnuda.startswith("-"):
                if "CONSENTIMIENTO" in desnuda:
                    story.append(Paragraph(html.escape(desnuda), h1))
                else:
                    story.append(Spacer(1, 0.1 * inch))
                    story.append(Paragraph(html.escape(desnuda), h2))
                    story.append(HRFlowable(width="100%", thickness=1, color=colors.black))
            else:
                story.append(Paragraph(_md_a_pdf(line), s["body_left"]))

        doc.build(story)
        logger.info("PDF de consentimiento generado: %s", filename)
        return filename
    except Exception as e:
        logger.error("Error generando PDF de consentimiento %s: %s", filename, e)
        raise
