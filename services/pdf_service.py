"""
services/pdf_service.py
=======================
Generación de PDFs (reporte deliberativo y consentimiento informado).

Mejora M4
---------
El reporte ahora INCRUSTA los gráficos. Antes el PDF contenía solo la frase
"los gráficos se muestran de forma interactiva en la aplicación web", lo que dejaba
sin evidencia visual al entregable formal del comité.

Kaleido ≥ 1.0 necesita un Chrome/Chromium instalado: `packages.txt` lo declara para
Streamlit Community Cloud y el devcontainer; en otros entornos puede indicarse su ruta
con la variable `BROWSER_PATH`.

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
    Table,
    TableStyle,
)

from core.indicadores import resumen_indicadores_texto
from core.modelos import safe_str
from services.charts import (
    figura_contribuciones,
    figura_divergencia,
    figura_distribucion_severidad,
    figura_matriz_calibracion,
    figura_paridad,
    generar_todas_las_figuras,
)

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
        # Plotly ≥ 6 retiró el argumento `engine` (kaleido es el único motor) y lanzaba
        # TypeError, así que ningún gráfico llegaba al PDF. Se llama sin él y solo se
        # reintenta con `engine` en versiones antiguas que lo necesiten.
        try:
            png_bytes = fig.to_image(format="png", scale=ESCALA_EXPORT)
        except TypeError:  # pragma: no cover - Plotly < 5 sin motor por defecto
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

    figuras["divergencia"] = figura_divergencia(perspectivas)
    figuras["contribuciones"] = figura_contribuciones(
        (data.get("AnalisisEtico") or {}).get("xai")
    )

    titulos = {
        "radar": "Ponderación por Perspectiva",
        "consenso": "Análisis de Consenso y Disenso",
        "equilibrio": "Análisis Comparativo de Principios",
        "divergencia": "Mapa de Divergencia entre Perspectivas",
        "contribuciones": "Atribución de la Severidad por Hallazgo (XAI)",
    }

    incrustadas = 0
    for clave in ("radar", "consenso", "equilibrio", "divergencia", "contribuciones"):
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


def _seccion_xai(xai: Optional[Dict[str, Any]], s: Dict[str, ParagraphStyle]) -> List[Any]:
    """Explicación del semáforo: atribución por hallazgo y contrafactuales."""
    if not xai or not xai.get("hallazgos"):
        return []
    story: List[Any] = [
        Paragraph("Explicación del Semáforo (IA Explicable)", s["h3"]),
        Paragraph(
            f"Puntos de severidad: <b>{safe_str(xai.get('puntos_severidad'))}</b> "
            f"(motor {html.escape(safe_str(xai.get('version_motor')))}). Cada punto es "
            "atribuible a una regla explícita del motor determinista.",
            s["body_left"],
        ),
    ]
    filas = [["Contribución", "Puntos", "Severidad si se resuelve"]]
    for c in xai.get("contrafactuales") or []:
        filas.append([
            safe_str(c.get("etiqueta")),
            safe_str(c.get("puntos_que_aporta")),
            safe_str(c.get("severidad_si_se_resuelve")),
        ])
    story.append(_tabla(filas))
    story.append(Spacer(1, 0.1 * inch))
    return story


def _tabla(filas: List[List[Any]], anchos: Optional[List[float]] = None) -> Table:
    """Tabla con estilo uniforme; el contenido se escapa para no romper el XML."""
    estilo_celda = ParagraphStyle(name="Celda", fontSize=8, fontName="Helvetica", leading=10)
    estilo_cab = ParagraphStyle(
        name="Cab", fontSize=8, fontName="Helvetica-Bold", leading=10, textColor=colors.white
    )
    datos = [
        [Paragraph(html.escape(safe_str(c)), estilo_cab if i == 0 else estilo_celda) for c in fila]
        for i, fila in enumerate(filas)
    ]
    tabla = Table(datos, colWidths=anchos, repeatRows=1)
    tabla.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return tabla


# --- Reporte deliberativo completo -------------------------------------------------

ORDEN_CAMPOS = (
    "ID del Caso",
    "Fecha Análisis",
    "Analista",
    "Resumen del Paciente",
    "Dominio Clínico",
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

            story.extend(_seccion_xai(analisis.get("xai"), s))

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

        indicadores = data.get("IndicadoresDeliberacion") or {}
        if indicadores:
            story.append(Paragraph("Indicadores de Deliberación", s["h2"]))
            for linea in resumen_indicadores_texto(indicadores):
                story.append(Paragraph(f"• {_md_a_pdf(linea)}", s["body_left"]))
        if data.get("Tiempo de Deliberación (s)"):
            minutos = safe_str(round(float(data["Tiempo de Deliberación (s)"]) / 60, 1))
            story.append(
                Paragraph(f"<b>Tiempo de registro y deliberación:</b> {minutos} min", s["body_left"])
            )

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

        validacion = data.get("ValidacionExperta") or {}
        if validacion:
            story.append(Paragraph("Validación Experta del Análisis", s["h2"]))
            for etiqueta, clave in (
                ("Utilidad", "utilidad"), ("Pertinencia", "pertinencia"),
                ("Fundamentación normativa", "fundamentacion"), ("Claridad", "claridad"),
            ):
                if validacion.get(clave):
                    story.append(Paragraph(
                        f"<b>{etiqueta}:</b> {html.escape(safe_str(validacion[clave]))}/5", s["body_left"]
                    ))
            if "aceptable" in validacion:
                story.append(Paragraph(
                    "<b>Recomendación aceptable:</b> " + ("Sí" if validacion["aceptable"] else "No"),
                    s["body_left"],
                ))
            if validacion.get("comentario"):
                story.append(Paragraph(_md_a_pdf(validacion["comentario"]), s["body"]))

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


# --- Informe de análisis de datos y auditoría de sesgos (Producto 1 DeliberIA) ------

def _fmt(valor: Any, decimales: int = 2, porcentaje: bool = False) -> str:
    if valor in (None, ""):
        return "—"
    try:
        v = float(valor)
    except (TypeError, ValueError):
        return safe_str(valor)
    return f"{v:.{decimales - 2 if decimales > 2 else 0}%}" if porcentaje else f"{v:.{decimales}f}"


def crear_informe_investigacion_pdf(
    resumen: Dict[str, Any],
    resumen_por_dominio: Dict[str, Dict[str, Any]],
    auditoria: Dict[str, Any],
    calidad: Dict[str, Any],
    filename: str,
    sus: Optional[Dict[str, Any]] = None,
    origen: str = "",
) -> str:
    """
    Genera el "Informe de análisis de datos y evidencia" del proyecto DeliberIA:
    indicadores de consenso/disenso/desempeño, auditoría formal de sesgos, calidad de
    la base y usabilidad. Solo recibe agregados SIN PII.
    """
    from datetime import datetime, timezone

    s = _estilos()
    doc = SimpleDocTemplate(filename, pagesize=letter, topMargin=inch / 2, bottomMargin=inch / 2)
    story: List[Any] = [
        Paragraph("Informe de Análisis de Datos y Auditoría de Sesgos", s["h1"]),
        Paragraph(
            "DeliberIA — BIOETHICARE 360º · Motor de deliberación ética computacional "
            f"· Generado {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC",
            s["nota"],
        ),
    ]
    if origen:
        story.append(Paragraph(f"<b>Origen de los datos:</b> {html.escape(origen)}", s["body_left"]))

    # 1. Resumen ejecutivo
    story.append(Paragraph("1. Resumen de indicadores", s["h2"]))
    consenso = resumen.get("consenso") or {}
    w = resumen.get("w_kendall") or {}
    tiempo = resumen.get("tiempo_deliberacion_min") or {}
    story.append(_tabla([
        ["Indicador", "Valor"],
        ["Casos analizados", safe_str(resumen.get("n"))],
        ["Proporción con severidad elevada (Moderado/Crítico)", _fmt(resumen.get("proporcion_severidad_elevada"), 3, True)],
        ["Índice de consenso medio (DE)", f"{_fmt(consenso.get('media'))} ({_fmt(consenso.get('desviacion'))})"],
        ["W de Kendall media", _fmt(w.get("media"))],
        ["Tiempo de deliberación, mediana (min)", _fmt(tiempo.get("mediana"), 1)],
        ["Concordancia dilema IA–comité", _fmt(resumen.get("concordancia_dilema_ia"), 3, True)],
        ["Concordancia riesgo IA–motor", _fmt(resumen.get("concordancia_riesgo_ia"), 3, True)],
        ["Recomendaciones aceptadas por expertos", _fmt(resumen.get("proporcion_recomendacion_aceptable"), 3, True)],
    ], anchos=[4.2 * inch, 2.3 * inch]))

    # 2. Por dominio clínico
    if resumen_por_dominio:
        story.append(Paragraph("2. Indicadores por dominio clínico", s["h2"]))
        filas = [["Dominio", "n", "% elevada", "Consenso", "W Kendall", "Tiempo med. (min)"]]
        for dominio, r in resumen_por_dominio.items():
            filas.append([
                dominio, safe_str(r.get("n")),
                _fmt(r.get("proporcion_severidad_elevada"), 3, True),
                _fmt((r.get("consenso") or {}).get("media")),
                _fmt((r.get("w_kendall") or {}).get("media")),
                _fmt((r.get("tiempo_deliberacion_min") or {}).get("mediana"), 1),
            ])
        story.append(_tabla(filas))
        img = figura_a_imagen(figura_distribucion_severidad(resumen_por_dominio), ancho=5.5 * inch)
        if img is not None:
            story.append(img)

    # 3. Auditoría de sesgos
    story.append(PageBreak())
    story.append(Paragraph("3. Auditoría formal de sesgos algorítmicos", s["h2"]))
    params = auditoria.get("parametros") or {}
    story.append(Paragraph(
        f"Protocolo <b>{html.escape(safe_str(auditoria.get('version_protocolo')))}</b>. "
        f"Veredicto global: <b>{html.escape(safe_str(auditoria.get('veredicto_global')))}</b>. "
        f"Subgrupos con n &lt; {safe_str(params.get('n_minimo_subgrupo'))} se excluyen de las "
        f"comparaciones; regla de los cuatro quintos = {safe_str(params.get('regla_cuatro_quintos'))}; "
        f"α = {safe_str(params.get('alfa'))}. Una disparidad es 'Alerta' solo si es práctica y "
        "estadísticamente significativa; si es solo una de las dos, 'Vigilar'.",
        s["body"],
    ))
    for titulo, clave in (("Alertas", "alertas"), ("Vigilar", "vigilancias")):
        items = auditoria.get(clave) or []
        if items:
            story.append(Paragraph(titulo, s["h3"]))
            for item in items:
                story.append(Paragraph(f"• {_md_a_pdf(item)}", s["body_left"]))

    story.append(Paragraph("3.1 Paridad de severidad elevada entre subgrupos", s["h3"]))
    filas = [["Atributo", "Dif. paridad", "Razón impacto", "χ² (gl)", "p", "Veredicto"]]
    for p in auditoria.get("paridad_severidad") or []:
        chi = p.get("chi2") or {}
        filas.append([
            p.get("etiqueta"), _fmt(p.get("diferencia_paridad")), _fmt(p.get("razon_impacto_dispar")),
            f"{_fmt(chi.get('chi2'))} ({safe_str(chi.get('gl', '—'))})" if chi else "—",
            _fmt(chi.get("p_valor"), 3) if chi else "—", p.get("veredicto"),
        ])
    story.append(_tabla(filas))
    for p in auditoria.get("paridad_severidad") or []:
        img = figura_a_imagen(figura_paridad(p), ancho=5 * inch)
        if img is not None:
            story.append(img)

    story.append(Paragraph("3.2 Paridad del consenso entre subgrupos", s["h3"]))
    filas = [["Atributo", "Brecha de consenso", "Veredicto"]]
    for c in auditoria.get("paridad_consenso") or []:
        filas.append([c.get("etiqueta"), _fmt(c.get("brecha")), c.get("veredicto")])
    story.append(_tabla(filas))

    cal = auditoria.get("calibracion_ia") or {}
    story.append(Paragraph("3.3 Calibración de la IA frente al motor determinista", s["h3"]))
    story.append(_tabla([
        ["Métrica", "Valor"],
        ["Pares comparados", safe_str(cal.get("n_pares_riesgo"))],
        ["Exactitud", _fmt(cal.get("exactitud"), 3, True)],
        ["Kappa de Cohen ponderado", _fmt(cal.get("kappa_ponderado"))],
        ["Tasa de sobreestimación", _fmt(cal.get("tasa_sobreestimacion"), 3, True)],
        ["Tasa de subestimación", _fmt(cal.get("tasa_subestimacion"), 3, True)],
        ["Concordancia de dilema IA–comité", _fmt(cal.get("concordancia_dilema"), 3, True)],
        ["Veredicto", safe_str(cal.get("veredicto"))],
    ], anchos=[4.2 * inch, 2.3 * inch]))
    img = figura_a_imagen(figura_matriz_calibracion(cal), ancho=4.5 * inch)
    if img is not None:
        story.append(img)

    story.append(Paragraph("3.4 Paridad del desempeño de la IA entre subgrupos", s["h3"]))
    filas = [["Atributo", "Brecha de exactitud", "Veredicto"]]
    for c in auditoria.get("paridad_calibracion") or []:
        filas.append([c.get("etiqueta"), _fmt(c.get("brecha_exactitud")), c.get("veredicto")])
    story.append(_tabla(filas))
    story.append(Paragraph(
        "Nota de interpretación: el semáforo es determinista y no recibe género, edad ni "
        "dominio. Una disparidad en 3.1 o 3.2 refleja patrones en las ponderaciones que "
        "registran las perspectivas (deliberación humana), no una regla del algoritmo. "
        "Las secciones 3.3 y 3.4 sí auditan al modelo de IA.",
        s["nota"],
    ))

    # 4. Calidad de la base
    story.append(Paragraph("4. Calidad de la base de datos", s["h2"]))
    story.append(_tabla([
        ["Control", "Resultado"],
        ["Registros", safe_str(calidad.get("n_registros"))],
        ["Seudónimos duplicados", safe_str(calidad.get("duplicados"))],
        ["Registros con perspectiva omitida", safe_str(calidad.get("registros_con_perspectiva_omitida"))],
        [f"k-anonimato (mínimo requerido {safe_str(calidad.get('k_minimo_requerido'))})", safe_str(calidad.get("k_anonimato"))],
        ["Apto para apertura (control automático)", "Sí" if calidad.get("apto_para_apertura") else "No"],
    ], anchos=[4.2 * inch, 2.3 * inch]))
    for obs in calidad.get("observaciones") or []:
        story.append(Paragraph(f"• {_md_a_pdf(obs)}", s["body_left"]))

    # 5. Usabilidad
    if sus and sus.get("n"):
        interp = sus.get("interpretacion") or {}
        story.append(Paragraph("5. Validación de usabilidad (SUS)", s["h2"]))
        ic = sus.get("ic95")
        story.append(Paragraph(
            f"n = {sus['n']}; media = <b>{_fmt(sus.get('media'), 1)}</b> "
            f"(DE {_fmt(sus.get('desviacion'), 1)}"
            + (f"; IC95% {ic[0]}–{ic[1]}" if ic else "")
            + f"). Adjetivo: {html.escape(safe_str(interp.get('adjetivo')))}; aceptabilidad: "
            f"{html.escape(safe_str(interp.get('aceptabilidad')))}; calificación: "
            f"{html.escape(safe_str(interp.get('calificacion')))} (referencia 68).",
            s["body"],
        ))

    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(
        "Este informe contiene solo agregados de registros seudonimizados. Apoya la "
        "investigación en bioética computacional; no sustituye la decisión del comité de "
        "ética ni del equipo tratante.",
        s["nota"],
    ))
    doc.build(story)
    logger.info("Informe de investigación generado: %s", filename)
    return filename
