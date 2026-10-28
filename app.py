"""
TRABAJO FINAL INTEGRADOR — EDI II (Ciencias de Datos)
Proyecto: Lector inteligente de comprobantes y facturas
Integrantes: Alan Bustamante, Christian Martínez

Descripción:
Aplicación web en Streamlit con arquitectura híbrida (Ollama Local + API en la Nube)
para la lectura mediante OCR (Tesseract), extracción estructurada de los 9 campos
definidos en la propuesta, revisión manual, exportación (CSV/JSON) y evaluación
de exactitud frente a una transcripción de referencia (Ground Truth).
"""

import json
import os
import re
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pytesseract
import requests
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# ==============================================================================
# 1. CONFIGURACIÓN GENERAL Y ESQUEMA DE CAMPOS DEL PROYECTO
# ==============================================================================

CAMPOS_COMPROBANTE = {
    "tipo_documento": "Tipo de documento",
    "emisor": "Nombre o razón social del emisor",
    "cuit": "CUIT del emisor",
    "fecha_emision": "Fecha de emisión",
    "numero_comprobante": "Número de comprobante",
    "importe_neto": "Importe neto o subtotal",
    "impuestos_iva": "IVA u otros impuestos discriminados",
    "importe_total": "Importe total",
    "moneda": "Moneda",
}

VALOR_NO_IDENTIFICADO = "No identificado"

PROMPT_SISTEMA = """Eres un asistente especializado en extracción contable de comprobantes, tickets y facturas de Argentina.
Tu única tarea es analizar el texto extraído por OCR de un comprobante y devolver EXCLUSIVAMENTE un objeto JSON válido con las siguientes 9 claves exactas:

{
  "tipo_documento": "Ej: Factura A, Factura B, Factura C, Ticket, Recibo o 'No identificado'",
  "emisor": "Nombre o razón social del comercio/emisor o 'No identificado'",
  "cuit": "CUIT del emisor con formato XX-XXXXXXXX-X o 'No identificado'",
  "fecha_emision": "Fecha en formato DD/MM/AAAA o 'No identificado'",
  "numero_comprobante": "Número de factura o ticket (ej: 0001-00012345) o 'No identificado'",
  "importe_neto": "Subtotal o importe neto numérico (ej: 12500.50) o 'No identificado'",
  "impuestos_iva": "Monto de IVA u otros impuestos (ej: 2625.10) o 'No identificado'",
  "importe_total": "Importe total final numérico (ej: 15125.60) o 'No identificado'",
  "moneda": "Moneda identificada (ej: ARS, USD, $) o 'No identificado'"
}

REGLAS ESTRICTAS:
1. NO inventes ni supongas ningún dato que no figure en el texto OCR.
2. Si un campo no aparece o es ilegible, coloca exactamente el texto "No identificado".
3. Devuelve ÚNICAMENTE el objeto JSON, sin explicaciones ni bloques markdown adicionales.
"""


# ==============================================================================
# 2. MÓDULO DE RECONOCIMIENTO ÓPTICO DE CARACTERES (TESSERACT OCR)
# ==============================================================================

def configurar_ruta_tesseract() -> Optional[str]:
    """Detecta automáticamente el ejecutable de Tesseract en Windows, Linux o Mac."""
    ruta_en_path = shutil.which("tesseract")
    if ruta_en_path:
        pytesseract.pytesseract.tesseract_cmd = ruta_en_path
        return ruta_en_path

    rutas_comunes_windows = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    for ruta in rutas_comunes_windows:
        if os.path.exists(ruta):
            pytesseract.pytesseract.tesseract_cmd = ruta
            return ruta
    return None


def preprocesar_imagen(imagen: Image.Image, aplicar_mejora: bool = True) -> Image.Image:
    """
    Aplica mejoras de imagen (escala de grises, autocontraste y nitidez)
    para reducir errores de lectura en tickets térmicos o fotos de celular.
    """
    img_rgb = imagen.convert("RGB")
    if not aplicar_mejora:
        return img_rgb

    # Conversión a escala de grises
    gris = ImageOps.grayscale(img_rgb)
    # Estiramiento de contraste automático
    gris = ImageOps.autocontrast(gris, cutoff=1)
    # Aumento de nitidez y contraste para resaltar caracteres borrosos
    realce_contraste = ImageEnhance.Contrast(gris).enhance(1.6)
    realce_nitidez = realce_contraste.filter(ImageFilter.SHARPEN)
    return realce_nitidez


def ejecutar_ocr(imagen: Image.Image, idioma: str = "spa") -> Tuple[str, Optional[str]]:
    """
    Ejecuta Tesseract OCR sobre la imagen suministrada.
    Si el paquete 'spa' no está instalado, reintenta automáticamente con 'eng'.
    """
    try:
        config_psm = "--oem 3 --psm 6"
        texto = pytesseract.image_to_string(imagen, lang=idioma, config=config_psm)
        return texto.strip(), None
    except pytesseract.TesseractError as err:
        if idioma != "eng":
            try:
                texto = pytesseract.image_to_string(imagen, lang="eng", config="--oem 3 --psm 6")
                aviso = (
                    "Se procesó con el paquete en inglés ('eng') porque no se detectó "
                    "el paquete 'spa' de Tesseract."
                )
                return texto.strip(), aviso
            except Exception as err_fallback:
                return "", f"Error al ejecutar Tesseract OCR: {err_fallback}"
        return "", f"Error de Tesseract OCR: {err}"
    except Exception as exc:
        return "", f"No se pudo ejecutar Tesseract OCR. Verifica su instalación. Detalle: {exc}"


# ==============================================================================
# 3. MÓDULO DE EXTRACCIÓN HÍBRIDA (OLLAMA LOCAL + NUBE + RESPALDO REGEX)
# ==============================================================================

def obtener_modelos_ollama(url_base: str = "http://localhost:11434") -> Tuple[List[str], Optional[str]]:
    """
    Consulta automáticamente la API local de Ollama para listar los modelos instalados
    sin que el usuario tenga que recordar el nombre o la versión.
    """
    try:
        respuesta = requests.get(f"{url_base.rstrip('/')}/api/tags", timeout=3)
        respuesta.raise_for_status()
        datos = respuesta.json()
        modelos = [m.get("name", "") for m in datos.get("models", []) if m.get("name")]
        return modelos, None
    except requests.exceptions.ConnectionError:
        return [], "No se detectó el servidor local de Ollama en ejecución (http://localhost:11434)."
    except Exception as exc:
        return [], f"Error al consultar Ollama: {exc}"


def normalizar_diccionario_extraido(datos_crudos: Dict) -> Dict[str, str]:
    """Garantiza que siempre existan los 9 campos esperados y reemplaza vacíos por 'No identificado'."""
    resultado = {}
    for clave in CAMPOS_COMPROBANTE:
        valor = datos_crudos.get(clave, VALOR_NO_IDENTIFICADO)
        if valor is None or str(valor).strip() == "" or str(valor).strip().lower() in {"null", "none", "n/a"}:
            resultado[clave] = VALOR_NO_IDENTIFICADO
        else:
            resultado[clave] = str(valor).strip()
    return resultado


def extraer_json_de_respuesta(texto_respuesta: str) -> Dict:
    """Extrae de forma segura un bloque JSON desde la respuesta del modelo de lenguaje."""
    try:
        return json.loads(texto_respuesta)
    except json.JSONDecodeError:
        coincidencia = re.search(r"\{.*\}", texto_respuesta, re.DOTALL)
        if coincidencia:
            try:
                return json.loads(coincidencia.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def extraer_con_ollama_local(
    texto_ocr: str,
    modelo: str,
    url_base: str = "http://localhost:11434",
) -> Tuple[Dict[str, str], Optional[str]]:
    """Envía el texto OCR al servidor local de Ollama y devuelve los 9 campos estructurados."""
    endpoint = f"{url_base.rstrip('/')}/api/generate"
    payload = {
        "model": modelo,
        "prompt": f"{PROMPT_SISTEMA}\n\nTEXTO OCR DEL COMPROBANTE:\n{texto_ocr}",
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.0},
    }
    try:
        resp = requests.post(endpoint, json=payload, timeout=90)
        resp.raise_for_status()
        contenido = resp.json().get("response", "{}")
        datos = extraer_json_de_respuesta(contenido)
        return normalizar_diccionario_extraido(datos), None
    except Exception as exc:
        return normalizar_diccionario_extraido({}), f"Error en Ollama local ({modelo}): {exc}"


def extraer_con_api_nube(
    texto_ocr: str,
    api_key: str,
    modelo: str = "llama-3.3-70b-versatile",
    endpoint_url: str = "https://api.groq.com/openai/v1/chat/completions",
) -> Tuple[Dict[str, str], Optional[str]]:
    """
    Envía el texto OCR a una API en la nube compatible con OpenAI (como Groq o OpenRouter)
    para permitir el uso desde cualquier celular en internet sin depender de hardware local.
    """
    if not api_key.strip():
        return normalizar_diccionario_extraido({}), "Falta ingresar la clave de API (API Key) para el modo Nube."

    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": modelo,
        "messages": [
            {"role": "system", "content": PROMPT_SISTEMA},
            {"role": "user", "content": f"TEXTO OCR DEL COMPROBANTE:\n{texto_ocr}"},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    try:
        resp = requests.post(endpoint_url, headers=headers, json=payload, timeout=45)
        resp.raise_for_status()
        contenido = resp.json()["choices"][0]["message"]["content"]
        datos = extraer_json_de_respuesta(contenido)
        return normalizar_diccionario_extraido(datos), None
    except Exception as exc:
        return normalizar_diccionario_extraido({}), f"Error en API Nube ({modelo}): {exc}"


def extraer_con_reglas_regex(texto_ocr: str) -> Dict[str, str]:
    """
    Extractor heurístico de respaldo basado en expresiones regulares.
    También se utiliza como refuerzo post-procesamiento para corregir CUIT y fechas.
    """
    datos = {clave: VALOR_NO_IDENTIFICADO for clave in CAMPOS_COMPROBANTE}

    # 1. Tipo de documento
    match_tipo = re.search(
        r"\b(FACTURA\s*[ABC]|TICKET\s*FACTURA\s*[ABC]|TICKET|RECIBO|NOTA\s*DE\s*CR[EÉ]DITO)\b",
        texto_ocr,
        re.IGNORECASE,
    )
    if match_tipo:
        datos["tipo_documento"] = match_tipo.group(1).upper()

    # 2. CUIT (formato XX-XXXXXXXX-X o 11 dígitos precedidos por CUIT)
    match_cuit = re.search(r"\b(20|23|24|27|30|33|34)[-\s]?(\d{8})[-\s]?(\d)\b", texto_ocr)
    if match_cuit:
        datos["cuit"] = f"{match_cuit.group(1)}-{match_cuit.group(2)}-{match_cuit.group(3)}"

    # 3. Fecha de emisión (DD/MM/AAAA o DD-MM-AAAA)
    match_fecha = re.search(r"\b(0?[1-9]|[12]\d|3[01])[/-](0?[1-9]|1[0-2])[/-](20\d{2}|\d{2})\b", texto_ocr)
    if match_fecha:
        dia, mes, anio = match_fecha.groups()
        anio = f"20{anio}" if len(anio) == 2 else anio
        datos["fecha_emision"] = f"{int(dia):02d}/{int(mes):02d}/{anio}"

    # 4. Número de comprobante (ej: 0001-00012345)
    match_nro = re.search(r"\b(\d{4,5})\s*[-–]\s*(\d{8})\b", texto_ocr)
    if match_nro:
        datos["numero_comprobante"] = f"{match_nro.group(1)}-{match_nro.group(2)}"

    # 5. Importe total
    match_total = re.search(
        r"TOTAL\s*(?:A\s*PAGAR)?\s*[:$]?\s*\$?\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\d+(?:[.,]\d{2}))",
        texto_ocr,
        re.IGNORECASE,
    )
    if match_total:
        datos["importe_total"] = match_total.group(1)

    # 6. Moneda
    if re.search(r"\b(USD|U\$S|D[OÓ]LARES)\b", texto_ocr, re.IGNORECASE):
        datos["moneda"] = "USD"
    elif "$" in texto_ocr or re.search(r"\b(ARS|PESOS)\b", texto_ocr, re.IGNORECASE):
        datos["moneda"] = "ARS"

    return datos


def aplicar_refuerzo_regex(datos_ia: Dict[str, str], texto_ocr: str) -> Dict[str, str]:
    """
    Mejora híbrida (Objetivo Específico 6): si el LLM omitió el CUIT, la fecha o el número
    de comprobante pero tienen un patrón exacto en el texto OCR, los completa con Regex.
    """
    respaldo = extraer_con_reglas_regex(texto_ocr)
    resultado = dict(datos_ia)
    for campo_clave in ["cuit", "fecha_emision", "numero_comprobante", "moneda"]:
        if resultado.get(campo_clave) == VALOR_NO_IDENTIFICADO and respaldo.get(campo_clave) != VALOR_NO_IDENTIFICADO:
            resultado[campo_clave] = respaldo[campo_clave]
    return resultado


# ==============================================================================
# 4. MÓDULO DE EVALUACIÓN DE EXACTITUD (COMPARACIÓN CON REFERENCIA MANUAL)
# ==============================================================================

def evaluar_extraccion(datos_extraidos: Dict[str, str], datos_referencia: Dict[str, str]) -> Tuple[pd.DataFrame, float]:
    """
    Compara cada campo extraído contra una transcripción manual de referencia (Ground Truth)
    y calcula el porcentaje de exactitud.
    """
    filas = []
    aciertos = 0
    total_campos = len(CAMPOS_COMPROBANTE)

    for clave, nombre_legible in CAMPOS_COMPROBANTE.items():
        val_ext = str(datos_extraidos.get(clave, VALOR_NO_IDENTIFICADO)).strip()
        val_ref = str(datos_referencia.get(clave, VALOR_NO_IDENTIFICADO)).strip()

        # Normalización suave para comparar montos e ignorar diferencias de mayúsculas/espacios
        norm_ext = val_ext.lower().replace("$", "").replace(" ", "")
        norm_ref = val_ref.lower().replace("$", "").replace(" ", "")

        coincide = norm_ext == norm_ref
        if coincide:
            aciertos += 1

        filas.append(
            {
                "Campo": nombre_legible,
                "Valor Extraído": val_ext,
                "Valor de Referencia (Real)": val_ref,
                "Resultado": "Correcto" if coincide else "Con diferencia",
            }
        )

    exactitud = round((aciertos / total_campos) * 100, 2) if total_campos > 0 else 0.0
    return pd.DataFrame(filas), exactitud


# ==============================================================================
# 5. INTERFAZ WEB EN STREAMLIT (COMPATIBLE CON PC Y NAVEGADOR MÓVIL)
# ==============================================================================

def inicializar_estado_sesion() -> None:
    if "texto_ocr" not in st.session_state:
        st.session_state["texto_ocr"] = ""
    if "datos_actuales" not in st.session_state:
        st.session_state["datos_actuales"] = {k: VALOR_NO_IDENTIFICADO for k in CAMPOS_COMPROBANTE}
    if "historial_comprobantes" not in st.session_state:
        st.session_state["historial_comprobantes"] = []


def aplicar_estilos_editoriales() -> None:
    """
    Inyecta CSS personalizado con estética editorial cálida, minimalista y luminosa
    (inspirada en el estilo 'Bloom Coaching') sin alterar la funcionalidad de la app.
    """
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400..700;1,9..40,400..700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        :root {
            --bg-main: #F7F4EF;
            --bg-sidebar: #EFECE6;
            --bg-card: #FFFFFF;
            --bg-card-warm: #FAF8F5;
            --border-subtle: #E2DDD5;
            --accent-blue: #87B1DC;
            --accent-blue-hover: #729ECA;
            --accent-blue-soft: rgba(135, 177, 220, 0.18);
            --text-primary: #2C2A29;
            --text-secondary: #6E6A64;

            /* Variables de Streamlit y Glide Data Grid (st.data_editor / st.dataframe) para forzar modo claro cálido */
            --primary-color: #87B1DC;
            --background-color: #F7F4EF;
            --secondary-background-color: #FAF8F5;
            --text-color: #2C2A29;
            --gdg-bg-cell: #FFFFFF;
            --gdg-bg-cell-medium: #FAF8F5;
            --gdg-bg-header: #EFECE6;
            --gdg-bg-header-has-focus: #E4DFD7;
            --gdg-bg-header-hovered: #E4DFD7;
            --gdg-bg-bubble: #EFECE6;
            --gdg-bg-bubble-selected: #87B1DC;
            --gdg-bg-search-result: rgba(135, 177, 220, 0.25);
            --gdg-border-color: #E2DDD5;
            --gdg-horizontal-border-color: #E2DDD5;
            --gdg-drilldown-border: #87B1DC;
            --gdg-text-dark: #2C2A29;
            --gdg-text-medium: #6E6A64;
            --gdg-text-light: #8A847C;
            --gdg-text-bubble: #2C2A29;
            --gdg-text-header: #2C2A29;
            --gdg-text-header-selected: #2C2A29;
            --gdg-accent-color: #87B1DC;
            --gdg-accent-fg: #2C2A29;
            --gdg-accent-light: rgba(135, 177, 220, 0.22);
            color-scheme: light !important;
        }

        /* Tipografía y fondo general cálido */
        html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"] {
            font-family: 'Plus Jakarta Sans', 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
            background-color: var(--bg-main) !important;
            color: var(--text-primary) !important;
        }

        [data-testid="stHeader"] {
            background-color: rgba(247, 244, 239, 0.88) !important;
            backdrop-filter: blur(8px);
            border-bottom: 1px solid var(--border-subtle);
        }

        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 3.5rem !important;
            max-width: 1280px !important;
        }

        /* Textos, títulos y etiquetas */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            color: var(--text-primary) !important;
            letter-spacing: -0.02em;
            font-weight: 700 !important;
        }

        p, li, span, label, [data-testid="stMarkdownContainer"] p {
            color: var(--text-primary);
        }

        [data-testid="stCaptionContainer"], small, .editorial-subtitle {
            color: var(--text-secondary) !important;
        }

        [data-testid="stWidgetLabel"] p {
            color: var(--text-primary) !important;
            font-weight: 600 !important;
            font-size: 0.92rem !important;
        }

        /* Barra lateral (Sidebar) en tono arena suave */
        [data-testid="stSidebar"] {
            background-color: var(--bg-sidebar) !important;
            border-right: 1px solid var(--border-subtle) !important;
        }

        [data-testid="stSidebar"] * {
            color: var(--text-primary);
        }

        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: var(--text-primary) !important;
            font-size: 1.05rem !important;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-top: 0.5rem;
        }

        /* Sección Hero (Encabezado estilo Landing Page Editorial) */
        .bloom-hero {
            position: relative;
            width: 100%;
            min-height: 290px;
            border-radius: 6px;
            overflow: hidden;
            margin-bottom: 2rem;
            background-image:
                linear-gradient(180deg, rgba(34, 31, 28, 0.72) 0%, rgba(34, 31, 28, 0.56) 50%, rgba(34, 31, 28, 0.76) 100%),
                url('https://images.unsplash.com/photo-1497366216548-37526070297c?auto=format&fit=crop&w=1600&q=80');
            background-size: cover;
            background-position: center;
            box-shadow: 0 12px 32px rgba(44, 42, 41, 0.08);
            border: 1px solid #DCD6CC;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            padding: 1.75rem 2.5rem 2.5rem 2.5rem;
        }

        .bloom-hero-topbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            width: 100%;
            border-bottom: 1px solid rgba(255, 255, 255, 0.22);
            padding-bottom: 0.9rem;
            gap: 1rem;
            flex-wrap: wrap;
        }

        .bloom-hero-tag {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.2em;
            color: #F7F4EF !important;
        }

        .bloom-hero-authors {
            font-family: 'DM Sans', sans-serif;
            font-size: 0.86rem;
            font-weight: 500;
            letter-spacing: 0.04em;
            color: #EFECE6 !important;
        }

        .bloom-hero-center {
            text-align: center;
            max-width: 820px;
            margin: 2.2rem auto 1.2rem auto;
        }

        .bloom-hero-title {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            font-size: clamp(1.9rem, 3.4vw, 2.85rem) !important;
            font-weight: 700 !important;
            color: #FFFFFF !important;
            line-height: 1.2 !important;
            margin: 0 0 0.85rem 0 !important;
            letter-spacing: -0.02em;
        }

        .bloom-underline {
            display: inline-block;
            position: relative;
            padding-bottom: 4px;
            border-bottom: 4px solid #87B1DC;
        }

        .bloom-hero-subtitle {
            font-family: 'DM Sans', sans-serif;
            font-size: 1.02rem;
            color: rgba(247, 244, 239, 0.9) !important;
            max-width: 650px;
            margin: 0 auto;
            font-weight: 400;
            line-height: 1.55;
        }

        /* Tarjetas y contenedores con borde sutil y sombra cálida */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: 6px !important;
            padding: 1.5rem 1.6rem !important;
            box-shadow: 0 6px 22px rgba(44, 42, 41, 0.04) !important;
        }

        /* Pestañas (st.tabs) estilo barra de navegación limpia y centrada */
        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            justify-content: center !important;
            gap: 2.2rem !important;
            background-color: transparent !important;
            border-bottom: 1px solid var(--border-subtle) !important;
            padding-bottom: 0 !important;
            margin-bottom: 1.5rem !important;
        }

        [data-testid="stTabs"] [data-baseweb="tab"] {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            font-size: 0.96rem !important;
            font-weight: 500 !important;
            color: var(--text-secondary) !important;
            background-color: transparent !important;
            border: none !important;
            padding: 0.85rem 0.5rem !important;
            transition: color 0.2s ease;
        }

        [data-testid="stTabs"] [data-baseweb="tab"]:hover {
            color: var(--text-primary) !important;
        }

        [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
            color: var(--text-primary) !important;
            font-weight: 700 !important;
            border-bottom: 3px solid var(--accent-blue) !important;
        }

        [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
            background-color: var(--accent-blue) !important;
            height: 3px !important;
        }

        /* Botones primarios, secundarios y de descarga en azul pastel (#87B1DC) y bordes 4px */
        .stButton > button[kind="primary"],
        [data-testid="stBaseButton-primary"] {
            background-color: var(--accent-blue) !important;
            color: var(--text-primary) !important;
            border: 1px solid #75A0CB !important;
            border-radius: 4px !important;
            font-weight: 600 !important;
            padding: 0.6rem 1.2rem !important;
            box-shadow: 0 2px 6px rgba(135, 177, 220, 0.25) !important;
            transition: all 0.2s ease !important;
        }

        .stButton > button[kind="primary"]:hover,
        [data-testid="stBaseButton-primary"]:hover {
            background-color: var(--accent-blue-hover) !important;
            color: var(--text-primary) !important;
            border-color: #628EB9 !important;
        }

        .stButton > button[kind="secondary"],
        [data-testid="stBaseButton-secondary"],
        .stDownloadButton > button {
            background-color: var(--bg-card-warm) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: 4px !important;
            font-weight: 600 !important;
            padding: 0.55rem 1rem !important;
            transition: all 0.2s ease !important;
        }

        .stButton > button[kind="secondary"]:hover,
        [data-testid="stBaseButton-secondary"]:hover,
        .stDownloadButton > button:hover {
            background-color: var(--accent-blue-soft) !important;
            border-color: var(--accent-blue) !important;
            color: var(--text-primary) !important;
        }

        .stButton > button *, .stDownloadButton > button * {
            color: var(--text-primary) !important;
        }

        /* Radio buttons y Checkboxes con acento azul pastel (#87B1DC) */
        div[role="radiogroup"] label > div:first-child {
            border-color: #B9B2A8 !important;
        }

        div[role="radiogroup"] label[data-baseweb="radio"] input:checked + div,
        [data-baseweb="radio"] > div:first-child {
            accent-color: var(--accent-blue) !important;
        }

        input[type="radio"], input[type="checkbox"] {
            accent-color: var(--accent-blue) !important;
        }

        /* Inputs de texto, áreas de texto y selectboxes en modo claro */
        [data-baseweb="input"] > div,
        [data-baseweb="textarea"] > div,
        [data-baseweb="select"] > div {
            background-color: #FFFFFF !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: 4px !important;
            color: var(--text-primary) !important;
        }

        [data-baseweb="input"] input,
        [data-baseweb="textarea"] textarea,
        [data-baseweb="select"] * {
            color: var(--text-primary) !important;
            background-color: transparent !important;
        }

        [data-baseweb="input"] > div:focus-within,
        [data-baseweb="textarea"] > div:focus-within,
        [data-baseweb="select"] > div:focus-within {
            border-color: var(--accent-blue) !important;
            box-shadow: 0 0 0 2px var(--accent-blue-soft) !important;
        }

        /* Menús desplegables (popovers de selectbox) */
        [data-baseweb="popover"], [data-baseweb="menu"], ul[role="listbox"] {
            background-color: #FFFFFF !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-subtle) !important;
        }

        li[role="option"] {
            color: var(--text-primary) !important;
            background-color: #FFFFFF !important;
        }

        li[role="option"]:hover, li[role="option"][aria-selected="true"] {
            background-color: var(--accent-blue-soft) !important;
        }

        /* File Uploader claro, cálido y minimalista */
        [data-testid="stFileUploader"] section,
        [data-testid="stFileUploaderDropzone"] {
            background-color: var(--bg-card-warm) !important;
            border: 1.5px dashed #C9C2B8 !important;
            border-radius: 4px !important;
            padding: 1.35rem !important;
            color: var(--text-primary) !important;
            transition: border-color 0.2s ease, background-color 0.2s ease;
        }

        [data-testid="stFileUploader"] section:hover,
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: var(--accent-blue) !important;
            background-color: #F3F6FA !important;
        }

        [data-testid="stFileUploaderDropzone"] span,
        [data-testid="stFileUploaderDropzone"] small,
        [data-testid="stFileUploaderDropzone"] div {
            color: var(--text-secondary) !important;
        }

        [data-testid="stFileUploaderDropzone"] button {
            background-color: #FFFFFF !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: 4px !important;
            font-weight: 600 !important;
        }

        [data-testid="stFileUploaderDropzone"] button:hover {
            border-color: var(--accent-blue) !important;
            background-color: var(--accent-blue-soft) !important;
        }

        /* Expander (st.expander) limpio y claro */
        [data-testid="stExpander"], details {
            background-color: var(--bg-card-warm) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: 4px !important;
            overflow: hidden;
            margin-bottom: 1rem !important;
        }

        [data-testid="stExpander"] summary, details > summary {
            background-color: var(--bg-card-warm) !important;
            color: var(--text-primary) !important;
            font-weight: 600 !important;
            padding: 0.75rem 1rem !important;
        }

        [data-testid="stExpander"] summary:hover, details > summary:hover {
            background-color: #F0ECE3 !important;
        }

        [data-testid="stExpanderDetails"] {
            background-color: #FFFFFF !important;
            border-top: 1px solid var(--border-subtle) !important;
            padding: 1rem !important;
        }

        /* Tablas editables (st.data_editor y st.dataframe) */
        [data-testid="stDataFrame"], [data-testid="stDataEditor"] {
            border: 1px solid var(--border-subtle) !important;
            border-radius: 4px !important;
            background-color: #FFFFFF !important;
            overflow: hidden;
        }

        /* Métricas (st.metric) en tarjeta editorial */
        [data-testid="stMetric"] {
            background-color: var(--bg-card-warm) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: 4px !important;
            padding: 1.1rem 1.25rem !important;
        }

        [data-testid="stMetricLabel"] {
            color: var(--text-secondary) !important;
            font-weight: 600 !important;
        }

        [data-testid="stMetricValue"] {
            color: var(--text-primary) !important;
            font-weight: 700 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def renderizar_hero_banner() -> None:
    """Renderiza el encabezado estilo Landing Page editorial con subrayado en azul pastel."""
    st.markdown(
        """
        <div class="bloom-hero">
            <div class="bloom-hero-topbar">
                <span class="bloom-hero-tag">EDI II &bull; CIENCIAS DE DATOS</span>
                <span class="bloom-hero-authors">Alan Bustamante y Christian Martínez</span>
            </div>
            <div class="bloom-hero-center">
                <h1 class="bloom-hero-title">
                    Lector Inteligente de Comprobantes y <span class="bloom-underline">Facturas</span>
                </h1>
                <p class="bloom-hero-subtitle">
                    Reconocimiento óptico de caracteres, extracción contable estructurada con IA híbrida,
                    revisión asistida y evaluación de exactitud.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Lector Inteligente de Comprobantes y Facturas",
        page_icon="🧾",
        layout="wide",
    )
    inicializar_estado_sesion()
    aplicar_estilos_editoriales()
    renderizar_hero_banner()

    # --------------------------------------------------------------------------
    # BARRA LATERAL: CONFIGURACIÓN HÍBRIDA Y DIAGNÓSTICO AUTOMÁTICO
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Configuración del Motor")

        # 1. Diagnóstico de Tesseract OCR
        ruta_tesseract = configurar_ruta_tesseract()
        if ruta_tesseract:
            st.success(f"Tesseract OCR listo ({ruta_tesseract})")
        else:
            st.error("No se encontró Tesseract OCR en el sistema.")
            ruta_manual = st.text_input(
                "Ruta manual a tesseract.exe:",
                placeholder=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            )
            if ruta_manual and os.path.exists(ruta_manual):
                pytesseract.pytesseract.tesseract_cmd = ruta_manual
                st.success("Ruta manual de Tesseract configurada.")

        st.subheader("Mejoras de Calidad (TFI Obj. 6)")
        aplicar_preprocesamiento = st.checkbox(
            "Preprocesar imagen (Contraste y Escala de Grises)",
            value=True,
            help="Mejora la lectura de tickets térmicos o fotos tomadas con celular.",
        )
        aplicar_hibrido_regex = st.checkbox(
            "Refuerzo con Expresiones Regulares (CUIT/Fechas)",
            value=True,
            help="Valida y rescata patrones exactos de CUIT, fecha y número de factura si el modelo los omite.",
        )

        st.subheader("Modo de Inteligencia Artificial")
        clave_entorno = os.environ.get("GROQ_API_KEY", "")
        if not clave_entorno:
            try:
                clave_entorno = st.secrets.get("GROQ_API_KEY", "")
            except Exception:
                clave_entorno = ""

        modo_ia = st.radio(
            "Selecciona dónde ejecutar el modelo:",
            options=[
                "1. Local (Ollama en esta PC)",
                "2. Nube (Escalable para Celulares 24/7)",
                "3. Solo Reglas Regex (Sin LLM)",
            ],
            index=1 if clave_entorno else 0,
        )

        modelo_seleccionado = ""
        url_ollama = "http://localhost:11434"
        api_key_nube = ""
        endpoint_nube = "https://api.groq.com/openai/v1/chat/completions"

        if modo_ia.startswith("1."):
            url_ollama = st.text_input("URL de Ollama:", value="http://localhost:11434")
            modelos_instalados, error_ollama = obtener_modelos_ollama(url_ollama)

            if modelos_instalados:
                st.success(f"Ollama conectado ({len(modelos_instalados)} modelo/s detectado/s)")
                modelo_seleccionado = st.selectbox(
                    "Modelo instalado en tu Ollama:",
                    options=modelos_instalados,
                )
            else:
                st.warning(error_ollama or "No tienes modelos descargados en Ollama todavía.")
                st.info(
                    "💡 **¿No sabes qué modelo tienes?**\n"
                    "Abre una terminal y ejecuta:\n"
                    "`ollama pull llama3.2`\n"
                    "Luego presiona **Recargar** en esta página."
                )
                modelo_seleccionado = st.text_input("Nombre manual del modelo:", value="llama3.2")

        elif modo_ia.startswith("2."):
            st.info(
                "En modo Nube, cualquier usuario puede entrar desde su celular sin descargar "
                "el modelo. Usa una API gratuita compatible con modelos abiertos (ej. Groq)."
            )
            api_key_nube = st.text_input(
                "API Key (Groq / OpenRouter):",
                value=clave_entorno,
                type="password",
                placeholder="gsk_...",
            )
            modelo_seleccionado = st.selectbox(
                "Modelo abierto en la nube:",
                options=[
                    "llama-3.3-70b-versatile",
                    "llama-3.1-8b-instant",
                    "qwen-qwq-32b",
                ],
            )
            endpoint_nube = st.text_input(
                "Endpoint API:",
                value="https://api.groq.com/openai/v1/chat/completions",
            )

    # --------------------------------------------------------------------------
    # PESTAÑAS PRINCIPALES DE LA APLICACIÓN
    # --------------------------------------------------------------------------
    tab_lector, tab_historial, tab_evaluacion = st.tabs(
        [
            "📄 1. Cargar y Extraer Comprobante",
            "🗂️ 2. Registro Acumulado y Exportación",
            "📊 3. Evaluación de Exactitud (TFI)",
        ]
    )

    # ==========================================================================
    # PESTAÑA 1: CARGA, OCR, EXTRACCIÓN Y REVISIÓN MANUAL
    # ==========================================================================
    with tab_lector:
        col_carga, col_resultados = st.columns([1, 1.2], gap="large")

        with col_carga:
            with st.container(border=True):
                st.subheader("1. Entrada del Comprobante")
                metodo_entrada = st.radio(
                    "Origen de la imagen:",
                    options=["Subir archivo (JPG, PNG)", "Tomar foto con cámara del celular / PC"],
                    horizontal=True,
                )

                archivo_imagen = None
                if metodo_entrada.startswith("Subir"):
                    archivo_imagen = st.file_uploader(
                        "Selecciona una imagen de factura o ticket:",
                        type=["png", "jpg", "jpeg", "webp"],
                    )
                else:
                    archivo_imagen = st.camera_input("Captura el comprobante con tu cámara:")

                if archivo_imagen is not None:
                    imagen_original = Image.open(archivo_imagen)
                    imagen_procesada = preprocesar_imagen(imagen_original, aplicar_mejora=aplicar_preprocesamiento)

                    vista_img = st.radio(
                        "Vista previa:",
                        options=["Imagen Original", "Imagen Preprocesada para OCR"],
                        horizontal=True,
                    )
                    if vista_img == "Imagen Original":
                        st.image(imagen_original, caption="Comprobante original", use_container_width=True)
                    else:
                        st.image(imagen_procesada, caption="Preprocesada para OCR", use_container_width=True)

                    if st.button("🚀 Procesar Comprobante (OCR + IA)", type="primary", use_container_width=True):
                        with st.spinner("Ejecutando OCR con Tesseract..."):
                            texto_detectado, alerta_ocr = ejecutar_ocr(imagen_procesada, idioma="spa")
                            st.session_state["texto_ocr"] = texto_detectado

                        if alerta_ocr:
                            st.warning(alerta_ocr)

                        if not texto_detectado or len(texto_detectado.strip()) < 10:
                            st.error(
                                "⚠️ La imagen no pudo procesarse correctamente o contiene muy poco texto legible. "
                                "Intenta con una foto más nítida o mejor iluminada."
                            )
                        else:
                            with st.spinner("Extrayendo campos estructurados..."):
                                error_ia = None
                                if modo_ia.startswith("1."):
                                    datos_ext, error_ia = extraer_con_ollama_local(
                                        texto_detectado,
                                        modelo=modelo_seleccionado,
                                        url_base=url_ollama,
                                    )
                                elif modo_ia.startswith("2."):
                                    datos_ext, error_ia = extraer_con_api_nube(
                                        texto_detectado,
                                        api_key=api_key_nube,
                                        modelo=modelo_seleccionado,
                                        endpoint_url=endpoint_nube,
                                    )
                                else:
                                    datos_ext = extraer_con_reglas_regex(texto_detectado)

                                if error_ia:
                                    st.warning(
                                        f"{error_ia}\nSe aplicó la extracción de respaldo por reglas (Regex)."
                                    )
                                    datos_ext = extraer_con_reglas_regex(texto_detectado)
                                elif aplicar_hibrido_regex:
                                    datos_ext = aplicar_refuerzo_regex(datos_ext, texto_detectado)

                                st.session_state["datos_actuales"] = datos_ext
                                st.success("¡Extracción completada! Revisa los campos en el panel derecho.")

        with col_resultados:
            with st.container(border=True):
                st.subheader("2. Revisión y Corrección Manual")

                with st.expander("🔍 Ver texto crudo reconocido por OCR", expanded=False):
                    texto_editable_ocr = st.text_area(
                        "Texto extraído por Tesseract:",
                        value=st.session_state["texto_ocr"],
                        height=180,
                    )
                    st.session_state["texto_ocr"] = texto_editable_ocr

                # Advertencia de campos críticos faltantes (Funcionalidad 7 del TFI)
                datos_act = st.session_state["datos_actuales"]
                campos_criticos = ["emisor", "fecha_emision", "importe_total"]
                faltantes = [
                    CAMPOS_COMPROBANTE[c]
                    for c in campos_criticos
                    if datos_act.get(c, VALOR_NO_IDENTIFICADO) == VALOR_NO_IDENTIFICADO
                ]
                if st.session_state["texto_ocr"] and faltantes:
                    st.warning(
                        "⚠️ **Atención:** No se pudieron identificar algunos campos clave: "
                        f"**{', '.join(faltantes)}**. Puedes completarlos manualmente en la tabla."
                    )

                # Construcción de tabla editable para revisión humana (Funcionalidades 4 y 5 del TFI)
                df_revision = pd.DataFrame(
                    [
                        {
                            "Clave": clave,
                            "Campo del Comprobante": etiqueta,
                            "Valor Detectado (Editable)": st.session_state["datos_actuales"].get(
                                clave, VALOR_NO_IDENTIFICADO
                            ),
                        }
                        for clave, etiqueta in CAMPOS_COMPROBANTE.items()
                    ]
                )

                df_editado = st.data_editor(
                    df_revision,
                    column_config={
                        "Clave": None,  # Oculta la columna técnica
                        "Campo del Comprobante": st.column_config.TextColumn(
                            "Campo del Comprobante", disabled=True, width="medium"
                        ),
                        "Valor Detectado (Editable)": st.column_config.TextColumn(
                            "Valor Revisado (Haz doble clic para corregir)", width="large"
                        ),
                    },
                    hide_index=True,
                    use_container_width=True,
                    key="editor_campos",
                )

                # Sincronizar los cambios hechos por el usuario
                datos_revisados = {
                    fila["Clave"]: str(fila["Valor Detectado (Editable)"]).strip() or VALOR_NO_IDENTIFICADO
                    for _, fila in df_editado.iterrows()
                }
                st.session_state["datos_actuales"] = datos_revisados

                col_btn1, col_btn2, col_btn3 = st.columns(3)

                with col_btn1:
                    if st.button("➕ Guardar en el Registro", type="primary", use_container_width=True):
                        registro_nuevo = {"fecha_carga": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                        registro_nuevo.update(datos_revisados)
                        st.session_state["historial_comprobantes"].append(registro_nuevo)
                        st.success("Comprobante guardado en la pestaña 'Registro Acumulado'.")

                with col_btn2:
                    json_individual = json.dumps(datos_revisados, ensure_ascii=False, indent=2)
                    st.download_button(
                        label="⬇️ Descargar JSON",
                        data=json_individual.encode("utf-8"),
                        file_name="comprobante_revisado.json",
                        mime="application/json",
                        use_container_width=True,
                    )

                with col_btn3:
                    df_export_ind = pd.DataFrame([datos_revisados])
                    csv_individual = df_export_ind.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label="⬇️ Descargar CSV",
                        data=csv_individual,
                        file_name="comprobante_revisado.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

    # ==========================================================================
    # PESTAÑA 2: REGISTRO ACUMULADO DE COMPROBANTES Y EXPORTACIÓN MASIVA
    # ==========================================================================
    with tab_historial:
        with st.container(border=True):
            st.subheader("🗂️ Comprobantes Registrados en la Sesión")
            historial = st.session_state["historial_comprobantes"]

            if not historial:
                st.info(
                    "Aún no has guardado comprobantes en el lote. "
                    "Procesa una factura en la pestaña 1 y presiona **'➕ Guardar en el Registro'**."
                )
            else:
                df_hist = pd.DataFrame(historial)
                st.dataframe(df_hist, use_container_width=True)

                col_exp1, col_exp2, col_limpiar = st.columns(3)
                with col_exp1:
                    csv_lote = df_hist.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "📥 Exportar Todos a CSV",
                        data=csv_lote,
                        file_name="registro_comprobantes.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                with col_exp2:
                    json_lote = json.dumps(historial, ensure_ascii=False, indent=2).encode("utf-8")
                    st.download_button(
                        "📥 Exportar Todos a JSON",
                        data=json_lote,
                        file_name="registro_comprobantes.json",
                        mime="application/json",
                        use_container_width=True,
                    )
                with col_limpiar:
                    if st.button("🗑️ Vaciar Registro", use_container_width=True):
                        st.session_state["historial_comprobantes"] = []
                        st.rerun()

    # ==========================================================================
    # PESTAÑA 3: EVALUACIÓN DE EXACTITUD CONTRA TRANSCRIPCIÓN DE REFERENCIA
    # ==========================================================================
    with tab_evaluacion:
        with st.container(border=True):
            st.subheader("📊 Evaluación de Exactitud (Ground Truth)")
            st.markdown(
                "Esta sección cumple con el **Objetivo Específico 5 del TFI**: comparar los datos extraídos "
                "por la aplicación frente a una **transcripción manual de referencia** para medir el porcentaje "
                "de aciertos."
            )

            df_ref_inicial = pd.DataFrame(
                [
                    {
                        "Clave": clave,
                        "Campo": etiqueta,
                        "Valor Extraído por la App": st.session_state["datos_actuales"].get(
                            clave, VALOR_NO_IDENTIFICADO
                        ),
                        "Transcripción de Referencia (Real)": st.session_state["datos_actuales"].get(
                            clave, VALOR_NO_IDENTIFICADO
                        ),
                    }
                    for clave, etiqueta in CAMPOS_COMPROBANTE.items()
                ]
            )

            st.markdown("##### Completa la columna **'Transcripción de Referencia (Real)'** con los datos verdaderos del comprobante:")
            df_ref_editado = st.data_editor(
                df_ref_inicial,
                column_config={
                    "Clave": None,
                    "Campo": st.column_config.TextColumn("Campo", disabled=True),
                    "Valor Extraído por la App": st.column_config.TextColumn(
                        "Valor Extraído por la App", disabled=True
                    ),
                    "Transcripción de Referencia (Real)": st.column_config.TextColumn(
                        "Transcripción de Referencia (Escribe el valor real)"
                    ),
                },
                hide_index=True,
                use_container_width=True,
                key="editor_ground_truth",
            )

            referencia_dict = {
                fila["Clave"]: str(fila["Transcripción de Referencia (Real)"]).strip()
                for _, fila in df_ref_editado.iterrows()
            }

            df_comparacion, porcentaje_exactitud = evaluar_extraccion(
                st.session_state["datos_actuales"], referencia_dict
            )

            col_metrica, col_detalle = st.columns([1, 2.5])
            with col_metrica:
                st.metric(
                    label="Exactitud de Extracción",
                    value=f"{porcentaje_exactitud}%",
                    help="Porcentaje de campos coincidentes sobre los 9 campos evaluados.",
                )
                csv_eval = df_comparacion.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "📥 Descargar Informe de Evaluación (CSV)",
                    data=csv_eval,
                    file_name="evaluacion_exactitud_tfi.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with col_detalle:
                st.dataframe(df_comparacion, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()