"""
TRABAJO FINAL INTEGRADOR — EDI II (Ciencias de Datos)
Proyecto: Lector inteligente de comprobantes y facturas
Integrantes: Alan Bustamante, Christian Martínez

Descripción:
Aplicación web en Streamlit con diseño editorial cálido y minimalista,
optimizador de imagen con OpenCV (Filtro Escáner) para Tesseract OCR,
extracción estructurada con IA híbrida (Groq / Google Gemini / Ollama / Regex),
autenticación de usuarios y persistencia en la nube con Supabase (Storage + PostgreSQL)
con soporte de Modo Demo Local automático.
"""

import base64
import io
import json
import os
import re
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytesseract
import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

try:
    import cv2

    OPENCV_DISPONIBLE = True
except ImportError:
    cv2 = None  # type: ignore
    OPENCV_DISPONIBLE = False

try:
    import fitz  # PyMuPDF para renderizar páginas PDF y extraer texto nativo

    PYMUPDF_DISPONIBLE = True
except ImportError:
    fitz = None  # type: ignore
    PYMUPDF_DISPONIBLE = False

try:
    from pypdf import PdfReader

    PYPDF_DISPONIBLE = True
except ImportError:
    PdfReader = None  # type: ignore
    PYPDF_DISPONIBLE = False

try:
    from supabase import Client, create_client

    SUPABASE_LIB_DISPONIBLE = True
except ImportError:
    Client = Any  # type: ignore
    create_client = None  # type: ignore
    SUPABASE_LIB_DISPONIBLE = False


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

# Modelo principal configurado de forma fija (hardcodeada) en Groq Cloud
MODELO_VISION_PRINCIPAL = "openai/gpt-oss-120b"
ENDPOINT_VISION_PRINCIPAL = "https://api.groq.com/openai/v1/chat/completions"

PROMPT_ITEMS_FACTURA = """Eres un sistema experto en extracción de datos estructurados a partir de imágenes de documentos contables y facturas (OCR e interpretación semántica).

Tu tarea es analizar la factura proporcionada y extraer EXCLUSIVAMENTE la tabla de ítems/productos facturados.

Debes devolver la información en un formato JSON estrictamente válido. Para cada línea, extrae:

1. "descripcion": (String) El texto exacto que describe el producto o servicio.
2. "cantidad_facturada": (Float) La cantidad de bultos o unidades que marca la línea.
3. "precio_linea_impreso": (Float) El valor monetario unitario impreso en esa línea.
4. "unidades_por_empaque": (Entero) Analiza la descripción. Si el texto indica que es un empaque múltiple (ej. "Gaseosa 6x3L", "Caja x 12", "Pack de 4"), extrae el número de unidades individuales que contiene (6, 12, 4). Si no hay indicios de ser un empaque múltiple, el valor por defecto es 1.

Devuelve ÚNICAMENTE el JSON, con esta estructura base:
{
  "items": [
    {
      "descripcion": "Manaos Cola 6x3L",
      "cantidad_facturada": 1.0,
      "precio_linea_impreso": 8200.0,
      "unidades_por_empaque": 6
    }
  ]
}
"""

PROMPT_SISTEMA = """Te estoy enviando múltiples fragmentos fotográficos (imágenes) que pertenecen a una ÚNICA factura larga o ticket. Tu tarea es analizar todas las imágenes en conjunto, unir mentalmente la información y extraer los datos solicitados en un único objeto JSON consolidado sin duplicar información.

Eres un sistema experto en extracción de datos estructurados a partir de imágenes de documentos contables y facturas (OCR e interpretación semántica).
Devuelve EXCLUSIVAMENTE un objeto JSON válido con las 9 claves generales del comprobante Y la clave "items" con el detalle de productos/servicios facturados:

{
  "tipo_documento": "Ej: Factura A, Factura B, Factura C, Ticket, Recibo o 'No identificado'",
  "emisor": "Nombre o razón social del comercio/emisor o 'No identificado'",
  "cuit": "CUIT del emisor con formato XX-XXXXXXXX-X o 'No identificado'",
  "fecha_emision": "Fecha en formato DD/MM/AAAA o 'No identificado'",
  "numero_comprobante": "Número de factura o ticket (ej: 0001-00012345) o 'No identificado'",
  "importe_neto": "Subtotal neto gravado sin impuestos en formato numérico con punto decimal (ej: 29813.34) o 'No identificado'",
  "impuestos_iva": "Suma total de impuestos discriminados (IVA 21%, IVA 10.5%, Impuesto Interno, Percepciones) en formato numérico con punto decimal (ej: 8853.37 o 6260.80) o 'No identificado'",
  "importe_total": "Importe total final en formato numérico con punto decimal (ej: 38666.71) o 'No identificado'",
  "moneda": "Moneda identificada (ej: ARS, USD, $) o 'No identificado'",
  "items": [
    {
      "descripcion": "Manaos Cola 6x3L",
      "cantidad_facturada": 1.0,
      "precio_linea_impreso": 8200.0,
      "unidades_por_empaque": 6
    }
  ]
}

REGLAS PARA LA TABLA DE ÍTEMS ("items"):
Para cada línea de producto o servicio facturado, extrae:
1. "descripcion": (String) El texto exacto que describe el producto o servicio.
2. "cantidad_facturada": (Float) La cantidad de bultos o unidades que marca la línea.
3. "precio_linea_impreso": (Float) El valor monetario unitario del bulto/artículo impreso en esa línea (si solo figura el subtotal de la línea, divídelo por cantidad_facturada).
4. "unidades_por_empaque": (Entero) Analiza la descripción. Si el texto indica que es un empaque múltiple (ej. "Gaseosa 6x3L", "Caja x 12", "Pack de 4", "24x473cc"), extrae el número de unidades individuales que contiene (6, 12, 4, 24). Si no hay indicios de ser un empaque múltiple, el valor por defecto es 1.

REGLAS CONTABLES Y DE CONSOLIDACIÓN MULTI-IMAGEN:
1. CONSOLIDACIÓN DE FRAGMENTOS SUPERPUESTOS: Si el encabezado aparece en las primeras fotos y los ítems/totales en las siguientes, combínalos en un único JSON final sin duplicar líneas en las zonas de solapamiento.
2. TABLAS DE PIE DE FACTURA EN FILAS SEPARADAS: En muchas Facturas A argentinas (ej: Salta Refrescos, Coca-Cola, distribuidoras, AFIP), los encabezados de la tabla ('Subtotal Neto', 'Impuesto Interno', 'IVA 21', 'Percepc. DGR', 'Percepc. Mun', 'TOTAL') aparecen en una línea y sus valores numéricos ('$29.813,34  $2.592,57  $6.260,80  $0,00  $0,00  $38.666,71') aparecen en la línea inmediatamente inferior:
   - El PRIMER monto positivo de esa fila de desglose (ej: 29813.34) corresponde a "Subtotal Neto" -> colócalo en "importe_neto".
   - Los montos intermedios mayores a cero antes del TOTAL (ej: Impuesto Interno 2592.57 + IVA 21 6260.80 = 8853.37) corresponden a "impuestos_iva".
3. NO CONFUNDIR 'SUBTOTAL' CON 'TOTAL': Si arriba de la tabla aparece 'SUBTOTAL $38.666,71' igual al 'TOTAL $38.666,71', NO uses ese valor como importe_neto si existe el Subtotal Neto menor (ej: 29813.34).
4. Convierte los importes argentinos (ej: $29.813,34) a formato decimal estándar con punto (ej: 29813.34).
5. Devuelve ÚNICAMENTE el objeto JSON válido, sin explicaciones ni bloques markdown.
"""


# ==============================================================================
# 2. MÓDULO DE RECONOCIMIENTO ÓPTICO (OPENCV + TESSERACT OCR)
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


def ordenar_puntos(pts: np.ndarray) -> np.ndarray:
    """
    Ordena las 4 esquinas detectadas del documento en el orden estricto:
    [0] arriba-izquierda (top-left)
    [1] arriba-derecha (top-right)
    [2] abajo-derecha (bottom-right)
    [3] abajo-izquierda (bottom-left)
    """
    rect = np.zeros((4, 2), dtype="float32")

    # Arriba-izquierda tiene la suma (x + y) más pequeña; abajo-derecha la más grande
    suma = pts.sum(axis=1)
    rect[0] = pts[np.argmin(suma)]
    rect[2] = pts[np.argmax(suma)]

    # Arriba-derecha tiene la diferencia (y - x) más pequeña; abajo-izquierda la más grande
    dif = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(dif)]
    rect[3] = pts[np.argmax(dif)]

    return rect


def escanear_documento_con_estado(imagen_bytes: bytes) -> Tuple[Image.Image, bool]:
    """
    Ejecuta el pipeline clásico de visión por computadora estilo CamScanner:
    1. Redimensionado temporal para acelerar procesamiento.
    2. Escala de grises + Gaussian Blur (5x5).
    3. Detección de bordes con Canny Edge Detection + dilatación suave.
    4. Búsqueda de contornos (cv2.findContours) ordenados por área descendente.
    5. Aproximación poligonal (cv2.approxPolyDP) verificando exactamente 4 vértices.
    6. Matriz de transformación (cv2.getPerspectiveTransform) y warpPerspective (bird's-eye view)
       sobre la imagen original en alta resolución.
    7. Manejo de errores: si no detecta 4 esquinas claras, devuelve la imagen original intacta.
    """
    try:
        pil_orig = ImageOps.exif_transpose(Image.open(io.BytesIO(imagen_bytes))).convert("RGB")
    except Exception:
        pil_orig = Image.open(io.BytesIO(imagen_bytes)).convert("RGB")

    if not OPENCV_DISPONIBLE or cv2 is None:
        return pil_orig, False

    try:
        orig_np = np.array(pil_orig)
        alto_orig, ancho_orig = orig_np.shape[:2]

        # 1. Redimensionar temporalmente para acelerar el procesamiento
        altura_proc = 650.0
        ratio = alto_orig / altura_proc
        ancho_proc = max(1, int(ancho_orig / ratio))
        img_redim = cv2.resize(orig_np, (ancho_proc, int(altura_proc)), interpolation=cv2.INTER_AREA)

        # 2. Convertir a escala de grises y aplicar Gaussian Blur
        gris = cv2.cvtColor(img_redim, cv2.COLOR_RGB2GRAY)
        desenfocada = cv2.GaussianBlur(gris, (5, 5), 0)

        # 3. Aplicar detección de bordes (Canny Edge Detection) y cierre morfológico suave
        bordes = cv2.Canny(desenfocada, 75, 200)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        bordes = cv2.dilate(bordes, kernel, iterations=1)

        # 4. Buscar contornos y ordenar de mayor a menor área
        contornos_info = cv2.findContours(bordes.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contornos = contornos_info[0] if len(contornos_info) == 2 else contornos_info[1]
        contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:5]

        area_total_proc = float(ancho_proc * int(altura_proc))
        contorno_pantalla = None

        # 5. Aproximar el polígono para verificar si tiene exactamente 4 vértices
        for c in contornos:
            area_c = cv2.contourArea(c)
            # Evitar recortar recuadros diminutos internos (exigir al menos 15% del área de la foto)
            if area_c < (0.15 * area_total_proc):
                continue

            perimetro = cv2.arcLength(c, True)
            aprox = cv2.approxPolyDP(c, 0.02 * perimetro, True)

            if len(aprox) == 4:
                contorno_pantalla = aprox
                break

        # Manejo de fallback si no se detectan 4 esquinas claras
        if contorno_pantalla is None:
            return pil_orig, False

        # 6. Calcular matriz de transformación y aplicar warpPerspective sobre la imagen de alta resolución
        pts_alta_res = contorno_pantalla.reshape(4, 2).astype("float32") * ratio
        rect = ordenar_puntos(pts_alta_res)
        (tl, tr, br, bl) = rect

        # Calcular el ancho máximo del nuevo documento enderezado
        ancho_abajo = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        ancho_arriba = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_ancho = max(int(ancho_abajo), int(ancho_arriba))

        # Calcular el alto máximo del nuevo documento enderezado
        alto_derecha = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        alto_izquierda = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_alto = max(int(alto_derecha), int(alto_izquierda))

        if max_ancho < 100 or max_alto < 100:
            return pil_orig, False

        destino = np.array(
            [
                [0, 0],
                [max_ancho - 1, 0],
                [max_ancho - 1, max_alto - 1],
                [0, max_alto - 1],
            ],
            dtype="float32",
        )

        matriz_m = cv2.getPerspectiveTransform(rect, destino)
        escaneo_plano = cv2.warpPerspective(orig_np, matriz_m, (max_ancho, max_alto))

        return Image.fromarray(escaneo_plano), True

    except Exception:
        # Si ocurre cualquier excepción inesperada, retornar la imagen original sin romper la app
        return pil_orig, False


def escanear_documento(imagen_bytes: bytes) -> Image.Image:
    """
    Función principal de escaneo estilo CamScanner: recibe los bytes de la imagen
    y devuelve la imagen recortada y enderezada (bird's-eye view), o la imagen original
    si no se detectan 4 esquinas claras.
    """
    imagen_resultado, _ = escanear_documento_con_estado(imagen_bytes)
    return imagen_resultado


def optimizar_para_tesseract(pil_image: Image.Image) -> Image.Image:
    """
    Optimizador de imagen especializado para Tesseract OCR (Filtro Escáner con OpenCV):
    1. Convierte la imagen a escala de grises.
    2. Aplica reescalado (upscaling 1.5x o 2x con cv2.INTER_CUBIC) si la imagen es pequeña
       para mejorar la lectura de tipografía pequeña en tickets térmicos.
    3. Aplica reducción de ruido preservando bordes (cv2.bilateralFilter).
    4. Aplica binarización adaptativa (cv2.adaptiveThreshold con blockSize=31, C=15)
       para eliminar sombras de fotos tomadas con celular sin romper caracteres numéricos.
    """
    img_rgb = pil_image.convert("RGB")

    if not OPENCV_DISPONIBLE or cv2 is None:
        # Respaldo con Pillow si OpenCV no está instalado aún en el entorno local
        gris_pil = ImageOps.grayscale(img_rgb)
        gris_pil = ImageOps.autocontrast(gris_pil, cutoff=1)
        realce = ImageEnhance.Contrast(gris_pil).enhance(1.8)
        return realce.filter(ImageFilter.SHARPEN)

    img_np = np.array(img_rgb)
    gris = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # 2. Reescalado (upscaling) para imágenes pequeñas o tickets de baja resolución
    alto, ancho = gris.shape[:2]
    dimension_max = max(alto, ancho)
    if dimension_max < 1000:
        factor = 2.0
    elif dimension_max < 1600:
        factor = 1.5
    else:
        factor = 1.0

    if factor > 1.0:
        gris = cv2.resize(
            gris,
            None,
            fx=factor,
            fy=factor,
            interpolation=cv2.INTER_CUBIC,
        )

    # 3. Reducción de ruido suave preservando trazos finos de números en tablas
    filtrado = cv2.bilateralFilter(gris, d=5, sigmaColor=45, sigmaSpace=45)

    # 4. Binarización adaptativa equilibrada (C=9) para no borrar tipografía delgada
    binarizada = cv2.adaptiveThreshold(
        filtrado,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=9,
    )

    return Image.fromarray(binarizada)


def calcular_nitidez_imagen(pil_image: Image.Image) -> float:
    """
    Estima el nivel de enfoque/nitidez de la foto mediante la varianza del Laplaciano (OpenCV)
    o la desviación estándar de bordes (Pillow). Valores muy bajos (< 45) indican foto borrosa.
    """
    try:
        img_gris = np.array(pil_image.convert("L"))
        if OPENCV_DISPONIBLE and cv2 is not None:
            return float(cv2.Laplacian(img_gris, cv2.CV_64F).var())
        bordes_pil = pil_image.convert("L").filter(ImageFilter.FIND_EDGES)
        return float(np.var(np.array(bordes_pil)))
    except Exception:
        return 100.0


def ejecutar_ocr(imagen: Image.Image, idioma: str = "spa") -> Tuple[str, Optional[str]]:
    """
    Ejecuta Tesseract OCR sobre la imagen con la configuración óptima para facturas
    (config='--oem 3 --psm 6'). Si 'spa' no está disponible, reintenta con 'eng'.
    Captura cualquier excepción devolviendo mensajes amigables para el usuario.
    """
    config_optima = "--oem 3 --psm 6"
    try:
        texto = pytesseract.image_to_string(imagen, lang=idioma, config=config_optima)
        return texto.strip(), None
    except pytesseract.TesseractError:
        if idioma != "eng":
            try:
                texto = pytesseract.image_to_string(imagen, lang="eng", config=config_optima)
                aviso = (
                    "ℹ️ Se procesó el comprobante con el diccionario base porque no se detectó "
                    "el paquete de idioma español ('spa') en el servidor."
                )
                return texto.strip(), aviso
            except Exception:
                return (
                    "",
                    "⚠️ No se pudo leer el texto de la imagen. Intentá subir una foto más nítida y con buena luz.",
                )
        return (
            "",
            "⚠️ El motor OCR tuvo dificultades para procesar la imagen. Verificá que el archivo no esté dañado.",
        )
    except Exception:
        return (
            "",
            "⚠️ El motor de lectura óptica (Tesseract OCR) no está disponible o no pudo procesar este archivo.",
        )


def procesar_archivo_pdf(
    pdf_bytes: bytes, indice_pagina: int = 0
) -> Tuple[Optional[Image.Image], str, int, Optional[str]]:
    """
    Procesa una factura en formato PDF:
    1. Si PyMuPDF (fitz) está disponible, renderiza la página seleccionada a una imagen PIL
       en alta resolución (220 DPI) para visualización y OCR, y además extrae el texto digital
       nativo embebido en el PDF (ideal para Facturas Electrónicas AFIP/ARCA).
    2. Si solo pypdf está disponible, extrae el texto digital del PDF y genera una vista previa.
    Retorna: (imagen_pil, texto_digital_pdf, total_paginas, mensaje_error)
    """
    if not pdf_bytes or len(pdf_bytes) < 50:
        return None, "", 0, "⚠️ El archivo PDF subido está vacío o incompleto."

    if PYMUPDF_DISPONIBLE and fitz is not None:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            total_paginas = len(doc)
            if total_paginas == 0:
                doc.close()
                return None, "", 0, "⚠️ El archivo PDF no contiene páginas válidas."

            idx = max(0, min(indice_pagina, total_paginas - 1))
            pagina = doc[idx]

            # Extraer texto digital de la página seleccionada (y del resto si es multipágina)
            texto_pag = pagina.get_text("text").strip()
            if total_paginas > 1:
                textos_todos = [doc[i].get_text("text").strip() for i in range(total_paginas)]
                texto_digital = "\n\n".join(t for t in textos_todos if t)
                if texto_pag:
                    texto_digital = texto_pag + "\n\n" + texto_digital
            else:
                texto_digital = texto_pag

            # Renderizar la página en alta definición (260 DPI) para OpenCV y Tesseract OCR
            pix = pagina.get_pixmap(dpi=260, alpha=False)
            imagen_pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            doc.close()
            return imagen_pil, texto_digital.strip(), total_paginas, None
        except Exception:
            return (
                None,
                "",
                0,
                "❌ No se pudo leer el archivo PDF. Verificá que no tenga contraseña ni esté dañado.",
            )

    if PYPDF_DISPONIBLE and PdfReader is not None:
        try:
            lector = PdfReader(io.BytesIO(pdf_bytes))
            total_paginas = len(lector.pages)
            if total_paginas == 0:
                return None, "", 0, "⚠️ El archivo PDF no contiene páginas válidas."

            idx = max(0, min(indice_pagina, total_paginas - 1))
            texto_digital = (lector.pages[idx].extract_text() or "").strip()

            # Crear lienzo de vista previa con el texto digital extraído
            lienzo = Image.new("RGB", (900, 1150), (255, 255, 255))
            dibujo = ImageDraw.Draw(lienzo)
            dibujo.rectangle([(25, 25), (875, 1125)], outline=(203, 213, 225), width=2)
            dibujo.text((50, 50), f"Vista Previa PDF Digital (Página {idx + 1}/{total_paginas})", fill=(30, 41, 59))
            y_pos = 100
            for linea in (texto_digital.splitlines()[:32] if texto_digital else ["PDF sin capa de texto digital"]):
                dibujo.text((50, y_pos), linea[:95], fill=(51, 65, 85))
                y_pos += 28
            return lienzo, texto_digital, total_paginas, None
        except Exception:
            return (
                None,
                "",
                0,
                "❌ No se pudo abrir el archivo PDF. Verificá que no esté protegido ni dañado.",
            )

    return (
        None,
        "",
        0,
        "⚠️ Para procesar facturas en PDF, instalá la librería ejecutando: `pip install pymupdf`.",
    )


# ==============================================================================
# 3. MÓDULO DE AUTENTICACIÓN Y PERSISTENCIA EN SUPABASE (STORAGE + DB)
# ==============================================================================

def obtener_credenciales_supabase() -> Tuple[str, str]:
    """Obtiene SUPABASE_URL y SUPABASE_KEY desde st.secrets o variables de entorno."""
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_KEY", "").strip()
    if not url or not key:
        try:
            url = url or str(st.secrets.get("SUPABASE_URL", "")).strip()
            key = key or str(st.secrets.get("SUPABASE_KEY", "")).strip()
        except Exception:
            pass
    return url, key


@st.cache_resource(show_spinner=False)
def inicializar_cliente_supabase(url: str, key: str) -> Optional[Client]:
    """Inicializa y cachea el cliente de Supabase si las credenciales son válidas."""
    if not SUPABASE_LIB_DISPONIBLE or not url or not key or create_client is None:
        return None
    try:
        return create_client(url, key)
    except Exception:
        return None


def sanitizar_nombre_archivo(nombre: str) -> str:
    """Limpia caracteres especiales del nombre de archivo antes de subirlo a Supabase Storage."""
    base = re.sub(r"[^a-zA-Z0-9._-]", "_", nombre)
    return base or "comprobante.png"


def guardar_comprobante_en_supabase(
    cliente: Client,
    user_id: str,
    user_email: str,
    bytes_imagen: bytes,
    nombre_archivo: str,
    mime_type: str,
    datos_revisados: Dict[str, str],
    texto_ocr: str,
) -> Tuple[bool, str, Optional[str]]:
    """
    Sube la foto al bucket 'comprobantes' en la ruta {user_id}/{timestamp}_{nombre_archivo}
    y guarda los 9 campos estructurados junto con la URL de la imagen en la tabla 'facturas'.
    """
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_limpio = sanitizar_nombre_archivo(nombre_archivo)
    ruta_storage = f"{user_id}/{timestamp_str}_{nombre_limpio}"
    imagen_url = ""

    # 1. Subir imagen al bucket 'comprobantes'
    try:
        cliente.storage.from_("comprobantes").upload(
            path=ruta_storage,
            file=bytes_imagen,
            file_options={"content-type": mime_type or "image/png", "upsert": "true"},
        )
        res_url = cliente.storage.from_("comprobantes").get_public_url(ruta_storage)
        imagen_url = res_url if isinstance(res_url, str) else str(res_url)
    except Exception as exc_storage:
        # Si el bucket requiere permisos adicionales, registramos el aviso sin bloquear el guardado en tabla
        imagen_url = f"storage://comprobantes/{ruta_storage}"
        aviso_storage = f"(Aviso Storage: {exc_storage})"
    else:
        aviso_storage = ""

    # 2. Insertar registro en la tabla 'facturas'
    registro_db = {
        "user_id": user_id,
        "user_email": user_email,
        "tipo_documento": datos_revisados.get("tipo_documento", VALOR_NO_IDENTIFICADO),
        "emisor": datos_revisados.get("emisor", VALOR_NO_IDENTIFICADO),
        "cuit": datos_revisados.get("cuit", VALOR_NO_IDENTIFICADO),
        "fecha_emision": datos_revisados.get("fecha_emision", VALOR_NO_IDENTIFICADO),
        "numero_comprobante": datos_revisados.get("numero_comprobante", VALOR_NO_IDENTIFICADO),
        "importe_neto": datos_revisados.get("importe_neto", VALOR_NO_IDENTIFICADO),
        "impuestos_iva": datos_revisados.get("impuestos_iva", VALOR_NO_IDENTIFICADO),
        "importe_total": datos_revisados.get("importe_total", VALOR_NO_IDENTIFICADO),
        "moneda": datos_revisados.get("moneda", VALOR_NO_IDENTIFICADO),
        "texto_ocr": texto_ocr,
        "imagen_url": imagen_url,
        "storage_path": ruta_storage,
    }

    try:
        cliente.table("facturas").insert(registro_db).execute()
        mensaje = "¡Comprobante e imagen guardados exitosamente en Supabase!"
        if aviso_storage:
            mensaje += f" {aviso_storage}"
        return True, mensaje, imagen_url
    except Exception as exc_db:
        return False, f"Error al guardar en la tabla 'facturas' de Supabase: {exc_db}", None


def consultar_facturas_usuario_supabase(cliente: Client, user_id: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Obtiene los comprobantes guardados por el usuario autenticado desde la tabla 'facturas'."""
    try:
        respuesta = (
            cliente.table("facturas")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )
        datos = respuesta.data if hasattr(respuesta, "data") and respuesta.data else []
        return datos, None
    except Exception as exc:
        return [], f"No se pudieron recuperar las facturas desde Supabase: {exc}"


def eliminar_facturas_supabase(
    cliente: Client,
    user_id: str,
    registros_a_borrar: List[Dict[str, Any]],
) -> Tuple[bool, str]:
    """
    Elimina uno o varios comprobantes seleccionados de la tabla 'facturas' de Supabase
    (filtrando por id y user_id) e intenta limpiar su archivo asociado en Storage.
    """
    if not registros_a_borrar:
        return False, "No seleccionaste ningún comprobante para eliminar."

    ids_borrar = [str(r["id"]) for r in registros_a_borrar if r.get("id") is not None]
    rutas_storage = [str(r["storage_path"]) for r in registros_a_borrar if r.get("storage_path")]

    if not ids_borrar:
        return False, "No se encontraron identificadores válidos para eliminar."

    borrados_reales = 0
    try:
        for id_factura in ids_borrar:
            res_del = (
                cliente.table("facturas")
                .delete()
                .eq("id", id_factura)
                .eq("user_id", user_id)
                .execute()
            )
            if hasattr(res_del, "data") and res_del.data:
                borrados_reales += len(res_del.data)
    except Exception as exc_del:
        return False, f"Error al eliminar el comprobante en Supabase: {exc_del}"

    if borrados_reales == 0:
        return (
            False,
            "⚠️ Supabase bloqueó el borrado por falta de permiso RLS DELETE en la tabla `facturas`. "
            "Ejecutá en el SQL Editor de Supabase: "
            "`CREATE POLICY \"Usuarios pueden eliminar sus propias facturas\" ON public.facturas FOR DELETE TO authenticated USING (auth.uid() = user_id);`",
        )

    if rutas_storage:
        try:
            cliente.storage.from_("comprobantes").remove(rutas_storage)
        except Exception:
            pass

    texto_plural = "comprobante eliminado" if borrados_reales == 1 else f"{borrados_reales} comprobantes eliminados"
    return True, f"✅ {texto_plural} correctamente del historial."


def formatear_fecha_guardado(valor_fecha: Any) -> Tuple[str, str]:
    """Convierte timestamps ISO o locales en ('DD/MM/AAAA HH:MM:SS', 'DD/MM/AAAA')."""
    texto = str(valor_fecha or "").strip()
    if not texto:
        return ("Sin fecha", "Sin fecha")
    try:
        limpio = texto.replace("Z", "+00:00")
        dt = datetime.fromisoformat(limpio)
        return (dt.strftime("%d/%m/%Y %H:%M:%S"), dt.strftime("%d/%m/%Y"))
    except Exception:
        try:
            dt = datetime.strptime(texto[:19], "%Y-%m-%d %H:%M:%S")
            return (dt.strftime("%d/%m/%Y %H:%M:%S"), dt.strftime("%d/%m/%Y"))
        except Exception:
            return (texto, texto[:10])


def renderizar_filtros_seleccionables(df_entrada: pd.DataFrame, prefijo: str) -> pd.DataFrame:
    """
    Renderiza filtros desplegables de selección (sin escribir) para todas las categorías
    del comprobante (Fecha de comprobante, Tipo, Emisor, CUIT, Moneda, N° Comprobante,
    Importe Neto, IVA, Importe Total y Fecha de Guardado).
    """
    clave_reset = f"reset_version_{prefijo}"
    if clave_reset not in st.session_state:
        st.session_state[clave_reset] = 0
    ver = st.session_state[clave_reset]

    definicion_categorias = [
        ("fecha_emision", "📅 Fecha Comprobante"),
        ("tipo_documento", "📄 Tipo de Documento"),
        ("emisor", "🏢 Emisor / Comercio"),
        ("cuit", "🪪 CUIT"),
        ("moneda", "💱 Moneda"),
        ("numero_comprobante", "🔢 N° Comprobante"),
        ("importe_neto", "💵 Importe Neto"),
        ("impuestos_iva", "🧾 IVA / Impuestos"),
        ("importe_total", "💰 Importe Total"),
        ("_dia_guardado", "🕒 Fecha de Guardado"),
    ]

    st.markdown("##### 🎛️ Filtrar historial seleccionando por categoría")

    selecciones: Dict[str, str] = {}

    # Fila 1 (5 categorías principales)
    cols_fila_1 = st.columns(5)
    for idx, (col_df, etiqueta) in enumerate(definicion_categorias[:5]):
        with cols_fila_1[idx]:
            if col_df in df_entrada.columns:
                valores_unicos = sorted(
                    {str(v).strip() for v in df_entrada[col_df].dropna().tolist() if str(v).strip()}
                )
            else:
                valores_unicos = []
            selecciones[col_df] = st.selectbox(
                etiqueta,
                options=["Todos"] + valores_unicos,
                index=0,
                key=f"sel_{prefijo}_{col_df}_{ver}",
            )

    # Fila 2 (5 categorías adicionales: numeración, importes y fecha de carga)
    cols_fila_2 = st.columns(5)
    for idx, (col_df, etiqueta) in enumerate(definicion_categorias[5:]):
        with cols_fila_2[idx]:
            if col_df in df_entrada.columns:
                valores_unicos = sorted(
                    {str(v).strip() for v in df_entrada[col_df].dropna().tolist() if str(v).strip()}
                )
            else:
                valores_unicos = []
            selecciones[col_df] = st.selectbox(
                etiqueta,
                options=["Todos"] + valores_unicos,
                index=0,
                key=f"sel_{prefijo}_{col_df}_{ver}",
            )

    df_filtrado = df_entrada.copy()
    filtros_activos = 0
    for col_df, valor_elegido in selecciones.items():
        if valor_elegido != "Todos" and col_df in df_filtrado.columns:
            filtros_activos += 1
            df_filtrado = df_filtrado[df_filtrado[col_df].astype(str).str.strip() == valor_elegido]

    col_info_f, col_btn_reset = st.columns([4, 1], vertical_alignment="center")
    with col_info_f:
        st.caption(
            f"Mostrando **{len(df_filtrado)}** de **{len(df_entrada)}** comprobante(s) guardado(s)"
            + (f" ({filtros_activos} filtro/s activo/s)." if filtros_activos else ".")
        )
    with col_btn_reset:
        if filtros_activos > 0:
            if st.button("🔄 Limpiar filtros", key=f"btn_clear_{prefijo}", use_container_width=True):
                st.session_state[clave_reset] += 1
                st.rerun()

    return df_filtrado


def convertir_monto_a_float(valor: Any) -> float:
    """
    Convierte cadenas de importes (ej. '18.948,87', '$ 18,948.87', '18948.87')
    en valores numéricos float para cálculos y gráficos en Pandas.
    Devuelve 0.0 si el valor es 'No identificado' o inválido.
    """
    if valor is None:
        return 0.0
    if isinstance(valor, (int, float)):
        return float(valor)

    texto = str(valor).strip()
    if not texto or texto.lower() in {"no identificado", "n/a", "null", "none", "-"}:
        return 0.0

    # Conservar únicamente dígitos, puntos, comas y signo negativo
    limpio = re.sub(r"[^\d.,-]", "", texto)
    if not limpio:
        return 0.0

    try:
        if "," in limpio and "." in limpio:
            # Determinar cuál aparece último como separador decimal
            if limpio.rfind(",") > limpio.rfind("."):
                # Formato 18.948,87 -> 18948.87
                limpio = limpio.replace(".", "").replace(",", ".")
            else:
                # Formato 18,948.87 -> 18948.87
                limpio = limpio.replace(",", "")
        elif "," in limpio:
            partes = limpio.split(",")
            if len(partes[-1]) in (1, 2):
                limpio = limpio.replace(",", ".")
            else:
                limpio = limpio.replace(",", "")
        elif limpio.count(".") > 1:
            partes = limpio.split(".")
            if len(partes[-1]) in (1, 2):
                limpio = "".join(partes[:-1]) + "." + partes[-1]
            else:
                limpio = "".join(partes)

        return round(float(limpio), 2)
    except Exception:
        return 0.0


def formatear_importe_metrica(valor: Any, moneda: str = "$") -> str:
    """Formatea un importe para mostrarlo de forma clara y elegante en st.metric."""
    texto_moneda = "$" if str(moneda).strip() in ("", VALOR_NO_IDENTIFICADO) else str(moneda).strip()
    monto_num = convertir_monto_a_float(valor)
    if monto_num > 0:
        return f"{texto_moneda} {monto_num:,.2f}"
    texto_val = str(valor or "").strip()
    if not texto_val or texto_val == VALOR_NO_IDENTIFICADO:
        return "No identificado"
    return f"{texto_moneda} {texto_val}"


def extraer_periodo_y_fecha(registro: Dict[str, Any]) -> Tuple[str, str]:
    """
    Extrae ('YYYY-MM', 'YYYY-MM-DD') a partir de 'fecha_emision' (DD/MM/YYYY o YYYY-MM-DD)
    o usa como respaldo 'created_at' / 'fecha_carga'.
    """
    fecha_em = str(registro.get("fecha_emision", "") or "").strip()
    formatos_comunes = ["%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%y", "%d-%m-%y", "%Y/%m/%d"]
    for fmt in formatos_comunes:
        try:
            dt = datetime.strptime(fecha_em, fmt)
            return dt.strftime("%Y-%m"), dt.strftime("%Y-%m-%d")
        except Exception:
            continue

    # Intentar extraer patrón DD/MM/YYYY dentro del texto
    match = re.search(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", fecha_em)
    if match:
        dia, mes, anio = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if anio < 100:
            anio += 2000
        try:
            dt = datetime(anio, mes, dia)
            return dt.strftime("%Y-%m"), dt.strftime("%Y-%m-%d")
        except Exception:
            pass

    # Respaldo con fecha de guardado en nube o sesión local
    respaldo = str(registro.get("created_at") or registro.get("fecha_carga") or "").strip()
    if respaldo:
        try:
            dt = datetime.fromisoformat(respaldo.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m"), dt.strftime("%Y-%m-%d")
        except Exception:
            try:
                dt = datetime.strptime(respaldo[:10], "%Y-%m-%d")
                return dt.strftime("%Y-%m"), dt.strftime("%Y-%m-%d")
            except Exception:
                pass

    hoy = datetime.now()
    return hoy.strftime("%Y-%m"), hoy.strftime("%Y-%m-%d")


def preparar_dataframe_analitica(registros: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Transforma la lista de comprobantes en un DataFrame de Pandas enriquecido
    con columnas numéricas ('importe_total_num', 'importe_neto_num', 'iva_num')
    y temporales ('periodo_mensual', 'fecha_iso') listas para graficar.
    """
    if not registros:
        return pd.DataFrame()

    filas_procesadas = []
    for reg in registros:
        periodo_mes, fecha_iso = extraer_periodo_y_fecha(reg)
        emisor_limpio = str(reg.get("emisor", VALOR_NO_IDENTIFICADO) or VALOR_NO_IDENTIFICADO).strip()
        if not emisor_limpio or emisor_limpio == VALOR_NO_IDENTIFICADO:
            emisor_limpio = "Proveedor sin especificar"

        tipo_limpio = str(reg.get("tipo_documento", VALOR_NO_IDENTIFICADO) or VALOR_NO_IDENTIFICADO).strip()
        if not tipo_limpio or tipo_limpio == VALOR_NO_IDENTIFICADO:
            tipo_limpio = "Otro Comprobante"

        filas_procesadas.append(
            {
                "periodo_mensual": periodo_mes,
                "fecha_iso": fecha_iso,
                "emisor": emisor_limpio,
                "tipo_documento": tipo_limpio,
                "cuit": str(reg.get("cuit", VALOR_NO_IDENTIFICADO)),
                "numero_comprobante": str(reg.get("numero_comprobante", VALOR_NO_IDENTIFICADO)),
                "importe_neto_num": convertir_monto_a_float(reg.get("importe_neto")),
                "iva_num": convertir_monto_a_float(reg.get("impuestos_iva")),
                "importe_total_num": convertir_monto_a_float(reg.get("importe_total")),
            }
        )

    return pd.DataFrame(filas_procesadas)


# ==============================================================================
# 4. MÓDULO DE VISIÓN MULTI-IMAGEN (FORMATO OPENAI / GROQ) Y REFUERZO CONTABLE
# ==============================================================================

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


def detectar_unidades_por_empaque(descripcion: str) -> int:
    """
    Analiza semánticamente la descripción de un ítem para detectar si corresponde a un
    empaque múltiple (ej. 'Manaos Cola 6x3L' -> 6, 'Caja x 12' -> 12, 'Pack de 4' -> 4,
    '24x473cc' -> 24, 'Fardo x 8' -> 8). Si no hay indicios, devuelve 1 por defecto.
    """
    if not descripcion:
        return 1
    texto = str(descripcion).strip()

    # 1. Patrones tipo "6x3L", "6 x 2.25L", "24x473cc", "12x500ml", "6X1500CC"
    m_mult_medida = re.search(
        r"\b(\d{1,3})\s*[xX\*]\s*\d+(?:[.,]\d+)?\s*(?:l|lt|lts|litros?|cc|ml|cm3|g|gr|grs|kg|oz)\b",
        texto,
        re.IGNORECASE,
    )
    if m_mult_medida:
        unidades = int(m_mult_medida.group(1))
        if 1 < unidades <= 200:
            return unidades

    # 2. Patrones explícitos de empaque: "Caja x 12", "Pack de 4", "Fardo x 6", "Bulto de 24", "Display x 12"
    m_empaque = re.search(
        r"\b(?:caja|pack|fardo|bulto|display|bandeja|estuche|bolsa|sixpack)\s*(?:de|x|por|\*)?\s*(\d{1,3})\b",
        texto,
        re.IGNORECASE,
    )
    if m_empaque:
        unidades = int(m_empaque.group(1))
        if 1 < unidades <= 500:
            return unidades

    # 3. Patrones tipo "x 12 un", "x6u", "x 6 uds", "6x" seguido de capacidad
    m_unid = re.search(
        r"(?:\b[xX]\s*(\d{1,3})\s*(?:u|un|unid|unidades|uds|bot|latas?)?\b|\b(\d{1,3})\s*[xX]\s*\d+\b)",
        texto,
        re.IGNORECASE,
    )
    if m_unid:
        val_str = m_unid.group(1) or m_unid.group(2)
        if val_str:
            unidades = int(val_str)
            if 1 < unidades <= 200:
                return unidades

    return 1


def normalizar_items_extraidos(items_crudos: Any) -> List[Dict[str, Any]]:
    """
    Normaliza la lista de ítems extraídos por la IA garantizando la estructura exacta:
    - descripcion (str)
    - cantidad_facturada (float)
    - precio_linea_impreso (float)
    - unidades_por_empaque (int >= 1)
    """
    if not isinstance(items_crudos, list):
        return []

    items_limpios: List[Dict[str, Any]] = []
    for item in items_crudos:
        if not isinstance(item, dict):
            continue

        desc = str(
            item.get("descripcion")
            or item.get("producto")
            or item.get("detalle")
            or item.get("concepto")
            or ""
        ).strip()
        if not desc or desc.lower() in {"null", "none", "n/a", "no identificado"}:
            continue

        cant_raw = item.get("cantidad_facturada", item.get("cantidad", 1.0))
        cant_val = convertir_monto_a_float(cant_raw)
        if cant_val <= 0:
            try:
                cant_val = float(str(cant_raw).replace(",", ".").strip())
            except Exception:
                cant_val = 1.0
        if cant_val <= 0:
            cant_val = 1.0

        precio_raw = item.get(
            "precio_linea_impreso",
            item.get("precio_unitario", item.get("precio", item.get("importe", 0.0))),
        )
        precio_val = convertir_monto_a_float(precio_raw)
        if precio_val < 0:
            precio_val = 0.0

        unid_raw = item.get("unidades_por_empaque", None)
        unid_detectada_desc = detectar_unidades_por_empaque(desc)
        try:
            unid_val = int(float(str(unid_raw).replace(",", ".").strip())) if unid_raw is not None else 0
        except Exception:
            unid_val = 0

        # Si la IA devolvió <= 1 pero la descripción indica explícitamente un empaque múltiple (ej. 6x3L), usar el detectado
        if unid_val <= 1 and unid_detectada_desc > 1:
            unid_val = unid_detectada_desc
        elif unid_val < 1:
            unid_val = 1

        items_limpios.append(
            {
                "descripcion": desc,
                "cantidad_facturada": round(float(cant_val), 2),
                "precio_linea_impreso": round(float(precio_val), 2),
                "unidades_por_empaque": int(unid_val),
            }
        )

    return items_limpios


def extraer_items_con_regex(texto_ocr: str) -> List[Dict[str, Any]]:
    """
    Extractor de respaldo para recuperar líneas de productos/artículos desde el texto OCR
    cuando no se dispone de respuesta del modelo o como refuerzo en facturas tabulares.
    """
    if not texto_ocr or not texto_ocr.strip():
        return []

    items_detectados: List[Dict[str, Any]] = []
    palabras_excluir = (
        "subtotal",
        "total",
        "cuit",
        "ingresos brutos",
        "inicio de actividades",
        "condicion de venta",
        "responsable inscripto",
        "cae",
        "vencimiento",
        "percepc",
        "impuesto interno",
        "son pesos",
        "punto de venta",
        "comprobante",
        "domicilio",
        "cliente",
        "razon social",
        "fecha",
    )

    for linea in texto_ocr.splitlines():
        linea_limpia = linea.strip()
        if len(linea_limpia) < 8:
            continue
        linea_low = linea_limpia.lower()
        if any(p in linea_low for p in palabras_excluir):
            continue

        # Patrón común de línea de factura: [Código opcional] [Descripción con letras] [Cantidad] [Precio] [Subtotal opcional]
        montos_linea = re.findall(r"\d{1,3}(?:[.,]\d{3})+[.,]\d{2}|\d+[.,]\d{2}", linea_limpia)
        if not montos_linea:
            continue

        precio_impreso = convertir_monto_a_float(montos_linea[0])
        if precio_impreso <= 0:
            continue

        # Extraer descripción quitando códigos iniciales y columnas numéricas finales
        sin_montos = re.sub(r"[\$]?\s*(?:\d{1,3}(?:[.,]\d{3})+[.,]\d{2}|\d+[.,]\d{2}).*$", "", linea_limpia).strip()
        # Buscar si al principio o al final de sin_montos está la cantidad facturada
        m_cant_ini = re.match(r"^(\d+(?:[.,]\d+)?)\s+([A-Za-zÁÉÍÓÚÑáéíóúñ].+)$", sin_montos)
        m_cod_desc_cant = re.match(r"^(?:\d{3,14}\s+)?([A-Za-zÁÉÍÓÚÑáéíóúñ].*?)\s+(\d+(?:[.,]\d+)?)$", sin_montos)

        cantidad_fact = 1.0
        descripcion_item = sin_montos

        if m_cod_desc_cant:
            descripcion_item = m_cod_desc_cant.group(1).strip()
            c_try = convertir_monto_a_float(m_cod_desc_cant.group(2))
            if 0 < c_try <= 1000:
                cantidad_fact = c_try
        elif m_cant_ini:
            c_try = convertir_monto_a_float(m_cant_ini.group(1))
            if 0 < c_try <= 500:
                cantidad_fact = c_try
                descripcion_item = m_cant_ini.group(2).strip()

        descripcion_item = re.sub(r"^\d{4,14}\s+", "", descripcion_item).strip(" -|:")
        if len(descripcion_item) < 3 or not re.search(r"[A-Za-zÁÉÍÓÚÑáéíóúñ]{3,}", descripcion_item):
            continue

        items_detectados.append(
            {
                "descripcion": descripcion_item,
                "cantidad_facturada": round(float(cantidad_fact), 2),
                "precio_linea_impreso": round(float(precio_impreso), 2),
                "unidades_por_empaque": detectar_unidades_por_empaque(descripcion_item),
            }
        )

    return items_detectados


def calcular_dataframe_items(items_lista: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convierte la lista de ítems ('items') en un DataFrame de Pandas y realiza el cálculo
    100% en Python:
      - df['precio_unitario_real'] = df['precio_linea_impreso'] / df['unidades_por_empaque']
      - df['cantidad_total_unidades'] = df['cantidad_facturada'] * df['unidades_por_empaque']
    """
    columnas_base = [
        "descripcion",
        "cantidad_facturada",
        "precio_linea_impreso",
        "unidades_por_empaque",
    ]
    if not items_lista:
        df_vacio = pd.DataFrame(columns=columnas_base + ["precio_unitario_real", "cantidad_total_unidades"])
        return df_vacio

    df = pd.DataFrame(items_lista)
    for col in columnas_base:
        if col not in df.columns:
            df[col] = "" if col == "descripcion" else (1 if col == "unidades_por_empaque" else 0.0)

    df["descripcion"] = df["descripcion"].astype(str).fillna("")
    df["cantidad_facturada"] = pd.to_numeric(df["cantidad_facturada"], errors="coerce").fillna(1.0).clip(lower=0.0)
    df["precio_linea_impreso"] = pd.to_numeric(df["precio_linea_impreso"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["unidades_por_empaque"] = (
        pd.to_numeric(df["unidades_por_empaque"], errors="coerce")
        .fillna(1)
        .clip(lower=1)
        .round(0)
        .astype(int)
    )

    # Calculamos el precio unitario real de cada artículo individual
    df["precio_unitario_real"] = (df["precio_linea_impreso"] / df["unidades_por_empaque"]).round(2)

    # Calculamos la cantidad total de artículos individuales que entraron al inventario
    df["cantidad_total_unidades"] = (df["cantidad_facturada"] * df["unidades_por_empaque"]).round(2)

    return df[columnas_base + ["precio_unitario_real", "cantidad_total_unidades"]]


def extraer_json_de_respuesta(texto_respuesta: str) -> Dict:
    """Extrae de forma segura un bloque JSON desde la respuesta del modelo de visión."""
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


def codificar_imagen_a_base64(
    imagen_pil: Image.Image, max_lado: int = 1600, calidad: int = 85
) -> str:
    """
    Convierte una imagen PIL a formato JPEG en memoria (redimensionando si supera max_lado
    para optimizar el peso del payload multi-imagen) y devuelve su cadena en Base64.
    """
    img_rgb = imagen_pil.convert("RGB")
    ancho, alto = img_rgb.size
    lado_mayor = max(ancho, alto)
    if lado_mayor > max_lado:
        escala = max_lado / float(lado_mayor)
        nuevo_tam = (max(1, int(ancho * escala)), max(1, int(alto * escala)))
        img_rgb = img_rgb.resize(nuevo_tam, Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    img_rgb.save(buffer, format="JPEG", quality=calidad, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def preparar_imagenes_base64(archivos_subidos: List[Any]) -> List[str]:
    """
    Toma la lista de archivos subidos por el usuario (accept_multiple_files=True)
    y convierte cada imagen (o páginas de un PDF) a una cadena Base64 lista para la API de Visión.
    """
    lista_base64: List[str] = []
    for archivo in archivos_subidos:
        try:
            bytes_arch = archivo.getvalue() if hasattr(archivo, "getvalue") else bytes(archivo)
            nombre = getattr(archivo, "name", "") or ""
            mime = getattr(archivo, "type", "") or ""
            es_pdf = nombre.lower().endswith(".pdf") or "pdf" in mime.lower() or bytes_arch[:4] == b"%PDF"

            if es_pdf:
                img_pdf, _, total_pags, err_pdf = procesar_archivo_pdf(bytes_arch, indice_pagina=0)
                if not err_pdf and img_pdf is not None:
                    lista_base64.append(codificar_imagen_a_base64(img_pdf))
                    for p_idx in range(1, min(total_pags, 5)):
                        img_extra, _, _, err_ex = procesar_archivo_pdf(bytes_arch, indice_pagina=p_idx)
                        if not err_ex and img_extra is not None:
                            lista_base64.append(codificar_imagen_a_base64(img_extra))
            else:
                img_pil = ImageOps.exif_transpose(Image.open(io.BytesIO(bytes_arch))).convert("RGB")
                lista_base64.append(codificar_imagen_a_base64(img_pil))
        except Exception:
            continue
    return lista_base64


def construir_payload_vision_multi_imagen(
    lista_imagenes_base64: List[str],
    prompt_texto: str = "",
    texto_ocr_apoyo: str = "",
    forzar_json: bool = True,
) -> Dict[str, Any]:
    """
    Genera la estructura del payload para la API de LLM de Visión (formato OpenAI / Groq).
    El campo 'content' del mensaje del usuario itera sobre todas las imágenes en base64 y adjunta
    un bloque {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
    por cada foto subida, además del bloque de texto con las instrucciones.
    """
    instruccion_usuario = (
        prompt_texto.strip()
        or (
            f"Te adjunto {len(lista_imagenes_base64)} fragmento(s) fotográfico(s) de una misma factura o ticket. "
            "Analiza todas las imágenes en conjunto, une la información de principio a fin sin duplicar montos "
            "y extrae los 9 campos contables en un único objeto JSON consolidado."
        )
    )
    if texto_ocr_apoyo.strip():
        instruccion_usuario += (
            f"\n\nTEXTO AUXILIAR EXTRAÍDO POR OCR DE LOS FRAGMENTOS:\n{texto_ocr_apoyo.strip()}"
        )

    contenido_usuario: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": instruccion_usuario,
        }
    ]

    # Iterar sobre la lista de imágenes base64 (respetando el tope de 5 imágenes por llamada en Groq)
    for img_base64 in lista_imagenes_base64[:5]:
        contenido_usuario.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_base64}",
                },
            }
        )

    payload: Dict[str, Any] = {
        "model": MODELO_VISION_PRINCIPAL,
        "messages": [
            {"role": "system", "content": PROMPT_SISTEMA},
            {"role": "user", "content": contenido_usuario},
        ],
        "temperature": 0.0,
    }
    if forzar_json:
        payload["response_format"] = {"type": "json_object"}

    return payload


def extraer_con_vision_multimodal(
    lista_imagenes_base64: List[str],
    api_key: str,
    texto_ocr_apoyo: str = "",
    endpoint_url: str = ENDPOINT_VISION_PRINCIPAL,
) -> Tuple[Dict[str, str], Optional[str]]:
    """
    Envía el payload multi-imagen al modelo de visión fijo (MODELO_VISION_PRINCIPAL)
    usando el formato estándar OpenAI / Groq (/openai/v1/chat/completions).
    """
    clave_limpia = api_key.strip().strip('"').strip("'")
    if not clave_limpia:
        return (
            normalizar_diccionario_extraido({}),
            "Falta ingresar la clave de API (GROQ_API_KEY) en el panel lateral.",
        )

    # Soporte transparente si el usuario pegó una clave de Google Gemini (AIza...)
    if clave_limpia.startswith("AIza"):
        try:
            url_gem = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={clave_limpia}"
            partes_gem: List[Dict[str, Any]] = [
                {"text": f"{PROMPT_SISTEMA}\n\nTEXTO OCR AUXILIAR:\n{texto_ocr_apoyo}"}
            ]
            for img_b64 in lista_imagenes_base64[:5]:
                partes_gem.append({"inline_data": {"mime_type": "image/jpeg", "data": img_b64}})
            resp_g = requests.post(
                url_gem,
                json={
                    "contents": [{"parts": partes_gem}],
                    "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"},
                },
                timeout=60,
            )
            if resp_g.ok:
                cand = resp_g.json().get("candidates", [])
                txt_g = cand[0]["content"]["parts"][0]["text"] if cand else "{}"
                datos_gem = extraer_json_de_respuesta(txt_g)
                items_gem = normalizar_items_extraidos(datos_gem.get("items", []))
                if not items_gem and texto_ocr_apoyo:
                    items_gem = extraer_items_con_regex(texto_ocr_apoyo)
                st.session_state["items_actuales"] = items_gem
                return normalizar_diccionario_extraido(datos_gem), None
        except Exception:
            pass

    headers = {
        "Authorization": f"Bearer {clave_limpia}",
        "Content-Type": "application/json",
    }

    payload = construir_payload_vision_multi_imagen(
        lista_imagenes_base64=lista_imagenes_base64,
        texto_ocr_apoyo=texto_ocr_apoyo,
        forzar_json=True,
    )

    def construir_payload_texto_consolidado(modelo_id: str, usar_json_mode: bool = True) -> Dict[str, Any]:
        cuerpo_txt: Dict[str, Any] = {
            "model": modelo_id,
            "messages": [
                {"role": "system", "content": PROMPT_SISTEMA},
                {
                    "role": "user",
                    "content": (
                        f"Fragmentos procesados ({len(lista_imagenes_base64)} imágenes de la misma factura).\n\n"
                        f"TEXTO CONSOLIDADO EXTRAÍDO DE TODOS LOS FRAGMENTOS:\n{texto_ocr_apoyo}"
                    ),
                },
            ],
            "temperature": 0.0,
        }
        if usar_json_mode:
            cuerpo_txt["response_format"] = {"type": "json_object"}
        return cuerpo_txt

    try:
        resp = requests.post(endpoint_url, headers=headers, json=payload, timeout=60)

        # Si el modelo (ej. openai/gpt-oss-120b) espera contenido consolidado en texto o sin response_format
        if resp.status_code in (400, 404, 422):
            resp_txt = requests.post(
                endpoint_url,
                headers=headers,
                json=construir_payload_texto_consolidado(MODELO_VISION_PRINCIPAL, usar_json_mode=True),
                timeout=60,
            )
            if resp_txt.status_code == 400:
                resp_txt = requests.post(
                    endpoint_url,
                    headers=headers,
                    json=construir_payload_texto_consolidado(MODELO_VISION_PRINCIPAL, usar_json_mode=False),
                    timeout=60,
                )
            if resp_txt.ok:
                resp = resp_txt
            else:
                # Respaldo con los otros modelos activos de la cuenta (qwen/qwen3.8-27b, openai/gpt-oss-20b)
                for mod_alt in ("qwen/qwen3.8-27b", "openai/gpt-oss-20b", "meta-llama/llama-4-scout-17b-16e-instruct"):
                    r_alt = requests.post(
                        endpoint_url,
                        headers=headers,
                        json=construir_payload_texto_consolidado(mod_alt, usar_json_mode=False),
                        timeout=60,
                    )
                    if r_alt.ok:
                        resp = r_alt
                        break

        if not resp.ok:
            detalle = ""
            try:
                detalle = resp.json().get("error", {}).get("message", "")
            except Exception:
                detalle = resp.text
            return (
                normalizar_diccionario_extraido({}),
                f"Aviso de API ({resp.status_code}): {detalle or resp.reason}",
            )

        contenido = resp.json()["choices"][0]["message"]["content"]
        datos = extraer_json_de_respuesta(contenido)
        items_ia = normalizar_items_extraidos(datos.get("items", []))
        if not items_ia and texto_ocr_apoyo:
            items_ia = extraer_items_con_regex(texto_ocr_apoyo)
        st.session_state["items_actuales"] = items_ia
        return normalizar_diccionario_extraido(datos), None
    except Exception as exc:
        return (
            normalizar_diccionario_extraido({}),
            f"Error de conexión con la API ({MODELO_VISION_PRINCIPAL}): {exc}",
        )


def extraer_montos_contables_ocr(
    texto_ocr: str, total_conocido: float = 0.0
) -> Tuple[str, str, str]:
    """
    Analiza el texto OCR (incluyendo tablas horizontales de pie de factura como en
    Facturas A de Salta Refrescos / AFIP donde los encabezados están arriba y los montos abajo)
    para identificar con precisión contable: (importe_neto, impuestos_iva, importe_total).
    """
    patron_monto = re.compile(r"(?<!\d)(\d{1,3}(?:[.,]\d{3})+[.,]\d{2}|\d{2,7}[.,]\d{2})(?!\d)")

    # 1. Buscar candidato a TOTAL explícito
    monto_total = total_conocido
    matches_total = re.findall(
        r"TOTAL\s*(?:A\s*PAGAR|FINAL|GENERAL)?\s*[:$]?\s*\$?\s*(\d{1,3}(?:[.,]\d{3})+[.,]\d{2}|\d+[.,]\d{2})",
        texto_ocr,
        re.IGNORECASE,
    )
    if matches_total and monto_total <= 0:
        candidatos_t = [convertir_monto_a_float(m) for m in matches_total]
        candidatos_t = [c for c in candidatos_t if c > 0]
        if candidatos_t:
            monto_total = max(candidatos_t)

    # Extraer montos línea por línea (enfocándonos en filas con importes decimales)
    lineas = texto_ocr.splitlines()
    filas_montos: List[List[float]] = []
    todos_montos_positivos: List[float] = []

    for lin in lineas:
        # Ignorar líneas que sean claramente CUIT o fechas o CAE
        if re.search(r"\b(CUIT|CAE|VTO|IIBB|ING\.?\s*BRUTOS)\b", lin, re.IGNORECASE):
            continue
        encontrados = [convertir_monto_a_float(m) for m in patron_monto.findall(lin)]
        positivos = [v for v in encontrados if v > 0.0]
        if positivos:
            filas_montos.append(positivos)
            todos_montos_positivos.extend(positivos)

    # Si aún no tenemos monto_total, revisar filas de resumen horizontal (3 o más montos en una línea)
    if monto_total <= 0 and filas_montos:
        for fila in reversed(filas_montos):
            if len(fila) >= 2:
                posible_max = max(fila)
                # Verificar si la suma de los anteriores da el último monto de la fila
                if len(fila) >= 3 and abs(sum(fila[:-1]) - fila[-1]) <= 1.50:
                    monto_total = fila[-1]
                    break
                if posible_max > monto_total:
                    monto_total = posible_max

    neto_val = 0.0
    iva_val = 0.0

    # 2. Estrategia A: Fila de tabla horizontal (ej: $29.813,34  $2.592,57  $6.260,80  $0,00  $38.666,71)
    for fila in reversed(filas_montos):
        if len(fila) >= 2:
            ultimo = fila[-1]
            previos = [x for x in fila[:-1] if x > 0 and abs(x - ultimo) > 0.01]
            if previos and abs(sum(previos) - ultimo) <= 1.50:
                if monto_total <= 0 or abs(ultimo - monto_total) <= 1.50:
                    monto_total = ultimo
                    neto_val = max(previos)
                    iva_val = round(monto_total - neto_val, 2)
                    break

    # 3. Estrategia B: Ecuación contable e identificación por alícuota de IVA (21%, 10.5%, 27%)
    if (neto_val <= 0 or iva_val <= 0) and todos_montos_positivos:
        # Tomar los montos de la mitad inferior del comprobante (donde siempre está el pie de factura)
        mitad = max(0, len(todos_montos_positivos) - 18)
        candidatos_pie = []
        for v in todos_montos_positivos[mitad:]:
            if v not in candidatos_pie:
                candidatos_pie.append(v)

        if monto_total <= 0 and candidatos_pie:
            monto_total = max(candidatos_pie)

        menores_al_total = [v for v in candidatos_pie if 0 < v < (monto_total - 0.05)]

        # B.1: Buscar si existe un par (Neto, IVA) donde Neto * 0.21 == IVA (o 10.5% / 27%)
        for cand_neto in sorted(menores_al_total, reverse=True):
            if monto_total > 0 and cand_neto < 0.40 * monto_total:
                continue
            for tasa in (0.21, 0.105, 0.27):
                iva_esperado = round(cand_neto * tasa, 2)
                for cand_iva in menores_al_total:
                    if cand_iva == cand_neto:
                        continue
                    tolerancia = max(0.50, cand_neto * 0.002)
                    if abs(cand_iva - iva_esperado) <= tolerancia:
                        neto_val = cand_neto
                        # Si la suma con otros impuestos da el total, usamos (Total - Neto) como total de impuestos
                        if monto_total > neto_val:
                            iva_val = round(monto_total - neto_val, 2)
                        else:
                            iva_val = cand_iva
                        break
                if neto_val > 0:
                    break
            if neto_val > 0:
                break

        # B.2: Si aún no se halló por tasa IVA, buscar subconjunto de 2 o 3 montos que sumen exactamente el TOTAL
        if neto_val <= 0 and monto_total > 0 and len(menores_al_total) >= 2:
            n_elem = len(menores_al_total)
            for i in range(n_elem):
                for j in range(i + 1, n_elem):
                    a, b = menores_al_total[i], menores_al_total[j]
                    if abs((a + b) - monto_total) <= 1.50:
                        neto_val = max(a, b)
                        iva_val = round(monto_total - neto_val, 2)
                        break
                    for k in range(j + 1, n_elem):
                        c = menores_al_total[k]
                        if abs((a + b + c) - monto_total) <= 1.50:
                            neto_val = max(a, b, c)
                            iva_val = round(monto_total - neto_val, 2)
                            break
                    if neto_val > 0:
                        break
                if neto_val > 0:
                    break

    # 4. Estrategia C: Etiquetas explícitas en línea ("Subtotal Neto", "Importe Neto", "IVA 21")
    if neto_val <= 0:
        match_neto = re.search(
            r"(?:SUBTOTAL\s*NETO|IMPORTE\s*NETO(?:\s*GRAVADO)?|NETO\s*GRAVADO)\s*[:$]?\s*\$?\s*(\d{1,3}(?:[.,]\d{3})+[.,]\d{2}|\d+[.,]\d{2})",
            texto_ocr,
            re.IGNORECASE,
        )
        if match_neto:
            cand = convertir_monto_a_float(match_neto.group(1))
            if cand > 0 and (monto_total <= 0 or cand <= monto_total):
                neto_val = cand

    if iva_val <= 0:
        matches_imp = re.findall(
            r"(?:IVA\s*(?:21|10[.,]5|27)?\s*%?|IMPUESTO\s*INTERNO)\s*[:$]?\s*\$?\s*(\d{1,3}(?:[.,]\d{3})+[.,]\d{2}|\d+[.,]\d{2})",
            texto_ocr,
            re.IGNORECASE,
        )
        if matches_imp:
            vals_imp = [convertir_monto_a_float(m) for m in matches_imp]
            vals_imp = [v for v in vals_imp if 0 < v < (monto_total if monto_total > 0 else 1e12)]
            if vals_imp:
                iva_val = round(sum(vals_imp), 2)

    # Completar por diferencia contable si tenemos 2 de los 3 valores
    if monto_total > 0 and neto_val > 0 and iva_val <= 0 and monto_total > neto_val:
        iva_val = round(monto_total - neto_val, 2)
    elif monto_total > 0 and iva_val > 0 and neto_val <= 0 and monto_total > iva_val:
        neto_val = round(monto_total - iva_val, 2)

    str_neto = f"{neto_val:.2f}" if neto_val > 0 else VALOR_NO_IDENTIFICADO
    str_iva = f"{iva_val:.2f}" if iva_val > 0 else VALOR_NO_IDENTIFICADO
    str_total = f"{monto_total:.2f}" if monto_total > 0 else VALOR_NO_IDENTIFICADO
    return str_neto, str_iva, str_total


def extraer_con_reglas_regex(texto_ocr: str) -> Dict[str, str]:
    """
    Extractor heurístico de respaldo basado en expresiones regulares y análisis contable de tablas.
    También se utiliza como refuerzo post-procesamiento para completar montos, CUIT y fechas.
    """
    datos = {clave: VALOR_NO_IDENTIFICADO for clave in CAMPOS_COMPROBANTE}

    match_tipo = re.search(
        r"\b(FACTURA\s*[ABC]|TICKET\s*FACTURA\s*[ABC]|TICKET|RECIBO|NOTA\s*DE\s*CR[EÉ]DITO)\b",
        texto_ocr,
        re.IGNORECASE,
    )
    if match_tipo:
        datos["tipo_documento"] = match_tipo.group(1).upper()

    match_cuit = re.search(r"\b(20|23|24|27|30|33|34)[-\s]?(\d{8})[-\s]?(\d)\b", texto_ocr)
    if match_cuit:
        datos["cuit"] = f"{match_cuit.group(1)}-{match_cuit.group(2)}-{match_cuit.group(3)}"

    match_fecha = re.search(r"\b(0?[1-9]|[12]\d|3[01])[/-](0?[1-9]|1[0-2])[/-](20\d{2}|\d{2})\b", texto_ocr)
    if match_fecha:
        dia, mes, anio = match_fecha.groups()
        anio = f"20{anio}" if len(anio) == 2 else anio
        datos["fecha_emision"] = f"{int(dia):02d}/{int(mes):02d}/{anio}"

    match_nro = re.search(r"\b(\d{4,5})\s*[-–]\s*(\d{8})\b", texto_ocr)
    if match_nro:
        datos["numero_comprobante"] = f"{match_nro.group(1)}-{match_nro.group(2)}"

    str_neto, str_iva, str_total = extraer_montos_contables_ocr(texto_ocr)
    datos["importe_neto"] = str_neto
    datos["impuestos_iva"] = str_iva
    datos["importe_total"] = str_total

    if re.search(r"\b(USD|U\$S|D[OÓ]LARES)\b", texto_ocr, re.IGNORECASE):
        datos["moneda"] = "USD"
    elif "$" in texto_ocr or re.search(r"\b(ARS|PESOS)\b", texto_ocr, re.IGNORECASE):
        datos["moneda"] = "ARS"

    return datos


def aplicar_refuerzo_regex(datos_ia: Dict[str, str], texto_ocr: str) -> Dict[str, str]:
    """
    Mejora híbrida: si el LLM omitió el CUIT, la fecha, el número de comprobante
    o los montos de una tabla horizontal (importe_neto, impuestos_iva, importe_total),
    los completa y valida matemáticamente con el analizador contable.
    """
    resultado = dict(datos_ia)
    total_ia = convertir_monto_a_float(resultado.get("importe_total"))
    neto_ocr, iva_ocr, total_ocr = extraer_montos_contables_ocr(texto_ocr, total_conocido=total_ia)
    respaldo = extraer_con_reglas_regex(texto_ocr)

    for campo_clave in ["cuit", "fecha_emision", "numero_comprobante", "moneda", "tipo_documento"]:
        if resultado.get(campo_clave) == VALOR_NO_IDENTIFICADO and respaldo.get(campo_clave) != VALOR_NO_IDENTIFICADO:
            resultado[campo_clave] = respaldo[campo_clave]

    # Completar o corregir importe_total
    if total_ia <= 0 and total_ocr != VALOR_NO_IDENTIFICADO:
        resultado["importe_total"] = total_ocr
        total_ia = convertir_monto_a_float(total_ocr)

    # Completar o corregir importe_neto (incluyendo si el LLM puso el total como subtotal por error)
    neto_ia = convertir_monto_a_float(resultado.get("importe_neto"))
    if (neto_ia <= 0 or (total_ia > 0 and abs(neto_ia - total_ia) < 0.01 and convertir_monto_a_float(neto_ocr) > 0 and convertir_monto_a_float(neto_ocr) < total_ia)):
        if neto_ocr != VALOR_NO_IDENTIFICADO:
            resultado["importe_neto"] = neto_ocr
            neto_ia = convertir_monto_a_float(neto_ocr)

    # Completar o corregir impuestos_iva
    iva_ia = convertir_monto_a_float(resultado.get("impuestos_iva"))
    if iva_ia <= 0:
        if iva_ocr != VALOR_NO_IDENTIFICADO:
            resultado["impuestos_iva"] = iva_ocr
        elif total_ia > 0 and neto_ia > 0 and total_ia > neto_ia:
            resultado["impuestos_iva"] = f"{round(total_ia - neto_ia, 2):.2f}"

    # Si tenemos total e impuestos pero faltó el neto, deducirlo contablemente
    if convertir_monto_a_float(resultado.get("importe_neto")) <= 0:
        iva_final = convertir_monto_a_float(resultado.get("impuestos_iva"))
        if total_ia > 0 and iva_final > 0 and total_ia > iva_final:
            resultado["importe_neto"] = f"{round(total_ia - iva_final, 2):.2f}"

    return resultado


# ==============================================================================
# 5. MÓDULO DE EVALUACIÓN DE EXACTITUD (COMPARACIÓN CON REFERENCIA MANUAL)
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
# 6. REDISEÑO UI/UX: ESTÉTICA AI FINTECH SAAS (INSPIRADA EN CALIPSO CAPTIA Y MONICA.IM)
# ==============================================================================

def inicializar_estado_sesion() -> None:
    if "texto_ocr" not in st.session_state:
        st.session_state["texto_ocr"] = ""
    if "datos_actuales" not in st.session_state:
        st.session_state["datos_actuales"] = {k: VALOR_NO_IDENTIFICADO for k in CAMPOS_COMPROBANTE}
    if "items_actuales" not in st.session_state:
        st.session_state["items_actuales"] = []
    if "historial_comprobantes" not in st.session_state:
        st.session_state["historial_comprobantes"] = []
    if "usuario_supabase" not in st.session_state:
        st.session_state["usuario_supabase"] = None
    if "mostrar_panel_auth" not in st.session_state:
        st.session_state["mostrar_panel_auth"] = False
    if "ultimo_archivo_bytes" not in st.session_state:
        st.session_state["ultimo_archivo_bytes"] = None
    if "ultimo_archivo_nombre" not in st.session_state:
        st.session_state["ultimo_archivo_nombre"] = "comprobante.png"
    if "ultimo_archivo_mime" not in st.session_state:
        st.session_state["ultimo_archivo_mime"] = "image/png"


def aplicar_estilos_saas() -> None:
    """
    Inyecta CSS personalizado con estética AI FinTech SaaS moderna, limpia y luminosa
    inspirada en Calipso Captia (bandeja de indexación B2B) y Monica.im Invoice Scanner
    (dropzone interactivo, degradados índigo-azul, píldoras de estado y tarjetas limpias).
    """
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

        :root {
            --bg-app: #F8FAFC;
            --bg-sidebar: #FFFFFF;
            --bg-card: #FFFFFF;
            --bg-subtle: #F1F5F9;
            --border-card: #E2E8F0;
            --border-indigo: #C7D2FE;
            --brand-primary: #4F46E5;
            --brand-secondary: #2563EB;
            --brand-violet: #7C3AED;
            --brand-soft: #EEF2FF;
            --text-main: #0F172A;
            --text-muted: #475569;
            --text-light: #64748B;
            --success-bg: #ECFDF5;
            --success-text: #059669;
            --warning-bg: #FFFBEB;
            --warning-text: #D97706;

            /* Variables nativas de Streamlit y Glide Data Grid (st.data_editor / st.dataframe) */
            --primary-color: #4F46E5;
            --background-color: #F8FAFC;
            --secondary-background-color: #FFFFFF;
            --text-color: #0F172A;
            --gdg-bg-cell: #FFFFFF;
            --gdg-bg-cell-medium: #F8FAFC;
            --gdg-bg-header: #F1F5F9;
            --gdg-bg-header-has-focus: #E2E8F0;
            --gdg-bg-header-hovered: #E2E8F0;
            --gdg-bg-bubble: #EEF2FF;
            --gdg-bg-bubble-selected: #4F46E5;
            --gdg-bg-search-result: rgba(79, 70, 229, 0.14);
            --gdg-border-color: #E2E8F0;
            --gdg-horizontal-border-color: #E2E8F0;
            --gdg-drilldown-border: #4F46E5;
            --gdg-text-dark: #0F172A;
            --gdg-text-medium: #475569;
            --gdg-text-light: #64748B;
            --gdg-text-bubble: #4F46E5;
            --gdg-text-header: #0F172A;
            --gdg-text-header-selected: #4F46E5;
            --gdg-accent-color: #4F46E5;
            --gdg-accent-fg: #FFFFFF;
            --gdg-accent-light: rgba(79, 70, 229, 0.12);
            color-scheme: light !important;
        }

        /* Fondo general luminoso con aura suave estilo Monica.im */
        html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"] {
            font-family: 'Inter', 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
            background-color: var(--bg-app) !important;
            background-image:
                radial-gradient(ellipse 80% 42% at 50% -8%, rgba(99, 102, 241, 0.13) 0%, rgba(56, 189, 248, 0.08) 45%, transparent 75%) !important;
            background-attachment: fixed !important;
            color: var(--text-main) !important;
        }

        [data-testid="stHeader"] {
            background-color: rgba(248, 250, 252, 0.82) !important;
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(226, 232, 240, 0.7);
        }

        .block-container {
            padding-top: 4.6rem !important;
            padding-bottom: 3.5rem !important;
            max-width: 1300px !important;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            color: var(--text-main) !important;
            letter-spacing: -0.025em;
            font-weight: 700 !important;
        }

        p, li, span, label, [data-testid="stMarkdownContainer"] p {
            color: var(--text-main);
        }

        [data-testid="stCaptionContainer"], small {
            color: var(--text-light) !important;
        }

        [data-testid="stWidgetLabel"] p {
            color: var(--text-main) !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
        }

        /* Barra de navegación superior SaaS */
        .saas-navbar {
            background: rgba(255, 255, 255, 0.96);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-card);
            border-radius: 12px;
            padding: 0.7rem 1.25rem;
            margin: 0;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            flex-wrap: wrap;
            box-shadow: 0 2px 12px rgba(15, 23, 42, 0.04);
        }

        .saas-brand {
            display: flex;
            align-items: center;
            gap: 0.7rem;
        }

        .saas-logo-icon {
            width: 34px;
            height: 34px;
            border-radius: 9px;
            background: linear-gradient(135deg, #4F46E5 0%, #2563EB 100%);
            color: #FFFFFF !important;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 1rem;
            box-shadow: 0 4px 10px rgba(79, 70, 229, 0.28);
        }

        .saas-brand-title {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-weight: 800;
            font-size: 1.02rem;
            color: var(--text-main) !important;
            letter-spacing: -0.02em;
        }

        .saas-credits {
            font-size: 0.82rem;
            color: var(--text-muted) !important;
            font-weight: 500;
        }

        .saas-status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            font-size: 0.78rem;
            font-weight: 600;
            padding: 0.3rem 0.75rem;
            border-radius: 999px;
            background: #EEF2FF;
            color: #4338CA !important;
            border: 1px solid #C7D2FE;
        }

        /* Hero SaaS limpio estilo Monica.im + Calipso Captia */
        .saas-hero {
            text-align: center;
            padding: 2.1rem 1.5rem 1.5rem 1.5rem;
            max-width: 920px;
            margin: 0 auto 0.6rem auto;
        }

        .saas-hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.38rem 0.95rem;
            border-radius: 999px;
            background: linear-gradient(90deg, rgba(79, 70, 229, 0.10) 0%, rgba(37, 99, 235, 0.10) 100%);
            border: 1px solid rgba(99, 102, 241, 0.28);
            color: #4338CA !important;
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 1rem;
        }

        .saas-hero-title {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            font-size: clamp(2rem, 3.6vw, 3rem) !important;
            font-weight: 800 !important;
            color: #0F172A !important;
            line-height: 1.15 !important;
            margin: 0 0 0.85rem 0 !important;
            letter-spacing: -0.03em !important;
        }

        .saas-gradient-text {
            background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #0284C7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .saas-hero-subtitle {
            font-size: 1.04rem;
            color: #475569 !important;
            max-width: 700px;
            margin: 0 auto 1.45rem auto;
            line-height: 1.6;
            font-weight: 400;
        }

        /* Tira de 3 pasos estilo Monica.im / Calipso Captia */
        .saas-steps-row {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.9rem;
            max-width: 900px;
            margin: 0 auto 1.2rem auto;
        }

        @media (max-width: 768px) {
            .saas-steps-row {
                grid-template-columns: 1fr;
            }
        }

        .saas-step-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 0.75rem 1rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            text-align: left;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.025);
        }

        .saas-step-num {
            min-width: 28px;
            height: 28px;
            border-radius: 8px;
            background: #EEF2FF;
            color: #4F46E5 !important;
            font-weight: 800;
            font-size: 0.82rem;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border: 1px solid #C7D2FE;
        }

        .saas-step-text strong {
            display: block;
            font-size: 0.85rem;
            color: #0F172A !important;
            font-weight: 700;
        }

        .saas-step-text span {
            font-size: 0.76rem;
            color: #64748B !important;
        }

        /* Encabezados de sección en tarjetas estilo Calipso Captia */
        .captia-card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            margin-bottom: 0.85rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid #F1F5F9;
            flex-wrap: wrap;
        }

        .captia-card-title {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 1.12rem;
            font-weight: 700;
            color: #0F172A !important;
            margin: 0;
        }

        .captia-chip {
            font-size: 0.75rem;
            font-weight: 700;
            padding: 0.26rem 0.7rem;
            border-radius: 999px;
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
        }

        .captia-chip-indigo {
            background: #EEF2FF;
            color: #4F46E5 !important;
            border: 1px solid #C7D2FE;
        }

        .captia-chip-success {
            background: #ECFDF5;
            color: #059669 !important;
            border: 1px solid #A7F3D0;
        }

        .captia-chip-warning {
            background: #FFFBEB;
            color: #D97706 !important;
            border: 1px solid #FDE68A;
        }

        /* Mini tarjetas KPI de resumen rápido en bandeja de validación */
        .captia-kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.65rem;
            margin-bottom: 1rem;
        }

        @media (max-width: 768px) {
            .captia-kpi-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }

        .captia-kpi-box {
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 0.6rem 0.75rem;
        }

        .captia-kpi-label {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748B !important;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .captia-kpi-val {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 0.9rem;
            font-weight: 700;
            color: #0F172A !important;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        /* Sidebar limpia estilo panel SaaS */
        [data-testid="stSidebar"] {
            background-color: var(--bg-sidebar) !important;
            border-right: 1px solid var(--border-card) !important;
        }

        [data-testid="stSidebar"] * {
            color: var(--text-main);
        }

        /* Tarjetas principales (st.container(border=True)) */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border-card) !important;
            border-radius: 16px !important;
            padding: 1.5rem 1.6rem !important;
            box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.03), 0 12px 28px -6px rgba(15, 23, 42, 0.05) !important;
        }

        /* Pestañas (st.tabs) estilo Segmented Control moderno */
        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            justify-content: center !important;
            gap: 0.5rem !important;
            background-color: #EEF2F6 !important;
            padding: 0.38rem !important;
            border-radius: 12px !important;
            border: 1px solid #E2E8F0 !important;
            max-width: 860px;
            margin: 0 auto 1.6rem auto !important;
        }

        [data-testid="stTabs"] [data-baseweb="tab"] {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            font-size: 0.9rem !important;
            font-weight: 600 !important;
            color: var(--text-muted) !important;
            background-color: transparent !important;
            border-radius: 9px !important;
            border: none !important;
            padding: 0.58rem 1.1rem !important;
            transition: all 0.2s ease !important;
        }

        [data-testid="stTabs"] [data-baseweb="tab"]:hover {
            color: var(--brand-primary) !important;
        }

        [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
            background-color: #FFFFFF !important;
            color: var(--brand-primary) !important;
            font-weight: 700 !important;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08) !important;
        }

        [data-testid="stTabs"] [data-baseweb="tab-highlight"],
        [data-testid="stTabs"] [data-baseweb="tab-border"] {
            display: none !important;
        }

        /* Botones principales con degradado Índigo-Azul estilo Monica.im / Captia */
        .stButton > button[kind="primary"],
        [data-testid="stBaseButton-primary"],
        .stFormSubmitButton > button {
            background: linear-gradient(135deg, #4F46E5 0%, #6366F1 50%, #2563EB 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            padding: 0.62rem 1.2rem !important;
            box-shadow: 0 6px 18px -4px rgba(79, 70, 229, 0.42) !important;
            transition: transform 0.15s ease, box-shadow 0.2s ease, opacity 0.2s ease !important;
        }

        .stButton > button[kind="primary"] *,
        [data-testid="stBaseButton-primary"] *,
        .stFormSubmitButton > button * {
            color: #FFFFFF !important;
        }

        .stButton > button[kind="primary"]:hover,
        [data-testid="stBaseButton-primary"]:hover,
        .stFormSubmitButton > button:hover {
            box-shadow: 0 10px 24px -4px rgba(79, 70, 229, 0.55) !important;
            opacity: 0.96 !important;
        }

        /* Botones secundarios y de exportación */
        .stButton > button[kind="secondary"],
        [data-testid="stBaseButton-secondary"],
        .stDownloadButton > button {
            background-color: #FFFFFF !important;
            color: var(--text-main) !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            padding: 0.58rem 1rem !important;
            transition: all 0.2s ease !important;
        }

        .stButton > button[kind="secondary"] *,
        [data-testid="stBaseButton-secondary"] *,
        .stDownloadButton > button * {
            color: var(--text-main) !important;
        }

        .stButton > button[kind="secondary"]:hover,
        [data-testid="stBaseButton-secondary"]:hover,
        .stDownloadButton > button:hover {
            background-color: var(--brand-soft) !important;
            border-color: var(--brand-primary) !important;
            color: var(--brand-primary) !important;
        }

        /* Toggles, Radio buttons y Checkboxes */
        input[type="radio"], input[type="checkbox"] {
            accent-color: var(--brand-primary) !important;
        }

        /* Inputs, Textareas y Selectboxes */
        [data-baseweb="input"] > div,
        [data-baseweb="textarea"] > div,
        [data-baseweb="select"] > div {
            background-color: #F8FAFC !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 10px !important;
            color: var(--text-main) !important;
        }

        [data-baseweb="input"] input,
        [data-baseweb="textarea"] textarea,
        [data-baseweb="select"] * {
            color: var(--text-main) !important;
            background-color: transparent !important;
        }

        [data-baseweb="select"] > div {
            cursor: pointer !important;
        }

        [data-baseweb="select"] input {
            caret-color: transparent !important;
            cursor: pointer !important;
        }

        [data-baseweb="input"] > div:focus-within,
        [data-baseweb="textarea"] > div:focus-within,
        [data-baseweb="select"] > div:focus-within {
            background-color: #FFFFFF !important;
            border-color: var(--brand-primary) !important;
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.14) !important;
        }

        [data-baseweb="popover"], [data-baseweb="menu"], ul[role="listbox"] {
            background-color: #FFFFFF !important;
            color: var(--text-main) !important;
            border: 1px solid var(--border-card) !important;
            border-radius: 10px !important;
        }

        li[role="option"] {
            color: var(--text-main) !important;
            background-color: #FFFFFF !important;
        }

        li[role="option"]:hover, li[role="option"][aria-selected="true"] {
            background-color: var(--brand-soft) !important;
            color: var(--brand-primary) !important;
        }

        /* Dropzone de carga de facturas estilo Monica.im Invoice Scanner */
        [data-testid="stFileUploader"] section,
        [data-testid="stFileUploaderDropzone"] {
            background: linear-gradient(180deg, #F8FAFF 0%, #F1F5FE 100%) !important;
            border: 2px dashed #A5B4FC !important;
            border-radius: 14px !important;
            padding: 1.65rem 1.25rem !important;
            color: var(--text-main) !important;
            transition: all 0.22s ease !important;
        }

        [data-testid="stFileUploader"] section:hover,
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: var(--brand-primary) !important;
            background: #EEF2FF !important;
            box-shadow: 0 8px 22px -6px rgba(79, 70, 229, 0.18) !important;
        }

        [data-testid="stFileUploaderDropzone"] button {
            background-color: #FFFFFF !important;
            color: var(--brand-primary) !important;
            border: 1px solid #C7D2FE !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
        }

        /* Expanders y Tablas */
        [data-testid="stExpander"], details {
            background-color: #F8FAFC !important;
            border: 1px solid var(--border-card) !important;
            border-radius: 10px !important;
            overflow: hidden;
            margin-bottom: 0.9rem !important;
        }

        [data-testid="stExpander"] summary, details > summary {
            background-color: #F8FAFC !important;
            color: var(--text-main) !important;
            font-weight: 600 !important;
            padding: 0.7rem 1rem !important;
        }

        [data-testid="stExpanderDetails"] {
            background-color: #FFFFFF !important;
            border-top: 1px solid var(--border-card) !important;
            padding: 1rem !important;
        }

        [data-testid="stDataFrame"], [data-testid="stDataEditor"] {
            border: 1px solid var(--border-card) !important;
            border-radius: 10px !important;
            background-color: #FFFFFF !important;
            overflow: hidden;
        }

        [data-testid="stMetric"] {
            background-color: #F8FAFC !important;
            border: 1px solid var(--border-card) !important;
            border-radius: 12px !important;
            padding: 0.65rem 0.75rem !important;
        }

        [data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"] p {
            font-size: 0.75rem !important;
            font-weight: 600 !important;
            color: var(--text-muted) !important;
            margin-bottom: 0.15rem !important;
        }

        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] > div {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            font-size: 0.98rem !important;
            font-weight: 700 !important;
            line-height: 1.25 !important;
            color: var(--text-main) !important;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            word-break: break-word !important;
        }

        [data-testid="stMetricDelta"],
        [data-testid="stMetricDelta"] > div {
            font-size: 0.74rem !important;
            font-weight: 600 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def renderizar_encabezado_y_auth(cliente_supabase: Optional[Client], supabase_configurado: bool) -> None:
    """
    Renderiza:
    1. Navbar superior estilo SaaS (Calipso Captia) con marca, créditos y botón Iniciar/Cerrar sesión.
    2. Modal/panel desplegable de autenticación con Supabase Auth.
    3. Hero limpio con texto degradado y tira de 3 pasos al estilo Monica.im Invoice Scanner.
    """
    usuario = st.session_state.get("usuario_supabase")

    col_nav, col_boton_auth = st.columns([4.1, 1.1], gap="small", vertical_alignment="center")

    with col_nav:
        estado_badge = (
            f"🟢 {usuario['email']}"
            if usuario
            else ("☁️ Nube Lista" if supabase_configurado else "⚡ Modo Demo Local")
        )
        st.markdown(
            f"""
            <div class="saas-navbar">
                <div class="saas-brand">
                    <span class="saas-logo-icon">🧾</span>
                    <div>
                        <div class="saas-brand-title">CaptiaOCR &bull; Escáner IA</div>
                        <div class="saas-credits">TFI EDI II (Ciencias de Datos) — Alan Bustamante y Christian Martínez</div>
                    </div>
                </div>
                <span class="saas-status-pill">{estado_badge}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_boton_auth:
        if usuario:
            if st.button("🚪 Cerrar sesión", key="btn_logout_top", use_container_width=True):
                if cliente_supabase:
                    try:
                        cliente_supabase.auth.sign_out()
                    except Exception:
                        pass
                st.session_state["usuario_supabase"] = None
                st.session_state["mostrar_panel_auth"] = False
                st.rerun()
        else:
            etiqueta_btn = "✕ Cerrar acceso" if st.session_state["mostrar_panel_auth"] else "👤 Iniciar sesión"
            if st.button(etiqueta_btn, key="btn_login_top", type="primary", use_container_width=True):
                st.session_state["mostrar_panel_auth"] = not st.session_state["mostrar_panel_auth"]
                st.rerun()

    if st.session_state["mostrar_panel_auth"] and not usuario:
        with st.container(border=True):
            st.subheader("🔐 Acceso a tu Repositorio en la Nube (Supabase Auth)")
            if not supabase_configurado or cliente_supabase is None:
                st.info(
                    "⚡ **Modo Demo Local activo:** Aún no se configuraron `SUPABASE_URL` y `SUPABASE_KEY` en "
                    "`st.secrets` (o variables de entorno). Puedes probar todo el escáner OpenCV, Tesseract OCR y "
                    "extracción con IA normalmente en memoria local."
                )
            else:
                tab_login, tab_registro = st.tabs(["Iniciar Sesión", "Crear Cuenta Nueva"])

                with tab_login:
                    with st.form("form_login_supabase", clear_on_submit=False):
                        email_in = st.text_input("Correo electrónico:", placeholder="usuario@empresa.com")
                        pass_in = st.text_input("Contraseña:", type="password", placeholder="••••••••")
                        submit_login = st.form_submit_button("Ingresar a mi bandeja", use_container_width=True)

                        if submit_login:
                            if not email_in.strip() or not pass_in:
                                st.warning("Completa tu correo electrónico y contraseña.")
                            else:
                                try:
                                    res = cliente_supabase.auth.sign_in_with_password(
                                        {"email": email_in.strip(), "password": pass_in}
                                    )
                                    if res and res.user:
                                        st.session_state["usuario_supabase"] = {
                                            "id": str(res.user.id),
                                            "email": str(res.user.email),
                                        }
                                        st.session_state["mostrar_panel_auth"] = False
                                        st.success(f"¡Bienvenido, {res.user.email}!")
                                        st.rerun()
                                except Exception as exc_auth:
                                    st.error(f"No se pudo iniciar sesión: {exc_auth}")

                with tab_registro:
                    with st.form("form_registro_supabase", clear_on_submit=False):
                        email_reg = st.text_input("Correo electrónico para registro:", placeholder="nombre@correo.com")
                        pass_reg = st.text_input(
                            "Contraseña (mínimo 6 caracteres):", type="password", placeholder="••••••••"
                        )
                        submit_reg = st.form_submit_button("Crear cuenta gratuita", use_container_width=True)

                        if submit_reg:
                            if not email_reg.strip() or len(pass_reg) < 6:
                                st.warning("Ingresa un correo válido y una contraseña de al menos 6 caracteres.")
                            else:
                                try:
                                    res_up = cliente_supabase.auth.sign_up(
                                        {"email": email_reg.strip(), "password": pass_reg}
                                    )
                                    if res_up and res_up.user:
                                        if res_up.session:
                                            st.session_state["usuario_supabase"] = {
                                                "id": str(res_up.user.id),
                                                "email": str(res_up.user.email),
                                            }
                                            st.session_state["mostrar_panel_auth"] = False
                                            st.success("¡Cuenta creada y sesión iniciada!")
                                            st.rerun()
                                        else:
                                            st.success(
                                                "¡Registro exitoso! Revisa tu correo si tienes activada la confirmación "
                                                "por email en Supabase, o inicia sesión en la pestaña 'Iniciar Sesión'."
                                            )
                                except Exception as exc_reg:
                                    st.error(f"Error al registrar usuario: {exc_reg}")

    st.markdown(
        """
        <div class="saas-hero">
            <div class="saas-hero-badge">✨ ESCÁNER INTELIGENTE DE FACTURAS CON IA + OCR</div>
            <h1 class="saas-hero-title">
                Automatizá la captura de tus <span class="saas-gradient-text">Comprobantes y Facturas</span>
            </h1>
            <p class="saas-hero-subtitle">
                Eliminá la carga manual: digitalizá tickets y facturas con filtro escáner OpenCV,
                interpretación inteligente por IA y circuito de validación antes de exportar a CSV, JSON o la nube.
            </p>
            <div class="saas-steps-row">
                <div class="saas-step-card">
                    <span class="saas-step-num">1</span>
                    <div class="saas-step-text">
                        <strong>Subí o fotografiá el ticket</strong>
                        <span>Mejora automática de nitidez y sombras</span>
                    </div>
                </div>
                <div class="saas-step-card">
                    <span class="saas-step-num">2</span>
                    <div class="saas-step-text">
                        <strong>Lectura híbrida OCR + IA</strong>
                        <span>Extrae CUIT, fechas, IVA y totales al instante</span>
                    </div>
                </div>
                <div class="saas-step-card">
                    <span class="saas-step-num">3</span>
                    <div class="saas-step-text">
                        <strong>Validá y exportá</strong>
                        <span>Revisión asistida, CSV/JSON y repositorio cloud</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# 7. APLICACIÓN PRINCIPAL DE STREAMLIT
# ==============================================================================

def main() -> None:
    st.set_page_config(
        page_title="Lector Inteligente de Comprobantes y Facturas",
        page_icon="🧾",
        layout="wide",
    )
    inicializar_estado_sesion()
    aplicar_estilos_saas()

    supabase_url, supabase_key = obtener_credenciales_supabase()
    cliente_supabase = inicializar_cliente_supabase(supabase_url, supabase_key)
    supabase_configurado = cliente_supabase is not None

    renderizar_encabezado_y_auth(cliente_supabase, supabase_configurado)

    # --------------------------------------------------------------------------
    # BARRA LATERAL: MOTOR DE VISIÓN FIJO Y CONFIGURACIÓN SIMPLIFICADA
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Motor de Visión IA")

        st.info(
            f"👁️ **Modelo de Visión Fijo:**\n`{MODELO_VISION_PRINCIPAL}`\n\n"
            "Soporta múltiples fotos superpuestas de una misma factura larga para consolidarlas en un único JSON."
        )

        clave_entorno = (
            os.environ.get("GROQ_API_KEY", "")
            or os.environ.get("GEMINI_API_KEY", "")
            or os.environ.get("GOOGLE_API_KEY", "")
        )
        if not clave_entorno:
            try:
                clave_entorno = (
                    st.secrets.get("GROQ_API_KEY", "")
                    or st.secrets.get("GEMINI_API_KEY", "")
                    or st.secrets.get("GOOGLE_API_KEY", "")
                )
            except Exception:
                clave_entorno = ""

        api_key_nube = st.text_input(
            "API Key (Groq Cloud):",
            value=clave_entorno,
            type="password",
            placeholder="Pega tu clave gsk_...",
        )

        st.subheader("Calidad y Validación")
        aplicar_hibrido_regex = st.checkbox(
            "Refuerzo contable con Regex (CUIT / Fechas / Subtotales)",
            value=True,
            help="Valida matemáticamente Subtotal Neto + IVA = Total y rescata CUIT/fechas exactas.",
        )

        # Diagnóstico de Tesseract OCR de apoyo
        ruta_tesseract = configurar_ruta_tesseract()
        if ruta_tesseract:
            st.success(f"OCR auxiliar activo ({ruta_tesseract})")
        else:
            st.caption("ℹ️ Modo Visión Directa (Tesseract OCR local opcional).")

        # Estado de Supabase / Modo Demo Local
        st.subheader("Repositorio en la Nube")
        usuario_actual = st.session_state.get("usuario_supabase")
        if usuario_actual and supabase_configurado:
            st.success(f"☁️ Conectado: {usuario_actual['email']}")
        elif supabase_configurado:
            st.info("☁️ Supabase listo. Iniciá sesión arriba a la derecha para sincronizar tus facturas.")
        else:
            st.caption("⚡ **Modo Demo Local activo** (Configurá `SUPABASE_URL` y `SUPABASE_KEY` para guardado cloud).")

    # --------------------------------------------------------------------------
    # PESTAÑAS PRINCIPALES ESTILO SEGMENTED CONTROL SAAS
    # --------------------------------------------------------------------------
    tab_lector, tab_historial, tab_analitica, tab_evaluacion = st.tabs(
        [
            "⚡ Escáner e Indexación IA",
            "🗂️ Repositorio de Comprobantes",
            "📈 Análisis de Gastos y Métricas",
            "📊 Auditoría de Exactitud (TFI)",
        ]
    )

    # ==========================================================================
    # PESTAÑA 1: PANTALLA DIVIDIDA (ESTILO MONICA.IM DROPZONE + CALIPSO CAPTIA)
    # ==========================================================================
    with tab_lector:
        col_carga, col_resultados = st.columns([1.02, 1.18], gap="large")

        with col_carga:
            with st.container(border=True):
                st.markdown(
                    """
                    <div class="captia-card-header">
                        <span class="captia-card-title">1. Captura Multi-Imagen del Comprobante</span>
                        <span class="captia-chip captia-chip-indigo">🧩 Visión Multi-Fragmento</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    usar_recorte_camscanner = st.toggle(
                        "Recorte y perspectiva (CamScanner)",
                        value=True,
                        help=(
                            "Ejecuta escanear_documento(imagen_bytes) con OpenCV: Canny Edge, "
                            "contorno de 4 vértices y warpPerspective (bird's-eye view)."
                        ),
                    )
                with col_t2:
                    usar_filtro_escaner = st.toggle(
                        "Filtro Escáner Binarizado (OCR)",
                        value=True,
                        help=(
                            "Ejecuta optimizar_para_tesseract(pil_image) con OpenCV: escala de grises, "
                            "upscaling cúbico, filtro bilateral y binarización adaptativa."
                        ),
                    )

                metodo_entrada = st.radio(
                    "Método de recepción:",
                    options=["Subir archivos (Múltiples fotos o PDF)", "Capturar con cámara (Celular / PC)"],
                    horizontal=True,
                )

                archivos_subidos: List[Any] = []
                if metodo_entrada.startswith("Subir"):
                    archivos_cargados = st.file_uploader(
                        "Arrastrá una o varias fotos superpuestas de la misma factura (JPG, PNG, WEBP o PDF):",
                        type=["pdf", "png", "jpg", "jpeg", "webp"],
                        accept_multiple_files=True,
                    )
                    if archivos_cargados:
                        archivos_subidos = list(archivos_cargados)
                else:
                    foto_camara = st.camera_input("Enfocá el comprobante con buena iluminación:")
                    if foto_camara is not None:
                        archivos_subidos = [foto_camara]

                if archivos_subidos:
                    # Preparar todos los fragmentos subidos (fotos superpuestas o páginas de PDF)
                    fragmentos_procesados: List[Dict[str, Any]] = []
                    textos_digitales_pdf: List[str] = []

                    for idx_arch, arch_item in enumerate(archivos_subidos):
                        try:
                            bytes_item = arch_item.getvalue()
                            nombre_item = getattr(arch_item, "name", f"fragmento_{idx_arch + 1}.png") or f"fragmento_{idx_arch + 1}.png"
                            mime_item = getattr(arch_item, "type", "") or ""
                            es_pdf_item = (
                                nombre_item.lower().endswith(".pdf")
                                or "pdf" in mime_item.lower()
                                or bytes_item[:4] == b"%PDF"
                            )

                            if not bytes_item or len(bytes_item) < 50:
                                st.warning(f"⚠️ El archivo '{nombre_item}' está vacío y fue omitido.")
                                continue

                            # Guardar el primer archivo como referencia para Supabase Storage
                            if idx_arch == 0:
                                st.session_state["ultimo_archivo_bytes"] = bytes_item
                                st.session_state["ultimo_archivo_nombre"] = nombre_item
                                st.session_state["ultimo_archivo_mime"] = (
                                    "application/pdf" if es_pdf_item else (mime_item or "image/png")
                                )

                            if es_pdf_item:
                                img_p0, txt_dig_0, tot_pags, err_pdf = procesar_archivo_pdf(bytes_item, indice_pagina=0)
                                if err_pdf or img_p0 is None:
                                    st.error(err_pdf or f"❌ No se pudo procesar el PDF '{nombre_item}'.")
                                    continue
                                if txt_dig_0.strip():
                                    textos_digitales_pdf.append(txt_dig_0.strip())

                                # Soportar todas las páginas del PDF (hasta 5 fragmentos)
                                paginas_a_procesar = [(0, img_p0, txt_dig_0)]
                                for p_i in range(1, min(tot_pags, 5)):
                                    img_pi, txt_pi, _, err_pi = procesar_archivo_pdf(bytes_item, indice_pagina=p_i)
                                    if not err_pi and img_pi is not None:
                                        paginas_a_procesar.append((p_i, img_pi, txt_pi))

                                for num_p, img_pag, txt_pag in paginas_a_procesar:
                                    img_orig_rgb = img_pag.convert("RGB")
                                    buf_png = io.BytesIO()
                                    img_orig_rgb.save(buf_png, format="PNG")
                                    if usar_recorte_camscanner and len(txt_pag.strip()) < 20:
                                        img_end, rec_ok = escanear_documento_con_estado(buf_png.getvalue())
                                    else:
                                        img_end, rec_ok = img_orig_rgb, False
                                    img_opt = optimizar_para_tesseract(img_end) if usar_filtro_escaner else img_end
                                    fragmentos_procesados.append(
                                        {
                                            "etiqueta": f"{nombre_item} (Pág. {num_p + 1}/{tot_pags})",
                                            "original": img_orig_rgb,
                                            "enderezada": img_end,
                                            "optimizada": img_opt,
                                            "recorte_ok": rec_ok,
                                            "es_pdf": True,
                                        }
                                    )
                            else:
                                img_orig_rgb = ImageOps.exif_transpose(
                                    Image.open(io.BytesIO(bytes_item))
                                ).convert("RGB")

                                if calcular_nitidez_imagen(img_orig_rgb) < 45.0:
                                    st.warning(
                                        f"⚠️ **Aviso de calidad ({nombre_item}):** La foto parece algo borrosa. "
                                        "Si algún dato falta, probá tomarla con mejor iluminación."
                                    )

                                if usar_recorte_camscanner:
                                    img_end, rec_ok = escanear_documento_con_estado(bytes_item)
                                else:
                                    img_end, rec_ok = img_orig_rgb, False

                                img_opt = optimizar_para_tesseract(img_end) if usar_filtro_escaner else img_end
                                fragmentos_procesados.append(
                                    {
                                        "etiqueta": f"Fragmento {idx_arch + 1}: {nombre_item}",
                                        "original": img_orig_rgb,
                                        "enderezada": img_end,
                                        "optimizada": img_opt,
                                        "recorte_ok": rec_ok,
                                        "es_pdf": False,
                                    }
                                )
                        except Exception:
                            st.error(
                                "❌ Uno de los archivos subidos no tiene un formato válido (JPG, PNG, WEBP o PDF)."
                            )

                    if fragmentos_procesados:
                        cant_frag = len(fragmentos_procesados)
                        if cant_frag > 1:
                            st.info(
                                f"🧩 **Modo Rompecabezas Activo ({cant_frag} fragmentos cargados):** "
                                "La IA de Visión analizará todas las imágenes en conjunto y unificará los datos sin duplicar información."
                            )

                        # Mostrar vista previa Antes vs. Después (del primer fragmento directamente y el resto en desplegable si son varios)
                        frag_principal = fragmentos_procesados[0]
                        col_antes, col_despues = st.columns(2)
                        with col_antes:
                            st.image(
                                frag_principal["original"],
                                caption=f"Antes — {frag_principal['etiqueta']}",
                                use_container_width=True,
                            )
                        with col_despues:
                            st.image(
                                frag_principal["enderezada"],
                                caption=(
                                    f"Después (OpenCV) — {frag_principal['etiqueta']}"
                                    if frag_principal["recorte_ok"]
                                    else f"Vista Procesada — {frag_principal['etiqueta']}"
                                ),
                                use_container_width=True,
                            )

                        if cant_frag > 1:
                            with st.expander(f"🖼️ Ver los {cant_frag - 1} fragmento(s) fotográfico(s) adicional(es)", expanded=False):
                                for frag_extra in fragmentos_procesados[1:]:
                                    c_a, c_d = st.columns(2)
                                    with c_a:
                                        st.image(frag_extra["original"], caption=f"Original — {frag_extra['etiqueta']}", use_container_width=True)
                                    with c_d:
                                        st.image(frag_extra["enderezada"], caption=f"Procesada — {frag_extra['etiqueta']}", use_container_width=True)

                        if st.button("✨ Escanear y Extraer Datos con IA", type="primary", use_container_width=True):
                            try:
                                with st.spinner("Procesando factura..."):
                                    # 1. Convertir cada imagen procesada a Base64 para el payload Multi-Imagen
                                    lista_imagenes_base64 = [
                                        codificar_imagen_a_base64(frag["enderezada"])
                                        for frag in fragmentos_procesados
                                    ]

                                    # 2. Ejecutar OCR auxiliar sobre cada fragmento para respaldo y validación contable
                                    bloques_ocr: List[str] = list(textos_digitales_pdf)
                                    for idx_f, frag in enumerate(fragmentos_procesados):
                                        txt_bin, _ = ejecutar_ocr(frag["optimizada"], idioma="spa")
                                        if txt_bin:
                                            bloques_ocr.append(f"[FRAGMENTO {idx_f + 1}]\n{txt_bin}")
                                        if usar_filtro_escaner:
                                            txt_nat, _ = ejecutar_ocr(frag["enderezada"], idioma="spa")
                                            if txt_nat and txt_nat.strip() != txt_bin.strip():
                                                bloques_ocr.append(f"[FRAGMENTO {idx_f + 1} - ESCALA NATURAL]\n{txt_nat}")

                                    texto_detectado = "\n\n".join(bloques_ocr).strip()
                                    st.session_state["texto_ocr"] = texto_detectado

                                    # 3. Enviar Payload Multi-Imagen al modelo de visión fijo (MODELO_VISION_PRINCIPAL)
                                    datos_ext, error_ia = extraer_con_vision_multimodal(
                                        lista_imagenes_base64=lista_imagenes_base64,
                                        api_key=api_key_nube,
                                        texto_ocr_apoyo=texto_detectado,
                                    )

                                    if error_ia:
                                        if texto_detectado and len(texto_detectado) >= 10:
                                            st.warning(
                                                f"ℹ️ {error_ia}\nSe aplicó automáticamente la extracción contable por reglas sobre los {cant_frag} fragmento(s)."
                                            )
                                            datos_ext = extraer_con_reglas_regex(texto_detectado)
                                            st.session_state["items_actuales"] = extraer_items_con_regex(texto_detectado)
                                        else:
                                            st.warning(f"⚠️ {error_ia}")
                                    elif aplicar_hibrido_regex and texto_detectado:
                                        datos_ext = aplicar_refuerzo_regex(datos_ext, texto_detectado)
                                        if not st.session_state.get("items_actuales"):
                                            st.session_state["items_actuales"] = extraer_items_con_regex(texto_detectado)

                                    st.session_state["datos_actuales"] = datos_ext
                                    st.success(
                                        f"✅ Factura procesada consolidando {cant_frag} fragmento(s). "
                                        "Revisá los montos y datos en el panel derecho."
                                    )
                            except Exception:
                                st.error(
                                    "❌ Ocurrió un inconveniente inesperado al analizar los fragmentos de la factura. "
                                    "Por favor, intentá nuevamente con imágenes más claras."
                                )

        with col_resultados:
            with st.container(border=True):
                datos_act = st.session_state["datos_actuales"]
                campos_detectados = sum(
                    1 for v in datos_act.values() if v and v != VALOR_NO_IDENTIFICADO
                )
                campos_criticos = ["emisor", "fecha_emision", "importe_total"]
                faltantes = [
                    CAMPOS_COMPROBANTE[c]
                    for c in campos_criticos
                    if datos_act.get(c, VALOR_NO_IDENTIFICADO) == VALOR_NO_IDENTIFICADO
                ]

                if not st.session_state["texto_ocr"] and campos_detectados == 0:
                    chip_html = '<span class="captia-chip captia-chip-indigo">⏳ Esperando documento</span>'
                elif not faltantes:
                    chip_html = f'<span class="captia-chip captia-chip-success">✅ Listo para aprobar ({campos_detectados}/9)</span>'
                else:
                    chip_html = f'<span class="captia-chip captia-chip-warning">⚠️ Revisión sugerida ({campos_detectados}/9)</span>'

                st.markdown(
                    f"""
                    <div class="captia-card-header">
                        <span class="captia-card-title">2. Bandeja de Validación Contable</span>
                        {chip_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Segmentación visual: métricas numéricas principales con st.metric
                moneda_act = datos_act.get("moneda", "$")
                col_met1, col_met2, col_met3 = st.columns(3)
                with col_met1:
                    st.metric(
                        label="💰 Total de la Factura",
                        value=formatear_importe_metrica(datos_act.get("importe_total"), moneda_act),
                        delta=(
                            datos_act.get("tipo_documento")
                            if datos_act.get("tipo_documento", VALOR_NO_IDENTIFICADO) != VALOR_NO_IDENTIFICADO
                            else None
                        ),
                    )
                with col_met2:
                    st.metric(
                        label="💵 Importe Neto",
                        value=formatear_importe_metrica(datos_act.get("importe_neto"), moneda_act),
                    )
                with col_met3:
                    st.metric(
                        label="🧾 Impuestos / IVA",
                        value=formatear_importe_metrica(datos_act.get("impuestos_iva"), moneda_act),
                    )

                st.markdown(
                    f"""
                    <div class="captia-kpi-grid">
                        <div class="captia-kpi-box">
                            <div class="captia-kpi-label">Emisor / Proveedor</div>
                            <div class="captia-kpi-val">{datos_act.get("emisor", VALOR_NO_IDENTIFICADO)}</div>
                        </div>
                        <div class="captia-kpi-box">
                            <div class="captia-kpi-label">CUIT Emisor</div>
                            <div class="captia-kpi-val">{datos_act.get("cuit", VALOR_NO_IDENTIFICADO)}</div>
                        </div>
                        <div class="captia-kpi-box">
                            <div class="captia-kpi-label">Fecha Emisión</div>
                            <div class="captia-kpi-val">{datos_act.get("fecha_emision", VALOR_NO_IDENTIFICADO)}</div>
                        </div>
                        <div class="captia-kpi-box">
                            <div class="captia-kpi-label">N° Comprobante</div>
                            <div class="captia-kpi-val">{datos_act.get("numero_comprobante", VALOR_NO_IDENTIFICADO)}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if st.session_state["texto_ocr"] and faltantes:
                    st.warning(
                        "⚠️ **Validación asistida:** Completá o verificá manualmente: "
                        f"**{', '.join(faltantes)}**."
                    )

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
                        "Clave": None,
                        "Campo del Comprobante": st.column_config.TextColumn(
                            "Campo Contable", disabled=True, width="medium"
                        ),
                        "Valor Detectado (Editable)": st.column_config.TextColumn(
                            "Valor Extraído (Doble clic para editar)", width="large"
                        ),
                    },
                    hide_index=True,
                    use_container_width=True,
                    key="editor_campos",
                )

                datos_revisados = {
                    fila["Clave"]: str(fila["Valor Detectado (Editable)"]).strip() or VALOR_NO_IDENTIFICADO
                    for _, fila in df_editado.iterrows()
                }
                st.session_state["datos_actuales"] = datos_revisados

                # ------------------------------------------------------------------
                # TABLA DE ÍTEMS FACTURADOS Y CÁLCULO DE COSTO UNITARIO CON PANDAS
                # ------------------------------------------------------------------
                st.markdown("---")
                col_tit_items, col_btn_demo_items = st.columns([3.2, 1.4], vertical_alignment="center")
                with col_tit_items:
                    st.markdown("##### 📦 Detalle de Ítems y Costo Unitario Real (Cálculo Python / Pandas)")
                    st.caption(
                        "Doble clic en cualquier celda para corregir valores (ej. cambiar unidades por empaque de `8` a `6`). "
                        "Python recalcula automáticamente el **Precio Unitario Real** y el **Total de Unidades**."
                    )
                with col_btn_demo_items:
                    if st.button(
                        "🧪 Cargar ejemplo (6x3L)",
                        key="btn_ejemplo_items",
                        use_container_width=True,
                        help="Carga el caso de ejemplo 'Manaos Cola 6x3L' para probar el recálculo automático en Pandas.",
                    ):
                        st.session_state["items_actuales"] = [
                            {
                                "descripcion": "Manaos Cola 6x3L",
                                "cantidad_facturada": 1.0,
                                "precio_linea_impreso": 8200.0,
                                "unidades_por_empaque": 6,
                            }
                        ]
                        st.rerun()

                df_items_actual = calcular_dataframe_items(st.session_state.get("items_actuales", []))

                df_items_editado = st.data_editor(
                    df_items_actual,
                    column_config={
                        "descripcion": st.column_config.TextColumn(
                            "Descripción del Producto",
                            help="Texto que describe el artículo o servicio (ej. 'Manaos Cola 6x3L').",
                            width="medium",
                            required=True,
                        ),
                        "cantidad_facturada": st.column_config.NumberColumn(
                            "Cant. Bultos",
                            help="Cantidad de bultos, fardos o packs que figura en la factura.",
                            min_value=0.0,
                            step=1.0,
                            format="%.2f",
                            width="small",
                            required=True,
                        ),
                        "precio_linea_impreso": st.column_config.NumberColumn(
                            "Precio Bulto (Factura)",
                            help="Precio unitario del fardo/bulto impreso en la línea de la factura.",
                            min_value=0.0,
                            step=10.0,
                            format="$ %.2f",
                            width="small",
                            required=True,
                        ),
                        "unidades_por_empaque": st.column_config.NumberColumn(
                            "Unid. x Empaque",
                            help="Unidades individuales dentro del empaque (ej. 6 en '6x3L', 12 en 'Caja x 12'). ¡Editalo para recalcular!",
                            min_value=1,
                            step=1,
                            format="%d u.",
                            width="small",
                            required=True,
                        ),
                        "precio_unitario_real": st.column_config.NumberColumn(
                            "💡 Costo Unitario Real",
                            help="Calculado por Python: precio_linea_impreso / unidades_por_empaque",
                            format="$ %.2f",
                            disabled=True,
                            width="small",
                        ),
                        "cantidad_total_unidades": st.column_config.NumberColumn(
                            "📊 Total Unidades",
                            help="Calculado por Python: cantidad_facturada * unidades_por_empaque",
                            format="%.0f u.",
                            disabled=True,
                            width="small",
                        ),
                    },
                    num_rows="dynamic",
                    hide_index=True,
                    use_container_width=True,
                    key="editor_items_factura",
                )

                # Sincronizar ediciones del usuario y recalcular al instante con Pandas
                items_editados_limpios: List[Dict[str, Any]] = []
                for _, fila_it in df_items_editado.iterrows():
                    desc_it = str(fila_it.get("descripcion", "") or "").strip()
                    if not desc_it or desc_it.lower() == "nan":
                        continue
                    try:
                        cant_it = float(fila_it.get("cantidad_facturada", 1.0) or 1.0)
                    except Exception:
                        cant_it = 1.0
                    try:
                        precio_it = float(fila_it.get("precio_linea_impreso", 0.0) or 0.0)
                    except Exception:
                        precio_it = 0.0
                    try:
                        unid_it = max(1, int(round(float(fila_it.get("unidades_por_empaque", 1) or 1))))
                    except Exception:
                        unid_it = 1

                    items_editados_limpios.append(
                        {
                            "descripcion": desc_it,
                            "cantidad_facturada": round(cant_it, 2),
                            "precio_linea_impreso": round(precio_it, 2),
                            "unidades_por_empaque": unid_it,
                        }
                    )

                if items_editados_limpios != st.session_state.get("items_actuales", []):
                    st.session_state["items_actuales"] = items_editados_limpios
                    st.rerun()

                df_items_final = calcular_dataframe_items(st.session_state.get("items_actuales", []))
                if not df_items_final.empty:
                    with st.expander("💡 Lectura para el Negocio / Inventario (Fardo vs. Unidad Individual)", expanded=True):
                        for _, r_item in df_items_final.iterrows():
                            st.markdown(
                                f"- **{r_item['descripcion']}**: Compraste **{r_item['cantidad_facturada']:g}** "
                                f"bulto(s)/fardo(s) a **${r_item['precio_linea_impreso']:,.2f}** *(lo que dice la factura)* "
                                f"➔ **Cada unidad individual te costó ${r_item['precio_unitario_real']:,.2f}** "
                                f"*(ingresan **{r_item['cantidad_total_unidades']:g}** unidades al inventario)*."
                            )
                        csv_items_bytes = df_items_final.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            label="⬇️ Exportar Tabla de Ítems y Costos Unitarios (CSV)",
                            data=csv_items_bytes,
                            file_name="items_costo_unitario.csv",
                            mime="text/csv",
                            key="btn_descargar_items_csv",
                            use_container_width=True,
                        )

                # Segmentación con st.expander para ocultar datos secundarios, JSON crudo y texto OCR
                with st.expander("🗂️ Ver JSON crudo estructurado y datos técnicos del OCR", expanded=False):
                    sub_tab_json, sub_tab_ocr = st.tabs(["📄 JSON Estructurado (IA + Ítems)", "📝 Texto Crudo (Tesseract OCR)"])
                    with sub_tab_json:
                        st.caption("Estructura JSON normalizada lista para almacenar o exportar (incluye 'items'):")
                        st.json(
                            {
                                **datos_revisados,
                                "items": st.session_state.get("items_actuales", []),
                            }
                        )
                    with sub_tab_ocr:
                        texto_editable_ocr = st.text_area(
                            "Salida cruda de Tesseract (--oem 3 --psm 6):",
                            value=st.session_state["texto_ocr"],
                            height=140,
                        )
                        st.session_state["texto_ocr"] = texto_editable_ocr

                col_btn1, col_btn2, col_btn3 = st.columns(3)

                with col_btn1:
                    if st.button("✅ Aprobar y Guardar", type="primary", use_container_width=True):
                        usuario_sesion = st.session_state.get("usuario_supabase")
                        bytes_actuales = st.session_state.get("ultimo_archivo_bytes")

                        if usuario_sesion and cliente_supabase is not None:
                            if not bytes_actuales:
                                st.warning("Primero subí o capturá un comprobante antes de guardar en la nube.")
                            else:
                                with st.spinner("Sincronizando imagen y factura en Supabase..."):
                                    exito, msg_supa, url_img = guardar_comprobante_en_supabase(
                                        cliente=cliente_supabase,
                                        user_id=usuario_sesion["id"],
                                        user_email=usuario_sesion["email"],
                                        bytes_imagen=bytes_actuales,
                                        nombre_archivo=st.session_state.get("ultimo_archivo_nombre", "comprobante.png"),
                                        mime_type=st.session_state.get("ultimo_archivo_mime", "image/png"),
                                        datos_revisados=datos_revisados,
                                        texto_ocr=st.session_state.get("texto_ocr", ""),
                                    )
                                if exito:
                                    st.success(msg_supa)
                                else:
                                    st.error(msg_supa)
                        else:
                            registro_local = {
                                "fecha_carga": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "modo": "Demo Local",
                            }
                            registro_local.update(datos_revisados)
                            st.session_state["historial_comprobantes"].append(registro_local)
                            st.success(
                                "Comprobante aprobado y guardado en el Repositorio de la sesión. "
                                "Iniciá sesión arriba para resguardarlo en la nube."
                            )

                with col_btn2:
                    payload_export_json = {
                        **datos_revisados,
                        "items": df_items_final.to_dict(orient="records") if not df_items_final.empty else [],
                    }
                    json_individual = json.dumps(payload_export_json, ensure_ascii=False, indent=2)
                    st.download_button(
                        label="⬇️ Exportar JSON",
                        data=json_individual.encode("utf-8"),
                        file_name="comprobante_revisado.json",
                        mime="application/json",
                        use_container_width=True,
                    )

                with col_btn3:
                    df_export_ind = pd.DataFrame([datos_revisados])
                    csv_individual = df_export_ind.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label="⬇️ Exportar CSV",
                        data=csv_individual,
                        file_name="comprobante_revisado.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

    # ==========================================================================
    # PESTAÑA 2: REPOSITORIO DOCUMENTAL CON FILTROS SELECCIONABLES Y BORRADO
    # ==========================================================================
    with tab_historial:
        with st.container(border=True):
            usuario_sesion = st.session_state.get("usuario_supabase")

            if usuario_sesion and cliente_supabase is not None:
                st.markdown(
                    f"""
                    <div class="captia-card-header">
                        <span class="captia-card-title">☁️ Repositorio Documental en la Nube ({usuario_sesion['email']})</span>
                        <span class="captia-chip captia-chip-success">🔒 Sincronizado con Supabase</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                registros_nube, error_consulta = consultar_facturas_usuario_supabase(
                    cliente_supabase, usuario_sesion["id"]
                )
                if error_consulta:
                    st.error(error_consulta)
                elif not registros_nube:
                    st.info(
                        "Aún no tenés comprobantes aprobados en tu repositorio cloud. "
                        "Escaneá una factura en la pestaña 1 y presioná **'✅ Aprobar y Guardar'**."
                    )
                else:
                    df_nube = pd.DataFrame(registros_nube)
                    fechas_fmt = df_nube.get("created_at", pd.Series([""] * len(df_nube))).apply(
                        formatear_fecha_guardado
                    )
                    df_nube["fecha_guardado"] = [par[0] for par in fechas_fmt]
                    df_nube["_dia_guardado"] = [par[1] for par in fechas_fmt]

                    # Filtros desplegables de selección para todas las categorías (sin escribir)
                    df_filtrado_nube = renderizar_filtros_seleccionables(df_nube, prefijo="nube")

                    if df_filtrado_nube.empty:
                        st.warning("No hay comprobantes que coincidan con la combinación de filtros seleccionada.")
                    else:
                        columnas_orden = [
                            "id",
                            "storage_path",
                            "fecha_guardado",
                            "tipo_documento",
                            "emisor",
                            "cuit",
                            "fecha_emision",
                            "numero_comprobante",
                            "importe_neto",
                            "impuestos_iva",
                            "importe_total",
                            "moneda",
                            "imagen_url",
                        ]
                        cols_presentes = [c for c in columnas_orden if c in df_filtrado_nube.columns]
                        df_tabla_nube = df_filtrado_nube[cols_presentes].copy()
                        df_tabla_nube.insert(0, "_eliminar", False)

                        cols_bloqueadas = [c for c in df_tabla_nube.columns if c != "_eliminar"]

                        st.caption("💡 Marcá la casilla **🗑️ Borrar** en cualquier fila si guardaste un comprobante por error y presioná **Eliminar seleccionados**.")
                        df_editor_nube = st.data_editor(
                            df_tabla_nube,
                            column_config={
                                "_eliminar": st.column_config.CheckboxColumn(
                                    "🗑️ Borrar",
                                    help="Seleccioná este comprobante para eliminarlo del historial",
                                    default=False,
                                    width="small",
                                ),
                                "id": None,
                                "storage_path": None,
                                "fecha_guardado": st.column_config.TextColumn("Fecha Guardado"),
                                "tipo_documento": st.column_config.TextColumn("Tipo"),
                                "emisor": st.column_config.TextColumn("Emisor / Comercio"),
                                "cuit": st.column_config.TextColumn("CUIT"),
                                "fecha_emision": st.column_config.TextColumn("Fecha Comprobante"),
                                "numero_comprobante": st.column_config.TextColumn("N° Comprobante"),
                                "importe_neto": st.column_config.TextColumn("Importe Neto"),
                                "impuestos_iva": st.column_config.TextColumn("IVA / Impuestos"),
                                "importe_total": st.column_config.TextColumn("Importe Total"),
                                "moneda": st.column_config.TextColumn("Moneda"),
                                "imagen_url": st.column_config.LinkColumn(
                                    "Foto", display_text="Ver comprobante"
                                ),
                            },
                            disabled=cols_bloqueadas,
                            hide_index=True,
                            use_container_width=True,
                            key="tabla_historial_nube",
                        )

                        filas_marcadas_nube = df_editor_nube[df_editor_nube["_eliminar"] == True]
                        cant_marcadas_nube = len(filas_marcadas_nube)

                        col_del, col_exp1, col_exp2 = st.columns([1.2, 1, 1])
                        with col_del:
                            etiqueta_borrar = (
                                f"🗑️ Eliminar seleccionados ({cant_marcadas_nube})"
                                if cant_marcadas_nube > 0
                                else "🗑️ Eliminar seleccionados"
                            )
                            if st.button(
                                etiqueta_borrar,
                                key="btn_borrar_nube",
                                disabled=(cant_marcadas_nube == 0),
                                use_container_width=True,
                            ):
                                registros_borrar = filas_marcadas_nube.to_dict(orient="records")
                                with st.spinner("Eliminando comprobante(s) de Supabase..."):
                                    ok_del, msg_del = eliminar_facturas_supabase(
                                        cliente_supabase,
                                        usuario_sesion["id"],
                                        registros_borrar,
                                    )
                                if ok_del:
                                    st.success(msg_del)
                                    st.rerun()
                                else:
                                    st.error(msg_del)

                        cols_exportar = [
                            c for c in cols_presentes if c not in ("id", "storage_path")
                        ]
                        df_export_nube = df_filtrado_nube[cols_exportar]

                        with col_exp1:
                            csv_nube = df_export_nube.to_csv(index=False).encode("utf-8-sig")
                            st.download_button(
                                "📥 Descargar Vista en CSV",
                                data=csv_nube,
                                file_name="mis_facturas_supabase.csv",
                                mime="text/csv",
                                use_container_width=True,
                            )
                        with col_exp2:
                            json_nube = json.dumps(
                                df_export_nube.to_dict(orient="records"),
                                ensure_ascii=False,
                                indent=2,
                                default=str,
                            ).encode("utf-8")
                            st.download_button(
                                "📥 Descargar Vista en JSON",
                                data=json_nube,
                                file_name="mis_facturas_supabase.json",
                                mime="application/json",
                                use_container_width=True,
                            )
            else:
                st.markdown(
                    """
                    <div class="captia-card-header">
                        <span class="captia-card-title">🗂️ Repositorio de Comprobantes de la Sesión</span>
                        <span class="captia-chip captia-chip-indigo">⚡ Modo Demo Local</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if supabase_configurado:
                    st.caption(
                        "💡 Estás visualizando la bandeja local temporal. "
                        "Hacé clic en **'👤 Iniciar sesión'** arriba a la derecha para acceder a tu repositorio permanente en la nube."
                    )
                historial = st.session_state["historial_comprobantes"]

                if not historial:
                    st.info(
                        "Todavía no aprobaste comprobantes en esta sesión. "
                        "Procesá una factura en la pestaña 1 y presioná **'✅ Aprobar y Guardar'**."
                    )
                else:
                    df_hist = pd.DataFrame(historial)
                    df_hist["_idx_origen"] = df_hist.index
                    fechas_loc = df_hist.get("fecha_carga", pd.Series([""] * len(df_hist))).apply(
                        formatear_fecha_guardado
                    )
                    df_hist["fecha_guardado"] = [par[0] for par in fechas_loc]
                    df_hist["_dia_guardado"] = [par[1] for par in fechas_loc]

                    # Filtros desplegables de selección para todas las categorías (sin escribir)
                    df_filtrado_loc = renderizar_filtros_seleccionables(df_hist, prefijo="local")

                    if df_filtrado_loc.empty:
                        st.warning("No hay comprobantes locales que coincidan con los filtros seleccionados.")
                    else:
                        cols_orden_loc = [
                            "_idx_origen",
                            "fecha_guardado",
                            "tipo_documento",
                            "emisor",
                            "cuit",
                            "fecha_emision",
                            "numero_comprobante",
                            "importe_neto",
                            "impuestos_iva",
                            "importe_total",
                            "moneda",
                        ]
                        cols_pres_loc = [c for c in cols_orden_loc if c in df_filtrado_loc.columns]
                        df_tabla_loc = df_filtrado_loc[cols_pres_loc].copy()
                        df_tabla_loc.insert(0, "_eliminar", False)

                        cols_bloq_loc = [c for c in df_tabla_loc.columns if c != "_eliminar"]

                        st.caption("💡 Marcá la casilla **🗑️ Borrar** en cualquier fila si guardaste un comprobante por error y presioná **Eliminar seleccionados**.")
                        df_editor_loc = st.data_editor(
                            df_tabla_loc,
                            column_config={
                                "_eliminar": st.column_config.CheckboxColumn(
                                    "🗑️ Borrar",
                                    help="Seleccioná este comprobante para eliminarlo del historial",
                                    default=False,
                                    width="small",
                                ),
                                "_idx_origen": None,
                                "fecha_guardado": st.column_config.TextColumn("Fecha Guardado"),
                                "tipo_documento": st.column_config.TextColumn("Tipo"),
                                "emisor": st.column_config.TextColumn("Emisor / Comercio"),
                                "cuit": st.column_config.TextColumn("CUIT"),
                                "fecha_emision": st.column_config.TextColumn("Fecha Comprobante"),
                                "numero_comprobante": st.column_config.TextColumn("N° Comprobante"),
                                "importe_neto": st.column_config.TextColumn("Importe Neto"),
                                "impuestos_iva": st.column_config.TextColumn("IVA / Impuestos"),
                                "importe_total": st.column_config.TextColumn("Importe Total"),
                                "moneda": st.column_config.TextColumn("Moneda"),
                            },
                            disabled=cols_bloq_loc,
                            hide_index=True,
                            use_container_width=True,
                            key="tabla_historial_local",
                        )

                        filas_marcadas_loc = df_editor_loc[df_editor_loc["_eliminar"] == True]
                        cant_marcadas_loc = len(filas_marcadas_loc)

                        col_del_l, col_exp1, col_exp2, col_limpiar = st.columns([1.2, 1, 1, 1])
                        with col_del_l:
                            etiqueta_borrar_loc = (
                                f"🗑️ Eliminar seleccionados ({cant_marcadas_loc})"
                                if cant_marcadas_loc > 0
                                else "🗑️ Eliminar seleccionados"
                            )
                            if st.button(
                                etiqueta_borrar_loc,
                                key="btn_borrar_local",
                                disabled=(cant_marcadas_loc == 0),
                                use_container_width=True,
                            ):
                                indices_a_quitar = set(filas_marcadas_loc["_idx_origen"].tolist())
                                st.session_state["historial_comprobantes"] = [
                                    item
                                    for idx, item in enumerate(st.session_state["historial_comprobantes"])
                                    if idx not in indices_a_quitar
                                ]
                                st.rerun()

                        cols_exp_loc = [c for c in cols_pres_loc if c != "_idx_origen"]
                        df_export_loc = df_filtrado_loc[cols_exp_loc]

                        with col_exp1:
                            csv_lote = df_export_loc.to_csv(index=False).encode("utf-8-sig")
                            st.download_button(
                                "📥 Exportar Vista a CSV",
                                data=csv_lote,
                                file_name="registro_comprobantes.csv",
                                mime="text/csv",
                                use_container_width=True,
                            )
                        with col_exp2:
                            json_lote = json.dumps(
                                df_export_loc.to_dict(orient="records"),
                                ensure_ascii=False,
                                indent=2,
                            ).encode("utf-8")
                            st.download_button(
                                "📥 Exportar Vista a JSON",
                                data=json_lote,
                                file_name="registro_comprobantes.json",
                                mime="application/json",
                                use_container_width=True,
                            )
                        with col_limpiar:
                            if st.button("🗑️ Vaciar Todo", use_container_width=True):
                                st.session_state["historial_comprobantes"] = []
                                st.rerun()

    # ==========================================================================
    # PESTAÑA 3: ANÁLISIS DE DATOS INTEGRADO (MÉTRICAS Y GRÁFICOS PANDAS)
    # ==========================================================================
    with tab_analitica:
        with st.container(border=True):
            usuario_sesion = st.session_state.get("usuario_supabase")
            fuente_etiqueta = "Modo Demo Local"
            registros_analitica: List[Dict[str, Any]] = []

            if usuario_sesion and cliente_supabase is not None:
                fuente_etiqueta = f"Supabase Cloud ({usuario_sesion['email']})"
                registros_nube_an, err_an = consultar_facturas_usuario_supabase(
                    cliente_supabase, usuario_sesion["id"]
                )
                if not err_an and registros_nube_an:
                    registros_analitica = registros_nube_an
            else:
                registros_analitica = list(st.session_state.get("historial_comprobantes", []))

            st.markdown(
                f"""
                <div class="captia-card-header">
                    <span class="captia-card-title">📈 Análisis de Gastos y Métricas Financieras</span>
                    <span class="captia-chip captia-chip-indigo">📊 Fuente: {fuente_etiqueta}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            usar_datos_demo = False
            if not registros_analitica:
                st.info(
                    "ℹ️ Aún no tenés comprobantes guardados en el historial para graficar. "
                    "Podés activar la vista previa de demostración abajo o aprobar facturas en la Pestaña 1."
                )
                usar_datos_demo = st.toggle(
                    "🧪 Previsualizar tablero con lote de facturas de ejemplo (Modo Demo)",
                    value=True,
                )

            if usar_datos_demo and not registros_analitica:
                registros_analitica = [
                    {
                        "fecha_emision": "05/07/2026",
                        "emisor": "Supermercado Mayorista Norte",
                        "tipo_documento": "FACTURA A",
                        "importe_neto": "42500.00",
                        "impuestos_iva": "8925.00",
                        "importe_total": "51425.00",
                    },
                    {
                        "fecha_emision": "18/07/2026",
                        "emisor": "Energía Eléctrica EJESA",
                        "tipo_documento": "FACTURA B",
                        "importe_neto": "19800.00",
                        "impuestos_iva": "4158.00",
                        "importe_total": "23958.00",
                    },
                    {
                        "fecha_emision": "04/08/2026",
                        "emisor": "Farmacia Norte Jujuy",
                        "tipo_documento": "TICKET",
                        "importe_neto": "18948.87",
                        "impuestos_iva": "0.00",
                        "importe_total": "18948.87",
                    },
                    {
                        "fecha_emision": "21/08/2026",
                        "emisor": "Supermercado Mayorista Norte",
                        "tipo_documento": "FACTURA A",
                        "importe_neto": "54000.00",
                        "impuestos_iva": "11340.00",
                        "importe_total": "65340.00",
                    },
                    {
                        "fecha_emision": "10/09/2026",
                        "emisor": "Librería e Insumos Jujuy",
                        "tipo_documento": "FACTURA C",
                        "importe_neto": "31200.00",
                        "impuestos_iva": "0.00",
                        "importe_total": "31200.00",
                    },
                    {
                        "fecha_emision": "19/10/2026",
                        "emisor": "Farmacia Norte Jujuy",
                        "tipo_documento": "TICKET",
                        "importe_neto": "24600.50",
                        "impuestos_iva": "0.00",
                        "importe_total": "24600.50",
                    },
                    {
                        "fecha_emision": "28/10/2026",
                        "emisor": "Energía Eléctrica EJESA",
                        "tipo_documento": "FACTURA B",
                        "importe_neto": "26400.00",
                        "impuestos_iva": "5544.00",
                        "importe_total": "31944.00",
                    },
                ]

            df_an = preparar_dataframe_analitica(registros_analitica)

            if not df_an.empty:
                gasto_total = float(df_an["importe_total_num"].sum())
                iva_total = float(df_an["iva_num"].sum())
                cantidad_comp = len(df_an)
                ticket_promedio = gasto_total / cantidad_comp if cantidad_comp > 0 else 0.0

                df_por_proveedor = (
                    df_an.groupby("emisor", as_index=False)
                    .agg(
                        Comprobantes=("importe_total_num", "count"),
                        Importe_Neto=("importe_neto_num", "sum"),
                        IVA_Total=("iva_num", "sum"),
                        Total_Facturado=("importe_total_num", "sum"),
                    )
                    .sort_values(by="Total_Facturado", ascending=False)
                )

                proveedor_top = (
                    str(df_por_proveedor.iloc[0]["emisor"])
                    if not df_por_proveedor.empty
                    else "Sin datos"
                )
                monto_top = (
                    float(df_por_proveedor.iloc[0]["Total_Facturado"])
                    if not df_por_proveedor.empty
                    else 0.0
                )

                # Fila de KPIs destacados con st.metric
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                with kpi1:
                    st.metric(
                        label="💰 Gasto Total Acumulado",
                        value=f"$ {gasto_total:,.2f}",
                        delta=f"IVA: $ {iva_total:,.2f}",
                    )
                with kpi2:
                    st.metric(
                        label="🧾 Promedio por Comprobante",
                        value=f"$ {ticket_promedio:,.2f}",
                    )
                with kpi3:
                    st.metric(
                        label="📄 Facturas Procesadas",
                        value=str(cantidad_comp),
                    )
                with kpi4:
                    st.metric(
                        label="🏢 Principal Proveedor",
                        value=proveedor_top[:22] + ("…" if len(proveedor_top) > 22 else ""),
                        delta=f"$ {monto_top:,.2f}",
                    )

                st.markdown("---")

                # Gráficos nativos de Streamlit con Pandas: st.line_chart y st.bar_chart
                col_graf_linea, col_graf_barras = st.columns(2, gap="large")

                with col_graf_linea:
                    st.markdown("##### 📈 Evolución de Gasto Mensual y Acumulado (`st.line_chart`)")
                    df_mensual = (
                        df_an.groupby("periodo_mensual", as_index=True)["importe_total_num"]
                        .sum()
                        .sort_index()
                        .to_frame(name="Gasto Mensual ($)")
                    )
                    df_mensual["Gasto Acumulado ($)"] = df_mensual["Gasto Mensual ($)"].cumsum()
                    st.line_chart(df_mensual, use_container_width=True)

                with col_graf_barras:
                    st.markdown("##### 📊 Montos Totales Agrupados por Proveedor (`st.bar_chart`)")
                    df_grafico_prov = df_por_proveedor.set_index("emisor")[["Total_Facturado"]].rename(
                        columns={"Total_Facturado": "Total Facturado ($)"}
                    )
                    st.bar_chart(df_grafico_prov, use_container_width=True)

                with st.expander("🗂️ Ver desglose por Tipo de Comprobante y Tabla Resumen por Proveedor", expanded=False):
                    col_tipo_g, col_tabla_g = st.columns([1, 1.4], gap="large")
                    with col_tipo_g:
                        st.markdown("**Distribución por Tipo de Comprobante**")
                        df_por_tipo = (
                            df_an.groupby("tipo_documento")["importe_total_num"]
                            .sum()
                            .to_frame(name="Importe Total ($)")
                        )
                        st.bar_chart(df_por_tipo, use_container_width=True)
                    with col_tabla_g:
                        st.markdown("**Resumen Contable Agrupado por Proveedor**")
                        st.dataframe(
                            df_por_proveedor.rename(
                                columns={
                                    "emisor": "Proveedor / Emisor",
                                    "Comprobantes": "Cant.",
                                    "Importe_Neto": "Neto ($)",
                                    "IVA_Total": "IVA ($)",
                                    "Total_Facturado": "Total ($)",
                                }
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )

    # ==========================================================================
    # PESTAÑA 4: EVALUACIÓN DE EXACTITUD CONTRA TRANSCRIPCIÓN DE REFERENCIA
    # ==========================================================================
    with tab_evaluacion:
        with st.container(border=True):
            st.markdown(
                """
                <div class="captia-card-header">
                    <span class="captia-card-title">📊 Auditoría de Exactitud (Ground Truth — TFI Obj. 5)</span>
                    <span class="captia-chip captia-chip-indigo">🎯 Control de Calidad IA</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                "Compará los datos extraídos automáticamente frente a la **transcripción manual de referencia** "
                "del comprobante para medir la tasa de exactitud del motor."
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

            st.markdown("##### Completá la columna **'Transcripción de Referencia (Real)'** con los valores reales del comprobante:")
            df_ref_editado = st.data_editor(
                df_ref_inicial,
                column_config={
                    "Clave": None,
                    "Campo": st.column_config.TextColumn("Campo", disabled=True),
                    "Valor Extraído por la App": st.column_config.TextColumn(
                        "Valor Extraído por la App", disabled=True
                    ),
                    "Transcripción de Referencia (Real)": st.column_config.TextColumn(
                        "Transcripción de Referencia (Escribí el valor real)"
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
                    "📥 Descargar Informe de Auditoría (CSV)",
                    data=csv_eval,
                    file_name="evaluacion_exactitud_tfi.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with col_detalle:
                st.dataframe(df_comparacion, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
