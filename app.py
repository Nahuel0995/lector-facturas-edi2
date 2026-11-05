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

    # 3. Reducción de ruido preservando la nitidez de los bordes de letras y números
    filtrado = cv2.bilateralFilter(gris, d=9, sigmaColor=75, sigmaSpace=75)

    # 4. Binarización adaptativa para eliminar sombras e iluminación despareja
    binarizada = cv2.adaptiveThreshold(
        filtrado,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=15,
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

            # Renderizar la página en alta definición (220 DPI) para OpenCV y Tesseract OCR
            pix = pagina.get_pixmap(dpi=220, alpha=False)
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
# 4. MÓDULO DE EXTRACCIÓN HÍBRIDA (GROQ / GEMINI / OLLAMA LOCAL + REGEX)
# ==============================================================================

def obtener_modelos_ollama(url_base: str = "http://localhost:11434") -> Tuple[List[str], Optional[str]]:
    """Consulta automáticamente la API local de Ollama para listar los modelos instalados."""
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


def detectar_proveedor_api(api_key: str) -> str:
    """Detecta automáticamente el proveedor según el formato de la API Key."""
    clave = api_key.strip().strip('"').strip("'")
    if clave.startswith("AIza"):
        return "google"
    if clave.startswith("sk-or-"):
        return "openrouter"
    return "groq"


def obtener_modelos_nube(api_key: str) -> Tuple[List[str], str, Optional[str]]:
    """
    Consulta en vivo a la API en la nube (Groq, Google Gemini u OpenRouter)
    para listar únicamente los modelos activos y habilitados en la cuenta del usuario.
    """
    clave = api_key.strip().strip('"').strip("'")
    if not clave:
        return [], "groq", None

    proveedor = detectar_proveedor_api(clave)

    if proveedor == "google":
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={clave}"
            resp = requests.get(url, timeout=10)
            if not resp.ok:
                detalle = resp.json().get("error", {}).get("message", resp.reason)
                return [], "google", f"Clave de Google inválida o sin acceso ({resp.status_code}): {detalle}"
            datos = resp.json().get("models", [])
            modelos = []
            for m in datos:
                metodos = m.get("supportedGenerationMethods", [])
                nombre = m.get("name", "").replace("models/", "")
                if "generateContent" in metodos and "gemini" in nombre.lower():
                    if not any(excl in nombre.lower() for excl in ["vision", "embedding", "imagen", "veo", "tts", "audio"]):
                        modelos.append(nombre)
            modelos.sort(key=lambda x: (0 if "2.5-flash" in x else 1 if "2.0-flash" in x else 2 if "flash" in x else 3, x))
            return modelos, "google", None
        except Exception as exc:
            return [], "google", f"Error al consultar modelos de Google Gemini: {exc}"

    url_modelos = (
        "https://openrouter.ai/api/v1/models"
        if proveedor == "openrouter"
        else "https://api.groq.com/openai/v1/models"
    )
    try:
        headers = {"Authorization": f"Bearer {clave}"}
        resp = requests.get(url_modelos, headers=headers, timeout=10)
        if not resp.ok:
            try:
                detalle = resp.json().get("error", {}).get("message", resp.reason)
            except Exception:
                detalle = resp.reason
            return [], proveedor, f"Error en API ({resp.status_code}): {detalle}"

        datos = resp.json().get("data", [])
        palabras_excluidas = (
            "whisper",
            "tts",
            "guard",
            "safeguard",
            "embed",
            "playai",
            "allam",
            "distil-whisper",
            "orpheus",
        )
        modelos = [
            m.get("id", "")
            for m in datos
            if m.get("id") and not any(p in m.get("id", "").lower() for p in palabras_excluidas)
        ]

        def prioridad_modelo(nombre_mod: str) -> Tuple[int, str]:
            n = nombre_mod.lower()
            if "llama-3.3" in n or "llama-4" in n:
                return (0, n)
            if "llama-3.1" in n or "llama3" in n:
                return (1, n)
            if "qwen" in n or "gemma" in n or "deepseek" in n:
                return (2, n)
            return (3, n)

        modelos.sort(key=prioridad_modelo)
        return modelos, proveedor, None
    except Exception as exc:
        return [], proveedor, f"No se pudieron listar los modelos de la nube: {exc}"


def extraer_con_google_gemini(
    texto_ocr: str,
    api_key: str,
    modelo: str = "gemini-2.5-flash",
) -> Tuple[Dict[str, str], Optional[str]]:
    """Extrae los 9 campos usando la API gratuita de Google AI Studio (Gemini)."""
    clave = api_key.strip().strip('"').strip("'")
    nombre_modelo = modelo.strip().replace("models/", "") or "gemini-2.5-flash"

    def invocar_gemini(mod: str) -> requests.Response:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{mod}:generateContent?key={clave}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": f"{PROMPT_SISTEMA}\n\nTEXTO OCR DEL COMPROBANTE:\n{texto_ocr}"}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
            },
        }
        return requests.post(url, json=payload, timeout=45)

    try:
        resp = invocar_gemini(nombre_modelo)
        if resp.status_code == 404:
            modelos_disp, _, _ = obtener_modelos_nube(clave)
            if modelos_disp:
                nombre_modelo = modelos_disp[0]
                resp = invocar_gemini(nombre_modelo)

        if not resp.ok:
            try:
                detalle = resp.json().get("error", {}).get("message", resp.reason)
            except Exception:
                detalle = resp.text
            return normalizar_diccionario_extraido({}), f"Error en Google Gemini ({resp.status_code}): {detalle}"

        candidatos = resp.json().get("candidates", [])
        texto_resp = candidatos[0]["content"]["parts"][0]["text"] if candidatos else "{}"
        datos = extraer_json_de_respuesta(texto_resp)
        return normalizar_diccionario_extraido(datos), None
    except Exception as exc:
        return normalizar_diccionario_extraido({}), f"Error de conexión con Google Gemini: {exc}"


def extraer_con_api_nube(
    texto_ocr: str,
    api_key: str,
    modelo: str = "llama-3.3-70b-versatile",
    endpoint_url: str = "https://api.groq.com/openai/v1/chat/completions",
) -> Tuple[Dict[str, str], Optional[str]]:
    """
    Envía el texto OCR a una API en la nube (Groq, Google Gemini u OpenRouter)
    para permitir el uso desde cualquier celular en internet sin depender de hardware local.
    """
    clave_limpia = api_key.strip().strip('"').strip("'")
    if not clave_limpia:
        return normalizar_diccionario_extraido({}), "Falta ingresar la clave de API (API Key) para el modo Nube."

    proveedor = detectar_proveedor_api(clave_limpia)
    if proveedor == "google":
        return extraer_con_google_gemini(texto_ocr, clave_limpia, modelo=modelo)

    if proveedor == "openrouter" and "groq.com" in endpoint_url:
        url_limpia = "https://openrouter.ai/api/v1/chat/completions"
    else:
        url_limpia = endpoint_url.strip().rstrip("/")
        if not url_limpia.endswith("/chat/completions"):
            if url_limpia.endswith("/v1"):
                url_limpia += "/chat/completions"
            elif "/v1" not in url_limpia:
                url_limpia += "/openai/v1/chat/completions"
            else:
                url_limpia += "/chat/completions"

    headers = {
        "Authorization": f"Bearer {clave_limpia}",
        "Content-Type": "application/json",
    }

    def enviar_peticion(mod_actual: str, forzar_json: bool = True) -> requests.Response:
        cuerpo = {
            "model": mod_actual.strip(),
            "messages": [
                {"role": "system", "content": PROMPT_SISTEMA},
                {"role": "user", "content": f"TEXTO OCR DEL COMPROBANTE:\n{texto_ocr}"},
            ],
            "temperature": 0.0,
        }
        if forzar_json:
            cuerpo["response_format"] = {"type": "json_object"}
        return requests.post(url_limpia, headers=headers, json=cuerpo, timeout=45)

    try:
        resp = enviar_peticion(modelo, forzar_json=True)

        if resp.status_code == 404:
            modelos_activos, _, _ = obtener_modelos_nube(clave_limpia)
            for mod_alternativo in modelos_activos[:3]:
                resp = enviar_peticion(mod_alternativo, forzar_json=True)
                if resp.status_code == 400:
                    resp = enviar_peticion(mod_alternativo, forzar_json=False)
                if resp.ok:
                    break

        if resp.status_code == 400:
            resp = enviar_peticion(modelo, forzar_json=False)

        if not resp.ok:
            detalle = ""
            try:
                error_json = resp.json()
                detalle = error_json.get("error", {}).get("message", "")
            except Exception:
                detalle = resp.text
            return (
                normalizar_diccionario_extraido({}),
                f"Error en API Nube ({resp.status_code}): {detalle or resp.reason}",
            )
        contenido = resp.json()["choices"][0]["message"]["content"]
        datos = extraer_json_de_respuesta(contenido)
        return normalizar_diccionario_extraido(datos), None
    except Exception as exc:
        return normalizar_diccionario_extraido({}), f"Error de conexión con API Nube ({modelo}): {exc}"


def extraer_con_reglas_regex(texto_ocr: str) -> Dict[str, str]:
    """
    Extractor heurístico de respaldo basado en expresiones regulares.
    También se utiliza como refuerzo post-procesamiento para corregir CUIT y fechas.
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

    match_total = re.search(
        r"TOTAL\s*(?:A\s*PAGAR)?\s*[:$]?\s*\$?\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\d+(?:[.,]\d{2}))",
        texto_ocr,
        re.IGNORECASE,
    )
    if match_total:
        datos["importe_total"] = match_total.group(1)

    if re.search(r"\b(USD|U\$S|D[OÓ]LARES)\b", texto_ocr, re.IGNORECASE):
        datos["moneda"] = "USD"
    elif "$" in texto_ocr or re.search(r"\b(ARS|PESOS)\b", texto_ocr, re.IGNORECASE):
        datos["moneda"] = "ARS"

    return datos


def aplicar_refuerzo_regex(datos_ia: Dict[str, str], texto_ocr: str) -> Dict[str, str]:
    """
    Mejora híbrida: si el LLM omitió el CUIT, la fecha o el número de comprobante
    pero tienen un patrón exacto en el texto OCR, los completa con Regex.
    """
    respaldo = extraer_con_reglas_regex(texto_ocr)
    resultado = dict(datos_ia)
    for campo_clave in ["cuit", "fecha_emision", "numero_comprobante", "moneda"]:
        if resultado.get(campo_clave) == VALOR_NO_IDENTIFICADO and respaldo.get(campo_clave) != VALOR_NO_IDENTIFICADO:
            resultado[campo_clave] = respaldo[campo_clave]
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
            padding: 1.1rem 1.25rem !important;
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
    # BARRA LATERAL: CONFIGURACIÓN HÍBRIDA Y DIAGNÓSTICO AUTOMÁTICO
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Motor de Captura IA")

        # 1. Diagnóstico de Tesseract OCR
        ruta_tesseract = configurar_ruta_tesseract()
        if ruta_tesseract:
            st.success(f"Tesseract OCR activo ({ruta_tesseract})")
        else:
            st.error("No se encontró Tesseract OCR en el sistema.")
            ruta_manual = st.text_input(
                "Ruta manual a tesseract.exe:",
                placeholder=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            )
            if ruta_manual and os.path.exists(ruta_manual):
                pytesseract.pytesseract.tesseract_cmd = ruta_manual
                st.success("Ruta manual de Tesseract configurada.")

        # 2. Estado de Supabase / Modo Demo Local
        st.subheader("Repositorio en la Nube")
        usuario_actual = st.session_state.get("usuario_supabase")
        if usuario_actual and supabase_configurado:
            st.success(f"☁️ Conectado: {usuario_actual['email']}")
        elif supabase_configurado:
            st.info("☁️ Supabase listo. Iniciá sesión arriba a la derecha para sincronizar tus facturas.")
        else:
            st.caption("⚡ **Modo Demo Local activo** (Configurá `SUPABASE_URL` y `SUPABASE_KEY` para guardado cloud).")

        st.subheader("Calidad y Validación")
        aplicar_hibrido_regex = st.checkbox(
            "Refuerzo fiscal con Regex (CUIT / Fechas)",
            value=True,
            help="Valida y rescata patrones exactos de CUIT, fecha y número de factura si el modelo los omite.",
        )

        st.subheader("Motor de Inteligencia Artificial")
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

        modo_ia = st.radio(
            "Modo de procesamiento:",
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
                "Acepta claves gratuitas de **Groq** (`gsk_...`), **Google Gemini** (`AIza...`) "
                "u **OpenRouter** (`sk-or-...`). Detecta proveedor y modelos automáticamente."
            )
            api_key_nube = st.text_input(
                "API Key (Groq o Google Gemini):",
                value=clave_entorno,
                type="password",
                placeholder="Pega tu clave gsk_... o AIza...",
            )

            modelos_nube, proveedor_detectado, error_modelos_nube = obtener_modelos_nube(api_key_nube)
            nombre_prov = {
                "google": "Google Gemini",
                "groq": "Groq Cloud",
                "openrouter": "OpenRouter",
            }.get(proveedor_detectado, "Nube")

            if modelos_nube:
                st.success(f"✅ {nombre_prov} activo ({len(modelos_nube)} modelos)")
                modelo_seleccionado = st.selectbox(
                    f"Modelo en {nombre_prov}:",
                    options=modelos_nube,
                )
            else:
                if api_key_nube.strip() and error_modelos_nube:
                    st.warning(error_modelos_nube)
                opciones_por_defecto = (
                    ["gemini-2.5-flash", "gemini-2.0-flash"]
                    if proveedor_detectado == "google"
                    else ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
                )
                modelo_seleccionado = st.selectbox(
                    "Modelo en la nube:",
                    options=opciones_por_defecto,
                )

            with st.expander("⚙️ Opciones avanzadas de API (opcional)", expanded=False):
                endpoint_nube = st.text_input(
                    "Endpoint API (solo para Groq/OpenRouter):",
                    value="https://api.groq.com/openai/v1/chat/completions",
                    help="Si usas Google Gemini (AIza...), el endpoint oficial de Google se selecciona automáticamente.",
                )

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
                        <span class="captia-card-title">1. Captura e Ingreso del Comprobante</span>
                        <span class="captia-chip captia-chip-indigo">📥 Escáner Inteligente</span>
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
                            "upscaling cúbico, filtro bilateral y binarización adaptativa (blockSize=31, C=15)."
                        ),
                    )

                metodo_entrada = st.radio(
                    "Método de recepción:",
                    options=["Subir archivo (PDF, JPG, PNG, WEBP)", "Capturar con cámara (Celular / PC)"],
                    horizontal=True,
                )

                archivo_imagen = None
                if metodo_entrada.startswith("Subir"):
                    archivo_imagen = st.file_uploader(
                        "Arrastrá tu factura en PDF o imagen (JPG, PNG, WEBP) aquí, o hacé clic para explorar:",
                        type=["pdf", "png", "jpg", "jpeg", "webp"],
                    )
                else:
                    archivo_imagen = st.camera_input("Enfocá el comprobante con buena iluminación:")

                if archivo_imagen is not None:
                    imagen_valida = False
                    imagen_original = None
                    imagen_enderezada = None
                    imagen_optimizada = None
                    recorte_exitoso = False
                    es_archivo_pdf = False
                    texto_digital_pdf = ""
                    total_paginas_pdf = 1

                    # Manejo de errores amigable al leer imagen o documento PDF
                    try:
                        bytes_img = archivo_imagen.getvalue()
                        nombre_subido = getattr(archivo_imagen, "name", "captura_camara.png") or "comprobante.png"
                        mime_subido = getattr(archivo_imagen, "type", "") or ""
                        es_archivo_pdf = (
                            nombre_subido.lower().endswith(".pdf")
                            or "pdf" in mime_subido.lower()
                            or bytes_img[:4] == b"%PDF"
                        )

                        if not bytes_img or len(bytes_img) < 50:
                            st.error(
                                "⚠️ El archivo subido está vacío o incompleto. "
                                "Por favor, seleccioná una factura válida en formato PDF, JPG, PNG o WEBP."
                            )
                        elif es_archivo_pdf:
                            imagen_pdf, texto_digital_pdf, total_paginas_pdf, error_pdf = procesar_archivo_pdf(
                                bytes_img, indice_pagina=0
                            )
                            if error_pdf or imagen_pdf is None:
                                st.error(
                                    error_pdf
                                    or "❌ No pudimos procesar este archivo PDF. Verificá que no esté dañado."
                                )
                            else:
                                if total_paginas_pdf > 1:
                                    pagina_sel = st.selectbox(
                                        f"📄 Documento PDF de {total_paginas_pdf} páginas — Seleccioná la página a visualizar:",
                                        options=list(range(1, total_paginas_pdf + 1)),
                                        index=0,
                                    )
                                    if pagina_sel > 1:
                                        imagen_pdf_pag, texto_pag_sel, _, err_p = procesar_archivo_pdf(
                                            bytes_img, indice_pagina=pagina_sel - 1
                                        )
                                        if not err_p and imagen_pdf_pag is not None:
                                            imagen_pdf = imagen_pdf_pag
                                            texto_digital_pdf = texto_pag_sel

                                imagen_original = imagen_pdf.convert("RGB")
                                st.session_state["ultimo_archivo_bytes"] = bytes_img
                                st.session_state["ultimo_archivo_nombre"] = nombre_subido
                                st.session_state["ultimo_archivo_mime"] = "application/pdf"
                                imagen_valida = True
                        else:
                            imagen_original = ImageOps.exif_transpose(
                                Image.open(io.BytesIO(bytes_img))
                            ).convert("RGB")
                            st.session_state["ultimo_archivo_bytes"] = bytes_img
                            st.session_state["ultimo_archivo_nombre"] = nombre_subido
                            st.session_state["ultimo_archivo_mime"] = mime_subido or "image/png"
                            imagen_valida = True
                    except Exception:
                        st.error(
                            "❌ No pudimos abrir este archivo. Asegurate de subir un comprobante válido "
                            "en formato **PDF, JPG, PNG o WEBP** que no esté dañado."
                        )

                    if imagen_valida and imagen_original is not None:
                        if es_archivo_pdf:
                            if len(texto_digital_pdf.strip()) >= 20:
                                st.info(
                                    f"📄 **Factura PDF Digital detectada ({total_paginas_pdf} pág.):** "
                                    "Se extraerá el texto digital nativo del PDF combinado con OCR para máxima precisión."
                                )
                            else:
                                st.info(
                                    f"📄 **Factura PDF Escaneada detectada ({total_paginas_pdf} pág.):** "
                                    "Página convertida a imagen de alta definición lista para OpenCV + Tesseract OCR."
                                )
                        else:
                            # Control preventivo de nitidez en fotos tomadas con celular/cámara
                            puntaje_nitidez = calcular_nitidez_imagen(imagen_original)
                            if puntaje_nitidez < 45.0:
                                st.warning(
                                    "⚠️ **Aviso de calidad:** La foto parece estar un poco borrosa o fuera de foco. "
                                    "Si algún dato no se detecta bien, te sugerimos tomar la foto con más luz y buen enfoque."
                                )

                        try:
                            # Convertir imagen_original a bytes PNG en memoria si viene de un PDF
                            if es_archivo_pdf:
                                buf_temp = io.BytesIO()
                                imagen_original.save(buf_temp, format="PNG")
                                bytes_para_opencv = buf_temp.getvalue()
                            else:
                                bytes_para_opencv = bytes_img

                            # Si es un PDF digital limpio, no hace falta deformar perspectiva salvo que sea un escaneo
                            if usar_recorte_camscanner and not (es_archivo_pdf and len(texto_digital_pdf.strip()) >= 20):
                                imagen_enderezada, recorte_exitoso = escanear_documento_con_estado(bytes_para_opencv)
                            else:
                                imagen_enderezada, recorte_exitoso = imagen_original, False

                            imagen_optimizada = (
                                optimizar_para_tesseract(imagen_enderezada)
                                if usar_filtro_escaner
                                else imagen_enderezada
                            )
                        except Exception:
                            st.warning(
                                "ℹ️ No se pudo aplicar el filtro avanzado de imagen; "
                                "se utilizará la vista original para la lectura."
                            )
                            imagen_enderezada = imagen_original
                            imagen_optimizada = imagen_original
                            recorte_exitoso = False

                        if usar_recorte_camscanner and not (es_archivo_pdf and len(texto_digital_pdf.strip()) >= 20):
                            if recorte_exitoso:
                                st.caption("📐 **CamScanner OpenCV:** 4 esquinas detectadas — fondo recortado y perspectiva enderezada (`warpPerspective`).")
                            else:
                                st.caption("ℹ️ **CamScanner OpenCV:** No se detectaron 4 bordes con contraste suficiente; se conserva el encuadre original sin errores.")

                        col_antes, col_despues = st.columns(2)
                        with col_antes:
                            st.image(
                                imagen_original,
                                caption="Vista Original (PDF renderizado)" if es_archivo_pdf else "Antes (Foto Original)",
                                use_container_width=True,
                            )
                        with col_despues:
                            st.image(
                                imagen_enderezada,
                                caption=(
                                    "Después (Recortada y Enderezada por OpenCV)"
                                    if recorte_exitoso
                                    else ("Vista Lista para Extracción" if es_archivo_pdf else "Después (Encuadre Original Conservado)")
                                ),
                                use_container_width=True,
                            )

                        if usar_filtro_escaner:
                            with st.expander("🔍 Ver imagen binarizada que recibe Tesseract OCR", expanded=False):
                                st.image(
                                    imagen_optimizada,
                                    caption="Salida de optimizar_para_tesseract() lista para OCR",
                                    use_container_width=True,
                                )

                        if st.button("✨ Escanear y Extraer Datos con IA", type="primary", use_container_width=True):
                            try:
                                with st.spinner("Procesando factura..."):
                                    texto_ocr_img, alerta_ocr = ejecutar_ocr(imagen_optimizada, idioma="spa")

                                    if es_archivo_pdf and len(texto_digital_pdf.strip()) >= 20:
                                        # Combinar texto nativo del PDF + texto OCR (por si el logo/emisor está como imagen)
                                        if texto_ocr_img and len(texto_ocr_img.strip()) >= 10:
                                            texto_detectado = (
                                                f"{texto_digital_pdf.strip()}\n\n"
                                                f"--- LECTURA OCR COMPLEMENTARIA ---\n{texto_ocr_img.strip()}"
                                            )
                                        else:
                                            texto_detectado = texto_digital_pdf.strip()
                                        alerta_ocr = None
                                    else:
                                        texto_detectado = texto_ocr_img

                                    st.session_state["texto_ocr"] = texto_detectado

                                    if alerta_ocr:
                                        st.warning(alerta_ocr)

                                    if not texto_detectado or len(texto_detectado.strip()) < 10:
                                        st.warning(
                                            "⚠️ No logramos detectar texto suficiente en el documento. "
                                            "Puede que la imagen esté borrosa, muy oscura o demasiado lejos. "
                                            "Intentá acercar la cámara o desactivar/activar el Filtro Escáner."
                                        )
                                    else:
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
                                                "ℹ️ El servicio de IA en la nube demoró o devolvió un aviso; "
                                                "se aplicó automáticamente la extracción de respaldo por reglas (Regex)."
                                            )
                                            datos_ext = extraer_con_reglas_regex(texto_detectado)
                                        elif aplicar_hibrido_regex:
                                            datos_ext = aplicar_refuerzo_regex(datos_ext, texto_detectado)

                                        st.session_state["datos_actuales"] = datos_ext
                                        st.success("✅ Comprobante procesado con éxito. Revisá los montos y datos en el panel derecho.")
                            except Exception:
                                st.error(
                                    "❌ Ocurrió un inconveniente inesperado al analizar la factura. "
                                    "Por favor, intentá nuevamente con un archivo más claro."
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

                # Segmentación con st.expander para ocultar datos secundarios, JSON crudo y texto OCR
                with st.expander("🗂️ Ver JSON crudo estructurado y datos técnicos del OCR", expanded=False):
                    sub_tab_json, sub_tab_ocr = st.tabs(["📄 JSON Estructurado (IA)", "📝 Texto Crudo (Tesseract OCR)"])
                    with sub_tab_json:
                        st.caption("Estructura JSON normalizada lista para almacenar o exportar:")
                        st.json(datos_revisados)
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
                    json_individual = json.dumps(datos_revisados, ensure_ascii=False, indent=2)
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
