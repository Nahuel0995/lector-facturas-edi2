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
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

try:
    import cv2

    OPENCV_DISPONIBLE = True
except ImportError:
    cv2 = None  # type: ignore
    OPENCV_DISPONIBLE = False

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


def ejecutar_ocr(imagen: Image.Image, idioma: str = "spa") -> Tuple[str, Optional[str]]:
    """
    Ejecuta Tesseract OCR sobre la imagen con la configuración óptima para facturas
    (config='--oem 3 --psm 6'). Si 'spa' no está disponible, reintenta con 'eng'.
    """
    config_optima = "--oem 3 --psm 6"
    try:
        texto = pytesseract.image_to_string(imagen, lang=idioma, config=config_optima)
        return texto.strip(), None
    except pytesseract.TesseractError as err:
        if idioma != "eng":
            try:
                texto = pytesseract.image_to_string(imagen, lang="eng", config=config_optima)
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
# 6. REDISEÑO UI/UX: ESTILO EDITORIAL CÁLIDO Y MINIMALISTA
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


def aplicar_estilos_editoriales() -> None:
    """
    Inyecta CSS personalizado con estética editorial cálida, minimalista y luminosa:
    - Fondo general color arena suave (#F7F4EF)
    - Contenedores/tarjetas en blanco cálido (#FFFFFF) con bordes sutiles (#E2DDD5)
    - Textos en gris carbón (#2C2A29)
    - Acento en azul cielo pastel (#87B1DC) con texto oscuro
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

            /* Variables de Streamlit y Glide Data Grid (st.data_editor / st.dataframe) */
            --primary-color: #87B1DC;
            --background-color: #F7F4EF;
            --secondary-background-color: #FFFFFF;
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
            padding-top: 1.5rem !important;
            padding-bottom: 3.5rem !important;
            max-width: 1280px !important;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            color: var(--text-primary) !important;
            letter-spacing: -0.02em;
            font-weight: 700 !important;
        }

        p, li, span, label, [data-testid="stMarkdownContainer"] p {
            color: var(--text-primary);
        }

        [data-testid="stCaptionContainer"], small {
            color: var(--text-secondary) !important;
        }

        [data-testid="stWidgetLabel"] p {
            color: var(--text-primary) !important;
            font-weight: 600 !important;
            font-size: 0.92rem !important;
        }

        /* Barra superior limpia de créditos */
        .editorial-topbar-credits {
            background-color: #FFFFFF;
            border: 1px solid var(--border-subtle);
            border-radius: 6px;
            padding: 0.65rem 1.15rem;
            font-size: 0.86rem;
            font-weight: 600;
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 2px 10px rgba(44, 42, 41, 0.03);
        }

        .editorial-topbar-credits strong {
            color: var(--text-primary);
        }

        /* Hero Banner Editorial Minimalista */
        .bloom-hero {
            position: relative;
            width: 100%;
            min-height: 240px;
            border-radius: 8px;
            overflow: hidden;
            margin: 0.6rem 0 1.6rem 0;
            background-image:
                linear-gradient(180deg, rgba(34, 31, 28, 0.70) 0%, rgba(34, 31, 28, 0.56) 50%, rgba(34, 31, 28, 0.74) 100%),
                url('https://images.unsplash.com/photo-1497366216548-37526070297c?auto=format&fit=crop&w=1600&q=80');
            background-size: cover;
            background-position: center;
            box-shadow: 0 10px 28px rgba(44, 42, 41, 0.07);
            border: 1px solid #DCD6CC;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            padding: 2.2rem 2rem;
        }

        .bloom-hero-eyebrow {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.22em;
            color: #EFECE6 !important;
            margin-bottom: 0.7rem;
            background: rgba(255, 255, 255, 0.12);
            padding: 0.3rem 0.85rem;
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .bloom-hero-title {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            font-size: clamp(1.85rem, 3.2vw, 2.75rem) !important;
            font-weight: 700 !important;
            color: #FFFFFF !important;
            line-height: 1.2 !important;
            margin: 0 0 0.8rem 0 !important;
            letter-spacing: -0.02em;
        }

        .bloom-underline {
            display: inline-block;
            padding-bottom: 3px;
            border-bottom: 4px solid #87B1DC;
        }

        .bloom-hero-subtitle {
            font-family: 'DM Sans', sans-serif;
            font-size: 1rem;
            color: rgba(247, 244, 239, 0.92) !important;
            max-width: 680px;
            margin: 0 auto;
            font-weight: 400;
            line-height: 1.55;
        }

        /* Sidebar cálida */
        [data-testid="stSidebar"] {
            background-color: var(--bg-sidebar) !important;
            border-right: 1px solid var(--border-subtle) !important;
        }

        [data-testid="stSidebar"] * {
            color: var(--text-primary);
        }

        /* Contenedores y tarjetas en blanco cálido (#FFFFFF) con bordes sutiles (#E2DDD5) */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: 6px !important;
            padding: 1.45rem 1.55rem !important;
            box-shadow: 0 6px 20px rgba(44, 42, 41, 0.04) !important;
        }

        /* Pestañas (st.tabs) con indicador azul cielo pastel (#87B1DC) */
        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            justify-content: center !important;
            gap: 2.2rem !important;
            background-color: transparent !important;
            border-bottom: 1px solid var(--border-subtle) !important;
            padding-bottom: 0 !important;
            margin-bottom: 1.4rem !important;
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

        /* Botones principales y secundarios con azul cielo pastel (#87B1DC) y texto oscuro (#2C2A29) */
        .stButton > button[kind="primary"],
        [data-testid="stBaseButton-primary"],
        .stFormSubmitButton > button {
            background-color: var(--accent-blue) !important;
            color: var(--text-primary) !important;
            border: 1px solid #75A0CB !important;
            border-radius: 4px !important;
            font-weight: 600 !important;
            padding: 0.58rem 1.15rem !important;
            box-shadow: 0 2px 6px rgba(135, 177, 220, 0.25) !important;
            transition: all 0.2s ease !important;
        }

        .stButton > button[kind="primary"]:hover,
        [data-testid="stBaseButton-primary"]:hover,
        .stFormSubmitButton > button:hover {
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

        .stButton > button *, .stDownloadButton > button *, .stFormSubmitButton > button * {
            color: var(--text-primary) !important;
        }

        /* Toggle, Radio buttons y Checkboxes en azul cielo pastel (#87B1DC) */
        input[type="radio"], input[type="checkbox"] {
            accent-color: var(--accent-blue) !important;
        }

        [data-baseweb="checkbox"] > div:first-child {
            border-color: #87B1DC !important;
        }

        /* Inputs, Textareas y Selectboxes */
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

        /* Popovers y listas desplegables */
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

        /* File Uploader y Expander */
        [data-testid="stFileUploader"] section,
        [data-testid="stFileUploaderDropzone"] {
            background-color: var(--bg-card-warm) !important;
            border: 1.5px dashed #C9C2B8 !important;
            border-radius: 4px !important;
            padding: 1.25rem !important;
            color: var(--text-primary) !important;
        }

        [data-testid="stFileUploader"] section:hover,
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: var(--accent-blue) !important;
            background-color: #F3F6FA !important;
        }

        [data-testid="stFileUploaderDropzone"] button {
            background-color: #FFFFFF !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: 4px !important;
            font-weight: 600 !important;
        }

        [data-testid="stExpander"], details {
            background-color: var(--bg-card-warm) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: 4px !important;
            overflow: hidden;
            margin-bottom: 0.9rem !important;
        }

        [data-testid="stExpander"] summary, details > summary {
            background-color: var(--bg-card-warm) !important;
            color: var(--text-primary) !important;
            font-weight: 600 !important;
            padding: 0.7rem 1rem !important;
        }

        [data-testid="stExpanderDetails"] {
            background-color: #FFFFFF !important;
            border-top: 1px solid var(--border-subtle) !important;
            padding: 1rem !important;
        }

        [data-testid="stDataFrame"], [data-testid="stDataEditor"] {
            border: 1px solid var(--border-subtle) !important;
            border-radius: 4px !important;
            background-color: #FFFFFF !important;
            overflow: hidden;
        }

        [data-testid="stMetric"] {
            background-color: var(--bg-card-warm) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: 4px !important;
            padding: 1.1rem 1.25rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def renderizar_encabezado_y_auth(cliente_supabase: Optional[Client], supabase_configurado: bool) -> None:
    """
    Renderiza:
    1. La barra superior limpia con los créditos del proyecto a la izquierda y el botón
       'Iniciar sesión / Cerrar sesión' a la derecha.
    2. El panel desplegable de Login / Registro con Supabase (o estado de Modo Demo Local).
    3. El Hero Banner minimalista con subrayado decorativo en azul pastel (#87B1DC).
    """
    usuario = st.session_state.get("usuario_supabase")

    col_creditos, col_boton_auth = st.columns([3.8, 1.2], gap="small")

    with col_creditos:
        estado_badge = (
            f"🟢 Sesión activa: {usuario['email']}"
            if usuario
            else ("☁️ Supabase listo" if supabase_configurado else "🧪 Modo Demo Local activo")
        )
        st.markdown(
            f"""
            <div class="editorial-topbar-credits">
                <span><strong>Trabajo Final Integrador — EDI II (Ciencias de Datos)</strong> &nbsp;|&nbsp; Integrantes: Alan Bustamante y Christian Martínez</span>
                <span>{estado_badge}</span>
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
            etiqueta_btn = "✕ Cerrar acceso" if st.session_state["mostrar_panel_auth"] else "👤 Iniciar sesión / Registro"
            if st.button(etiqueta_btn, key="btn_login_top", type="primary", use_container_width=True):
                st.session_state["mostrar_panel_auth"] = not st.session_state["mostrar_panel_auth"]
                st.rerun()

    # Panel de autenticación (Iniciar Sesión / Registrarse) cuando el usuario lo despliega
    if st.session_state["mostrar_panel_auth"] and not usuario:
        with st.container(border=True):
            st.subheader("🔐 Acceso de Usuarios (Supabase Auth)")
            if not supabase_configurado or cliente_supabase is None:
                st.info(
                    "🧪 **Modo Demo Local activo:** Aún no se configuraron `SUPABASE_URL` y `SUPABASE_KEY` en "
                    "`st.secrets` (o variables de entorno). Puedes usar todas las funciones de diseño, filtro escáner OpenCV, "
                    "OCR y extracción con Groq normalmente en memoria local."
                )
            else:
                tab_login, tab_registro = st.tabs(["Iniciar Sesión", "Crear Cuenta Nueva"])

                with tab_login:
                    with st.form("form_login_supabase", clear_on_submit=False):
                        email_in = st.text_input("Correo electrónico:", placeholder="usuario@ejemplo.com")
                        pass_in = st.text_input("Contraseña:", type="password", placeholder="••••••••")
                        submit_login = st.form_submit_button("Ingresar a mi cuenta", use_container_width=True)

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
                        submit_reg = st.form_submit_button("Registrarme en la plataforma", use_container_width=True)

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

    # Hero Banner Principal
    st.markdown(
        """
        <div class="bloom-hero">
            <div class="bloom-hero-eyebrow">EDI II &bull; CIENCIAS DE DATOS</div>
            <h1 class="bloom-hero-title">
                Lector Inteligente de Comprobantes y <span class="bloom-underline">Facturas</span>
            </h1>
            <p class="bloom-hero-subtitle">
                Preprocesamiento con filtro escáner OpenCV, lectura OCR con Tesseract,
                extracción contable estructurada con Groq / IA y resguardo en la nube con Supabase.
            </p>
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
    aplicar_estilos_editoriales()

    supabase_url, supabase_key = obtener_credenciales_supabase()
    cliente_supabase = inicializar_cliente_supabase(supabase_url, supabase_key)
    supabase_configurado = cliente_supabase is not None

    renderizar_encabezado_y_auth(cliente_supabase, supabase_configurado)

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

        # 2. Estado de Supabase / Modo Demo Local
        st.subheader("Base de Datos y Storage")
        usuario_actual = st.session_state.get("usuario_supabase")
        if usuario_actual and supabase_configurado:
            st.success(f"☁️ Conectado como: {usuario_actual['email']}")
        elif supabase_configurado:
            st.info("☁️ Supabase disponible. Inicia sesión arriba a la derecha para guardar tus fotos en la nube.")
        else:
            st.caption("🧪 **Modo Demo Local activo** (Configura `SUPABASE_URL` y `SUPABASE_KEY` para persistencia en la nube).")

        st.subheader("Mejoras de Calidad (TFI Obj. 6)")
        aplicar_hibrido_regex = st.checkbox(
            "Refuerzo con Expresiones Regulares (CUIT/Fechas)",
            value=True,
            help="Valida y rescata patrones exactos de CUIT, fecha y número de factura si el modelo los omite.",
        )

        st.subheader("Modo de Inteligencia Artificial")
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
                "Acepta claves gratuitas de **Groq** (`gsk_...`), **Google AI Studio / Gemini** (`AIza...`) "
                "u **OpenRouter** (`sk-or-...`). Detecta automáticamente el proveedor y tus modelos activos."
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
                st.success(f"✅ Conectado a {nombre_prov} ({len(modelos_nube)} modelos disponibles)")
                modelo_seleccionado = st.selectbox(
                    f"Modelo activo en {nombre_prov}:",
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
    # PESTAÑA 1: CARGA, OPTIMIZACIÓN OPENCV, OCR, EXTRACCIÓN Y REVISIÓN MANUAL
    # ==========================================================================
    with tab_lector:
        col_carga, col_resultados = st.columns([1.05, 1.15], gap="large")

        with col_carga:
            with st.container(border=True):
                st.subheader("1. Entrada del Comprobante")

                usar_filtro_escaner = st.toggle(
                    "Mejorar calidad de foto para OCR (Filtro Escáner)",
                    value=True,
                    help=(
                        "Ejecuta optimizar_para_tesseract(pil_image) con OpenCV: escala de grises, "
                        "upscaling cúbico, filtro bilateral y binarización adaptativa (blockSize=31, C=15)."
                    ),
                )

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
                    # Guardar bytes originales en sesión para subirlos a Supabase Storage al confirmar
                    bytes_img = archivo_imagen.getvalue()
                    st.session_state["ultimo_archivo_bytes"] = bytes_img
                    st.session_state["ultimo_archivo_nombre"] = getattr(archivo_imagen, "name", "captura_camara.png")
                    st.session_state["ultimo_archivo_mime"] = getattr(archivo_imagen, "type", "image/png") or "image/png"

                    imagen_original = Image.open(io.BytesIO(bytes_img))
                    imagen_optimizada = (
                        optimizar_para_tesseract(imagen_original)
                        if usar_filtro_escaner
                        else imagen_original.convert("RGB")
                    )

                    with st.expander("🔍 Comparar Foto Original vs. Foto Optimizada (Filtro Escáner)", expanded=True):
                        col_img_orig, col_img_opt = st.columns(2)
                        with col_img_orig:
                            st.image(imagen_original, caption="Foto Original", use_container_width=True)
                        with col_img_opt:
                            st.image(
                                imagen_optimizada,
                                caption="Optimizada para Tesseract" if usar_filtro_escaner else "Sin Filtro Escáner",
                                use_container_width=True,
                            )

                    if st.button("🚀 Procesar Comprobante (OCR + IA)", type="primary", use_container_width=True):
                        with st.spinner("Ejecutando Tesseract OCR (--oem 3 --psm 6)..."):
                            texto_detectado, alerta_ocr = ejecutar_ocr(imagen_optimizada, idioma="spa")
                            st.session_state["texto_ocr"] = texto_detectado

                        if alerta_ocr:
                            st.warning(alerta_ocr)

                        if not texto_detectado or len(texto_detectado.strip()) < 10:
                            st.error(
                                "⚠️ La imagen no pudo procesarse correctamente o contiene muy poco texto legible. "
                                "Intenta con una foto más nítida o cambia el estado del Filtro Escáner."
                            )
                        else:
                            with st.spinner("Extrayendo los 9 campos estructurados con IA..."):
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

                with st.expander("📝 Ver texto crudo reconocido por Tesseract OCR", expanded=False):
                    texto_editable_ocr = st.text_area(
                        "Texto extraído por Tesseract (--oem 3 --psm 6):",
                        value=st.session_state["texto_ocr"],
                        height=170,
                    )
                    st.session_state["texto_ocr"] = texto_editable_ocr

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

                datos_revisados = {
                    fila["Clave"]: str(fila["Valor Detectado (Editable)"]).strip() or VALOR_NO_IDENTIFICADO
                    for _, fila in df_editado.iterrows()
                }
                st.session_state["datos_actuales"] = datos_revisados

                col_btn1, col_btn2, col_btn3 = st.columns(3)

                with col_btn1:
                    if st.button("➕ Guardar Comprobante", type="primary", use_container_width=True):
                        usuario_sesion = st.session_state.get("usuario_supabase")
                        bytes_actuales = st.session_state.get("ultimo_archivo_bytes")

                        # Si el usuario inició sesión en Supabase, sube imagen a Storage y guarda en tabla 'facturas'
                        if usuario_sesion and cliente_supabase is not None:
                            if not bytes_actuales:
                                st.warning("Primero sube o captura una imagen de comprobante antes de guardar en la nube.")
                            else:
                                with st.spinner("Subiendo imagen a Supabase Storage y registrando factura..."):
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
                            # Modo Demo Local en memoria de sesión
                            registro_local = {
                                "fecha_carga": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "modo": "Demo Local",
                            }
                            registro_local.update(datos_revisados)
                            st.session_state["historial_comprobantes"].append(registro_local)
                            st.success(
                                "Comprobante guardado en 'Registro Acumulado' (Modo Demo Local). "
                                "Inicia sesión arriba para guardarlo permanentemente en Supabase."
                            )

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
    # PESTAÑA 2: REGISTRO ACUMULADO (SUPABASE POR USUARIO O MODO DEMO LOCAL)
    # ==========================================================================
    with tab_historial:
        with st.container(border=True):
            usuario_sesion = st.session_state.get("usuario_supabase")

            if usuario_sesion and cliente_supabase is not None:
                st.subheader(f"☁️ Mis Comprobantes en la Nube ({usuario_sesion['email']})")
                registros_nube, error_consulta = consultar_facturas_usuario_supabase(
                    cliente_supabase, usuario_sesion["id"]
                )
                if error_consulta:
                    st.error(error_consulta)
                elif not registros_nube:
                    st.info(
                        "Aún no tienes comprobantes guardados en tu cuenta de Supabase. "
                        "Procesa una factura en la pestaña 1 y presiona **'➕ Guardar Comprobante'**."
                    )
                else:
                    columnas_mostrar = [
                        "created_at",
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
                    df_nube = pd.DataFrame(registros_nube)
                    cols_existentes = [c for c in columnas_mostrar if c in df_nube.columns]
                    df_vista = df_nube[cols_existentes] if cols_existentes else df_nube

                    st.dataframe(
                        df_vista,
                        column_config={
                            "imagen_url": st.column_config.LinkColumn(
                                "Foto del Comprobante", display_text="Ver imagen"
                            )
                        },
                        use_container_width=True,
                        hide_index=True,
                    )

                    col_exp1, col_exp2 = st.columns(2)
                    with col_exp1:
                        csv_nube = df_vista.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            "📥 Exportar Mis Facturas a CSV",
                            data=csv_nube,
                            file_name="mis_facturas_supabase.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                    with col_exp2:
                        json_nube = json.dumps(registros_nube, ensure_ascii=False, indent=2, default=str).encode("utf-8")
                        st.download_button(
                            "📥 Exportar Mis Facturas a JSON",
                            data=json_nube,
                            file_name="mis_facturas_supabase.json",
                            mime="application/json",
                            use_container_width=True,
                        )
            else:
                st.subheader("🗂️ Comprobantes Registrados en la Sesión (Modo Demo Local)")
                if supabase_configurado:
                    st.caption(
                        "💡 Estás viendo los comprobantes temporales de la sesión local. "
                        "Haz clic en **'👤 Iniciar sesión / Registro'** arriba a la derecha para ver y guardar tus propios comprobantes en Supabase."
                    )
                historial = st.session_state["historial_comprobantes"]

                if not historial:
                    st.info(
                        "Aún no has guardado comprobantes en el lote. "
                        "Procesa una factura en la pestaña 1 y presiona **'➕ Guardar Comprobante'**."
                    )
                else:
                    df_hist = pd.DataFrame(historial)
                    st.dataframe(df_hist, use_container_width=True, hide_index=True)

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
                        if st.button("🗑️ Vaciar Registro Local", use_container_width=True):
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
