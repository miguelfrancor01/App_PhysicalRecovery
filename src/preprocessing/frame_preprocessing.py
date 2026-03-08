"""
Módulo de preprocesamiento básico para frames de video.

Este módulo prepara los frames capturados con OpenCV antes de enviarlos a los
módulos de detección y estimación de pose. Su propósito es realizar únicamente
transformaciones sencillas para que la imagen quede lista para el modelo, sin
modificar la lógica de los módulos de pose.

Responsabilidades:
- Validar el frame recibido.
- Redimensionar opcionalmente la imagen para reducir carga computacional.
- Aplicar un suavizado ligero opcional.
- Convertir el frame de BGR a RGB.
- Convertir el frame RGB a formato PIL.
"""

import cv2
import numpy as np
from PIL import Image


def validar_frame(frame, frame_index=-1):
    """
    Verifica que el frame recibido tenga un formato válido.

    Args:
        frame (np.ndarray): Frame capturado por OpenCV.
        frame_index (int, optional): Índice del frame dentro del video o stream.

    Returns:
        dict: Diccionario con el resultado de la validación. Contiene:
            - valid (bool): Indica si el frame es válido.
            - message (str): Mensaje descriptivo del resultado.
            - frame_index (int): Índice del frame evaluado.
    """
    if frame is None:
        return {
            "valid": False,
            "message": "El frame es None.",
            "frame_index": frame_index,
        }

    if not isinstance(frame, np.ndarray):
        return {
            "valid": False,
            "message": f"Tipo de frame inválido: {type(frame)}",
            "frame_index": frame_index,
        }

    if frame.size == 0:
        return {
            "valid": False,
            "message": "El frame está vacío.",
            "frame_index": frame_index,
        }

    if len(frame.shape) != 3 or frame.shape[2] != 3:
        return {
            "valid": False,
            "message": f"Se esperaba una imagen BGR de 3 canales, pero se recibió shape={frame.shape}.",
            "frame_index": frame_index,
        }

    return {
        "valid": True,
        "message": "ok",
        "frame_index": frame_index,
    }


def redimensionar_frame(frame, ancho_objetivo=640):
    """
    Redimensiona el frame manteniendo la relación de aspecto si el ancho
    original es mayor al ancho objetivo.

    Args:
        frame (np.ndarray): Frame en formato BGR proveniente de OpenCV.
        ancho_objetivo (int, optional): Ancho máximo permitido para el frame.

    Returns:
        np.ndarray: Frame redimensionado si el ancho supera el objetivo,
        o el frame original si no es necesario redimensionar.
    """
    alto, ancho = frame.shape[:2]

    if ancho <= ancho_objetivo:
        return frame

    escala = ancho_objetivo / float(ancho)
    nuevo_alto = int(alto * escala)

    frame_redimensionado = cv2.resize(
        frame,
        (ancho_objetivo, nuevo_alto),
        interpolation=cv2.INTER_LINEAR
    )

    return frame_redimensionado


def suavizar_frame(frame, aplicar_suavizado=False, kernel=(3, 3)):
    """
    Aplica un desenfoque gaussiano ligero para reducir ruido visual.

    Args:
        frame (np.ndarray): Frame en formato BGR.
        aplicar_suavizado (bool, optional): Indica si se debe aplicar suavizado.
        kernel (tuple, optional): Tamaño del kernel para el filtro gaussiano.

    Returns:
        np.ndarray: Frame suavizado si se activó la opción, o el frame original.
    """
    if not aplicar_suavizado:
        return frame

    return cv2.GaussianBlur(frame, kernel, 0)


def convertir_bgr_a_rgb(frame_bgr):
    """
    Convierte un frame de OpenCV desde BGR a RGB.

    Args:
        frame_bgr (np.ndarray): Frame en formato BGR.

    Returns:
        np.ndarray: Frame convertido al espacio de color RGB.
    """
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def convertir_rgb_a_pil(frame_rgb):
    """
    Convierte un frame RGB a una imagen PIL.

    Args:
        frame_rgb (np.ndarray): Frame en formato RGB.

    Returns:
        PIL.Image.Image: Imagen en formato PIL.
    """
    return Image.fromarray(frame_rgb)


def procesar_frame_para_modelo(
    frame,
    frame_index=-1,
    redimensionar=False,
    ancho_objetivo=640,
    aplicar_suavizado=False,
    kernel_suavizado=(3, 3),
):
    """
    Ejecuta el preprocesamiento mínimo necesario para dejar un frame listo
    para los modelos de detección y pose.

    Args:
        frame (np.ndarray): Frame capturado desde OpenCV.
        frame_index (int, optional): Índice del frame dentro del flujo de video.
        redimensionar (bool, optional): Indica si el frame debe redimensionarse.
        ancho_objetivo (int, optional): Ancho máximo del frame en caso de redimensionar.
        aplicar_suavizado (bool, optional): Indica si se debe aplicar suavizado gaussiano.
        kernel_suavizado (tuple, optional): Kernel utilizado para el filtro gaussiano.

    Returns:
        dict: Diccionario con los resultados del preprocesamiento. Contiene:
            - valid (bool): Indica si el procesamiento fue exitoso.
            - message (str): Mensaje descriptivo del resultado.
            - frame_index (int): Índice del frame procesado.
            - frame_procesado (np.ndarray | None): Frame final usado para inferencia.
            - frame_rgb (np.ndarray | None): Frame convertido a RGB.
            - image_pil (PIL.Image.Image | None): Imagen convertida a formato PIL.
    """
    validacion = validar_frame(frame, frame_index)

    if not validacion["valid"]:
        return {
            "valid": False,
            "message": validacion["message"],
            "frame_index": frame_index,
            "frame_procesado": None,
            "frame_rgb": None,
            "image_pil": None,
        }

    frame_procesado = frame.copy()

    if redimensionar:
        frame_procesado = redimensionar_frame(
            frame_procesado,
            ancho_objetivo=ancho_objetivo
        )

    frame_procesado = suavizar_frame(
        frame_procesado,
        aplicar_suavizado=aplicar_suavizado,
        kernel=kernel_suavizado
    )

    try:
        frame_rgb = convertir_bgr_a_rgb(frame_procesado)
    except cv2.error as exc:
        return {
            "valid": False,
            "message": f"Error al convertir BGR a RGB: {exc}",
            "frame_index": frame_index,
            "frame_procesado": None,
            "frame_rgb": None,
            "image_pil": None,
        }

    try:
        image_pil = convertir_rgb_a_pil(frame_rgb)
    except Exception as exc:
        return {
            "valid": False,
            "message": f"Error al convertir RGB a PIL: {exc}",
            "frame_index": frame_index,
            "frame_procesado": None,
            "frame_rgb": None,
            "image_pil": None,
        }

    return {
        "valid": True,
        "message": "ok",
        "frame_index": frame_index,
        "frame_procesado": frame_procesado,
        "frame_rgb": frame_rgb,
        "image_pil": image_pil,
    }