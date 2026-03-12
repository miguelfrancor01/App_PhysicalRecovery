import numpy as np
from PIL import Image

from preprocessing.frame_preprocessing import procesar_frame_para_modelo


def test_procesar_frame_para_modelo_realiza_pipeline_completo_correctamente():
    """
    Prueba unitaria para verificar el flujo completo de
    `procesar_frame_para_modelo`.

    Esta prueba valida que el pipeline de preprocesamiento ejecute
    correctamente las siguientes etapas:

    1. Validación exitosa del frame de entrada.
    2. Redimensionamiento de la imagen conservando proporciones.
    3. Aplicación de suavizado opcional.
    4. Conversión del espacio de color BGR -> RGB.
    5. Conversión de la imagen RGB a formato PIL.
    6. Retorno de la estructura de salida esperada.

    También se verifica que el redimensionamiento mantenga la relación
    de aspecto y que la conversión de canales BGR->RGB sea correcta.
    """

    # Frame BGR sintético de 100x200 (ancho mayor al objetivo)
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    # Color BGR arbitrario para validar conversión de canales
    # B=10, G=20, R=30
    frame[:, :] = [10, 20, 30]

    resultado = procesar_frame_para_modelo(
        frame=frame,
        frame_index=5,
        redimensionar=True,
        ancho_objetivo=100,
        aplicar_suavizado=True,
        kernel_suavizado=(3, 3),
    )

    # Validación general del resultado
    assert resultado["valid"] is True
    assert resultado["message"] == "ok"
    assert resultado["frame_index"] == 5

    # Verificar tipos de salida
    assert isinstance(resultado["frame_procesado"], np.ndarray)
    assert isinstance(resultado["frame_rgb"], np.ndarray)
    assert isinstance(resultado["image_pil"], Image.Image)

    # Verificar redimensionamiento conservando proporción:
    # original = 100x200 -> nuevo = 50x100
    assert resultado["frame_procesado"].shape == (50, 100, 3)
    assert resultado["frame_rgb"].shape == (50, 100, 3)

    # Verificar conversión BGR -> RGB en un píxel
    pixel_rgb = resultado["frame_rgb"][0, 0]
    assert np.array_equal(pixel_rgb, np.array([30, 20, 10], dtype=np.uint8))