from unittest.mock import MagicMock

import numpy as np
import torch
from PIL import Image

from pose_module.detector import detect_persons


class DummyInputs(dict):
    """Clase auxiliar utilizada en pruebas unitarias para simular el objeto de
    entrada (`inputs`) que normalmente devuelve el processor de HuggingFace.

    En el pipeline real de estimación de pose, el processor genera un
    diccionario con tensores (por ejemplo, `pixel_values`) que posteriormente
    se envían al dispositivo de cómputo mediante el método `.to(device)`
    de PyTorch.

    Esta clase hereda de `dict` para comportarse como el objeto real de
    inputs y define un método `.to()` que:

    - registra el dispositivo recibido (`device_received`)
    - retorna el propio objeto, imitando el comportamiento de PyTorch

    Esto permite que las pruebas unitarias verifiquen que el pipeline envía
    correctamente los inputs al dispositivo (CPU/GPU) sin necesidad de crear
    tensores reales ni ejecutar operaciones reales de PyTorch.
    """

    def __init__(self, *args, **kwargs):
        """Inicializa el diccionario de inputs simulado.

        Args:
        ----
            *args: Argumentos posicionales heredados de dict.
            **kwargs: Argumentos nombrados heredados de dict.

        """
        super().__init__(*args, **kwargs)
        self.device_received = None

    def to(self, device):
        """Simula el método `.to(device)` utilizado por tensores de PyTorch.

        En lugar de mover tensores a un dispositivo, este método simplemente
        registra el dispositivo recibido para que las pruebas unitarias
        puedan verificar que el pipeline realiza correctamente la operación.

        Args:
        ----
            device (str | torch.device): Dispositivo de cómputo.

        Returns:
        -------
            DummyInputs: El mismo objeto, para imitar el comportamiento
            encadenado de `.to()` en PyTorch.

        """
        self.device_received = device
        return self


def test_detect_persons_filtra_personas_y_convierte_cajas_a_xywh():
    """Prueba unitaria para verificar el comportamiento principal de
    `detect_persons`.

    Esta prueba valida que la función:

    1. Procese correctamente la imagen mediante el processor.
    2. Ejecute el modelo de detección.
    3. Filtre únicamente las detecciones con `label == 0` (personas).
    4. Convierta correctamente las cajas delimitadoras de formato
       `xyxy` a `xywh`.

    Además, verifica que se utilicen correctamente el dispositivo de
    cómputo y los parámetros de postprocesamiento del detector.
    """
    image = Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8))
    device = "cpu"

    processor = MagicMock()
    model = MagicMock()

    inputs_mock = DummyInputs({"pixel_values": "fake_tensor"})
    processor.return_value = inputs_mock

    model.return_value = {"logits": "fake_outputs"}

    # Dos cajas:
    # - la primera con label 0 (persona)
    # - la segunda con label 2 (otra clase, debe descartarse)
    boxes = torch.tensor(
        [
            [10.0, 20.0, 50.0, 80.0],  # persona
            [100.0, 30.0, 160.0, 90.0],  # no persona
        ]
    )
    labels = torch.tensor([0, 2])

    processor.post_process_object_detection.return_value = [
        {
            "boxes": boxes,
            "labels": labels,
        }
    ]

    result = detect_persons(
        image=image, processor=processor, model=model, device=device
    )

    # Verifica llamada al processor
    processor.assert_called_once_with(images=image, return_tensors="pt")

    # Verifica envío al device
    assert inputs_mock.device_received == device

    # Verifica ejecución del modelo
    model.assert_called_once_with(**inputs_mock)

    # Verifica postproceso
    processor.post_process_object_detection.assert_called_once()
    _, kwargs = processor.post_process_object_detection.call_args
    assert kwargs["threshold"] == 0.3
    assert torch.equal(
        kwargs["target_sizes"], torch.tensor([(image.height, image.width)])
    )

    # Solo debe quedar la caja de la persona
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 4)

    # Conversión esperada:
    # [x1, y1, x2, y2] = [10, 20, 50, 80]
    # [x, y, w, h] = [10, 20, 40, 60]
    expected = np.array([[10.0, 20.0, 40.0, 60.0]])
    assert np.allclose(result, expected)
