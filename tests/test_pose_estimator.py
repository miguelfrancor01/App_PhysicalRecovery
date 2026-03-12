import numpy as np
import torch
from PIL import Image
from unittest.mock import MagicMock

from pose_module.pose_estimator import estimate_pose


class DummyInputs(dict):
    """
    Clase auxiliar utilizada en pruebas unitarias para simular el objeto
    de entrada (`inputs`) que normalmente devuelve el processor de
    HuggingFace.

    En el pipeline real de estimación de pose, el processor genera un
    diccionario con tensores (por ejemplo, `pixel_values`) que luego se
    envían al dispositivo de cómputo mediante el método `.to(device)`
    de PyTorch.

    Esta clase hereda de `dict` para comportarse como el objeto real de
    inputs y define un método `.to()` que:

    - registra el dispositivo recibido (`device_received`),
    - retorna el propio objeto, imitando el comportamiento de PyTorch.

    Esto permite que las pruebas unitarias verifiquen que el pipeline
    envía correctamente los inputs al dispositivo (CPU/GPU) sin crear
    tensores reales ni ejecutar operaciones de PyTorch.
    """

    def __init__(self, *args, **kwargs):
        """
        Inicializa el diccionario de inputs simulado.

        Args:
            *args: Argumentos posicionales heredados de `dict`.
            **kwargs: Argumentos nombrados heredados de `dict`.
        """
        super().__init__(*args, **kwargs)
        self.device_received = None

    def to(self, device):
        """
        Simula el método `.to(device)` utilizado por tensores de PyTorch.

        En lugar de mover tensores a un dispositivo real, este método
        registra el dispositivo recibido para que las pruebas unitarias
        puedan verificar que el pipeline realiza correctamente esta
        operación.

        Args:
            device (str | torch.device): Dispositivo de cómputo.

        Returns:
            DummyInputs: El mismo objeto, para imitar el comportamiento
            encadenado del método `.to()` en PyTorch.
        """
        self.device_received = device
        return self


def test_estimate_pose_pipeline():
    """
    Prueba unitaria para verificar el flujo completo de `estimate_pose`.

    La prueba valida que el pipeline de estimación de pose ejecute
    correctamente las siguientes etapas:

    1. El processor recibe correctamente la imagen y las cajas de persona.
    2. Los inputs generados se envían al dispositivo de cómputo.
    3. Se añade el campo `dataset_index` al diccionario de inputs.
    4. El modelo recibe los inputs correctos para inferencia.
    5. Se ejecuta el postprocesamiento de estimación de pose.
    6. La función retorna correctamente el primer resultado de
       `pose_results`.
    """

    image = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    person_boxes = [[10, 10, 40, 50]]
    device = "cpu"

    processor = MagicMock()

    inputs_mock = DummyInputs({"pixel_values": "fake_tensor"})
    processor.return_value = inputs_mock

    processor.post_process_pose_estimation.return_value = [
        {"keypoints": [[1, 2], [3, 4]], "scores": [0.9, 0.8]}
    ]

    model = MagicMock()
    model.return_value = {"logits": "fake"}

    result = estimate_pose(
        image=image,
        person_boxes=person_boxes,
        processor=processor,
        model=model,
        device=device
    )

    processor.assert_called_once_with(
        image,
        boxes=[person_boxes],
        return_tensors="pt"
    )

    assert inputs_mock.device_received == device

    assert "dataset_index" in inputs_mock
    assert torch.equal(
        inputs_mock["dataset_index"],
        torch.tensor([0], device=device)
    )

    model.assert_called_once_with(**inputs_mock)

    processor.post_process_pose_estimation.assert_called_once_with(
        {"logits": "fake"},
        boxes=[person_boxes],
        threshold=0.3
    )

    assert result == {"keypoints": [[1, 2], [3, 4]], "scores": [0.9, 0.8]}