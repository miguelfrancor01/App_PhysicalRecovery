"""Módulo de detección de personas usando RT-DETR.

Recibe una imagen PIL y retorna los bounding boxes de las personas
detectadas en formato [x, y, width, height].
"""

import numpy as np
import torch


def detect_persons(image, processor, model, device):
    """Detecta personas en una imagen usando RT-DETR.

    Procesa la imagen con el processor y el modelo de detección de objetos,
    filtra únicamente la clase 'persona' (label 0) y convierte los bounding
    boxes del formato [x1, y1, x2, y2] a [x, y, width, height].

    Args:
    ----
        image: Imagen PIL sobre la que se realiza la detección.
        processor: Processor de HuggingFace para RT-DETR.
        model: Modelo RT-DETR cargado y listo para inferencia.
        device: Dispositivo de cómputo ('cuda' o 'cpu').

    Returns:
    -------
        numpy.ndarray: Array de bounding boxes con forma (N, 4) en formato
        [x, y, width, height], donde N es el número de personas detectadas.

    """
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_object_detection(
        outputs, target_sizes=torch.tensor([(image.height, image.width)]), threshold=0.3
    )[0]

    person_boxes = results["boxes"][results["labels"] == 0]
    person_boxes = person_boxes.cpu().numpy()

    person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
    person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]

    return person_boxes
