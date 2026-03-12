"""Módulo de estimación de pose usando ViTPose.

Recibe una imagen PIL y los bounding boxes de personas detectadas,
y retorna los keypoints estimados para cada persona.
"""

import torch


def estimate_pose(image, person_boxes, processor, model, device):
    """Estima los keypoints corporales de las personas detectadas en una imagen.

    Preprocesa la imagen recortando las regiones de interés según los
    bounding boxes, ejecuta el modelo ViTPose y retorna los keypoints
    con sus puntuaciones de confianza.

    Args:
    ----
        image: Imagen PIL de entrada sobre la que se estima la pose.
        person_boxes (numpy.ndarray): Bounding boxes de personas en formato
            [x, y, width, height] con forma (N, 4).
        processor: Processor de HuggingFace para ViTPose.
        model: Modelo ViTPose cargado y listo para inferencia.
        device: Dispositivo de cómputo ('cuda' o 'cpu').

    Returns:
    -------
        list: Lista de resultados de pose para cada persona detectada,
        donde cada elemento contiene los keypoints y sus puntuaciones
        de confianza.

    """
    inputs = processor(image, boxes=[person_boxes], return_tensors="pt").to(device)

    inputs["dataset_index"] = torch.tensor([0], device=device)

    with torch.no_grad():
        outputs = model(**inputs)

    pose_results = processor.post_process_pose_estimation(
        outputs, boxes=[person_boxes], threshold=0.3
    )

    return pose_results[0]
