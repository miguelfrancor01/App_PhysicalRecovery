"""Módulo de carga de modelos de detección y estimación de pose.

Define los identificadores de los modelos en HuggingFace y expone
una función para cargarlos y moverlos al dispositivo de cómputo.
"""

from transformers import (
    AutoProcessor,
    RTDetrForObjectDetection,
    VitPoseForPoseEstimation,
)


DET_MODEL = "PekingU/rtdetr_r18vd"
POSE_MODEL = "usyd-community/vitpose-plus-base"


def load_models(device):
    """Carga los modelos de detección de personas y estimación de pose.

    Descarga o usa la caché local de HuggingFace para RT-DETR (detección)
    y ViTPose-plus (estimación de keypoints), y los mueve al dispositivo
    especificado.

    Args:
    ----
        device: Dispositivo de cómputo ('cuda' o 'cpu') al que se
                envían los modelos tras su carga.

    Returns:
    -------
        tuple: (det_processor, det_model, pose_processor, pose_model)
            - det_processor: Processor de RT-DETR para preprocesar imágenes.
            - det_model: Modelo RT-DETR para detección de personas.
            - pose_processor: Processor de ViTPose para preprocesar crops.
            - pose_model: Modelo ViTPose para estimación de keypoints.

    """
    print("Loading detection model...")

    det_processor = AutoProcessor.from_pretrained(DET_MODEL)

    det_model = RTDetrForObjectDetection.from_pretrained(DET_MODEL).to(device)

    print("Loading pose model...")

    pose_processor = AutoProcessor.from_pretrained(POSE_MODEL, use_fast=False)

    pose_model = VitPoseForPoseEstimation.from_pretrained(POSE_MODEL).to(device)

    return det_processor, det_model, pose_processor, pose_model
