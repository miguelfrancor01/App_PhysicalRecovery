import torch
import numpy as np
from transformers import AutoProcessor, RTDetrForObjectDetection


MODEL = "PekingU/rtdetr_r50vd_coco_o365"


def load_detection_model(device):

    processor = AutoProcessor.from_pretrained(MODEL)

    model = RTDetrForObjectDetection.from_pretrained(
        MODEL,
        device_map=device
    )

    return processor, model


def detect_persons(image, processor, model, device):

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([(image.height, image.width)]),
        threshold=0.3
    )[0]

    person_boxes = results["boxes"][results["labels"] == 0]
    person_boxes = person_boxes.cpu().numpy()

    # convert VOC → COCO
    person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
    person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]

    return person_boxes