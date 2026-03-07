from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation

DET_MODEL = "PekingU/rtdetr_r50vd_coco_o365"
POSE_MODEL = "usyd-community/vitpose-plus-base"


def load_models(device):

    print("Loading detection model...")

    det_processor = AutoProcessor.from_pretrained(DET_MODEL)

    det_model = RTDetrForObjectDetection.from_pretrained(
        DET_MODEL
    ).to(device)

    print("Loading pose model...")

    pose_processor = AutoProcessor.from_pretrained(
        POSE_MODEL,
        use_fast=False
    )

    pose_model = VitPoseForPoseEstimation.from_pretrained(
        POSE_MODEL
    ).to(device)

    return det_processor, det_model, pose_processor, pose_model