import torch
from transformers import AutoProcessor, VitPoseForPoseEstimation


MODEL = "usyd-community/vitpose-plus-base"


def load_pose_model(device):

    processor = AutoProcessor.from_pretrained(MODEL)

    model = VitPoseForPoseEstimation.from_pretrained(
        MODEL,
        device_map=device
    )

    return processor, model


def estimate_pose(image, person_boxes, processor, model, device):

    inputs = processor(
        image,
        boxes=[person_boxes],
        return_tensors="pt"
    ).to(device)

    inputs["dataset_index"] = torch.tensor([0], device=device)

    with torch.no_grad():
        outputs = model(**inputs)

    pose_results = processor.post_process_pose_estimation(
        outputs,
        boxes=[person_boxes],
        threshold=0.3
    )

    return pose_results[0]


def print_pose_results(pose_results, model):

    for i, person_pose in enumerate(pose_results):

        print(f"Person #{i}")

        for keypoint, label, score in zip(
            person_pose["keypoints"],
            person_pose["labels"],
            person_pose["scores"]
        ):

            name = model.config.id2label[label.item()]
            x, y = keypoint

            print(
                f" - {name}: x={x.item():.2f}, y={y.item():.2f}, score={score.item():.2f}"
            )