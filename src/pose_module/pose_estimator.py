import torch


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