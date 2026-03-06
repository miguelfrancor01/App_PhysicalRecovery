import torch
import cv2
import numpy as np
from PIL import Image

from src.utils import load_image, save_image
from src.detection_model import load_detection_model, detect_persons
from src.pose_model import load_pose_model, estimate_pose, print_pose_results
from src.draw_pose import draw_pose
from src.camera_capture import capture_image_from_camera



def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    image_path = "images/person.jpg"
    output_path = "outputs/result.jpg"

    #image = load_image(image_path)
    image = capture_image_from_camera()
    image_rgb=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert numpy -> PIL
    image = Image.fromarray(image_rgb)

    
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # -------- HUMAN DETECTION --------
    det_processor, det_model = load_detection_model(device)

    person_boxes = detect_persons(
        image,
        det_processor,
        det_model,
        device
    )

    if len(person_boxes) == 0:
        print("No persons detected.")
        return

    # -------- POSE ESTIMATION --------
    pose_processor, pose_model = load_pose_model(device)

    pose_results = estimate_pose(
        image,
        person_boxes,
        pose_processor,
        pose_model,
        device
    )

    print_pose_results(pose_results, pose_model)

    # -------- DRAW SKELETON --------
    for person_pose in pose_results:

        keypoints = person_pose["keypoints"]
        scores = person_pose["scores"]

        image_cv = draw_pose(image_cv, keypoints, scores)

    save_image(image_cv, output_path)

    print(f"\nSaved result → {output_path}")


if __name__ == "__main__":
    main()