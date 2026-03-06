import torch
import cv2
import numpy as np
from PIL import Image
import time

from src.utils import save_image
from src.detection_model import load_detection_model, detect_persons
from src.pose_model import load_pose_model, estimate_pose
from src.draw_pose import draw_pose


PROCESS_EVERY_N_FRAMES = 3


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # -------- Se cargan los modelos --------
    print("Loading detection model...")
    det_processor, det_model = load_detection_model(device)

    print("Loading pose model...")
    pose_processor, pose_model = load_pose_model(device)

    print("Models loaded. Opening camera...")

    # -------- incializar la camara --------
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    print("Camera opened. Press Q to quit.")

    frame_count = 0
    last_pose_results = []
    fps = 0
    fps_timer = time.time()

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Failed to read frame.")
            break

        frame_count += 1

        # -------- procesamiento cada x frames --------
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:

            # Convertir BGR -> RGB -> PIL
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)

            # Detectar persona
            person_boxes = detect_persons(
                image_pil,
                det_processor,
                det_model,
                device
            )

            # Estimatar pose
            if len(person_boxes) > 0:
                pose_results = estimate_pose(
                    image_pil,
                    person_boxes,
                    pose_processor,
                    pose_model,
                    device
                )
                last_pose_results = pose_results
            else:
                last_pose_results = []

        # -------- dibujar --------
        display_frame = frame.copy()

        for person_pose in last_pose_results:
            keypoints = person_pose["keypoints"]
            scores = person_pose["scores"]
            display_frame = draw_pose(display_frame, keypoints, scores)

        # -------- contador de fps --------
        fps_count = frame_count
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps = fps_count / elapsed
            frame_count = 0
            fps_timer = time.time()

        cv2.putText(
            display_frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display_frame,
            f"Device: {device.upper()}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.imshow("Pose Estimation - Press Q to quit", display_frame)

        # Q -> salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()