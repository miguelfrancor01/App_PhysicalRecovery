import torch
import cv2
import time
from PIL import Image

from src.pose_module.model_loader import load_models
from src.pose_module.detector import detect_persons
from src.pose_module.pose_estimator import estimate_pose
from src.pose_module.visualizer import draw_pose
from src.report_module.session_data import SessionData
from src.report_module.report_generator import generate_report


PROCESS_EVERY_N_FRAMES = 3


def main():

    # =================================
    # DEVICE CONFIGURATION
    # =================================

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n=================================")
    print("Initializing Pose Estimation System")
    print("=================================")

    print(f"Using device: {device}")

    if device == "cuda":
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")

    print("=================================\n")

    # =================================
    # LOAD MODELS
    # =================================

    print("Loading AI models...")

    print("-> Loading detection model (RT-DETR)")
    print("-> Loading pose estimation model (VitPose)")

    det_processor, det_model, pose_processor, pose_model = load_models(device)

    print("Models loaded successfully.\n")

    # =================================
    # CREATE SESSION DATA
    # =================================

    print("Initializing session data...\n")

    session = SessionData()

    # =================================
    # OPEN VIDEO SOURCE
    # =================================

    print("Opening video source...")

    cap = cv2.VideoCapture("videos/ejercicio1.mp4")

    if not cap.isOpened():
        raise RuntimeError("Cannot open video file")

    print("Video opened successfully.")
    print("Controls:")
    print("   Q → Quit program")
    print("   R → Generate report\n")

    # =================================
    # VARIABLES
    # =================================

    frame_count = 0
    last_pose_results = []

    fps = 0
    fps_timer = time.time()

    print("Starting main processing loop...\n")

    # =================================
    # MAIN LOOP
    # =================================

    while True:

        ret, frame = cap.read()

        if not ret:
            print("End of video reached.")
            break

        frame_count += 1

        # =================================
        # RUN AI EVERY N FRAMES
        # =================================

        if frame_count % PROCESS_EVERY_N_FRAMES == 0:

            # Convert BGR → RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert RGB → PIL
            image_pil = Image.fromarray(image_rgb)

            # -----------------------------
            # PERSON DETECTION
            # -----------------------------

            person_boxes = detect_persons(
                image_pil,
                det_processor,
                det_model,
                device
            )

            # -----------------------------
            # POSE ESTIMATION
            # -----------------------------

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

        # =================================
        # DRAW POSE
        # =================================

        display_frame = frame.copy()

        for i, person_pose in enumerate(last_pose_results):

            keypoints = person_pose["keypoints"]
            scores = person_pose["scores"]
            labels = person_pose["labels"]

            print(f"\nPerson #{i}")

            for kp, score, label in zip(keypoints, scores, labels):

                name = pose_model.config.id2label[label.item()]
                x, y = kp

                print(
                    f" - {name}: x={x.item():.2f}, y={y.item():.2f}, score={score.item():.2f}"
                )

            display_frame = draw_pose(
                display_frame,
                keypoints,
                scores
            )

        # =================================
        # FPS COUNTER
        # =================================

        elapsed = time.time() - fps_timer

        if elapsed >= 1.0:

            fps = frame_count / elapsed
            frame_count = 0
            fps_timer = time.time()

        # =================================
        # OVERLAY TEXT
        # =================================

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

        # =================================
        # SHOW FRAME
        # =================================

        cv2.imshow("Pose Estimation - Press Q to quit", display_frame)

        key = cv2.waitKey(1) & 0xFF

        # Quit program
        if key == ord("q"):
            print("\nExiting program...")
            break

        # Generate report
        if key == ord("r"):
            print("\nGenerating session report...")
            generate_report(session)
            print("Report generated successfully.\n")

    # =================================
    # CLEANUP
    # =================================

    cap.release()
    cv2.destroyAllWindows()

    print("Program finished successfully.")


if __name__ == "__main__":
    main()