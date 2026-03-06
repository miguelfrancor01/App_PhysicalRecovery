import cv2

def capture_image_from_camera(camera_index=0, save_path=None):
    """
    Capture one image from the webcam.

    Press SPACE to take the photo.
    Press ESC to cancel.

    Args:
        camera_index: camera device index (default 0)
        save_path: optional path to save the captured image

    Returns:
        Captured image (numpy array)
    """

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    print("Camera opened. Press SPACE to capture photo or ESC to exit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            raise RuntimeError("Failed to read frame from camera")

        cv2.imshow("Camera - Press SPACE", frame)

        key = cv2.waitKey(1)

        # ESC -> cancel
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            raise RuntimeError("Capture cancelled")

        # SPACE -> capture
        if key == 32:
            image = frame.copy()
            break

    cap.release()
    cv2.destroyAllWindows()

    if save_path:
        cv2.imwrite(save_path, image)

    return image