import cv2


def open_camera(index=0):

    cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    return cap