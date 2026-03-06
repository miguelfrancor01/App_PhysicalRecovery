import os
from PIL import Image
import cv2


def load_image(path):
    return Image.open(path).convert("RGB")


def save_image(image, path):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    cv2.imwrite(path, image)