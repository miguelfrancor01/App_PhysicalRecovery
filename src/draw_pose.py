import cv2


HEAD = (0, 1, 2, 3, 4)
ARMS = (5, 6, 7, 8, 9, 10)
LEGS = (11, 12, 13, 14, 15, 16)

SKELETON = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6),
    (5, 11), (6, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

COLORS = {
    "head": (0, 255, 255),
    "arms": (255, 0, 0),
    "legs": (0, 255, 0),
}


def get_color(index):

    if index in HEAD:
        return COLORS["head"]

    if index in ARMS:
        return COLORS["arms"]

    if index in LEGS:
        return COLORS["legs"]

    return (255, 255, 255)


def draw_pose(image, keypoints, scores, threshold=0.3):

    for i, (kp, score) in enumerate(zip(keypoints, scores)):

        if score < threshold:
            continue

        x, y = int(kp[0]), int(kp[1])

        color = get_color(i)

        cv2.circle(image, (x, y), 4, color, -1)

    for a, b in SKELETON:

        if scores[a] < threshold or scores[b] < threshold:
            continue

        x1, y1 = int(keypoints[a][0]), int(keypoints[a][1])
        x2, y2 = int(keypoints[b][0]), int(keypoints[b][1])

        cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 2)

    return image