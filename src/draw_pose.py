import cv2


# -------------------------------
# BODY PART GROUPS (keypoints)
# -------------------------------

HEAD = [0, 1, 2, 3, 4]
ARMS = [5, 6, 7, 8, 9, 10]
LEGS = [11, 12, 13, 14, 15, 16]
TORSO = [5, 6, 11, 12]


# -------------------------------
# SKELETON CONNECTIONS
# -------------------------------

SKELETON_CONNECTIONS = {

    "head": [
        (0, 1), (0, 2), (1, 3), (2, 4)
    ],

    "arms": [
        (5, 7), (7, 9),
        (6, 8), (8, 10)
    ],

    "torso": [
        (5, 6),
        (5, 11),
        (6, 12)
    ],

    "legs": [
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16)
    ]
}


# -------------------------------
# COLORS (BGR for OpenCV)
# -------------------------------

COLORS = {

    "head": (0, 255, 255),     # yellow
    "arms": (255, 0, 0),       # blue
    "torso": (0, 165, 255),    # orange
    "legs": (0, 255, 0),       # green
}


# -------------------------------
# DRAW FUNCTION
# -------------------------------

def draw_pose(image, keypoints, scores, threshold=0.3):

    # Draw keypoints
    for i, (kp, score) in enumerate(zip(keypoints, scores)):

        if score < threshold:
            continue

        x, y = int(kp[0]), int(kp[1])

        # Determine color by body group
        if i in HEAD:
            color = COLORS["head"]

        elif i in ARMS:
            color = COLORS["arms"]

        elif i in LEGS:
            color = COLORS["legs"]

        else:
            color = COLORS["torso"]

        cv2.circle(image, (x, y), 4, color, -1)

    # Draw skeleton connections
    for group, connections in SKELETON_CONNECTIONS.items():

        color = COLORS[group]

        for a, b in connections:

            if scores[a] < threshold or scores[b] < threshold:
                continue

            x1, y1 = int(keypoints[a][0]), int(keypoints[a][1])
            x2, y2 = int(keypoints[b][0]), int(keypoints[b][1])

            cv2.line(image, (x1, y1), (x2, y2), color, 2)

    return image