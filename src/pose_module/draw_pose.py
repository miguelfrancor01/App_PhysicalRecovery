import cv2


# -------------------------------
# SKELETON CONNECTIONS (COCO-17 reference)
# -------------------------------

BASE_SKELETON = {

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
# COLORS (BGR OpenCV)
# -------------------------------

COLORS = {

    "head": (0, 255, 255),   # yellow
    "arms": (255, 0, 0),     # blue
    "torso": (0, 165, 255),  # orange
    "legs": (0, 255, 0),     # green
}


# -------------------------------
# JOINT GROUP DETECTION
# -------------------------------

def get_body_groups(num_joints):

    groups = {
        "head": [i for i in range(0, min(5, num_joints))],
        "arms": [i for i in range(5, min(11, num_joints))],
        "torso": [i for i in range(5, min(13, num_joints))],
        "legs": [i for i in range(11, min(17, num_joints))]
    }

    return groups


# -------------------------------
# FILTER VALID CONNECTIONS
# -------------------------------

def get_valid_connections(num_joints):

    valid = {}

    for group, connections in BASE_SKELETON.items():

        valid[group] = []

        for a, b in connections:

            if a < num_joints and b < num_joints:
                valid[group].append((a, b))

    return valid


# -------------------------------
# DRAW FUNCTION
# -------------------------------

def draw_pose(image, keypoints, scores, threshold=0.3):

    num_joints = len(keypoints)

    body_groups = get_body_groups(num_joints)
    skeleton = get_valid_connections(num_joints)

    # -------------------------------
    # Draw keypoints
    # -------------------------------

    for i, (kp, score) in enumerate(zip(keypoints, scores)):

        if score < threshold:
            continue

        x, y = int(kp[0]), int(kp[1])

        color = COLORS["torso"]

        for group, joints in body_groups.items():
            if i in joints:
                color = COLORS[group]
                break

        cv2.circle(image, (x, y), 4, color, -1)

    # -------------------------------
    # Draw skeleton connections
    # -------------------------------

    for group, connections in skeleton.items():

        color = COLORS[group]

        for a, b in connections:

            if scores[a] < threshold or scores[b] < threshold:
                continue

            x1, y1 = int(keypoints[a][0]), int(keypoints[a][1])
            x2, y2 = int(keypoints[b][0]), int(keypoints[b][1])

            cv2.line(image, (x1, y1), (x2, y2), color, 2)

    return image