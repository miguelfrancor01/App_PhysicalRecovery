"""
Módulo para renderizar keypoints y esqueleto humano sobre una imagen
utilizando el formato COCO de 17 puntos.

El dibujo se realiza usando OpenCV y se colorean las distintas partes
del cuerpo para facilitar la interpretación visual.
"""

import cv2


# ---------------------------------------------------------------------
# Conexiones del esqueleto (referencia COCO-17)
# ---------------------------------------------------------------------

BASE_SKELETON = {
    "head": [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
    ],
    "arms": [
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
    ],
    "torso": [
        (5, 6),
        (5, 11),
        (6, 12),
    ],
    "legs": [
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ],
}


# ---------------------------------------------------------------------
# Colores en formato BGR (OpenCV)
# ---------------------------------------------------------------------

COLORS = {
    "head": (0, 255, 255),     # Amarillo
    "arms": (255, 0, 0),       # Azul
    "torso": (0, 165, 255),    # Naranja
    "legs": (0, 255, 0),       # Verde
}


# ---------------------------------------------------------------------
# Identificación de grupos corporales
# ---------------------------------------------------------------------

def get_body_groups(num_joints):
    """
    Determina qué articulaciones pertenecen a cada grupo corporal.

    Parameters
    ----------
    num_joints : int
        Número total de keypoints detectados.

    Returns
    -------
    dict
        Diccionario con listas de índices de articulaciones
        para cada grupo corporal.
    """

    groups = {
        "head": [i for i in range(0, min(5, num_joints))],
        "arms": [i for i in range(5, min(11, num_joints))],
        "torso": [i for i in range(5, min(13, num_joints))],
        "legs": [i for i in range(11, min(17, num_joints))],
    }

    return groups


# ---------------------------------------------------------------------
# Filtrar conexiones válidas del esqueleto
# ---------------------------------------------------------------------

def get_valid_connections(num_joints):
    """
    Filtra las conexiones del esqueleto que pueden dibujarse
    según el número de articulaciones disponibles.

    Parameters
    ----------
    num_joints : int
        Número de keypoints detectados.

    Returns
    -------
    dict
        Diccionario con las conexiones válidas por grupo corporal.
    """

    valid = {}

    for group, connections in BASE_SKELETON.items():
        valid[group] = []

        for a, b in connections:
            if a < num_joints and b < num_joints:
                valid[group].append((a, b))

    return valid


# ---------------------------------------------------------------------
# Renderizado de pose
# ---------------------------------------------------------------------

def draw_pose(image, keypoints, scores, threshold=0.3):
    """
    Dibuja los keypoints y el esqueleto humano sobre una imagen.

    Parameters
    ----------
    image : numpy.ndarray
        Imagen en formato OpenCV (BGR).
    keypoints : array-like
        Coordenadas de los keypoints con forma (N, 2).
    scores : array-like
        Nivel de confianza asociado a cada keypoint.
    threshold : float, optional
        Umbral mínimo de confianza para visualizar un keypoint.
        El valor por defecto es 0.3.

    Returns
    -------
    numpy.ndarray
        Imagen con los keypoints y conexiones dibujadas.
    """

    num_joints = len(keypoints)

    body_groups = get_body_groups(num_joints)
    skeleton = get_valid_connections(num_joints)

    # -------------------------------------------------------------
    # Dibujar keypoints
    # -------------------------------------------------------------

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

    # -------------------------------------------------------------
    # Dibujar conexiones del esqueleto
    # -------------------------------------------------------------

    for group, connections in skeleton.items():

        color = COLORS[group]

        for a, b in connections:

            if scores[a] < threshold or scores[b] < threshold:
                continue

            x1, y1 = int(keypoints[a][0]), int(keypoints[a][1])
            x2, y2 = int(keypoints[b][0]), int(keypoints[b][1])

            cv2.line(image, (x1, y1), (x2, y2), color, 2)

    return image