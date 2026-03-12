"""Módulo para evaluar un ejercicio de elevación del brazo izquierdo.

El sistema detecta repeticiones del movimiento y calcula la calificación
final basada en el ángulo máximo alcanzado en cada repetición.

La evaluación final se obtiene al finalizar el video.
"""

import numpy as np


UP_THRESHOLD = 40
DOWN_THRESHOLD = 30
EXPECTED_REPS = 4


in_rep = False
max_angle = 0.0
repetition_angles = []


def _compute_arm_angle(keypoints):
    """Calcula el ángulo del brazo izquierdo.

    Parameters
    ----------
    keypoints : numpy.ndarray
        Keypoints en formato COCO.

    Returns
    -------
    float
        Ángulo del brazo en grados.

    """
    shoulder = keypoints[5][:2]
    wrist = keypoints[9][:2]
    hip = keypoints[11][:2]

    a = np.array(hip)
    b = np.array(shoulder)
    c = np.array(wrist)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)

    angle = np.degrees(np.arccos(cosine))

    return float(angle)


def _angle_to_score(angle):
    """Convierte un ángulo a calificación porcentual.

    90 grados o más equivale a 100 %.

    Parameters
    ----------
    angle : float

    Returns
    -------
    float

    """
    return min((angle / 90.0) * 100.0, 100.0)


def evaluate_pose(keypoints):
    """Procesa un frame del video y actualiza el estado de repeticiones.

    Esta función debe llamarse en cada frame donde exista una pose.

    Parameters
    ----------
    keypoints : numpy.ndarray
        Keypoints de la persona.

    Returns
    -------
    None

    """
    global in_rep
    global max_angle
    global repetition_angles

    angle = _compute_arm_angle(keypoints)

    if not in_rep and angle > UP_THRESHOLD:
        in_rep = True
        max_angle = angle

    elif in_rep:
        max_angle = max(max_angle, angle)

        if angle < DOWN_THRESHOLD:
            if len(repetition_angles) < EXPECTED_REPS:
                repetition_angles.append(max_angle)

            in_rep = False
            max_angle = 0.0
    print("Reps actuales:", repetition_angles)


def get_final_rating():
    """Calcula la calificación final del ejercicio.

    Returns
    -------
    dict
        Información completa de la evaluación.

    """
    if not repetition_angles:
        return {
            "repetitions_detected": 0,
            "angles": [],
            "scores": [],
            "final_score": 0.0,
        }

    scores = [_angle_to_score(a) for a in repetition_angles]

    final_score = float(np.mean(scores))

    return {
        "repetitions_detected": len(repetition_angles),
        "angles": repetition_angles,
        "scores": scores,
        "final_score": final_score,
    }


def reset_session():
    """Reinicia el estado interno del evaluador."""
    global in_rep
    global max_angle
    global repetition_angles

    in_rep = False
    max_angle = 0.0
    repetition_angles = []
