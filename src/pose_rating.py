"""
Módulo para evaluar un ejercicio de elevación del brazo izquierdo.

El sistema detecta repeticiones del movimiento y calcula la calificación
final basada en el ángulo máximo alcanzado en cada repetición.

La evaluación final se obtiene al finalizar el video.
"""

import numpy as np


UP_THRESHOLD = 90
DOWN_THRESHOLD = 70
EXPECTED_REPS = 5


in_rep = False
max_angle = 0.0
repetition_angles = []


def _compute_arm_angle(keypoints):
    """
    Calcula el ángulo del brazo izquierdo.

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

    # Definición de vectores anatómicos
    v_trunk = hip - shoulder   # Vector hacia abajo del cuerpo
    v_arm = wrist - shoulder   # Vector hacia la extremidad

    norm_t = np.linalg.norm(v_trunk)
    norm_a = np.linalg.norm(v_arm)

    if norm_t == 0 or norm_a == 0:
        return 0.0

    # Cálculo del ángulo mediante producto punto
    cosine = np.dot(v_trunk, v_arm) / (norm_t * norm_a)
    cosine = np.clip(cosine, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine))

    return float(angle)


def _angle_to_score(angle):
    """
    Transforma el ángulo máximo alcanzado en un puntaje porcentual.
    Considera 90 grados como el rango de movimiento (ROM) óptimo.
    """
    return min((angle / 90.0) * 100.0, 100.0)


def evaluate_pose(keypoints):
    """
    Procesa un frame del video y actualiza el estado de repeticiones.

    Esta función debe llamarse en cada frame donde exista una pose.

    Parameters
    ----------
    keypoints : numpy.ndarray
        Keypoints de la persona.

    Returns
    -------
    None
    """

    global in_rep, repetition_angles
    angle = _compute_arm_angle(keypoints)

    # 1. DETECTAR SUBIDA (DISPARO INMEDIATO)
    if not in_rep and angle >= UP_THRESHOLD:
        in_rep = True
        # Registramos la repeticion apenas cruza la linea
        if len(repetition_angles) < EXPECTED_REPS:
            repetition_angles.append(angle)
            print(f"DEBUG: Repeticion detectada! Total: {len(repetition_angles)}")
    
    # 2. ACTUALIZAR EL MEJOR ANGULO DE LA REPE ACTUAL
    elif in_rep:
        if angle > repetition_angles[-1]:
            repetition_angles[-1] = angle
        
        # 3. RESETEAR ESTADO PARA LA SIGUIENTE REPE (BAJADA)
        # Solo permitimos una nueva repeticion cuando el brazo baje de 70
        if angle < DOWN_THRESHOLD:
            in_rep = False
            print("DEBUG: Brazo abajo, listo para la siguiente.")

def get_final_rating():
    """
    Calcula la calificación final del ejercicio.

    Returns
    -------
    dict
        Información completa de la evaluación.
    """

    if not repetition_angles:
        return {"repetitions_detected": 0, "final_score": 0.0}

    # Calculamos el score de cada repeticion (maximo 100%)
    scores = [min((a / 90.0) * 100.0, 100.0) for a in repetition_angles]
    return {
        "repetitions_detected": len(repetition_angles),
        "final_score": float(np.mean(scores))
    }


def reset_session():
    """
    Reinicia el estado interno del evaluador.
    """

    global in_rep
    global max_angle
    global repetition_angles

    in_rep = False
    max_angle = 0.0
    repetition_angles = []