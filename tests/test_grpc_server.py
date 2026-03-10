import numpy as np
import pytest
from src.grpc_server import PoseServicer


def test_stream_pose_generates_valid_response(monkeypatch):
    """
    Test unitario para el método StreamPose del servidor gRPC.

    Objetivo
    --------
    Verificar que el servidor procese correctamente un frame recibido y genere
    una respuesta PoseResponse válida cuando se detecta una persona en la imagen.

    Este test valida el flujo principal del servidor sin ejecutar modelos reales
    de visión por computadora. Para ello se reemplazan (mockean) las funciones
    externas mediante monkeypatch.

    Flujo validado
    --------------
    1. Decodificación del frame recibido.
    2. Preprocesamiento del frame.
    3. Detección de personas.
    4. Estimación de pose.
    5. Evaluación del ejercicio.
    6. Construcción del PoseResponse con:
        - keypoints
        - ángulo actual
        - número de repeticiones
        - puntuación final
    """

    # -----------------------------
    # Mock del preprocesamiento
    # -----------------------------
    monkeypatch.setattr(
        "src.grpc_server.procesar_frame_para_modelo",
        lambda frame, frame_index, redimensionar, ancho_objetivo: {
            "valid": True,
            "image_pil": "fake_image"
        }
    )

    # -----------------------------
    # Mock del detector de personas
    # -----------------------------
    monkeypatch.setattr(
        "src.grpc_server.detect_persons",
        lambda image, processor, model, device: np.array([[10, 20, 30, 40]])
    )

    # -----------------------------
    # Mock del estimador de pose
    # -----------------------------
    monkeypatch.setattr(
        "src.grpc_server.estimate_pose",
        lambda image, boxes, processor, model, device: {
            "keypoints": [[100, 200], [150, 250]],
            "scores": [0.9, 0.8]
        }
    )

    # -----------------------------
    # Mock de funciones de evaluación
    # -----------------------------
    monkeypatch.setattr("src.grpc_server.reset_session", lambda: None)
    monkeypatch.setattr("src.grpc_server.evaluate_pose", lambda k: None)

    monkeypatch.setattr(
        "src.grpc_server.get_final_rating",
        lambda: {"repetitions_detected": 2, "final_score": 80}
    )

    monkeypatch.setattr(
        "src.grpc_server._compute_arm_angle",
        lambda keypoints: 95.0
    )

    # -----------------------------
    # Mock de decodificación de imagen
    # -----------------------------
    monkeypatch.setattr(
        "src.grpc_server.np.frombuffer",
        lambda data, dtype: np.array([1, 2, 3])
    )

    monkeypatch.setattr(
        "src.grpc_server.cv2.imdecode",
        lambda arr, flag: np.zeros((10, 10, 3))
    )

    # -----------------------------
    # Crear instancia del servicer
    # -----------------------------
    servicer = PoseServicer()

    # -----------------------------
    # Simular request gRPC
    # -----------------------------
    class FakeRequest:
        """
        Objeto que simula un mensaje PoseRequest recibido por el servidor.
        """
        def __init__(self):
            self.image_data = b"fake_image_bytes"
            self.frame_id = 1

    requests = [FakeRequest()]

    # Ejecutar el método del servidor
    responses = list(servicer.StreamPose(iter(requests), None))

    # -----------------------------
    # Validaciones
    # -----------------------------
    assert len(responses) == 1

    response = responses[0]

    # Verificar ID del frame
    assert response.frame_id == 1

    # Verificar métricas calculadas
    assert response.current_angle == 95.0
    assert response.repetitions == 2
    assert response.final_score == 80

    # Verificar estructura de personas detectadas
    assert len(response.people) == 1

    person = response.people[0]

    # Debe haber dos keypoints simulados
    assert len(person.keypoints) == 2

    # Validar valores del primer keypoint
    kp0 = person.keypoints[0]

    assert kp0.id == 0
    assert kp0.x == 100
    assert kp0.y == 200
    assert kp0.score == pytest.approx(0.9)