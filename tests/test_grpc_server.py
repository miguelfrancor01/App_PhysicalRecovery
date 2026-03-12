import sys
import types

import numpy as np
import pytest


def test_stream_pose_generates_valid_response(monkeypatch):
    """Prueba unitaria para validar el flujo principal de `StreamPose`.

    El objetivo de esta prueba es verificar el comportamiento del servidor
    gRPC cuando recibe un frame válido desde un cliente.

    Para evitar cargar modelos reales de visión por computadora, se crean
    módulos simulados y se registran temporalmente en `sys.modules`. De esta
    forma, cuando `grpc_server` importa dependencias como
    `pose_module.model_loader`, `pose_module.detector` o
    `pose_module.pose_estimator`, Python utiliza estas implementaciones
    ficticias en lugar de las reales.

    Esto permite probar únicamente la lógica del servidor sin ejecutar
    inferencia real ni depender de recursos pesados como GPU.

    La prueba valida que:

    - el frame recibido se procese correctamente,
    - se detecte una persona en la imagen simulada,
    - se estimen keypoints ficticios,
    - se calcule el ángulo del brazo,
    - el servidor genere una respuesta `PoseResponse` consistente.
    """
    # ------------------------------------------------------------------
    # Crear módulos simulados para evitar cargar dependencias reales
    # ------------------------------------------------------------------

    pose_module_pkg = types.ModuleType("pose_module")

    model_loader_mod = types.ModuleType("pose_module.model_loader")
    model_loader_mod.load_models = lambda device: (
        "fake_det_processor",
        "fake_det_model",
        "fake_pose_processor",
        "fake_pose_model",
    )

    detector_mod = types.ModuleType("pose_module.detector")
    detector_mod.detect_persons = lambda image, processor, model, device: np.array(
        [[10, 20, 30, 40]]
    )

    pose_estimator_mod = types.ModuleType("pose_module.pose_estimator")
    pose_estimator_mod.estimate_pose = lambda image, boxes, processor, model, device: {
        "keypoints": [[100, 200], [150, 250]],
        "scores": [0.9, 0.8],
    }

    pose_rating_mod = types.ModuleType("pose_rating")
    pose_rating_mod._compute_arm_angle = lambda keypoints: 95.0

    monkeypatch.setitem(sys.modules, "pose_module", pose_module_pkg)
    monkeypatch.setitem(sys.modules, "pose_module.model_loader", model_loader_mod)
    monkeypatch.setitem(sys.modules, "pose_module.detector", detector_mod)
    monkeypatch.setitem(sys.modules, "pose_module.pose_estimator", pose_estimator_mod)
    monkeypatch.setitem(sys.modules, "pose_rating", pose_rating_mod)

    # ------------------------------------------------------------------
    # # Volver a cargar el módulo grpc_server para que utilice los módulos
    # simulados definidos anteriormente en sys.modules.
    # ------------------------------------------------------------------

    sys.modules.pop("grpc_server", None)
    from grpc_server import PoseServicer

    # ------------------------------------------------------------------
    # Mock de funciones internas del servidor
    # ------------------------------------------------------------------

    monkeypatch.setattr(
        "grpc_server.procesar_frame_para_modelo",
        lambda frame, frame_index, redimensionar, ancho_objetivo: {
            "valid": True,
            "image_pil": "fake_image",
        },
    )

    monkeypatch.setattr(
        "grpc_server.detect_persons",
        lambda image, processor, model, device: np.array([[10, 20, 30, 40]]),
    )

    monkeypatch.setattr(
        "grpc_server.estimate_pose",
        lambda image, boxes, processor, model, device: {
            "keypoints": [[100, 200], [150, 250]],
            "scores": [0.9, 0.8],
        },
    )

    monkeypatch.setattr(
        "grpc_server._compute_arm_angle",
        lambda keypoints: 95.0,
    )

    monkeypatch.setattr(
        "grpc_server.np.frombuffer",
        lambda data, dtype: np.array([1, 2, 3], dtype=np.uint8),
    )

    monkeypatch.setattr(
        "grpc_server.cv2.imdecode",
        lambda arr, flag: np.zeros((10, 10, 3), dtype=np.uint8),
    )

    # ------------------------------------------------------------------
    # Crear instancia del servidor gRPC
    # ------------------------------------------------------------------

    servicer = PoseServicer()

    class FakeRequest:
        """Simula un mensaje `PoseRequest` recibido por el servidor gRPC."""

        def __init__(self):
            self.image_data = b"fake_image_bytes"
            self.frame_id = 1

    requests = [FakeRequest()]

    # Ejecutar el método StreamPose
    responses = list(servicer.StreamPose(iter(requests), None))

    # ------------------------------------------------------------------
    # Validaciones
    # ------------------------------------------------------------------

    assert len(responses) == 1

    response = responses[0]

    # Verificar ID del frame
    assert response.frame_id == 1

    # Verificar ángulo calculado
    assert response.current_angle == 95.0

    # Verificar que se detectó una persona
    assert len(response.people) == 1

    person = response.people[0]

    # Deben existir dos keypoints simulados
    assert len(person.keypoints) == 2

    kp0 = person.keypoints[0]

    # Validar valores del primer keypoint
    assert kp0.id == 0
    assert kp0.x == 100
    assert kp0.y == 200
    assert kp0.score == pytest.approx(0.9)
