import sys
import types
import numpy as np
import pytest

# -------------------------------------------------
# Teniendo en cuenta que en las pruebas unitarias no se quieren cargar los módulos reales porque
# estos inicializan modelos de visión por computadora, se crearon
# módulos simulados usando types.ModuleType y se registran en sys.modules.
#
# De esta forma, cuando src.grpc_server realiza esos imports, Python
# utiliza estas implementaciones ficticias, permitiendo probar únicamente
# la lógica del servidor sin ejecutar inferencia real.
# -------------------------------------------------
# -------------------------------------------------
pose_module_pkg = types.ModuleType("pose_module")
sys.modules["pose_module"] = pose_module_pkg

model_loader_mod = types.ModuleType("pose_module.model_loader")
model_loader_mod.load_models = lambda device: (
    "fake_det_processor",
    "fake_det_model",
    "fake_pose_processor",
    "fake_pose_model",
)
sys.modules["pose_module.model_loader"] = model_loader_mod

detector_mod = types.ModuleType("pose_module.detector")
detector_mod.detect_persons = lambda image, processor, model, device: np.array([[10, 20, 30, 40]])
sys.modules["pose_module.detector"] = detector_mod

pose_estimator_mod = types.ModuleType("pose_module.pose_estimator")
pose_estimator_mod.estimate_pose = lambda image, boxes, processor, model, device: {
    "keypoints": [[100, 200], [150, 250]],
    "scores": [0.9, 0.8]
}
sys.modules["pose_module.pose_estimator"] = pose_estimator_mod

pose_rating_mod = types.ModuleType("pose_rating")
pose_rating_mod._compute_arm_angle = lambda keypoints: 95.0
sys.modules["pose_rating"] = pose_rating_mod

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
    # Mock del cálculo de ángulo
    # -----------------------------
    monkeypatch.setattr(
        "src.grpc_server._compute_arm_angle",
        lambda keypoints: 95.0
    )

    # -----------------------------
    # Mock de decodificación de imagen
    # -----------------------------
    monkeypatch.setattr(
        "src.grpc_server.np.frombuffer",
        lambda data, dtype: np.array([1, 2, 3], dtype=np.uint8)
    )

    monkeypatch.setattr(
        "src.grpc_server.cv2.imdecode",
        lambda arr, flag: np.zeros((10, 10, 3), dtype=np.uint8)
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

    # Verificar ángulo calculado
    assert response.current_angle == 95.0

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