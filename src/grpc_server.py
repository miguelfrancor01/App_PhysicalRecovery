import os
import sys
from concurrent import futures

import cv2
import grpc
import numpy as np
import torch


current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import pose_pb2
import pose_pb2_grpc
from pose_rating import _compute_arm_angle
from preprocessing.frame_preprocessing import procesar_frame_para_modelo

from pose_module.detector import detect_persons
from pose_module.model_loader import load_models
from pose_module.pose_estimator import estimate_pose


class PoseServicer(pose_pb2_grpc.PoseServiceServicer):
    """Implementación del servicio gRPC para estimación de pose.

    Este servicio recibe frames codificados desde el cliente, los decodifica,
    ejecuta el pipeline de preprocesamiento, detección de personas y
    estimación de pose, y devuelve una respuesta con los keypoints detectados
    y el ángulo actual calculado.

    La lógica de evaluación de repeticiones no se realiza en este servidor;
    esa responsabilidad pertenece al cliente o a otro módulo externo.
    """

    def __init__(self):
        """Inicializa el servicio cargando los modelos de detección y pose
        en el dispositivo disponible (CPU o GPU).
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.det_p, self.det_m, self.pose_p, self.pose_m = load_models(self.device)
        print(f"Servidor gRPC iniciado en {self.device} con pose_module")

    def StreamPose(self, request_iterator, context):
        """Procesa un flujo de frames entrantes y devuelve respuestas de pose.

        Por cada frame recibido, el servidor:

        1. Decodifica la imagen enviada por el cliente.
        2. Ejecuta el preprocesamiento del frame.
        3. Detecta personas en la imagen.
        4. Estima los keypoints corporales.
        5. Calcula el ángulo actual del brazo.
        6. Devuelve una respuesta `PoseResponse` con los resultados.

        Args:
        ----
            request_iterator: Iterador de mensajes `PoseRequest`.
            context: Contexto de la llamada gRPC.

        Yields:
        ------
            pose_pb2.PoseResponse: Respuesta con el ID del frame, keypoints
            detectados y ángulo actual. Si ocurre un error o el frame no es
            válido, se retorna una respuesta vacía con el `frame_id`.

        """
        # El servidor no llama evaluate_pose() ni reset_session().
        # Su única responsabilidad es:
        # 1. Detectar personas en el frame.
        # 2. Estimar keypoints.
        # 3. Calcular el ángulo actual.
        # 4. Devolver esos resultados a app.py.
        #
        # El conteo de repeticiones lo maneja app.py con pose_rating local.

        for request in request_iterator:
            try:
                nparr = np.frombuffer(request.image_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    yield pose_pb2.PoseResponse(frame_id=request.frame_id)
                    continue

                resultado_pre = procesar_frame_para_modelo(
                    frame,
                    frame_index=request.frame_id,
                    redimensionar=True,
                    ancho_objetivo=640,
                )

                if not resultado_pre["valid"]:
                    yield pose_pb2.PoseResponse(frame_id=request.frame_id)
                    continue

                image_pil = resultado_pre["image_pil"]
                response = pose_pb2.PoseResponse(frame_id=request.frame_id)

                person_boxes = detect_persons(
                    image_pil,
                    self.det_p,
                    self.det_m,
                    self.device,
                )

                if len(person_boxes) > 0:
                    pose_results = estimate_pose(
                        image_pil,
                        person_boxes,
                        self.pose_p,
                        self.pose_m,
                        self.device,
                    )

                    person_pose = (
                        pose_results[0]
                        if isinstance(pose_results, list)
                        else pose_results
                    )
                    keypoints = person_pose["keypoints"]
                    scores = person_pose["scores"]

                    # Llenar keypoints en el mensaje de respuesta.
                    person_msg = response.people.add()
                    for i, (kp, sc) in enumerate(zip(keypoints, scores, strict=False)):
                        kp_item = person_msg.keypoints.add()
                        kp_item.id = i
                        kp_item.x = float(kp[0])
                        kp_item.y = float(kp[1])
                        kp_item.score = float(sc)

                    # Calcular el ángulo actual para mostrarlo en la UI.
                    response.current_angle = float(_compute_arm_angle(keypoints))

                yield response

            except Exception as e:
                print(f"Error en servidor frame {request.frame_id}: {e}")
                yield pose_pb2.PoseResponse(frame_id=request.frame_id)


def serve():
    """Inicia el servidor gRPC y lo deja escuchando en el puerto 50051."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pose_pb2_grpc.add_PoseServiceServicer_to_server(PoseServicer(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("Servidor gRPC escuchando en puerto 50051...")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
