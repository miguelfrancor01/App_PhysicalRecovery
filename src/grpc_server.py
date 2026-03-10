import os
import sys
import grpc
from concurrent import futures
import cv2
import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import pose_pb2
import pose_pb2_grpc

from pose_module.model_loader import load_models
from pose_module.detector import detect_persons
from pose_module.pose_estimator import estimate_pose

from pose_rating import _compute_arm_angle
from preprocessing.frame_preprocessing import procesar_frame_para_modelo


class PoseServicer(pose_pb2_grpc.PoseServiceServicer):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.det_p, self.det_m, self.pose_p, self.pose_m = load_models(self.device)
        print(f"Servidor gRPC iniciado en {self.device} con pose_module")

    def StreamPose(self, request_iterator, context):
        # El servidor YA NO llama evaluate_pose() ni reset_session()
        # Su única responsabilidad es:
        #   1. Detectar personas en el frame
        #   2. Estimar keypoints
        #   3. Calcular el ángulo actual
        #   4. Devolver todo eso a app.py
        # El conteo de repeticiones lo maneja app.py con pose_rating local

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
                    ancho_objetivo=640
                )

                if not resultado_pre["valid"]:
                    yield pose_pb2.PoseResponse(frame_id=request.frame_id)
                    continue

                image_pil = resultado_pre["image_pil"]
                response  = pose_pb2.PoseResponse(frame_id=request.frame_id)

                person_boxes = detect_persons(image_pil, self.det_p, self.det_m, self.device)

                if len(person_boxes) > 0:
                    pose_results = estimate_pose(image_pil, person_boxes, self.pose_p, self.pose_m, self.device)

                    person_pose = pose_results[0] if isinstance(pose_results, list) else pose_results
                    keypoints   = person_pose["keypoints"]
                    scores      = person_pose["scores"]

                    # Llenar keypoints en el mensaje
                    person_msg = response.people.add()
                    for i, (kp, sc) in enumerate(zip(keypoints, scores)):
                        kp_item       = person_msg.keypoints.add()
                        kp_item.id    = i
                        kp_item.x     = float(kp[0])
                        kp_item.y     = float(kp[1])
                        kp_item.score = float(sc)

                    # Solo calcular el ángulo para mostrarlo en la UI
                    response.current_angle = float(_compute_arm_angle(keypoints))

                yield response

            except Exception as e:
                print(f"Error en servidor frame {request.frame_id}: {e}")
                yield pose_pb2.PoseResponse(frame_id=request.frame_id)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pose_pb2_grpc.add_PoseServiceServicer_to_server(PoseServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Servidor gRPC escuchando en puerto 50051...")
    server.wait_for_termination()


if __name__ == '__main__':
    serve()