import os
import sys
import grpc
from concurrent import futures
import cv2
import numpy as np
from PIL import Image
import torch

# Asegurar rutas de importacion
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, ".."))

import pose_pb2
import pose_pb2_grpc
from pose_module.model_loader import load_models
from pose_module.detector import detect_persons
from pose_module.pose_estimator import estimate_pose
from pose_rating import evaluate_pose, get_final_rating, reset_session, _compute_arm_angle

class PoseServicer(pose_pb2_grpc.PoseServiceServicer):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Carga identica a tu main.py
        self.det_p, self.det_m, self.pose_p, self.pose_m = load_models(self.device)
        print(f"Modelos cargados en {self.device}")

    def StreamPose(self, request_iterator, context):
        reset_session()
        for request in request_iterator:
            try:
                # 1. Decodificacion de imagen
                nparr = np.frombuffer(request.image_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None: continue

                # 2. Logica exacta de tu main.py (BGR -> RGB -> PIL)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(image_rgb)

                # 3. Deteccion
                person_boxes = detect_persons(image_pil, self.det_p, self.det_m, self.device)
                
                response = pose_pb2.PoseResponse(frame_id=request.frame_id)

                if len(person_boxes) > 0:
                    # 4. Estimacion de Pose
                    pose_results = estimate_pose(image_pil, person_boxes, self.pose_p, self.pose_m, self.device)
                    
                    # Usamos el primer resultado (asumiendo una persona como en tu main)
                    person_pose = pose_results[0]
                    keypoints = person_pose["keypoints"]
                    scores = person_pose["scores"]

                    # 5. Calificacion (Llamada a pose_rating)
                    evaluate_pose(keypoints)
                    stats = get_final_rating()

                    # 6. Empaquetar respuesta para gRPC
                    person_msg = response.people.add()
                    for i, (kp, sc) in enumerate(zip(keypoints, scores)):
                        # Los keypoints vienen como tensores o arrays, convertimos a float
                        person_msg.keypoints.add(
                            id=i, 
                            x=float(kp[0]), 
                            y=float(kp[1]), 
                            score=float(sc)
                        )
                    
                    # Seteo de metricas reales
                    response.current_angle = float(_compute_arm_angle(keypoints))
                    response.repetitions = int(stats["repetitions_detected"])
                    response.final_score = float(stats["final_score"])

                yield response
            except Exception as e:
                print(f"Error en servidor: {e}")
                yield pose_pb2.PoseResponse(frame_id=request.frame_id)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pose_pb2_grpc.add_PoseServiceServicer_to_server(PoseServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Servidor gRPC escuchando...")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()