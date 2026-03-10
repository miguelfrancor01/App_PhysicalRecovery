import os
import sys
import grpc
from concurrent import futures
import cv2
import numpy as np
import torch

# Asegurar que el sistema encuentre la carpeta src y sus submodulos
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import pose_pb2
import pose_pb2_grpc

# --- POSE_MODULE ---
from pose_module.model_loader import load_models
from pose_module.detector import detect_persons
from pose_module.pose_estimator import estimate_pose

#  módulos necesarios en la raíz de src/
from pose_rating import evaluate_pose, get_final_rating, reset_session, _compute_arm_angle
from preprocessing.frame_preprocessing import procesar_frame_para_modelo

class PoseServicer(pose_pb2_grpc.PoseServiceServicer):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Carga de modelos desde el subpaquete pose_module
        # det_p: processor, det_m: model, pose_p: processor, pose_m: model
        self.det_p, self.det_m, self.pose_p, self.pose_m = load_models(self.device)
        
        print(f"Servidor gRPC iniciado en {self.device} con pose_module")

    def StreamPose(self, request_iterator, context):
        
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
                response = pose_pb2.PoseResponse(frame_id=request.frame_id)

                # Detección usando el módulo reincorporado
                person_boxes = detect_persons(image_pil, self.det_p, self.det_m, self.device)
                
                if len(person_boxes) > 0:
                    # Estimación usando el módulo reincorporado
                    pose_results = estimate_pose(image_pil, person_boxes, self.pose_p, self.pose_m, self.device)
                    
                    # Manejo de la estructura de retorno de pose_estimator
                    person_pose = pose_results[0] if isinstance(pose_results, list) else pose_results
                    keypoints = person_pose["keypoints"]
                    scores = person_pose["scores"]

                    
                    

                    # Llenado de mensaje gRPC
                    person_msg = response.people.add()
                    for i, (kp, sc) in enumerate(zip(keypoints, scores)):
                        kp_item = person_msg.keypoints.add()
                        kp_item.id = i
                        kp_item.x = float(kp[0])
                        kp_item.y = float(kp[1])
                        kp_item.score = float(sc)
                    angle = _compute_arm_angle(keypoints)

                    evaluate_pose(keypoints)
                    stats = get_final_rating()

                    response.current_angle = float(angle)
                    response.repetitions = int(stats["repetitions_detected"])

                    # No enviar score final aún
                    response.final_score = 0.0
                    
                 
                else:
                    stats = get_final_rating()
                    response.repetitions = int(stats["repetitions_detected"])
                    response.final_score = 0.0

                yield response

            except Exception as e:
                print(f"Error en servidor: {e}")
                yield pose_pb2.PoseResponse(frame_id=request.frame_id)
        
        # Cuando termina el stream del video
        final_stats = get_final_rating()

        yield pose_pb2.PoseResponse(
            frame_id=-1,
            repetitions=int(final_stats["repetitions_detected"]),
            final_score=float(final_stats["final_score"])
        )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pose_pb2_grpc.add_PoseServiceServicer_to_server(PoseServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Servidor gRPC escuchando en puerto 50051...")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()