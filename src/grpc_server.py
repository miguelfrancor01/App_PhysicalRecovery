"""
Servidor gRPC para el procesamiento de Pose en tiempo real.
Este módulo actúa como el orquestador central, recibiendo imágenes de la App,
ejecutando los modelos de IA y devolviendo métricas de rehabilitación.
"""

import os
import sys
import grpc
from concurrent import futures
import cv2
import numpy as np
import torch

# Asegurar rutas
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import pose_pb2
import pose_pb2_grpc
from pose_module.model_loader import load_models
from pose_module.detector import detect_persons
from pose_module.pose_estimator import estimate_pose
from pose_rating import evaluate_pose, get_final_rating, reset_session, _compute_arm_angle
from preprocessing.frame_preprocessing import procesar_frame_para_modelo

class PoseServicer(pose_pb2_grpc.PoseServiceServicer):
    """
    Implementación de los servicios definidos en el archivo proto.
    Maneja la carga de modelos y la lógica de inferencia por cada frame recibido.
    """

    def __init__(self):
        """
        Inicializa el servidor configurando el dispositivo (CPU/GPU) y 
        cargando los modelos de detección y estimación en memoria.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.det_p, self.det_m, self.pose_p, self.pose_m = load_models(self.device)
        print(f"Servidor gRPC iniciado en {self.device}")

    def StreamPose(self, request_iterator, context):
        """
        Maneja un flujo bidireccional de datos de pose.

        Args:
            request_iterator (iterator): Iterador que entrega objetos PoseRequest 
                                         con datos de imagen.
            context (grpc.ServicerContext): Contexto de la llamada gRPC.

        Yields:
            pose_pb2.PoseResponse: Objeto que contiene puntos clave, repeticiones, 
                                   ángulo actual y puntaje de la sesión.
        """
        # Reiniciar las variables globales de pose_rating al iniciar un nuevo stream
        reset_session()
        
        for request in request_iterator:
            try:
                # Decodificación de la imagen recibida
                nparr = np.frombuffer(request.image_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None: continue

                # Preprocesamiento de la imagen para los modelos de IA
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

                # Detección de personas en la escena
                person_boxes = detect_persons(image_pil, self.det_p, self.det_m, self.device)
                
                if len(person_boxes) > 0:
                    # Estimación de puntos clave (Pose Estimation)
                    pose_results = estimate_pose(image_pil, person_boxes, self.pose_p, self.pose_m, self.device)
                    person_pose = pose_results[0]
                    keypoints = person_pose["keypoints"]
                    scores = person_pose["scores"]

                    # 1. Evaluación biomecánica (Lógica de repeticiones y score)
                    evaluate_pose(keypoints)
                    
                    # 2. Obtención de estadísticas acumuladas
                    stats = get_final_rating()

                    # 3. Empaquetado de la respuesta gRPC
                    person_msg = response.people.add()
                    for i, (kp, sc) in enumerate(zip(keypoints, scores)):
                        person_msg.keypoints.add(id=i, x=float(kp[0]), y=float(kp[1]), score=float(sc))
                    
                    response.current_angle = float(_compute_arm_angle(keypoints))
                    response.repetitions = int(stats["repetitions_detected"])
                    response.final_score = float(stats["final_score"])
                else:
                    # Si no hay nadie detectado, se mantienen las estadísticas previas
                    stats = get_final_rating()
                    response.repetitions = int(stats["repetitions_detected"])
                    response.final_score = float(stats["final_score"])

                yield response
            except Exception as e:
                print(f"Error servidor: {e}")
                yield pose_pb2.PoseResponse(frame_id=request.frame_id)

def serve():
    """
    Configura e inicia el servidor gRPC en el puerto 50051.
    Utiliza un ThreadPoolExecutor para manejar múltiples peticiones concurrentes.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pose_pb2_grpc.add_PoseServiceServicer_to_server(PoseServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Servidor gRPC escuchando en el puerto 50051...")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()