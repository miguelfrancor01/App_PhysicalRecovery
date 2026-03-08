import os
import sys
import streamlit as st
import cv2
import grpc
import numpy as np
import av 

# Configuración de rutas
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import pose_pb2
import pose_pb2_grpc
from pose_module.visualizer import draw_pose

st.set_page_config(page_title="Detección de pose", layout="wide")
st.title("Vit Pose - Detección de pose en video con gRPC")

VIDEO_DIR = "videos"
video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
selected_video = st.sidebar.selectbox("Video:", video_files if video_files else ["Vacio"])
run_btn = st.sidebar.button("Procesar")

placeholder = st.empty()
m1, m2, m3 = st.columns(3)
# Contenedores para evitar duplicados
c1 = m1.empty()
c2 = m2.empty()
c3 = m3.empty()

if run_btn and video_files:
    video_path = os.path.join(VIDEO_DIR, selected_video)
    try:
        channel = grpc.insecure_channel('localhost:50051')
        stub = pose_pb2_grpc.PoseServiceStub(channel)
        
        container = av.open(video_path)
        
        f_id = 0
        for frame in container.decode(video=0):
            # 1. Obtener frame del video
            img = frame.to_ndarray(format='bgr24')
            
            # 2. Preparar envio gRPC
            _, buffer = cv2.imencode('.jpg', img)
            request = pose_pb2.PoseRequest(image_data=buffer.tobytes(), frame_id=f_id)
            
            # 3. Obtener respuesta del servidor
            responses = stub.StreamPose(iter([request]))
            
            for res in responses:
                # 4. DIBUJAR POSE (Igual que en tu main.py)
                if len(res.people) > 0:
                    for person in res.people:
                        # Reconstruir arrays de numpy para el visualizador
                        kpts = np.array([[kp.x, kp.y] for kp in person.keypoints])
                        scores = np.array([kp.score for kp in person.keypoints])
                        
                        # Aplicar dibujo sobre el frame
                        img = draw_pose(img, kpts, scores)

                # 5. Mostrar en pantalla
                placeholder.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                # 6. Actualizar metricas
                c1.metric("Angulo", f"{res.current_angle:.1f} deg")
                c2.metric("Reps", res.repetitions)
                c3.metric("Puntaje", f"{res.final_score:.1f}%")
                break 

            f_id += 1
            
    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        if 'container' in locals(): container.close()