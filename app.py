"""
Módulo de Interfaz de Usuario - Physical Recovery AI.

Este script gestiona la experiencia del usuario mediante Streamlit, permitiendo
la selección de ejercicios, la visualización del procesamiento de pose en 
tiempo real a través de gRPC y la generación de reportes clínicos en PDF.
"""

import streamlit as st
import cv2
import grpc
import numpy as np
import av
import os
import sys
import time

# Rutas de modulos
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path: sys.path.insert(0, src_path)

import pose_pb2
import pose_pb2_grpc
from pose_module.visualizer import draw_pose
from report_module.session_data import SessionData
from report_module.report_generator import generate_report

# --- CONFIGURACION DE PAGINA ---
st.set_page_config(page_title="App Physical Recovery", layout="wide")

# Inicialización de variables de estado de sesión para persistencia
if 'fase' not in st.session_state: st.session_state.fase = 'config'
if 'stats' not in st.session_state: st.session_state.stats = {"reps": 0, "score": 0.0}
if 'pdf_path' not in st.session_state: st.session_state.pdf_path = None

st.title("App Physical Recovery")
st.markdown("### Monitoreo de rehabilitación con VitaPose y gRPC")

# --- BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    """
    Configuración del panel lateral.
    Permite al usuario seleccionar el video de entrada y descargar el reporte final.
    """
    st.header("Panel de control")
    ejercicio = st.selectbox("Ejercicio:", ["Elevación brazo izquierdo"])
    
    VIDEO_DIR = "videos"
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(('.mp4', '.avi'))]
    selected_video = st.selectbox("Archivo de video:", video_files if video_files else ["N/A"])
    
    st.divider()
    
    # Manejo de Reporte PDF: Se activa solo cuando el archivo ha sido generado
    if st.session_state.fase == 'finalizado' and st.session_state.pdf_path:
        st.subheader("Reporte generado")
        with open(st.session_state.pdf_path, "rb") as f:
            st.download_button(
                label="Descargar reporte PDF",
                data=f,
                file_name=os.path.basename(st.session_state.pdf_path),
                mime="application/pdf",
                use_container_width=True
            )

    if st.button("Reiniciar aplicativo", use_container_width=True):
        st.session_state.fase = 'config'
        st.session_state.stats = {"reps": 0, "score": 0.0}
        st.session_state.pdf_path = None
        st.rerun() 

# --- FLUJO DE PANTALLAS ---

if st.session_state.fase == 'config':
    """
    Fase de configuración inicial.
    Estado de espera donde el usuario confirma la selección del video.
    """
    st.info("Seleccione su video y presione 'Comenzar evaluación' para iniciar el procesamiento. El vídeo debe posicionarse en ubicación sagital y empezar a levantar su brazo izquierdo.")
    
    if selected_video != "N/A":
        if st.button("Comenzar evaluación", type="primary", use_container_width=True):
            st.session_state.fase = 'procesando'
            st.rerun()

elif st.session_state.fase == 'procesando':
    """
    Fase de procesamiento en tiempo real.
    Establece conexión gRPC, decodifica el video y actualiza las métricas en pantalla.
    """
    col_vid, col_met = st.columns([2, 1])
    placeholder = col_vid.empty() # Espacio para el video procesado
    
    with col_met:
        st.subheader("Métricas de desempeño")
        m_reps = st.empty()
        m_angle = st.empty()
        m_score = st.empty()
        st.divider()
        stop_btn = st.button("Detener sesión", type="primary", use_container_width=True)

    try:
        # Inicio de métricas de rendimiento
        start_time = time.time()
        
        # Conexión al servidor gRPC
        channel = grpc.insecure_channel('localhost:50051')
        stub = pose_pb2_grpc.PoseServiceStub(channel)
        
        # Apertura del contenedor de video mediante PyAV
        container = av.open(os.path.join(VIDEO_DIR, selected_video))
        
        frames_count = 0
        for frame in container.decode(video=0):
            frames_count += 1
            img_raw = frame.to_ndarray(format='bgr24')
            img_proc = cv2.resize(img_raw, (640, 360)) # Normalización de tamaño
            
            # Codificación de imagen para envío gRPC
            _, buffer = cv2.imencode('.jpg', img_proc)
            request = pose_pb2.PoseRequest(image_data=buffer.tobytes(), frame_id=frames_count)
            
            # Consumo del servicio StreamPose
            responses = stub.StreamPose(iter([request]))
            
            for res in responses:
                # Dibujado de esqueleto si se detectan personas
                if len(res.people) > 0:
                    for p in res.people:
                        kpts = np.array([[kp.x, kp.y] for kp in p.keypoints])
                        img_proc = draw_pose(img_proc, kpts, np.array([kp.score for kp in p.keypoints]))

                # Actualización de la interfaz
                placeholder.image(cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB))
                
                # Sincronización de estados y métricas
                st.session_state.stats.update({"reps": res.repetitions, "score": res.final_score})
                m_reps.metric("Repeticiones", res.repetitions)
                m_angle.metric("Ángulo actual", f"{res.current_angle:.1f}°")
                m_score.metric("Puntaje", f"{res.final_score:.1f}%")
                
                # Condición de parada automática por meta cumplida
                if res.repetitions >= 5: st.session_state.fase = 'finalizado'
                break
            
            # Salida manual o por cumplimiento de reps
            if stop_btn or st.session_state.fase == 'finalizado': break
        
        # Cálculo de estadísticas finales de la sesión
        duration = time.time() - start_time
        container.close()
        
        # --- GENERACIÓN DE REPORTE PDF ---
        # Creación del objeto de datos de sesión para el generador
        session = SessionData()
        session.exercise_name = ejercicio
        session.total_reps = st.session_state.stats["reps"]
        session.correct_reps = st.session_state.stats["reps"]
        session.incorrect_reps = 0
        session.duration_seconds = int(duration)
        session.avg_fps = int(frames_count / duration) if duration > 0 else 0
        
        # Ejecución del generador de reportes
        generate_report(session)
        
        # Localización del PDF generado para habilitar la descarga
        report_files = [os.path.join("reports", f) for f in os.listdir("reports") if f.endswith(".pdf")]
        if report_files:
            st.session_state.pdf_path = max(report_files, key=os.path.getctime)

        st.session_state.fase = 'finalizado'
        st.rerun()

    except Exception as e:
        st.error(f"Error gRPC: {e}")

elif st.session_state.fase == 'finalizado':
    """
    Fase de finalización.
    Muestra el resumen final del desempeño y orienta al usuario hacia el reporte.
    """
    st.success(f"Sesión completada. Puntaje alcanzado: {st.session_state.stats['score']:.1f}%")
    st.info("Puede descargar el reporte PDF desde el panel lateral izquierdo.")