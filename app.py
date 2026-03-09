import streamlit as st
import cv2
import grpc
import numpy as np
import av
import os
import sys
import time
import tempfile

# --- CONFIGURACIÓN DE RUTAS ABSOLUTAS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, 'src')

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# --- IMPORTACIONES DE MÓDULOS  ---
try:
    import pose_pb2
    import pose_pb2_grpc
    
    from pose_module.draw_pose import draw_pose
    from report_module.session_data import SessionData
    from report_module.report_generator import generate_report
except ImportError as e:
    st.error(f"Error de dependencias internas: {e}")
    st.stop()

# --- CONFIGURACIÓN DE LA INTERFAZ ---
st.set_page_config(page_title="App Physical Recovery", layout="wide")

if 'fase' not in st.session_state: st.session_state.fase = 'config'
if 'stats' not in st.session_state: st.session_state.stats = {"reps": 0, "score": 0.0}
if 'pdf_path' not in st.session_state: st.session_state.pdf_path = None

st.title("App Physical Recovery")
st.markdown("### Sistema de análisis de pose Vit Pose con gRPC")

# --- BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    st.header("Configuración de sesión")
    ejercicio = st.selectbox("Tipo de ejercicio:", ["Elevación brazo izquierdo"])
    
    st.divider()
    
    # Carga de video por el usuario
    st.subheader("Subir video")
    uploaded_file = st.file_uploader("Formatos soportados: MP4", type=["mp4"])
    
    st.divider()
    
    # Botones de descarga y reinicio
    if st.session_state.fase == 'finalizado' and st.session_state.pdf_path:
        if os.path.exists(st.session_state.pdf_path):
            with open(st.session_state.pdf_path, "rb") as f:
                st.download_button(
                    label="Descargar reporte PDF",
                    data=f,
                    file_name=os.path.basename(st.session_state.pdf_path),
                    mime="application/pdf",
                    use_container_width=True
                )

    if st.button("Reiniciar aplicación", use_container_width=True):
        st.session_state.fase = 'config'
        st.session_state.stats = {"reps": 0, "score": 0.0}
        st.session_state.pdf_path = None
        st.rerun()

# --- LÓGICA DE PANTALLAS ---

if st.session_state.fase == 'config':
    st.info("Cargue un video en el panel lateral para comenzar el análisis biomecánico.")
    if uploaded_file is not None:
        st.success(f"Archivo listo: {uploaded_file.name}")
        if st.button("Comenzar evaluación", type="primary", use_container_width=True):
            st.session_state.fase = 'procesando'
            st.rerun()

elif st.session_state.fase == 'procesando':
    col_vid, col_met = st.columns([2, 1])
    placeholder = col_vid.empty()
    
    historial_angulos = []
    ultimo_frame_con_pose = None

    with col_met:
        st.subheader("Métricas")
        m_reps = st.empty()
        m_angle = st.empty()
        m_score = st.empty()
        st.divider()
        stop_btn = st.button("Finalizar y generar reporte", type="primary", use_container_width=True)

    # --- MANEJO DE ARCHIVO TEMPORAL (WINDOWS SAFE) ---
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    video_path = tfile.name
    
    try:
        # Escribimos los bytes y cerramos el puntero de escritura inmediatamente
        tfile.write(uploaded_file.read())
        tfile.close() 

        # Conexión al servidor gRPC
        channel = grpc.insecure_channel('localhost:50051')
        stub = pose_pb2_grpc.PoseServiceStub(channel)
        
        # Procesamiento del video
        container = av.open(video_path)
        start_time = time.time()
        frames_count = 0

        try:
            for frame in container.decode(video=0):
                frames_count += 1
                img_raw = frame.to_ndarray(format='bgr24')
                img_disp = cv2.resize(img_raw, (640, 360))
                
                # Codificación para gRPC
                _, buffer = cv2.imencode('.jpg', img_disp)
                request = pose_pb2.PoseRequest(image_data=buffer.tobytes(), frame_id=frames_count)
                
                # Llamada al servidor
                responses = stub.StreamPose(iter([request]))
                for res in responses:
                    if len(res.people) > 0:
                        for p in res.people:
                            # Extracción de puntos para dibujo
                            kpts = np.array([[kp.x, kp.y] for kp in p.keypoints])
                            confs = np.array([kp.score for kp in p.keypoints])
                            img_disp = draw_pose(img_disp, kpts, confs)
                        
                        if frames_count % 15 == 0:
                            historial_angulos.append(res.current_angle)
                        ultimo_frame_con_pose = img_disp.copy()

                    # Actualización de UI
                    placeholder.image(cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB))
                    m_reps.metric("Repeticiones", res.repetitions)
                    m_angle.metric("Ángulo", f"{res.current_angle:.1f}°")
                    m_score.metric("Puntaje precisión", f"{res.final_score:.1f}%")
                    
                    st.session_state.stats = {"reps": res.repetitions, "score": res.final_score}
                    break
                
                if stop_btn: break
        finally:
            # Liberamos el video antes de intentar borrar el archivo del disco
            container.close()

        # --- FASE DE CIERRE Y REPORTE ---
        duration = time.time() - start_time
        
        # Captura de pantalla para el PDF
        img_temp_report = "session_capture.jpg"
        if ultimo_frame_con_pose is not None:
            cv2.imwrite(img_temp_report, ultimo_frame_con_pose)

        # Configuración de datos del reporte
        session = SessionData()
        session.exercise_name = ejercicio
        session.duration_seconds = int(duration)
        session.avg_fps = int(frames_count / duration) if duration > 0 else 0
        
        results = {
            "repetitions_detected": st.session_state.stats["reps"],
            "angles": historial_angulos[-8:],
            "scores": [st.session_state.stats["score"]],
            "final_score": st.session_state.stats["score"]
        }

        # Generar PDF
        generate_report(session, results, image_path=img_temp_report)
        
        # Buscar el archivo generado
        if os.path.exists("reports"):
            reports = [os.path.join("reports", f) for f in os.listdir("reports") if f.endswith(".pdf")]
            if reports:
                st.session_state.pdf_path = max(reports, key=os.path.getctime)

        # Borrado del video temporal
        if os.path.exists(video_path):
            try: os.remove(video_path)
            except: pass

        st.session_state.fase = 'finalizado'
        st.rerun()

    except Exception as e:
        st.error(f"Error durante el procesamiento: {e}")
        if os.path.exists(video_path):
            try: os.remove(video_path)
            except: pass

elif st.session_state.fase == 'finalizado':
    st.success(f"Evaluación completada con éxito. Total repeticiones: {st.session_state.stats['reps']}")
    st.balloons()