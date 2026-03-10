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
SRC_DIR  = os.path.join(BASE_DIR, 'src')

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# --- IMPORTACIONES ---
try:
    import pose_pb2
    import pose_pb2_grpc

    from pose_module.draw_pose import draw_pose
    from report_module.session_data import SessionData
    from report_module.report_generator import generate_report

    # pose_rating vive SOLO en app.py
    # El servidor ya no lo usa → no hay doble conteo ni conflicto de reset
    from pose_rating import evaluate_pose, get_final_rating, reset_session

except ImportError as e:
    st.error(f"Error de dependencias internas: {e}")
    st.stop()

# --- CONFIGURACIÓN DE LA INTERFAZ ---
st.set_page_config(page_title="App Physical Recovery", layout="wide")

if 'fase'             not in st.session_state: st.session_state.fase             = 'config'
if 'stats'            not in st.session_state: st.session_state.stats            = {"reps": 0, "score": 0.0}
if 'pdf_path'         not in st.session_state: st.session_state.pdf_path         = None
if 'exercise_results' not in st.session_state: st.session_state.exercise_results = None

st.title("App Physical Recovery")
st.markdown("### Sistema de análisis de pose ViTPose con gRPC")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("Configuración de sesión")
    ejercicio = st.selectbox("Tipo de ejercicio:", ["Elevación brazo izquierdo"])

    st.divider()

    st.subheader("Subir video")
    uploaded_file = st.file_uploader("Formatos soportados: MP4", type=["mp4"])

    st.divider()

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
        st.session_state.fase             = 'config'
        st.session_state.stats            = {"reps": 0, "score": 0.0}
        st.session_state.pdf_path         = None
        st.session_state.exercise_results = None
        st.rerun()


# ══════════════════════════════════════════════════════════════
# PANTALLA: CONFIG
# ══════════════════════════════════════════════════════════════
if st.session_state.fase == 'config':
    st.info("Cargue un video en el panel lateral para comenzar el análisis biomecánico.")
    if uploaded_file is not None:
        st.success(f"Archivo listo: {uploaded_file.name}")
        if st.button("Comenzar evaluación", type="primary", use_container_width=True):
            st.session_state.fase = 'procesando'
            st.rerun()


# ══════════════════════════════════════════════════════════════
# PANTALLA: PROCESANDO
# ══════════════════════════════════════════════════════════════
elif st.session_state.fase == 'procesando':

    col_vid, col_met = st.columns([2, 1])
    placeholder = col_vid.empty()
    ultimo_frame_con_pose = None

    with col_met:
        st.subheader("Métricas en vivo")
        m_reps  = st.empty()
        m_angle = st.empty()
        m_score = st.empty()
        st.divider()
        stop_btn = st.button("Finalizar y generar reporte", type="primary", use_container_width=True)

    tfile      = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    video_path = tfile.name

    try:
        tfile.write(uploaded_file.read())
        tfile.close()

        # Reset SIEMPRE antes de empezar un video nuevo
        # Como pose_rating solo vive aquí, este reset es suficiente
        reset_session()

        channel = grpc.insecure_channel('localhost:50051')
        stub    = pose_pb2_grpc.PoseServiceStub(channel)

        container    = av.open(video_path)
        start_time   = time.time()
        frames_count = 0

        try:
            for frame in container.decode(video=0):
                frames_count += 1
                img_raw  = frame.to_ndarray(format='bgr24')
                img_disp = cv2.resize(img_raw, (640, 360))

                _, buffer = cv2.imencode('.jpg', img_disp)
                request   = pose_pb2.PoseRequest(
                    image_data=buffer.tobytes(),
                    frame_id=frames_count
                )

                responses = stub.StreamPose(iter([request]))

                for res in responses:

                    if len(res.people) > 0:
                        for p in res.people:
                            kpts  = np.array([[kp.x, kp.y] for kp in p.keypoints])
                            confs = np.array([kp.score for kp in p.keypoints])
                            img_disp = draw_pose(img_disp, kpts, confs)

                            # pose_rating solo se llama aquí, una vez por frame
                            evaluate_pose(kpts)

                        ultimo_frame_con_pose = img_disp.copy()

                    # FPS
                    elapsed = time.time() - start_time
                    fps     = frames_count / elapsed if elapsed > 0 else 0
                    cv2.putText(img_disp, f"FPS: {fps:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Conteo local (no del servidor) → siempre empieza en 0
                    reps_local = get_final_rating()["repetitions_detected"]

                    placeholder.image(cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB))
                    m_reps.metric("Repeticiones",   reps_local)
                    m_angle.metric("Ángulo actual", f"{res.current_angle:.1f}°")

                    st.session_state.stats["reps"] = reps_local
                    break

                if stop_btn:
                    break

        finally:
            container.close()

        # Resultados finales desde pose_rating local
        results = get_final_rating()

        m_score.metric("Puntaje final", f"{results['final_score']:.1f}%")

        st.session_state.exercise_results = results
        st.session_state.stats = {
            "reps":  results["repetitions_detected"],
            "score": results["final_score"]
        }

        # Reporte PDF
        duration        = time.time() - start_time
        img_temp_report = "session_capture.jpg"
        if ultimo_frame_con_pose is not None:
            cv2.imwrite(img_temp_report, ultimo_frame_con_pose)

        session                  = SessionData()
        session.exercise_name    = ejercicio
        session.duration_seconds = int(duration)
        session.avg_fps          = int(frames_count / duration) if duration > 0 else 0

        generate_report(session, results, image_path=img_temp_report)

        if os.path.exists("reports"):
            pdfs = [os.path.join("reports", f)
                    for f in os.listdir("reports") if f.endswith(".pdf")]
            if pdfs:
                st.session_state.pdf_path = max(pdfs, key=os.path.getctime)

        if os.path.exists(video_path):
            try: os.remove(video_path)
            except Exception: pass

        st.session_state.fase = 'finalizado'
        st.rerun()

    except Exception as e:
        st.error(f"Error durante el procesamiento: {e}")
        if os.path.exists(video_path):
            try: os.remove(video_path)
            except Exception: pass


# ══════════════════════════════════════════════════════════════
# PANTALLA: FINALIZADO
# ══════════════════════════════════════════════════════════════
elif st.session_state.fase == 'finalizado':
    results = st.session_state.exercise_results or {}

    st.success(
        f"Evaluación completada — "
        f"Repeticiones: {results.get('repetitions_detected', 0)} | "
        f"Score global: {results.get('final_score', 0.0):.1f}%"
    )

    angles = results.get("angles", [])
    scores = results.get("scores", [])

    if angles and scores:
        st.subheader("Detalle por repetición")
        header_cols = st.columns(3)
        header_cols[0].markdown("**Repetición**")
        header_cols[1].markdown("**Ángulo máximo**")
        header_cols[2].markdown("**Score**")

        for i, (ang, sc) in enumerate(zip(angles, scores)):
            row_cols = st.columns(3)
            row_cols[0].write(f"Rep. {i + 1}")
            row_cols[1].write(f"{ang:.2f}°")
            row_cols[2].write(f"{sc:.2f}%")

    st.balloons()