"""
mlflow_experiments.py
=====================
Experimentos MLflow para el proyecto de Pose Estimation con ViTPose+.

Este módulo define y ejecuta cinco experimentos de seguimiento usando MLflow,
cubriendo los principales parámetros configurables del pipeline de estimación
de pose: detección de personas, confianza de keypoints, preprocesamiento,
concurrencia del servidor gRPC y comparación de detectores.

Experimentos definidos:
    1. Threshold de detección de personas      (detector.py)
    2. Threshold de confianza de keypoints     (pose_estimator.py)
    3. Configuraciones de preprocesamiento     (frame_preprocessing.py)
    4. Workers del servidor gRPC               (grpc_server.py)
    5. Comparación de detectores de personas   (model_loader.py)

Uso:
    python mlflow_experiments.py            # Corre los 5 experimentos
    python mlflow_experiments.py --exp 1    # Corre sólo el experimento 1
    mlflow ui                               # Abre la UI en localhost:5000
"""

import argparse
import os
import tempfile
import time
from concurrent import futures

import cv2
import mlflow
import mlflow.transformers
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    RTDetrForObjectDetection,
    VitPoseForPoseEstimation,
)

# ---------------------------------------------------------------------------
# Constantes globales
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI = "http://localhost:5000"
"""str: URI de seguimiento de MLflow. Usar carpeta local o servidor remoto."""

TEST_IMAGE_PATH = "session_capture.jpg"
"""str: Ruta a la imagen de prueba. Ajustar según el proyecto."""

POSE_MODEL_ID = "usyd-community/vitpose-plus-base"
"""str: Identificador del modelo de estimación de pose en Hugging Face Hub."""

DETECTOR_MODELS = {
    "rtdetr_r50vd_coco_o365": "PekingU/rtdetr_r50vd_coco_o365",
    "rtdetr_r18vd": "PekingU/rtdetr_r18vd",
}
"""dict[str, str]: Detectores disponibles para el Experimento 5."""


# ---------------------------------------------------------------------------
# Funciones auxiliares privadas
# ---------------------------------------------------------------------------


def _get_device() -> str:
    """Determina el dispositivo de cómputo disponible.

    Returns
    -------
    str
        ``"cuda"`` si hay GPU disponible, ``"cpu"`` en caso contrario.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_test_image() -> Image.Image:
    """Carga la imagen de prueba desde disco o genera una imagen sintética.

    Si ``TEST_IMAGE_PATH`` no existe, crea un arreglo NumPy aleatorio de
    480x640 píxeles para permitir la ejecución sin archivos reales.

    Returns
    -------
    PIL.Image.Image
        Imagen en formato RGB lista para ser procesada por los modelos.
    """
    if os.path.exists(TEST_IMAGE_PATH):
        frame = cv2.imread(TEST_IMAGE_PATH)
    else:
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print(f"[AVISO] '{TEST_IMAGE_PATH}' no encontrado. Usando imagen sintetica.")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def _boxes_to_coco(boxes_voc: np.ndarray) -> np.ndarray:
    """Convierte bounding boxes del formato VOC al formato COCO.

    El formato VOC usa ``(x1, y1, x2, y2)`` mientras que COCO usa
    ``(x1, y1, w, h)``.

    Parameters
    ----------
    boxes_voc : numpy.ndarray
        Array de forma ``(N, 4)`` con bounding boxes en formato VOC.

    Returns
    -------
    numpy.ndarray
        Array de forma ``(N, 4)`` con bounding boxes en formato COCO.
    """
    boxes = boxes_voc.copy()
    boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
    boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
    return boxes


def _detect(
    image: Image.Image,
    processor,
    model,
    device: str,
    threshold: float,
) -> np.ndarray:
    """Detecta personas en una imagen y retorna sus bounding boxes.

    Ejecuta el detector RTDetr sobre la imagen, filtra unicamente las
    detecciones de la clase persona (label 0 en COCO) y convierte
    el resultado al formato COCO.

    Parameters
    ----------
    image : PIL.Image.Image
        Imagen de entrada en formato RGB.
    processor : transformers.AutoProcessor
        Procesador del modelo detector.
    model : transformers.RTDetrForObjectDetection
        Modelo detector de objetos.
    device : str
        Dispositivo de computo (``"cuda"`` o ``"cpu"``).
    threshold : float
        Umbral de confianza minimo para aceptar una deteccion.

    Returns
    -------
    numpy.ndarray
        Array de forma ``(N, 4)`` con bounding boxes en formato COCO.
        Puede ser un array vacio si no se detecta ninguna persona.
    """
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([(image.height, image.width)]),
        threshold=threshold,
    )[0]

    person_boxes = results["boxes"][results["labels"] == 0].cpu().numpy()

    if len(person_boxes) == 0:
        return person_boxes

    return _boxes_to_coco(person_boxes)


def _estimate_pose(
    image: Image.Image,
    person_boxes: np.ndarray,
    processor,
    model,
    device: str,
    kp_threshold: float,
    dataset_index: int = 0,
):
    """Estima la pose de las personas detectadas en una imagen.

    Ejecuta el modelo ViTPose+ sobre los recortes de personas y filtra
    los keypoints cuya confianza sea menor a ``kp_threshold``.

    Parameters
    ----------
    image : PIL.Image.Image
        Imagen de entrada en formato RGB.
    person_boxes : numpy.ndarray
        Bounding boxes de personas en formato COCO ``(x1, y1, w, h)``.
    processor : transformers.AutoProcessor
        Procesador del modelo de estimacion de pose.
    model : transformers.VitPoseForPoseEstimation
        Modelo de estimacion de pose.
    device : str
        Dispositivo de computo (``"cuda"`` o ``"cpu"``).
    kp_threshold : float
        Umbral de confianza minimo para conservar un keypoint.
    dataset_index : int, optional
        Indice del experto MoE a utilizar (0-5). Por defecto es ``0``
        que corresponde al dataset MS COCO.

    Returns
    -------
    list[dict]
        Lista de diccionarios con los keypoints y scores de cada persona
        detectada, tal como los retorna ``post_process_pose_estimation``.
    """
    inputs = processor(image, boxes=[person_boxes], return_tensors="pt").to(device)
    inputs["dataset_index"] = torch.tensor([dataset_index], device=device)

    with torch.no_grad():
        outputs = model(**inputs)

    return processor.post_process_pose_estimation(
        outputs,
        boxes=[person_boxes],
        threshold=kp_threshold,
    )[0]


def _compute_arm_angle(keypoints) -> float:
    """Calcula el angulo del brazo izquierdo usando tres keypoints COCO.

    Replica la logica de ``pose_rating._compute_arm_angle``. Usa el
    hombro izquierdo (indice 5), la muneca izquierda (indice 9) y la
    cadera izquierda (indice 11) como vertices del angulo.

    Parameters
    ----------
    keypoints : array-like
        Secuencia de keypoints en formato COCO. Cada elemento debe
        soportar indexacion numerica para extraer las coordenadas ``x``
        e ``y``.

    Returns
    -------
    float
        Angulo del brazo en grados. Retorna ``0.0`` si alguno de los
        vectores tiene norma cero.
    """
    kps = np.array([[float(kp[0]), float(kp[1])] for kp in keypoints])
    shoulder = kps[5]
    wrist = kps[9]
    hip = kps[11]

    ba = hip - shoulder
    bc = wrist - shoulder

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    cosine = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _save_temp_image(image: Image.Image, filename: str = "result.jpg") -> str:
    """Guarda una imagen PIL en el directorio temporal del sistema.

    Parameters
    ----------
    image : PIL.Image.Image
        Imagen a guardar.
    filename : str, optional
        Nombre del archivo de salida. Por defecto es ``"result.jpg"``.

    Returns
    -------
    str
        Ruta absoluta al archivo guardado.
    """
    path = os.path.join(tempfile.gettempdir(), filename)
    image.save(path)
    return path


# ===========================================================================
# Experimento 1 - Threshold de deteccion de personas
# ===========================================================================


def experimento_1_threshold_deteccion() -> None:
    """Evalua el impacto del umbral de deteccion sobre el detector RTDetr.

    Varia el parametro ``threshold`` del detector de personas entre los
    valores ``[0.2, 0.3, 0.4, 0.5]`` y registra en MLflow cuantas personas
    se detectan y el tiempo de inferencia para cada configuracion.

    Cada valor de threshold genera un run independiente dentro del
    experimento ``Exp1_Threshold_Deteccion_Personas``.

    Parametros registrados
    ----------------------
    detector_model : str
        Nombre del modelo detector utilizado.
    detector_threshold : float
        Valor del umbral evaluado en el run.
    device : str
        Dispositivo de computo usado (``"cuda"`` o ``"cpu"``).

    Metricas registradas
    --------------------
    persons_detected : int
        Numero de personas detectadas con el umbral dado.
    inference_time_ms : float
        Tiempo de inferencia del detector en milisegundos.

    Artefactos registrados
    ----------------------
    input/test_frame.jpg
        Imagen de prueba utilizada durante el experimento.
    """
    mlflow.set_experiment("Exp1_Threshold_Deteccion_Personas")

    device = _get_device()
    image = _load_test_image()
    thresholds = [0.2, 0.3, 0.4, 0.5]

    processor = AutoProcessor.from_pretrained(
        DETECTOR_MODELS["rtdetr_r50vd_coco_o365"]
    )
    model = RTDetrForObjectDetection.from_pretrained(
        DETECTOR_MODELS["rtdetr_r50vd_coco_o365"]
    ).to(device)

    print("\n[Exp 1] Threshold de deteccion de personas")

    for thr in thresholds:
        with mlflow.start_run(run_name=f"det_threshold_{thr}"):
            mlflow.log_param("detector_model", "rtdetr_r50vd_coco_o365")
            mlflow.log_param("detector_threshold", thr)
            mlflow.log_param("device", device)

            t0 = time.perf_counter()
            boxes = _detect(image, processor, model, device, threshold=thr)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            mlflow.log_metric("persons_detected", len(boxes))
            mlflow.log_metric("inference_time_ms", round(elapsed_ms, 2))

            img_path = _save_temp_image(image, "test_frame.jpg")
            mlflow.log_artifact(img_path, artifact_path="input")

            print(
                f"  threshold={thr} -> personas={len(boxes)}, "
                f"tiempo={elapsed_ms:.1f}ms"
            )

    print("[Exp 1] Completado.\n")


# ===========================================================================
# Experimento 2 - Threshold de confianza de keypoints
# ===========================================================================


def experimento_2_threshold_keypoints() -> None:
    """Evalua el impacto del umbral de confianza sobre los keypoints estimados.

    Varia el parametro ``threshold`` de ``post_process_pose_estimation``
    entre ``[0.2, 0.3, 0.4, 0.5]`` y registra cuantos keypoints superan
    el filtro, su puntuacion promedio y el angulo del brazo resultante.

    La deteccion de personas se realiza una sola vez con ``threshold=0.3``
    para aislar el efecto del umbral de keypoints. Si no se detecta ninguna
    persona, el experimento se omite con una advertencia en consola.

    Cada valor de threshold genera un run independiente dentro del
    experimento ``Exp2_Threshold_Confianza_Keypoints``.

    Parametros registrados
    ----------------------
    pose_model : str
        Identificador del modelo de estimacion de pose.
    keypoint_threshold : float
        Valor del umbral de confianza de keypoints evaluado.
    detector_threshold : float
        Umbral fijo usado en la deteccion de personas (``0.3``).
    device : str
        Dispositivo de computo usado.

    Metricas registradas
    --------------------
    keypoints_detected : int
        Numero de keypoints que superaron el umbral.
    avg_keypoint_score : float
        Puntuacion de confianza promedio de los keypoints detectados.
    arm_angle_deg : float
        Angulo del brazo izquierdo calculado en grados.
    inference_time_ms : float
        Tiempo de inferencia del modelo de pose en milisegundos.

    Artefactos registrados
    ----------------------
    input/test_frame.jpg
        Imagen de prueba utilizada durante el experimento.
    """
    mlflow.set_experiment("Exp2_Threshold_Confianza_Keypoints")

    device = _get_device()
    image = _load_test_image()
    kp_thresholds = [0.2, 0.3, 0.4, 0.5]

    det_processor = AutoProcessor.from_pretrained(
        DETECTOR_MODELS["rtdetr_r50vd_coco_o365"]
    )
    det_model = RTDetrForObjectDetection.from_pretrained(
        DETECTOR_MODELS["rtdetr_r50vd_coco_o365"]
    ).to(device)

    pose_processor = AutoProcessor.from_pretrained(POSE_MODEL_ID, use_fast=False)
    pose_model = VitPoseForPoseEstimation.from_pretrained(POSE_MODEL_ID).to(device)

    boxes = _detect(image, det_processor, det_model, device, threshold=0.3)

    print("[Exp 2] Threshold de confianza de keypoints")

    if len(boxes) == 0:
        print("  [AVISO] No se detectaron personas. Omitiendo experimento 2.")
        return

    for kp_thr in kp_thresholds:
        with mlflow.start_run(run_name=f"kp_threshold_{kp_thr}"):
            mlflow.log_param("pose_model", POSE_MODEL_ID)
            mlflow.log_param("keypoint_threshold", kp_thr)
            mlflow.log_param("detector_threshold", 0.3)
            mlflow.log_param("device", device)

            t0 = time.perf_counter()
            pose_results = _estimate_pose(
                image,
                boxes,
                pose_processor,
                pose_model,
                device,
                kp_threshold=kp_thr,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000

            person = (
                pose_results[0] if isinstance(pose_results, list) else pose_results
            )
            kps = person["keypoints"]
            scores = person["scores"]

            avg_score = float(np.mean([float(s) for s in scores]))
            angle = _compute_arm_angle(kps)

            mlflow.log_metric("keypoints_detected", len(kps))
            mlflow.log_metric("avg_keypoint_score", round(avg_score, 4))
            mlflow.log_metric("arm_angle_deg", round(angle, 2))
            mlflow.log_metric("inference_time_ms", round(elapsed_ms, 2))

            img_path = _save_temp_image(image, "test_frame.jpg")
            mlflow.log_artifact(img_path, artifact_path="input")

            print(
                f"  kp_threshold={kp_thr} -> kps={len(kps)}, "
                f"avg_score={avg_score:.3f}, angulo={angle:.1f}°, "
                f"tiempo={elapsed_ms:.1f}ms"
            )

    print("[Exp 2] Completado.\n")


# ===========================================================================
# Experimento 3 - Configuraciones de preprocesamiento
# ===========================================================================


def experimento_3_preprocesamiento() -> None:
    """Evalua el impacto de distintas configuraciones de preprocesamiento.

    Combina las opciones de redimensionamiento y suavizado gaussiano del
    modulo ``frame_preprocessing`` y mide el tiempo total del pipeline
    (preprocesamiento + deteccion + pose) y la calidad del angulo estimado.

    Las cinco configuraciones evaluadas son:

    +---------------+-----------------+------------------+
    | redimensionar | ancho_objetivo  | aplicar_suavizado|
    +===============+=================+==================+
    | False         | 640             | False            |
    +---------------+-----------------+------------------+
    | True          | 640             | False            |
    +---------------+-----------------+------------------+
    | True          | 480             | False            |
    +---------------+-----------------+------------------+
    | True          | 640             | True             |
    +---------------+-----------------+------------------+
    | True          | 480             | True             |
    +---------------+-----------------+------------------+

    Cada configuracion genera un run dentro del experimento
    ``Exp3_Configuraciones_Preprocesamiento``.

    Parametros registrados
    ----------------------
    redimensionar : bool
        Indica si se aplica redimensionamiento al frame.
    ancho_objetivo : int
        Ancho maximo en pixeles tras el redimensionamiento.
    aplicar_suavizado : bool
        Indica si se aplica suavizado gaussiano al frame.
    device : str
        Dispositivo de computo usado.

    Metricas registradas
    --------------------
    pipeline_time_ms : float
        Tiempo total del pipeline completo en milisegundos.
    arm_angle_deg : float
        Angulo del brazo izquierdo calculado en grados.
    keypoints_detected : int
        Numero de keypoints detectados.
    frame_width_px : int
        Ancho real del frame tras el preprocesamiento.

    Artefactos registrados
    ----------------------
    preprocessed/preprocessed_frame.jpg
        Frame resultante tras aplicar el preprocesamiento.
    """
    try:
        from preprocessing.frame_preprocessing import procesar_frame_para_modelo
    except ImportError:
        from frame_preprocessing import procesar_frame_para_modelo

    mlflow.set_experiment("Exp3_Configuraciones_Preprocesamiento")

    device = _get_device()

    if os.path.exists(TEST_IMAGE_PATH):
        frame_bgr = cv2.imread(TEST_IMAGE_PATH)
    else:
        frame_bgr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    det_processor = AutoProcessor.from_pretrained(
        DETECTOR_MODELS["rtdetr_r50vd_coco_o365"]
    )
    det_model = RTDetrForObjectDetection.from_pretrained(
        DETECTOR_MODELS["rtdetr_r50vd_coco_o365"]
    ).to(device)
    pose_processor = AutoProcessor.from_pretrained(POSE_MODEL_ID, use_fast=False)
    pose_model = VitPoseForPoseEstimation.from_pretrained(POSE_MODEL_ID).to(device)

    configs = [
        {"redimensionar": False, "ancho_objetivo": 640, "aplicar_suavizado": False},
        {"redimensionar": True, "ancho_objetivo": 640, "aplicar_suavizado": False},
        {"redimensionar": True, "ancho_objetivo": 480, "aplicar_suavizado": False},
        {"redimensionar": True, "ancho_objetivo": 640, "aplicar_suavizado": True},
        {"redimensionar": True, "ancho_objetivo": 480, "aplicar_suavizado": True},
    ]

    print("[Exp 3] Configuraciones de preprocesamiento")

    for cfg in configs:
        run_name = (
            f"resize={cfg['redimensionar']}_"
            f"w={cfg['ancho_objetivo']}_"
            f"blur={cfg['aplicar_suavizado']}"
        )

        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("redimensionar", cfg["redimensionar"])
            mlflow.log_param("ancho_objetivo", cfg["ancho_objetivo"])
            mlflow.log_param("aplicar_suavizado", cfg["aplicar_suavizado"])
            mlflow.log_param("device", device)

            t0 = time.perf_counter()

            resultado = procesar_frame_para_modelo(
                frame_bgr,
                frame_index=0,
                redimensionar=cfg["redimensionar"],
                ancho_objetivo=cfg["ancho_objetivo"],
                aplicar_suavizado=cfg["aplicar_suavizado"],
            )

            if not resultado["valid"]:
                mlflow.log_metric("pipeline_time_ms", 0)
                mlflow.log_metric("arm_angle_deg", 0)
                mlflow.log_metric("keypoints_detected", 0)
                print(f"  {run_name} -> frame invalido")
                continue

            image_pil = resultado["image_pil"]
            boxes = _detect(
                image_pil, det_processor, det_model, device, threshold=0.3
            )

            angle = 0.0
            kps_count = 0

            if len(boxes) > 0:
                pose_results = _estimate_pose(
                    image_pil,
                    boxes,
                    pose_processor,
                    pose_model,
                    device,
                    kp_threshold=0.3,
                )
                person = (
                    pose_results[0]
                    if isinstance(pose_results, list)
                    else pose_results
                )
                kps_count = len(person["keypoints"])
                angle = _compute_arm_angle(person["keypoints"])

            elapsed_ms = (time.perf_counter() - t0) * 1000

            mlflow.log_metric("pipeline_time_ms", round(elapsed_ms, 2))
            mlflow.log_metric("arm_angle_deg", round(angle, 2))
            mlflow.log_metric("keypoints_detected", kps_count)
            mlflow.log_metric(
                "frame_width_px", resultado["frame_procesado"].shape[1]
            )

            pre_path = _save_temp_image(image_pil, "preprocessed_frame.jpg")
            mlflow.log_artifact(pre_path, artifact_path="preprocessed")

            print(
                f"  {run_name} -> tiempo={elapsed_ms:.1f}ms, "
                f"angulo={angle:.1f}°, kps={kps_count}"
            )

    print("[Exp 3] Completado.\n")


# ===========================================================================
# Experimento 4 - Workers del servidor gRPC
# ===========================================================================


def experimento_4_grpc_workers() -> None:
    """Evalua el impacto del numero de workers del ThreadPoolExecutor gRPC.

    Simula el procesamiento concurrente de frames tal como lo haria el
    servidor gRPC (``grpc_server.py``), variando el parametro ``max_workers``
    del ``ThreadPoolExecutor`` entre ``[1, 2, 4, 8]``.

    No levanta un servidor gRPC real; el objetivo es aislar el efecto de
    la concurrencia sobre la latencia y el throughput del pipeline de
    deteccion y estimacion de pose.

    Cada configuracion de workers genera un run dentro del experimento
    ``Exp4_Workers_Servidor_gRPC``.

    Parametros registrados
    ----------------------
    max_workers : int
        Numero de workers del ThreadPoolExecutor evaluado.
    num_frames : int
        Numero de frames simulados por configuracion (fijo en 20).
    device : str
        Dispositivo de computo usado.

    Metricas registradas
    --------------------
    avg_latency_ms : float
        Latencia promedio por frame en milisegundos.
    min_latency_ms : float
        Latencia minima registrada en milisegundos.
    max_latency_ms : float
        Latencia maxima registrada en milisegundos.
    throughput_fps : float
        Throughput total expresado en frames por segundo.
    total_time_ms : float
        Tiempo total de procesamiento de todos los frames en milisegundos.

    Artefactos registrados
    ----------------------
    logs/latencies.txt
        Archivo de texto con la latencia individual de cada frame simulado.
    """
    mlflow.set_experiment("Exp4_Workers_Servidor_gRPC")

    device = _get_device()
    image = _load_test_image()

    det_processor = AutoProcessor.from_pretrained(
        DETECTOR_MODELS["rtdetr_r50vd_coco_o365"]
    )
    det_model = RTDetrForObjectDetection.from_pretrained(
        DETECTOR_MODELS["rtdetr_r50vd_coco_o365"]
    ).to(device)
    pose_processor = AutoProcessor.from_pretrained(POSE_MODEL_ID, use_fast=False)
    pose_model = VitPoseForPoseEstimation.from_pretrained(POSE_MODEL_ID).to(device)

    num_frames = 20
    worker_opts = [1, 2, 4, 8]

    def process_single_frame(_: int) -> float:
        """Procesa un frame simulado y retorna su latencia en milisegundos.

        Parameters
        ----------
        _ : int
            Indice del frame (no utilizado; requerido por ``executor.map``).

        Returns
        -------
        float
            Tiempo de procesamiento del frame en milisegundos.
        """
        t0 = time.perf_counter()
        boxes = _detect(image, det_processor, det_model, device, threshold=0.3)
        if len(boxes) > 0:
            _estimate_pose(
                image,
                boxes,
                pose_processor,
                pose_model,
                device,
                kp_threshold=0.3,
            )
        return (time.perf_counter() - t0) * 1000

    print("[Exp 4] Workers del servidor gRPC")

    for n_workers in worker_opts:
        with mlflow.start_run(run_name=f"grpc_workers_{n_workers}"):
            mlflow.log_param("max_workers", n_workers)
            mlflow.log_param("num_frames", num_frames)
            mlflow.log_param("device", device)

            t_total_start = time.perf_counter()
            with futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                latencies = list(
                    executor.map(process_single_frame, range(num_frames))
                )
            total_ms = (time.perf_counter() - t_total_start) * 1000

            avg_lat = float(np.mean(latencies))
            throughput = num_frames / (total_ms / 1000)

            mlflow.log_metric("avg_latency_ms", round(avg_lat, 2))
            mlflow.log_metric("min_latency_ms", round(float(np.min(latencies)), 2))
            mlflow.log_metric("max_latency_ms", round(float(np.max(latencies)), 2))
            mlflow.log_metric("throughput_fps", round(throughput, 3))
            mlflow.log_metric("total_time_ms", round(total_ms, 2))

            log_path = os.path.join(tempfile.gettempdir(), "latencies.txt")
            with open(log_path, "w", encoding="utf-8") as log_file:
                log_file.write(f"workers={n_workers}\n")
                for i, lat in enumerate(latencies):
                    log_file.write(f"frame_{i:03d}: {lat:.2f}ms\n")
            mlflow.log_artifact(log_path, artifact_path="logs")

            print(
                f"  workers={n_workers} -> avg={avg_lat:.1f}ms, "
                f"throughput={throughput:.2f}fps, total={total_ms:.0f}ms"
            )

    print("[Exp 4] Completado.\n")


# ===========================================================================
# Experimento 5 - Comparacion de detectores de personas
# ===========================================================================


def experimento_5_comparacion_detectores() -> None:
    """Compara el rendimiento de distintos detectores de personas.

    Evalua los detectores definidos en ``DETECTOR_MODELS`` midiendo su
    tiempo de inferencia, la cantidad de personas detectadas y el angulo
    del brazo resultante al ejecutar el pipeline completo con ViTPose+.

    Al finalizar, registra cada detector en el MLflow Model Registry usando
    ``mlflow.transformers.log_model`` e imprime en consola cual fue el
    detector mas rapido entre los que lograron al menos una deteccion.

    Cada detector genera un run dentro del experimento
    ``Exp5_Comparacion_Detectores``.

    Parametros registrados
    ----------------------
    detector_name : str
        Nombre corto del detector evaluado.
    detector_model_id : str
        Identificador completo del detector en Hugging Face Hub.
    pose_model_id : str
        Identificador del modelo de estimacion de pose.
    detector_threshold : float
        Umbral fijo de deteccion (``0.3``).
    kp_threshold : float
        Umbral fijo de confianza de keypoints (``0.3``).
    device : str
        Dispositivo de computo usado.

    Metricas registradas
    --------------------
    persons_detected : int
        Numero de personas detectadas por el detector.
    det_time_ms : float
        Tiempo de inferencia del detector en milisegundos.
    pose_time_ms : float
        Tiempo de inferencia del modelo de pose en milisegundos.
    total_pipeline_ms : float
        Tiempo total del pipeline (deteccion + pose) en milisegundos.
    keypoints_detected : int
        Numero de keypoints detectados para la primera persona.
    arm_angle_deg : float
        Angulo del brazo izquierdo calculado en grados.

    Artefactos registrados
    ----------------------
    MLflow Model Registry
        Cada detector es registrado con el nombre
        ``PoseEstimation_Detector_<nombre>``.
    """
    mlflow.set_experiment("Exp5_Comparacion_Detectores")

    device = _get_device()
    image = _load_test_image()

    pose_processor = AutoProcessor.from_pretrained(POSE_MODEL_ID, use_fast=False)
    pose_model = VitPoseForPoseEstimation.from_pretrained(POSE_MODEL_ID).to(device)

    results_summary = {}

    print("[Exp 5] Comparacion de detectores de personas")

    for det_name, det_id in DETECTOR_MODELS.items():
        with mlflow.start_run(run_name=f"detector_{det_name}"):
            mlflow.log_param("detector_name", det_name)
            mlflow.log_param("detector_model_id", det_id)
            mlflow.log_param("pose_model_id", POSE_MODEL_ID)
            mlflow.log_param("detector_threshold", 0.3)
            mlflow.log_param("kp_threshold", 0.3)
            mlflow.log_param("device", device)

            det_processor = AutoProcessor.from_pretrained(det_id)
            det_model = RTDetrForObjectDetection.from_pretrained(det_id).to(device)

            t_det = time.perf_counter()
            boxes = _detect(image, det_processor, det_model, device, threshold=0.3)
            det_ms = (time.perf_counter() - t_det) * 1000

            angle = 0.0
            kps_count = 0
            pose_ms = 0.0

            if len(boxes) > 0:
                t_pose = time.perf_counter()
                pose_results = _estimate_pose(
                    image,
                    boxes,
                    pose_processor,
                    pose_model,
                    device,
                    kp_threshold=0.3,
                )
                pose_ms = (time.perf_counter() - t_pose) * 1000
                person = (
                    pose_results[0]
                    if isinstance(pose_results, list)
                    else pose_results
                )
                kps_count = len(person["keypoints"])
                angle = _compute_arm_angle(person["keypoints"])

            total_ms = det_ms + pose_ms

            mlflow.log_metric("persons_detected", len(boxes))
            mlflow.log_metric("det_time_ms", round(det_ms, 2))
            mlflow.log_metric("pose_time_ms", round(pose_ms, 2))
            mlflow.log_metric("total_pipeline_ms", round(total_ms, 2))
            mlflow.log_metric("keypoints_detected", kps_count)
            mlflow.log_metric("arm_angle_deg", round(angle, 2))

            results_summary[det_name] = {
                "total_ms": total_ms,
                "persons": len(boxes),
                "angle": angle,
            }

            components = {"model": det_model, "image_processor": det_processor}
            mlflow.transformers.log_model(
                transformers_model=components,
                name=f"detector_{det_name}",
                task="object-detection",
                registered_model_name=f"PoseEstimation_Detector_{det_name}",
            )

            print(
                f"  {det_name} -> personas={len(boxes)}, "
                f"det={det_ms:.1f}ms, total={total_ms:.1f}ms, "
                f"angulo={angle:.1f}°"
            )

    valid = {k: v for k, v in results_summary.items() if v["persons"] > 0}
    if valid:
        best = min(valid, key=lambda k: valid[k]["total_ms"])
        print(f"\n  Detector recomendado (mas rapido con detecciones): {best}")
        print(
            f"  Registrado en MLflow Model Registry como "
            f"'PoseEstimation_Detector_{best}'"
        )

    print("[Exp 5] Completado.\n")


# ===========================================================================
# Punto de entrada
# ===========================================================================


def main() -> None:
    """Punto de entrada principal del script de experimentos MLflow.

    Parsea los argumentos de linea de comandos para determinar que
    experimentos ejecutar, configura el tracking URI de MLflow y delega
    la ejecucion a cada funcion de experimento correspondiente.

    Si no se especifica ``--exp``, se ejecutan los cinco experimentos en
    orden secuencial.

    Command-line arguments
    ----------------------
    --exp : int, optional
        Numero del experimento a ejecutar (1-5). Si se omite, se ejecutan
        todos.

    Examples
    --------
    Ejecutar todos los experimentos::

        python mlflow_experiments.py

    Ejecutar unicamente el experimento 3::

        python mlflow_experiments.py --exp 3

    Ver resultados en la UI de MLflow::

        mlflow ui
    """
    parser = argparse.ArgumentParser(
        description="Corre los experimentos MLflow del proyecto de Pose Estimation."
    )
    parser.add_argument(
        "--exp",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=None,
        help="Numero del experimento a correr (1-5). Sin este flag corre todos.",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    experiments = {
        1: experimento_1_threshold_deteccion,
        2: experimento_2_threshold_keypoints,
        3: experimento_3_preprocesamiento,
        4: experimento_4_grpc_workers,
        5: experimento_5_comparacion_detectores,
    }

    to_run = [args.exp] if args.exp else [1, 2, 3, 4, 5]

    print("=" * 60)
    print("  MLflow - Pose Estimation Experiments")
    print(f"  Tracking URI : {MLFLOW_TRACKING_URI}")
    print(f"  Device       : {_get_device()}")
    print(f"  Experimentos : {to_run}")
    print("=" * 60)

    for exp_num in to_run:
        experiments[exp_num]()

    print("=" * 60)
    print("  Todos los experimentos completados.")
    print("  Ejecuta : mlflow ui")
    print("  Abre    : http://localhost:5000")
    print("=" * 60)


if __name__ == "__main__":
    main()
