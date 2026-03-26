# App Physical Recovery: Sistema de Monitoreo de Rehabilitación de Miembro Superior con Visión por Computadora

**Estudiantes:**
- Miguel Angel Franco Restrepo (22506163)  
- Saulo Quiñones Góngora (22506635)  
- Adrian Felipe Vargas Rojas (22505561)
- Juan Sebastián Peña Valderrama (22502483)  

**Curso:** Desarrollo de Proyectos de IA

**Institución:** Universidad Autónoma de Occidente

**Periodo:** 2026‑1

**Repositorio:** App_PhysicalRecovery

---

## 1. Resumen del proyecto

Este proyecto implementa un sistema de visión por computadora para el monitoreo de ejercicios de rehabilitación de miembro superior, utilizando modelos de Deep Learning. El sistema procesa frames de video en tiempo real, detecta personas, estima la pose corporal y calcula métricas biomecánicas (ángulo del brazo) para evaluar si el ejercicio se está ejecutando correctamente.

La arquitectura está dividida en dos componentes desacoplados:

- **Servidor gRPC** (`grpc_server.py`): ejecuta el pipeline de inferencia (detección + estimación de pose + cálculo de ángulo).
- **Interfaz Streamlit** (`app.py`): actúa como cliente gRPC, gestiona la evaluación de repeticiones y genera el reporte final en PDF.

**Métricas de evaluación:**
- Ángulo del brazo izquierdo por frame
- Conteo de repeticiones detectadas
- Puntaje porcentual por repetición (basado en ángulo máximo alcanzado)
- Calificación final promediada

---

## 2. Objetivo de la práctica

El objetivo principal es construir un sistema modular de monitoreo de ejercicios de rehabilitación, donde:

- Cada módulo cumple una única responsabilidad (preprocesar, detectar, estimar, evaluar, reportar).
- La interfaz gráfica no ejecuta inferencia: la delega íntegramente al servidor gRPC.
- La comunicación cliente-servidor se realiza mediante un protocolo eficiente de streaming bidireccional (gRPC + Protocol Buffers).
- El pipeline es reproducible y trazable mediante integración con **MLflow** para el seguimiento de experimentos.
- El flujo puede validarse con pruebas unitarias usando `pytest`.

---

## 3. Estructura del proyecto

```
App_PhysicalRecovery/
│
├── app.py                              # Interfaz Streamlit (cliente gRPC)
├── main.py                             # Script de ejecución local (sin gRPC)
├── benchmark_mlflow.py                 # Benchmark y registro de experimentos en MLflow
├── pose.proto                          # Definición del protocolo gRPC
├── pyproject.toml                      # Dependencias del proyecto (UV)
├── uv.lock                             # Lockfile reproducible (UV)
├── Makefile                            # Comandos abreviados para ejecución del proyecto
├── .pre-commit-config.yaml             # Hooks de calidad de código (Ruff, pytest, saneamiento)
├── mlflow.db                           # Base de datos local de experimentos MLflow
├── session_capture.jpg                 # Captura de sesión de ejemplo
│
├── videos/                             # Videos de prueba para el ejercicio de rehabilitación
│
├── src/
│   ├── __init__.py
│   ├── grpc_server.py                  # Servidor gRPC: orquesta el pipeline de inferencia
│   ├── pose_rating.py                  # Evaluación de ejercicio: ángulos, reps y puntaje
│   ├── pose_pb2.py                     # Código generado por protoc (mensajes)
│   ├── pose_pb2_grpc.py                # Código generado por protoc (servicios)
│   ├── mlflow_experiments.py           # Registro y ejecución de experimentos MLflow
│   │
│   ├── preprocessing/
│   │   └── frame_preprocessing.py     # Validación, resize, suavizado y conversión de frames
│   │
│   ├── pose_module/
│   │   ├── __init__.py
│   │   ├── model_loader.py             # Carga de RT-DETR y ViTPose desde HuggingFace
│   │   ├── detector.py                 # Detección de personas con RT-DETR
│   │   ├── pose_estimator.py           # Estimación de keypoints con ViTPose
│   │   └── draw_pose.py               # Visualización del esqueleto sobre el frame
│   │
│   └── report_module/
│       ├── __init__.py
│       ├── session_data.py             # Estructura de datos de la sesión
│       └── report_generator.py        # Generación del reporte final en PDF
│
├── tests/
│   ├── test_detector.py                # Pruebas de detección de personas
│   ├── test_pose_estimator.py          # Pruebas del pipeline de estimación de pose
│   ├── test_frame_preprocessing.py     # Pruebas del preprocesamiento de frames
│   ├── test_grpc_server.py             # Pruebas del servidor gRPC
│   ├── test_model_loader.py            # Pruebas de carga de modelos
│   └── test_draw_pose.py              # Pruebas de visualización de pose
│
├── images/
│   └── person.jpg                      # Imagen de ejemplo para pruebas
│
└── README.md
```

> **Nota sobre la carpeta `videos/`:** Esta carpeta contiene los videos de prueba utilizados para verificar el funcionamiento del sistema (por ejemplo, `ejercicio1.mp4`). No se incluye en el repositorio debido a su tamaño. Para ejecutar la aplicación, cree esta carpeta en la raíz del proyecto y coloque allí los videos del ejercicio de rehabilitación.

---

## 4. Requerimientos y entorno (UV)

### 4.1 ¿Por qué se usa UV y no `requirements.txt`?

Este proyecto usa **UV** como gestor moderno de entornos y dependencias. En lugar de mantener un `requirements.txt` manual, UV trabaja con:

- **`pyproject.toml`**: declara las dependencias del proyecto (fuente de verdad).
- **`uv.lock`**: bloquea versiones exactas para garantizar reproducibilidad.

Al utilizar este gestor se fomentan prácticas adecuadas de desarrollo de software, dado que:

- Previene inconsistencias entre entornos de desarrollo.
- Permite la recreación determinística del entorno.
- Disminuye errores derivados de diferencias locales de configuración.

No es necesario un `requirements.txt`, ya que **`pyproject.toml` + `uv.lock`** cubren la instalación completa.

### 4.2 Creación del entorno e instalación de dependencias

Clone el repositorio:

```bash
git clone https://github.com/<usuario>/App_PhysicalRecovery.git
cd App_PhysicalRecovery
```

Cree el entorno virtual e instale las dependencias con **UV**:

```bash
uv venv
uv sync
```

**Activar el entorno (opcional):**

- Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

- macOS/Linux:
```bash
source .venv/bin/activate
```

Se recomienda ejecutar `uv sync` para instalar exactamente las versiones registradas en `uv.lock`.

**Nota sobre el entorno de desarrollo:**

El proyecto requiere **Python >= 3.10**. Las dependencias principales incluyen modelos de HuggingFace Transformers que se descargan automáticamente en el primer uso y se almacenan en caché local.

Dependencias principales (versiones mínimas declaradas):

```
- accelerate >= 1.12.0
- av >= 16.1.0
- grpcio < 1.78.1
- grpcio-tools < 1.78.1
- mlflow >= 3.10.1
- numpy >= 2.2.6
- opencv-python >= 4.13.0.92
- pillow >= 12.1.1
- reportlab >= 4.4.10
- scipy >= 1.15.3
- streamlit >= 1.54.0
- torch >= 2.10.0
- torchvision >= 0.25.0
- transformers >= 5.2.0
```

---

## 5. Modelos de Deep Learning

El sistema utiliza dos modelos preentrenados descargados automáticamente desde **HuggingFace** en el primer arranque:

| Modelo | Identificador HuggingFace | Propósito |
|---|---|---|
| RT-DETR | `PekingU/rtdetr_r18vd` | Detección de personas |
| ViTPose-plus | `usyd-community/vitpose-plus-base` | Estimación de 17 keypoints (COCO) |

### Keypoints COCO-17 utilizados para la evaluación

La función de cálculo de ángulo del brazo izquierdo utiliza los siguientes keypoints:

| Índice | Punto clave |
|---|---|
| 5 | Hombro izquierdo |
| 9 | Muñeca izquierda |
| 11 | Cadera izquierda |

El ángulo se calcula como el ángulo en el hombro entre el vector cadera→hombro y el vector muñeca→hombro.

---

## 6. Protocolo gRPC

La comunicación entre la interfaz y el servidor se define en `pose.proto`:

- **`PoseRequest`**: envía el frame codificado como bytes y su identificador.
- **`PoseResponse`**: retorna los keypoints detectados por persona y el ángulo calculado del brazo.

Para regenerar los archivos Python del protocolo:

```bash
python -m grpc_tools.protoc -I. --python_out=src/ --grpc_python_out=src/ pose.proto
```

---

## 7. Ejecución con Makefile

El proyecto incluye un `Makefile` con comandos abreviados para las tareas más frecuentes. Todos los comandos deben ejecutarse desde la raíz del proyecto con el entorno virtual activo.

### 7.1 Iniciar el servidor gRPC

```bash
make server
```

Inicia el servidor de inferencia en el puerto `50051`. En el primer arranque descargará los modelos desde HuggingFace (requiere conexión a internet). **Este comando debe ejecutarse antes de iniciar el cliente.**

### 7.2 Iniciar la interfaz Streamlit (cliente)

En otra terminal:

```bash
make client
```

Levanta la interfaz gráfica en el navegador. Asegúrese de que el servidor gRPC esté activo antes de ejecutar este comando.

### 7.3 Iniciar el servidor de MLflow

```bash
make serverMl
```

Inicia la UI de MLflow en el puerto `5000` para visualizar experimentos registrados. Acceda desde el navegador en `http://localhost:5000`.

### 7.4 Ejecutar experimentos MLflow

```bash
make experiments
```

Ejecuta el script `src/mlflow_experiments.py`, que lanza y registra los experimentos de evaluación del modelo en MLflow.

### 7.5 Limpiar la consola

```bash
make clear
```

Limpia la consola de Windows (`cls`).

---

## 8. Uso de la interfaz (Streamlit)

1. Asegúrese de que el servidor gRPC esté activo (`make server`) antes de abrir la interfaz.
2. Cargue un video del ejercicio de rehabilitación desde la carpeta `videos/` (formatos compatibles: `.mp4`, `.avi`, `.mov`).
3. La interfaz enviará cada frame al servidor y mostrará en tiempo real:
   - El video con el esqueleto superpuesto.
   - El ángulo actual del brazo.
   - El contador de repeticiones detectadas.
4. Al finalizar el video, se calculará el puntaje final de la sesión.
5. Se generará automáticamente un reporte en PDF con los resultados.

---

## 9. Módulos clave

- **`frame_preprocessing.py`**: valida el frame recibido, aplica redimensionamiento opcional conservando la relación de aspecto, suavizado gaussiano opcional, y convierte de BGR (OpenCV) a RGB y luego a formato PIL.

- **`model_loader.py`**: carga los procesadores y modelos de RT-DETR y ViTPose desde HuggingFace y los mueve al dispositivo disponible (CPU o GPU).

- **`detector.py`**: ejecuta la detección de objetos con RT-DETR, filtra únicamente la clase persona (label 0) y convierte las cajas del formato `xyxy` a `xywh`.

- **`pose_estimator.py`**: recibe la imagen y los bounding boxes de personas, ejecuta ViTPose y retorna los 17 keypoints con sus puntajes de confianza.

- **`draw_pose.py`**: dibuja los keypoints y las conexiones del esqueleto COCO-17 sobre el frame, coloreando cada segmento corporal de manera diferenciada.

- **`pose_rating.py`**: calcula el ángulo del brazo izquierdo, detecta repeticiones mediante un sistema de umbral de entrada/salida, y genera la calificación final como promedio del puntaje de cada repetición.

- **`grpc_server.py`**: orquesta el pipeline completo (preprocesamiento → detección → estimación → ángulo) y sirve los resultados mediante streaming gRPC bidireccional.

---

## 10. Evaluación del ejercicio

El módulo `pose_rating.py` implementa la lógica de evaluación con los siguientes parámetros:

| Parámetro | Valor | Descripción |
|---|---|---|
| `UP_THRESHOLD` | 40° | Ángulo mínimo para iniciar una repetición |
| `DOWN_THRESHOLD` | 30° | Ángulo por debajo del cual termina la repetición |
| `EXPECTED_REPS` | 4 | Número de repeticiones esperadas por sesión |

La calificación de cada repetición se calcula como `(ángulo_máximo / 90°) × 100%`, donde 90° equivale al 100% del puntaje. La calificación final es el promedio de todas las repeticiones detectadas.

---

## 11. Experimentos con MLflow

El proyecto integra **MLflow** para el registro y seguimiento de experimentos. Los resultados se almacenan en `mlflow.db`.

Para registrar y visualizar los experimentos, ejecute en terminales separadas:

```bash
# Terminal 1: iniciar la UI de MLflow
make serverMl

# Terminal 2: ejecutar los experimentos
make experiments
```

Acceda a la interfaz de MLflow desde el navegador en `http://localhost:5000`.

---

## 12. Docker

El sistema incluye un `Dockerfile` y un `docker-compose.yml` que permiten desplegar ambos servicios (servidor gRPC e interfaz Streamlit) en contenedores aislados y reproducibles, sin necesidad de configurar el entorno local manualmente.

### 12.1 Estructura de archivos Docker

```
App_PhysicalRecovery/
├── Dockerfile          # Imagen única compartida por ambos servicios
└── docker-compose.yml  # Orquestación de grpc-server y frontend
```

Se utiliza una **imagen única** para los dos contenedores. El punto de entrada (`command`) definido en `docker-compose.yml` determina qué proceso levanta cada uno.

### 12.2 Dockerfile

La imagen se construye sobre `python:3.11-slim` e instala las dependencias del sistema necesarias para OpenCV y procesamiento de video:

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY . .

# Compila el archivo .proto y mueve los stubs generados a src/
RUN uv run python -m grpc_tools.protoc \
    -I. \
    --python_out=. \
    --grpc_python_out=. \
    pose.proto

RUN mv pose_pb2.py src/ && mv pose_pb2_grpc.py src/

EXPOSE 50051
EXPOSE 8501
```

Puntos clave de la imagen:

- Usa **UV** como gestor de dependencias (`uv sync --frozen`) para instalaciones rápidas y reproducibles.
- Compila `pose.proto` automáticamente durante el build, garantizando que los stubs gRPC siempre estén sincronizados con el protocolo.
- Expone los dos puertos del sistema: `50051` (gRPC) y `8501` (Streamlit).

### 12.3 docker-compose.yml

```yaml
services:
  grpc-server:
    build: .
    container_name: recovery-grpc-server
    volumes:
      - .:/app
      - /app/.venv
    ports:
      - "50051:50051"
    environment:
      - CUDA_VISIBLE_DEVICES=-1
      - UV_LINK_MODE=copy
      - PYTHONUNBUFFERED=1
    command: uv run python src/grpc_server.py

  frontend:
    build: .
    container_name: recovery-frontend
    volumes:
      - .:/app
      - /app/.venv
    ports:
      - "8501:8501"
    depends_on:
      - grpc-server
    environment:
      - GRPC_SERVER_ADDRESS=grpc-server:50051
      - UV_LINK_MODE=copy
      - PYTHONUNBUFFERED=1
    command: uv run streamlit run src/app.py --server.address=0.0.0.0
```

El servicio `frontend` se conecta al servidor gRPC usando el nombre de red interno `grpc-server:50051`, definido mediante la variable de entorno `GRPC_SERVER_ADDRESS`. La directiva `depends_on` garantiza que el servidor arranque antes que la interfaz.

### 12.4 Construcción y ejecución

```bash
# Construir las imágenes
docker-compose build

# Levantar ambos servicios
docker-compose up

# Levantar en segundo plano
docker-compose up -d

# Detener los contenedores
docker-compose down
```

Una vez levantados, la interfaz Streamlit estará disponible en `http://localhost:8501` y el servidor gRPC escuchará en el puerto `50051`.

### 12.5 Consideraciones

- **CPU only:** la variable `CUDA_VISIBLE_DEVICES=-1` fuerza la inferencia en CPU. Para entornos con GPU, eliminar esta variable y agregar el runtime `nvidia` en el servicio `grpc-server`.
- **Caché de modelos:** los modelos de HuggingFace se descargan en el primer arranque. Para evitar descargas repetidas, montar un volumen sobre `~/.cache/huggingface`.
- **Videos de prueba:** la carpeta `videos/` queda disponible dentro del contenedor gracias al volumen `.:/app`, por lo que los archivos locales son accesibles directamente desde la interfaz.

---

## 13. Calidad de código (pre-commit + Ruff)

El proyecto usa **pre-commit** para garantizar calidad y consistencia del código antes de cada `git commit`. Los hooks se configuran en `.pre-commit-config.yaml` y las reglas de Ruff se definen en `pyproject.toml`.

### 13.1 Instalación

Con el entorno activo, ejecutar una sola vez por clon del repositorio:

```bash
uv run pre-commit install
```

A partir de ese momento, cada `git commit` ejecutará los hooks automáticamente. Para correrlos manualmente sobre todos los archivos:

```bash
uv run pre-commit run --all-files
```

### 13.2 Hooks configurados

El pipeline de pre-commit tiene tres capas:

**1. Saneamiento general de archivos** (`pre-commit-hooks v4.6.0`): elimina espacios en blanco al final de línea, asegura salto de línea final en todos los archivos, valida sintaxis de archivos YAML y TOML, detecta archivos mayores a 1 MB, y bloquea commits directos a las ramas `main` y `master`.

**2. Ruff** (`ruff-pre-commit v0.4.9`): ejecuta linting con corrección automática (`ruff --fix`) y formateo consistente del código Python (`ruff-format`). Reemplaza en una sola herramienta a flake8, black, isort y pylint. Las reglas activas incluyen: estilo PEP 8 (`E/W`), imports no usados (`F`), orden de imports (`I`), convenciones de nombres (`N`), docstrings (`D`), modernización de sintaxis (`UP`), patrones problemáticos (`B`) y simplificaciones (`SIM`).

**3. pytest**: ejecuta la suite completa de tests antes de cada commit, bloqueándolo si algún test falla.

### 13.3 Configuración de Ruff en `pyproject.toml`

El `pyproject.toml` actualizado incorpora la configuración completa de Ruff con las siguientes decisiones destacables:

- **Longitud de línea:** 88 caracteres (estándar de black).
- **Archivos excluidos del análisis:** `pose_pb2.py` y `pose_pb2_grpc.py`, ya que son generados automáticamente por `protoc` y no son código propio del proyecto.
- **Regla N802 ignorada:** la función `StreamPose` usa CamelCase porque el contrato del archivo `.proto` de gRPC lo exige; no puede renombrarse.
- **Regla E501 ignorada:** `pose_pb2.py` contiene líneas generadas automáticamente que superan ampliamente el límite; forzar su corte rompería el archivo.
- **Regla F401 ignorada:** los imports en `pose_module/__init__.py` son re-exports intencionales (patrón de módulo fachada), no imports sin usar.
- **Regla E402 ignorada:** `grpc_server.py` manipula `sys.path` antes de importar módulos propios, lo que obliga a tener imports fuera del encabezado del archivo.
- **isort configurado** con los paquetes internos del proyecto (`pose_module`, `report_module`) como `known-first-party`, separándolos visualmente de las dependencias de terceros como `torch`, `transformers` o `streamlit`.

---

## 14. Pruebas

El proyecto incluye pruebas unitarias con **pytest**. Se usan `MagicMock` y `monkeypatch` para evitar la carga de modelos reales durante las pruebas.

Para ejecutarlas:

```bash
uv run pytest
```

Cobertura de pruebas:

| Archivo de prueba | Módulo cubierto |
|---|---|
| `test_detector.py` | Detección de personas y conversión de cajas |
| `test_pose_estimator.py` | Pipeline de estimación de keypoints |
| `test_frame_preprocessing.py` | Validación, resize y conversión de frames |
| `test_grpc_server.py` | Flujo completo del servidor gRPC |
| `test_model_loader.py` | Carga de modelos y procesadores |
| `test_draw_pose.py` | Visualización del esqueleto |

---

## 15. Diagrama UML

A continuación se presenta el diagrama UML de la arquitectura modular del sistema, mostrando la organización de los módulos, sus responsabilidades y las dependencias entre componentes. Se ilustra el flujo principal desde la interfaz Streamlit hasta el servidor gRPC, el pipeline de inferencia y la generación del reporte.

<img width="2188" height="1114" alt="UML - App physical recovery (1)" src="https://github.com/user-attachments/assets/42c6cafb-6825-4638-b855-9d3f80949f74" />

## 16. Tablero Kanban


La gestión de tareas se llevó a cabo con un tablero Kanban que permitió visualizar el flujo de trabajo en columnas de backlog, en progreso y completado. El historial completo de actividades está disponible en: [Tablero Kanban](https://n9.cl/3wu03).

---

## 17. Uso académico

Este proyecto es de uso educativo. No reemplaza la supervisión de un profesional de salud en procesos de rehabilitación física.

## 18. Licencia

Este proyecto está licenciado bajo la licencia Apache-2.0. Consulte el archivo `LICENSE` para obtener más detalles.

