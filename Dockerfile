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

RUN uv sync --frozen && uv cache clean

COPY . .

# fix: .venv/bin/python en vez de uv run python
RUN .venv/bin/python -m grpc_tools.protoc \
    -I. --python_out=. --grpc_python_out=. pose.proto \
    && mv pose_pb2.py src/ && mv pose_pb2_grpc.py src/

EXPOSE 50051
EXPOSE 8501