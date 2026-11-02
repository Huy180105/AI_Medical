# MedFusion-CXR — Multimodal Multi-Agent Clinical Decision Support System
# Vision: DenseNet-121 (NIH ChestX-ray14) · Clinical: XGBoost + TreeSHAP · Fusion: Gated (GMU prototype)
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# libgomp1 is required at runtime by XGBoost / LightGBM (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first to avoid pulling multi-GB CUDA wheels
RUN pip install --upgrade pip \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .

# Model weights are bind-mounted into /app/models at runtime (see docker-compose.yml)
RUN mkdir -p /app/models /app/output /app/data/uploads /app/.cache/huggingface

ENV API_HOST=0.0.0.0 \
    API_PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -fsS http://localhost:8000/api/v1/treatment/benchmark > /dev/null || exit 1

CMD ["python", "main.py", "--mode", "api"]
