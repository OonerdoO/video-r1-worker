FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV MODEL_NAME=Video-R1/Video-R1-7B
ENV HF_HOME=/runpod-volume/huggingface

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Installer vLLM dernière version (supporte Qwen2.5-VL)
RUN pip install --no-cache-dir \
    vllm>=0.6.0 \
    transformers>=4.45.0 \
    huggingface-hub>=0.25.0 \
    accelerate \
    qwen-vl-utils \
    runpod \
    Pillow \
    opencv-python-headless \
    numpy \
    decord \
    av

# Créer le répertoire de travail
WORKDIR /app

# Copier le handler
COPY handler.py /app/handler.py

# Point d'entrée
CMD ["python", "-u", "handler.py"]
