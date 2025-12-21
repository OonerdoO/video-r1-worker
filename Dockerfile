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
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Installer les dépendances Python de base
RUN pip install --no-cache-dir \
    runpod \
    Pillow \
    opencv-python-headless \
    numpy \
    wandb==0.18.3 \
    tensorboardx \
    torchvision

# Installer qwen_vl_utils avec support decord
RUN pip install --no-cache-dir qwen_vl_utils[decord]

# Installer flash-attn
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# Installer vLLM 0.7.2 (version officielle Video-R1)
RUN pip install --no-cache-dir vllm==0.7.2

# Installer la version spécifique de transformers compatible avec Video-R1
# Cette version est nécessaire car Qwen2.5-VL change fréquemment
RUN pip install --no-cache-dir git+https://github.com/huggingface/transformers.git@336dc69d63d56f232a183a3e7f52790429b871ef

# Créer le répertoire de travail
WORKDIR /app

# Copier le handler
COPY handler.py /app/handler.py

# Point d'entrée
CMD ["python", "-u", "handler.py"]
