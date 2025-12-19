FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Définir les variables d'environnement
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MODEL_NAME=Video-R1/Video-R1-7B

# Installer Python et dépendances système
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Créer un lien symbolique pour python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# Mettre à jour pip
RUN python -m pip install --upgrade pip

# Installer les dépendances Python
RUN pip install --no-cache-dir \
    torch>=2.1.0 \
    vllm==0.7.2 \
    transformers>=4.40.0 \
    huggingface-hub>=0.20.0 \
    accelerate \
    qwen-vl-utils \
    runpod \
    Pillow \
    opencv-python-headless \
    numpy \
    decord

# Créer le répertoire de travail
WORKDIR /app

# Copier le handler
COPY handler.py /app/handler.py

# Pré-télécharger le modèle (optionnel, réduit le temps de démarrage)
# RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Video-R1/Video-R1-7B')"

# Point d'entrée
CMD ["python", "-u", "handler.py"]
