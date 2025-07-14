# Use NVIDIA CUDA 12.1 base image with cuDNN
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set library paths
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda-12.1
ENV HF_HUB_CACHE=/app/models
ENV WHISPER_MODEL_CACHE=/app/models

# Install Python dependencies with pinned versions
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt




# Pre-download model with validation
RUN mkdir -p /app/models && \
    python -c "\
    from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='openai/whisper-large-v3', \
    local_dir='/app/models/large-v3', \
    local_dir_use_symlinks=False, \
    token=os.getenv('HF_TOKEN'))"
    

COPY app.py .

CMD ["python", "app.py"]
