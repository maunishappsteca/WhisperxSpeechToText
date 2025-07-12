# Use NVIDIA CUDA 12.1 base image (supports Ada GPUs)
FROM nvidia/cuda:12.1.1-base-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    python3-pip \
    libcudnn8=8.9.4.*-1+cuda12.1 \  # Explicit cuDNN for Ada
    && rm -rf /var/lib/apt/lists/*

# Set library paths for Ada GPUs
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda-12.1

# Install Python dependencies with CUDA 12.1 support
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download model (same as before)
RUN mkdir -p /app/models && \
    python -c "\
    from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='openai/whisper-large-v3', \
    local_dir='/app/models/large-v3', \
    local_dir_use_symlinks=False)"

COPY app.py .

CMD ["python", "app.py"]
