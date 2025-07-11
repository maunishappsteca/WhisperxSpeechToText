# Base image with CUDA 11.8
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    ffmpeg \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Copy requirements first for better caching
COPY requirements.txt .
COPY app.py .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Load environment variables
RUN touch .env

# Pre-download the default model to reduce cold start time
RUN python -c "import os; import whisperx; whisperx.load_model(os.getenv('WHISPER_MODEL', 'large-v3'), device='cuda')"

# Run the application
CMD ["python", "app.py"]
