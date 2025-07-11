# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy Python files
COPY requirements.txt .
COPY app.py .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the model
# RUN python -c "import os; import whisperx; whisperx.load_model(os.getenv('WHISPER_MODEL', 'large-v3'), device='cpu')"

# Pre-download model to container (using CPU during build)
ENV WHISPER_MODEL=large-v3
RUN python -c "import os; import whisperx; whisperx.load_model(os.getenv('WHISPER_MODEL'), device='cpu', download_root='/app/models')"

# Set environment variable for runtime model location
ENV WHISPER_MODEL_CACHE=/app/models

CMD ["python", "app.py"]
