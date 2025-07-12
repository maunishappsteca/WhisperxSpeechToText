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
# RUN pip install --no-cache-dir -r requirements.txt



# Pre-download the model
# RUN python -c "import os; import whisperx; whisperx.load_model(os.getenv('WHISPER_MODEL', 'large-v3'), device='cpu')"

# Create model cache directory (adjust path if using different WHISPER_MODEL_CACHE)
# RUN mkdir -p /app/models && chmod -R 777 /app/models    



# Set environment variables for model name and cache directory
ENV WHISPER_MODEL=large-v3
ENV WHISPER_MODEL_CACHE=/app/models
ENV WHISPER_LANGUAGE=en

RUN mkdir -p $WHISPER_MODEL_CACHE && chmod -R 777 $WHISPER_MODEL_CACHE

# Pre-download the whisper model using whisper and whisperx
RUN python -c "\
import os; \
import whisper; \
import whisperx; \
model_name = os.environ.get('WHISPER_MODEL', 'large-v3'); \
download_root = os.environ.get('WHISPER_MODEL_CACHE', '/app/models'); \
print(f'Downloading Whisper model: {model_name}'); \
whisper.load_model(model_name, download_root=download_root); \
print('Passing loaded model to WhisperX...');




    
CMD ["python", "app.py"]














