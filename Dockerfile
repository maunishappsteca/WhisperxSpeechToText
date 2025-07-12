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

# Create model cache directory (adjust path if using different WHISPER_MODEL_CACHE)
# RUN mkdir -p /app/models && chmod -R 777 /app/models



# Create model cache directory (with fallback)
RUN mkdir -p "$(python -c 'import os; print(os.getenv(\"WHISPER_MODEL_CACHE\", \"/app/models\"))')" && \
    chmod -R 777 "$(python -c 'import os; print(os.getenv(\"WHISPER_MODEL_CACHE\", \"/app/models\"))')"
    

# Pre-download model to container (using CPU during build)
# RUN python -c "import os; import whisperx; whisperx.load_model(os.getenv('WHISPER_MODEL'), device='cpu', download_root=os.getenv('WHISPER_MODEL_CACHE'))"



# Pre-download model using environment variables
RUN python -c "import os; import whisperx; \
    model = os.getenv('WHISPER_MODEL', 'large-v3'); \
    cache = os.getenv('WHISPER_MODEL_CACHE', '/app/models'); \
    print(f'Downloading {model} to {cache}'); \
    whisperx.load_model(model, device='cpu', download_root=cache)"
    

# Verify model download
RUN echo "Verifying model download..." && \
    MODEL_PATH=$(python -c "import os; print(os.path.join(os.getenv('WHISPER_MODEL_CACHE', '/app/models'), os.getenv('WHISPER_MODEL', 'large-v3'))") && \
    echo "Checking path: $MODEL_PATH" && \
    if [ -d "$MODEL_PATH" ]; then \
        echo "Model downloaded successfully:" && \
        ls -lh "$MODEL_PATH" && \
        du -sh "$MODEL_PATH" && \
        [ -f "$MODEL_PATH/model.bin" ] && echo "Model files verified"; \
    else \
        echo "ERROR: Model download failed - directory not found"; \
        exit 1; \
    fi
    

# Cleanup temporary files
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
    
CMD ["python", "app.py"]














