import os
import uuid
import subprocess
import whisperx
import runpod
import boto3
import gc
from typing import Optional
from botocore.exceptions import ClientError

# --- Configuration ---
COMPUTE_TYPE = "int8"  # Using int8 for CPU optimization
BATCH_SIZE = 4  # Reduced batch size for CPU
S3_BUCKET = os.environ.get("S3_BUCKET_NAME")
MODEL_CACHE_DIR = os.getenv("WHISPER_MODEL_CACHE", "/app/models")

# Initialize S3 client
s3 = boto3.client('s3') if S3_BUCKET else None

def convert_to_wav(input_path: str) -> str:
    """Convert media file to 16kHz mono WAV"""
    output_path = f"/tmp/{uuid.uuid4()}.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-vn", "-ac", "1", "-ar", "16000",
        "-acodec", "pcm_s16le",
        "-loglevel", "error",
        output_path
    ], check=True)
    return output_path

def load_model(model_size: str, language: Optional[str]):
    """Load Whisper model with CPU optimization"""
    return whisperx.load_model(
        model_size,
        device="cpu",
        compute_type=COMPUTE_TYPE,
        download_root=MODEL_CACHE_DIR,
        language=language if language and language != "-" else None
    )

def transcribe_audio(audio_path: str, model_size: str, language: Optional[str], align: bool):
    """Core transcription logic"""
    model = load_model(model_size, language)
    result = model.transcribe(audio_path, batch_size=BATCH_SIZE)
    detected_language = result.get("language", language if language else "en")
    
    if align and detected_language != "unknown":
        try:
            align_model, metadata = whisperx.load_align_model(
                language_code=detected_language,
                device="cpu"
            )
            result = whisperx.align(
                result["segments"],
                align_model,
                metadata,
                audio_path,
                device="cpu",
                return_char_alignments=False
            )
        except Exception as e:
            print(f"Alignment skipped: {str(e)}")
    
    return {
        "text": " ".join(seg["text"] for seg in result["segments"]),
        "segments": result["segments"],
        "language": detected_language,
        "model": model_size
    }

def handler(job):
    """RunPod serverless handler"""
    try:
        input_data = job["input"]
        file_name = input_data["file_name"]
        
        # 1. Download from S3
        local_path = f"/tmp/{uuid.uuid4()}_{os.path.basename(file_name)}"
        s3.download_file(S3_BUCKET, file_name, local_path)
        
        # 2. Convert to WAV if needed
        if not file_name.lower().endswith('.wav'):
            audio_path = convert_to_wav(local_path)
            os.remove(local_path)
        else:
            audio_path = local_path
        
        # 3. Transcribe
        result = transcribe_audio(
            audio_path,
            input_data.get("model_size", "large-v3"),
            input_data.get("language", None),
            input_data.get("align", False)
        )
        
        # 4. Cleanup
        os.remove(audio_path)
        gc.collect()
        
        return result
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print("Starting WhisperX CPU Endpoint...")
    
    if os.environ.get("RUNPOD_SERVERLESS_MODE") == "true":
        runpod.serverless.start({"handler": handler})
    else:
        # Test with mock input
        test_result = handler({
            "input": {
                "file_name": "test.wav",
                "model_size": "base",
                "align": True
            }
        })
        print("Test Result:", test_result)
