import os
import uuid
import subprocess
import whisperx
import runpod
import boto3
import torch
import gc
from typing import Optional
from botocore.exceptions import ClientError
from huggingface_hub import snapshot_download

# --- Configuration ---
COMPUTE_TYPE = "float16"  # Changed to float16 for better performance/memory balance
BATCH_SIZE = 4  # Reduced from 8 to prevent OOM
S3_BUCKET = os.environ.get("S3_BUCKET_NAME")
MODEL_CACHE_DIR = os.getenv("WHISPER_MODEL_CACHE", "/app/models")

# Initialize S3 client with better timeout settings
s3 = None
if S3_BUCKET:
    s3_config = {
        'aws_access_key_id': os.environ.get("AWS_ACCESS_KEY_ID"),
        'aws_secret_access_key': os.environ.get("AWS_SECRET_ACCESS_KEY"),
        'region_name': os.environ.get("AWS_REGION", "us-east-1"),
        'config': boto3.session.Config(
            connect_timeout=30,
            read_timeout=300,
            retries={'max_attempts': 3}
        )
    }
    s3 = boto3.client('s3', **s3_config)

# --- Utility Functions ---
def convert_to_wav(input_path: str) -> str:
    """Convert media file to 16kHz mono WAV with better error handling"""
    output_path = f"/tmp/{uuid.uuid4()}.wav"
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-vn", "-ac", "1", "-ar", "16000",
            "-acodec", "pcm_s24le",
            "-loglevel", "error",
            output_path
        ], check=True, timeout=300)  # Added timeout
        return output_path
    except subprocess.TimeoutExpired:
        raise RuntimeError("FFmpeg conversion timed out after 5 minutes")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg conversion failed: {e.stderr.decode('utf-8') if e.stderr else str(e)}")

def cleanup_gpu_memory():
    """Clear GPU memory and cache"""
    torch.cuda.empty_cache()
    gc.collect()

def validate_model_cache(model_size: str):
    """Validate the model cache exists and is accessible"""
    expected_path = os.path.join(MODEL_CACHE_DIR, model_size)
    if not os.path.exists(expected_path):
        raise FileNotFoundError(f"Model not found at {expected_path}")
    required_files = ["config.json", "model.bin"]
    for f in required_files:
        if not os.path.exists(os.path.join(expected_path, f)):
            raise FileNotFoundError(f"Required model file {f} missing in {expected_path}")

# --- Model Loader ---
def load_cached_model(model_size: str, device: str, language: Optional[str]):
    """Load pre-downloaded model with validation and memory management"""
    try:
        validate_model_cache(model_size)
        print(f"Loading {model_size} from cache at {MODEL_CACHE_DIR}")
        
        # Clear memory before loading
        cleanup_gpu_memory()
        
        model = whisperx.load_model(
            model_size,
            device=device,
            compute_type=COMPUTE_TYPE,
            download_root=MODEL_CACHE_DIR,
            language=language if language and language != "-" else None
        )
        
        return model
    except Exception as e:
        cleanup_gpu_memory()
        raise RuntimeError(f"Model loading failed: {str(e)}")

# --- Main Transcription Logic ---
def process_transcription(file_name: str, model_size: str, language: Optional[str], align: bool):
    temp_files = []
    try:
        # Validate inputs
        if not file_name or not isinstance(file_name, str):
            return {"error": "Invalid file_name provided"}
            
        if not s3:
            return {"error": "S3 client not initialized"}

        # Step 1: Download from S3
        local_path = f"/tmp/{uuid.uuid4()}_{os.path.basename(file_name)}"
        print(f"Downloading {file_name} from S3 → {local_path}")
        try:
            s3.download_file(S3_BUCKET, file_name, local_path)
        except ClientError as e:
            return {"error": f"S3 download failed: {e.response['Error']['Message']}"}
        temp_files.append(local_path)

        # Step 2: Convert to WAV if needed
        try:
            if not file_name.lower().endswith('.wav'):
                print("Converting media file to WAV...")
                audio_path = convert_to_wav(local_path)
                temp_files.append(audio_path)
            else:
                audio_path = local_path
        except Exception as e:
            return {"error": f"Audio conversion failed: {str(e)}"}

        # Step 3: Load Whisper model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model '{model_size}' on device: {device}")
        try:
            model = load_cached_model(model_size, device, language)
        except Exception as e:
            return {"error": f"Model loading failed: {str(e)}"}

        # Step 4: Transcription
        print("Transcribing...")
        try:
            result = model.transcribe(audio_path, batch_size=BATCH_SIZE)
            detected_language = result.get("language", "unknown")
            used_language = detected_language if language == "-" else language
        except Exception as e:
            return {"error": f"Transcription failed: {str(e)}"}

        # Step 5: Alignment
        if align and used_language and used_language != "unknown":
            print(f"Aligning segments for language: {used_language}")
            try:
                align_model, metadata = whisperx.load_align_model(
                    language_code=used_language,
                    device=device
                )
                result = whisperx.align(
                    result["segments"],
                    align_model,
                    metadata,
                    audio_path,
                    device=device,
                    return_char_alignments=False
                )
                # Clean up alignment model
                del align_model, metadata
                cleanup_gpu_memory()
            except Exception as e:
                print(f"Alignment failed (proceeding without alignment): {str(e)}")

        # Step 6: Return results
        return {
            "text": " ".join(seg["text"] for seg in result["segments"]),
            "segments": result["segments"],
            "detected_language": detected_language if language == "-" else None,
            "used_language": used_language,
            "model_used": model_size,
            "compute_type": COMPUTE_TYPE,
            "device_used": device,
            "processed_file": file_name
        }

    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}
    finally:
        print("Cleaning up resources...")
        # Clean up temporary files
        for f in temp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception as e:
                print(f"Warning: Could not delete {f}: {str(e)}")
        # Clean up GPU memory
        cleanup_gpu_memory()

# --- RunPod Handler ---
def handler(job):
    print("Job received:", job)

    if not S3_BUCKET:
        return {"error": "S3_BUCKET_NAME environment variable not set"}

    try:
        input_data = job.get("input", {})
        if not input_data:
            return {"error": "No input data provided"}

        required_params = ["file_name"]
        missing_params = [p for p in required_params if p not in input_data]
        if missing_params:
            return {"error": f"Missing required parameters: {', '.join(missing_params)}"}

        return process_transcription(
            file_name=input_data["file_name"],
            model_size=input_data.get("model_size", "large-v3"),
            language=input_data.get("language", "-"),
            align=input_data.get("align", False)
        )

    except Exception as e:
        return {"error": f"Handler error: {str(e)}"}

# --- Entry Point ---
if __name__ == "__main__":
    print("Starting WhisperX Endpoint...")
    print(f"Torch CUDA available: {torch.cuda.is_available()}")
    print(f"Torch CUDA version: {torch.version.cuda}")
    print(f"Torch cuDNN enabled: {torch.backends.cudnn.enabled}")

    if os.environ.get("RUNPOD_SERVERLESS_MODE") == "true":
        runpod.serverless.start({"handler": handler})
    else:
        # Local test
        print("Running local test...")
        test_input = {
            "input": {
                "file_name": "test-audio.wav",
                "model_size": "large-v3",
                "language": "-",
                "align": True
            }
        }
        print("Test Result:", handler(test_input))
