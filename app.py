import os
import uuid
import subprocess
import whisperx
import runpod
import boto3
from typing import Optional
from botocore.exceptions import ClientError
from huggingface_hub import snapshot_download

# --- Configuration ---
COMPUTE_TYPE = "float32"  # Use "int8" for faster/lower memory
BATCH_SIZE = 4
S3_BUCKET = os.environ.get("S3_BUCKET_NAME")
MODEL_CACHE_DIR = os.getenv("WHISPER_MODEL_CACHE", "/app/models")  # e.g. /app/models

# Initialize S3 client (optional)
s3 = boto3.client(
    's3',
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    region_name=os.environ.get("AWS_REGION", "us-east-1")
) if S3_BUCKET else None


# --- Utility: Convert to 16kHz mono WAV ---
def convert_to_wav(input_path: str) -> str:
    output_path = f"/tmp/{uuid.uuid4()}.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-vn", "-ac", "1", "-ar", "16000",
        "-acodec", "pcm_s24le",
        "-loglevel", "error",
        output_path
    ], check=True)
    return output_path


# --- Model Loader ---
# --- Improved Model Loader ---
def load_cached_model(model_size: str, device: str, language: Optional[str]):
    """Load pre-downloaded model with validation"""
    expected_path = os.path.join(MODEL_CACHE_DIR, model_size)
    
    if not os.path.exists(expected_path):
        raise RuntimeError(f"Model not found at {expected_path}. Must pre-download during build.")
    
    print(f"Loading {model_size} from cache at {expected_path}")
    return whisperx.load_model(
        model_size,
        device=device,
        compute_type=COMPUTE_TYPE,
        download_root=MODEL_CACHE_DIR  # Critical for cache usage
    )



# --- Main Transcription Logic ---
def process_transcription(file_name: str, model_size: str, language: Optional[str], align: bool):
    temp_files = []
    try:
        # Step 1: Download from S3
        local_path = f"/tmp/{uuid.uuid4()}_{os.path.basename(file_name)}"
        print(f"Downloading {file_name} from S3 → {local_path}")
        s3.download_file(S3_BUCKET, file_name, local_path)
        temp_files.append(local_path)

        # Step 2: Convert to WAV
        if file_name.lower().endswith(('.mov', '.mp4', '.avi', '.mkv', '.mp3')):
            print("Converting media file to WAV...")
            audio_path = convert_to_wav(local_path)
            temp_files.append(audio_path)
        else:
            audio_path = local_path

        # Step 3: Load Whisper model
        device = "cuda" if os.environ.get("RUNPOD_SERVERLESS_MODE") == "true" else "cpu"
        print(f"Loading model '{model_size}' on device: {device}")
        model = load_cached_model(model_size, device, language)

        # Step 4: Transcription
        print("Transcribing...")
        result = model.transcribe(audio_path, batch_size=BATCH_SIZE)
        detected_language = result.get("language", "unknown")
        used_language = detected_language if language == "-" else language

        # Step 5: Alignment
        if align and used_language and used_language != "unknown":
            print(f"Aligning segments for language: {used_language}")
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

    except ClientError as e:
        return {"error": f"S3 download failed: {e.response['Error']['Message']}"}
    except subprocess.CalledProcessError as e:
        return {"error": f"FFmpeg conversion failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}
    finally:
        print("Cleaning up temporary files...")
        for f in temp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
                    print(f"Deleted: {f}")
            except Exception as cleanup_err:
                print(f"Warning: Could not delete {f}: {cleanup_err}")


# --- RunPod Handler ---
def handler(job):
    print("Job received:", job)

    if not S3_BUCKET:
        return {"error": "S3_BUCKET_NAME environment variable not set"}

    try:
        input_data = job["input"]
        required_params = ["file_name"]
        if not all(k in input_data for k in required_params):
            return {"error": f"Missing required parameters. Needed: {required_params}"}

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
