import os
import uuid
import subprocess
import whisperx
import runpod
import boto3
from typing import Optional
from botocore.exceptions import ClientError

# --- Configuration ---
COMPUTE_TYPE = "float32"  # Using float32 for precision with large-v3
BATCH_SIZE = 4
S3_BUCKET = os.environ.get("S3_BUCKET_NAME")

# Initialize S3 client
s3 = boto3.client(
    's3',
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    region_name=os.environ.get("AWS_REGION", "us-east-1")
) if S3_BUCKET else None

def convert_to_wav(input_path: str) -> str:
    """Convert media to 16kHz WAV using FFmpeg"""
    output_path = f"/tmp/{uuid.uuid4()}.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-vn", "-ac", "1", "-ar", "16000",
        "-acodec", "pcm_s24le",  # 24-bit for better alignment precision
        "-loglevel", "error",
        output_path
    ], check=True)
    return output_path

def process_transcription(file_name: str, model_size: str, language: Optional[str], align: bool):
    """Core transcription workflow with explicit file handling"""
    temp_files = []
    try:
        # =====================
        # 1. DOWNLOAD FROM S3
        # =====================
        local_path = f"/tmp/{uuid.uuid4()}_{os.path.basename(file_name)}"
        print(f"Downloading {file_name} from S3 to {local_path}")
        s3.download_file(S3_BUCKET, file_name, local_path)
        temp_files.append(local_path)  # Track for cleanup

        # =====================
        # 2. CONVERT IF NEEDED
        # =====================
        if file_name.lower().endswith(('.mov', '.mp4', '.avi', '.mkv', '.mp3')):
            print("Converting media file to WAV...")
            audio_path = convert_to_wav(local_path)
            temp_files.append(audio_path)  # Track converted file
        else:
            audio_path = local_path

        # =====================
        # 3. LOAD MODEL
        # =====================
        device = "cuda" if runpod.utils.rp_utils.is_gpu_available() else "cpu"
        print(f"Loading {model_size} model on {device}...")
        model = whisperx.load_model(
            model_size,
            device=device,
            compute_type=COMPUTE_TYPE,
            language=None if language == "-" else language
        )

        # =====================
        # 4. TRANSCRIBE
        # =====================
        print("Starting transcription...")
        result = model.transcribe(audio_path, batch_size=BATCH_SIZE)
        detected_language = result.get("language", "unknown")
        used_language = detected_language if language == "-" else language

        # =====================
        # 5. ALIGNMENT (IF ENABLED)
        # =====================
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
        return {"error": str(e)}
    finally:
        # =====================
        # 6. CLEANUP FILES
        # =====================
        print("Cleaning up temporary files...")
        for f in temp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
                    print(f"Deleted: {f}")
            except Exception as cleanup_err:
                print(f"Warning: Failed to delete {f}: {cleanup_err}")

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

# --- Main Execution Block ---
if __name__ == "__main__":
    print("Initializing WhisperX Serverless Endpoint")
    
    if os.environ.get("RUNPOD_SERVERLESS_MODE") == "true":
        runpod.serverless.start({"handler": handler})
    else:
        # Local testing simulation
        test_input = {
            "input": {
                "file_name": "test-audio.wav",
                "model_size": "large-v3",
                "language": "-",
                "align": True
            }
        }
        print("Local test result:", handler(test_input))