import argparse
import os
from pathlib import Path
import shutil
import sys

from huggingface_hub import hf_hub_download
from ultralytics import YOLO
try:
    from modelscope import snapshot_download, model_file_download
    MODELSCOPE_AVAILABLE = True
except ImportError:
    MODELSCOPE_AVAILABLE = False
    print("Warning: modelscope not available. Install with: pip install modelscope")


def download_from_hf(repo_id, filename, local_dir="models"):
    """Download specified file from Hugging Face"""
    print(f"Downloading {filename} from {repo_id}...")

    # Ensure local directory exists
    os.makedirs(local_dir, exist_ok=True)

    try:
        # Download file
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print(f"Download completed: {downloaded_path}")
        return downloaded_path
    except Exception as e:
        print(f"Download failed: {e}")
        return None


def download_from_modelscope(repo_id, filename=None, local_dir="models"):
    """Download model or specific file from ModelScope"""
    if not MODELSCOPE_AVAILABLE:
        print("Error: modelscope is not available. Please install it with: pip install modelscope")
        return None

    # Ensure local directory exists
    os.makedirs(local_dir, exist_ok=True)

    try:
        if filename:
            # Download specific file
            print(f"Downloading {filename} from ModelScope: {repo_id}...")
            downloaded_path = model_file_download(
                model_id=repo_id,
                file_path=filename,
                cache_dir=local_dir
            )
            print(f"File download completed: {downloaded_path}")
        else:
            # Download entire model
            print(f"Downloading entire model from ModelScope: {repo_id}...")
            downloaded_path = snapshot_download(
                model_id=repo_id,
                cache_dir=local_dir
            )
            print(f"Model download completed: {downloaded_path}")

        return downloaded_path
    except Exception as e:
        print(f"Download failed: {e}")
        return None


def download_rapidocr_models(local_dir="models"):
    """Download RapidOCR models from ModelScope"""
    if not MODELSCOPE_AVAILABLE:
        print("Error: modelscope is not available. Please install it with: pip install modelscope")
        return None

    print("Downloading RapidOCR models from ModelScope...")

    # Ensure local directory exists
    os.makedirs(local_dir, exist_ok=True)

    try:
        # Download RapidOCR model
        downloaded_path = snapshot_download(
            model_id="RapidAI/RapidOCR",
            cache_dir=local_dir
        )
        print(f"RapidOCR model download completed: {downloaded_path}")

        # Check if the specific ONNX files exist
        det_model_path = os.path.join(downloaded_path, "onnx/PP-OCRv5/det/ch_PP-OCRv5_mobile_det.onnx")
        rec_model_path = os.path.join(downloaded_path, "onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile_infer.onnx")

        if os.path.exists(det_model_path):
            print(f"Detection model found: {det_model_path}")
        else:
            print(f"Warning: Detection model not found at {det_model_path}")

        if os.path.exists(rec_model_path):
            print(f"Recognition model found: {rec_model_path}")
        else:
            print(f"Warning: Recognition model not found at {rec_model_path}")

        return downloaded_path
    except Exception as e:
        print(f"Download failed: {e}")
        return None


def convert_model_to_onnx(model_path, output_name=None, output_dir="models"):
    """Convert YOLO model to ONNX format"""
    print(f"Converting model: {model_path}")

    try:
        # Load model
        model = YOLO(model_path)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Generate output filename
        if output_name is None:
            model_name = Path(model_path).stem
            output_name = f"{model_name}.onnx"
        elif not output_name.endswith('.onnx'):
            output_name = f"{output_name}.onnx"

        output_path = os.path.join(output_dir, output_name)

        # Export to ONNX
        model.export(format="onnx", device='cpu')

        # Move generated ONNX file to specified location
        generated_onnx = model_path.replace('.pt', '.onnx')
        if os.path.exists(generated_onnx) and generated_onnx != output_path:
            os.rename(generated_onnx, output_path)

        print(f"Conversion completed: {output_path}")
        return output_path
    except Exception as e:
        print(f"Conversion failed: {e}")
        return None


def move_and_cleanup_modelscope_files(det_path, rec_path, models_dir):
    """Move downloaded files to models directory and cleanup temporary directories"""
    try:
        # Get the base directory where ModelScope downloaded the files
        det_dir = os.path.dirname(det_path)
        rec_dir = os.path.dirname(rec_path)

        # Extract filenames
        det_filename = os.path.basename(det_path)
        rec_filename = os.path.basename(rec_path)

        # Define target paths in models directory
        target_det_path = os.path.join(models_dir, det_filename)
        target_rec_path = os.path.join(models_dir, rec_filename)

        print(f"Moving detection model to: {target_det_path}")
        shutil.move(det_path, target_det_path)

        print(f"Moving recognition model to: {target_rec_path}")
        shutil.move(rec_path, target_rec_path)

        # Cleanup temporary directories
        print("Cleaning up temporary directories...")

        # Find and remove RapidAI directory
        rapidai_dir = None
        for root, dirs, files in os.walk(models_dir):
            if "RapidAI" in dirs:
                rapidai_dir = os.path.join(root, "RapidAI")
                break

        if rapidai_dir and os.path.exists(rapidai_dir):
            print(f"Removing temporary directory: {rapidai_dir}")
            shutil.rmtree(rapidai_dir)

        # Find and remove ._____temp directory
        temp_dir = None
        for root, dirs, files in os.walk(models_dir):
            if "._____temp" in dirs:
                temp_dir = os.path.join(root, "._____temp")
                break

        if temp_dir and os.path.exists(temp_dir):
            print(f"Removing temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)

        # Remove .cache directory
        cache_dir = os.path.join(models_dir, ".cache")
        if os.path.exists(cache_dir):
            print(f"Removing cache directory: {cache_dir}")
            shutil.rmtree(cache_dir)

        print("Cleanup completed successfully!")
        return target_det_path, target_rec_path

    except Exception as e:
        print(f"Error during file moving and cleanup: {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(description="Download models from HuggingFace/ModelScope and convert to ONNX format")
    parser.add_argument("--source", choices=["hf", "modelscope", "rapidocr"], default="modelscope",
                       help="Source to download from: hf (HuggingFace), modelscope, or rapidocr (default: modelscope)")
    parser.add_argument("repo_id", nargs='?', default="RapidAI/RapidOCR",
                       help="Repository ID (for HuggingFace or ModelScope, default: RapidAI/RapidOCR)")
    parser.add_argument("filename", nargs='?', default="onnx/PP-OCRv5/det/ch_PP-OCRv5_mobile_det.onnx",
                       help="Filename to download (for HuggingFace and ModelScope, default: detection model)")
    parser.add_argument("--output-name", "-o", help="Output model name (without extension)")
    parser.add_argument("--models-dir", "-d", default="models",
                       help="Model storage directory (default: models)")

    args = parser.parse_args()

    if args.source == "rapidocr":
        # Download RapidOCR models (Paddle models)
        print("Downloading Paddle OCR models (RapidOCR)...")
        downloaded_path = download_rapidocr_models(args.models_dir)
        if downloaded_path is None:
            sys.exit(1)
        print(f"Paddle OCR models downloaded to: {downloaded_path}")
        print("Models available:")
        print("- Detection: onnx/PP-OCRv5/det/ch_PP-OCRv5_mobile_det.onnx")
        print("- Recognition: onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile_infer.onnx")

    elif args.source == "modelscope":
        # Download from ModelScope (entire model or specific file)
        if args.repo_id == "RapidAI/RapidOCR" and args.filename == "onnx/PP-OCRv5/det/ch_PP-OCRv5_mobile_det.onnx":
            # Default behavior: download both detection and recognition models
            print("Downloading Paddle OCR models (detection and recognition)...")

            # Download detection model
            print("Downloading detection model...")
            det_path = download_from_modelscope("RapidAI/RapidOCR", "onnx/PP-OCRv5/det/ch_PP-OCRv5_mobile_det.onnx", args.models_dir)
            if det_path is None:
                sys.exit(1)
            print(f"Detection model downloaded to: {det_path}")

            # Download recognition model
            print("Downloading recognition model...")
            rec_path = download_from_modelscope("RapidAI/RapidOCR", "onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile_infer.onnx", args.models_dir)
            if rec_path is None:
                sys.exit(1)
            print(f"Recognition model downloaded to: {rec_path}")

            # Move files to models directory and cleanup
            print("Moving files to models directory and cleaning up...")
            final_det_path, final_rec_path = move_and_cleanup_modelscope_files(det_path, rec_path, args.models_dir)

            if final_det_path and final_rec_path:
                print("Both Paddle OCR models downloaded and moved successfully!")
                print("Models available:")
                print(f"- Detection: {final_det_path}")
                print(f"- Recognition: {final_rec_path}")
            else:
                print("Error: Failed to move files or cleanup directories")
                sys.exit(1)

        else:
            # Download specific file or entire model
            downloaded_path = download_from_modelscope(args.repo_id, args.filename, args.models_dir)
            if downloaded_path is None:
                sys.exit(1)
            if args.filename:
                print(f"ModelScope file downloaded to: {downloaded_path}")
            else:
                print(f"ModelScope model downloaded to: {downloaded_path}")

    else:
        # Download from HuggingFace and convert
        downloaded_path = download_from_hf(args.repo_id, args.filename, args.models_dir)
        if downloaded_path is None:
            sys.exit(1)

        # Convert model
        output_path = convert_model_to_onnx(downloaded_path, args.output_name, args.models_dir)
        if output_path is None:
            sys.exit(1)

        print(f"All operations completed! ONNX model saved at: {output_path}")


if __name__ == "__main__":
    main()
