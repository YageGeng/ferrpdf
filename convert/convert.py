import sys
import os
from pathlib import Path
import argparse
from huggingface_hub import hf_hub_download
from ultralytics import YOLO


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


def main():
    parser = argparse.ArgumentParser(description="Download model from HuggingFace and convert to ONNX format")
    parser.add_argument("repo_id", nargs='?', default="hantian/yolo-doclaynet", 
                       help="HuggingFace repository ID")
    parser.add_argument("filename", nargs='?', default="yolov12s-doclaynet.pt", 
                       help="Filename to download")
    parser.add_argument("--output-name", "-o", help="Output model name (without extension)")
    parser.add_argument("--models-dir", "-d", default="models", 
                       help="Model storage directory (default: models)")
    
    args = parser.parse_args()
    
    # Download model
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