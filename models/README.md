# Model Download and Conversion Script

This script downloads model files from Hugging Face, ModelScope, and specifically RapidOCR models (Paddle OCR), with support for converting YOLO models to ONNX format.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Default: Download Paddle OCR detection and recognition models
python download.py

# Or explicitly specify
python download.py --source modelscope
```

### Parameters

- `--source`: Source to download from: `hf` (HuggingFace), `modelscope`, or `rapidocr` (default: `modelscope`)
- `repo_id`: Repository ID (optional, default: "RapidAI/RapidOCR" for ModelScope)
- `filename`: File name to download (for HuggingFace and ModelScope, optional, default: detection model)
- `--output-name, -o`: Output model name (without extension, for HuggingFace conversion only)
- `--models-dir, -d`: Model storage directory (default: models)

### Examples

#### 1. Download Paddle OCR Models (Default - Recommended for OCR)

```bash
# Download Paddle OCR models from ModelScope (default behavior)
python download.py

# Or explicitly specify ModelScope
python download.py --source modelscope
```

This will download both Paddle OCR models:
- `onnx/PP-OCRv5/det/ch_PP-OCRv5_mobile_det.onnx` (detection model)
- `onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile_infer.onnx` (recognition model)

**Note**: The script will automatically move the downloaded files to the `models/` directory and clean up temporary directories (`RapidAI/`, `._____temp/`, and `.cache/`).

## Output

- Downloaded model files are saved in the specified model directory
- For HuggingFace models: Converted ONNX files are also saved in the same directory
- For Paddle OCR: ONNX models are automatically moved to the models directory and ready to use
- Temporary directories are automatically cleaned up after download (RapidAI/, ._____temp/, .cache/)
- By default, all files are saved in the `models/` directory

#### 2. Download from ModelScope

```bash
# Download entire model from ModelScope
python download.py --source modelscope "some/model/id"

# Download specific file from ModelScope
python download.py --source modelscope "some/model/id" "path/to/specific/file.onnx"

# Download only detection model
python download.py --source modelscope "RapidAI/RapidOCR" "onnx/PP-OCRv5/det/ch_PP-OCRv5_mobile_det.onnx"

# Download only recognition model
python download.py --source modelscope "RapidAI/RapidOCR" "onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile_infer.onnx"
```

#### 3. Download and Convert from HuggingFace

```bash
# Download and convert YOLOv8n model
python download.py --source hf ultralytics/yolov8 yolov8n.pt

# Specify output filename
python download.py --source hf ultralytics/yolov8 yolov8n.pt --output-name my_yolo_model

# Specify custom model directory
python download.py --source hf ultralytics/yolov8 yolov8n.pt --models-dir ./custom_models

# Download and convert YOLOv11 model
python download.py --source hf ultralytics/yolo11 yolo11n.pt -o yolo11_nano
```

## Supported Sources

### 1. Paddle OCR (ModelScope - Default)
- **Repository**: `RapidAI/RapidOCR`
- **Models**: Pre-trained OCR detection and recognition models
- **Format**: ONNX (ready to use)
- **Use Case**: Document OCR, text detection and recognition
- **Default**: Yes, this is the default download source
- **Default Files**: Downloads both detection and recognition models automatically

### 2. ModelScope
- **Repository**: Any ModelScope model ID
- **Models**: Various AI models available on ModelScope
- **Format**: Depends on the model
- **Use Case**: General model downloading

### 3. HuggingFace
- **Repository**: Any HuggingFace repository
- **Models**: PyTorch models (converted to ONNX)
- **Format**: PyTorch (.pt) → ONNX (.onnx)
- **Use Case**: YOLO models and other PyTorch models

## Notes

- Ensure sufficient disk space for storing downloaded model files
- For HuggingFace models: Conversion process may take time depending on model size
- Network connection required for downloading from HuggingFace/ModelScope
- For HuggingFace models: Conversion process uses CPU, may be slow for large models
- Paddle OCR models are already in ONNX format and ready for inference

## Error Handling

The script includes error handling mechanisms:
- If download fails, the script will display error message and exit
- If conversion fails (HuggingFace only), the script will display error message and exit
- Graceful handling when modelscope is not installed
- All operations have corresponding progress indicators

## Supported Model Formats

### Input Formats
- **HuggingFace**: PyTorch (.pt) model files
- **ModelScope**: Various formats depending on the model
- **Paddle OCR**: Pre-packaged ONNX models

### Output Formats
- **HuggingFace**: ONNX (.onnx) model files
- **ModelScope**: Original format (no conversion)
- **Paddle OCR**: ONNX (.onnx) model files (ready to use)

## Advanced Usage

### Custom Repositories and Models

```bash
# Download from specific HuggingFace repository
python download.py --source hf hantian/yolo-doclaynet yolov8n-doclaynet.pt

# Download entire model from ModelScope
python download.py --source modelscope "damo/cv_resnet50_face-detection_retinaface"

# Download specific file from ModelScope
python download.py --source modelscope "RapidAI/RapidOCR" "onnx/PP-OCRv5/det/ch_PP-OCRv5_mobile_det.onnx"

# Download specific file from other ModelScope models
python download.py --source modelscope "damo/cv_resnet50_face-detection_retinaface" "model.pth"

# Download specialized trained models
python download.py --source hf ultralyticsplus/yolov8s_worldv2 yolov8s-worldv2.pt
```

### Batch Processing Multiple Files

Create a configuration file `models_config.txt`:
```
rapidocr
hf,ultralytics/yolov8,yolov8n.pt,yolo8n_detection
hf,ultralytics/yolov8,yolov8s.pt,yolo8s_detection
hf,hantian/yolo-doclaynet,yolov8n-doclaynet.pt,yolo8n_layout
modelscope,damo/cv_resnet50_face-detection_retinaface
modelscope,RapidAI/RapidOCR,onnx/PP-OCRv5/det/ch_PP-OCRv5_mobile_det.onnx
modelscope,RapidAI/RapidOCR,onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile_infer.onnx
```

Then use script for batch processing:
```bash
while IFS=',' read -r source repo file output || [ -n "$source" ]; do
    if [ "$source" = "rapidocr" ]; then
        python download.py --source rapidocr
    elif [ "$source" = "modelscope" ]; then
        if [ -n "$file" ]; then
            python download.py --source modelscope "$repo" "$file"
        else
            python download.py --source modelscope "$repo"
        fi
    else
        python download.py --source hf "$repo" "$file" --output-name "$output"
    fi
done < models_config.txt
```

### Environment Variable Configuration

Set default model directory:
```bash
export YOLO_MODELS_DIR="/path/to/your/models"
python download.py --source hf ultralytics/yolov8 yolov8n.pt --models-dir "$YOLO_MODELS_DIR"
```

## Performance Optimization

1. **Disk Space**: Ensure sufficient space (each model ~6MB-200MB)
2. **Network Connection**: Stable network connection required for initial download
3. **CPU Resources**: Conversion process consumes CPU resources (HuggingFace only), recommend running during low load
4. **Memory Usage**: Large model conversion may require 4GB+ memory (HuggingFace only)

## Troubleshooting

### Common Issues

1. **Download Timeout**
```bash
# Set longer timeout for HuggingFace
export HF_HUB_DOWNLOAD_TIMEOUT=300
python download.py --source hf ultralytics/yolov8 yolov8n.pt
```

2. **Memory Insufficient (HuggingFace conversion only)**
```bash
# For large models, try reducing batch size
python download.py --source hf ultralytics/yolov8 yolov8n.pt  # Process one at a time
```

3. **Network Proxy Settings**
```bash
# If proxy is needed
export https_proxy=http://your-proxy:port
export http_proxy=http://your-proxy:port
python download.py --source rapidocr
```

4. **ModelScope Not Available**
```bash
# Install modelscope if not available
pip install modelscope
python download.py --source rapidocr
```

## Requirements

- Python 3.8+
- PyTorch 2.7.0+
- ultralytics 8.3+
- huggingface_hub 0.33.0+
- safetensors 0.5.0+
- modelscope 1.9.0+ (for ModelScope and Paddle OCR support)