# Model Conversion Script

This script downloads YOLO model files from Hugging Face and converts them to ONNX format.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python convert.py [repo_id] [filename]
```

### Parameters

- `repo_id`: HuggingFace repository ID (optional, default: "hantian/yolo-doclaynet")
- `filename`: File name to download (optional, default: "yolov12s-doclaynet.pt")
- `--output-name, -o`: Output model name (without extension)
- `--models-dir, -d`: Model storage directory (default: "models")

### Examples

1. **Use default parameters:**
```bash
python convert.py
```

2. **Download and convert YOLOv8n model:**
```bash
python convert.py ultralytics/yolov8 yolov8n.pt
```

3. **Specify output filename:**
```bash
python convert.py ultralytics/yolov8 yolov8n.pt --output-name my_yolo_model
```

4. **Specify custom model directory:**
```bash
python convert.py ultralytics/yolov8 yolov8n.pt --models-dir ./custom_models
```

5. **Download and convert YOLOv11 model:**
```bash
python convert.py ultralytics/yolo11 yolo11n.pt -o yolo11_nano
```

## Output

- Downloaded model files are saved in the specified model directory
- Converted ONNX files are also saved in the same directory
- By default, all files are saved in the `models/` directory

## Notes

- Ensure sufficient disk space for storing downloaded model files
- Conversion process may take time depending on model size
- Network connection required for downloading from Hugging Face
- Conversion process uses CPU, may be slow for large models

## Error Handling

The script includes error handling mechanisms:
- If download fails, the script will display error message and exit
- If conversion fails, the script will display error message and exit
- All operations have corresponding progress indicators

## Supported Model Formats

- Input: PyTorch (.pt) model files
- Output: ONNX (.onnx) model files

## Advanced Usage

### Custom Repositories and Models

```bash
# Download from specific repository
python convert.py hantian/yolo-doclaynet yolov8n-doclaynet.pt

# Download specialized trained models
python convert.py ultralyticsplus/yolov8s_worldv2 yolov8s-worldv2.pt
```

### Batch Processing Multiple Files

Create a configuration file `models_config.txt`:
```
ultralytics/yolov8,yolov8n.pt,yolo8n_detection
ultralytics/yolov8,yolov8s.pt,yolo8s_detection
hantian/yolo-doclaynet,yolov8n-doclaynet.pt,yolo8n_layout
```

Then use script for batch processing:
```bash
while IFS=',' read -r repo file output || [ -n "$repo" ]; do
    python convert.py "$repo" "$file" --output-name "$output"
done < models_config.txt
```

### Environment Variable Configuration

Set default model directory:
```bash
export YOLO_MODELS_DIR="/path/to/your/models"
python convert.py ultralytics/yolov8 yolov8n.pt --models-dir "$YOLO_MODELS_DIR"
```

## Performance Optimization

1. **Disk Space**: Ensure sufficient space (each model ~6MB-200MB)
2. **Network Connection**: Stable network connection required for initial download
3. **CPU Resources**: Conversion process consumes CPU resources, recommend running during low load
4. **Memory Usage**: Large model conversion may require 4GB+ memory

## Troubleshooting

### Common Issues

1. **Download Timeout**
```bash
# Set longer timeout
export HF_HUB_DOWNLOAD_TIMEOUT=300
python convert.py ultralytics/yolov8 yolov8n.pt
```

2. **Memory Insufficient**
```bash
# For large models, try reducing batch size
python convert.py ultralytics/yolov8 yolov8n.pt  # Process one at a time
```

3. **Network Proxy Settings**
```bash
# If proxy is needed
export https_proxy=http://your-proxy:port
export http_proxy=http://your-proxy:port
python convert.py ultralytics/yolov8 yolov8n.pt
```

## Requirements

- Python 3.8+
- PyTorch 2.7.0+
- ultralytics 8.3+
- huggingface_hub 0.33.0+
- safetensors 0.5.0+