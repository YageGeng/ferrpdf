# FerrPDF

A Rust-based PDF content extraction tool that combines PDF rendering with deep learning-based document layout detection and OCR text recognition. This project is inspired by and builds upon the excellent work of [ferrules](https://github.com/AmineDiro/ferrules).

## Features

- **PDF to Image Conversion**: Convert PDF pages to high-quality images using PDFium with configurable rendering options
- **Layout Analysis**: Detect and classify document elements using YOLOv12 model trained on DocLayNet, with advanced sorting by page index
- **Text Detection & OCR**: Extract text content using PaddleOCR with PaddlePaddle models, including layout-aware detection and multi-language support
- **Bounding Box Visualization**: Draw detected layout elements and text regions with color-coded bounding boxes, including detection and OCR results
- **Reading Order Sorting**: Automatically sort detected elements in natural reading order, optimized for multi-column layouts
- **Multi-column Support**: Handle both single and multi-column document layouts with intelligent merging of text lines
- **Text Line Merging**: Intelligently merge overlapping text detections into coherent text lines, with configurable thresholds
- **Debug Mode**: Save intermediate results such as layout bounding boxes and text detection visualizations for debugging purposes

## Detected Layout Elements

The tool can detect and classify the following document elements:

- Caption
- Footnote
- Formula
- List-Item
- Page-Footer
- Page-Header
- Picture
- Section-Header
- Table
- Text
- Title

## OCR Capabilities

### Text Detection
- **PaddleOCR Detection**: Uses PaddlePaddle's text detection model to locate text regions
- **Layout-Aware Detection**: Detects text within specific layout regions (e.g., text blocks, tables)
- **Confidence Scoring**: Each text detection includes confidence probability
- **Bounding Box Extraction**: Precise text region coordinates with proper scaling

### Text Recognition
- **PaddleOCR Recognition**: Uses PaddlePaddle's text recognition model for OCR
- **Multi-language Support**: Supports various languages and character sets
- **High Accuracy**: State-of-the-art recognition accuracy on printed text

## Installation

### Prerequisites

1. **PDFium Library**: The tool requires PDFium binaries. Set the environment variable:
   ```bash
   export PDFIUM_DYNAMIC_LIB_PATH=/path/to/pdfium/lib
   ```

2. **Rust**: Install Rust from [rustup.rs](https://rustup.rs/)

3. **ONNX Runtime**: Required for model inference (automatically handled by dependencies)

4. **Arch Linux CUDA Requirements**: If enabling CUDA features on Arch Linux, ensure the following packages are installed:
   - `onnxruntime`
   - `cuda`
   - `cudnn`
   - `nccl`

### Build

```bash
cd ferrpdf/ferrpdf-core
cargo build --release
```

## Usage

### Analyze Tool (Recommended)

The `analyze` tool combines PDF rendering, layout analysis, and OCR text extraction in a single command:

```bash
# Analyze first page of a PDF with OCR
cargo run --bin analyze -- input.pdf

# Analyze specific page (0-based indexing)
cargo run --bin analyze -- input.pdf --page 2

# Specify output directory
cargo run --bin analyze -- input.pdf --output results --debug

# Full example
cargo run --bin analyze -- "document.pdf" --page 0 --output analysis_results --debug
```

#### Options

- `<INPUT>`: Input PDF file path (required)
- `-p, --page <PAGE>`: Page number to analyze (0-based, default: 0)
- `-o, --output <OUTPUT>`: Output directory (default: "images")

#### Output Files

- `analysis-{page}.jpg`: Layout analysis result with bounding boxes and labels
- `detection-{page}-{N}.jpg`: Text detection visualization with bounding boxes and confidence scores

### Individual Tools

#### PDF to Image Conversion

```bash
cargo run --bin pdf2img
```

#### Layout Analysis on Existing Images

```bash
cargo run --bin layout
```

## Configuration

### Model Parameters

The following constants can be adjusted in `src/consts.rs`:

- `PROBA_THRESHOLD`: Minimum confidence threshold (default: 0.2)
- `NMS_IOU_THRESHOLD`: IoU threshold for Non-Maximum Suppression (default: 0.45)
- `REQUIRED_WIDTH/HEIGHT`: Model input dimensions (1024x1024)

### PaddleOCR Configuration

PaddleOCR detection and recognition models can be configured with custom parameters:

- **Detection Thresholds**: Adjust text detection confidence and IoU thresholds
- **Line Merging**: Configure text line merging parameters for better text extraction
- **Post-processing**: Customize DB (Differentiable Binarization) post-processing parameters

### NMS Algorithm

The tool uses an advanced NMS algorithm that:

- Merges overlapping bounding boxes instead of removing them
- Uses both IoU and overlap ratio for better detection of nested elements
- Handles bounding boxes of different sizes effectively

## Examples

### Basic Usage

```bash
# Analyze a research paper with OCR
cargo run --bin analyze -- "research_paper.pdf"

# Analyze page 3 of a document
cargo run --bin analyze -- "document.pdf" --page 3

# Save results to custom directory
cargo run --bin analyze -- "document.pdf" --output my_analysis
```

## Architecture

### Core Components

- **`analysis/bbox.rs`**: Bounding box operations and geometric calculations
- **`analysis/labels.rs`**: Document element classification labels
- **`inference/session.rs`**: ONNX Runtime integration and model inference
- **`layout/element.rs`**: Layout element data structures
- **`inference/paddle/`**: PaddleOCR detection and recognition models
- **`inference/yolov12/`**: YOLOv12 layout detection model

### Key Algorithms

1. **Image Preprocessing**: Resize and normalize images for model input
2. **YOLO Inference**: Run YOLOv12 model for layout element detection
3. **Layout-Aware Text Detection**: Use PaddleOCR to detect text within layout regions
4. **Text Recognition**: Extract text content using PaddleOCR recognition model
5. **Text Line Merging**: Intelligently merge overlapping text detections
6. **Advanced NMS**: Merge overlapping detections with union bounding boxes
7. **Reading Order Sorting**: Sort elements for natural document flow

### Model Pipeline

```
PDF Page → PDFium Rendering → Configurable Image Dimensions → YOLOv12 Layout Detection
                                                      ↓
                                              Layout Regions → PaddleOCR Text Detection
                                                      ↓
                                              Text Regions → PaddleOCR Text Recognition
                                                      ↓
                                              Extracted Text + Coordinates
                                                      ↓
                                      Debug Visualization (Optional)
```

## Development

### Running Tests

```bash
cargo test
```

### Debugging

Enable debug mode to save intermediate results such as layout bounding boxes and text detection visualizations:

```bash
cargo run --bin analyze -- --page 1 input.pdf --debug
```

### Linting

```bash
cargo clippy
```

### Documentation

```bash
cargo doc --open
```

## Acknowledgments

This project is inspired by and builds upon the excellent work of [ferrules](https://github.com/AmineDiro/ferrules), a comprehensive document analysis framework. We extend our gratitude to the ferrules project for providing the foundation and inspiration for this PDF content extraction tool.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## Troubleshooting

### Common Issues

1. **PDFium Library Not Found**
   ```
   Solution: Set PDFIUM_DYNAMIC_LIB_PATH environment variable
   ```

2. **PDF Not Found Error**
   ```
   Solution: Check file path and ensure PDF file exists
   ```

3. **Page Index Out of Range**
   ```
   Solution: Use 0-based page indexing (first page = 0)
   ```

4. **Low Detection Accuracy**
   ```
   Solution: Adjust PROBA_THRESHOLD or try different NMS_IOU_THRESHOLD
   ```

5. **OCR Text Recognition Issues**
   ```
   Solution: Ensure text is clear and properly rendered, adjust detection thresholds
   ```

6. **Debug Output Not Generated**
   ```
   Solution: Ensure debug mode is enabled with the --debug flag and specify a valid output directory
   ```

### Performance Tips

- Use release builds for better performance: `cargo build --release`
- Consider batch processing for multiple PDFs
- Adjust model thresholds based on document type
- For large documents, process pages individually to manage memory usage
