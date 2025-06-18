# FerrPDF

A Rust-based PDF layout analysis tool that combines PDF rendering with deep learning-based document layout detection.

## Features

- **PDF to Image Conversion**: Convert PDF pages to high-quality images using PDFium
- **Layout Analysis**: Detect and classify document elements using YOLOv12 model trained on DocLayNet
- **Bounding Box Visualization**: Draw detected layout elements with color-coded bounding boxes
- **Reading Order Sorting**: Automatically sort detected elements in natural reading order
- **Multi-column Support**: Handle both single and multi-column document layouts

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

## Installation

### Prerequisites

1. **PDFium Library**: The tool requires PDFium binaries. Set the environment variable:
   ```bash
   export PDFIUM_DYNAMIC_LIB_PATH=/path/to/pdfium/lib
   ```

2. **Rust**: Install Rust from [rustup.rs](https://rustup.rs/)

### Build

```bash
cd ferrpdf/ferrpdf-core
cargo build --release
```

## Usage

### Analyze Tool (Recommended)

The `analyze` tool combines PDF rendering and layout analysis in a single command:

```bash
# Analyze first page of a PDF
cargo run --bin analyze -- input.pdf

# Analyze specific page (0-based indexing)
cargo run --bin analyze -- input.pdf --page 2

# Specify output directory
cargo run --bin analyze -- input.pdf --output results

# Full example
cargo run --bin analyze -- "document.pdf" --page 0 --output analysis_results
```

#### Options

- `<INPUT>`: Input PDF file path (required)
- `-p, --page <PAGE>`: Page number to analyze (0-based, default: 0)
- `-o, --output <OUTPUT>`: Output directory (default: "images")

#### Output Files

- `page-{N}.jpg`: Original page rendered as image
- `analysis-page-{N}.jpg`: Analysis result with bounding boxes and labels

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

### NMS Algorithm

The tool uses an advanced NMS algorithm that:

- Merges overlapping bounding boxes instead of removing them
- Uses both IoU and overlap ratio for better detection of nested elements
- Handles bounding boxes of different sizes effectively

## Examples

### Basic Usage

```bash
# Analyze a research paper
cargo run --bin analyze -- "research_paper.pdf"

# Analyze page 3 of a document
cargo run --bin analyze -- "document.pdf" --page 3

# Save results to custom directory
cargo run --bin analyze -- "document.pdf" --output my_analysis
```

### Expected Output

```
=== PDF Layout Analysis Summary ===
Input PDF: document.pdf
Analyzed page: 0
Total detections: 12
Confidence threshold: 0.2
NMS IoU threshold: 0.45

Detected elements:
  1. Title (95.2%)
  2. Text (89.7%)
  3. Section-Header (87.3%)
  4. Text (85.9%)
  5. Figure (82.1%)
  6. Caption (78.4%)
  7. Text (76.8%)
  8. Table (74.5%)
  9. Text (72.1%)
  10. Page-Footer (68.9%)
```

## Architecture

### Core Components

- **`analysis/bbox.rs`**: Bounding box operations and geometric calculations
- **`analysis/labels.rs`**: Document element classification labels
- **`inference/session.rs`**: ONNX Runtime integration and model inference
- **`layout/element.rs`**: Layout element data structures

### Key Algorithms

1. **Image Preprocessing**: Resize and normalize images for model input
2. **YOLO Inference**: Run YOLOv12 model for object detection
3. **Bounding Box Extraction**: Convert model outputs to layout elements
4. **Advanced NMS**: Merge overlapping detections with union bounding boxes
5. **Reading Order Sorting**: Sort elements for natural document flow

## Development

### Running Tests

```bash
cargo test
```

### Linting

```bash
cargo clippy
```

### Documentation

```bash
cargo doc --open
```

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

### Performance Tips

- Use release builds for better performance: `cargo build --release`
- Consider batch processing for multiple PDFs
- Adjust model thresholds based on document type