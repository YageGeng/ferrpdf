use crate::analysis::labels::Label;

/// The number of values representing bounding box coordinates in YOLO format.
///
/// YOLO format uses 4 values: [center_x, center_y, width, height]
/// This constant defines the offset where class probability data begins
/// in the model output tensor.
pub const CXYWH_OFFSET: usize = 4;

/// The number of different document element classes that the model can detect.
///
/// This value is determined by the Label enum and represents the total
/// number of document layout categories (e.g., text, title, figure, table, etc.)
/// that the YOLOv12 model was trained to recognize.
pub const LABEL_SIZE: usize = Label::label_size();

/// The total size of each detection vector in the model output.
///
/// Each detection contains:
/// - 4 bounding box coordinates (CXYWH_OFFSET)
/// - N class probabilities (LABEL_SIZE)
///
/// This represents the feature dimension for each detected object.
pub const LABEL_PROBA_SIZE: usize = CXYWH_OFFSET + LABEL_SIZE;

/// The expected output tensor shape from the YOLOv12 model.
///
/// Format: [batch_size, feature_size, num_detections]
/// - batch_size: 1 (single image processing)
/// - feature_size: LABEL_PROBA_SIZE (bbox coords + class probabilities)
/// - num_detections: 21504 (maximum number of detections per image)
///
/// The 21504 value comes from the model's anchor grid configuration
/// across different scales and aspect ratios.
pub const OUTPUT_SIZE: [usize; 3] = [1, LABEL_PROBA_SIZE, 21504];

/// Minimum confidence threshold for accepting a detection.
///
/// Detections with confidence scores below this threshold are filtered out
/// to reduce false positives. The value 0.2 (20%) provides a balance between
/// detection sensitivity and precision for document layout analysis.
///
/// This threshold can be adjusted based on the specific use case:
/// - Lower values (0.1-0.15): More sensitive, may include more false positives
/// - Higher values (0.3-0.5): More conservative, may miss some true detections
pub const PROBA_THRESHOLD: f32 = 0.1;

pub const YOLOV12_INPUT_IMAGE_WIDTH: usize = 1024;
pub const YOLOV12_INPUT_IMAGE_HEIGHT: usize = 1024;

/// Background fill value for image preprocessing.
///
/// When resizing images, areas not covered by the original image
/// are filled with this normalized value (144/255 ≈ 0.565).
/// This value is chosen to represent a neutral gray background.
pub const BACKGROUND_FILL_VALUE: f32 = 144.0 / 255.0;

// Pre-calculate normalization constants for performance
// Original: (r/255.0 - 0.5) / 0.5 = r/127.5 - 1.0
pub const NORMALIZATION_SCALE: f32 = 1.0 / 127.5;

/// Environment variable name for specifying the path to the PDFium dynamic library.
///
/// If the environment variable is not set, the library will attempt to locate
/// the PDFium dynamic library in the system's default library paths.
pub const PDFIUM_LIB_PATH_ENV_NAME: &str = "PDFIUM_DYNAMIC_LIB_PATH";

/// Maximum number of concurrent render requests to PDFium.
///
/// This value controls the number of concurrent requests that can be processed
/// by PDFium. Adjusting this value can help manage resource usage and performance.
pub const MAX_CONCURRENT_RENDER_QUEUE: usize = 10;

pub const FONT: &[u8] = include_bytes!("../../fonts/DejaVuSans.ttf");
