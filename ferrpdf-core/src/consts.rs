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
pub const PROBA_THRESHOLD: f32 = 0.2;

/// IoU threshold for Non-Maximum Suppression (NMS).
///
/// When two bounding boxes have an Intersection over Union (IoU) score
/// greater than this threshold, the detection with lower confidence is
/// suppressed to eliminate duplicate detections of the same object.
///
/// The value 0.5 (50% overlap) is a common choice that provides good
/// balance between removing duplicates and preserving distinct objects:
/// - Lower values (0.3-0.4): More aggressive suppression, may remove valid detections
/// - Higher values (0.6-0.8): Less aggressive, may keep more duplicates
pub const NMS_IOU_THRESHOLD: f32 = 0.45;

/// Required input width for the YOLOv12 model.
///
/// This is the standard width that input images must be resized to
/// before feeding into the model. The model expects square inputs
/// with consistent dimensions for optimal performance.
pub const REQUIRED_WIDTH: u32 = 1024;

/// Required input height for the YOLOv12 model.
///
/// This is the standard height that input images must be resized to
/// before feeding into the model. The model expects square inputs
/// with consistent dimensions for optimal performance.
pub const REQUIRED_HEIGHT: u32 = 1024;

/// Number of color channels in the input image.
///
/// RGB images have 3 channels: Red, Green, Blue.
/// This constant is used for tensor shape calculations.
pub const INPUT_CHANNELS: usize = 3;

/// Batch size for model inference.
///
/// Currently set to 1 for single image processing.
/// Can be increased for batch processing if needed.
pub const BATCH_SIZE: usize = 1;

/// Background fill value for image preprocessing.
///
/// When resizing images, areas not covered by the original image
/// are filled with this normalized value (144/255 ≈ 0.565).
/// This value is chosen to represent a neutral gray background.
pub const BACKGROUND_FILL_VALUE: f32 = 144.0 / 255.0;

pub const FONT: &[u8] = include_bytes!("../../fonts/DejaVuSans.ttf");
