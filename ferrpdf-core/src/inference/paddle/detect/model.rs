use crate::inference::model::Model;
use derive_builder::Builder;
use ndarray::{ArrayBase, Dim, OwnedRepr};

const PADDLE_DETECT: &[u8] = include_bytes!("../../../../../models/ch_PP-OCRv5_mobile_det.onnx");

pub struct PaddleDet {
    config: PaddleDetConfig,
}

pub type PaddleDetInput = ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>;
pub type PaddleDetOutput = ArrayBase<OwnedRepr<f32>, Dim<[usize; 3]>>;

/// Configuration for PaddleOCR text detection model
///
/// This configuration controls various aspects of the text detection process,
/// including model input requirements, post-processing thresholds, and output formatting.
#[derive(Debug, Clone, Builder)]
#[builder[setter(into)]]
pub struct PaddleDetConfig {
    /// Required input width for the model (model expects this exact width)
    ///
    /// The model requires a fixed input size. Images will be resized to fit within
    /// this width while maintaining aspect ratio. Larger values may improve accuracy
    /// but increase memory usage and processing time.
    pub required_width: usize,

    /// Required input height for the model (model expects this exact height)
    ///
    /// The model requires a fixed input size. Images will be resized to fit within
    /// this height while maintaining aspect ratio. Larger values may improve accuracy
    /// but increase memory usage and processing time.
    pub required_height: usize,

    /// Batch size for model inference
    ///
    /// Currently only supports batch size of 1. This parameter is kept for future
    /// batch processing capabilities.
    pub batch_size: usize,

    /// Number of input channels (RGB = 3)
    ///
    /// The model expects RGB images with 3 channels. This should not be changed
    /// unless the model architecture is modified.
    pub input_channels: usize,

    /// Background fill value for input tensor padding
    ///
    /// When resizing images to fit the required dimensions, padding may be added.
    /// This value (normalized to [-1, 1] range) is used to fill the padding areas.
    /// Default is 0.5 (gray in normalized space).
    pub background_fill_value: f32,

    /// Confidence threshold for text detection (0.0 to 1.0)
    ///
    /// Only text regions with confidence scores above this threshold will be
    /// considered as valid detections. Higher values result in fewer but more
    /// confident detections. Lower values may detect more text but include more
    /// false positives.
    ///
    /// Typical range: 0.1 - 0.5
    /// Default: 0.3
    pub det_db_thresh: f32,

    /// Box threshold for text detection (0.0 to 1.0)
    ///
    /// This threshold is used during the binarization step of DB (Differentiable
    /// Binarization) post-processing. It controls the sensitivity of text region
    /// detection. Higher values require stronger text signals for detection.
    ///
    /// Typical range: 0.3 - 0.8
    /// Default: 0.6
    pub det_db_box_thresh: f32,

    /// Maximum number of text candidates to process
    ///
    /// Limits the number of potential text regions that will be processed during
    /// post-processing. This helps control memory usage and processing time for
    /// images with many text regions.
    ///
    /// Default: 1000
    pub max_candidates: usize,

    /// Minimum side length threshold for text detection (in pixels)
    ///
    /// Text regions with side lengths smaller than this threshold will be filtered out.
    /// This helps remove very small noise or artifacts that might be detected as text.
    ///
    /// Typical range: 1.0 - 10.0
    /// Default: 3.0
    pub max_side_thresh: f32,

    /// Padding to add around detected text bounding boxes (in pixels)
    ///
    /// This padding is added to each detected text region to provide some margin
    /// around the actual text. This can help ensure that the entire text is captured
    /// and may improve OCR accuracy. The padding is clamped to image boundaries.
    ///
    /// Typical range: 0.0 - 10.0
    /// Default: 6.0
    pub text_padding: f32,

    /// Y-coordinate tolerance threshold for reading order sorting (in pixels)
    ///
    /// When sorting text detections by reading order, this threshold determines
    /// the maximum vertical distance between two text boxes to be considered
    /// on the same line. Text boxes within this tolerance are sorted horizontally
    /// (left-to-right), while boxes beyond this tolerance are sorted vertically
    /// (top-to-bottom).
    ///
    /// - Higher values: More text boxes are considered on the same line
    /// - Lower values: More strict line separation
    ///
    /// This is crucial for proper reading order in multi-line text documents.
    ///
    /// Represents the maximum difference in y-values for elements to be considered on the same line.
    /// Typical range: 3.0 - 15.0
    /// Default: 10.0
    pub y_tolerance_threshold: f32,
    /// Represents the maximum difference in x-values for elements to be considered for merging.
    /// Typical range: 0.0 - 15.0
    /// Default: 10.0
    pub x_merge_threshold: f32,
}

impl Default for PaddleDetConfig {
    fn default() -> Self {
        Self {
            required_width: 960,
            required_height: 960,
            batch_size: 1,
            input_channels: 3,
            background_fill_value: 0.5,
            det_db_thresh: 0.3,
            det_db_box_thresh: 0.6,
            max_candidates: 1000,
            max_side_thresh: 3.0,
            text_padding: 6.0,
            y_tolerance_threshold: 10.0,
            x_merge_threshold: 10.0,
        }
    }
}

impl PaddleDet {
    pub fn new() -> Self {
        Self {
            config: PaddleDetConfig::default(),
        }
    }

    pub fn with_config(config: PaddleDetConfig) -> Self {
        Self { config }
    }
}

impl Default for PaddleDet {
    fn default() -> Self {
        Self::new()
    }
}

impl Model for PaddleDet {
    type Input = PaddleDetInput;
    type Output = PaddleDetOutput;
    type Config = PaddleDetConfig;

    const INPUT_NAME: &'static str = "x";
    const OUTPUT_NAME: &'static str = "fetch_name_0";
    const MODEL_NAME: &'static str = "ch_PP-OCRv5_mobile_det";

    fn load(&self) -> &[u8] {
        PADDLE_DETECT
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }
}
