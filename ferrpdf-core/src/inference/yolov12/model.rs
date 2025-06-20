use ndarray::{ArrayBase, Dim, OwnedRepr};

use crate::{consts::*, inference::model::Model};

const YOLOV12: &[u8] = include_bytes!("../../../../models/yolov12s-doclaynet.onnx");

pub struct Yolov12 {
    config: Yolov12Config,
}

pub type Yolov12Input = ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>;
pub type Yolov12Output = ArrayBase<OwnedRepr<f32>, Dim<[usize; 3]>>;

/// Configuration for YOLOv12 document layout analysis model
///
/// This configuration controls various aspects of the document layout detection process,
/// including model input requirements, post-processing thresholds, and output filtering.
#[derive(Debug, Clone)]
pub struct Yolov12Config {
    /// Required input width for the model (model expects this exact width)
    ///
    /// The model requires a fixed input size. Images will be resized to fit within
    /// this width while maintaining aspect ratio. Larger values may improve accuracy
    /// but increase memory usage and processing time.
    ///
    /// Default: 1024
    pub required_width: usize,

    /// Required input height for the model (model expects this exact height)
    ///
    /// The model requires a fixed input size. Images will be resized to fit within
    /// this height while maintaining aspect ratio. Larger values may improve accuracy
    /// but increase memory usage and processing time.
    ///
    /// Default: 1024
    pub required_height: usize,

    /// Batch size for model inference
    ///
    /// Currently only supports batch size of 1. This parameter is kept for future
    /// batch processing capabilities.
    ///
    /// Default: 1
    pub batch_size: usize,

    /// Number of input channels (RGB = 3)
    ///
    /// The model expects RGB images with 3 channels. This should not be changed
    /// unless the model architecture is modified.
    ///
    /// Default: 3
    pub input_channels: usize,

    /// Background fill value for input tensor padding
    ///
    /// When resizing images to fit the required dimensions, padding may be added.
    /// This value (normalized to [0, 1] range) is used to fill the padding areas.
    /// Default is 144.0/255.0 (light gray).
    ///
    /// Default: 144.0 / 255.0
    pub background_fill_value: f32,

    /// Output tensor size [channels, height, width]
    ///
    /// Defines the expected output tensor dimensions from the model. This is used
    /// to reshape and process the model output correctly.
    ///
    /// Default: OUTPUT_SIZE constant
    pub output_size: [usize; 3],

    /// Size of center coordinates and dimensions (cx, cy, w, h)
    ///
    /// Number of values used to represent the bounding box center coordinates
    /// and dimensions in the model output.
    ///
    /// Default: CXYWH_OFFSET constant
    pub cxywh_size: usize,

    /// Size of label probability values
    ///
    /// Number of probability values for different layout element classes
    /// (e.g., text, title, figure, table, etc.).
    ///
    /// Default: LABEL_PROBA_SIZE constant
    pub label_proba_size: usize,

    /// Confidence threshold for layout detection (0.0 to 1.0)
    ///
    /// Only layout elements with confidence scores above this threshold will be
    /// considered as valid detections. Higher values result in fewer but more
    /// confident detections. Lower values may detect more elements but include more
    /// false positives.
    ///
    /// Typical range: 0.1 - 0.8
    /// Default: PROBA_THRESHOLD constant
    pub proba_threshold: f32,

    /// IoU threshold for Non-Maximum Suppression (NMS)
    ///
    /// During NMS, overlapping detections with IoU above this threshold will be
    /// filtered out, keeping only the highest confidence detection. Higher values
    /// allow more overlapping detections to coexist.
    ///
    /// Typical range: 0.3 - 0.7
    /// Default: 0.45
    pub iou_threshold: f32,

    /// Y-coordinate tolerance threshold for layout element grouping
    ///
    /// Used to group layout elements that are horizontally aligned (similar Y coordinates).
    /// Elements with Y-coordinate differences less than this threshold are considered
    /// to be on the same horizontal line.
    ///
    /// Unit: pixels
    /// Typical range: 5.0 - 20.0
    /// Default: 10.0
    pub y_tolerance_threshold: f32,
}

impl Default for Yolov12Config {
    fn default() -> Self {
        Self {
            required_width: 1024,
            required_height: 1024,
            batch_size: 1,
            input_channels: 3,
            background_fill_value: 144.0 / 255.0,
            output_size: OUTPUT_SIZE,
            cxywh_size: CXYWH_OFFSET,
            label_proba_size: LABEL_PROBA_SIZE,
            proba_threshold: PROBA_THRESHOLD,
            iou_threshold: 0.45,
            y_tolerance_threshold: 10.0,
        }
    }
}

impl Yolov12 {
    pub fn new() -> Self {
        Self {
            config: Yolov12Config::default(),
        }
    }

    pub fn with_config(config: Yolov12Config) -> Self {
        Self { config }
    }
}

impl Default for Yolov12 {
    fn default() -> Self {
        Self::new()
    }
}

impl Model for Yolov12 {
    type Input = Yolov12Input;

    type Output = Yolov12Output;
    type Config = Yolov12Config;

    const INPUT_NAME: &'static str = "images";

    const OUTPUT_NAME: &'static str = "output0";

    const MODEL_NAME: &'static str = "yolov12s-doclaynet";

    fn load(&self) -> &[u8] {
        YOLOV12
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }
}
