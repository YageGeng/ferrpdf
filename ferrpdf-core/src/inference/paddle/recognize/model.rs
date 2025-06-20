use crate::inference::model::Model;
use ndarray::{ArrayBase, Dim, OwnedRepr};

const PADDLE_RECOGNIZE: &[u8] =
    include_bytes!("../../../../../models/ch_PP-OCRv5_rec_mobile_infer.onnx");

pub struct PaddleRec {
    config: PaddleRecConfig,
}

pub type PaddleInput = ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>;
pub type PaddleOutput = ArrayBase<OwnedRepr<f32>, Dim<[usize; 3]>>;

/// Configuration for PaddleOCR text recognition model
///
/// This configuration controls various aspects of the text recognition (OCR) process,
/// including model input requirements and preprocessing parameters.
#[derive(Debug, Clone)]
pub struct PaddleRecConfig {
    /// Required input height for the model (model expects this exact height)
    ///
    /// The model requires a fixed height for text line images. Text regions will be
    /// resized to this height while maintaining aspect ratio. The width is calculated
    /// automatically based on the aspect ratio of the input text region.
    ///
    /// Typical values: 32, 48, 64
    /// Default: 48
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
    /// When resizing text regions to fit the required dimensions, padding may be added.
    /// This value (normalized to [-1, 1] range) is used to fill the padding areas.
    /// Default is 0.5 (gray in normalized space).
    ///
    /// Default: 0.5
    pub background_fill_value: f32,
}

impl Default for PaddleRecConfig {
    fn default() -> Self {
        Self {
            required_height: 48,
            batch_size: 1,
            input_channels: 3,
            background_fill_value: 0.5,
        }
    }
}

impl PaddleRec {
    pub fn new() -> Self {
        Self {
            config: PaddleRecConfig::default(),
        }
    }

    pub fn with_config(config: PaddleRecConfig) -> Self {
        Self { config }
    }
}

impl Default for PaddleRec {
    fn default() -> Self {
        Self::new()
    }
}

impl Model for PaddleRec {
    type Input = PaddleInput;

    type Output = PaddleOutput;

    type Config = PaddleRecConfig;

    const INPUT_NAME: &'static str = "x";

    const OUTPUT_NAME: &'static str = "fetch_name_0";

    const MODEL_NAME: &'static str = "ch_PP-OCRv5_rec_mobile_infer";

    fn load(&self) -> &[u8] {
        PADDLE_RECOGNIZE
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }
}
