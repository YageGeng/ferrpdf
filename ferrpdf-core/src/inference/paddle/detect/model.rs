use crate::inference::model::Model;
use ndarray::{ArrayBase, Dim, OwnedRepr};

const PADDLE_DETECT: &[u8] = include_bytes!("../../../../../models/ch_PP-OCRv5_mobile_det.onnx");

pub struct PaddleDet {
    config: PaddleDetConfig,
}

pub type PaddleDetInput = ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>;
pub type PaddleDetOutput = ArrayBase<OwnedRepr<f32>, Dim<[usize; 3]>>;

pub struct PaddleDetConfig {
    pub required_width: usize,
    pub required_height: usize,
    pub batch_size: usize,
    pub input_channels: usize,
    pub background_fill_value: f32,
    pub det_db_thresh: f32,
    pub det_db_box_thresh: f32,
    pub det_db_unclip_ratio: f32,
    pub max_candidates: usize,
    /// Minimum side length threshold for text detection
    pub max_side_thresh: f32,
    /// Padding to add around detected text bounding boxes (in pixels)
    pub text_padding: f32,
    pub merge_distance_thresh: f32,
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
            det_db_unclip_ratio: 1.5,
            max_candidates: 1000,
            max_side_thresh: 3.0,
            text_padding: 3.0,
            merge_distance_thresh: 5.0,
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
