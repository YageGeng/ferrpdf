use crate::inference::model::Model;
use ndarray::{ArrayBase, Dim, OwnedRepr};

const PADDLE: &[u8] = include_bytes!("../../../../models/ch_PP-OCRv5_rec_mobile_infer.onnx");

pub struct Paddle {
    config: PaddleConfig,
}

pub type PaddleInput = ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>;
pub type PaddleOutput = ArrayBase<OwnedRepr<f32>, Dim<[usize; 3]>>;

pub struct PaddleConfig {
    pub required_height: usize,
    pub batch_size: usize,
    pub input_channels: usize,
    pub background_fill_value: f32,
}

impl Default for PaddleConfig {
    fn default() -> Self {
        Self {
            required_height: 48,
            batch_size: 1,
            input_channels: 3,
            background_fill_value: 0.5,
        }
    }
}

impl Paddle {
    pub fn new() -> Self {
        Self {
            config: PaddleConfig::default(),
        }
    }

    pub fn with_config(config: PaddleConfig) -> Self {
        Self { config }
    }
}

impl Default for Paddle {
    fn default() -> Self {
        Self::new()
    }
}

impl Model for Paddle {
    type Input = PaddleInput;

    type Output = PaddleOutput;

    type Config = PaddleConfig;

    const INPUT_NAME: &'static str = "x";

    const OUTPUT_NAME: &'static str = "fetch_name_0";

    const MODEL_NAME: &'static str = "ch_PP-OCRv5_rec_mobile_infer";

    fn load(&self) -> &[u8] {
        PADDLE
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }
}
