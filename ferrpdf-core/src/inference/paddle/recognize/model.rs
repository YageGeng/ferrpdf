use crate::inference::model::Model;
use ndarray::{ArrayBase, Dim, OwnedRepr};

const PADDLE_RECOGNIZE: &[u8] =
    include_bytes!("../../../../../models/ch_PP-OCRv5_rec_mobile_infer.onnx");

pub struct PaddleRec {
    config: PaddleRecConfig,
}

pub type PaddleInput = ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>;
pub type PaddleOutput = ArrayBase<OwnedRepr<f32>, Dim<[usize; 3]>>;

pub struct PaddleRecConfig {
    pub required_height: usize,
    pub batch_size: usize,
    pub input_channels: usize,
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
