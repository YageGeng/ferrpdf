use crate::inference::model::Model;

const PADDLE: &[u8] = include_bytes!("../../../../models/ch_PP-OCRv5_rec_mobile_infer.onnx");

pub struct Paddle {
    config: PaddleConfig,
}

pub struct PaddleConfig {}

impl Paddle {
    pub fn new(config: PaddleConfig) -> Self {
        Self { config }
    }
}

impl Model for Paddle {
    type Input = ();

    type Output = ();

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
