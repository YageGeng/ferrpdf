use ndarray::{ArrayBase, Dim, OwnedRepr};

use crate::{consts::*, inference::model::Model};

const YOLOV12: &[u8] = include_bytes!("../../../../models/yolov12s-doclaynet.onnx");

pub struct Yolov12 {
    config: Yolov12Config,
}

pub type Yolov12Input = ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>;
pub type Yolov12Output = ArrayBase<OwnedRepr<f32>, Dim<[usize; 3]>>;

pub struct Yolov12Config {
    pub required_width: usize,
    pub required_height: usize,
    pub batch_size: usize,
    pub input_channels: usize,
    pub background_fill_value: f32,
    pub output_size: [usize; 3],
    pub cxywh_size: usize,
    pub label_proba_size: usize,
    pub proba_threshold: f32,
    pub iou_threshold: f32,
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
