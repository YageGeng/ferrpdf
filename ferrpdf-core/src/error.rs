use snafu::prelude::*;

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum FerrpdfError {
    #[snafu(display("Ort Session init stage `{}` error: {}", stage, source))]
    OrtInit {
        source: ort::error::Error,
        stage: String,
    },
    #[snafu(display("Build Tensor for `{}` error: {}", stage, source))]
    Tensor {
        source: ort::error::Error,
        stage: String,
    },
    #[snafu(display("Onnx Inference error: {}", source))]
    Inference { source: ort::error::Error },
    #[snafu(display("Onnx Output can not found {}", output_name))]
    NotFoundOutput { output_name: String },
    #[snafu(display("Ndarray Shape error at stage `{}`: {}", stage, source))]
    Shape {
        source: ndarray::ShapeError,
        stage: String,
    },
}
