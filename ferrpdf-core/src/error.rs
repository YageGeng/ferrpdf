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
    #[snafu(display("Onnx RunOption for {} error: {}", stage, source))]
    RunConfig {
        source: ort::error::Error,
        stage: String,
    },
    #[snafu(display("Onnx Output can not found {}", output_name))]
    NotFoundOutput { output_name: String },
    #[snafu(display("Ndarray Shape error at stage `{}`: {}", stage, source))]
    Shape {
        source: ndarray::ShapeError,
        stage: String,
    },
    #[snafu(display("Load Font error: {}", source))]
    Font { source: ab_glyph::InvalidFont },
    #[snafu(display("Image Write error: {}", source))]
    ImageWrite {
        source: image::ImageError,
        path: String,
    },
    #[snafu(display("Write `{}` error: {}", path, source))]
    IoWrite {
        source: std::io::Error,
        path: String,
    },
    #[snafu(display("Environment `{}` Not Found, error {}", name, source))]
    EnvNotFound {
        source: std::env::VarError,
        name: String,
    },
    #[snafu(display("Pdfium `{}` error {}", stage, source))]
    Pdfium {
        source: pdfium_render::prelude::PdfiumError,
        stage: String,
    },
    #[snafu(display("Parse pdf error on `{}` for {}, msg {}", stage, path, message))]
    ParserErr {
        stage: String,
        path: String,
        message: String,
    },
}
