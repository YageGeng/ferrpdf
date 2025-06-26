use image::DynamicImage;
use ort::{
    execution_providers::CPUExecutionProvider,
    session::{
        RunOptions, Session,
        builder::{GraphOptimizationLevel, SessionBuilder},
    },
};
use snafu::ResultExt;
use std::future::Future;

use crate::error::{FerrpdfError, OrtInitSnafu};

pub trait Model {
    type Input;
    type Output;
    type Config;

    const INPUT_NAME: &'static str;
    const OUTPUT_NAME: &'static str;
    const MODEL_NAME: &'static str;

    fn load(&self) -> &[u8];
    fn config(&self) -> &Self::Config;
}

pub trait OnnxSession<M: Model> {
    type Output;
    type Extra;

    fn preprocess(&self, image: &DynamicImage) -> Result<M::Input, FerrpdfError>;

    fn postprocess(
        &self,
        output: M::Output,
        extra: Self::Extra,
    ) -> Result<Self::Output, FerrpdfError>;

    fn infer(
        &mut self,
        input: M::Input,
        input_name: &str,
        output_name: &str,
    ) -> Result<M::Output, FerrpdfError>;

    fn infer_async(
        &mut self,
        input: M::Input,
        input_name: &str,
        output_name: &str,
        options: &RunOptions,
    ) -> impl Future<Output = Result<M::Output, FerrpdfError>>;

    fn run(
        &mut self,
        image: &DynamicImage,
        extra: Self::Extra,
    ) -> Result<Self::Output, FerrpdfError> {
        let input = self.preprocess(image)?;

        let output = self.infer(input, M::INPUT_NAME, M::OUTPUT_NAME)?;

        self.postprocess(output, extra)
    }

    fn run_async(
        &mut self,
        image: &DynamicImage,
        extra: Self::Extra,
        options: &RunOptions,
    ) -> impl Future<Output = Result<Self::Output, FerrpdfError>> {
        async move {
            let input = self.preprocess(image)?;

            let output = self
                .infer_async(input, M::INPUT_NAME, M::OUTPUT_NAME, options)
                .await?;

            self.postprocess(output, extra)
        }
    }
}

/// common session builder
pub fn session_builder() -> Result<SessionBuilder, FerrpdfError> {
    let session_builder = Session::builder()
        .context(OrtInitSnafu { stage: "builder" })?
        .with_execution_providers(vec![
            #[cfg(all(feature = "coreml", target_os = "macos"))]
            {
                use ort::execution_providers::CoreMLExecutionProvider;
                use ort::execution_providers::coreml::*;
                CoreMLExecutionProvider::default()
                    .with_model_format(CoreMLModelFormat::MLProgram)
                    .build()
            },
            #[cfg(all(feature = "cuda"))]
            {
                use ort::execution_providers::CUDAExecutionProvider;
                CUDAExecutionProvider::default().build()
            },
            CPUExecutionProvider::default().build(),
        ])
        .context(OrtInitSnafu { stage: "provider" })?
        .with_optimization_level(GraphOptimizationLevel::Level1)
        .context(OrtInitSnafu {
            stage: "optimization",
        })?
        .with_intra_threads(4)
        .context(OrtInitSnafu {
            stage: "intra-threads",
        })?;
    // .with_inter_threads(4)
    // .context(OrtInitSnafu {
    //     stage: "inter-threads",
    // })?
    // .with_parallel_execution(true)
    // .context(OrtInitSnafu {
    //     stage: "parallel-enable",
    // })?;

    Ok(session_builder)
}
