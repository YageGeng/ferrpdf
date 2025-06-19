use image::DynamicImage;

use crate::error::FerrpdfError;

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

    fn run(
        &mut self,
        image: &DynamicImage,
        extra: Self::Extra,
    ) -> Result<Self::Output, FerrpdfError> {
        let input = self.preprocess(image)?;

        let output = self.infer(input, M::INPUT_NAME, M::OUTPUT_NAME)?;

        self.postprocess(output, extra)
    }
}
