use ort::session::{Session, builder::SessionBuilder};
use snafu::ResultExt;

use crate::{
    error::*,
    inference::{
        model::{Model, OnnxSession},
        paddle::model::Paddle,
    },
};

pub struct PaddleSession<M: Model> {
    session: Session,
    model: M,
}

impl PaddleSession<Paddle> {
    pub fn new(session: SessionBuilder, model: Paddle) -> Result<Self, FerrpdfError> {
        let session = session
            .commit_from_memory(model.load())
            .context(OrtInitSnafu { stage: "commit" })?;

        println!("{:?}", session.inputs.first());
        println!("{:?}", session.outputs.first());

        Ok(Self { session, model })
    }
}

impl OnnxSession<Paddle> for PaddleSession<Paddle> {
    type Output = ();

    type Extra = ();

    fn preprocess(
        &self,
        image: &image::DynamicImage,
    ) -> Result<<Paddle as Model>::Input, FerrpdfError> {
        Ok(())
    }

    fn postprocess(
        &self,
        output: <Paddle as Model>::Output,
        extra: Self::Extra,
    ) -> Result<Self::Output, FerrpdfError> {
        Ok(())
    }

    fn infer(
        &mut self,
        input: <Paddle as Model>::Input,
        input_name: &str,
        output_name: &str,
    ) -> Result<<Paddle as Model>::Output, FerrpdfError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::inference::paddle::model::PaddleConfig;

    use super::*;
    use ort::{
        execution_providers::CPUExecutionProvider,
        session::{Session, builder::GraphOptimizationLevel},
    };

    #[test]
    fn test_paddle_session() -> Result<(), Box<dyn std::error::Error>> {
        let session_builder = Session::builder()?
            .with_execution_providers(vec![CPUExecutionProvider::default().build()])?
            .with_optimization_level(GraphOptimizationLevel::Level1)?;

        let model = Paddle::new(PaddleConfig {});
        let session = PaddleSession::new(session_builder, model)?;
        Ok(())
    }
}
