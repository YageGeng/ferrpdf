use crate::error::FerrpdfError;

pub mod detect;
pub mod extra;
pub mod layout;
pub mod ocr;
pub mod render;

pub trait Task<'a> {
    type Output;
    type Extra: 'a;

    fn run(&self, extra: Self::Extra) -> Result<Self::Output, FerrpdfError>;
}
