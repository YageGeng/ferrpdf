use serde::Serialize;

use crate::analysis::{bbox::Bbox, labels::Label};

#[derive(Clone, Serialize, Debug)]
pub struct Layout {
    pub bbox: Bbox,
    pub label: Label,
    pub page_no: usize,
    pub bbox_id: usize,
    pub proba: f32,
    pub ocr: Option<Ocr>,
    pub text: Option<String>,
}

pub struct TextBlock {
    pub content: String,
    pub font_size: f32,
    pub font_weight: f32,
    pub font_family: String,
}

#[derive(Clone, Serialize, Debug)]
pub struct Ocr {
    /// pdfium extra text coverage threshold
    pub text_coverage: f32,
    /// is the text come from ocr
    pub is_ocr: bool,
}
