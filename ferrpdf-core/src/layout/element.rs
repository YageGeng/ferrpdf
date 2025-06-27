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

#[derive(Clone, Serialize, Debug)]
pub struct Ocr {
    /// is the text come from ocr
    pub is_ocr: bool,
}
