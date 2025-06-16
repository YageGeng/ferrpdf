use crate::analysis::{bbox::Bbox, labels::Label};

#[derive(Clone)]
pub struct Layout {
    pub bbox: Bbox,
    pub label: Label,
    pub page_no: u32,
    pub proba: f32,
    pub text: Option<String>,
}
