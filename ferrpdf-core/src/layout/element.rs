use serde::Serialize;

use crate::analysis::{bbox::Bbox, labels::Label};

#[derive(Clone, Serialize)]
pub struct Layout {
    pub bbox: Bbox,
    pub label: Label,
    pub page_no: usize,
    pub bbox_id: usize,
    pub proba: f32,
    pub text: Option<String>,
}
