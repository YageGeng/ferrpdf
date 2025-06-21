use crate::layout::element::Layout;

#[derive(Debug, Clone)]
pub struct Page {
    pub width: f32,
    pub height: f32,
    pub blocks: Vec<Layout>,
    pub page_no: usize,
}
