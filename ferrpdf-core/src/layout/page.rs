use std::alloc::Layout;

pub struct Page {
    pub width: f32,
    pub height: f32,
    pub blocks: Vec<Layout>,
    pub page_no: usize,
}
