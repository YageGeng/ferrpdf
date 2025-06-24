use tokio::sync::mpsc::Sender;
use tracing::*;
use uuid::Uuid;

use crate::{
    error::FerrpdfError,
    inference::yolov12::{YoloSession, Yolov12},
    layout::element::Layout,
};

use super::{Task, render::PdfImage};

pub struct LayoutTask {
    pub page_id: Uuid,
    pub pdf_image: PdfImage,
    pub span: Span,
}

pub struct LayoutQueue {
    layout_queue: Sender<LayoutTask>,
}

impl LayoutQueue {
    pub fn new(layout_queue: Sender<LayoutTask>) -> Self {
        Self { layout_queue }
    }

    pub async fn add_task(&self, page_id: Uuid, pdf_image: PdfImage, span: Span) {
        let layout_task = LayoutTask {
            page_id,
            pdf_image,
            span,
        };
        self.layout_queue.send(layout_task).await.unwrap();
    }
}

impl<'a> Task<'a> for LayoutTask {
    type Output = Layout;

    type Extra = &'a mut YoloSession<Yolov12>;

    fn run(&self, _extra: Self::Extra) -> Result<Self::Output, FerrpdfError> {
        todo!()
    }
}
