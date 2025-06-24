use std::{ops::Range, time::Instant};

use bytes::Bytes;
use glam::Vec2;
use image::DynamicImage;
use pdfium_render::prelude::{PdfPageRenderRotation, PdfRenderConfig, Pdfium};
use snafu::ResultExt;
use tokio::{
    sync::mpsc::{self, Receiver, Sender},
    task,
};
use tracing::*;
use uuid::Uuid;

use crate::{consts::*, error::*};

use super::Task;

pub struct RenderTask {
    pub document: Bytes,
    pub page_id: Uuid,
    pub passowrd: Option<String>,
    pub range: Range<usize>,
    pub sender: Sender<RenderResult>,
    pub span: Span,
}

pub struct RenderResult {
    pub page_id: Uuid,
    pub images: Vec<Result<PdfImage, FerrpdfError>>,
}

pub struct PdfImage {
    /// image
    pub image: DynamicImage,
    /// target size
    pub target_size: Vec2,
    /// pdf size
    pub pdf_size: Vec2,
    /// scale factor
    pub scale: f32,
    /// page number
    pub page_no: usize,
    /// span
    pub span: Span,
}

/// render queue
pub struct RenderQueue {
    render_queue: Sender<RenderTask>,
}

impl RenderQueue {
    pub async fn new() -> Result<Self, FerrpdfError> {
        let (render_sender, render_rx) = mpsc::channel::<RenderTask>(MAX_CONCURRENT_RENDER_QUEUE);

        task::spawn_blocking(move || {
            // todo panic
            pdf_render(render_rx).map_err(|err| {
                error!("init pdfium error: {}", err);
            })
        });

        Ok(Self {
            render_queue: render_sender,
        })
    }

    pub async fn add_task(&self, task: RenderTask) {
        // todo: Snafu
        self.render_queue.send(task).await.unwrap();
    }
}

impl<'a> Task<'a> for RenderTask {
    type Output = RenderResult;

    type Extra = &'a Pdfium;

    fn run(&self, pdfium: Self::Extra) -> Result<Self::Output, FerrpdfError> {
        let _guard = self.span.enter();

        let pdf_id = self.page_id.to_string();
        let page_start = self.range.start;
        info!(
            "start render task for pdf {pdf_id}, range {:?}.",
            self.range
        );

        let pdf = pdfium
            .load_pdf_from_byte_slice(&self.document, self.passowrd.as_deref())
            .context(PdfiumSnafu { stage: "load-pdf" })?;
        if page_start > pdf.pages().len() as _ {
            info!("render pdf {pdf_id} task range is empty");
            return Ok(RenderResult {
                page_id: self.page_id,
                images: Vec::new(),
            });
        }

        let render_start = Instant::now();
        let render_result = pdf
            .pages()
            .iter()
            .skip(page_start)
            .take(self.range.end - page_start)
            .enumerate()
            .map(|(page_no, page)| {
                info!("start render pdf {pdf_id} page {}.", page_no + page_start);

                let width = page.width().value;
                let height = page.height().value;

                let target_width = YOLOV12_INPUT_IMAGE_WIDTH as f32;
                let target_height = YOLOV12_INPUT_IMAGE_HEIGHT as f32;
                let target_size = Vec2::new(target_width, target_height);
                let pdf_size = Vec2::new(width, height);

                let scale = f32::min(width / target_width, height / target_height);

                let render_config = PdfRenderConfig::new()
                    .scale_page_by_factor(scale)
                    .rotate_if_landscape(PdfPageRenderRotation::Degrees90, true);

                let instans = Instant::now();
                let pdfimage = page
                    .render_with_config(&render_config)
                    .context(PdfiumSnafu { stage: "render" })
                    .map(|image| image.as_image())
                    .map(|image| PdfImage {
                        image,
                        target_size,
                        pdf_size,
                        scale,
                        page_no,
                        span: Span::current(),
                    });

                info!(
                    "render pdf {pdf_id} page {} in {}",
                    page_no + page_start,
                    instans.elapsed().as_micros()
                );
                pdfimage
            })
            .collect::<Vec<_>>();

        info!(
            "render pdf {pdf_id} const {}",
            render_start.elapsed().as_millis()
        );

        Ok(RenderResult {
            page_id: self.page_id,
            images: render_result,
        })
    }
}

fn pdf_render(mut render_rx: Receiver<RenderTask>) -> Result<(), FerrpdfError> {
    let pdfium_lib_path = std::env::var(PDFIUM_LIB_PATH_ENV_NAME).context(EnvNotFoundSnafu {
        name: PDFIUM_LIB_PATH_ENV_NAME,
    })?;
    let pdfium = Pdfium::new(
        Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path(
            &pdfium_lib_path,
        ))
        .context(PdfiumSnafu {
            stage: "load-dyn-lib",
        })?,
    );

    while let Some(render_task) = render_rx.blocking_recv() {
        let _result = render_task.run(&pdfium);
        // todo send result to layout
    }

    Ok(())
}
