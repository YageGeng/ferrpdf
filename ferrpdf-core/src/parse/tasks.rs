use std::{
    ops::{Deref, Range},
    sync::Arc,
};

use futures::future;
use glam::Vec2;
use image::DynamicImage;
use pdfium_render::prelude::{PdfDocument, PdfPageRenderRotation, PdfRenderConfig};
use plsfix::fix_text;
use snafu::ResultExt;
use tracing::*;
use uuid::Uuid;

use crate::{
    analysis::labels::Label,
    consts::*,
    error::{FerrpdfError, PdfiumSnafu},
    inference::{
        paddle::detect::TextDetection,
        yolov12::{DocMeta, PdfLayouts},
    },
    layout::element::{Layout, Ocr},
    parse::{
        parser::TextExtraMode,
        task_queue::{
            ImageRenderingInput, ImageRenderingOutput, LayoutRecognitionInput,
            LayoutRecognitionOutput, LackTextBlock, OcrInput, OcrOutput, PdfPage, Task,
            TaskContext, TaskInput, TaskOutput, TaskPriority, TaskType, TextDetectionInput,
            TextDetectionOutput, TextExtractionInput, TextExtractionOutput,
        },
    },
};

/// Task for rendering PDF pages to images
#[derive(Debug)]
pub struct ImageRenderingTask {
    id: Uuid,
    priority: TaskPriority,
}

impl ImageRenderingTask {
    pub fn new(priority: TaskPriority) -> Self {
        Self {
            id: Uuid::new_v4(),
            priority,
        }
    }
}

#[async_trait::async_trait]
impl Task for ImageRenderingTask {
    type Input = TaskInput;
    type Output = TaskOutput;

    fn task_type(&self) -> TaskType {
        TaskType::ImageRendering
    }

    fn priority(&self) -> TaskPriority {
        self.priority
    }

    fn id(&self) -> Uuid {
        self.id
    }

    async fn execute(
        &self,
        input: Self::Input,
        _context: &TaskContext,
    ) -> Result<Self::Output, FerrpdfError> {
        let input = match input {
            TaskInput::ImageRendering(input) => input,
            _ => {
                return Err(FerrpdfError::ParserErr {
                    stage: "image-rendering".to_string(),
                    path: "unknown".to_string(),
                    message: "Invalid input type for ImageRenderingTask".to_string(),
                });
            }
        };

        info!("Starting image rendering for pages {:?}", input.page_range);
        let pdf_images = render_pdf_pages(&input.document, &input.page_range)?;
        info!("Completed image rendering, generated {} images", pdf_images.len());

        Ok(TaskOutput::ImageRendering(ImageRenderingOutput {
            pdf_pages: pdf_images,
        }))
    }
}

/// Task for layout recognition using YOLO
#[derive(Debug)]
pub struct LayoutRecognitionTask {
    id: Uuid,
    priority: TaskPriority,
}

impl LayoutRecognitionTask {
    pub fn new(priority: TaskPriority) -> Self {
        Self {
            id: Uuid::new_v4(),
            priority,
        }
    }
}

#[async_trait::async_trait]
impl Task for LayoutRecognitionTask {
    type Input = TaskInput;
    type Output = TaskOutput;

    fn task_type(&self) -> TaskType {
        TaskType::LayoutRecognition
    }

    fn priority(&self) -> TaskPriority {
        self.priority
    }

    fn id(&self) -> Uuid {
        self.id
    }

    async fn execute(
        &self,
        input: Self::Input,
        context: &TaskContext,
    ) -> Result<Self::Output, FerrpdfError> {
        let input = match input {
            TaskInput::LayoutRecognition(input) => input,
            _ => {
                return Err(FerrpdfError::ParserErr {
                    stage: "layout-recognition".to_string(),
                    path: "unknown".to_string(),
                    message: "Invalid input type for LayoutRecognitionTask".to_string(),
                });
            }
        };

        info!("Starting layout analysis for {} pages", input.pdf_pages.len());
        let layouts = analyze_layouts(&input.pdf_pages, context).await?;
        info!("Completed layout analysis, found {} layout groups", layouts.len());

        Ok(TaskOutput::LayoutRecognition(LayoutRecognitionOutput {
            layouts,
        }))
    }
}

/// Task for text extraction from PDF
#[derive(Debug)]
pub struct TextExtractionTask {
    id: Uuid,
    priority: TaskPriority,
}

impl TextExtractionTask {
    pub fn new(priority: TaskPriority) -> Self {
        Self {
            id: Uuid::new_v4(),
            priority,
        }
    }
}

#[async_trait::async_trait]
impl Task for TextExtractionTask {
    type Input = TaskInput;
    type Output = TaskOutput;

    fn task_type(&self) -> TaskType {
        TaskType::TextExtraction
    }

    fn priority(&self) -> TaskPriority {
        self.priority
    }

    fn id(&self) -> Uuid {
        self.id
    }

    async fn execute(
        &self,
        input: Self::Input,
        context: &TaskContext,
    ) -> Result<Self::Output, FerrpdfError> {
        let input = match input {
            TaskInput::TextExtraction(input) => input,
            _ => {
                return Err(FerrpdfError::ParserErr {
                    stage: "text-extraction".to_string(),
                    path: "unknown".to_string(),
                    message: "Invalid input type for TextExtractionTask".to_string(),
                });
            }
        };

        info!("Starting text extraction for {} layout groups", input.layouts.len());
        let (updated_layouts, lack_text_blocks) = extract_pdf_text(
            &input.document,
            input.layouts,
            input.text_extra_mode,
            context,
        )?;
        info!(
            "Completed text extraction, {} blocks need OCR",
            lack_text_blocks.len()
        );

        Ok(TaskOutput::TextExtraction(TextExtractionOutput {
            updated_layouts,
            lack_text_blocks,
        }))
    }
}

/// Task for text detection
#[derive(Debug)]
pub struct TextDetectionTask {
    id: Uuid,
    priority: TaskPriority,
}

impl TextDetectionTask {
    pub fn new(priority: TaskPriority) -> Self {
        Self {
            id: Uuid::new_v4(),
            priority,
        }
    }
}

#[async_trait::async_trait]
impl Task for TextDetectionTask {
    type Input = TaskInput;
    type Output = TaskOutput;

    fn task_type(&self) -> TaskType {
        TaskType::TextDetection
    }

    fn priority(&self) -> TaskPriority {
        self.priority
    }

    fn id(&self) -> Uuid {
        self.id
    }

    async fn execute(
        &self,
        input: Self::Input,
        context: &TaskContext,
    ) -> Result<Self::Output, FerrpdfError> {
        let input = match input {
            TaskInput::TextDetection(input) => input,
            _ => {
                return Err(FerrpdfError::ParserErr {
                    stage: "text-detection".to_string(),
                    path: "unknown".to_string(),
                    message: "Invalid input type for TextDetectionTask".to_string(),
                });
            }
        };

        info!("Starting text detection for {} blocks", input.lack_text_blocks.len());
        let text_detections = detect_text_in_blocks(&input.lack_text_blocks, &input.pdf_pages, context).await?;
        info!("Completed text detection");

        Ok(TaskOutput::TextDetection(TextDetectionOutput {
            text_detections,
        }))
    }
}

/// Task for OCR
#[derive(Debug)]
pub struct OcrTask {
    id: Uuid,
    priority: TaskPriority,
}

impl OcrTask {
    pub fn new(priority: TaskPriority) -> Self {
        Self {
            id: Uuid::new_v4(),
            priority,
        }
    }
}

#[async_trait::async_trait]
impl Task for OcrTask {
    type Input = TaskInput;
    type Output = TaskOutput;

    fn task_type(&self) -> TaskType {
        TaskType::Ocr
    }

    fn priority(&self) -> TaskPriority {
        self.priority
    }

    fn id(&self) -> Uuid {
        self.id
    }

    async fn execute(
        &self,
        input: Self::Input,
        context: &TaskContext,
    ) -> Result<Self::Output, FerrpdfError> {
        let input = match input {
            TaskInput::Ocr(input) => input,
            _ => {
                return Err(FerrpdfError::ParserErr {
                    stage: "ocr".to_string(),
                    path: "unknown".to_string(),
                    message: "Invalid input type for OcrTask".to_string(),
                });
            }
        };

        info!("Starting OCR for detected text regions");
        let recognized_texts = perform_ocr(&input.text_detections, &input.pdf_pages, context).await?;
        info!("Completed OCR");

        Ok(TaskOutput::Ocr(OcrOutput {
            recognized_texts,
        }))
    }
}

// Helper functions extracted from the original parser implementation

fn render_pdf_pages(
    document: &PdfDocument<'_>,
    range: &Range<u16>,
) -> Result<Vec<PdfPage>, FerrpdfError> {
    let page_number = document.pages().len();

    info!("Rendering pages within range: {:?}", range);
    let pdf_images = document
        .pages()
        .iter()
        .enumerate()
        .skip(range.start as _)
        .take(usize::min(range.len(), (page_number - range.start) as _))
        .flat_map(|(idx, page)| {
            let width = page.width().value;
            let height = page.height().value;

            let target_width = YOLOV12_INPUT_IMAGE_WIDTH as f32;
            let target_height = YOLOV12_INPUT_IMAGE_HEIGHT as f32;
            let target_size = Vec2::new(target_width, target_height);
            let pdf_size = Vec2::new(width, height);

            let scale = f32::min(target_width / width, target_height / height);

            let render_config = PdfRenderConfig::new()
                .scale_page_by_factor(scale)
                .rotate_if_landscape(PdfPageRenderRotation::Degrees90, true);

            page.render_with_config(&render_config)
                .map(|img| img.as_image())
                .map(|pdf_image| PdfPage {
                    metadata: DocMeta {
                        scale,
                        pdf_size,
                        page: idx,
                        image_size: target_size,
                    },
                    image: pdf_image,
                })
        })
        .collect::<Vec<_>>();

    Ok(pdf_images)
}

async fn analyze_layouts(
    pdf_images: &[PdfPage],
    context: &TaskContext,
) -> Result<Vec<PdfLayouts>, FerrpdfError> {
    let layout = Arc::clone(&context.layout_session);

    let layout_tasks = pdf_images
        .iter()
        .map(|pdf_page| {
            async {
                let mut layout = layout.lock().await;
                layout
                    .run_async(&pdf_page.image, pdf_page.metadata, &context.run_options)
                    .in_current_span()
                    .await
            }
            .in_current_span()
        })
        .collect::<Vec<_>>();

    info!("Executing layout analysis tasks.");
    let mut result = future::join_all(layout_tasks)
        .await
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

    // sort by page index
    info!("Sorting layout analysis results by page index.");
    result.sort_by(|a, b| a.metadata.page.cmp(&b.metadata.page));

    Ok(result)
}

fn extract_pdf_text(
    document: &PdfDocument<'_>,
    mut layouts: Vec<PdfLayouts>,
    extra_mode: TextExtraMode,
    context: &TaskContext,
) -> Result<(Vec<PdfLayouts>, Vec<LackTextBlock>), FerrpdfError> {
    info!("Identifying layouts requiring OCR.");
    let mut need_ocr_layouts = Vec::new();

    for pdf_layouts in layouts.iter_mut() {
        let metadata = pdf_layouts.metadata;
        let page = if extra_mode.need_pdf() {
            info!("Fetching page {} from PDF document.", metadata.page);
            let page = document
                .pages()
                .get(metadata.page as u16)
                .context(PdfiumSnafu { stage: "get-page" })?;
            Some(page)
        } else {
            None
        };

        for (layout_index, layout) in pdf_layouts.layouts.iter_mut().enumerate() {
            let pdf_text = if extra_mode.need_pdf() {
                let page_text = page
                    .as_ref()
                    .unwrap()
                    .text()
                    .context(PdfiumSnafu { stage: "text" })?;
                let pdf_bbox = layout.bbox.to_pdf_rect(metadata.pdf_size.y);
                let mut text = page_text.inside_rect(pdf_bbox);

                if context.config.auto_clean_text {
                    text = fix_text(&text, None);
                }
                Some(text)
            } else {
                None
            };

            layout.ocr = Some(Ocr { is_ocr: false });

            let is_text_empty = pdf_text.as_ref().is_none_or(|text| text.is_empty());
            layout.text = pdf_text;

            if extra_mode.need_ocr()
                && (matches!(layout.label, Label::Picture) || is_text_empty)
            {
                need_ocr_layouts.push(LackTextBlock {
                    layout_index,
                    page_no: metadata.page,
                    scale: metadata.scale,
                    bbox: layout.bbox.clone(),
                });
            }
        }
    }

    info!(
        "Completed text extraction. {} layouts require OCR.",
        need_ocr_layouts.len()
    );
    Ok((layouts, need_ocr_layouts))
}

async fn detect_text_in_blocks(
    lack_text_blocks: &[LackTextBlock],
    pdf_images: &[PdfPage],
    context: &TaskContext,
) -> Result<Vec<(usize, usize, Result<Vec<TextDetection>, FerrpdfError>)>, FerrpdfError> {
    info!(
        "Starting text detection tasks for {} blocks.",
        lack_text_blocks.len()
    );

    let detect_tasks = lack_text_blocks
        .iter()
        .enumerate()
        .map(|(idx, lack_block)| async move {
            let mut detect_session = context.detect_session.lock().await;
            let page_no = lack_block.page_no;
            let image = &pdf_images[page_no].image;

            // Create a temporary layout for detection
            let temp_layout = Layout {
                bbox: lack_block.bbox.clone(),
                label: Label::Text, // Default label for detection
                page_no: lack_block.page_no,
                bbox_id: idx,
                proba: 1.0,
                ocr: None,
                text: None,
            };

            let detections = detect_session
                .detect_text_in_layout_async(
                    image,
                    &temp_layout,
                    lack_block.scale,
                    &context.run_options,
                )
                .instrument(info_span!("detect"))
                .await;
            (idx, page_no, detections)
        })
        .collect::<Vec<_>>();

    info!("Start text detection tasks.");
    let text_detections = future::join_all(detect_tasks).in_current_span().await;
    info!("Completed text detection tasks.");

    Ok(text_detections)
}

async fn perform_ocr(
    text_detections: &[(usize, usize, Result<Vec<TextDetection>, FerrpdfError>)],
    pdf_images: &[PdfPage],
    context: &TaskContext,
) -> Result<Vec<(usize, Vec<Result<String, FerrpdfError>>)>, FerrpdfError> {
    info!("Starting text recognition tasks.");

    let recognize_tasks = text_detections
        .iter()
        .filter_map(|(idx, page_no, detect_result)| {
            detect_result
                .as_ref()
                .ok()
                .and_then(|detections| {
                    if detections.is_empty() {
                        None
                    } else {
                        Some((*idx, *page_no, detections.clone()))
                    }
                })
        })
        .map(|(idx, page_no, text_detections)| async move {
            let image = &pdf_images[page_no].image;
            let image = Arc::new(image);

            let recognize_tasks = text_detections
                .into_iter()
                .map(|text_detection| {
                    let image = Arc::clone(&image);
                    async move {
                        let mut ocr_session = context.recognizer_session.lock().await;
                        ocr_session
                            .recognize_text_region_async(&image, text_detection.bbox, &context.run_options)
                            .instrument(info_span!("recognize"))
                            .await
                    }
                })
                .collect::<Vec<_>>();

            let recognition_results = future::join_all(recognize_tasks).in_current_span().await;
            (idx, recognition_results)
        })
        .collect::<Vec<_>>();

    info!("Start text recognition tasks.");
    let results = future::join_all(recognize_tasks)
        .in_current_span()
        .await;
    info!("Completed text recognition tasks.");

    Ok(results)
}