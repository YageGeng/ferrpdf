use std::{
    collections::HashMap,
    ops::Range,
    path::{Path, PathBuf},
    sync::Arc,
};

use futures::future;
use glam::Vec2;
use image::DynamicImage;
use ort::session::RunOptions;
use pdfium_render::prelude::{PdfDocument, PdfPageRenderRotation, PdfRenderConfig, Pdfium};
use plsfix::fix_text;
use snafu::ResultExt;
use tokio::sync::Mutex;
use tracing::*;
use uuid::Uuid;

use crate::{
    analysis::labels::Label,
    consts::*,
    error::{EnvNotFoundSnafu, FerrpdfError, PdfiumSnafu, RunConfigSnafu},
    inference::{
        model::{OnnxSession, session_builder},
        paddle::{
            detect::{PaddleDet, PaddleDetSession, TextDetection},
            recognize::{PaddleRec, PaddleRecSession},
        },
        yolov12::{DocMeta, PdfLayouts, YoloSession, Yolov12},
    },
    layout::element::{Layout, Ocr},
    parse::parser::{ParserConfig, Pdf, TextExtraMode},
};

/// Task types for different processing stages
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskType {
    ImageRendering,
    LayoutRecognition,  
    TextExtraction,
    TextDetection,
    Ocr,
}

/// Task result that contains the output and metadata
#[derive(Debug)]
pub struct TaskResult<T> {
    pub task_type: TaskType,
    pub task_id: Uuid,
    pub result: Result<T, FerrpdfError>,
    pub duration: std::time::Duration,
}

/// Enhanced PDF page with additional metadata
#[derive(Debug)]
pub struct EnhancedPdfPage {
    pub metadata: DocMeta,
    pub image: DynamicImage,
    pub page_index: usize,
}

/// Text block that lacks text and needs OCR
#[derive(Debug)]
pub struct LackTextBlock<'a> {
    pub page_no: usize,
    pub layout: Arc<Mutex<&'a mut Layout>>,
    pub scale: f32,
}

/// Refactored PDF parser using task-based architecture
pub struct TaskBasedPdfParser {
    pub pdfium: Pdfium,
    pub layout: Arc<Mutex<YoloSession<Yolov12>>>,
    pub detect: Arc<Mutex<PaddleDetSession<PaddleDet>>>,
    pub recognizer: Arc<Mutex<PaddleRecSession<PaddleRec>>>,
    pub config: ParserConfig,
}

impl TaskBasedPdfParser {
    #[tracing::instrument(skip_all)]
    pub fn new() -> Result<Self, FerrpdfError> {
        // Get pdfium library path
        info!("Fetching PDFium library path from environment variable.");
        let pdfium_lib_path =
            std::env::var(PDFIUM_LIB_PATH_ENV_NAME).context(EnvNotFoundSnafu {
                name: PDFIUM_LIB_PATH_ENV_NAME,
            })?;

        // Create pdfium instance
        info!("Creating PDFium instance.");
        let pdfium = Pdfium::new(
            Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path(
                &pdfium_lib_path,
            ))
            .context(PdfiumSnafu {
                stage: "load-dyn-lib",
            })?,
        );

        // Create sessions
        info!("Initializing layout, detection, and recognition sessions.");
        let layout = YoloSession::new(session_builder()?, Yolov12::new())?;
        let detect = PaddleDetSession::new(session_builder()?, PaddleDet::new())?;
        let recognizer = PaddleRecSession::new(session_builder()?, PaddleRec::new())?;

        Ok(Self {
            pdfium,
            layout: Arc::new(Mutex::new(layout)),
            detect: Arc::new(Mutex::new(detect)),
            recognizer: Arc::new(Mutex::new(recognizer)),
            config: ParserConfig::default(),
        })
    }

    #[tracing::instrument(skip_all,fields(pdf = %pdf.uuid))]
    pub async fn parse(&self, pdf: &Pdf) -> Result<Vec<PdfLayouts>, FerrpdfError> {
        info!("Starting task-based PDF parsing pipeline.");
        
        // Task 1: Load PDF document
        let document = self.execute_load_document_task(pdf).await?;

        // Task 2: Image Rendering
        let pdf_images = self.execute_image_rendering_task(&document, &pdf.range).await?;

        // Task 3: Layout Recognition  
        let mut layouts = self.execute_layout_recognition_task(&pdf_images).await?;

        // Debug output if requested
        if let Some(path) = pdf.debug.as_ref() {
            info!("Drawing layout bounding boxes for debugging.");
            self.draw_layout_bbox(path, &layouts, &pdf_images)?;
        }

        // Task 4: Text Extraction
        let lack_text_blocks = self.execute_text_extraction_task(&document, &mut layouts, pdf.text_extra_mode).await?;

        // Task 5 & 6: Text Detection and OCR (if needed)
        if !lack_text_blocks.is_empty() {
            info!("Performing text detection and OCR on missing text blocks.");
            self.execute_detection_and_ocr_tasks(&lack_text_blocks, &pdf_images, pdf).await?;
        }

        Ok(layouts)
    }

    /// Task 1: Load PDF Document
    #[tracing::instrument(skip_all)]
    async fn execute_load_document_task(&self, pdf: &Pdf) -> Result<PdfDocument<'_>, FerrpdfError> {
        let start = std::time::Instant::now();
        let task_id = Uuid::new_v4();
        
        info!("Executing Task 1: Load PDF Document (ID: {})", task_id);
        
        let result = self.load_pdf(pdf);
        let duration = start.elapsed();
        
        match &result {
            Ok(_) => info!("Task 1 completed successfully in {:?}", duration),
            Err(e) => error!("Task 1 failed after {:?}: {}", duration, e),
        }
        
        result
    }

    /// Task 2: Image Rendering
    #[tracing::instrument(skip_all)]
    async fn execute_image_rendering_task(
        &self,
        document: &PdfDocument<'_>,
        range: &Range<u16>,
    ) -> Result<Vec<EnhancedPdfPage>, FerrpdfError> {
        let start = std::time::Instant::now();
        let task_id = Uuid::new_v4();
        
        info!("Executing Task 2: Image Rendering (ID: {})", task_id);
        
        let result = self.render_pdf_to_images(document, range);
        let duration = start.elapsed();
        
        match &result {
            Ok(pages) => info!("Task 2 completed successfully in {:?}, rendered {} pages", duration, pages.len()),
            Err(e) => error!("Task 2 failed after {:?}: {}", duration, e),
        }
        
        result
    }

    /// Task 3: Layout Recognition
    #[tracing::instrument(skip_all)]
    async fn execute_layout_recognition_task(
        &self,
        pdf_images: &[EnhancedPdfPage],
    ) -> Result<Vec<PdfLayouts>, FerrpdfError> {
        let start = std::time::Instant::now();
        let task_id = Uuid::new_v4();
        
        info!("Executing Task 3: Layout Recognition (ID: {})", task_id);
        
        let result = self.analyze_page_layouts(pdf_images).await;
        let duration = start.elapsed();
        
        match &result {
            Ok(layouts) => info!("Task 3 completed successfully in {:?}, analyzed {} layout groups", duration, layouts.len()),
            Err(e) => error!("Task 3 failed after {:?}: {}", duration, e),
        }
        
        result
    }

    /// Task 4: Text Extraction
    #[tracing::instrument(skip_all)]
    async fn execute_text_extraction_task<'a>(
        &self,
        document: &PdfDocument<'_>,
        layouts: &'a mut [PdfLayouts],
        extra_mode: TextExtraMode,
    ) -> Result<Vec<LackTextBlock<'a>>, FerrpdfError> {
        let start = std::time::Instant::now();
        let task_id = Uuid::new_v4();
        
        info!("Executing Task 4: Text Extraction (ID: {})", task_id);
        
        let result = self.extract_text_from_pdf(document, layouts, extra_mode);
        let duration = start.elapsed();
        
        match &result {
            Ok(lack_blocks) => info!("Task 4 completed successfully in {:?}, {} blocks need OCR", duration, lack_blocks.len()),
            Err(e) => error!("Task 4 failed after {:?}: {}", duration, e),
        }
        
        result
    }

    /// Task 5 & 6: Combined Text Detection and OCR
    #[tracing::instrument(skip_all)]
    async fn execute_detection_and_ocr_tasks(
        &self,
        lack_text_blocks: &[LackTextBlock<'_>],
        pdf_images: &[EnhancedPdfPage],
        pdf: &Pdf,
    ) -> Result<(), FerrpdfError> {
        let start = std::time::Instant::now();
        let task_id = Uuid::new_v4();
        
        info!("Executing Task 5 & 6: Text Detection and OCR (ID: {})", task_id);
        
        let result = self.perform_detection_and_ocr(lack_text_blocks, pdf_images, pdf).await;
        let duration = start.elapsed();
        
        match &result {
            Ok(_) => info!("Task 5 & 6 completed successfully in {:?}", duration),
            Err(e) => error!("Task 5 & 6 failed after {:?}: {}", duration, e),
        }
        
        result
    }

    // Individual task implementations (decoupled from main flow)

    #[tracing::instrument(skip_all, fields(result))]
    fn load_pdf<'a, 'b: 'a>(&'a self, pdf: &'b Pdf) -> Result<PdfDocument<'a>, FerrpdfError> {
        info!("Loading PDF from file: {:?}", pdf.path);
        let document = self
            .pdfium
            .load_pdf_from_file(&pdf.path, pdf.password.as_deref())
            .context(PdfiumSnafu {
                stage: "load-pdf-by-path",
            })?;

        let page_number = document.pages().len();
        let range = pdf.range.clone();
        
        if page_number < range.start {
            error!(
                "Page number is less than the start of the range for PDF: {:?}",
                pdf.path
            );
            return Err(FerrpdfError::ParserErr {
                stage: "parse-pdf".to_string(),
                path: pdf.path.to_string_lossy().to_string(),
                message: "Page number is less than the start of the range".to_string(),
            });
        }

        Ok(document)
    }

    #[tracing::instrument(skip_all, fields(result))]
    fn render_pdf_to_images(
        &self,
        document: &PdfDocument<'_>,
        range: &Range<u16>,
    ) -> Result<Vec<EnhancedPdfPage>, FerrpdfError> {
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
                    .map(|pdf_image| EnhancedPdfPage {
                        metadata: DocMeta {
                            scale,
                            pdf_size,
                            page: idx,
                            image_size: target_size,
                        },
                        image: pdf_image,
                        page_index: idx,
                    })
            })
            .collect::<Vec<_>>();

        Ok(pdf_images)
    }

    #[tracing::instrument(skip_all, fields(result))]
    async fn analyze_page_layouts(
        &self,
        pdf_images: &[EnhancedPdfPage],
    ) -> Result<Vec<PdfLayouts>, FerrpdfError> {
        info!("Creating run options for layout analysis.");
        let run_option = RunOptions::new().context(RunConfigSnafu {
            stage: "run-options",
        })?;

        info!("Cloning layout session for analysis.");
        let layout = Arc::clone(&self.layout);

        let layout_tasks = pdf_images
            .iter()
            .map(|pdf_page| {
                async {
                    let mut layout = layout.lock().await;
                    layout
                        .run_async(&pdf_page.image, pdf_page.metadata, &run_option)
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

        // Sort by page index
        info!("Sorting layout analysis results by page index.");
        result.sort_by(|a, b| a.metadata.page.cmp(&b.metadata.page));

        Ok(result)
    }

    #[tracing::instrument(skip_all, fields(result))]
    fn extract_text_from_pdf<'a>(
        &self,
        document: &PdfDocument<'_>,
        layouts: &'a mut [PdfLayouts],
        extra_mode: TextExtraMode,
    ) -> Result<Vec<LackTextBlock<'a>>, FerrpdfError> {
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

            for layout in pdf_layouts.layouts.iter_mut() {
                let pdf_text = if extra_mode.need_pdf() {
                    let page_text = page
                        .as_ref()
                        .unwrap()
                        .text()
                        .context(PdfiumSnafu { stage: "text" })?;
                    let pdf_bbox = layout.bbox.to_pdf_rect(metadata.pdf_size.y);
                    let mut text = page_text.inside_rect(pdf_bbox);

                    if self.config.auto_clean_text {
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
                        layout: Arc::new(Mutex::new(layout)),
                        page_no: metadata.page,
                        scale: metadata.scale,
                    });
                }
            }
        }

        info!(
            "Completed text extraction. {} layouts require OCR.",
            need_ocr_layouts.len()
        );
        Ok(need_ocr_layouts)
    }

    #[tracing::instrument(skip_all, fields(result))]
    async fn perform_detection_and_ocr(
        &self,
        lack_text_blocks: &[LackTextBlock<'_>],
        pdf_images: &[EnhancedPdfPage],
        pdf: &Pdf,
    ) -> Result<(), FerrpdfError> {
        info!("Creating run options for text detection and OCR.");
        let run_options = RunOptions::new().context(RunConfigSnafu {
            stage: "run-options",
        })?;

        let run_options = Arc::new(run_options);

        info!(
            "Starting text detection tasks for {} blocks.",
            lack_text_blocks.len()
        );
        
        let detect_tasks = lack_text_blocks
            .iter()
            .enumerate()
            .map(|(idx, lack_block)| (idx, lack_block, Arc::clone(&run_options)))
            .map(|(idx, lack_block, run_options)| async move {
                let mut detect_session = self.detect.lock().await;
                let page_no = lack_block.page_no;
                let image = &pdf_images[page_no].image;

                let layout = lack_block.layout.lock().await;
                let detections = detect_session
                    .detect_text_in_layout_async(
                        image,
                        layout.as_ref(),
                        lack_block.scale,
                        &run_options,
                    )
                    .instrument(info_span!("detect"))
                    .await;
                (idx, page_no, detections)
            })
            .collect::<Vec<_>>();

        info!("Start text detection tasks.");
        let text_detections = future::join_all(detect_tasks).in_current_span().await;
        info!("Completed text detection tasks.");

        if let Some(path) = pdf.debug.as_ref() {
            info!("Drawing detection bounding boxes for debugging.");
            self.draw_detection_bbox(path, &text_detections, pdf_images)?;
        }

        info!("Starting text recognition tasks.");
        let recognize_tasks = text_detections
            .into_iter()
            .flat_map(|(idx, page_no, detect_result)| {
                detect_result
                    .map(|detections| {
                        // Skip empty detections
                        if detections.is_empty() {
                            None
                        } else {
                            Some((idx, page_no, detections, Arc::clone(&run_options)))
                        }
                    })
                    .map_err(|err| {
                        error!("Error detecting text in layout: {}", err);
                    })
                    .ok()
                    .flatten()
            })
            .map(|(idx, page_no, text_detections, run_options)| async move {
                let image = &pdf_images[page_no].image;
                let image = Arc::new(image);

                let recognize_tasks = text_detections
                    .into_iter()
                    .map(|text_detection| {
                        (text_detection, Arc::clone(&image), Arc::clone(&run_options))
                    })
                    .map(|(text_detection, image, run_options)| async move {
                        let mut ocr_session = self.recognizer.lock().await;
                        ocr_session
                            .recognize_text_region_async(&image, text_detection.bbox, &run_options)
                            .instrument(info_span!("recognize"))
                            .await
                    })
                    .collect::<Vec<_>>();
                (
                    idx,
                    future::join_all(recognize_tasks).in_current_span().await,
                )
            })
            .collect::<Vec<_>>();

        info!("Start text recognition tasks.");
        for (idx, recognition) in future::join_all(recognize_tasks)
            .in_current_span()
            .await
            .into_iter()
        {
            let mut text =
                recognition
                    .into_iter()
                    .flatten()
                    .fold(String::new(), |mut acc, text| {
                        acc.push_str(&text);
                        acc.push('\n');
                        acc
                    });

            if self.config.auto_clean_text {
                text = fix_text(&text, None);
            }

            let mut layout = lack_text_blocks[idx].layout.lock().await;

            layout.text = Some(text);
            if let Some(ocr) = layout.ocr.as_mut() {
                ocr.is_ocr = true;
            }
        }
        info!("Completed text recognition tasks.");

        Ok(())
    }

    // Debug helper methods (unchanged from original)
    
    #[tracing::instrument(skip_all)]
    fn draw_layout_bbox<P: AsRef<Path>>(
        &self,
        output: P,
        layout: &[PdfLayouts],
        images: &[EnhancedPdfPage],
    ) -> Result<(), FerrpdfError> {
        let path = output.as_ref();
        for (idx, PdfLayouts { layouts, metadata }) in layout.iter().enumerate() {
            debug!("layout-{}.png", metadata.page);
            YoloSession::draw(
                path.join(format!("layout-{}.png", metadata.page)),
                metadata,
                layouts,
                &images[idx].image,
            )?;
        }
        Ok(())
    }

    #[tracing::instrument(skip_all)]
    #[allow(clippy::type_complexity)]
    fn draw_detection_bbox<P: AsRef<Path>>(
        &self,
        output: P,
        layout: &[(usize, usize, Result<Vec<TextDetection>, FerrpdfError>)],
        images: &[EnhancedPdfPage],
    ) -> Result<(), FerrpdfError> {
        let grouped_detections = layout
            .iter()
            .filter_map(|(_, page_no, detections_result)| {
                detections_result
                    .as_ref()
                    .ok()
                    .map(|detections| (*page_no, detections.clone()))
            })
            .fold(
                HashMap::new(),
                |mut acc: HashMap<usize, Vec<TextDetection>>, (page_no, detections)| {
                    acc.entry(page_no).or_default().extend(detections);
                    acc
                },
            );

        let path = output.as_ref();
        for (page_no, detections) in grouped_detections.into_iter() {
            if detections.is_empty() {
                continue;
            }
            let pdf_image = &images[page_no];
            debug!("draw detections-{}", pdf_image.metadata.page);

            PaddleDetSession::draw(
                path.join(format!("detections-{}.png", pdf_image.metadata.page,)),
                &detections,
                &pdf_image.image,
            )?;
        }

        Ok(())
    }
}

impl Drop for TaskBasedPdfParser {
    fn drop(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_type_display() {
        assert_eq!(format!("{:?}", TaskType::ImageRendering), "ImageRendering");
        assert_eq!(format!("{:?}", TaskType::LayoutRecognition), "LayoutRecognition");
        assert_eq!(format!("{:?}", TaskType::TextExtraction), "TextExtraction");
        assert_eq!(format!("{:?}", TaskType::TextDetection), "TextDetection");
        assert_eq!(format!("{:?}", TaskType::Ocr), "Ocr");
    }
}