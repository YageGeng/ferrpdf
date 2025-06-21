use std::collections::VecDeque;
use std::time::Duration;

use glam::Vec2;
use image::DynamicImage;
use ort::session::Session;
use pdfium_render::prelude::*;
use snafu::ResultExt;
use tracing::{error, info};

use crate::analysis::bbox::Bbox;
use crate::error::{EnvNotFoundSnafu, FerrpdfError, PdfiumSnafu};
use crate::inference::model::OnnxSession;
use crate::inference::paddle::detect::*;
use crate::inference::paddle::recognize::*;
use crate::inference::yolov12::*;
use crate::layout::{element::Layout, page::Page};

/// Configuration for the PDF parser
#[derive(Debug, Clone)]
pub struct ParserConfig {
    /// Minimum text density ratio (text_length / bbox_area) to trigger OCR
    /// If the ratio is below this threshold, OCR will be used
    pub text_density_threshold: f32,
    /// Maximum number of concurrent tasks in the queue
    pub max_concurrent_tasks: usize,
    /// Timeout for task processing (in seconds)
    pub task_timeout: Duration,
    /// Whether to enable debug mode (save intermediate images)
    pub debug_mode: bool,
    /// Output directory for debug images
    pub debug_output_dir: Option<String>,
}

impl Default for ParserConfig {
    fn default() -> Self {
        Self {
            text_density_threshold: 0.1, // 10% text density threshold
            max_concurrent_tasks: 4,
            task_timeout: Duration::from_secs(30),
            debug_mode: false,
            debug_output_dir: None,
        }
    }
}

/// Task for processing a single page
#[derive(Debug)]
pub struct PageTask {
    pub pdf_path: String,
    pub page_num: usize,
    pub priority: u32, // Higher number = higher priority
}

impl PageTask {
    pub fn new(pdf_path: String, page_num: usize, priority: u32) -> Self {
        Self {
            pdf_path,
            page_num,
            priority,
        }
    }
}

/// Result of page processing
#[derive(Debug, Clone)]
pub struct PageResult {
    pub page_num: usize,
    pub page: Page,
    pub processing_time: Duration,
    pub success: bool,
    pub error: Option<String>,
}

/// Task queue for processing PDF pages with concurrency and priority
pub struct TaskQueue {
    tasks: VecDeque<PageTask>,
}

impl TaskQueue {
    pub fn new(_config: ParserConfig) -> Self {
        Self {
            tasks: VecDeque::new(),
        }
    }

    /// Add a task to the queue
    pub fn add_task(&mut self, task: PageTask) {
        self.tasks.push_back(task);
    }

    /// Get the next task from the queue
    pub fn get_next_task(&mut self) -> Option<PageTask> {
        self.tasks.pop_front()
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

    /// Get queue size
    pub fn size(&self) -> usize {
        self.tasks.len()
    }
}

/// Main PDF parser that encapsulates pdfium and manages inference sessions
pub struct PdfParser {
    pdfium: Pdfium,
    layout_session: YoloSession<Yolov12>,
    text_det_session: PaddleDetSession<PaddleDet>,
    ocr_session: PaddleRecSession<PaddleRec>,
    task_queue: TaskQueue,
    config: ParserConfig,
}

impl PdfParser {
    /// Create a new PDF parser with the given configuration
    pub fn new(config: ParserConfig) -> Result<Self, FerrpdfError> {
        info!("Initializing PDF parser");

        let pdfium = Self::initialize_pdfium()?;
        let layout_session = Self::initialize_layout_session()?;
        let text_det_session = Self::initialize_text_det_session()?;
        let ocr_session = Self::initialize_ocr_session()?;
        let task_queue = TaskQueue::new(config.clone());

        info!("PDF parser initialized successfully");
        Ok(Self {
            pdfium,
            layout_session,
            text_det_session,
            ocr_session,
            task_queue,
            config,
        })
    }

    /// Process a PDF file and return all pages
    pub fn process_pdf(&mut self, pdf_path: &str) -> Result<Vec<Page>, FerrpdfError> {
        info!("Processing PDF: {}", pdf_path);

        // Load document once - extract pdfium reference to avoid borrow conflicts
        let pdfium = &self.pdfium;
        let document =
            pdfium
                .load_pdf_from_file(pdf_path, None)
                .map_err(|e| FerrpdfError::IoWrite {
                    source: std::io::Error::other(e.to_string()),
                    path: pdf_path.to_string(),
                })?;

        let total_pages = document.pages().len() as usize;
        info!("PDF has {} pages", total_pages);

        let mut pages = Vec::new();

        // Process each page
        for page_num in 0..total_pages {
            info!("Processing page {}/{}", page_num + 1, total_pages);

            // Get the page directly to avoid borrow checker issues
            let page =
                document
                    .pages()
                    .get(page_num as u16)
                    .map_err(|e| FerrpdfError::IoWrite {
                        source: std::io::Error::other(e.to_string()),
                        path: "pdf_page".to_string(),
                    })?;

            // Process the page step by step to avoid borrow conflicts
            let page_result = Self::process_page_static(
                &page,
                page_num,
                &mut self.layout_session,
                &mut self.text_det_session,
                &mut self.ocr_session,
                &self.config,
            );

            match page_result {
                Ok(page) => pages.push(page),
                Err(e) => {
                    error!("Failed to process page {}: {}", page_num, e);
                    // Create an empty page as fallback
                    let empty_page = Page {
                        width: 0.0,
                        height: 0.0,
                        blocks: Vec::new(),
                        page_no: page_num,
                    };
                    pages.push(empty_page);
                }
            }
        }

        info!("PDF processing completed. Processed {} pages", pages.len());
        Ok(pages)
    }

    /// Static method to process a single page to avoid borrow conflicts
    fn process_page_static(
        page: &PdfPage,
        page_num: usize,
        layout_session: &mut YoloSession<Yolov12>,
        text_det_session: &mut PaddleDetSession<PaddleDet>,
        ocr_session: &mut PaddleRecSession<PaddleRec>,
        config: &ParserConfig,
    ) -> Result<Page, FerrpdfError> {
        // Step 1: Render page to image
        let (image, scale) = Self::render_page_to_image_static(page)?;
        let page_size = Vec2::new(page.width().value, page.height().value);

        // Step 2: Perform layout analysis
        let layouts =
            Self::perform_layout_analysis_static(layout_session, &image, scale, page_num)?;

        // Step 3: Extract text from layouts using pdfium
        let mut layouts_with_text = Self::extract_text_from_pdfium_static(page, layouts)?;

        // Step 4: Evaluate text density and use OCR if needed
        Self::enhance_text_with_ocr_static(
            &mut layouts_with_text,
            &image,
            scale,
            text_det_session,
            ocr_session,
            config,
        )?;

        // Step 5: Create page result
        Ok(Page {
            width: page_size.x,
            height: page_size.y,
            blocks: layouts_with_text,
            page_no: page_num,
        })
    }

    /// Static method to render page to image
    fn render_page_to_image_static(page: &PdfPage) -> Result<(DynamicImage, f32), FerrpdfError> {
        let width = page.width().value;
        let height = page.height().value;
        let scale = f32::min(1024.0 / width, 1024.0 / height);

        let render_config = PdfRenderConfig::new()
            .scale_page_by_factor(scale)
            .rotate_if_landscape(PdfPageRenderRotation::Degrees90, true);

        let image = page
            .render_with_config(&render_config)
            .map_err(|e| FerrpdfError::IoWrite {
                source: std::io::Error::other(e.to_string()),
                path: "render".to_string(),
            })?
            .as_image()
            .into_rgb8();

        Ok((DynamicImage::ImageRgb8(image), scale))
    }

    /// Static method to perform layout analysis
    fn perform_layout_analysis_static(
        layout_session: &mut YoloSession<Yolov12>,
        image: &DynamicImage,
        scale: f32,
        page_num: usize,
    ) -> Result<Vec<Layout>, FerrpdfError> {
        let image_size = Vec2::new(image.width() as f32, image.height() as f32);
        let pdf_size = image_size / scale;

        // Create document metadata for layout analysis
        let doc_meta = crate::inference::yolov12::session::DocMeta {
            pdf_size,
            image_size,
            scale,
            page: page_num,
        };

        let layouts = layout_session.run(image, doc_meta)?;
        info!("Layout analysis found {} elements", layouts.len());

        Ok(layouts)
    }

    /// Static method to extract text from pdfium
    fn extract_text_from_pdfium_static(
        page: &PdfPage,
        mut layouts: Vec<Layout>,
    ) -> Result<Vec<Layout>, FerrpdfError> {
        for layout in &mut layouts {
            // Extract text from the region
            let text = Self::extract_text_from_region_static(page, &layout.bbox)?;
            layout.text = Some(text);
        }

        Ok(layouts)
    }

    /// Static method to extract text from region
    fn extract_text_from_region_static(
        page: &PdfPage,
        _bbox: &Bbox,
    ) -> Result<String, FerrpdfError> {
        // For now, extract all text from the page
        // TODO: Implement region-based text extraction when pdfium API is available
        let text = page
            .text()
            .map_err(|e| FerrpdfError::IoWrite {
                source: std::io::Error::other(e.to_string()),
                path: "text_extract".to_string(),
            })?
            .to_string();

        Ok(text)
    }

    /// Static method to enhance text with OCR
    fn enhance_text_with_ocr_static(
        layouts: &mut [Layout],
        image: &DynamicImage,
        scale: f32,
        text_det_session: &mut PaddleDetSession<PaddleDet>,
        ocr_session: &mut PaddleRecSession<PaddleRec>,
        config: &ParserConfig,
    ) -> Result<(), FerrpdfError> {
        for layout in layouts {
            if let Some(text) = &layout.text {
                // Calculate text density ratio
                let text_length = text.len() as f32;
                let bbox_area = layout.bbox.area();
                let text_density = if bbox_area > 0.0 {
                    text_length / bbox_area
                } else {
                    0.0
                };

                // If text density is below threshold, use OCR
                if text_density < config.text_density_threshold {
                    info!(
                        "Low text density ({:.3}) for layout, using OCR",
                        text_density
                    );

                    let ocr_text = Self::extract_text_with_ocr_static(
                        layout.bbox,
                        image,
                        scale,
                        text_det_session,
                        ocr_session,
                    )?;
                    if !ocr_text.is_empty() {
                        layout.text = Some(ocr_text);
                        info!("OCR enhanced text: '{}'", layout.text.as_ref().unwrap());
                    }
                }
            }
        }

        Ok(())
    }

    /// Static method to extract text with OCR
    fn extract_text_with_ocr_static(
        bbox: Bbox,
        image: &DynamicImage,
        _scale: f32,
        text_det_session: &mut PaddleDetSession<PaddleDet>,
        ocr_session: &mut PaddleRecSession<PaddleRec>,
    ) -> Result<String, FerrpdfError> {
        // Crop the image region
        let cropped_image = Self::crop_image_region_static(image, &bbox)?;

        // Detect text lines
        let detections = text_det_session.run(&cropped_image, DetExtra::default())?;

        if detections.is_empty() {
            return Ok(String::new());
        }

        // Recognize text for each detection
        let mut ocr_texts = Vec::new();
        for detection in detections {
            let text_bbox = detection.bbox;
            let text_image = Self::crop_image_region_static(&cropped_image, &text_bbox)?;

            // Perform OCR - use () as Extra type for paddle recognize
            let ocr_result = ocr_session.run(&text_image, ())?;
            ocr_texts.push(ocr_result);
        }

        Ok(ocr_texts.join(" "))
    }

    /// Static method to crop image region
    fn crop_image_region_static(
        image: &DynamicImage,
        bbox: &Bbox,
    ) -> Result<DynamicImage, FerrpdfError> {
        let bbox = bbox.clamp(
            Vec2::ZERO,
            Vec2::new(image.width() as f32, image.height() as f32),
        );

        let x = bbox.min.x as u32;
        let y = bbox.min.y as u32;
        let width = (bbox.max.x - bbox.min.x) as u32;
        let height = (bbox.max.y - bbox.min.y) as u32;

        if width == 0 || height == 0 {
            return Err(FerrpdfError::IoWrite {
                source: std::io::Error::other("Invalid crop region".to_string()),
                path: "crop".to_string(),
            });
        }

        let cropped = image.crop_imm(x, y, width, height);
        Ok(cropped)
    }

    /// Process tasks in the queue asynchronously
    pub fn process_queue(&mut self) -> Result<Vec<PageResult>, FerrpdfError> {
        let mut results = Vec::new();

        // Process tasks sequentially for now
        // TODO: Implement proper concurrent processing
        while let Some(task) = self.task_queue.get_next_task() {
            info!("Processing page {} from {}", task.page_num, task.pdf_path);

            let start_time = std::time::Instant::now();
            let result = Self::process_task(&task, &self.config);
            let processing_time = start_time.elapsed();

            let (page, success, error) = match result {
                Ok(page) => (page, true, None),
                Err(e) => (
                    Page {
                        width: 0.0,
                        height: 0.0,
                        blocks: Vec::new(),
                        page_no: task.page_num,
                    },
                    false,
                    Some(e.to_string()),
                ),
            };

            let page_result = PageResult {
                page_num: task.page_num,
                page,
                processing_time,
                success,
                error,
            };

            results.push(page_result);
        }

        Ok(results)
    }

    /// Process a single task (static method for worker threads)
    fn process_task(_task: &PageTask, _config: &ParserConfig) -> Result<Page, FerrpdfError> {
        // This would need to be implemented with proper session management
        // For now, return an error as this is a placeholder
        Err(FerrpdfError::IoWrite {
            source: std::io::Error::other("Task processing not implemented".to_string()),
            path: "task".to_string(),
        })
    }

    // Initialization methods
    fn initialize_pdfium() -> Result<Pdfium, FerrpdfError> {
        let pdfium_lib_path =
            std::env::var("PDFIUM_DYNAMIC_LIB_PATH").context(EnvNotFoundSnafu {
                name: "PDFIUM_DYNAMIC_LIB_PATH",
            })?;
        let pdfium = Pdfium::new(
            Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path(
                &pdfium_lib_path,
            ))
            .context(PdfiumSnafu {
                stage: "load-library",
            })?,
        );
        Ok(pdfium)
    }

    fn initialize_layout_session() -> Result<YoloSession<Yolov12>, FerrpdfError> {
        let session_builder = Session::builder().map_err(|e| FerrpdfError::OrtInit {
            source: e,
            stage: "layout".to_string(),
        })?;

        let model = Yolov12::new();
        YoloSession::new(session_builder, model)
    }

    fn initialize_text_det_session() -> Result<PaddleDetSession<PaddleDet>, FerrpdfError> {
        let session_builder = Session::builder().map_err(|e| FerrpdfError::OrtInit {
            source: e,
            stage: "text_det".to_string(),
        })?;

        let model = PaddleDet::new();
        PaddleDetSession::new(session_builder, model)
    }

    fn initialize_ocr_session() -> Result<PaddleRecSession<PaddleRec>, FerrpdfError> {
        let session_builder = Session::builder().map_err(|e| FerrpdfError::OrtInit {
            source: e,
            stage: "ocr".to_string(),
        })?;

        let model = PaddleRec::new();
        PaddleRecSession::new(session_builder, model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser_config_default() {
        let config = ParserConfig::default();
        assert_eq!(config.text_density_threshold, 0.1);
        assert_eq!(config.max_concurrent_tasks, 4);
        assert!(!config.debug_mode);
    }

    #[test]
    fn test_task_queue() {
        let config = ParserConfig::default();
        let mut queue = TaskQueue::new(config);

        // Add some tasks
        let task1 = PageTask {
            pdf_path: "test1.pdf".to_string(),
            page_num: 0,
            priority: 1,
        };
        let task2 = PageTask {
            pdf_path: "test2.pdf".to_string(),
            page_num: 1,
            priority: 2,
        };

        queue.add_task(task1);
        queue.add_task(task2);

        assert_eq!(queue.size(), 2);
        assert!(!queue.is_empty());

        // Get tasks
        let task = queue.get_next_task().unwrap();
        assert_eq!(task.pdf_path, "test1.pdf");
        assert_eq!(queue.size(), 1);

        let task = queue.get_next_task().unwrap();
        assert_eq!(task.pdf_path, "test2.pdf");
        assert_eq!(queue.size(), 0);
        assert!(queue.is_empty());
    }
}
