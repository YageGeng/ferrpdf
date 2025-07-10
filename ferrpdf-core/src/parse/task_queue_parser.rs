use std::{
    collections::HashMap,
    ops::Range,
    path::{Path, PathBuf},
    sync::Arc,
};

use derive_builder::Builder;
use ort::session::RunOptions;
use pdfium_render::prelude::{PdfDocument, Pdfium};
use snafu::ResultExt;
use tokio::sync::{Mutex, RwLock};
use tracing::*;
use uuid::Uuid;

use crate::{
    error::{EnvNotFoundSnafu, FerrpdfError, PdfiumSnafu, RunConfigSnafu},
    inference::{
        model::{OnnxSession, session_builder},
        paddle::{
            detect::{PaddleDet, PaddleDetSession},
            recognize::{PaddleRec, PaddleRecSession},
        },
        yolov12::{PdfLayouts, YoloSession, Yolov12},
    },
    layout::element::{Layout, Ocr},
    parse::{
        parser::{ParserConfig, Pdf, TextExtraMode},
        task_queue::{
            ImageRenderingInput, ImageRenderingOutput, LayoutRecognitionInput,
            LayoutRecognitionOutput, LackTextBlock, OcrInput, OcrOutput, PdfPage, Task,
            TaskContext, TaskInput, TaskOutput, TaskPriority, TaskQueue, TaskStatus, TaskType,
            TaskWrapper, TextDetectionInput, TextDetectionOutput, TextExtractionInput,
            TextExtractionOutput,
        },
        tasks::{
            ImageRenderingTask, LayoutRecognitionTask, OcrTask, TextDetectionTask,
            TextExtractionTask,
        },
    },
    consts::*,
};

/// Enhanced PDF parser that uses a task queue for decoupled processing
pub struct TaskQueuePdfParser {
    pub pdfium: Pdfium,
    pub layout: Arc<Mutex<YoloSession<Yolov12>>>,
    pub detect: Arc<Mutex<PaddleDetSession<PaddleDet>>>,
    pub recognizer: Arc<Mutex<PaddleRecSession<PaddleRec>>>,
    pub config: ParserConfig,
    pub max_concurrent_tasks: usize,
}

impl TaskQueuePdfParser {
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
            max_concurrent_tasks: 4, // Default to 4 concurrent tasks
        })
    }

    pub fn with_config(mut self, config: ParserConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_max_concurrent_tasks(mut self, max_concurrent_tasks: usize) -> Self {
        self.max_concurrent_tasks = max_concurrent_tasks;
        self
    }

    #[tracing::instrument(skip_all, fields(pdf = %pdf.uuid))]
    pub async fn parse(&self, pdf: &Pdf) -> Result<Vec<PdfLayouts>, FerrpdfError> {
        info!("Loading PDF document.");
        let document = self.load_pdf(pdf)?;
        let document = Arc::new(document);

        // Create run options
        let run_options = RunOptions::new().context(RunConfigSnafu {
            stage: "run-options",
        })?;

        // Create task queue
        let task_queue = TaskQueue::new(
            Arc::clone(&self.layout),
            Arc::clone(&self.detect),
            Arc::clone(&self.recognizer),
            self.config.clone(),
            run_options,
            self.max_concurrent_tasks,
        );

        // Create and enqueue tasks
        info!("Creating and enqueuing processing tasks.");
        
        // Task 1: Image Rendering
        let image_rendering_task = self.create_image_rendering_task(
            Arc::clone(&document),
            pdf.range.clone(),
        ).await?;
        task_queue.enqueue_task(image_rendering_task).await;

        // Execute image rendering first (it's a dependency for other tasks)
        let mut results = task_queue.execute_all().await?;
        
        // Extract image rendering result
        let pdf_pages = self.extract_pdf_pages(&mut results)?;

        // Task 2: Layout Recognition
        let layout_recognition_task = self.create_layout_recognition_task(pdf_pages.clone()).await?;
        task_queue.enqueue_task(layout_recognition_task).await;

        // Execute layout recognition
        let mut results = task_queue.execute_all().await?;
        let mut layouts = self.extract_layouts(&mut results)?;

        // Debug output if requested
        if let Some(path) = pdf.debug.as_ref() {
            info!("Drawing layout bounding boxes for debugging.");
            self.draw_layout_bbox(path, &layouts, &pdf_pages)?;
        }

        // Task 3: Text Extraction
        let text_extraction_task = self.create_text_extraction_task(
            Arc::clone(&document),
            layouts,
            pdf.text_extra_mode,
        ).await?;
        task_queue.enqueue_task(text_extraction_task).await;

        // Execute text extraction
        let mut results = task_queue.execute_all().await?;
        let (updated_layouts, lack_text_blocks) = self.extract_text_extraction_results(&mut results)?;
        layouts = updated_layouts;

        // If there are blocks that need OCR, proceed with detection and OCR tasks
        if !lack_text_blocks.is_empty() {
            info!("Performing text detection and OCR on missing text blocks.");
            
            // Task 4: Text Detection
            let text_detection_task = self.create_text_detection_task(
                lack_text_blocks.clone(),
                pdf_pages.clone(),
            ).await?;
            task_queue.enqueue_task(text_detection_task).await;

            // Execute text detection
            let mut results = task_queue.execute_all().await?;
            let text_detections = self.extract_text_detections(&mut results)?;

            // Debug output if requested
            if let Some(path) = pdf.debug.as_ref() {
                info!("Drawing detection bounding boxes for debugging.");
                self.draw_detection_bbox(path, &text_detections, &pdf_pages)?;
            }

            // Task 5: OCR
            let ocr_task = self.create_ocr_task(
                text_detections,
                pdf_pages.clone(),
                lack_text_blocks.clone(),
            ).await?;
            task_queue.enqueue_task(ocr_task).await;

            // Execute OCR
            let mut results = task_queue.execute_all().await?;
            let recognized_texts = self.extract_ocr_results(&mut results)?;

            // Apply OCR results back to layouts
            self.apply_ocr_results(&mut layouts, &lack_text_blocks, &recognized_texts)?;
        }

        Ok(layouts)
    }

    // Helper methods for creating tasks
    async fn create_image_rendering_task(
        &self,
        document: Arc<PdfDocument<'static>>,
        page_range: Range<u16>,
    ) -> Result<TaskWrapper, FerrpdfError> {
        let task = ImageRenderingTask::new(TaskPriority::High);
        let task_id = task.id();
        let task_type = task.task_type();
        let priority = task.priority();

        Ok(TaskWrapper {
            id: task_id,
            task_type,
            priority,
            status: Arc::new(RwLock::new(TaskStatus::Pending)),
            task: Box::new(task),
        })
    }

    async fn create_layout_recognition_task(
        &self,
        pdf_pages: Vec<PdfPage>,
    ) -> Result<TaskWrapper, FerrpdfError> {
        let task = LayoutRecognitionTask::new(TaskPriority::High);
        let task_id = task.id();
        let task_type = task.task_type();
        let priority = task.priority();

        Ok(TaskWrapper {
            id: task_id,
            task_type,
            priority,
            status: Arc::new(RwLock::new(TaskStatus::Pending)),
            task: Box::new(task),
        })
    }

    async fn create_text_extraction_task(
        &self,
        document: Arc<PdfDocument<'static>>,
        layouts: Vec<PdfLayouts>,
        text_extra_mode: TextExtraMode,
    ) -> Result<TaskWrapper, FerrpdfError> {
        let task = TextExtractionTask::new(TaskPriority::Normal);
        let task_id = task.id();
        let task_type = task.task_type();
        let priority = task.priority();

        Ok(TaskWrapper {
            id: task_id,
            task_type,
            priority,
            status: Arc::new(RwLock::new(TaskStatus::Pending)),
            task: Box::new(task),
        })
    }

    async fn create_text_detection_task(
        &self,
        lack_text_blocks: Vec<LackTextBlock>,
        pdf_pages: Vec<PdfPage>,
    ) -> Result<TaskWrapper, FerrpdfError> {
        let task = TextDetectionTask::new(TaskPriority::Normal);
        let task_id = task.id();
        let task_type = task.task_type();
        let priority = task.priority();

        Ok(TaskWrapper {
            id: task_id,
            task_type,
            priority,
            status: Arc::new(RwLock::new(TaskStatus::Pending)),
            task: Box::new(task),
        })
    }

    async fn create_ocr_task(
        &self,
        text_detections: Vec<(usize, usize, Result<Vec<crate::inference::paddle::detect::TextDetection>, FerrpdfError>)>,
        pdf_pages: Vec<PdfPage>,
        lack_text_blocks: Vec<LackTextBlock>,
    ) -> Result<TaskWrapper, FerrpdfError> {
        let task = OcrTask::new(TaskPriority::Normal);
        let task_id = task.id();
        let task_type = task.task_type();
        let priority = task.priority();

        Ok(TaskWrapper {
            id: task_id,
            task_type,
            priority,
            status: Arc::new(RwLock::new(TaskStatus::Pending)),
            task: Box::new(task),
        })
    }

    // Helper methods for extracting results
    fn extract_pdf_pages(&self, results: &mut Vec<TaskOutput>) -> Result<Vec<PdfPage>, FerrpdfError> {
        if let Some(TaskOutput::ImageRendering(output)) = results.pop() {
            Ok(output.pdf_pages)
        } else {
            Err(FerrpdfError::ParserErr {
                stage: "extract-pdf-pages".to_string(),
                path: "unknown".to_string(),
                message: "Failed to extract PDF pages from task results".to_string(),
            })
        }
    }

    fn extract_layouts(&self, results: &mut Vec<TaskOutput>) -> Result<Vec<PdfLayouts>, FerrpdfError> {
        if let Some(TaskOutput::LayoutRecognition(output)) = results.pop() {
            Ok(output.layouts)
        } else {
            Err(FerrpdfError::ParserErr {
                stage: "extract-layouts".to_string(),
                path: "unknown".to_string(),
                message: "Failed to extract layouts from task results".to_string(),
            })
        }
    }

    fn extract_text_extraction_results(
        &self,
        results: &mut Vec<TaskOutput>,
    ) -> Result<(Vec<PdfLayouts>, Vec<LackTextBlock>), FerrpdfError> {
        if let Some(TaskOutput::TextExtraction(output)) = results.pop() {
            Ok((output.updated_layouts, output.lack_text_blocks))
        } else {
            Err(FerrpdfError::ParserErr {
                stage: "extract-text-extraction".to_string(),
                path: "unknown".to_string(),
                message: "Failed to extract text extraction results".to_string(),
            })
        }
    }

    fn extract_text_detections(
        &self,
        results: &mut Vec<TaskOutput>,
    ) -> Result<Vec<(usize, usize, Result<Vec<crate::inference::paddle::detect::TextDetection>, FerrpdfError>)>, FerrpdfError> {
        if let Some(TaskOutput::TextDetection(output)) = results.pop() {
            Ok(output.text_detections)
        } else {
            Err(FerrpdfError::ParserErr {
                stage: "extract-text-detections".to_string(),
                path: "unknown".to_string(),
                message: "Failed to extract text detections".to_string(),
            })
        }
    }

    fn extract_ocr_results(
        &self,
        results: &mut Vec<TaskOutput>,
    ) -> Result<Vec<(usize, Vec<Result<String, FerrpdfError>>)>, FerrpdfError> {
        if let Some(TaskOutput::Ocr(output)) = results.pop() {
            Ok(output.recognized_texts)
        } else {
            Err(FerrpdfError::ParserErr {
                stage: "extract-ocr-results".to_string(),
                path: "unknown".to_string(),
                message: "Failed to extract OCR results".to_string(),
            })
        }
    }

    fn apply_ocr_results(
        &self,
        layouts: &mut [PdfLayouts],
        lack_text_blocks: &[LackTextBlock],
        recognized_texts: &[(usize, Vec<Result<String, FerrpdfError>>)],
    ) -> Result<(), FerrpdfError> {
        for (block_idx, recognition_results) in recognized_texts {
            if let Some(lack_block) = lack_text_blocks.get(*block_idx) {
                let mut text = recognition_results
                    .iter()
                    .filter_map(|result| result.as_ref().ok())
                    .fold(String::new(), |mut acc, text| {
                        acc.push_str(text);
                        acc.push('\n');
                        acc
                    });

                if self.config.auto_clean_text {
                    text = plsfix::fix_text(&text, None);
                }

                // Find the layout and update it
                for pdf_layouts in layouts.iter_mut() {
                    if pdf_layouts.metadata.page == lack_block.page_no {
                        if let Some(layout) = pdf_layouts.layouts.get_mut(lack_block.layout_index) {
                            layout.text = Some(text.clone());
                            if let Some(ocr) = layout.ocr.as_mut() {
                                ocr.is_ocr = true;
                            }
                        }
                        break;
                    }
                }
            }
        }
        Ok(())
    }

    // Copied methods from original parser (unchanged)
    
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

    #[tracing::instrument(skip_all)]
    fn draw_layout_bbox<P: AsRef<Path>>(
        &self,
        output: P,
        layout: &[PdfLayouts],
        images: &[PdfPage],
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
        layout: &[(usize, usize, Result<Vec<crate::inference::paddle::detect::TextDetection>, FerrpdfError>)],
        images: &[PdfPage],
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
                |mut acc: HashMap<usize, Vec<crate::inference::paddle::detect::TextDetection>>, (page_no, detections)| {
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

impl Drop for TaskQueuePdfParser {
    fn drop(&mut self) {}
}