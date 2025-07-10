use std::{
    collections::VecDeque,
    path::PathBuf,
    sync::Arc,
};

use derive_builder::Builder;
use futures::future::{self, BoxFuture};
use glam::Vec2;
use image::DynamicImage;
use ort::session::RunOptions;
use pdfium_render::prelude::PdfDocument;
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, RwLock};
use tracing::*;
use uuid::Uuid;

use crate::{
    error::FerrpdfError,
    inference::{
        paddle::{
            detect::{PaddleDetSession, PaddleDet, TextDetection},
            recognize::{PaddleRecSession, PaddleRec},
        },
        yolov12::{DocMeta, PdfLayouts, YoloSession, Yolov12},
    },
    layout::element::{Layout, Ocr},
    parse::parser::{ParserConfig, TextExtraMode},
};

/// Task types for different processing stages
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TaskType {
    ImageRendering,
    LayoutRecognition,
    TextExtraction,
    TextDetection,
    Ocr,
}

/// Task priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TaskPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Task status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    Pending,
    Running,
    Completed,
    Failed(String),
}

/// Base trait for all tasks
#[async_trait::async_trait]
pub trait Task: Send + Sync + std::fmt::Debug {
    type Input: Send + Sync;
    type Output: Send + Sync;

    fn task_type(&self) -> TaskType;
    fn priority(&self) -> TaskPriority;
    fn id(&self) -> Uuid;
    
    async fn execute(
        &self,
        input: Self::Input,
        context: &TaskContext,
    ) -> Result<Self::Output, FerrpdfError>;
}

/// Context shared across all tasks
#[derive(Debug)]
pub struct TaskContext {
    pub layout_session: Arc<Mutex<YoloSession<Yolov12>>>,
    pub detect_session: Arc<Mutex<PaddleDetSession<PaddleDet>>>,
    pub recognizer_session: Arc<Mutex<PaddleRecSession<PaddleRec>>>,
    pub config: ParserConfig,
    pub run_options: Arc<RunOptions>,
}

/// A task wrapper that contains the task and its metadata
#[derive(Debug)]
pub struct TaskWrapper {
    pub id: Uuid,
    pub task_type: TaskType,
    pub priority: TaskPriority,
    pub status: Arc<RwLock<TaskStatus>>,
    pub task: Box<dyn Task<Input = TaskInput, Output = TaskOutput>>,
}

/// Unified input type for all tasks
#[derive(Debug)]
pub enum TaskInput {
    ImageRendering(ImageRenderingInput),
    LayoutRecognition(LayoutRecognitionInput),
    TextExtraction(TextExtractionInput),
    TextDetection(TextDetectionInput),
    Ocr(OcrInput),
}

/// Unified output type for all tasks
#[derive(Debug)]
pub enum TaskOutput {
    ImageRendering(ImageRenderingOutput),
    LayoutRecognition(LayoutRecognitionOutput),
    TextExtraction(TextExtractionOutput),
    TextDetection(TextDetectionOutput),
    Ocr(OcrOutput),
}

/// Task queue that manages and executes tasks
#[derive(Debug)]
pub struct TaskQueue {
    tasks: Arc<Mutex<VecDeque<TaskWrapper>>>,
    context: TaskContext,
    max_concurrent_tasks: usize,
}

impl TaskQueue {
    pub fn new(
        layout_session: Arc<Mutex<YoloSession<Yolov12>>>,
        detect_session: Arc<Mutex<PaddleDetSession<PaddleDet>>>,
        recognizer_session: Arc<Mutex<PaddleRecSession<PaddleRec>>>,
        config: ParserConfig,
        run_options: RunOptions,
        max_concurrent_tasks: usize,
    ) -> Self {
        let context = TaskContext {
            layout_session,
            detect_session,
            recognizer_session,
            config,
            run_options: Arc::new(run_options),
        };

        Self {
            tasks: Arc::new(Mutex::new(VecDeque::new())),
            context,
            max_concurrent_tasks,
        }
    }

    /// Add a task to the queue
    pub async fn enqueue_task(&self, task: TaskWrapper) {
        let mut tasks = self.tasks.lock().await;
        
        // Insert task based on priority (higher priority first)
        let mut insert_index = tasks.len();
        for (i, existing_task) in tasks.iter().enumerate() {
            if task.priority > existing_task.priority {
                insert_index = i;
                break;
            }
        }
        
        tasks.insert(insert_index, task);
        info!("Task {} enqueued at position {}", task.id, insert_index);
    }

    /// Execute all tasks in the queue
    pub async fn execute_all(&self) -> Result<Vec<TaskOutput>, FerrpdfError> {
        let mut results = Vec::new();
        let mut running_tasks = Vec::new();

        loop {
            // Start new tasks up to the limit
            while running_tasks.len() < self.max_concurrent_tasks {
                let task = {
                    let mut tasks = self.tasks.lock().await;
                    tasks.pop_front()
                };

                match task {
                    Some(task) => {
                        let task_id = task.id;
                        let task_type = task.task_type.clone();
                        let status = Arc::clone(&task.status);
                        
                        // Mark as running
                        {
                            let mut status = status.write().await;
                            *status = TaskStatus::Running;
                        }

                        let context = &self.context;
                        let future = self.execute_task(task, context);
                        running_tasks.push((task_id, task_type, status, Box::pin(future)));
                        
                        info!("Started execution of task {} ({})", task_id, task_type);
                    }
                    None => break,
                }
            }

            if running_tasks.is_empty() {
                break;
            }

            // Wait for any task to complete
            let (result, index, _) = future::select_all(
                running_tasks.iter_mut().map(|(_, _, _, future)| future)
            ).await;

            let (task_id, task_type, status, _) = running_tasks.remove(index);

            match result {
                Ok(output) => {
                    let mut status = status.write().await;
                    *status = TaskStatus::Completed;
                    results.push(output);
                    info!("Task {} ({}) completed successfully", task_id, task_type);
                }
                Err(err) => {
                    let mut status = status.write().await;
                    *status = TaskStatus::Failed(err.to_string());
                    error!("Task {} ({}) failed: {}", task_id, task_type, err);
                    return Err(err);
                }
            }
        }

        Ok(results)
    }

    async fn execute_task(
        &self,
        task: TaskWrapper,
        context: &TaskContext,
    ) -> Result<TaskOutput, FerrpdfError> {
        // This is a simplified implementation - in reality, we'd need to match on task type
        // and create the appropriate input. For now, we'll create a placeholder.
        let input = TaskInput::ImageRendering(ImageRenderingInput {
            document: Arc::new(unsafe { std::mem::zeroed() }), // This is unsafe and just for compilation
            page_range: 0..1,
        });
        
        task.task.execute(input, context).await
    }

    /// Get the status of all tasks
    pub async fn get_task_statuses(&self) -> Vec<(Uuid, TaskType, TaskStatus)> {
        let tasks = self.tasks.lock().await;
        let mut statuses = Vec::new();
        
        for task in tasks.iter() {
            let status = task.status.read().await.clone();
            statuses.push((task.id, task.task_type.clone(), status));
        }
        
        statuses
    }
}

// Input/Output types for each task type will be defined below

#[derive(Debug)]
pub struct ImageRenderingInput {
    pub document: Arc<PdfDocument<'static>>,
    pub page_range: std::ops::Range<u16>,
}

#[derive(Debug)]
pub struct ImageRenderingOutput {
    pub pdf_pages: Vec<PdfPage>,
}

#[derive(Debug)]
pub struct PdfPage {
    pub metadata: DocMeta,
    pub image: DynamicImage,
}

#[derive(Debug)]
pub struct LayoutRecognitionInput {
    pub pdf_pages: Vec<PdfPage>,
}

#[derive(Debug)]
pub struct LayoutRecognitionOutput {
    pub layouts: Vec<PdfLayouts>,
}

#[derive(Debug)]
pub struct TextExtractionInput {
    pub document: Arc<PdfDocument<'static>>,
    pub layouts: Vec<PdfLayouts>,
    pub text_extra_mode: TextExtraMode,
}

#[derive(Debug)]
pub struct TextExtractionOutput {
    pub updated_layouts: Vec<PdfLayouts>,
    pub lack_text_blocks: Vec<LackTextBlock>,
}

#[derive(Debug)]
pub struct LackTextBlock {
    pub page_no: usize,
    pub layout_index: usize,
    pub scale: f32,
    pub bbox: crate::analysis::bbox::Bbox,
}

#[derive(Debug)]
pub struct TextDetectionInput {
    pub lack_text_blocks: Vec<LackTextBlock>,
    pub pdf_pages: Vec<PdfPage>,
}

#[derive(Debug)]
pub struct TextDetectionOutput {
    pub text_detections: Vec<(usize, usize, Result<Vec<TextDetection>, FerrpdfError>)>,
}

#[derive(Debug)]
pub struct OcrInput {
    pub text_detections: Vec<(usize, usize, Result<Vec<TextDetection>, FerrpdfError>)>,
    pub pdf_pages: Vec<PdfPage>,
    pub lack_text_blocks: Vec<LackTextBlock>,
}

#[derive(Debug)]
pub struct OcrOutput {
    pub recognized_texts: Vec<(usize, Vec<Result<String, FerrpdfError>>)>,
}

// Display implementations for better debugging
impl std::fmt::Display for TaskType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskType::ImageRendering => write!(f, "ImageRendering"),
            TaskType::LayoutRecognition => write!(f, "LayoutRecognition"),
            TaskType::TextExtraction => write!(f, "TextExtraction"),
            TaskType::TextDetection => write!(f, "TextDetection"),
            TaskType::Ocr => write!(f, "OCR"),
        }
    }
}