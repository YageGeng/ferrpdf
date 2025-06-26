use std::error::Error;
use std::path::{Path, PathBuf};

use clap::Parser;
use ferrpdf_core::layout::element::Layout;
use glam::Vec2;
use image::DynamicImage;
use ort::session::Session;
use pdfium_render::prelude::*;
use tracing::{error, info};

use ferrpdf_core::inference::model::OnnxSession;
use ferrpdf_core::inference::paddle::detect::*;
use ferrpdf_core::inference::paddle::recognize::*;
use ferrpdf_core::inference::yolov12::*;
use ort::execution_providers::CPUExecutionProvider;
use ort::session::builder::GraphOptimizationLevel;

#[derive(Parser)]
#[command(name = "analyze")]
#[command(about = "PDF layout analysis tool")]
struct Args {
    #[arg(help = "Input PDF file path")]
    input: String,

    #[arg(
        long,
        short('p'),
        help = "Specify pages to parse using Rust range syntax: '0..5' for pages 0-4, '1..' for pages 1 onwards, '..5' for first 5 pages, '3' for single page. If not specified, all pages will be processed."
    )]
    page: Option<String>,

    #[arg(short, long, default_value = "images", help = "Output directory")]
    output: String,

    #[arg(
        short,
        long,
        default_value = "false",
        help = "Enable debug mode to save debug images"
    )]
    debug: bool,
}

/// Main analyzer struct that holds initialized resources
pub struct Analyzer {
    session: YoloSession<Yolov12>,
    text_det_session: PaddleDetSession<PaddleDet>,
    ocr_session: PaddleRecSession<PaddleRec>,
    pdfium: Pdfium,
}

impl Analyzer {
    /// Initialize the analyzer with YOLOv12 session, text detection session, OCR session, and PDFium
    pub fn new() -> Result<Self, Box<dyn Error>> {
        info!("Initializing analyzer resources");

        let session = Self::initialize_yolo_session()?;
        let text_det_session = Self::initialize_text_det_session()?;
        let ocr_session = Self::initialize_ocr_session()?;
        let pdfium = Self::initialize_pdfium()?;

        info!("Analyzer initialized successfully");
        Ok(Self {
            session,
            text_det_session,
            ocr_session,
            pdfium,
        })
    }

    /// Analyze a single page of a PDF document
    pub fn analyze_page(
        &mut self,
        pdf_path: &str,
        page_num: usize,
        output_dir: &Path,
        debug: bool,
    ) -> Result<Vec<Layout>, Box<dyn Error>> {
        info!("Analyzing page {} of {}", page_num, pdf_path);

        // Step 1: Load PDF and convert page to image
        let (image, scale) = {
            let (image_data, _document) = self.load_pdf_page(pdf_path, page_num)?;
            image_data
        };

        // Step 2: Perform layout analysis
        let (mut layouts, doc_meta) = self.perform_layout_analysis(&image, scale, page_num)?;

        // Step 3: Save analysis results (only if debug mode is enabled)
        if debug {
            self.save_analysis_result(&doc_meta, &image, &layouts, output_dir, page_num)?;
        }

        // Step 4: Load PDF again for text extraction to avoid borrow conflicts
        let document = self.pdfium.load_pdf_from_file(pdf_path, None)?;
        Self::extract_text_from_layouts(&mut layouts, &document, page_num)?;

        // Step 5: Detect text lines and extract text with OCR for better accuracy
        // Create a clone of the image to avoid borrowing issues
        let image_clone = image.clone();
        drop(document); // Explicitly drop the document to release the borrow
        let text_detections =
            self.extract_text_with_detection_and_ocr(&mut layouts, &image_clone, scale)?;

        // Step 6: Save text detection visualization (only if debug mode is enabled)
        if debug && !text_detections.is_empty() {
            self.save_text_detection_result(&image_clone, &text_detections, output_dir, page_num)?;
        }

        Ok(layouts)
    }

    /// Load a PDF page and convert it to an image
    fn load_pdf_page(
        &self,
        pdf_path: &str,
        page_num: usize,
    ) -> Result<((DynamicImage, f32), PdfDocument), Box<dyn Error>> {
        info!("Loading PDF page {} from {}", page_num, pdf_path);

        let document = self.pdfium.load_pdf_from_file(pdf_path, None)?;

        if page_num >= document.pages().len() as usize {
            return Err(format!(
                "Page {} not found. PDF has {} pages (0-based indexing)",
                page_num,
                document.pages().len()
            )
            .into());
        }

        let page = document.pages().get(page_num as u16)?;
        let image_data = self.render_page_to_image(&page)?;

        Ok((image_data, document))
    }

    /// Render a PDF page to an image with appropriate scaling
    fn render_page_to_image(&self, page: &PdfPage) -> Result<(DynamicImage, f32), Box<dyn Error>> {
        let width = page.width().value;
        let height = page.height().value;
        let scale = f32::min(1024.0 / width, 1024.0 / height);

        info!(
            "Rendering page - Width: {}, Height: {}, Scale: {}",
            width, height, scale
        );

        let render_config = PdfRenderConfig::new()
            .scale_page_by_factor(scale)
            .rotate_if_landscape(PdfPageRenderRotation::Degrees90, true);

        let image = page
            .render_with_config(&render_config)?
            .as_image()
            .into_rgb8();

        Ok((DynamicImage::ImageRgb8(image), scale))
    }

    /// Perform layout analysis on the image
    fn perform_layout_analysis(
        &mut self,
        image: &DynamicImage,
        scale: f32,
        page: usize,
    ) -> Result<(Vec<Layout>, DocMeta), Box<dyn Error>> {
        info!("Performing layout analysis");

        let image_size = Vec2::new(image.width() as f32, image.height() as f32);
        let pdf_size = image_size / scale;

        let doc_meta = DocMeta {
            pdf_size,
            image_size,
            scale,
            page,
        };

        let layouts = self.session.run(image, doc_meta)?;

        info!(
            "Layout analysis completed. Found {} elements",
            layouts.layouts.len()
        );
        Ok((layouts.layouts, doc_meta))
    }

    /// Save analysis results to output directory
    fn save_analysis_result(
        &self,
        doc_meta: &DocMeta,
        image: &DynamicImage,
        layouts: &[Layout],
        output_dir: &Path,
        page_num: usize,
    ) -> Result<(), Box<dyn Error>> {
        let output_path = output_dir.join(format!("analysis-page-{}.jpg", page_num));

        info!("Saving analysis result to: {:?}", output_path);

        self.session.draw(&output_path, doc_meta, layouts, image)?;
        info!("Analysis result saved successfully");

        Ok(())
    }

    /// Save text detection results to output directory
    fn save_text_detection_result(
        &self,
        image: &DynamicImage,
        text_detections: &[TextDetection],
        output_dir: &Path,
        page_num: usize,
    ) -> Result<(), Box<dyn Error>> {
        let output_path = output_dir.join(format!("text-detection-page-{}.jpg", page_num));

        info!("Saving text detection result to: {:?}", output_path);

        self.text_det_session
            .draw(&output_path, text_detections, image)?;
        info!("Text detection result saved successfully");

        Ok(())
    }

    /// Extract text from detected layout elements
    fn extract_text_from_layouts(
        layouts: &mut [Layout],
        document: &PdfDocument,
        page_num: usize,
    ) -> Result<(), Box<dyn Error>> {
        info!("Extracting text from detected layout elements");

        let page = document.pages().get(page_num as u16)?;
        let page_text = page.text()?;
        let page_height = page.height().value;

        for layout in layouts.iter_mut() {
            let pdf_bbox = layout.bbox.to_pdf_rect(page_height);
            layout.text = Some(page_text.inside_rect(pdf_bbox));
        }

        let text_extracted_count = layouts.iter().filter(|l| l.text.is_some()).count();
        info!(
            "Successfully extracted text from {} out of {} layout elements",
            text_extracted_count,
            layouts.len()
        );

        Ok(())
    }

    /// Initialize YOLOv12 session with appropriate execution providers
    fn initialize_yolo_session() -> Result<YoloSession<Yolov12>, Box<dyn Error>> {
        info!("Initializing YOLOv12 session");

        let session_builder = Session::builder()?
            .with_execution_providers(vec![
                #[cfg(all(feature = "coreml", target_os = "macos"))]
                {
                    use ort::execution_providers::CoreMLExecutionProvider;
                    use ort::execution_providers::coreml::*;
                    CoreMLExecutionProvider::default()
                        .with_model_format(CoreMLModelFormat::MLProgram)
                        .build()
                },
                #[cfg(all(feature = "cuda"))]
                {
                    use ort::execution_providers::CUDAExecutionProvider;
                    CUDAExecutionProvider::default().build()
                },
                CPUExecutionProvider::default().build(),
            ])?
            .with_optimization_level(GraphOptimizationLevel::Level1)?;

        let model = Yolov12::new();
        let session = YoloSession::new(session_builder, model)?;

        info!("YOLOv12 session initialized successfully");
        Ok(session)
    }

    /// Initialize PDFium library
    fn initialize_pdfium() -> Result<Pdfium, Box<dyn Error>> {
        info!("Initializing PDFium library");

        let pdfium_lib_path = std::env::var("PDFIUM_DYNAMIC_LIB_PATH")
            .map_err(|_| "PDFIUM_DYNAMIC_LIB_PATH environment variable not found")?;

        let pdfium = Pdfium::new(
            Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path(
                &pdfium_lib_path,
            ))
            .map_err(|e| format!("Failed to load PDFium bindings: {}", e))?,
        );

        info!("PDFium initialized successfully");
        Ok(pdfium)
    }

    /// Initialize text detection session for text line detection
    fn initialize_text_det_session() -> Result<PaddleDetSession<PaddleDet>, Box<dyn Error>> {
        info!("Initializing text detection session");

        let session_builder = Session::builder()?
            .with_execution_providers(vec![
                #[cfg(all(feature = "coreml", target_os = "macos"))]
                {
                    use ort::execution_providers::CoreMLExecutionProvider;
                    use ort::execution_providers::coreml::*;
                    CoreMLExecutionProvider::default()
                        .with_model_format(CoreMLModelFormat::MLProgram)
                        .build()
                },
                #[cfg(all(feature = "cuda"))]
                {
                    use ort::execution_providers::CUDAExecutionProvider;
                    CUDAExecutionProvider::default().build()
                },
                CPUExecutionProvider::default().build(),
            ])?
            .with_optimization_level(GraphOptimizationLevel::Level1)?;

        let model = PaddleDet::new();
        let session = PaddleDetSession::new(session_builder, model)?;

        info!("Text detection session initialized successfully");
        Ok(session)
    }

    /// Initialize OCR session for text recognition
    fn initialize_ocr_session() -> Result<PaddleRecSession<PaddleRec>, Box<dyn Error>> {
        info!("Initializing OCR session");

        let session_builder = Session::builder()?
            .with_execution_providers(vec![
                #[cfg(all(feature = "coreml", target_os = "macos"))]
                {
                    use ort::execution_providers::CoreMLExecutionProvider;
                    use ort::execution_providers::coreml::*;
                    CoreMLExecutionProvider::default()
                        .with_model_format(CoreMLModelFormat::MLProgram)
                        .build()
                },
                #[cfg(all(feature = "cuda"))]
                {
                    use ort::execution_providers::CUDAExecutionProvider;
                    CUDAExecutionProvider::default().build()
                },
                CPUExecutionProvider::default().build(),
            ])?
            .with_optimization_level(GraphOptimizationLevel::Level1)?;

        let model = PaddleRec::new();
        let session = PaddleRecSession::new(session_builder, model)?;

        info!("OCR session initialized successfully");
        Ok(session)
    }

    /// Extract text from layout regions using text detection followed by OCR
    fn extract_text_with_detection_and_ocr(
        &mut self,
        layouts: &mut [Layout],
        image: &DynamicImage,
        scale: f32,
    ) -> Result<Vec<TextDetection>, Box<dyn Error>> {
        info!("Extracting text using text detection and OCR");

        // Step 1: Detect text lines within each layout region
        // Note: layouts bbox coordinates are in PDF space, need to convert to image space
        let text_detections = self
            .text_det_session
            .detect_text_in_layouts(image, layouts, scale)?;

        let mut total_text_lines = 0;
        let mut ocr_extracted_count = 0;
        let mut all_detections = Vec::new();

        // Step 2: Apply OCR to each detected text line
        for (layout_idx, layout_text_lines) in text_detections.iter().enumerate() {
            if layout_idx >= layouts.len() {
                continue;
            }

            total_text_lines += layout_text_lines.len();
            let mut layout_texts = Vec::new();

            for text_detection in layout_text_lines {
                // TextDetection already has a bbox field, no need to convert from points
                let bbox = text_detection.bbox;
                if let Ok(text) = self.ocr_session.recognize_text_region(image, &bbox) {
                    println!("{}", text);
                    let cleaned_text = text.trim();
                    if !cleaned_text.is_empty() {
                        layout_texts.push(cleaned_text.to_string());
                        ocr_extracted_count += 1;
                    }
                }
                // Add all detections to the result
                all_detections.push(text_detection.clone());
            }

            // Combine all text lines for this layout
            if !layout_texts.is_empty() {
                layouts[layout_idx].text = Some(layout_texts.join(" "));
            }
        }

        info!(
            "Detected {} text lines across {} layouts, OCR extracted text from {} lines",
            total_text_lines,
            layouts.len(),
            ocr_extracted_count
        );

        Ok(all_detections)
    }

    /// Extract text from layout regions using OCR (fallback method)
    #[allow(unused)]
    fn extract_text_with_ocr(
        &mut self,
        layouts: &mut [Layout],
        image: &DynamicImage,
        scale: f32,
    ) -> Result<(), Box<dyn Error>> {
        info!("Extracting text using OCR");

        // Filter layouts that are likely to contain text
        let text_layouts: Vec<usize> = layouts.iter().enumerate().map(|(i, _)| i).collect();

        let mut ocr_extracted_count = 0;
        for &idx in &text_layouts {
            if let Ok(text) = self
                .ocr_session
                .recognize_text_region(image, &layouts[idx].bbox.scale(scale))
            {
                println!("{}", text);
                let cleaned_text = text.trim();
                if !cleaned_text.is_empty() {
                    layouts[idx].text = Some(cleaned_text.to_string());
                    ocr_extracted_count += 1;
                }
            }
        }

        info!(
            "OCR extracted text from {} out of {} text regions",
            ocr_extracted_count,
            text_layouts.len()
        );

        Ok(())
    }
}

/// Configuration and validation utilities
struct AnalysisConfig {
    input_path: String,
    page: Option<String>,
    output_dir: PathBuf,
    debug: bool,
}

impl AnalysisConfig {
    fn from_args(args: Args) -> Result<Self, Box<dyn Error>> {
        Self::validate_input(&args.input)?;
        let output_dir = Self::setup_output_directory(&args.output)?;

        Ok(Self {
            input_path: args.input,
            page: args.page,
            output_dir,
            debug: args.debug,
        })
    }

    fn validate_input(input_path: &str) -> Result<(), Box<dyn Error>> {
        if !Path::new(input_path).exists() {
            error!("Input PDF not found: {}", input_path);
            return Err(format!("Input PDF not found: {}", input_path).into());
        }

        if !input_path.to_lowercase().ends_with(".pdf") {
            error!("Input file must be a PDF");
            return Err("Input file must be a PDF".into());
        }

        Ok(())
    }

    fn validate_page_range(&self, total_pages: usize) -> Result<(), Box<dyn Error>> {
        if let Some(page_range_str) = &self.page {
            // Check if it's a single page number
            if let Ok(page) = page_range_str.parse::<usize>() {
                if page >= total_pages {
                    return Err(format!(
                        "Page {} is out of range. PDF has {} pages (0-based indexing)",
                        page, total_pages
                    )
                    .into());
                }
                return Ok(());
            }

            // Try to parse as Rust range syntax
            if page_range_str.contains("..") {
                let parts: Vec<&str> = page_range_str.split("..").collect();

                match parts.len() {
                    2 => {
                        let start_str = parts[0];
                        let end_str = parts[1];

                        if start_str.is_empty() && end_str.is_empty() {
                            return Err("Invalid range format: '..'. Use format like '0..5', '1..', '..5', or '3'".into());
                        } else if start_str.is_empty() {
                            // Format: "..5" - first 5 pages
                            let end = end_str
                                .parse::<usize>()
                                .map_err(|_| format!("Invalid end page number: {}", end_str))?;
                            if end > total_pages {
                                return Err(format!(
                                    "End page {} is out of range. PDF has {} pages (0-based indexing)",
                                    end, total_pages
                                ).into());
                            }
                        } else if end_str.is_empty() {
                            // Format: "1.." - from page 1 onwards
                            let start = start_str
                                .parse::<usize>()
                                .map_err(|_| format!("Invalid start page number: {}", start_str))?;
                            if start >= total_pages {
                                return Err(format!(
                                    "Start page {} is out of range. PDF has {} pages (0-based indexing)",
                                    start, total_pages
                                ).into());
                            }
                        } else {
                            // Format: "0..5" - pages 0 to 4
                            let start = start_str
                                .parse::<usize>()
                                .map_err(|_| format!("Invalid start page number: {}", start_str))?;
                            let end = end_str
                                .parse::<usize>()
                                .map_err(|_| format!("Invalid end page number: {}", end_str))?;

                            if start >= total_pages || end > total_pages {
                                return Err(format!(
                                    "Page range {} is out of range. PDF has {} pages (0-based indexing)",
                                    page_range_str, total_pages
                                ).into());
                            }

                            if start >= end {
                                return Err(format!(
                                    "Invalid page range: start page {} must be less than end page {}",
                                    start, end
                                ).into());
                            }
                        }
                    }
                    _ => {
                        return Err(format!(
                            "Invalid range format: {}. Use format like '0..5', '1..', '..5', or '3'",
                            page_range_str
                        ).into());
                    }
                }
            } else {
                return Err(format!(
                    "Invalid page format: {}. Use format like '0..5', '1..', '..5', or '3'",
                    page_range_str
                )
                .into());
            }
        }
        Ok(())
    }

    fn get_page_range(&self, total_pages: usize) -> Vec<usize> {
        let mut pages = Vec::new();
        if let Some(page_range_str) = &self.page {
            // Check if it's a single page number
            if let Ok(page) = page_range_str.parse::<usize>() {
                pages.push(page);
            } else if page_range_str.contains("..") {
                // Parse as Rust range syntax
                let parts: Vec<&str> = page_range_str.split("..").collect();

                if parts.len() == 2 {
                    let start_str = parts[0];
                    let end_str = parts[1];

                    if start_str.is_empty() && !end_str.is_empty() {
                        // Format: "..5" - first 5 pages
                        if let Ok(end) = end_str.parse::<usize>() {
                            pages.extend(0..end);
                        }
                    } else if !start_str.is_empty() && end_str.is_empty() {
                        // Format: "1.." - from page 1 onwards
                        if let Ok(start) = start_str.parse::<usize>() {
                            pages.extend(start..total_pages);
                        }
                    } else if !start_str.is_empty() && !end_str.is_empty() {
                        // Format: "0..5" - pages 0 to 4
                        if let (Ok(start), Ok(end)) =
                            (start_str.parse::<usize>(), end_str.parse::<usize>())
                        {
                            pages.extend(start..end);
                        }
                    }
                }
            }
        } else {
            // If no page range is specified, process all pages
            pages.extend(0..total_pages);
        }
        pages.sort_unstable();
        pages.dedup();
        pages.into_iter().filter(|&p| p < total_pages).collect()
    }

    fn setup_output_directory(output_dir: &str) -> Result<PathBuf, Box<dyn Error>> {
        let output_path = PathBuf::from(output_dir);

        if !output_path.exists() {
            std::fs::create_dir_all(&output_path)?;
            info!("Created output directory: {}", output_dir);
        }

        Ok(output_path)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();
    info!("Starting PDF layout analysis");
    info!("Input PDF: {}", args.input);
    info!("Page range: {:?}", args.page);
    info!("Output directory: {}", args.output);
    info!("Debug mode: {}", args.debug);

    // Parse and validate configuration
    let config = AnalysisConfig::from_args(args)?;

    // Initialize analyzer (session and pdfium are created once)
    let mut analyzer = Analyzer::new()?;

    // Get total number of pages in the PDF
    let document = analyzer
        .pdfium
        .load_pdf_from_file(&config.input_path, None)?;
    let total_pages = document.pages().len() as usize;
    drop(document); // Release the document borrow

    // Validate page range
    config.validate_page_range(total_pages)?;

    // Get the page range to process
    let page_range = config.get_page_range(total_pages);
    info!(
        "Processing pages: {:?} (total pages in PDF: {})",
        page_range, total_pages
    );

    // Analyze each page in the range
    for &page_num in &page_range {
        info!("Processing page {} of {}", page_num + 1, page_range.len());
        let _layouts = analyzer.analyze_page(
            &config.input_path,
            page_num,
            &config.output_dir,
            config.debug,
        )?;
    }

    info!(
        "Analysis completed successfully for {} pages!",
        page_range.len()
    );
    Ok(())
}
