use std::error::Error;
use std::path::{Path, PathBuf};

use clap::Parser;
use ferrpdf_core::analysis::bbox::Bbox;
use ferrpdf_core::layout::element::Layout;
use glam::Vec2;
use image::{DynamicImage, GenericImageView};
use pdfium_render::prelude::*;
use tracing::{error, info, warn};

use ferrpdf_core::consts::*;
use ferrpdf_core::inference::session::OrtSession;

#[derive(Parser)]
#[command(name = "analyze")]
#[command(about = "PDF layout analysis tool")]
struct Args {
    #[arg(help = "Input PDF file path")]
    input: String,

    #[arg(
        short,
        long,
        default_value = "0",
        help = "Page number to analyze (0-based)"
    )]
    page: usize,

    #[arg(short, long, default_value = "images", help = "Output directory")]
    output: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    info!("Starting PDF layout analysis");
    info!("Input PDF: {}", args.input);
    info!("Page: {}", args.page);
    info!("Output directory: {}", args.output);

    validate_input(&args.input)?;
    let output_dir = setup_output_directory(&args.output)?;
    let pdfium = initialize_pdfium()?;

    let ((image, scale), document) = convert_pdf_to_image(&pdfium, &args.input, args.page)?;
    let mut layouts = analyze_layout(&image, args.page)?;

    // save output result
    save_analysis_result(&image, &layouts, &output_dir, args.page)?;

    // extra text
    extract_text_from_layouts(&mut layouts, &document, args.page, scale)?;

    print_summary(&args.input, args.page, &layouts);

    println!("{}", serde_json::to_string(&layouts).unwrap());

    info!("Analysis completed successfully!");
    Ok(())
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

fn setup_output_directory(output_dir: &str) -> Result<PathBuf, Box<dyn Error>> {
    let output_path = PathBuf::from(output_dir);

    if !output_path.exists() {
        std::fs::create_dir_all(&output_path)?;
        info!("Created output directory: {}", output_dir);
    }

    Ok(output_path)
}

fn initialize_pdfium() -> Result<Pdfium, Box<dyn Error>> {
    let pdfium_lib_path = std::env::var("PDFIUM_DYNAMIC_LIB_PATH")
        .map_err(|_| "PDFIUM_DYNAMIC_LIB_PATH environment variable not found")?;

    let pdfium = Pdfium::new(
        Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path(
            &pdfium_lib_path,
        ))
        .map_err(|e| format!("Failed to load PDFium bindings: {}", e))?,
    );

    Ok(pdfium)
}

fn convert_pdf_to_image<'a>(
    pdfium: &'a Pdfium,
    pdf_path: &str,
    page_num: usize,
) -> Result<((DynamicImage, f32), PdfDocument<'a>), Box<dyn Error>> {
    info!("Converting PDF page {} to image", page_num);

    let document = pdfium.load_pdf_from_file(pdf_path, None)?;

    if page_num >= document.pages().len() as usize {
        return Err(format!(
            "Page {} not found. PDF has {} pages (0-based indexing)",
            page_num,
            document.pages().len()
        )
        .into());
    }

    let page = document.pages().get(page_num as u16)?;
    let image = render_page_to_image(&page)?;

    Ok((image, document))
}

fn render_page_to_image(page: &PdfPage) -> Result<(DynamicImage, f32), Box<dyn Error>> {
    let width = page.width().value;
    let height = page.height().value;
    let scale = f32::min(1024.0 / width, 1024.0 / height);
    println!("PDF Width: {}, Height: {}, Scale: {}", width, height, scale);

    let render_config = PdfRenderConfig::new()
        .scale_page_by_factor(scale)
        .rotate_if_landscape(PdfPageRenderRotation::Degrees90, true);

    let image = page
        .render_with_config(&render_config)?
        .as_image()
        .into_rgb8();

    Ok((DynamicImage::ImageRgb8(image), scale))
}

fn analyze_layout(
    image: &DynamicImage,
    page: usize,
) -> Result<Vec<ferrpdf_core::layout::element::Layout>, Box<dyn Error>> {
    info!("Starting layout analysis");

    let (original_width, original_height) = image.dimensions();
    let original_size = Vec2::new(original_width as f32, original_height as f32);

    info!("Image size: {}x{}", original_width, original_height);

    let mut session = initialize_onnx_session()?;
    let input_tensor = preprocess_image(image);
    let output = run_inference(&mut session, input_tensor)?;
    let mut layouts = extract_bounding_boxes(output, original_size, page);
    apply_post_processing(&mut layouts);

    info!(
        "Layout analysis completed. Found {} elements",
        layouts.len()
    );
    Ok(layouts)
}

fn initialize_onnx_session() -> Result<OrtSession, Box<dyn Error>> {
    info!("Initializing ONNX Runtime session");
    let session = OrtSession::new()?;
    info!("ONNX Runtime session initialized successfully");
    Ok(session)
}

fn preprocess_image(image: &DynamicImage) -> ndarray::Array4<f32> {
    info!("Preprocessing image for model input");
    let tensor = OrtSession::preprocess(image);
    info!(
        "Image preprocessing completed. Tensor shape: {:?}",
        tensor.shape()
    );
    tensor
}

type InferenceOutput = ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 3]>>;

fn run_inference(
    session: &mut OrtSession,
    input_tensor: ndarray::Array4<f32>,
) -> Result<InferenceOutput, Box<dyn Error>> {
    info!("Running model inference");
    let start_time = std::time::Instant::now();
    let output = session.run(input_tensor)?;
    let inference_time = start_time.elapsed();
    info!("Inference completed in {}ms", inference_time.as_millis());
    Ok(output)
}

fn extract_bounding_boxes(
    output: InferenceOutput,
    original_size: Vec2,
    page: usize,
) -> Vec<ferrpdf_core::layout::element::Layout> {
    info!("Extracting layout elements from model output");
    let layouts = OrtSession::extra_bbox(page, original_size, output);
    info!("Found {} raw layout detections", layouts.len());
    layouts
}

fn apply_post_processing(layouts: &mut Vec<ferrpdf_core::layout::element::Layout>) {
    info!(
        "Applying Non-Maximum Suppression with IoU threshold: {}",
        NMS_IOU_THRESHOLD
    );
    let layouts_before_nms = layouts.len();
    OrtSession::nms(layouts, NMS_IOU_THRESHOLD);
    let layouts_after_nms = layouts.len();
    info!(
        "NMS completed. Kept {} out of {} detections",
        layouts_after_nms, layouts_before_nms
    );

    info!("Sorting layout elements by reading order");
    OrtSession::sort_by_reading_order(layouts);
    info!("Layout elements sorted by reading order");

    if layouts.is_empty() {
        warn!("No layout elements detected after filtering");
    }
}

fn save_analysis_result(
    image: &DynamicImage,
    layouts: &[Layout],
    output_dir: &Path,
    page_num: usize,
) -> Result<(), Box<dyn Error>> {
    let output_path = output_dir.join(format!("analysis-page-{}.jpg", page_num));

    info!("Drawing bounding boxes and saving analysis result");
    OrtSession::draw_bbox(image, layouts, &output_path)?;
    info!("Analysis result saved to: {:?}", output_path);

    Ok(())
}

fn extract_text_from_layouts(
    layouts: &mut [Layout],
    document: &PdfDocument,
    page_num: usize,
    scale: f32,
) -> Result<(), Box<dyn Error>> {
    info!("Extracting text from detected layout elements");

    let page = document.pages().get(page_num as u16)?;
    let page_text = page.text()?;
    let page_height = page.height().value;
    let page_width = page.width().value;

    for layout in layouts.iter_mut() {
        // Convert bbox coordinates from image coordinates to PDF coordinates
        let pdf_bbox =
            convert_image_bbox_to_pdf_rect(&mut layout.bbox, (page_height, page_width), scale);
        // Extract text within the bounding box
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

fn convert_image_bbox_to_pdf_rect(bbox: &mut Bbox, page_size: (f32, f32), scale: f32) -> PdfRect {
    let (pdf_height, _) = page_size;
    let rescale = 1. / scale;

    bbox.scale(rescale);

    let pdf_min_x = bbox.min.x;
    let pdf_max_x = bbox.max.x;
    let pdf_min_y = bbox.min.y;
    let pdf_max_y = bbox.max.y;

    let pdf_bottom = pdf_height - pdf_max_y;
    let pdf_top = pdf_height - pdf_min_y;

    PdfRect::new(
        PdfPoints::new(pdf_bottom),
        PdfPoints::new(pdf_min_x),
        PdfPoints::new(pdf_top),
        PdfPoints::new(pdf_max_x),
    )
}

#[allow(dead_code)]
fn print_summary(
    input_path: &str,
    page_num: usize,
    layouts: &[ferrpdf_core::layout::element::Layout],
) {
    println!("\n=== PDF Layout Analysis Summary ===");
    println!("Input PDF: {}", input_path);
    println!("Analyzed page: {}", page_num);
    println!("Total detections: {}", layouts.len());
    println!("Confidence threshold: {}", PROBA_THRESHOLD);
    println!("NMS IoU threshold: {}", NMS_IOU_THRESHOLD);

    if !layouts.is_empty() {
        println!("\nDetected elements:");
        for (i, layout) in layouts.iter().enumerate() {
            println!(
                "  {}. {} ({:.1}%)",
                i + 1,
                layout.label.name(),
                layout.proba * 100.0
            );
        }
    } else {
        println!("\nNo layout elements detected.");
    }
}
