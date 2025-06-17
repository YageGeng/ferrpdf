use std::error::Error;
use std::path::Path;

use glam::Vec2;
use image::GenericImageView;
use tracing::{error, info, warn};

use ferrpdf_core::consts::*;
use ferrpdf_core::inference::session::OrtSession;

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize tracing for logging
    tracing_subscriber::fmt::init();

    info!("Starting layout inference on test-page-0.jpg");

    // Load the input image
    let input_path = "images/test-page-0.jpg";
    let output_path = "images/output.jpg";

    if !Path::new(input_path).exists() {
        error!("Input image not found: {}", input_path);
        return Err(format!("Input image not found: {}", input_path).into());
    }

    info!("Loading image from: {}", input_path);
    let image = image::open(input_path)?;
    let (original_width, original_height) = image.dimensions();
    let original_size = Vec2::new(original_width as f32, original_height as f32);

    info!(
        "Image loaded successfully. Original size: {}x{}",
        original_width, original_height
    );

    // Create ONNX Runtime session
    info!("Initializing ONNX Runtime session...");
    let mut session = OrtSession::new()?;
    info!("ONNX Runtime session initialized successfully");

    // Preprocess the image
    info!("Preprocessing image for model input...");
    let input_tensor = OrtSession::preprocess(&image);
    info!(
        "Image preprocessing completed. Tensor shape: {:?}",
        input_tensor.shape()
    );

    // Run inference
    info!("Running model inference...");
    let start_time = std::time::Instant::now();
    let output = session.run(input_tensor)?;
    let inference_time = start_time.elapsed();
    info!("Inference completed in {}ms", inference_time.as_millis());

    // Extract bounding boxes from model output
    info!("Extracting layout elements from model output...");
    let mut layouts = OrtSession::extra_bbox(1, original_size, output);
    info!("Found {} raw layout detections", layouts.len());

    // Apply Non-Maximum Suppression to remove duplicate detections
    info!(
        "Applying Non-Maximum Suppression with IoU threshold: {}",
        NMS_IOU_THRESHOLD
    );
    let layouts_before_nms = layouts.len();
    OrtSession::nms(&mut layouts, NMS_IOU_THRESHOLD);
    let layouts_after_nms = layouts.len();
    info!(
        "NMS completed. Kept {} out of {} detections",
        layouts_after_nms, layouts_before_nms
    );

    // Sort by reading order (top-to-bottom, left-to-right, multi-column aware)
    info!("Sorting layout elements by reading order...");
    OrtSession::sort_by_reading_order(&mut layouts);
    info!("Layout elements sorted by reading order");

    // Log detected layout elements
    if layouts.is_empty() {
        warn!("No layout elements detected after filtering");
    }

    // Draw bounding boxes on the image and save
    info!("Drawing bounding boxes and saving output image...");
    OrtSession::draw_bbox(&image, &layouts, output_path)?;
    info!("Output image saved to: {}", output_path);

    // Print summary
    println!("\n=== Layout Inference Summary ===");
    println!("Input image: {}", input_path);
    println!("Output image: {}", output_path);
    println!(
        "Original image size: {}x{}",
        original_width, original_height
    );
    println!("Inference time: {}ms", inference_time.as_millis());
    println!("Raw detections: {}", layouts_before_nms);
    println!("Final detections: {}", layouts_after_nms);
    println!("Confidence threshold: {}", PROBA_THRESHOLD);
    println!("NMS IoU threshold: {}", NMS_IOU_THRESHOLD);

    if !layouts.is_empty() {
        println!("\nDetected elements:");
        for layout in &layouts {
            println!("  - {} ({:.1}%)", layout.label.name(), layout.proba * 100.0);
        }
    }

    info!("Layout inference completed successfully!");
    Ok(())
}
