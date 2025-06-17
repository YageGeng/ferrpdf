use glam::Vec2;
use image::{DynamicImage, GenericImageView, Rgb, imageops::FilterType};
use imageproc::drawing::{draw_filled_rect_mut, draw_hollow_rect_mut};
use imageproc::rect::Rect;
use ndarray::{Array4, ArrayBase, Axis, Dim, OwnedRepr};
use ort::{
    execution_providers::CPUExecutionProvider,
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};
use snafu::{OptionExt, ResultExt};
use std::path::Path;

use crate::{
    analysis::{bbox::Bbox, labels::Label},
    consts::*,
    error::*,
    layout::element::Layout,
};

/// Embedded YOLOv12 model binary for document layout analysis
/// The model is trained on DocLayNet dataset for detecting document elements
pub const YOLOV12: &[u8] = include_bytes!("../../../models/yolov12s-doclaynet.onnx");

/// ONNX Runtime session wrapper for YOLOv12 document layout detection
pub struct OrtSession {
    /// Name of the primary output tensor from the model
    pub output: String,
    /// The underlying ONNX Runtime session
    pub session: Session,
}

impl OrtSession {
    /// Creates a new ONNX Runtime session for document layout detection
    ///
    /// # Returns
    ///
    /// * `Ok(OrtSession)` - Successfully initialized session
    /// * `Err(FerrpdfError)` - Failed to initialize session at any stage
    pub fn new() -> Result<Self, FerrpdfError> {
        // Build ONNX Runtime session with CPU execution provider and optimization
        let session = Session::builder()
            .context(OrtInitSnafu { stage: "Builder" })?
            .with_execution_providers(vec![CPUExecutionProvider::default().build()])
            .context(OrtInitSnafu {
                stage: "Execution Providers",
            })?
            .with_optimization_level(GraphOptimizationLevel::Level1)
            .context(OrtInitSnafu {
                stage: "Optimization Level",
            })?
            .commit_from_memory(YOLOV12)
            .context(OrtInitSnafu { stage: "Commit" })?;

        // Get the name of the first output tensor, fallback to "output0" if not found
        let output = session
            .outputs
            .first()
            .map(|output| &output.name)
            .cloned()
            .unwrap_or("output0".to_string());

        Ok(Self { output, session })
    }

    /// Preprocesses an image for model input by resizing, normalizing, and converting to tensor format.
    ///
    /// This method performs the following operations:
    /// 1. Calculates optimal scaling to fit the image within model input dimensions
    /// 2. Resizes the image while maintaining aspect ratio
    /// 3. Creates a normalized 4D tensor with background padding
    /// 4. Converts RGB pixel values to normalized float values (0.0-1.0)
    ///
    /// # Arguments
    ///
    /// * `img` - Input image to preprocess
    ///
    /// # Returns
    ///
    /// 4D tensor with shape [batch_size, channels, height, width] ready for model inference
    ///
    /// # Example
    ///
    /// ```no_run
    /// use image::DynamicImage;
    /// # use ferrpdf_core::inference::session::OrtSession;
    ///
    /// let img = DynamicImage::new_rgb8(800, 600);
    /// let tensor = OrtSession::preprocess(&img);
    /// assert_eq!(tensor.shape(), &[1, 3, 1024, 1024]);
    /// ```
    pub fn preprocess(image: &DynamicImage) -> Array4<f32> {
        /// Calculates optimal scaling dimensions to fit an image within target dimensions while maintaining aspect ratio.
        fn scale_wh(w0: f32, h0: f32, target_w: f32, target_h: f32) -> (f32, f32, f32) {
            let scale = f32::min(target_w / w0, target_h / h0);
            let w_new = (w0 * scale).round();
            let h_new = (h0 * scale).round();
            (scale, w_new, h_new)
        }

        let (w0, h0) = image.dimensions();
        let (_, w_new, h_new) = scale_wh(
            w0 as f32,
            h0 as f32,
            REQUIRED_WIDTH as f32,
            REQUIRED_HEIGHT as f32,
        );

        // Resize image to calculated dimensions
        let resized_img = image.resize_exact(w_new as u32, h_new as u32, FilterType::Triangle);

        // Create tensor with background fill value
        let mut input_tensor = Array4::ones([
            BATCH_SIZE,
            INPUT_CHANNELS,
            REQUIRED_HEIGHT as usize,
            REQUIRED_WIDTH as usize,
        ]);
        input_tensor.fill(BACKGROUND_FILL_VALUE);

        // Fill tensor with normalized pixel values
        for (x, y, pixel) in resized_img.pixels() {
            let x = x as usize;
            let y = y as usize;
            let [r, g, b, _] = pixel.0;
            input_tensor[[0, 0, y, x]] = r as f32 / 255.0;
            input_tensor[[0, 1, y, x]] = g as f32 / 255.0;
            input_tensor[[0, 2, y, x]] = b as f32 / 255.0;
        }

        input_tensor
    }

    /// Draws bounding boxes and labels on an image to visualize detection results.
    ///
    /// This method creates a visual representation of the detected layout elements
    /// by drawing colored bounding boxes around each detection and labeling them
    /// with their class names and confidence scores.
    ///
    /// # Arguments
    ///
    /// * `img` - The input image to draw on
    /// * `layouts` - Vector of detected layout elements to visualize
    /// * `output_path` - Path where to save the output image
    ///
    /// # Features
    ///
    /// - Color-coded bounding boxes based on element type
    /// - Thick border lines for better visibility
    /// - Text labels with class names and confidence scores
    /// - Automatic text positioning to avoid overlap
    ///
    pub fn draw_bbox<P: AsRef<Path>>(
        img: &DynamicImage,
        layouts: &[Layout],
        output_path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut output_img = img.to_rgb8();

        // Use simple bitmap text rendering approach
        for layout in layouts {
            let x = layout.bbox.min.x as i32;
            let y = layout.bbox.min.y as i32;
            let width = (layout.bbox.max.x - layout.bbox.min.x) as u32;
            let height = (layout.bbox.max.y - layout.bbox.min.y) as u32;

            if width == 0 || height == 0 {
                continue;
            }

            // Draw bounding box with thicker lines
            let color = Rgb(layout.label.color());

            // Draw multiple rectangles to create thicker lines
            for offset in 0..3 {
                let thick_rect = Rect::at(x - offset, y - offset)
                    .of_size(width + (offset * 2) as u32, height + (offset * 2) as u32);
                draw_hollow_rect_mut(&mut output_img, thick_rect, color);
            }

            // Draw label indicator as colored rectangle above bbox
            let label_text = format!("{} {:.2}", layout.label.name(), layout.proba);
            let text_x = x.max(5); // Ensure text is not too close to edge
            let text_y = (y - 25).max(5); // Position text above bbox

            // Calculate label indicator dimensions
            let label_width = (label_text.len() as f32 * 8.0) as u32; // Approximate width
            let label_height = 20u32; // Fixed height

            // Draw label indicator rectangle
            if text_x >= 0
                && text_y >= 0
                && text_x + label_width as i32 <= output_img.width() as i32
                && text_y + label_height as i32 <= output_img.height() as i32
            {
                // Draw colored background rectangle as label indicator
                let label_rect = Rect::at(text_x, text_y).of_size(label_width, label_height);
                draw_filled_rect_mut(&mut output_img, label_rect, color);

                // Draw border around label for better visibility
                draw_hollow_rect_mut(&mut output_img, label_rect, Rgb([255, 255, 255]));
            }
        }

        // Save the output image
        output_img.save(output_path)?;

        Ok(())
    }

    /// Runs inference on the input image tensor
    ///
    /// # Arguments
    ///
    /// * `input` - 4D tensor with shape [batch, channels, height, width]
    ///
    /// # Returns
    ///
    /// * `Ok(Array)` - 3D output tensor with detection results
    /// * `Err(FerrpdfError)` - Inference or tensor processing failed
    pub fn run(
        &mut self,
        input: ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>,
    ) -> Result<ArrayBase<OwnedRepr<f32>, Dim<[usize; 3]>>, FerrpdfError> {
        // Run inference with the input tensor named "images"
        let output = self
            .session
            .run(ort::inputs![
                "images"=> TensorRef::from_array_view(&input).context(TensorSnafu{stage: "input"})?
            ])
            .context(InferenceSnafu {})?;

        // Extract the output tensor and convert to ndarray
        let tensor = output
            .get(&self.output)
            .context(NotFoundOutputSnafu {
                output_name: &self.output,
            })?
            .try_extract_array::<f32>()
            .context(TensorSnafu { stage: "extract" })?;

        // Reshape tensor to expected output dimensions and return owned copy
        let output = tensor
            .to_shape(OUTPUT_SIZE)
            .context(ShapeSnafu { stage: "output" })?
            .to_owned();

        Ok(output)
    }

    /// Extracts bounding boxes and layout elements from model output
    ///
    /// # Arguments
    ///
    /// * `page_no` - Page number for the detected elements
    /// * `original_size` - Original image dimensions for coordinate scaling
    /// * `output` - 3D tensor output from the YOLO model
    ///
    /// # Returns
    ///
    /// Vector of detected layout elements with bounding boxes and labels
    pub fn extra_bbox(
        page_no: usize,
        original_size: Vec2,
        output: ArrayBase<OwnedRepr<f32>, Dim<[usize; 3]>>,
    ) -> Vec<Layout> {
        let mut layouts = Vec::new();

        // Get the first batch slice (assuming batch size = 1)
        let output = output.slice(ndarray::s![0, .., ..]);

        // Iterate through each prediction in the output
        for prediction in output.axis_iter(Axis(1)) {
            // Extract bounding box coordinates (center_x, center_y, width, height)
            let bbox = prediction.slice(ndarray::s![0..CXYWH_OFFSET]);
            // Extract class probabilities for different document elements
            let labels = prediction.slice(ndarray::s![CXYWH_OFFSET..LABEL_PROBA_SIZE]);

            // Find the class with the highest probability
            let (max_prob_idx, &proba) = labels
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();

            // Skip detections below confidence threshold
            if proba < PROBA_THRESHOLD {
                continue;
            }

            // Extract YOLO format coordinates (center x, center y, width, height)
            let cx = bbox[0_usize];
            let cy = bbox[1_usize];
            let w = bbox[2_usize];
            let h = bbox[3_usize];

            // Convert from YOLO center-size format to bounding box and clamp to image bounds
            let bbox = Bbox::from_center_size(Vec2::new(cx, cy), Vec2::new(w, h))
                .clamp(Vec2::new(0.0, 0.0), original_size);

            // Create layout element with detected information
            let layout = Layout {
                bbox,
                label: Label::from(max_prob_idx),
                page_no,
                proba,
                text: None, // Text content will be extracted later
            };

            layouts.push(layout);
        }

        layouts
    }

    /// Applies Non-Maximum Suppression (NMS) to filter overlapping detections.
    ///
    /// This highly optimized algorithm performs in-place swapping during iteration
    /// to avoid any auxiliary data structures or additional memory operations.
    /// It removes duplicate detections by comparing Intersection over Union (IoU)
    /// scores between bounding boxes. For each pair of overlapping boxes (IoU > threshold),
    /// only the detection with higher confidence is kept.
    ///
    /// # Arguments
    ///
    /// * `raw_layouts` - Mutable vector of layout detections to be filtered
    /// * `iou_threshold` - IoU threshold above which detections are considered overlapping
    ///
    /// # Algorithm
    ///
    /// 1. Sort detections by confidence score (highest first)
    /// 2. Use two-pointer approach: keep_index tracks valid detections
    /// 3. For each position, compare with previous valid detections
    /// 4. If no overlap, advance keep_index; otherwise skip current detection
    /// 5. Truncate array to final valid length
    pub fn nms(raw_layouts: &mut Vec<Layout>, iou_threshold: f32) {
        if raw_layouts.is_empty() || raw_layouts.len() == 1 {
            return;
        }

        // Sort layouts by confidence score in descending order
        // Higher confidence detections will be processed first and have priority
        raw_layouts.sort_by(|layout1, layout2| {
            layout2
                .proba
                .partial_cmp(&layout1.proba)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut keep_index = 0; // Index to track the next position for kept detections

        for current_index in 0..raw_layouts.len() {
            let mut should_keep = true;

            // Compare current detection with all previously kept detections
            for kept_index in 0..keep_index {
                let iou = raw_layouts[current_index]
                    .bbox
                    .iou(&raw_layouts[kept_index].bbox);

                // If IoU exceeds threshold, suppress current detection
                // Since array is sorted by confidence, previously kept detections have higher confidence
                if iou > iou_threshold {
                    should_keep = false;
                    break;
                }
            }

            // If current detection should be kept, move it to the keep position
            if should_keep {
                if keep_index != current_index {
                    raw_layouts.swap(keep_index, current_index);
                }
                keep_index += 1;
            }
        }

        // Truncate to remove suppressed elements
        raw_layouts.truncate(keep_index);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::labels::Label;
    use glam::Vec2;

    /// Helper function to create a test layout with given parameters
    fn create_test_layout(
        bbox_center: Vec2,
        bbox_size: Vec2,
        proba: f32,
        label: Label,
        page_no: usize,
    ) -> Layout {
        Layout {
            bbox: Bbox::from_center_size(bbox_center, bbox_size),
            label,
            page_no,
            proba,
            text: None,
        }
    }

    #[test]
    fn test_nms_empty_input() {
        let mut layouts = Vec::new();
        OrtSession::nms(&mut layouts, 0.5);
        assert_eq!(layouts.len(), 0);
    }

    #[test]
    fn test_nms_single_detection() {
        let mut layouts = vec![create_test_layout(
            Vec2::new(50.0, 50.0),
            Vec2::new(100.0, 100.0),
            0.9,
            Label::Text,
            0,
        )];

        OrtSession::nms(&mut layouts, 0.5);
        assert_eq!(layouts.len(), 1);
        assert_eq!(layouts[0].proba, 0.9);
    }

    #[test]
    fn test_nms_no_overlap() {
        // Create two non-overlapping bounding boxes
        let mut layouts = vec![
            create_test_layout(
                Vec2::new(50.0, 50.0), // Center at (50, 50)
                Vec2::new(50.0, 50.0), // Size 50x50, so bbox is (25,25) to (75,75)
                0.9,
                Label::Text,
                0,
            ),
            create_test_layout(
                Vec2::new(150.0, 150.0), // Center at (150, 150)
                Vec2::new(50.0, 50.0),   // Size 50x50, so bbox is (125,125) to (175,175)
                0.8,
                Label::Title,
                0,
            ),
        ];

        OrtSession::nms(&mut layouts, 0.5);

        // Both should be kept since they don't overlap
        assert_eq!(layouts.len(), 2);
        // Should be sorted by confidence (highest first)
        assert_eq!(layouts[0].proba, 0.9);
        assert_eq!(layouts[1].proba, 0.8);
    }

    #[test]
    fn test_nms_complete_overlap() {
        // Create two identical bounding boxes with different confidence
        let mut layouts = vec![
            create_test_layout(
                Vec2::new(50.0, 50.0),
                Vec2::new(100.0, 100.0),
                0.8, // Lower confidence
                Label::Text,
                0,
            ),
            create_test_layout(
                Vec2::new(50.0, 50.0),
                Vec2::new(100.0, 100.0),
                0.9, // Higher confidence
                Label::Text,
                0,
            ),
        ];

        OrtSession::nms(&mut layouts, 0.5);

        // Only one should remain (the one with higher confidence)
        assert_eq!(layouts.len(), 1);
        assert_eq!(layouts[0].proba, 0.9);
    }

    #[test]
    fn test_nms_partial_overlap() {
        // Create two partially overlapping bounding boxes
        let mut layouts = vec![
            create_test_layout(
                Vec2::new(50.0, 50.0), // Center at (50, 50)
                Vec2::new(60.0, 60.0), // Size 60x60, so bbox is (20,20) to (80,80)
                0.9,
                Label::Text,
                0,
            ),
            create_test_layout(
                Vec2::new(70.0, 70.0), // Center at (70, 70)
                Vec2::new(60.0, 60.0), // Size 60x60, so bbox is (40,40) to (100,100)
                0.8,                   // Overlap area is (40,40) to (80,80) = 1600
                Label::Title,          // Union area is 3600 + 3600 - 1600 = 5600
                0,                     // IoU = 1600/5600 ≈ 0.286 < 0.5
            ),
        ];

        OrtSession::nms(&mut layouts, 0.5);

        // Both should be kept since IoU < 0.5
        assert_eq!(layouts.len(), 2);
        assert_eq!(layouts[0].proba, 0.9);
        assert_eq!(layouts[1].proba, 0.8);
    }

    #[test]
    fn test_nms_high_overlap() {
        // Create two highly overlapping bounding boxes
        let mut layouts = vec![
            create_test_layout(
                Vec2::new(50.0, 50.0), // Center at (50, 50)
                Vec2::new(60.0, 60.0), // Size 60x60, so bbox is (20,20) to (80,80)
                0.9,
                Label::Text,
                0,
            ),
            create_test_layout(
                Vec2::new(55.0, 55.0), // Center at (55, 55) - very close
                Vec2::new(60.0, 60.0), // Size 60x60, so bbox is (25,25) to (85,85)
                0.8,                   // Large overlap, IoU > 0.5
                Label::Title,
                0,
            ),
        ];

        OrtSession::nms(&mut layouts, 0.5);

        // Only one should remain (the one with higher confidence)
        assert_eq!(layouts.len(), 1);
        assert_eq!(layouts[0].proba, 0.9);
    }

    #[test]
    fn test_nms_multiple_detections() {
        // Test with more complex scenario: 5 detections with various overlaps
        let mut layouts = vec![
            // High confidence, should be kept
            create_test_layout(
                Vec2::new(50.0, 50.0),
                Vec2::new(60.0, 60.0),
                0.95,
                Label::Text,
                0,
            ),
            // Overlaps with first, should be suppressed
            create_test_layout(
                Vec2::new(52.0, 52.0),
                Vec2::new(60.0, 60.0),
                0.85,
                Label::Text,
                0,
            ),
            // No overlap, should be kept
            create_test_layout(
                Vec2::new(200.0, 200.0),
                Vec2::new(50.0, 50.0),
                0.9,
                Label::Title,
                0,
            ),
            // Overlaps with third, should be suppressed
            create_test_layout(
                Vec2::new(205.0, 205.0),
                Vec2::new(50.0, 50.0),
                0.7,
                Label::Title,
                0,
            ),
            // No overlap, should be kept
            create_test_layout(
                Vec2::new(350.0, 350.0),
                Vec2::new(40.0, 40.0),
                0.8,
                Label::Picture,
                0,
            ),
        ];

        OrtSession::nms(&mut layouts, 0.5);

        // Should keep 3 detections (1st, 3rd, 5th)
        assert_eq!(layouts.len(), 3);

        // Check they are in correct confidence order
        assert_eq!(layouts[0].proba, 0.95); // First detection
        assert_eq!(layouts[1].proba, 0.9); // Third detection
        assert_eq!(layouts[2].proba, 0.8); // Fifth detection

        // Verify the labels are correct
        assert_eq!(layouts[0].label, Label::Text);
        assert_eq!(layouts[1].label, Label::Title);
        assert_eq!(layouts[2].label, Label::Picture);
    }

    #[test]
    fn test_nms_confidence_sorting() {
        // Test that detections are properly sorted by confidence before NMS
        let mut layouts = vec![
            create_test_layout(
                Vec2::new(50.0, 50.0),
                Vec2::new(60.0, 60.0),
                0.3, // Lowest confidence
                Label::Text,
                0,
            ),
            create_test_layout(
                Vec2::new(150.0, 150.0),
                Vec2::new(60.0, 60.0),
                0.9, // Highest confidence
                Label::Title,
                0,
            ),
            create_test_layout(
                Vec2::new(250.0, 250.0),
                Vec2::new(60.0, 60.0),
                0.6, // Medium confidence
                Label::Picture,
                0,
            ),
        ];

        OrtSession::nms(&mut layouts, 0.5);

        // All should be kept (no overlaps), but order should be by confidence
        assert_eq!(layouts.len(), 3);
        assert_eq!(layouts[0].proba, 0.9); // Highest first
        assert_eq!(layouts[1].proba, 0.6); // Medium second
        assert_eq!(layouts[2].proba, 0.3); // Lowest last
    }

    #[test]
    fn test_nms_different_thresholds() {
        // Test with same setup but different IoU thresholds
        let create_overlapping_layouts = || {
            vec![
                create_test_layout(
                    Vec2::new(50.0, 50.0),
                    Vec2::new(60.0, 60.0),
                    0.9,
                    Label::Text,
                    0,
                ),
                create_test_layout(
                    Vec2::new(65.0, 65.0), // Moderate overlap
                    Vec2::new(60.0, 60.0),
                    0.8,
                    Label::Title,
                    0,
                ),
            ]
        };

        // Test with strict threshold (low IoU threshold)
        let mut layouts_strict = create_overlapping_layouts();
        OrtSession::nms(&mut layouts_strict, 0.3);
        assert_eq!(layouts_strict.len(), 1); // Should suppress second detection

        // Test with lenient threshold (high IoU threshold)
        let mut layouts_lenient = create_overlapping_layouts();
        OrtSession::nms(&mut layouts_lenient, 0.7);
        assert_eq!(layouts_lenient.len(), 2); // Should keep both detections
    }

    #[test]
    fn test_nms_preserves_non_bbox_fields() {
        // Test that NMS preserves all fields except the filtered detections
        let mut layouts = vec![
            create_test_layout(
                Vec2::new(50.0, 50.0),
                Vec2::new(60.0, 60.0),
                0.9,
                Label::Text,
                1, // page_no = 1
            ),
            create_test_layout(
                Vec2::new(52.0, 52.0), // Overlapping
                Vec2::new(60.0, 60.0),
                0.8,
                Label::Title,
                2, // page_no = 2
            ),
        ];

        // Add text to first layout
        layouts[0].text = Some("Test text".to_string());

        OrtSession::nms(&mut layouts, 0.5);

        // Should keep only the first (higher confidence)
        assert_eq!(layouts.len(), 1);
        assert_eq!(layouts[0].proba, 0.9);
        assert_eq!(layouts[0].label, Label::Text);
        assert_eq!(layouts[0].page_no, 1);
        assert_eq!(layouts[0].text, Some("Test text".to_string()));
    }

    #[test]
    fn test_preprocess_tensor_shape() {
        // Test that preprocess produces correct tensor shape
        let img = DynamicImage::new_rgb8(800, 600);
        let tensor = OrtSession::preprocess(&img);

        assert_eq!(
            tensor.shape(),
            &[
                BATCH_SIZE,
                INPUT_CHANNELS,
                REQUIRED_HEIGHT as usize,
                REQUIRED_WIDTH as usize
            ]
        );
        assert_eq!(tensor.shape(), &[1, 3, 1024, 1024]);
    }

    #[test]
    fn test_preprocess_normalization() {
        // Create a simple 2x2 white image
        let img = DynamicImage::new_rgb8(2, 2);
        let tensor = OrtSession::preprocess(&img);

        // Check that tensor contains normalized values
        for &value in tensor.iter() {
            assert!(
                (0.0..=1.0).contains(&value),
                "Pixel value {} is not normalized",
                value
            );
        }
    }

    #[test]
    fn test_preprocess_background_fill() {
        // Create a small image that will leave background areas after resizing
        let img = DynamicImage::new_rgb8(100, 50); // Aspect ratio that won't fill perfectly
        let tensor = OrtSession::preprocess(&img);

        // First verify tensor has been initialized with background value
        let total_pixels =
            BATCH_SIZE * INPUT_CHANNELS * REQUIRED_HEIGHT as usize * REQUIRED_WIDTH as usize;
        assert_eq!(tensor.len(), total_pixels);

        // The image will be scaled and positioned, leaving some background pixels
        // Check that all values are either background fill or normalized pixel values (0-1)
        for &value in tensor.iter() {
            assert!(
                (value - BACKGROUND_FILL_VALUE).abs() < 0.001 || (0.0..=1.0).contains(&value),
                "Pixel value {} is neither background fill nor normalized pixel",
                value
            );
        }
    }
}
