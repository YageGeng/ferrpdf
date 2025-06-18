use ab_glyph::{FontRef, PxScale};
use glam::Vec2;
use image::{DynamicImage, GenericImageView, Rgb, imageops::FilterType};
use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut};
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
pub const FONT: &[u8] = include_bytes!("../../../fonts/DejaVuSans.ttf");

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
    /// with their class names and confidence scores using TrueType font rendering.
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
    /// - High-quality text labels using DejaVu Sans font
    /// - Automatic text positioning to avoid overlap
    ///
    pub fn draw_bbox<P: AsRef<Path>>(
        img: &DynamicImage,
        layouts: &[Layout],
        output_path: P,
    ) -> Result<(), FerrpdfError> {
        let mut output_img = img.to_rgb8();

        let font = FontRef::try_from_slice(FONT).context(FontSnafu {})?;

        // Define font scale (size)
        let font_scale = PxScale::from(16.0);

        for (idx, layout) in layouts.iter().enumerate() {
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

            // Prepare label text
            let label_text = format!("{} {:.2} {}", layout.label.name(), layout.proba, idx);

            // Calculate text position
            let text_x = x.max(5); // Ensure text is not too close to edge
            let text_y = y; // Position text above bbox with more space

            let text_color = Rgb([255, 0, 0]); // Red text color
            // Draw text with white color for good contrast
            if text_x >= 0 && text_y >= 0 {
                // Draw the text using TrueType font
                draw_text_mut(
                    &mut output_img,
                    text_color,
                    text_x,
                    text_y,
                    font_scale,
                    &font,
                    &label_text,
                );
            }
        }

        // Save the output image
        output_img
            .save(output_path.as_ref())
            .context(ImageWriteSnafu {
                path: output_path.as_ref().to_string_lossy(),
            })?;

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
        let mut label_idx = 0;
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
                bbox_id: label_idx,
                proba,
                text: None, // Text content will be extracted later
            };
            label_idx += 1;

            layouts.push(layout);
        }

        layouts
    }

    /// Applies Non-Maximum Suppression (NMS) with bounding box merging for overlapping detections.
    ///
    /// This algorithm merges overlapping detections using both IoU and overlap ratio comparisons.
    /// IoU is used for general overlap detection, while overlap ratio (intersection/min_area)
    /// provides more lenient detection for cases where small bboxes are contained within larger ones.
    /// For each pair of overlapping boxes, the detection with higher confidence is kept and its
    /// bounding box is expanded to encompass both original detections (union operation).
    ///
    /// Additionally, when two bounding boxes have the same label and one completely
    /// contains the other, the larger bbox is kept and expanded if necessary.
    ///
    /// # Arguments
    ///
    /// * `raw_layouts` - Mutable vector of layout detections to be processed
    /// * `iou_threshold` - IoU threshold above which detections are considered overlapping
    ///
    /// # Algorithm
    ///
    /// 1. Sort detections by confidence score (highest first)
    /// 2. For each detection, compare with all previously kept detections using original bboxes
    /// 3. Check for containment (same label), IoU overlap, or overlap ratio
    /// 4. Collect merge operations without modifying bboxes during comparison
    /// 5. Apply all merge operations to expand kept detections
    /// 6. Keep only the detections that weren't merged into others
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

        // Store original bboxes to avoid using modified bboxes during comparison
        let original_bboxes: Vec<Bbox> = raw_layouts.iter().map(|l| l.bbox).collect();

        let mut keep_flags = vec![true; raw_layouts.len()];
        let mut merge_operations = Vec::new(); // (merge_from_idx, merge_to_idx)

        #[allow(clippy::needless_range_loop)]
        for current_index in 0..raw_layouts.len() {
            if !keep_flags[current_index] {
                continue; // Already marked for removal
            }

            // Compare current detection with all previously kept detections
            for kept_index in 0..current_index {
                // Dont skip already suppressed detections, maybe other will merge into it
                // if !keep_flags[kept_index] {
                //     continue; // Skip already suppressed detections
                // }

                // Use original bboxes for comparison to avoid repeated expansion
                let current_bbox = original_bboxes[current_index];
                let kept_bbox = original_bboxes[kept_index];

                // Check if one bbox contains the other (regardless of label)
                // If kept bbox contains current bbox, merge current into kept
                if kept_bbox.contains(&current_bbox) {
                    merge_operations.push((current_index, kept_index));
                    keep_flags[current_index] = false;
                    break;
                }

                // If current bbox contains kept bbox, merge current into kept (kept has higher confidence)
                if current_bbox.contains(&kept_bbox) {
                    merge_operations.push((current_index, kept_index));
                    keep_flags[current_index] = false;
                    break;
                }

                let iou = current_bbox.iou(&kept_bbox);
                let overlap_ratio = current_bbox.overlap_ratio(&kept_bbox);

                // If IoU exceeds threshold OR overlap ratio is high (indicating small bbox in large bbox),
                // merge current into kept detection. Use more lenient overlap_ratio threshold.
                // Since array is sorted by confidence, kept detections have higher confidence
                if iou > iou_threshold || overlap_ratio > iou_threshold {
                    merge_operations.push((current_index, kept_index));
                    keep_flags[current_index] = false;
                    break;
                }
            }
        }

        // Apply all merge operations to expand kept detections
        for (merge_from_idx, merge_to_idx) in merge_operations {
            let merge_from_bbox = original_bboxes[merge_from_idx];
            raw_layouts[merge_to_idx].bbox = raw_layouts[merge_to_idx].bbox.union(&merge_from_bbox);
        }

        // Keep only the detections that weren't merged into others
        let mut keep_index = 0;
        #[allow(clippy::needless_range_loop)]
        for current_index in 0..raw_layouts.len() {
            if keep_flags[current_index] {
                if keep_index != current_index {
                    raw_layouts.swap(keep_index, current_index);
                }
                keep_index += 1;
            }
        }

        // Truncate to remove suppressed elements
        raw_layouts.truncate(keep_index);
    }

    /// Sorts layout elements by reading order: top-to-bottom, left-to-right.
    /// For multi-column documents, processes left column first, then right column.
    ///
    /// This method should be called after NMS if you want the results sorted in reading order.
    /// It provides better organization for text extraction and document processing workflows.
    ///
    /// # Arguments
    ///
    /// * `layouts` - Mutable vector of layout elements to be sorted
    ///
    /// # Algorithm
    ///
    /// 1. Detect if document has multiple columns by analyzing X-coordinate distribution
    /// 2. If multi-column: separate layouts into columns, sort each column, then merge
    /// 3. If single-column: sort directly by Y-coordinate (top-to-bottom), then X-coordinate (left-to-right)
    /// 4. Use tolerance for Y-coordinates to handle elements on the same line
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrpdf_core::inference::session::OrtSession;
    /// # let mut layouts = Vec::new();
    /// // After running NMS
    /// OrtSession::nms(&mut layouts, 0.5);
    ///
    /// // Optionally sort by reading order
    /// OrtSession::sort_by_reading_order(&mut layouts);
    /// ```
    pub fn sort_by_reading_order(layouts: &mut Vec<Layout>) {
        if layouts.is_empty() {
            return;
        }

        // Check if we have a multi-column layout
        if Self::is_multi_column_layout(layouts) {
            Self::sort_multi_column_layout(layouts);
        } else {
            Self::sort_single_column_layout(layouts);
        }
    }

    /// Detects if the layout has multiple columns by analyzing X-coordinate distribution
    fn is_multi_column_layout(layouts: &[Layout]) -> bool {
        if layouts.len() < 4 {
            return false; // Too few elements to determine multi-column
        }

        // Get all center X coordinates
        let mut x_centers: Vec<f32> = layouts.iter().map(|l| l.bbox.center().x).collect();
        x_centers.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Find the overall X range
        let min_x = x_centers[0];
        let max_x = x_centers[x_centers.len() - 1];
        let total_width = max_x - min_x;

        if total_width < 100.0 {
            return false; // Too narrow to be multi-column
        }

        // Split into potential left and right halves
        let mid_x = min_x + total_width / 2.0;
        let left_count = x_centers.iter().filter(|&&x| x < mid_x).count();
        let right_count = x_centers.len() - left_count;

        // Consider it multi-column if both sides have significant elements
        // and the distribution isn't too uneven
        let min_elements_per_column = layouts.len() / 4; // At least 25% in smaller column
        left_count >= min_elements_per_column && right_count >= min_elements_per_column
    }

    /// Sorts multi-column layout: left column first (top-to-bottom), then right column (top-to-bottom)
    fn sort_multi_column_layout(layouts: &mut Vec<Layout>) {
        // Find the boundary between columns
        let bbox_widths: Vec<(f32, f32)> = layouts
            .iter()
            .map(|l| (l.bbox.min.x, l.bbox.max.x))
            .collect();
        let (min_x, max_x) = bbox_widths.iter().fold(
            (f32::INFINITY, f32::NEG_INFINITY),
            |(min_x, max_x), &(o_min_x, o_max_x)| (min_x.min(o_min_x), max_x.max(o_max_x)),
        );
        let mid_x = (max_x - min_x) / 2.0;

        let _one_of_six = mid_x / 3.0;
        let mid_min_x = mid_x - _one_of_six;
        let mid_max_x = mid_x + _one_of_six;

        // Separate into left and right columns
        let mut left_column = Vec::new();
        let mut right_column = Vec::new();

        for layout in layouts.drain(..) {
            match layout.label {
                Label::Title | Label::PageHeader => {
                    left_column.push(layout);
                }
                Label::PageFooter => {
                    right_column.push(layout);
                }
                _ if layout.bbox.center().x > mid_min_x && layout.bbox.center().x < mid_max_x => {
                    left_column.push(layout);
                }
                _ if layout.bbox.center().x < mid_x => {
                    left_column.push(layout);
                }
                _ => {
                    right_column.push(layout);
                }
            }
        }

        // Sort each column by reading order (top-to-bottom, left-to-right within same line)
        Self::sort_single_column_layout(&mut left_column);
        Self::sort_single_column_layout(&mut right_column);

        // Merge back: left column first, then right column
        layouts.extend(left_column);
        layouts.extend(right_column);
    }

    /// Sorts single column layout by reading order: top-to-bottom, left-to-right for same-line elements
    fn sort_single_column_layout(layouts: &mut [Layout]) {
        const Y_TOLERANCE: f32 = 10.0; // Tolerance for considering elements on the same line

        layouts.sort_by(|a, b| {
            // TODO: order with Label
            if a.label == Label::PageHeader || a.label == Label::Title {
                return std::cmp::Ordering::Less;
            } else if a.label == Label::PageFooter {
                return std::cmp::Ordering::Greater;
            }

            let a_center = a.bbox.center();
            let b_center = b.bbox.center();

            // Primary sort: by Y coordinate (top to bottom)
            let y_diff = a_center.y - b_center.y;

            if y_diff.abs() <= Y_TOLERANCE {
                // Elements are on roughly the same line, sort by X coordinate (left to right)
                a_center
                    .x
                    .partial_cmp(&b_center.x)
                    .unwrap_or(std::cmp::Ordering::Equal)
            } else {
                // Different lines, sort by Y coordinate
                y_diff
                    .partial_cmp(&0.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
        });
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
            bbox_id: 0,
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
        // Bbox should remain the same since they were identical
        assert_eq!(layouts.len(), 1);
        assert_eq!(layouts[0].proba, 0.9);
        assert_eq!(layouts[0].bbox.area(), 10000.0); // 100x100
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
        // Bbox should be expanded to union of both: (20,20) to (85,85)
        assert_eq!(layouts.len(), 1);
        assert_eq!(layouts[0].proba, 0.9);
        assert_eq!(layouts[0].bbox.min, Vec2::new(20.0, 20.0));
        assert_eq!(layouts[0].bbox.max, Vec2::new(85.0, 85.0));
        assert_eq!(layouts[0].bbox.area(), 4225.0); // 65x65
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
            // Overlaps with first, should be merged
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
            // Overlaps with third, should be merged
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

        // Should keep 3 detections (1st merged, 3rd merged, 5th)
        assert_eq!(layouts.len(), 3);

        // Check they are in correct confidence order
        assert_eq!(layouts[0].proba, 0.95); // First detection (merged)
        assert_eq!(layouts[1].proba, 0.9); // Third detection (merged)
        assert_eq!(layouts[2].proba, 0.8); // Fifth detection

        // Verify the labels are correct
        assert_eq!(layouts[0].label, Label::Text);
        assert_eq!(layouts[1].label, Label::Title);
        assert_eq!(layouts[2].label, Label::Picture);

        // Check that first bbox was expanded to include the overlapping one
        // First bbox: center (50,50), size (60,60) -> (20,20) to (80,80)
        // Second bbox: center (52,52), size (60,60) -> (22,22) to (82,82)
        // Union should be (20,20) to (82,82)
        assert_eq!(layouts[0].bbox.min, Vec2::new(20.0, 20.0));
        assert_eq!(layouts[0].bbox.max, Vec2::new(82.0, 82.0));
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

    #[test]
    fn test_nms_containment_same_label() {
        // Test that smaller bbox is merged when contained by larger bbox with same label
        let mut layouts = vec![
            // Larger bbox with higher confidence - should be kept
            create_test_layout(
                Vec2::new(50.0, 50.0),
                Vec2::new(100.0, 100.0), // Large bbox: (0,0) to (100,100)
                0.9,
                Label::Text,
                0,
            ),
            // Smaller bbox completely inside larger one - should be merged
            create_test_layout(
                Vec2::new(50.0, 50.0),
                Vec2::new(60.0, 60.0), // Small bbox: (20,20) to (80,80)
                0.8,
                Label::Text, // Same label
                0,
            ),
        ];

        OrtSession::nms(&mut layouts, 0.5);

        // Only the larger bbox should remain, unchanged since smaller was contained
        assert_eq!(layouts.len(), 1);
        assert_eq!(layouts[0].proba, 0.9);
        assert_eq!(layouts[0].bbox.area(), 10000.0); // 100x100, unchanged
        assert_eq!(layouts[0].bbox.min, Vec2::new(0.0, 0.0));
        assert_eq!(layouts[0].bbox.max, Vec2::new(100.0, 100.0));
    }

    #[test]
    fn test_nms_containment_different_labels() {
        // Test that containment logic only applies to same-label bboxes
        let mut layouts = vec![
            // Larger bbox
            create_test_layout(
                Vec2::new(50.0, 50.0),
                Vec2::new(100.0, 100.0),
                0.9,
                Label::Text,
                0,
            ),
            // Smaller bbox inside larger one but different label - should be kept
            create_test_layout(
                Vec2::new(50.0, 50.0),
                Vec2::new(60.0, 60.0),
                0.8,
                Label::Title, // Different label
                0,
            ),
        ];

        OrtSession::nms(&mut layouts, 0.5);

        // With overlap_ratio, the smaller bbox will be merged into the larger one
        // even with different labels since overlap_ratio = 1.0 > 0.7
        assert_eq!(layouts.len(), 1);
        assert_eq!(layouts[0].proba, 0.9);
        // The bbox should be expanded to include the smaller bbox (already contained)
        assert_eq!(layouts[0].bbox.area(), 10000.0); // 100x100, unchanged since smaller was contained
    }

    #[test]
    fn test_nms_containment_lower_confidence_container() {
        // Test containment when container has lower confidence than contained
        let mut layouts = vec![
            // Smaller bbox with higher confidence - should be kept and expanded
            create_test_layout(
                Vec2::new(50.0, 50.0),
                Vec2::new(60.0, 60.0),
                0.9,
                Label::Text,
                0,
            ),
            // Larger bbox with lower confidence that contains the first - should be merged
            create_test_layout(
                Vec2::new(50.0, 50.0),
                Vec2::new(100.0, 100.0),
                0.8,
                Label::Text, // Same label
                0,
            ),
        ];

        OrtSession::nms(&mut layouts, 0.5);

        // Only the higher confidence bbox should remain, but expanded to union
        assert_eq!(layouts.len(), 1);
        assert_eq!(layouts[0].proba, 0.9);
        // Union of (20,20)-(80,80) and (0,0)-(100,100) = (0,0)-(100,100)
        assert_eq!(layouts[0].bbox.area(), 10000.0); // 100x100
        assert_eq!(layouts[0].bbox.min, Vec2::new(0.0, 0.0));
        assert_eq!(layouts[0].bbox.max, Vec2::new(100.0, 100.0));
    }

    #[test]
    fn test_nms_partial_overlap_no_containment() {
        // Test that partial overlap without containment uses IoU threshold
        let mut layouts = vec![
            create_test_layout(
                Vec2::new(50.0, 50.0), // Center at (50, 50)
                Vec2::new(60.0, 60.0), // Size 60x60, bbox (20,20) to (80,80)
                0.9,
                Label::Text,
                0,
            ),
            create_test_layout(
                Vec2::new(70.0, 70.0), // Center at (70, 70)
                Vec2::new(60.0, 60.0), // Size 60x60, bbox (40,40) to (100,100)
                0.8,                   // Overlap but no containment
                Label::Text,           // Same label
                0,
            ),
        ];

        // With IoU threshold 0.5, these should both be kept (IoU < 0.5)
        OrtSession::nms(&mut layouts, 0.5);
        assert_eq!(layouts.len(), 2);

        // With lower IoU threshold 0.2, second should be merged
        let mut layouts2 = vec![
            create_test_layout(
                Vec2::new(50.0, 50.0),
                Vec2::new(60.0, 60.0),
                0.9,
                Label::Text,
                0,
            ),
            create_test_layout(
                Vec2::new(70.0, 70.0),
                Vec2::new(60.0, 60.0),
                0.8,
                Label::Text,
                0,
            ),
        ];
        OrtSession::nms(&mut layouts2, 0.2);
        assert_eq!(layouts2.len(), 1);
        assert_eq!(layouts2[0].proba, 0.9);
        // Union of (20,20)-(80,80) and (40,40)-(100,100) = (20,20)-(100,100)
        assert_eq!(layouts2[0].bbox.min, Vec2::new(20.0, 20.0));
        assert_eq!(layouts2[0].bbox.max, Vec2::new(100.0, 100.0));
    }

    #[test]
    fn test_nms_mixed_containment_and_iou() {
        // Test complex scenario with both containment and IoU merging
        let mut layouts = vec![
            // Large container with highest confidence
            create_test_layout(
                Vec2::new(100.0, 100.0),
                Vec2::new(200.0, 200.0), // (0,0) to (200,200)
                0.95,
                Label::Text,
                0,
            ),
            // Small contained box - should be merged by containment
            create_test_layout(
                Vec2::new(100.0, 100.0),
                Vec2::new(100.0, 100.0), // (50,50) to (150,150)
                0.85,
                Label::Text, // Same label
                0,
            ),
            // Separate high-confidence box
            create_test_layout(
                Vec2::new(300.0, 300.0),
                Vec2::new(80.0, 80.0),
                0.9,
                Label::Title,
                0,
            ),
            // Box overlapping with third but different label - should be kept
            create_test_layout(
                Vec2::new(320.0, 320.0),
                Vec2::new(80.0, 80.0),
                0.8,
                Label::Picture, // Different label
                0,
            ),
            // Box with high IoU overlap with first but different label - should be kept
            create_test_layout(
                Vec2::new(110.0, 110.0),
                Vec2::new(180.0, 180.0),
                0.7,
                Label::Table, // Different label
                0,
            ),
        ];

        OrtSession::nms(&mut layouts, 0.5);

        // Should keep: 1st (merged), 3rd (separate), 4th (different label)
        // 5th should be merged with 1st due to high IoU overlap
        assert_eq!(layouts.len(), 2);

        // Check that the merged layout is not present
        for layout in &layouts {
            // The contained layout with confidence 0.85 should be gone
            assert_ne!(layout.proba, 0.85);
        }

        // Check that all kept label combinations are preserved
        let labels: Vec<_> = layouts.iter().map(|l| l.label.clone()).collect();
        assert!(labels.contains(&Label::Text));
        assert!(labels.contains(&Label::Title));
        // Label::Table should be merged with Text due to high IoU overlap
        assert!(!labels.contains(&Label::Table));
    }

    #[test]
    fn test_sort_single_column_reading_order() {
        // Test single column layout sorting: top-to-bottom, left-to-right
        let mut layouts = vec![
            // Bottom element
            create_test_layout(
                Vec2::new(50.0, 300.0),
                Vec2::new(100.0, 50.0),
                0.9,
                Label::Text,
                0,
            ),
            // Top element
            create_test_layout(
                Vec2::new(50.0, 100.0),
                Vec2::new(100.0, 50.0),
                0.8,
                Label::Title,
                0,
            ),
            // Middle element
            create_test_layout(
                Vec2::new(50.0, 200.0),
                Vec2::new(100.0, 50.0),
                0.7,
                Label::Text,
                0,
            ),
        ];

        OrtSession::sort_single_column_layout(&mut layouts);

        // Should be sorted top-to-bottom by Y coordinate
        assert_eq!(layouts[0].bbox.center().y, 100.0); // Top
        assert_eq!(layouts[1].bbox.center().y, 200.0); // Middle
        assert_eq!(layouts[2].bbox.center().y, 300.0); // Bottom
    }

    #[test]
    fn test_sort_same_line_elements() {
        // Test elements on the same line should be sorted left-to-right
        let mut layouts = vec![
            // Right element on same line
            create_test_layout(
                Vec2::new(200.0, 100.0),
                Vec2::new(80.0, 40.0),
                0.9,
                Label::Text,
                0,
            ),
            // Left element on same line
            create_test_layout(
                Vec2::new(50.0, 105.0), // Slightly different Y but within tolerance
                Vec2::new(80.0, 40.0),
                0.8,
                Label::Text,
                0,
            ),
            // Middle element on same line
            create_test_layout(
                Vec2::new(125.0, 102.0),
                Vec2::new(80.0, 40.0),
                0.7,
                Label::Text,
                0,
            ),
        ];

        OrtSession::sort_single_column_layout(&mut layouts);

        // Should be sorted left-to-right by X coordinate (same line)
        assert_eq!(layouts[0].bbox.center().x, 50.0); // Left
        assert_eq!(layouts[1].bbox.center().x, 125.0); // Middle
        assert_eq!(layouts[2].bbox.center().x, 200.0); // Right
    }

    #[test]
    fn test_is_multi_column_layout_detection() {
        // Test single column (should return false)
        let single_column = vec![
            create_test_layout(
                Vec2::new(100.0, 50.0),
                Vec2::new(80.0, 40.0),
                0.9,
                Label::Text,
                0,
            ),
            create_test_layout(
                Vec2::new(100.0, 150.0),
                Vec2::new(80.0, 40.0),
                0.8,
                Label::Text,
                0,
            ),
            create_test_layout(
                Vec2::new(100.0, 250.0),
                Vec2::new(80.0, 40.0),
                0.7,
                Label::Text,
                0,
            ),
            create_test_layout(
                Vec2::new(100.0, 350.0),
                Vec2::new(80.0, 40.0),
                0.6,
                Label::Text,
                0,
            ),
        ];
        assert!(!OrtSession::is_multi_column_layout(&single_column));

        // Test multi-column (should return true)
        let multi_column = vec![
            // Left column
            create_test_layout(
                Vec2::new(100.0, 50.0),
                Vec2::new(80.0, 40.0),
                0.9,
                Label::Text,
                0,
            ),
            create_test_layout(
                Vec2::new(100.0, 150.0),
                Vec2::new(80.0, 40.0),
                0.8,
                Label::Text,
                0,
            ),
            create_test_layout(
                Vec2::new(100.0, 250.0),
                Vec2::new(80.0, 40.0),
                0.7,
                Label::Text,
                0,
            ),
            // Right column
            create_test_layout(
                Vec2::new(400.0, 50.0),
                Vec2::new(80.0, 40.0),
                0.6,
                Label::Text,
                0,
            ),
            create_test_layout(
                Vec2::new(400.0, 150.0),
                Vec2::new(80.0, 40.0),
                0.5,
                Label::Text,
                0,
            ),
            create_test_layout(
                Vec2::new(400.0, 250.0),
                Vec2::new(80.0, 40.0),
                0.4,
                Label::Text,
                0,
            ),
        ];
        assert!(OrtSession::is_multi_column_layout(&multi_column));

        // Test too few elements (should return false)
        let too_few = vec![
            create_test_layout(
                Vec2::new(100.0, 50.0),
                Vec2::new(80.0, 40.0),
                0.9,
                Label::Text,
                0,
            ),
            create_test_layout(
                Vec2::new(400.0, 50.0),
                Vec2::new(80.0, 40.0),
                0.8,
                Label::Text,
                0,
            ),
        ];
        assert!(!OrtSession::is_multi_column_layout(&too_few));
    }

    #[test]
    fn test_sort_multi_column_layout() {
        // Test multi-column layout: left column first (top-to-bottom), then right column (top-to-bottom)
        let mut layouts = vec![
            // Right column, bottom
            create_test_layout(
                Vec2::new(400.0, 300.0),
                Vec2::new(80.0, 40.0),
                0.9,
                Label::Text,
                0,
            ),
            // Left column, bottom
            create_test_layout(
                Vec2::new(100.0, 300.0),
                Vec2::new(80.0, 40.0),
                0.8,
                Label::Text,
                0,
            ),
            // Right column, top
            create_test_layout(
                Vec2::new(400.0, 100.0),
                Vec2::new(80.0, 40.0),
                0.7,
                Label::Title,
                0,
            ),
            // Left column, top
            create_test_layout(
                Vec2::new(100.0, 100.0),
                Vec2::new(80.0, 40.0),
                0.6,
                Label::Title,
                0,
            ),
            // Left column, middle
            create_test_layout(
                Vec2::new(100.0, 200.0),
                Vec2::new(80.0, 40.0),
                0.5,
                Label::Text,
                0,
            ),
            // Right column, middle
            create_test_layout(
                Vec2::new(400.0, 200.0),
                Vec2::new(80.0, 40.0),
                0.4,
                Label::Text,
                0,
            ),
        ];

        OrtSession::sort_multi_column_layout(&mut layouts);

        // Expected order: Left column (top to bottom), then Right column (top to bottom)
        let expected_positions = [
            (100.0, 100.0), // Left top
            (400.0, 100.0), // Right top
            (100.0, 200.0), // Left middle
            (100.0, 300.0), // Left bottom
            (400.0, 200.0), // Right middle
            (400.0, 300.0), // Right bottom
        ];

        for (i, layout) in layouts.iter().enumerate() {
            let center = layout.bbox.center();
            assert_eq!((center.x, center.y), expected_positions[i]);
        }
    }

    #[test]
    fn test_reading_order_with_separate_sorting() {
        // Test NMS followed by separate reading order sorting
        let mut layouts = vec![
            // Create elements in random order
            create_test_layout(
                Vec2::new(100.0, 300.0),
                Vec2::new(80.0, 40.0),
                0.9,
                Label::Text,
                0,
            ), // Left bottom
            create_test_layout(
                Vec2::new(400.0, 100.0),
                Vec2::new(80.0, 40.0),
                0.8,
                Label::Title,
                0,
            ), // Right top
            create_test_layout(
                Vec2::new(100.0, 100.0),
                Vec2::new(80.0, 40.0),
                0.7,
                Label::Title,
                0,
            ), // Left top
            create_test_layout(
                Vec2::new(400.0, 300.0),
                Vec2::new(80.0, 40.0),
                0.6,
                Label::Text,
                0,
            ), // Right bottom
            create_test_layout(
                Vec2::new(100.0, 200.0),
                Vec2::new(80.0, 40.0),
                0.5,
                Label::Text,
                0,
            ), // Left middle
            create_test_layout(
                Vec2::new(400.0, 200.0),
                Vec2::new(80.0, 40.0),
                0.4,
                Label::Text,
                0,
            ), // Right middle
        ];

        // Apply NMS first
        OrtSession::nms(&mut layouts, 0.5);

        // Verify we have all elements (no overlap so no suppression)
        assert_eq!(layouts.len(), 6);

        // Now apply reading order sorting separately
        OrtSession::sort_by_reading_order(&mut layouts);

        // After sorting, should be in reading order: left column (top-to-bottom), then right column (top-to-bottom)
        let positions: Vec<_> = layouts.iter().map(|l| l.bbox.center()).collect();

        // Check reading order for multi-column layout
        assert_eq!(positions[0], Vec2::new(100.0, 100.0)); // Left top
        assert_eq!(positions[1], Vec2::new(400.0, 100.0)); // Right top
        assert_eq!(positions[2], Vec2::new(100.0, 200.0)); // Left middle
        assert_eq!(positions[3], Vec2::new(100.0, 300.0)); // Left bottom
        assert_eq!(positions[4], Vec2::new(400.0, 200.0)); // Right middle
        assert_eq!(positions[5], Vec2::new(400.0, 300.0)); // Right bottom
    }

    #[test]
    fn test_reading_order_single_column_with_nms() {
        // Test single column reading order after NMS
        let mut layouts = vec![
            // Create elements in random order (single column)
            create_test_layout(
                Vec2::new(100.0, 300.0),
                Vec2::new(80.0, 40.0),
                0.9,
                Label::Text,
                0,
            ), // Bottom
            create_test_layout(
                Vec2::new(100.0, 100.0),
                Vec2::new(80.0, 40.0),
                0.8,
                Label::Title,
                0,
            ), // Top
            create_test_layout(
                Vec2::new(100.0, 200.0),
                Vec2::new(80.0, 40.0),
                0.7,
                Label::Text,
                0,
            ), // Middle
            // Same line elements (should be sorted left-to-right)
            create_test_layout(
                Vec2::new(200.0, 400.0),
                Vec2::new(60.0, 30.0),
                0.6,
                Label::Text,
                0,
            ), // Same line, right
            create_test_layout(
                Vec2::new(50.0, 405.0),
                Vec2::new(60.0, 30.0),
                0.5,
                Label::Text,
                0,
            ), // Same line, left
        ];

        // Apply NMS first
        OrtSession::nms(&mut layouts, 0.5);
        assert_eq!(layouts.len(), 5);

        // Now apply reading order sorting separately
        OrtSession::sort_by_reading_order(&mut layouts);

        // Should be in reading order: top-to-bottom, left-to-right for same line
        let positions: Vec<_> = layouts.iter().map(|l| l.bbox.center()).collect();

        assert_eq!(positions[0], Vec2::new(100.0, 100.0)); // Top
        assert_eq!(positions[1], Vec2::new(100.0, 200.0)); // Middle
        assert_eq!(positions[2], Vec2::new(100.0, 300.0)); // Bottom
        assert_eq!(positions[3], Vec2::new(50.0, 405.0)); // Same line, left
        assert_eq!(positions[4], Vec2::new(200.0, 400.0)); // Same line, right
    }

    #[test]
    fn test_reading_order_edge_cases() {
        // Test edge cases for reading order

        // Empty vector
        let mut empty_layouts = Vec::new();
        OrtSession::sort_by_reading_order(&mut empty_layouts);
        assert_eq!(empty_layouts.len(), 0);

        // Single element
        let mut single_layout = vec![create_test_layout(
            Vec2::new(100.0, 100.0),
            Vec2::new(80.0, 40.0),
            0.9,
            Label::Text,
            0,
        )];
        OrtSession::sort_by_reading_order(&mut single_layout);
        assert_eq!(single_layout.len(), 1);

        // Two elements
        let mut two_layouts = vec![
            create_test_layout(
                Vec2::new(100.0, 200.0),
                Vec2::new(80.0, 40.0),
                0.9,
                Label::Text,
                0,
            ), // Bottom
            create_test_layout(
                Vec2::new(100.0, 100.0),
                Vec2::new(80.0, 40.0),
                0.8,
                Label::Title,
                0,
            ), // Top
        ];
        OrtSession::sort_by_reading_order(&mut two_layouts);
        assert_eq!(two_layouts[0].bbox.center().y, 100.0); // Top first
        assert_eq!(two_layouts[1].bbox.center().y, 200.0); // Bottom second
    }

    #[test]
    fn test_nms_bbox_merging_behavior() {
        // Test that NMS properly merges overlapping bboxes instead of just removing them
        let mut layouts = vec![
            // First detection: center (100, 100), size (80, 60) -> bbox (60,70) to (140,130)
            create_test_layout(
                Vec2::new(100.0, 100.0),
                Vec2::new(80.0, 60.0),
                0.9, // Higher confidence
                Label::Text,
                0,
            ),
            // Second detection: center (120, 110), size (70, 50) -> bbox (85,85) to (155,135)
            // This overlaps with first detection
            create_test_layout(
                Vec2::new(120.0, 110.0),
                Vec2::new(70.0, 50.0),
                0.7, // Lower confidence
                Label::Text,
                0,
            ),
        ];

        // Calculate expected union bbox manually:
        // First bbox: (60,70) to (140,130)
        // Second bbox: (85,85) to (155,135)
        // Union should be: (60,70) to (155,135)
        // IoU = ~0.42, so we need threshold < 0.42 for merging to occur
        OrtSession::nms(&mut layouts, 0.4);

        // Should keep only one detection (the one with higher confidence)
        assert_eq!(layouts.len(), 1);
        assert_eq!(layouts[0].proba, 0.9); // Higher confidence kept

        // The bbox should be the union of both original bboxes
        // First bbox: (60,70) to (140,130)
        // Second bbox: (85,85) to (155,135)
        // Union should be: (60,70) to (155,135)
        assert_eq!(layouts[0].bbox.min, Vec2::new(60.0, 70.0));
        assert_eq!(layouts[0].bbox.max, Vec2::new(155.0, 135.0));

        // Verify the bbox is larger than either original bbox
        let original_area1 = 80.0 * 60.0; // 4800
        let original_area2 = 70.0 * 50.0; // 3500
        assert!(layouts[0].bbox.area() > original_area1);
        assert!(layouts[0].bbox.area() > original_area2);
    }

    #[test]
    fn test_nms_multiple_merges() {
        // Test merging behavior with multiple overlapping detections
        let mut layouts = vec![
            // Central high-confidence detection
            create_test_layout(
                Vec2::new(100.0, 100.0),
                Vec2::new(60.0, 60.0), // (70,70) to (130,130)
                0.95,
                Label::Text,
                0,
            ),
            // Overlapping detection 1
            create_test_layout(
                Vec2::new(80.0, 90.0),
                Vec2::new(50.0, 50.0), // (55,65) to (105,115)
                0.8,
                Label::Text,
                0,
            ),
            // Overlapping detection 2
            create_test_layout(
                Vec2::new(120.0, 110.0),
                Vec2::new(40.0, 40.0), // (100,90) to (140,130)
                0.7,
                Label::Text,
                0,
            ),
            // Non-overlapping detection (should be kept separate)
            create_test_layout(
                Vec2::new(200.0, 200.0),
                Vec2::new(30.0, 30.0), // (185,185) to (215,215)
                0.6,
                Label::Title,
                0,
            ),
        ];

        // Use lower threshold to ensure overlapping text detections get merged
        OrtSession::nms(&mut layouts, 0.2);

        // Should have 2 detections: merged Text and separate Title
        assert_eq!(layouts.len(), 2);

        // First should be the merged text detection with highest confidence
        assert_eq!(layouts[0].proba, 0.95);
        assert_eq!(layouts[0].label, Label::Text);

        // Second should be the separate title detection
        assert_eq!(layouts[1].proba, 0.6);
        assert_eq!(layouts[1].label, Label::Title);

        // The merged bbox should encompass all three original text detections
        // Union of (70,70)-(130,130), (55,65)-(105,115), (100,90)-(140,130)
        // Should be (55,65) to (140,130)
        assert_eq!(layouts[0].bbox.min, Vec2::new(55.0, 65.0));
        assert_eq!(layouts[0].bbox.max, Vec2::new(140.0, 130.0));

        // The title bbox should remain unchanged
        assert_eq!(layouts[1].bbox.min, Vec2::new(185.0, 185.0));
        assert_eq!(layouts[1].bbox.max, Vec2::new(215.0, 215.0));
    }

    #[test]
    fn test_nms_no_repeated_merging() {
        // Test that bboxes are not repeatedly expanded during the merging process
        // This ensures we use original bboxes for IoU calculation, not already-merged ones
        let mut layouts = vec![
            // High confidence detection at center
            create_test_layout(
                Vec2::new(100.0, 100.0),
                Vec2::new(60.0, 60.0), // (70,70) to (130,130)
                0.9,
                Label::Text,
                0,
            ),
            // Medium confidence detection overlapping with first
            create_test_layout(
                Vec2::new(110.0, 110.0),
                Vec2::new(50.0, 50.0), // (85,85) to (135,135)
                0.7,
                Label::Text,
                0,
            ),
            // Lower confidence detection that overlaps with both original positions
            // but should only be compared against original bboxes, not expanded ones
            create_test_layout(
                Vec2::new(120.0, 120.0),
                Vec2::new(40.0, 40.0), // (100,100) to (140,140)
                0.5,
                Label::Text,
                0,
            ),
        ];

        OrtSession::nms(&mut layouts, 0.3);

        // With overlap_ratio threshold of 0.7, more merging may occur
        // All three layouts may be merged into one since they have overlaps
        assert!(layouts.len() <= 2);
        assert_eq!(layouts[0].proba, 0.9); // Highest confidence preserved

        // The first layout should be expanded to encompass merged detections
        assert!(layouts[0].bbox.area() >= 3600.0); // At least as large as original 60x60
    }

    #[test]
    fn test_nms_overlap_ratio_merging() {
        // Test that overlap_ratio enables merging of small bboxes inside large ones
        // even when IoU is low due to size differences
        let mut layouts = vec![
            // Large bbox with high confidence
            create_test_layout(
                Vec2::new(500.0, 500.0),
                Vec2::new(800.0, 800.0), // (100,100) to (900,900), area = 640,000
                0.9,
                Label::Text,
                0,
            ),
            // Small bbox completely inside the large one
            create_test_layout(
                Vec2::new(300.0, 300.0),
                Vec2::new(80.0, 80.0), // (260,260) to (340,340), area = 6,400
                0.7,
                Label::Title, // Different label to test cross-label merging
                0,
            ),
            // Another small bbox inside the large one
            create_test_layout(
                Vec2::new(700.0, 700.0),
                Vec2::new(60.0, 60.0), // (670,670) to (730,730), area = 3,600
                0.5,
                Label::Picture,
                0,
            ),
        ];

        // Calculate IoU and overlap_ratio to demonstrate the difference
        let large_bbox = layouts[0].bbox;
        let small_bbox1 = layouts[1].bbox;
        let small_bbox2 = layouts[2].bbox;

        let iou1 = large_bbox.iou(&small_bbox1);
        let iou2 = large_bbox.iou(&small_bbox2);
        let overlap_ratio1 = large_bbox.overlap_ratio(&small_bbox1);
        let overlap_ratio2 = large_bbox.overlap_ratio(&small_bbox2);

        // Overlap ratios should be 1.0 (complete containment) while IoUs are very low
        assert!((overlap_ratio1 - 1.0).abs() < 0.001);
        assert!((overlap_ratio2 - 1.0).abs() < 0.001);
        assert!(iou1 < 0.02); // Very low IoU due to size difference
        assert!(iou2 < 0.02);

        // Use standard IoU threshold that wouldn't merge these boxes
        OrtSession::nms(&mut layouts, 0.5);

        // Should have only 1 layout remaining due to overlap_ratio merging (threshold 0.7)
        assert_eq!(layouts.len(), 1);
        assert_eq!(layouts[0].proba, 0.9); // Highest confidence preserved

        // The final bbox should be expanded to encompass all original boxes
        // Should be at least as large as the original large bbox
        assert!(layouts[0].bbox.area() >= 640000.0);

        // The bbox should be the union of all three original bboxes
        assert_eq!(layouts[0].bbox.min, Vec2::new(100.0, 100.0));
        assert_eq!(layouts[0].bbox.max, Vec2::new(900.0, 900.0));
    }
}
