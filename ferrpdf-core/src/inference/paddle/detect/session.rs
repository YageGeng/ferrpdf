use derive_builder::Builder;
use image::Rgb;
use image::{DynamicImage, GenericImageView, imageops::FilterType};
use imageproc::{drawing::draw_hollow_rect_mut, rect::Rect};
use ndarray::prelude::*;
use ort::session::RunOptions;
use ort::session::{Session, builder::SessionBuilder};
use ort::value::TensorRef;
use snafu::{OptionExt, ResultExt};
use std::path::Path;

use crate::consts::*;
use crate::{
    analysis::bbox::Bbox,
    error::*,
    inference::model::{Model, OnnxSession},
    layout::element::Layout,
};

use super::model::PaddleDet;

pub struct PaddleDetSession<M: Model> {
    session: Session,
    model: M,
}

/// Text detection result with bounding box and confidence score
#[derive(Debug, Clone)]
pub struct TextDetection {
    pub bbox: Bbox,
    pub proba: f32,
}

/// Extra parameters for text detection
#[derive(Debug, Clone, Builder)]
#[builder(setter(into))]
#[derive(Default)]
pub struct DetExtra {
    /// Original image dimensions
    pub original_shape: (u32, u32),
    /// Resized image dimensions for model input
    pub resized_shape: (u32, u32),
    /// Layout bounding box in image coordinates (optional, for region-based detection)
    pub layout_bbox: Option<Bbox>,
}

impl PaddleDetSession<PaddleDet> {
    pub fn new(session: SessionBuilder, model: PaddleDet) -> Result<Self, FerrpdfError> {
        let span = tracing::info_span!("init-paddle-detect-session");
        let _guard = span.enter();
        let session = session
            .commit_from_memory(model.load())
            .context(OrtInitSnafu { stage: "commit" })?;

        Ok(Self { session, model })
    }

    /// Detect text lines in an entire image
    pub fn detect_text_lines(
        &mut self,
        image: &DynamicImage,
    ) -> Result<Vec<TextDetection>, FerrpdfError> {
        let (orig_w, orig_h) = image.dimensions();
        let config = self.model.config();
        let scale = f32::min(
            config.required_width as f32 / orig_w as f32,
            config.required_height as f32 / orig_h as f32,
        );
        let new_w = (orig_w as f32 * scale) as u32;
        let new_h = (orig_h as f32 * scale) as u32;

        let extra = DetExtra {
            original_shape: (orig_w, orig_h),
            resized_shape: (new_w, new_h),
            layout_bbox: None,
        };

        self.run(image, extra)
    }

    /// Detect text lines within a specific layout region
    pub fn detect_text_in_layout(
        &mut self,
        image: &DynamicImage,
        layout: &Layout,
        pdf_to_image_scale: f32,
    ) -> Result<Vec<TextDetection>, FerrpdfError> {
        // Convert layout bbox from PDF coordinates to image coordinates
        let image_bbox = layout.bbox.scale(pdf_to_image_scale);

        // Crop the image to the layout region
        let cropped_image = self.crop_image_region(image, &image_bbox)?;

        // Get original and resized shapes for the cropped region
        let (orig_w, orig_h) = cropped_image.dimensions();
        let config = self.model.config();
        let scale = f32::min(
            config.required_width as f32 / orig_w as f32,
            config.required_height as f32 / orig_h as f32,
        );
        let new_w = (orig_w as f32 * scale) as u32;
        let new_h = (orig_h as f32 * scale) as u32;

        let extra = DetExtra {
            original_shape: (orig_w, orig_h),
            resized_shape: (new_w, new_h),
            layout_bbox: Some(image_bbox),
        };

        // Run detection on cropped image
        let detections = self.run(&cropped_image, extra)?;

        // Transform detections back to original image coordinates
        let transformed_detections = detections
            .into_iter()
            .map(|det| TextDetection {
                bbox: det.bbox.translate(image_bbox.min),
                proba: det.proba,
            })
            .collect();

        Ok(transformed_detections)
    }

    /// Detect text lines within multiple layout regions
    pub fn detect_text_in_layouts(
        &mut self,
        image: &DynamicImage,
        layouts: &[Layout],
        pdf_to_image_scale: f32,
    ) -> Result<Vec<Vec<TextDetection>>, FerrpdfError> {
        let mut results = Vec::new();

        for layout in layouts {
            let detections = self.detect_text_in_layout(image, layout, pdf_to_image_scale)?;
            results.push(detections);
        }

        Ok(results)
    }

    pub async fn detect_text_in_layout_async(
        &mut self,
        image: &DynamicImage,
        layout: &Layout,
        pdf_to_image_scale: f32,
        run_options: &RunOptions,
    ) -> Result<Vec<TextDetection>, FerrpdfError> {
        // Convert layout bbox from PDF coordinates to image coordinates
        let image_bbox = layout.bbox.scale(pdf_to_image_scale);

        // Crop the image to the layout region
        let cropped_image = self.crop_image_region(image, &image_bbox)?;

        // Get original and resized shapes for the cropped region
        let (orig_w, orig_h) = cropped_image.dimensions();
        let config = self.model.config();
        let scale = f32::min(
            config.required_width as f32 / orig_w as f32,
            config.required_height as f32 / orig_h as f32,
        );
        let new_w = (orig_w as f32 * scale) as u32;
        let new_h = (orig_h as f32 * scale) as u32;

        let extra = DetExtra {
            original_shape: (orig_w, orig_h),
            resized_shape: (new_w, new_h),
            layout_bbox: Some(image_bbox),
        };

        // Run detection on cropped image
        let detections = self.run_async(&cropped_image, extra, run_options).await?;

        // Transform detections back to original image coordinates
        let transformed_detections = detections
            .into_iter()
            .map(|det| TextDetection {
                bbox: det.bbox.translate(image_bbox.min),
                proba: det.proba,
            })
            .collect();

        Ok(transformed_detections)
    }

    /// Crop image to a specific bounding box region
    fn crop_image_region(
        &self,
        image: &DynamicImage,
        bbox: &Bbox,
    ) -> Result<DynamicImage, FerrpdfError> {
        let image_width = image.width() as f32;
        let image_height = image.height() as f32;

        // Clamp bbox to image boundaries
        let clamped_bbox = bbox.clamp(
            glam::Vec2::new(0.0, 0.0),
            glam::Vec2::new(image_width, image_height),
        );

        // Calculate crop coordinates
        let x = clamped_bbox.min.x.max(0.0) as u32;
        let y = clamped_bbox.min.y.max(0.0) as u32;
        let width = (clamped_bbox.max.x - clamped_bbox.min.x).max(1.0) as u32;
        let height = (clamped_bbox.max.y - clamped_bbox.min.y).max(1.0) as u32;

        // Ensure crop doesn't exceed image boundaries
        let width = width.min(image.width().saturating_sub(x));
        let height = height.min(image.height().saturating_sub(y));

        if width == 0 || height == 0 {
            // Return a minimal 1x1 image if bbox is invalid
            return Ok(DynamicImage::new_rgb8(1, 1));
        }

        let cropped = image.crop_imm(x, y, width, height);
        Ok(cropped)
    }

    /// Post-process detection results using DB (Differentiable Binarization) algorithm
    fn db_postprocess(&self, pred: &Array3<f32>, extra: &DetExtra) -> Vec<TextDetection> {
        let config = self.model.config();

        // 1. Extract and prepare probability map
        let (pred_img, cbuf_img) = self.extract_probability_maps(pred);

        // 2. Apply thresholding and morphological operations
        let dilated_img = self.apply_threshold_and_dilation(&cbuf_img, config.det_db_box_thresh);

        // 3. Find contours in the processed image
        let contours = self.find_text_contours(&dilated_img);

        // 4. Process each contour to extract text detections
        let mut detections = Vec::new();
        for contour in contours {
            if let Some(detection) =
                self.process_contour_to_detection(&contour, &pred_img, extra, config)
            {
                detections.push(detection);
            }
        }

        // 5. Merge overlapping text boxes
        self.merge_bboxes(detections)
    }

    /// Extract probability maps from model prediction
    fn extract_probability_maps(
        &self,
        pred: &Array3<f32>,
    ) -> (
        image::ImageBuffer<image::Luma<f32>, Vec<f32>>,
        image::GrayImage,
    ) {
        use image::{GrayImage, Luma};

        // Extract probability map from first channel
        let prob_map = pred.slice(ndarray::s![0, .., ..]);
        let (h, w) = prob_map.dim();

        // Convert to vectors for image processing
        let pred_data: Vec<f32> = prob_map.iter().copied().collect();
        let cbuf_data: Vec<u8> = pred_data.iter().map(|&x| (x * 255.0) as u8).collect();

        // Create image buffers
        let pred_img: image::ImageBuffer<Luma<f32>, Vec<f32>> =
            image::ImageBuffer::from_vec(w as u32, h as u32, pred_data).unwrap();
        let cbuf_img = GrayImage::from_vec(w as u32, h as u32, cbuf_data).unwrap();

        (pred_img, cbuf_img)
    }

    /// Apply thresholding and dilation operations
    fn apply_threshold_and_dilation(
        &self,
        cbuf_img: &image::GrayImage,
        box_thresh: f32,
    ) -> image::GrayImage {
        use imageproc::contrast::threshold;
        use imageproc::distance_transform::Norm;
        use imageproc::morphology::dilate;

        // Apply binary thresholding
        let threshold_img = threshold(
            cbuf_img,
            (box_thresh * 255.0) as u8,
            imageproc::contrast::ThresholdType::Binary,
        );

        // Apply dilation to connect nearby text regions
        dilate(&threshold_img, Norm::LInf, 1)
    }

    /// Find text contours in the processed image
    fn find_text_contours(
        &self,
        dilated_img: &image::GrayImage,
    ) -> Vec<imageproc::contours::Contour<i32>> {
        use imageproc::contours::find_contours;

        find_contours(dilated_img)
    }

    /// Process a single contour to extract text detection
    fn process_contour_to_detection(
        &self,
        contour: &imageproc::contours::Contour<i32>,
        pred_img: &image::ImageBuffer<image::Luma<f32>, Vec<f32>>,
        extra: &DetExtra,
        config: &<PaddleDet as Model>::Config,
    ) -> Option<TextDetection> {
        // Skip contours with too few points
        if contour.points.len() <= 2 {
            return None;
        }

        // Get minimum area rectangle and check size constraints
        let mut max_side = 0.0;
        let min_box = self.get_mini_box(&contour.points, &mut max_side);
        if max_side < config.max_side_thresh {
            return None;
        }

        // Calculate confidence score
        let score = self.get_score(contour, pred_img);
        if score < config.det_db_thresh {
            return None;
        }

        // Apply unclip operation
        let clip_box = self.unclip(&min_box);
        if clip_box.is_empty() {
            return None;
        }

        // Check size constraints after unclip
        let mut max_side_clip = 0.0;
        let clip_min_box = self.get_mini_box(&clip_box, &mut max_side_clip);
        if max_side_clip < config.max_side_thresh + 2.0 {
            return None;
        }

        // Convert to bounding box and scale to original coordinates
        let bbox = self.points_to_bbox(&clip_min_box, extra);

        Some(TextDetection { bbox, proba: score })
    }

    // Convert contour points to bounding box
    fn points_to_bbox(&self, points: &[imageproc::point::Point<f32>], extra: &DetExtra) -> Bbox {
        if points.is_empty() {
            return Bbox::new(glam::Vec2::ZERO, glam::Vec2::ZERO);
        }

        // Calculate bounding box from points
        let (min_x, min_y, max_x, max_y) = self.calculate_points_bounds(points);

        // Scale coordinates to original image space
        let (min, max) = self.scale_coordinates_to_original(min_x, min_y, max_x, max_y, extra);

        // Apply text padding
        self.apply_text_padding(min, max, extra)
    }

    /// Calculate bounding box from contour points
    fn calculate_points_bounds(
        &self,
        points: &[imageproc::point::Point<f32>],
    ) -> (f32, f32, f32, f32) {
        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;

        for point in points {
            min_x = min_x.min(point.x);
            min_y = min_y.min(point.y);
            max_x = max_x.max(point.x);
            max_y = max_y.max(point.y);
        }

        (min_x, min_y, max_x, max_y)
    }

    /// Scale coordinates from resized to original image space
    fn scale_coordinates_to_original(
        &self,
        min_x: f32,
        min_y: f32,
        max_x: f32,
        max_y: f32,
        extra: &DetExtra,
    ) -> (glam::Vec2, glam::Vec2) {
        let scale_x = extra.original_shape.0 as f32 / extra.resized_shape.0 as f32;
        let scale_y = extra.original_shape.1 as f32 / extra.resized_shape.1 as f32;

        let min = glam::Vec2::new(min_x * scale_x, min_y * scale_y);
        let max = glam::Vec2::new(max_x * scale_x, max_y * scale_y);

        (min, max)
    }

    /// Apply text padding to bounding box
    fn apply_text_padding(&self, min: glam::Vec2, max: glam::Vec2, extra: &DetExtra) -> Bbox {
        let config = self.model.config();
        let padding = config.text_padding;

        let padded_min = glam::Vec2::new((min.x - padding).max(0.0), (min.y - padding).max(0.0));
        let padded_max = glam::Vec2::new(
            (max.x + padding).min(extra.original_shape.0 as f32),
            (max.y + padding).min(extra.original_shape.1 as f32),
        );

        Bbox::new(padded_min, padded_max)
    }

    // Minimum bounding rectangle
    fn get_mini_box(
        &self,
        points: &[imageproc::point::Point<i32>],
        min_edge_size: &mut f32,
    ) -> Vec<imageproc::point::Point<f32>> {
        use imageproc::geometry::min_area_rect;

        // Get minimum area rectangle
        let rect = min_area_rect(points);

        // Convert to float points and calculate dimensions
        let rect_points = self.convert_rect_to_float_points(&rect);
        let (width, height) = self.calculate_rect_dimensions(&rect_points);
        *min_edge_size = width.min(height);

        // Sort and reorder points for consistent output
        self.sort_and_reorder_rect_points(rect_points)
    }

    /// Convert integer rectangle points to float points
    fn convert_rect_to_float_points(
        &self,
        rect: &[imageproc::point::Point<i32>],
    ) -> Vec<imageproc::point::Point<f32>> {
        rect.iter()
            .map(|p| imageproc::point::Point::new(p.x as f32, p.y as f32))
            .collect()
    }

    /// Calculate width and height of rectangle
    fn calculate_rect_dimensions(
        &self,
        rect_points: &[imageproc::point::Point<f32>],
    ) -> (f32, f32) {
        let width = ((rect_points[0].x - rect_points[1].x).powi(2)
            + (rect_points[0].y - rect_points[1].y).powi(2))
        .sqrt();
        let height = ((rect_points[1].x - rect_points[2].x).powi(2)
            + (rect_points[1].y - rect_points[2].y).powi(2))
        .sqrt();

        (width, height)
    }

    /// Sort and reorder rectangle points for consistent output
    fn sort_and_reorder_rect_points(
        &self,
        mut rect_points: Vec<imageproc::point::Point<f32>>,
    ) -> Vec<imageproc::point::Point<f32>> {
        use std::cmp::Ordering;

        // Sort by x-coordinate
        rect_points.sort_by(|a, b| {
            if a.x > b.x {
                return Ordering::Greater;
            }
            if a.x == b.x {
                return Ordering::Equal;
            }
            Ordering::Less
        });

        // Determine indices for reordering
        let (index_1, index_4) = if rect_points[1].y > rect_points[0].y {
            (0, 1)
        } else {
            (1, 0)
        };

        let (index_2, index_3) = if rect_points[3].y > rect_points[2].y {
            (2, 3)
        } else {
            (3, 2)
        };

        // Reorder points
        vec![
            rect_points[index_1],
            rect_points[index_2],
            rect_points[index_3],
            rect_points[index_4],
        ]
    }

    // Calculate score
    fn get_score(
        &self,
        contour: &imageproc::contours::Contour<i32>,
        f_map_mat: &image::ImageBuffer<image::Luma<f32>, Vec<f32>>,
    ) -> f32 {
        // Get bounding box of contour
        let (xmin, xmax, ymin, ymax) = self.get_contour_bounds(contour, f_map_mat);

        // Calculate ROI dimensions
        let roi_width = xmax - xmin + 1;
        let roi_height = ymax - ymin + 1;

        if roi_width <= 0 || roi_height <= 0 {
            return 0.0;
        }

        // Create mask for the contour region
        let mask = self.create_contour_mask(contour, xmin, ymin, roi_width, roi_height);

        // Crop the probability map to ROI
        let cropped_img = self.crop_probability_map(f_map_mat, xmin, ymin, roi_width, roi_height);

        // Calculate mean score within the contour
        self.calculate_mean_score(&cropped_img, &mask)
    }

    /// Get bounding box of contour with bounds checking
    fn get_contour_bounds(
        &self,
        contour: &imageproc::contours::Contour<i32>,
        f_map_mat: &image::ImageBuffer<image::Luma<f32>, Vec<f32>>,
    ) -> (i32, i32, i32, i32) {
        // Initialize boundary values
        let mut xmin = i32::MAX;
        let mut xmax = i32::MIN;
        let mut ymin = i32::MAX;
        let mut ymax = i32::MIN;

        // Find bounding box of contour
        for point in contour.points.iter() {
            xmin = xmin.min(point.x);
            xmax = xmax.max(point.x);
            ymin = ymin.min(point.y);
            ymax = ymax.max(point.y);
        }

        let width = f_map_mat.width() as i32;
        let height = f_map_mat.height() as i32;

        // Create a Bbox and use its clamp method for consistency
        let bbox = Bbox::new(
            glam::Vec2::new(xmin as f32, ymin as f32),
            glam::Vec2::new(xmax as f32, ymax as f32),
        );

        let clamped_bbox = bbox.clamp(
            glam::Vec2::ZERO,
            glam::Vec2::new((width - 1) as f32, (height - 1) as f32),
        );

        (
            clamped_bbox.min.x as i32,
            clamped_bbox.max.x as i32,
            clamped_bbox.min.y as i32,
            clamped_bbox.max.y as i32,
        )
    }

    /// Create binary mask for contour region
    fn create_contour_mask(
        &self,
        contour: &imageproc::contours::Contour<i32>,
        xmin: i32,
        ymin: i32,
        roi_width: i32,
        roi_height: i32,
    ) -> image::GrayImage {
        let mut mask = image::GrayImage::new(roi_width as u32, roi_height as u32);

        // Translate contour points to ROI coordinates
        let pts: Vec<imageproc::point::Point<i32>> = contour
            .points
            .iter()
            .map(|p| imageproc::point::Point::new(p.x - xmin, p.y - ymin))
            .collect();

        // Draw polygon mask
        imageproc::drawing::draw_polygon_mut(&mut mask, pts.as_slice(), image::Luma([255]));

        mask
    }

    /// Crop probability map to ROI region
    fn crop_probability_map(
        &self,
        f_map_mat: &image::ImageBuffer<image::Luma<f32>, Vec<f32>>,
        xmin: i32,
        ymin: i32,
        roi_width: i32,
        roi_height: i32,
    ) -> image::ImageBuffer<image::Luma<f32>, Vec<f32>> {
        image::imageops::crop_imm(
            f_map_mat,
            xmin as u32,
            ymin as u32,
            roi_width as u32,
            roi_height as u32,
        )
        .to_image()
    }

    /// Calculate mean score within the contour mask
    fn calculate_mean_score(
        &self,
        cropped_img: &image::ImageBuffer<image::Luma<f32>, Vec<f32>>,
        mask: &image::GrayImage,
    ) -> f32 {
        let mut sum = 0.0;
        let mut count = 0;

        for (x, y, pixel) in cropped_img.enumerate_pixels() {
            if mask.get_pixel(x, y)[0] > 0 {
                sum += pixel[0];
                count += 1;
            }
        }

        if count == 0 { 0.0 } else { sum / count as f32 }
    }

    // Unclip - simplified version without geo-clipper dependency
    fn unclip(
        &self,
        box_points: &[imageproc::point::Point<f32>],
    ) -> Vec<imageproc::point::Point<i32>> {
        // Simplified implementation - return original points as integers
        box_points
            .iter()
            .map(|p| imageproc::point::Point::new(p.x as i32, p.y as i32))
            .collect()
    }

    /// Merge text boxes based on IoU threshold using the most efficient strategy.
    /// - For n <= 2000, use sweep line (O(n log n)), which is fast and memory efficient.
    /// - For n > 2000, use grid-based merging (O(n)~O(n log n)), suitable for massive bbox sets.
    ///   You can easily switch or extend this logic as needed.
    fn merge_bboxes(&self, detections: Vec<TextDetection>) -> Vec<TextDetection> {
        let n = detections.len();
        let config = self.model.config();
        let overlap_threshold = config.overlap_ratio_threshold;

        // Auto-select the most efficient algorithm
        if n > 2000 {
            // For very large n, use grid-based merging
            self.merge_bboxes_grid(detections, overlap_threshold)
        } else {
            // Default: improved sweep line with better overlap detection
            self.merge_bboxes_improved(detections, overlap_threshold)
        }
    }

    /// Improved bbox merging that can handle multiple bboxes merging into one large bbox
    /// Uses a more robust approach that considers both horizontal and vertical overlaps
    fn merge_bboxes_improved(
        &self,
        mut detections: Vec<TextDetection>,
        overlap_threshold: f32,
    ) -> Vec<TextDetection> {
        if detections.len() <= 1 {
            return detections;
        }

        // Sort by area (larger bboxes first) to prioritize merging into larger ones
        detections.sort_by(|a, b| {
            let area_a = a.bbox.area();
            let area_b = b.bbox.area();
            area_b
                .partial_cmp(&area_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut merged = Vec::new();
        let mut used = vec![false; detections.len()];

        // Process each detection starting from the largest
        for i in 0..detections.len() {
            if used[i] {
                continue;
            }

            // Try to merge current bbox with all other unused bboxes
            let (merged_bbox, max_proba) =
                self.merge_with_all_overlapping(&detections, i, &mut used, overlap_threshold);

            merged.push(TextDetection {
                bbox: merged_bbox,
                proba: max_proba,
            });
        }

        merged
    }

    /// Merge a bbox with all overlapping bboxes (not just active ones)
    fn merge_with_all_overlapping(
        &self,
        detections: &[TextDetection],
        start_idx: usize,
        used: &mut [bool],
        overlap_threshold: f32,
    ) -> (Bbox, f32) {
        let mut current_bbox = detections[start_idx].bbox;
        let mut max_proba = detections[start_idx].proba;
        used[start_idx] = true;

        // Keep merging until no more overlaps are found
        let mut changed = true;
        while changed {
            changed = false;

            // Check all unused detections for overlaps
            for (j, other) in detections.iter().enumerate() {
                if used[j] {
                    continue;
                }

                let overlap_ratio = current_bbox.overlap_ratio(&other.bbox);

                // Also check if the other bbox is mostly contained within current bbox
                let other_area = other.bbox.area();
                let intersection_area = current_bbox.intersection(&other.bbox);
                let containment_ratio = intersection_area / other_area;

                if overlap_ratio >= overlap_threshold || containment_ratio >= 0.5 {
                    // Merge with this bbox
                    current_bbox = current_bbox.union(&other.bbox);
                    max_proba = max_proba.max(other.proba);
                    used[j] = true;
                    changed = true;
                }
            }
        }

        (current_bbox, max_proba)
    }

    /// Alternative: Grid-based merging algorithm - O(n) average case
    fn merge_bboxes_grid(
        &self,
        detections: Vec<TextDetection>,
        overlap_threshold: f32,
    ) -> Vec<TextDetection> {
        if detections.is_empty() {
            return detections;
        }

        // Calculate grid size based on typical bbox size
        let grid_size = self.calculate_optimal_grid_size(&detections);

        // Create spatial grid
        let mut grid: std::collections::HashMap<(i32, i32), Vec<usize>> =
            std::collections::HashMap::new();

        // Assign detections to grid cells
        for (idx, detection) in detections.iter().enumerate() {
            let cells = self.get_bbox_grid_cells(&detection.bbox, grid_size);
            for cell in cells {
                grid.entry(cell).or_default();
                grid.get_mut(&cell).unwrap().push(idx);
            }
        }

        // Merge detections within each grid cell
        let mut merged = Vec::new();
        let mut used = vec![false; detections.len()];

        for cell_detections in grid.values() {
            if cell_detections.len() <= 1 {
                continue;
            }

            // Merge detections in this cell
            for &idx in cell_detections {
                if used[idx] {
                    continue;
                }

                let (merged_bbox, max_proba) = self.merge_detections_in_cell(
                    &detections,
                    idx,
                    cell_detections,
                    &mut used,
                    overlap_threshold,
                );

                merged.push(TextDetection {
                    bbox: merged_bbox,
                    proba: max_proba,
                });
            }
        }

        // Add unmerged detections
        for (idx, detection) in detections.iter().enumerate() {
            if !used[idx] {
                merged.push(detection.clone());
            }
        }

        merged
    }

    /// Calculate optimal grid size based on bbox statistics
    fn calculate_optimal_grid_size(&self, detections: &[TextDetection]) -> f32 {
        if detections.is_empty() {
            return 50.0; // Default grid size
        }

        // Calculate average bbox size
        let total_width: f32 = detections.iter().map(|d| d.bbox.max.x - d.bbox.min.x).sum();
        let total_height: f32 = detections.iter().map(|d| d.bbox.max.y - d.bbox.min.y).sum();
        let avg_size = (total_width + total_height) / (2.0 * detections.len() as f32);

        // Grid size should be 2-3 times the average bbox size
        (avg_size * 2.5).clamp(20.0, 100.0)
    }

    /// Get grid cells that a bbox overlaps with
    fn get_bbox_grid_cells(&self, bbox: &Bbox, grid_size: f32) -> Vec<(i32, i32)> {
        let min_x = (bbox.min.x / grid_size).floor() as i32;
        let max_x = (bbox.max.x / grid_size).floor() as i32;
        let min_y = (bbox.min.y / grid_size).floor() as i32;
        let max_y = (bbox.max.y / grid_size).floor() as i32;

        let mut cells = Vec::new();
        for x in min_x..=max_x {
            for y in min_y..=max_y {
                cells.push((x, y));
            }
        }
        cells
    }

    /// Merge detections within a grid cell
    fn merge_detections_in_cell(
        &self,
        detections: &[TextDetection],
        start_idx: usize,
        cell_indices: &[usize],
        used: &mut [bool],
        overlap_threshold: f32,
    ) -> (Bbox, f32) {
        let mut current_bbox = detections[start_idx].bbox;
        let mut max_proba = detections[start_idx].proba;
        used[start_idx] = true;

        for &idx in cell_indices {
            if used[idx] || idx == start_idx {
                continue;
            }

            let overlap_ratio = current_bbox.overlap_ratio(&detections[idx].bbox);
            if overlap_ratio >= overlap_threshold {
                current_bbox = current_bbox.union(&detections[idx].bbox);
                max_proba = max_proba.max(detections[idx].proba);
                used[idx] = true;
            }
        }

        (current_bbox, max_proba)
    }

    #[tracing::instrument(skip_all)]
    pub fn draw<P: AsRef<Path>>(
        output: P,
        detections: &[TextDetection],
        image: &DynamicImage,
    ) -> Result<(), FerrpdfError> {
        let mut output_img = image.to_rgb8();

        for detection in detections.iter() {
            let bbox = detection.bbox;
            let x = bbox.min.x as i32;
            let y = bbox.min.y as i32;

            let width = (bbox.max.x - bbox.min.x) as u32;
            let height = (bbox.max.y - bbox.min.y) as u32;

            if width == 0 || height == 0 {
                continue;
            }

            // Draw bounding box with thicker lines
            let color = Rgb([255, 0, 0]); // Red color for text detection

            // Draw multiple rectangles to create thicker lines
            for offset in 0..3 {
                let thick_rect = Rect::at(x - offset, y - offset)
                    .of_size(width + (offset * 2) as u32, height + (offset * 2) as u32);
                draw_hollow_rect_mut(&mut output_img, thick_rect, color);
            }
        }

        // Save the output image
        output_img.save(output.as_ref()).context(ImageWriteSnafu {
            path: output.as_ref().to_string_lossy(),
        })?;

        Ok(())
    }

    /// Sorts text detections by reading order: top-to-bottom, left-to-right for same-line elements
    pub fn sort_by_reading_order(&self, detections: &mut [TextDetection]) {
        if detections.len() < 2 {
            return;
        }

        let config = self.model.config();
        let y_tolerance_threshold = config.y_tolerance_threshold;

        // fix: user-provided comparison function does not correctly implement a total order
        detections.sort_unstable_by(|a, b| {
            let a_center = a.bbox.center();
            let b_center = b.bbox.center();

            let y_category_a = (a_center.y / y_tolerance_threshold).floor() as i32;
            let y_category_b = (b_center.y / y_tolerance_threshold).floor() as i32;

            match y_category_a.cmp(&y_category_b) {
                std::cmp::Ordering::Equal => match a_center.x.partial_cmp(&b_center.x) {
                    Some(ordering) => ordering,
                    None => std::cmp::Ordering::Equal,
                },
                ordering => ordering,
            }
        });
    }
}

impl OnnxSession<PaddleDet> for PaddleDetSession<PaddleDet> {
    type Output = Vec<TextDetection>;
    type Extra = DetExtra;

    fn preprocess(
        &self,
        image: &DynamicImage,
    ) -> Result<<PaddleDet as Model>::Input, FerrpdfError> {
        let config = self.model.config();

        // Convert to RGB if needed
        let img_src = image.to_rgb8();
        let (orig_w, orig_h) = img_src.dimensions();

        // Calculate scale to fit within required dimensions while maintaining aspect ratio
        let scale = f32::min(
            config.required_width as f32 / orig_w as f32,
            config.required_height as f32 / orig_h as f32,
        );

        let new_w = (orig_w as f32 * scale) as u32;
        let new_h = (orig_h as f32 * scale) as u32;

        // Resize image
        let resized_img = image::imageops::resize(&img_src, new_w, new_h, FilterType::Triangle);

        // Create input tensor with padding
        let mut input_tensor = Array4::zeros([
            config.batch_size,
            config.input_channels,
            config.required_height,
            config.required_width,
        ]);

        // Fill with background value
        input_tensor.fill(config.background_fill_value);

        // Copy resized image to tensor
        for (x, y, pixel) in resized_img.enumerate_pixels() {
            let x = x as usize;
            let y = y as usize;
            let [r, g, b] = pixel.0;

            // Optimized normalization: r * NORMALIZATION_SCALE -1.0
            input_tensor[[0, 0, y, x]] = r as f32 * NORMALIZATION_SCALE - 1.0;
            input_tensor[[0, 1, y, x]] = g as f32 * NORMALIZATION_SCALE - 1.0;
            input_tensor[[0, 2, y, x]] = b as f32 * NORMALIZATION_SCALE - 1.0;
        }

        Ok(input_tensor)
    }

    fn postprocess(
        &self,
        output: <PaddleDet as Model>::Output,
        extra: Self::Extra,
    ) -> Result<Self::Output, FerrpdfError> {
        let mut detections = self.db_postprocess(&output, &extra);
        self.sort_by_reading_order(&mut detections);
        Ok(detections)
    }

    fn infer(
        &mut self,
        input: <PaddleDet as Model>::Input,
        input_name: &str,
        output_name: &str,
    ) -> Result<<PaddleDet as Model>::Output, FerrpdfError> {
        // Run inference
        let output = self
            .session
            .run(ort::inputs![
                input_name => TensorRef::from_array_view(&input).context(TensorSnafu{stage: "detec-tinput"})?
            ])
            .context(InferenceSnafu {})?;

        // Extract the output tensor and convert to ndarray
        let tensor = output
            .get(output_name)
            .context(NotFoundOutputSnafu { output_name })?
            .try_extract_array::<f32>()
            .context(TensorSnafu {
                stage: "detect-extract",
            })?;

        // Convert to owned array with proper shape
        let _shape = tensor.shape();

        // Remove batch dimension if present and convert to 3D
        let slice_3d = tensor.slice(ndarray::s![0, .., .., ..]);
        let output_array = Array3::from_shape_vec(
            (
                slice_3d.shape()[0],
                slice_3d.shape()[1],
                slice_3d.shape()[2],
            ),
            slice_3d.iter().cloned().collect(),
        )
        .context(ShapeSnafu { stage: "detect" })?;
        Ok(output_array)
    }

    async fn infer_async(
        &mut self,
        input: <PaddleDet as Model>::Input,
        input_name: &str,
        output_name: &str,
        options: &RunOptions,
    ) -> Result<<PaddleDet as Model>::Output, FerrpdfError> {
        // Run inference
        let output = self
        .session
        .run_async(ort::inputs![
            input_name => TensorRef::from_array_view(&input).context(TensorSnafu{stage: "detect-input"})?
        ], options)
        .context(InferenceSnafu {})?.await.context(InferenceSnafu{})?;

        // Extract the output tensor and convert to ndarray
        let tensor = output
            .get(output_name)
            .context(NotFoundOutputSnafu { output_name })?
            .try_extract_array::<f32>()
            .context(TensorSnafu {
                stage: "detect-extract",
            })?;

        // Convert to owned array with proper shape
        let _shape = tensor.shape();

        // Remove batch dimension if present and convert to 3D
        let slice_3d = tensor.slice(ndarray::s![0, .., .., ..]);
        let output_array = Array3::from_shape_vec(
            (
                slice_3d.shape()[0],
                slice_3d.shape()[1],
                slice_3d.shape()[2],
            ),
            slice_3d.iter().cloned().collect(),
        )
        .context(ShapeSnafu { stage: "detect" })?;

        Ok(output_array)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ort::{
        execution_providers::CPUExecutionProvider,
        session::{Session, builder::GraphOptimizationLevel},
    };

    #[test]
    fn test_paddle_det_session() -> Result<(), Box<dyn std::error::Error>> {
        let session_builder = Session::builder()?
            .with_execution_providers(vec![CPUExecutionProvider::default().build()])?
            .with_optimization_level(GraphOptimizationLevel::Level1)?;

        let model = PaddleDet::new();
        assert!(PaddleDetSession::new(session_builder, model).is_ok());

        println!("PaddleDet session initialized successfully");
        Ok(())
    }

    #[test]
    fn test_paddle_det_preprocessing() -> Result<(), Box<dyn std::error::Error>> {
        let test_image = image::DynamicImage::new_rgb8(800, 600);

        let session_builder = Session::builder()?
            .with_execution_providers(vec![CPUExecutionProvider::default().build()])?
            .with_optimization_level(GraphOptimizationLevel::Level1)?;

        let model = PaddleDet::new();
        let session = PaddleDetSession::new(session_builder, model)?;

        // Test preprocessing
        let preprocessed = session.preprocess(&test_image)?;
        println!("Preprocessed tensor shape: {:?}", preprocessed.shape());

        // Test cropping functionality
        let bbox = Bbox::new(glam::Vec2::new(100.0, 50.0), glam::Vec2::new(700.0, 550.0));
        let cropped = session.crop_image_region(&test_image, &bbox)?;
        println!(
            "Cropped image dimensions: {}x{}",
            cropped.width(),
            cropped.height()
        );

        Ok(())
    }

    #[test]
    fn test_text_detection_workflow() -> Result<(), Box<dyn std::error::Error>> {
        use crate::{analysis::labels::Label, layout::element::Layout};

        let test_image = image::DynamicImage::new_rgb8(1000, 800);

        let session_builder = Session::builder()?
            .with_execution_providers(vec![CPUExecutionProvider::default().build()])?
            .with_optimization_level(GraphOptimizationLevel::Level1)?;

        let model = PaddleDet::new();
        let mut session = PaddleDetSession::new(session_builder, model)?;

        // Create test layouts
        let layouts = vec![
            Layout {
                bbox: Bbox::new(glam::Vec2::new(50.0, 50.0), glam::Vec2::new(450.0, 200.0)),
                label: Label::Text,
                page_no: 0,
                bbox_id: 0,
                proba: 0.9,
                ocr: None,
                text: None,
            },
            Layout {
                bbox: Bbox::new(glam::Vec2::new(50.0, 250.0), glam::Vec2::new(450.0, 400.0)),
                label: Label::Text,
                page_no: 0,
                bbox_id: 1,
                proba: 0.85,
                ocr: None,
                text: None,
            },
        ];

        // Test text detection in layouts (will return empty for test image)
        let scale = 1.0; // Test scale factor
        let result = session.detect_text_in_layouts(&test_image, &layouts, scale);
        println!("Text detection test completed: {:?}", result.is_ok());

        if let Ok(detections) = result {
            println!("Number of layout regions processed: {}", detections.len());
            for (i, layout_detections) in detections.iter().enumerate() {
                println!(
                    "Layout {}: {} text lines detected",
                    i,
                    layout_detections.len()
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_coordinate_system_transformations() {
        // Test coordinate system transformations between PDF and image coordinates
        let pdf_size = glam::Vec2::new(612.0, 792.0); // Standard letter size in points
        let image_size = glam::Vec2::new(816.0, 1056.0); // Scaled to fit target resolution
        let pdf_to_image_scale = image_size.x / pdf_size.x; // ~1.33

        // Create a bbox in PDF coordinates
        let pdf_bbox = Bbox::new(glam::Vec2::new(100.0, 200.0), glam::Vec2::new(300.0, 400.0));

        // Transform to image coordinates
        let image_bbox = pdf_bbox.scale(pdf_to_image_scale);

        // Transform back to PDF coordinates
        let back_to_pdf = image_bbox.scale(1.0 / pdf_to_image_scale);

        // Should be approximately equal (within floating point precision)
        let tolerance = 0.1;
        assert!((pdf_bbox.min.x - back_to_pdf.min.x).abs() < tolerance);
        assert!((pdf_bbox.min.y - back_to_pdf.min.y).abs() < tolerance);
        assert!((pdf_bbox.max.x - back_to_pdf.max.x).abs() < tolerance);
        assert!((pdf_bbox.max.y - back_to_pdf.max.y).abs() < tolerance);

        // Validate that image coordinates are within image bounds
        assert!(image_bbox.min.x >= 0.0 && image_bbox.min.x <= image_size.x);
        assert!(image_bbox.min.y >= 0.0 && image_bbox.min.y <= image_size.y);
        assert!(image_bbox.max.x >= image_bbox.min.x && image_bbox.max.x <= image_size.x);
        assert!(image_bbox.max.y >= image_bbox.min.y && image_bbox.max.y <= image_size.y);

        println!("PDF bbox: {:?}", pdf_bbox);
        println!("Image bbox: {:?}", image_bbox);
        println!("Back to PDF: {:?}", back_to_pdf);
        println!("Scale factor: {:.3}", pdf_to_image_scale);
    }

    #[test]
    fn test_bbox_scaling_edge_cases() {
        let scale = 2.0;

        // Test zero-size bbox
        let zero_bbox = Bbox::new(glam::Vec2::new(0.0, 0.0), glam::Vec2::new(0.0, 0.0));
        let scaled_zero = zero_bbox.scale(scale);
        assert_eq!(scaled_zero.min, glam::Vec2::new(0.0, 0.0));
        assert_eq!(scaled_zero.max, glam::Vec2::new(0.0, 0.0));

        // Test unit bbox
        let unit_bbox = Bbox::new(glam::Vec2::new(0.0, 0.0), glam::Vec2::new(1.0, 1.0));
        let scaled_unit = unit_bbox.scale(scale);
        assert_eq!(scaled_unit.min, glam::Vec2::new(0.0, 0.0));
        assert_eq!(scaled_unit.max, glam::Vec2::new(2.0, 2.0));

        // Test negative coordinates
        let negative_bbox = Bbox::new(glam::Vec2::new(-10.0, -5.0), glam::Vec2::new(10.0, 5.0));
        let scaled_negative = negative_bbox.scale(scale);
        assert_eq!(scaled_negative.min, glam::Vec2::new(-20.0, -10.0));
        assert_eq!(scaled_negative.max, glam::Vec2::new(20.0, 10.0));
    }

    #[test]
    fn test_text_detection_struct() {
        // Test TextDetection struct creation
        let bbox = Bbox::new(glam::Vec2::new(10.0, 20.0), glam::Vec2::new(100.0, 50.0));
        let detection = TextDetection { bbox, proba: 0.95 };

        assert_eq!(detection.bbox.min, glam::Vec2::new(10.0, 20.0));
        assert_eq!(detection.bbox.max, glam::Vec2::new(100.0, 50.0));
        assert_eq!(detection.proba, 0.95);
    }

    #[test]
    fn test_det_extra_struct() {
        // Test DetExtra struct creation
        let extra = DetExtra {
            original_shape: (800, 600),
            resized_shape: (960, 720),
            layout_bbox: Some(Bbox::new(
                glam::Vec2::new(100.0, 100.0),
                glam::Vec2::new(200.0, 200.0),
            )),
        };

        assert_eq!(extra.original_shape, (800, 600));
        assert_eq!(extra.resized_shape, (960, 720));
        assert!(extra.layout_bbox.is_some());
    }

    #[test]
    fn test_paddle_det_config() {
        use crate::inference::paddle::detect::model::PaddleDetConfig;

        // Test default configuration
        let config = PaddleDetConfig::default();
        assert_eq!(config.max_side_thresh, 3.0);
        assert_eq!(config.text_padding, 6.0);
        assert_eq!(config.det_db_thresh, 0.3);
        assert_eq!(config.det_db_box_thresh, 0.6);

        // Test custom configuration
        let config = PaddleDetConfig {
            overlap_ratio_threshold: 0.7, // 70% IoU threshold
            ..PaddleDetConfig::default()
        };

        assert_eq!(config.max_side_thresh, 3.0);
        assert_eq!(config.text_padding, 6.0);
        assert_eq!(config.det_db_thresh, 0.3);
        assert_eq!(config.det_db_box_thresh, 0.6);
        assert_eq!(config.overlap_ratio_threshold, 0.7);
    }

    #[test]
    fn test_text_padding_application() -> Result<(), Box<dyn std::error::Error>> {
        use crate::inference::paddle::detect::model::PaddleDet;
        use crate::inference::paddle::detect::model::PaddleDetConfig;

        // Create a session with custom padding
        let session_builder = Session::builder()?
            .with_execution_providers(vec![CPUExecutionProvider::default().build()])?
            .with_optimization_level(GraphOptimizationLevel::Level1)?;

        let config = PaddleDetConfig {
            text_padding: 5.0,
            ..PaddleDetConfig::default()
        };
        let model = PaddleDet::with_config(config);
        let session = PaddleDetSession::new(session_builder, model)?;

        // Test that padding is applied correctly in bbox conversion
        let extra = DetExtra {
            original_shape: (1000, 800),
            resized_shape: (960, 720),
            layout_bbox: None,
        };

        // Create test points that would result in a bbox at (100, 100) to (200, 150)
        let test_points = vec![
            imageproc::point::Point::new(100.0, 100.0),
            imageproc::point::Point::new(200.0, 100.0),
            imageproc::point::Point::new(200.0, 150.0),
            imageproc::point::Point::new(100.0, 150.0),
        ];

        let bbox = session.points_to_bbox(&test_points, &extra);

        // With 5.0 padding, the bbox should be expanded by 5 pixels on each side
        // Scale factor: 1000/960 ≈ 1.042, 800/720 ≈ 1.111
        let scale_x = 1000.0_f32 / 960.0;
        let scale_y = 800.0_f32 / 720.0;

        let expected_min_x = (100.0 * scale_x - 5.0).max(0.0);
        let expected_min_y = (100.0 * scale_y - 5.0).max(0.0);
        let expected_max_x = (200.0 * scale_x + 5.0).min(1000.0);
        let expected_max_y = (150.0 * scale_y + 5.0).min(800.0);

        assert!((bbox.min.x - expected_min_x).abs() < 0.1);
        assert!((bbox.min.y - expected_min_y).abs() < 0.1);
        assert!((bbox.max.x - expected_max_x).abs() < 0.1);
        assert!((bbox.max.y - expected_max_y).abs() < 0.1);

        Ok(())
    }

    #[test]
    fn test_overlapping_bbox_removal() -> Result<(), Box<dyn std::error::Error>> {
        use crate::inference::paddle::detect::model::PaddleDet;
        use crate::inference::paddle::detect::model::PaddleDetConfig;

        let session_builder = Session::builder()?
            .with_execution_providers(vec![CPUExecutionProvider::default().build()])?
            .with_optimization_level(GraphOptimizationLevel::Level1)?;

        let config = PaddleDetConfig {
            overlap_ratio_threshold: 0.3, // Lower threshold to ensure overlapping detection
            ..PaddleDetConfig::default()
        };
        let model = PaddleDet::with_config(config);
        let session = PaddleDetSession::new(session_builder, model)?;

        // Create overlapping detections
        let detections = vec![
            TextDetection {
                bbox: Bbox::new(glam::Vec2::new(10.0, 20.0), glam::Vec2::new(100.0, 40.0)), // Higher confidence
                proba: 0.9,
            },
            TextDetection {
                bbox: Bbox::new(glam::Vec2::new(50.0, 25.0), glam::Vec2::new(150.0, 45.0)), // Overlapping, lower confidence
                proba: 0.7,
            },
            TextDetection {
                bbox: Bbox::new(glam::Vec2::new(200.0, 20.0), glam::Vec2::new(300.0, 40.0)), // Non-overlapping
                proba: 0.8,
            },
        ];

        let result = session.merge_bboxes(detections);

        // Should have 2 non-overlapping detections
        assert_eq!(result.len(), 2);

        // First should be the highest confidence one
        assert_eq!(result[0].proba, 0.9);
        assert_eq!(result[0].bbox.min.x, 10.0);
        assert_eq!(result[0].bbox.max.x, 150.0); // Merged with overlapping box

        // Second should be the non-overlapping one
        assert_eq!(result[1].proba, 0.8);
        assert_eq!(result[1].bbox.min.x, 200.0);
        assert_eq!(result[1].bbox.max.x, 300.0);

        println!("Overlapping bbox removal test passed");
        Ok(())
    }

    #[test]
    fn test_y_tolerance_threshold_config() {
        use crate::inference::paddle::detect::model::PaddleDetConfig;
        use glam::Vec2;

        // Test default value
        let config = PaddleDetConfig::default();
        assert_eq!(config.y_tolerance_threshold, 5.0);

        // Test custom value
        let paddle_det_config = PaddleDetConfig {
            y_tolerance_threshold: 10.0,
            ..Default::default()
        };
        assert_eq!(paddle_det_config.y_tolerance_threshold, 10.0);

        // Test that the session uses the config value
        let model = PaddleDet::with_config(paddle_det_config);
        let session = PaddleDetSession::new(SessionBuilder::new().unwrap(), model).unwrap();

        // Create test detections with different y-coordinates
        let mut detections = vec![
            TextDetection {
                bbox: Bbox::new(Vec2::new(0.0, 0.0), Vec2::new(10.0, 10.0)),
                proba: 0.9,
            },
            TextDetection {
                bbox: Bbox::new(Vec2::new(20.0, 5.0), Vec2::new(30.0, 15.0)), // y_diff = 5
                proba: 0.8,
            },
            TextDetection {
                bbox: Bbox::new(Vec2::new(40.0, 15.0), Vec2::new(50.0, 25.0)), // y_diff = 10
                proba: 0.7,
            },
        ];

        // Sort by reading order
        session.sort_by_reading_order(&mut detections);

        // Verify sorting behavior based on y_tolerance_threshold
        // With threshold = 10.0, the first two should be considered on same line
        // and sorted by x-coordinate, while the third should be below
        assert_eq!(detections[0].bbox.center().x, 5.0); // First bbox (x=5)
        assert_eq!(detections[1].bbox.center().x, 25.0); // Second bbox (x=25)
        assert_eq!(detections[2].bbox.center().x, 45.0); // Third bbox (x=45)
    }
}
