use std::path::Path;

use ab_glyph::{FontRef, PxScale};
use glam::Vec2;
use image::{DynamicImage, GenericImageView, Rgb, imageops::FilterType};
use imageproc::{
    drawing::{draw_hollow_rect_mut, draw_text_mut},
    rect::Rect,
};
use ndarray::prelude::*;
use ort::{
    session::{Session, builder::SessionBuilder},
    value::TensorRef,
};
use snafu::{OptionExt, ResultExt};

use crate::{
    analysis::{bbox::Bbox, labels::Label},
    consts::FONT,
    error::*,
    inference::{
        model::{Model, OnnxSession},
        yolov12::model::Yolov12,
    },
    layout::element::Layout,
};

pub struct YoloSession<M: Model> {
    session: Session,
    model: M,
}

/// Document metadata
pub struct DocMeta {
    pub pdf_size: Vec2,
    pub image_size: Vec2,
    pub scale: f32,
    pub page: usize,
}

impl YoloSession<Yolov12> {
    pub fn new(session: SessionBuilder, model: Yolov12) -> Result<Self, FerrpdfError> {
        let session = session
            .commit_from_memory(model.load())
            .context(OrtInitSnafu { stage: "commit" })?;

        Ok(Self { session, model })
    }
}

impl OnnxSession<Yolov12> for YoloSession<Yolov12> {
    type Output = Vec<Layout>;
    type Extra = DocMeta;

    fn preprocess(&self, image: &DynamicImage) -> Result<<Yolov12 as Model>::Input, FerrpdfError> {
        /// Calculates optimal scaling dimensions to fit an image within target dimensions while maintaining aspect ratio.
        fn scale_wh(w0: f32, h0: f32, target_w: f32, target_h: f32) -> (f32, f32, f32) {
            let scale = f32::min(target_w / w0, target_h / h0);
            let w_new = (w0 * scale).round();
            let h_new = (h0 * scale).round();
            (scale, w_new, h_new)
        }

        let model_config = self.model.config();

        let (w0, h0) = image.dimensions();
        let (_, w_new, h_new) = scale_wh(
            w0 as f32,
            h0 as f32,
            model_config.required_width as f32,
            model_config.required_height as f32,
        );

        // Resize image to calculated dimensions
        let resized_img = image.resize_exact(w_new as u32, h_new as u32, FilterType::Triangle);

        // Create tensor with background fill value
        let mut input_tensor = Array4::ones([
            model_config.batch_size,
            model_config.input_channels,
            model_config.required_height,
            model_config.required_width,
        ]);
        input_tensor.fill(model_config.background_fill_value);

        // Fill tensor with normalized pixel values
        for (x, y, pixel) in resized_img.pixels() {
            let x = x as usize;
            let y = y as usize;
            let [r, g, b, _] = pixel.0;
            input_tensor[[0, 0, y, x]] = r as f32 / 255.0;
            input_tensor[[0, 1, y, x]] = g as f32 / 255.0;
            input_tensor[[0, 2, y, x]] = b as f32 / 255.0;
        }

        Ok(input_tensor)
    }

    fn postprocess(
        &self,
        output: <Yolov12 as Model>::Output,
        extra: Self::Extra,
    ) -> Result<Self::Output, FerrpdfError> {
        // extra bbox
        let mut layouts = self.extra_bbox(output, &extra);

        // merge overlapping boxes
        self.nms(&mut layouts);

        // sort bbox order by top-left first
        self.sort_by_reading_order(&mut layouts);

        Ok(layouts)
    }

    fn infer(
        &mut self,
        input: <Yolov12 as Model>::Input,
        input_name: &str,
        output_name: &str,
    ) -> Result<<Yolov12 as Model>::Output, FerrpdfError> {
        // Run inference with the input tensor named "images"
        let output = self
            .session
            .run(ort::inputs![
                input_name => TensorRef::from_array_view(&input).context(TensorSnafu{stage: "input"})?
            ])
            .context(InferenceSnafu {})?;

        // Extract the output tensor and convert to ndarray
        let tensor = output
            .get(output_name)
            .context(NotFoundOutputSnafu { output_name })?
            .try_extract_array::<f32>()
            .context(TensorSnafu { stage: "extract" })?;

        // Reshape tensor to expected output dimensions and return owned copy
        let output = tensor
            .to_shape(self.model.config().output_size)
            .context(ShapeSnafu { stage: "output" })?
            .to_owned();

        Ok(output)
    }
}

impl YoloSession<Yolov12> {
    fn extra_bbox(&self, output: <Yolov12 as Model>::Output, extra: &DocMeta) -> Vec<Layout> {
        let mut layouts = Vec::new();

        // Get the first batch slice (assuming batch size = 1)
        let output = output.slice(ndarray::s![0, .., ..]);

        // Iterate through each prediction in the output
        let mut label_idx = 0;
        let config = self.model.config();
        for prediction in output.axis_iter(Axis(1)) {
            // Extract bounding box coordinates (center_x, center_y, width, height)
            let bbox = prediction.slice(ndarray::s![0..config.cxywh_size]);
            // Extract class probabilities for different document elements
            let labels = prediction.slice(ndarray::s![config.cxywh_size..config.label_size]);

            // Find the class with the highest probability
            let (max_prob_idx, &proba) = labels
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();

            // Skip detections below confidence threshold
            if proba < config.proba_threshold {
                continue;
            }

            // Extract YOLO format coordinates (center x, center y, width, height)
            let cx = bbox[0_usize];
            let cy = bbox[1_usize];
            let w = bbox[2_usize];
            let h = bbox[3_usize];

            // Convert from YOLO center-size format to bounding box and clamp to image bounds
            let bbox = Bbox::from_center_size(Vec2::new(cx, cy), Vec2::new(w, h))
                .clamp(Vec2::new(0.0, 0.0), extra.image_size)
                .scale(1. / extra.scale); // transform to pdf coordinates

            // Create layout element with detected information
            let layout = Layout {
                bbox,
                label: Label::from(max_prob_idx),
                page_no: extra.page,
                bbox_id: label_idx,
                proba,
                text: None, // Text content will be extracted later
            };
            label_idx += 1;

            layouts.push(layout);
        }

        layouts
    }

    fn nms(&self, raw_layouts: &mut Vec<Layout>) {
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

        let mut keep_flags = vec![true; raw_layouts.len()];

        // Process each layout and check for merges with previous kept layouts
        for current_index in 0..raw_layouts.len() {
            if !keep_flags[current_index] {
                continue; // Already marked for removal
            }

            let current_bbox = raw_layouts[current_index].bbox;

            // Check against all previously processed layouts that are still kept
            for kept_index in 0..current_index {
                if !keep_flags[kept_index] {
                    continue; // Skip layouts that were already merged/removed
                }

                let overlap_ratio = current_bbox.overlap_ratio(&raw_layouts[kept_index].bbox);

                // If overlap ratio is high, merge current into kept detection
                // Since array is sorted by confidence, kept detections have higher confidence
                if overlap_ratio > self.model.config().iou_threshold {
                    // Merge the bbox by taking the union - update the kept layout's bbox
                    raw_layouts[kept_index].bbox =
                        raw_layouts[kept_index].bbox.union(&current_bbox);
                    // Mark current layout for removal
                    keep_flags[current_index] = false;
                    break;
                }
            }
        }

        // Remove layouts marked for deletion by swapping with kept ones
        let mut write_index = 0;
        for (read_index, keep_flag) in keep_flags.iter().enumerate().take(raw_layouts.len()) {
            if *keep_flag {
                if write_index != read_index {
                    raw_layouts.swap(write_index, read_index);
                }
                write_index += 1;
            }
        }

        // Truncate to remove the elements that were swapped to the end
        raw_layouts.truncate(write_index);
    }

    #[allow(dead_code)]
    fn draw<P: AsRef<Path>>(
        &self,
        output: P,
        extra: &DocMeta,
        layouts: &[Layout],
        image: &DynamicImage,
    ) -> Result<(), FerrpdfError> {
        // pub fn draw_bbox<P: AsRef<Path>>(
        //     img: &DynamicImage,
        //     layouts: &[Layout],
        //     output_path: P,
        // ) -> Result<(), FerrpdfError> {
        let mut output_img = image.to_rgb8();

        let font = FontRef::try_from_slice(FONT).context(FontSnafu {})?;

        // Define font scale (size)
        let font_scale = PxScale::from(16.0);

        for (idx, layout) in layouts.iter().enumerate() {
            // transform to image size
            let bbox = layout.bbox.scale(extra.scale);
            let x = bbox.min.x as i32;
            let y = bbox.min.y as i32;

            let width = (bbox.max.x - bbox.min.x) as u32;
            let height = (bbox.max.y - bbox.min.y) as u32;

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
        output_img.save(output.as_ref()).context(ImageWriteSnafu {
            path: output.as_ref().to_string_lossy(),
        })?;

        Ok(())
    }

    fn sort_by_reading_order(&self, layouts: &mut Vec<Layout>) {
        if layouts.is_empty() {
            return;
        }

        // Check if we have a multi-column layout
        if Self::is_multi_column_layout(layouts) {
            self.sort_multi_column_layout(layouts);
        } else {
            self.sort_single_column_layout(layouts);
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
    fn sort_multi_column_layout(&self, layouts: &mut Vec<Layout>) {
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
        self.sort_single_column_layout(&mut left_column);
        self.sort_single_column_layout(&mut right_column);

        // Merge back: left column first, then right column
        layouts.extend(left_column);
        layouts.extend(right_column);
    }

    /// Sorts single column layout by reading order: top-to-bottom, left-to-right for same-line elements
    fn sort_single_column_layout(&self, layouts: &mut [Layout]) {
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

            if y_diff.abs() <= self.model.config().y_tolerance_threshold {
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
