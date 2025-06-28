use std::path::Path;

use ab_glyph::{FontRef, PxScale};
use derive_builder::Builder;
use glam::Vec2;
use image::{DynamicImage, GenericImageView, Rgb, imageops::FilterType};
use imageproc::{
    drawing::{draw_hollow_rect_mut, draw_text_mut},
    rect::Rect,
};
use ndarray::prelude::*;
use ort::{
    session::{RunOptions, Session, builder::SessionBuilder},
    value::TensorRef,
};
use serde::Serialize;
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
#[derive(Clone, Copy, Builder, Debug, Serialize)]
#[builder(setter(into))]
pub struct DocMeta {
    pub pdf_size: Vec2,
    pub image_size: Vec2,
    pub scale: f32,
    pub page: usize,
}

#[derive(Debug, Serialize)]
pub struct PdfLayouts {
    pub layouts: Vec<Layout>,
    pub metadata: DocMeta,
}

impl YoloSession<Yolov12> {
    pub fn new(session: SessionBuilder, model: Yolov12) -> Result<Self, FerrpdfError> {
        let span = tracing::info_span!("init-yolosession");
        let _guard = span.enter();
        let session = session
            .commit_from_memory(model.load())
            .context(OrtInitSnafu { stage: "commit" })?;

        Ok(Self { session, model })
    }
}

impl OnnxSession<Yolov12> for YoloSession<Yolov12> {
    type Output = PdfLayouts;
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
        let resized_img = image.resize_exact(w_new as u32, h_new as u32, FilterType::CatmullRom);

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

        Ok(PdfLayouts {
            layouts,
            metadata: extra,
        })
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
                input_name => TensorRef::from_array_view(&input).context(TensorSnafu{stage: "layout-input"})?
            ])
            .context(InferenceSnafu {})?;

        // Extract the output tensor and convert to ndarray
        let tensor = output
            .get(output_name)
            .context(NotFoundOutputSnafu { output_name })?
            .try_extract_array::<f32>()
            .context(TensorSnafu {
                stage: "layout-extract",
            })?;

        // Reshape tensor to expected output dimensions and return owned copy
        let output = tensor
            .to_shape(self.model.config().output_size)
            .context(ShapeSnafu { stage: "layout" })?
            .to_owned();

        Ok(output)
    }

    async fn infer_async(
        &mut self,
        input: <Yolov12 as Model>::Input,
        input_name: &str,
        output_name: &str,
        options: &RunOptions,
    ) -> Result<<Yolov12 as Model>::Output, FerrpdfError> {
        // Run inference with the input tensor named "images"
        let output = self
            .session
            .run_async(ort::inputs![
                input_name => TensorRef::from_array_view(&input).context(TensorSnafu{stage: "layout-input"})?
            ], options)
            .context(InferenceSnafu {})?.await.context(InferenceSnafu{})?;

        // Extract the output tensor and convert to ndarray
        let tensor = output
            .get(output_name)
            .context(NotFoundOutputSnafu { output_name })?
            .try_extract_array::<f32>()
            .context(TensorSnafu {
                stage: "layout-extract",
            })?;

        // Reshape tensor to expected output dimensions and return owned copy
        let output = tensor
            .to_shape(self.model.config().output_size)
            .context(ShapeSnafu { stage: "layout" })?
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
        let scale = 1. / extra.scale;
        for prediction in output.axis_iter(Axis(1)) {
            // Extract bounding box coordinates (center_x, center_y, width, height)
            let bbox = prediction.slice(ndarray::s![0..config.cxywh_size]);
            // Extract class probabilities for different document elements
            let labels = prediction.slice(ndarray::s![config.cxywh_size..config.label_proba_size]);

            // Find the class with the highest probability
            let (max_prob_idx, &proba) = labels
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
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
                .clamp(Vec2::ZERO, extra.image_size)
                .scale(scale); // transform to pdf coordinates

            // Create layout element with detected information
            let layout = Layout {
                bbox,
                label: Label::from(max_prob_idx),
                page_no: extra.page,
                bbox_id: label_idx,
                proba,
                ocr: None,
                text: None, // Text content will be extracted later
            };
            label_idx += 1;

            layouts.push(layout);
        }

        layouts
    }

    fn nms(&self, raw_layouts: &mut Vec<Layout>) {
        let len = raw_layouts.len();
        if len < 2 {
            return;
        }

        // Sort layouts by area (larger bboxes first) to prioritize merging into larger ones
        raw_layouts.sort_by(|a, b| b.bbox.area().total_cmp(&a.bbox.area()));

        let mut removed = vec![false; raw_layouts.len()];

        let iou_threshold = self.model.config().iou_threshold;
        // Process each layout starting from the largest
        for i in 0..len {
            if removed[i] {
                continue;
            }

            let mut new_bbox = raw_layouts[i].bbox;
            let mut new_label = raw_layouts[i].label;
            for j in (i + 1)..len {
                let next_layout = &raw_layouts[j];

                if new_bbox.overlap_ratio(&next_layout.bbox) > iou_threshold {
                    new_bbox = new_bbox.union(&next_layout.bbox);
                    if next_layout.proba > raw_layouts[i].proba {
                        new_label = next_layout.label;
                    }
                    removed[j] = true;
                }
            }

            raw_layouts[i].bbox = new_bbox;
            raw_layouts[i].label = new_label;
        }

        let mut idx = 0;
        raw_layouts.retain(|_| {
            let result = !removed[idx];
            idx += 1;
            result
        });
    }

    #[tracing::instrument(skip_all)]
    pub fn draw<P: AsRef<Path>>(
        output: P,
        extra: &DocMeta,
        layouts: &[Layout],
        image: &DynamicImage,
    ) -> Result<(), FerrpdfError> {
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
        x_centers.sort_by(|a, b| a.total_cmp(b));

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

    /// Sorts single column layout by reading order with special handling for headers, titles, and footers
    fn sort_single_column_layout(&self, layouts: &mut [Layout]) {
        layouts.sort_by(|a, b| {
            // Special priority ordering:
            // 1. PageHeader (highest priority, sorted by Y coordinate)
            // 2. Title (second priority, sorted by Y coordinate)
            // 3. Regular elements (sorted by Y coordinate, then X coordinate)
            // 4. PageFooter (lowest priority, sorted by Y coordinate)

            match (&a.label, &b.label) {
                // PageHeader vs PageHeader: compare Y coordinates
                (Label::PageHeader, Label::PageHeader) => {
                    a.bbox.center().y.total_cmp(&b.bbox.center().y)
                }

                // PageHeader vs Title: PageHeader always comes first
                (Label::PageHeader, Label::Title) => std::cmp::Ordering::Less,

                // PageHeader vs others: PageHeader always comes first
                (Label::PageHeader, _) => std::cmp::Ordering::Less,

                // Title vs PageHeader: Title comes after PageHeader
                (Label::Title, Label::PageHeader) => std::cmp::Ordering::Greater,

                // Title vs Title: compare Y coordinates
                (Label::Title, Label::Title) => a.bbox.center().y.total_cmp(&b.bbox.center().y),

                // Title vs PageFooter: Title comes before PageFooter
                (Label::Title, Label::PageFooter) => std::cmp::Ordering::Less,

                // Title vs others: Title comes before regular elements
                (Label::Title, _) => std::cmp::Ordering::Less,

                // PageFooter vs PageHeader/Title: PageFooter always comes last
                (Label::PageFooter, Label::PageHeader | Label::Title) => {
                    std::cmp::Ordering::Greater
                }

                // PageFooter vs PageFooter: compare Y coordinates
                (Label::PageFooter, Label::PageFooter) => {
                    a.bbox.center().y.total_cmp(&b.bbox.center().y)
                }

                // PageFooter vs others: PageFooter always comes last
                (Label::PageFooter, _) => std::cmp::Ordering::Greater,

                // Regular elements vs PageHeader/Title: regular elements come after
                (_, Label::PageHeader | Label::Title) => std::cmp::Ordering::Greater,

                // Regular elements vs PageFooter: regular elements come before
                (_, Label::PageFooter) => std::cmp::Ordering::Less,

                // Regular elements vs regular elements: normal reading order
                (_, _) => {
                    let a_center = a.bbox.center();
                    let b_center = b.bbox.center();

                    // Primary sort: by Y coordinate (top to bottom)
                    let y_diff = a_center.y - b_center.y;

                    if y_diff.abs() <= self.model.config().y_tolerance_threshold {
                        // Elements are on roughly the same line, sort by X coordinate (left to right)
                        a_center.x.total_cmp(&b_center.x)
                    } else {
                        // Different lines, sort by Y coordinate
                        y_diff.total_cmp(&0.0)
                    }
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use ort::{
        execution_providers::CPUExecutionProvider, session::builder::GraphOptimizationLevel,
    };

    use super::*;

    #[test]
    fn test_layout_sorting() -> Result<(), Box<dyn std::error::Error>> {
        use crate::analysis::labels::Label;
        use crate::layout::element::Layout;

        let session_builder = Session::builder()?
            .with_execution_providers(vec![CPUExecutionProvider::default().build()])?
            .with_optimization_level(GraphOptimizationLevel::Level1)?;

        let model = Yolov12::new();
        let session = YoloSession::new(session_builder, model)?;

        // Create test layouts with different labels and positions
        let mut layouts = vec![
            Layout {
                bbox: crate::analysis::bbox::Bbox::new(
                    glam::Vec2::new(100.0, 50.0),
                    glam::Vec2::new(200.0, 80.0),
                ),
                label: Label::Text,
                page_no: 0,
                bbox_id: 0,
                proba: 0.9,
                ocr: None,
                text: None,
            },
            Layout {
                bbox: crate::analysis::bbox::Bbox::new(
                    glam::Vec2::new(50.0, 30.0),
                    glam::Vec2::new(150.0, 60.0),
                ),
                label: Label::Title,
                page_no: 0,
                bbox_id: 1,
                proba: 0.95,
                ocr: None,
                text: None,
            },
            Layout {
                bbox: crate::analysis::bbox::Bbox::new(
                    glam::Vec2::new(20.0, 10.0),
                    glam::Vec2::new(120.0, 40.0),
                ),
                label: Label::PageHeader,
                page_no: 0,
                bbox_id: 2,
                proba: 0.98,
                ocr: None,
                text: None,
            },
            Layout {
                bbox: crate::analysis::bbox::Bbox::new(
                    glam::Vec2::new(80.0, 200.0),
                    glam::Vec2::new(180.0, 230.0),
                ),
                label: Label::PageFooter,
                page_no: 0,
                bbox_id: 3,
                proba: 0.85,
                ocr: None,
                text: None,
            },
            Layout {
                bbox: crate::analysis::bbox::Bbox::new(
                    glam::Vec2::new(150.0, 100.0),
                    glam::Vec2::new(250.0, 130.0),
                ),
                label: Label::Text,
                page_no: 0,
                bbox_id: 4,
                proba: 0.88,
                ocr: None,
                text: None,
            },
        ];

        // Sort the layouts
        session.sort_single_column_layout(&mut layouts);

        // Verify the order: PageHeader -> Title -> Text -> PageFooter
        assert_eq!(layouts[0].label, Label::PageHeader);
        assert_eq!(layouts[1].label, Label::Title);
        assert_eq!(layouts[2].label, Label::Text);
        assert_eq!(layouts[3].label, Label::Text);
        assert_eq!(layouts[4].label, Label::PageFooter);

        println!("Layout sorting test passed");
        Ok(())
    }
}
