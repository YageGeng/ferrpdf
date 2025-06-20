use ab_glyph::PxScale;
use image::Rgb;
use image::{DynamicImage, GenericImageView, imageops::FilterType};
use imageproc::{drawing::draw_hollow_rect_mut, rect::Rect};
use ndarray::prelude::*;
use ort::session::{Session, builder::SessionBuilder};
use ort::value::TensorRef;
use snafu::{OptionExt, ResultExt};
use std::path::Path;

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
#[derive(Debug, Clone)]
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
        use image::{GrayImage, Luma};
        use imageproc::contours::find_contours;
        use imageproc::contrast::threshold;
        use imageproc::distance_transform::Norm;
        use imageproc::morphology::dilate;

        let config = self.model.config();
        let mut detections = Vec::new();

        // 1. Extract probability map
        let prob_map = pred.slice(ndarray::s![0, .., ..]);
        let (h, w) = prob_map.dim();
        let pred_data: Vec<f32> = prob_map.iter().copied().collect();
        let cbuf_data: Vec<u8> = pred_data.iter().map(|&x| (x * 255.0) as u8).collect();

        let pred_img: image::ImageBuffer<Luma<f32>, Vec<f32>> =
            image::ImageBuffer::from_vec(w as u32, h as u32, pred_data).unwrap();
        let cbuf_img = GrayImage::from_vec(w as u32, h as u32, cbuf_data).unwrap();

        // 2. Thresholding
        let threshold_img = threshold(
            &cbuf_img,
            (config.det_db_box_thresh * 255.0) as u8,
            imageproc::contrast::ThresholdType::Binary,
        );

        // 3. Dilation
        let dilate_img = dilate(&threshold_img, Norm::LInf, 1);

        // 4. Contour finding
        let img_contours = find_contours(&dilate_img);

        let max_side_thresh = 3.0;

        for contour in img_contours {
            if contour.points.len() <= 2 {
                continue;
            }

            // 5. Minimum area rectangle
            let mut max_side = 0.0;
            let min_box = self.get_mini_box(&contour.points, &mut max_side);
            if max_side < max_side_thresh {
                continue;
            }

            // 6. Calculate score
            let score = self.get_score(&contour, &pred_img);
            if score < config.det_db_thresh {
                continue;
            }

            // 7. Unclip
            let clip_box = self.unclip(&min_box, config.det_db_unclip_ratio);
            if clip_box.is_empty() {
                continue;
            }

            let mut max_side_clip = 0.0;
            let clip_min_box = self.get_mini_box(&clip_box, &mut max_side_clip);
            if max_side_clip < max_side_thresh + 2.0 {
                continue;
            }

            // 8. Convert to bounding box and scale to original coordinates
            let bbox = self.points_to_bbox(&clip_min_box, extra);

            detections.push(TextDetection { bbox, proba: score });
        }

        detections
    }

    // Convert contour points to bounding box
    fn points_to_bbox(&self, points: &[imageproc::point::Point<f32>], extra: &DetExtra) -> Bbox {
        if points.is_empty() {
            return Bbox::new(glam::Vec2::ZERO, glam::Vec2::ZERO);
        }

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

        // Scale from resized coordinates to original coordinates
        let scale_x = extra.original_shape.0 as f32 / extra.resized_shape.0 as f32;
        let scale_y = extra.original_shape.1 as f32 / extra.resized_shape.1 as f32;

        let min = glam::Vec2::new(min_x * scale_x, min_y * scale_y);
        let max = glam::Vec2::new(max_x * scale_x, max_y * scale_y);

        Bbox::new(min, max)
    }

    // === 辅助方法 ===

    // 最小外接矩形
    fn get_mini_box(
        &self,
        points: &[imageproc::point::Point<i32>],
        min_edge_size: &mut f32,
    ) -> Vec<imageproc::point::Point<f32>> {
        use imageproc::geometry::min_area_rect;
        use std::cmp::Ordering;

        let rect = min_area_rect(points);

        let mut rect_points: Vec<imageproc::point::Point<f32>> = rect
            .iter()
            .map(|p| imageproc::point::Point::new(p.x as f32, p.y as f32))
            .collect();

        let width = ((rect_points[0].x - rect_points[1].x).powi(2)
            + (rect_points[0].y - rect_points[1].y).powi(2))
        .sqrt();
        let height = ((rect_points[1].x - rect_points[2].x).powi(2)
            + (rect_points[1].y - rect_points[2].y).powi(2))
        .sqrt();

        *min_edge_size = width.min(height);

        rect_points.sort_by(|a, b| {
            if a.x > b.x {
                return Ordering::Greater;
            }
            if a.x == b.x {
                return Ordering::Equal;
            }
            Ordering::Less
        });

        let mut box_points = Vec::new();
        let index_1;
        let index_4;
        if rect_points[1].y > rect_points[0].y {
            index_1 = 0;
            index_4 = 1;
        } else {
            index_1 = 1;
            index_4 = 0;
        }

        let index_2;
        let index_3;
        if rect_points[3].y > rect_points[2].y {
            index_2 = 2;
            index_3 = 3;
        } else {
            index_2 = 3;
            index_3 = 2;
        }

        box_points.push(rect_points[index_1]);
        box_points.push(rect_points[index_2]);
        box_points.push(rect_points[index_3]);
        box_points.push(rect_points[index_4]);

        box_points
    }

    // 计算分数
    fn get_score(
        &self,
        contour: &imageproc::contours::Contour<i32>,
        f_map_mat: &image::ImageBuffer<image::Luma<f32>, Vec<f32>>,
    ) -> f32 {
        // 初始化边界值
        let mut xmin = i32::MAX;
        let mut xmax = i32::MIN;
        let mut ymin = i32::MAX;
        let mut ymax = i32::MIN;

        // 找到轮廓的边界框
        for point in contour.points.iter() {
            let x = point.x;
            let y = point.y;

            if x < xmin {
                xmin = x;
            }
            if x > xmax {
                xmax = x;
            }
            if y < ymin {
                ymin = y;
            }
            if y > ymax {
                ymax = y;
            }
        }

        let width = f_map_mat.width() as i32;
        let height = f_map_mat.height() as i32;

        let xmin = xmin.max(0).min(width - 1);
        let xmax = xmax.max(0).min(width - 1);
        let ymin = ymin.max(0).min(height - 1);
        let ymax = ymax.max(0).min(height - 1);

        let roi_width = xmax - xmin + 1;
        let roi_height = ymax - ymin + 1;

        if roi_width <= 0 || roi_height <= 0 {
            return 0.0;
        }

        let mut mask = image::GrayImage::new(roi_width as u32, roi_height as u32);

        let mut pts = Vec::<imageproc::point::Point<i32>>::new();
        for point in contour.points.iter() {
            pts.push(imageproc::point::Point::new(point.x - xmin, point.y - ymin));
        }

        imageproc::drawing::draw_polygon_mut(&mut mask, pts.as_slice(), image::Luma([255]));

        let cropped_img = image::imageops::crop_imm(
            f_map_mat,
            xmin as u32,
            ymin as u32,
            roi_width as u32,
            roi_height as u32,
        )
        .to_image();

        // 计算均值
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

    // unclip - simplified version without geo-clipper dependency
    fn unclip(
        &self,
        box_points: &[imageproc::point::Point<f32>],
        unclip_ratio: f32,
    ) -> Vec<imageproc::point::Point<i32>> {
        // Simplified implementation - return original points as integers
        box_points
            .iter()
            .map(|p| imageproc::point::Point::new(p.x as i32, p.y as i32))
            .collect()
    }

    pub fn draw<P: AsRef<Path>>(
        &self,
        output: P,
        detections: &[TextDetection],
        image: &DynamicImage,
    ) -> Result<(), FerrpdfError> {
        let mut output_img = image.to_rgb8();

        // Define font scale (size)
        let font_scale = PxScale::from(16.0);

        for (idx, detection) in detections.iter().enumerate() {
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

            // Normalize to [0, 1] and then to [-1, 1] range
            input_tensor[[0, 0, y, x]] = (r as f32 / 255.0 - 0.5) / 0.5;
            input_tensor[[0, 1, y, x]] = (g as f32 / 255.0 - 0.5) / 0.5;
            input_tensor[[0, 2, y, x]] = (b as f32 / 255.0 - 0.5) / 0.5;
        }

        Ok(input_tensor)
    }

    fn postprocess(
        &self,
        output: <PaddleDet as Model>::Output,
        extra: Self::Extra,
    ) -> Result<Self::Output, FerrpdfError> {
        let detections = self.db_postprocess(&output, &extra);
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
                input_name => TensorRef::from_array_view(&input).context(TensorSnafu{stage: "input"})?
            ])
            .context(InferenceSnafu {})?;

        // Extract the output tensor and convert to ndarray
        let tensor = output
            .get(output_name)
            .context(NotFoundOutputSnafu { output_name })?
            .try_extract_array::<f32>()
            .context(TensorSnafu { stage: "extract" })?;

        // Convert to owned array with proper shape
        let shape = tensor.shape();
        let output_array = if shape.len() == 4 {
            // Remove batch dimension if present and convert to 3D
            let slice_3d = tensor.slice(ndarray::s![0, .., .., ..]);
            Array3::from_shape_vec(
                (
                    slice_3d.shape()[0],
                    slice_3d.shape()[1],
                    slice_3d.shape()[2],
                ),
                slice_3d.iter().cloned().collect(),
            )
            .unwrap()
        } else if shape.len() == 3 {
            Array3::from_shape_vec(
                (shape[0], shape[1], shape[2]),
                tensor.iter().cloned().collect(),
            )
            .unwrap()
        } else {
            // Reshape to 3D if needed
            let total_elements = tensor.len();
            let h = shape[shape.len() - 2];
            let w = shape[shape.len() - 1];
            let c = total_elements / (h * w);
            Array3::from_shape_vec((c, h, w), tensor.iter().cloned().collect()).unwrap()
        };

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
        let session = PaddleDetSession::new(session_builder, model)?;

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
                text: None,
            },
            Layout {
                bbox: Bbox::new(glam::Vec2::new(50.0, 250.0), glam::Vec2::new(450.0, 400.0)),
                label: Label::Text,
                page_no: 0,
                bbox_id: 1,
                proba: 0.85,
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
}
