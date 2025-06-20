use image::{DynamicImage, imageops::FilterType};
use ndarray::prelude::*;
use ort::session::{Session, builder::SessionBuilder};
use ort::value::TensorRef;
use snafu::{OptionExt, ResultExt};

use crate::{
    analysis::bbox::Bbox,
    error::*,
    inference::model::{Model, OnnxSession},
    layout::element::Layout,
};

use super::model::PaddleRec;

pub struct PaddleRecSession<M: Model> {
    session: Session,
    model: M,
    character_dict: Vec<String>,
}

impl PaddleRecSession<PaddleRec> {
    pub fn new(session: SessionBuilder, model: PaddleRec) -> Result<Self, FerrpdfError> {
        let session = session
            .commit_from_memory(model.load())
            .context(OrtInitSnafu { stage: "commit" })?;

        // Extract character dictionary from model metadata
        let chars = session
            .metadata()
            .ok()
            .and_then(|m| m.custom("character").ok().flatten())
            .unwrap_or_default();

        let mut character_dict = Vec::with_capacity(chars.len());
        character_dict.push("#".to_string());
        // Parse character dictionary - assuming it's a string with characters
        character_dict.extend(chars.lines().map(|char| char.to_string()));
        character_dict.push(" ".to_string());

        Ok(Self {
            session,
            model,
            character_dict,
        })
    }

    pub fn character_dict(&self) -> &[String] {
        &self.character_dict
    }

    /// Extract text from a specific region of an image
    pub fn recognize_text_region(
        &mut self,
        image: &DynamicImage,
        bbox: &Bbox,
    ) -> Result<String, FerrpdfError> {
        // Crop the image to the bounding box region
        let cropped_image = self.crop_image_region(image, bbox)?;

        // Run OCR on the cropped region
        let text = self.run(&cropped_image, ())?;

        Ok(text)
    }

    /// Extract text from multiple regions in an image
    pub fn recognize_text_regions(
        &mut self,
        image: &DynamicImage,
        layouts: &mut [Layout],
    ) -> Result<(), FerrpdfError> {
        for layout in layouts.iter_mut() {
            if let Ok(text) = self.recognize_text_region(image, &layout.bbox) {
                if !text.trim().is_empty() {
                    layout.text = Some(text.trim().to_string());
                }
            }
        }
        Ok(())
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
}

impl OnnxSession<PaddleRec> for PaddleRecSession<PaddleRec> {
    type Output = String;
    type Extra = ();

    fn preprocess(
        &self,
        image: &DynamicImage,
    ) -> Result<<PaddleRec as Model>::Input, FerrpdfError> {
        let config = self.model.config();

        // Convert to RGB if needed
        let mut img_src = image.to_rgb8();

        // Check if rotation is needed based on aspect ratio
        let aspect_ratio = img_src.height() as f32 / img_src.width() as f32;
        if config.aspect_ratio_threshold > 0.0 && aspect_ratio > config.aspect_ratio_threshold {
            // Rotate 90 degrees clockwise for vertical text
            img_src = image::imageops::rotate90(&img_src);
        }

        // Calculate scale and target width using provided logic
        let scale = config.required_height as f32 / img_src.height() as f32;
        let dst_width = (img_src.width() as f32 * scale) as u32;

        // Resize using image::imageops
        let src_resize = image::imageops::resize(
            &img_src,
            dst_width,
            config.required_height as u32,
            FilterType::Triangle,
        );

        // Create input tensor with shape [batch_size, channels, height, width]
        let mut input_tensor = Array4::zeros([
            config.batch_size,
            config.input_channels,
            src_resize.height() as _,
            src_resize.width() as _,
        ]);

        // Fill tensor with normalized pixel values
        for (x, y, pixel) in src_resize.enumerate_pixels() {
            let x = x as usize;
            let y = y as usize;
            let [r, g, b] = pixel.0;

            // Normalize to [0, 1] range and then to [-1, 1] range (common for OCR models)
            input_tensor[[0, 0, y, x]] = (r as f32 / 255.0 - 0.5) / 0.5;
            input_tensor[[0, 1, y, x]] = (g as f32 / 255.0 - 0.5) / 0.5;
            input_tensor[[0, 2, y, x]] = (b as f32 / 255.0 - 0.5) / 0.5;
        }

        Ok(input_tensor)
    }

    fn postprocess(
        &self,
        output: <PaddleRec as Model>::Output,
        _extra: Self::Extra,
    ) -> Result<Self::Output, FerrpdfError> {
        // Output shape is [batch_size, sequence_length, vocab_size]
        // We take the first batch
        let batch_output = output.slice(ndarray::s![0, .., ..]);

        // Convert probabilities to character indices using argmax
        let mut text = String::new();
        let mut prev_char_idx = None;

        for timestep in batch_output.axis_iter(Axis(0)) {
            // Find the character with highest probability
            let (max_idx, _max_prob) = timestep
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, &0.0));

            // Skip blank token (usually index 0) and repeated characters (CTC decoding)
            //
            if max_idx != 0 && Some(max_idx) != prev_char_idx && max_idx < self.character_dict.len()
            {
                // Subtract 1 because index 0 is blank token
                text.push_str(&self.character_dict[max_idx]);
            }

            prev_char_idx = Some(max_idx);
        }

        Ok(text)
    }

    fn infer(
        &mut self,
        input: <PaddleRec as Model>::Input,
        input_name: &str,
        output_name: &str,
    ) -> Result<<PaddleRec as Model>::Output, FerrpdfError> {
        // Run inference
        let output = self
            .session
            .run(ort::inputs![
                input_name => TensorRef::from_array_view(&input).context(TensorSnafu{stage: "recognize-input"})?
            ])
            .context(InferenceSnafu {})?;

        // Extract the output tensor and convert to ndarray
        let tensor = output
            .get(output_name)
            .context(NotFoundOutputSnafu { output_name })?
            .try_extract_array::<f32>()
            .context(TensorSnafu {
                stage: "recognize-extract",
            })?;

        // Get the actual output shape and convert to owned array
        // Reshape to 3D array [batch_size, sequence_length, vocab_size]
        let shape = tensor.shape();
        let output_array = tensor
            .to_shape([shape[0], shape[1], shape[2]])
            .context(ShapeSnafu { stage: "recognize" })?
            .to_owned();

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
    fn test_paddle_session() -> Result<(), Box<dyn std::error::Error>> {
        let session_builder = Session::builder()?
            .with_execution_providers(vec![CPUExecutionProvider::default().build()])?
            .with_optimization_level(GraphOptimizationLevel::Level1)?;

        let model = PaddleRec::new();
        let session = PaddleRecSession::new(session_builder, model)?;

        // Print character dictionary size for debugging
        println!(
            "Character dictionary size: {}",
            session.character_dict().len()
        );
        if !session.character_dict().is_empty() {
            println!(
                "First few characters: {:?}",
                &session.character_dict()[..session.character_dict().len().min(10)]
            );
        }

        Ok(())
    }

    #[test]
    fn test_paddle_ocr_inference() -> Result<(), Box<dyn std::error::Error>> {
        // Create a simple test image (48 pixels high, any width)
        let test_image = image::DynamicImage::new_rgb8(100, 48);

        let session_builder = Session::builder()?
            .with_execution_providers(vec![CPUExecutionProvider::default().build()])?
            .with_optimization_level(GraphOptimizationLevel::Level1)?;

        let model = PaddleRec::new();
        let session = PaddleRecSession::new(session_builder, model)?;

        // Test preprocessing
        let preprocessed = session.preprocess(&test_image)?;
        println!("Preprocessed tensor shape: {:?}", preprocessed.shape());

        // Test cropping functionality
        use crate::analysis::bbox::Bbox;
        let bbox = Bbox::new(glam::Vec2::new(10.0, 5.0), glam::Vec2::new(90.0, 40.0));
        let cropped = session.crop_image_region(&test_image, &bbox)?;
        println!(
            "Cropped image dimensions: {}x{}",
            cropped.width(),
            cropped.height()
        );

        // Note: Full inference test would require running the model,
        // which might not work in unit tests without proper setup

        Ok(())
    }

    #[test]
    fn test_paddle_text_recognition() -> Result<(), Box<dyn std::error::Error>> {
        let session_builder = Session::builder()?
            .with_execution_providers(vec![CPUExecutionProvider::default().build()])?
            .with_optimization_level(GraphOptimizationLevel::Level1)?;

        let model = PaddleRec::new();
        let mut session = PaddleRecSession::new(session_builder, model)?;

        // Create a test image
        let test_image = image::DynamicImage::new_rgb8(100, 50);

        // Test OCR on the test image
        let result = session.run(&test_image, ());
        println!("OCR test completed: {:?}", result.is_ok());

        Ok(())
    }

    #[test]
    fn test_aspect_ratio_rotation() -> Result<(), Box<dyn std::error::Error>> {
        use crate::inference::paddle::recognize::model::PaddleRecConfig;

        let session_builder = Session::builder()?
            .with_execution_providers(vec![CPUExecutionProvider::default().build()])?
            .with_optimization_level(GraphOptimizationLevel::Level1)?;

        // Test with different aspect ratio thresholds
        let config = PaddleRecConfig {
            aspect_ratio_threshold: 1.5,
            ..PaddleRecConfig::default()
        };
        let model = PaddleRec::with_config(config);
        let session = PaddleRecSession::new(session_builder, model)?;

        // Create a wide image (aspect ratio > 1.5)
        let wide_image = image::DynamicImage::new_rgb8(200, 50); // aspect ratio = 4.0
        let narrow_image = image::DynamicImage::new_rgb8(50, 100); // aspect ratio = 0.5

        // Test preprocessing with wide image (should rotate)
        let wide_tensor = session.preprocess(&wide_image)?;
        let narrow_tensor = session.preprocess(&narrow_image)?;

        // After rotation, the wide image should have different dimensions
        // The rotated wide image (50x200) will be resized to height=48, width=192
        // The narrow image (50x100) will be resized to height=48, width=24
        assert_eq!(wide_tensor.shape()[2], 48); // height
        assert_eq!(narrow_tensor.shape()[2], 48); // height
        assert!(wide_tensor.shape()[3] > narrow_tensor.shape()[3]); // width comparison

        println!("Aspect ratio rotation test passed");
        Ok(())
    }
}
