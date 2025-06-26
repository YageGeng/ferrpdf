use std::{
    ops::{Deref, Range},
    path::{Path, PathBuf},
    sync::Arc,
};

use derive_builder::Builder;
use futures::future;
use glam::Vec2;
use image::DynamicImage;
use ort::session::RunOptions;
use pdfium_render::prelude::{PdfDocument, PdfPageRenderRotation, PdfRenderConfig, Pdfium};
use plsfix::fix_text;
use snafu::ResultExt;
use tokio::sync::Mutex;
use tracing::*;

use crate::{
    analysis::labels::Label,
    consts::*,
    error::{EnvNotFoundSnafu, FerrpdfError, PdfiumSnafu, RunConfigSnafu},
    inference::{
        model::{OnnxSession, session_builder},
        paddle::{
            detect::{PaddleDet, PaddleDetSession, TextDetection},
            recognize::{PaddleRec, PaddleRecSession},
        },
        yolov12::{DocMeta, PdfLayouts, YoloSession, Yolov12},
    },
    layout::element::{Layout, Ocr},
};

pub struct PdfParser {
    pub pdfium: Pdfium,
    pub layout: Arc<Mutex<YoloSession<Yolov12>>>,
    pub detect: Arc<Mutex<PaddleDetSession<PaddleDet>>>,
    pub recognizer: Arc<Mutex<PaddleRecSession<PaddleRec>>>,
    pub config: ParserConfig,
}

#[derive(Debug, Builder)]
pub struct ParserConfig {
    pub auto_clean_text: bool,
    pub coverage_threshold: f32,
}

impl Default for ParserConfig {
    fn default() -> Self {
        Self {
            auto_clean_text: true,
            coverage_threshold: 0.002,
        }
    }
}

pub struct Pdf {
    pub path: PathBuf,
    pub password: Option<String>,
    pub range: Range<u16>,
    pub debug: Option<PathBuf>,
}

pub struct PdfPage {
    pub metadata: DocMeta,
    pub image: DynamicImage,
}

pub struct LackTextBlock<'a> {
    pub page_no: usize,
    pub layout: Arc<Mutex<&'a mut Layout>>,
    pub scale: f32,
}

impl PdfParser {
    pub fn new() -> Result<Self, FerrpdfError> {
        // Get pdfium library path
        let pdfium_lib_path =
            std::env::var(PDFIUM_LIB_PATH_ENV_NAME).context(EnvNotFoundSnafu {
                name: PDFIUM_LIB_PATH_ENV_NAME,
            })?;

        // Create pdfium instance
        let pdfium = Pdfium::new(
            Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path(
                &pdfium_lib_path,
            ))
            .context(PdfiumSnafu {
                stage: "load-dyn-lib",
            })?,
        );

        // Create LayoutSession
        let layout = YoloSession::new(session_builder()?, Yolov12::new())?;
        let detect = PaddleDetSession::new(session_builder()?, PaddleDet::new())?;
        let recognizer = PaddleRecSession::new(session_builder()?, PaddleRec::new())?;

        Ok(Self {
            pdfium,
            layout: Arc::new(Mutex::new(layout)),
            detect: Arc::new(Mutex::new(detect)),
            recognizer: Arc::new(Mutex::new(recognizer)),
            config: ParserConfig::default(),
        })
    }

    pub async fn parse(&self, pdf: &Pdf) -> Result<Vec<PdfLayouts>, FerrpdfError> {
        let document = self.load_pdf(pdf)?;

        let pdf_images = self.render(&document, &pdf.range)?;

        let mut layouts = self.layout_analyze(&pdf_images).await?;

        if let Some(path) = pdf.debug.as_ref() {
            Self::draw_layout_bbox(path, &layouts, &pdf_images)?;
        }

        let lack_text_block = self.extra_pdf_text(&document, &mut layouts)?;
        if !lack_text_block.is_empty() {
            self.detect_and_ocr(&lack_text_block, &pdf_images, pdf)
                .await?;
        }

        Ok(layouts)
    }

    fn draw_layout_bbox<P: AsRef<Path>>(
        output: P,
        layout: &[PdfLayouts],
        images: &[PdfPage],
    ) -> Result<(), FerrpdfError> {
        let path = output.as_ref();
        for (idx, PdfLayouts { layouts, metadata }) in layout.iter().enumerate() {
            YoloSession::draw(
                path.join(format!("layout-{}.png", metadata.page)),
                metadata,
                layouts,
                &images[idx].image,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::type_complexity)]
    fn draw_detection_bbox<P: AsRef<Path>>(
        output: P,
        layout: &[(usize, usize, Result<Vec<TextDetection>, FerrpdfError>)],
        images: &[PdfPage],
    ) -> Result<(), FerrpdfError> {
        let path = output.as_ref();
        for (idx, page_no, detections) in layout.iter().flat_map(|(idx, page_no, result)| {
            result
                .as_ref()
                .ok()
                .map(|detections| (idx, page_no, detections))
        }) {
            let pdf_image = &images[*page_no];

            PaddleDetSession::draw(
                path.join(format!(
                    "detections-{}-{}.png",
                    pdf_image.metadata.page, idx
                )),
                detections,
                &pdf_image.image,
            )?;
        }
        Ok(())
    }

    // load pdf
    fn load_pdf<'a, 'b: 'a>(&'a self, pdf: &'b Pdf) -> Result<PdfDocument<'a>, FerrpdfError> {
        let document = self
            .pdfium
            .load_pdf_from_file(&pdf.path, pdf.password.as_deref())
            .context(PdfiumSnafu {
                stage: "load-pdf-by-path",
            })?;

        let page_number = document.pages().len();

        let range = pdf.range.clone();
        if page_number < range.start {
            return Err(FerrpdfError::ParserErr {
                stage: "parse-pdf".to_string(),
                path: pdf.path.to_string_lossy().to_string(),
                message: "Page number is less than the start of the range".to_string(),
            });
        }

        Ok(document)
    }

    // render pdf to jpeg
    fn render(
        &self,
        document: &PdfDocument<'_>,
        range: &Range<u16>,
    ) -> Result<Vec<PdfPage>, FerrpdfError> {
        let page_number = document.pages().len();

        let pdf_images = document
            .pages()
            .iter()
            .enumerate()
            .skip(range.start as _)
            .take(usize::min(range.len(), (page_number - range.start) as _))
            .flat_map(|(idx, page)| {
                let width = page.width().value;
                let height = page.height().value;

                let target_width = YOLOV12_INPUT_IMAGE_WIDTH as f32;
                let target_height = YOLOV12_INPUT_IMAGE_HEIGHT as f32;
                let target_size = Vec2::new(target_width, target_height);
                let pdf_size = Vec2::new(width, height);

                let scale = f32::min(target_width / width, target_height / height);

                let render_config = PdfRenderConfig::new()
                    .scale_page_by_factor(scale)
                    .rotate_if_landscape(PdfPageRenderRotation::Degrees90, true);

                page.render_with_config(&render_config)
                    .map(|img| img.as_image())
                    .map(|pdf_image| PdfPage {
                        metadata: DocMeta {
                            scale,
                            pdf_size,
                            page: idx,
                            image_size: target_size,
                        },
                        image: pdf_image,
                    })
            })
            .collect::<Vec<_>>();

        Ok(pdf_images)
    }

    async fn layout_analyze(
        &self,
        pdf_images: &[PdfPage],
    ) -> Result<Vec<PdfLayouts>, FerrpdfError> {
        let run_option = RunOptions::new().context(RunConfigSnafu {
            stage: "run-options",
        })?;

        let layout = Arc::clone(&self.layout);

        let layout_tasks = pdf_images
            .iter()
            .map(|pdf_page| async {
                let mut layout = layout.lock().await;
                layout
                    .run_async(&pdf_page.image, pdf_page.metadata, &run_option)
                    .await
            })
            .collect::<Vec<_>>();

        let mut result = future::join_all(layout_tasks)
            .await
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        // sort by page index
        result.sort_by(|a, b| a.metadata.page.cmp(&b.metadata.page));

        Ok(result)
    }

    fn extra_pdf_text<'a>(
        &self,
        document: &PdfDocument<'_>,
        layouts: &'a mut [PdfLayouts],
    ) -> Result<Vec<LackTextBlock<'a>>, FerrpdfError> {
        let mut need_ocr_layouts = Vec::new();
        for pdf_layouts in layouts.iter_mut() {
            let metadata = pdf_layouts.metadata;
            let page = document
                .pages()
                .get(metadata.page as u16)
                .context(PdfiumSnafu { stage: "get-page" })?;

            for layout in pdf_layouts.layouts.iter_mut() {
                let page_text = page.text().context(PdfiumSnafu { stage: "text" })?;
                let pdf_bbox = layout.bbox.to_pdf_rect(metadata.pdf_size.y);
                let mut text = page_text.inside_rect(pdf_bbox);

                if self.config.auto_clean_text {
                    text = fix_text(&text, None);
                }

                let text_len = text.len();
                let area = layout.bbox.area();

                // update text coverage
                let text_coverage = text_len as f32 / area;

                layout.ocr = Some(Ocr {
                    text_coverage,
                    is_ocr: false,
                });

                layout.text = Some(text);

                if matches!(layout.label, Label::Picture)
                    || text_coverage <= self.config.coverage_threshold
                {
                    need_ocr_layouts.push(LackTextBlock {
                        layout: Arc::new(Mutex::new(layout)),
                        page_no: metadata.page,
                        scale: metadata.scale,
                    });
                }
            }
        }

        Ok(need_ocr_layouts)
    }

    async fn detect_and_ocr(
        &self,
        lack_text_blocks: &[LackTextBlock<'_>],
        pdf_images: &[PdfPage],
        pdf: &Pdf,
    ) -> Result<(), FerrpdfError> {
        let run_options = RunOptions::new().context(RunConfigSnafu {
            stage: "run-options",
        })?;

        let run_options = Arc::new(run_options);

        let detect_tasks = lack_text_blocks
            .iter()
            .enumerate()
            .map(|(idx, lack_block)| (idx, lack_block, Arc::clone(&run_options)))
            .map(|(idx, lack_block, run_options)| async move {
                let mut detect_session = self.detect.lock().await;
                let page_no = lack_block.page_no;
                let image = &pdf_images[page_no].image;

                let layout = lack_block.layout.lock().await;
                let detections = detect_session
                    .detect_text_in_layout_async(
                        image,
                        layout.deref(),
                        lack_block.scale,
                        &run_options,
                    )
                    .await;
                (idx, page_no, detections)
            })
            .collect::<Vec<_>>();

        let text_detections = future::join_all(detect_tasks).await;

        if let Some(path) = pdf.debug.as_ref() {
            Self::draw_detection_bbox(path, &text_detections, pdf_images)?;
        }

        let recognize_tasks = text_detections
            .into_iter()
            .flat_map(|(idx, page_no, detect_result)| {
                if let Err(err) = detect_result.as_ref() {
                    error!("Error detecting text in layout: {}", err);
                }

                detect_result
                    .ok()
                    .map(|detections| (idx, page_no, detections, Arc::clone(&run_options)))
            })
            .map(|(idx, page_no, text_detections, run_options)| async move {
                let image = &pdf_images[page_no].image;
                let image = Arc::new(image);

                let recognize_tasks = text_detections
                    .into_iter()
                    .map(|text_detection| {
                        (text_detection, Arc::clone(&image), Arc::clone(&run_options))
                    })
                    .map(|(text_detection, image, run_options)| async move {
                        let mut ocr_session = self.recognizer.lock().await;
                        ocr_session
                            .recognize_text_region_async(&image, text_detection.bbox, &run_options)
                            .await
                    })
                    .collect::<Vec<_>>();
                (idx, future::join_all(recognize_tasks).await)
            })
            .collect::<Vec<_>>();

        for (idx, recognition) in future::join_all(recognize_tasks).await.into_iter() {
            let mut text =
                recognition
                    .into_iter()
                    .flatten()
                    .fold(String::new(), |mut acc, text| {
                        acc.push_str(&text);
                        acc.push('\n');
                        acc
                    });

            if self.config.auto_clean_text {
                text = fix_text(&text, None);
            }

            let mut layout = lack_text_blocks[idx].layout.lock().await;

            layout.text = Some(text);
            if let Some(ocr) = layout.ocr.as_mut() {
                ocr.is_ocr = true;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_control_chars() {
        // STX
        assert!('\u{0002}'.is_ascii_control());
        assert!('\u{0002}'.is_control());
        assert!(!'\u{0002}'.is_ascii_alphanumeric());
        assert!(!'\u{0002}'.is_ascii_whitespace());
        assert!(!'\u{0002}'.is_whitespace());

        // space
        assert!(' '.is_whitespace());
        assert!(!' '.is_control());
        // \n
        assert!('\n'.is_whitespace());
        assert!('\n'.is_control());
        assert!('\n'.is_ascii_control());
        // \r
        assert!('\r'.is_whitespace());
        assert!('\r'.is_control());
        // \t
        assert!('\t'.is_whitespace());
        assert!('\t'.is_control());
    }

    #[tokio::test]
    async fn test_parse() -> Result<(), FerrpdfError> {
        let parser = PdfParser::new().unwrap();
        let pdf = Pdf{path:"/home/isbest/Downloads/Docs/DocBank: A Benchmark Dataset for Document Layout Analysis.pdf".into(),password:None,range:0..1, debug: None };

        let _ = parser.parse(&pdf).await;

        Ok(())
    }
}
