use std::{
    collections::HashMap,
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
use uuid::Uuid;

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
}

#[derive(clap::ValueEnum, Debug, Default, Clone, Copy)]
pub enum TextExtraMode {
    #[default]
    /// Prefer PDF parsing, fallback to OCR if PDF parsing fails.
    Auto,
    /// Use OCR exclusively for text extraction.
    Ocr,
    /// Use PDF parsing exclusively for text extraction.
    Pdf,
}

impl TextExtraMode {
    pub fn need_ocr(&self) -> bool {
        matches!(self, TextExtraMode::Ocr | TextExtraMode::Auto)
    }

    pub fn need_pdf(&self) -> bool {
        matches!(self, TextExtraMode::Pdf | TextExtraMode::Auto)
    }
}

impl Default for ParserConfig {
    fn default() -> Self {
        Self {
            auto_clean_text: true,
        }
    }
}

pub struct Pdf {
    pub path: PathBuf,
    pub password: Option<String>,
    pub range: Range<u16>,
    pub debug: Option<PathBuf>,
    pub uuid: Uuid,
    pub text_extra_mode: TextExtraMode,
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
    #[tracing::instrument(skip_all)]
    pub fn new() -> Result<Self, FerrpdfError> {
        // Get pdfium library path
        info!("Fetching PDFium library path from environment variable.");
        let pdfium_lib_path =
            std::env::var(PDFIUM_LIB_PATH_ENV_NAME).context(EnvNotFoundSnafu {
                name: PDFIUM_LIB_PATH_ENV_NAME,
            })?;

        // Create pdfium instance
        info!("Creating PDFium instance.");
        let pdfium = Pdfium::new(
            Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path(
                &pdfium_lib_path,
            ))
            .context(PdfiumSnafu {
                stage: "load-dyn-lib",
            })?,
        );

        // Create LayoutSession
        info!("Initializing layout, detection, and recognition sessions.");
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

    #[tracing::instrument(skip_all,fields(pdf = %pdf.uuid))]
    pub async fn parse(&self, pdf: &Pdf) -> Result<Vec<PdfLayouts>, FerrpdfError> {
        info!("Loading PDF document.");
        let document = self.load_pdf(pdf)?;

        info!("Rendering PDF pages to images.");
        let pdf_images = self.render(&document, &pdf.range)?;

        info!("Analyzing layouts in rendered PDF images.");
        let mut layouts = self.layout_analyze(&pdf_images).await?;

        if let Some(path) = pdf.debug.as_ref() {
            info!("Drawing layout bounding boxes for debugging.");
            Self::draw_layout_bbox(path, &layouts, &pdf_images)?;
        }

        info!("Extracting text from PDF document.");
        let lack_text_block = self.extra_pdf_text(&document, &mut layouts, pdf.text_extra_mode)?;
        if !lack_text_block.is_empty() {
            info!("Performing text detection and OCR on missing text blocks.");
            self.detect_and_ocr(&lack_text_block, &pdf_images, pdf)
                .await?;
        }

        Ok(layouts)
    }

    #[tracing::instrument(skip_all)]
    fn draw_layout_bbox<P: AsRef<Path>>(
        output: P,
        layout: &[PdfLayouts],
        images: &[PdfPage],
    ) -> Result<(), FerrpdfError> {
        let path = output.as_ref();
        for (idx, PdfLayouts { layouts, metadata }) in layout.iter().enumerate() {
            debug!("layout-{}.png", metadata.page);
            YoloSession::draw(
                path.join(format!("layout-{}.png", metadata.page)),
                metadata,
                layouts,
                &images[idx].image,
            )?;
        }
        Ok(())
    }

    #[tracing::instrument(skip_all)]
    #[allow(clippy::type_complexity)]
    fn draw_detection_bbox<P: AsRef<Path>>(
        output: P,
        layout: &[(usize, usize, Result<Vec<TextDetection>, FerrpdfError>)],
        images: &[PdfPage],
    ) -> Result<(), FerrpdfError> {
        let grouped_detections = layout
            .iter()
            .filter_map(|(_, page_no, detections_result)| {
                detections_result
                    .as_ref()
                    .ok()
                    .map(|detections| (*page_no, detections.clone()))
            })
            .fold(
                HashMap::new(),
                |mut acc: HashMap<usize, Vec<TextDetection>>, (page_no, detections)| {
                    acc.entry(page_no).or_default().extend(detections);
                    acc
                },
            );

        let path = output.as_ref();
        for (page_no, detections) in grouped_detections.into_iter() {
            if detections.is_empty() {
                continue;
            }
            let pdf_image = &images[page_no];
            debug!("draw detections-{}", pdf_image.metadata.page);

            PaddleDetSession::draw(
                path.join(format!("detections-{}.png", pdf_image.metadata.page,)),
                &detections,
                &pdf_image.image,
            )?;
        }

        Ok(())
    }

    // load pdf
    #[tracing::instrument(skip_all, fields(result))]
    fn load_pdf<'a, 'b: 'a>(&'a self, pdf: &'b Pdf) -> Result<PdfDocument<'a>, FerrpdfError> {
        info!("Loading PDF from file: {:?}", pdf.path);
        let document = self
            .pdfium
            .load_pdf_from_file(&pdf.path, pdf.password.as_deref())
            .context(PdfiumSnafu {
                stage: "load-pdf-by-path",
            })?;

        let page_number = document.pages().len();

        let range = pdf.range.clone();
        if page_number < range.start {
            error!(
                "Page number is less than the start of the range for PDF: {:?}",
                pdf.path
            );
            return Err(FerrpdfError::ParserErr {
                stage: "parse-pdf".to_string(),
                path: pdf.path.to_string_lossy().to_string(),
                message: "Page number is less than the start of the range".to_string(),
            });
        }

        Ok(document)
    }

    // render pdf to jpeg
    #[tracing::instrument(skip_all, fields(result))]
    fn render(
        &self,
        document: &PdfDocument<'_>,
        range: &Range<u16>,
    ) -> Result<Vec<PdfPage>, FerrpdfError> {
        let page_number = document.pages().len();

        info!("Rendering pages within range: {:?}", range);
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

    #[tracing::instrument(skip_all, fields(result))]
    async fn layout_analyze(
        &self,
        pdf_images: &[PdfPage],
    ) -> Result<Vec<PdfLayouts>, FerrpdfError> {
        info!("Creating run options for layout analysis.");
        let run_option = RunOptions::new().context(RunConfigSnafu {
            stage: "run-options",
        })?;

        info!("Cloning layout session for analysis.");
        let layout = Arc::clone(&self.layout);

        let layout_tasks = pdf_images
            .iter()
            .map(|pdf_page| {
                async {
                    let mut layout = layout.lock().await;
                    layout
                        .run_async(&pdf_page.image, pdf_page.metadata, &run_option)
                        .in_current_span()
                        .await
                }
                .in_current_span()
            })
            .collect::<Vec<_>>();

        info!("Executing layout analysis tasks.");
        let mut result = future::join_all(layout_tasks)
            .await
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        // sort by page index
        info!("Sorting layout analysis results by page index.");
        result.sort_by(|a, b| a.metadata.page.cmp(&b.metadata.page));

        Ok(result)
    }

    #[tracing::instrument(skip_all, fields(result))]
    fn extra_pdf_text<'a>(
        &self,
        document: &PdfDocument<'_>,
        layouts: &'a mut [PdfLayouts],
        extra_mode: TextExtraMode,
    ) -> Result<Vec<LackTextBlock<'a>>, FerrpdfError> {
        info!("Identifying layouts requiring OCR.");
        let mut need_ocr_layouts = Vec::new();
        for pdf_layouts in layouts.iter_mut() {
            let metadata = pdf_layouts.metadata;
            let page = if extra_mode.need_pdf() {
                info!("Fetching page {} from PDF document.", metadata.page);
                let page = document
                    .pages()
                    .get(metadata.page as u16)
                    .context(PdfiumSnafu { stage: "get-page" })?;
                Some(page)
            } else {
                None
            };

            for layout in pdf_layouts.layouts.iter_mut() {
                let pdf_text = if extra_mode.need_pdf() {
                    let page_text = page
                        .as_ref()
                        .unwrap()
                        .text()
                        .context(PdfiumSnafu { stage: "text" })?;
                    let pdf_bbox = layout.bbox.to_pdf_rect(metadata.pdf_size.y);
                    let mut text = page_text.inside_rect(pdf_bbox);

                    if self.config.auto_clean_text {
                        text = fix_text(&text, None);
                    }
                    Some(text)
                } else {
                    None
                };

                layout.ocr = Some(Ocr { is_ocr: false });

                let is_text_empty = pdf_text.as_ref().is_none_or(|text| text.is_empty());
                layout.text = pdf_text;

                if extra_mode.need_ocr()
                    && (matches!(layout.label, Label::Picture) || is_text_empty)
                {
                    need_ocr_layouts.push(LackTextBlock {
                        layout: Arc::new(Mutex::new(layout)),
                        page_no: metadata.page,
                        scale: metadata.scale,
                    });
                }
            }
        }

        info!(
            "Completed text extraction. {} layouts require OCR.",
            need_ocr_layouts.len()
        );
        Ok(need_ocr_layouts)
    }

    #[tracing::instrument(skip_all, fields(result))]
    async fn detect_and_ocr(
        &self,
        lack_text_blocks: &[LackTextBlock<'_>],
        pdf_images: &[PdfPage],
        pdf: &Pdf,
    ) -> Result<(), FerrpdfError> {
        info!("Creating run options for text detection and OCR.");
        let run_options = RunOptions::new().context(RunConfigSnafu {
            stage: "run-options",
        })?;

        let run_options = Arc::new(run_options);

        info!(
            "Starting text detection tasks for {} blocks.",
            lack_text_blocks.len()
        );
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
                    .instrument(info_span!("detect"))
                    .await;
                (idx, page_no, detections)
            })
            .collect::<Vec<_>>();

        info!("Start text detection tasks.");
        let text_detections = future::join_all(detect_tasks).in_current_span().await;
        info!("Completed text detection tasks.");

        if let Some(path) = pdf.debug.as_ref() {
            info!("Drawing detection bounding boxes for debugging.");
            Self::draw_detection_bbox(path, &text_detections, pdf_images)?;
        }

        info!("Starting text recognition tasks.");
        let recognize_tasks = text_detections
            .into_iter()
            .flat_map(|(idx, page_no, detect_result)| {
                detect_result
                    .map(|detections| {
                        // skip empty detections
                        if detections.is_empty() {
                            None
                        } else {
                            Some((idx, page_no, detections, Arc::clone(&run_options)))
                        }
                    })
                    .map_err(|err| {
                        error!("Error detecting text in layout: {}", err);
                    })
                    .ok()
                    .flatten()
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
                            .instrument(info_span!("recognize"))
                            .await
                    })
                    .collect::<Vec<_>>();
                (
                    idx,
                    future::join_all(recognize_tasks).in_current_span().await,
                )
            })
            .collect::<Vec<_>>();

        info!("Start text recognition tasks.");
        for (idx, recognition) in future::join_all(recognize_tasks)
            .in_current_span()
            .await
            .into_iter()
        {
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
        info!("Completed text recognition tasks.");

        Ok(())
    }
}

impl Drop for PdfParser {
    fn drop(&mut self) {}
}

#[cfg(test)]
mod tests {
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
}
