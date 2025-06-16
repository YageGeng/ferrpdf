use std::env;
use std::path::{Path, PathBuf};

use pdfium_render::prelude::*;

fn export_pdf_to_jpegs<T: AsRef<Path>>(
    path: &T,
    password: Option<&str>,
) -> Result<(), PdfiumError> {
    let pdfium = Pdfium::new(
        Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path(
            &std::env::var("PDFIUM_DYNAMIC_LIB_PATH")
                .expect("PDFIUM_DYNAMIC_LIB_PATH env not found"),
        ))
        .expect("can't load pdfiurm bindings"),
    );

    let document = pdfium.load_pdf_from_file(path.as_ref(), password)?;

    let images_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("images");

    if !images_dir.exists() {
        std::fs::create_dir_all(&images_dir).expect("无法创建图片输出目录");
    }

    let render_config = PdfRenderConfig::new()
        .set_target_width(2048)
        .set_maximum_height(2048)
        .rotate_if_landscape(PdfPageRenderRotation::Degrees90, true);

    for (index, page) in document.pages().iter().enumerate() {
        let output_path = images_dir.join(format!("test-page-{}.jpg", index));

        if let Ok(text) = page.text() {
            println!("{}", text.all());
        }

        page.render_with_config(&render_config)?
            .as_image() // Renders this page to an image::DynamicImage...
            .into_rgb8() // ... then converts it to an image::Image...
            .save_with_format(&output_path, image::ImageFormat::Jpeg) // ... and saves it to a file.
            .map_err(|_| PdfiumError::ImageError)?;
    }

    Ok(())
}

pub fn main() {
    let pdf_path = "/Users/isbset/Downloads/物种分布模型-物种分布模型在预测海洋微体生物分布中的应用-Cong JY.pdf";

    export_pdf_to_jpegs(&pdf_path, None).unwrap_or_else(|e| {
        eprintln!("处理 PDF 时出错: {:?}", e);
    });
}
