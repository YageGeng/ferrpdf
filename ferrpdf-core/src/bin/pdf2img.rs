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

    for (index, page) in document.pages().iter().take(1).enumerate() {
        let output_path = images_dir.join(format!("test-page-{}.jpg", index));

        let width = page.width().value;
        let height = page.height().value;

        let scale = f32::min(1024.0 / width, 1024.0 / height);

        let render_config = PdfRenderConfig::new()
            .scale_page_by_factor(scale)
            .rotate_if_landscape(PdfPageRenderRotation::Degrees90, true);

        println!("Annotaion:");
        for anno in page.annotations().iter() {
            if let Some(link) = anno.as_link_annotation() {
                let bbox = link.bounds().unwrap();
                println!("Ref: {}", page.text()?.inside_rect(bbox));
            }
        }

        println!("Content: {}", page.objects().len());
        for (idx, obj) in page.objects().iter().enumerate() {
            let bbox = obj.bounds().unwrap();
            println!("{idx}:{}", page.text()?.inside_rect(bbox))
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
    // let pdf_path =
    //     "/home/isbest/Downloads/DocBank: A Benchmark Dataset for Document Layout Analysis.pdf";

    // export_pdf_to_jpegs(&pdf_path, None).unwrap_or_else(|e| {
    //     eprintln!("处理 PDF 时出错: {:?}", e);
    // });

    // println!("\n\n=======================\n");
    let pdf_path =
        "/home/isbest/Downloads/DocBank: A Benchmark Dataset for Document Layout Analysis.pdf";

    export_pdf_to_jpegs(&pdf_path, None).unwrap_or_else(|e| {
        eprintln!("处理 PDF 时出错: {:?}", e);
    });
}
