use std::{error::Error, path::Path, time::Instant};

use image::{GenericImageView, imageops::FilterType};
use ort::{
    execution_providers::CUDAExecutionProvider,
    session::{Session, SessionOutputs},
    value::TensorRef,
    *,
};

pub fn init() -> Result<(), Box<dyn Error>> {
    ort::init()
        .with_execution_providers([CUDAExecutionProvider::default().build()])
        .commit()?;

    let now = Instant::now();
    let original_img = image::open(Path::new("images/test-page-0.jpg")).unwrap();
    let img = original_img.resize_exact(1024, 1024, FilterType::CatmullRom);

    let mut input = ndarray::Array::zeros((1, 3, 1024, 1024));

    for pixel in img.pixels() {
        let x = pixel.0 as _;
        let y = pixel.1 as _;
        let [r, g, b, _] = pixel.2.0;
        input[[0, 0, y, x]] = (r as f32) / 255.;
        input[[0, 1, y, x]] = (g as f32) / 255.;
        input[[0, 2, y, x]] = (b as f32) / 255.;
    }
    println!("IMAGE Const {}", now.elapsed().as_millis());

    let mut session = Session::builder()?.commit_from_file("models/yolov12s-doclaynet.onnx")?;

    // Run YOLOv12 inference
    let now = Instant::now();
    let outputs: SessionOutputs =
        session.run(inputs!["images" => TensorRef::from_array_view(&input)?])?;
    let output = outputs["output0"]
        .try_extract_array::<f32>()?
        .t()
        .into_owned();
    println!("Const {}", now.elapsed().as_millis());

    let output = output.slice(ndarray::s![.., .., 0]);

    for prediction in output.axis_iter(ndarray::Axis(0)) {
        const CXYWH_OFFSET: usize = 4;
        let bbox = prediction.slice(ndarray::s![0..CXYWH_OFFSET]);

        let classes = prediction.slice(ndarray::s![CXYWH_OFFSET..CXYWH_OFFSET + LABEL.len()]);
        let (max_prob_idx, &proba) = classes
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        if proba < 0.5 {
            continue;
        }

        println!("LABEL {}, PROBA {}", LABEL[max_prob_idx], proba);
        println!(
            "x {}, y {}, w {}, h {}",
            bbox[0_usize], bbox[1_usize], bbox[2_usize], bbox[3_usize]
        );
    }

    Ok(())
}

pub const LABEL: [&str; 11] = [
    "Caption",
    "Footnote",
    "Formula",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Section-header",
    "Table",
    "Text",
    "Title",
];
