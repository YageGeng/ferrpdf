use std::{error::Error, path::Path, time::Instant};

use glam::Vec2;
use image::{GenericImageView, Rgb, imageops::FilterType};
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;
use ort::{
    execution_providers::CUDAExecutionProvider,
    session::{Session, SessionOutputs},
    value::TensorRef,
    *,
};

use crate::{
    analysis::{bbox::Bbox, labels::Label},
    layout::element::Layout,
};

fn non_max_suppression(mut layouts: Vec<Layout>, iou_threshold: f32) -> Vec<Layout> {
    layouts.sort_by(|a, b| b.proba.partial_cmp(&a.proba).unwrap());

    let mut result = Vec::new();

    while !layouts.is_empty() {
        let best = layouts.remove(0);
        result.push(best.clone());

        layouts.retain(|layout| {
            layout.label != best.label || best.bbox.iou(&layout.bbox) < iou_threshold
        });
    }

    result
}

pub fn init() -> Result<(), Box<dyn Error>> {
    ort::init()
        .with_execution_providers([CUDAExecutionProvider::default().build()])
        .commit()?;

    let now = Instant::now();
    let original_img = image::open(Path::new("images/test-page-6.jpg")).unwrap();
    let img = original_img.resize_exact(1024, 1024, FilterType::CatmullRom);

    // Create a copy for drawing bounding boxes
    let mut output_img = img.to_rgb8();

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

    let mut layouts = Vec::new();

    for prediction in output.axis_iter(ndarray::Axis(0)) {
        const CXYWH_OFFSET: usize = 4;
        let bbox = prediction.slice(ndarray::s![0..CXYWH_OFFSET]);

        let classes = prediction.slice(ndarray::s![
            CXYWH_OFFSET..CXYWH_OFFSET + Label::label_size()
        ]);
        let (max_prob_idx, &proba) = classes
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        if proba < 0.1 {
            continue;
        }

        let cx = bbox[0_usize];
        let cy = bbox[1_usize];
        let w = bbox[2_usize];
        let h = bbox[3_usize];

        // Create bbox from YOLO center-size format and clamp to image bounds
        let bbox = Bbox::from_center_size(Vec2::new(cx, cy), Vec2::new(w, h))
            .clamp(Vec2::new(0.0, 0.0), Vec2::new(1023.0, 1023.0));

        let layout = Layout {
            bbox,
            label: Label::from(max_prob_idx),
            page_no: 1,
            proba,
            text: None,
        };

        layouts.push(layout);
    }

    // Apply non-maximum suppression to remove duplicate detections
    let filtered_layouts = non_max_suppression(layouts, 0.5);

    println!("Found {} layouts after filtering", filtered_layouts.len());

    // Draw filtered layouts
    for layout in filtered_layouts {
        println!("LABEL {}, PROBA {:.3}", layout.label.name(), layout.proba);

        let x = layout.bbox.min.x as i32;
        let y = layout.bbox.min.y as i32;
        let width = (layout.bbox.max.x - layout.bbox.min.x) as u32;
        let height = (layout.bbox.max.y - layout.bbox.min.y) as u32;

        println!(
            "Drawing box: x={}, y={}, width={}, height={}",
            x, y, width, height
        );

        // Draw bounding box with thicker lines
        if width > 0 && height > 0 {
            let color = Rgb(layout.label.color());

            // Draw multiple rectangles to create thicker lines
            for offset in 0..3 {
                let thick_rect = Rect::at(x - offset, y - offset)
                    .of_size(width + (offset * 2) as u32, height + (offset * 2) as u32);
                draw_hollow_rect_mut(&mut output_img, thick_rect, color);
            }
        }
    }

    // Save the output image
    output_img.save("images/output.jpg").unwrap();
    println!("Output image saved to images/output.jpg");

    Ok(())
}
