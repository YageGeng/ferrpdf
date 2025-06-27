use std::path::PathBuf;

use clap::Parser;
use ferrpdf_core::{
    error::FerrpdfError,
    parse::parser::{Pdf, PdfParser},
};
use uuid::Uuid;

#[derive(Parser)]
#[command(name = "analyze")]
#[command(about = "PDF layout analysis tool")]
struct Args {
    #[arg(help = "Input PDF file path")]
    input: String,

    #[arg(
        long,
        short('p'),
        help = "Specify pages to parse using Rust range syntax: '0..5' for pages 0-4, '1..' for pages 1 onwards, '..5' for first 5 pages, '3' for single page. If not specified, all pages will be processed."
    )]
    page: Option<String>,

    #[arg(short, long, default_value = "images", help = "Output directory")]
    output: String,

    #[arg(
        short,
        long,
        default_value = "false",
        help = "Enable debug mode to save debug images"
    )]
    debug: bool,
}

#[tokio::main]
async fn main() -> Result<(), FerrpdfError> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .with_thread_ids(false)
        .with_env_filter("debug,ort=error")
        .compact()
        .init();

    let args = Args::parse();

    let parser = PdfParser::new()?;
    let range = parse_page_range(&args.page);

    let debug = if args.debug {
        Some(PathBuf::from(args.output))
    } else {
        None
    };

    let pdf = Pdf {
        path: PathBuf::from(args.input),
        uuid: Uuid::new_v4(),
        password: None,
        range,
        debug,
    };

    let layouts = parser.parse(&pdf).await?;

    println!("{}", serde_json::to_string(&layouts).unwrap());

    Ok(())
}

fn parse_page_range(page: &Option<String>) -> std::ops::Range<u16> {
    match page {
        Some(range_str) => {
            let parts: Vec<&str> = range_str.split("..").collect();
            match parts.as_slice() {
                [start, end] => {
                    let start = start.parse::<u16>().unwrap_or(0);
                    let end = end.parse::<u16>().unwrap_or(u16::MAX);
                    start..end
                }
                [start] => {
                    let start = start.parse::<u16>().ok().unwrap_or(u16::MAX);
                    0..start
                }
                _ => 0..u16::MAX,
            }
        }
        None => 0..u16::MAX,
    }
}
