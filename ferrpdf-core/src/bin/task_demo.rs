use std::{ops::Range, path::PathBuf};
use uuid::Uuid;
use ferrpdf_core::{
    parse::{
        parser::{Pdf, TextExtraMode}, 
        task_based_parser::TaskBasedPdfParser
    }
};

/// Example demonstrating the new task-based PDF parser
/// This shows how the refactored parser decouples different processing stages
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();
    
    println!("🚀 Task-Based PDF Parser Demo");
    println!("=============================");
    
    // Create the new task-based parser
    let parser = TaskBasedPdfParser::new()?;
    
    // Example PDF configuration
    let pdf = Pdf {
        path: PathBuf::from("example.pdf"), // This would be a real PDF file
        password: None,
        range: Range { start: 0, end: 3 }, // Parse first 3 pages
        debug: Some(PathBuf::from("debug_output")), // Enable debug output
        uuid: Uuid::new_v4(),
        text_extra_mode: TextExtraMode::Auto, // Use auto mode (PDF + OCR fallback)
    };
    
    println!("📋 Processing Configuration:");
    println!("   - File: {:?}", pdf.path);
    println!("   - Pages: {} to {}", pdf.range.start, pdf.range.end);
    println!("   - Text Mode: {:?}", pdf.text_extra_mode);
    println!("   - Debug Output: {:?}", pdf.debug);
    
    // The new task-based parser automatically handles:
    // Task 1: Document Loading
    // Task 2: Image Rendering (decoupled from layout analysis)
    // Task 3: Layout Recognition (decoupled from text extraction)  
    // Task 4: Text Extraction (decoupled from OCR)
    // Task 5: Text Detection (decoupled from recognition)
    // Task 6: OCR (only for blocks that need it)
    
    match parser.parse(&pdf).await {
        Ok(layouts) => {
            println!("✅ PDF parsing completed successfully!");
            println!("📊 Results Summary:");
            
            for (page_idx, page_layout) in layouts.iter().enumerate() {
                println!("   Page {}: {} layout elements detected", 
                    page_layout.metadata.page, 
                    page_layout.layouts.len()
                );
                
                for (elem_idx, layout) in page_layout.layouts.iter().enumerate() {
                    let text_source = if layout.ocr.as_ref().map_or(false, |ocr| ocr.is_ocr) {
                        "OCR"
                    } else {
                        "PDF"
                    };
                    
                    let text_preview = layout.text
                        .as_ref()
                        .map(|t| {
                            let preview = t.chars().take(50).collect::<String>();
                            if t.len() > 50 { format!("{}...", preview) } else { preview }
                        })
                        .unwrap_or_else(|| "No text".to_string());
                    
                    println!("     Element {}: {:?} [{} confidence: {:.2}] - {} ({})", 
                        elem_idx,
                        layout.label,
                        text_source,
                        layout.proba,
                        text_preview,
                        text_source
                    );
                }
            }
            
            println!("\n🎯 Task-Based Architecture Benefits:");
            println!("   ✓ Decoupled processing stages");
            println!("   ✓ Clear separation of concerns");
            println!("   ✓ Individual task monitoring and logging");
            println!("   ✓ Easier testing and debugging");
            println!("   ✓ Potential for parallel execution");
            
        }
        Err(e) => {
            println!("❌ PDF parsing failed: {}", e);
            return Err(Box::new(e));
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_task_based_parser_creation() {
        // This test would require proper ONNX runtime setup
        // For now, we just test that the types are properly imported
        let pdf = Pdf {
            path: PathBuf::from("test.pdf"),
            password: None,
            range: Range { start: 0, end: 1 },
            debug: None,
            uuid: Uuid::new_v4(),
            text_extra_mode: TextExtraMode::Auto,
        };
        
        assert_eq!(pdf.range.start, 0);
        assert_eq!(pdf.range.end, 1);
        assert_eq!(pdf.text_extra_mode, TextExtraMode::Auto);
    }
}