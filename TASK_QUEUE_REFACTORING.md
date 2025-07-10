# Task Queue Parser Refactoring

## 概述 (Overview)

本次重构将原有的 `parser/parser.rs` 从单体式的顺序处理模式重构为基于任务队列的解耦架构，实现了以下几个核心任务的分离：

This refactoring transforms the original `parser/parser.rs` from a monolithic sequential processing model to a decoupled task queue architecture, achieving separation of the following core tasks:

1. **图片渲染 (Image Rendering)**
2. **布局识别 (Layout Recognition)**  
3. **文字提取 (Text Extraction)**
4. **文字检测 (Text Detection)**
5. **OCR识别 (OCR Recognition)**

## 架构设计 (Architecture Design)

### 原有架构问题 (Original Architecture Issues)

```rust
// 原有的单体式处理流程 (Original monolithic process)
pub async fn parse(&self, pdf: &Pdf) -> Result<Vec<PdfLayouts>, FerrpdfError> {
    let document = self.load_pdf(pdf)?;                    // 文档加载
    let pdf_images = self.render(&document, &pdf.range)?; // 图片渲染
    let mut layouts = self.layout_analyze(&pdf_images).await?; // 布局识别
    let lack_text_block = self.extra_pdf_text(&document, &mut layouts, pdf.text_extra_mode)?; // 文字提取
    if !lack_text_block.is_empty() {
        self.detect_and_ocr(&lack_text_block, &pdf_images, pdf).await?; // 检测和OCR
    }
    Ok(layouts)
}
```

**问题:**
- 紧耦合：各个处理阶段相互依赖，难以独立测试和调试
- 顺序执行：无法利用并行处理提升性能
- 缺乏监控：难以跟踪每个处理阶段的状态和性能
- 难以扩展：添加新的处理步骤需要修改核心逻辑

### 新的任务队列架构 (New Task Queue Architecture)

```rust
// 基于任务的解耦架构 (Task-based decoupled architecture)
pub async fn parse(&self, pdf: &Pdf) -> Result<Vec<PdfLayouts>, FerrpdfError> {
    // Task 1: 文档加载 (Document Loading)
    let document = self.execute_load_document_task(pdf).await?;
    
    // Task 2: 图片渲染 (Image Rendering) 
    let pdf_images = self.execute_image_rendering_task(&document, &pdf.range).await?;
    
    // Task 3: 布局识别 (Layout Recognition)
    let mut layouts = self.execute_layout_recognition_task(&pdf_images).await?;
    
    // Task 4: 文字提取 (Text Extraction)
    let lack_text_blocks = self.execute_text_extraction_task(&document, &mut layouts, pdf.text_extra_mode).await?;
    
    // Task 5 & 6: 文字检测和OCR (Text Detection and OCR)
    if !lack_text_blocks.is_empty() {
        self.execute_detection_and_ocr_tasks(&lack_text_blocks, &pdf_images, pdf).await?;
    }
    
    Ok(layouts)
}
```

## 核心组件 (Core Components)

### 1. 任务类型定义 (Task Type Definitions)

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskType {
    ImageRendering,     // 图片渲染
    LayoutRecognition,  // 布局识别
    TextExtraction,     // 文字提取
    TextDetection,      // 文字检测
    Ocr,               // OCR识别
}
```

### 2. 任务结果跟踪 (Task Result Tracking)

```rust
#[derive(Debug)]
pub struct TaskResult<T> {
    pub task_type: TaskType,
    pub task_id: Uuid,
    pub result: Result<T, FerrpdfError>,
    pub duration: std::time::Duration,
}
```

### 3. 增强的PDF页面结构 (Enhanced PDF Page Structure)

```rust
#[derive(Debug)]
pub struct EnhancedPdfPage {
    pub metadata: DocMeta,
    pub image: DynamicImage,
    pub page_index: usize,  // 新增页面索引便于跟踪
}
```

## 实现细节 (Implementation Details)

### 任务执行器 (Task Executors)

每个任务都有独立的执行器，包含完整的日志记录和错误处理：

```rust
async fn execute_image_rendering_task(
    &self,
    document: &PdfDocument<'_>,
    range: &Range<u16>,
) -> Result<Vec<EnhancedPdfPage>, FerrpdfError> {
    let start = std::time::Instant::now();
    let task_id = Uuid::new_v4();
    
    info!("Executing Task 2: Image Rendering (ID: {})", task_id);
    
    let result = self.render_pdf_to_images(document, range);
    let duration = start.elapsed();
    
    match &result {
        Ok(pages) => info!("Task 2 completed successfully in {:?}, rendered {} pages", duration, pages.len()),
        Err(e) => error!("Task 2 failed after {:?}: {}", duration, e),
    }
    
    result
}
```

### 解耦的处理函数 (Decoupled Processing Functions)

原有的处理逻辑被提取为独立的函数，便于测试和重用：

```rust
// 图片渲染 (Image Rendering)
fn render_pdf_to_images(&self, document: &PdfDocument<'_>, range: &Range<u16>) -> Result<Vec<EnhancedPdfPage>, FerrpdfError>

// 布局分析 (Layout Analysis)
async fn analyze_page_layouts(&self, pdf_images: &[EnhancedPdfPage]) -> Result<Vec<PdfLayouts>, FerrpdfError>

// 文字提取 (Text Extraction) 
fn extract_text_from_pdf(&self, document: &PdfDocument<'_>, layouts: &mut [PdfLayouts], extra_mode: TextExtraMode) -> Result<Vec<LackTextBlock>, FerrpdfError>

// 检测和OCR (Detection and OCR)
async fn perform_detection_and_ocr(&self, lack_text_blocks: &[LackTextBlock], pdf_images: &[EnhancedPdfPage], pdf: &Pdf) -> Result<(), FerrpdfError>
```

## 优势 (Advantages)

### 1. 关注点分离 (Separation of Concerns)
- 每个任务专注于单一职责
- 易于理解和维护
- 便于单元测试

### 2. 可观测性 (Observability) 
- 每个任务都有独立的日志记录
- 性能监控和错误跟踪
- 任务执行时间统计

### 3. 可扩展性 (Extensibility)
- 容易添加新的处理任务
- 可以插入中间处理步骤
- 支持条件执行

### 4. 并行化潜力 (Parallelization Potential)
- 为未来的并行执行做好准备
- 可以根据任务依赖关系优化执行顺序

### 5. 向后兼容 (Backward Compatibility)
- 保持相同的公共API
- 现有代码无需修改
- 渐进式迁移

## 使用示例 (Usage Example)

```rust
use ferrpdf_core::parse::task_based_parser::TaskBasedPdfParser;

// 创建新的任务式解析器 (Create new task-based parser)
let parser = TaskBasedPdfParser::new()?;

// 解析PDF - API保持不变 (Parse PDF - API remains unchanged)  
let layouts = parser.parse(&pdf).await?;

// 日志输出显示各个任务的执行情况 (Logs show execution of each task)
// INFO  Executing Task 1: Load PDF Document (ID: ...)
// INFO  Executing Task 2: Image Rendering (ID: ...)  
// INFO  Executing Task 3: Layout Recognition (ID: ...)
// INFO  Executing Task 4: Text Extraction (ID: ...)
// INFO  Executing Task 5 & 6: Text Detection and OCR (ID: ...)
```

## 未来扩展 (Future Extensions)

### 1. 完整的任务队列系统 (Full Task Queue System)
- 任务优先级管理
- 并发任务执行
- 任务重试机制
- 任务依赖管理

### 2. 插件化架构 (Plugin Architecture)
- 可插拔的处理模块
- 自定义任务类型
- 第三方扩展支持

### 3. 性能优化 (Performance Optimization)
- 智能任务调度
- 资源使用优化
- 缓存机制

### 4. 监控和调试 (Monitoring and Debugging)
- 实时任务状态监控
- 性能指标收集
- 详细的调试信息

## 测试策略 (Testing Strategy)

### 1. 单元测试 (Unit Tests)
每个任务执行器都可以独立测试：

```rust
#[tokio::test]
async fn test_image_rendering_task() {
    let parser = TaskBasedPdfParser::new()?;
    let document = load_test_document();
    let result = parser.execute_image_rendering_task(&document, &(0..1)).await;
    assert!(result.is_ok());
}
```

### 2. 集成测试 (Integration Tests)
测试完整的处理流水线：

```rust
#[tokio::test]
async fn test_complete_pipeline() {
    let parser = TaskBasedPdfParser::new()?;
    let pdf = create_test_pdf();
    let layouts = parser.parse(&pdf).await?;
    assert!(!layouts.is_empty());
}
```

### 3. 性能测试 (Performance Tests)
测量每个任务的执行时间和资源使用。

## 总结 (Summary)

这次重构成功地将原有的单体式PDF解析器转换为基于任务的解耦架构，实现了以下目标：

- ✅ 解耦了图片渲染、布局识别、文字提取、文字检测和OCR任务
- ✅ 提供了清晰的任务边界和职责分离
- ✅ 增强了可观测性和调试能力
- ✅ 保持了向后兼容性
- ✅ 为未来的并行化和扩展奠定了基础

新的架构不仅解决了原有代码的耦合问题，还为未来的功能扩展和性能优化提供了良好的基础。