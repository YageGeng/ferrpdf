#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Label {
    Caption,
    Footnote,
    Formula,
    ListItem,
    PageFooter,
    PageHeader,
    Picture,
    SectionHeader,
    Table,
    Text,
    Title,
}

impl Label {
    pub const fn name(&self) -> &str {
        match self {
            Label::Caption => "Caption",
            Label::Footnote => "Footnote",
            Label::Formula => "Formula",
            Label::ListItem => "List-Item",
            Label::PageFooter => "Page-Footer",
            Label::PageHeader => "Page-Header",
            Label::Picture => "Picture",
            Label::SectionHeader => "Section-Header",
            Label::Table => "Table",
            Label::Text => "Text",
            Label::Title => "Title",
        }
    }

    pub const fn idx(&self) -> usize {
        match self {
            Label::Caption => 0,
            Label::Footnote => 1,
            Label::Formula => 2,
            Label::ListItem => 3,
            Label::PageFooter => 4,
            Label::PageHeader => 5,
            Label::Picture => 6,
            Label::SectionHeader => 7,
            Label::Table => 8,
            Label::Text => 9,
            Label::Title => 10,
        }
    }

    pub const fn color(&self) -> [u8; 3] {
        match self {
            Label::Caption => [255, 0, 0],         // Red
            Label::Footnote => [0, 255, 0],        // Green
            Label::Formula => [0, 0, 255],         // Blue
            Label::ListItem => [255, 255, 0],      // Yellow
            Label::PageFooter => [255, 0, 255],    // Magenta
            Label::PageHeader => [0, 255, 255],    // Cyan
            Label::Picture => [128, 0, 128],       // Purple
            Label::SectionHeader => [255, 165, 0], // Orange
            Label::Table => [128, 128, 128],       // Gray
            Label::Text => [0, 128, 0],            // Dark Green
            Label::Title => [255, 20, 147],        // Deep Pink
        }
    }

    pub const fn label_size() -> usize {
        11
    }
}

macro_rules! impl_from_for_label {
    ($($t:ty),*) => {
        $(
            impl From<$t> for Label {
                fn from(idx: $t) -> Self {
                    match idx {
                        0 => Label::Caption,
                        1 => Label::Footnote,
                        2 => Label::Formula,
                        3 => Label::ListItem,
                        4 => Label::PageFooter,
                        5 => Label::PageHeader,
                        6 => Label::Picture,
                        7 => Label::SectionHeader,
                        8 => Label::Table,
                        9 => Label::Text,
                        10 => Label::Title,
                        _ => panic!("Invalid label index: {}", idx),
                    }
                }
            }
        )*
    };
}

impl_from_for_label!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize
);
