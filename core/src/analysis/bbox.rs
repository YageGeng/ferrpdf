#[derive(Clone, Copy, Debug)]
pub struct Bbox {
    /// The minimum point of the bounding box.
    pub min: glam::Vec2,
    /// The maximum point of the bounding box.
    pub max: glam::Vec2,
}

impl Bbox {
    /// Calculate the intersection area between two bounding boxes.
    pub fn intersection(&self, other: &Bbox) -> f32 {
        let width = (self.max.x.min(other.max.x) - self.min.x.max(other.min.x)).max(0.0);
        let height = (self.max.y.min(other.max.y) - self.min.y.max(other.min.y)).max(0.0);
        width * height
    }

    /// Calculate the union area of two bounding boxes.
    pub fn union(&self, other: &Bbox) -> f32 {
        let area1 = (self.max.x - self.min.x) * (self.max.y - self.min.y);
        let area2 = (other.max.x - other.min.x) * (other.max.y - other.min.y);
        area1 + area2 - self.intersection(other)
    }

    /// Calculate the Intersection over Union (IoU) ratio.
    pub fn iou(&self, other: &Bbox) -> f32 {
        let intersection = self.intersection(other);
        let union = self.union(other);
        if union > 0.0 {
            intersection / union
        } else {
            0.0
        }
    }
}
