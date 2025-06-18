/// A 2D axis-aligned bounding box represented by minimum and maximum points.
///
/// This structure is commonly used in computer graphics and computer vision
/// for collision detection, object detection, and spatial queries.
#[derive(Clone, Copy, Debug)]
pub struct Bbox {
    /// The minimum point of the bounding box (bottom-left corner).
    pub min: glam::Vec2,
    /// The maximum point of the bounding box (top-right corner).
    pub max: glam::Vec2,
}

impl Bbox {
    /// Creates a new bounding box from minimum and maximum points.
    ///
    /// # Arguments
    /// * `min` - The minimum point (bottom-left corner)
    /// * `max` - The maximum point (top-right corner)
    ///
    /// # Example
    /// ```
    /// use glam::Vec2;
    /// use ferrpdf_core::analysis::bbox::Bbox;
    /// let bbox = Bbox::new(Vec2::new(0.0, 0.0), Vec2::new(10.0, 5.0));
    /// ```
    pub fn new(min: glam::Vec2, max: glam::Vec2) -> Self {
        Self { min, max }
    }

    /// Creates a new bounding box from a minimum point and size vector.
    ///
    /// # Arguments
    /// * `min` - The minimum point (bottom-left corner)
    /// * `size` - The size vector (width and height)
    ///
    /// # Example
    /// ```
    /// use glam::Vec2;
    /// use ferrpdf_core::analysis::bbox::Bbox;
    /// let bbox = Bbox::new_from_min_size(Vec2::new(1.0, 2.0), Vec2::new(5.0, 3.0));
    /// // Creates a bbox from (1,2) to (6,5)
    /// ```
    pub fn new_from_min_size(min: glam::Vec2, size: glam::Vec2) -> Self {
        Self {
            min,
            max: min + size,
        }
    }

    /// Creates a new bounding box from a center point and size vector.
    ///
    /// This constructor is commonly used with YOLO-style detection outputs where
    /// bounding boxes are represented as (center_x, center_y, width, height).
    ///
    /// # Arguments
    /// * `center` - The center point of the bounding box
    /// * `size` - The size vector (width and height)
    ///
    /// # Example
    /// ```
    /// use glam::Vec2;
    /// use ferrpdf_core::analysis::bbox::Bbox;
    /// // Create a bbox centered at (100, 200) with size 50x80
    /// let bbox = Bbox::from_center_size(Vec2::new(100.0, 200.0), Vec2::new(50.0, 80.0));
    /// // Results in bbox from (75, 160) to (125, 240)
    /// ```
    pub fn from_center_size(center: glam::Vec2, size: glam::Vec2) -> Self {
        let half_size = size / 2.0;
        // TODO: Bad Case
        Self {
            min: center - half_size,
            max: center + half_size,
        }
    }

    /// Calculates the area of the bounding box.
    ///
    /// # Returns
    /// The area as a f32 value (width × height)
    ///
    /// # Example
    /// ```
    /// use glam::Vec2;
    /// use ferrpdf_core::analysis::bbox::Bbox;
    /// let bbox = Bbox::new_from_min_size(Vec2::ZERO, Vec2::new(4.0, 3.0));
    /// assert_eq!(bbox.area(), 12.0);
    /// ```
    pub fn area(&self) -> f32 {
        let length = self.max - self.min;

        length.x * length.y
    }

    /// Calculates the center point of the bounding box.
    ///
    /// # Returns
    /// The center point as a Vec2
    ///
    /// # Example
    /// ```
    /// use glam::Vec2;
    /// use ferrpdf_core::analysis::bbox::Bbox;
    /// let bbox = Bbox::new(Vec2::new(0.0, 0.0), Vec2::new(4.0, 2.0));
    /// assert_eq!(bbox.center(), Vec2::new(2.0, 1.0));
    /// ```
    pub fn center(&self) -> glam::Vec2 {
        (self.min + self.max) / 2.0
    }

    /// Calculates the area of intersection between this bounding box and another.
    ///
    /// This method computes the overlapping area between two axis-aligned bounding boxes.
    /// If the boxes don't overlap, it returns 0.0.
    ///
    /// # Arguments
    /// * `other` - The other bounding box to intersect with
    ///
    /// # Returns
    /// The intersection area as a f32 value, or 0.0 if no intersection exists
    ///
    /// # Algorithm
    /// 1. Find the intersection rectangle by taking:
    ///    - Maximum of minimum points (intersection's bottom-left)
    ///    - Minimum of maximum points (intersection's top-right)
    /// 2. If the resulting rectangle is valid (positive width and height), calculate area
    /// 3. Otherwise, return 0.0 (no intersection)
    ///
    /// # Example
    /// ```
    /// use glam::Vec2;
    /// use ferrpdf_core::analysis::bbox::Bbox;
    /// let bbox1 = Bbox::new(Vec2::new(0.0, 0.0), Vec2::new(4.0, 4.0));
    /// let bbox2 = Bbox::new(Vec2::new(2.0, 2.0), Vec2::new(6.0, 6.0));
    /// assert_eq!(bbox1.intersection(&bbox2), 4.0); // 2x2 intersection area
    /// ```
    pub fn intersection(&self, other: &Self) -> f32 {
        let min = self.min.max(other.min);
        let max = self.max.min(other.max);

        if max.x > min.x && max.y > min.y {
            (max.x - min.x) * (max.y - min.y)
        } else {
            0.
        }
    }

    /// Calculates the Intersection over Union (IoU) between this bounding box and another.
    ///
    /// IoU is a measure of overlap between two bounding boxes, commonly used in computer vision
    /// for object detection evaluation and non-maximum suppression.
    ///
    /// # Arguments
    /// * `other` - The other bounding box to compare with
    ///
    /// # Returns
    /// The IoU value as a f32 between 0.0 and 1.0:
    /// - 0.0: No overlap
    /// - 1.0: Perfect overlap (identical boxes)
    /// - 0.5+: Generally considered good overlap in object detection
    ///
    /// # Formula
    /// IoU = Intersection Area / Union Area
    /// Union Area = Area1 + Area2 - Intersection Area
    ///
    /// # Example
    /// ```
    /// use glam::Vec2;
    /// use ferrpdf_core::analysis::bbox::Bbox;
    /// let bbox1 = Bbox::new(Vec2::new(0.0, 0.0), Vec2::new(2.0, 2.0));
    /// let bbox2 = Bbox::new(Vec2::new(0.0, 0.0), Vec2::new(2.0, 2.0));
    /// assert_eq!(bbox1.iou(&bbox2), 1.0); // Identical boxes
    /// ```
    pub fn iou(&self, other: &Self) -> f32 {
        let intersection_area = self.intersection(other);
        let union_area = self.area() + other.area() - intersection_area;

        if union_area > 0.0 {
            intersection_area / union_area
        } else {
            0.0
        }
    }

    /// Converts this bounding box from image coordinates to Cartesian coordinates.
    ///
    /// Image coordinates have the origin (0,0) at the top-left corner with Y increasing downward.
    /// Cartesian coordinates have the origin (0,0) at the bottom-left corner with Y increasing upward.
    ///
    /// # Arguments
    /// * `image_height` - The height of the image in pixels
    ///
    /// # Returns
    /// A new `Bbox` with coordinates converted to Cartesian system
    ///
    /// # Transformation
    /// For each point (x, y) in image coordinates:
    /// - x_cartesian = x_image (X coordinate remains the same)
    /// - y_cartesian = image_height - y_image (Y coordinate is flipped)
    ///
    /// # Example
    /// ```
    /// use glam::Vec2;
    /// use ferrpdf_core::analysis::bbox::Bbox;
    /// // Image coordinates: top-left (10, 20) to bottom-right (50, 80)
    /// let image_bbox = Bbox::new(Vec2::new(10.0, 20.0), Vec2::new(50.0, 80.0));
    /// let cartesian_bbox = image_bbox.to_cartesian(100.0);
    /// // Cartesian coordinates: bottom-left (10, 20) to top-right (50, 80)
    /// assert_eq!(cartesian_bbox.min, Vec2::new(10.0, 20.0));
    /// assert_eq!(cartesian_bbox.max, Vec2::new(50.0, 80.0));
    /// ```
    pub fn to_cartesian(&self, image_height: f32) -> Self {
        let cartesian_min = glam::Vec2::new(self.min.x, image_height - self.max.y);
        let cartesian_max = glam::Vec2::new(self.max.x, image_height - self.min.y);

        Self::new(cartesian_min, cartesian_max)
    }

    /// Clamps the bounding box coordinates to stay within the specified bounds.
    ///
    /// This method ensures that the bounding box doesn't extend beyond the given
    /// minimum and maximum bounds, which is useful for constraining detection
    /// results to image boundaries.
    ///
    /// # Arguments
    /// * `min_bounds` - The minimum allowed coordinates (typically (0, 0))
    /// * `max_bounds` - The maximum allowed coordinates (typically image dimensions)
    ///
    /// # Returns
    /// A new `Bbox` with coordinates clamped to the specified bounds
    ///
    /// # Example
    /// ```
    /// use glam::Vec2;
    /// use ferrpdf_core::analysis::bbox::Bbox;
    /// let bbox = Bbox::new(Vec2::new(-10.0, -5.0), Vec2::new(1030.0, 1030.0));
    /// let clamped = bbox.clamp(Vec2::new(0.0, 0.0), Vec2::new(1023.0, 1023.0));
    /// assert_eq!(clamped.min, Vec2::new(0.0, 0.0));
    /// assert_eq!(clamped.max, Vec2::new(1023.0, 1023.0));
    /// ```
    pub fn clamp(&self, min_bounds: glam::Vec2, max_bounds: glam::Vec2) -> Self {
        Self {
            min: self.min.max(min_bounds),
            max: self.max.min(max_bounds),
        }
    }

    /// Clamps this bounding box coordinates in-place to stay within the specified bounds.
    ///
    /// This is the mutable version of `clamp()` that modifies the current bounding box
    /// instead of creating a new one.
    ///
    /// # Arguments
    /// * `min_bounds` - The minimum allowed coordinates (typically (0, 0))
    /// * `max_bounds` - The maximum allowed coordinates (typically image dimensions)
    ///
    /// # Example
    /// ```
    /// use glam::Vec2;
    /// use ferrpdf_core::analysis::bbox::Bbox;
    /// let mut bbox = Bbox::new(Vec2::new(-10.0, -5.0), Vec2::new(1030.0, 1030.0));
    /// bbox.clamp_mut(Vec2::new(0.0, 0.0), Vec2::new(1023.0, 1023.0));
    /// assert_eq!(bbox.min, Vec2::new(0.0, 0.0));
    /// assert_eq!(bbox.max, Vec2::new(1023.0, 1023.0));
    /// ```
    pub fn clamp_mut(&mut self, min_bounds: glam::Vec2, max_bounds: glam::Vec2) {
        self.min.x = self.min.x.max(min_bounds.x);
        self.min.y = self.min.y.max(min_bounds.y);
        self.max.x = self.max.x.min(max_bounds.x);
        self.max.y = self.max.y.min(max_bounds.y);
    }

    /// Checks if this bounding box completely contains another bounding box.
    ///
    /// A bounding box is considered to contain another if all four corners
    /// of the contained box are within or on the boundary of the containing box.
    ///
    /// # Arguments
    /// * `other` - The bounding box to check for containment
    ///
    /// # Returns
    /// `true` if this bounding box completely contains the other box, `false` otherwise
    ///
    /// # Example
    /// ```
    /// use glam::Vec2;
    /// use ferrpdf_core::analysis::bbox::Bbox;
    ///
    /// let outer = Bbox::new(Vec2::new(0.0, 0.0), Vec2::new(10.0, 10.0));
    /// let inner = Bbox::new(Vec2::new(2.0, 3.0), Vec2::new(7.0, 8.0));
    /// let separate = Bbox::new(Vec2::new(12.0, 12.0), Vec2::new(15.0, 15.0));
    ///
    /// assert!(outer.contains(&inner));   // outer contains inner
    /// assert!(!inner.contains(&outer));  // inner does not contain outer
    /// assert!(!outer.contains(&separate)); // outer does not contain separate
    /// ```
    pub fn contains(&self, other: &Self) -> bool {
        self.min.x <= other.min.x
            && self.min.y <= other.min.y
            && self.max.x >= other.max.x
            && self.max.y >= other.max.y
    }

    /// Creates a union bounding box that encompasses both this bounding box and another.
    ///
    /// The union bounding box is the smallest axis-aligned rectangle that completely
    /// contains both input bounding boxes. This is useful for merging overlapping
    /// detections in object detection algorithms.
    ///
    /// # Arguments
    /// * `other` - The other bounding box to union with
    ///
    /// # Returns
    /// A new `Bbox` that represents the union of both bounding boxes
    ///
    /// # Example
    /// ```
    /// use glam::Vec2;
    /// use ferrpdf_core::analysis::bbox::Bbox;
    ///
    /// let bbox1 = Bbox::new(Vec2::new(0.0, 0.0), Vec2::new(5.0, 5.0));
    /// let bbox2 = Bbox::new(Vec2::new(3.0, 3.0), Vec2::new(8.0, 8.0));
    /// let union = bbox1.union(&bbox2);
    ///
    /// assert_eq!(union.min, Vec2::new(0.0, 0.0));
    /// assert_eq!(union.max, Vec2::new(8.0, 8.0));
    /// ```
    pub fn union(&self, other: &Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// Calculates the overlap ratio between this bounding box and another using the smaller area as denominator.
    ///
    /// This method provides a more lenient overlap detection compared to IoU, especially useful
    /// when dealing with bounding boxes of very different sizes. Unlike IoU which uses the union
    /// area as denominator, this method uses the smaller of the two areas, making it more sensitive
    /// to detecting when a smaller bbox is contained within or significantly overlaps with a larger one.
    ///
    /// # Arguments
    /// * `other` - The other bounding box to compare with
    ///
    /// # Returns
    /// The overlap ratio as a f32 between 0.0 and 1.0:
    /// - 0.0: No overlap
    /// - 1.0: The smaller bbox is completely contained within the larger one
    /// - Values > 0.5: Significant overlap relative to the smaller bbox
    ///
    /// # Formula
    /// overlap_ratio = intersection_area / min(area1, area2)
    ///
    /// # Example
    /// ```
    /// use glam::Vec2;
    /// use ferrpdf_core::analysis::bbox::Bbox;
    ///
    /// // Large bbox containing a smaller one
    /// let large = Bbox::new(Vec2::new(0.0, 0.0), Vec2::new(100.0, 100.0));
    /// let small = Bbox::new(Vec2::new(10.0, 10.0), Vec2::new(30.0, 30.0));
    ///
    /// let overlap_ratio = large.overlap_ratio(&small);
    /// assert_eq!(overlap_ratio, 1.0); // Small bbox is completely contained
    ///
    /// let iou = large.iou(&small);
    /// assert!(overlap_ratio > iou); // More lenient than IoU
    /// ```
    pub fn overlap_ratio(&self, other: &Self) -> f32 {
        let intersection_area = self.intersection(other);
        let min_area = self.area().min(other.area());

        if min_area > 0.0 {
            intersection_area / min_area
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bbox_area() {
        // Test basic rectangle area calculation
        let bbox = Bbox::new_from_min_size(glam::Vec2::ZERO, glam::Vec2::new(2.0, 3.0));
        assert_eq!(bbox.area(), 6.0);

        // Test square area
        let square = Bbox::new_from_min_size(glam::Vec2::new(1.0, 1.0), glam::Vec2::new(5.0, 5.0));
        assert_eq!(square.area(), 25.0);

        // Test unit square
        let unit = Bbox::new(glam::Vec2::ZERO, glam::Vec2::ONE);
        assert_eq!(unit.area(), 1.0);

        // Test zero area (degenerate case)
        let line = Bbox::new(glam::Vec2::ZERO, glam::Vec2::new(5.0, 0.0));
        assert_eq!(line.area(), 0.0);
    }

    #[test]
    fn test_bbox_center() {
        // Test center of rectangle starting at origin
        let bbox = Bbox::new_from_min_size(glam::Vec2::ZERO, glam::Vec2::new(2.0, 3.0));
        assert_eq!(bbox.center(), glam::Vec2::new(1.0, 1.5));

        // Test center of offset rectangle
        let offset_bbox = Bbox::new(glam::Vec2::new(10.0, 20.0), glam::Vec2::new(14.0, 26.0));
        assert_eq!(offset_bbox.center(), glam::Vec2::new(12.0, 23.0));

        // Test center of unit square at origin
        let unit = Bbox::new(glam::Vec2::ZERO, glam::Vec2::ONE);
        assert_eq!(unit.center(), glam::Vec2::new(0.5, 0.5));

        // Test center with negative coordinates
        let negative = Bbox::new(glam::Vec2::new(-4.0, -2.0), glam::Vec2::new(0.0, 2.0));
        assert_eq!(negative.center(), glam::Vec2::new(-2.0, 0.0));
    }

    #[test]
    fn test_bbox_from_center_size() {
        // Test basic center-size construction
        let bbox =
            Bbox::from_center_size(glam::Vec2::new(100.0, 200.0), glam::Vec2::new(50.0, 80.0));
        assert_eq!(bbox.min, glam::Vec2::new(75.0, 160.0));
        assert_eq!(bbox.max, glam::Vec2::new(125.0, 240.0));
        assert_eq!(bbox.center(), glam::Vec2::new(100.0, 200.0));
        assert_eq!(bbox.area(), 4000.0); // 50 * 80

        // Test square bbox from center
        let square = Bbox::from_center_size(glam::Vec2::new(10.0, 10.0), glam::Vec2::new(6.0, 6.0));
        assert_eq!(square.min, glam::Vec2::new(7.0, 7.0));
        assert_eq!(square.max, glam::Vec2::new(13.0, 13.0));
        assert_eq!(square.center(), glam::Vec2::new(10.0, 10.0));
        assert_eq!(square.area(), 36.0);

        // Test unit bbox at origin
        let unit = Bbox::from_center_size(glam::Vec2::new(0.5, 0.5), glam::Vec2::new(1.0, 1.0));
        assert_eq!(unit.min, glam::Vec2::new(0.0, 0.0));
        assert_eq!(unit.max, glam::Vec2::new(1.0, 1.0));
        assert_eq!(unit.center(), glam::Vec2::new(0.5, 0.5));

        // Test with negative center coordinates
        let negative_center =
            Bbox::from_center_size(glam::Vec2::new(-5.0, -3.0), glam::Vec2::new(4.0, 2.0));
        assert_eq!(negative_center.min, glam::Vec2::new(-7.0, -4.0));
        assert_eq!(negative_center.max, glam::Vec2::new(-3.0, -2.0));
        assert_eq!(negative_center.center(), glam::Vec2::new(-5.0, -3.0));

        // Test with zero size (degenerate case)
        let point = Bbox::from_center_size(glam::Vec2::new(5.0, 7.0), glam::Vec2::new(0.0, 0.0));
        assert_eq!(point.min, glam::Vec2::new(5.0, 7.0));
        assert_eq!(point.max, glam::Vec2::new(5.0, 7.0));
        assert_eq!(point.area(), 0.0);

        // Test fractional coordinates
        let fractional =
            Bbox::from_center_size(glam::Vec2::new(2.5, 3.7), glam::Vec2::new(1.4, 2.6));
        assert_eq!(fractional.min, glam::Vec2::new(1.8, 2.4));
        assert_eq!(fractional.max, glam::Vec2::new(3.2, 5.0));
        assert_eq!(fractional.center(), glam::Vec2::new(2.5, 3.7));

        // Test YOLO-style detection coordinates (common use case)
        let yolo_detection =
            Bbox::from_center_size(glam::Vec2::new(512.0, 384.0), glam::Vec2::new(100.0, 200.0));
        assert_eq!(yolo_detection.min, glam::Vec2::new(462.0, 284.0));
        assert_eq!(yolo_detection.max, glam::Vec2::new(562.0, 484.0));
        assert_eq!(yolo_detection.area(), 20000.0);

        // Test very small bbox
        let tiny = Bbox::from_center_size(glam::Vec2::new(1.0, 1.0), glam::Vec2::new(0.1, 0.1));
        assert_eq!(tiny.min, glam::Vec2::new(0.95, 0.95));
        assert_eq!(tiny.max, glam::Vec2::new(1.05, 1.05));
        assert!((tiny.area() - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_bbox_intersection_area() {
        // Two partially overlapping boxes (2×2 intersection)
        let bbox1 = Bbox::new(glam::Vec2::new(0.0, 0.0), glam::Vec2::new(4.0, 4.0));
        let bbox2 = Bbox::new(glam::Vec2::new(2.0, 2.0), glam::Vec2::new(6.0, 6.0));
        assert_eq!(bbox1.intersection(&bbox2), 4.0);

        // Non-overlapping boxes (no intersection)
        let bbox3 = Bbox::new(glam::Vec2::new(0.0, 0.0), glam::Vec2::new(2.0, 2.0));
        let bbox4 = Bbox::new(glam::Vec2::new(3.0, 3.0), glam::Vec2::new(5.0, 5.0));
        assert_eq!(bbox3.intersection(&bbox4), 0.0);

        // Identical boxes (complete overlap)
        let bbox5 = Bbox::new(glam::Vec2::new(1.0, 1.0), glam::Vec2::new(3.0, 3.0));
        let bbox6 = Bbox::new(glam::Vec2::new(1.0, 1.0), glam::Vec2::new(3.0, 3.0));
        assert_eq!(bbox5.intersection(&bbox6), 4.0);

        // One box completely inside another
        let outer = Bbox::new(glam::Vec2::new(0.0, 0.0), glam::Vec2::new(10.0, 10.0));
        let inner = Bbox::new(glam::Vec2::new(2.0, 3.0), glam::Vec2::new(5.0, 7.0));
        assert_eq!(outer.intersection(&inner), 12.0); // 3×4 = 12
        assert_eq!(inner.intersection(&outer), 12.0); // Should be symmetric

        // Edge touching (no area intersection)
        let left = Bbox::new(glam::Vec2::new(0.0, 0.0), glam::Vec2::new(2.0, 2.0));
        let right = Bbox::new(glam::Vec2::new(2.0, 0.0), glam::Vec2::new(4.0, 2.0));
        assert_eq!(left.intersection(&right), 0.0);

        // Partial overlap with different aspect ratios
        let wide = Bbox::new(glam::Vec2::new(0.0, 0.0), glam::Vec2::new(6.0, 2.0));
        let tall = Bbox::new(glam::Vec2::new(1.0, 0.0), glam::Vec2::new(3.0, 4.0));
        assert_eq!(wide.intersection(&tall), 4.0); // 2×2 = 4

        // Corner overlap
        let corner1 = Bbox::new(glam::Vec2::new(0.0, 0.0), glam::Vec2::new(3.0, 3.0));
        let corner2 = Bbox::new(glam::Vec2::new(2.0, 2.0), glam::Vec2::new(5.0, 5.0));
        assert_eq!(corner1.intersection(&corner2), 1.0); // 1×1 = 1

        // Negative coordinate test
        let neg1 = Bbox::new(glam::Vec2::new(-2.0, -2.0), glam::Vec2::new(1.0, 1.0));
        let neg2 = Bbox::new(glam::Vec2::new(-1.0, -1.0), glam::Vec2::new(2.0, 2.0));
        assert_eq!(neg1.intersection(&neg2), 4.0); // 2×2 = 4
    }

    #[test]
    fn test_bbox_iou() {
        // Identical boxes (perfect overlap, IoU = 1.0)
        let bbox1 = Bbox::new(glam::Vec2::new(0.0, 0.0), glam::Vec2::new(4.0, 4.0));
        let bbox2 = Bbox::new(glam::Vec2::new(0.0, 0.0), glam::Vec2::new(4.0, 4.0));
        assert_eq!(bbox1.iou(&bbox2), 1.0);

        // Non-overlapping boxes (IoU = 0.0)
        let bbox3 = Bbox::new(glam::Vec2::new(0.0, 0.0), glam::Vec2::new(2.0, 2.0));
        let bbox4 = Bbox::new(glam::Vec2::new(3.0, 3.0), glam::Vec2::new(5.0, 5.0));
        assert_eq!(bbox3.iou(&bbox4), 0.0);

        // Partial overlap - calculate IoU manually
        // bbox1: (0,0) to (4,4), area = 16
        // bbox2: (2,2) to (6,6), area = 16
        // intersection: (2,2) to (4,4), area = 4
        // union: 16 + 16 - 4 = 28
        // IoU: 4/28 = 1/7 ≈ 0.1428571
        let bbox5 = Bbox::new(glam::Vec2::new(0.0, 0.0), glam::Vec2::new(4.0, 4.0));
        let bbox6 = Bbox::new(glam::Vec2::new(2.0, 2.0), glam::Vec2::new(6.0, 6.0));
        let expected_iou = 4.0 / 28.0;
        assert!((bbox5.iou(&bbox6) - expected_iou).abs() < 1e-6);

        // One box completely inside another
        // outer: (0,0) to (10,10), area = 100
        // inner: (2,3) to (5,7), area = 12
        // intersection: (2,3) to (5,7), area = 12 (the inner box)
        // union: 100 + 12 - 12 = 100
        // IoU: 12/100 = 0.12
        let outer = Bbox::new(glam::Vec2::new(0.0, 0.0), glam::Vec2::new(10.0, 10.0));
        let inner = Bbox::new(glam::Vec2::new(2.0, 3.0), glam::Vec2::new(5.0, 7.0));
        assert_eq!(outer.iou(&inner), 0.12);
        assert_eq!(inner.iou(&outer), 0.12); // Should be symmetric

        // Half overlap case
        // bbox1: (0,0) to (4,2), area = 8
        // bbox2: (2,0) to (6,2), area = 8
        // intersection: (2,0) to (4,2), area = 4
        // union: 8 + 8 - 4 = 12
        // IoU: 4/12 = 1/3 ≈ 0.3333333
        let half1 = Bbox::new(glam::Vec2::new(0.0, 0.0), glam::Vec2::new(4.0, 2.0));
        let half2 = Bbox::new(glam::Vec2::new(2.0, 0.0), glam::Vec2::new(6.0, 2.0));
        let expected_half_iou = 1.0 / 3.0;
        assert!((half1.iou(&half2) - expected_half_iou).abs() < 1e-6);

        // Edge touching (no overlap, IoU = 0.0)
        let left = Bbox::new(glam::Vec2::new(0.0, 0.0), glam::Vec2::new(2.0, 2.0));
        let right = Bbox::new(glam::Vec2::new(2.0, 0.0), glam::Vec2::new(4.0, 2.0));
        assert_eq!(left.iou(&right), 0.0);

        // Small overlap with different sizes
        // small: (0,0) to (2,2), area = 4
        // large: (1,1) to (5,5), area = 16
        // intersection: (1,1) to (2,2), area = 1
        // union: 4 + 16 - 1 = 19
        // IoU: 1/19 ≈ 0.0526316
        let small = Bbox::new(glam::Vec2::new(0.0, 0.0), glam::Vec2::new(2.0, 2.0));
        let large = Bbox::new(glam::Vec2::new(1.0, 1.0), glam::Vec2::new(5.0, 5.0));
        let expected_small_iou = 1.0 / 19.0;
        assert!((small.iou(&large) - expected_small_iou).abs() < 1e-6);

        // Zero area boxes (degenerate case, IoU = 0.0)
        let line1 = Bbox::new(glam::Vec2::new(0.0, 0.0), glam::Vec2::new(5.0, 0.0));
        let line2 = Bbox::new(glam::Vec2::new(2.0, 0.0), glam::Vec2::new(7.0, 0.0));
        assert_eq!(line1.iou(&line2), 0.0);

        // Negative coordinates
        let neg1 = Bbox::new(glam::Vec2::new(-4.0, -4.0), glam::Vec2::new(0.0, 0.0));
        let neg2 = Bbox::new(glam::Vec2::new(-2.0, -2.0), glam::Vec2::new(2.0, 2.0));
        // neg1 area: 16, neg2 area: 16
        // intersection: (-2,-2) to (0,0), area = 4
        // union: 16 + 16 - 4 = 28
        // IoU: 4/28 = 1/7
        let expected_neg_iou = 4.0 / 28.0;
        assert!((neg1.iou(&neg2) - expected_neg_iou).abs() < 1e-6);
    }

    #[test]
    fn test_bbox_to_cartesian() {
        let image_height = 100.0;

        // Test basic conversion
        // Image coords: top-left (10, 20) to bottom-right (50, 80)
        // Cartesian coords: bottom-left (10, 20) to top-right (50, 80)
        let image_bbox = Bbox::new(glam::Vec2::new(10.0, 20.0), glam::Vec2::new(50.0, 80.0));
        let cartesian_bbox = image_bbox.to_cartesian(image_height);
        assert_eq!(cartesian_bbox.min, glam::Vec2::new(10.0, 20.0));
        assert_eq!(cartesian_bbox.max, glam::Vec2::new(50.0, 80.0));

        // Test bbox at image top (y=0 in image coords)
        // Image coords: (0, 0) to (20, 10)
        // Cartesian coords: (0, 90) to (20, 100)
        let top_bbox = Bbox::new(glam::Vec2::new(0.0, 0.0), glam::Vec2::new(20.0, 10.0));
        let top_cartesian = top_bbox.to_cartesian(image_height);
        assert_eq!(top_cartesian.min, glam::Vec2::new(0.0, 90.0));
        assert_eq!(top_cartesian.max, glam::Vec2::new(20.0, 100.0));

        // Test bbox at image bottom (y=max in image coords)
        // Image coords: (0, 90) to (20, 100)
        // Cartesian coords: (0, 0) to (20, 10)
        let bottom_bbox = Bbox::new(glam::Vec2::new(0.0, 90.0), glam::Vec2::new(20.0, 100.0));
        let bottom_cartesian = bottom_bbox.to_cartesian(image_height);
        assert_eq!(bottom_cartesian.min, glam::Vec2::new(0.0, 0.0));
        assert_eq!(bottom_cartesian.max, glam::Vec2::new(20.0, 10.0));

        // Test centered bbox
        // Image coords: (25, 40) to (75, 60)
        // Cartesian coords: (25, 40) to (75, 60)
        let center_bbox = Bbox::new(glam::Vec2::new(25.0, 40.0), glam::Vec2::new(75.0, 60.0));
        let center_cartesian = center_bbox.to_cartesian(image_height);
        assert_eq!(center_cartesian.min, glam::Vec2::new(25.0, 40.0));
        assert_eq!(center_cartesian.max, glam::Vec2::new(75.0, 60.0));

        // Test full image bbox
        // Image coords: (0, 0) to (100, 100)
        // Cartesian coords: (0, 0) to (100, 100)
        let full_bbox = Bbox::new(glam::Vec2::new(0.0, 0.0), glam::Vec2::new(100.0, 100.0));
        let full_cartesian = full_bbox.to_cartesian(image_height);
        assert_eq!(full_cartesian.min, glam::Vec2::new(0.0, 0.0));
        assert_eq!(full_cartesian.max, glam::Vec2::new(100.0, 100.0));

        // Test single pixel bbox at origin
        // Image coords: (0, 0) to (1, 1)
        // Cartesian coords: (0, 99) to (1, 100)
        let pixel_bbox = Bbox::new(glam::Vec2::new(0.0, 0.0), glam::Vec2::new(1.0, 1.0));
        let pixel_cartesian = pixel_bbox.to_cartesian(image_height);
        assert_eq!(pixel_cartesian.min, glam::Vec2::new(0.0, 99.0));
        assert_eq!(pixel_cartesian.max, glam::Vec2::new(1.0, 100.0));

        // Test different image height
        let small_image_height = 50.0;
        let small_bbox = Bbox::new(glam::Vec2::new(10.0, 10.0), glam::Vec2::new(20.0, 20.0));
        let small_cartesian = small_bbox.to_cartesian(small_image_height);
        assert_eq!(small_cartesian.min, glam::Vec2::new(10.0, 30.0));
        assert_eq!(small_cartesian.max, glam::Vec2::new(20.0, 40.0));
    }

    #[test]
    fn test_bbox_clamp() {
        let min_bounds = glam::Vec2::new(0.0, 0.0);
        let max_bounds = glam::Vec2::new(1023.0, 1023.0);

        // Test bbox that exceeds bounds on all sides
        let oversized = Bbox::new(
            glam::Vec2::new(-10.0, -5.0),
            glam::Vec2::new(1030.0, 1030.0),
        );
        let clamped = oversized.clamp(min_bounds, max_bounds);
        assert_eq!(clamped.min, glam::Vec2::new(0.0, 0.0));
        assert_eq!(clamped.max, glam::Vec2::new(1023.0, 1023.0));

        // Test bbox already within bounds (should remain unchanged)
        let within_bounds = Bbox::new(glam::Vec2::new(100.0, 200.0), glam::Vec2::new(500.0, 600.0));
        let unchanged = within_bounds.clamp(min_bounds, max_bounds);
        assert_eq!(unchanged.min, glam::Vec2::new(100.0, 200.0));
        assert_eq!(unchanged.max, glam::Vec2::new(500.0, 600.0));

        // Test bbox that only exceeds left/top bounds
        let left_top_exceed =
            Bbox::new(glam::Vec2::new(-50.0, -30.0), glam::Vec2::new(200.0, 300.0));
        let left_top_clamped = left_top_exceed.clamp(min_bounds, max_bounds);
        assert_eq!(left_top_clamped.min, glam::Vec2::new(0.0, 0.0));
        assert_eq!(left_top_clamped.max, glam::Vec2::new(200.0, 300.0));

        // Test bbox that only exceeds right/bottom bounds
        let right_bottom_exceed = Bbox::new(
            glam::Vec2::new(800.0, 900.0),
            glam::Vec2::new(1100.0, 1200.0),
        );
        let right_bottom_clamped = right_bottom_exceed.clamp(min_bounds, max_bounds);
        assert_eq!(right_bottom_clamped.min, glam::Vec2::new(800.0, 900.0));
        assert_eq!(right_bottom_clamped.max, glam::Vec2::new(1023.0, 1023.0));

        // Test with different bounds
        let custom_min = glam::Vec2::new(50.0, 100.0);
        let custom_max = glam::Vec2::new(500.0, 400.0);
        let custom_bbox = Bbox::new(glam::Vec2::new(0.0, 0.0), glam::Vec2::new(600.0, 500.0));
        let custom_clamped = custom_bbox.clamp(custom_min, custom_max);
        assert_eq!(custom_clamped.min, glam::Vec2::new(50.0, 100.0));
        assert_eq!(custom_clamped.max, glam::Vec2::new(500.0, 400.0));

        // Test edge case: bbox at exact bounds
        let exact_bounds = Bbox::new(min_bounds, max_bounds);
        let exact_clamped = exact_bounds.clamp(min_bounds, max_bounds);
        assert_eq!(exact_clamped.min, min_bounds);
        assert_eq!(exact_clamped.max, max_bounds);
    }

    #[test]
    fn test_bbox_clamp_mut() {
        let min_bounds = glam::Vec2::new(0.0, 0.0);
        let max_bounds = glam::Vec2::new(1023.0, 1023.0);

        // Test in-place clamping of oversized bbox
        let mut oversized = Bbox::new(
            glam::Vec2::new(-10.0, -5.0),
            glam::Vec2::new(1030.0, 1030.0),
        );
        oversized.clamp_mut(min_bounds, max_bounds);
        assert_eq!(oversized.min, glam::Vec2::new(0.0, 0.0));
        assert_eq!(oversized.max, glam::Vec2::new(1023.0, 1023.0));

        // Test in-place clamping of bbox already within bounds
        let mut within_bounds =
            Bbox::new(glam::Vec2::new(100.0, 200.0), glam::Vec2::new(500.0, 600.0));
        within_bounds.clamp_mut(min_bounds, max_bounds);
        assert_eq!(within_bounds.min, glam::Vec2::new(100.0, 200.0));
        assert_eq!(within_bounds.max, glam::Vec2::new(500.0, 600.0));

        // Test in-place clamping with partial bounds violation
        let mut partial_exceed = Bbox::new(
            glam::Vec2::new(-20.0, 500.0),
            glam::Vec2::new(1100.0, 800.0),
        );
        partial_exceed.clamp_mut(min_bounds, max_bounds);
        assert_eq!(partial_exceed.min, glam::Vec2::new(0.0, 500.0));
        assert_eq!(partial_exceed.max, glam::Vec2::new(1023.0, 800.0));

        // Test with custom bounds
        let custom_min = glam::Vec2::new(10.0, 20.0);
        let custom_max = glam::Vec2::new(100.0, 200.0);
        let mut custom_bbox = Bbox::new(glam::Vec2::new(0.0, 0.0), glam::Vec2::new(150.0, 250.0));
        custom_bbox.clamp_mut(custom_min, custom_max);
        assert_eq!(custom_bbox.min, glam::Vec2::new(10.0, 20.0));
        assert_eq!(custom_bbox.max, glam::Vec2::new(100.0, 200.0));
    }

    #[test]
    fn test_bbox_contains() {
        // Test complete containment
        let outer = Bbox::new(glam::Vec2::new(0.0, 0.0), glam::Vec2::new(10.0, 10.0));
        let inner = Bbox::new(glam::Vec2::new(2.0, 3.0), glam::Vec2::new(7.0, 8.0));
        assert!(outer.contains(&inner));
        assert!(!inner.contains(&outer));

        // Test identical boxes (should contain each other)
        let bbox1 = Bbox::new(glam::Vec2::new(1.0, 1.0), glam::Vec2::new(5.0, 5.0));
        let bbox2 = Bbox::new(glam::Vec2::new(1.0, 1.0), glam::Vec2::new(5.0, 5.0));
        assert!(bbox1.contains(&bbox2));
        assert!(bbox2.contains(&bbox1));

        // Test edge touching (should contain)
        let container = Bbox::new(glam::Vec2::new(0.0, 0.0), glam::Vec2::new(10.0, 10.0));
        let edge_box = Bbox::new(glam::Vec2::new(0.0, 0.0), glam::Vec2::new(10.0, 10.0));
        assert!(container.contains(&edge_box));

        // Test partial overlap but not containment
        let bbox3 = Bbox::new(glam::Vec2::new(0.0, 0.0), glam::Vec2::new(5.0, 5.0));
        let bbox4 = Bbox::new(glam::Vec2::new(3.0, 3.0), glam::Vec2::new(8.0, 8.0));
        assert!(!bbox3.contains(&bbox4));
        assert!(!bbox4.contains(&bbox3));

        // Test completely separate boxes
        let separate1 = Bbox::new(glam::Vec2::new(0.0, 0.0), glam::Vec2::new(2.0, 2.0));
        let separate2 = Bbox::new(glam::Vec2::new(5.0, 5.0), glam::Vec2::new(7.0, 7.0));
        assert!(!separate1.contains(&separate2));
        assert!(!separate2.contains(&separate1));

        // Test one dimension contained, other not
        let wide = Bbox::new(glam::Vec2::new(0.0, 0.0), glam::Vec2::new(10.0, 2.0));
        let tall = Bbox::new(glam::Vec2::new(1.0, 0.0), glam::Vec2::new(3.0, 5.0));
        assert!(!wide.contains(&tall)); // tall extends beyond wide in y
        assert!(!tall.contains(&wide)); // wide extends beyond tall in x

        // Test with negative coordinates
        let neg_outer = Bbox::new(glam::Vec2::new(-10.0, -10.0), glam::Vec2::new(0.0, 0.0));
        let neg_inner = Bbox::new(glam::Vec2::new(-5.0, -7.0), glam::Vec2::new(-2.0, -1.0));
        assert!(neg_outer.contains(&neg_inner));
        assert!(!neg_inner.contains(&neg_outer));

        // Test point-like bbox (zero area)
        let point = Bbox::new(glam::Vec2::new(5.0, 5.0), glam::Vec2::new(5.0, 5.0));
        let container_point = Bbox::new(glam::Vec2::new(0.0, 0.0), glam::Vec2::new(10.0, 10.0));
        assert!(container_point.contains(&point));
        assert!(!point.contains(&container_point));

        // Test floating point precision
        let precise1 = Bbox::new(glam::Vec2::new(0.1, 0.1), glam::Vec2::new(9.9, 9.9));
        let precise2 = Bbox::new(glam::Vec2::new(0.2, 0.2), glam::Vec2::new(9.8, 9.8));
        assert!(precise1.contains(&precise2));
        assert!(!precise2.contains(&precise1));
    }

    #[test]
    fn test_bbox_union() {
        // Test basic union of two overlapping boxes
        let bbox1 = Bbox::new(glam::Vec2::new(0.0, 0.0), glam::Vec2::new(5.0, 5.0));
        let bbox2 = Bbox::new(glam::Vec2::new(3.0, 3.0), glam::Vec2::new(8.0, 8.0));
        let union = bbox1.union(&bbox2);

        assert_eq!(union.min, glam::Vec2::new(0.0, 0.0));
        assert_eq!(union.max, glam::Vec2::new(8.0, 8.0));
        assert_eq!(union.area(), 64.0); // 8x8

        // Test union with non-overlapping boxes
        let bbox3 = Bbox::new(glam::Vec2::new(0.0, 0.0), glam::Vec2::new(2.0, 2.0));
        let bbox4 = Bbox::new(glam::Vec2::new(5.0, 5.0), glam::Vec2::new(7.0, 7.0));
        let union2 = bbox3.union(&bbox4);

        assert_eq!(union2.min, glam::Vec2::new(0.0, 0.0));
        assert_eq!(union2.max, glam::Vec2::new(7.0, 7.0));
        assert_eq!(union2.area(), 49.0); // 7x7

        // Test union of identical boxes
        let bbox5 = Bbox::new(glam::Vec2::new(1.0, 1.0), glam::Vec2::new(3.0, 3.0));
        let bbox6 = Bbox::new(glam::Vec2::new(1.0, 1.0), glam::Vec2::new(3.0, 3.0));
        let union3 = bbox5.union(&bbox6);

        assert_eq!(union3.min, glam::Vec2::new(1.0, 1.0));
        assert_eq!(union3.max, glam::Vec2::new(3.0, 3.0));
        assert_eq!(union3.area(), 4.0); // Same as original

        // Test union where one box contains another
        let outer = Bbox::new(glam::Vec2::new(0.0, 0.0), glam::Vec2::new(10.0, 10.0));
        let inner = Bbox::new(glam::Vec2::new(2.0, 3.0), glam::Vec2::new(5.0, 7.0));
        let union4 = outer.union(&inner);

        assert_eq!(union4.min, glam::Vec2::new(0.0, 0.0));
        assert_eq!(union4.max, glam::Vec2::new(10.0, 10.0));
        assert_eq!(union4.area(), 100.0); // Same as outer box

        // Test union with negative coordinates
        let neg1 = Bbox::new(glam::Vec2::new(-5.0, -3.0), glam::Vec2::new(-1.0, 1.0));
        let neg2 = Bbox::new(glam::Vec2::new(-2.0, -1.0), glam::Vec2::new(3.0, 4.0));
        let union5 = neg1.union(&neg2);

        assert_eq!(union5.min, glam::Vec2::new(-5.0, -3.0));
        assert_eq!(union5.max, glam::Vec2::new(3.0, 4.0));
        assert_eq!(union5.area(), 56.0); // 8x7

        // Test union symmetry (a.union(b) == b.union(a))
        let bbox7 = Bbox::new(glam::Vec2::new(1.0, 2.0), glam::Vec2::new(4.0, 6.0));
        let bbox8 = Bbox::new(glam::Vec2::new(3.0, 1.0), glam::Vec2::new(7.0, 5.0));
        let union_ab = bbox7.union(&bbox8);
        let union_ba = bbox8.union(&bbox7);

        assert_eq!(union_ab.min, union_ba.min);
        assert_eq!(union_ab.max, union_ba.max);
        assert_eq!(union_ab.area(), union_ba.area());
    }

    #[test]
    fn test_bbox_overlap_ratio() {
        // Test complete containment (smaller bbox completely inside larger one)
        let large = Bbox::new(glam::Vec2::new(0.0, 0.0), glam::Vec2::new(100.0, 100.0));
        let small = Bbox::new(glam::Vec2::new(10.0, 10.0), glam::Vec2::new(30.0, 30.0));

        let overlap_ratio = large.overlap_ratio(&small);
        assert_eq!(overlap_ratio, 1.0); // Small bbox is completely contained

        // Should be symmetric for complete containment
        let overlap_ratio_reversed = small.overlap_ratio(&large);
        assert_eq!(overlap_ratio_reversed, 1.0);

        // Compare with IoU - overlap_ratio should be much higher for containment
        let iou = large.iou(&small);
        assert!(overlap_ratio > iou);

        // Test partial overlap with different sizes
        let bbox1 = Bbox::new(glam::Vec2::new(0.0, 0.0), glam::Vec2::new(60.0, 60.0)); // Area: 3600
        let bbox2 = Bbox::new(glam::Vec2::new(40.0, 40.0), glam::Vec2::new(80.0, 80.0)); // Area: 1600
        // Intersection: (40,40) to (60,60) = 400
        // Min area: 1600
        // Expected overlap ratio: 400/1600 = 0.25

        let overlap_ratio_partial = bbox1.overlap_ratio(&bbox2);
        assert!((overlap_ratio_partial - 0.25).abs() < 0.001);

        // Test with identical bboxes
        let identical1 = Bbox::new(glam::Vec2::new(5.0, 5.0), glam::Vec2::new(15.0, 15.0));
        let identical2 = Bbox::new(glam::Vec2::new(5.0, 5.0), glam::Vec2::new(15.0, 15.0));
        let overlap_ratio_identical = identical1.overlap_ratio(&identical2);
        assert_eq!(overlap_ratio_identical, 1.0);

        // Test with no overlap
        let separate1 = Bbox::new(glam::Vec2::new(0.0, 0.0), glam::Vec2::new(10.0, 10.0));
        let separate2 = Bbox::new(glam::Vec2::new(20.0, 20.0), glam::Vec2::new(30.0, 30.0));
        let overlap_ratio_none = separate1.overlap_ratio(&separate2);
        assert_eq!(overlap_ratio_none, 0.0);

        // Test edge case: zero area bbox
        let zero_area = Bbox::new(glam::Vec2::new(5.0, 5.0), glam::Vec2::new(5.0, 5.0));
        let normal = Bbox::new(glam::Vec2::new(0.0, 0.0), glam::Vec2::new(10.0, 10.0));
        let overlap_ratio_zero = zero_area.overlap_ratio(&normal);
        assert_eq!(overlap_ratio_zero, 0.0);

        // Test case where overlap_ratio is more useful than IoU for small-in-large detection
        let very_large = Bbox::new(glam::Vec2::new(0.0, 0.0), glam::Vec2::new(1000.0, 1000.0)); // 1M area
        let very_small = Bbox::new(glam::Vec2::new(100.0, 100.0), glam::Vec2::new(110.0, 110.0)); // 100 area

        let overlap_ratio_large_small = very_large.overlap_ratio(&very_small);
        let iou_large_small = very_large.iou(&very_small);

        assert_eq!(overlap_ratio_large_small, 1.0); // Complete containment
        assert!(iou_large_small < 0.001); // Very low IoU due to size difference
        assert!(overlap_ratio_large_small > iou_large_small);
    }
}
