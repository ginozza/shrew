// Transform — data augmentation / preprocessing pipeline

use crate::dataset::Sample;

/// A transform applied to each sample before batching.
pub trait Transform: Send + Sync {
    /// Apply the transform to a sample, returning the modified sample.
    fn apply(&self, sample: Sample) -> Sample;
}

// Built-in transforms

/// Normalize features to [0, 1] by dividing by a given scale factor.
///
/// Commonly used for image pixels: `Normalize::new(255.0)`.
#[derive(Debug, Clone)]
pub struct Normalize {
    scale: f64,
}

impl Normalize {
    pub fn new(scale: f64) -> Self {
        Self { scale }
    }
}

impl Transform for Normalize {
    fn apply(&self, mut sample: Sample) -> Sample {
        for v in &mut sample.features {
            *v /= self.scale;
        }
        sample
    }
}

/// Standardize features to zero mean and unit variance.
#[derive(Debug, Clone)]
pub struct Standardize {
    pub mean: f64,
    pub std: f64,
}

impl Standardize {
    pub fn new(mean: f64, std: f64) -> Self {
        Self { mean, std }
    }
}

impl Transform for Standardize {
    fn apply(&self, mut sample: Sample) -> Sample {
        for v in &mut sample.features {
            *v = (*v - self.mean) / self.std;
        }
        sample
    }
}

/// One-hot encode the target label into a vector of size `num_classes`.
#[derive(Debug, Clone)]
pub struct OneHotEncode {
    pub num_classes: usize,
}

impl OneHotEncode {
    pub fn new(num_classes: usize) -> Self {
        Self { num_classes }
    }
}

impl Transform for OneHotEncode {
    fn apply(&self, mut sample: Sample) -> Sample {
        let class_idx = sample.target[0] as usize;
        let mut one_hot = vec![0.0; self.num_classes];
        if class_idx < self.num_classes {
            one_hot[class_idx] = 1.0;
        }
        sample.target = one_hot;
        sample.target_shape = vec![self.num_classes];
        sample
    }
}

/// Chain multiple transforms.
pub struct Compose {
    transforms: Vec<Box<dyn Transform>>,
}

impl Compose {
    pub fn new(transforms: Vec<Box<dyn Transform>>) -> Self {
        Self { transforms }
    }
}

impl Transform for Compose {
    fn apply(&self, mut sample: Sample) -> Sample {
        for t in &self.transforms {
            sample = t.apply(sample);
        }
        sample
    }
}

/// Reshape the feature tensor to a different shape (without changing data).
///
/// Useful for converting flat MNIST images `[784]` to 2D `[1, 28, 28]`
/// for convolutional networks.
///
/// # Examples
/// ```ignore
/// // MNIST: [784] → [1, 28, 28] for Conv2d input
/// let reshape = ReshapeFeatures::new(vec![1, 28, 28]);
/// ```
#[derive(Debug, Clone)]
pub struct ReshapeFeatures {
    pub new_shape: Vec<usize>,
}

impl ReshapeFeatures {
    pub fn new(new_shape: Vec<usize>) -> Self {
        Self { new_shape }
    }
}

impl Transform for ReshapeFeatures {
    fn apply(&self, mut sample: Sample) -> Sample {
        // Verify element count matches
        let old_count: usize = sample.feature_shape.iter().product();
        let new_count: usize = self.new_shape.iter().product();
        assert_eq!(
            old_count, new_count,
            "ReshapeFeatures: old shape {:?} ({}) != new shape {:?} ({})",
            sample.feature_shape, old_count, self.new_shape, new_count,
        );
        sample.feature_shape = self.new_shape.clone();
        sample
    }
}
