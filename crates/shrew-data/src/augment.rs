// Image Augmentation — random transforms for data augmentation
//
// All augmentations operate on `Sample::features` treating them as images
// in [C, H, W] layout (channel-first, row-major).

use rand::thread_rng;
use rand::Rng;

use crate::dataset::Sample;
use crate::transform::Transform;

// RandomHorizontalFlip

/// Randomly flip an image horizontally with probability `p`.
///
/// Expects `feature_shape = [C, H, W]`.
#[derive(Debug, Clone)]
pub struct RandomHorizontalFlip {
    pub p: f64,
}

impl RandomHorizontalFlip {
    pub fn new(p: f64) -> Self {
        Self { p }
    }
}

impl Transform for RandomHorizontalFlip {
    fn apply(&self, mut sample: Sample) -> Sample {
        let mut rng = thread_rng();
        if rng.gen::<f64>() >= self.p {
            return sample;
        }
        let shape = &sample.feature_shape;
        if shape.len() != 3 {
            return sample;
        }
        let (c, h, w) = (shape[0], shape[1], shape[2]);
        let mut flipped = vec![0.0; c * h * w];
        for ch in 0..c {
            for row in 0..h {
                for col in 0..w {
                    let src = ch * h * w + row * w + col;
                    let dst = ch * h * w + row * w + (w - 1 - col);
                    flipped[dst] = sample.features[src];
                }
            }
        }
        sample.features = flipped;
        sample
    }
}

// RandomVerticalFlip

/// Randomly flip an image vertically with probability `p`.
///
/// Expects `feature_shape = [C, H, W]`.
#[derive(Debug, Clone)]
pub struct RandomVerticalFlip {
    pub p: f64,
}

impl RandomVerticalFlip {
    pub fn new(p: f64) -> Self {
        Self { p }
    }
}

impl Transform for RandomVerticalFlip {
    fn apply(&self, mut sample: Sample) -> Sample {
        let mut rng = thread_rng();
        if rng.gen::<f64>() >= self.p {
            return sample;
        }
        let shape = &sample.feature_shape;
        if shape.len() != 3 {
            return sample;
        }
        let (c, h, w) = (shape[0], shape[1], shape[2]);
        let mut flipped = vec![0.0; c * h * w];
        for ch in 0..c {
            for row in 0..h {
                for col in 0..w {
                    let src = ch * h * w + row * w + col;
                    let dst = ch * h * w + (h - 1 - row) * w + col;
                    flipped[dst] = sample.features[src];
                }
            }
        }
        sample.features = flipped;
        sample
    }
}

// RandomCrop

/// Randomly crop an image to `[crop_h, crop_w]`, optionally with zero-padding.
///
/// Expects `feature_shape = [C, H, W]`.  If `padding > 0`, the image is first
/// padded with zeros on all sides by `padding` pixels.
#[derive(Debug, Clone)]
pub struct RandomCrop {
    pub crop_h: usize,
    pub crop_w: usize,
    pub padding: usize,
}

impl RandomCrop {
    pub fn new(crop_h: usize, crop_w: usize, padding: usize) -> Self {
        Self {
            crop_h,
            crop_w,
            padding,
        }
    }
}

impl Transform for RandomCrop {
    fn apply(&self, mut sample: Sample) -> Sample {
        let shape = &sample.feature_shape;
        if shape.len() != 3 {
            return sample;
        }
        let (c, h, w) = (shape[0], shape[1], shape[2]);
        let pad = self.padding;
        let padded_h = h + 2 * pad;
        let padded_w = w + 2 * pad;

        // Build padded image
        let mut padded = vec![0.0; c * padded_h * padded_w];
        for ch in 0..c {
            for row in 0..h {
                for col in 0..w {
                    let src = ch * h * w + row * w + col;
                    let dst = ch * padded_h * padded_w + (row + pad) * padded_w + (col + pad);
                    padded[dst] = sample.features[src];
                }
            }
        }

        // Random crop position
        let mut rng = thread_rng();
        let max_y = padded_h.saturating_sub(self.crop_h);
        let max_x = padded_w.saturating_sub(self.crop_w);
        let y0 = if max_y > 0 {
            rng.gen_range(0..=max_y)
        } else {
            0
        };
        let x0 = if max_x > 0 {
            rng.gen_range(0..=max_x)
        } else {
            0
        };

        let mut cropped = vec![0.0; c * self.crop_h * self.crop_w];
        for ch in 0..c {
            for row in 0..self.crop_h {
                for col in 0..self.crop_w {
                    let src = ch * padded_h * padded_w + (y0 + row) * padded_w + (x0 + col);
                    let dst = ch * self.crop_h * self.crop_w + row * self.crop_w + col;
                    cropped[dst] = padded[src];
                }
            }
        }

        sample.features = cropped;
        sample.feature_shape = vec![c, self.crop_h, self.crop_w];
        sample
    }
}

// RandomNoise — additive Gaussian noise

/// Add Gaussian noise to features: `x' = x + N(0, std)`.
#[derive(Debug, Clone)]
pub struct RandomNoise {
    pub std_dev: f64,
}

impl RandomNoise {
    pub fn new(std_dev: f64) -> Self {
        Self { std_dev }
    }
}

impl Transform for RandomNoise {
    fn apply(&self, mut sample: Sample) -> Sample {
        use rand_distr::{Distribution, Normal};
        let normal = Normal::new(0.0, self.std_dev).unwrap();
        let mut rng = thread_rng();
        for v in &mut sample.features {
            *v += normal.sample(&mut rng);
        }
        sample
    }
}

// RandomErasing — randomly erase a rectangular region (cutout)

/// Erase a random rectangular region, replacing with a constant value.
///
/// Uses a random rectangle whose area is `[min_area_ratio, max_area_ratio]`
/// of the total, with aspect ratio in `[min_aspect, max_aspect]`.
/// Expects `feature_shape = [C, H, W]`.
#[derive(Debug, Clone)]
pub struct RandomErasing {
    pub p: f64,
    pub fill_value: f64,
    pub min_area_ratio: f64,
    pub max_area_ratio: f64,
}

impl RandomErasing {
    pub fn new(p: f64) -> Self {
        Self {
            p,
            fill_value: 0.0,
            min_area_ratio: 0.02,
            max_area_ratio: 0.33,
        }
    }
}

impl Transform for RandomErasing {
    fn apply(&self, mut sample: Sample) -> Sample {
        let mut rng = thread_rng();
        if rng.gen::<f64>() >= self.p {
            return sample;
        }
        let shape = &sample.feature_shape;
        if shape.len() != 3 {
            return sample;
        }
        let (c, h, w) = (shape[0], shape[1], shape[2]);
        let area = (h * w) as f64;

        // Pick random area and aspect ratio
        let target_area = area * rng.gen_range(self.min_area_ratio..self.max_area_ratio);
        let aspect = rng.gen_range(0.3f64..3.3f64);
        let erase_h = (target_area * aspect).sqrt().round() as usize;
        let erase_w = (target_area / aspect).sqrt().round() as usize;

        if erase_h >= h || erase_w >= w {
            return sample;
        }

        let y0 = rng.gen_range(0..h - erase_h);
        let x0 = rng.gen_range(0..w - erase_w);

        for ch in 0..c {
            for row in y0..y0 + erase_h {
                for col in x0..x0 + erase_w {
                    sample.features[ch * h * w + row * w + col] = self.fill_value;
                }
            }
        }
        sample
    }
}

// ColorJitter — random brightness/contrast for images normalised to [0,1]

/// Randomly adjust brightness and contrast.
///
/// brightness: `x' = x + uniform(-brightness, +brightness)`
/// contrast:   `x' = mean + (x - mean) * factor` where factor ∈ `[1 - contrast, 1 + contrast]`
#[derive(Debug, Clone)]
pub struct ColorJitter {
    pub brightness: f64,
    pub contrast: f64,
}

impl ColorJitter {
    pub fn new(brightness: f64, contrast: f64) -> Self {
        Self {
            brightness,
            contrast,
        }
    }
}

impl Transform for ColorJitter {
    fn apply(&self, mut sample: Sample) -> Sample {
        let mut rng = thread_rng();

        // Brightness
        if self.brightness > 0.0 {
            let delta = rng.gen_range(-self.brightness..self.brightness);
            for v in &mut sample.features {
                *v += delta;
            }
        }

        // Contrast
        if self.contrast > 0.0 {
            let factor = rng.gen_range(1.0 - self.contrast..1.0 + self.contrast);
            let mean: f64 = sample.features.iter().sum::<f64>() / sample.features.len() as f64;
            for v in &mut sample.features {
                *v = mean + (*v - mean) * factor;
            }
        }

        sample
    }
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;

    fn make_image_sample(c: usize, h: usize, w: usize) -> Sample {
        let n = c * h * w;
        Sample {
            features: (0..n).map(|i| i as f64).collect(),
            feature_shape: vec![c, h, w],
            target: vec![0.0],
            target_shape: vec![1],
        }
    }

    #[test]
    fn horizontal_flip_deterministic() {
        // p=1.0 always flips
        let flip = RandomHorizontalFlip::new(1.0);
        let sample = make_image_sample(1, 2, 3);
        // Original: [0,1,2, 3,4,5]
        let result = flip.apply(sample);
        // Flipped:  [2,1,0, 5,4,3]
        assert_eq!(result.features, vec![2.0, 1.0, 0.0, 5.0, 4.0, 3.0]);
    }

    #[test]
    fn vertical_flip_deterministic() {
        let flip = RandomVerticalFlip::new(1.0);
        let sample = make_image_sample(1, 2, 3);
        // Original: [0,1,2, 3,4,5]
        let result = flip.apply(sample);
        // Flipped:  [3,4,5, 0,1,2]
        assert_eq!(result.features, vec![3.0, 4.0, 5.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn random_crop_no_padding_same_size() {
        let crop = RandomCrop::new(2, 3, 0);
        let sample = make_image_sample(1, 2, 3);
        let result = crop.apply(sample);
        assert_eq!(result.feature_shape, vec![1, 2, 3]);
        assert_eq!(result.features.len(), 6);
    }

    #[test]
    fn random_crop_with_padding() {
        let crop = RandomCrop::new(4, 4, 1);
        let sample = make_image_sample(1, 4, 4);
        let result = crop.apply(sample);
        assert_eq!(result.feature_shape, vec![1, 4, 4]);
        assert_eq!(result.features.len(), 16);
    }

    #[test]
    fn random_noise_changes_values() {
        let noise = RandomNoise::new(1.0);
        let sample = make_image_sample(1, 2, 2);
        let result = noise.apply(sample.clone());
        // Values should be different (with astronomical probability)
        let changed = result
            .features
            .iter()
            .zip(sample.features.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(changed);
    }

    #[test]
    fn random_erasing_p1() {
        let erasing = RandomErasing::new(1.0);
        let sample = make_image_sample(1, 8, 8);
        let result = erasing.apply(sample);
        // At least some zeros should be introduced
        let num_zeros = result.features.iter().filter(|&&v| v == 0.0).count();
        // Index 0 is naturally 0.0, so at least one more zero from erasing
        assert!(num_zeros >= 2, "Expected erased zeros, got {}", num_zeros);
    }

    #[test]
    fn color_jitter() {
        let jitter = ColorJitter::new(0.1, 0.1);
        let sample = make_image_sample(1, 2, 2);
        let result = jitter.apply(sample.clone());
        let changed = result
            .features
            .iter()
            .zip(sample.features.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(changed);
    }
}
