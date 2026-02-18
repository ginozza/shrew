// Dataset Combinators — compose, filter, subset, concatenate datasets

use crate::dataset::{Dataset, Sample};
use crate::transform::Transform;

// SubsetDataset — view of selected indices

/// A dataset that exposes only the samples at the given indices.
///
/// This is useful for train/val/test splitting.
pub struct SubsetDataset<D: Dataset> {
    inner: D,
    indices: Vec<usize>,
}

impl<D: Dataset> SubsetDataset<D> {
    /// Create a subset of `inner` containing only the samples at `indices`.
    ///
    /// # Panics
    /// Panics (lazily, at `get` time) if any index is out of range.
    pub fn new(inner: D, indices: Vec<usize>) -> Self {
        Self { inner, indices }
    }
}

impl<D: Dataset> Dataset for SubsetDataset<D> {
    fn len(&self) -> usize {
        self.indices.len()
    }

    fn get(&self, index: usize) -> Sample {
        self.inner.get(self.indices[index])
    }

    fn feature_shape(&self) -> &[usize] {
        self.inner.feature_shape()
    }

    fn target_shape(&self) -> &[usize] {
        self.inner.target_shape()
    }

    fn name(&self) -> &str {
        self.inner.name()
    }
}

// ConcatDataset — concatenate multiple datasets

/// Concatenate two or more datasets end-to-end.
///
/// All datasets must share the same feature and target shapes.
pub struct ConcatDataset {
    datasets: Vec<Box<dyn Dataset>>,
    cumulative_sizes: Vec<usize>,
    feature_shape: Vec<usize>,
    target_shape: Vec<usize>,
}

impl ConcatDataset {
    /// Create a concatenation of the given datasets.
    ///
    /// # Panics
    /// Panics if `datasets` is empty.
    pub fn new(datasets: Vec<Box<dyn Dataset>>) -> Self {
        assert!(
            !datasets.is_empty(),
            "ConcatDataset: need at least one dataset"
        );

        let feature_shape = datasets[0].feature_shape().to_vec();
        let target_shape = datasets[0].target_shape().to_vec();

        let mut cumulative_sizes = Vec::with_capacity(datasets.len());
        let mut total = 0;
        for ds in &datasets {
            total += ds.len();
            cumulative_sizes.push(total);
        }

        Self {
            datasets,
            cumulative_sizes,
            feature_shape,
            target_shape,
        }
    }

    /// Locate which dataset and local index a global index maps to.
    fn locate(&self, index: usize) -> (usize, usize) {
        for (ds_idx, &cum) in self.cumulative_sizes.iter().enumerate() {
            if index < cum {
                let offset = if ds_idx == 0 {
                    0
                } else {
                    self.cumulative_sizes[ds_idx - 1]
                };
                return (ds_idx, index - offset);
            }
        }
        panic!(
            "ConcatDataset: index {} out of range (total {})",
            index,
            self.cumulative_sizes.last().unwrap_or(&0)
        );
    }
}

impl Dataset for ConcatDataset {
    fn len(&self) -> usize {
        *self.cumulative_sizes.last().unwrap_or(&0)
    }

    fn get(&self, index: usize) -> Sample {
        let (ds_idx, local_idx) = self.locate(index);
        self.datasets[ds_idx].get(local_idx)
    }

    fn feature_shape(&self) -> &[usize] {
        &self.feature_shape
    }

    fn target_shape(&self) -> &[usize] {
        &self.target_shape
    }

    fn name(&self) -> &str {
        "concat"
    }
}

// MapDataset — apply a transform lazily

/// Wraps a dataset and applies a `Transform` lazily on each `get()`.
pub struct MapDataset<D: Dataset> {
    inner: D,
    transform: Box<dyn Transform>,
    /// Feature shape after transform (caller-provided).
    feat_shape: Vec<usize>,
    /// Target shape after transform (caller-provided).
    tgt_shape: Vec<usize>,
}

impl<D: Dataset> MapDataset<D> {
    /// Create a MapDataset.
    ///
    /// `feat_shape` and `tgt_shape` describe the shapes *after* the
    /// transform is applied.  If the transform doesn't change shapes,
    /// you can clone them from the inner dataset.
    pub fn new(
        inner: D,
        transform: Box<dyn Transform>,
        feat_shape: Vec<usize>,
        tgt_shape: Vec<usize>,
    ) -> Self {
        Self {
            inner,
            transform,
            feat_shape,
            tgt_shape,
        }
    }

    /// Convenience: create a MapDataset whose shapes are unchanged.
    pub fn same_shape(inner: D, transform: Box<dyn Transform>) -> Self {
        let feat_shape = inner.feature_shape().to_vec();
        let tgt_shape = inner.target_shape().to_vec();
        Self {
            inner,
            transform,
            feat_shape,
            tgt_shape,
        }
    }
}

impl<D: Dataset> Dataset for MapDataset<D> {
    fn len(&self) -> usize {
        self.inner.len()
    }

    fn get(&self, index: usize) -> Sample {
        let sample = self.inner.get(index);
        self.transform.apply(sample)
    }

    fn feature_shape(&self) -> &[usize] {
        &self.feat_shape
    }

    fn target_shape(&self) -> &[usize] {
        &self.tgt_shape
    }

    fn name(&self) -> &str {
        self.inner.name()
    }
}

// VecDataset — in-memory dataset from raw vectors

/// A simple in-memory dataset backed by a `Vec<Sample>`.
///
/// Useful for building datasets programmatically or loading from CSV/JSON.
pub struct VecDataset {
    samples: Vec<Sample>,
    feature_shape: Vec<usize>,
    target_shape: Vec<usize>,
    dataset_name: String,
}

impl VecDataset {
    /// Create a VecDataset from a vector of samples.
    ///
    /// # Panics
    /// Panics if `samples` is empty.
    pub fn new(samples: Vec<Sample>, name: &str) -> Self {
        assert!(!samples.is_empty(), "VecDataset: need at least one sample");
        let feature_shape = samples[0].feature_shape.clone();
        let target_shape = samples[0].target_shape.clone();
        Self {
            samples,
            feature_shape,
            target_shape,
            dataset_name: name.to_string(),
        }
    }

    /// Build from feature/target matrices.
    ///
    /// `features`: `[n_samples, n_features]` row-major
    /// `targets`:  `[n_samples, n_targets]` row-major
    pub fn from_flat(
        features: &[f64],
        feature_shape: &[usize],
        targets: &[f64],
        target_shape: &[usize],
        name: &str,
    ) -> Self {
        let feat_per_sample: usize = feature_shape.iter().product();
        let tgt_per_sample: usize = target_shape.iter().product();
        let n = features.len() / feat_per_sample;
        assert_eq!(features.len(), n * feat_per_sample);
        assert_eq!(targets.len(), n * tgt_per_sample);

        let samples: Vec<Sample> = (0..n)
            .map(|i| Sample {
                features: features[i * feat_per_sample..(i + 1) * feat_per_sample].to_vec(),
                feature_shape: feature_shape.to_vec(),
                target: targets[i * tgt_per_sample..(i + 1) * tgt_per_sample].to_vec(),
                target_shape: target_shape.to_vec(),
            })
            .collect();

        Self {
            samples,
            feature_shape: feature_shape.to_vec(),
            target_shape: target_shape.to_vec(),
            dataset_name: name.to_string(),
        }
    }
}

impl Dataset for VecDataset {
    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, index: usize) -> Sample {
        self.samples[index].clone()
    }

    fn feature_shape(&self) -> &[usize] {
        &self.feature_shape
    }

    fn target_shape(&self) -> &[usize] {
        &self.target_shape
    }

    fn name(&self) -> &str {
        &self.dataset_name
    }
}

// Train / Validation / Test Split

/// Split a dataset into (train, val) or (train, val, test) subsets.
///
/// Returns `SubsetDataset` views over the original dataset.
///
/// # Arguments
/// * `dataset` — the source dataset
/// * `ratios` — slice of 2 or 3 floats that sum to 1.0, e.g. `[0.8, 0.2]`
///   or `[0.7, 0.15, 0.15]`
/// * `seed` — random seed for reproducible shuffling of indices
pub fn train_test_split<D>(dataset: D, ratios: &[f64], seed: u64) -> Vec<SubsetDataset<D>>
where
    D: Dataset + Clone,
{
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;

    assert!(
        ratios.len() >= 2 && ratios.len() <= 3,
        "train_test_split: ratios must have 2 or 3 elements"
    );
    let sum: f64 = ratios.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "train_test_split: ratios must sum to 1.0, got {}",
        sum
    );

    let n = dataset.len();
    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng = StdRng::seed_from_u64(seed);
    indices.shuffle(&mut rng);

    let mut splits = Vec::new();
    let mut offset = 0;
    for (i, &ratio) in ratios.iter().enumerate() {
        let count = if i == ratios.len() - 1 {
            n - offset // give remainder to last split
        } else {
            (n as f64 * ratio).round() as usize
        };
        let end = (offset + count).min(n);
        splits.push(SubsetDataset::new(
            dataset.clone(),
            indices[offset..end].to_vec(),
        ));
        offset = end;
    }

    splits
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny helper dataset for testing.
    #[derive(Clone)]
    struct TinyDataset {
        n: usize,
    }

    impl Dataset for TinyDataset {
        fn len(&self) -> usize {
            self.n
        }
        fn get(&self, idx: usize) -> Sample {
            Sample {
                features: vec![idx as f64],
                feature_shape: vec![1],
                target: vec![(idx % 3) as f64],
                target_shape: vec![1],
            }
        }
        fn feature_shape(&self) -> &[usize] {
            &[1]
        }
        fn target_shape(&self) -> &[usize] {
            &[1]
        }
    }

    #[test]
    fn subset_dataset() {
        let ds = TinyDataset { n: 10 };
        let sub = SubsetDataset::new(ds, vec![2, 5, 7]);
        assert_eq!(sub.len(), 3);
        assert_eq!(sub.get(0).features[0], 2.0);
        assert_eq!(sub.get(1).features[0], 5.0);
        assert_eq!(sub.get(2).features[0], 7.0);
    }

    #[test]
    fn concat_dataset() {
        let ds1 = TinyDataset { n: 5 };
        let ds2 = TinyDataset { n: 3 };
        let concat = ConcatDataset::new(vec![Box::new(ds1), Box::new(ds2)]);
        assert_eq!(concat.len(), 8);
        // First 5 come from ds1, next 3 from ds2
        assert_eq!(concat.get(0).features[0], 0.0);
        assert_eq!(concat.get(4).features[0], 4.0);
        assert_eq!(concat.get(5).features[0], 0.0); // ds2 index 0
        assert_eq!(concat.get(7).features[0], 2.0); // ds2 index 2
    }

    #[test]
    fn map_dataset() {
        use crate::transform::Normalize;
        let ds = TinyDataset { n: 4 };
        let mapped = MapDataset::same_shape(ds, Box::new(Normalize::new(10.0)));
        assert_eq!(mapped.len(), 4);
        let s = mapped.get(2);
        assert!((s.features[0] - 0.2).abs() < 1e-10);
    }

    #[test]
    fn vec_dataset() {
        let features = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let targets = vec![0.0, 1.0, 0.0];
        let ds = VecDataset::from_flat(&features, &[2], &targets, &[1], "test");
        assert_eq!(ds.len(), 3);
        assert_eq!(ds.get(0).features, vec![1.0, 2.0]);
        assert_eq!(ds.get(1).features, vec![3.0, 4.0]);
        assert_eq!(ds.get(2).target, vec![0.0]);
    }

    #[test]
    fn train_test_split_two_way() {
        let ds = TinyDataset { n: 100 };
        let splits = train_test_split(ds, &[0.8, 0.2], 42);
        assert_eq!(splits.len(), 2);
        assert_eq!(splits[0].len() + splits[1].len(), 100);
        assert_eq!(splits[0].len(), 80);
        assert_eq!(splits[1].len(), 20);
    }

    #[test]
    fn train_test_split_three_way() {
        let ds = TinyDataset { n: 100 };
        let splits = train_test_split(ds, &[0.7, 0.15, 0.15], 42);
        assert_eq!(splits.len(), 3);
        assert_eq!(splits[0].len() + splits[1].len() + splits[2].len(), 100);
    }

    #[test]
    fn train_test_split_reproducible() {
        let ds1 = TinyDataset { n: 50 };
        let ds2 = TinyDataset { n: 50 };
        let s1 = train_test_split(ds1, &[0.8, 0.2], 123);
        let s2 = train_test_split(ds2, &[0.8, 0.2], 123);
        // Same seed → same indices → same samples
        for i in 0..s1[0].len() {
            assert_eq!(s1[0].get(i).features, s2[0].get(i).features);
        }
    }
}
