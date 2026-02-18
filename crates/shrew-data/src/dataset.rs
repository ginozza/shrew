// Dataset trait — unified interface for any data source

/// A single sample: a pair of (input features, label/target).
///
/// Both are stored as `Vec<f64>` with their associated shapes so they can be
/// batched into tensors later.
#[derive(Debug, Clone)]
pub struct Sample {
    /// Input feature vector (flattened).
    pub features: Vec<f64>,
    /// Shape of the feature tensor (e.g. `[784]` for MNIST, `[3,32,32]` for CIFAR).
    pub feature_shape: Vec<usize>,
    /// Target / label value(s) (flattened).  For classification this is typically
    /// a single-element vec holding the class index as `f64`.
    pub target: Vec<f64>,
    /// Shape of the target tensor (e.g. `[1]` for a class index, `[10]` for one-hot).
    pub target_shape: Vec<usize>,
}

/// A dataset is an indexed collection of samples.
///
/// Implementations must be `Send + Sync` so DataLoader can read from multiple
/// threads when parallel prefetching is enabled.
pub trait Dataset: Send + Sync {
    /// Total number of samples in the dataset.
    fn len(&self) -> usize;

    /// Whether the dataset is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Retrieve the sample at position `index`.
    ///
    /// # Panics
    /// May panic if `index >= self.len()`.
    fn get(&self, index: usize) -> Sample;

    /// The shape of a single feature sample (without batch dim).
    fn feature_shape(&self) -> &[usize];

    /// The shape of a single target sample (without batch dim).
    fn target_shape(&self) -> &[usize];

    /// Optional human-readable name.
    fn name(&self) -> &str {
        "dataset"
    }
}
