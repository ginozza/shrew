// DataLoader — batching, shuffling, iteration

use std::collections::HashMap;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{thread_rng, SeedableRng};

use rayon::prelude::*;

use shrew_core::backend::Backend;
use shrew_core::tensor::Tensor;
use shrew_core::DType;

use crate::dataset::{Dataset, Sample};
use crate::transform::Transform;

/// Configuration for the DataLoader.
#[derive(Debug, Clone)]
pub struct DataLoaderConfig {
    /// Number of samples per batch.
    pub batch_size: usize,
    /// Whether to shuffle indices each epoch.
    pub shuffle: bool,
    /// Whether to drop the last incomplete batch.
    pub drop_last: bool,
    /// DType for the created tensors.
    pub dtype: DType,
    /// Number of parallel workers for sample fetching (0 = sequential).
    pub num_workers: usize,
    /// Optional random seed for reproducible shuffling.
    pub seed: Option<u64>,
}

impl Default for DataLoaderConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            shuffle: true,
            drop_last: false,
            dtype: DType::F32,
            num_workers: 0,
            seed: None,
        }
    }
}

impl DataLoaderConfig {
    pub fn batch_size(mut self, bs: usize) -> Self {
        self.batch_size = bs;
        self
    }

    pub fn shuffle(mut self, s: bool) -> Self {
        self.shuffle = s;
        self
    }

    pub fn drop_last(mut self, d: bool) -> Self {
        self.drop_last = d;
        self
    }

    pub fn dtype(mut self, d: DType) -> Self {
        self.dtype = d;
        self
    }

    pub fn num_workers(mut self, n: usize) -> Self {
        self.num_workers = n;
        self
    }

    pub fn seed(mut self, s: u64) -> Self {
        self.seed = Some(s);
        self
    }
}

/// A DataLoader wraps a Dataset and produces batches of tensors.
///
/// Each batch is a `HashMap<String, Tensor<B>>` with keys `"input"` and
/// `"target"`, matching the format expected by `Executor::run()` and
/// `Trainer::train()`.
pub struct DataLoader<'a, B: Backend> {
    dataset: &'a dyn Dataset,
    config: DataLoaderConfig,
    transforms: Vec<Box<dyn Transform>>,
    device: B::Device,
    indices: Vec<usize>,
}

impl<'a, B: Backend> DataLoader<'a, B> {
    /// Create a new DataLoader over a dataset.
    pub fn new(dataset: &'a dyn Dataset, device: B::Device, config: DataLoaderConfig) -> Self {
        let indices: Vec<usize> = (0..dataset.len()).collect();
        Self {
            dataset,
            config,
            transforms: Vec::new(),
            device,
            indices,
        }
    }

    /// Add a transform to apply to each sample.
    pub fn with_transform(mut self, t: Box<dyn Transform>) -> Self {
        self.transforms.push(t);
        self
    }

    /// The number of batches per epoch.
    pub fn num_batches(&self) -> usize {
        if self.config.drop_last {
            self.dataset.len() / self.config.batch_size
        } else {
            self.dataset.len().div_ceil(self.config.batch_size)
        }
    }

    /// Total number of samples.
    pub fn len(&self) -> usize {
        self.dataset.len()
    }

    /// Whether the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.dataset.is_empty()
    }

    /// Reshuffle indices (call at the start of each epoch).
    pub fn reshuffle(&mut self) {
        if self.config.shuffle {
            match self.config.seed {
                Some(seed) => {
                    let mut rng = StdRng::seed_from_u64(seed);
                    self.indices.shuffle(&mut rng);
                }
                None => {
                    let mut rng = thread_rng();
                    self.indices.shuffle(&mut rng);
                }
            }
        }
    }

    /// Fetch a slice of samples, optionally in parallel via rayon.
    fn fetch_samples(&self, indices: &[usize]) -> Vec<Sample> {
        if self.config.num_workers > 0 && indices.len() > 1 {
            // Parallel fetch + transform
            indices
                .par_iter()
                .map(|&i| {
                    let mut s = self.dataset.get(i);
                    for t in &self.transforms {
                        s = t.apply(s);
                    }
                    s
                })
                .collect()
        } else {
            // Sequential
            indices
                .iter()
                .map(|&i| {
                    let mut s = self.dataset.get(i);
                    for t in &self.transforms {
                        s = t.apply(s);
                    }
                    s
                })
                .collect()
        }
    }

    /// Produce all batches for one epoch as a Vec of HashMap tensors.
    ///
    /// Each HashMap has keys matching the input/target names you'll pass
    /// to the executor.  By default: `"input"` and `"target"`.
    pub fn epoch_batches(
        &mut self,
        input_name: &str,
        target_name: &str,
    ) -> Result<Vec<HashMap<String, Tensor<B>>>, shrew_core::Error> {
        self.reshuffle();

        let bs = self.config.batch_size;
        let n = self.dataset.len();
        let num_batches = self.num_batches();
        let mut batches = Vec::with_capacity(num_batches);

        for batch_idx in 0..num_batches {
            let start = batch_idx * bs;
            let end = (start + bs).min(n);
            let actual_bs = end - start;

            // Collect samples (potentially in parallel)
            let batch_indices: Vec<usize> = (start..end).map(|i| self.indices[i]).collect();
            let samples = self.fetch_samples(&batch_indices);

            // Stack features into a batch tensor [actual_bs, ...feature_shape]
            let feat_shape = samples[0].feature_shape.clone();
            let tgt_shape = samples[0].target_shape.clone();

            let mut feat_data: Vec<f64> = Vec::with_capacity(actual_bs * samples[0].features.len());
            let mut tgt_data: Vec<f64> = Vec::with_capacity(actual_bs * samples[0].target.len());

            for s in &samples {
                feat_data.extend_from_slice(&s.features);
                tgt_data.extend_from_slice(&s.target);
            }

            // Build batch shapes: [actual_bs, ...original_shape]
            let mut batch_feat_shape = vec![actual_bs];
            batch_feat_shape.extend_from_slice(&feat_shape);

            let mut batch_tgt_shape = vec![actual_bs];
            batch_tgt_shape.extend_from_slice(&tgt_shape);

            let feat_tensor = Tensor::<B>::from_f64_slice(
                &feat_data,
                batch_feat_shape,
                self.config.dtype,
                &self.device,
            )?;

            let tgt_tensor = Tensor::<B>::from_f64_slice(
                &tgt_data,
                batch_tgt_shape,
                self.config.dtype,
                &self.device,
            )?;

            let mut batch_map = HashMap::new();
            batch_map.insert(input_name.to_string(), feat_tensor);
            batch_map.insert(target_name.to_string(), tgt_tensor);

            batches.push(batch_map);
        }

        Ok(batches)
    }

    /// Iterate over batches one at a time (lower memory than `epoch_batches`).
    pub fn iter_batches(
        &mut self,
        input_name: &str,
        target_name: &str,
    ) -> BatchIterator<'_, 'a, B> {
        self.reshuffle();
        BatchIterator {
            loader: self,
            batch_idx: 0,
            input_name: input_name.to_string(),
            target_name: target_name.to_string(),
        }
    }
}

/// Iterator that yields one batch at a time.
pub struct BatchIterator<'l, 'a, B: Backend> {
    loader: &'l DataLoader<'a, B>,
    batch_idx: usize,
    input_name: String,
    target_name: String,
}

impl<'l, 'a, B: Backend> Iterator for BatchIterator<'l, 'a, B> {
    type Item = Result<HashMap<String, Tensor<B>>, shrew_core::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        let bs = self.loader.config.batch_size;
        let n = self.loader.dataset.len();
        let start = self.batch_idx * bs;

        if start >= n {
            return None;
        }

        if self.loader.config.drop_last && start + bs > n {
            return None;
        }

        let end = (start + bs).min(n);
        let actual_bs = end - start;
        self.batch_idx += 1;

        // Collect and transform samples (potentially in parallel)
        let batch_indices: Vec<usize> = (start..end).map(|i| self.loader.indices[i]).collect();
        let samples = self.loader.fetch_samples(&batch_indices);

        let feat_shape = samples[0].feature_shape.clone();
        let tgt_shape = samples[0].target_shape.clone();

        let mut feat_data: Vec<f64> = Vec::with_capacity(actual_bs * samples[0].features.len());
        let mut tgt_data: Vec<f64> = Vec::with_capacity(actual_bs * samples[0].target.len());

        for s in &samples {
            feat_data.extend_from_slice(&s.features);
            tgt_data.extend_from_slice(&s.target);
        }

        let mut batch_feat_shape = vec![actual_bs];
        batch_feat_shape.extend_from_slice(&feat_shape);

        let mut batch_tgt_shape = vec![actual_bs];
        batch_tgt_shape.extend_from_slice(&tgt_shape);

        let feat_tensor = match Tensor::<B>::from_f64_slice(
            &feat_data,
            batch_feat_shape,
            self.loader.config.dtype,
            &self.loader.device,
        ) {
            Ok(t) => t,
            Err(e) => return Some(Err(e)),
        };

        let tgt_tensor = match Tensor::<B>::from_f64_slice(
            &tgt_data,
            batch_tgt_shape,
            self.loader.config.dtype,
            &self.loader.device,
        ) {
            Ok(t) => t,
            Err(e) => return Some(Err(e)),
        };

        let mut batch_map = HashMap::new();
        batch_map.insert(self.input_name.clone(), feat_tensor);
        batch_map.insert(self.target_name.clone(), tgt_tensor);

        Some(Ok(batch_map))
    }
}
