// AsyncDataLoader — Prefetching data loader with background workers
//
// Spawns a pool of background threads that pre-load and transform batches ahead
// of the consumer.  The consumer pulls ready batches from a channel, overlapping
// data loading with GPU computation.
//
// Usage:
//
//   let loader = AsyncDataLoader::<CpuBackend>::new(
//       &dataset,
//       CpuDevice,
//       AsyncDataLoaderConfig::default()
//           .batch_size(64)
//           .prefetch_factor(2)
//           .num_workers(4),
//   );
//
//   for epoch in 0..num_epochs {
//       for batch in loader.iter_epoch("input", "target") {
//           let batch = batch?;
//           // train on batch ...
//       }
//   }

use std::collections::HashMap;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{thread_rng, SeedableRng};

use shrew_core::backend::Backend;
use shrew_core::error::Error;
use shrew_core::tensor::Tensor;
use shrew_core::DType;

use crate::dataset::{Dataset, Sample};
use crate::transform::Transform;

// Configuration

/// Configuration for the async prefetching data loader.
#[derive(Debug, Clone)]
pub struct AsyncDataLoaderConfig {
    /// Number of samples per batch.
    pub batch_size: usize,
    /// Whether to shuffle indices each epoch.
    pub shuffle: bool,
    /// Whether to drop the last incomplete batch.
    pub drop_last: bool,
    /// DType for the created tensors.
    pub dtype: DType,
    /// Number of background workers (threads) for loading + transforming.
    /// 0 = no background threads (falls back to sync loading, with prefetch
    /// still happening on a single background thread).
    pub num_workers: usize,
    /// How many batches to pre-load ahead of the consumer.
    /// Total buffered batches = prefetch_factor * max(num_workers, 1).
    pub prefetch_factor: usize,
    /// Optional random seed for reproducible shuffling.
    pub seed: Option<u64>,
}

impl Default for AsyncDataLoaderConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            shuffle: true,
            drop_last: false,
            dtype: DType::F32,
            num_workers: 2,
            prefetch_factor: 2,
            seed: None,
        }
    }
}

impl AsyncDataLoaderConfig {
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
    pub fn prefetch_factor(mut self, pf: usize) -> Self {
        self.prefetch_factor = pf;
        self
    }
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = Some(s);
        self
    }
}

// Batch type alias

/// A single batch: maps names (e.g. "input", "target") to tensors.
pub type Batch<B> = HashMap<String, Tensor<B>>;

// AsyncDataLoader

/// A data loader that prefetches batches on background threads.
///
/// On each call to [`iter_epoch`](AsyncDataLoader::iter_epoch), the loader:
/// 1. Optionally reshuffles indices.
/// 2. Spawns worker threads that load, transform, and collate batches.
/// 3. Returns an iterator that pulls ready batches from a bounded channel.
///
/// The channel capacity is `prefetch_factor * max(num_workers, 1)`, so at most
/// that many batches are materialised in memory at any time.
///
/// The dataset is held via `Arc<dyn Dataset>` so it can be safely shared with
/// background worker threads.
pub struct AsyncDataLoader<B: Backend> {
    dataset: Arc<dyn Dataset>,
    config: AsyncDataLoaderConfig,
    transforms: Vec<Arc<dyn Transform>>,
    device: B::Device,
    indices: Vec<usize>,
}

impl<B: Backend> AsyncDataLoader<B>
where
    B::Device: Clone + Send + Sync + 'static,
{
    /// Create a new async data loader.
    pub fn new(
        dataset: Arc<dyn Dataset>,
        device: B::Device,
        config: AsyncDataLoaderConfig,
    ) -> Self {
        let n = dataset.len();
        let indices: Vec<usize> = (0..n).collect();
        Self {
            dataset,
            config,
            transforms: Vec::new(),
            device,
            indices,
        }
    }

    /// Add a transform.
    pub fn with_transform(mut self, t: Arc<dyn Transform>) -> Self {
        self.transforms.push(t);
        self
    }

    /// Number of batches per epoch.
    pub fn num_batches(&self) -> usize {
        if self.config.drop_last {
            self.dataset.len() / self.config.batch_size
        } else {
            self.dataset.len().div_ceil(self.config.batch_size)
        }
    }

    /// Reshuffle indices.
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

    /// Iterate over one epoch of prefetched batches.
    ///
    /// Spawns background workers that load batches into a bounded channel.
    /// The returned iterator yields `Result<Batch<B>>` — one per batch.
    ///
    /// The background workers are joined when the iterator is dropped.
    #[allow(clippy::type_complexity)]
    pub fn iter_epoch(&mut self, input_name: &str, target_name: &str) -> PrefetchIterator<B> {
        self.reshuffle();

        let bs = self.config.batch_size;
        let n = self.dataset.len();
        let num_batches = self.num_batches();
        let workers = self.config.num_workers.max(1);
        let capacity = self.config.prefetch_factor * workers;

        // Build the list of batch index ranges
        let mut batch_ranges: Vec<Vec<usize>> = Vec::with_capacity(num_batches);
        for b in 0..num_batches {
            let start = b * bs;
            let end = (start + bs).min(n);
            batch_ranges.push(self.indices[start..end].to_vec());
        }

        let (tx, rx) = mpsc::sync_channel::<Result<Batch<B>, Error>>(capacity);

        // Shared work queue: each worker pops a batch index to process
        let work_queue: Arc<Mutex<std::vec::IntoIter<(usize, Vec<usize>)>>> = Arc::new(Mutex::new(
            batch_ranges
                .into_iter()
                .enumerate()
                .collect::<Vec<_>>()
                .into_iter(),
        ));

        // Snapshot all the info workers need
        let dtype = self.config.dtype;
        let device = self.device.clone();
        let transforms = self.transforms.clone();
        let input_name = input_name.to_string();
        let target_name = target_name.to_string();
        let dataset = self.dataset.clone();

        let mut handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            let wq = work_queue.clone();
            let tx = tx.clone();
            let dev = device.clone();
            let tfs = transforms.clone();
            let in_name = input_name.clone();
            let tgt_name = target_name.clone();
            let ds = dataset.clone();

            let handle = thread::spawn(move || {
                let dataset: &dyn Dataset = &*ds;

                loop {
                    // Pop next batch from the shared queue
                    let item = {
                        let mut q = wq.lock().unwrap();
                        q.next()
                    };
                    let (_batch_idx, sample_indices) = match item {
                        Some(x) => x,
                        None => break, // no more work
                    };

                    // Fetch and transform samples
                    let samples: Vec<Sample> = sample_indices
                        .iter()
                        .map(|&i| {
                            let mut s = dataset.get(i);
                            for t in &tfs {
                                s = t.apply(s);
                            }
                            s
                        })
                        .collect();

                    // Collate into a batch of tensors
                    let result = collate_batch::<B>(&samples, dtype, &dev, &in_name, &tgt_name);

                    // Send to consumer — if receiver is dropped, stop
                    if tx.send(result).is_err() {
                        break;
                    }
                }
            });
            handles.push(handle);
        }

        // Drop the original sender so the channel closes when all workers finish
        drop(tx);

        PrefetchIterator {
            rx,
            handles: Some(handles),
            remaining: num_batches,
        }
    }
}

// PrefetchIterator

/// An iterator that yields prefetched batches from background workers.
///
/// Workers are joined when the iterator is fully consumed or dropped.
pub struct PrefetchIterator<B: Backend> {
    rx: mpsc::Receiver<Result<Batch<B>, Error>>,
    handles: Option<Vec<thread::JoinHandle<()>>>,
    remaining: usize,
}

impl<B: Backend> Iterator for PrefetchIterator<B> {
    type Item = Result<Batch<B>, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        match self.rx.recv() {
            Ok(batch) => {
                self.remaining -= 1;
                Some(batch)
            }
            Err(_) => {
                // Channel closed — workers done (possibly early)
                self.remaining = 0;
                None
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<B: Backend> ExactSizeIterator for PrefetchIterator<B> {}

impl<B: Backend> Drop for PrefetchIterator<B> {
    fn drop(&mut self) {
        // Drain the channel to unblock workers
        while self.rx.try_recv().is_ok() {}
        // Join all worker threads
        if let Some(handles) = self.handles.take() {
            for h in handles {
                let _ = h.join();
            }
        }
    }
}

// Collation helper

/// Collate a slice of samples into a batch of tensors.
fn collate_batch<B: Backend>(
    samples: &[Sample],
    dtype: DType,
    device: &B::Device,
    input_name: &str,
    target_name: &str,
) -> Result<Batch<B>, Error> {
    let batch_size = samples.len();
    if batch_size == 0 {
        return Ok(HashMap::new());
    }

    // Flatten features
    let feat_shape = &samples[0].feature_shape;
    let feat_len: usize = feat_shape.iter().product();
    let mut features = Vec::with_capacity(batch_size * feat_len);
    for s in samples {
        features.extend_from_slice(&s.features);
    }

    // Flatten targets
    let tgt_shape = &samples[0].target_shape;
    let tgt_len: usize = tgt_shape.iter().product();
    let mut targets = Vec::with_capacity(batch_size * tgt_len);
    for s in samples {
        targets.extend_from_slice(&s.target);
    }

    // Build shapes: [batch_size, ...feature_shape] and [batch_size, ...target_shape]
    let mut f_shape = vec![batch_size];
    f_shape.extend_from_slice(feat_shape);
    let mut t_shape = vec![batch_size];
    t_shape.extend_from_slice(tgt_shape);

    let input_tensor = Tensor::<B>::from_f64_slice(&features, f_shape, dtype, device)?;
    let target_tensor = Tensor::<B>::from_f64_slice(&targets, t_shape, dtype, device)?;

    let mut batch = HashMap::with_capacity(2);
    batch.insert(input_name.to_string(), input_tensor);
    batch.insert(target_name.to_string(), target_tensor);
    Ok(batch)
}
