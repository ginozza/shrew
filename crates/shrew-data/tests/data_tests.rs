// Tests for shrew-data: Dataset, DataLoader, Transforms, MNIST

use shrew_core::DType;
use shrew_cpu::{CpuBackend, CpuDevice};
use shrew_data::dataset::{Dataset, Sample};
use shrew_data::loader::{DataLoader, DataLoaderConfig};
use shrew_data::mnist::{build_idx1_bytes, build_idx3_bytes, MnistDataset, MnistSplit};
use shrew_data::transform::{Compose, Normalize, OneHotEncode, Standardize};
use shrew_data::Transform;

// Simple in-memory dataset for testing

struct ToyDataset {
    samples: Vec<(Vec<f64>, f64)>,
}

impl ToyDataset {
    fn new(n: usize) -> Self {
        let samples: Vec<(Vec<f64>, f64)> = (0..n)
            .map(|i| {
                let x = i as f64;
                (vec![x, x * 2.0], (i % 3) as f64)
            })
            .collect();
        Self { samples }
    }
}

impl Dataset for ToyDataset {
    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, index: usize) -> Sample {
        let (ref features, label) = self.samples[index];
        Sample {
            features: features.clone(),
            feature_shape: vec![2],
            target: vec![label],
            target_shape: vec![1],
        }
    }

    fn feature_shape(&self) -> &[usize] {
        &[2]
    }

    fn target_shape(&self) -> &[usize] {
        &[1]
    }

    fn name(&self) -> &str {
        "toy"
    }
}

// Dataset trait tests

#[test]
fn test_toy_dataset_basics() {
    let ds = ToyDataset::new(10);
    assert_eq!(ds.len(), 10);
    assert!(!ds.is_empty());
    assert_eq!(ds.name(), "toy");
    assert_eq!(ds.feature_shape(), &[2]);
    assert_eq!(ds.target_shape(), &[1]);
}

#[test]
fn test_toy_dataset_get() {
    let ds = ToyDataset::new(5);
    let s = ds.get(3);
    assert_eq!(s.features, vec![3.0, 6.0]);
    assert_eq!(s.target, vec![0.0]); // 3 % 3 = 0
    assert_eq!(s.feature_shape, vec![2]);
}

// Transform tests

#[test]
fn test_normalize_transform() {
    let t = Normalize::new(255.0);
    let sample = Sample {
        features: vec![0.0, 127.5, 255.0],
        feature_shape: vec![3],
        target: vec![5.0],
        target_shape: vec![1],
    };
    let out = t.apply(sample);
    assert!((out.features[0] - 0.0).abs() < 1e-9);
    assert!((out.features[1] - 0.5).abs() < 1e-9);
    assert!((out.features[2] - 1.0).abs() < 1e-9);
    // target unchanged
    assert_eq!(out.target, vec![5.0]);
}

#[test]
fn test_standardize_transform() {
    let t = Standardize::new(100.0, 50.0);
    let sample = Sample {
        features: vec![100.0, 150.0, 50.0],
        feature_shape: vec![3],
        target: vec![0.0],
        target_shape: vec![1],
    };
    let out = t.apply(sample);
    assert!((out.features[0] - 0.0).abs() < 1e-9); // (100-100)/50
    assert!((out.features[1] - 1.0).abs() < 1e-9); // (150-100)/50
    assert!((out.features[2] - -1.0).abs() < 1e-9); // (50-100)/50
}

#[test]
fn test_onehot_encode() {
    let t = OneHotEncode::new(10);
    let sample = Sample {
        features: vec![1.0, 2.0],
        feature_shape: vec![2],
        target: vec![3.0],
        target_shape: vec![1],
    };
    let out = t.apply(sample);
    assert_eq!(out.target.len(), 10);
    assert_eq!(out.target[3], 1.0);
    assert_eq!(out.target[0], 0.0);
    assert_eq!(out.target_shape, vec![10]);
    // features unchanged
    assert_eq!(out.features, vec![1.0, 2.0]);
}

#[test]
fn test_compose_transforms() {
    let t = Compose::new(vec![
        Box::new(Normalize::new(255.0)),
        Box::new(OneHotEncode::new(5)),
    ]);
    let sample = Sample {
        features: vec![255.0, 0.0],
        feature_shape: vec![2],
        target: vec![2.0],
        target_shape: vec![1],
    };
    let out = t.apply(sample);
    // Normalize applied
    assert!((out.features[0] - 1.0).abs() < 1e-9);
    assert!((out.features[1] - 0.0).abs() < 1e-9);
    // OneHot applied
    assert_eq!(out.target.len(), 5);
    assert_eq!(out.target[2], 1.0);
}

// DataLoader tests

#[test]
fn test_dataloader_num_batches() {
    let ds = ToyDataset::new(10);
    let config = DataLoaderConfig::default().batch_size(3).shuffle(false);
    let loader = DataLoader::<CpuBackend>::new(&ds, CpuDevice, config);
    // 10 / 3 = 3 full + 1 partial = 4
    assert_eq!(loader.num_batches(), 4);
}

#[test]
fn test_dataloader_num_batches_drop_last() {
    let ds = ToyDataset::new(10);
    let config = DataLoaderConfig::default()
        .batch_size(3)
        .shuffle(false)
        .drop_last(true);
    let loader = DataLoader::<CpuBackend>::new(&ds, CpuDevice, config);
    // 10 / 3 = 3 (discard the last 1)
    assert_eq!(loader.num_batches(), 3);
}

#[test]
fn test_dataloader_epoch_batches_no_shuffle() {
    let ds = ToyDataset::new(6);
    let config = DataLoaderConfig::default()
        .batch_size(2)
        .shuffle(false)
        .dtype(DType::F64);
    let mut loader = DataLoader::<CpuBackend>::new(&ds, CpuDevice, config);

    let batches = loader.epoch_batches("x", "y").unwrap();
    assert_eq!(batches.len(), 3);

    // First batch: samples 0 and 1
    let b0 = &batches[0];
    let x = b0.get("x").expect("missing x");
    let y = b0.get("y").expect("missing y");

    assert_eq!(x.dims(), &[2, 2]); // batch=2, features=2
    assert_eq!(y.dims(), &[2, 1]); // batch=2, target=1

    let x_data = x.to_f64_vec().unwrap();
    // Sample 0: [0.0, 0.0], Sample 1: [1.0, 2.0]
    assert_eq!(x_data, vec![0.0, 0.0, 1.0, 2.0]);
}

#[test]
fn test_dataloader_epoch_batches_partial() {
    let ds = ToyDataset::new(5);
    let config = DataLoaderConfig::default()
        .batch_size(3)
        .shuffle(false)
        .dtype(DType::F64);
    let mut loader = DataLoader::<CpuBackend>::new(&ds, CpuDevice, config);

    let batches = loader.epoch_batches("input", "target").unwrap();
    assert_eq!(batches.len(), 2);

    // First batch: 3 samples
    let x0 = batches[0].get("input").unwrap();
    assert_eq!(x0.dims(), &[3, 2]);

    // Second batch: 2 samples (partial)
    let x1 = batches[1].get("input").unwrap();
    assert_eq!(x1.dims(), &[2, 2]);
}

#[test]
fn test_dataloader_with_transform() {
    let ds = ToyDataset::new(4);
    let config = DataLoaderConfig::default()
        .batch_size(4)
        .shuffle(false)
        .dtype(DType::F64);
    let mut loader = DataLoader::<CpuBackend>::new(&ds, CpuDevice, config)
        .with_transform(Box::new(OneHotEncode::new(3)));

    let batches = loader.epoch_batches("x", "y").unwrap();
    assert_eq!(batches.len(), 1);

    let y = batches[0].get("y").unwrap();
    assert_eq!(y.dims(), &[4, 3]); // 4 samples, 3-class one-hot

    let y_data = y.to_f64_vec().unwrap();
    // Labels: 0%3=0, 1%3=1, 2%3=2, 3%3=0
    // One-hot: [1,0,0], [0,1,0], [0,0,1], [1,0,0]
    assert_eq!(
        y_data,
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,]
    );
}

#[test]
fn test_dataloader_iter_batches() {
    let ds = ToyDataset::new(7);
    let config = DataLoaderConfig::default()
        .batch_size(3)
        .shuffle(false)
        .dtype(DType::F64);
    let mut loader = DataLoader::<CpuBackend>::new(&ds, CpuDevice, config);

    let mut count = 0;
    let mut total_samples = 0;
    for batch in loader.iter_batches("x", "y") {
        let batch = batch.unwrap();
        let x = batch.get("x").unwrap();
        total_samples += x.dims()[0];
        count += 1;
    }
    assert_eq!(count, 3); // 7/3 = 2 full + 1 partial
    assert_eq!(total_samples, 7);
}

#[test]
fn test_dataloader_shuffle_changes_order() {
    let ds = ToyDataset::new(100);
    let config = DataLoaderConfig::default()
        .batch_size(100)
        .shuffle(true)
        .dtype(DType::F64);
    let mut loader = DataLoader::<CpuBackend>::new(&ds, CpuDevice, config);

    let batch1 = loader.epoch_batches("x", "y").unwrap();
    let data1 = batch1[0].get("x").unwrap().to_f64_vec().unwrap();

    let batch2 = loader.epoch_batches("x", "y").unwrap();
    let data2 = batch2[0].get("x").unwrap().to_f64_vec().unwrap();

    // With 100 samples, the probability of two shuffles being identical is negligible
    assert_ne!(data1, data2, "shuffle should produce different orderings");
}

// MNIST integration tests (using synthetic data)

#[test]
fn test_mnist_synthetic_as_dataset() {
    let ds = MnistDataset::synthetic(50, MnistSplit::Train);
    assert_eq!(ds.len(), 50);
    assert_eq!(ds.name(), "MNIST-train");

    let s = ds.get(0);
    assert_eq!(s.features.len(), 784); // 28*28
    assert_eq!(s.feature_shape, vec![784]);
    assert!(s.target[0] >= 0.0 && s.target[0] <= 9.0);
}

#[test]
fn test_mnist_with_dataloader() {
    let ds = MnistDataset::synthetic(100, MnistSplit::Train);
    let config = DataLoaderConfig::default()
        .batch_size(32)
        .shuffle(false)
        .dtype(DType::F32);
    let mut loader = DataLoader::<CpuBackend>::new(&ds, CpuDevice, config)
        .with_transform(Box::new(Normalize::new(255.0)));

    let batches = loader.epoch_batches("images", "labels").unwrap();
    assert_eq!(batches.len(), 4); // ceil(100/32) = 4

    // First batch: 32 images
    let imgs = batches[0].get("images").unwrap();
    assert_eq!(imgs.dims(), &[32, 784]);

    let lbls = batches[0].get("labels").unwrap();
    assert_eq!(lbls.dims(), &[32, 1]);

    // Check normalization: all pixel values should be in [0, 1]
    let img_data = imgs.to_f64_vec().unwrap();
    for &v in &img_data {
        assert!(v >= 0.0 && v <= 1.0, "pixel {v} not in [0,1]");
    }
}

#[test]
fn test_mnist_with_onehot() {
    let ds = MnistDataset::synthetic(16, MnistSplit::Test);
    let config = DataLoaderConfig::default()
        .batch_size(16)
        .shuffle(false)
        .dtype(DType::F64);
    let mut loader = DataLoader::<CpuBackend>::new(&ds, CpuDevice, config)
        .with_transform(Box::new(Normalize::new(255.0)))
        .with_transform(Box::new(OneHotEncode::new(10)));

    let batches = loader.epoch_batches("x", "target").unwrap();
    let target = batches[0].get("target").unwrap();
    assert_eq!(target.dims(), &[16, 10]); // 16 samples, 10-class one-hot
}

#[test]
fn test_mnist_idx_parsing_full_roundtrip() {
    // Build a tiny MNIST-like dataset with 3 4×4 images
    let img0 = vec![0u8; 16];
    let img1 = vec![128u8; 16];
    let img2 = vec![255u8; 16];
    let img_bytes = build_idx3_bytes(&[&img0, &img1, &img2], 4, 4);
    let lbl_bytes = build_idx1_bytes(&[0, 5, 9]);

    let ds = MnistDataset::from_raw(&img_bytes, &lbl_bytes, MnistSplit::Train).unwrap();
    assert_eq!(ds.num_samples(), 3);
    assert_eq!(ds.image_dims(), (4, 4));

    // Use DataLoader to batch all 3
    let config = DataLoaderConfig::default()
        .batch_size(3)
        .shuffle(false)
        .dtype(DType::F64);
    let mut loader = DataLoader::<CpuBackend>::new(&ds, CpuDevice, config);
    let batches = loader.epoch_batches("x", "y").unwrap();

    let x = batches[0].get("x").unwrap();
    assert_eq!(x.dims(), &[3, 16]); // 3 images, 4×4=16 pixels each

    let y = batches[0].get("y").unwrap();
    let y_data = y.to_f64_vec().unwrap();
    assert_eq!(y_data, vec![0.0, 5.0, 9.0]);
}
