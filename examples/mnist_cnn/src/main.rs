// MNIST CNN — Convolutional Neural Network for Digit Classification (Shrew)
//
// This example trains a CNN on the MNIST handwritten-digit dataset (0–9).
//
// Architecture:
//   Conv2d(1→16, 3×3, pad=1) → BatchNorm2d(16) → ReLU → MaxPool2d(2×2)
//   Conv2d(16→32, 3×3, pad=1) → BatchNorm2d(32) → ReLU → MaxPool2d(2×2)
//   Flatten → Linear(32·7·7 → 128) → ReLU → Linear(128 → 10) → CrossEntropy
//
// Features demonstrated:
//   1. Conv2d, BatchNorm2d, MaxPool2d, Flatten layers
//   2. ReshapeFeatures transform (784 → [1, 28, 28])
//   3. Training loop with CNN modules
//   4. BatchNorm train/eval mode switching
//   5. Save/load checkpoints
//
// Usage:
//   cargo run -p mnist-cnn-example                               # synthetic data
//   cargo run -p mnist-cnn-example -- --data-dir path/to/mnist   # real MNIST data
//   cargo run -p mnist-cnn-example -- --epochs 10                # more epochs
//   cargo run -p mnist-cnn-example -- --save model.shrew         # save after training

use shrew::nn::Module;
use shrew::prelude::*;
use shrew_data::mnist::MnistSplit;
use shrew_data::{
    transform::{Normalize, OneHotEncode, ReshapeFeatures},
    DataLoader, DataLoaderConfig, MnistDataset,
};

// Configuration

struct Config {
    data_dir: Option<String>,
    epochs: usize,
    batch_size: usize,
    lr: f64,
    train_samples: usize,
    test_samples: usize,
    save_path: Option<String>,
    load_path: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            data_dir: None,
            epochs: 5,
            batch_size: 32,
            lr: 0.001,
            train_samples: 2000,
            test_samples: 500,
            save_path: None,
            load_path: None,
        }
    }
}

fn parse_args() -> Config {
    let mut cfg = Config::default();
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--data-dir" => {
                i += 1;
                cfg.data_dir = Some(args[i].clone());
            }
            "--epochs" => {
                i += 1;
                cfg.epochs = args[i].parse().expect("invalid --epochs");
            }
            "--batch-size" => {
                i += 1;
                cfg.batch_size = args[i].parse().expect("invalid --batch-size");
            }
            "--lr" => {
                i += 1;
                cfg.lr = args[i].parse().expect("invalid --lr");
            }
            "--samples" => {
                i += 1;
                cfg.train_samples = args[i].parse().expect("invalid --samples");
                cfg.test_samples = cfg.train_samples / 4;
            }
            "--save" => {
                i += 1;
                cfg.save_path = Some(args[i].clone());
            }
            "--load" => {
                i += 1;
                cfg.load_path = Some(args[i].clone());
            }
            "--help" | "-h" => {
                println!("MNIST CNN training example for Shrew");
                println!();
                println!("Options:");
                println!("  --data-dir <path>   Path to MNIST IDX files");
                println!("  --epochs <n>        Number of training epochs (default: 5)");
                println!("  --batch-size <n>    Batch size (default: 32)");
                println!("  --lr <f>            Learning rate (default: 0.001)");
                println!("  --samples <n>       Synthetic training samples (default: 2000)");
                println!("  --save <path>       Save trained weights to .shrew checkpoint");
                println!("  --load <path>       Load pre-trained weights from .shrew checkpoint");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }
    cfg
}

// CNN Model

/// A simple CNN for MNIST:
///   conv1 → bn1 → relu → pool1
///   conv2 → bn2 → relu → pool2
///   flatten → fc1 → relu → fc2
struct MnistCnn {
    conv1: Conv2d<CpuBackend>,
    bn1: BatchNorm2d<CpuBackend>,
    pool1: MaxPool2d,

    conv2: Conv2d<CpuBackend>,
    bn2: BatchNorm2d<CpuBackend>,
    pool2: MaxPool2d,

    flatten: Flatten,
    fc1: Linear<CpuBackend>,
    fc2: Linear<CpuBackend>,
}

impl MnistCnn {
    fn new(dev: &CpuDevice) -> shrew::Result<Self> {
        let dtype = DType::F64;
        Ok(MnistCnn {
            // Block 1: 1→16 channels, 28×28 → 14×14 after pool
            conv1: Conv2d::new(1, 16, [3, 3], [1, 1], [1, 1], true, dtype, dev)?,
            bn1: BatchNorm2d::new(16, 1e-5, 0.1, dtype, dev)?,
            pool1: MaxPool2d::new([2, 2], [2, 2], [0, 0]),

            // Block 2: 16→32 channels, 14×14 → 7×7 after pool
            conv2: Conv2d::new(16, 32, [3, 3], [1, 1], [1, 1], true, dtype, dev)?,
            bn2: BatchNorm2d::new(32, 1e-5, 0.1, dtype, dev)?,
            pool2: MaxPool2d::new([2, 2], [2, 2], [0, 0]),

            // Classifier: 32·7·7 = 1568 → 128 → 10
            flatten: Flatten::default_flat(),
            fc1: Linear::new(32 * 7 * 7, 128, true, dtype, dev)?,
            fc2: Linear::new(128, 10, true, dtype, dev)?,
        })
    }

    fn forward(&self, x: &CpuTensor) -> shrew::Result<CpuTensor> {
        // Block 1
        let x = self.conv1.forward(x)?;
        let x = self.bn1.forward(&x)?;
        let x = x.relu()?;
        let x = self.pool1.forward(&x)?;

        // Block 2
        let x = self.conv2.forward(&x)?;
        let x = self.bn2.forward(&x)?;
        let x = x.relu()?;
        let x = self.pool2.forward(&x)?;

        // Classifier
        let x = self.flatten.forward(&x)?;
        let x = self.fc1.forward(&x)?;
        let x = x.relu()?;
        let x = self.fc2.forward(&x)?;

        Ok(x)
    }

    /// Collect all learnable parameters.
    fn parameters(&self) -> Vec<CpuTensor> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.bn1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.bn2.parameters());
        params.extend(self.fc1.parameters());
        params.extend(self.fc2.parameters());
        params
    }

    /// Named parameters for checkpoint save/load.
    fn named_parameters(&self) -> Vec<(String, CpuTensor)> {
        let mut named = Vec::new();
        for (i, p) in self.conv1.parameters().iter().enumerate() {
            named.push((format!("conv1.{i}"), p.clone()));
        }
        for (i, p) in self.bn1.parameters().iter().enumerate() {
            named.push((format!("bn1.{i}"), p.clone()));
        }
        for (i, p) in self.conv2.parameters().iter().enumerate() {
            named.push((format!("conv2.{i}"), p.clone()));
        }
        for (i, p) in self.bn2.parameters().iter().enumerate() {
            named.push((format!("bn2.{i}"), p.clone()));
        }
        for (i, p) in self.fc1.parameters().iter().enumerate() {
            named.push((format!("fc1.{i}"), p.clone()));
        }
        for (i, p) in self.fc2.parameters().iter().enumerate() {
            named.push((format!("fc2.{i}"), p.clone()));
        }
        named
    }

    /// Switch to eval mode (BatchNorm uses running stats).
    fn eval(&self) {
        self.bn1.eval();
        self.bn2.eval();
    }

    /// Switch to training mode (BatchNorm uses batch stats).
    fn train(&self) {
        self.bn1.train();
        self.bn2.train();
    }

    fn total_params(&self) -> usize {
        self.parameters().iter().map(|t| t.elem_count()).sum()
    }
}

// Helpers

/// Compute argmax-based classification accuracy.
fn accuracy(logits: &CpuTensor, targets_onehot: &CpuTensor) -> shrew::Result<f64> {
    let logits_data = logits.to_f64_vec()?;
    let targets_data = targets_onehot.to_f64_vec()?;
    let batch = logits.dims()[0];
    let classes = logits.dims()[1];

    let mut correct = 0usize;
    for b in 0..batch {
        let row = &logits_data[b * classes..(b + 1) * classes];
        let pred = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        let tgt_row = &targets_data[b * classes..(b + 1) * classes];
        let true_class = tgt_row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        if pred == true_class {
            correct += 1;
        }
    }
    Ok(correct as f64 / batch as f64)
}

// Main

fn main() -> shrew::Result<()> {
    let cfg = parse_args();
    let dev = CpuDevice;

    println!("=== Shrew — MNIST CNN Digit Classification ===");
    println!();

    // ─────────────────────────────────────────────────────────────────────
    // 1. Load dataset
    // ─────────────────────────────────────────────────────────────────────
    let (train_ds, test_ds) = match &cfg.data_dir {
        Some(dir) => {
            println!("Loading MNIST from: {dir}");
            let train = MnistDataset::load(dir, MnistSplit::Train)
                .map_err(|e| shrew_core::Error::msg(format!("Failed to load MNIST: {e}")))?;
            let test = MnistDataset::load(dir, MnistSplit::Test)
                .map_err(|e| shrew_core::Error::msg(format!("Failed to load MNIST: {e}")))?;
            println!(
                "  Train: {} images ({}x{})",
                train.num_samples(),
                train.image_dims().0,
                train.image_dims().1
            );
            println!("  Test:  {} images", test.num_samples());
            (train, test)
        }
        None => {
            println!(
                "Using synthetic MNIST data ({} train, {} test)",
                cfg.train_samples, cfg.test_samples
            );
            println!("  Tip: use --data-dir <path> for real MNIST");
            let train = MnistDataset::synthetic(cfg.train_samples, MnistSplit::Train);
            let test = MnistDataset::synthetic(cfg.test_samples, MnistSplit::Test);
            (train, test)
        }
    };
    println!();

    // ─────────────────────────────────────────────────────────────────────
    // 2. Create DataLoaders with transforms
    //    Key difference from MLP: we reshape [784] → [1, 28, 28] for CNN
    // ─────────────────────────────────────────────────────────────────────
    let train_config = DataLoaderConfig::default()
        .batch_size(cfg.batch_size)
        .shuffle(true)
        .dtype(DType::F64);

    let test_config = DataLoaderConfig::default()
        .batch_size(cfg.batch_size)
        .shuffle(false)
        .dtype(DType::F64);

    let mut train_loader = DataLoader::<CpuBackend>::new(&train_ds, CpuDevice, train_config)
        .with_transform(Box::new(Normalize::new(255.0)))
        .with_transform(Box::new(ReshapeFeatures::new(vec![1, 28, 28])))
        .with_transform(Box::new(OneHotEncode::new(10)));

    let mut test_loader = DataLoader::<CpuBackend>::new(&test_ds, CpuDevice, test_config)
        .with_transform(Box::new(Normalize::new(255.0)))
        .with_transform(Box::new(ReshapeFeatures::new(vec![1, 28, 28])))
        .with_transform(Box::new(OneHotEncode::new(10)));

    println!("DataLoader:");
    println!("  Batch size: {}", cfg.batch_size);
    println!("  Train batches/epoch: {}", train_loader.num_batches());
    println!("  Test batches: {}", test_loader.num_batches());
    println!("  Transforms: Normalize(1/255) → Reshape([1,28,28]) → OneHotEncode(10)");
    println!();

    // ─────────────────────────────────────────────────────────────────────
    // 3. Build the CNN model
    // ─────────────────────────────────────────────────────────────────────
    let model = MnistCnn::new(&dev)?;

    println!("Architecture:");
    println!("  Conv2d(1→16, 3×3, pad=1) → BN(16) → ReLU → MaxPool(2×2)");
    println!("  Conv2d(16→32, 3×3, pad=1) → BN(32) → ReLU → MaxPool(2×2)");
    println!("  Flatten → Linear(1568→128) → ReLU → Linear(128→10)");
    println!("  Total parameters: {}", model.total_params());
    println!();

    // ─────────────────────────────────────────────────────────────────────
    // 4. Set up optimizer
    // ─────────────────────────────────────────────────────────────────────
    let all_params = model.parameters();
    let mut optimizer = Adam::<CpuBackend>::new(all_params, cfg.lr);

    // Load pre-trained weights if requested
    if let Some(ref load_path) = cfg.load_path {
        println!("Loading weights from: {load_path}");
        let loaded = shrew::checkpoint::load_tensors::<CpuBackend>(load_path, &dev)?;
        let param_names = model.named_parameters();
        for (name, tensor) in &loaded {
            if let Some(idx) = param_names.iter().position(|(n, _)| n == name) {
                let params = optimizer.params_mut();
                params[idx] = tensor.clone().set_variable();
            }
        }
        println!("  Loaded {} parameters", loaded.len());
    }

    println!("Optimizer: Adam (lr={})", cfg.lr);
    println!();

    // ─────────────────────────────────────────────────────────────────────
    // 5. Training loop
    // ─────────────────────────────────────────────────────────────────────
    println!("Training for {} epochs...", cfg.epochs);
    println!("{:-<65}", "");

    for epoch in 0..cfg.epochs {
        model.train();
        let batches = train_loader.epoch_batches("input", "target")?;
        let num_batches = batches.len();
        let mut epoch_loss = 0.0;
        let mut epoch_correct = 0usize;
        let mut epoch_total = 0usize;

        for (batch_idx, batch) in batches.iter().enumerate() {
            let x = batch.get("input").unwrap(); // [B, 1, 28, 28]
            let y = batch.get("target").unwrap(); // [B, 10] one-hot

            // Forward
            let logits = model.forward(x)?; // [B, 10]

            // Loss
            let loss = cross_entropy_loss(&logits, y)?;
            epoch_loss += loss.to_scalar_f64()?;

            // Accuracy
            let acc = accuracy(&logits, y)?;
            let bs = x.dims()[0];
            epoch_correct += (acc * bs as f64) as usize;
            epoch_total += bs;

            // Backward
            let grads = loss.backward()?;

            // Step
            optimizer.step(&grads)?;

            // Progress every 10 batches
            if (batch_idx + 1) % 10 == 0 || batch_idx + 1 == num_batches {
                print!(
                    "\r  Epoch {}/{} | Batch {}/{} | Loss: {:.4}",
                    epoch + 1,
                    cfg.epochs,
                    batch_idx + 1,
                    num_batches,
                    epoch_loss / (batch_idx + 1) as f64
                );
            }
        }

        let avg_loss = epoch_loss / num_batches as f64;
        let train_acc = epoch_correct as f64 / epoch_total as f64 * 100.0;
        println!(
            "\r  Epoch {}/{} | Loss: {:.4} | Train Acc: {:.1}%          ",
            epoch + 1,
            cfg.epochs,
            avg_loss,
            train_acc
        );
    }

    println!("{:-<65}", "");
    println!();

    // ─────────────────────────────────────────────────────────────────────
    // 6. Evaluate on test set (eval mode — BatchNorm uses running stats)
    // ─────────────────────────────────────────────────────────────────────
    println!("Evaluating on test set...");
    model.eval();
    let test_batches = test_loader.epoch_batches("input", "target")?;
    let mut test_correct = 0usize;
    let mut test_total = 0usize;
    let mut test_loss = 0.0;

    for batch in &test_batches {
        let x = batch.get("input").unwrap();
        let y = batch.get("target").unwrap();

        let logits = model.forward(x)?;
        let loss = cross_entropy_loss(&logits, y)?;
        test_loss += loss.to_scalar_f64()?;

        let acc = accuracy(&logits, y)?;
        test_correct += (acc * x.dims()[0] as f64) as usize;
        test_total += x.dims()[0];
    }

    let test_acc = test_correct as f64 / test_total as f64 * 100.0;
    let avg_test_loss = test_loss / test_batches.len() as f64;

    println!("  Test Loss: {:.4}", avg_test_loss);
    println!(
        "  Test Accuracy: {:.1}% ({}/{})",
        test_acc, test_correct, test_total
    );
    println!();

    // ─────────────────────────────────────────────────────────────────────
    // 7. Show sample predictions
    // ─────────────────────────────────────────────────────────────────────
    if let Some(batch) = test_batches.first() {
        let x = batch.get("input").unwrap();
        let y = batch.get("target").unwrap();
        let logits = model.forward(x)?;

        let logits_data = logits.to_f64_vec()?;
        let y_data = y.to_f64_vec()?;
        let n_show = 10.min(x.dims()[0]);

        println!("Sample predictions (first {n_show}):");
        for i in 0..n_show {
            let row = &logits_data[i * 10..(i + 1) * 10];
            let pred = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            let tgt_row = &y_data[i * 10..(i + 1) * 10];
            let true_c = tgt_row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            let mark = if pred == true_c { "OK" } else { "MISS" };
            println!("  [{mark:>4}] predicted: {pred}  actual: {true_c}");
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // 8. Save trained weights
    // ─────────────────────────────────────────────────────────────────────
    if let Some(ref save_path) = cfg.save_path {
        let named = model.named_parameters();
        shrew::checkpoint::save_tensors(save_path, &named)?;
        println!("Saved {} parameters to: {save_path}", named.len());
    }

    println!();
    println!("=== Done! ===");

    Ok(())
}
