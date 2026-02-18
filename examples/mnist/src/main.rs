// =============================================================================
// MNIST Handwritten Digit Classification — Shrew Deep Learning Library
// =============================================================================
//
// This example trains a simple MLP (Multi-Layer Perceptron) on the MNIST
// dataset of handwritten digits (0–9).
//
// Architecture:
//   Input(784) → Linear(784,128) → ReLU → Linear(128,10) → CrossEntropy
//
// Features demonstrated:
//   1. MnistDataset loading (synthetic or real IDX files)
//   2. DataLoader with batching, shuffling, and transforms
//   3. Normalize + OneHotEncode transforms
//   4. Manual training loop with Adam optimizer
//   5. Cross-entropy loss for multi-class classification
//   6. Evaluation accuracy on test set
//
// Usage:
//   cargo run -p mnist-example                            # synthetic data (quick demo)
//   cargo run -p mnist-example -- --data-dir path/to/mnist  # real MNIST data
//   cargo run -p mnist-example -- --epochs 10             # more epochs
//   cargo run -p mnist-example -- --samples 5000          # more training samples

use shrew::nn::Module;
use shrew::prelude::*;
use shrew_data::mnist::MnistSplit;
use shrew_data::{
    transform::{Normalize, OneHotEncode},
    DataLoader, DataLoaderConfig, MnistDataset,
};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

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
            batch_size: 64,
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
                println!("MNIST training example for Shrew");
                println!();
                println!("Options:");
                println!("  --data-dir <path>   Path to MNIST IDX files");
                println!("  --epochs <n>        Number of training epochs (default: 5)");
                println!("  --batch-size <n>    Batch size (default: 64)");
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

// ─────────────────────────────────────────────────────────────────────────────
// Forward pass helper
// ─────────────────────────────────────────────────────────────────────────────

/// Forward pass: Input → Linear(784,128) → ReLU → Linear(128,10)
///
/// params layout:
///   [0] = w1  [128, 784]
///   [1] = b1  [1, 128]
///   [2] = w2  [10, 128]
///   [3] = b2  [1, 10]
fn forward(x: &CpuTensor, params: &[CpuTensor]) -> shrew::Result<CpuTensor> {
    // Layer 1: hidden = ReLU(x @ W1^T + b1)
    let w1 = &params[0];
    let b1 = &params[1];
    let wt1 = w1.t()?.contiguous()?;
    let h = x.matmul(&wt1)?.add(b1)?.relu()?;

    // Layer 2: logits = h @ W2^T + b2
    let w2 = &params[2];
    let b2 = &params[3];
    let wt2 = w2.t()?.contiguous()?;
    let logits = h.matmul(&wt2)?.add(b2)?;

    Ok(logits)
}

/// Compute classification accuracy: fraction of correct predictions.
fn accuracy(logits: &CpuTensor, targets_onehot: &CpuTensor) -> shrew::Result<f64> {
    let logits_data = logits.to_f64_vec()?;
    let targets_data = targets_onehot.to_f64_vec()?;
    let batch = logits.dims()[0];
    let classes = logits.dims()[1];

    let mut correct = 0usize;
    for b in 0..batch {
        // Predicted class = argmax of logits
        let row = &logits_data[b * classes..(b + 1) * classes];
        let pred = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        // True class = argmax of one-hot target
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

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

fn main() -> shrew::Result<()> {
    let cfg = parse_args();
    let dev = CpuDevice;

    println!("=== Shrew — MNIST Handwritten Digit Classification ===");
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
        .with_transform(Box::new(OneHotEncode::new(10)));

    let mut test_loader = DataLoader::<CpuBackend>::new(&test_ds, CpuDevice, test_config)
        .with_transform(Box::new(Normalize::new(255.0)))
        .with_transform(Box::new(OneHotEncode::new(10)));

    println!("DataLoader:");
    println!("  Batch size: {}", cfg.batch_size);
    println!("  Train batches/epoch: {}", train_loader.num_batches());
    println!("  Test batches: {}", test_loader.num_batches());
    println!("  Transforms: Normalize(1/255) → OneHotEncode(10)");
    println!();

    // ─────────────────────────────────────────────────────────────────────
    // 3. Create network layers
    // ─────────────────────────────────────────────────────────────────────
    let l1 = Linear::<CpuBackend>::new(784, 128, true, DType::F64, &dev)?;
    let l2 = Linear::<CpuBackend>::new(128, 10, true, DType::F64, &dev)?;

    let total_params: usize = l1
        .parameters()
        .iter()
        .chain(l2.parameters().iter())
        .map(|t| t.elem_count())
        .sum();

    println!("Network: Linear(784→128) → ReLU → Linear(128→10)");
    println!(
        "  Layer 1: weight {:?} + bias {:?}",
        l1.weight().dims(),
        l1.bias().unwrap().dims()
    );
    println!(
        "  Layer 2: weight {:?} + bias {:?}",
        l2.weight().dims(),
        l2.bias().unwrap().dims()
    );
    println!("  Total parameters: {total_params}");
    println!();

    // ─────────────────────────────────────────────────────────────────────
    // 4. Set up optimizer
    // ─────────────────────────────────────────────────────────────────────
    let mut all_params: Vec<CpuTensor> = Vec::new();
    all_params.extend(l1.parameters());
    all_params.extend(l2.parameters());

    let mut optimizer = Adam::<CpuBackend>::new(all_params, cfg.lr);

    // Load pre-trained weights if requested
    if let Some(ref load_path) = cfg.load_path {
        println!("Loading weights from: {load_path}");
        let loaded = shrew::checkpoint::load_tensors::<CpuBackend>(load_path, &dev)?;
        // Map loaded params back into optimizer
        // Order: w1, b1, w2, b2
        let param_names = ["w1", "b1", "w2", "b2"];
        for (name, tensor) in &loaded {
            if let Some(idx) = param_names.iter().position(|&n| n == name) {
                let params = optimizer.params_mut();
                params[idx] = tensor.clone().set_variable();
            }
        }
        println!("  Loaded {} parameters", loaded.len());
    }

    println!("Optimizer: Adam (lr={}, beta1=0.9, beta2=0.999)", cfg.lr);
    println!();

    // ─────────────────────────────────────────────────────────────────────
    // 5. Training loop
    // ─────────────────────────────────────────────────────────────────────
    println!("Training for {} epochs...", cfg.epochs);
    println!("{:-<60}", "");

    for epoch in 0..cfg.epochs {
        let batches = train_loader.epoch_batches("input", "target")?;
        let num_batches = batches.len();
        let mut epoch_loss = 0.0;
        let mut epoch_correct = 0usize;
        let mut epoch_total = 0usize;

        for batch in &batches {
            let x = batch.get("input").unwrap();
            let y = batch.get("target").unwrap();

            // Forward
            let logits = forward(x, optimizer.params())?;

            // Loss
            let loss = cross_entropy_loss(&logits, y)?;
            epoch_loss += loss.to_scalar_f64()?;

            // Accuracy
            let logits_data = logits.to_f64_vec()?;
            let y_data = y.to_f64_vec()?;
            let bs = x.dims()[0];
            let classes = 10;
            for b in 0..bs {
                let row = &logits_data[b * classes..(b + 1) * classes];
                let pred = row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0;
                let tgt_row = &y_data[b * classes..(b + 1) * classes];
                let true_c = tgt_row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0;
                if pred == true_c {
                    epoch_correct += 1;
                }
                epoch_total += 1;
            }

            // Backward
            let grads = loss.backward()?;

            // Step
            optimizer.step(&grads)?;
        }

        let avg_loss = epoch_loss / num_batches as f64;
        let train_acc = epoch_correct as f64 / epoch_total as f64 * 100.0;

        println!(
            "  Epoch {}/{} | Loss: {:.4} | Train Acc: {:.1}%",
            epoch + 1,
            cfg.epochs,
            avg_loss,
            train_acc
        );
    }

    println!("{:-<60}", "");
    println!();

    // ─────────────────────────────────────────────────────────────────────
    // 6. Evaluate on test set
    // ─────────────────────────────────────────────────────────────────────
    println!("Evaluating on test set...");
    let test_batches = test_loader.epoch_batches("input", "target")?;
    let mut test_correct = 0usize;
    let mut test_total = 0usize;
    let mut test_loss = 0.0;

    for batch in &test_batches {
        let x = batch.get("input").unwrap();
        let y = batch.get("target").unwrap();

        let logits = forward(x, optimizer.params())?;
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
        let logits = forward(x, optimizer.params())?;

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
        let params = optimizer.params();
        let named: Vec<(String, CpuTensor)> = vec![
            ("w1".to_string(), params[0].clone()),
            ("b1".to_string(), params[1].clone()),
            ("w2".to_string(), params[2].clone()),
            ("b2".to_string(), params[3].clone()),
        ];
        shrew::checkpoint::save_tensors(save_path, &named)?;
        println!("Saved {} parameters to: {save_path}", named.len());
    }

    println!();
    println!("=== Done! ===");

    Ok(())
}
