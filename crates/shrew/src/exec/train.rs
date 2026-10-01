// =============================================================================
// Trainer — Training loop runner powered by the graph executor
// =============================================================================
//
// Reads the @training block from an IrProgram and runs the training loop:
//   1. Initialize parameters
//   2. For each epoch:
//      a. Forward pass through the model graph
//      b. Compute loss
//      c. Backward pass (autograd)
//      d. Optimizer step
//      e. Log metrics
//
// Uses the Executor for graph evaluation and the shrew-optim optimizers.

use std::collections::HashMap;

use shrew_core::backend::Backend;
use shrew_core::error::Result;
use shrew_core::tensor::Tensor;

use shrew_ir::graph::IrProgram;

use shrew_data::dataset::Dataset;
use shrew_nn::{cross_entropy_loss, mse_loss};
use shrew_optim::{Adam, AdamW, Optimizer, SGD};

use super::engine::{Executor, RuntimeConfig};

// ─────────────────────────────────────────────────────────────────────────────
// Training result types
// ─────────────────────────────────────────────────────────────────────────────

/// Summary of a full training run.
#[derive(Debug, Clone)]
pub struct TrainResult {
    /// Per-epoch logs.
    pub epochs: Vec<EpochLog>,
    /// Final loss value.
    pub final_loss: f64,
}

/// Log for a single training epoch.
#[derive(Debug, Clone)]
pub struct EpochLog {
    /// Epoch number (0-indexed).
    pub epoch: usize,
    /// Average loss for this epoch.
    pub loss: f64,
}

impl std::fmt::Display for TrainResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Training complete — {} epochs", self.epochs.len())?;
        for log in &self.epochs {
            writeln!(f, "  epoch {}: loss = {:.6}", log.epoch, log.loss)?;
        }
        write!(f, "  final loss: {:.6}", self.final_loss)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Trainer
// ─────────────────────────────────────────────────────────────────────────────

/// High-level training loop runner.
///
/// Reads the @training configuration from an IrProgram and orchestrates
/// forward, backward, and optimizer steps.
pub struct Trainer<B: Backend> {
    /// The graph executor.
    pub executor: Executor<B>,
    /// Name of the model graph.
    model_graph: String,
    /// Loss function name.
    loss_fn: String,
    /// Number of epochs.
    epochs: usize,
    /// Batch size (for reference, data batching is external).
    pub batch_size: usize,
}

impl<B: Backend> Trainer<B> {
    /// Create a Trainer from an IrProgram.
    ///
    /// Reads the @training block and fails if no training config exists.
    pub fn from_program(
        program: IrProgram,
        device: B::Device,
        config: RuntimeConfig,
    ) -> Result<Self> {
        let training = program.training.as_ref().ok_or_else(|| {
            shrew_core::Error::msg("Program has no @training block. Cannot create Trainer.")
        })?;

        let model_graph = training.model_graph.clone();
        let loss_fn = training.loss.clone();
        let epochs = training.epochs as usize;
        let batch_size = training.batch_size as usize;

        let executor = Executor::<B>::new(program, device, config)?;

        Ok(Self {
            executor,
            model_graph,
            loss_fn,
            epochs,
            batch_size,
        })
    }

    /// Run the training loop with an iterator of input batches.
    ///
    /// Each batch is a `HashMap<String, Tensor<B>>` mapping input names to
    /// tensors. Returns a `TrainResult` with per-epoch loss logs.
    ///
    /// `targets_key` is the name of the target tensor in each batch map.
    pub fn train(
        &mut self,
        data: &[HashMap<String, Tensor<B>>],
        targets_key: &str,
    ) -> Result<TrainResult> {
        let training = self
            .executor
            .program()
            .training
            .as_ref()
            .ok_or_else(|| shrew_core::Error::msg("No @training config"))?;

        // Create optimizer
        let params = self.executor.graph_params(&self.model_graph);
        let lr = training.optimizer.lr;

        let mut optimizer: Box<dyn Optimizer<B>> = match training.optimizer.kind.as_str() {
            "SGD" | "sgd" => {
                let momentum = training
                    .optimizer
                    .extra
                    .get("momentum")
                    .and_then(|v| match v {
                        shrew_ir::graph::ConfigValue::Float(f) => Some(*f),
                        _ => None,
                    })
                    .unwrap_or(0.0);
                Box::new(SGD::new(params.clone(), lr, momentum, 0.0))
            }
            "Adam" | "adam" => Box::new(Adam::new(params.clone(), lr)),
            "AdamW" | "adamw" => Box::new(AdamW::new(params.clone(), lr, 0.01)),
            other => {
                return Err(shrew_core::Error::msg(format!(
                    "Unknown optimizer type: '{}'. Supported: SGD, Adam, AdamW",
                    other
                )));
            }
        };

        // Set training mode
        self.executor.config_mut().training = true;

        let mut epoch_logs = Vec::new();

        for epoch in 0..self.epochs {
            let mut epoch_loss = 0.0;
            let mut n_batches = 0;

            for batch in data {
                // Forward pass
                let result = self.executor.run(&self.model_graph, batch)?;

                // Get model output
                let output = result
                    .output()
                    .ok_or_else(|| shrew_core::Error::msg("Model graph produced no output"))?;

                // Get target from batch
                let target = batch.get(targets_key).ok_or_else(|| {
                    shrew_core::Error::msg(format!(
                        "Target tensor '{}' not found in batch",
                        targets_key
                    ))
                })?;

                // Compute loss
                let loss = match self.loss_fn.as_str() {
                    "cross_entropy" | "CrossEntropy" => cross_entropy_loss(output, target)?,
                    "mse" | "mse_loss" | "MSE" => mse_loss(output, target)?,
                    other => {
                        return Err(shrew_core::Error::msg(format!(
                            "Unknown loss function: '{}'. Supported: cross_entropy, mse",
                            other
                        )));
                    }
                };

                let loss_val = loss.to_scalar_f64()?;
                epoch_loss += loss_val;
                n_batches += 1;

                // Backward pass
                let grads = loss.backward()?;

                // Optimizer step → new parameters
                let new_params = optimizer.step(&grads)?;

                // Update parameters in executor
                self.executor.update_params(&self.model_graph, &new_params);
            }

            let avg_loss = if n_batches > 0 {
                epoch_loss / n_batches as f64
            } else {
                0.0
            };

            epoch_logs.push(EpochLog {
                epoch,
                loss: avg_loss,
            });
        }

        let final_loss = epoch_logs.last().map_or(0.0, |l| l.loss);
        Ok(TrainResult {
            epochs: epoch_logs,
            final_loss,
        })
    }

    /// Run a single forward pass (inference mode).
    pub fn infer(&self, inputs: &HashMap<String, Tensor<B>>) -> Result<HashMap<String, Tensor<B>>> {
        let result = self.executor.run(&self.model_graph, inputs)?;
        Ok(result.outputs)
    }

    /// Get the model graph name.
    pub fn model_graph_name(&self) -> &str {
        &self.model_graph
    }

    /// Get the loss function name.
    pub fn loss_fn_name(&self) -> &str {
        &self.loss_fn
    }

    /// Get the number of epochs.
    pub fn epochs(&self) -> usize {
        self.epochs
    }

    /// Train using the dataset configuration specified in the @training block.
    pub fn train_auto(&mut self) -> Result<TrainResult> {
        let training = self
            .executor
            .program()
            .training
            .as_ref()
            .ok_or_else(|| shrew_core::Error::msg("No @training configuration found in program"))?;

        let dataset_cfg = training.dataset.as_ref().ok_or_else(|| {
            shrew_core::Error::msg(
                "No 'dataset' specified in @training block. Specify `dataset: \"path/to/data.csv\";` or `dataset: \"xor\";`"
            )
        })?;

        // 1. Determine input tensor name for the model graph
        let input_names = self
            .executor
            .program()
            .get_graph(&self.model_graph)
            .map(|g| g.inputs.iter().map(|id| g.node(*id).name.clone()).collect::<Vec<_>>())
            .unwrap_or_default();
        let input_key = input_names.first().cloned().unwrap_or_else(|| "x".to_string());
        let target_key = "targets";

        // 2. Load dataset batches
        let batches = self.load_batches(dataset_cfg, &input_key, target_key)?;

        // 3. Run training loop
        self.train(&batches, target_key)
    }

    fn load_batches(
        &self,
        dataset_cfg: &shrew_ir::graph::DatasetConfig,
        input_key: &str,
        target_key: &str,
    ) -> Result<Vec<HashMap<String, Tensor<B>>>> {
        let dtype = self.executor.config().default_dtype;
        let device = self.executor.device();

        if dataset_cfg.format == "xor" || dataset_cfg.path == "xor" {
            let x_vals = vec![
                0.0, 0.0,
                0.0, 1.0,
                1.0, 0.0,
                1.0, 1.0,
            ];
            let y_vals = vec![0.0, 1.0, 1.0, 0.0];

            let x_tensor = Tensor::<B>::from_f64_slice(&x_vals, (4, 2), dtype, device)?;
            let y_tensor = Tensor::<B>::from_f64_slice(&y_vals, (4, 1), dtype, device)?;

            let mut batch = HashMap::new();
            batch.insert(input_key.to_string(), x_tensor);
            batch.insert(target_key.to_string(), y_tensor);
            return Ok(vec![batch]);
        }

        let csv_cfg = shrew_data::csv_dataset::CsvConfig {
            has_header: dataset_cfg.has_header,
            feature_cols: dataset_cfg.feature_cols.clone(),
            target_cols: dataset_cfg.target_cols.clone(),
            delimiter: b',',
        };

        let csv_path = if std::path::Path::new(&dataset_cfg.path).exists() {
            dataset_cfg.path.clone()
        } else if let Ok(cwd) = std::env::current_dir() {
            let p1 = cwd.join(&dataset_cfg.path);
            let p2 = cwd.join("..").join(&dataset_cfg.path);
            let p3 = cwd.join("../..").join(&dataset_cfg.path);
            if p1.exists() {
                p1.to_string_lossy().to_string()
            } else if p2.exists() {
                p2.to_string_lossy().to_string()
            } else if p3.exists() {
                p3.to_string_lossy().to_string()
            } else {
                dataset_cfg.path.clone()
            }
        } else {
            dataset_cfg.path.clone()
        };

        let ds = shrew_data::csv_dataset::CsvDataset::load(&csv_path, csv_cfg)
            .map_err(|e| shrew_core::Error::msg(e))?;

        let n_samples = ds.len();
        if n_samples == 0 {
            return Err(shrew_core::Error::msg(format!(
                "Dataset at '{}' is empty",
                dataset_cfg.path
            )));
        }

        let batch_size = self.batch_size.max(1);
        let mut batches = Vec::new();

        for chunk_start in (0..n_samples).step_by(batch_size) {
            let chunk_end = (chunk_start + batch_size).min(n_samples);
            let b_len = chunk_end - chunk_start;

            let feat_len = ds.feature_shape().iter().product::<usize>();
            let tgt_len = ds.target_shape().iter().product::<usize>();

            let mut feat_buf = Vec::with_capacity(b_len * feat_len);
            let mut tgt_buf = Vec::with_capacity(b_len * tgt_len);

            for i in chunk_start..chunk_end {
                let sample = ds.get(i);
                feat_buf.extend_from_slice(&sample.features);
                tgt_buf.extend_from_slice(&sample.target);
            }

            let x_tensor = Tensor::<B>::from_f64_slice(&feat_buf, vec![b_len, feat_len], dtype, device)?;
            let y_tensor = Tensor::<B>::from_f64_slice(&tgt_buf, vec![b_len, tgt_len], dtype, device)?;

            let mut batch = HashMap::new();
            batch.insert(input_key.to_string(), x_tensor);
            batch.insert(target_key.to_string(), y_tensor);
            batches.push(batch);
        }

        Ok(batches)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience functions
// ─────────────────────────────────────────────────────────────────────────────

/// Parse, lower, validate, optimize, and prepare an executor from .sw source.
pub fn load_program<B: Backend>(
    source: &str,
    device: B::Device,
    config: RuntimeConfig,
) -> Result<Executor<B>> {
    let ast =
        shrew_ir::parse(source).map_err(|e| shrew_core::Error::msg(format!("Parse error: {e}")))?;
    let mut ir = shrew_ir::lower(&ast)
        .map_err(|e| shrew_core::Error::msg(format!("Lowering error: {e}")))?;

    // Validate
    if let Err(errors) = shrew_ir::validate(&ir) {
        let msg = errors
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        return Err(shrew_core::Error::msg(format!("Validation errors:\n{msg}")));
    }

    // Infer shapes and optimize
    shrew_ir::infer_shapes(&mut ir);
    shrew_ir::optimize(&mut ir);

    Executor::<B>::new(ir, device, config)
}

/// Parse, lower, validate, optimize, and prepare a trainer from .sw source.
pub fn load_trainer<B: Backend>(
    source: &str,
    device: B::Device,
    config: RuntimeConfig,
) -> Result<Trainer<B>> {
    let ast =
        shrew_ir::parse(source).map_err(|e| shrew_core::Error::msg(format!("Parse error: {e}")))?;
    let mut ir = shrew_ir::lower(&ast)
        .map_err(|e| shrew_core::Error::msg(format!("Lowering error: {e}")))?;

    if let Err(errors) = shrew_ir::validate(&ir) {
        let msg = errors
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        return Err(shrew_core::Error::msg(format!("Validation errors:\n{msg}")));
    }

    shrew_ir::infer_shapes(&mut ir);
    shrew_ir::optimize(&mut ir);

    Trainer::<B>::from_program(ir, device, config)
}

/// Parse, lower, and train a .sw program directly from a file path using its embedded configuration.
pub fn train_file<B: Backend>(
    path: &str,
    device: B::Device,
    config: RuntimeConfig,
) -> Result<(Trainer<B>, TrainResult)> {
    let source = std::fs::read_to_string(path)
        .map_err(|e| shrew_core::Error::msg(format!("Failed to read '{path}': {e}")))?;
    let mut trainer = load_trainer::<B>(&source, device, config)?;
    let result = trainer.train_auto()?;
    Ok((trainer, result))
}

