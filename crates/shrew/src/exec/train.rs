// Trainer — Training loop runner powered by the graph executor
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

use shrew_nn::{cross_entropy_loss, mse_loss};
use shrew_optim::{Adam, AdamW, Optimizer, SGD};

use super::engine::{Executor, RuntimeConfig};

// Training result types

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

// Trainer

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
}

// Convenience functions

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
