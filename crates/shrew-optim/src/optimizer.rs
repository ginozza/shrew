// Optimizer trait — The interface all optimizers implement
//
// Every optimizer takes the current parameters + their gradients and
// produces updated parameter values. The trait is simple:
//
//   fn step(&mut self, grads: &GradStore<B>) → updated parameters
//
// DESIGN DECISION: Immutable parameter update
//
// Since our tensors are immutable (Arc-wrapped), optimizers can't modify
// parameters in-place. Instead, step() returns new tensors with updated values.
// The training loop is responsible for replacing the old parameters.
//
// This is actually cleaner than PyTorch's in-place mutation approach:
//   old_params → optimizer.step(grads) → new_params (functional update)
//
// The trade-off is slightly more allocation, but it avoids complex mutation
// semantics and plays well with Rust's ownership model.

use std::collections::HashMap;

use shrew_core::backend::Backend;
use shrew_core::backprop::GradStore;
use shrew_core::error::Result;
use shrew_core::tensor::Tensor;

/// Trait that all optimizers implement.
///
/// Optimizers update model parameters given their gradients.
///
/// # Type Parameters
/// - `B`: the compute backend
pub trait Optimizer<B: Backend> {
    /// Perform one optimization step.
    ///
    /// Given the current parameters and their gradients (from backward()),
    /// compute and return the updated parameter values.
    ///
    /// Returns a vector of updated parameters in the same order as
    /// the parameters passed to the optimizer's constructor.
    fn step(&mut self, grads: &GradStore<B>) -> Result<Vec<Tensor<B>>>;

    /// Return the current learning rate.
    fn learning_rate(&self) -> f64;

    /// Set a new learning rate (for learning rate scheduling).
    fn set_learning_rate(&mut self, lr: f64);
}

// OptimizerState — Serializable state dictionary for checkpoint save/load

/// A serializable snapshot of an optimizer's internal state.
///
/// Contains named scalar values (e.g., step count, learning rate)
/// and named f64 buffers (e.g., momentum vectors, second moment estimates).
///
/// This follows the PyTorch `state_dict()` / `load_state_dict()` pattern
/// but is Rust-native and format-agnostic.
#[derive(Debug, Clone)]
pub struct OptimizerState {
    /// Optimizer type name (e.g., "Adam", "SGD") for validation on load.
    pub optimizer_type: String,
    /// Named scalar values (step count, hyperparameters, etc.)
    pub scalars: HashMap<String, f64>,
    /// Named f64 buffers (momentum vectors, second moment estimates, etc.)
    /// Each buffer is flattened to a single Vec<f64>; the key encodes
    /// the parameter index: e.g., "m.0", "m.1", "v.0", "v.1".
    pub buffers: HashMap<String, Vec<f64>>,
}

impl OptimizerState {
    /// Create an empty state dict for the given optimizer type.
    pub fn new(optimizer_type: impl Into<String>) -> Self {
        OptimizerState {
            optimizer_type: optimizer_type.into(),
            scalars: HashMap::new(),
            buffers: HashMap::new(),
        }
    }

    /// Insert a scalar value.
    pub fn set_scalar(&mut self, key: impl Into<String>, value: f64) {
        self.scalars.insert(key.into(), value);
    }

    /// Insert a buffer.
    pub fn set_buffer(&mut self, key: impl Into<String>, data: Vec<f64>) {
        self.buffers.insert(key.into(), data);
    }

    /// Get a scalar value.
    pub fn get_scalar(&self, key: &str) -> Option<f64> {
        self.scalars.get(key).copied()
    }

    /// Get a buffer.
    pub fn get_buffer(&self, key: &str) -> Option<&Vec<f64>> {
        self.buffers.get(key)
    }
}

/// Trait for optimizers that can save and restore their internal state.
///
/// This enables training checkpoint save/load — not just model weights,
/// but also the optimizer's momentum buffers, step counters, etc.,
/// allowing training to resume exactly where it left off.
pub trait Stateful {
    /// Export the optimizer's internal state as a serializable dictionary.
    fn state_dict(&self) -> OptimizerState;

    /// Restore the optimizer's internal state from a previously saved dictionary.
    ///
    /// Returns an error if the state dict is incompatible (wrong optimizer type,
    /// missing keys, wrong buffer sizes).
    fn load_state_dict(&mut self, state: &OptimizerState) -> Result<()>;
}
