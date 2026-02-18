// Module trait — The interface every neural network layer implements
//
// In PyTorch, `nn.Module` is the base class for all neural network layers.
// In Shrew, `Module` is a trait that every layer implements.
//
// The key method is forward() — it takes input tensor(s) and returns output.
// The parameters() method returns all trainable tensors (for optimizers).
//
// WHY A TRAIT?
//
// Unlike PyTorch's class hierarchy, Rust uses traits for polymorphism.
// Each module (Linear, Embedding, etc.) is a plain struct that implements
// the Module trait. This is idiomatic Rust and enables static dispatch.
//
// GENERIC OVER BACKEND:
//
// All modules are generic over B: Backend, so the same module definition
// can run on CPU or GPU. The tensors inside the module live on whatever
// backend B is.

use shrew_core::backend::Backend;
use shrew_core::error::Result;
use shrew_core::tensor::Tensor;

/// The fundamental trait for all neural network layers.
///
/// Every layer in Shrew implements this trait, providing:
/// - `forward()`: compute output from input (the actual computation)
/// - `parameters()`: list all trainable tensors (for optimizer updates)
/// - `set_training()` / `is_training()`: toggle train/eval mode
/// - `num_parameters()` / `trainable_params_count()`: count parameters
///
/// # Example
/// ```ignore
/// struct MyLayer<B: Backend> {
///     linear: Linear<B>,
/// }
///
/// impl<B: Backend> Module<B> for MyLayer<B> {
///     fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
///         self.linear.forward(x)?.relu()
///     }
///     fn parameters(&self) -> Vec<Tensor<B>> {
///         self.linear.parameters()
///     }
/// }
/// ```
pub trait Module<B: Backend> {
    /// Compute the output tensor from the input tensor.
    /// This defines the layer's computation (forward pass).
    fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>>;

    /// Return all trainable parameters of this module.
    /// The optimizer uses these to update weights during training.
    fn parameters(&self) -> Vec<Tensor<B>>;

    /// Set training or evaluation mode.
    ///
    /// Override in modules that behave differently in train vs eval
    /// (e.g., Dropout, BatchNorm). Default is a no-op.
    ///
    /// Uses interior mutability (`Cell<bool>`) so `&self` suffices.
    fn set_training(&self, _training: bool) {
        // Default: no-op. Override in Dropout, BatchNorm, etc.
    }

    /// Whether the module is in training mode (default: true).
    fn is_training(&self) -> bool {
        true
    }

    /// Convenience: set training mode.
    fn train(&self) {
        self.set_training(true);
    }

    /// Convenience: set evaluation mode.
    fn eval(&self) {
        self.set_training(false);
    }

    /// Total number of scalar parameters in this module.
    fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.elem_count()).sum()
    }

    /// Number of trainable (variable) parameters.
    fn trainable_params_count(&self) -> usize {
        self.parameters()
            .iter()
            .filter(|p| p.is_variable())
            .map(|p| p.elem_count())
            .sum()
    }

    /// Freeze all parameters: returns new parameter tensors with
    /// `is_variable = false`, preventing gradient accumulation.
    ///
    /// The caller must rebuild the module with the frozen tensors.
    fn frozen_parameters(&self) -> Vec<Tensor<B>> {
        self.parameters().into_iter().map(|p| p.freeze()).collect()
    }

    /// Return all trainable parameters with human-readable names.
    ///
    /// Leaf modules (Linear, Conv2d, etc.) override this to provide
    /// meaningful names like `"weight"` / `"bias"`.  Composite modules
    /// should concatenate sub-module names with a `"."` separator, e.g.
    /// `"fc1.weight"`, `"attn.w_q.weight"`.
    ///
    /// The default uses positional indices (`param_0`, `param_1`, …).
    fn named_parameters(&self) -> Vec<(String, Tensor<B>)> {
        self.parameters()
            .into_iter()
            .enumerate()
            .map(|(i, p)| (format!("param_{i}"), p))
            .collect()
    }

    /// Returns a `state_dict`-style map of parameter name → tensor.
    ///
    /// This is the idiomatic way to serialize a module.
    fn state_dict(&self) -> Vec<(String, Tensor<B>)> {
        self.named_parameters()
    }
}
