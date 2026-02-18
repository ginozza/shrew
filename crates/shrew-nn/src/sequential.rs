// Sequential — A chain of modules applied one after another
//
// Sequential is the simplest way to build a neural network: a list of layers
// applied in order. It's equivalent to PyTorch's nn.Sequential.
//
// Example:
//   let model = Sequential::new()
//       .add(linear1)
//       .add(ReLU)
//       .add(linear2);
//
//   let output = model.forward(&input)?;
//
// The output of each layer becomes the input to the next.

use shrew_core::backend::Backend;
use shrew_core::error::Result;
use shrew_core::tensor::Tensor;

use crate::module::Module;

/// A container that chains modules sequentially.
///
/// Each module's output becomes the next module's input.
/// Sequential itself implements Module, so it can be nested.
pub struct Sequential<B: Backend> {
    layers: Vec<Box<dyn Module<B>>>,
}

impl<B: Backend> Sequential<B> {
    /// Create an empty Sequential.
    pub fn new() -> Self {
        Sequential { layers: Vec::new() }
    }

    /// Add a layer to the end of the sequence. Returns self for chaining.
    #[allow(clippy::should_implement_trait)]
    pub fn add<M: Module<B> + 'static>(mut self, module: M) -> Self {
        self.layers.push(Box::new(module));
        self
    }

    /// Number of layers.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Whether the sequential is empty.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

impl<B: Backend> Default for Sequential<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> Module<B> for Sequential<B> {
    fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        let mut out = x.clone();
        for layer in &self.layers {
            out = layer.forward(&out)?;
        }
        Ok(out)
    }

    fn parameters(&self) -> Vec<Tensor<B>> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    fn named_parameters(&self) -> Vec<(String, Tensor<B>)> {
        let mut named = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            for (k, v) in layer.named_parameters() {
                named.push((format!("layers.{i}.{k}"), v));
            }
        }
        named
    }

    /// Propagate training mode to all child layers.
    fn set_training(&self, training: bool) {
        for layer in &self.layers {
            layer.set_training(training);
        }
    }
}
