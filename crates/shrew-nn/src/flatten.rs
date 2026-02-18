// Flatten — Flatten spatial dimensions into a single feature dimension
//
// Flattens a contiguous range of dimensions of the input tensor.
// Commonly used between convolutional and fully connected layers.
//
// By default, flattens all dimensions except the batch dimension:
//   [N, C, H, W] → [N, C*H*W]
//
// The `start_dim` and `end_dim` parameters control which dimensions
// to flatten (1-indexed, inclusive). Default: start_dim=1, end_dim=-1.

use shrew_core::backend::Backend;
use shrew_core::error::Result;
use shrew_core::shape::Shape;
use shrew_core::tensor::Tensor;

use crate::module::Module;

/// Flatten layer: collapses dimensions `[start_dim..=end_dim]` into one.
///
/// Default (start_dim=1): `[N, C, H, W]` → `[N, C*H*W]`.
///
/// # Examples
/// ```ignore
/// let flatten = Flatten::new(1); // flatten from dim 1 onward
/// let x: [2, 8, 4, 4] tensor
/// let y = flatten.forward(&x)?; // [2, 128]
/// ```
pub struct Flatten {
    start_dim: usize,
}

impl Flatten {
    /// Create a Flatten that collapses from `start_dim` through the last dim.
    ///
    /// - `start_dim = 1`: flatten everything except batch → `[N, ...]` → `[N, flat]`
    /// - `start_dim = 0`: flatten everything → `[total]`
    pub fn new(start_dim: usize) -> Self {
        Flatten { start_dim }
    }

    /// Default flatten: from dim 1 onward (preserves batch dimension).
    pub fn default_flat() -> Self {
        Flatten { start_dim: 1 }
    }
}

impl Default for Flatten {
    fn default() -> Self {
        Self::default_flat()
    }
}

impl<B: Backend> Module<B> for Flatten {
    fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        let dims = x.dims();
        if self.start_dim >= dims.len() {
            return Ok(x.clone()); // nothing to flatten
        }

        // Keep dims before start_dim, multiply out the rest
        let mut new_dims: Vec<usize> = dims[..self.start_dim].to_vec();
        let flat: usize = dims[self.start_dim..].iter().product();
        new_dims.push(flat);

        x.reshape(Shape::new(new_dims))
    }

    fn parameters(&self) -> Vec<Tensor<B>> {
        vec![] // No learnable parameters
    }
}
