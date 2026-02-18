// Dropout — Regularization via random zeroing
//
// During training, Dropout randomly sets elements to zero with probability p.
// The remaining elements are scaled by 1/(1-p) to preserve the expected value.
// This prevents co-adaptation of neurons and improves generalization.
//
// During inference (eval mode), Dropout does nothing (identity function).
//
// The training flag uses Cell<bool> for interior mutability, so set_training
// can be called through the Module trait's &self interface.

use std::cell::Cell;

use shrew_core::backend::Backend;
use shrew_core::error::Result;
use shrew_core::tensor::Tensor;

use crate::module::Module;

/// Applies dropout regularization.
///
/// During training: randomly zeros elements with probability `p`,
/// scales remaining by `1/(1-p)`.
///
/// During eval: identity (no-op).
pub struct Dropout {
    /// Probability of an element being zeroed.
    p: f64,
    /// Whether we're in training mode (Cell for interior mutability).
    training: Cell<bool>,
}

impl Dropout {
    /// Create a new Dropout layer.
    pub fn new(p: f64) -> Self {
        assert!(
            (0.0..1.0).contains(&p),
            "Dropout probability must be in [0, 1)"
        );
        Dropout {
            p,
            training: Cell::new(true),
        }
    }

    /// Set training/eval mode directly (works without specifying backend).
    pub fn set_training(&self, training: bool) {
        self.training.set(training);
    }

    /// Whether module is in training mode (works without specifying backend).
    pub fn is_training(&self) -> bool {
        self.training.get()
    }

    /// Apply dropout: randomly zero elements during training.
    pub fn forward_t<B: Backend>(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        if !self.training.get() || self.p == 0.0 {
            return Ok(x.clone());
        }

        let scale = 1.0 / (1.0 - self.p);

        // Generate random mask on-device (no host round-trip)
        let mask = Tensor::<B>::rand(x.shape().clone(), x.dtype(), x.device())?;
        let threshold = Tensor::<B>::full(x.shape().clone(), self.p, x.dtype(), x.device())?;

        // keep_mask: U8 tensor — 1 where mask >= p (keep), 0 where drop
        let keep_mask = mask.ge(&threshold)?;

        // Build zero tensor & scaled input
        let zeros = Tensor::<B>::zeros(x.shape().clone(), x.dtype(), x.device())?;
        let scaled_x = x.affine(scale, 0.0)?;

        // Select: where keep → scaled_x, where drop → 0
        Tensor::<B>::where_cond(&keep_mask, &scaled_x, &zeros)
    }
}

// Module impl — note that Dropout has no trainable parameters.
impl<B: Backend> Module<B> for Dropout {
    fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        self.forward_t(x)
    }

    fn parameters(&self) -> Vec<Tensor<B>> {
        vec![] // No trainable parameters
    }

    fn set_training(&self, training: bool) {
        self.training.set(training);
    }

    fn is_training(&self) -> bool {
        self.training.get()
    }
}
