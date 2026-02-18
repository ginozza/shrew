// EMA — Exponential Moving Average of Model Parameters
//
// EMA maintains a shadow copy of model parameters that is an exponential
// moving average of the training parameters. This smoothed version of the
// model often generalizes better than the final training weights.
//
// Update rule (after each optimizer step):
//   shadow_θ = decay * shadow_θ + (1 - decay) * θ
//
// Typical decay: 0.999 (close to 1 means slower update → more smoothing)
//
// USAGE:
//   - During training: update EMA after each optimizer step
//   - During evaluation: use EMA parameters instead of training parameters
//
// This technique is used in:
//   - Image generation (DDPM, StyleGAN)
//   - Semi-supervised learning (Mean Teacher)
//   - Large language models (some fine-tuning recipes)
//
// DESIGN: The EMA stores copies of parameter data (as Vec<f64>) so it
// doesn't interfere with training. Use `apply()` to write EMA weights
// into the model parameters, and `restore()` to put training weights back.

use shrew_core::backend::Backend;
use shrew_core::error::Result;
use shrew_core::tensor::Tensor;

/// Exponential Moving Average of model parameters.
///
/// Maintains a shadow copy that is a smoothed version of training parameters.
///
/// # Example
/// ```ignore
/// let mut ema = EMA::new(model.parameters(), 0.999);
///
/// // Training loop:
/// optimizer.step(&grads)?;
/// ema.update(&model.parameters())?;
///
/// // Evaluation:
/// ema.apply()?;             // Write EMA weights into model
/// let output = model.forward(input)?;
/// ema.restore()?;           // Restore training weights
/// ```
pub struct EMA<B: Backend> {
    /// References to the model parameters (used for apply/restore)
    params: Vec<Tensor<B>>,
    /// Shadow parameters (EMA values)
    shadow: Vec<Vec<f64>>,
    /// Saved training parameters (for restore after apply)
    backup: Vec<Vec<f64>>,
    /// Decay rate (e.g., 0.999)
    decay: f64,
    /// Number of updates performed
    num_updates: u64,
}

impl<B: Backend> EMA<B> {
    /// Create a new EMA tracker.
    ///
    /// # Arguments
    /// - `params`: The model parameters to track
    /// - `decay`: Decay rate (typical: 0.999 or 0.9999)
    pub fn new(params: Vec<Tensor<B>>, decay: f64) -> Result<Self> {
        let shadow: Result<Vec<Vec<f64>>> = params.iter().map(|p| p.to_f64_vec()).collect();
        let shadow = shadow?;

        Ok(EMA {
            params,
            shadow,
            backup: Vec::new(),
            decay,
            num_updates: 0,
        })
    }

    /// Update the EMA shadow parameters with current model parameters.
    ///
    /// Call this after each optimizer step.
    pub fn update(&mut self, current_params: &[Tensor<B>]) -> Result<()> {
        self.num_updates += 1;

        for (i, param) in current_params.iter().enumerate() {
            let data = param.to_f64_vec()?;
            for (s, &d) in self.shadow[i].iter_mut().zip(data.iter()) {
                *s = self.decay * *s + (1.0 - self.decay) * d;
            }
        }

        Ok(())
    }

    /// Update using an adjusted decay that ramps up during early training.
    ///
    /// The effective decay is: min(decay, (1 + num_updates) / (10 + num_updates))
    /// This prevents the EMA from being too biased toward initial values.
    pub fn update_with_warmup(&mut self, current_params: &[Tensor<B>]) -> Result<()> {
        self.num_updates += 1;

        let effective_decay = self
            .decay
            .min((1.0 + self.num_updates as f64) / (10.0 + self.num_updates as f64));

        for (i, param) in current_params.iter().enumerate() {
            let data = param.to_f64_vec()?;
            for (s, &d) in self.shadow[i].iter_mut().zip(data.iter()) {
                *s = effective_decay * *s + (1.0 - effective_decay) * d;
            }
        }

        Ok(())
    }

    /// Apply EMA parameters to the model (for evaluation).
    ///
    /// This saves the current training parameters so they can be restored
    /// with `restore()`.
    pub fn apply(&mut self) -> Result<()> {
        // Save current training weights
        self.backup = Vec::with_capacity(self.params.len());
        for param in &self.params {
            self.backup.push(param.to_f64_vec()?);
        }

        // Write EMA weights into model parameters
        for (param, shadow) in self.params.iter().zip(self.shadow.iter()) {
            param.update_data_inplace(shadow)?;
        }

        Ok(())
    }

    /// Restore training parameters after `apply()`.
    pub fn restore(&mut self) -> Result<()> {
        if self.backup.is_empty() {
            return Ok(());
        }

        for (param, backup) in self.params.iter().zip(self.backup.iter()) {
            param.update_data_inplace(backup)?;
        }

        self.backup.clear();
        Ok(())
    }

    /// Get the decay rate.
    pub fn decay(&self) -> f64 {
        self.decay
    }

    /// Get the number of updates performed.
    pub fn num_updates(&self) -> u64 {
        self.num_updates
    }

    /// Get the shadow (EMA) values for a specific parameter index.
    pub fn shadow_values(&self, index: usize) -> &[f64] {
        &self.shadow[index]
    }
}
