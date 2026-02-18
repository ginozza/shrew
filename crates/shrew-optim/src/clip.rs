// Gradient Clipping — Prevent exploding gradients during training
//
// Gradient clipping limits the magnitude of gradients before the optimizer
// step, preventing catastrophically large updates.
//
// Two strategies:
//   1. ClipByNorm: Scale all gradients so their global L2 norm ≤ max_norm
//      (used by GPT, BERT, and most modern architectures)
//   2. ClipByValue: Clamp each gradient element to [-max_value, max_value]
//
// USAGE:
//   let grads = loss.backward()?;
//   let clipped = clip_grad_norm::<CpuBackend>(&grads, &params, 1.0)?;
//   optimizer.step(&clipped)?;

use shrew_core::backend::Backend;
use shrew_core::backprop::GradStore;
use shrew_core::error::Result;
use shrew_core::tensor::Tensor;

// Clip by global L2 norm

/// Clip gradients by their global L2 norm.
///
/// If the total L2 norm of all gradients exceeds `max_norm`, all gradients
/// are scaled down proportionally so the total norm equals `max_norm`.
/// If the norm is already ≤ `max_norm`, gradients are returned unchanged.
///
/// Returns `(clipped_grads, total_norm)`.
///
/// # Arguments
/// - `grads`: gradient store from `loss.backward()`
/// - `params`: the parameters whose gradients to clip
/// - `max_norm`: maximum allowed L2 norm
///
/// # Example
/// ```ignore
/// let grads = loss.backward()?;
/// let (clipped, norm) = clip_grad_norm(&grads, &params, 1.0)?;
/// println!("Gradient norm: {norm:.4}");
/// optimizer.step(&clipped)?;
/// ```
pub fn clip_grad_norm<B: Backend>(
    grads: &GradStore<B>,
    params: &[Tensor<B>],
    max_norm: f64,
) -> Result<(GradStore<B>, f64)> {
    // 1. Compute global L2 norm: sqrt(sum of all grad elements squared)
    let mut total_norm_sq = 0.0f64;
    for param in params {
        if let Some(grad) = grads.get(param) {
            let data = grad.to_f64_vec()?;
            for &v in &data {
                total_norm_sq += v * v;
            }
        }
    }
    let total_norm = total_norm_sq.sqrt();

    // 2. If norm ≤ max_norm, return as-is
    if total_norm <= max_norm {
        return Ok((grads.clone(), total_norm));
    }

    // 3. Scale factor: max_norm / total_norm
    let scale = max_norm / (total_norm + 1e-6);

    // 4. Build new GradStore with scaled gradients
    let mut clipped = GradStore::<B>::new();
    for param in params {
        if let Some(grad) = grads.get(param) {
            let scaled = grad.affine(scale, 0.0)?;
            clipped.accumulate(param.id(), scaled)?;
        }
    }

    Ok((clipped, total_norm))
}

/// Compute the global L2 norm of all gradients without clipping.
///
/// Useful for monitoring gradient magnitudes during training.
pub fn grad_norm<B: Backend>(grads: &GradStore<B>, params: &[Tensor<B>]) -> Result<f64> {
    let mut total_norm_sq = 0.0f64;
    for param in params {
        if let Some(grad) = grads.get(param) {
            let data = grad.to_f64_vec()?;
            for &v in &data {
                total_norm_sq += v * v;
            }
        }
    }
    Ok(total_norm_sq.sqrt())
}

// Clip by value

/// Clamp each gradient element to `[-max_value, max_value]`.
///
/// This is a simpler but less commonly used strategy than norm clipping.
///
/// Returns `clipped_grads`.
pub fn clip_grad_value<B: Backend>(
    grads: &GradStore<B>,
    params: &[Tensor<B>],
    max_value: f64,
) -> Result<GradStore<B>> {
    let mut clipped = GradStore::<B>::new();
    for param in params {
        if let Some(grad) = grads.get(param) {
            let data = grad.to_f64_vec()?;
            let clamped: Vec<f64> = data
                .iter()
                .map(|&v| v.max(-max_value).min(max_value))
                .collect();
            let clamped_tensor = Tensor::<B>::from_f64_slice(
                &clamped,
                grad.shape().clone(),
                grad.dtype(),
                grad.device(),
            )?;
            clipped.accumulate(param.id(), clamped_tensor)?;
        }
    }
    Ok(clipped)
}

// Gradient Accumulation — Simulate larger batch sizes
//
// Gradient accumulation lets you simulate a larger effective batch size
// without increasing memory usage. Instead of stepping the optimizer
// after every batch, you accumulate gradients over N mini-batches
// and then step once with the averaged gradient.
//
// This is essential when:
//   - Your GPU memory can only fit small batches
//   - You need large effective batch sizes (e.g., for Transformers)
//
// Without this helper, the pattern requires manually managing a GradStore
// accumulator. This struct encapsulates that logic cleanly.

/// Gradient accumulation helper.
///
/// Accumulates gradients over multiple mini-batches and provides
/// the averaged gradient for the optimizer step.
///
/// # Example
/// ```ignore
/// let mut accum = GradAccumulator::<CpuBackend>::new(4); // 4 accumulation steps
///
/// for (i, batch) in batches.iter().enumerate() {
///     let loss = model.forward(batch)?;
///     let grads = loss.backward()?;
///     
///     if let Some(avg_grads) = accum.step(&grads, &params)? {
///         optimizer.step(&avg_grads)?;
///     }
/// }
/// ```
pub struct GradAccumulator<B: Backend> {
    /// Number of steps to accumulate before yielding
    accum_steps: u64,
    /// Current step within the accumulation window
    current_step: u64,
    /// Accumulated gradient sums (param_id → gradient tensor)
    accumulated: Option<GradStore<B>>,
}

impl<B: Backend> GradAccumulator<B> {
    /// Create a new gradient accumulator.
    ///
    /// # Arguments
    /// - `accum_steps`: Number of mini-batches to accumulate before stepping
    pub fn new(accum_steps: u64) -> Self {
        assert!(accum_steps > 0, "accum_steps must be > 0");
        GradAccumulator {
            accum_steps,
            current_step: 0,
            accumulated: None,
        }
    }

    /// Add gradients from one mini-batch.
    ///
    /// Returns `Some(averaged_grads)` when `accum_steps` batches have been
    /// accumulated, otherwise returns `None`.
    ///
    /// The returned gradients are divided by `accum_steps` to produce
    /// the average gradient.
    pub fn step(
        &mut self,
        grads: &GradStore<B>,
        params: &[Tensor<B>],
    ) -> Result<Option<GradStore<B>>> {
        self.current_step += 1;

        // Accumulate
        let acc = self.accumulated.get_or_insert_with(GradStore::new);
        for param in params {
            if let Some(grad) = grads.get(param) {
                acc.accumulate(param.id(), grad.clone())?;
            }
        }

        // Check if we've accumulated enough
        if self.current_step >= self.accum_steps {
            let accumulated = self.accumulated.take().unwrap();
            self.current_step = 0;

            // Average: divide each gradient by accum_steps
            let scale = 1.0 / self.accum_steps as f64;
            let mut averaged = GradStore::<B>::new();
            for param in params {
                if let Some(grad) = accumulated.get(param) {
                    let avg = grad.affine(scale, 0.0)?;
                    averaged.accumulate(param.id(), avg)?;
                }
            }

            Ok(Some(averaged))
        } else {
            Ok(None)
        }
    }

    /// Reset the accumulator, discarding any accumulated gradients.
    pub fn reset(&mut self) {
        self.current_step = 0;
        self.accumulated = None;
    }

    /// Get the number of accumulation steps.
    pub fn accum_steps(&self) -> u64 {
        self.accum_steps
    }

    /// Get the current step within the accumulation window.
    pub fn current_step(&self) -> u64 {
        self.current_step
    }
}
