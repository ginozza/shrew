// Loss Functions
//
// Loss functions measure the difference between predictions and targets.
// The loss is a scalar value that the optimizer tries to minimize.
//
// All loss functions return a SCALAR tensor so that backward() works directly.
//
// KEY LOSSES:
//
// 1. MSE (Mean Squared Error): mean((pred - target)²)
//    Used for regression tasks. Penalizes large errors quadratically.
//
// 2. Cross-Entropy: -mean(sum_classes(target * log(softmax(pred))))
//    Used for classification. The standard loss for neural networks that
//    output class probabilities.
//
// 3. L1 Loss (Mean Absolute Error): mean(|pred - target|)
//    More robust to outliers than MSE.
//
// 4. Smooth L1 / Huber Loss: smooth transition between L1 and L2 at beta.
//    Used in object detection (Faster R-CNN).
//
// 5. BCE (Binary Cross-Entropy): binary classification with probabilities.
//
// 6. BCE with Logits: combines sigmoid + BCE for numerical stability.
//
// 7. NLL Loss: negative log-likelihood with class indices.

use shrew_core::backend::Backend;
use shrew_core::error::Result;
use shrew_core::tensor::Tensor;

/// Reduction mode for loss functions.
///
/// Controls how the per-element losses are aggregated:
/// - `Mean` (default): average over all elements
/// - `Sum`: sum over all elements
/// - `None`: return per-element losses (no reduction)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Reduction {
    /// Return the mean of all per-element losses (default).
    #[default]
    Mean,
    /// Return the sum of all per-element losses.
    Sum,
    /// Return per-element losses without reduction.
    None,
}

/// Apply the reduction to a per-element loss tensor.
fn apply_reduction<B: Backend>(loss: &Tensor<B>, reduction: Reduction) -> Result<Tensor<B>> {
    match reduction {
        Reduction::Mean => loss.mean_all(),
        Reduction::Sum => loss.sum_all(),
        Reduction::None => Ok(loss.clone()),
    }
}

/// Mean Squared Error loss: mean((prediction - target)²)
///
/// Both prediction and target must have the same shape.
/// Returns a scalar tensor.
///
/// # Example
/// ```ignore
/// let loss = mse_loss(&y_pred, &y_true)?;
/// let grads = loss.backward()?;
/// ```
pub fn mse_loss<B: Backend>(prediction: &Tensor<B>, target: &Tensor<B>) -> Result<Tensor<B>> {
    let diff = prediction.sub(target)?;
    let sq = diff.square()?;
    sq.mean_all()
}

/// MSE Loss with configurable reduction.
pub fn mse_loss_with_reduction<B: Backend>(
    prediction: &Tensor<B>,
    target: &Tensor<B>,
    reduction: Reduction,
) -> Result<Tensor<B>> {
    let diff = prediction.sub(target)?;
    let sq = diff.square()?;
    apply_reduction(&sq, reduction)
}

/// Cross-entropy loss with log-softmax for numerical stability.
///
/// Computes: -mean( sum_over_classes( target * log_softmax(prediction) ) )
///
/// # Arguments
/// - `logits`: raw scores [batch, num_classes] (NOT softmax-ed)
/// - `target`: one-hot encoded targets [batch, num_classes]
///
/// # Numerical Stability
/// Uses the tensor-level `log_softmax` which computes:
///   log_softmax(x)_i = x_i - max(x) - log(sum(exp(x - max(x))))
/// This is built entirely from differentiable tensor ops, so gradients
/// flow back through logits automatically.
pub fn cross_entropy_loss<B: Backend>(logits: &Tensor<B>, target: &Tensor<B>) -> Result<Tensor<B>> {
    let dims = logits.dims();
    if dims.len() != 2 {
        return Err(shrew_core::Error::msg(format!(
            "cross_entropy expects 2D logits [batch, classes], got {:?}",
            dims
        )));
    }

    // log_softmax along class dimension (dim=1) — fully differentiable
    let log_sm = logits.log_softmax(1)?;

    // Cross-entropy = -mean(sum_classes(target * log_softmax))
    let prod = target.mul(&log_sm)?;
    let sum_classes = prod.sum(1, false)?; // [batch]
    let mean_batch = sum_classes.mean_all()?; // scalar
    mean_batch.neg()
}

/// L1 Loss (Mean Absolute Error): mean(|prediction - target|)
///
/// More robust to outliers than MSE because errors grow linearly, not
/// quadratically. Commonly used in regression with noisy targets.
///
/// Both prediction and target must have the same shape.
/// Returns a scalar tensor.
pub fn l1_loss<B: Backend>(prediction: &Tensor<B>, target: &Tensor<B>) -> Result<Tensor<B>> {
    let diff = prediction.sub(target)?;
    let abs_diff = diff.abs()?;
    abs_diff.mean_all()
}

/// L1 Loss with configurable reduction.
pub fn l1_loss_with_reduction<B: Backend>(
    prediction: &Tensor<B>,
    target: &Tensor<B>,
    reduction: Reduction,
) -> Result<Tensor<B>> {
    let diff = prediction.sub(target)?;
    let abs_diff = diff.abs()?;
    apply_reduction(&abs_diff, reduction)
}

/// Smooth L1 Loss (Huber Loss):
///
/// ```text
///             ⎧ 0.5 * (x)² / beta   if |x| < beta
/// loss(x) =  ⎨
///             ⎩ |x| - 0.5 * beta     otherwise
/// ```
/// where x = prediction - target.
///
/// Transitions smoothly from L2 (near zero) to L1 (far from zero) at `beta`.
/// Used in Faster R-CNN and SSD for bounding box regression.
///
/// # Arguments
/// - `prediction`: predicted values (any shape)
/// - `target`: ground truth values (same shape)
/// - `beta`: threshold at which to switch from L2 to L1 (must be > 0)
pub fn smooth_l1_loss<B: Backend>(
    prediction: &Tensor<B>,
    target: &Tensor<B>,
    beta: f64,
) -> Result<Tensor<B>> {
    if beta <= 0.0 {
        return Err(shrew_core::Error::msg("smooth_l1_loss: beta must be > 0"));
    }
    let diff = prediction.sub(target)?;
    let abs_diff = diff.abs()?;

    // L2 branch: 0.5 * x² / beta
    let l2_part = diff.square()?.affine(0.5 / beta, 0.0)?;

    // L1 branch: |x| - 0.5 * beta
    let l1_part = abs_diff.affine(1.0, -0.5 * beta)?;

    // Mask: 1 where |x| < beta, 0 otherwise
    let beta_tensor = Tensor::<B>::full(
        abs_diff.shape().clone(),
        beta,
        abs_diff.dtype(),
        abs_diff.device(),
    )?;
    let mask = abs_diff.lt(&beta_tensor)?;

    // Combine: where(mask, l2_part, l1_part)
    let result = Tensor::<B>::where_cond(&mask, &l2_part, &l1_part)?;
    result.mean_all()
}

/// Binary Cross-Entropy loss for probabilities in [0, 1].
///
/// Computes: -mean(target * log(pred) + (1 - target) * log(1 - pred))
///
/// # Arguments
/// - `prediction`: predicted probabilities in (0, 1) — typically sigmoid output
/// - `target`: binary targets in {0, 1}
///
/// # Warning
/// Numerically unstable if prediction values are exactly 0 or 1.
/// Use `bce_with_logits_loss` for better stability.
pub fn bce_loss<B: Backend>(prediction: &Tensor<B>, target: &Tensor<B>) -> Result<Tensor<B>> {
    // Clamp predictions to avoid log(0)
    let eps = 1e-7;
    let pred_clamped = prediction.clamp(eps, 1.0 - eps)?;

    // -[target * log(pred) + (1-target) * log(1-pred)]
    let log_pred = pred_clamped.log()?;

    let ones = Tensor::<B>::ones(
        prediction.shape().clone(),
        prediction.dtype(),
        prediction.device(),
    )?;
    let one_minus_pred = ones.sub(&pred_clamped)?;
    let log_one_minus_pred = one_minus_pred.log()?;

    let one_minus_target = ones.sub(target)?;

    // target * log(pred) + (1-target) * log(1-pred)
    let term1 = target.mul(&log_pred)?;
    let term2 = one_minus_target.mul(&log_one_minus_pred)?;
    let sum = term1.add(&term2)?;

    sum.mean_all()?.neg()
}

/// Binary Cross-Entropy with Logits (numerically stable).
///
/// Combines sigmoid + BCE in a single formula:
///   loss = mean(max(x, 0) - x*t + log(1 + exp(-|x|)))
///
/// This is numerically stable for any logit value.
///
/// # Arguments
/// - `logits`: raw scores (before sigmoid), any shape
/// - `target`: binary targets in {0, 1}, same shape
pub fn bce_with_logits_loss<B: Backend>(
    logits: &Tensor<B>,
    target: &Tensor<B>,
) -> Result<Tensor<B>> {
    // Stable formula: max(x,0) - x*t + log(1 + exp(-|x|))
    // = relu(x) - x*t + softplus(-|x|)
    let relu_x = logits.relu()?;
    let x_times_t = logits.mul(target)?;
    let abs_x = logits.abs()?;
    let neg_abs = abs_x.neg()?;
    let exp_neg_abs = neg_abs.exp()?;

    let ones = Tensor::<B>::ones(logits.shape().clone(), logits.dtype(), logits.device())?;
    let one_plus_exp = ones.add(&exp_neg_abs)?;
    let log_term = one_plus_exp.log()?;

    // relu(x) - x*t + log(1 + exp(-|x|))
    let loss = relu_x.sub(&x_times_t)?.add(&log_term)?;
    loss.mean_all()
}

/// Negative Log-Likelihood Loss with integer class indices.
///
/// Computes: -mean(log_probs[i, target[i]]) for each sample i.
///
/// # Arguments
/// - `log_probs`: log-probabilities [batch, num_classes] (output of log_softmax)
/// - `targets`: class indices as f64 [batch] — each value in 0..num_classes
///
/// Typically used as: `nll_loss(&logits.log_softmax(1)?, &targets)`
///
/// Note: unlike cross_entropy_loss which takes one-hot targets,
/// this takes integer class indices (more memory efficient).
pub fn nll_loss<B: Backend>(log_probs: &Tensor<B>, targets: &Tensor<B>) -> Result<Tensor<B>> {
    let dims = log_probs.dims();
    if dims.len() != 2 {
        return Err(shrew_core::Error::msg(format!(
            "nll_loss expects 2D log_probs [batch, classes], got {:?}",
            dims
        )));
    }
    let batch = dims[0];
    let num_classes = dims[1];

    // Convert targets to one-hot, then use element-wise multiply
    let target_vals = targets.to_f64_vec()?;
    let mut one_hot = vec![0.0f64; batch * num_classes];
    for i in 0..batch {
        let cls = target_vals[i] as usize;
        if cls >= num_classes {
            return Err(shrew_core::Error::msg(format!(
                "nll_loss: target index {} out of range for {} classes",
                cls, num_classes
            )));
        }
        one_hot[i * num_classes + cls] = 1.0;
    }
    let one_hot_tensor = Tensor::<B>::from_f64_slice(
        &one_hot,
        (batch, num_classes),
        log_probs.dtype(),
        log_probs.device(),
    )?;

    // -mean(sum_classes(one_hot * log_probs))
    let prod = one_hot_tensor.mul(log_probs)?;
    let sum_classes = prod.sum(1, false)?;
    let mean_batch = sum_classes.mean_all()?;
    mean_batch.neg()
}
