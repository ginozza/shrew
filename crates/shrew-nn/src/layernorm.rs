// LayerNorm — Layer Normalization
//
// Layer Normalization normalizes the activations WITHIN each sample
// (across the feature dimension), independent of the batch.
//
// FORMULA:
//   y = (x - mean(x)) / sqrt(var(x) + ε) * γ + β
//
// Where:
//   - mean(x) and var(x) are computed over the last `normalized_shape` dims
//   - γ (gamma/weight) and β (beta/bias) are learnable parameters
//   - ε is a small constant for numerical stability (default 1e-5)
//
// WHY LayerNorm?
//
// In transformers, LayerNorm is used instead of BatchNorm because:
// 1. It normalizes per-sample, not per-batch → works with variable batch sizes
// 2. It doesn't need running statistics at inference time
// 3. It's invariant to the batch composition
//
// Every transformer layer applies LayerNorm:
//   x = x + Attention(LayerNorm(x))    ← pre-norm style (GPT-2, LLaMA)
//   x = LayerNorm(x + Attention(x))    ← post-norm style (original Transformer)
//
// SHAPES:
//   Input:  [*, normalized_shape]  (e.g., [batch, seq_len, d_model])
//   Output: same shape as input
//   γ, β:  [normalized_shape]     (e.g., [d_model])
//
// The normalization is applied over the last `len(normalized_shape)` dimensions.
// For a typical transformer with d_model=512: LayerNorm(512) normalizes the
// last dimension of each [batch, seq_len, 512] tensor.

use shrew_core::backend::Backend;
use shrew_core::dtype::DType;
use shrew_core::error::Result;
use shrew_core::tensor::Tensor;

use crate::module::Module;

/// Layer Normalization: normalizes over the last N dimensions.
///
/// # Example
/// ```ignore
/// let ln = LayerNorm::<CpuBackend>::new(512, 1e-5, DType::F64, &dev)?;
/// let x = CpuTensor::rand((2, 10, 512), DType::F64, &dev)?;
/// let y = ln.forward(&x)?; // same shape, normalized
/// ```
pub struct LayerNorm<B: Backend> {
    /// Learnable scale parameter γ: [normalized_size]
    weight: Tensor<B>,
    /// Learnable shift parameter β: [normalized_size]
    bias: Tensor<B>,
    /// Size of the last dimension to normalize over.
    normalized_size: usize,
    /// Small constant for numerical stability.
    eps: f64,
}

impl<B: Backend> LayerNorm<B> {
    /// Create a new LayerNorm layer.
    ///
    /// # Arguments
    /// - `normalized_size`: size of the last dimension to normalize
    /// - `eps`: numerical stability constant (typically 1e-5)
    /// - `dtype`: data type for parameters
    /// - `device`: device to create parameters on
    pub fn new(normalized_size: usize, eps: f64, dtype: DType, device: &B::Device) -> Result<Self> {
        // γ initialized to 1 (identity scale)
        let weight = Tensor::<B>::ones(normalized_size, dtype, device)?.set_variable();
        // β initialized to 0 (no shift)
        let bias = Tensor::<B>::zeros(normalized_size, dtype, device)?.set_variable();

        Ok(LayerNorm {
            weight,
            bias,
            normalized_size,
            eps,
        })
    }

    /// Create from existing weight and bias tensors.
    pub fn from_tensors(weight: Tensor<B>, bias: Tensor<B>, eps: f64) -> Result<Self> {
        let normalized_size = weight.elem_count();
        Ok(LayerNorm {
            weight: weight.set_variable(),
            bias: bias.set_variable(),
            normalized_size,
            eps,
        })
    }

    pub fn eps(&self) -> f64 {
        self.eps
    }

    pub fn normalized_size(&self) -> usize {
        self.normalized_size
    }
}

impl<B: Backend> Module<B> for LayerNorm<B> {
    /// Forward pass: normalize over last dimension, then scale + shift.
    ///
    /// For input [batch, seq, d_model]:
    ///   1. mean = mean(x, dim=-1, keepdim=true)   → [batch, seq, 1]
    ///   2. var  = var(x, dim=-1, keepdim=true)     → [batch, seq, 1]
    ///   3. x_norm = (x - mean) / sqrt(var + eps)   → [batch, seq, d_model]
    ///   4. output = x_norm * γ + β                 → [batch, seq, d_model]
    fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        let rank = x.rank();
        if rank == 0 {
            return Err(shrew_core::Error::msg(
                "LayerNorm: input must have at least 1 dimension",
            ));
        }
        let last_dim = rank - 1;

        // Step 1: Compute mean over last dimension
        let mu = x.mean(last_dim, true)?; // [..., 1]

        // Step 2: Compute variance over last dimension
        // var(x) = mean((x - mean)²)
        let centered = x.sub(&mu)?; // broadcasting: [..., D] - [..., 1] → [..., D]
        let sq = centered.square()?;
        let variance = sq.mean(last_dim, true)?; // [..., 1]

        // Step 3: Normalize
        // x_norm = centered / sqrt(var + eps)
        let std = variance.affine(1.0, self.eps)?.sqrt()?; // [..., 1]
        let x_norm = centered.div(&std)?; // broadcasting

        // Step 4: Scale and shift
        // Output = x_norm * γ + β
        // γ and β are shape [D], x_norm is [..., D]
        // Broadcasting handles this: [D] broadcasts to [..., D]
        let output = x_norm.mul(&self.weight)?.add(&self.bias)?;

        Ok(output)
    }

    fn parameters(&self) -> Vec<Tensor<B>> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn named_parameters(&self) -> Vec<(String, Tensor<B>)> {
        vec![
            ("weight".to_string(), self.weight.clone()),
            ("bias".to_string(), self.bias.clone()),
        ]
    }
}
