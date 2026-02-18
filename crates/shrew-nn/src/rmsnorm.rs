// RMSNorm — Root Mean Square Layer Normalization
//
// RMSNorm is a simplification of LayerNorm that removes the mean centering.
// It only normalizes by the RMS (root mean square) of the values.
//
// FORMULA:
//   x_norm = x / sqrt(mean(x²) + ε)
//   y = γ * x_norm
//
// WHY RMSNorm?
//
// RMSNorm is used in modern LLMs (LLaMA, Mistral, Gemma) because:
// 1. It's computationally simpler (no mean subtraction)
// 2. Empirically comparable performance to LayerNorm
// 3. Slightly faster due to fewer operations
//
// SHAPES:
//   Input:  [*, normalized_size]
//   Output: same shape as input
//   γ:      [normalized_size]

use shrew_core::backend::Backend;
use shrew_core::dtype::DType;
use shrew_core::error::Result;
use shrew_core::tensor::Tensor;

use crate::module::Module;

/// RMS Normalization layer (used in LLaMA, Mistral, etc.).
///
/// Normalizes by root-mean-square without mean centering:
///   y = x / sqrt(mean(x²) + ε) * γ
///
/// # Example
/// ```ignore
/// let rms = RMSNorm::<CpuBackend>::new(512, 1e-5, DType::F64, &dev)?;
/// let x = CpuTensor::rand((2, 10, 512), DType::F64, &dev)?;
/// let y = rms.forward(&x)?;
/// ```
pub struct RMSNorm<B: Backend> {
    /// Learnable scale parameter γ: [normalized_size]
    weight: Tensor<B>,
    normalized_size: usize,
    eps: f64,
}

impl<B: Backend> RMSNorm<B> {
    /// Create a new RMSNorm layer.
    pub fn new(normalized_size: usize, eps: f64, dtype: DType, device: &B::Device) -> Result<Self> {
        let weight = Tensor::<B>::ones(normalized_size, dtype, device)?.set_variable();
        Ok(RMSNorm {
            weight,
            normalized_size,
            eps,
        })
    }

    pub fn normalized_size(&self) -> usize {
        self.normalized_size
    }
    pub fn eps(&self) -> f64 {
        self.eps
    }
}

impl<B: Backend> Module<B> for RMSNorm<B> {
    fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        let rank = x.rank();
        if rank == 0 {
            return Err(shrew_core::Error::msg(
                "RMSNorm: input must have at least 1 dimension",
            ));
        }
        let last_dim = rank - 1;

        // RMS = sqrt(mean(x²) + eps)
        let x_sq = x.square()?;
        let mean_sq = x_sq.mean(last_dim, true)?; // [..., 1]
        let rms = mean_sq.affine(1.0, self.eps)?.sqrt()?; // [..., 1]

        // Normalize
        let x_norm = x.div(&rms)?; // broadcasting

        // Scale by γ
        x_norm.mul(&self.weight)
    }

    fn parameters(&self) -> Vec<Tensor<B>> {
        vec![self.weight.clone()]
    }

    fn named_parameters(&self) -> Vec<(String, Tensor<B>)> {
        vec![("weight".to_string(), self.weight.clone())]
    }
}
