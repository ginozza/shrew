// Activation modules — Wrappers around tensor activation functions
//
// These are thin wrappers that turn tensor-level activations (like tensor.relu())
// into Module implementations. This lets you compose activations in Sequential.
//
// Example:
//   let model = Sequential::new(vec![
//       Box::new(linear1),
//       Box::new(ReLU),
//       Box::new(linear2),
//   ]);

use shrew_core::backend::Backend;
use shrew_core::error::Result;
use shrew_core::tensor::Tensor;

use crate::module::Module;

/// ReLU activation: max(0, x)
pub struct ReLU;

impl<B: Backend> Module<B> for ReLU {
    fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        x.relu()
    }
    fn parameters(&self) -> Vec<Tensor<B>> {
        vec![]
    }
}

/// GELU activation (Gaussian Error Linear Unit)
/// Used in Transformers (BERT, GPT, etc.)
pub struct GeLU;

impl<B: Backend> Module<B> for GeLU {
    fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        x.gelu()
    }
    fn parameters(&self) -> Vec<Tensor<B>> {
        vec![]
    }
}

/// SiLU / Swish activation: x * σ(x)
/// Used in modern architectures (EfficientNet, LLaMA, etc.)
pub struct SiLU;

impl<B: Backend> Module<B> for SiLU {
    fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        x.silu()
    }
    fn parameters(&self) -> Vec<Tensor<B>> {
        vec![]
    }
}

/// Sigmoid activation: 1 / (1 + e^(-x))
pub struct Sigmoid;

impl<B: Backend> Module<B> for Sigmoid {
    fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        x.sigmoid()
    }
    fn parameters(&self) -> Vec<Tensor<B>> {
        vec![]
    }
}

/// Tanh activation
pub struct Tanh;

impl<B: Backend> Module<B> for Tanh {
    fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        x.tanh()
    }
    fn parameters(&self) -> Vec<Tensor<B>> {
        vec![]
    }
}

/// LeakyReLU activation: max(negative_slope * x, x)
///
/// Allows a small gradient when the unit is not active (x < 0).
/// Default negative_slope = 0.01.
pub struct LeakyReLU {
    negative_slope: f64,
}

impl LeakyReLU {
    /// Create with default negative_slope = 0.01.
    pub fn new() -> Self {
        LeakyReLU {
            negative_slope: 0.01,
        }
    }

    /// Create with custom negative_slope.
    pub fn with_slope(negative_slope: f64) -> Self {
        LeakyReLU { negative_slope }
    }
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> Module<B> for LeakyReLU {
    fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        // LeakyReLU(x) = x if x >= 0, negative_slope * x otherwise
        let zeros = Tensor::<B>::zeros_like(x)?;
        let mask = x.ge(&zeros)?; // mask = (x >= 0)
        let scaled = x.affine(self.negative_slope, 0.0)?; // negative_slope * x
        Tensor::<B>::where_cond(&mask, x, &scaled)
    }
    fn parameters(&self) -> Vec<Tensor<B>> {
        vec![]
    }
}

/// ELU activation: x if x > 0, alpha * (exp(x) - 1) otherwise
///
/// Smoother than ReLU for negative values. Default alpha = 1.0.
pub struct ELU {
    alpha: f64,
}

impl ELU {
    /// Create with default alpha = 1.0.
    pub fn new() -> Self {
        ELU { alpha: 1.0 }
    }

    /// Create with custom alpha.
    pub fn with_alpha(alpha: f64) -> Self {
        ELU { alpha }
    }
}

impl Default for ELU {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> Module<B> for ELU {
    fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        // ELU(x) = x if x > 0, alpha * (exp(x) - 1) otherwise
        let zeros = Tensor::<B>::zeros_like(x)?;
        let mask = x.gt(&zeros)?;
        let exp_x = x.exp()?;
        let ones = Tensor::<B>::ones_like(x)?;
        let exp_minus_1 = exp_x.sub(&ones)?;
        let neg_part = exp_minus_1.affine(self.alpha, 0.0)?;
        Tensor::<B>::where_cond(&mask, x, &neg_part)
    }
    fn parameters(&self) -> Vec<Tensor<B>> {
        vec![]
    }
}

/// Mish activation: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
///
/// A self-regularizing non-monotonic activation function.
/// Used in YOLOv4 and other modern architectures.
pub struct Mish;

impl<B: Backend> Module<B> for Mish {
    fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        // softplus(x) = ln(1 + exp(x))
        let exp_x = x.exp()?;
        let ones = Tensor::<B>::ones_like(x)?;
        let one_plus_exp = ones.add(&exp_x)?;
        let softplus = one_plus_exp.log()?;
        // mish(x) = x * tanh(softplus(x))
        let tanh_sp = softplus.tanh()?;
        x.mul(&tanh_sp)
    }
    fn parameters(&self) -> Vec<Tensor<B>> {
        vec![]
    }
}
