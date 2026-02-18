// Linear — Fully-connected (dense) layer
//
// The most fundamental neural network layer: y = xW^T + b
//
// Linear(in_features, out_features) transforms an input of shape
// [..., in_features] to [..., out_features] — a matrix multiplication
// followed by an optional bias addition.
//
// WEIGHT INITIALIZATION:
//
// We use Kaiming (He) uniform initialization, which is designed for ReLU
// networks. The weights are drawn from U(-k, k) where k = sqrt(1/in_features).
// This prevents the signal from vanishing or exploding as it passes through
// many layers — critical for training deep networks.
//
// PARAMETER SHAPES:
//
//   weight: [out_features, in_features]  — stored transposed for efficient matmul
//   bias:   [1, out_features]            — broadcast across batch dimension
//
// COMPUTATION:
//
//   y = x @ weight^T + bias
//   Input:  [batch, in_features]
//   Output: [batch, out_features]

use shrew_core::backend::Backend;
use shrew_core::dtype::DType;
use shrew_core::error::Result;
use shrew_core::tensor::Tensor;

use crate::module::Module;

/// A fully-connected (dense) layer: y = xW^T + b.
///
/// # Type Parameters
/// - `B`: the compute backend
///
/// # Examples
/// ```ignore
/// let linear = Linear::<CpuBackend>::new(784, 128, true, DType::F32, &dev)?;
/// let x = CpuTensor::rand((32, 784), DType::F32, &dev)?; // batch of 32
/// let y = linear.forward(&x)?; // shape: [32, 128]
/// ```
pub struct Linear<B: Backend> {
    /// Weight matrix: [out_features, in_features]
    weight: Tensor<B>,
    /// Optional bias vector: [1, out_features]
    bias: Option<Tensor<B>>,
    in_features: usize,
    out_features: usize,
}

impl<B: Backend> Linear<B> {
    /// Create a new Linear layer with Kaiming uniform initialization.
    ///
    /// # Arguments
    /// - `in_features`: size of each input sample
    /// - `out_features`: size of each output sample
    /// - `use_bias`: whether to add a learnable bias
    /// - `dtype`: data type for parameters
    /// - `device`: device to create parameters on
    pub fn new(
        in_features: usize,
        out_features: usize,
        use_bias: bool,
        dtype: DType,
        device: &B::Device,
    ) -> Result<Self> {
        // Kaiming uniform: U(-k, k) where k = sqrt(1/in_features)
        // This is the standard initialization for layers followed by ReLU.
        let k = (1.0 / in_features as f64).sqrt();

        // weight = rand_uniform * 2k - k  →  uniform in [-k, k]
        let weight = Tensor::<B>::rand((out_features, in_features), dtype, device)?
            .affine(2.0 * k, -k)?
            .set_variable();

        let bias = if use_bias {
            // Bias initialized to uniform [-k, k] as well
            let b = Tensor::<B>::rand((1, out_features), dtype, device)?
                .affine(2.0 * k, -k)?
                .set_variable();
            Some(b)
        } else {
            None
        };

        Ok(Linear {
            weight,
            bias,
            in_features,
            out_features,
        })
    }

    /// Create a Linear layer from existing weight and bias tensors.
    /// Useful for loading pre-trained models.
    pub fn from_tensors(weight: Tensor<B>, bias: Option<Tensor<B>>) -> Result<Self> {
        let dims = weight.dims();
        if dims.len() != 2 {
            return Err(shrew_core::Error::msg(format!(
                "Linear weight must be 2D, got shape {:?}",
                dims
            )));
        }
        let out_features = dims[0];
        let in_features = dims[1];
        Ok(Linear {
            weight: weight.set_variable(),
            bias: bias.map(|b| b.set_variable()),
            in_features,
            out_features,
        })
    }

    /// The input feature dimension.
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// The output feature dimension.
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Direct access to the weight tensor.
    pub fn weight(&self) -> &Tensor<B> {
        &self.weight
    }

    /// Direct access to the bias tensor (if any).
    pub fn bias(&self) -> Option<&Tensor<B>> {
        self.bias.as_ref()
    }
}

impl<B: Backend> Module<B> for Linear<B> {
    /// Forward pass: y = x @ W^T + b
    ///
    /// Input shape:  [batch, in_features]
    /// Output shape: [batch, out_features]
    fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        // x: [batch, in_features]
        // weight: [out_features, in_features]
        // weight^T: [in_features, out_features]
        // x @ weight^T: [batch, out_features]
        let wt = self.weight.t()?.contiguous()?;
        let output = x.matmul(&wt)?;

        match &self.bias {
            Some(bias) => {
                // bias shape: [1, out_features] — broadcasts over batch dim
                output.add(bias)
            }
            None => Ok(output),
        }
    }

    fn parameters(&self) -> Vec<Tensor<B>> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref b) = self.bias {
            params.push(b.clone());
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, Tensor<B>)> {
        let mut named = vec![("weight".to_string(), self.weight.clone())];
        if let Some(ref b) = self.bias {
            named.push(("bias".to_string(), b.clone()));
        }
        named
    }
}
