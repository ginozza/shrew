// BatchNorm2d — 2D Batch Normalization
//
// Batch Normalization normalizes activations ACROSS the batch for each channel,
// stabilizing and accelerating training of deep convolutional networks.
//
// FORMULA (training mode):
//   x_hat = (x - mean_batch) / sqrt(var_batch + ε)
//   y = γ * x_hat + β
//
// Where mean_batch and var_batch are computed per-channel over (N, H, W).
//
// RUNNING STATISTICS:
//   During training, we maintain exponential moving averages:
//     running_mean = (1 - momentum) * running_mean + momentum * mean_batch
//     running_var  = (1 - momentum) * running_var  + momentum * var_batch
//
//   During eval, we use running_mean/running_var instead of batch stats.
//
// SHAPES:
//   Input:  [N, C, H, W]
//   Output: [N, C, H, W] (same shape)
//   γ, β:  [C]
//
// WHY BatchNorm2d?
//
// In CNNs, BatchNorm dramatically helps training:
// 1. Reduces internal covariate shift
// 2. Allows higher learning rates
// 3. Acts as a regularizer (reduces need for dropout)
// 4. Makes networks less sensitive to weight initialization

use shrew_core::backend::Backend;
use shrew_core::dtype::DType;
use shrew_core::error::Result;
use shrew_core::shape::Shape;
use shrew_core::tensor::Tensor;

use crate::module::Module;

/// 2D Batch Normalization layer for convolutional networks.
///
/// Normalizes each channel across the batch: for input `[N, C, H, W]`,
/// mean and variance are computed over `(N, H, W)` for each of `C` channels.
///
/// # Examples
/// ```ignore
/// let bn = BatchNorm2d::<CpuBackend>::new(16, 1e-5, 0.1, DType::F64, &dev)?;
/// let x: [batch, 16, H, W] tensor
/// let y = bn.forward(&x)?; // normalized, same shape
/// ```
pub struct BatchNorm2d<B: Backend> {
    /// Learnable scale (gamma): [C]
    weight: Tensor<B>,
    /// Learnable shift (beta): [C]
    bias: Tensor<B>,
    /// Running mean (not trainable): [C]
    running_mean: std::cell::RefCell<Vec<f64>>,
    /// Running variance (not trainable): [C]
    running_var: std::cell::RefCell<Vec<f64>>,
    /// Number of channels.
    num_features: usize,
    /// Numerical stability constant.
    eps: f64,
    /// Momentum for running statistics update.
    momentum: f64,
    /// Whether we're in training mode (use batch stats) or eval (use running stats).
    training: std::cell::Cell<bool>,
}

impl<B: Backend> BatchNorm2d<B> {
    /// Create a new BatchNorm2d layer.
    ///
    /// # Arguments
    /// - `num_features`: number of channels (C)
    /// - `eps`: numerical stability constant (typically 1e-5)
    /// - `momentum`: EMA momentum for running stats (typically 0.1)
    /// - `dtype`: data type for learnable parameters
    /// - `device`: device
    pub fn new(
        num_features: usize,
        eps: f64,
        momentum: f64,
        dtype: DType,
        device: &B::Device,
    ) -> Result<Self> {
        // γ initialized to 1 (scale)
        let weight =
            Tensor::<B>::ones(Shape::new(vec![num_features]), dtype, device)?.set_variable();
        // β initialized to 0 (shift)
        let bias =
            Tensor::<B>::zeros(Shape::new(vec![num_features]), dtype, device)?.set_variable();

        Ok(BatchNorm2d {
            weight,
            bias,
            running_mean: std::cell::RefCell::new(vec![0.0; num_features]),
            running_var: std::cell::RefCell::new(vec![1.0; num_features]),
            num_features,
            eps,
            momentum,
            training: std::cell::Cell::new(true),
        })
    }

    /// Set training mode (use batch statistics).
    pub fn train(&self) {
        self.training.set(true);
    }

    /// Set evaluation mode (use running statistics).
    pub fn eval(&self) {
        self.training.set(false);
    }

    /// Whether the module is in training mode.
    pub fn is_training(&self) -> bool {
        self.training.get()
    }

    pub fn num_features(&self) -> usize {
        self.num_features
    }

    pub fn eps(&self) -> f64 {
        self.eps
    }

    pub fn weight(&self) -> &Tensor<B> {
        &self.weight
    }

    pub fn bias(&self) -> &Tensor<B> {
        &self.bias
    }

    /// Create from existing weight and bias tensors (for executor/model loading).
    /// Initializes running stats to mean=0, var=1.
    pub fn from_tensors(weight: Tensor<B>, bias: Tensor<B>, eps: f64) -> Result<Self> {
        let num_features = weight.elem_count();
        Ok(BatchNorm2d {
            weight: weight.set_variable(),
            bias: bias.set_variable(),
            running_mean: std::cell::RefCell::new(vec![0.0; num_features]),
            running_var: std::cell::RefCell::new(vec![1.0; num_features]),
            num_features,
            eps,
            momentum: 0.1,
            training: std::cell::Cell::new(true),
        })
    }
}

impl<B: Backend> Module<B> for BatchNorm2d<B> {
    /// Forward pass: batch-normalize each channel.
    ///
    /// Training:  use batch mean/var, update running stats.
    /// Eval:      use running mean/var.
    fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        if x.rank() != 4 {
            return Err(shrew_core::Error::msg(format!(
                "BatchNorm2d: expected 4D input [N,C,H,W], got rank {}",
                x.rank()
            )));
        }

        let dims = x.dims();
        let (n, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);

        if c != self.num_features {
            return Err(shrew_core::Error::msg(format!(
                "BatchNorm2d: expected {} channels, got {}",
                self.num_features, c
            )));
        }

        if self.training.get() {
            // Compute per-channel mean and variance entirely on-device.
            // Reshape [N,C,H,W] → [N,C,H*W] so we can reduce over dim=0 and dim=2.
            let x_flat = x.reshape(Shape::new(vec![n, c, h * w]))?;

            // Mean over N and spatial: first mean over dim=2 → [N,C], then mean over dim=0 → [C]
            let mean_spatial = x_flat.mean(2, false)?; // [N, C]
            let mean_batch = mean_spatial.mean(0, false)?; // [C]

            // Variance: E[(x - mean)^2] — compute on device
            let mean_bcast = mean_batch.reshape(Shape::new(vec![1, c, 1, 1]))?;
            let diff = x.sub(&mean_bcast)?;
            let diff_sq = diff.mul(&diff)?;
            let diff_sq_flat = diff_sq.reshape(Shape::new(vec![n, c, h * w]))?;
            let var_spatial = diff_sq_flat.mean(2, false)?; // [N, C]
            let var_batch = var_spatial.mean(0, false)?; // [C]

            // Update running statistics (small host transfer — only C floats)
            {
                let mean_host = mean_batch.to_f64_vec()?;
                let var_host = var_batch.to_f64_vec()?;
                let mut rm = self.running_mean.borrow_mut();
                let mut rv = self.running_var.borrow_mut();
                for ci in 0..c {
                    rm[ci] = (1.0 - self.momentum) * rm[ci] + self.momentum * mean_host[ci];
                    rv[ci] = (1.0 - self.momentum) * rv[ci] + self.momentum * var_host[ci];
                }
            }

            self.apply_norm_tensors(x, &mean_batch, &var_batch, c)
        } else {
            // Eval mode: use running stats
            let rm = self.running_mean.borrow();
            let rv = self.running_var.borrow();
            let mean_t =
                Tensor::<B>::from_f64_slice(&rm, Shape::new(vec![c]), x.dtype(), x.device())?;
            let var_t =
                Tensor::<B>::from_f64_slice(&rv, Shape::new(vec![c]), x.dtype(), x.device())?;
            self.apply_norm_tensors(x, &mean_t, &var_t, c)
        }
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

    fn set_training(&self, training: bool) {
        self.training.set(training);
    }

    fn is_training(&self) -> bool {
        self.training.get()
    }
}

impl<B: Backend> BatchNorm2d<B> {
    /// Apply normalization using on-device mean/var tensors (shape [C]).
    ///
    /// All computation stays on the device — no host round-trip.
    ///   x_hat = (x - mean[1,C,1,1]) * invstd[1,C,1,1]
    ///   y = gamma[1,C,1,1] * x_hat + beta[1,C,1,1]
    fn apply_norm_tensors(
        &self,
        x: &Tensor<B>,
        mean: &Tensor<B>,
        var: &Tensor<B>,
        c: usize,
    ) -> Result<Tensor<B>> {
        // Broadcast shapes: [C] → [1, C, 1, 1]
        let mean_b = mean.reshape(Shape::new(vec![1, c, 1, 1]))?;
        let var_b = var.reshape(Shape::new(vec![1, c, 1, 1]))?;

        // invstd = 1 / sqrt(var + eps)
        let var_eps = var_b.affine(1.0, self.eps)?; // var + eps
        let invstd = var_eps.sqrt()?.powf(-1.0)?; // 1/sqrt(var+eps)

        // x_hat = (x - mean) * invstd
        let x_hat = x.sub(&mean_b)?.mul(&invstd)?;

        // y = gamma * x_hat + beta
        let gamma = self.weight.reshape(Shape::new(vec![1, c, 1, 1]))?;
        let beta = self.bias.reshape(Shape::new(vec![1, c, 1, 1]))?;
        x_hat.mul(&gamma)?.add(&beta)
    }
}
