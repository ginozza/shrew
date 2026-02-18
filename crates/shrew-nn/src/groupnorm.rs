// GroupNorm — Group Normalization
//
// Group Normalization divides channels into groups and normalizes within each
// group. It's a generalization of LayerNorm and InstanceNorm:
//
//   - GroupNorm(1, C)   = LayerNorm over channels
//   - GroupNorm(C, C)   = InstanceNorm
//   - GroupNorm(G, C)   = normalize each of G groups of C/G channels
//
// Unlike BatchNorm, GroupNorm is independent of batch size, making it
// ideal for small-batch or single-sample training (e.g., object detection).
//
// SHAPES:
//   Input:  [N, C, *] (any spatial dimensions after channels)
//   Output: [N, C, *] (same shape)
//   weight: [C], bias: [C]
//   num_groups must divide C evenly.

use shrew_core::backend::Backend;
use shrew_core::dtype::DType;
use shrew_core::error::Result;
use shrew_core::shape::Shape;
use shrew_core::tensor::Tensor;

use crate::module::Module;

/// Group Normalization layer.
///
/// # Examples
/// ```ignore
/// let gn = GroupNorm::<CpuBackend>::new(8, 32, 1e-5, DType::F64, &dev)?;
/// let x = CpuTensor::rand((2, 32, 16, 16), DType::F64, &dev)?;
/// let y = gn.forward(&x)?; // [2, 32, 16, 16]
/// ```
pub struct GroupNorm<B: Backend> {
    weight: Tensor<B>,
    bias: Tensor<B>,
    num_groups: usize,
    num_channels: usize,
    eps: f64,
}

impl<B: Backend> GroupNorm<B> {
    /// Create a new GroupNorm layer.
    ///
    /// # Arguments
    /// - `num_groups`: number of groups to divide channels into
    /// - `num_channels`: total number of channels (must be divisible by num_groups)
    /// - `eps`: numerical stability constant
    /// - `dtype`: data type
    /// - `device`: compute device
    pub fn new(
        num_groups: usize,
        num_channels: usize,
        eps: f64,
        dtype: DType,
        device: &B::Device,
    ) -> Result<Self> {
        if !num_channels.is_multiple_of(num_groups) {
            return Err(shrew_core::Error::msg(format!(
                "GroupNorm: num_channels ({}) must be divisible by num_groups ({})",
                num_channels, num_groups
            )));
        }
        let weight = Tensor::<B>::ones(num_channels, dtype, device)?.set_variable();
        let bias = Tensor::<B>::zeros(num_channels, dtype, device)?.set_variable();
        Ok(GroupNorm {
            weight,
            bias,
            num_groups,
            num_channels,
            eps,
        })
    }

    pub fn num_groups(&self) -> usize {
        self.num_groups
    }
    pub fn num_channels(&self) -> usize {
        self.num_channels
    }
}

impl<B: Backend> Module<B> for GroupNorm<B> {
    fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        let dims = x.dims();
        if dims.len() < 2 {
            return Err(shrew_core::Error::msg(
                "GroupNorm: input must be at least 2D [N, C, ...]",
            ));
        }
        let n = dims[0];
        let c = dims[1];
        if c != self.num_channels {
            return Err(shrew_core::Error::msg(format!(
                "GroupNorm: expected {} channels, got {}",
                self.num_channels, c
            )));
        }
        let channels_per_group = c / self.num_groups;

        // Flatten spatial dims: [N, C, *] → product of spatial dims
        let spatial: usize = dims[2..].iter().product();
        let group_size = channels_per_group * spatial;

        // Reshape to [N, G, channels_per_group * spatial]
        let x_flat = x.reshape(Shape::new(vec![n, self.num_groups, group_size]))?;

        // Mean and var within each group
        let mu = x_flat.mean(2, true)?; // [N, G, 1]
        let centered = x_flat.sub(&mu)?;
        let var = centered.square()?.mean(2, true)?; // [N, G, 1]
        let std = var.affine(1.0, self.eps)?.sqrt()?;
        let x_norm = centered.div(&std)?;

        // Reshape back to original shape
        let x_norm = x_norm.reshape(Shape::new(dims.to_vec()))?;

        // Scale and shift: build weight/bias to match [1, C, 1, 1, ...]
        let mut w_shape = vec![1usize; dims.len()];
        w_shape[1] = c;
        let gamma = self.weight.reshape(Shape::new(w_shape.clone()))?;
        let beta = self.bias.reshape(Shape::new(w_shape))?;

        x_norm.mul(&gamma)?.add(&beta)
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
