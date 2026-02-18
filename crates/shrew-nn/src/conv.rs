// Conv2d & MaxPool2d — 2D convolutional layers
//
// Conv2d applies a set of learnable 2D convolution filters to an input
// tensor of shape [N, C_in, H, W], producing [N, C_out, H_out, W_out].
//
// MaxPool2d performs 2D max-pooling (spatial down-sampling) by taking the
// maximum value in each sliding window.
//
// WEIGHT INITIALIZATION (Conv2d):
//
//   Kaiming (He) uniform: U(-k, k) where k = sqrt(1 / (C_in * kH * kW)).
//   This is the standard for layers followed by ReLU.
//
// PARAMETER SHAPES (Conv2d):
//
//   weight: [C_out, C_in, kH, kW]
//   bias:   [C_out]                 (optional)
//
// OUTPUT SIZE FORMULA:
//
//   H_out = floor((H + 2*padding_h - kernel_h) / stride_h) + 1
//   W_out = floor((W + 2*padding_w - kernel_w) / stride_w) + 1

use shrew_core::backend::Backend;
use shrew_core::dtype::DType;
use shrew_core::error::Result;
use shrew_core::shape::Shape;
use shrew_core::tensor::Tensor;

use crate::module::Module;

/// 2D convolutional layer.
///
/// Applies a set of learnable filters to a 4D input `[N, C_in, H, W]`,
/// producing output of shape `[N, C_out, H_out, W_out]`.
///
/// # Examples
/// ```ignore
/// let conv = Conv2d::<CpuBackend>::new(1, 16, [3, 3], [1, 1], [1, 1], true, DType::F32, &dev)?;
/// let x = CpuTensor::rand((4, 1, 28, 28), DType::F32, &dev)?;
/// let y = conv.forward(&x)?; // [4, 16, 28, 28]
/// ```
pub struct Conv2d<B: Backend> {
    /// Convolution filters: [C_out, C_in, kH, kW]
    weight: Tensor<B>,
    /// Optional bias: [C_out]
    bias: Option<Tensor<B>>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
}

impl<B: Backend> Conv2d<B> {
    /// Create a new Conv2d layer with Kaiming uniform initialization.
    ///
    /// # Arguments
    /// - `in_channels`:  number of input channels (C_in)
    /// - `out_channels`: number of output channels / filters (C_out)
    /// - `kernel_size`:  `[kH, kW]` spatial size of each filter
    /// - `stride`:       `[sH, sW]` stride of the convolution
    /// - `padding`:      `[pH, pW]` zero-padding added to both sides
    /// - `use_bias`:     whether to include an additive bias
    /// - `dtype`:        data type for parameters
    /// - `device`:       device to create parameters on
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        use_bias: bool,
        dtype: DType,
        device: &B::Device,
    ) -> Result<Self> {
        let [kh, kw] = kernel_size;
        let fan_in = in_channels * kh * kw;
        let k = (1.0 / fan_in as f64).sqrt();

        // weight: uniform in [-k, k], shape [C_out, C_in, kH, kW]
        let weight = Tensor::<B>::rand(
            Shape::new(vec![out_channels, in_channels, kh, kw]),
            dtype,
            device,
        )?
        .affine(2.0 * k, -k)?
        .set_variable();

        let bias = if use_bias {
            let b = Tensor::<B>::rand(Shape::new(vec![out_channels]), dtype, device)?
                .affine(2.0 * k, -k)?
                .set_variable();
            Some(b)
        } else {
            None
        };

        Ok(Conv2d {
            weight,
            bias,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        })
    }

    /// Create a Conv2d from existing weight and bias tensors (e.g. for loading).
    pub fn from_tensors(
        weight: Tensor<B>,
        bias: Option<Tensor<B>>,
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> Result<Self> {
        let dims = weight.dims();
        if dims.len() != 4 {
            return Err(shrew_core::Error::msg(format!(
                "Conv2d weight must be 4D [C_out,C_in,kH,kW], got {:?}",
                dims
            )));
        }
        let out_channels = dims[0];
        let in_channels = dims[1];
        let kernel_size = [dims[2], dims[3]];
        Ok(Conv2d {
            weight: weight.set_variable(),
            bias: bias.map(|b| b.set_variable()),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        })
    }

    pub fn in_channels(&self) -> usize {
        self.in_channels
    }
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }
    pub fn kernel_size(&self) -> [usize; 2] {
        self.kernel_size
    }
    pub fn stride(&self) -> [usize; 2] {
        self.stride
    }
    pub fn padding(&self) -> [usize; 2] {
        self.padding
    }
    pub fn weight(&self) -> &Tensor<B> {
        &self.weight
    }
    pub fn bias(&self) -> Option<&Tensor<B>> {
        self.bias.as_ref()
    }
}

impl<B: Backend> Module<B> for Conv2d<B> {
    /// Forward pass: 2D convolution.
    ///
    /// Input:  `[N, C_in, H, W]`
    /// Output: `[N, C_out, H_out, W_out]`
    fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        x.conv2d(&self.weight, self.bias.as_ref(), self.stride, self.padding)
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

// MaxPool2d

/// 2D max-pooling layer.
///
/// Slides a window of `kernel_size` over the input's spatial dimensions
/// and takes the max in each window.
///
/// Input:  `[N, C, H, W]`
/// Output: `[N, C, H_out, W_out]`
pub struct MaxPool2d {
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
}

impl MaxPool2d {
    /// Create a new MaxPool2d layer.
    ///
    /// # Arguments
    /// - `kernel_size`: `[kH, kW]`
    /// - `stride`:      `[sH, sW]` — typically equal to kernel_size
    /// - `padding`:     `[pH, pW]`
    pub fn new(kernel_size: [usize; 2], stride: [usize; 2], padding: [usize; 2]) -> Self {
        MaxPool2d {
            kernel_size,
            stride,
            padding,
        }
    }
}

impl<B: Backend> Module<B> for MaxPool2d {
    fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        x.max_pool2d(self.kernel_size, self.stride, self.padding)
    }

    fn parameters(&self) -> Vec<Tensor<B>> {
        vec![] // No learnable parameters
    }
}

// AvgPool2d

/// 2D average-pooling layer.
///
/// Slides a window of `kernel_size` over the input's spatial dimensions
/// and takes the mean in each window.
///
/// Input:  `[N, C, H, W]`
/// Output: `[N, C, H_out, W_out]`
pub struct AvgPool2d {
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
}

impl AvgPool2d {
    /// Create a new AvgPool2d layer.
    ///
    /// # Arguments
    /// - `kernel_size`: `[kH, kW]`
    /// - `stride`:      `[sH, sW]` — typically equal to kernel_size
    /// - `padding`:     `[pH, pW]`
    pub fn new(kernel_size: [usize; 2], stride: [usize; 2], padding: [usize; 2]) -> Self {
        AvgPool2d {
            kernel_size,
            stride,
            padding,
        }
    }
}

impl<B: Backend> Module<B> for AvgPool2d {
    fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        x.avg_pool2d(self.kernel_size, self.stride, self.padding)
    }

    fn parameters(&self) -> Vec<Tensor<B>> {
        vec![] // No learnable parameters
    }
}

// Conv1d

/// 1D convolution layer.
///
/// Input:  `[N, C_in, L]`
/// Output: `[N, C_out, L_out]`
///
/// where `L_out = (L + 2*padding - kernel_size) / stride + 1`.
///
/// Weight shape: `[C_out, C_in, K]`
/// Bias shape:   `[C_out]` (optional)
#[allow(dead_code)]
pub struct Conv1d<B: Backend> {
    weight: Tensor<B>,
    bias: Option<Tensor<B>>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
}

impl<B: Backend> Conv1d<B> {
    /// Create a new Conv1d layer with Kaiming-uniform initialization.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        use_bias: bool,
        dtype: DType,
        device: &B::Device,
    ) -> Result<Self> {
        // Kaiming uniform initialization
        let k = 1.0 / (in_channels as f64 * kernel_size as f64).sqrt();
        let w_shape = Shape::new(vec![out_channels, in_channels, kernel_size]);
        let weight = Tensor::<B>::rand(w_shape, dtype, device)?
            .affine(2.0 * k, -k)?
            .set_variable();

        let bias = if use_bias {
            let b_shape = Shape::new(vec![out_channels]);
            Some(
                Tensor::<B>::rand(b_shape, dtype, device)?
                    .affine(2.0 * k, -k)?
                    .set_variable(),
            )
        } else {
            None
        };

        Ok(Conv1d {
            weight,
            bias,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        })
    }
}

impl<B: Backend> Module<B> for Conv1d<B> {
    fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        x.conv1d(&self.weight, self.bias.as_ref(), self.stride, self.padding)
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

// AdaptiveAvgPool2d

/// Adaptive 2D Average Pooling — pools to a fixed output size.
///
/// Automatically computes kernel_size, stride, and padding to produce
/// the desired output spatial dimensions, regardless of input size.
///
/// Input:  `[N, C, H_in, W_in]`
/// Output: `[N, C, H_out, W_out]`
///
/// Common use: `AdaptiveAvgPool2d([1, 1])` — global average pooling.
pub struct AdaptiveAvgPool2d {
    output_size: [usize; 2],
}

impl AdaptiveAvgPool2d {
    /// Create an AdaptiveAvgPool2d with the desired output spatial size.
    pub fn new(output_size: [usize; 2]) -> Self {
        AdaptiveAvgPool2d { output_size }
    }
}

impl<B: Backend> Module<B> for AdaptiveAvgPool2d {
    fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        let dims = x.dims();
        if dims.len() != 4 {
            return Err(shrew_core::Error::msg(format!(
                "AdaptiveAvgPool2d: expected 4D [N,C,H,W], got {:?}",
                dims
            )));
        }
        let h_in = dims[2];
        let w_in = dims[3];
        let [h_out, w_out] = self.output_size;

        if h_out == 0 || w_out == 0 {
            return Err(shrew_core::Error::msg(
                "AdaptiveAvgPool2d: output_size must be > 0",
            ));
        }

        // Compute kernel, stride, padding to achieve desired output size.
        // Formula: output = floor((input + 2*pad - kernel) / stride) + 1
        // Simplest: stride = input / output, kernel = input - (output-1)*stride
        let stride_h = h_in / h_out;
        let stride_w = w_in / w_out;
        let kernel_h = h_in - (h_out - 1) * stride_h;
        let kernel_w = w_in - (w_out - 1) * stride_w;

        x.avg_pool2d([kernel_h, kernel_w], [stride_h, stride_w], [0, 0])
    }

    fn parameters(&self) -> Vec<Tensor<B>> {
        vec![]
    }
}
