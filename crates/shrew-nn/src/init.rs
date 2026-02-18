// nn::init — Parameter Initialization Utilities
//
// Standalone functions for creating initialized tensors, following PyTorch's
// `torch.nn.init` module. These are useful when building custom layers or
// when you need fine-grained control over initialization.
//
// AVAILABLE INITIALIZERS:
//
//   uniform(shape, low, high)       — U(low, high)
//   normal(shape, mean, std)        — N(mean, std)
//   constant(shape, val)            — all elements = val
//   zeros(shape)                    — all zeros
//   ones(shape)                     — all ones
//   xavier_uniform(shape, gain)     — Glorot uniform
//   xavier_normal(shape, gain)      — Glorot normal
//   kaiming_uniform(shape, a, mode) — He uniform (for ReLU)
//   kaiming_normal(shape, a, mode)  — He normal  (for ReLU)
//
// All functions return Tensor<B> with `set_variable()` already called,
// making them ready for gradient tracking.

use shrew_core::backend::Backend;
use shrew_core::dtype::DType;
use shrew_core::error::Result;
use shrew_core::shape::Shape;
use shrew_core::tensor::Tensor;

/// Fan computation mode for Kaiming initialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FanMode {
    /// Use fan_in (input features). Default, preserves variance in forward pass.
    FanIn,
    /// Use fan_out (output features). Preserves variance in backward pass.
    FanOut,
}

/// Compute (fan_in, fan_out) from a shape.
///
/// - For 1-D: fan_in = fan_out = dims[0]
/// - For 2-D: fan_in = dims[1], fan_out = dims[0]
/// - For 3-D+: fan_in = dims[1] * product(dims[2..]),
///   fan_out = dims[0] * product(dims[2..])
///   (convolution-style: dims[0]=out_channels, dims[1]=in_channels, rest=kernel)
fn compute_fans(shape: &Shape) -> (f64, f64) {
    let dims = shape.dims();
    match dims.len() {
        0 => (1.0, 1.0),
        1 => (dims[0] as f64, dims[0] as f64),
        2 => (dims[1] as f64, dims[0] as f64),
        _ => {
            let receptive_field: usize = dims[2..].iter().product();
            let fan_in = dims[1] as f64 * receptive_field as f64;
            let fan_out = dims[0] as f64 * receptive_field as f64;
            (fan_in, fan_out)
        }
    }
}

/// Initialize a tensor from a uniform distribution U(low, high).
pub fn uniform<B: Backend>(
    shape: impl Into<Shape>,
    low: f64,
    high: f64,
    dtype: DType,
    device: &B::Device,
) -> Result<Tensor<B>> {
    let shape = shape.into();
    let range = high - low;
    let t = Tensor::<B>::rand(shape, dtype, device)?
        .affine(range, low)?
        .set_variable();
    Ok(t)
}

/// Initialize a tensor from a normal distribution N(mean, std).
pub fn normal<B: Backend>(
    shape: impl Into<Shape>,
    mean: f64,
    std: f64,
    dtype: DType,
    device: &B::Device,
) -> Result<Tensor<B>> {
    let shape = shape.into();
    let t = Tensor::<B>::randn(shape, dtype, device)?
        .affine(std, mean)?
        .set_variable();
    Ok(t)
}

/// Initialize a tensor with a constant value.
pub fn constant<B: Backend>(
    shape: impl Into<Shape>,
    val: f64,
    dtype: DType,
    device: &B::Device,
) -> Result<Tensor<B>> {
    let t = Tensor::<B>::full(shape, val, dtype, device)?.set_variable();
    Ok(t)
}

/// Initialize a tensor with all zeros (as a variable).
pub fn zeros<B: Backend>(
    shape: impl Into<Shape>,
    dtype: DType,
    device: &B::Device,
) -> Result<Tensor<B>> {
    let t = Tensor::<B>::zeros(shape, dtype, device)?.set_variable();
    Ok(t)
}

/// Initialize a tensor with all ones (as a variable).
pub fn ones<B: Backend>(
    shape: impl Into<Shape>,
    dtype: DType,
    device: &B::Device,
) -> Result<Tensor<B>> {
    let t = Tensor::<B>::ones(shape, dtype, device)?.set_variable();
    Ok(t)
}

/// Xavier (Glorot) uniform initialization.
///
/// Draws from U(-a, a) where a = gain * sqrt(6 / (fan_in + fan_out)).
/// Designed to keep variance constant across layers with linear activations.
///
/// # Arguments
/// - `gain`: scaling factor (1.0 for linear/sigmoid, sqrt(2) for ReLU)
pub fn xavier_uniform<B: Backend>(
    shape: impl Into<Shape>,
    gain: f64,
    dtype: DType,
    device: &B::Device,
) -> Result<Tensor<B>> {
    let shape = shape.into();
    let (fan_in, fan_out) = compute_fans(&shape);
    let a = gain * (6.0 / (fan_in + fan_out)).sqrt();
    uniform::<B>(shape, -a, a, dtype, device)
}

/// Xavier (Glorot) normal initialization.
///
/// Draws from N(0, std) where std = gain * sqrt(2 / (fan_in + fan_out)).
pub fn xavier_normal<B: Backend>(
    shape: impl Into<Shape>,
    gain: f64,
    dtype: DType,
    device: &B::Device,
) -> Result<Tensor<B>> {
    let shape = shape.into();
    let (fan_in, fan_out) = compute_fans(&shape);
    let std = gain * (2.0 / (fan_in + fan_out)).sqrt();
    normal::<B>(shape, 0.0, std, dtype, device)
}

/// Kaiming (He) uniform initialization.
///
/// Draws from U(-bound, bound) where bound = sqrt(3 * gain² / fan).
/// Designed for layers followed by ReLU (or variants).
///
/// # Arguments
/// - `a`: negative slope of the rectifier (0 for ReLU, 0.01 for LeakyReLU)
/// - `mode`: `FanIn` or `FanOut`
pub fn kaiming_uniform<B: Backend>(
    shape: impl Into<Shape>,
    a: f64,
    mode: FanMode,
    dtype: DType,
    device: &B::Device,
) -> Result<Tensor<B>> {
    let shape = shape.into();
    let (fan_in, fan_out) = compute_fans(&shape);
    let fan = match mode {
        FanMode::FanIn => fan_in,
        FanMode::FanOut => fan_out,
    };
    let gain_sq = 2.0 / (1.0 + a * a);
    let bound = (3.0 * gain_sq / fan).sqrt();
    uniform::<B>(shape, -bound, bound, dtype, device)
}

/// Kaiming (He) normal initialization.
///
/// Draws from N(0, std) where std = sqrt(gain² / fan).
///
/// # Arguments
/// - `a`: negative slope of the rectifier (0 for ReLU)
/// - `mode`: `FanIn` or `FanOut`
pub fn kaiming_normal<B: Backend>(
    shape: impl Into<Shape>,
    a: f64,
    mode: FanMode,
    dtype: DType,
    device: &B::Device,
) -> Result<Tensor<B>> {
    let shape = shape.into();
    let (fan_in, fan_out) = compute_fans(&shape);
    let fan = match mode {
        FanMode::FanIn => fan_in,
        FanMode::FanOut => fan_out,
    };
    let gain_sq = 2.0 / (1.0 + a * a);
    let std = (gain_sq / fan).sqrt();
    normal::<B>(shape, 0.0, std, dtype, device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use shrew_cpu::{CpuBackend, CpuDevice};

    type T = Tensor<CpuBackend>;

    #[test]
    fn test_xavier_uniform_shape() {
        let dev = CpuDevice;
        let t = xavier_uniform::<CpuBackend>((128, 64), 1.0, DType::F32, &dev).unwrap();
        assert_eq!(t.dims(), &[128, 64]);
        assert_eq!(t.dtype(), DType::F32);
    }

    #[test]
    fn test_xavier_normal_shape() {
        let dev = CpuDevice;
        let t = xavier_normal::<CpuBackend>((64, 32), 1.0, DType::F64, &dev).unwrap();
        assert_eq!(t.dims(), &[64, 32]);
    }

    #[test]
    fn test_kaiming_uniform_bounds() {
        let dev = CpuDevice;
        // fan_in = 100 for shape (50, 100), gain = sqrt(2) for ReLU (a=0)
        let t: T = kaiming_uniform((50, 100), 0.0, FanMode::FanIn, DType::F64, &dev).unwrap();
        let v = t.to_f64_vec().unwrap();
        let bound = (3.0 * 2.0 / 100.0_f64).sqrt(); // sqrt(6/100)
        for &x in &v {
            assert!(
                x >= -bound - 1e-6 && x <= bound + 1e-6,
                "value {} out of bounds [-{}, {}]",
                x,
                bound,
                bound
            );
        }
    }

    #[test]
    fn test_kaiming_normal_shape() {
        let dev = CpuDevice;
        let t: T = kaiming_normal((32, 16), 0.0, FanMode::FanOut, DType::F32, &dev).unwrap();
        assert_eq!(t.dims(), &[32, 16]);
    }

    #[test]
    fn test_constant_values() {
        let dev = CpuDevice;
        let t: T = constant((3, 4), 7.0, DType::F64, &dev).unwrap();
        let v = t.to_f64_vec().unwrap();
        for &x in &v {
            assert!((x - 7.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_zeros_values() {
        let dev = CpuDevice;
        let t: T = zeros(5, DType::F64, &dev).unwrap();
        let v = t.to_f64_vec().unwrap();
        for &x in &v {
            assert!(x.abs() < 1e-10);
        }
    }

    #[test]
    fn test_ones_values() {
        let dev = CpuDevice;
        let t: T = ones((2, 3), DType::F64, &dev).unwrap();
        let v = t.to_f64_vec().unwrap();
        for &x in &v {
            assert!((x - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_uniform_range() {
        let dev = CpuDevice;
        let t: T = uniform((1000,), -2.0, 3.0, DType::F64, &dev).unwrap();
        let v = t.to_f64_vec().unwrap();
        for &x in &v {
            assert!(x >= -2.0 - 1e-6 && x <= 3.0 + 1e-6);
        }
    }

    #[test]
    fn test_normal_stats() {
        let dev = CpuDevice;
        let t: T = normal((10000,), 5.0, 0.1, DType::F64, &dev).unwrap();
        let v = t.to_f64_vec().unwrap();
        let mean: f64 = v.iter().sum::<f64>() / v.len() as f64;
        assert!((mean - 5.0).abs() < 0.05, "mean {} too far from 5.0", mean);
    }

    #[test]
    fn test_compute_fans_conv() {
        // Conv2d: [out_ch=16, in_ch=3, kh=5, kw=5]
        let shape = Shape::from((16, 3, 5, 5));
        let (fan_in, fan_out) = compute_fans(&shape);
        assert_eq!(fan_in, 3.0 * 25.0); // 75
        assert_eq!(fan_out, 16.0 * 25.0); // 400
    }
}
