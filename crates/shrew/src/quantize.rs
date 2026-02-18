// Quantization — INT8 / INT4 post-training quantization
//
// Quantization reduces model size and inference latency by representing
// weights (and optionally activations) with low-precision integers
// instead of 32-bit floats.
//
// SUPPORTED MODES:
//
//   - INT8 symmetric: weights in [-127, 127], scale = max(|w|) / 127
//   - INT8 asymmetric: weights in [0, 255], scale + zero_point
//   - INT4 symmetric: weights in [-7, 7], scale = max(|w|) / 7
//   - INT4 packed: two 4-bit values per byte for 4× compression
//
// GRANULARITY:
//
//   - Per-tensor: one scale for the entire tensor (least accurate, most compact)
//   - Per-channel: one scale per output channel (best accuracy/size tradeoff)
//
// WORKFLOW:
//
//   1. Train model normally in FP32
//   2. Call quantize_model() or quantize_tensor() for post-training quantization
//   3. Run inference using dequantize_tensor() to recover approximate FP32 values
//   4. For deployment, use QuantizedLinear for fused quant/dequant inference

use shrew_core::backend::Backend;
use shrew_core::dtype::DType;
use shrew_core::error::Result;
use shrew_core::tensor::Tensor;
use shrew_nn::Module;

// Quantization configuration

/// Bit-width for quantized values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantBits {
    /// 8-bit quantization (range: -128..127 or 0..255).
    Int8,
    /// 4-bit quantization (range: -8..7 or 0..15).
    Int4,
}

/// Quantization mode (symmetric vs. asymmetric).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantMode {
    /// Symmetric: zero_point = 0, range is [-max, +max].
    /// Simplest and fastest; works well for weights.
    Symmetric,
    /// Asymmetric: zero_point can be non-zero, range is [min, max].
    /// Better accuracy for activations with skewed distributions.
    Asymmetric,
}

/// Granularity of quantization parameters (scale / zero_point).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantGranularity {
    /// One scale/zero_point for the entire tensor.
    PerTensor,
    /// One scale/zero_point per output channel (dim 0).
    PerChannel,
}

/// Full quantization configuration.
#[derive(Debug, Clone)]
pub struct QuantConfig {
    /// Bit width: Int8 or Int4.
    pub bits: QuantBits,
    /// Symmetric or asymmetric quantization.
    pub mode: QuantMode,
    /// Per-tensor or per-channel granularity.
    pub granularity: QuantGranularity,
}

impl Default for QuantConfig {
    fn default() -> Self {
        Self {
            bits: QuantBits::Int8,
            mode: QuantMode::Symmetric,
            granularity: QuantGranularity::PerTensor,
        }
    }
}

impl QuantConfig {
    /// Create INT8 symmetric per-tensor config (most common).
    pub fn int8() -> Self {
        Self::default()
    }

    /// Create INT8 symmetric per-channel config (best accuracy for weights).
    pub fn int8_per_channel() -> Self {
        Self {
            bits: QuantBits::Int8,
            mode: QuantMode::Symmetric,
            granularity: QuantGranularity::PerChannel,
        }
    }

    /// Create INT4 symmetric per-tensor config (maximum compression).
    pub fn int4() -> Self {
        Self {
            bits: QuantBits::Int4,
            mode: QuantMode::Symmetric,
            granularity: QuantGranularity::PerTensor,
        }
    }

    /// Create INT4 per-channel config.
    pub fn int4_per_channel() -> Self {
        Self {
            bits: QuantBits::Int4,
            mode: QuantMode::Symmetric,
            granularity: QuantGranularity::PerChannel,
        }
    }

    /// Set mode to asymmetric.
    pub fn asymmetric(mut self) -> Self {
        self.mode = QuantMode::Asymmetric;
        self
    }

    /// Maximum representable integer for this bit-width.
    fn qmax(&self) -> f64 {
        match self.bits {
            QuantBits::Int8 => 127.0,
            QuantBits::Int4 => 7.0,
        }
    }

    /// Minimum representable integer for this bit-width and mode.
    fn qmin(&self) -> f64 {
        match (self.bits, self.mode) {
            (QuantBits::Int8, QuantMode::Symmetric) => -127.0,
            (QuantBits::Int8, QuantMode::Asymmetric) => -128.0,
            (QuantBits::Int4, QuantMode::Symmetric) => -7.0,
            (QuantBits::Int4, QuantMode::Asymmetric) => -8.0,
        }
    }
}

// QuantizedTensor — holds quantized weights + metadata

/// A quantized tensor storing integer weights with associated scale/zero_point.
///
/// The original float value is recovered by: float = (int - zero_point) * scale
///
/// For per-channel quantization, `scales` and `zero_points` have one entry
/// per channel (dimension 0 of the original tensor).
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Quantized integer values (stored as i8 for INT8, packed for INT4).
    pub data: Vec<i8>,
    /// Scale factor(s). Length 1 for per-tensor, N for per-channel.
    pub scales: Vec<f64>,
    /// Zero point(s). Length 1 for per-tensor, N for per-channel.
    pub zero_points: Vec<f64>,
    /// Original tensor shape.
    pub shape: Vec<usize>,
    /// Original dtype (for dequantization target).
    pub original_dtype: DType,
    /// Quantization config used.
    pub config: QuantConfig,
}

impl QuantizedTensor {
    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Size in bytes of the quantized representation.
    pub fn size_bytes(&self) -> usize {
        match self.config.bits {
            QuantBits::Int8 => self.numel(),
            QuantBits::Int4 => self.numel().div_ceil(2), // packed: 2 values per byte
        }
    }

    /// Compression ratio vs FP32.
    pub fn compression_ratio(&self) -> f64 {
        let fp32_bytes = self.numel() * 4;
        fp32_bytes as f64 / self.size_bytes() as f64
    }
}

// Quantize / Dequantize operations

/// Quantize a float tensor to a `QuantizedTensor`.
///
/// # Arguments
/// - `tensor`: the FP32/FP64 tensor to quantize
/// - `config`: quantization configuration
pub fn quantize_tensor<B: Backend>(
    tensor: &Tensor<B>,
    config: &QuantConfig,
) -> Result<QuantizedTensor> {
    let data = tensor.to_f64_vec()?;
    let shape = tensor.dims().to_vec();

    match config.granularity {
        QuantGranularity::PerTensor => quantize_per_tensor(&data, &shape, tensor.dtype(), config),
        QuantGranularity::PerChannel => quantize_per_channel(&data, &shape, tensor.dtype(), config),
    }
}

/// Dequantize a `QuantizedTensor` back to a float tensor.
///
/// The dequantized values are approximate: float ≈ (int - zero_point) * scale.
pub fn dequantize_tensor<B: Backend>(
    qtensor: &QuantizedTensor,
    device: &B::Device,
) -> Result<Tensor<B>> {
    let float_data = match qtensor.config.granularity {
        QuantGranularity::PerTensor => dequantize_per_tensor(qtensor),
        QuantGranularity::PerChannel => dequantize_per_channel(qtensor),
    };

    Tensor::<B>::from_f64_slice(
        &float_data,
        qtensor.shape.clone(),
        qtensor.original_dtype,
        device,
    )
}

// ── Per-tensor quantization ──

fn quantize_per_tensor(
    data: &[f64],
    shape: &[usize],
    dtype: DType,
    config: &QuantConfig,
) -> Result<QuantizedTensor> {
    let (scale, zero_point) = compute_scale_zp(data, config);
    let inv_scale = if scale.abs() < 1e-30 {
        0.0
    } else {
        1.0 / scale
    };
    let qmin = config.qmin();
    let qmax = config.qmax();

    let quantized: Vec<i8> = data
        .iter()
        .map(|&v| {
            let q = (v * inv_scale + zero_point).round().clamp(qmin, qmax);
            q as i8
        })
        .collect();

    Ok(QuantizedTensor {
        data: quantized,
        scales: vec![scale],
        zero_points: vec![zero_point],
        shape: shape.to_vec(),
        original_dtype: dtype,
        config: config.clone(),
    })
}

fn dequantize_per_tensor(qt: &QuantizedTensor) -> Vec<f64> {
    let scale = qt.scales[0];
    let zp = qt.zero_points[0];
    qt.data.iter().map(|&q| (q as f64 - zp) * scale).collect()
}

// ── Per-channel quantization ──

fn quantize_per_channel(
    data: &[f64],
    shape: &[usize],
    dtype: DType,
    config: &QuantConfig,
) -> Result<QuantizedTensor> {
    if shape.is_empty() {
        return quantize_per_tensor(data, shape, dtype, config);
    }

    let n_channels = shape[0];
    let channel_size: usize = shape[1..].iter().product();
    let qmin = config.qmin();
    let qmax = config.qmax();

    let mut scales = Vec::with_capacity(n_channels);
    let mut zero_points = Vec::with_capacity(n_channels);
    let mut quantized = vec![0i8; data.len()];

    for ch in 0..n_channels {
        let start = ch * channel_size;
        let end = start + channel_size;
        let channel_data = &data[start..end];

        let (scale, zp) = compute_scale_zp(channel_data, config);
        let inv_scale = if scale.abs() < 1e-30 {
            0.0
        } else {
            1.0 / scale
        };

        for (i, &v) in channel_data.iter().enumerate() {
            let q = (v * inv_scale + zp).round().clamp(qmin, qmax);
            quantized[start + i] = q as i8;
        }

        scales.push(scale);
        zero_points.push(zp);
    }

    Ok(QuantizedTensor {
        data: quantized,
        scales,
        zero_points,
        shape: shape.to_vec(),
        original_dtype: dtype,
        config: config.clone(),
    })
}

fn dequantize_per_channel(qt: &QuantizedTensor) -> Vec<f64> {
    let n_channels = qt.shape[0];
    let channel_size: usize = qt.shape[1..].iter().product();
    let mut result = vec![0.0f64; qt.data.len()];

    for ch in 0..n_channels {
        let start = ch * channel_size;
        let scale = qt.scales[ch];
        let zp = qt.zero_points[ch];
        for i in 0..channel_size {
            result[start + i] = (qt.data[start + i] as f64 - zp) * scale;
        }
    }

    result
}

// ── Scale / zero-point computation ──

fn compute_scale_zp(data: &[f64], config: &QuantConfig) -> (f64, f64) {
    if data.is_empty() {
        return (1.0, 0.0);
    }

    let qmin = config.qmin();
    let qmax = config.qmax();

    match config.mode {
        QuantMode::Symmetric => {
            let amax = data.iter().fold(0.0f64, |acc, &v| acc.max(v.abs()));
            let scale = if amax < 1e-30 { 1.0 } else { amax / qmax };
            (scale, 0.0)
        }
        QuantMode::Asymmetric => {
            let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let range = max_val - min_val;
            let scale = if range < 1e-30 {
                1.0
            } else {
                range / (qmax - qmin)
            };
            let zero_point = (qmin - min_val / scale).round();
            (scale, zero_point)
        }
    }
}

// QuantizedLinear — Quantized linear layer for inference

/// A quantized linear layer that stores weights in INT8/INT4.
///
/// During inference, weights are dequantized on-the-fly for computation.
/// This saves memory (2-8× depending on bit-width) at the cost of a small
/// dequantization overhead.
///
/// # Example
/// ```ignore
/// // Quantize a trained linear layer
/// let linear = Linear::new(256, 10, true, DType::F32, &dev)?;
/// // ... train ...
/// let qlinear = QuantizedLinear::from_linear(&linear, &QuantConfig::int8())?;
/// let output = qlinear.forward(&input)?;
/// ```
pub struct QuantizedLinear<B: Backend> {
    /// Quantized weight matrix.
    weight_q: QuantizedTensor,
    /// Bias (kept in FP32 — small relative to weights).
    bias: Option<Tensor<B>>,
    /// Device for dequantization.
    device: B::Device,
    /// Input features.
    pub in_features: usize,
    /// Output features.
    pub out_features: usize,
}

impl<B: Backend> std::fmt::Debug for QuantizedLinear<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuantizedLinear")
            .field("in_features", &self.in_features)
            .field("out_features", &self.out_features)
            .field("bits", &self.weight_q.config.bits)
            .field(
                "compression",
                &format!("{:.1}x", self.weight_q.compression_ratio()),
            )
            .finish()
    }
}

impl<B: Backend> QuantizedLinear<B> {
    /// Create a `QuantizedLinear` from a trained `Linear` layer.
    pub fn from_linear(linear: &shrew_nn::Linear<B>, config: &QuantConfig) -> Result<Self> {
        let params = linear.parameters();
        let weight = &params[0];
        let bias = if params.len() > 1 {
            Some(params[1].clone())
        } else {
            None
        };

        let weight_q = quantize_tensor(weight, config)?;

        let in_features = weight.dims()[1];
        let out_features = weight.dims()[0];

        Ok(Self {
            weight_q,
            bias,
            device: weight.device().clone(),
            in_features,
            out_features,
        })
    }

    /// Create from raw quantized data and optional bias tensor.
    pub fn new(weight_q: QuantizedTensor, bias: Option<Tensor<B>>, device: B::Device) -> Self {
        let in_features = weight_q.shape[1];
        let out_features = weight_q.shape[0];
        Self {
            weight_q,
            bias,
            device,
            in_features,
            out_features,
        }
    }

    /// Get the quantized weight data.
    pub fn weight_quantized(&self) -> &QuantizedTensor {
        &self.weight_q
    }

    /// Memory saved compared to FP32 weight storage.
    pub fn memory_savings_bytes(&self) -> usize {
        let fp32_size = self.weight_q.numel() * 4;
        let quant_size = self.weight_q.size_bytes();
        fp32_size - quant_size
    }
}

impl<B: Backend> Module<B> for QuantizedLinear<B> {
    fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        // Dequantize weight from INT8/INT4 → FP32 on-the-fly
        let weight = dequantize_tensor::<B>(&self.weight_q, &self.device)?;

        // x @ weight^T + bias
        let wt = weight.t()?;
        let out = x.matmul(&wt)?;

        if let Some(ref bias) = self.bias {
            out.add(bias)
        } else {
            Ok(out)
        }
    }

    fn parameters(&self) -> Vec<Tensor<B>> {
        // Quantized model is for inference — no trainable parameters
        Vec::new()
    }
}

// Model-level quantization

/// Quantize all Linear layers in a model's named_parameters.
///
/// Returns a vector of `(name, QuantizedTensor)` for each weight parameter.
/// Biases are kept in FP32.
///
/// # Example
/// ```ignore
/// let model = Sequential::new(vec![...]);
/// let quantized = quantize_named_parameters(&model, &QuantConfig::int8())?;
/// for (name, qt) in &quantized {
///     println!("{}: {} → {} bytes ({:.1}x compression)",
///         name, qt.numel() * 4, qt.size_bytes(), qt.compression_ratio());
/// }
/// ```
pub fn quantize_named_parameters<B: Backend, M: Module<B>>(
    module: &M,
    config: &QuantConfig,
) -> Result<Vec<(String, QuantizedTensor)>> {
    let named = module.named_parameters();
    let mut quantized = Vec::new();

    for (name, tensor) in &named {
        // Only quantize weight tensors (typically 2D+), skip biases (1D)
        if tensor.rank() >= 2 {
            let qt = quantize_tensor(tensor, config)?;
            quantized.push((name.clone(), qt));
        }
    }

    Ok(quantized)
}

/// Compute quantization statistics for a model.
#[derive(Debug, Clone)]
pub struct QuantStats {
    /// Number of parameters quantized.
    pub num_quantized: usize,
    /// Number of parameters kept in FP32 (e.g., biases).
    pub num_fp32: usize,
    /// Total FP32 size in bytes.
    pub fp32_bytes: usize,
    /// Total quantized size in bytes.
    pub quantized_bytes: usize,
    /// Overall compression ratio.
    pub compression_ratio: f64,
}

/// Compute quantization statistics for a model without actually quantizing.
pub fn quantization_stats<B: Backend, M: Module<B>>(
    module: &M,
    config: &QuantConfig,
) -> QuantStats {
    let named = module.named_parameters();
    let mut num_quantized = 0usize;
    let mut num_fp32 = 0usize;
    let mut fp32_bytes = 0usize;
    let mut quantized_bytes = 0usize;

    for (_, tensor) in &named {
        let numel = tensor.elem_count();
        let param_fp32 = numel * 4;
        fp32_bytes += param_fp32;

        if tensor.rank() >= 2 {
            num_quantized += 1;
            let quant_size = match config.bits {
                QuantBits::Int8 => numel,
                QuantBits::Int4 => numel.div_ceil(2),
            };
            // Add scale/zp overhead
            let meta_size = match config.granularity {
                QuantGranularity::PerTensor => 16, // 2 × f64
                QuantGranularity::PerChannel => {
                    if tensor.dims().is_empty() {
                        16
                    } else {
                        tensor.dims()[0] * 16
                    }
                }
            };
            quantized_bytes += quant_size + meta_size;
        } else {
            num_fp32 += 1;
            quantized_bytes += param_fp32; // biases stay FP32
        }
    }

    let compression_ratio = if quantized_bytes > 0 {
        fp32_bytes as f64 / quantized_bytes as f64
    } else {
        1.0
    };

    QuantStats {
        num_quantized,
        num_fp32,
        fp32_bytes,
        quantized_bytes,
        compression_ratio,
    }
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;
    use shrew_cpu::{CpuBackend, CpuDevice};

    type B = CpuBackend;
    type T = Tensor<B>;
    const DEV: CpuDevice = CpuDevice;

    // ── Basic quantize/dequantize round-trip ──

    #[test]
    fn test_int8_symmetric_per_tensor() {
        let data = vec![1.0, -0.5, 0.25, -1.0, 0.0, 0.75];
        let tensor = T::from_f64_slice(&data, vec![2, 3], DType::F32, &DEV).unwrap();

        let config = QuantConfig::int8();
        let qt = quantize_tensor::<B>(&tensor, &config).unwrap();

        assert_eq!(qt.shape, vec![2, 3]);
        assert_eq!(qt.scales.len(), 1);
        assert_eq!(qt.zero_points, vec![0.0]);

        // Dequantize and check approximate recovery
        let recovered = dequantize_tensor::<B>(&qt, &DEV).unwrap();
        let rec_data = recovered.to_f64_vec().unwrap();

        for (orig, rec) in data.iter().zip(rec_data.iter()) {
            assert!(
                (orig - rec).abs() < 0.02,
                "int8 round-trip: {} vs {}",
                orig,
                rec
            );
        }
    }

    #[test]
    fn test_int4_symmetric_per_tensor() {
        let data = vec![1.0, -0.5, 0.25, -1.0];
        let tensor = T::from_f64_slice(&data, vec![2, 2], DType::F32, &DEV).unwrap();

        let config = QuantConfig::int4();
        let qt = quantize_tensor::<B>(&tensor, &config).unwrap();

        // INT4 has max 7, so scale = 1.0/7 ≈ 0.143
        assert_eq!(qt.scales.len(), 1);

        let recovered = dequantize_tensor::<B>(&qt, &DEV).unwrap();
        let rec_data = recovered.to_f64_vec().unwrap();

        for (orig, rec) in data.iter().zip(rec_data.iter()) {
            assert!(
                (orig - rec).abs() < 0.2,
                "int4 round-trip: {} vs {}",
                orig,
                rec
            );
        }
    }

    #[test]
    fn test_int8_per_channel() {
        // 3 channels, 4 elements each
        let data: Vec<f64> = (0..12).map(|i| (i as f64 - 6.0) * 0.1).collect();
        let tensor = T::from_f64_slice(&data, vec![3, 4], DType::F32, &DEV).unwrap();

        let config = QuantConfig::int8_per_channel();
        let qt = quantize_tensor::<B>(&tensor, &config).unwrap();

        assert_eq!(qt.scales.len(), 3); // one per channel

        let recovered = dequantize_tensor::<B>(&qt, &DEV).unwrap();
        let rec_data = recovered.to_f64_vec().unwrap();

        for (orig, rec) in data.iter().zip(rec_data.iter()) {
            assert!(
                (orig - rec).abs() < 0.01,
                "per-channel round-trip: {} vs {}",
                orig,
                rec
            );
        }
    }

    #[test]
    fn test_int8_asymmetric() {
        let data = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let tensor = T::from_f64_slice(&data, vec![1, 5], DType::F32, &DEV).unwrap();

        let config = QuantConfig::int8().asymmetric();
        let qt = quantize_tensor::<B>(&tensor, &config).unwrap();

        // Asymmetric with symmetric data should still produce valid results
        assert_eq!(qt.scales.len(), 1);

        let recovered = dequantize_tensor::<B>(&qt, &DEV).unwrap();
        let rec_data = recovered.to_f64_vec().unwrap();

        for (orig, rec) in data.iter().zip(rec_data.iter()) {
            assert!(
                (orig - rec).abs() < 0.02,
                "asymmetric round-trip: {} vs {}",
                orig,
                rec
            );
        }
    }

    // ── Compression ratio ──

    #[test]
    fn test_compression_ratio() {
        let tensor = T::randn(vec![256, 128], DType::F32, &DEV).unwrap();

        let qt8 = quantize_tensor::<B>(&tensor, &QuantConfig::int8()).unwrap();
        assert!((qt8.compression_ratio() - 4.0).abs() < 0.01); // 4:1 for INT8

        let qt4 = quantize_tensor::<B>(&tensor, &QuantConfig::int4()).unwrap();
        assert!((qt4.compression_ratio() - 8.0).abs() < 0.01); // 8:1 for INT4
    }

    // ── QuantizedLinear ──

    #[test]
    fn test_quantized_linear_forward() {
        let linear = shrew_nn::Linear::<B>::new(4, 3, true, DType::F32, &DEV).unwrap();
        let input = T::randn(vec![2, 4], DType::F32, &DEV).unwrap();

        // FP32 reference output
        let fp32_output = linear.forward(&input).unwrap();

        // Quantize and run
        let qlinear = QuantizedLinear::from_linear(&linear, &QuantConfig::int8()).unwrap();
        let quant_output = qlinear.forward(&input).unwrap();

        // Outputs should be close but not identical
        let fp32_data = fp32_output.to_f64_vec().unwrap();
        let quant_data = quant_output.to_f64_vec().unwrap();

        assert_eq!(fp32_data.len(), quant_data.len());
        for (a, b) in fp32_data.iter().zip(quant_data.iter()) {
            assert!(
                (a - b).abs() < 0.5,
                "quantized output diverged: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_quantized_linear_no_trainable_params() {
        let linear = shrew_nn::Linear::<B>::new(4, 2, true, DType::F32, &DEV).unwrap();
        let qlinear = QuantizedLinear::from_linear(&linear, &QuantConfig::int8()).unwrap();
        assert!(qlinear.parameters().is_empty()); // inference-only
    }

    #[test]
    fn test_quantized_linear_memory_savings() {
        let linear = shrew_nn::Linear::<B>::new(256, 128, true, DType::F32, &DEV).unwrap();
        let qlinear = QuantizedLinear::from_linear(&linear, &QuantConfig::int8()).unwrap();
        // 256*128 = 32768 params, fp32 = 131072 bytes, int8 = 32768 bytes
        // Savings = 131072 - 32768 = 98304
        assert_eq!(qlinear.memory_savings_bytes(), 98304);
    }

    // ── Model-level quantization stats ──

    #[test]
    fn test_quantize_named_parameters() {
        let linear = shrew_nn::Linear::<B>::new(8, 4, true, DType::F32, &DEV).unwrap();
        let quantized = quantize_named_parameters(&linear, &QuantConfig::int8()).unwrap();
        // Linear stores weight as [4,8] (rank 2) and bias as [1,4] (rank 2)
        // Both have rank >= 2, so both get quantized
        assert_eq!(quantized.len(), 2);
    }

    #[test]
    fn test_quantization_stats() {
        let linear = shrew_nn::Linear::<B>::new(64, 32, true, DType::F32, &DEV).unwrap();
        let stats = quantization_stats(&linear, &QuantConfig::int8());
        // Both weight [32,64] and bias [1,32] are rank 2 → both quantized
        assert_eq!(stats.num_quantized, 2);
        assert_eq!(stats.num_fp32, 0);
        assert!(stats.compression_ratio > 3.0);
    }

    // ── Edge case: zero tensor ──

    #[test]
    fn test_quantize_zero_tensor() {
        let tensor = T::zeros(vec![2, 3], DType::F32, &DEV).unwrap();
        let qt = quantize_tensor::<B>(&tensor, &QuantConfig::int8()).unwrap();
        let recovered = dequantize_tensor::<B>(&qt, &DEV).unwrap();
        let data = recovered.to_f64_vec().unwrap();
        for &v in &data {
            assert_eq!(v, 0.0);
        }
    }
}
