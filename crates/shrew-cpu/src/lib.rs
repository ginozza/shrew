//! # shrew-cpu
//!
//! CPU backend implementation for Shrew.
//!
//! This crate implements the [`Backend`](shrew_core::Backend) trait for CPU execution.
//! Uses [`gemm`] for SIMD-accelerated matrix multiplication (AVX2/AVX-512/FMA)
//! and [`rayon`] for parallel batched matmul and large elementwise ops.
// reference implementation: everything runs on the CPU using standard Rust
// iterators and (eventually) SIMD optimizations.
//
// Architecture:
//   CpuBackend — the Backend implementor
//   CpuDevice  — trivial device (there's only one CPU)
//   CpuStorage — enum over typed Vec<T> for each DType

pub mod ops;

use shrew_core::backend::{
    Backend, BackendDevice, BackendStorage, BinaryOp, CmpOp, ReduceOp, UnaryOp,
};
use shrew_core::dtype::DType;
use shrew_core::error::{Error, Result};
use shrew_core::layout::Layout;
use shrew_core::shape::Shape;

// CpuDevice — The CPU "device" (trivial: there's only one CPU)

/// The CPU device. Since every machine has exactly one CPU (from our
/// perspective), this is a zero-sized type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuDevice;

impl BackendDevice for CpuDevice {
    fn name(&self) -> String {
        "cpu".to_string()
    }
}

// CpuStorage — Typed storage for tensor data in CPU memory
//
// We use an enum over Vec<T> rather than a type-erased Vec<u8> because:
// 1. We can iterate with the correct type without unsafe casting
// 2. Pattern matching makes the code explicit and safe
// 3. The compiler can optimize typed operations better

/// Storage of tensor data in CPU memory.
///
/// Each variant holds a Vec of the corresponding Rust type.
/// Operations pattern-match on this enum to dispatch to typed code.
#[derive(Debug, Clone)]
pub enum CpuStorage {
    F16(Vec<half::f16>),
    BF16(Vec<half::bf16>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    U8(Vec<u8>),
    U32(Vec<u32>),
    I64(Vec<i64>),
}

impl BackendStorage for CpuStorage {
    fn dtype(&self) -> DType {
        match self {
            CpuStorage::F16(_) => DType::F16,
            CpuStorage::BF16(_) => DType::BF16,
            CpuStorage::F32(_) => DType::F32,
            CpuStorage::F64(_) => DType::F64,
            CpuStorage::U8(_) => DType::U8,
            CpuStorage::U32(_) => DType::U32,
            CpuStorage::I64(_) => DType::I64,
        }
    }

    fn len(&self) -> usize {
        match self {
            CpuStorage::F16(v) => v.len(),
            CpuStorage::BF16(v) => v.len(),
            CpuStorage::F32(v) => v.len(),
            CpuStorage::F64(v) => v.len(),
            CpuStorage::U8(v) => v.len(),
            CpuStorage::U32(v) => v.len(),
            CpuStorage::I64(v) => v.len(),
        }
    }
}

// CpuBackend — The main CPU backend struct

/// CPU backend. Implements Backend by running operations on CPU via iterators.
#[derive(Debug, Clone)]
pub struct CpuBackend;

// Half-precision helpers: promote to F32, compute, demote back

/// If storage is F16 or BF16, return the target half dtype.
fn half_dtype(s: &CpuStorage) -> Option<DType> {
    match s {
        CpuStorage::F16(_) => Some(DType::F16),
        CpuStorage::BF16(_) => Some(DType::BF16),
        _ => None,
    }
}

/// Promote F16/BF16 storage to contiguous F32. Returns unchanged for other types.
fn promote_f32(s: &CpuStorage, layout: &Layout) -> (CpuStorage, Layout) {
    match s {
        CpuStorage::F16(data) => {
            let f32_data: Vec<f32> = layout
                .strided_indices()
                .map(|idx| data[idx].to_f32())
                .collect();
            (
                CpuStorage::F32(f32_data),
                Layout::contiguous(layout.shape().clone()),
            )
        }
        CpuStorage::BF16(data) => {
            let f32_data: Vec<f32> = layout
                .strided_indices()
                .map(|idx| data[idx].to_f32())
                .collect();
            (
                CpuStorage::F32(f32_data),
                Layout::contiguous(layout.shape().clone()),
            )
        }
        _ => (s.clone(), layout.clone()),
    }
}

/// Demote F32 result back to the target half dtype.
fn demote_f32(s: CpuStorage, target: DType) -> Result<CpuStorage> {
    match (&s, target) {
        (CpuStorage::F32(data), DType::F16) => Ok(CpuStorage::F16(
            data.iter().map(|&v| half::f16::from_f32(v)).collect(),
        )),
        (CpuStorage::F32(data), DType::BF16) => Ok(CpuStorage::BF16(
            data.iter().map(|&v| half::bf16::from_f32(v)).collect(),
        )),
        _ => Ok(s),
    }
}

impl Backend for CpuBackend {
    type Device = CpuDevice;
    type Storage = CpuStorage;

    fn zeros(shape: &Shape, dtype: DType, _device: &CpuDevice) -> Result<CpuStorage> {
        let n = shape.elem_count();
        Ok(match dtype {
            DType::F16 => CpuStorage::F16(vec![half::f16::ZERO; n]),
            DType::BF16 => CpuStorage::BF16(vec![half::bf16::ZERO; n]),
            DType::F32 => CpuStorage::F32(vec![0.0f32; n]),
            DType::F64 => CpuStorage::F64(vec![0.0f64; n]),
            DType::U8 => CpuStorage::U8(vec![0u8; n]),
            DType::U32 => CpuStorage::U32(vec![0u32; n]),
            DType::I64 => CpuStorage::I64(vec![0i64; n]),
        })
    }

    fn ones(shape: &Shape, dtype: DType, _device: &CpuDevice) -> Result<CpuStorage> {
        let n = shape.elem_count();
        Ok(match dtype {
            DType::F16 => CpuStorage::F16(vec![half::f16::ONE; n]),
            DType::BF16 => CpuStorage::BF16(vec![half::bf16::ONE; n]),
            DType::F32 => CpuStorage::F32(vec![1.0f32; n]),
            DType::F64 => CpuStorage::F64(vec![1.0f64; n]),
            DType::U8 => CpuStorage::U8(vec![1u8; n]),
            DType::U32 => CpuStorage::U32(vec![1u32; n]),
            DType::I64 => CpuStorage::I64(vec![1i64; n]),
        })
    }

    fn full(shape: &Shape, val: f64, dtype: DType, _device: &CpuDevice) -> Result<CpuStorage> {
        let n = shape.elem_count();
        Ok(match dtype {
            DType::F16 => CpuStorage::F16(vec![half::f16::from_f64(val); n]),
            DType::BF16 => CpuStorage::BF16(vec![half::bf16::from_f64(val); n]),
            DType::F32 => CpuStorage::F32(vec![val as f32; n]),
            DType::F64 => CpuStorage::F64(vec![val; n]),
            DType::U8 => CpuStorage::U8(vec![val as u8; n]),
            DType::U32 => CpuStorage::U32(vec![val as u32; n]),
            DType::I64 => CpuStorage::I64(vec![val as i64; n]),
        })
    }

    fn from_f64_slice(data: &[f64], dtype: DType, _device: &CpuDevice) -> Result<CpuStorage> {
        Ok(match dtype {
            DType::F16 => CpuStorage::F16(data.iter().map(|&v| half::f16::from_f64(v)).collect()),
            DType::BF16 => {
                CpuStorage::BF16(data.iter().map(|&v| half::bf16::from_f64(v)).collect())
            }
            DType::F32 => CpuStorage::F32(data.iter().map(|&v| v as f32).collect()),
            DType::F64 => CpuStorage::F64(data.to_vec()),
            DType::U8 => CpuStorage::U8(data.iter().map(|&v| v as u8).collect()),
            DType::U32 => CpuStorage::U32(data.iter().map(|&v| v as u32).collect()),
            DType::I64 => CpuStorage::I64(data.iter().map(|&v| v as i64).collect()),
        })
    }

    fn rand_uniform(shape: &Shape, dtype: DType, _device: &CpuDevice) -> Result<CpuStorage> {
        use rand::Rng;
        let n = shape.elem_count();
        let mut rng = rand::thread_rng();
        Ok(match dtype {
            DType::F16 => CpuStorage::F16(
                (0..n)
                    .map(|_| half::f16::from_f32(rng.gen::<f32>()))
                    .collect(),
            ),
            DType::BF16 => CpuStorage::BF16(
                (0..n)
                    .map(|_| half::bf16::from_f32(rng.gen::<f32>()))
                    .collect(),
            ),
            DType::F32 => CpuStorage::F32((0..n).map(|_| rng.gen::<f32>()).collect()),
            DType::F64 => CpuStorage::F64((0..n).map(|_| rng.gen::<f64>()).collect()),
            _ => {
                return Err(Error::msg(format!(
                    "rand_uniform not supported for {:?}",
                    dtype
                )))
            }
        })
    }

    fn rand_normal(shape: &Shape, dtype: DType, _device: &CpuDevice) -> Result<CpuStorage> {
        use rand::Rng;
        use rand_distr::StandardNormal;
        let n = shape.elem_count();
        let mut rng = rand::thread_rng();
        Ok(match dtype {
            DType::F16 => CpuStorage::F16(
                (0..n)
                    .map(|_| half::f16::from_f32(rng.sample::<f32, _>(StandardNormal)))
                    .collect(),
            ),
            DType::BF16 => CpuStorage::BF16(
                (0..n)
                    .map(|_| half::bf16::from_f32(rng.sample::<f32, _>(StandardNormal)))
                    .collect(),
            ),
            DType::F32 => CpuStorage::F32(
                (0..n)
                    .map(|_| rng.sample::<f32, _>(StandardNormal))
                    .collect(),
            ),
            DType::F64 => CpuStorage::F64(
                (0..n)
                    .map(|_| rng.sample::<f64, _>(StandardNormal))
                    .collect(),
            ),
            _ => {
                return Err(Error::msg(format!(
                    "rand_normal not supported for {:?}",
                    dtype
                )))
            }
        })
    }

    fn binary_op(
        op: BinaryOp,
        lhs: &CpuStorage,
        lhs_layout: &Layout,
        rhs: &CpuStorage,
        rhs_layout: &Layout,
    ) -> Result<CpuStorage> {
        let target = half_dtype(lhs).or(half_dtype(rhs));
        if let Some(dt) = target {
            let (l, ll) = promote_f32(lhs, lhs_layout);
            let (r, rl) = promote_f32(rhs, rhs_layout);
            let result = ops::binary_op(op, &l, &ll, &r, &rl)?;
            return demote_f32(result, dt);
        }
        ops::binary_op(op, lhs, lhs_layout, rhs, rhs_layout)
    }

    fn unary_op(op: UnaryOp, input: &CpuStorage, layout: &Layout) -> Result<CpuStorage> {
        if let Some(dt) = half_dtype(input) {
            let (s, l) = promote_f32(input, layout);
            let result = ops::unary_op(op, &s, &l)?;
            return demote_f32(result, dt);
        }
        ops::unary_op(op, input, layout)
    }

    fn reduce_op(
        op: ReduceOp,
        input: &CpuStorage,
        layout: &Layout,
        dims: &[usize],
        keep_dim: bool,
    ) -> Result<CpuStorage> {
        if let Some(dt) = half_dtype(input) {
            let (s, l) = promote_f32(input, layout);
            let result = ops::reduce_op(op, &s, &l, dims, keep_dim)?;
            // ArgMax/ArgMin return I64, don't demote those
            if matches!(op, ReduceOp::ArgMax | ReduceOp::ArgMin) {
                return Ok(result);
            }
            return demote_f32(result, dt);
        }
        ops::reduce_op(op, input, layout, dims, keep_dim)
    }

    fn matmul(
        lhs: &CpuStorage,
        lhs_layout: &Layout,
        rhs: &CpuStorage,
        rhs_layout: &Layout,
    ) -> Result<CpuStorage> {
        let target = half_dtype(lhs).or(half_dtype(rhs));
        if let Some(dt) = target {
            let (l, ll) = promote_f32(lhs, lhs_layout);
            let (r, rl) = promote_f32(rhs, rhs_layout);
            let result = ops::matmul(&l, &ll, &r, &rl)?;
            return demote_f32(result, dt);
        }
        ops::matmul(lhs, lhs_layout, rhs, rhs_layout)
    }

    fn to_contiguous(input: &CpuStorage, layout: &Layout) -> Result<CpuStorage> {
        match input {
            CpuStorage::F16(data) => {
                let out: Vec<half::f16> = layout.strided_indices().map(|i| data[i]).collect();
                Ok(CpuStorage::F16(out))
            }
            CpuStorage::BF16(data) => {
                let out: Vec<half::bf16> = layout.strided_indices().map(|i| data[i]).collect();
                Ok(CpuStorage::BF16(out))
            }
            _ => ops::to_contiguous(input, layout),
        }
    }

    fn to_f64_vec(input: &CpuStorage, layout: &Layout) -> Result<Vec<f64>> {
        match input {
            CpuStorage::F16(data) => {
                Ok(layout.strided_indices().map(|i| data[i].to_f64()).collect())
            }
            CpuStorage::BF16(data) => {
                Ok(layout.strided_indices().map(|i| data[i].to_f64()).collect())
            }
            _ => ops::to_f64_vec(input, layout),
        }
    }

    fn cmp_op(
        op: CmpOp,
        lhs: &CpuStorage,
        lhs_layout: &Layout,
        rhs: &CpuStorage,
        rhs_layout: &Layout,
    ) -> Result<CpuStorage> {
        let target = half_dtype(lhs).or(half_dtype(rhs));
        if target.is_some() {
            let (l, ll) = promote_f32(lhs, lhs_layout);
            let (r, rl) = promote_f32(rhs, rhs_layout);
            // cmp_op returns U8, no demotion needed
            return ops::cmp_op(op, &l, &ll, &r, &rl);
        }
        ops::cmp_op(op, lhs, lhs_layout, rhs, rhs_layout)
    }

    fn affine(input: &CpuStorage, layout: &Layout, mul: f64, add: f64) -> Result<CpuStorage> {
        if let Some(dt) = half_dtype(input) {
            let (s, l) = promote_f32(input, layout);
            let result = ops::affine(&s, &l, mul, add)?;
            return demote_f32(result, dt);
        }
        ops::affine(input, layout, mul, add)
    }

    fn index_select(
        input: &CpuStorage,
        input_layout: &Layout,
        indices: &CpuStorage,
        indices_layout: &Layout,
        dim: usize,
    ) -> Result<CpuStorage> {
        if let Some(dt) = half_dtype(input) {
            let (s, l) = promote_f32(input, input_layout);
            let result = ops::index_select(&s, &l, indices, indices_layout, dim)?;
            return demote_f32(result, dt);
        }
        ops::index_select(input, input_layout, indices, indices_layout, dim)
    }

    fn powf(input: &CpuStorage, layout: &Layout, exponent: f64) -> Result<CpuStorage> {
        if let Some(dt) = half_dtype(input) {
            let (s, l) = promote_f32(input, layout);
            let result = ops::powf(&s, &l, exponent)?;
            return demote_f32(result, dt);
        }
        ops::powf(input, layout, exponent)
    }

    fn clamp(input: &CpuStorage, layout: &Layout, min: f64, max: f64) -> Result<CpuStorage> {
        if let Some(dt) = half_dtype(input) {
            let (s, l) = promote_f32(input, layout);
            let result = ops::clamp(&s, &l, min, max)?;
            return demote_f32(result, dt);
        }
        ops::clamp(input, layout, min, max)
    }

    fn where_cond(
        mask: &CpuStorage,
        mask_layout: &Layout,
        on_true: &CpuStorage,
        on_true_layout: &Layout,
        on_false: &CpuStorage,
        on_false_layout: &Layout,
    ) -> Result<CpuStorage> {
        let target = half_dtype(on_true).or(half_dtype(on_false));
        if let Some(dt) = target {
            let (t, tl) = promote_f32(on_true, on_true_layout);
            let (f, fl) = promote_f32(on_false, on_false_layout);
            let result = ops::where_cond(mask, mask_layout, &t, &tl, &f, &fl)?;
            return demote_f32(result, dt);
        }
        ops::where_cond(
            mask,
            mask_layout,
            on_true,
            on_true_layout,
            on_false,
            on_false_layout,
        )
    }

    fn gather(
        input: &CpuStorage,
        input_layout: &Layout,
        index: &CpuStorage,
        index_layout: &Layout,
        dim: usize,
    ) -> Result<CpuStorage> {
        if let Some(dt) = half_dtype(input) {
            let (s, l) = promote_f32(input, input_layout);
            let result = ops::gather(&s, &l, index, index_layout, dim)?;
            return demote_f32(result, dt);
        }
        ops::gather(input, input_layout, index, index_layout, dim)
    }

    fn cat(inputs: &[(&CpuStorage, &Layout)], out_shape: &Shape, dim: usize) -> Result<CpuStorage> {
        // Check if any input is half
        let target = inputs.iter().find_map(|(s, _)| half_dtype(s));
        if let Some(dt) = target {
            let promoted: Vec<(CpuStorage, Layout)> =
                inputs.iter().map(|(s, l)| promote_f32(s, l)).collect();
            let refs: Vec<(&CpuStorage, &Layout)> = promoted.iter().map(|(s, l)| (s, l)).collect();
            let result = ops::cat(&refs, out_shape, dim)?;
            return demote_f32(result, dt);
        }
        ops::cat(inputs, out_shape, dim)
    }
}

/// Convenience type alias for CPU tensors.
pub type CpuTensor = shrew_core::Tensor<CpuBackend>;
