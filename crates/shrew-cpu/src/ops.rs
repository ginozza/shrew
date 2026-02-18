// CPU Operations — The actual math implementations
//
// This module contains all the numerical operations for CPU tensors.
// Each function takes storage (the raw data) + layout (shape/strides/offset)
// and returns new storage with the result.
//
// KEY CONCEPT: Layout-aware iteration
//
// Tensors might not be contiguous in memory (e.g., after transpose or narrow).
// We use Layout::strided_indices() to iterate over elements in logical order,
// mapping each to the correct flat index in the underlying storage.
//
// For example, a transposed [3,2] matrix with strides [1,3]:
//   Logical [0,0] → storage[0], [0,1] → storage[3],
//   Logical [1,0] → storage[1], [1,1] → storage[4], etc.
//
// PERFORMANCE NOTE:
// matmul uses the `gemm` crate for SIMD-accelerated BLAS (auto AVX2/AVX-512/FMA).
// Elementwise ops use contiguous fast-paths + rayon parallelism for large tensors.

use crate::CpuStorage;
use shrew_core::backend::{BinaryOp, CmpOp, ReduceOp, UnaryOp};
use shrew_core::error::{Error, Result};
use shrew_core::layout::Layout;
use shrew_core::shape::Shape;

/// Minimum number of elements to trigger rayon parallel iteration.
const PAR_THRESHOLD: usize = 100_000;

// Helper: Extract typed slice from CpuStorage

/// Macro to implement an operation for all float types.
/// Uses contiguous fast-path when possible, with rayon for large tensors.
/// Both F32 and F64 compute natively — no f32→f64 upcast overhead.
macro_rules! map_float_storage {
    ($storage:expr, $layout:expr, |$val:ident| $body:expr) => {
        match $storage {
            CpuStorage::F32(data) => {
                let n = $layout.elem_count();
                let result: Vec<f32> = if $layout.is_contiguous() {
                    let off = $layout.offset();
                    let slice = &data[off..off + n];
                    if n >= PAR_THRESHOLD {
                        use rayon::prelude::*;
                        slice
                            .par_iter()
                            .map(|v| {
                                let $val = *v;
                                $body
                            })
                            .collect()
                    } else {
                        slice
                            .iter()
                            .map(|v| {
                                let $val = *v;
                                $body
                            })
                            .collect()
                    }
                } else {
                    $layout
                        .strided_indices()
                        .map(|idx| {
                            let $val = data[idx];
                            $body
                        })
                        .collect()
                };
                CpuStorage::F32(result)
            }
            CpuStorage::F64(data) => {
                let n = $layout.elem_count();
                let result: Vec<f64> = if $layout.is_contiguous() {
                    let off = $layout.offset();
                    let slice = &data[off..off + n];
                    if n >= PAR_THRESHOLD {
                        use rayon::prelude::*;
                        slice
                            .par_iter()
                            .map(|v| {
                                let $val = *v;
                                $body
                            })
                            .collect()
                    } else {
                        slice
                            .iter()
                            .map(|v| {
                                let $val = *v;
                                $body
                            })
                            .collect()
                    }
                } else {
                    $layout
                        .strided_indices()
                        .map(|idx| {
                            let $val = data[idx];
                            $body
                        })
                        .collect()
                };
                CpuStorage::F64(result)
            }
            _ => return Err(Error::msg("operation only supported on float types")),
        }
    };
}

// Binary operations: add, sub, mul, div

#[allow(clippy::needless_range_loop)]
pub fn binary_op(
    op: BinaryOp,
    lhs: &CpuStorage,
    lhs_layout: &Layout,
    rhs: &CpuStorage,
    rhs_layout: &Layout,
) -> Result<CpuStorage> {
    use shrew_core::shape::Shape;

    let lhs_shape = lhs_layout.shape();
    let rhs_shape = rhs_layout.shape();

    // Compute broadcast output shape
    let out_shape = Shape::broadcast_shape(lhs_shape, rhs_shape)?;
    let out_count = out_shape.elem_count();
    let out_strides = out_shape.stride_contiguous();
    let out_rank = out_shape.rank();

    // Compute broadcast strides for each input
    // We need strides into the CONTIGUOUS data (after layout application):
    // First, we read data respecting each layout's strides into contiguous arrays,
    // then apply broadcast strides on top.
    let lhs_bcast_strides = lhs_shape.broadcast_strides(&out_shape);
    let rhs_bcast_strides = rhs_shape.broadcast_strides(&out_shape);

    // Fast path: no broadcasting needed (same element count)
    if lhs_layout.elem_count() == rhs_layout.elem_count() && lhs_layout.elem_count() == out_count {
        // Original non-broadcast path — optimized for contiguous layouts
        macro_rules! binary_typed_fast {
            ($lhs_data:expr, $rhs_data:expr, $T:ty, $ctor:path) => {{
                let n = lhs_layout.elem_count();

                // Super-fast path: both contiguous → direct slice ops, optionally parallel
                if lhs_layout.is_contiguous() && rhs_layout.is_contiguous() {
                    let lo = lhs_layout.offset();
                    let ro = rhs_layout.offset();
                    let l_slice = &$lhs_data[lo..lo + n];
                    let r_slice = &$rhs_data[ro..ro + n];

                    let result: Vec<$T> = if n >= PAR_THRESHOLD {
                        use rayon::prelude::*;
                        l_slice
                            .par_iter()
                            .zip(r_slice.par_iter())
                            .map(|(&a, &b)| match op {
                                BinaryOp::Add => a + b,
                                BinaryOp::Sub => a - b,
                                BinaryOp::Mul => a * b,
                                BinaryOp::Div => a / b,
                            })
                            .collect()
                    } else {
                        l_slice
                            .iter()
                            .zip(r_slice.iter())
                            .map(|(&a, &b)| match op {
                                BinaryOp::Add => a + b,
                                BinaryOp::Sub => a - b,
                                BinaryOp::Mul => a * b,
                                BinaryOp::Div => a / b,
                            })
                            .collect()
                    };
                    return Ok($ctor(result));
                }

                // Strided path: at least one input is non-contiguous
                let lhs_iter = lhs_layout.strided_indices();
                let rhs_iter = rhs_layout.strided_indices();
                let result: Vec<$T> = lhs_iter
                    .zip(rhs_iter)
                    .map(|(li, ri)| {
                        let a = $lhs_data[li];
                        let b = $rhs_data[ri];
                        match op {
                            BinaryOp::Add => a + b,
                            BinaryOp::Sub => a - b,
                            BinaryOp::Mul => a * b,
                            BinaryOp::Div => a / b,
                        }
                    })
                    .collect();
                Ok($ctor(result))
            }};
        }

        return match (lhs, rhs) {
            (CpuStorage::F32(l), CpuStorage::F32(r)) => {
                binary_typed_fast!(l, r, f32, CpuStorage::F32)
            }
            (CpuStorage::F64(l), CpuStorage::F64(r)) => {
                binary_typed_fast!(l, r, f64, CpuStorage::F64)
            }
            (CpuStorage::I64(l), CpuStorage::I64(r)) => {
                binary_typed_fast!(l, r, i64, CpuStorage::I64)
            }
            (CpuStorage::U32(l), CpuStorage::U32(r)) => {
                binary_typed_fast!(l, r, u32, CpuStorage::U32)
            }
            _ => Err(Error::msg("binary op: dtype mismatch between lhs and rhs")),
        };
    }

    // Slow path: broadcasting required
    // First, materialize both inputs as contiguous f64 arrays via layout
    // Then index with broadcast strides
    let lhs_data = to_f64_vec(lhs, lhs_layout)?;
    let rhs_data = to_f64_vec(rhs, rhs_layout)?;

    // Contiguous strides for the materialized data
    let lhs_cont_strides = lhs_shape.stride_contiguous();
    let rhs_cont_strides = rhs_shape.stride_contiguous();

    let mut result = vec![0.0f64; out_count];

    for flat in 0..out_count {
        // Convert flat index to multi-dimensional index in output
        let mut remainder = flat;
        let mut lhs_idx = 0usize;
        let mut rhs_idx = 0usize;

        for d in 0..out_rank {
            let coord = if out_strides[d] > 0 {
                remainder / out_strides[d]
            } else {
                0
            };
            if out_strides[d] > 0 {
                remainder %= out_strides[d];
            }

            // Map to lhs index: if broadcast stride is 0, coordinate is 0
            let lhs_coord = if lhs_bcast_strides[d] > 0 { coord } else { 0 };
            let rhs_coord = if rhs_bcast_strides[d] > 0 { coord } else { 0 };

            // Map coords to flat index in the contiguous data
            let lhs_offset = d
                .checked_sub(out_rank - lhs_shape.rank())
                .unwrap_or(out_rank);
            let rhs_offset = d
                .checked_sub(out_rank - rhs_shape.rank())
                .unwrap_or(out_rank);

            if lhs_offset < lhs_shape.rank() {
                lhs_idx += lhs_coord * lhs_cont_strides[lhs_offset];
            }
            if rhs_offset < rhs_shape.rank() {
                rhs_idx += rhs_coord * rhs_cont_strides[rhs_offset];
            }
        }

        let a = lhs_data[lhs_idx];
        let b = rhs_data[rhs_idx];
        result[flat] = match op {
            BinaryOp::Add => a + b,
            BinaryOp::Sub => a - b,
            BinaryOp::Mul => a * b,
            BinaryOp::Div => a / b,
        };
    }

    // Convert back to the appropriate dtype
    match lhs {
        CpuStorage::F32(_) => Ok(CpuStorage::F32(result.iter().map(|&v| v as f32).collect())),
        CpuStorage::F64(_) => Ok(CpuStorage::F64(result)),
        CpuStorage::I64(_) => Ok(CpuStorage::I64(result.iter().map(|&v| v as i64).collect())),
        CpuStorage::U32(_) => Ok(CpuStorage::U32(result.iter().map(|&v| v as u32).collect())),
        _ => Err(Error::msg("binary op: unsupported dtype")),
    }
}

// Unary operations: neg, abs, exp, log, sqrt, relu, sigmoid, tanh, gelu, etc.

pub fn unary_op(op: UnaryOp, input: &CpuStorage, layout: &Layout) -> Result<CpuStorage> {
    let result = match op {
        UnaryOp::Neg => map_float_storage!(input, layout, |x| -x),
        UnaryOp::Abs => map_float_storage!(input, layout, |x| x.abs()),
        UnaryOp::Exp => map_float_storage!(input, layout, |x| x.exp()),
        UnaryOp::Log => map_float_storage!(input, layout, |x| x.ln()),
        UnaryOp::Sqrt => map_float_storage!(input, layout, |x| x.sqrt()),
        UnaryOp::Square => map_float_storage!(input, layout, |x| x * x),
        UnaryOp::Sin => map_float_storage!(input, layout, |x| x.sin()),
        UnaryOp::Cos => map_float_storage!(input, layout, |x| x.cos()),

        // ReLU: max(0, x)
        // The simplest and most widely used activation function.
        // Gradient: 1 if x > 0, else 0
        UnaryOp::Relu => map_float_storage!(input, layout, |x| if x > 0.0 { x } else { 0.0 }),

        // Sigmoid: σ(x) = 1 / (1 + e^(-x))
        // Squashes any value to range (0, 1).
        // Gradient: σ(x) * (1 - σ(x))
        UnaryOp::Sigmoid => map_float_storage!(input, layout, |x| 1.0 / (1.0 + (-x).exp())),

        // Tanh: (e^x - e^(-x)) / (e^x + e^(-x))
        // Squashes to (-1, 1). Zero-centered unlike sigmoid.
        // Gradient: 1 - tanh²(x)
        UnaryOp::Tanh => map_float_storage!(input, layout, |x| x.tanh()),

        // GELU: x * Φ(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
        // Used in GPT, BERT, and most modern transformers.
        // Smooth approximation of ReLU.
        UnaryOp::Gelu => {
            map_float_storage!(input, layout, |x| {
                0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044715 * x * x * x)).tanh())
            })
        }

        // SiLU / Swish: x * σ(x) = x / (1 + e^(-x))
        // Used in many modern architectures (EfficientNet, LLaMA).
        UnaryOp::Silu => map_float_storage!(input, layout, |x| x / (1.0 + (-x).exp())),

        // Floor: largest integer ≤ x
        UnaryOp::Floor => map_float_storage!(input, layout, |x| x.floor()),
        // Ceil: smallest integer ≥ x
        UnaryOp::Ceil => map_float_storage!(input, layout, |x| x.ceil()),
        // Round: nearest integer (half-to-even)
        UnaryOp::Round => map_float_storage!(input, layout, |x| x.round()),
    };
    Ok(result)
}

// Reduction operations: sum, mean, max, min, argmax, argmin
//
// Reductions collapse one or more dimensions of a tensor.
// - sum([2,3], dim=1) → [2]    (sum each row)
// - mean([2,3], dim=0) → [3]   (mean of each column)
// - sum_all([2,3]) → scalar     (sum everything)

pub fn reduce_op(
    op: ReduceOp,
    input: &CpuStorage,
    layout: &Layout,
    dims: &[usize],
    _keep_dim: bool,
) -> Result<CpuStorage> {
    // If dims is empty, reduce over all elements → scalar
    if dims.is_empty() {
        return reduce_all(op, input, layout);
    }

    // For single-dimension reduction, use the optimized path
    if dims.len() == 1 {
        return reduce_dim(op, input, layout, dims[0]);
    }

    // Multi-dim reduction: reduce one dim at a time
    // (This is not the most efficient approach but it's correct and clear)
    Err(Error::msg(
        "multi-dim reduction not yet implemented (use single dim or reduce_all)",
    ))
}

/// Reduce ALL elements to a single scalar.
fn reduce_all(op: ReduceOp, input: &CpuStorage, layout: &Layout) -> Result<CpuStorage> {
    macro_rules! reduce_all_typed {
        ($data:expr, $T:ty, $ctor:path) => {{
            let iter = layout.strided_indices().map(|idx| $data[idx]);
            let result: $T = match op {
                ReduceOp::Sum => iter.fold(0.0 as $T, |acc, x| acc + x),
                ReduceOp::Mean => {
                    let n = layout.elem_count() as $T;
                    iter.fold(0.0 as $T, |acc, x| acc + x) / n
                }
                ReduceOp::Max => iter.fold(<$T>::NEG_INFINITY, |acc, x| acc.max(x)),
                ReduceOp::Min => iter.fold(<$T>::INFINITY, |acc, x| acc.min(x)),
                ReduceOp::ArgMax => {
                    let (idx, _) = layout
                        .strided_indices()
                        .map(|idx| $data[idx])
                        .enumerate()
                        .fold((0usize, <$T>::NEG_INFINITY), |(best_i, best_v), (i, v)| {
                            if v > best_v {
                                (i, v)
                            } else {
                                (best_i, best_v)
                            }
                        });
                    return Ok(CpuStorage::I64(vec![idx as i64]));
                }
                ReduceOp::ArgMin => {
                    let (idx, _) = layout
                        .strided_indices()
                        .map(|idx| $data[idx])
                        .enumerate()
                        .fold((0usize, <$T>::INFINITY), |(best_i, best_v), (i, v)| {
                            if v < best_v {
                                (i, v)
                            } else {
                                (best_i, best_v)
                            }
                        });
                    return Ok(CpuStorage::I64(vec![idx as i64]));
                }
            };
            Ok($ctor(vec![result]))
        }};
    }

    match input {
        CpuStorage::F32(data) => reduce_all_typed!(data, f32, CpuStorage::F32),
        CpuStorage::F64(data) => reduce_all_typed!(data, f64, CpuStorage::F64),
        _ => Err(Error::msg("reduce only supported on float types")),
    }
}

/// Reduce along a single dimension.
///
/// Example: reduce_dim(Sum, [2,3] data, dim=1)
///   - For each "slice" along dim 1, sum the elements
///   - Result has shape [2] (dim 1 collapsed)
///
/// We iterate over all positions in the output shape, and for each,
/// sum (or max, etc.) over the reduction dimension.
fn reduce_dim(op: ReduceOp, input: &CpuStorage, layout: &Layout, dim: usize) -> Result<CpuStorage> {
    let dims = layout.dims();
    let rank = layout.rank();
    if dim >= rank {
        return Err(Error::DimOutOfRange { dim, rank });
    }

    let reduce_size = dims[dim];

    // Compute output shape (shape with dim removed)
    let out_dims: Vec<usize> = dims
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != dim)
        .map(|(_, &d)| d)
        .collect();
    let out_count = out_dims.iter().product::<usize>().max(1);

    macro_rules! reduce_dim_typed {
        ($data:expr, $T:ty, $ctor:path) => {{
            let strides = layout.strides();
            let offset = layout.offset();

            let mut result = Vec::with_capacity(out_count);

            // We need to iterate over all positions in the output shape,
            // and for each, iterate over the reduction dimension.
            // Strategy: iterate multi-dim index over output shape, then inner loop over reduce_dim.
            let out_shape = shrew_core::Shape::new(out_dims.clone());
            let out_strides_for_iter = out_shape.stride_contiguous();

            for out_flat_idx in 0..out_count {
                // Convert flat output index to multi-dim output index
                let mut out_md = vec![0usize; out_dims.len()];
                let mut remainder = out_flat_idx;
                for i in 0..out_dims.len() {
                    if out_strides_for_iter[i] > 0 {
                        out_md[i] = remainder / out_strides_for_iter[i];
                        remainder %= out_strides_for_iter[i];
                    }
                }

                // Map output multi-dim index back to input multi-dim index
                // (insert the reduction dimension)
                let mut input_md = vec![0usize; rank];
                let mut j = 0;
                for i in 0..rank {
                    if i == dim {
                        input_md[i] = 0; // will be varied in inner loop
                    } else {
                        input_md[i] = out_md[j];
                        j += 1;
                    }
                }

                // Compute base flat index in storage
                let mut base_idx = offset;
                for i in 0..rank {
                    base_idx += input_md[i] * strides[i];
                }
                let reduce_stride = strides[dim];

                // Inner loop over reduction dimension
                match op {
                    ReduceOp::Sum => {
                        let mut acc = 0.0 as $T;
                        for k in 0..reduce_size {
                            acc += $data[base_idx + k * reduce_stride];
                        }
                        result.push(acc);
                    }
                    ReduceOp::Mean => {
                        let mut acc = 0.0 as $T;
                        for k in 0..reduce_size {
                            acc += $data[base_idx + k * reduce_stride];
                        }
                        result.push(acc / reduce_size as $T);
                    }
                    ReduceOp::Max => {
                        let mut acc = <$T>::NEG_INFINITY;
                        for k in 0..reduce_size {
                            let v = $data[base_idx + k * reduce_stride];
                            if v > acc {
                                acc = v;
                            }
                        }
                        result.push(acc);
                    }
                    ReduceOp::Min => {
                        let mut acc = <$T>::INFINITY;
                        for k in 0..reduce_size {
                            let v = $data[base_idx + k * reduce_stride];
                            if v < acc {
                                acc = v;
                            }
                        }
                        result.push(acc);
                    }
                    ReduceOp::ArgMax => {
                        let mut best_k = 0usize;
                        let mut best_v = <$T>::NEG_INFINITY;
                        for k in 0..reduce_size {
                            let v = $data[base_idx + k * reduce_stride];
                            if v > best_v {
                                best_v = v;
                                best_k = k;
                            }
                        }
                        // Tricky: argmax returns I64 indices, but we're in a float-typed macro.
                        // We'll store as float temporarily and convert at the end.
                        result.push(best_k as $T);
                    }
                    ReduceOp::ArgMin => {
                        let mut best_k = 0usize;
                        let mut best_v = <$T>::INFINITY;
                        for k in 0..reduce_size {
                            let v = $data[base_idx + k * reduce_stride];
                            if v < best_v {
                                best_v = v;
                                best_k = k;
                            }
                        }
                        result.push(best_k as $T);
                    }
                }
            }

            // ArgMax/ArgMin should return I64 storage
            match op {
                ReduceOp::ArgMax | ReduceOp::ArgMin => {
                    Ok(CpuStorage::I64(result.iter().map(|&v| v as i64).collect()))
                }
                _ => Ok($ctor(result)),
            }
        }};
    }

    match input {
        CpuStorage::F32(data) => reduce_dim_typed!(data, f32, CpuStorage::F32),
        CpuStorage::F64(data) => reduce_dim_typed!(data, f64, CpuStorage::F64),
        _ => Err(Error::msg("reduce only supported on float types")),
    }
}

// Matrix multiplication: C = A @ B
//
// For [m, k] @ [k, n] → [m, n]:
//   C[i, j] = Σ_k A[i, k] * B[k, j]
//
// This is the naive O(m*n*k) implementation. It correctly handles
// non-contiguous layouts via strides. We delegate to the `gemm` crate which
// auto-detects AVX2/AVX-512/FMA for high-performance SIMD-accelerated GEMM.
// For batched matmul with batch_size > 1, we parallelize across batches with rayon.

pub fn matmul(
    lhs: &CpuStorage,
    lhs_layout: &Layout,
    rhs: &CpuStorage,
    rhs_layout: &Layout,
) -> Result<CpuStorage> {
    let lhs_dims = lhs_layout.dims();
    let rhs_dims = rhs_layout.dims();

    // Extract matrix dimensions
    let rank = lhs_dims.len();
    let m = lhs_dims[rank - 2];
    let k = lhs_dims[rank - 1];
    let n = rhs_dims[rhs_dims.len() - 1];

    // Batch size (product of all dims except last 2)
    let batch_size: usize = lhs_dims[..rank - 2].iter().product::<usize>().max(1);

    /// Perform a single GEMM: C[m×n] = A[m×k] × B[k×n] using the `gemm` crate.
    /// All data is contiguous row-major.
    ///
    /// gemm computes: dst = alpha * dst + beta * lhs * rhs
    /// (when read_dst=false, alpha is forced to zero → dst = beta * lhs * rhs)
    ///
    /// Strides: element(i,j) = ptr[i*rs + j*cs].
    /// For row-major: cs = 1, rs = number_of_columns.
    macro_rules! gemm_one_batch {
        ($a_data:expr, $b_data:expr, $c_data:expr, $m:expr, $k:expr, $n:expr, $T:ty) => {
            unsafe {
                gemm::gemm(
                    $m, // m: rows of dst/lhs
                    $n, // n: cols of dst/rhs
                    $k, // k: cols of lhs = rows of rhs
                    $c_data.as_mut_ptr(),
                    1,           // dst_cs = 1 (row-major)
                    $n as isize, // dst_rs = n (row-major)
                    false,       // read_dst: overwrite (alpha forced to 0)
                    $a_data.as_ptr(),
                    1,           // lhs_cs = 1
                    $k as isize, // lhs_rs = k
                    $b_data.as_ptr(),
                    1,           // rhs_cs = 1
                    $n as isize, // rhs_rs = n
                    (0.0 as $T), // alpha: coeff for existing dst (ignored, read_dst=false)
                    (1.0 as $T), // beta: coeff for lhs*rhs product
                    false,
                    false,
                    false, // no conjugation
                    gemm::Parallelism::None,
                );
            }
        };
    }

    macro_rules! matmul_gemm {
        ($lhs_data:expr, $rhs_data:expr, $T:ty, $ctor:path) => {{
            // Make contiguous copies if layouts are non-contiguous
            let lhs_contig: Vec<$T>;
            let rhs_contig: Vec<$T>;
            let a_slice: &[$T] = if lhs_layout.is_contiguous() && lhs_layout.offset() == 0 {
                $lhs_data
            } else {
                lhs_contig = lhs_layout
                    .strided_indices()
                    .map(|idx| $lhs_data[idx])
                    .collect();
                &lhs_contig
            };
            let b_slice: &[$T] = if rhs_layout.is_contiguous() && rhs_layout.offset() == 0 {
                $rhs_data
            } else {
                rhs_contig = rhs_layout
                    .strided_indices()
                    .map(|idx| $rhs_data[idx])
                    .collect();
                &rhs_contig
            };

            let mk = m * k;
            let kn = k * n;
            let mn = m * n;

            if batch_size > 1 {
                // Parallel across batches using rayon
                use rayon::prelude::*;
                let mut result = vec![<$T as Default>::default(); batch_size * mn];
                result
                    .par_chunks_mut(mn)
                    .enumerate()
                    .for_each(|(b, c_chunk)| {
                        let a_batch = &a_slice[b * mk..(b + 1) * mk];
                        let b_batch = &b_slice[b * kn..(b + 1) * kn];
                        gemm_one_batch!(a_batch, b_batch, c_chunk, m, k, n, $T);
                    });
                Ok($ctor(result))
            } else {
                let mut result = vec![<$T as Default>::default(); mn];
                gemm_one_batch!(&a_slice[..], &b_slice[..], &mut result[..], m, k, n, $T);
                Ok($ctor(result))
            }
        }};
    }

    match (lhs, rhs) {
        (CpuStorage::F32(l), CpuStorage::F32(r)) => matmul_gemm!(l, r, f32, CpuStorage::F32),
        (CpuStorage::F64(l), CpuStorage::F64(r)) => matmul_gemm!(l, r, f64, CpuStorage::F64),
        _ => Err(Error::msg("matmul only supported for f32/f64")),
    }
}

// Data movement

/// Create a contiguous copy of storage following the given layout.
pub fn to_contiguous(input: &CpuStorage, layout: &Layout) -> Result<CpuStorage> {
    macro_rules! contiguous_typed {
        ($data:expr, $T:ty, $ctor:path) => {{
            let result: Vec<$T> = layout.strided_indices().map(|idx| $data[idx]).collect();
            $ctor(result)
        }};
    }

    Ok(match input {
        CpuStorage::F16(data) => contiguous_typed!(data, half::f16, CpuStorage::F16),
        CpuStorage::BF16(data) => contiguous_typed!(data, half::bf16, CpuStorage::BF16),
        CpuStorage::F32(data) => contiguous_typed!(data, f32, CpuStorage::F32),
        CpuStorage::F64(data) => contiguous_typed!(data, f64, CpuStorage::F64),
        CpuStorage::U8(data) => contiguous_typed!(data, u8, CpuStorage::U8),
        CpuStorage::U32(data) => contiguous_typed!(data, u32, CpuStorage::U32),
        CpuStorage::I64(data) => contiguous_typed!(data, i64, CpuStorage::I64),
    })
}

/// Convert storage to Vec<f64>, respecting layout (strides, offset).
pub fn to_f64_vec(input: &CpuStorage, layout: &Layout) -> Result<Vec<f64>> {
    Ok(match input {
        CpuStorage::F16(data) => layout
            .strided_indices()
            .map(|idx| data[idx].to_f64())
            .collect(),
        CpuStorage::BF16(data) => layout
            .strided_indices()
            .map(|idx| data[idx].to_f64())
            .collect(),
        CpuStorage::F32(data) => layout
            .strided_indices()
            .map(|idx| data[idx] as f64)
            .collect(),
        CpuStorage::F64(data) => layout.strided_indices().map(|idx| data[idx]).collect(),
        CpuStorage::U8(data) => layout
            .strided_indices()
            .map(|idx| data[idx] as f64)
            .collect(),
        CpuStorage::U32(data) => layout
            .strided_indices()
            .map(|idx| data[idx] as f64)
            .collect(),
        CpuStorage::I64(data) => layout
            .strided_indices()
            .map(|idx| data[idx] as f64)
            .collect(),
    })
}

// Comparison ops

pub fn cmp_op(
    op: CmpOp,
    lhs: &CpuStorage,
    lhs_layout: &Layout,
    rhs: &CpuStorage,
    rhs_layout: &Layout,
) -> Result<CpuStorage> {
    if lhs_layout.elem_count() != rhs_layout.elem_count() {
        return Err(Error::ShapeMismatch {
            expected: lhs_layout.shape().clone(),
            got: rhs_layout.shape().clone(),
        });
    }

    macro_rules! cmp_typed {
        ($lhs_data:expr, $rhs_data:expr) => {{
            let result: Vec<u8> = lhs_layout
                .strided_indices()
                .zip(rhs_layout.strided_indices())
                .map(|(li, ri)| {
                    let a = $lhs_data[li] as f64;
                    let b = $rhs_data[ri] as f64;
                    let r = match op {
                        CmpOp::Eq => a == b,
                        CmpOp::Ne => a != b,
                        CmpOp::Gt => a > b,
                        CmpOp::Ge => a >= b,
                        CmpOp::Lt => a < b,
                        CmpOp::Le => a <= b,
                    };
                    if r {
                        1u8
                    } else {
                        0u8
                    }
                })
                .collect();
            Ok(CpuStorage::U8(result))
        }};
    }

    match (lhs, rhs) {
        (CpuStorage::F32(l), CpuStorage::F32(r)) => cmp_typed!(l, r),
        (CpuStorage::F64(l), CpuStorage::F64(r)) => cmp_typed!(l, r),
        (CpuStorage::I64(l), CpuStorage::I64(r)) => cmp_typed!(l, r),
        (CpuStorage::U32(l), CpuStorage::U32(r)) => cmp_typed!(l, r),
        _ => Err(Error::msg("cmp_op: dtype mismatch")),
    }
}

// Affine transform: result = input * mul + add

pub fn affine(input: &CpuStorage, layout: &Layout, mul: f64, add: f64) -> Result<CpuStorage> {
    match input {
        CpuStorage::F32(data) => {
            let m = mul as f32;
            let a = add as f32;
            let n = layout.elem_count();
            let result: Vec<f32> = if layout.is_contiguous() {
                let off = layout.offset();
                let slice = &data[off..off + n];
                if n >= PAR_THRESHOLD {
                    use rayon::prelude::*;
                    slice.par_iter().map(|&v| v * m + a).collect()
                } else {
                    slice.iter().map(|&v| v * m + a).collect()
                }
            } else {
                layout
                    .strided_indices()
                    .map(|idx| data[idx] * m + a)
                    .collect()
            };
            Ok(CpuStorage::F32(result))
        }
        CpuStorage::F64(data) => {
            let n = layout.elem_count();
            let result: Vec<f64> = if layout.is_contiguous() {
                let off = layout.offset();
                let slice = &data[off..off + n];
                if n >= PAR_THRESHOLD {
                    use rayon::prelude::*;
                    slice.par_iter().map(|&v| v * mul + add).collect()
                } else {
                    slice.iter().map(|&v| v * mul + add).collect()
                }
            } else {
                layout
                    .strided_indices()
                    .map(|idx| data[idx] * mul + add)
                    .collect()
            };
            Ok(CpuStorage::F64(result))
        }
        _ => Err(Error::msg("affine only supported on float types")),
    }
}

// Index select (gather along dimension)
//
// Selects elements along `dim` using the given index tensor.

pub fn index_select(
    input: &CpuStorage,
    input_layout: &Layout,
    indices: &CpuStorage,
    indices_layout: &Layout,
    dim: usize,
) -> Result<CpuStorage> {
    // Get indices as usize values
    let idx_vec: Vec<usize> = match indices {
        CpuStorage::U32(v) => indices_layout
            .strided_indices()
            .map(|i| v[i] as usize)
            .collect(),
        CpuStorage::I64(v) => indices_layout
            .strided_indices()
            .map(|i| v[i] as usize)
            .collect(),
        CpuStorage::F32(v) => indices_layout
            .strided_indices()
            .map(|i| v[i] as usize)
            .collect(),
        CpuStorage::F64(v) => indices_layout
            .strided_indices()
            .map(|i| v[i] as usize)
            .collect(),
        _ => return Err(Error::msg("index_select: unsupported index dtype")),
    };

    let in_dims = input_layout.dims();
    let rank = in_dims.len();
    let in_strides = input_layout.strides();

    // Compute output shape: same as input except dim is replaced by num_indices
    let num_indices = idx_vec.len();
    let mut out_dims = in_dims.to_vec();
    out_dims[dim] = num_indices;
    let out_n: usize = out_dims.iter().product();

    // Compute output strides for index decomposition
    let mut out_strides = vec![0usize; rank];
    let mut stride = 1;
    for d in (0..rank).rev() {
        out_strides[d] = stride;
        stride *= out_dims[d];
    }

    macro_rules! impl_index_select {
        ($data:expr, $ty:ty, $variant:ident) => {{
            let data = $data;
            let mut out = vec![<$ty>::default(); out_n];
            for flat_idx in 0..out_n {
                // Decompose flat output index into coordinates
                let mut remaining = flat_idx;
                let mut src_offset = input_layout.offset();
                for d in 0..rank {
                    let coord = remaining / out_strides[d];
                    remaining %= out_strides[d];
                    if d == dim {
                        // Use the looked-up index for this dim
                        src_offset += idx_vec[coord] * in_strides[d];
                    } else {
                        src_offset += coord * in_strides[d];
                    }
                }
                out[flat_idx] = data[src_offset];
            }
            Ok(CpuStorage::$variant(out))
        }};
    }

    match input {
        CpuStorage::F32(data) => impl_index_select!(data, f32, F32),
        CpuStorage::F64(data) => impl_index_select!(data, f64, F64),
        CpuStorage::U8(data) => impl_index_select!(data, u8, U8),
        CpuStorage::U32(data) => impl_index_select!(data, u32, U32),
        CpuStorage::I64(data) => impl_index_select!(data, i64, I64),
        _ => Err(Error::msg("index_select: unsupported input dtype")),
    }
}

// Powf — element-wise power (scalar exponent)

pub fn powf(input: &CpuStorage, layout: &Layout, exponent: f64) -> Result<CpuStorage> {
    match input {
        CpuStorage::F32(data) => {
            let exp = exponent as f32;
            let n = layout.elem_count();
            let result: Vec<f32> = if layout.is_contiguous() {
                let off = layout.offset();
                let slice = &data[off..off + n];
                if n >= PAR_THRESHOLD {
                    use rayon::prelude::*;
                    slice.par_iter().map(|&v| v.powf(exp)).collect()
                } else {
                    slice.iter().map(|&v| v.powf(exp)).collect()
                }
            } else {
                layout
                    .strided_indices()
                    .map(|idx| data[idx].powf(exp))
                    .collect()
            };
            Ok(CpuStorage::F32(result))
        }
        CpuStorage::F64(data) => {
            let n = layout.elem_count();
            let result: Vec<f64> = if layout.is_contiguous() {
                let off = layout.offset();
                let slice = &data[off..off + n];
                if n >= PAR_THRESHOLD {
                    use rayon::prelude::*;
                    slice.par_iter().map(|&v| v.powf(exponent)).collect()
                } else {
                    slice.iter().map(|&v| v.powf(exponent)).collect()
                }
            } else {
                layout
                    .strided_indices()
                    .map(|idx| data[idx].powf(exponent))
                    .collect()
            };
            Ok(CpuStorage::F64(result))
        }
        _ => Err(Error::msg("powf only supported on float types")),
    }
}

// Clamp — element-wise clamp to [min, max]

pub fn clamp(
    input: &CpuStorage,
    layout: &Layout,
    min_val: f64,
    max_val: f64,
) -> Result<CpuStorage> {
    match input {
        CpuStorage::F32(data) => {
            let lo = min_val as f32;
            let hi = max_val as f32;
            let n = layout.elem_count();
            let result: Vec<f32> = if layout.is_contiguous() {
                let off = layout.offset();
                let slice = &data[off..off + n];
                if n >= PAR_THRESHOLD {
                    use rayon::prelude::*;
                    slice.par_iter().map(|&v| v.max(lo).min(hi)).collect()
                } else {
                    slice.iter().map(|&v| v.max(lo).min(hi)).collect()
                }
            } else {
                layout
                    .strided_indices()
                    .map(|idx| data[idx].max(lo).min(hi))
                    .collect()
            };
            Ok(CpuStorage::F32(result))
        }
        CpuStorage::F64(data) => {
            let n = layout.elem_count();
            let result: Vec<f64> = if layout.is_contiguous() {
                let off = layout.offset();
                let slice = &data[off..off + n];
                if n >= PAR_THRESHOLD {
                    use rayon::prelude::*;
                    slice
                        .par_iter()
                        .map(|&v| v.max(min_val).min(max_val))
                        .collect()
                } else {
                    slice.iter().map(|&v| v.max(min_val).min(max_val)).collect()
                }
            } else {
                layout
                    .strided_indices()
                    .map(|idx| data[idx].max(min_val).min(max_val))
                    .collect()
            };
            Ok(CpuStorage::F64(result))
        }
        _ => Err(Error::msg("clamp only supported on float types")),
    }
}

// Cat — Concatenate storages along a dimension (native typed, no f64 roundtrip)

pub fn cat(inputs: &[(&CpuStorage, &Layout)], out_shape: &Shape, dim: usize) -> Result<CpuStorage> {
    if inputs.is_empty() {
        return Err(Error::msg("cat: empty input list"));
    }

    let rank = out_shape.dims().len();
    let out_count = out_shape.elem_count();
    let out_strides = out_shape.stride_contiguous();

    // Dispatch by dtype of first input
    match &inputs[0].0 {
        CpuStorage::F32(_) => {
            let mut out = vec![0.0f32; out_count];
            let mut offset_in_dim = 0usize;
            for &(storage, layout) in inputs {
                let data = match storage {
                    CpuStorage::F32(d) => d,
                    _ => return Err(Error::msg("cat: dtype mismatch among inputs")),
                };
                let t_shape = layout.shape();
                let t_dims = t_shape.dims();
                let t_cont_strides = t_shape.stride_contiguous();
                for (flat, src_idx) in layout.strided_indices().enumerate() {
                    // Decompose flat logical index to coords
                    let mut remaining = flat;
                    let mut out_flat = 0usize;
                    for d in 0..rank {
                        let coord = if t_cont_strides[d] > 0 {
                            remaining / t_cont_strides[d]
                        } else {
                            0
                        };
                        if t_cont_strides[d] > 0 {
                            remaining %= t_cont_strides[d];
                        }
                        let shifted = if d == dim {
                            coord + offset_in_dim
                        } else {
                            coord
                        };
                        out_flat += shifted * out_strides[d];
                    }
                    out[out_flat] = data[src_idx];
                }
                offset_in_dim += t_dims[dim];
            }
            Ok(CpuStorage::F32(out))
        }
        CpuStorage::F64(_) => {
            let mut out = vec![0.0f64; out_count];
            let mut offset_in_dim = 0usize;
            for &(storage, layout) in inputs {
                let data = match storage {
                    CpuStorage::F64(d) => d,
                    _ => return Err(Error::msg("cat: dtype mismatch among inputs")),
                };
                let t_shape = layout.shape();
                let t_dims = t_shape.dims();
                let t_cont_strides = t_shape.stride_contiguous();
                for (flat, src_idx) in layout.strided_indices().enumerate() {
                    let mut remaining = flat;
                    let mut out_flat = 0usize;
                    for d in 0..rank {
                        let coord = if t_cont_strides[d] > 0 {
                            remaining / t_cont_strides[d]
                        } else {
                            0
                        };
                        if t_cont_strides[d] > 0 {
                            remaining %= t_cont_strides[d];
                        }
                        let shifted = if d == dim {
                            coord + offset_in_dim
                        } else {
                            coord
                        };
                        out_flat += shifted * out_strides[d];
                    }
                    out[out_flat] = data[src_idx];
                }
                offset_in_dim += t_dims[dim];
            }
            Ok(CpuStorage::F64(out))
        }
        _ => Err(Error::msg("cat: only float types supported")),
    }
}

// Gather — select elements along a dimension using an index tensor

pub fn gather(
    input: &CpuStorage,
    input_layout: &Layout,
    index: &CpuStorage,
    index_layout: &Layout,
    dim: usize,
) -> Result<CpuStorage> {
    let input_data = to_f64_vec(input, input_layout)?;
    let index_data = to_f64_vec(index, index_layout)?;

    let input_dims = input_layout.shape().dims();
    let index_dims = index_layout.shape().dims();
    let rank = input_dims.len();

    if rank != index_dims.len() {
        return Err(Error::msg(format!(
            "gather: input rank {} != index rank {}",
            rank,
            index_dims.len()
        )));
    }
    if dim >= rank {
        return Err(Error::msg(format!(
            "gather: dim {} out of range for rank {}",
            dim, rank
        )));
    }

    // Compute contiguous strides for input
    let mut input_strides = vec![1usize; rank];
    for d in (0..rank - 1).rev() {
        input_strides[d] = input_strides[d + 1] * input_dims[d + 1];
    }

    let mut index_strides = vec![1usize; rank];
    for d in (0..rank - 1).rev() {
        index_strides[d] = index_strides[d + 1] * index_dims[d + 1];
    }

    let total = index_data.len();
    let mut result = vec![0.0f64; total];

    for flat_idx in 0..total {
        // Decompose flat_idx into multi-dimensional index for output/index tensor
        let mut remaining = flat_idx;
        let mut coords = vec![0usize; rank];
        for d in 0..rank {
            coords[d] = remaining / index_strides[d];
            remaining %= index_strides[d];
        }

        // The index value along `dim`
        let gathered_dim = index_data[flat_idx] as usize;
        if gathered_dim >= input_dims[dim] {
            return Err(Error::msg(format!(
                "gather: index {} out of range for dim {} (size {})",
                gathered_dim, dim, input_dims[dim]
            )));
        }

        // Build input coordinates: same as output except at `dim`
        let mut input_coords = coords.clone();
        input_coords[dim] = gathered_dim;

        // Compute flat index into input
        let input_flat: usize = input_coords
            .iter()
            .zip(input_strides.iter())
            .map(|(&c, &s)| c * s)
            .sum();

        result[flat_idx] = input_data[input_flat];
    }

    // Output dtype matches input
    match input {
        CpuStorage::F32(_) => Ok(CpuStorage::F32(result.iter().map(|&v| v as f32).collect())),
        CpuStorage::F64(_) => Ok(CpuStorage::F64(result)),
        _ => Err(Error::msg("gather: unsupported dtype")),
    }
}

// Where / conditional select

pub fn where_cond(
    mask: &CpuStorage,
    mask_layout: &Layout,
    on_true: &CpuStorage,
    on_true_layout: &Layout,
    on_false: &CpuStorage,
    on_false_layout: &Layout,
) -> Result<CpuStorage> {
    // Read mask as f64 values (works for U8, F32, F64, etc.)
    let mask_data = to_f64_vec(mask, mask_layout)?;
    let true_data = to_f64_vec(on_true, on_true_layout)?;
    let false_data = to_f64_vec(on_false, on_false_layout)?;

    let n = mask_data.len();
    if true_data.len() != n || false_data.len() != n {
        return Err(Error::msg(format!(
            "where_cond: shape mismatch — mask {}, on_true {}, on_false {}",
            n,
            true_data.len(),
            false_data.len()
        )));
    }

    let result: Vec<f64> = mask_data
        .iter()
        .zip(true_data.iter())
        .zip(false_data.iter())
        .map(|((&m, &t), &f)| if m != 0.0 { t } else { f })
        .collect();

    // Output dtype = on_true dtype
    let out_dtype = match on_true {
        CpuStorage::F32(_) => shrew_core::dtype::DType::F32,
        CpuStorage::F64(_) => shrew_core::dtype::DType::F64,
        _ => return Err(Error::msg("where_cond: unsupported dtype for on_true")),
    };
    match out_dtype {
        shrew_core::dtype::DType::F32 => {
            Ok(CpuStorage::F32(result.iter().map(|&v| v as f32).collect()))
        }
        shrew_core::dtype::DType::F64 => Ok(CpuStorage::F64(result)),
        _ => Err(Error::msg(format!(
            "where_cond: unsupported dtype {:?}",
            out_dtype
        ))),
    }
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuDevice, CpuTensor};
    use shrew_core::dtype::DType;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn assert_vec_approx(got: &[f64], expected: &[f64], tol: f64) {
        assert_eq!(got.len(), expected.len(), "length mismatch");
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                approx_eq(*g, *e, tol),
                "index {}: got {} expected {} (tol {})",
                i,
                g,
                e,
                tol
            );
        }
    }

    #[test]
    fn test_zeros_ones() -> Result<()> {
        let dev = CpuDevice;
        let z = CpuTensor::zeros((2, 3), DType::F32, &dev)?;
        assert_eq!(z.to_f64_vec()?, vec![0.0; 6]);

        let o = CpuTensor::ones((2, 3), DType::F32, &dev)?;
        assert_eq!(o.to_f64_vec()?, vec![1.0; 6]);
        Ok(())
    }

    #[test]
    fn test_from_slice() -> Result<()> {
        let dev = CpuDevice;
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = CpuTensor::from_f64_slice(&data, (2, 3), DType::F64, &dev)?;
        assert_eq!(t.to_f64_vec()?, data);
        assert_eq!(t.dims(), &[2, 3]);
        Ok(())
    }

    #[test]
    fn test_add() -> Result<()> {
        let dev = CpuDevice;
        let a = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0], (2, 2), DType::F32, &dev)?;
        let b = CpuTensor::from_f64_slice(&[10.0, 20.0, 30.0, 40.0], (2, 2), DType::F32, &dev)?;
        let c = a.add(&b)?;
        assert_vec_approx(&c.to_f64_vec()?, &[11.0, 22.0, 33.0, 44.0], 1e-5);
        Ok(())
    }

    #[test]
    fn test_mul() -> Result<()> {
        let dev = CpuDevice;
        let a = CpuTensor::from_f64_slice(&[2.0, 3.0, 4.0, 5.0], (2, 2), DType::F32, &dev)?;
        let b = CpuTensor::from_f64_slice(&[10.0, 10.0, 10.0, 10.0], (2, 2), DType::F32, &dev)?;
        let c = a.mul(&b)?;
        assert_vec_approx(&c.to_f64_vec()?, &[20.0, 30.0, 40.0, 50.0], 1e-5);
        Ok(())
    }

    #[test]
    fn test_relu() -> Result<()> {
        let dev = CpuDevice;
        let a = CpuTensor::from_f64_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], 6, DType::F32, &dev)?;
        let r = a.relu()?;
        assert_vec_approx(&r.to_f64_vec()?, &[0.0, 0.0, 0.0, 1.0, 2.0, 3.0], 1e-5);
        Ok(())
    }

    #[test]
    fn test_sigmoid() -> Result<()> {
        let dev = CpuDevice;
        let a = CpuTensor::from_f64_slice(&[0.0], (), DType::F64, &dev)?;
        let s = a.sigmoid()?;
        assert!(approx_eq(s.to_scalar_f64()?, 0.5, 1e-10));
        Ok(())
    }

    #[test]
    fn test_sum_all() -> Result<()> {
        let dev = CpuDevice;
        let a =
            CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), DType::F32, &dev)?;
        let s = a.sum_all()?;
        assert!(approx_eq(s.to_scalar_f64()?, 21.0, 1e-5));
        Ok(())
    }

    #[test]
    fn test_sum_dim() -> Result<()> {
        let dev = CpuDevice;
        // [[1, 2, 3],
        //  [4, 5, 6]]
        let a =
            CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), DType::F32, &dev)?;

        // Sum along dim 0 (columns): [5, 7, 9]
        let s0 = a.sum(0, false)?;
        assert_vec_approx(&s0.to_f64_vec()?, &[5.0, 7.0, 9.0], 1e-5);

        // Sum along dim 1 (rows): [6, 15]
        let s1 = a.sum(1, false)?;
        assert_vec_approx(&s1.to_f64_vec()?, &[6.0, 15.0], 1e-5);
        Ok(())
    }

    #[test]
    fn test_mean_dim() -> Result<()> {
        let dev = CpuDevice;
        let a =
            CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), DType::F32, &dev)?;
        let m = a.mean(1, false)?;
        assert_vec_approx(&m.to_f64_vec()?, &[2.0, 5.0], 1e-5);
        Ok(())
    }

    #[test]
    fn test_matmul_2x3_3x2() -> Result<()> {
        let dev = CpuDevice;
        // A = [[1, 2, 3],
        //      [4, 5, 6]]
        let a =
            CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), DType::F64, &dev)?;
        // B = [[7,  8],
        //      [9,  10],
        //      [11, 12]]
        let b = CpuTensor::from_f64_slice(
            &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            (3, 2),
            DType::F64,
            &dev,
        )?;
        // C = A @ B = [[58, 64], [139, 154]]
        let c = a.matmul(&b)?;
        assert_eq!(c.dims(), &[2, 2]);
        assert_vec_approx(&c.to_f64_vec()?, &[58.0, 64.0, 139.0, 154.0], 1e-10);
        Ok(())
    }

    #[test]
    fn test_transpose() -> Result<()> {
        let dev = CpuDevice;
        // [[1, 2, 3],
        //  [4, 5, 6]]
        let a =
            CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), DType::F64, &dev)?;
        let at = a.t()?;
        assert_eq!(at.dims(), &[3, 2]);
        // Transposed: [[1, 4], [2, 5], [3, 6]]
        assert_vec_approx(&at.to_f64_vec()?, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 1e-10);
        Ok(())
    }

    #[test]
    fn test_reshape() -> Result<()> {
        let dev = CpuDevice;
        let a =
            CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), DType::F64, &dev)?;
        let b = a.reshape((3, 2))?;
        assert_eq!(b.dims(), &[3, 2]);
        assert_eq!(b.to_f64_vec()?, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        Ok(())
    }

    #[test]
    fn test_narrow() -> Result<()> {
        let dev = CpuDevice;
        // [[1, 2, 3, 4],
        //  [5, 6, 7, 8]]
        let a = CpuTensor::from_f64_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            (2, 4),
            DType::F64,
            &dev,
        )?;
        // Narrow dim 1, start=1, len=2 → [[2, 3], [6, 7]]
        let b = a.narrow(1, 1, 2)?;
        assert_eq!(b.dims(), &[2, 2]);
        assert_vec_approx(&b.to_f64_vec()?, &[2.0, 3.0, 6.0, 7.0], 1e-10);
        Ok(())
    }

    #[test]
    fn test_gelu() -> Result<()> {
        let dev = CpuDevice;
        let a = CpuTensor::from_f64_slice(&[0.0, 1.0, -1.0], 3, DType::F64, &dev)?;
        let g = a.gelu()?;
        let vals = g.to_f64_vec()?;
        // GELU(0) ≈ 0, GELU(1) ≈ 0.8412, GELU(-1) ≈ -0.1588
        assert!(approx_eq(vals[0], 0.0, 1e-5));
        assert!(approx_eq(vals[1], 0.8412, 1e-3));
        assert!(approx_eq(vals[2], -0.1588, 1e-3));
        Ok(())
    }

    #[test]
    fn test_affine() -> Result<()> {
        let dev = CpuDevice;
        let a = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0], 3, DType::F32, &dev)?;
        let b = a.affine(2.0, 1.0)?; // 2*x + 1
        assert_vec_approx(&b.to_f64_vec()?, &[3.0, 5.0, 7.0], 1e-5);
        Ok(())
    }

    #[test]
    fn test_unsqueeze_squeeze() -> Result<()> {
        let dev = CpuDevice;
        let a = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0], 3, DType::F32, &dev)?;
        let b = a.unsqueeze(0)?;
        assert_eq!(b.dims(), &[1, 3]);
        let c = b.squeeze_all();
        assert_eq!(c.dims(), &[3]);
        Ok(())
    }

    // =========================================================================
    // Autograd / backward tests
    // =========================================================================

    /// Numerical gradient check: compare analytical gradient with finite differences.
    /// f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
    fn numerical_grad(
        f: impl Fn(&CpuTensor) -> Result<CpuTensor>,
        x: &CpuTensor,
        h: f64,
    ) -> Result<Vec<f64>> {
        let x_data = x.to_f64_vec()?;
        let mut grad = vec![0.0; x_data.len()];
        for i in 0..x_data.len() {
            let mut x_plus = x_data.clone();
            let mut x_minus = x_data.clone();
            x_plus[i] += h;
            x_minus[i] -= h;
            let t_plus =
                CpuTensor::from_f64_slice(&x_plus, x.shape().clone(), x.dtype(), x.device())?;
            let t_minus =
                CpuTensor::from_f64_slice(&x_minus, x.shape().clone(), x.dtype(), x.device())?;
            let f_plus = f(&t_plus)?.to_scalar_f64()?;
            let f_minus = f(&t_minus)?.to_scalar_f64()?;
            grad[i] = (f_plus - f_minus) / (2.0 * h);
        }
        Ok(grad)
    }

    #[test]
    fn test_backward_mul() -> Result<()> {
        // f(a, b) = sum(a * b)
        // grad_a = b, grad_b = a
        let dev = CpuDevice;
        let a = CpuTensor::from_f64_slice(&[2.0, 3.0], 2, DType::F64, &dev)?.set_variable();
        let b = CpuTensor::from_f64_slice(&[4.0, 5.0], 2, DType::F64, &dev)?.set_variable();
        let c = a.mul(&b)?;
        let loss = c.sum_all()?;
        let grads = loss.backward()?;

        let grad_a = grads.get(&a).unwrap().to_f64_vec()?;
        let grad_b = grads.get(&b).unwrap().to_f64_vec()?;
        assert_vec_approx(&grad_a, &[4.0, 5.0], 1e-10); // grad_a = b
        assert_vec_approx(&grad_b, &[2.0, 3.0], 1e-10); // grad_b = a
        Ok(())
    }

    #[test]
    fn test_backward_add_sub() -> Result<()> {
        // f(a, b) = sum(a + b - a) = sum(b)
        let dev = CpuDevice;
        let a = CpuTensor::from_f64_slice(&[1.0, 2.0], 2, DType::F64, &dev)?.set_variable();
        let b = CpuTensor::from_f64_slice(&[3.0, 4.0], 2, DType::F64, &dev)?.set_variable();
        let c = a.add(&b)?;
        let d = c.sub(&a)?; // d = b
        let loss = d.sum_all()?;
        let grads = loss.backward()?;

        let grad_a = grads.get(&a).unwrap().to_f64_vec()?;
        let grad_b = grads.get(&b).unwrap().to_f64_vec()?;
        // d = (a + b) - a → grad_a = 1 (from add) + (-1) (from sub) = 0
        // grad_b = 1
        assert_vec_approx(&grad_a, &[0.0, 0.0], 1e-10);
        assert_vec_approx(&grad_b, &[1.0, 1.0], 1e-10);
        Ok(())
    }

    #[test]
    fn test_backward_div() -> Result<()> {
        // f(a, b) = sum(a / b)
        // grad_a = 1/b, grad_b = -a/b²
        let dev = CpuDevice;
        let a = CpuTensor::from_f64_slice(&[6.0], (), DType::F64, &dev)?.set_variable();
        let b = CpuTensor::from_f64_slice(&[3.0], (), DType::F64, &dev)?.set_variable();
        let c = a.div(&b)?; // 6/3 = 2
        let grads = c.backward()?;

        let grad_a = grads.get(&a).unwrap().to_scalar_f64()?;
        let grad_b = grads.get(&b).unwrap().to_scalar_f64()?;
        assert!(approx_eq(grad_a, 1.0 / 3.0, 1e-10)); // 1/b
        assert!(approx_eq(grad_b, -6.0 / 9.0, 1e-10)); // -a/b²
        Ok(())
    }

    #[test]
    fn test_backward_square_mean() -> Result<()> {
        // MSE-like: f(x) = mean(x²)
        // f'(x_i) = 2*x_i / n
        let dev = CpuDevice;
        let x = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0], 3, DType::F64, &dev)?.set_variable();
        let sq = x.square()?;
        let loss = sq.mean_all()?;
        let grads = loss.backward()?;

        let grad_x = grads.get(&x).unwrap().to_f64_vec()?;
        // grad = 2*x / n = [2/3, 4/3, 6/3]
        assert_vec_approx(&grad_x, &[2.0 / 3.0, 4.0 / 3.0, 2.0], 1e-10);
        Ok(())
    }

    #[test]
    fn test_backward_relu() -> Result<()> {
        let dev = CpuDevice;
        let x = CpuTensor::from_f64_slice(&[-1.0, 0.5, 2.0], 3, DType::F64, &dev)?.set_variable();
        let y = x.relu()?;
        let loss = y.sum_all()?;
        let grads = loss.backward()?;

        let grad_x = grads.get(&x).unwrap().to_f64_vec()?;
        // relu'(-1) = 0, relu'(0.5) = 1, relu'(2) = 1
        assert_vec_approx(&grad_x, &[0.0, 1.0, 1.0], 1e-10);
        Ok(())
    }

    #[test]
    fn test_backward_exp_log() -> Result<()> {
        // f(x) = sum(log(exp(x))) = sum(x)  → grad = 1
        let dev = CpuDevice;
        let x = CpuTensor::from_f64_slice(&[1.0, 2.0], 2, DType::F64, &dev)?.set_variable();
        let e = x.exp()?;
        let l = e.log()?;
        let loss = l.sum_all()?;
        let grads = loss.backward()?;

        let grad_x = grads.get(&x).unwrap().to_f64_vec()?;
        assert_vec_approx(&grad_x, &[1.0, 1.0], 1e-6);
        Ok(())
    }

    #[test]
    fn test_backward_sigmoid() -> Result<()> {
        let dev = CpuDevice;
        let x = CpuTensor::from_f64_slice(&[0.0], (), DType::F64, &dev)?.set_variable();
        let s = x.sigmoid()?;
        let grads = s.backward()?;

        let grad_x = grads.get(&x).unwrap().to_scalar_f64()?;
        // σ'(0) = σ(0) * (1 - σ(0)) = 0.5 * 0.5 = 0.25
        assert!(approx_eq(grad_x, 0.25, 1e-10));
        Ok(())
    }

    #[test]
    fn test_backward_tanh() -> Result<()> {
        let dev = CpuDevice;
        let x = CpuTensor::from_f64_slice(&[0.0], (), DType::F64, &dev)?.set_variable();
        let t = x.tanh()?;
        let grads = t.backward()?;

        let grad_x = grads.get(&x).unwrap().to_scalar_f64()?;
        // tanh'(0) = 1 - tanh²(0) = 1 - 0 = 1
        assert!(approx_eq(grad_x, 1.0, 1e-10));
        Ok(())
    }

    #[test]
    fn test_backward_matmul() -> Result<()> {
        // C = A @ B, loss = sum(C)
        // grad_A = ones @ B^T, grad_B = A^T @ ones
        let dev = CpuDevice;
        let a = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0], (2, 2), DType::F64, &dev)?
            .set_variable();
        let b = CpuTensor::from_f64_slice(&[5.0, 6.0, 7.0, 8.0], (2, 2), DType::F64, &dev)?
            .set_variable();
        let c = a.matmul(&b)?;
        let loss = c.sum_all()?;
        let grads = loss.backward()?;

        // grad_A = ones(2,2) @ B^T
        // B^T = [[5,7],[6,8]]
        // grad_A = [[5+7, 6+8], [5+7, 6+8]] = [[12, 14], [12, 14]]
        // Wait: ones @ B^T = [[1,1],[1,1]] @ [[5,7],[6,8]] = [[11,15],[11,15]]
        let grad_a = grads.get(&a).unwrap().to_f64_vec()?;
        assert_vec_approx(&grad_a, &[11.0, 15.0, 11.0, 15.0], 1e-10);

        // grad_B = A^T @ ones(2,2)
        // A^T = [[1,3],[2,4]]
        // grad_B = [[1+3, 1+3],[2+4, 2+4]] = [[4,4],[6,6]]
        let grad_b = grads.get(&b).unwrap().to_f64_vec()?;
        assert_vec_approx(&grad_b, &[4.0, 4.0, 6.0, 6.0], 1e-10);
        Ok(())
    }

    #[test]
    fn test_backward_reshape() -> Result<()> {
        let dev = CpuDevice;
        let x = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0], (2, 2), DType::F64, &dev)?
            .set_variable();
        let y = x.reshape(4)?;
        let loss = y.sum_all()?;
        let grads = loss.backward()?;

        let grad_x = grads.get(&x).unwrap().to_f64_vec()?;
        assert_eq!(grad_x.len(), 4);
        assert_vec_approx(&grad_x, &[1.0, 1.0, 1.0, 1.0], 1e-10);
        Ok(())
    }

    #[test]
    fn test_backward_transpose() -> Result<()> {
        let dev = CpuDevice;
        let x =
            CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), DType::F64, &dev)?
                .set_variable();
        // y = x^T, loss = sum(y * constant)
        let yt = x.t()?;
        let c =
            CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2), DType::F64, &dev)?;
        let prod = yt.contiguous()?.mul(&c)?;
        let loss = prod.sum_all()?;
        let grads = loss.backward()?;

        let grad_x = grads.get(&x).unwrap().to_f64_vec()?;
        // The transposed grad should be c^T = [[1,3,5],[2,4,6]]
        assert_vec_approx(&grad_x, &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0], 1e-10);
        Ok(())
    }

    #[test]
    fn test_backward_affine() -> Result<()> {
        let dev = CpuDevice;
        let x = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0], 3, DType::F64, &dev)?.set_variable();
        let y = x.affine(3.0, 1.0)?; // y = 3x + 1
        let loss = y.sum_all()?;
        let grads = loss.backward()?;

        let grad_x = grads.get(&x).unwrap().to_f64_vec()?;
        // d(3x+1)/dx = 3
        assert_vec_approx(&grad_x, &[3.0, 3.0, 3.0], 1e-10);
        Ok(())
    }

    #[test]
    fn test_backward_chain_rule() -> Result<()> {
        // f(x) = (2x + 1)², loss = mean(f(x))
        // f'(x) = 2 * (2x + 1) * 2 = 4(2x+1)
        // grad_x_i = f'(x_i) / n
        let dev = CpuDevice;
        let x = CpuTensor::from_f64_slice(&[1.0, 2.0], 2, DType::F64, &dev)?.set_variable();
        let affined = x.affine(2.0, 1.0)?; // 2x + 1 = [3, 5]
        let sq = affined.square()?; // [9, 25]
        let loss = sq.mean_all()?; // 17
        let grads = loss.backward()?;

        let grad_x = grads.get(&x).unwrap().to_f64_vec()?;
        // grad = 4*(2x+1) / 2 = 2*(2x+1)
        // x=1: 2*3 = 6, x=2: 2*5 = 10
        assert_vec_approx(&grad_x, &[6.0, 10.0], 1e-10);
        Ok(())
    }

    #[test]
    fn test_backward_sum_dim() -> Result<()> {
        let dev = CpuDevice;
        // [[1, 2, 3],
        //  [4, 5, 6]]
        let x =
            CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), DType::F64, &dev)?
                .set_variable();
        let s = x.sum(1, false)?; // [6, 15]
        let loss = s.sum_all()?; // 21
        let grads = loss.backward()?;

        let grad_x = grads.get(&x).unwrap().to_f64_vec()?;
        // Sum along dim=1 broadcasts gradient back: all 1s
        assert_vec_approx(&grad_x, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 1e-10);
        Ok(())
    }

    #[test]
    fn test_backward_narrow() -> Result<()> {
        let dev = CpuDevice;
        let x =
            CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0], 4, DType::F64, &dev)?.set_variable();
        let y = x.narrow(0, 1, 2)?; // [2, 3]
        let loss = y.sum_all()?; // 5
        let grads = loss.backward()?;

        let grad_x = grads.get(&x).unwrap().to_f64_vec()?;
        // Gradient only flows to elements [1] and [2]
        assert_vec_approx(&grad_x, &[0.0, 1.0, 1.0, 0.0], 1e-10);
        Ok(())
    }

    #[test]
    fn test_backward_numerical_check() -> Result<()> {
        // Compare analytical gradients with numerical finite differences
        let dev = CpuDevice;
        let x = CpuTensor::from_f64_slice(&[1.0, -0.5, 2.0], 3, DType::F64, &dev)?.set_variable();

        // f(x) = sum(x² * sigmoid(x))
        let f = |t: &CpuTensor| -> Result<CpuTensor> {
            let sq = t.square()?;
            let sig = t.sigmoid()?;
            let prod = sq.mul(&sig)?;
            prod.sum_all()
        };

        let loss = f(&x)?;
        let grads = loss.backward()?;
        let analytical = grads.get(&x).unwrap().to_f64_vec()?;
        let numerical = numerical_grad(f, &x, 1e-5)?;

        assert_vec_approx(&analytical, &numerical, 1e-4);
        Ok(())
    }

    #[test]
    fn test_backward_shared_variable() -> Result<()> {
        // f(a) = sum(a * a) = sum(a²) → grad_a = 2a
        // This tests that a tensor used twice accumulates gradients correctly.
        let dev = CpuDevice;
        let a = CpuTensor::from_f64_slice(&[3.0, 4.0], 2, DType::F64, &dev)?.set_variable();
        let c = a.mul(&a)?;
        let loss = c.sum_all()?;
        let grads = loss.backward()?;

        let grad_a = grads.get(&a).unwrap().to_f64_vec()?;
        assert_vec_approx(&grad_a, &[6.0, 8.0], 1e-10);
        Ok(())
    }

    #[test]
    fn test_backward_detach() -> Result<()> {
        // Detach should stop gradient flow
        let dev = CpuDevice;
        let x = CpuTensor::from_f64_slice(&[2.0], (), DType::F64, &dev)?.set_variable();
        let y = x.mul(&x)?; // y = x² = 4
        let z = y.detach(); // z = 4, but disconnected from graph
        let w = z.mul(&x)?; // w = 4 * x = 8 (only x has grad here)
        let grads = w.backward()?;

        let grad_x = grads.get(&x).unwrap().to_scalar_f64()?;
        // w = z * x, dw/dx = z = 4 (z is treated as a constant)
        // Without detach, it would be d(x²*x)/dx = 3x² = 12
        assert!(approx_eq(grad_x, 4.0, 1e-10));
        Ok(())
    }

    // =================================================================
    // Tests for new tensor ops: powf, clamp, where_cond, stack, arange, triu/tril
    // =================================================================

    #[test]
    fn test_powf() -> Result<()> {
        let dev = CpuDevice;
        let a = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0], (2, 2), DType::F64, &dev)?;
        let r = a.powf(2.0)?;
        assert_vec_approx(&r.to_f64_vec()?, &[1.0, 4.0, 9.0, 16.0], 1e-10);
        Ok(())
    }

    #[test]
    fn test_powf_sqrt() -> Result<()> {
        let dev = CpuDevice;
        let a = CpuTensor::from_f64_slice(&[4.0, 9.0, 16.0, 25.0], 4, DType::F64, &dev)?;
        let r = a.powf(0.5)?;
        assert_vec_approx(&r.to_f64_vec()?, &[2.0, 3.0, 4.0, 5.0], 1e-10);
        Ok(())
    }

    #[test]
    fn test_powf_backward() -> Result<()> {
        let dev = CpuDevice;
        // f(x) = x^3, f'(x) = 3x^2
        let x = CpuTensor::from_f64_slice(&[2.0, 3.0], 2, DType::F64, &dev)?.set_variable();
        let y = x.powf(3.0)?;
        let loss = y.sum_all()?;
        let grads = loss.backward()?;
        let grad_x = grads.get(&x).unwrap().to_f64_vec()?;
        // 3*2^2 = 12, 3*3^2 = 27
        assert_vec_approx(&grad_x, &[12.0, 27.0], 1e-10);
        Ok(())
    }

    #[test]
    fn test_clamp() -> Result<()> {
        let dev = CpuDevice;
        let a = CpuTensor::from_f64_slice(&[-5.0, -1.0, 0.0, 1.0, 5.0, 10.0], 6, DType::F64, &dev)?;
        let r = a.clamp(-2.0, 3.0)?;
        assert_vec_approx(&r.to_f64_vec()?, &[-2.0, -1.0, 0.0, 1.0, 3.0, 3.0], 1e-10);
        Ok(())
    }

    #[test]
    fn test_clamp_backward() -> Result<()> {
        let dev = CpuDevice;
        // Gradient should be 1 where input is in (min, max), 0 at boundaries
        let x = CpuTensor::from_f64_slice(&[-5.0, 0.0, 5.0], 3, DType::F64, &dev)?.set_variable();
        let y = x.clamp(-2.0, 3.0)?;
        let loss = y.sum_all()?;
        let grads = loss.backward()?;
        let grad_x = grads.get(&x).unwrap().to_f64_vec()?;
        // -5 is clamped (grad=0), 0 is in range (grad=1), 5 is clamped (grad=0)
        assert_vec_approx(&grad_x, &[0.0, 1.0, 0.0], 1e-10);
        Ok(())
    }

    #[test]
    fn test_where_cond() -> Result<()> {
        let dev = CpuDevice;
        let mask = CpuTensor::from_f64_slice(&[1.0, 0.0, 1.0, 0.0], 4, DType::F64, &dev)?;
        let on_true = CpuTensor::from_f64_slice(&[10.0, 20.0, 30.0, 40.0], 4, DType::F64, &dev)?;
        let on_false =
            CpuTensor::from_f64_slice(&[100.0, 200.0, 300.0, 400.0], 4, DType::F64, &dev)?;
        let r = CpuTensor::where_cond(&mask, &on_true, &on_false)?;
        assert_vec_approx(&r.to_f64_vec()?, &[10.0, 200.0, 30.0, 400.0], 1e-10);
        Ok(())
    }

    #[test]
    fn test_where_cond_with_cmp() -> Result<()> {
        let dev = CpuDevice;
        let x = CpuTensor::from_f64_slice(&[-1.0, 2.0, -3.0, 4.0], 4, DType::F64, &dev)?;
        let zero = CpuTensor::zeros(4, DType::F64, &dev)?;
        let mask = x.gt(&zero)?; // [0, 1, 0, 1]
        let pos = CpuTensor::from_f64_slice(&[1.0, 1.0, 1.0, 1.0], 4, DType::F64, &dev)?;
        let neg = CpuTensor::from_f64_slice(&[-1.0, -1.0, -1.0, -1.0], 4, DType::F64, &dev)?;
        let r = CpuTensor::where_cond(&mask, &pos, &neg)?;
        assert_vec_approx(&r.to_f64_vec()?, &[-1.0, 1.0, -1.0, 1.0], 1e-10);
        Ok(())
    }

    #[test]
    fn test_where_cond_backward() -> Result<()> {
        let dev = CpuDevice;
        let mask = CpuTensor::from_f64_slice(&[1.0, 0.0, 1.0], 3, DType::F64, &dev)?;
        let a = CpuTensor::from_f64_slice(&[2.0, 3.0, 4.0], 3, DType::F64, &dev)?.set_variable();
        let b = CpuTensor::from_f64_slice(&[5.0, 6.0, 7.0], 3, DType::F64, &dev)?.set_variable();
        let r = CpuTensor::where_cond(&mask, &a, &b)?;
        let loss = r.sum_all()?;
        let grads = loss.backward()?;
        let grad_a = grads.get(&a).unwrap().to_f64_vec()?;
        let grad_b = grads.get(&b).unwrap().to_f64_vec()?;
        // mask=[1,0,1]: grad_a gets [1,0,1], grad_b gets [0,1,0]
        assert_vec_approx(&grad_a, &[1.0, 0.0, 1.0], 1e-10);
        assert_vec_approx(&grad_b, &[0.0, 1.0, 0.0], 1e-10);
        Ok(())
    }

    #[test]
    fn test_stack() -> Result<()> {
        let dev = CpuDevice;
        let a = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0], 3, DType::F64, &dev)?;
        let b = CpuTensor::from_f64_slice(&[4.0, 5.0, 6.0], 3, DType::F64, &dev)?;
        // stack along dim 0: [2, 3]
        let s = CpuTensor::stack(&[a.clone(), b.clone()], 0)?;
        assert_eq!(s.dims(), &[2, 3]);
        assert_vec_approx(&s.to_f64_vec()?, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1e-10);
        // stack along dim 1: [3, 2]
        let s1 = CpuTensor::stack(&[a, b], 1)?;
        assert_eq!(s1.dims(), &[3, 2]);
        assert_vec_approx(&s1.to_f64_vec()?, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 1e-10);
        Ok(())
    }

    #[test]
    fn test_arange() -> Result<()> {
        let dev = CpuDevice;
        let t = CpuTensor::arange(5, DType::F64, &dev)?;
        assert_eq!(t.dims(), &[5]);
        assert_vec_approx(&t.to_f64_vec()?, &[0.0, 1.0, 2.0, 3.0, 4.0], 1e-10);
        Ok(())
    }

    #[test]
    fn test_arange_step() -> Result<()> {
        let dev = CpuDevice;
        let t = CpuTensor::arange_step(0.0, 1.0, 0.25, DType::F64, &dev)?;
        assert_eq!(t.dims(), &[4]);
        assert_vec_approx(&t.to_f64_vec()?, &[0.0, 0.25, 0.5, 0.75], 1e-10);
        Ok(())
    }

    #[test]
    fn test_triu() -> Result<()> {
        let dev = CpuDevice;
        let t = CpuTensor::triu(3, 3, 0, DType::F64, &dev)?;
        assert_eq!(t.dims(), &[3, 3]);
        assert_vec_approx(
            &t.to_f64_vec()?,
            &[1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            1e-10,
        );
        Ok(())
    }

    #[test]
    fn test_triu_diagonal() -> Result<()> {
        let dev = CpuDevice;
        // diagonal=1: above main diagonal only
        let t = CpuTensor::triu(3, 3, 1, DType::F64, &dev)?;
        assert_vec_approx(
            &t.to_f64_vec()?,
            &[0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            1e-10,
        );
        Ok(())
    }

    #[test]
    fn test_tril() -> Result<()> {
        let dev = CpuDevice;
        let t = CpuTensor::tril(3, 3, 0, DType::F64, &dev)?;
        assert_eq!(t.dims(), &[3, 3]);
        assert_vec_approx(
            &t.to_f64_vec()?,
            &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
            1e-10,
        );
        Ok(())
    }

    #[test]
    fn test_tril_negative_diagonal() -> Result<()> {
        let dev = CpuDevice;
        // diagonal=-1: below main diagonal only
        let t = CpuTensor::tril(3, 3, -1, DType::F64, &dev)?;
        assert_vec_approx(
            &t.to_f64_vec()?,
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            1e-10,
        );
        Ok(())
    }
}
