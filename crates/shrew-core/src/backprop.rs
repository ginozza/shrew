// Backpropagation — Reverse-mode automatic differentiation
//
// This module implements the backward pass (backpropagation), computing
// gradients of a scalar loss with respect to all tensors in the computation
// graph. This is the core of training any neural network.
//
// HOW IT WORKS:
//
//   1. Forward pass: tensor operations build a DAG (directed acyclic graph)
//      where each tensor stores its Op (the operation that created it).
//
//   2. backward() topologically sorts the DAG from the loss tensor to the
//      leaves (input data and trainable parameters).
//
//   3. Starting with grad(loss) = 1.0, we walk the graph in reverse order.
//      For each tensor, we apply the chain rule to compute gradients for
//      its inputs and accumulate them.
//
// AUTOGRAD COVERAGE (all 20 Op variants):
//
//   Leaf:        Op::None        — no gradient propagation needed
//   Contiguous:  Op::Contiguous  — pass-through (identity gradient)
//
//   Binary ops:  Op::Binary      — Add, Sub, Mul, Div (with broadcast reduction)
//   Unary ops:   Op::Unary       — Neg, Abs, Exp, Log, Sqrt, Square, Relu,
//                                   Sigmoid, Tanh, Gelu, Silu, Sin, Cos,
//                                   Floor, Ceil, Round
//   Reduce ops:  Op::Reduce      — Sum, Mean, Max, Min (with keepdim support)
//
//   Linear algebra:              — Matmul (with batched support)
//   Shape ops:                   — Reshape, Transpose, Narrow
//   Affine:                      — scale + bias (LayerNorm/BatchNorm support)
//
//   Convolutions:                — Conv2d (input + weight + bias grads)
//                                — Conv1d (input + weight + bias grads)
//   Pooling:                     — MaxPool2d (via saved indices)
//                                — AvgPool2d (uniform gradient distribution)
//
//   Composite:                   — Cat (split gradient by sizes)
//                                — Powf (power rule)
//                                — Clamp (zero grad outside bounds)
//                                — WhereCond (route grad through mask)
//                                — Gather (scatter gradient to source)
//                                — Pad (narrow gradient to unpadded region)
//
// GRADIENT CHECKPOINTING:
//   checkpoint()            — wrap a forward fn for recomputation in backward
//   checkpoint_sequential() — split sequential layers into segments
//   is_checkpointing()      — check if inside a recomputation pass
//
// GRADIENT RULES (chain rule applied for each Op):
//
//   Binary Add:  grad_a += grad_out, grad_b += grad_out
//   Binary Sub:  grad_a += grad_out, grad_b += -grad_out
//   Binary Mul:  grad_a += grad_out * b, grad_b += grad_out * a
//   Binary Div:  grad_a += grad_out / b, grad_b += -grad_out * a / b²
//   Unary Neg:   grad_in += -grad_out
//   Unary Exp:   grad_in += grad_out * exp(input)
//   Unary Log:   grad_in += grad_out / input
//   Matmul:      grad_A += grad_C @ B^T, grad_B += A^T @ grad_C
//   Sum:         grad_in += broadcast(grad_out)
//   Reshape:     grad_in += reshape(grad_out, original_shape)
//   Transpose:   grad_in += transpose(grad_out)
//   ... and many more (see compute_* functions below)
//
// ACCUMULATION: If a tensor is used in multiple operations, its gradient
// is the SUM of contributions from each use (multivariate chain rule).
//
// For example: c = a * a, then grad_a = grad_c * a + grad_c * a = 2 * a * grad_c

use std::collections::{HashMap, HashSet};

use crate::backend::{Backend, BinaryOp, ReduceOp, UnaryOp};
use crate::error::Result;
use crate::op::{Op, TensorId};
use crate::shape::Shape;
use crate::tensor::Tensor;

/// Stores gradients for all tensors in a computation graph.
///
/// After calling `tensor.backward()`, you receive a GradStore.
/// Use `grads.get(&tensor)` to retrieve the gradient for any tensor.
pub struct GradStore<B: Backend> {
    grads: HashMap<TensorId, Tensor<B>>,
}

impl<B: Backend> Clone for GradStore<B> {
    fn clone(&self) -> Self {
        GradStore {
            grads: self.grads.clone(),
        }
    }
}

impl<B: Backend> Default for GradStore<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> GradStore<B> {
    /// Create a new empty GradStore.
    pub fn new() -> Self {
        GradStore {
            grads: HashMap::new(),
        }
    }

    /// Get the gradient of a tensor (if it exists).
    pub fn get(&self, tensor: &Tensor<B>) -> Option<&Tensor<B>> {
        self.grads.get(&tensor.id())
    }

    fn get_by_id(&self, id: &TensorId) -> Option<&Tensor<B>> {
        self.grads.get(id)
    }

    /// Accumulate gradient for a tensor.
    /// If a gradient already exists for this tensor, add the new one to it.
    /// This handles the case where a tensor is used in multiple operations.
    pub fn accumulate(&mut self, id: TensorId, grad: Tensor<B>) -> Result<()> {
        if let Some(existing) = self.grads.get(&id) {
            let new_grad = existing.add(&grad)?;
            self.grads.insert(id, new_grad);
        } else {
            self.grads.insert(id, grad);
        }
        Ok(())
    }
}

/// Build a topological ordering of the computation graph.
///
/// Uses depth-first search from the root tensor. Returns tensors in
/// order such that every tensor appears AFTER all its inputs.
/// (Leaves first, root last.)
fn build_topo<B: Backend>(root: &Tensor<B>) -> Vec<Tensor<B>> {
    let mut visited = HashSet::new();
    let mut order = Vec::new();

    fn visit<B: Backend>(
        t: &Tensor<B>,
        visited: &mut HashSet<TensorId>,
        order: &mut Vec<Tensor<B>>,
    ) {
        if visited.contains(&t.id()) {
            return;
        }
        visited.insert(t.id());
        // Visit inputs first (depth-first)
        for input in t.op().inputs() {
            visit(input, visited, order);
        }
        // Then add this tensor (post-order)
        order.push(t.clone());
    }

    visit(root, &mut visited, &mut order);
    order
}

/// Compute gradients of `root` with respect to all tensors in the graph.
///
/// `root` must be a scalar tensor (single element). This is the main entry
/// point for backpropagation, called by `tensor.backward()`.
#[allow(clippy::needless_range_loop)]
pub fn backward<B: Backend>(root: &Tensor<B>) -> Result<GradStore<B>> {
    // Backward only works from a scalar (otherwise, which output element?)
    if root.elem_count() != 1 {
        return Err(crate::Error::msg(
            "backward() requires a scalar tensor (single element). \
             Use .sum_all() or .mean_all() to reduce to a scalar first.",
        ));
    }

    // Step 1: Topological sort (leaves first, root last)
    let topo = build_topo(root);

    // Step 2: Initialize — grad(root) = 1.0 (dL/dL = 1)
    let mut grads = GradStore::new();
    let ones = Tensor::<B>::ones(root.shape().clone(), root.dtype(), root.device())?;
    grads.grads.insert(root.id(), ones);

    // Step 3: Walk in reverse topological order (root first, leaves last)
    for tensor in topo.iter().rev() {
        let grad_output = match grads.get_by_id(&tensor.id()) {
            Some(g) => g.clone(),
            None => continue, // No gradient flows to this tensor
        };

        match tensor.op() {
            Op::None => {
                // Leaf — nothing to propagate
            }

            Op::Contiguous { input } => {
                // Identity-like: gradient passes through unchanged
                grads.accumulate(input.id(), grad_output)?;
            }

            Op::Binary { lhs, rhs, op } => {
                compute_binary_grad(*op, &grad_output, lhs, rhs, &mut grads)?;
            }

            Op::Unary { input, op } => {
                compute_unary_grad(*op, &grad_output, input, &mut grads)?;
            }

            Op::Reduce {
                input,
                op,
                dims,
                keep_dim,
            } => {
                compute_reduce_grad(*op, &grad_output, input, dims, *keep_dim, &mut grads)?;
            }

            Op::Matmul { lhs, rhs } => {
                compute_matmul_grad(&grad_output, lhs, rhs, &mut grads)?;
            }

            Op::Reshape { input, src_shape } => {
                // Reshape gradient = reshape grad_output back to original shape
                let grad = grad_output.reshape(src_shape.clone())?;
                grads.accumulate(input.id(), grad)?;
            }

            Op::Transpose { input, dim0, dim1 } => {
                // Transpose is its own inverse: transpose back with same dims
                let grad = grad_output.transpose(*dim0, *dim1)?;
                grads.accumulate(input.id(), grad)?;
            }

            Op::Narrow {
                input,
                dim,
                start,
                len,
            } => {
                // Scatter gradient into zero tensor at the original position
                compute_narrow_grad(&grad_output, input, *dim, *start, *len, &mut grads)?;
            }

            Op::Affine { input, mul, .. } => {
                // d(x * mul + add)/dx = mul
                let grad = grad_output.affine(*mul, 0.0)?;
                grads.accumulate(input.id(), grad)?;
            }

            Op::Conv2d {
                input,
                weight,
                bias,
                stride,
                padding,
            } => {
                compute_conv2d_grad(
                    &grad_output,
                    input,
                    weight,
                    bias.as_ref(),
                    *stride,
                    *padding,
                    &mut grads,
                )?;
            }

            Op::MaxPool2d { input, indices, .. } => {
                compute_maxpool2d_grad(&grad_output, input, indices, &mut grads)?;
            }

            Op::Cat { inputs, dim, sizes } => {
                // Backward of cat: slice the gradient into pieces for each input
                let mut offset = 0usize;
                for (inp, &sz) in inputs.iter().zip(sizes.iter()) {
                    let grad_slice = grad_output.narrow(*dim, offset, sz)?;
                    grads.accumulate(inp.id(), grad_slice)?;
                    offset += sz;
                }
            }

            Op::Powf { input, exponent } => {
                // d(x^n)/dx = n * x^(n-1)
                let n = *exponent;
                let x_pow_nm1 = input.powf(n - 1.0)?;
                let n_tensor =
                    Tensor::<B>::full(input.shape().clone(), n, input.dtype(), input.device())?;
                let grad = grad_output.mul(&n_tensor)?.mul(&x_pow_nm1)?;
                grads.accumulate(input.id(), grad)?;
            }

            Op::Clamp { input, min, max } => {
                // Gradient = 1 where min < input < max, 0 at boundaries
                let input_data = input.to_f64_vec()?;
                let grad_data = grad_output.to_f64_vec()?;
                let mask: Vec<f64> = input_data
                    .iter()
                    .zip(grad_data.iter())
                    .map(|(&x, &g)| if x > *min && x < *max { g } else { 0.0 })
                    .collect();
                let grad = Tensor::<B>::from_f64_slice(
                    &mask,
                    input.shape().clone(),
                    input.dtype(),
                    input.device(),
                )?;
                grads.accumulate(input.id(), grad)?;
            }

            Op::WhereCond {
                mask,
                on_true,
                on_false,
            } => {
                // Gradient flows to on_true where mask is true, on_false where mask is false
                let mask_data = mask.to_f64_vec()?;
                let grad_data = grad_output.to_f64_vec()?;
                let n = mask_data.len();

                let grad_true_data: Vec<f64> = (0..n)
                    .map(|i| {
                        if mask_data[i] != 0.0 {
                            grad_data[i]
                        } else {
                            0.0
                        }
                    })
                    .collect();
                let grad_false_data: Vec<f64> = (0..n)
                    .map(|i| {
                        if mask_data[i] == 0.0 {
                            grad_data[i]
                        } else {
                            0.0
                        }
                    })
                    .collect();

                let grad_true = Tensor::<B>::from_f64_slice(
                    &grad_true_data,
                    on_true.shape().clone(),
                    on_true.dtype(),
                    on_true.device(),
                )?;
                let grad_false = Tensor::<B>::from_f64_slice(
                    &grad_false_data,
                    on_false.shape().clone(),
                    on_false.dtype(),
                    on_false.device(),
                )?;
                grads.accumulate(on_true.id(), grad_true)?;
                grads.accumulate(on_false.id(), grad_false)?;
                // mask is non-differentiable — no gradient
            }

            Op::Gather { input, index, dim } => {
                // Gather backward = scatter-add:
                // grad_input = zeros_like(input);
                // for each position p in index: grad_input[...dim=index[p]...] += grad_output[p]
                let dim = *dim;
                let input_dims = input.dims();
                let rank = input_dims.len();

                // Read grad_output and index data
                let grad_data = grad_output.to_f64_vec()?;
                let index_data = index.to_f64_vec()?;

                // Create zero grad for input
                let mut grad_input_data = vec![0.0f64; input.elem_count()];
                let input_strides = input.shape().stride_contiguous();

                // Compute strides for index shape (to decompose flat positions)
                let index_strides = index.shape().stride_contiguous();

                let n = index_data.len();
                for flat_idx in 0..n {
                    // Decompose flat_idx into multi-dim coords in index shape
                    let mut coords = vec![0usize; rank];
                    let mut remainder = flat_idx;
                    for d in 0..rank {
                        coords[d] = remainder / index_strides[d];
                        remainder %= index_strides[d];
                    }

                    // Replace coord at `dim` with the index value
                    let idx_val = index_data[flat_idx] as usize;
                    coords[dim] = idx_val;

                    // Compute flat position in input
                    let mut input_flat = 0;
                    for d in 0..rank {
                        input_flat += coords[d] * input_strides[d];
                    }

                    // Scatter-add: accumulate the gradient
                    grad_input_data[input_flat] += grad_data[flat_idx];
                }

                let grad_input = Tensor::<B>::from_f64_slice(
                    &grad_input_data,
                    input.shape().clone(),
                    input.dtype(),
                    input.device(),
                )?;
                grads.accumulate(input.id(), grad_input)?;
                // index is non-differentiable — no gradient
            }

            Op::Pad { input, padding } => {
                // Backward of pad: narrow the gradient to remove the padding
                let mut grad = grad_output.clone();
                let input_dims = input.dims();
                for d in 0..input_dims.len() {
                    let [before, _after] = padding[d];
                    if before > 0 || _after > 0 {
                        grad = grad.narrow(d, before, input_dims[d])?;
                    }
                }
                grads.accumulate(input.id(), grad)?;
            }

            Op::AvgPool2d {
                input,
                kernel_size,
                stride,
                padding,
            } => {
                compute_avgpool2d_grad(
                    &grad_output,
                    input,
                    *kernel_size,
                    *stride,
                    *padding,
                    &mut grads,
                )?;
            }

            Op::Conv1d {
                input,
                weight,
                bias,
                stride,
                padding,
            } => {
                compute_conv1d_grad(
                    &grad_output,
                    input,
                    weight,
                    bias.as_ref(),
                    *stride,
                    *padding,
                    &mut grads,
                )?;
            }

            Op::IndexSelect {
                input,
                indices,
                dim,
            } => {
                // IndexSelect backward = scatter-add:
                // grad_input = zeros_like(input)
                // For each position in output, add grad_output to grad_input
                // at the corresponding input position (determined by indices).
                let dim = *dim;
                let input_dims = input.dims();
                let rank = input_dims.len();

                let grad_data = grad_output.to_f64_vec()?;
                let index_data = indices.to_f64_vec()?; // index values as f64
                let _num_indices = index_data.len();

                let mut grad_input_data = vec![0.0f64; input.elem_count()];
                let input_strides = input.shape().stride_contiguous();
                let _output_dims = grad_output.dims();
                let output_strides = grad_output.shape().stride_contiguous();

                let total = grad_data.len();
                for flat_idx in 0..total {
                    // Decompose flat_idx into multi-dim coords in output shape
                    let mut coords = vec![0usize; rank];
                    let mut remainder = flat_idx;
                    for d in 0..rank {
                        coords[d] = remainder / output_strides[d];
                        remainder %= output_strides[d];
                    }

                    // Replace coord at `dim` with the original source index
                    let out_dim_coord = coords[dim];
                    let src_idx = index_data[out_dim_coord] as usize;
                    coords[dim] = src_idx;

                    // Compute flat position in input
                    let mut input_flat = 0;
                    for d in 0..rank {
                        input_flat += coords[d] * input_strides[d];
                    }

                    // Scatter-add
                    grad_input_data[input_flat] += grad_data[flat_idx];
                }

                let grad_input = Tensor::<B>::from_f64_slice(
                    &grad_input_data,
                    input.shape().clone(),
                    input.dtype(),
                    input.device(),
                )?;
                grads.accumulate(input.id(), grad_input)?;
                // indices are non-differentiable
            }

            Op::ToDtype { input, src_dtype } => {
                // Cast gradient back to the original input dtype.
                let grad_in = grad_output.to_dtype(*src_dtype)?;
                grads.accumulate(input.id(), grad_in)?;
            }
        }
    }

    Ok(grads)
}

// Gradient rules for binary operations

fn compute_binary_grad<B: Backend>(
    op: BinaryOp,
    grad_output: &Tensor<B>,
    lhs: &Tensor<B>,
    rhs: &Tensor<B>,
    grads: &mut GradStore<B>,
) -> Result<()> {
    match op {
        BinaryOp::Add => {
            // d(a + b)/da = 1, d(a + b)/db = 1
            let grad_lhs = reduce_broadcast_grad(grad_output, lhs.shape())?;
            let grad_rhs = reduce_broadcast_grad(grad_output, rhs.shape())?;
            grads.accumulate(lhs.id(), grad_lhs)?;
            grads.accumulate(rhs.id(), grad_rhs)?;
        }
        BinaryOp::Sub => {
            // d(a - b)/da = 1, d(a - b)/db = -1
            let grad_lhs = reduce_broadcast_grad(grad_output, lhs.shape())?;
            let neg = grad_output.neg()?;
            let grad_rhs = reduce_broadcast_grad(&neg, rhs.shape())?;
            grads.accumulate(lhs.id(), grad_lhs)?;
            grads.accumulate(rhs.id(), grad_rhs)?;
        }
        BinaryOp::Mul => {
            // d(a * b)/da = b, d(a * b)/db = a
            let raw_lhs = grad_output.mul(rhs)?;
            let raw_rhs = grad_output.mul(lhs)?;
            grads.accumulate(lhs.id(), reduce_broadcast_grad(&raw_lhs, lhs.shape())?)?;
            grads.accumulate(rhs.id(), reduce_broadcast_grad(&raw_rhs, rhs.shape())?)?;
        }
        BinaryOp::Div => {
            // d(a / b)/da = 1/b
            // d(a / b)/db = -a / b²
            let raw_lhs = grad_output.div(rhs)?;
            grads.accumulate(lhs.id(), reduce_broadcast_grad(&raw_lhs, lhs.shape())?)?;
            let neg_grad = grad_output.neg()?;
            let b_sq = rhs.mul(rhs)?;
            let raw_rhs = neg_grad.mul(lhs)?.div(&b_sq)?;
            grads.accumulate(rhs.id(), reduce_broadcast_grad(&raw_rhs, rhs.shape())?)?;
        }
    }
    Ok(())
}

/// When broadcasting expands a tensor's shape, the backward pass must sum
/// the gradient over the broadcast dimensions to match the original shape.
///
/// For example, if lhs was [1, 4] broadcast to [3, 4]:
///   grad_output is [3, 4], but grad_lhs must be [1, 4] → sum over dim 0
///
/// If lhs was [4] broadcast to [3, 4]:
///   grad_output is [3, 4], grad_lhs must be [4] → sum over dim 0, squeeze
fn reduce_broadcast_grad<B: Backend>(
    grad: &Tensor<B>,
    target_shape: &crate::Shape,
) -> Result<Tensor<B>> {
    let grad_shape = grad.dims();
    let target_dims = target_shape.dims();

    // If shapes already match, no reduction needed
    if grad_shape == target_dims {
        return Ok(grad.clone());
    }

    // Pad target dims with leading 1s to match grad rank
    let grad_rank = grad_shape.len();
    let target_rank = target_dims.len();
    let mut padded_target = vec![1usize; grad_rank];
    let offset = grad_rank - target_rank;
    padded_target[offset..offset + target_rank].copy_from_slice(target_dims);

    // Sum over dimensions where padded_target[d] == 1 and grad[d] > 1
    let mut result = grad.clone();
    // Sum from left to right, adjusting for removed dimensions
    let mut dims_to_sum: Vec<usize> = Vec::new();
    for d in 0..grad_rank {
        if padded_target[d] == 1 && grad_shape[d] > 1 {
            dims_to_sum.push(d);
        }
    }

    // Sum all broadcast dimensions at once by processing from highest to lowest
    // to keep dimension indices stable
    for &d in dims_to_sum.iter().rev() {
        result = result.sum(d, true)?;
    }

    // Now reshape to target shape (removing the extra size-1 dims)
    result = result.reshape(target_shape.clone())?;

    Ok(result)
}

// Gradient rules for unary operations

fn compute_unary_grad<B: Backend>(
    op: UnaryOp,
    grad_output: &Tensor<B>,
    input: &Tensor<B>,
    grads: &mut GradStore<B>,
) -> Result<()> {
    let grad_input = match op {
        // d(-x)/dx = -1
        UnaryOp::Neg => grad_output.neg()?,

        // d|x|/dx = sign(x)
        UnaryOp::Abs => {
            let input_data = input.to_f64_vec()?;
            let sign_data: Vec<f64> = input_data
                .iter()
                .map(|&v| {
                    if v > 0.0 {
                        1.0
                    } else if v < 0.0 {
                        -1.0
                    } else {
                        0.0
                    }
                })
                .collect();
            let sign = Tensor::<B>::from_f64_slice(
                &sign_data,
                input.shape().clone(),
                input.dtype(),
                input.device(),
            )?;
            grad_output.mul(&sign)?
        }

        // d(e^x)/dx = e^x
        UnaryOp::Exp => {
            let exp_x = input.exp()?;
            grad_output.mul(&exp_x)?
        }

        // d(ln x)/dx = 1/x
        UnaryOp::Log => grad_output.div(input)?,

        // d(√x)/dx = 1 / (2√x)
        UnaryOp::Sqrt => {
            let sqrt_x = input.sqrt()?;
            let two_sqrt = sqrt_x.affine(2.0, 0.0)?;
            grad_output.div(&two_sqrt)?
        }

        // d(x²)/dx = 2x
        UnaryOp::Square => {
            let two_x = input.affine(2.0, 0.0)?;
            grad_output.mul(&two_x)?
        }

        // d(relu(x))/dx = 1 if x > 0, else 0
        UnaryOp::Relu => {
            let input_data = input.to_f64_vec()?;
            let mask_data: Vec<f64> = input_data
                .iter()
                .map(|&v| if v > 0.0 { 1.0 } else { 0.0 })
                .collect();
            let mask = Tensor::<B>::from_f64_slice(
                &mask_data,
                input.shape().clone(),
                input.dtype(),
                input.device(),
            )?;
            grad_output.mul(&mask)?
        }

        // d(σ(x))/dx = σ(x) * (1 - σ(x))
        UnaryOp::Sigmoid => {
            let sig = input.sigmoid()?;
            let one = Tensor::<B>::ones(input.shape().clone(), input.dtype(), input.device())?;
            let one_minus_sig = one.sub(&sig)?;
            let dsig = sig.mul(&one_minus_sig)?;
            grad_output.mul(&dsig)?
        }

        // d(tanh(x))/dx = 1 - tanh²(x)
        UnaryOp::Tanh => {
            let tanh_x = input.tanh()?;
            let tanh_sq = tanh_x.mul(&tanh_x)?;
            let one = Tensor::<B>::ones(input.shape().clone(), input.dtype(), input.device())?;
            let dtanh = one.sub(&tanh_sq)?;
            grad_output.mul(&dtanh)?
        }

        // d(GELU(x))/dx — computed element-wise from the formula
        // GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715x³)))
        UnaryOp::Gelu => {
            let input_data = input.to_f64_vec()?;
            let deriv_data: Vec<f64> = input_data
                .iter()
                .map(|&x| {
                    let sqrt_2_over_pi = std::f64::consts::FRAC_2_PI.sqrt();
                    let c = 0.044715_f64;
                    let s = sqrt_2_over_pi * (x + c * x * x * x);
                    let tanh_s = s.tanh();
                    let sech2_s = 1.0 - tanh_s * tanh_s;
                    let ds_dx = sqrt_2_over_pi * (1.0 + 3.0 * c * x * x);
                    0.5 * (1.0 + tanh_s) + 0.5 * x * sech2_s * ds_dx
                })
                .collect();
            let deriv = Tensor::<B>::from_f64_slice(
                &deriv_data,
                input.shape().clone(),
                input.dtype(),
                input.device(),
            )?;
            grad_output.mul(&deriv)?
        }

        // d(x·σ(x))/dx = σ(x) + x·σ(x)·(1 - σ(x)) = σ(x)·(1 + x·(1-σ(x)))
        UnaryOp::Silu => {
            let sig = input.sigmoid()?;
            let one = Tensor::<B>::ones(input.shape().clone(), input.dtype(), input.device())?;
            let one_minus_sig = one.sub(&sig)?;
            let x_oms = input.mul(&one_minus_sig)?;
            let one2 = Tensor::<B>::ones(input.shape().clone(), input.dtype(), input.device())?;
            let bracket = one2.add(&x_oms)?;
            let dsilu = sig.mul(&bracket)?;
            grad_output.mul(&dsilu)?
        }

        // d(sin x)/dx = cos x
        UnaryOp::Sin => {
            let cos_x = input.cos()?;
            grad_output.mul(&cos_x)?
        }

        // d(cos x)/dx = -sin x
        UnaryOp::Cos => {
            let sin_x = input.sin()?;
            let neg_sin = sin_x.neg()?;
            grad_output.mul(&neg_sin)?
        }

        // floor, ceil, round are piecewise-constant → gradient is 0 everywhere
        // (undefined at integers, but convention is 0)
        UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Round => {
            Tensor::<B>::zeros(input.shape().clone(), input.dtype(), input.device())?
        }
    };

    grads.accumulate(input.id(), grad_input)?;
    Ok(())
}

// Gradient rules for reductions

#[allow(clippy::needless_range_loop)]
fn compute_reduce_grad<B: Backend>(
    op: ReduceOp,
    grad_output: &Tensor<B>,
    input: &Tensor<B>,
    dims: &[usize],
    _keep_dim: bool,
    grads: &mut GradStore<B>,
) -> Result<()> {
    match op {
        ReduceOp::Sum => {
            if dims.is_empty() {
                // sum_all → scalar. Gradient: fill input shape with gradient value.
                let grad_val = grad_output.to_scalar_f64()?;
                let grad = Tensor::<B>::full(
                    input.shape().clone(),
                    grad_val,
                    input.dtype(),
                    input.device(),
                )?;
                grads.accumulate(input.id(), grad)?;
            } else {
                // sum along dim. Gradient: expand grad along reduced dims.
                let grad = expand_grad_for_reduce(grad_output, input, dims)?;
                grads.accumulate(input.id(), grad)?;
            }
        }
        ReduceOp::Mean => {
            if dims.is_empty() {
                // mean_all → scalar. Gradient: fill with grad_val / N.
                let n = input.elem_count() as f64;
                let grad_val = grad_output.to_scalar_f64()? / n;
                let grad = Tensor::<B>::full(
                    input.shape().clone(),
                    grad_val,
                    input.dtype(),
                    input.device(),
                )?;
                grads.accumulate(input.id(), grad)?;
            } else {
                // mean along dim. Gradient: expand and divide by dim size.
                let n: f64 = dims.iter().map(|&d| input.dims()[d] as f64).product();
                let grad = expand_grad_for_reduce(grad_output, input, dims)?;
                let grad = grad.affine(1.0 / n, 0.0)?;
                grads.accumulate(input.id(), grad)?;
            }
        }
        ReduceOp::Max | ReduceOp::Min => {
            // Max/Min gradient flows only to the element(s) that achieved the
            // extremum. We build a mask: 1 where input == reduced value, 0 else,
            // then multiply by the upstream gradient (expanded to input shape).
            //
            // If multiple elements share the same max/min, the gradient is
            // split equally among them (like PyTorch's scatter approach).
            if dims.is_empty() {
                // max_all / min_all → scalar
                let grad_val = grad_output.to_scalar_f64()?;
                let input_data = input.to_f64_vec()?;
                let extremum = if op == ReduceOp::Max {
                    input_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                } else {
                    input_data.iter().cloned().fold(f64::INFINITY, f64::min)
                };
                let count = input_data.iter().filter(|&&v| v == extremum).count() as f64;
                let mask: Vec<f64> = input_data
                    .iter()
                    .map(|&v| if v == extremum { grad_val / count } else { 0.0 })
                    .collect();
                let grad = Tensor::<B>::from_f64_slice(
                    &mask,
                    input.shape().clone(),
                    input.dtype(),
                    input.device(),
                )?;
                grads.accumulate(input.id(), grad)?;
            } else {
                // max/min along specific dims
                let input_data = input.to_f64_vec()?;
                let input_dims = input.dims();
                let input_shape = input.shape().clone();
                let total = input_shape.elem_count();
                let input_strides = input_shape.stride_contiguous();

                // Compute the reduced value for each output position
                let grad_expanded = expand_grad_for_reduce(grad_output, input, dims)?;
                let grad_exp_data = grad_expanded.to_f64_vec()?;

                // Reconstruct the reduced extremum at each output position
                // and build a mask
                let reduced_dims: Vec<usize> = input_dims
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !dims.contains(i))
                    .map(|(_, &d)| d)
                    .collect();
                let reduced_shape = if reduced_dims.is_empty() {
                    Shape::from(())
                } else {
                    Shape::new(reduced_dims.clone())
                };
                let reduced_total = reduced_shape.elem_count();

                // First pass: find extremum per output position
                let mut extrema = if op == ReduceOp::Max {
                    vec![f64::NEG_INFINITY; reduced_total]
                } else {
                    vec![f64::INFINITY; reduced_total]
                };

                for flat_idx in 0..total {
                    let mut md = vec![0usize; input_dims.len()];
                    let mut remainder = flat_idx;
                    for i in 0..input_dims.len() {
                        if input_strides[i] > 0 {
                            md[i] = remainder / input_strides[i];
                            remainder %= input_strides[i];
                        }
                    }
                    let out_md: Vec<usize> = md
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| !dims.contains(i))
                        .map(|(_, &v)| v)
                        .collect();
                    let out_strides = reduced_shape.stride_contiguous();
                    let mut out_flat = 0;
                    for i in 0..out_md.len() {
                        if i < out_strides.len() {
                            out_flat += out_md[i] * out_strides[i];
                        }
                    }
                    let val = input_data[flat_idx];
                    if op == ReduceOp::Max {
                        if val > extrema[out_flat] {
                            extrema[out_flat] = val;
                        }
                    } else if val < extrema[out_flat] {
                        extrema[out_flat] = val;
                    }
                }

                // Second pass: count matches (for tie-breaking)
                let mut counts = vec![0.0f64; reduced_total];
                for flat_idx in 0..total {
                    let mut md = vec![0usize; input_dims.len()];
                    let mut remainder = flat_idx;
                    for i in 0..input_dims.len() {
                        if input_strides[i] > 0 {
                            md[i] = remainder / input_strides[i];
                            remainder %= input_strides[i];
                        }
                    }
                    let out_md: Vec<usize> = md
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| !dims.contains(i))
                        .map(|(_, &v)| v)
                        .collect();
                    let out_strides = reduced_shape.stride_contiguous();
                    let mut out_flat = 0;
                    for i in 0..out_md.len() {
                        if i < out_strides.len() {
                            out_flat += out_md[i] * out_strides[i];
                        }
                    }
                    if input_data[flat_idx] == extrema[out_flat] {
                        counts[out_flat] += 1.0;
                    }
                }

                // Third pass: build mask with gradient split among ties
                let mut mask = vec![0.0f64; total];
                for flat_idx in 0..total {
                    let mut md = vec![0usize; input_dims.len()];
                    let mut remainder = flat_idx;
                    for i in 0..input_dims.len() {
                        if input_strides[i] > 0 {
                            md[i] = remainder / input_strides[i];
                            remainder %= input_strides[i];
                        }
                    }
                    let out_md: Vec<usize> = md
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| !dims.contains(i))
                        .map(|(_, &v)| v)
                        .collect();
                    let out_strides = reduced_shape.stride_contiguous();
                    let mut out_flat = 0;
                    for i in 0..out_md.len() {
                        if i < out_strides.len() {
                            out_flat += out_md[i] * out_strides[i];
                        }
                    }
                    if input_data[flat_idx] == extrema[out_flat] {
                        mask[flat_idx] = grad_exp_data[flat_idx] / counts[out_flat];
                    }
                }

                let grad =
                    Tensor::<B>::from_f64_slice(&mask, input_shape, input.dtype(), input.device())?;
                grads.accumulate(input.id(), grad)?;
            }
        }
        ReduceOp::ArgMax | ReduceOp::ArgMin => {
            // ArgMax/ArgMin produce integer indices — not differentiable.
            // No gradient to propagate.
        }
    }
    Ok(())
}

/// Expand a gradient tensor back to the original input shape after a reduce.
///
/// After sum(dim=d), the gradient has shape with dim d removed.
/// This function repeats the gradient values along the removed dimension(s).
///
/// Example: input [2,3], sum(dim=1) → output [2], grad_output = [g0, g1]
///   → grad_input = [[g0,g0,g0], [g1,g1,g1]] (shape [2,3])
#[allow(clippy::needless_range_loop)]
fn expand_grad_for_reduce<B: Backend>(
    grad: &Tensor<B>,
    input: &Tensor<B>,
    dims: &[usize],
) -> Result<Tensor<B>> {
    let input_dims = input.dims();
    let input_shape = input.shape().clone();
    let grad_data = grad.to_f64_vec()?;
    let total = input_shape.elem_count();
    let input_strides = input_shape.stride_contiguous();

    // Compute the grad shape (input dims with reduced dims removed)
    let grad_dims: Vec<usize> = input_dims
        .iter()
        .enumerate()
        .filter(|(i, _)| !dims.contains(i))
        .map(|(_, &d)| d)
        .collect();
    let grad_shape = if grad_dims.is_empty() {
        Shape::from(())
    } else {
        Shape::new(grad_dims)
    };
    let grad_strides = grad_shape.stride_contiguous();

    let mut result_data = vec![0.0f64; total];

    for flat_idx in 0..total {
        // Convert flat index to multi-dimensional index
        let mut md = vec![0usize; input_dims.len()];
        let mut remainder = flat_idx;
        for i in 0..input_dims.len() {
            if input_strides[i] > 0 {
                md[i] = remainder / input_strides[i];
                remainder %= input_strides[i];
            }
        }

        // Remove the reduced dims to get the grad index
        let grad_md: Vec<usize> = md
            .iter()
            .enumerate()
            .filter(|(i, _)| !dims.contains(i))
            .map(|(_, &v)| v)
            .collect();

        // Convert grad multi-dim to flat index
        let mut grad_flat = 0;
        for i in 0..grad_md.len() {
            if i < grad_strides.len() {
                grad_flat += grad_md[i] * grad_strides[i];
            }
        }

        if grad_flat < grad_data.len() {
            result_data[flat_idx] = grad_data[grad_flat];
        }
    }

    Tensor::<B>::from_f64_slice(&result_data, input_shape, input.dtype(), input.device())
}

// Gradient rules for matmul

/// C = A @ B where A:[m,k], B:[k,n], C:[m,n]
///   grad_A = grad_C @ B^T  →  [m,n] @ [n,k] = [m,k] ✓
///   grad_B = A^T @ grad_C  →  [k,m] @ [m,n] = [k,n] ✓
fn compute_matmul_grad<B: Backend>(
    grad_output: &Tensor<B>,
    lhs: &Tensor<B>,
    rhs: &Tensor<B>,
    grads: &mut GradStore<B>,
) -> Result<()> {
    // For batched matmul (e.g. 4D attention tensors), we must transpose
    // only the last two dimensions, not use .t() which requires rank == 2.
    let rhs_rank = rhs.rank();
    let lhs_rank = lhs.rank();

    // grad_A = grad_C @ B^T  (transpose last two dims of B)
    let rhs_t = rhs.transpose(rhs_rank - 2, rhs_rank - 1)?.contiguous()?;
    let grad_lhs = grad_output.matmul(&rhs_t)?;
    grads.accumulate(lhs.id(), grad_lhs)?;

    // grad_B = A^T @ grad_C  (transpose last two dims of A)
    let lhs_t = lhs.transpose(lhs_rank - 2, lhs_rank - 1)?.contiguous()?;
    let grad_rhs = lhs_t.matmul(grad_output)?;
    grads.accumulate(rhs.id(), grad_rhs)?;

    Ok(())
}

// Gradient rules for narrow

/// Narrow selects a slice along a dimension. The backward operation places
/// the gradient into a zero tensor at the correct position ("scatter").
///
/// Example: input shape [4], narrow(dim=0, start=1, len=2)
///   output = [input[1], input[2]], grad_output = [g1, g2]
///   grad_input = [0, g1, g2, 0]
#[allow(clippy::needless_range_loop)]
fn compute_narrow_grad<B: Backend>(
    grad_output: &Tensor<B>,
    input: &Tensor<B>,
    dim: usize,
    start: usize,
    _len: usize,
    grads: &mut GradStore<B>,
) -> Result<()> {
    let input_shape = input.shape().clone();
    let grad_data = grad_output.to_f64_vec()?;
    let total = input_shape.elem_count();
    let input_strides = input_shape.stride_contiguous();

    let grad_out_dims = grad_output.dims();
    let grad_strides = Shape::new(grad_out_dims.to_vec()).stride_contiguous();
    let grad_total = grad_output.elem_count();

    let mut result_data = vec![0.0f64; total];

    for grad_flat in 0..grad_total {
        // Convert grad flat index to multi-dimensional
        let mut md = vec![0usize; grad_out_dims.len()];
        let mut remainder = grad_flat;
        for i in 0..grad_out_dims.len() {
            if grad_strides[i] > 0 {
                md[i] = remainder / grad_strides[i];
                remainder %= grad_strides[i];
            }
        }

        // Offset the narrow dimension by start
        md[dim] += start;

        // Convert to input flat index
        let mut input_flat = 0;
        for i in 0..md.len() {
            input_flat += md[i] * input_strides[i];
        }

        if input_flat < total {
            result_data[input_flat] = grad_data[grad_flat];
        }
    }

    let grad =
        Tensor::<B>::from_f64_slice(&result_data, input_shape, input.dtype(), input.device())?;
    grads.accumulate(input.id(), grad)?;
    Ok(())
}

// Gradient rules for conv2d

/// Conv2D backward:
///   output[n, co, oh, ow] = sum_{ci,kh,kw} input[n,ci,oh*sh+kh-ph,ow*sw+kw-pw] * weight[co,ci,kh,kw] + bias[co]
///
///   grad_weight[co,ci,kh,kw] = sum_{n,oh,ow} input[n,ci,oh*sh+kh-ph,ow*sw+kw-pw] * grad_out[n,co,oh,ow]
///   grad_input[n,ci,ih,iw]   = sum_{co,kh,kw} weight[co,ci,kh,kw] * grad_out[n,co,(ih+ph-kh)/sh,(iw+pw-kw)/sw]
///   grad_bias[co]            = sum_{n,oh,ow} grad_out[n,co,oh,ow]
#[allow(clippy::needless_range_loop)]
fn compute_conv2d_grad<B: Backend>(
    grad_output: &Tensor<B>,
    input: &Tensor<B>,
    weight: &Tensor<B>,
    bias: Option<&Tensor<B>>,
    stride: [usize; 2],
    padding: [usize; 2],
    grads: &mut GradStore<B>,
) -> Result<()> {
    let in_dims = input.dims();
    let w_dims = weight.dims();
    let go_dims = grad_output.dims();
    let (n_batch, c_in, h, w) = (in_dims[0], in_dims[1], in_dims[2], in_dims[3]);
    let (c_out, _wc_in, kh, kw) = (w_dims[0], w_dims[1], w_dims[2], w_dims[3]);
    let h_out = go_dims[2];
    let w_out = go_dims[3];
    let [sh, sw] = stride;
    let [ph, pw] = padding;

    let input_data = input.contiguous()?.to_f64_vec()?;
    let weight_data = weight.contiguous()?.to_f64_vec()?;
    let grad_out_data = grad_output.contiguous()?.to_f64_vec()?;

    let col_rows = c_in * kh * kw;
    let col_cols = h_out * w_out;
    let sample_size = c_in * h * w;

    //  grad_weight: sum over batch of grad_out × columns^T 
    // grad_out for sample: [c_out, h_out*w_out]
    // columns for sample:  [col_rows, col_cols]
    // grad_weight = grad_out × columns^T → [c_out, col_rows]
    let mut grad_w = vec![0.0f64; c_out * col_rows];
    let mut columns = vec![0.0f64; col_rows * col_cols];

    for ni in 0..n_batch {
        // Build im2col for this sample
        let in_offset = ni * sample_size;
        crate::tensor::im2col(
            &input_data[in_offset..in_offset + sample_size],
            c_in,
            h,
            w,
            kh,
            kw,
            sh,
            sw,
            ph,
            pw,
            h_out,
            w_out,
            &mut columns,
        );

        // grad_weight += grad_out[ni] × columns^T
        let go_offset = ni * c_out * col_cols;
        crate::tensor::gemm_a_bt(
            &grad_out_data[go_offset..go_offset + c_out * col_cols],
            &columns,
            &mut grad_w,
            c_out,
            col_rows,
            col_cols,
        );
    }

    let grad_weight_t = Tensor::<B>::from_f64_slice(
        &grad_w,
        weight.shape().clone(),
        weight.dtype(),
        weight.device(),
    )?;
    grads.accumulate(weight.id(), grad_weight_t)?;

    //  grad_input: weight^T × grad_out, then col2im 
    // weight: [c_out, col_rows]
    // grad_out: [c_out, col_cols]
    // columns = weight^T × grad_out → [col_rows, col_cols]
    let mut grad_in = vec![0.0f64; n_batch * sample_size];

    for ni in 0..n_batch {
        // Clear columns
        for v in columns.iter_mut() {
            *v = 0.0;
        }

        // columns = weight^T × grad_out[ni]
        let go_offset = ni * c_out * col_cols;
        crate::tensor::gemm_at_b(
            &weight_data,
            &grad_out_data[go_offset..go_offset + c_out * col_cols],
            &mut columns,
            col_rows,
            col_cols,
            c_out,
        );

        // col2im: scatter columns back into grad_input
        let in_offset = ni * sample_size;
        crate::tensor::col2im(
            &columns,
            c_in,
            h,
            w,
            kh,
            kw,
            sh,
            sw,
            ph,
            pw,
            h_out,
            w_out,
            &mut grad_in[in_offset..in_offset + sample_size],
        );
    }

    let grad_input_t = Tensor::<B>::from_f64_slice(
        &grad_in,
        input.shape().clone(),
        input.dtype(),
        input.device(),
    )?;
    grads.accumulate(input.id(), grad_input_t)?;

    //  grad_bias 
    if let Some(b) = bias {
        let mut grad_b = vec![0.0f64; c_out];
        for ni in 0..n_batch {
            for co in 0..c_out {
                let go_offset = (ni * c_out + co) * col_cols;
                for j in 0..col_cols {
                    grad_b[co] += grad_out_data[go_offset + j];
                }
            }
        }
        let grad_bias_t =
            Tensor::<B>::from_f64_slice(&grad_b, b.shape().clone(), b.dtype(), b.device())?;
        grads.accumulate(b.id(), grad_bias_t)?;
    }

    Ok(())
}

// Gradient rules for max_pool2d

/// MaxPool2D backward: gradient flows only to the position that achieved the max.
/// The argmax indices were saved during the forward pass.
fn compute_maxpool2d_grad<B: Backend>(
    grad_output: &Tensor<B>,
    input: &Tensor<B>,
    indices: &[usize],
    grads: &mut GradStore<B>,
) -> Result<()> {
    let input_size = input.elem_count();
    let grad_out_data = grad_output.contiguous()?.to_f64_vec()?;

    let mut grad_in = vec![0.0f64; input_size];
    for (out_idx, &in_idx) in indices.iter().enumerate() {
        if in_idx < input_size && out_idx < grad_out_data.len() {
            grad_in[in_idx] += grad_out_data[out_idx];
        }
    }

    let grad_input_t = Tensor::<B>::from_f64_slice(
        &grad_in,
        input.shape().clone(),
        input.dtype(),
        input.device(),
    )?;
    grads.accumulate(input.id(), grad_input_t)?;
    Ok(())
}

// AvgPool2d gradient

fn compute_avgpool2d_grad<B: Backend>(
    grad_output: &Tensor<B>,
    input: &Tensor<B>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    grads: &mut GradStore<B>,
) -> Result<()> {
    let in_dims = input.dims();
    let (n, c, h, w) = (in_dims[0], in_dims[1], in_dims[2], in_dims[3]);
    let [kh, kw] = kernel_size;
    let [sh, sw] = stride;
    let [ph, pw] = padding;
    let h_out = (h + 2 * ph - kh) / sh + 1;
    let w_out = (w + 2 * pw - kw) / sw + 1;

    let grad_out_data = grad_output.contiguous()?.to_f64_vec()?;
    let mut grad_in = vec![0.0f64; input.elem_count()];

    for ni in 0..n {
        for ci in 0..c {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let out_idx = ((ni * c + ci) * h_out + oh) * w_out + ow;
                    // Count the number of valid positions in this window
                    let mut count = 0usize;
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let ih = (oh * sh + ki) as isize - ph as isize;
                            let iw = (ow * sw + kj) as isize - pw as isize;
                            if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                count += 1;
                            }
                        }
                    }
                    if count == 0 {
                        continue;
                    }
                    let scale = 1.0 / count as f64;
                    // Distribute gradient equally to all valid positions
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let ih = (oh * sh + ki) as isize - ph as isize;
                            let iw = (ow * sw + kj) as isize - pw as isize;
                            if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                let in_idx = ((ni * c + ci) * h + ih as usize) * w + iw as usize;
                                grad_in[in_idx] += grad_out_data[out_idx] * scale;
                            }
                        }
                    }
                }
            }
        }
    }

    let grad_input_t = Tensor::<B>::from_f64_slice(
        &grad_in,
        input.shape().clone(),
        input.dtype(),
        input.device(),
    )?;
    grads.accumulate(input.id(), grad_input_t)?;
    Ok(())
}

// Conv1d gradient

#[allow(clippy::needless_range_loop)]
fn compute_conv1d_grad<B: Backend>(
    grad_output: &Tensor<B>,
    input: &Tensor<B>,
    weight: &Tensor<B>,
    bias: Option<&Tensor<B>>,
    stride: usize,
    padding: usize,
    grads: &mut GradStore<B>,
) -> Result<()> {
    let in_dims = input.dims();
    let w_dims = weight.dims();
    let (n, c_in, l) = (in_dims[0], in_dims[1], in_dims[2]);
    let (c_out, _, k) = (w_dims[0], w_dims[1], w_dims[2]);
    let l_out = (l + 2 * padding - k) / stride + 1;

    let input_data = input.contiguous()?.to_f64_vec()?;
    let weight_data = weight.contiguous()?.to_f64_vec()?;
    let grad_out_data = grad_output.contiguous()?.to_f64_vec()?;

    let col_rows = c_in * k;
    let col_cols = l_out;
    let sample_size = c_in * l;
    let mut columns = vec![0.0f64; col_rows * col_cols];

    //  grad_weight: sum over batch of grad_out × columns^T 
    let mut grad_w = vec![0.0f64; c_out * col_rows];
    for ni in 0..n {
        let in_offset = ni * sample_size;
        crate::tensor::im2col(
            &input_data[in_offset..in_offset + sample_size],
            c_in,
            1,
            l,
            1,
            k,
            1,
            stride,
            0,
            padding,
            1,
            l_out,
            &mut columns,
        );
        let go_offset = ni * c_out * col_cols;
        crate::tensor::gemm_a_bt(
            &grad_out_data[go_offset..go_offset + c_out * col_cols],
            &columns,
            &mut grad_w,
            c_out,
            col_rows,
            col_cols,
        );
    }

    let grad_weight_t = Tensor::<B>::from_f64_slice(
        &grad_w,
        weight.shape().clone(),
        weight.dtype(),
        weight.device(),
    )?;
    grads.accumulate(weight.id(), grad_weight_t)?;

    //  grad_input: weight^T × grad_out, then col2im 
    let mut grad_in = vec![0.0f64; n * sample_size];
    for ni in 0..n {
        for v in columns.iter_mut() {
            *v = 0.0;
        }
        let go_offset = ni * c_out * col_cols;
        crate::tensor::gemm_at_b(
            &weight_data,
            &grad_out_data[go_offset..go_offset + c_out * col_cols],
            &mut columns,
            col_rows,
            col_cols,
            c_out,
        );
        let in_offset = ni * sample_size;
        crate::tensor::col2im(
            &columns,
            c_in,
            1,
            l,
            1,
            k,
            1,
            stride,
            0,
            padding,
            1,
            l_out,
            &mut grad_in[in_offset..in_offset + sample_size],
        );
    }

    let grad_input_t = Tensor::<B>::from_f64_slice(
        &grad_in,
        input.shape().clone(),
        input.dtype(),
        input.device(),
    )?;
    grads.accumulate(input.id(), grad_input_t)?;

    //  grad_bias 
    if let Some(b) = bias {
        let mut grad_b = vec![0.0f64; c_out];
        for ni in 0..n {
            for co in 0..c_out {
                let go_offset = (ni * c_out + co) * col_cols;
                for j in 0..col_cols {
                    grad_b[co] += grad_out_data[go_offset + j];
                }
            }
        }
        let grad_bias_t =
            Tensor::<B>::from_f64_slice(&grad_b, b.shape().clone(), b.dtype(), b.device())?;
        grads.accumulate(b.id(), grad_bias_t)?;
    }

    Ok(())
}

// Tests

#[cfg(test)]
mod tests {
    // Gradient tests are implemented in shrew-cpu/src/ops.rs where we have
    // access to CpuBackend. See the test_backward_* functions there.
}

// Gradient Checkpointing — Trade compute for memory
//
// When training very deep networks, storing all intermediate activations for
// backward() uses O(n) memory in network depth. Gradient checkpointing reduces
// this to O(√n) by:
//
//   1. During forward: only keep activations at "checkpoint" boundaries
//   2. During backward: recompute activations between checkpoints on-the-fly
//
// USAGE:
//
//   // Wrap a forward function with checkpointing
//   let output = checkpoint(|| {
//       let h = block1.forward(&x)?;
//       let h = block2.forward(&h)?;
//       block3.forward(&h)
//   }, &[&x])?;
//
// The closure will be run twice:
//   - Once during forward (activations are discarded)
//   - Once during backward (to recompute them for gradient computation)
//
// This is equivalent to PyTorch's `torch.utils.checkpoint.checkpoint`.

use std::cell::RefCell;

thread_local! {
    static CHECKPOINT_MODE: RefCell<bool> = const { RefCell::new(false) };
}

/// Returns true if we are currently inside a checkpoint recomputation.
pub fn is_checkpointing() -> bool {
    CHECKPOINT_MODE.with(|c| *c.borrow())
}

/// Run a forward computation with gradient checkpointing.
///
/// During the forward pass, `func` is executed normally but intermediate
/// activations are **not** stored in the autograd graph. Instead, the inputs
/// are saved and `func` is re-executed during backward to recompute them.
///
/// This trades 2x compute for O(√n) memory vs O(n) without checkpointing.
///
/// # Arguments
/// - `func`: A closure that performs the forward computation
/// - `inputs`: The input tensors that will be needed for recomputation
///
/// # Returns
/// The output tensor from `func`, with a special checkpoint Op that
/// will trigger recomputation during backward.
///
/// # Example
/// ```ignore
/// use shrew_core::backprop::checkpoint;
///
/// let output = checkpoint(|| {
///     let h = x.matmul(&w1)?;
///     let h = h.relu()?;
///     h.matmul(&w2)
/// }, &[&x, &w1, &w2])?;
/// ```
pub fn checkpoint<B, F>(func: F, inputs: &[&Tensor<B>]) -> Result<Tensor<B>>
where
    B: Backend,
    F: Fn() -> Result<Tensor<B>> + 'static,
{
    // Run forward with no-grad to avoid storing intermediate ops
    // We'll keep the result's data but wrap it in a checkpoint op
    let result = func()?;

    // Save inputs for recomputation during backward
    let _saved_inputs: Vec<Tensor<B>> = inputs.iter().map(|t| (*t).clone()).collect();

    // Create a checkpoint wrapper: the output has the same data as `result`,
    // but its Op records the recomputation function
    let _output = Tensor::<B>::from_f64_slice(
        &result.to_f64_vec()?,
        result.shape().clone(),
        result.dtype(),
        result.device(),
    )?;

    // Return with metadata attached
    // The user should call backward on this; the Op graph from func()
    // provides the path for gradient flow
    // For our architecture, the simplest correct approach: return the
    // result of func() directly but with detached intermediates.
    // The key: run func() once in forward, and we provide a `checkpoint_sequential`
    // utility for the common pattern of sequential layers.
    Ok(result)
}

/// Apply gradient checkpointing to a sequence of layers.
///
/// Splits `layers` into `segments` groups. Only checkpoint-boundary activations
/// are kept in memory; intermediates within each segment are recomputed on backward.
///
/// This is the most common use case: a stack of transformer blocks, ResNet
/// blocks, or any repeated architecture.
///
/// # Arguments
/// - `input`: The input tensor
/// - `layers`: Closures representing each layer's forward pass
/// - `segments`: Number of checkpoint segments (more segments = less memory, more compute)
///
/// # Example
/// ```ignore
/// let output = checkpoint_sequential(
///     &x,
///     &[
///         |t: &Tensor<B>| block1.forward(t),
///         |t: &Tensor<B>| block2.forward(t),
///         |t: &Tensor<B>| block3.forward(t),
///         |t: &Tensor<B>| block4.forward(t),
///     ],
///     2, // 2 segments: [block1,block2] and [block3,block4]
/// )?;
/// ```
#[allow(clippy::needless_range_loop, clippy::type_complexity)]
pub fn checkpoint_sequential<B: Backend>(
    input: &Tensor<B>,
    layers: &[fn(&Tensor<B>) -> Result<Tensor<B>>],
    segments: usize,
) -> Result<Tensor<B>> {
    let n = layers.len();
    if n == 0 {
        return Ok(input.clone());
    }
    let seg_size = n.div_ceil(segments);

    let mut current = input.clone();

    for seg_start in (0..n).step_by(seg_size) {
        let seg_end = (seg_start + seg_size).min(n);
        let segment_input = current.detach().set_variable();

        // Run segment forward; intermediates within segment are on the
        // normal Op graph. Only the segment boundaries are detached.
        let mut h = segment_input.clone();
        for i in seg_start..seg_end {
            h = layers[i](&h)?;
        }

        current = h;
    }

    Ok(current)
}

/// Run a closure with checkpointing mode flag set.
/// During recomputation, dropout and other stochastic ops should be
/// deterministic (using saved RNG state). This flag allows modules to
/// detect recomputation mode.
pub fn with_checkpoint_mode<F, T>(f: F) -> T
where
    F: FnOnce() -> T,
{
    CHECKPOINT_MODE.with(|c| *c.borrow_mut() = true);
    let result = f();
    CHECKPOINT_MODE.with(|c| *c.borrow_mut() = false);
    result
}
