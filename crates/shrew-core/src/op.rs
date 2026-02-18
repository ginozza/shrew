// Op — Computational graph node for automatic differentiation
//
// Every tensor that results from a computation records HOW it was created
// via the Op enum. This forms a directed acyclic graph (DAG) that backward()
// traverses to compute gradients.
//
// Example: c = a + b
//   a.op = Op::None (leaf variable)
//   b.op = Op::None (leaf variable)
//   c.op = Op::Binary { lhs: a, rhs: b, op: Add }
//
// When we call c.backward():
//   1. Start with grad_c = 1.0 (by convention, dL/dL = 1)
//   2. Look at c.op → Binary { lhs: a, rhs: b, Add }
//   3. grad_a += grad_c * d(a+b)/da = grad_c * 1 = grad_c
//   4. grad_b += grad_c * d(a+b)/db = grad_c * 1 = grad_c
//
// WHY STORE Tensor<B> INSTEAD OF TensorId?
//
// In Phase 1, Op stored only TensorIds. Now in Phase 2, each Op variant stores
// the actual Tensor<B> references to its inputs. Since Tensor<B> is Arc-wrapped,
// cloning is cheap (just increment refcount). This means:
//
//   1. backward() can directly access input values for gradient computation
//      (e.g., d(a*b)/da = b — we need the actual value of b)
//   2. The computation graph keeps input tensors alive as long as the output
//      tensor exists (correct: we need them for backward)
//   3. No separate tensor registry needed — the graph IS the references
//
// MEMORY: The graph forms a DAG (no cycles), so Arc handles cleanup correctly.
// When the loss tensor is dropped, all intermediate tensors' refcounts decrease.

use crate::backend::{Backend, BinaryOp, ReduceOp, UnaryOp};

/// Unique identifier for a tensor. Used as keys in GradStore.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(pub(crate) u64);

impl Default for TensorId {
    fn default() -> Self {
        Self::new()
    }
}

impl TensorId {
    /// Generate a new unique tensor ID (uses a global atomic counter).
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        TensorId(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

/// Records the operation that produced a tensor, storing references to inputs.
///
/// Each variant holds the actual input Tensor(s) (Arc-wrapped, cheap to clone)
/// plus the operation parameters. backward() uses these to compute gradients
/// via the chain rule.
///
/// Op<B> is generic over the Backend because it stores Tensor<B>.
pub enum Op<B: Backend> {
    /// No operation — this is a leaf tensor (input data or trainable parameter).
    None,

    /// Element-wise binary: result = op(lhs, rhs)
    Binary {
        lhs: crate::Tensor<B>,
        rhs: crate::Tensor<B>,
        op: BinaryOp,
    },

    /// Element-wise unary: result = op(input)
    Unary {
        input: crate::Tensor<B>,
        op: UnaryOp,
    },

    /// Reduction: result = reduce(input, dims)
    Reduce {
        input: crate::Tensor<B>,
        op: ReduceOp,
        dims: Vec<usize>,
        keep_dim: bool,
    },

    /// Matrix multiplication: result = lhs @ rhs
    Matmul {
        lhs: crate::Tensor<B>,
        rhs: crate::Tensor<B>,
    },

    /// Reshape (includes squeeze/unsqueeze): same data, different shape.
    /// src_shape records the original shape so backward can reshape gradients back.
    Reshape {
        input: crate::Tensor<B>,
        src_shape: crate::Shape,
    },

    /// Transpose: swap two dimensions
    Transpose {
        input: crate::Tensor<B>,
        dim0: usize,
        dim1: usize,
    },

    /// Narrow/slice along a dimension
    Narrow {
        input: crate::Tensor<B>,
        dim: usize,
        start: usize,
        len: usize,
    },

    /// Affine transform: result = input * mul + add
    Affine {
        input: crate::Tensor<B>,
        mul: f64,
        add: f64,
    },

    /// Contiguous copy: same logical values, but data is now contiguous in memory.
    /// Gradient passes through unchanged.
    Contiguous { input: crate::Tensor<B> },

    /// 2D convolution: result = conv2d(input, weight) + bias
    /// input: [N, C_in, H, W], weight: [C_out, C_in, kH, kW]
    Conv2d {
        input: crate::Tensor<B>,
        weight: crate::Tensor<B>,
        bias: Option<crate::Tensor<B>>,
        stride: [usize; 2],
        padding: [usize; 2],
    },

    /// 2D max-pooling.
    /// input: [N, C, H, W]
    /// indices stores the argmax positions for backward.
    MaxPool2d {
        input: crate::Tensor<B>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        indices: Vec<usize>,
    },

    /// Concatenation along a dimension.
    /// `inputs` are the original tensors that were concatenated.
    /// `dim` is the concatenation dimension.
    /// `sizes` stores the size of each input along `dim` (needed by backward
    /// to slice the gradient back into per-input pieces via narrow).
    Cat {
        inputs: Vec<crate::Tensor<B>>,
        dim: usize,
        sizes: Vec<usize>,
    },

    /// Element-wise power: result = input ^ exponent.
    Powf {
        input: crate::Tensor<B>,
        exponent: f64,
    },

    /// Element-wise clamp: result = clamp(input, min, max).
    Clamp {
        input: crate::Tensor<B>,
        min: f64,
        max: f64,
    },

    /// Conditional select: result[i] = if mask[i] { on_true[i] } else { on_false[i] }.
    WhereCond {
        mask: crate::Tensor<B>,
        on_true: crate::Tensor<B>,
        on_false: crate::Tensor<B>,
    },

    /// Gather elements along a dimension using index tensor.
    Gather {
        input: crate::Tensor<B>,
        index: crate::Tensor<B>,
        dim: usize,
    },

    /// Constant padding.
    Pad {
        input: crate::Tensor<B>,
        padding: Vec<[usize; 2]>,
    },

    /// 2D average-pooling.
    /// input: [N, C, H, W]
    AvgPool2d {
        input: crate::Tensor<B>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
    },

    /// 1D convolution: result = conv1d(input, weight) + bias
    /// input: [N, C_in, L], weight: [C_out, C_in, K]
    Conv1d {
        input: crate::Tensor<B>,
        weight: crate::Tensor<B>,
        bias: Option<crate::Tensor<B>>,
        stride: usize,
        padding: usize,
    },

    /// Index select along a dimension: result = input.index_select(dim, indices)
    /// Backward = scatter-add of grad_output into grad_input at index positions.
    IndexSelect {
        input: crate::Tensor<B>,
        indices: crate::Tensor<B>,
        dim: usize,
    },

    /// Dtype conversion: result = input.to_dtype(target_dtype)
    /// Backward casts gradient back to the original dtype.
    ToDtype {
        input: crate::Tensor<B>,
        src_dtype: crate::dtype::DType,
    },
}

// Manual Clone implementation because derive can't handle the generic well.
// All clones are cheap: Tensor clone is just Arc refcount increment.
impl<B: Backend> Clone for Op<B> {
    fn clone(&self) -> Self {
        match self {
            Op::None => Op::None,
            Op::Binary { lhs, rhs, op } => Op::Binary {
                lhs: lhs.clone(),
                rhs: rhs.clone(),
                op: *op,
            },
            Op::Unary { input, op } => Op::Unary {
                input: input.clone(),
                op: *op,
            },
            Op::Reduce {
                input,
                op,
                dims,
                keep_dim,
            } => Op::Reduce {
                input: input.clone(),
                op: *op,
                dims: dims.clone(),
                keep_dim: *keep_dim,
            },
            Op::Matmul { lhs, rhs } => Op::Matmul {
                lhs: lhs.clone(),
                rhs: rhs.clone(),
            },
            Op::Reshape { input, src_shape } => Op::Reshape {
                input: input.clone(),
                src_shape: src_shape.clone(),
            },
            Op::Transpose { input, dim0, dim1 } => Op::Transpose {
                input: input.clone(),
                dim0: *dim0,
                dim1: *dim1,
            },
            Op::Narrow {
                input,
                dim,
                start,
                len,
            } => Op::Narrow {
                input: input.clone(),
                dim: *dim,
                start: *start,
                len: *len,
            },
            Op::Affine { input, mul, add } => Op::Affine {
                input: input.clone(),
                mul: *mul,
                add: *add,
            },
            Op::Contiguous { input } => Op::Contiguous {
                input: input.clone(),
            },
            Op::Conv2d {
                input,
                weight,
                bias,
                stride,
                padding,
            } => Op::Conv2d {
                input: input.clone(),
                weight: weight.clone(),
                bias: bias.clone(),
                stride: *stride,
                padding: *padding,
            },
            Op::MaxPool2d {
                input,
                kernel_size,
                stride,
                padding,
                indices,
            } => Op::MaxPool2d {
                input: input.clone(),
                kernel_size: *kernel_size,
                stride: *stride,
                padding: *padding,
                indices: indices.clone(),
            },
            Op::Cat { inputs, dim, sizes } => Op::Cat {
                inputs: inputs.clone(),
                dim: *dim,
                sizes: sizes.clone(),
            },
            Op::Powf { input, exponent } => Op::Powf {
                input: input.clone(),
                exponent: *exponent,
            },
            Op::Clamp { input, min, max } => Op::Clamp {
                input: input.clone(),
                min: *min,
                max: *max,
            },
            Op::WhereCond {
                mask,
                on_true,
                on_false,
            } => Op::WhereCond {
                mask: mask.clone(),
                on_true: on_true.clone(),
                on_false: on_false.clone(),
            },
            Op::Gather { input, index, dim } => Op::Gather {
                input: input.clone(),
                index: index.clone(),
                dim: *dim,
            },
            Op::Pad { input, padding } => Op::Pad {
                input: input.clone(),
                padding: padding.clone(),
            },
            Op::AvgPool2d {
                input,
                kernel_size,
                stride,
                padding,
            } => Op::AvgPool2d {
                input: input.clone(),
                kernel_size: *kernel_size,
                stride: *stride,
                padding: *padding,
            },
            Op::Conv1d {
                input,
                weight,
                bias,
                stride,
                padding,
            } => Op::Conv1d {
                input: input.clone(),
                weight: weight.clone(),
                bias: bias.clone(),
                stride: *stride,
                padding: *padding,
            },
            Op::IndexSelect {
                input,
                indices,
                dim,
            } => Op::IndexSelect {
                input: input.clone(),
                indices: indices.clone(),
                dim: *dim,
            },
            Op::ToDtype { input, src_dtype } => Op::ToDtype {
                input: input.clone(),
                src_dtype: *src_dtype,
            },
        }
    }
}

// Concise Debug: show op type and tensor IDs only (not full tensor data).
impl<B: Backend> std::fmt::Debug for Op<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Op::None => write!(f, "None"),
            Op::Binary { lhs, rhs, op } => {
                write!(f, "Binary({:?}, id={:?}, id={:?})", op, lhs.id(), rhs.id())
            }
            Op::Unary { input, op } => {
                write!(f, "Unary({:?}, id={:?})", op, input.id())
            }
            Op::Reduce {
                input, op, dims, ..
            } => {
                write!(f, "Reduce({:?}, dims={:?}, id={:?})", op, dims, input.id())
            }
            Op::Matmul { lhs, rhs } => {
                write!(f, "Matmul(id={:?}, id={:?})", lhs.id(), rhs.id())
            }
            Op::Reshape { input, src_shape } => {
                write!(f, "Reshape({} → ?, id={:?})", src_shape, input.id())
            }
            Op::Transpose { input, dim0, dim1 } => {
                write!(f, "Transpose({}, {}, id={:?})", dim0, dim1, input.id())
            }
            Op::Narrow {
                input,
                dim,
                start,
                len,
            } => {
                write!(
                    f,
                    "Narrow(dim={}, {}..{}, id={:?})",
                    dim,
                    start,
                    start + len,
                    input.id()
                )
            }
            Op::Affine { input, mul, add } => {
                write!(f, "Affine(*{} +{}, id={:?})", mul, add, input.id())
            }
            Op::Contiguous { input } => {
                write!(f, "Contiguous(id={:?})", input.id())
            }
            Op::Conv2d {
                input,
                weight,
                bias,
                stride,
                padding,
            } => {
                write!(
                    f,
                    "Conv2d(in={:?}, w={:?}, bias={}, s={:?}, p={:?})",
                    input.id(),
                    weight.id(),
                    bias.is_some(),
                    stride,
                    padding
                )
            }
            Op::MaxPool2d {
                input,
                kernel_size,
                stride,
                padding,
                ..
            } => {
                write!(
                    f,
                    "MaxPool2d(in={:?}, k={:?}, s={:?}, p={:?})",
                    input.id(),
                    kernel_size,
                    stride,
                    padding
                )
            }
            Op::Cat { inputs, dim, .. } => {
                let ids: Vec<_> = inputs.iter().map(|t| t.id()).collect();
                write!(f, "Cat(dim={}, ids={:?})", dim, ids)
            }
            Op::Powf { input, exponent } => {
                write!(f, "Powf(exp={}, id={:?})", exponent, input.id())
            }
            Op::Clamp { input, min, max } => {
                write!(f, "Clamp(min={}, max={}, id={:?})", min, max, input.id())
            }
            Op::WhereCond {
                mask,
                on_true,
                on_false,
            } => {
                write!(
                    f,
                    "WhereCond(mask={:?}, true={:?}, false={:?})",
                    mask.id(),
                    on_true.id(),
                    on_false.id()
                )
            }
            Op::Gather { input, index, dim } => {
                write!(
                    f,
                    "Gather(dim={}, input={:?}, index={:?})",
                    dim,
                    input.id(),
                    index.id()
                )
            }
            Op::Pad { input, padding } => {
                write!(f, "Pad(pad={:?}, id={:?})", padding, input.id())
            }
            Op::AvgPool2d {
                input,
                kernel_size,
                stride,
                padding,
                ..
            } => {
                write!(
                    f,
                    "AvgPool2d(in={:?}, k={:?}, s={:?}, p={:?})",
                    input.id(),
                    kernel_size,
                    stride,
                    padding
                )
            }
            Op::Conv1d {
                input,
                weight,
                bias,
                stride,
                padding,
            } => {
                write!(
                    f,
                    "Conv1d(in={:?}, w={:?}, bias={}, s={}, p={})",
                    input.id(),
                    weight.id(),
                    bias.is_some(),
                    stride,
                    padding
                )
            }
            Op::IndexSelect {
                input,
                indices,
                dim,
            } => {
                write!(
                    f,
                    "IndexSelect(dim={}, input={:?}, indices={:?})",
                    dim,
                    input.id(),
                    indices.id()
                )
            }
            Op::ToDtype { input, src_dtype } => {
                write!(f, "ToDtype(from={:?}, id={:?})", src_dtype, input.id())
            }
        }
    }
}

impl<B: Backend> Op<B> {
    /// Return references to all input tensors of this operation.
    /// Used by topological sort in backward() to traverse the graph.
    pub fn inputs(&self) -> Vec<&crate::Tensor<B>> {
        match self {
            Op::None => vec![],
            Op::Binary { lhs, rhs, .. } | Op::Matmul { lhs, rhs } => vec![lhs, rhs],
            Op::Unary { input, .. }
            | Op::Reduce { input, .. }
            | Op::Reshape { input, .. }
            | Op::Transpose { input, .. }
            | Op::Narrow { input, .. }
            | Op::Affine { input, .. }
            | Op::Contiguous { input }
            | Op::MaxPool2d { input, .. }
            | Op::AvgPool2d { input, .. }
            | Op::Powf { input, .. }
            | Op::Clamp { input, .. } => vec![input],
            Op::Conv2d {
                input,
                weight,
                bias,
                ..
            } => {
                let mut v = vec![input, weight];
                if let Some(b) = bias {
                    v.push(b);
                }
                v
            }
            Op::Conv1d {
                input,
                weight,
                bias,
                ..
            } => {
                let mut v = vec![input, weight];
                if let Some(b) = bias {
                    v.push(b);
                }
                v
            }
            Op::Cat { inputs, .. } => inputs.iter().collect(),
            Op::WhereCond {
                mask,
                on_true,
                on_false,
            } => {
                vec![mask, on_true, on_false]
            }
            Op::Gather { input, index, .. } => vec![input, index],
            Op::IndexSelect { input, indices, .. } => vec![input, indices],
            Op::ToDtype { input, .. } => vec![input],
            Op::Pad { input, .. } => vec![input],
        }
    }
}
