// Shape Inference — Propagate tensor shapes through the computation graph
//
// Shape inference fills in Unknown output types by propagating shapes forward
// through the graph in topological order. This enables:
//
//   1. Early detection of shape mismatches
//   2. Memory planning (allocate exact buffer sizes)
//   3. Kernel selection (choose optimal implementation for shapes)
//
// After inference, every node output_type should ideally be a concrete
// IrType::Tensor (except for truly dynamic/symbolic shapes).
//
// RULES:
//   - Identity: output = input shape
//   - Add/Sub/Mul/Div: shapes must broadcast, output = broadcast shape
//   - MatMul: [.., M, K] × [.., K, N] → [.., M, N]
//   - Relu/Gelu/etc: shape preserved
//   - Softmax: shape preserved
//   - LayerNorm: shape preserved
//   - Transpose: last two dims swapped
//   - Sum/Mean: dims removed (or kept if keepdim)
//   - Reshape: target shape
//   - Concat: sum along concat dim
//   - Dropout: shape preserved
//   - Embedding: indices_shape + [embed_dim]
//   - Linear: [.., in_features] → [.., out_features]

use crate::graph::*;

// Public API

/// Run shape inference on all graphs in the program.
/// Modifies node output_type in place.
pub fn infer_shapes(program: &mut IrProgram) {
    for graph in &mut program.graphs {
        infer_graph_shapes(graph);
    }
}

/// Run shape inference on a single graph.
pub fn infer_graph_shapes(graph: &mut IrGraph) {
    let order = graph.topo_order();
    for id in order {
        let inferred = infer_node_type(graph, id);
        if let Some(ty) = inferred {
            graph.node_mut(id).output_type = ty;
        }
    }
}

// Per-node inference

fn infer_node_type(graph: &IrGraph, id: NodeId) -> Option<IrType> {
    let node = graph.node(id);

    // If already a concrete type, keep it
    if !matches!(node.output_type, IrType::Unknown) {
        return None;
    }

    let inputs: Vec<&IrType> = node
        .inputs
        .iter()
        .map(|&i| &graph.node(i).output_type)
        .collect();

    match &node.op {
        //  Identity: propagate from single input 
        OpKind::Identity => inputs.first().map(|t| (*t).clone()),

        //  Unary element-wise: shape preserved 
        OpKind::Neg
        | OpKind::Relu
        | OpKind::Gelu
        | OpKind::Silu
        | OpKind::Sigmoid
        | OpKind::Tanh
        | OpKind::Exp
        | OpKind::Log
        | OpKind::Sqrt
        | OpKind::Not => inputs.first().map(|t| (*t).clone()),

        //  Softmax: shape preserved 
        OpKind::Softmax { .. } => inputs.first().map(|t| (*t).clone()),

        //  Dropout: shape preserved 
        OpKind::Dropout { .. } => inputs.first().map(|t| (*t).clone()),

        //  LayerNorm / BatchNorm: shape preserved 
        OpKind::LayerNorm { .. } | OpKind::BatchNorm { .. } => inputs.first().map(|t| (*t).clone()),

        //  Binary element-wise: broadcast 
        OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div | OpKind::Mod | OpKind::Pow => {
            if inputs.len() == 2 {
                broadcast_shapes(inputs[0], inputs[1])
            } else {
                None
            }
        }

        //  Comparison: same shape, bool dtype 
        OpKind::Equal
        | OpKind::NotEqual
        | OpKind::Less
        | OpKind::Greater
        | OpKind::LessEqual
        | OpKind::GreaterEqual => {
            if inputs.len() == 2 {
                if let Some(IrType::Tensor { shape, .. }) = broadcast_shapes(inputs[0], inputs[1]) {
                    Some(IrType::Tensor {
                        shape,
                        dtype: DType::Bool,
                    })
                } else {
                    None
                }
            } else {
                None
            }
        }

        //  Logical: same shape, bool 
        OpKind::And | OpKind::Or => {
            if inputs.len() == 2 {
                if let Some(IrType::Tensor { shape, .. }) = broadcast_shapes(inputs[0], inputs[1]) {
                    Some(IrType::Tensor {
                        shape,
                        dtype: DType::Bool,
                    })
                } else {
                    None
                }
            } else {
                None
            }
        }

        //  MatMul: [.., M, K] × [.., K, N] → [.., M, N] 
        OpKind::MatMul => {
            if inputs.len() == 2 {
                infer_matmul(inputs[0], inputs[1])
            } else {
                None
            }
        }

        //  Transpose: swap last two dims 
        OpKind::Transpose => {
            if let Some(IrType::Tensor { shape, dtype }) = inputs.first() {
                if shape.len() >= 2 {
                    let mut new_shape = shape.clone();
                    let n = new_shape.len();
                    new_shape.swap(n - 1, n - 2);
                    Some(IrType::Tensor {
                        shape: new_shape,
                        dtype: *dtype,
                    })
                } else {
                    Some(IrType::Tensor {
                        shape: shape.clone(),
                        dtype: *dtype,
                    })
                }
            } else {
                None
            }
        }

        //  Permute: reorder dimensions 
        OpKind::Permute { dims } => {
            if let Some(IrType::Tensor { shape, dtype }) = inputs.first() {
                let new_shape: Vec<Dim> = dims
                    .iter()
                    .map(|&d| {
                        let idx = if d < 0 { shape.len() as i64 + d } else { d } as usize;
                        shape.get(idx).cloned().unwrap_or(Dim::Dynamic)
                    })
                    .collect();
                Some(IrType::Tensor {
                    shape: new_shape,
                    dtype: *dtype,
                })
            } else {
                None
            }
        }

        //  Reshape / View: target shape (resolve -1 if possible) 
        OpKind::Reshape { target_shape } | OpKind::View { target_shape } => {
            if let Some(IrType::Tensor { dtype, .. }) = inputs.first() {
                Some(IrType::Tensor {
                    shape: target_shape.clone(),
                    dtype: *dtype,
                })
            } else {
                None
            }
        }

        //  Expand: target shape 
        OpKind::Expand { target_shape } => {
            if let Some(IrType::Tensor { dtype, .. }) = inputs.first() {
                Some(IrType::Tensor {
                    shape: target_shape.clone(),
                    dtype: *dtype,
                })
            } else {
                None
            }
        }

        //  Reduction ops 
        OpKind::Sum { dims, keepdim }
        | OpKind::Mean { dims, keepdim }
        | OpKind::Variance { dims, keepdim } => {
            if let Some(IrType::Tensor { shape, dtype }) = inputs.first() {
                Some(infer_reduction(shape, dims, *keepdim, *dtype))
            } else {
                None
            }
        }

        OpKind::Max { dim, keepdim } | OpKind::Min { dim, keepdim } => {
            if let Some(IrType::Tensor { shape, dtype }) = inputs.first() {
                Some(infer_reduction(shape, &[*dim], *keepdim, *dtype))
            } else {
                None
            }
        }

        //  Concat: sum along dim 
        OpKind::Concat { dim } => infer_concat(&inputs, *dim),

        //  Embedding: indices → [.., embed_dim] 
        OpKind::Embedding => {
            // If we have table and indices: table=[V, D], indices=[..] → [.., D]
            if inputs.len() >= 2 {
                if let (
                    IrType::Tensor {
                        shape: table_shape,
                        dtype,
                    },
                    IrType::Tensor {
                        shape: idx_shape, ..
                    },
                ) = (inputs[0], inputs[1])
                {
                    if let Some(embed_dim) = table_shape.last() {
                        let mut out_shape = idx_shape.clone();
                        out_shape.push(embed_dim.clone());
                        return Some(IrType::Tensor {
                            shape: out_shape,
                            dtype: *dtype,
                        });
                    }
                }
            }
            None
        }

        //  Linear 
        OpKind::Linear { .. } => {
            // input=[.., in_features], weight=[out, in] → [.., out_features]
            if inputs.len() >= 2 {
                if let (
                    IrType::Tensor {
                        shape: in_shape,
                        dtype,
                    },
                    IrType::Tensor { shape: w_shape, .. },
                ) = (inputs[0], inputs[1])
                {
                    if !in_shape.is_empty() && w_shape.len() == 2 {
                        let mut out_shape = in_shape[..in_shape.len() - 1].to_vec();
                        out_shape.push(w_shape[0].clone());
                        return Some(IrType::Tensor {
                            shape: out_shape,
                            dtype: *dtype,
                        });
                    }
                }
            }
            None
        }

        //  Loss functions: output is scalar 
        OpKind::CrossEntropy | OpKind::MseLoss => {
            if let Some(IrType::Tensor { dtype, .. }) = inputs.first() {
                Some(IrType::Scalar(*dtype))
            } else {
                None
            }
        }

        //  Attention / Transformer: shape preserved (simplified) 
        OpKind::MultiHeadAttention { .. } | OpKind::TransformerBlock { .. } => {
            inputs.first().map(|t| (*t).clone())
        }

        //  Repeat: shape preserved 
        OpKind::Repeat { .. } => inputs.first().map(|t| (*t).clone()),

        //  Constants: already typed at creation 
        OpKind::Constant(_) => None,

        //  Custom / Call: can't infer 
        OpKind::Custom { .. } | OpKind::Call { .. } => None,

        //  Split / Range: complex, skip 
        OpKind::Split { .. } | OpKind::Range => None,
    }
}

// Shape helpers

/// Broadcast two tensor types. Returns the output type if compatible.
fn broadcast_shapes(left: &IrType, right: &IrType) -> Option<IrType> {
    if let (
        IrType::Tensor {
            shape: ls,
            dtype: ld,
        },
        IrType::Tensor {
            shape: rs,
            dtype: _rd,
        },
    ) = (left, right)
    {
        // dtype of output = left dtype (assume matching)
        let dtype = *ld;
        let max_rank = ls.len().max(rs.len());
        let mut result = Vec::with_capacity(max_rank);

        for i in 0..max_rank {
            let l_idx = if i < ls.len() {
                Some(&ls[ls.len() - 1 - i])
            } else {
                None
            };
            let r_idx = if i < rs.len() {
                Some(&rs[rs.len() - 1 - i])
            } else {
                None
            };

            let dim = match (l_idx, r_idx) {
                (Some(l), None) => l.clone(),
                (None, Some(r)) => r.clone(),
                (Some(l), Some(r)) => broadcast_dim(l, r)?,
                (None, None) => unreachable!(),
            };
            result.push(dim);
        }

        result.reverse();
        Some(IrType::Tensor {
            shape: result,
            dtype,
        })
    } else {
        None
    }
}

/// Broadcast a single dimension pair.
fn broadcast_dim(a: &Dim, b: &Dim) -> Option<Dim> {
    match (a, b) {
        (Dim::Fixed(1), other) | (other, Dim::Fixed(1)) => Some(other.clone()),
        (Dim::Fixed(x), Dim::Fixed(y)) if x == y => Some(Dim::Fixed(*x)),
        (Dim::Fixed(_), Dim::Fixed(_)) => None, // incompatible
        (Dim::Symbolic(s), Dim::Symbolic(t)) if s == t => Some(Dim::Symbolic(s.clone())),
        (Dim::Dynamic, _) | (_, Dim::Dynamic) => Some(Dim::Dynamic),
        (Dim::Symbolic(s), _) | (_, Dim::Symbolic(s)) => Some(Dim::Symbolic(s.clone())),
    }
}

/// MatMul shape: [.., M, K] × [.., K, N] → [.., M, N]
fn infer_matmul(left: &IrType, right: &IrType) -> Option<IrType> {
    if let (IrType::Tensor { shape: ls, dtype }, IrType::Tensor { shape: rs, .. }) = (left, right) {
        if ls.len() < 2 || rs.len() < 2 {
            // 1D matmul: not fully handled, return Dynamic
            return Some(IrType::Tensor {
                shape: vec![Dim::Dynamic],
                dtype: *dtype,
            });
        }

        // Batch dims = broadcast of everything except last 2
        let l_batch = &ls[..ls.len() - 2];
        let r_batch = &rs[..rs.len() - 2];
        let batch = broadcast_batch_dims(l_batch, r_batch);

        let m = ls[ls.len() - 2].clone();
        let n = rs[rs.len() - 1].clone();

        let mut shape = batch;
        shape.push(m);
        shape.push(n);

        Some(IrType::Tensor {
            shape,
            dtype: *dtype,
        })
    } else {
        None
    }
}

fn broadcast_batch_dims(a: &[Dim], b: &[Dim]) -> Vec<Dim> {
    let max_len = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_len);
    for i in 0..max_len {
        let l = if i < a.len() {
            Some(&a[a.len() - 1 - i])
        } else {
            None
        };
        let r = if i < b.len() {
            Some(&b[b.len() - 1 - i])
        } else {
            None
        };
        let dim = match (l, r) {
            (Some(l), None) => l.clone(),
            (None, Some(r)) => r.clone(),
            (Some(l), Some(r)) => broadcast_dim(l, r).unwrap_or(Dim::Dynamic),
            (None, None) => unreachable!(),
        };
        result.push(dim);
    }
    result.reverse();
    result
}

/// Infer the result of a reduction op.
fn infer_reduction(shape: &[Dim], dims: &[i64], keepdim: bool, dtype: DType) -> IrType {
    let rank = shape.len();
    let normalized: Vec<usize> = dims
        .iter()
        .map(|&d| {
            if d < 0 {
                (rank as i64 + d) as usize
            } else {
                d as usize
            }
        })
        .collect();

    if keepdim {
        let new_shape: Vec<Dim> = shape
            .iter()
            .enumerate()
            .map(|(i, d)| {
                if normalized.contains(&i) {
                    Dim::Fixed(1)
                } else {
                    d.clone()
                }
            })
            .collect();
        IrType::Tensor {
            shape: new_shape,
            dtype,
        }
    } else {
        let new_shape: Vec<Dim> = shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !normalized.contains(i))
            .map(|(_, d)| d.clone())
            .collect();
        if new_shape.is_empty() {
            IrType::Scalar(dtype)
        } else {
            IrType::Tensor {
                shape: new_shape,
                dtype,
            }
        }
    }
}

/// Infer concat shape.
fn infer_concat(inputs: &[&IrType], dim: i64) -> Option<IrType> {
    if inputs.is_empty() {
        return None;
    }

    // Use the first input as a template
    if let IrType::Tensor {
        shape: first_shape,
        dtype,
    } = inputs[0]
    {
        let rank = first_shape.len();
        let d = if dim < 0 {
            (rank as i64 + dim) as usize
        } else {
            dim as usize
        };

        if d >= rank {
            return None;
        }

        let mut result_shape = first_shape.clone();

        // Sum the concat dimension across all inputs
        let mut total = dim_value(&first_shape[d]);
        for &input in &inputs[1..] {
            if let IrType::Tensor { shape, .. } = input {
                if shape.len() == rank {
                    total = match (total, dim_value(&shape[d])) {
                        (Some(a), Some(b)) => Some(a + b),
                        _ => None,
                    };
                }
            }
        }

        result_shape[d] = match total {
            Some(n) => Dim::Fixed(n),
            None => Dim::Dynamic,
        };

        Some(IrType::Tensor {
            shape: result_shape,
            dtype: *dtype,
        })
    } else {
        None
    }
}

fn dim_value(dim: &Dim) -> Option<i64> {
    match dim {
        Dim::Fixed(n) => Some(*n),
        _ => None,
    }
}
