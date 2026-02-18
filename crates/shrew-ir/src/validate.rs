// Graph Validation — Checks well-formedness of the IR before optimization
//
// Validation catches structural errors that the parser/lowering cannot detect:
//
//   1. Dangling inputs — node references a NodeId that doesn't exist
//   2. Cycles — the graph must be a DAG
//   3. Duplicate names — every node name must be unique within a graph
//   4. Input/output validity — listed inputs/outputs exist in the graph
//   5. Parameter validity — params reference real nodes with correct types
//   6. Type consistency — binary ops require compatible types
//   7. Program-level — training/inference reference existing graphs

use crate::graph::*;
use std::collections::{HashMap, HashSet};

// Public API

/// Validate an entire IrProgram. Returns all errors found (does not stop at first).
pub fn validate(program: &IrProgram) -> std::result::Result<(), Vec<ValidationError>> {
    let mut errors = Vec::new();

    for graph in &program.graphs {
        validate_graph(graph, &mut errors);
    }

    validate_program_refs(program, &mut errors);

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Validate a single graph. Returns Ok(()) or the list of errors.
pub fn validate_graph_standalone(graph: &IrGraph) -> std::result::Result<(), Vec<ValidationError>> {
    let mut errors = Vec::new();
    validate_graph(graph, &mut errors);
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

// Validation errors

/// A validation error with context.
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Which graph this error belongs to (empty for program-level).
    pub graph: String,
    /// Which node, if applicable.
    pub node: Option<String>,
    /// The error kind.
    pub kind: ValidationErrorKind,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let loc = if let Some(node) = &self.node {
            format!("{}::{}", self.graph, node)
        } else if !self.graph.is_empty() {
            self.graph.clone()
        } else {
            "program".to_string()
        };
        write!(f, "[{loc}] {}", self.kind)
    }
}

/// Specific validation error kinds.
#[derive(Debug, Clone)]
pub enum ValidationErrorKind {
    /// A node references a NodeId that doesn't exist.
    DanglingInput { node_id: NodeId, input_id: NodeId },
    /// The graph contains a cycle.
    CycleDetected,
    /// Two nodes share the same name.
    DuplicateName { name: String },
    /// An input listed in graph.inputs doesn't exist.
    InvalidInput { node_id: NodeId },
    /// An output listed in graph.outputs doesn't exist.
    InvalidOutput { node_id: NodeId },
    /// A parameter references a non-existent node.
    InvalidParamNode { param_name: String, node_id: NodeId },
    /// A parameter doesn't have a Tensor type.
    ParamNotTensor { param_name: String },
    /// Binary op has wrong number of inputs.
    BinaryOpArity { expected: usize, got: usize },
    /// Unary op has wrong number of inputs.
    UnaryOpArity { expected: usize, got: usize },
    /// Type mismatch on binary op inputs.
    TypeMismatch { left: IrType, right: IrType },
    /// Training config references a graph that doesn't exist.
    TrainingGraphNotFound { name: String },
    /// Inference config references a graph that doesn't exist.
    InferenceGraphNotFound { name: String },
    /// Graph has no outputs.
    NoOutputs,
    /// Reduction op with out-of-range dimension.
    InvalidDim { dim: i64, rank: usize },
}

impl std::fmt::Display for ValidationErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DanglingInput { node_id, input_id } => {
                write!(f, "node {node_id} references non-existent input {input_id}")
            }
            Self::CycleDetected => write!(f, "cycle detected — graph is not a DAG"),
            Self::DuplicateName { name } => write!(f, "duplicate node name \"{name}\""),
            Self::InvalidInput { node_id } => write!(f, "graph input {node_id} does not exist"),
            Self::InvalidOutput { node_id } => write!(f, "graph output {node_id} does not exist"),
            Self::InvalidParamNode {
                param_name,
                node_id,
            } => write!(
                f,
                "parameter \"{param_name}\" references non-existent node {node_id}"
            ),
            Self::ParamNotTensor { param_name } => {
                write!(f, "parameter \"{param_name}\" must have Tensor type")
            }
            Self::BinaryOpArity { expected, got } => {
                write!(f, "binary op expects {expected} inputs, got {got}")
            }
            Self::UnaryOpArity { expected, got } => {
                write!(f, "unary op expects {expected} input, got {got}")
            }
            Self::TypeMismatch { left, right } => write!(f, "type mismatch: {left} vs {right}"),
            Self::TrainingGraphNotFound { name } => {
                write!(f, "@training references non-existent graph \"{name}\"")
            }
            Self::InferenceGraphNotFound { name } => {
                write!(f, "@inference references non-existent graph \"{name}\"")
            }
            Self::NoOutputs => write!(f, "graph has no outputs"),
            Self::InvalidDim { dim, rank } => {
                write!(f, "dimension {dim} out of range for rank-{rank} tensor")
            }
        }
    }
}

// Graph-level validation

fn validate_graph(graph: &IrGraph, errors: &mut Vec<ValidationError>) {
    let gname = &graph.name;

    // 1. No outputs check
    if graph.outputs.is_empty() {
        errors.push(ValidationError {
            graph: gname.clone(),
            node: None,
            kind: ValidationErrorKind::NoOutputs,
        });
    }

    // 2. Duplicate node names
    check_duplicate_names(graph, errors);

    // 3. Dangling input references
    let has_dangling = check_dangling_inputs(graph, errors);

    // 4. Graph inputs/outputs reference valid nodes
    check_io_validity(graph, errors);

    // 5. Parameter validity
    check_params(graph, errors);

    // 6. Op arity checks
    check_op_arity(graph, errors);

    // 7. Type consistency on binary ops (skip if dangling inputs)
    if !has_dangling {
        check_type_consistency(graph, errors);
    }

    // 8. Cycle detection (skip if dangling — topo_order would panic)
    if !has_dangling {
        check_acyclic(graph, errors);
    }

    // 9. Dimension bounds (skip if dangling)
    if !has_dangling {
        check_dim_bounds(graph, errors);
    }
}

fn check_duplicate_names(graph: &IrGraph, errors: &mut Vec<ValidationError>) {
    let mut seen: HashMap<&str, usize> = HashMap::new();
    for node in &graph.nodes {
        let count = seen.entry(&node.name).or_insert(0);
        *count += 1;
        if *count == 2 {
            // Report on second occurrence
            errors.push(ValidationError {
                graph: graph.name.clone(),
                node: Some(node.name.clone()),
                kind: ValidationErrorKind::DuplicateName {
                    name: node.name.clone(),
                },
            });
        }
    }
}

fn check_dangling_inputs(graph: &IrGraph, errors: &mut Vec<ValidationError>) -> bool {
    let max_id = graph.nodes.len();
    let mut found = false;
    for node in &graph.nodes {
        for &inp in &node.inputs {
            if inp.0 >= max_id {
                found = true;
                errors.push(ValidationError {
                    graph: graph.name.clone(),
                    node: Some(node.name.clone()),
                    kind: ValidationErrorKind::DanglingInput {
                        node_id: node.id,
                        input_id: inp,
                    },
                });
            }
        }
    }
    found
}

fn check_io_validity(graph: &IrGraph, errors: &mut Vec<ValidationError>) {
    let max_id = graph.nodes.len();
    for &id in &graph.inputs {
        if id.0 >= max_id {
            errors.push(ValidationError {
                graph: graph.name.clone(),
                node: None,
                kind: ValidationErrorKind::InvalidInput { node_id: id },
            });
        }
    }
    for out in &graph.outputs {
        if out.node_id.0 >= max_id {
            errors.push(ValidationError {
                graph: graph.name.clone(),
                node: None,
                kind: ValidationErrorKind::InvalidOutput {
                    node_id: out.node_id,
                },
            });
        }
    }
}

fn check_params(graph: &IrGraph, errors: &mut Vec<ValidationError>) {
    let max_id = graph.nodes.len();
    for param in &graph.params {
        if param.node_id.0 >= max_id {
            errors.push(ValidationError {
                graph: graph.name.clone(),
                node: None,
                kind: ValidationErrorKind::InvalidParamNode {
                    param_name: param.name.clone(),
                    node_id: param.node_id,
                },
            });
            continue;
        }
        // Params should be Tensor type (or Unknown before inference)
        match &param.ty {
            IrType::Tensor { .. } | IrType::Unknown => {}
            _other => {
                errors.push(ValidationError {
                    graph: graph.name.clone(),
                    node: Some(param.name.clone()),
                    kind: ValidationErrorKind::ParamNotTensor {
                        param_name: param.name.clone(),
                    },
                });
            }
        }
    }
}

fn check_op_arity(graph: &IrGraph, errors: &mut Vec<ValidationError>) {
    for node in &graph.nodes {
        let (min, max) = expected_arity(&node.op);
        let got = node.inputs.len();
        if got < min || got > max {
            let kind = if is_binary_like(&node.op) {
                ValidationErrorKind::BinaryOpArity { expected: min, got }
            } else if is_unary_like(&node.op) {
                ValidationErrorKind::UnaryOpArity { expected: min, got }
            } else {
                // For other ops, use binary arity error as generic
                ValidationErrorKind::BinaryOpArity { expected: min, got }
            };
            errors.push(ValidationError {
                graph: graph.name.clone(),
                node: Some(node.name.clone()),
                kind,
            });
        }
    }
}

/// Return (min_inputs, max_inputs) for an op.
fn expected_arity(op: &OpKind) -> (usize, usize) {
    match op {
        // Zero inputs
        OpKind::Constant(_) | OpKind::Range => (0, 2),

        // Exactly one input — unary ops
        OpKind::Neg
        | OpKind::Relu
        | OpKind::Gelu
        | OpKind::Silu
        | OpKind::Sigmoid
        | OpKind::Tanh
        | OpKind::Exp
        | OpKind::Log
        | OpKind::Sqrt
        | OpKind::Transpose
        | OpKind::Not => (1, 1),

        // One-input reductions / shape ops
        OpKind::Sum { .. }
        | OpKind::Mean { .. }
        | OpKind::Max { .. }
        | OpKind::Min { .. }
        | OpKind::Variance { .. } => (1, 1),
        OpKind::Reshape { .. }
        | OpKind::View { .. }
        | OpKind::Permute { .. }
        | OpKind::Expand { .. } => (1, 1),
        OpKind::Softmax { .. } => (1, 1),
        OpKind::Dropout { .. } => (1, 1),

        // Exactly two inputs — binary ops
        OpKind::Add
        | OpKind::Sub
        | OpKind::Mul
        | OpKind::Div
        | OpKind::Mod
        | OpKind::Pow
        | OpKind::MatMul => (2, 2),
        OpKind::Equal
        | OpKind::NotEqual
        | OpKind::Less
        | OpKind::Greater
        | OpKind::LessEqual
        | OpKind::GreaterEqual => (2, 2),
        OpKind::And | OpKind::Or => (2, 2),

        // Normalization: input + weight + bias (2-3)
        OpKind::LayerNorm { .. } | OpKind::BatchNorm { .. } => (1, 3),

        // Embedding: table + indices (2), or just indices (1)
        OpKind::Embedding => (1, 2),

        // Linear: input [+ weight [+ bias]] (1-3)
        OpKind::Linear { .. } => (1, 3),

        // Concat/Split: variable
        OpKind::Concat { .. } => (1, 64),
        OpKind::Split { .. } => (1, 1),

        // Loss functions: predictions + targets
        OpKind::CrossEntropy | OpKind::MseLoss => (2, 2),

        // Attention blocks: variable input
        OpKind::MultiHeadAttention { .. } => (1, 6),
        OpKind::TransformerBlock { .. } => (1, 6),

        // Repeat: the body subgraph input
        OpKind::Repeat { .. } => (1, 64),

        // Identity: exactly one
        OpKind::Identity => (0, 1),

        // Custom/Call: any number
        OpKind::Custom { .. } | OpKind::Call { .. } => (0, 64),
    }
}

fn is_binary_like(op: &OpKind) -> bool {
    matches!(
        op,
        OpKind::Add
            | OpKind::Sub
            | OpKind::Mul
            | OpKind::Div
            | OpKind::Mod
            | OpKind::Pow
            | OpKind::MatMul
            | OpKind::Equal
            | OpKind::NotEqual
            | OpKind::Less
            | OpKind::Greater
            | OpKind::LessEqual
            | OpKind::GreaterEqual
            | OpKind::And
            | OpKind::Or
    )
}

fn is_unary_like(op: &OpKind) -> bool {
    matches!(
        op,
        OpKind::Neg
            | OpKind::Relu
            | OpKind::Gelu
            | OpKind::Silu
            | OpKind::Sigmoid
            | OpKind::Tanh
            | OpKind::Exp
            | OpKind::Log
            | OpKind::Sqrt
            | OpKind::Transpose
            | OpKind::Not
    )
}

fn check_type_consistency(graph: &IrGraph, errors: &mut Vec<ValidationError>) {
    for node in &graph.nodes {
        if !is_binary_like(&node.op) || node.inputs.len() != 2 {
            continue;
        }
        let left_ty = &graph.nodes[node.inputs[0].0].output_type;
        let right_ty = &graph.nodes[node.inputs[1].0].output_type;

        // Skip if either type is Unknown (not yet inferred)
        if matches!(left_ty, IrType::Unknown) || matches!(right_ty, IrType::Unknown) {
            continue;
        }

        // For tensor ops, check dtype compatibility
        if let (IrType::Tensor { dtype: ld, .. }, IrType::Tensor { dtype: rd, .. }) =
            (left_ty, right_ty)
        {
            if ld != rd {
                errors.push(ValidationError {
                    graph: graph.name.clone(),
                    node: Some(node.name.clone()),
                    kind: ValidationErrorKind::TypeMismatch {
                        left: left_ty.clone(),
                        right: right_ty.clone(),
                    },
                });
            }
        }
    }
}

fn check_acyclic(graph: &IrGraph, errors: &mut Vec<ValidationError>) {
    let order = graph.topo_order();
    // If topo_order returns fewer nodes than exist, there's a cycle
    if order.len() < graph.nodes.len() {
        errors.push(ValidationError {
            graph: graph.name.clone(),
            node: None,
            kind: ValidationErrorKind::CycleDetected,
        });
    }
}

fn check_dim_bounds(graph: &IrGraph, errors: &mut Vec<ValidationError>) {
    for node in &graph.nodes {
        // Check softmax dim
        if let OpKind::Softmax { dim } = &node.op {
            if let Some(rank) = output_rank(graph, &node.inputs) {
                if !is_valid_dim(*dim, rank) {
                    errors.push(ValidationError {
                        graph: graph.name.clone(),
                        node: Some(node.name.clone()),
                        kind: ValidationErrorKind::InvalidDim { dim: *dim, rank },
                    });
                }
            }
        }
        // Check reduction dims
        match &node.op {
            OpKind::Sum { dims, .. }
            | OpKind::Mean { dims, .. }
            | OpKind::Variance { dims, .. } => {
                if node.inputs.len() == 1 {
                    if let Some(rank) = node_rank(graph, node.inputs[0]) {
                        for d in dims {
                            if !is_valid_dim(*d, rank) {
                                errors.push(ValidationError {
                                    graph: graph.name.clone(),
                                    node: Some(node.name.clone()),
                                    kind: ValidationErrorKind::InvalidDim { dim: *d, rank },
                                });
                            }
                        }
                    }
                }
            }
            OpKind::Max { dim, .. } | OpKind::Min { dim, .. } => {
                if node.inputs.len() == 1 {
                    if let Some(rank) = node_rank(graph, node.inputs[0]) {
                        if !is_valid_dim(*dim, rank) {
                            errors.push(ValidationError {
                                graph: graph.name.clone(),
                                node: Some(node.name.clone()),
                                kind: ValidationErrorKind::InvalidDim { dim: *dim, rank },
                            });
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

/// Get the rank of a node's output if it's a tensor.
fn node_rank(graph: &IrGraph, id: NodeId) -> Option<usize> {
    match &graph.nodes[id.0].output_type {
        IrType::Tensor { shape, .. } => Some(shape.len()),
        _ => None,
    }
}

/// Get the rank of the first input node.
fn output_rank(graph: &IrGraph, inputs: &[NodeId]) -> Option<usize> {
    inputs.first().and_then(|id| node_rank(graph, *id))
}

/// Check if a dimension index is valid for the given rank.
/// Supports negative indexing (e.g., -1 = last dim).
fn is_valid_dim(dim: i64, rank: usize) -> bool {
    let rank = rank as i64;
    dim >= -rank && dim < rank
}

// Program-level validation

fn validate_program_refs(program: &IrProgram, errors: &mut Vec<ValidationError>) {
    let graph_names: HashSet<&str> = program.graphs.iter().map(|g| g.name.as_str()).collect();

    if let Some(training) = &program.training {
        if !graph_names.contains(training.model_graph.as_str()) {
            errors.push(ValidationError {
                graph: String::new(),
                node: None,
                kind: ValidationErrorKind::TrainingGraphNotFound {
                    name: training.model_graph.clone(),
                },
            });
        }
    }

    if let Some(inference) = &program.inference {
        if !graph_names.contains(inference.model_graph.as_str()) {
            errors.push(ValidationError {
                graph: String::new(),
                node: None,
                kind: ValidationErrorKind::InferenceGraphNotFound {
                    name: inference.model_graph.clone(),
                },
            });
        }
    }
}
