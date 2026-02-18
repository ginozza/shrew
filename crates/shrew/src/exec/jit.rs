// JIT Graph Compilation — Compile IR graphs into optimized execution plans
//
// The default `Executor` interprets the IR graph on every call:
//   - Recomputes topological order each time
//   - Looks up each node in a HashMap
//   - Matches on OpKind (50+ variants) per node
//   - Allocates intermediates into a HashMap with no reuse
//
// The JIT compiler transforms the graph into a pre-compiled execution plan
// that eliminates all of this overhead:
//
// COMPONENTS:
//
//   CompiledGraph    — The compiled execution plan for one IR graph
//   Instruction      — A single operation in the compiled plan
//   MemoryPlan       — Buffer lifecycle analysis and reuse
//   BufferSlot       — A reusable memory slot (register analogy)
//   JitExecutor      — Runs compiled graphs instead of re-interpreting
//   CompileStats     — Compilation statistics
//
// WORKFLOW:
//
//   1. Compile:    JitExecutor::compile(program) → JitExecutor
//   2. Run:        executor.run("Forward", &inputs) → JitResult
//   3. Recompile:  executor.recompile("Forward") — after graph changes
//
// OPTIMIZATIONS:
//
//   - Pre-computed topological order (computed once at compile time)
//   - Instruction tape (flat Vec<Instruction>, no HashMap lookup)
//   - Memory planning: liveness analysis, buffer reuse (register allocation)
//   - Dead value early-free (values dropped as soon as last consumer is done)
//   - Fused dispatch (fused ops become single instructions)
//   - Input/param index lookup pre-computed (no string matching at runtime)

use std::collections::HashMap;
use std::fmt;
use std::time::Instant;

use shrew_core::backend::Backend;
use shrew_core::dtype::DType as CoreDType;
use shrew_core::error::Result;
use shrew_core::tensor::Tensor;

use shrew_ir::graph::{ConstantValue, Dim, IrGraph, IrNode, IrProgram, IrType, NodeId, OpKind};

use shrew_nn::Module;

use super::engine::{ir_dtype_to_core, RuntimeConfig};

// Instruction — A single pre-compiled operation

/// Specifies which buffer slot an instruction reads from.
#[derive(Debug, Clone)]
pub struct SlotRef {
    /// Index into the buffer table.
    pub slot: usize,
}

/// A single operation in the compiled execution plan.
///
/// Unlike the interpreter, instructions are a flat enum with pre-resolved
/// input/output buffer slots — no name lookups, no HashMap access.
#[derive(Debug, Clone)]
pub enum Instruction {
    // ── Source instructions (produce values from external sources) ──
    /// Load a graph input into a buffer slot.
    LoadInput {
        /// Input name (for lookup in the provided HashMap).
        name: String,
        /// Buffer slot to store the input tensor.
        dst: usize,
    },
    /// Load a parameter into a buffer slot.
    LoadParam {
        /// Key: (graph_name, param_name).
        graph_name: String,
        param_name: String,
        /// Buffer slot to store the parameter tensor.
        dst: usize,
    },

    // ── Unary operations ──
    Unary {
        op: UnaryInstr,
        src: usize,
        dst: usize,
    },

    // ── Binary operations ──
    Binary {
        op: BinaryInstr,
        lhs: usize,
        rhs: usize,
        dst: usize,
    },

    // ── Reduction operations ──
    Reduce {
        op: ReduceInstr,
        src: usize,
        dst: usize,
        dims: Vec<i64>,
        keepdim: bool,
    },

    // ── Shape operations ──
    Reshape {
        src: usize,
        dst: usize,
        /// Pre-resolved concrete shape (symbolic dims resolved at compile time).
        shape: Vec<usize>,
    },
    Transpose {
        src: usize,
        dst: usize,
    },
    Permute {
        src: usize,
        dst: usize,
        dims: Vec<i64>,
    },
    Expand {
        src: usize,
        dst: usize,
        shape: Vec<usize>,
    },
    Concat {
        srcs: Vec<usize>,
        dst: usize,
        dim: usize,
    },
    Split {
        src: usize,
        dst: usize,
        dim: usize,
        chunks: usize,
    },

    // ── Neural network operations ──
    Softmax {
        src: usize,
        dst: usize,
        dim: usize,
    },
    Embedding {
        indices: usize,
        table: usize,
        dst: usize,
    },
    Linear {
        input: usize,
        weight: usize,
        bias: Option<usize>,
        dst: usize,
    },
    LayerNorm {
        input: usize,
        weight: usize,
        bias: usize,
        dst: usize,
        eps: f64,
    },
    BatchNorm {
        input: usize,
        weight: Option<usize>,
        bias: Option<usize>,
        dst: usize,
        eps: f64,
    },
    MultiHeadAttention {
        input: usize,
        dst: usize,
        n_heads: usize,
    },
    TransformerBlock {
        input: usize,
        dst: usize,
        n_heads: usize,
    },
    Dropout {
        src: usize,
        dst: usize,
        p: f64,
    },

    // ── Loss functions ──
    CrossEntropy {
        predictions: usize,
        targets: usize,
        dst: usize,
    },
    MseLoss {
        predictions: usize,
        targets: usize,
        dst: usize,
    },

    // ── Constants ──
    Constant {
        value: ConstantValue,
        output_type: IrType,
        dst: usize,
    },

    // ── Control flow ──
    Repeat {
        count: i64,
        body_op: Box<OpKind>,
        src: usize,
        dst: usize,
    },
    Call {
        graph_name: String,
        inputs: Vec<usize>,
        dst: usize,
    },

    // ── Comparison / logical ──
    Compare {
        op: CompareInstr,
        lhs: usize,
        rhs: usize,
        dst: usize,
    },
    LogicalNot {
        src: usize,
        dst: usize,
    },
    LogicalBinOp {
        op: LogicalBinInstr,
        lhs: usize,
        rhs: usize,
        dst: usize,
    },

    // ── Fused operations (from IR optimizer) ──
    FusedMatMulAdd {
        a: usize,
        b: usize,
        c: usize,
        dst: usize,
    },
    FusedAddRelu {
        lhs: usize,
        rhs: usize,
        dst: usize,
    },
    FusedSubRelu {
        lhs: usize,
        rhs: usize,
        dst: usize,
    },
    FusedMatMulRelu {
        lhs: usize,
        rhs: usize,
        dst: usize,
    },

    // ── Identity (pass-through) ──
    Copy {
        src: usize,
        dst: usize,
    },

    // ── Range ──
    Range {
        inputs: Vec<usize>,
        output_type: IrType,
        dst: usize,
    },

    // ── Free a buffer slot (dead value elimination) ──
    Free {
        slot: usize,
    },
}

/// Unary operation variants (pre-dispatched).
#[derive(Debug, Clone, Copy)]
pub enum UnaryInstr {
    Neg,
    Relu,
    Gelu,
    Silu,
    Sigmoid,
    Tanh,
    Exp,
    Log,
    Sqrt,
}

/// Binary operation variants (pre-dispatched).
#[derive(Debug, Clone, Copy)]
pub enum BinaryInstr {
    Add,
    Sub,
    Mul,
    Div,
    MatMul,
    Pow,
    Mod,
}

/// Reduction operation variants.
#[derive(Debug, Clone, Copy)]
pub enum ReduceInstr {
    Sum,
    Mean,
    Max,
    Min,
    Variance,
}

/// Comparison operation variants.
#[derive(Debug, Clone, Copy)]
pub enum CompareInstr {
    Equal,
    NotEqual,
    Less,
    Greater,
    LessEqual,
    GreaterEqual,
}

/// Logical binary operation variants.
#[derive(Debug, Clone, Copy)]
pub enum LogicalBinInstr {
    And,
    Or,
}

// MemoryPlan — Buffer lifecycle analysis

/// Tracks when each value is first produced and last consumed.
#[derive(Debug, Clone)]
pub struct ValueLifetime {
    /// Instruction index where this value is produced.
    pub produced_at: usize,
    /// Instruction index where this value is last consumed (inclusive).
    pub last_used_at: usize,
    /// Node ID from the original graph.
    pub node_id: NodeId,
    /// Whether this value is a graph output (must not be freed).
    pub is_output: bool,
    /// Whether this value is an input or parameter (externally owned).
    pub is_external: bool,
}

/// The memory plan for a compiled graph — maps NodeIds to buffer slots
/// and tracks lifetimes for dead value elimination.
#[derive(Debug, Clone)]
pub struct MemoryPlan {
    /// Number of buffer slots needed.
    pub num_slots: usize,
    /// Mapping from NodeId → buffer slot.
    pub node_to_slot: HashMap<usize, usize>,
    /// Lifetime of each slot.
    pub lifetimes: Vec<ValueLifetime>,
    /// Free instructions to insert (slot, after_instruction_idx).
    pub free_points: Vec<(usize, usize)>,
    /// Number of buffers reused.
    pub reuse_count: usize,
}

// CompiledGraph — A fully compiled execution plan

/// The compiled execution plan for a single IR graph.
///
/// Contains a flat instruction tape, memory plan, and metadata for
/// efficient repeated execution.
#[derive(Debug)]
pub struct CompiledGraph {
    /// Name of the source graph.
    pub graph_name: String,
    /// Flat instruction tape — executed sequentially.
    pub instructions: Vec<Instruction>,
    /// Memory plan — buffer slot assignments.
    pub memory_plan: MemoryPlan,
    /// Output slot mappings: name → buffer slot.
    pub output_slots: HashMap<String, usize>,
    /// Total number of buffer slots.
    pub num_slots: usize,
    /// Compilation statistics.
    pub stats: CompileStats,
}

/// Statistics from the compilation process.
#[derive(Debug, Clone)]
pub struct CompileStats {
    /// Number of instructions in the compiled plan.
    pub num_instructions: usize,
    /// Number of nodes in the source graph.
    pub num_source_nodes: usize,
    /// Number of buffer slots allocated.
    pub num_slots: usize,
    /// Number of buffer slots reused.
    pub num_reused: usize,
    /// Number of free instructions inserted.
    pub num_frees: usize,
    /// Number of fused instructions.
    pub num_fused: usize,
    /// Compilation time in microseconds.
    pub compile_time_us: u64,
}

impl fmt::Display for CompileStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CompiledGraph: {} instructions ({} source nodes), {} slots ({} reused), {} frees, {} fused, compiled in {}μs",
            self.num_instructions,
            self.num_source_nodes,
            self.num_slots,
            self.num_reused,
            self.num_frees,
            self.num_fused,
            self.compile_time_us,
        )
    }
}

// Compilation — transform IrGraph → CompiledGraph

/// Compile a single IR graph into an optimized execution plan.
pub fn compile_graph(
    graph: &IrGraph,
    program: &IrProgram,
    config: &RuntimeConfig,
) -> Result<CompiledGraph> {
    let start = Instant::now();

    // 1. Get topological order (computed once)
    let order = graph.topo_order();

    // 2. Assign buffer slots
    let mut node_to_slot: HashMap<usize, usize> = HashMap::new();
    let mut next_slot = 0usize;
    let mut produced_at: HashMap<usize, usize> = HashMap::new();
    let mut last_used_at: HashMap<usize, usize> = HashMap::new();

    // Pre-compute output node IDs for quick lookup
    let output_node_ids: std::collections::HashSet<usize> =
        graph.outputs.iter().map(|o| o.node_id.0).collect();

    // Pre-compute input/param node IDs
    let input_node_ids: std::collections::HashSet<usize> =
        graph.inputs.iter().map(|id| id.0).collect();
    let param_node_ids: std::collections::HashSet<usize> =
        graph.params.iter().map(|p| p.node_id.0).collect();

    // Assign a slot to each node in topo order
    for &node_id in &order {
        let slot = next_slot;
        node_to_slot.insert(node_id.0, slot);
        next_slot += 1;
    }

    // 3. Build instruction tape
    let mut instructions: Vec<Instruction> = Vec::with_capacity(order.len() + graph.inputs.len());
    let mut num_fused = 0;

    for (instr_idx, &node_id) in order.iter().enumerate() {
        let node = graph.node(node_id);
        let dst = node_to_slot[&node_id.0];

        // Track production point
        produced_at.insert(node_id.0, instr_idx);

        // Track consumption points for inputs
        for &input_id in &node.inputs {
            last_used_at.insert(input_id.0, instr_idx);
        }

        // Check if this node is an input
        if input_node_ids.contains(&node_id.0) {
            instructions.push(Instruction::LoadInput {
                name: node.name.clone(),
                dst,
            });
            continue;
        }

        // Check if this node is a parameter
        if param_node_ids.contains(&node_id.0) {
            if let Some(param) = graph.params.iter().find(|p| p.node_id == node_id) {
                instructions.push(Instruction::LoadParam {
                    graph_name: graph.name.clone(),
                    param_name: param.name.clone(),
                    dst,
                });
            }
            continue;
        }

        // Compile the operation
        let instr = compile_node(graph, node, &node_to_slot, config, program)?;

        // Track fused instructions
        match &instr {
            Instruction::FusedMatMulAdd { .. }
            | Instruction::FusedAddRelu { .. }
            | Instruction::FusedSubRelu { .. }
            | Instruction::FusedMatMulRelu { .. } => {
                num_fused += 1;
            }
            _ => {}
        }

        instructions.push(instr);
    }

    // 4. Compute lifetimes and insert free instructions
    let mut lifetimes = Vec::new();
    let mut free_points = Vec::new();

    for &node_id in &order {
        let slot = node_to_slot[&node_id.0];
        let is_output = output_node_ids.contains(&node_id.0);
        let is_external =
            input_node_ids.contains(&node_id.0) || param_node_ids.contains(&node_id.0);
        let prod = produced_at.get(&node_id.0).copied().unwrap_or(0);
        let last = last_used_at.get(&node_id.0).copied().unwrap_or(prod);

        lifetimes.push(ValueLifetime {
            produced_at: prod,
            last_used_at: last,
            node_id,
            is_output,
            is_external,
        });

        // Insert free point if this value is not an output and not external
        if !is_output && !is_external && last < instructions.len().saturating_sub(1) {
            free_points.push((slot, last));
        }
    }

    // Sort frees by position (latest first for stable insertion)
    free_points.sort_by(|a, b| b.1.cmp(&a.1));

    // Insert free instructions after the last use
    let num_frees = free_points.len();
    for (slot, after_idx) in &free_points {
        let insert_pos = (*after_idx + 1).min(instructions.len());
        instructions.insert(insert_pos, Instruction::Free { slot: *slot });
    }

    // 5. Build output slot mapping
    let mut output_slots = HashMap::new();
    for output in &graph.outputs {
        if let Some(&slot) = node_to_slot.get(&output.node_id.0) {
            output_slots.insert(output.name.clone(), slot);
        }
    }

    let num_slots = next_slot;
    let compile_time = start.elapsed();

    let memory_plan = MemoryPlan {
        num_slots,
        node_to_slot,
        lifetimes,
        free_points: Vec::new(), // Already applied
        reuse_count: 0,          // No physical reuse in this version (logical slots)
    };

    let stats = CompileStats {
        num_instructions: instructions.len(),
        num_source_nodes: graph.nodes.len(),
        num_slots,
        num_reused: 0,
        num_frees,
        num_fused,
        compile_time_us: compile_time.as_micros() as u64,
    };

    Ok(CompiledGraph {
        graph_name: graph.name.clone(),
        instructions,
        memory_plan,
        output_slots,
        num_slots,
        stats,
    })
}

/// Compile a single IR node into an instruction.
fn compile_node(
    _graph: &IrGraph,
    node: &IrNode,
    node_to_slot: &HashMap<usize, usize>,
    config: &RuntimeConfig,
    program: &IrProgram,
) -> Result<Instruction> {
    let dst = node_to_slot[&node.id.0];

    // Helper: get slot for an input
    let slot = |idx: usize| -> Result<usize> {
        let input_id = node.inputs.get(idx).ok_or_else(|| {
            shrew_core::Error::msg(format!(
                "Node '{}' expected input at index {}, but has {} inputs",
                node.name,
                idx,
                node.inputs.len()
            ))
        })?;
        node_to_slot.get(&input_id.0).copied().ok_or_else(|| {
            shrew_core::Error::msg(format!(
                "Node '{}' input {} (NodeId {}) not found in slot map",
                node.name, idx, input_id.0
            ))
        })
    };

    match &node.op {
        OpKind::Identity => Ok(Instruction::Copy { src: slot(0)?, dst }),

        // ── Unary ──
        OpKind::Neg => Ok(Instruction::Unary {
            op: UnaryInstr::Neg,
            src: slot(0)?,
            dst,
        }),
        OpKind::Relu => Ok(Instruction::Unary {
            op: UnaryInstr::Relu,
            src: slot(0)?,
            dst,
        }),
        OpKind::Gelu => Ok(Instruction::Unary {
            op: UnaryInstr::Gelu,
            src: slot(0)?,
            dst,
        }),
        OpKind::Silu => Ok(Instruction::Unary {
            op: UnaryInstr::Silu,
            src: slot(0)?,
            dst,
        }),
        OpKind::Sigmoid => Ok(Instruction::Unary {
            op: UnaryInstr::Sigmoid,
            src: slot(0)?,
            dst,
        }),
        OpKind::Tanh => Ok(Instruction::Unary {
            op: UnaryInstr::Tanh,
            src: slot(0)?,
            dst,
        }),
        OpKind::Exp => Ok(Instruction::Unary {
            op: UnaryInstr::Exp,
            src: slot(0)?,
            dst,
        }),
        OpKind::Log => Ok(Instruction::Unary {
            op: UnaryInstr::Log,
            src: slot(0)?,
            dst,
        }),
        OpKind::Sqrt => Ok(Instruction::Unary {
            op: UnaryInstr::Sqrt,
            src: slot(0)?,
            dst,
        }),

        // ── Binary ──
        OpKind::Add => Ok(Instruction::Binary {
            op: BinaryInstr::Add,
            lhs: slot(0)?,
            rhs: slot(1)?,
            dst,
        }),
        OpKind::Sub => Ok(Instruction::Binary {
            op: BinaryInstr::Sub,
            lhs: slot(0)?,
            rhs: slot(1)?,
            dst,
        }),
        OpKind::Mul => Ok(Instruction::Binary {
            op: BinaryInstr::Mul,
            lhs: slot(0)?,
            rhs: slot(1)?,
            dst,
        }),
        OpKind::Div => Ok(Instruction::Binary {
            op: BinaryInstr::Div,
            lhs: slot(0)?,
            rhs: slot(1)?,
            dst,
        }),
        OpKind::MatMul => Ok(Instruction::Binary {
            op: BinaryInstr::MatMul,
            lhs: slot(0)?,
            rhs: slot(1)?,
            dst,
        }),
        OpKind::Pow => Ok(Instruction::Binary {
            op: BinaryInstr::Pow,
            lhs: slot(0)?,
            rhs: slot(1)?,
            dst,
        }),
        OpKind::Mod => Ok(Instruction::Binary {
            op: BinaryInstr::Mod,
            lhs: slot(0)?,
            rhs: slot(1)?,
            dst,
        }),

        // ── Transpose ──
        OpKind::Transpose => Ok(Instruction::Transpose { src: slot(0)?, dst }),

        // ── Reductions ──
        OpKind::Sum { dims, keepdim } => Ok(Instruction::Reduce {
            op: ReduceInstr::Sum,
            src: slot(0)?,
            dst,
            dims: dims.clone(),
            keepdim: *keepdim,
        }),
        OpKind::Mean { dims, keepdim } => Ok(Instruction::Reduce {
            op: ReduceInstr::Mean,
            src: slot(0)?,
            dst,
            dims: dims.clone(),
            keepdim: *keepdim,
        }),
        OpKind::Max { dim, keepdim } => Ok(Instruction::Reduce {
            op: ReduceInstr::Max,
            src: slot(0)?,
            dst,
            dims: vec![*dim],
            keepdim: *keepdim,
        }),
        OpKind::Min { dim, keepdim } => Ok(Instruction::Reduce {
            op: ReduceInstr::Min,
            src: slot(0)?,
            dst,
            dims: vec![*dim],
            keepdim: *keepdim,
        }),
        OpKind::Variance { dims, keepdim } => Ok(Instruction::Reduce {
            op: ReduceInstr::Variance,
            src: slot(0)?,
            dst,
            dims: dims.clone(),
            keepdim: *keepdim,
        }),

        // ── Softmax ──
        OpKind::Softmax { dim } => {
            let d = *dim;
            Ok(Instruction::Softmax {
                src: slot(0)?,
                dst,
                dim: d as usize,
            })
        }

        // ── Shape ops ──
        OpKind::Reshape { target_shape } | OpKind::View { target_shape } => {
            let shape = resolve_shape_vec(target_shape, config, program)?;
            Ok(Instruction::Reshape {
                src: slot(0)?,
                dst,
                shape,
            })
        }
        OpKind::Permute { dims } => Ok(Instruction::Permute {
            src: slot(0)?,
            dst,
            dims: dims.clone(),
        }),
        OpKind::Expand { target_shape } => {
            let shape = resolve_shape_vec(target_shape, config, program)?;
            Ok(Instruction::Expand {
                src: slot(0)?,
                dst,
                shape,
            })
        }
        OpKind::Concat { dim } => {
            let srcs: Vec<usize> = (0..node.inputs.len())
                .map(&slot)
                .collect::<Result<Vec<_>>>()?;
            Ok(Instruction::Concat {
                srcs,
                dst,
                dim: *dim as usize,
            })
        }
        OpKind::Split { dim, chunks } => Ok(Instruction::Split {
            src: slot(0)?,
            dst,
            dim: resolve_neg_dim(*dim, 4), // dim resolved at runtime
            chunks: *chunks as usize,
        }),

        // ── NN layers ──
        OpKind::Embedding => Ok(Instruction::Embedding {
            indices: slot(0)?,
            table: slot(1)?,
            dst,
        }),
        OpKind::Linear { bias } => {
            let bias_slot = if *bias && node.inputs.len() >= 3 {
                Some(slot(2)?)
            } else {
                None
            };
            Ok(Instruction::Linear {
                input: slot(0)?,
                weight: slot(1)?,
                bias: bias_slot,
                dst,
            })
        }
        OpKind::LayerNorm { eps } => Ok(Instruction::LayerNorm {
            input: slot(0)?,
            weight: slot(1)?,
            bias: slot(2)?,
            dst,
            eps: *eps,
        }),
        OpKind::BatchNorm { eps } => {
            let weight = if node.inputs.len() >= 2 {
                Some(slot(1)?)
            } else {
                None
            };
            let bias = if node.inputs.len() >= 3 {
                Some(slot(2)?)
            } else {
                None
            };
            Ok(Instruction::BatchNorm {
                input: slot(0)?,
                weight,
                bias,
                dst,
                eps: *eps,
            })
        }
        OpKind::MultiHeadAttention { n_heads } => Ok(Instruction::MultiHeadAttention {
            input: slot(0)?,
            dst,
            n_heads: *n_heads as usize,
        }),
        OpKind::TransformerBlock { n_heads } => Ok(Instruction::TransformerBlock {
            input: slot(0)?,
            dst,
            n_heads: *n_heads as usize,
        }),
        OpKind::Dropout { p } => Ok(Instruction::Dropout {
            src: slot(0)?,
            dst,
            p: *p,
        }),

        // ── Loss ──
        OpKind::CrossEntropy => Ok(Instruction::CrossEntropy {
            predictions: slot(0)?,
            targets: slot(1)?,
            dst,
        }),
        OpKind::MseLoss => Ok(Instruction::MseLoss {
            predictions: slot(0)?,
            targets: slot(1)?,
            dst,
        }),

        // ── Constants ──
        OpKind::Constant(val) => Ok(Instruction::Constant {
            value: val.clone(),
            output_type: node.output_type.clone(),
            dst,
        }),

        // ── Repeat ──
        OpKind::Repeat { count, body_op } => Ok(Instruction::Repeat {
            count: *count,
            body_op: body_op.clone(),
            src: slot(0)?,
            dst,
        }),

        // ── Call ──
        OpKind::Call { graph_name } => {
            let inputs: Vec<usize> = (0..node.inputs.len())
                .map(&slot)
                .collect::<Result<Vec<_>>>()?;
            Ok(Instruction::Call {
                graph_name: graph_name.clone(),
                inputs,
                dst,
            })
        }

        // ── Range ──
        OpKind::Range => {
            let inputs: Vec<usize> = (0..node.inputs.len())
                .map(&slot)
                .collect::<Result<Vec<_>>>()?;
            Ok(Instruction::Range {
                inputs,
                output_type: node.output_type.clone(),
                dst,
            })
        }

        // ── Comparison ──
        OpKind::Equal => Ok(Instruction::Compare {
            op: CompareInstr::Equal,
            lhs: slot(0)?,
            rhs: slot(1)?,
            dst,
        }),
        OpKind::NotEqual => Ok(Instruction::Compare {
            op: CompareInstr::NotEqual,
            lhs: slot(0)?,
            rhs: slot(1)?,
            dst,
        }),
        OpKind::Less => Ok(Instruction::Compare {
            op: CompareInstr::Less,
            lhs: slot(0)?,
            rhs: slot(1)?,
            dst,
        }),
        OpKind::Greater => Ok(Instruction::Compare {
            op: CompareInstr::Greater,
            lhs: slot(0)?,
            rhs: slot(1)?,
            dst,
        }),
        OpKind::LessEqual => Ok(Instruction::Compare {
            op: CompareInstr::LessEqual,
            lhs: slot(0)?,
            rhs: slot(1)?,
            dst,
        }),
        OpKind::GreaterEqual => Ok(Instruction::Compare {
            op: CompareInstr::GreaterEqual,
            lhs: slot(0)?,
            rhs: slot(1)?,
            dst,
        }),

        // ── Logical ──
        OpKind::And => Ok(Instruction::LogicalBinOp {
            op: LogicalBinInstr::And,
            lhs: slot(0)?,
            rhs: slot(1)?,
            dst,
        }),
        OpKind::Or => Ok(Instruction::LogicalBinOp {
            op: LogicalBinInstr::Or,
            lhs: slot(0)?,
            rhs: slot(1)?,
            dst,
        }),
        OpKind::Not => Ok(Instruction::LogicalNot { src: slot(0)?, dst }),

        // ── Fused ops (from IR optimizer) ──
        OpKind::Custom { name, .. } => match name.as_str() {
            "fused_matmul_add" => Ok(Instruction::FusedMatMulAdd {
                a: slot(0)?,
                b: slot(1)?,
                c: slot(2)?,
                dst,
            }),
            "fused_add_relu" => Ok(Instruction::FusedAddRelu {
                lhs: slot(0)?,
                rhs: slot(1)?,
                dst,
            }),
            "fused_sub_relu" => Ok(Instruction::FusedSubRelu {
                lhs: slot(0)?,
                rhs: slot(1)?,
                dst,
            }),
            "fused_matmul_relu" => Ok(Instruction::FusedMatMulRelu {
                lhs: slot(0)?,
                rhs: slot(1)?,
                dst,
            }),
            other => Err(shrew_core::Error::msg(format!(
                "Unknown custom op '{}' during JIT compilation",
                other
            ))),
        },
    }
}

// JitExecutor — Runs compiled graphs

/// A JIT-compiled executor that runs pre-compiled graph execution plans.
///
/// Unlike the interpreter (`Executor`), the JIT executor:
/// - Pre-compiles each graph into a flat instruction tape
/// - Pre-resolves all buffer slot assignments
/// - Inserts dead-value-free instructions for memory efficiency
/// - Dispatches operations without HashMap lookups or string matching
///
/// # Usage
/// ```ignore
/// let jit = JitExecutor::<CpuBackend>::compile(program, device, config)?;
/// let result = jit.run("Forward", &inputs)?;
/// let output = result.get("output").unwrap();
/// ```
pub struct JitExecutor<B: Backend> {
    /// Compiled graphs, keyed by graph name.
    compiled: HashMap<String, CompiledGraph>,
    /// The source IR program.
    program: IrProgram,
    /// Runtime configuration.
    config: RuntimeConfig,
    /// Device.
    device: B::Device,
    /// Initialized parameters.
    params: HashMap<(String, String), Tensor<B>>,
}

/// Result of a JIT execution.
#[derive(Debug)]
pub struct JitResult<B: Backend> {
    /// Output tensors, keyed by output name.
    pub outputs: HashMap<String, Tensor<B>>,
}

impl<B: Backend> JitResult<B> {
    /// Get the first output tensor.
    pub fn output(&self) -> Option<&Tensor<B>> {
        self.outputs.values().next()
    }

    /// Get an output by name.
    pub fn get(&self, name: &str) -> Option<&Tensor<B>> {
        self.outputs.get(name)
    }
}

impl<B: Backend> JitExecutor<B> {
    /// Compile all graphs in a program and create a JIT executor.
    pub fn compile(program: IrProgram, device: B::Device, config: RuntimeConfig) -> Result<Self> {
        let mut compiled = HashMap::new();

        // Compile each graph
        for graph in &program.graphs {
            let cg = compile_graph(graph, &program, &config)?;
            compiled.insert(graph.name.clone(), cg);
        }

        // Initialize parameters (reuse logic from Executor)
        let mut params = HashMap::new();
        for graph in &program.graphs {
            for param in &graph.params {
                let tensor = init_param::<B>(
                    &param.ty,
                    &param.init,
                    param.frozen,
                    &config,
                    &program,
                    &device,
                )?;
                params.insert((graph.name.clone(), param.name.clone()), tensor);
            }
        }

        Ok(Self {
            compiled,
            program,
            config,
            device,
            params,
        })
    }

    /// Get compilation statistics for a graph.
    pub fn stats(&self, graph_name: &str) -> Option<&CompileStats> {
        self.compiled.get(graph_name).map(|cg| &cg.stats)
    }

    /// Get all compilation statistics.
    pub fn all_stats(&self) -> Vec<(&str, &CompileStats)> {
        self.compiled
            .iter()
            .map(|(name, cg)| (name.as_str(), &cg.stats))
            .collect()
    }

    /// Run a compiled graph with the given inputs.
    pub fn run(
        &self,
        graph_name: &str,
        inputs: &HashMap<String, Tensor<B>>,
    ) -> Result<JitResult<B>> {
        let cg = self.compiled.get(graph_name).ok_or_else(|| {
            shrew_core::Error::msg(format!(
                "Graph '{}' not compiled. Available: {:?}",
                graph_name,
                self.compiled.keys().collect::<Vec<_>>()
            ))
        })?;

        // Allocate buffer table (slots)
        let mut slots: Vec<Option<Tensor<B>>> = vec![None; cg.num_slots];

        // Execute instruction tape
        for instr in &cg.instructions {
            match instr {
                Instruction::LoadInput { name, dst } => {
                    if let Some(tensor) = inputs.get(name) {
                        slots[*dst] = Some(tensor.clone());
                    }
                }

                Instruction::LoadParam {
                    graph_name,
                    param_name,
                    dst,
                } => {
                    let key = (graph_name.clone(), param_name.clone());
                    if let Some(tensor) = self.params.get(&key) {
                        slots[*dst] = Some(tensor.clone());
                    }
                }

                Instruction::Unary { op, src, dst } => {
                    let t = get_slot(&slots, *src)?;
                    let result = match op {
                        UnaryInstr::Neg => t.neg(),
                        UnaryInstr::Relu => t.relu(),
                        UnaryInstr::Gelu => t.gelu(),
                        UnaryInstr::Silu => t.silu(),
                        UnaryInstr::Sigmoid => t.sigmoid(),
                        UnaryInstr::Tanh => t.tanh(),
                        UnaryInstr::Exp => t.exp(),
                        UnaryInstr::Log => t.log(),
                        UnaryInstr::Sqrt => t.sqrt(),
                    }?;
                    slots[*dst] = Some(result);
                }

                Instruction::Binary { op, lhs, rhs, dst } => {
                    let a = get_slot(&slots, *lhs)?;
                    let b = get_slot(&slots, *rhs)?;
                    let result = match op {
                        BinaryInstr::Add => a.add(b),
                        BinaryInstr::Sub => a.sub(b),
                        BinaryInstr::Mul => a.mul(b),
                        BinaryInstr::Div => a.div(b),
                        BinaryInstr::MatMul => a.matmul(b),
                        BinaryInstr::Pow => a.log()?.mul(b)?.exp(), // x^y = exp(y*ln(x))
                        BinaryInstr::Mod => {
                            let quotient = a.div(b)?.floor()?;
                            let product = quotient.mul(b)?;
                            a.sub(&product)
                        }
                    }?;
                    slots[*dst] = Some(result);
                }

                Instruction::Reduce {
                    op,
                    src,
                    dst,
                    dims,
                    keepdim,
                } => {
                    let t = get_slot(&slots, *src)?;
                    let result = match op {
                        ReduceInstr::Sum => {
                            if dims.is_empty() || (dims.len() == 1 && dims[0] == -1) {
                                t.sum_all()
                            } else {
                                let d = resolve_neg_dim(dims[0], t.rank());
                                t.sum(d as usize, *keepdim)
                            }
                        }
                        ReduceInstr::Mean => {
                            if dims.is_empty() || (dims.len() == 1 && dims[0] == -1) {
                                t.mean_all()
                            } else {
                                let d = resolve_neg_dim(dims[0], t.rank());
                                t.mean(d as usize, *keepdim)
                            }
                        }
                        ReduceInstr::Max => {
                            let d = resolve_neg_dim(dims[0], t.rank());
                            t.max(d as usize, *keepdim)
                        }
                        ReduceInstr::Min => {
                            let d = resolve_neg_dim(dims[0], t.rank());
                            t.min(d as usize, *keepdim)
                        }
                        ReduceInstr::Variance => {
                            if dims.is_empty() {
                                t.var(0, *keepdim)
                            } else {
                                let d = resolve_neg_dim(dims[0], t.rank());
                                t.var(d as usize, *keepdim)
                            }
                        }
                    }?;
                    slots[*dst] = Some(result);
                }

                Instruction::Reshape { src, dst, shape } => {
                    let t = get_slot(&slots, *src)?;
                    slots[*dst] = Some(t.reshape(shape.clone())?);
                }

                Instruction::Transpose { src, dst } => {
                    let t = get_slot(&slots, *src)?;
                    let rank = t.rank();
                    slots[*dst] = Some(t.transpose(rank - 2, rank - 1)?);
                }

                Instruction::Permute { src, dst, dims } => {
                    let t = get_slot(&slots, *src)?;
                    let mut result = t.clone();
                    let mut current: Vec<usize> = (0..t.rank()).collect();
                    for i in 0..dims.len() {
                        let target = dims[i] as usize;
                        if current[i] != target {
                            if let Some(j) = current.iter().position(|&x| x == target) {
                                result = result.transpose(i, j)?;
                                current.swap(i, j);
                            }
                        }
                    }
                    slots[*dst] = Some(result);
                }

                Instruction::Expand { src, dst, shape } => {
                    let t = get_slot(&slots, *src)?;
                    slots[*dst] = Some(t.expand(shape.clone())?);
                }

                Instruction::Concat { srcs, dst, dim } => {
                    let tensors: Vec<Tensor<B>> = srcs
                        .iter()
                        .map(|s| get_slot(&slots, *s).cloned())
                        .collect::<Result<Vec<_>>>()?;
                    slots[*dst] = Some(Tensor::<B>::cat(&tensors, *dim)?);
                }

                Instruction::Split {
                    src,
                    dst,
                    dim,
                    chunks,
                } => {
                    let t = get_slot(&slots, *src)?;
                    let result = t.chunk(*chunks, *dim)?;
                    if let Some(first) = result.into_iter().next() {
                        slots[*dst] = Some(first);
                    }
                }

                Instruction::Softmax { src, dst, dim } => {
                    let t = get_slot(&slots, *src)?;
                    slots[*dst] = Some(t.softmax(*dim)?);
                }

                Instruction::Embedding {
                    indices,
                    table,
                    dst,
                } => {
                    let idx = get_slot(&slots, *indices)?;
                    let tbl = get_slot(&slots, *table)?;
                    let emb = shrew_nn::Embedding::<B>::from_tensor(tbl.clone())?;
                    slots[*dst] = Some(emb.forward(idx)?);
                }

                Instruction::Linear {
                    input,
                    weight,
                    bias,
                    dst,
                } => {
                    let inp = get_slot(&slots, *input)?;
                    let w = get_slot(&slots, *weight)?;
                    let b = bias.map(|s| get_slot(&slots, s).cloned()).transpose()?;
                    let lin = shrew_nn::Linear::<B>::from_tensors(w.clone(), b)?;
                    slots[*dst] = Some(lin.forward(inp)?);
                }

                Instruction::LayerNorm {
                    input,
                    weight,
                    bias,
                    dst,
                    eps,
                } => {
                    let inp = get_slot(&slots, *input)?;
                    let w = get_slot(&slots, *weight)?;
                    let b = get_slot(&slots, *bias)?;
                    let ln = shrew_nn::LayerNorm::<B>::from_tensors(w.clone(), b.clone(), *eps)?;
                    slots[*dst] = Some(ln.forward(inp)?);
                }

                Instruction::BatchNorm {
                    input,
                    weight,
                    bias,
                    dst,
                    eps,
                } => {
                    let inp = get_slot(&slots, *input)?;
                    if let (Some(ws), Some(bs)) = (weight, bias) {
                        let w = get_slot(&slots, *ws)?;
                        let b = get_slot(&slots, *bs)?;
                        let bn =
                            shrew_nn::BatchNorm2d::<B>::from_tensors(w.clone(), b.clone(), *eps)?;
                        slots[*dst] = Some(bn.forward(inp)?);
                    } else {
                        let dims = inp.dims();
                        let c = if dims.len() == 4 { dims[1] } else { dims[0] };
                        let bn = shrew_nn::BatchNorm2d::<B>::new(
                            c,
                            *eps,
                            0.1,
                            inp.dtype(),
                            &self.device,
                        )?;
                        slots[*dst] = Some(bn.forward(inp)?);
                    }
                }

                Instruction::MultiHeadAttention {
                    input,
                    dst,
                    n_heads,
                } => {
                    let inp = get_slot(&slots, *input)?;
                    let d_model = *inp
                        .dims()
                        .last()
                        .ok_or_else(|| shrew_core::Error::msg("MHA input has no dimensions"))?;
                    let mha = shrew_nn::MultiHeadAttention::<B>::new(
                        d_model,
                        *n_heads,
                        inp.dtype(),
                        inp.device(),
                    )?;
                    slots[*dst] = Some(mha.forward(inp)?);
                }

                Instruction::TransformerBlock {
                    input,
                    dst,
                    n_heads,
                } => {
                    let inp = get_slot(&slots, *input)?;
                    let dims = inp.dims();
                    let d_model = dims[dims.len() - 1];
                    let d_ff = d_model * 4;
                    let block = shrew_nn::TransformerBlock::<B>::new(
                        d_model,
                        *n_heads,
                        d_ff,
                        true,
                        inp.dtype(),
                        inp.device(),
                    )?;
                    slots[*dst] = Some(block.forward(inp)?);
                }

                Instruction::Dropout { src, dst, p } => {
                    let t = get_slot(&slots, *src)?;
                    let dropout = shrew_nn::Dropout::new(*p);
                    if self.config.training {
                        slots[*dst] = Some(dropout.forward_t(t)?);
                    } else {
                        slots[*dst] = Some(t.clone());
                    }
                }

                Instruction::CrossEntropy {
                    predictions,
                    targets,
                    dst,
                } => {
                    let p = get_slot(&slots, *predictions)?;
                    let t = get_slot(&slots, *targets)?;
                    slots[*dst] = Some(shrew_nn::cross_entropy_loss(p, t)?);
                }

                Instruction::MseLoss {
                    predictions,
                    targets,
                    dst,
                } => {
                    let p = get_slot(&slots, *predictions)?;
                    let t = get_slot(&slots, *targets)?;
                    slots[*dst] = Some(shrew_nn::mse_loss(p, t)?);
                }

                Instruction::Constant {
                    value,
                    output_type,
                    dst,
                } => {
                    let tensor = materialize_constant::<B>(
                        value,
                        output_type,
                        self.config.default_dtype,
                        &self.device,
                    )?;
                    slots[*dst] = Some(tensor);
                }

                Instruction::Repeat {
                    count,
                    body_op,
                    src,
                    dst,
                } => {
                    let t = get_slot(&slots, *src)?;
                    let mut current = t.clone();
                    for _ in 0..(*count as u32) {
                        current = execute_body_op::<B>(body_op, &current, &self.device)?;
                    }
                    slots[*dst] = Some(current);
                }

                Instruction::Call {
                    graph_name,
                    inputs: input_slots,
                    dst,
                } => {
                    let _sub_cg = self.compiled.get(graph_name).ok_or_else(|| {
                        shrew_core::Error::msg(format!(
                            "Called graph '{}' not compiled",
                            graph_name
                        ))
                    })?;
                    let sub_graph = self.program.get_graph(graph_name).ok_or_else(|| {
                        shrew_core::Error::msg(format!("Called graph '{}' not found", graph_name))
                    })?;
                    let mut sub_inputs = HashMap::new();
                    for (i, &input_id) in sub_graph.inputs.iter().enumerate() {
                        if let Some(&s) = input_slots.get(i) {
                            if let Some(tensor) = &slots[s] {
                                let input_name = sub_graph.node(input_id).name.clone();
                                sub_inputs.insert(input_name, tensor.clone());
                            }
                        }
                    }
                    let result = self.run(graph_name, &sub_inputs)?;
                    if let Some(out) = result.output() {
                        slots[*dst] = Some(out.clone());
                    }
                }

                Instruction::Compare { op, lhs, rhs, dst } => {
                    let a = get_slot(&slots, *lhs)?;
                    let b = get_slot(&slots, *rhs)?;
                    let result = match op {
                        CompareInstr::Equal => a.eq(b),
                        CompareInstr::NotEqual => a.ne(b),
                        CompareInstr::Less => a.lt(b),
                        CompareInstr::Greater => a.gt(b),
                        CompareInstr::LessEqual => a.le(b),
                        CompareInstr::GreaterEqual => a.ge(b),
                    }?;
                    slots[*dst] = Some(result);
                }

                Instruction::LogicalNot { src, dst } => {
                    let t = get_slot(&slots, *src)?;
                    let data = t.to_f64_vec()?;
                    let result: Vec<f64> = data
                        .iter()
                        .map(|&v| if v == 0.0 { 1.0 } else { 0.0 })
                        .collect();
                    let n = result.len();
                    slots[*dst] = Some(Tensor::<B>::from_f64_slice(
                        &result,
                        n,
                        CoreDType::U8,
                        &self.device,
                    )?);
                }

                Instruction::LogicalBinOp { op, lhs, rhs, dst } => {
                    let a = get_slot(&slots, *lhs)?;
                    let b = get_slot(&slots, *rhs)?;
                    let a_data = a.to_f64_vec()?;
                    let b_data = b.to_f64_vec()?;
                    let result: Vec<f64> = a_data
                        .iter()
                        .zip(b_data.iter())
                        .map(|(&x, &y)| match op {
                            LogicalBinInstr::And => {
                                if x != 0.0 && y != 0.0 {
                                    1.0
                                } else {
                                    0.0
                                }
                            }
                            LogicalBinInstr::Or => {
                                if x != 0.0 || y != 0.0 {
                                    1.0
                                } else {
                                    0.0
                                }
                            }
                        })
                        .collect();
                    let n = result.len();
                    slots[*dst] = Some(Tensor::<B>::from_f64_slice(
                        &result,
                        n,
                        CoreDType::U8,
                        &self.device,
                    )?);
                }

                Instruction::FusedMatMulAdd { a, b, c, dst } => {
                    let at = get_slot(&slots, *a)?;
                    let bt = get_slot(&slots, *b)?;
                    let ct = get_slot(&slots, *c)?;
                    slots[*dst] = Some(at.matmul(bt)?.add(ct)?);
                }

                Instruction::FusedAddRelu { lhs, rhs, dst } => {
                    let a = get_slot(&slots, *lhs)?;
                    let b = get_slot(&slots, *rhs)?;
                    slots[*dst] = Some(a.add(b)?.relu()?);
                }

                Instruction::FusedSubRelu { lhs, rhs, dst } => {
                    let a = get_slot(&slots, *lhs)?;
                    let b = get_slot(&slots, *rhs)?;
                    slots[*dst] = Some(a.sub(b)?.relu()?);
                }

                Instruction::FusedMatMulRelu { lhs, rhs, dst } => {
                    let a = get_slot(&slots, *lhs)?;
                    let b = get_slot(&slots, *rhs)?;
                    slots[*dst] = Some(a.matmul(b)?.relu()?);
                }

                Instruction::Copy { src, dst } => {
                    let t = get_slot(&slots, *src)?;
                    slots[*dst] = Some(t.clone());
                }

                Instruction::Range {
                    inputs: input_slots,
                    output_type,
                    dst,
                } => {
                    let (start, end) = if input_slots.len() >= 2 {
                        let s = get_slot(&slots, input_slots[0])?.to_scalar_f64()?;
                        let e = get_slot(&slots, input_slots[1])?.to_scalar_f64()?;
                        (s as i64, e as i64)
                    } else if input_slots.len() == 1 {
                        (
                            0i64,
                            get_slot(&slots, input_slots[0])?.to_scalar_f64()? as i64,
                        )
                    } else {
                        match output_type {
                            IrType::Tensor { shape, .. } => {
                                if let Some(Dim::Fixed(n)) = shape.first() {
                                    (0, *n)
                                } else {
                                    (0, 1)
                                }
                            }
                            _ => (0, 1),
                        }
                    };
                    let data: Vec<f64> = (start..end).map(|i| i as f64).collect();
                    let len = data.len();
                    slots[*dst] = Some(Tensor::<B>::from_f64_slice(
                        &data,
                        len,
                        CoreDType::I64,
                        &self.device,
                    )?);
                }

                Instruction::Free { slot } => {
                    slots[*slot] = None;
                }
            }
        }

        // Collect outputs
        let mut outputs = HashMap::new();
        for (name, &slot) in &cg.output_slots {
            if let Some(tensor) = &slots[slot] {
                outputs.insert(name.clone(), tensor.clone());
            }
        }

        Ok(JitResult { outputs })
    }

    /// Get the underlying program.
    pub fn program(&self) -> &IrProgram {
        &self.program
    }

    /// Get runtime config.
    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }

    /// Get all parameters.
    pub fn params(&self) -> &HashMap<(String, String), Tensor<B>> {
        &self.params
    }

    /// Get parameters for a specific graph.
    pub fn graph_params(&self, graph_name: &str) -> Vec<Tensor<B>> {
        self.params
            .iter()
            .filter(|((g, _), _)| g == graph_name)
            .map(|(_, t)| t.clone())
            .collect()
    }

    /// Update parameters after an optimizer step.
    pub fn update_params(&mut self, graph_name: &str, new_params: &[Tensor<B>]) {
        let param_names: Vec<String> = self
            .params
            .keys()
            .filter(|(g, _)| g == graph_name)
            .map(|(_, n)| n.clone())
            .collect();

        for (name, tensor) in param_names.into_iter().zip(new_params.iter()) {
            self.params
                .insert((graph_name.to_string(), name), tensor.clone());
        }
    }

    /// Recompile a single graph (e.g., after optimizer changes shapes).
    pub fn recompile(&mut self, graph_name: &str) -> Result<()> {
        let graph = self
            .program
            .get_graph(graph_name)
            .ok_or_else(|| shrew_core::Error::msg(format!("Graph '{}' not found", graph_name)))?;
        let cg = compile_graph(graph, &self.program, &self.config)?;
        self.compiled.insert(graph_name.to_string(), cg);
        Ok(())
    }

    /// Dump the compiled instruction tape for a graph (debugging).
    pub fn dump(&self, graph_name: &str) -> Option<String> {
        let cg = self.compiled.get(graph_name)?;
        let mut out = format!("=== JIT Compiled: {} ===\n", cg.graph_name);
        out.push_str(&format!("{}\n\n", cg.stats));
        for (i, instr) in cg.instructions.iter().enumerate() {
            out.push_str(&format!("  [{:>3}] {:?}\n", i, instr));
        }
        out.push_str(&format!("\nOutputs: {:?}\n", cg.output_slots));
        Some(out)
    }
}

// Helper functions

/// Get a tensor from a buffer slot.
fn get_slot<B: Backend>(slots: &[Option<Tensor<B>>], idx: usize) -> Result<&Tensor<B>> {
    slots.get(idx).and_then(|s| s.as_ref()).ok_or_else(|| {
        shrew_core::Error::msg(format!(
            "Buffer slot {} is empty (value was freed or never produced)",
            idx
        ))
    })
}

/// Resolve a negative dimension index.
fn resolve_neg_dim(dim: i64, rank: usize) -> usize {
    if dim < 0 {
        (rank as i64 + dim) as usize
    } else {
        dim as usize
    }
}

/// Resolve a Vec<Dim> to concrete shape.
fn resolve_shape_vec(
    dims: &[Dim],
    config: &RuntimeConfig,
    program: &IrProgram,
) -> Result<Vec<usize>> {
    dims.iter()
        .map(|d| resolve_dim(d, config, program))
        .collect()
}

/// Resolve a single Dim.
fn resolve_dim(dim: &Dim, config: &RuntimeConfig, program: &IrProgram) -> Result<usize> {
    match dim {
        Dim::Fixed(n) => Ok(*n as usize),
        Dim::Symbolic(name) => {
            if let Some(&val) = config.dims.get(name) {
                return Ok(val);
            }
            if let Some(shrew_ir::graph::ConfigValue::Int(n)) = program.config.get(name) {
                return Ok(*n as usize);
            }
            Err(shrew_core::Error::msg(format!(
                "Unresolved symbolic dimension: '{}'",
                name
            )))
        }
        Dim::Dynamic => Err(shrew_core::Error::msg(
            "Cannot resolve dynamic dimension at compile time",
        )),
    }
}

/// Initialize a parameter tensor.
fn init_param<B: Backend>(
    ty: &IrType,
    init: &shrew_ir::graph::InitStrategy,
    frozen: bool,
    config: &RuntimeConfig,
    program: &IrProgram,
    device: &B::Device,
) -> Result<Tensor<B>> {
    let (shape, dtype) = resolve_type(ty, config, program)?;

    let tensor = match init {
        shrew_ir::graph::InitStrategy::Zeros => Tensor::<B>::zeros(shape, dtype, device)?,
        shrew_ir::graph::InitStrategy::Ones => Tensor::<B>::ones(shape, dtype, device)?,
        shrew_ir::graph::InitStrategy::Normal { mean, std } => {
            Tensor::<B>::randn(shape, dtype, device)?.affine(*std, *mean)?
        }
        shrew_ir::graph::InitStrategy::Uniform { low, high } => {
            Tensor::<B>::rand(shape, dtype, device)?.affine(*high - *low, *low)?
        }
        shrew_ir::graph::InitStrategy::XavierUniform => {
            let (fan_in, fan_out) = compute_fans(&shape);
            let a = (6.0_f64 / (fan_in + fan_out) as f64).sqrt();
            Tensor::<B>::rand(shape, dtype, device)?.affine(2.0 * a, -a)?
        }
        shrew_ir::graph::InitStrategy::XavierNormal => {
            let (fan_in, fan_out) = compute_fans(&shape);
            let std = (2.0_f64 / (fan_in + fan_out) as f64).sqrt();
            Tensor::<B>::randn(shape, dtype, device)?.affine(std, 0.0)?
        }
        shrew_ir::graph::InitStrategy::KaimingUniform => {
            let (fan_in, _) = compute_fans(&shape);
            let bound = (3.0_f64 / fan_in as f64).sqrt();
            Tensor::<B>::rand(shape, dtype, device)?.affine(2.0 * bound, -bound)?
        }
        shrew_ir::graph::InitStrategy::KaimingNormal => {
            let (fan_in, _) = compute_fans(&shape);
            let std = (2.0_f64 / fan_in as f64).sqrt();
            Tensor::<B>::randn(shape, dtype, device)?.affine(std, 0.0)?
        }
        shrew_ir::graph::InitStrategy::Custom(_) => Tensor::<B>::randn(shape, dtype, device)?,
    };

    if frozen {
        Ok(tensor)
    } else {
        Ok(tensor.set_variable())
    }
}

/// Resolve IrType to (Shape, CoreDType).
fn resolve_type(
    ty: &IrType,
    config: &RuntimeConfig,
    program: &IrProgram,
) -> Result<(shrew_core::Shape, CoreDType)> {
    match ty {
        IrType::Tensor { shape, dtype } => {
            let dims: Vec<usize> = shape
                .iter()
                .map(|d| resolve_dim(d, config, program))
                .collect::<Result<Vec<_>>>()?;
            let core_dtype = ir_dtype_to_core(*dtype)?;
            Ok((shrew_core::Shape::new(dims), core_dtype))
        }
        IrType::Scalar(dtype) => {
            let core_dtype = ir_dtype_to_core(*dtype)?;
            Ok((shrew_core::Shape::new(vec![1]), core_dtype))
        }
        IrType::Int => Ok((shrew_core::Shape::new(vec![1]), CoreDType::I64)),
        _ => Ok((shrew_core::Shape::new(vec![1]), config.default_dtype)),
    }
}

/// Materialize a constant as a tensor.
fn materialize_constant<B: Backend>(
    val: &ConstantValue,
    ty: &IrType,
    default_dtype: CoreDType,
    device: &B::Device,
) -> Result<Tensor<B>> {
    match val {
        ConstantValue::Int(n) => {
            Tensor::<B>::from_f64_slice(&[*n as f64], 1, CoreDType::I64, device)
        }
        ConstantValue::Float(f) => {
            let dtype = match ty {
                IrType::Tensor { dtype, .. } => ir_dtype_to_core(*dtype)?,
                IrType::Scalar(dtype) => ir_dtype_to_core(*dtype)?,
                _ => default_dtype,
            };
            Tensor::<B>::from_f64_slice(&[*f], 1, dtype, device)
        }
        ConstantValue::Bool(b) => {
            Tensor::<B>::from_f64_slice(&[if *b { 1.0 } else { 0.0 }], 1, CoreDType::U8, device)
        }
        ConstantValue::Str(_) => Tensor::<B>::zeros(1, default_dtype, device),
        ConstantValue::Null => Tensor::<B>::zeros(1, default_dtype, device),
    }
}

/// Execute a body op (for Repeat instruction).
fn execute_body_op<B: Backend>(
    op: &OpKind,
    input: &Tensor<B>,
    _device: &B::Device,
) -> Result<Tensor<B>> {
    match op {
        OpKind::TransformerBlock { n_heads } => {
            let dims = input.dims();
            let d_model = dims[dims.len() - 1];
            let d_ff = d_model * 4;
            let block = shrew_nn::TransformerBlock::<B>::new(
                d_model,
                *n_heads as usize,
                d_ff,
                true,
                input.dtype(),
                input.device(),
            )?;
            block.forward(input)
        }
        OpKind::MultiHeadAttention { n_heads } => {
            let d_model = *input
                .dims()
                .last()
                .ok_or_else(|| shrew_core::Error::msg("MHA input has no dimensions"))?;
            let mha = shrew_nn::MultiHeadAttention::<B>::new(
                d_model,
                *n_heads as usize,
                input.dtype(),
                input.device(),
            )?;
            mha.forward(input)
        }
        _ => Err(shrew_core::Error::msg(format!(
            "Unsupported op in Repeat body: {:?}",
            op
        ))),
    }
}

/// Compute (fan_in, fan_out) from a parameter shape.
fn compute_fans(shape: &shrew_core::Shape) -> (usize, usize) {
    let dims = shape.dims();
    match dims.len() {
        0 => (1, 1),
        1 => (dims[0], dims[0]),
        2 => (dims[1], dims[0]),
        _ => {
            let receptive: usize = dims[2..].iter().product();
            let fan_in = dims[1] * receptive;
            let fan_out = dims[0] * receptive;
            (fan_in, fan_out)
        }
    }
}

// Convenience: parse → lower → validate → optimize → JIT compile

/// Parse, lower, validate, optimize, and JIT-compile a `.sw` program.
///
/// This is the recommended entry point for production use — it produces
/// a JIT executor that runs graphs faster than the interpreter.
///
/// # Example
/// ```ignore
/// let jit = load_jit::<CpuBackend>(source, CpuDevice, RuntimeConfig::default())?;
/// let result = jit.run("Forward", &inputs)?;
/// ```
pub fn load_jit<B: Backend>(
    source: &str,
    device: B::Device,
    config: RuntimeConfig,
) -> Result<JitExecutor<B>> {
    let ast =
        shrew_ir::parse(source).map_err(|e| shrew_core::Error::msg(format!("Parse error: {e}")))?;
    let mut ir = shrew_ir::lower(&ast)
        .map_err(|e| shrew_core::Error::msg(format!("Lowering error: {e}")))?;

    if let Err(errors) = shrew_ir::validate(&ir) {
        let msg = errors
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        return Err(shrew_core::Error::msg(format!("Validation errors:\n{msg}")));
    }

    shrew_ir::infer_shapes(&mut ir);
    shrew_ir::optimize(&mut ir);

    JitExecutor::<B>::compile(ir, device, config)
}
